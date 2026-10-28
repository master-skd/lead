import torch

from lead.tfv6.semantic_intent_encoder import SemanticIntentEncoder
from lead.tfv6.semantic_intent_slots import prepare_planner_intents, slots_to_tensors


def _proposal(obligation):
    return {
        "scene_rules": {
            "signal": "none", "stop_sign": "no",
            "pedestrian_conflict": "no", "vehicle_conflict": "no",
        },
        "intents": [
            {"maneuver": "keep_lane", "obligations": [obligation], "confidence": 0.8},
            {"maneuver": "change_lane_left", "obligations": [], "confidence": 0.7},
        ],
    }


def test_one_vector_per_intent_and_padding_is_zero():
    packed = prepare_planner_intents(_proposal("follow_lead"), "follow_lane", capacity=3)
    tensors = slots_to_tensors([packed])
    encoder = SemanticIntentEncoder(32)
    navigation = torch.tensor([[0., 0., 0., 1., 0., 0.]])
    out = encoder(
        tensors["semantic_intent_maneuver_ids"],
        tensors["semantic_intent_obligations"],
        tensors["semantic_intent_scene_rules"],
        navigation, tensors["semantic_intent_valid"],
    )
    assert out.shape == (1, 3, 32)
    assert torch.count_nonzero(out[0, 2]) == 0
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_branch_obligation_changes_its_token_and_backpropagates():
    a = slots_to_tensors([prepare_planner_intents(
        _proposal("follow_lead"), "follow_lane", capacity=2,
    )])
    b = slots_to_tensors([prepare_planner_intents(
        _proposal("yield_vehicle"), "follow_lane", capacity=2,
    )])
    encoder = SemanticIntentEncoder(32)
    navigation = torch.tensor([[0., 0., 0., 1., 0., 0.]])

    def encode(batch):
        return encoder(
            batch["semantic_intent_maneuver_ids"], batch["semantic_intent_obligations"],
            batch["semantic_intent_scene_rules"], navigation,
            batch["semantic_intent_valid"],
        )

    out_a, out_b = encode(a), encode(b)
    assert not torch.allclose(out_a[0, 0], out_b[0, 0])
    torch.sum(out_a[0, 0]).backward()
    assert encoder.obligations.weight.grad is not None
    assert torch.count_nonzero(encoder.obligations.weight.grad) > 0
