"""Small CPU integration test for P7 query, WTA and paired speed wiring."""

import torch

from lead.tfv6.planning_decoder import PlanningDecoder
from lead.tfv6.joint_online_training import select_route_speed_pair
from lead.tfv6.route_safety_rescorer import highest_confidence_valid
from lead.training.config_training import TrainingConfig


def test_semantic_queries_produce_k_routes_and_k_speeds_with_gradient():
    config = TrainingConfig(loaded_config={
        "radar_detection": False, "use_radar_detection": False,
    })
    config.use_mixed_precision_training = False
    config.transfuser_token_dim = 32
    config.transfuser_num_bev_cross_attention_heads = 4
    config.transfuser_num_bev_cross_attention_layers = 1
    config.num_route_points_prediction = 3
    config.multimodal_planner = True
    config.multimodal_planner_k = 3
    config.multimodal_target_speed = True
    config.joint_scorer_replaces_conf = False
    config.use_semantic_intent_planner = True
    config.use_control_conditioning = False
    config.use_radars = False
    config.use_collision_cost = False

    decoder = PlanningDecoder(8, config, torch.device("cpu"))
    decoder.train()
    batch = 2
    valid = torch.tensor([[True, True, False], [True, False, False]])
    data = {
        "semantic_intent_maneuver_ids": torch.tensor([[1, 2, 0], [1, 0, 0]]),
        "semantic_intent_obligations": torch.zeros(batch, 3, 6),
        "semantic_intent_scene_rules": torch.tensor([[1, 1, 1, 1], [2, 1, 1, 1]]),
        "semantic_intent_valid": valid,
        "speed": torch.tensor([5.0, 4.0]),
        "command": torch.tensor([[0., 0., 0., 1., 0., 0.]] * batch),
        "target_point": torch.zeros(batch, 2),
        "target_point_previous": torch.zeros(batch, 2),
        "target_point_next": torch.zeros(batch, 2),
    }
    decoder(torch.randn(batch, 8, 4, 4), None, None, data, {})
    routes = data["route_multimodal"]
    speeds = data["target_speed_multimodal_distribution"]
    assert routes.shape == (batch, 3, 3, 2)
    assert speeds.shape[:2] == (batch, 3)
    assert data["route_conf"].shape == (batch, 3)

    gt = torch.zeros(batch, 3, 2)
    winner = decoder._route_winner(gt, data)
    assert valid[torch.arange(batch), winner].all()
    losses = {}
    decoder._multimodal_route_loss(gt, data, losses, {})
    assert "loss_route_anchor" not in losses
    assert "loss_route_corridor" not in losses
    matched_speed = speeds[torch.arange(batch), winner]
    (sum(losses.values()) + matched_speed.square().mean()).backward()
    assert decoder.semantic_intent_encoder.maneuver.weight.grad is not None
    assert torch.count_nonzero(decoder.semantic_intent_encoder.maneuver.weight.grad) > 0


def test_p7_rejects_missing_training_intents():
    config = TrainingConfig(loaded_config={
        "radar_detection": False, "use_radar_detection": False,
    })
    config.use_mixed_precision_training = False
    config.multimodal_planner = True
    config.multimodal_target_speed = True
    config.joint_scorer_replaces_conf = False
    config.use_semantic_intent_planner = True
    config.use_radars = False
    decoder = PlanningDecoder(8, config, torch.device("cpu"))
    decoder.train()
    data = {
        "semantic_intent_maneuver_ids": torch.zeros(1, config.multimodal_planner_k, dtype=torch.long),
        "semantic_intent_obligations": torch.zeros(1, config.multimodal_planner_k, 6),
        "semantic_intent_scene_rules": torch.zeros(1, 4, dtype=torch.long),
        "semantic_intent_valid": torch.zeros(1, config.multimodal_planner_k, dtype=torch.bool),
        "speed": torch.ones(1),
        "command": torch.tensor([[0., 0., 0., 1., 0., 0.]]),
        "target_point": torch.zeros(1, 2),
        "target_point_previous": torch.zeros(1, 2),
        "target_point_next": torch.zeros(1, 2),
    }
    try:
        decoder(torch.randn(1, 8, 4, 4), None, None, data, {})
    except ValueError as exc:
        assert "at least one valid intent" in str(exc)
    else:
        raise AssertionError("an empty P7 training sample must be handled explicitly")


def test_confidence_selection_keeps_route_and_speed_on_same_valid_arm():
    routes = torch.tensor([[[[1., 0.]], [[2., 0.]], [[3., 0.]]]])
    logits = torch.tensor([[[1., 0.], [2., 0.], [3., 0.]]])
    speeds = torch.tensor([[4., 7., 10.]])
    confidence = torch.tensor([[0.2, 0.8, 100.0]])
    valid = torch.tensor([[True, True, False]])
    selected = highest_confidence_valid(confidence, valid)
    route, dist, speed = select_route_speed_pair(routes, logits, speeds, selected)
    assert selected.item() == 1
    assert route[0, 0, 0].item() == 2.0
    assert dist[0, 0].item() == 2.0
    assert speed.item() == 7.0
