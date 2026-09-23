from types import SimpleNamespace

import numpy as np
import torch

from lead.tfv6.future_collision import speed_profile_distances
from lead.tfv6.velocity_scorer import (
    VelocityScorer,
    apply_velocity_scorer_gate,
    build_velocity_candidates,
    load_velocity_scorer,
)
from scripts.p4.eval_b3a_velocity_scorer import evaluate_gate
from scripts.p4.eval_b3a_velocity_vocab_oracle import (
    interval_speeds_from_distances,
)


def test_velocity_scorer_returns_per_candidate_logits_and_gradients():
    scorer = VelocityScorer(
        route_feature_dim=4, profile_steps=3, hidden_dim=8, dropout=0.0
    )
    route = torch.randn(2, 4)
    current = torch.tensor([2.0, 4.0])
    raw = torch.tensor([3.0, 3.0])
    candidates = torch.randn(2, 5, 3).abs()
    preference, collision = scorer(route, current, raw, candidates)
    assert preference.shape == (2, 5)
    assert collision.shape == (2, 5)
    (preference.mean() + collision.mean()).backward()
    assert all(parameter.grad is not None for parameter in scorer.parameters())


def test_velocity_scorer_checkpoint_round_trip():
    scorer = VelocityScorer(
        route_feature_dim=4, profile_steps=3, hidden_dim=8, dropout=0.0
    )
    loaded = load_velocity_scorer(
        {"scorer_config": scorer.checkpoint_config(), "model": scorer.state_dict()}
    )
    route = torch.randn(1, 4)
    current = torch.tensor([2.0])
    raw = torch.tensor([3.0])
    candidates = torch.randn(1, 2, 3).abs()
    expected = scorer(route, current, raw, candidates)
    actual = loaded(route, current, raw, candidates)
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        torch.testing.assert_close(expected_tensor, actual_tensor)


def test_gate_keeps_safe_raw_and_rescues_triggered_collision():
    arrays = SimpleNamespace(
        keys=np.asarray(["safe", "rescue", "unresolved"]),
        candidate_valid=np.ones((3, 3), dtype=bool),
        collision=np.asarray(
            [[False, False, True], [True, False, True], [True, True, True]]
        ),
        imitation_error=np.asarray(
            [[0.1, 0.2, 1.0], [1.0, 0.2, 2.0], [1.0, 2.0, 3.0]],
            dtype=np.float32,
        ),
    )
    preference = np.asarray([[3.0, 2.0, 1.0], [0.0, 3.0, 1.0], [0.0, 3.0, 1.0]])
    risk = np.asarray([[0.1, 0.1, 0.9], [0.9, 0.1, 0.9], [0.9, 0.9, 0.9]])
    result = evaluate_gate(arrays, preference, risk, 0.7, 0.3)
    assert result["switch_rate"] == 1 / 3
    assert result["collision_before"] == 2 / 3
    assert result["collision_after"] == 1 / 3
    assert result["rescued_count"] == 1
    assert result["false_switch_rate"] == 0.0
    assert result["introduced_collision_rate"] == 0.0


def test_online_raw_candidate_matches_feature_extraction_profile():
    current = torch.tensor([0.0, 3.0, 10.0])
    target = torch.tensor([8.0, 0.0, 12.0])
    vocabulary = torch.zeros(2, 8)
    candidates, valid = build_velocity_candidates(current, target, vocabulary)
    expected = np.stack(
        [
            interval_speeds_from_distances(
                speed_profile_distances(float(v0), float(v1)), 0.25
            )
            for v0, v1 in zip(current, target, strict=True)
        ]
    )
    np.testing.assert_allclose(candidates[:, 0].numpy(), expected, atol=1e-5)
    assert valid[:, 0].all()


def test_velocity_vocabulary_rejects_unreachable_later_jump():
    current = torch.tensor([3.0])
    target = torch.tensor([3.0])
    vocabulary = torch.tensor(
        [
            [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
        ]
    )
    _, valid = build_velocity_candidates(
        current, target, vocabulary, full_profile_reachability=True
    )
    # Candidate zero is always the raw fallback; the first residual candidate
    # jumps by 12 m/s^2 halfway through and must no longer reach the scorer.
    torch.testing.assert_close(valid, torch.tensor([[True, False, True]]))


class _FixedVelocityScorer(torch.nn.Module):
    def __init__(self, preference: torch.Tensor, risk: torch.Tensor):
        super().__init__()
        self.preference = preference
        self.collision_logit = torch.logit(risk)

    def forward(self, route, current, raw, candidates):
        del route, current, raw
        count = len(candidates)
        return self.preference[:count], self.collision_logit[:count]


def test_online_gate_preserves_safe_raw_and_uses_selected_terminal_speed():
    scorer = _FixedVelocityScorer(
        preference=torch.tensor([[0.0, 4.0], [0.0, 4.0]]),
        risk=torch.tensor([[0.1, 0.1], [0.9, 0.1]]),
    )
    # The first interval is reachable under -4.95 m/s^2; later bins may commit to stop.
    vocabulary = torch.tensor([[-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.0, -3.0]])
    result = apply_velocity_scorer_gate(
        scorer,
        torch.zeros(2, 4),
        torch.tensor([3.0, 3.0]),
        torch.tensor([[5.0], [5.0]]),
        vocabulary,
    )
    torch.testing.assert_close(result.target_speed, torch.tensor([[5.0], [0.0]]))
    torch.testing.assert_close(result.selected_index, torch.tensor([0, 1]))
    torch.testing.assert_close(result.switched, torch.tensor([False, True]))
    torch.testing.assert_close(result.fallback, torch.tensor([False, False]))


def test_online_gate_falls_back_to_raw_when_no_safe_alternative():
    scorer = _FixedVelocityScorer(
        preference=torch.tensor([[0.0, 4.0]]),
        risk=torch.tensor([[0.9, 0.7]]),
    )
    result = apply_velocity_scorer_gate(
        scorer,
        torch.zeros(1, 4),
        torch.tensor([3.0]),
        torch.tensor([[5.0]]),
        torch.zeros(1, 8),
    )
    torch.testing.assert_close(result.target_speed, torch.tensor([[5.0]]))
    torch.testing.assert_close(result.switched, torch.tensor([False]))
    torch.testing.assert_close(result.fallback, torch.tensor([True]))
