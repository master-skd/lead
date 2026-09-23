from argparse import Namespace

import numpy as np
import torch

from lead.tfv6.trajectory_scene_scorer import (
    SCORE_HEADS,
    TrajectorySceneScorer,
    build_trajectory_candidate_states,
    compress_planner_scene_tokens,
    select_scene_scored_trajectory,
    trajectory_score_loss,
)
from scripts.p4.audit_b3a_trajectory_scene_scorer import (
    _within_frame_safety_ranking,
)
from scripts.p4.train_b3a_trajectory_scene_scorer import _safety_gate_failures


def test_candidate_states_compose_path_and_velocity():
    route = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    velocity = torch.tensor([[[2.0, 2.0], [1.0, 2.0]]])
    state = build_trajectory_candidate_states(
        route, velocity, torch.tensor([2.0]), interval_s=0.5
    )
    assert state.shape == (1, 2, 2, 6)
    torch.testing.assert_close(
        state[0, 0, :, :2], torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    )
    torch.testing.assert_close(
        state[0, 0, :, 2:4], torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    )
    torch.testing.assert_close(state[0, 1, :, 5], torch.tensor([-4.0, 2.0]))


def test_scene_token_compression_preserves_non_spatial_tail():
    tokens = torch.arange(2 * 25 * 4, dtype=torch.float32).reshape(2, 25, 4)
    compressed = compress_planner_scene_tokens(
        tokens,
        spatial_shape=(3, 4),
        pooled_shape=(2, 2),
        has_intent_tokens=True,
    )
    assert compressed.shape == (2, 9, 4)
    torch.testing.assert_close(compressed[:, -1], tokens[:, -1])


def test_trajectory_scene_scorer_outputs_all_heads_and_gradients():
    scorer = TrajectorySceneScorer(
        state_dim=6,
        scene_dim=12,
        hidden_dim=16,
        profile_steps=4,
        num_heads=4,
        temporal_layers=1,
        interaction_layers=1,
        dropout=0.0,
    )
    state = torch.randn(2, 3, 4, 6)
    scene = torch.randn(2, 7, 12)
    valid = torch.tensor([[True, True, False], [True, False, True]])
    logits = scorer(state, scene, valid)
    assert set(logits) == set(SCORE_HEADS)
    assert all(value.shape == (2, 3) for value in logits.values())
    labels = {name: torch.rand(2, 3) for name in SCORE_HEADS}
    loss, losses = trajectory_score_loss(logits, labels, valid)
    loss.backward()
    assert set(losses) == {*SCORE_HEADS, "total"}
    assert all(parameter.grad is not None for parameter in scorer.parameters())


def test_selection_filters_safety_then_ranks_progress():
    probability = {
        "collision_free": torch.tensor([[0.9, 0.4, 0.8]]),
        "ttc": torch.tensor([[0.9, 0.9, 0.8]]),
        "progress": torch.tensor([[0.2, 1.0, 0.8]]),
        "comfort": torch.tensor([[0.9, 0.9, 0.8]]),
        "imitation": torch.tensor([[0.9, 0.9, 0.2]]),
    }
    logits = {name: torch.logit(value) for name, value in probability.items()}
    result = select_scene_scored_trajectory(logits, torch.ones(1, 3, dtype=torch.bool))
    # Candidate 1 has highest progress but fails collision-free filtering.
    torch.testing.assert_close(result.selected_index, torch.tensor([2]))
    torch.testing.assert_close(result.fallback, torch.tensor([False]))


def test_within_frame_ranking_uses_counterfactual_pairs():
    scores = np.array([[0.9, 0.1, 0.8], [0.2, 0.3, 0.4]])
    valid = np.array([[True, True, True], [True, True, False]])
    collision = np.array([[False, True, False], [False, True, False]])
    ranking = _within_frame_safety_ranking(scores, valid, collision)
    assert ranking == {"mixed_frames": 2, "pairs": 3, "safe_above_unsafe": 2 / 3}


def test_safety_gate_rejects_collision_regression_despite_good_auc():
    args = Namespace(
        max_switch_rate=0.1,
        max_fallback_rate=0.05,
        min_collision_auroc=0.8,
        min_within_frame_safety_rank=0.6,
        max_mae_increase=0.1,
    )
    metrics = {
        "raw_collision_rate": 0.025,
        "selected_collision_rate": 0.02,
        "rescue_rate": 0.01,
        "introduced_collision_rate": 0.005,
        "switch_rate": 0.05,
        "fallback_rate": 0.01,
        "collision_auroc": 0.9,
        "within_frame_safe_above_unsafe": 0.7,
        "raw_profile_mae_mps": 0.5,
        "selected_profile_mae_mps": 0.55,
    }
    assert _safety_gate_failures(metrics, args) == []
    metrics["selected_collision_rate"] = 0.04
    assert "selected collisions exceed raw" in _safety_gate_failures(metrics, args)


def test_pairwise_safety_loss_prefers_safe_candidate_within_a_frame():
    labels = {name: torch.ones(1, 2) for name in SCORE_HEADS}
    labels["collision_free"] = torch.tensor([[1.0, 0.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    logits = {name: torch.zeros(1, 2) for name in SCORE_HEADS}
    logits["collision_free"] = torch.tensor([[0.0, 0.0]], requires_grad=True)
    loss, parts = trajectory_score_loss(
        logits,
        labels,
        valid,
        collision_pair_weight=2.0,
        ttc_pair_weight=0.0,
    )
    assert "collision_free_rank" in parts
    loss.backward()
    assert logits["collision_free"].grad[0, 0] < 0
    assert logits["collision_free"].grad[0, 1] > 0
