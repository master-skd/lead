from types import SimpleNamespace

import numpy as np
import torch

from lead.tfv6.route_safety_head import (
    RouteSafetyHead,
    binary_average_precision,
    binary_auroc,
)
from scripts.p4.eval_b2d_safety_head import evaluate_gate


def test_route_safety_head_supports_frame_and_arm_batches():
    head = RouteSafetyHead(route_feature_dim=8, hidden_dim=4, dropout=0.0)
    assert head(torch.randn(5, 8), torch.randn(5), torch.randn(5)).shape == (5,)
    assert head(torch.randn(3, 2, 8), torch.randn(3), torch.randn(3)).shape == (3, 2)


def test_binary_ranking_metrics_are_perfect_with_ties():
    score = np.array([0.1, 0.1, 0.8, 0.9])
    label = np.array([False, False, True, True])
    assert binary_auroc(score, label) == 1.0
    assert binary_average_precision(score, label) == 1.0


def test_gate_only_uses_low_risk_alternate_and_reports_false_switch():
    arrays = SimpleNamespace(
        keys=np.array(["a", "b", "c"]),
        valid=np.ones((3, 2), dtype=bool),
        confidence=np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]),
        collision=np.array([[True, False], [False, False], [True, True]]),
        ade=np.array([[0.1, 0.2], [0.1, 0.3], [0.1, 0.4]]),
    )
    risk = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
    result = evaluate_gate(arrays, risk, unsafe_threshold=0.8, safe_threshold=0.2)
    assert result["rescued_count"] == 1
    assert result["rescue_possible_count"] == 1
    assert np.isclose(result["collision_before"], 2 / 3)
    assert np.isclose(result["collision_after"], 1 / 3)
    assert np.isclose(result["false_switch_rate"], 1 / 3)
