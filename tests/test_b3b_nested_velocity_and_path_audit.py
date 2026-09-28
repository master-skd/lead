import json

import numpy as np
import pytest

from scripts.p4.audit_b3b_local_paths import (
    add_base_vocabulary_selection,
    max_prefix_curvature,
    prefix_corridor_max,
    yaw_rate_limited,
)
from scripts.p4.build_b3b_nested_velocity_vocab import build_vocab


def test_nested_velocity_vocab_preserves_base_prefix(tmp_path):
    base = np.array([[0.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    extra = np.array([[-2.0, -2.0]], dtype=np.float32)
    p16, p32 = tmp_path / "k16.npy", tmp_path / "k32.npy"
    np.save(p16, base)
    np.save(p32, extra)
    output = tmp_path / "nested.npy"
    report = build_vocab([p16, p32], output)
    assert report["prefix_preserved"]
    np.testing.assert_array_equal(np.load(output), np.concatenate((base, extra)))
    assert output.with_suffix(".json").exists()


def test_curvature_and_corridor_prefix_screen():
    straight = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    bend = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0]])
    assert max_prefix_curvature(straight, 3.0) == pytest.approx(0.0)
    assert max_prefix_curvature(bend, 3.0) > 0.3
    assert prefix_corridor_max(straight, np.array([0.0, 0.2, 0.8]), 2.1) == pytest.approx(0.2)


def test_nested_audit_reselects_preserved_base_candidates(tmp_path):
    vocab = tmp_path / "nested.npy"
    vocab.with_suffix(".json").write_text(json.dumps({
        "prefix_preserved": True,
        "sources": [{"shape": [1, 8]}, {"shape": [1, 8]}],
    }))
    report = {
        "vocabulary": str(vocab),
        "original_path_count": 1,
        "policy": {"max_off_corridor_cost": 0.1, "min_progress_ratio": 0.5},
    }
    arrays = {
        "key": np.array(["frame"]),
        "winner": np.array([0]),
        "raw_collision": np.array([True]),
        "route_valid": np.array([[True, False]]),
        "direction_ok": np.array([[True, True]]),
        "corridor_cost": np.zeros((1, 2)),
        "route_length": np.array([[10.0, 10.0]]),
        "candidate_valid": np.ones((1, 3), dtype=bool),
        "candidate_progress": np.array([[5.0, 4.0, 3.0]]),
        "candidate_pid_target": np.array([[1.0, 1.0, 1.0]]),
        "raw_progress": np.array([5.0]),
        "raw_target_speed": np.array([1.0]),
        "candidate_collision": np.array([[
            [True, True, False], [True, True, True],
        ]]),
    }
    assert add_base_vocabulary_selection(arrays, report) == 1
    assert not arrays["conservative_base_k1_expanded_rescue"][0]


def test_yaw_rate_limit_starts_at_current_ego_heading():
    yaws = np.deg2rad(np.array([30.0, 30.0, 30.0]))
    np.testing.assert_allclose(
        np.rad2deg(yaw_rate_limited(yaws, 30.0)), [7.5, 15.0, 22.5], atol=1e-5
    )
