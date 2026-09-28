import json

import numpy as np
import pytest

from scripts.p4.merge_b3b_joint_oracle import merge


def _write_shard(tmp_path, name, start, arrays, *, offsets=None):
    path = tmp_path / f"{name}.json"
    frame_path = path.with_suffix(".frames.npz")
    np.savez_compressed(frame_path, **arrays)
    report = {
        "start": start,
        "end": start + len(arrays["key"]),
        "n_frames": len(arrays["key"]),
        "candidate_shape": [2, 2],
        "original_path_count": 1,
        "local_offsets_m": offsets if offsets is not None else [0.75],
        "variant_ramp_m": 8.0,
        "checkpoint": "ckpt",
        "manifest": "manifest",
        "vocabulary": "vocab",
        "future_cache_dir": "future",
        "selection_privilege": "GT",
        "corridor_source": "intent",
        "policy": {"max_off_corridor_cost": 0.1},
        "frame_records": str(frame_path),
    }
    path.write_text(json.dumps(report))
    return path


def _arrays():
    arrays = {
        "key": np.array(["a", "b", "c"]),
        "winner": np.array([0, 0, 0]),
        "multi": np.array([False, True, False]),
        "raw_collision": np.array([True, True, False]),
        "raw_progress": np.array([10.0, 10.0, 10.0]),
        "route_valid": np.ones((3, 2), dtype=bool),
        "direction_ok": np.ones((3, 2), dtype=bool),
        "corridor_cost": np.zeros((3, 2)),
        "candidate_valid": np.ones((3, 2), dtype=bool),
        "candidate_progress": np.array([[10.0, 2.0], [10.0, 8.0], [10.0, 9.0]]),
        "route_length": np.ones((3, 2)) * 15,
    }
    for policy in ("reachable", "conservative"):
        arrays[f"{policy}_fixed_rescue"] = np.array([False, True, False])
        arrays[f"{policy}_joint_rescue"] = np.array([True, True, False])
        arrays[f"{policy}_joint_arm"] = np.array([1, 0, 0])
        arrays[f"{policy}_joint_mode"] = np.array([1, 1, 0])
        arrays[f"{policy}_expanded_rescue"] = np.array([True, True, False])
        arrays[f"{policy}_expanded_arm"] = np.array([1, 0, 0])
        arrays[f"{policy}_expanded_mode"] = np.array([1, 1, 0])
    return arrays


def test_merge_recomputes_global_quantile_and_counts(tmp_path):
    arrays = _arrays()
    shard_a = _write_shard(tmp_path, "a", 5, {k: v[:1] for k, v in arrays.items()})
    shard_b = _write_shard(tmp_path, "b", 6, {k: v[1:] for k, v in arrays.items()})
    report = merge([shard_b, shard_a], tmp_path / "merged.json")
    assert (report["start"], report["end"], report["n_frames"]) == (5, 8, 3)
    result = report["results"]["reachable"]["all"]
    assert result["raw_collision_count"] == 2
    assert result["incremental_joint_rescue_count"] == 1
    assert result["rescued_progress_ratio_median"] == pytest.approx(0.5)
    assert report["results"]["reachable"]["expanded_all"][
        "local_path_selected_on_rescue_count"
    ] == 1
    with np.load(report["frame_records"], allow_pickle=False) as merged:
        np.testing.assert_array_equal(merged["key"], arrays["key"])


def test_merge_rejects_missing_interval(tmp_path):
    arrays = _arrays()
    shard_a = _write_shard(tmp_path, "a", 0, {k: v[:1] for k, v in arrays.items()})
    shard_b = _write_shard(tmp_path, "b", 2, {k: v[1:] for k, v in arrays.items()})
    with pytest.raises(ValueError, match="gap or overlap"):
        merge([shard_a, shard_b], tmp_path / "merged.json")
