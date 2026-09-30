from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.p4.extract_b3b_joint_scene import (
    FIELDS, pack_batch, parse_intervals, verify_shard,
)


def test_parse_disjoint_persistent_worker_intervals():
    assert parse_intervals(None, None, "12000:24000,0:12000,36000:48000") == [
        (0, 12000), (12000, 24000), (36000, 48000),
    ]
    assert parse_intervals(0, 512, None) == [(0, 512)]
    with pytest.raises(ValueError, match="overlap"):
        parse_intervals(None, None, "0:12000,6000:18000")
    with pytest.raises(ValueError, match="combined"):
        parse_intervals(0, 512, "0:512")


def test_same_forward_batch_contains_scene_and_paths():
    batch, arms, points = 2, 6, 4
    routes = torch.zeros(batch, arms, points, 2)
    routes[..., 0] = torch.arange(1, points + 1)
    anchor = torch.zeros(batch, arms, 4)
    anchor[:, 0, 3] = 1
    prediction = SimpleNamespace(
        pred_scene_tokens=torch.ones(batch, 10, 8),
        pred_route_multimodal=routes,
        pred_route_anchor=anchor,
        pred_route_conf=torch.zeros(batch, arms),
        pred_route_selected_idx=torch.zeros(batch, dtype=torch.long),
        pred_target_speed_scalar=torch.tensor([4.0, 5.0]),
    )
    data = {
        "speed": torch.tensor([3.0, 4.0]),
        "route": routes[:, 0].clone(),
        "command": torch.tensor([3, 4]),
    }
    packed = pack_batch(
        prediction, data, ["a", "b"], (-1.5, -0.75, 0.75, 1.5),
        spatial_shape=(2, 2), has_intent_tokens=True,
    )
    assert set(packed) == set(FIELDS)
    assert packed["scene_tokens"].shape[0] == batch
    assert packed["routes"].shape == (batch, 10, points, 2)
    assert packed["route_valid"].sum(axis=1).tolist() == [5, 5]
    assert packed["raw_target_speed"].tolist() == [4.0, 5.0]
    assert not packed["anchor_fallback"].any()


def test_executed_fallback_remains_a_valid_candidate():
    routes = torch.ones(1, 6, 3, 2)
    prediction = SimpleNamespace(
        pred_scene_tokens=torch.ones(1, 10, 8),
        pred_route_multimodal=routes,
        pred_route_anchor=torch.zeros(1, 6, 4),
        pred_route_conf=torch.zeros(1, 6),
        pred_route_selected_idx=torch.tensor([0]),
        pred_target_speed_scalar=torch.tensor([2.0]),
    )
    data = {"speed": torch.tensor([1.0]), "route": routes[:, 0],
            "command": torch.tensor([0])}
    packed = pack_batch(
        prediction, data, ["a"], (-1.5, -0.75, 0.75, 1.5),
        spatial_shape=(2, 2), has_intent_tokens=True,
    )
    assert packed["anchor_fallback"].tolist() == [True]
    assert packed["route_valid"].sum() == 5


def test_verify_rejects_wrong_source_checkpoint(tmp_path):
    checkpoint = tmp_path / "model_0019.pth"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.jsonl"
    nearest = tmp_path / "sparse.jsonl"
    manifest.write_text("")
    nearest.write_text("")
    path = tmp_path / "joint_scene_000000_000001.npz"
    fields = {name: np.zeros((1, 1), dtype=np.float16) for name in FIELDS}
    fields["keys"] = np.array(["a"])
    fields["scene_tokens"] = np.zeros((1, 2, 8), dtype=np.float16)
    fields["routes"] = np.zeros((1, 10, 3, 2), dtype=np.float16)
    fields["route_valid"] = np.ones((1, 10), dtype=bool)
    fields["route_conf"] = np.zeros((1, 6), dtype=np.float16)
    np.savez_compressed(
        path, **fields, schema_version=1, same_forward=True,
        source_checkpoint=str(checkpoint.resolve()),
        checkpoint_size=checkpoint.stat().st_size,
        checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
        source_manifest=str(manifest.resolve()),
        nearest_vlm_manifest=str(nearest.resolve()),
        start=0, end=1,
        local_offsets_m=np.array([-1.5, -0.75, 0.75, 1.5]),
    )
    assert verify_shard(path, checkpoint, manifest, nearest, 0, 1,
                        (-1.5, -0.75, 0.75, 1.5)) == 1
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_shard(path, checkpoint, manifest, nearest, 0, 1,
                     (-1.5, -0.75, 0.75, 1.5))
