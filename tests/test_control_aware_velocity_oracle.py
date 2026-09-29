import numpy as np
import pytest

from lead.data_loader.future_actor_cache import FutureActorFrame
from lead.expert.config_expert import ExpertConfig
from lead.tfv6.control_aware_velocity_oracle import (
    Dynamics,
    rollout_scalar_target,
    scalar_actions,
)
from scripts.p4.audit_b3a_control_aware_actions import evaluate_frame
from scripts.p4.audit_b3a_short_horizon_actions import audit_horizon
from scripts.p4.verify_b3a_selected_paths import verify_shard


def _empty_actors():
    return FutureActorFrame(
        positions=np.zeros((0, 8, 2), np.float32),
        yaws=np.zeros((0, 8), np.float32),
        extents=np.zeros((0, 3), np.float32),
        z=np.zeros(0, np.float32),
        valid=np.zeros((0, 8), bool),
        actor_ids=np.zeros(0, np.int64),
        class_ids=np.zeros(0, np.uint8),
        ego_extent=np.array([2.45, 0.95, 0.75], np.float32),
    )


def test_actions_deduplicate_stops_and_preserve_raw():
    targets, valid = scalar_actions(0.0)
    assert targets[0] == 0.0
    assert valid.tolist() == [True, False, False, False, False, True]
    targets, valid = scalar_actions(6.0)
    assert targets.tolist() == [6.0, 5.5, 5.0, 4.0, 0.0, 7.0]
    assert valid.all()


def test_controller_rollout_exposes_full_brake_from_scalar_collapse():
    config = ExpertConfig()
    baseline = rollout_scalar_target(5.21, 6.06, expert_config=config)
    slower = rollout_scalar_target(5.21, 4.65, expert_config=config)
    assert baseline.acceleration_mps2[0] > 0.0
    assert slower.acceleration_mps2[0] < 0.0
    assert slower.brake_ticks > baseline.brake_ticks
    assert slower.brake_ticks > 0
    assert slower.progress_m < baseline.progress_m
    assert len(slower.acceleration_mps2) == 40
    assert len(slower.xy_distance_m) == 8


def test_short_horizon_coverage_uses_executable_moving_action():
    targets, valid = scalar_actions(3.0)
    collision = np.zeros((1, len(targets)), bool)
    collision[0, 0] = True
    ttc = np.full(collision.shape, 2.0, np.float32)
    ttc[0, 0] = 0.75
    summary, details = audit_horizon(
        horizon_s=1.0,
        keys=np.array(["scenario__route__0001"]),
        targets=targets[None],
        valid=valid[None],
        collisions_2s=collision,
        ttc_s=ttc,
        current_speeds=np.array([3.0]),
        route_ends=np.array([20.0]),
        dynamics=Dynamics(),
        max_progress_loss_m=2.0,
    )
    assert summary["raw_unsafe_qualified_frames"] == 1
    assert summary["moving_rescue_frames"] == 1
    assert summary["stop_only_rescue_frames"] == 0
    assert details["moving_rescue"].tolist() == [True]


def test_fixed_path_labels_are_multidimensional_and_keep_baseline_when_safe():
    velocity = np.full((1, 8), 5.0, np.float32)
    states = np.zeros((1, 8, 6), np.float32)
    states[0, :, 0] = np.arange(1, 9) * 1.25
    states[0, :, 3] = 1.0
    states[0, :, 4] = 5.0
    result = evaluate_frame(
        current_speed=5.0,
        raw_target=5.0,
        candidate_velocity=velocity,
        candidate_states=states,
        candidate_valid=np.array([True]),
        actors=_empty_actors(),
        dynamics=Dynamics(),
        expert_config=ExpertConfig(),
        margin_m=0.2,
        max_progress_loss_m=2.0,
    )
    assert result["selected_index"] == 0
    assert result["safe_any"]
    assert not result["collision"].any()
    assert result["progress_m"][0] > result["progress_m"][4]
    assert 0.0 < result["comfort"][0] <= 1.0


def test_oracle_only_rescues_when_a_collision_is_avoided_with_bounded_progress():
    velocity = np.full((1, 8), 3.0, np.float32)
    states = np.zeros((1, 8, 6), np.float32)
    states[0, :, 0] = np.arange(1, 9) * 0.75
    states[0, :, 3] = 1.0
    actors = FutureActorFrame(
        positions=np.tile(np.array([[[8.0, 0.0]]], np.float32), (1, 8, 1)),
        yaws=np.zeros((1, 8), np.float32),
        extents=np.array([[1.0, 0.8, 1.0]], np.float32),
        z=np.zeros(1, np.float32),
        valid=np.ones((1, 8), bool),
        actor_ids=np.array([1], np.int64),
        class_ids=np.array([1], np.uint8),
        ego_extent=np.array([2.45, 0.95, 0.75], np.float32),
    )
    result = evaluate_frame(
        current_speed=3.0,
        raw_target=3.0,
        candidate_velocity=velocity,
        candidate_states=states,
        candidate_valid=np.array([True]),
        actors=actors,
        dynamics=Dynamics(),
        expert_config=ExpertConfig(),
        margin_m=0.2,
        max_progress_loss_m=3.0,
    )
    assert result["collision"][0]
    assert result["selected_index"] == 2
    assert not result["collision"][result["selected_index"]]


def test_full_selected_path_avoids_candidate_sample_endpoint_overflow():
    velocity = np.full((1, 8), 5.0, np.float32)
    states = np.zeros((1, 8, 6), np.float32)
    states[0, :, 0] = np.arange(1, 9) * 1.25
    states[0, :, 3] = 1.0
    path = np.stack((np.arange(1, 31, dtype=np.float32), np.zeros(30)), axis=1)
    result = evaluate_frame(
        current_speed=8.0,
        raw_target=10.0,
        candidate_velocity=velocity,
        candidate_states=states,
        candidate_valid=np.array([True]),
        selected_path=path,
        actors=_empty_actors(),
        dynamics=Dynamics(),
        expert_config=ExpertConfig(),
        margin_m=0.2,
        max_progress_loss_m=2.0,
    )
    assert result["progress_m"][0] > 10.0
    assert not result["route_overflow"][0]


def test_selected_path_verifier_checks_key_and_geometry_alignment(tmp_path):
    feature = tmp_path / "velocity_features_000000_000001.npz"
    sidecar = tmp_path / "selected_paths_000000_000001.npz"
    path = np.stack((np.arange(1, 9, dtype=np.float32) * 0.75, np.zeros(8)), axis=1)
    states = np.zeros((1, 1, 8, 6), np.float32)
    states[0, 0, :, :2] = path
    np.savez_compressed(
        feature,
        keys=np.array(["frame_a"]),
        current_speed=np.array([3.0]),
        raw_target_speed=np.array([3.0]),
        selected_arm=np.array([2]),
        source_checkpoint=np.array("checkpoint.pth"),
        source_manifest=np.array("manifest.jsonl"),
        nearest_vlm_manifest=np.array("vlm.jsonl"),
        candidate_velocity=np.full((1, 1, 8), 3.0),
        candidate_states=states,
    )
    np.savez_compressed(
        sidecar,
        keys=np.array(["frame_a"]),
        current_speed=np.array([3.0]),
        raw_target_speed=np.array([3.0]),
        selected_arm=np.array([2]),
        source_checkpoint=np.array("checkpoint.pth"),
        source_manifest=np.array("manifest.jsonl"),
        nearest_vlm_manifest=np.array("vlm.jsonl"),
        selected_path=path[None],
    )
    assert verify_shard(feature, sidecar)["sampled_candidate_point_error_max_m"] < 1e-5
    np.savez_compressed(
        sidecar,
        keys=np.array(["wrong_key"]),
        current_speed=np.array([3.0]),
        raw_target_speed=np.array([3.0]),
        selected_arm=np.array([2]),
        source_checkpoint=np.array("checkpoint.pth"),
        source_manifest=np.array("manifest.jsonl"),
        nearest_vlm_manifest=np.array("vlm.jsonl"),
        selected_path=path[None],
    )
    with pytest.raises(ValueError, match="keys/order mismatch"):
        verify_shard(feature, sidecar)
