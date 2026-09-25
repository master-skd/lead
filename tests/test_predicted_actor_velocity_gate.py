import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lead.tfv6.predicted_actor_velocity_gate import (
    extrapolate_detected_actors,
    predicted_candidate_collisions,
    select_safe_slowdown,
)


def test_detected_vehicle_future_and_safe_slowdown():
    boxes = np.array([[8.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.9]], dtype=np.float32)
    actors = extrapolate_detected_actors(boxes, score_threshold=0.3)
    states = np.zeros((3, 8, 6), dtype=np.float32)
    states[..., 3] = 1.0  # cos(yaw)
    states[0, :, 0] = np.arange(1, 9, dtype=np.float32) * 2.0
    states[1, :, 0] = 0.0
    states[2, :, 0] = np.arange(1, 9, dtype=np.float32)
    velocity = np.array([[8.0] * 8, [0.0] * 8, [4.0] * 8])
    valid = np.ones(3, dtype=bool)
    collision = predicted_candidate_collisions(states, valid, actors)
    np.testing.assert_array_equal(collision, [True, False, True])
    assert select_safe_slowdown(valid, collision, velocity) == 1
    assert select_safe_slowdown(valid, [False, False, True], velocity) == 0


def test_low_confidence_detection_does_not_trigger_gate():
    boxes = np.array([[8.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1]], dtype=np.float32)
    actors = extrapolate_detected_actors(boxes, score_threshold=0.3)
    assert actors.num_actors == 0


def test_controller_guard_excludes_late_braking_profile_and_preserves_raw_brake():
    velocity = np.asarray(
        [
            [4.0] * 8,  # raw
            [5.0] + [3.5] * 7,  # shorter distance, but fast first tick
            [3.5] * 8,  # genuine slowdown
        ],
        dtype=np.float32,
    )
    valid = np.ones(3, dtype=bool)
    collision = np.asarray([True, False, False])
    assert select_safe_slowdown(valid, collision, velocity) == 1
    assert (
        select_safe_slowdown(
            valid,
            collision,
            velocity,
            current_speed_mps=4.0,
            raw_target_speed_mps=4.0,
        )
        == 2
    )
    assert (
        select_safe_slowdown(
            valid,
            collision,
            velocity,
            current_speed_mps=4.0,
            raw_target_speed_mps=0.0,
        )
        == 0
    )
    with pytest.raises(ValueError, match="provided together"):
        select_safe_slowdown(valid, collision, velocity, current_speed_mps=4.0)


def test_predicted_actor_audit_runs_on_cached_candidates(tmp_path, monkeypatch):
    from scripts.p4.eval_b3a_predicted_actor_gate import main

    feature_dir = tmp_path / "features"
    detection_dir = tmp_path / "detections"
    feature_dir.mkdir()
    detection_dir.mkdir()
    keys = np.asarray(["scenario__route__0001", "scenario__route__0002"])
    states = np.zeros((2, 2, 8, 6), dtype=np.float32)
    states[..., 3] = 1.0
    states[:, 0, :, 0] = np.arange(1, 9) * 2.0
    velocity = np.tile(np.asarray([[8.0] * 8, [0.0] * 8]), (2, 1, 1))
    np.savez_compressed(
        feature_dir / "velocity_features_000000_000002.npz",
        keys=keys,
        candidate_states=states,
        candidate_valid=np.ones((2, 2), dtype=bool),
        candidate_velocity=velocity,
        collision=np.asarray([[True, False], [False, False]]),
        imitation_error=np.asarray([[0.1, 2.0], [0.1, 2.0]]),
        current_speed=np.asarray([8.0, 8.0]),
        raw_target_speed=np.asarray([8.0, 8.0]),
    )
    boxes = np.asarray(
        [
            [[8.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.9]],
            [[8.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        detection_dir / "predicted_boxes_000000_000002.npz",
        keys=keys,
        boxes=boxes,
        source_checkpoint=np.asarray("fake.pth"),
    )
    output = tmp_path / "gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_b3a_predicted_actor_gate.py",
            "--feature-cache-dir",
            str(feature_dir),
            "--predicted-box-dir",
            str(detection_dir),
            "--out",
            str(output),
            "--score-thresholds",
            "0.3",
            "--controller-speed-guard",
        ],
    )
    main()
    result = json.loads(output.read_text())["threshold_sweep"][0]
    assert result["raw_gt_collision_rate"] == 0.5
    assert result["oracle_gt_collision_rate"] == 0.0
    assert result["selected_gt_collision_rate"] == 0.0
    assert result["switch_rate"] == 0.5
    assert json.loads(output.read_text())["controller_speed_guard"] is True


def test_closed_loop_gate_keeps_path_and_selects_only_safe_slowdown():
    from lead.inference.config_open_loop import OpenLoopConfig
    from lead.inference.open_loop_inference import OpenLoopInference
    from lead.tfv6.tfv6 import Prediction
    from lead.training.config_training import TrainingConfig

    config = TrainingConfig()
    config.use_planning_decoder = True
    config.predict_temporal_spatial_waypoints = False
    config.use_navsim_data = False
    config.route_selection_mode = "confidence"
    config.route_predicted_actor_velocity_gate = True
    inference = OpenLoopInference.__new__(OpenLoopInference)
    inference.config_training = config
    inference.config_open_loop = OpenLoopConfig(raise_error_on_missing_key=False)
    inference.device = torch.device("cpu")
    inference.predicted_actor_velocity_vocabulary = torch.zeros((1, 8))

    logits = torch.full((1, len(config.target_speed_classes)), -10.0)
    logits[:, 2] = 10.0  # 8 m/s raw target
    route = torch.stack(
        (torch.arange(1, 11, dtype=torch.float32), torch.zeros(10)), dim=-1
    )[None]
    prediction = Prediction(
        pred_future_waypoints=None,
        pred_target_speed_distribution=logits,
        pred_target_speed_scalar=torch.tensor([8.0]),
        pred_route=route,
        pred_semantic=None,
        pred_bev_semantic=None,
        pred_depth=None,
        pred_bounding_box=None,
        pred_radar_features=None,
        pred_radar_predictions=None,
        pred_bounding_box_navsim=None,
        pred_bev_semantic_navsim=None,
        pred_headings=None,
    )
    box = np.asarray([[12.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.9]])
    image_box = box.copy()
    image_box[:, :2] -= [config.min_x_meter, config.min_y_meter]
    image_box[:, :4] *= config.pixels_per_meter
    prediction.pred_bounding_box = SimpleNamespace(
        pred_bounding_box_image_system=image_box[None].astype(np.float32)
    )
    result = inference.ensemble_planning_decoder(
        [prediction], {"speed": torch.tensor([3.0])}
    )
    torch.testing.assert_close(result[0], route)
    torch.testing.assert_close(result[13], torch.tensor([1]))
    torch.testing.assert_close(result[14], torch.tensor([True]))
    torch.testing.assert_close(result[2], torch.tensor([[3.0]]))
    torch.testing.assert_close(result[11], torch.tensor([1.0]))
    torch.testing.assert_close(result[12], torch.tensor([0.0]))

    prediction.pred_bounding_box.pred_bounding_box_image_system[0, 0, 8] = 0.1
    no_actor = inference.ensemble_planning_decoder(
        [prediction], {"speed": torch.tensor([3.0])}
    )
    torch.testing.assert_close(no_actor[13], torch.tensor([0]))
    torch.testing.assert_close(no_actor[2], no_actor[6])
