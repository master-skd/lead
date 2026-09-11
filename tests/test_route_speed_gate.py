import pytest
import torch
from torch import nn

from lead.tfv6.route_speed_gate import (
    apply_collision_speed_gate,
    collision_risk_to_speed_factor,
)
from lead.training.config_training import TrainingConfig


def test_speed_factor_is_piecewise_linear():
    risk = torch.tensor([0.1, 0.5, 0.7, 0.9, 1.0])
    factor = collision_risk_to_speed_factor(risk, 0.5, 0.9)
    torch.testing.assert_close(factor, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]))


def test_minimum_factor_is_respected():
    factor = collision_risk_to_speed_factor(torch.tensor([0.9]), 0.5, 0.9, 0.25)
    torch.testing.assert_close(factor, torch.tensor([0.25]))


def test_apply_gate_only_changes_speed_not_any_route_state():
    config = TrainingConfig()
    config.route_speed_gate_low_threshold = 0.5
    config.route_speed_gate_high_threshold = 0.9
    config.route_speed_gate_minimum_factor = 0.0
    speed = torch.tensor([[8.0], [8.0], [8.0]])
    gated, factor = apply_collision_speed_gate(speed, torch.tensor([0.2, 0.7, 0.95]), config)
    torch.testing.assert_close(factor, torch.tensor([1.0, 0.5, 0.0]))
    torch.testing.assert_close(gated, torch.tensor([[8.0], [4.0], [0.0]]))


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError):
        collision_risk_to_speed_factor(torch.tensor([0.5]), 0.9, 0.5)
    with pytest.raises(ValueError):
        collision_risk_to_speed_factor(torch.tensor([0.5]), 0.1, 0.9, 1.1)


def test_post_ensemble_gate_changes_pid_speed_but_not_route():
    from lead.inference.config_open_loop import OpenLoopConfig
    from lead.inference.open_loop_inference import OpenLoopInference
    from lead.tfv6.tfv6 import Prediction

    config = TrainingConfig()
    config.use_planning_decoder = True
    config.predict_temporal_spatial_waypoints = False
    config.use_navsim_data = False
    config.route_speed_safety_gate = True
    config.route_speed_gate_low_threshold = 0.5
    config.route_speed_gate_high_threshold = 0.9
    config.route_speed_gate_minimum_factor = 0.0
    open_config = OpenLoopConfig(raise_error_on_missing_key=False)
    inference = OpenLoopInference.__new__(OpenLoopInference)
    inference.config_training = config
    inference.config_open_loop = open_config
    inference.device = torch.device("cpu")

    logits = torch.full((1, len(config.target_speed_classes)), -10.0)
    logits[:, 2] = 10.0  # 8 m/s class
    route = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2)
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
        pred_route_collision_risk=torch.tensor([0.7]),
    )
    result = inference.ensemble_planning_decoder([prediction])
    gated_speed, raw_speed, factor = result[2], result[6], result[7]
    torch.testing.assert_close(result[0], route)
    torch.testing.assert_close(factor, torch.tensor([0.5]))
    torch.testing.assert_close(gated_speed, raw_speed * 0.5)


class _FeatureLogitHead(nn.Module):
    def forward(self, route_features, current_speed, target_speed):
        del current_speed, target_speed
        return route_features[..., 0]


def test_future_gate_uses_selected_arm_and_never_changes_route():
    from lead.inference.config_open_loop import OpenLoopConfig
    from lead.inference.open_loop_inference import OpenLoopInference
    from lead.tfv6.tfv6 import Prediction

    config = TrainingConfig()
    config.use_planning_decoder = True
    config.predict_temporal_spatial_waypoints = False
    config.use_navsim_data = False
    config.route_speed_safety_gate = False
    config.route_future_safety_gate = True
    config.route_future_gate_low_threshold = 0.5
    config.route_future_gate_high_threshold = 0.9
    config.route_future_gate_minimum_factor = 0.0
    inference = OpenLoopInference.__new__(OpenLoopInference)
    inference.config_training = config
    inference.config_open_loop = OpenLoopConfig(raise_error_on_missing_key=False)
    inference.device = torch.device("cpu")
    inference.route_safety_head = _FeatureLogitHead()

    logits = torch.full((1, len(config.target_speed_classes)), -10.0)
    logits[:, 2] = 10.0
    route = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2)
    # Arm 0 is safe (sigmoid(-4)); selected arm 1 has risk 0.7.
    route_features = torch.tensor([[[-4.0], [0.8472979]]])
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
        pred_route_features=route_features,
        pred_route_selected_idx=torch.tensor([1]),
    )
    result = inference.ensemble_planning_decoder(
        [prediction], {"speed": torch.tensor([3.0])},
    )
    torch.testing.assert_close(result[0], route)
    torch.testing.assert_close(result[8], torch.tensor([0.7]))
    torch.testing.assert_close(result[10], torch.tensor([0.5]))
    torch.testing.assert_close(result[2], result[6] * 0.5)


def test_combined_gate_takes_more_conservative_factor():
    from lead.inference.config_open_loop import OpenLoopConfig
    from lead.inference.open_loop_inference import OpenLoopInference
    from lead.tfv6.tfv6 import Prediction

    config = TrainingConfig()
    config.use_planning_decoder = True
    config.predict_temporal_spatial_waypoints = False
    config.use_navsim_data = False
    config.route_speed_safety_gate = True
    config.route_speed_gate_low_threshold = 0.5
    config.route_speed_gate_high_threshold = 0.9
    config.route_future_safety_gate = True
    config.route_future_gate_low_threshold = 0.5
    config.route_future_gate_high_threshold = 0.9
    inference = OpenLoopInference.__new__(OpenLoopInference)
    inference.config_training = config
    inference.config_open_loop = OpenLoopConfig(raise_error_on_missing_key=False)
    inference.device = torch.device("cpu")
    inference.route_safety_head = _FeatureLogitHead()

    logits = torch.full((1, len(config.target_speed_classes)), -10.0)
    logits[:, 2] = 10.0
    prediction = Prediction(
        pred_future_waypoints=None,
        pred_target_speed_distribution=logits,
        pred_target_speed_scalar=torch.tensor([8.0]),
        pred_route=torch.zeros(1, 10, 2),
        pred_semantic=None,
        pred_bev_semantic=None,
        pred_depth=None,
        pred_bounding_box=None,
        pred_radar_features=None,
        pred_radar_predictions=None,
        pred_bounding_box_navsim=None,
        pred_bev_semantic_navsim=None,
        pred_headings=None,
        pred_route_collision_risk=torch.tensor([0.8]),  # current factor 0.25
        pred_route_features=torch.tensor([[[0.0], [0.8472979]]]),
        pred_route_selected_idx=torch.tensor([1]),  # future factor 0.5
    )
    result = inference.ensemble_planning_decoder(
        [prediction], {"speed": torch.tensor([3.0])},
    )
    torch.testing.assert_close(result[9], torch.tensor([0.25]))
    torch.testing.assert_close(result[10], torch.tensor([0.5]))
    torch.testing.assert_close(result[7], torch.tensor([0.25]))
    torch.testing.assert_close(result[2], result[6] * 0.25)
