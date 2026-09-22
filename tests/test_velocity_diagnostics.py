import json
from types import SimpleNamespace

import pytest
import torch

from lead.inference.velocity_diagnostics import build_velocity_diagnostic_record


def _prediction():
    return SimpleNamespace(
        velocity_candidate_risk=torch.tensor([[0.8, 0.2, 0.1]]),
        velocity_candidate_valid=torch.tensor([[True, True, False]]),
        velocity_candidate_preference=torch.tensor([[1.0, 3.0, 5.0]]),
        velocity_candidate_profiles=torch.tensor(
            [[[4.0, 4.0], [3.0, 2.0], [5.0, 6.0]]]
        ),
        velocity_selected_index=torch.tensor([1]),
        velocity_switched=torch.tensor([True]),
        velocity_fallback=torch.tensor([False]),
        velocity_raw_risk=torch.tensor([0.8]),
        velocity_selected_risk=torch.tensor([0.2]),
        raw_target_speed_scalar=torch.tensor([[4.0]]),
        pred_target_speed_scalar=torch.tensor([[2.5]]),
        velocity_selected_profile=torch.tensor([[3.0, 2.0]]),
        pred_trajectory=torch.tensor([[[0.75, 0.0], [1.25, 0.0]]]),
        route_steer=0.1,
        target_speed_throttle=0.2,
        target_speed_brake=0.0,
    )


def test_build_velocity_diagnostic_record_retains_candidate_decision():
    record = build_velocity_diagnostic_record(
        _prediction(),
        step=42,
        current_speed_mps=3.5,
        safe_threshold=0.5,
        selection_mode="profile_select",
        final_steer=0.1,
        final_throttle=0.3,
        final_brake=0.0,
        stuck_detector=7,
        force_move_remaining=0,
    )

    assert record is not None
    assert record["step"] == 42
    assert record["selected_index"] == 1
    assert record["valid_candidate_count"] == 2
    assert record["safe_candidate_count"] == 1
    assert record["selected_preference_logit"] == 3.0
    assert record["minimum_valid_collision_risk"] == pytest.approx(0.2)
    assert record["selected_profile_mps"] == [3.0, 2.0]
    assert record["candidate_valid"] == [True, True, False]
    assert record["controller"]["final_throttle"] == 0.3
    json.dumps(record, allow_nan=False)


def test_build_velocity_diagnostic_record_skips_disabled_scorer():
    prediction = _prediction()
    prediction.velocity_candidate_risk = None

    assert (
        build_velocity_diagnostic_record(
            prediction,
            step=1,
            current_speed_mps=0.0,
            safe_threshold=0.5,
            selection_mode="profile_select",
            final_steer=0.0,
            final_throttle=0.0,
            final_brake=1.0,
            stuck_detector=0,
            force_move_remaining=0,
        )
        is None
    )
