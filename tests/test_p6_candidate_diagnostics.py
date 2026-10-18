import json
from types import SimpleNamespace as Obj

import pytest
import torch

from lead.inference.p6_candidate_diagnostics import (
    build_p6_candidate_record,
    driving_lane_fractions,
)
from scripts.p4.summarize_p6_candidate_trace import summarize


def _record(*, multimodal_speed=False):
    routes = torch.tensor([[[[1.0, 0.0], [2.0, 0.0]],
                            [[1.0, 1.0], [2.0, 2.0]]]])
    scores = {name: torch.tensor([[0.0, 1.0]]) for name in (
        "collision_free", "ttc", "drivable", "task", "progress", "comfort", "imitation"
    )}
    data = {
        "anchor": torch.tensor([[[0.0, 1.0, 1.0, 1.0], [0.7, 0.7, 1.0, 1.0]]]),
        "joint_score_logits": scores,
        "joint_candidate_states": torch.zeros(1, 2, 8, 6),
        "target_point": torch.tensor([[3.0, 0.0]]),
        "target_point_next": torch.tensor([[6.0, 0.0]]),
        "command": torch.tensor([[1.0, 0, 0, 0, 0, 0]]),
    }
    prediction = Obj(
        pred_route_multimodal=routes,
        pred_route_selected_idx=torch.tensor([1]),
        pred_target_speed_multimodal_scalar=(
            torch.tensor([[3.0, 4.0]]) if multimodal_speed else None
        ),
        pred_target_speed_scalar=torch.tensor([4.0]),
        pred_visual_intent=torch.zeros(1, 1, 2, 2),
    )
    executed = Obj(pred_target_speed_scalar=torch.tensor([[4.0]]),
                   route_steer=0.1, target_speed_throttle=0.4,
                   target_speed_brake=0.0)
    record = build_p6_candidate_record(
        step=5, route_id="3904", game_time_s=0.25,
        ego_world=(10.0, 20.0, 0.2, 90.0), current_speed_mps=1.0,
        data=data, prediction=prediction, closed_loop_prediction=executed,
        control=Obj(steer=0.1, throttle=0.4, brake=0.0),
        stuck_detector=2, force_move_remaining=0,
        driving_lane_fraction=[1.0, 0.25],
    )
    assert prediction.pred_route_multimodal is routes
    return record


def test_trace_serializes_k_candidates_scores_and_selected_control():
    record = _record()
    assert record["selection"]["index"] == 1
    assert record["candidates"]["target_speeds_mps"] == [4.0, 4.0]
    assert len(record["candidates"]["routes_ego_m"]) == 2
    assert record["candidates"]["utility"][1] > record["candidates"]["utility"][0]
    assert record["controller"]["commanded"]["throttle"] == 0.4
    json.dumps(record, allow_nan=False)


def test_trace_pairs_multimodal_speeds_and_checks_valid_arm():
    record = _record(multimodal_speed=True)
    assert record["candidates"]["target_speeds_mps"] == [3.0, 4.0]


def test_lane_fraction_uses_ego_pose_rotation():
    routes = torch.tensor([[[[1.0, 0.0], [2.0, 0.0]],
                            [[1.0, 2.0], [2.0, 2.0]]]])
    observed = []

    def lane(x, y):
        observed.append((x, y))
        return x >= 9.0

    fractions = driving_lane_fractions(routes, 10.0, 20.0, 90.0, lane)
    assert fractions == [1.0, 0.0]
    assert observed[0] == pytest.approx((10.0, 21.0))


def test_summary_reports_selected_vs_available_lane_without_oracle_claim():
    record = _record()
    result = summarize([record])
    assert result["map_sampled_frames"] == 1
    assert result["map_lane_miss_frames"] == [(5, 1.0, 0.25)]
    assert result["stopped_frames"] == 0
