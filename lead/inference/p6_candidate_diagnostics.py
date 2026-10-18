"""Read-only P6 candidate/selection/control trace for short B2D audits."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch


UTILITY_WEIGHTS = {
    "collision_free": 2.0,
    "ttc": 1.0,
    "drivable": 1.0,
    "task": 1.0,
    "progress": 0.5,
    "comfort": 0.25,
    "imitation": 0.5,
}


def _list(value: Any) -> list | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu()
        if value.ndim and value.shape[0] == 1:
            value = value[0]
        return value.tolist()
    return np.asarray(value).tolist()


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().reshape(-1)[0].cpu())
    return float(value)


def driving_lane_fractions(
    routes: torch.Tensor,
    ego_x: float,
    ego_y: float,
    ego_yaw_deg: float,
    is_driving_lane: Callable[[float, float], bool],
) -> list[float]:
    """Sample 8 positions per arm against CARLA's actual driving-lane map.

    This is a geometric diagnostic, not a safety oracle: it does not check
    traffic, route direction, or whether an adjacent lane is permissible.
    """
    paths = np.asarray(_list(routes), dtype=np.float32)
    if paths.ndim != 3 or paths.shape[-1] != 2:
        raise ValueError("routes must be [K,P,2] with batch size one")
    angle = math.radians(ego_yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    indices = np.unique(np.linspace(0, paths.shape[1] - 1, 8).round().astype(int))
    fractions = []
    for path in paths:
        on_lane = []
        for x, y in path[indices]:
            world_x = ego_x + cosine * float(x) - sine * float(y)
            world_y = ego_y + sine * float(x) + cosine * float(y)
            on_lane.append(bool(is_driving_lane(world_x, world_y)))
        fractions.append(float(np.mean(on_lane)))
    return fractions


def build_p6_candidate_record(
    *,
    step: int,
    route_id: str,
    game_time_s: float,
    ego_world: tuple[float, float, float, float],
    current_speed_mps: float,
    data: dict,
    prediction: Any,
    closed_loop_prediction: Any,
    control: Any,
    stuck_detector: int,
    force_move_remaining: int,
    driving_lane_fraction: list[float] | None = None,
) -> dict[str, Any]:
    """Serialize one model decision without changing any model/control tensor."""
    if prediction.pred_route_multimodal is None or prediction.pred_route_selected_idx is None:
        raise ValueError("P6 multi-route prediction is required")
    routes = _list(prediction.pred_route_multimodal)
    selected = int(prediction.pred_route_selected_idx.detach().reshape(-1)[0].cpu())
    logits = data["joint_score_logits"]
    scores = {name: torch.sigmoid(value.detach().float())[0].cpu().tolist()
              for name, value in logits.items()}
    utility = [sum(weight * scores[name][arm] for name, weight in UTILITY_WEIGHTS.items())
               for arm in range(len(routes))]
    anchors = _list(data["anchor"])
    valid = [bool(anchor[3] > 0.5) for anchor in anchors]
    if not 0 <= selected < len(routes) or not valid[selected]:
        raise ValueError("selected P6 route is not a valid candidate")
    arm_speeds = _list(prediction.pred_target_speed_multimodal_scalar)
    if arm_speeds is None:
        arm_speeds = [float(_scalar(prediction.pred_target_speed_scalar))] * len(routes)
    intent = prediction.pred_visual_intent
    intent_coverage = None
    if intent is not None:
        intent_coverage = float((torch.sigmoid(intent.detach().float()) > 0.5).float().mean().cpu())
    return {
        "schema_version": 1,
        "step": int(step),
        "route_id": str(route_id),
        "game_time_s": float(game_time_s),
        "ego_world": {"x": float(ego_world[0]), "y": float(ego_world[1]),
                      "z": float(ego_world[2]), "yaw_deg": float(ego_world[3])},
        "current_speed_mps": float(current_speed_mps),
        "navigation": {
            "target_point_ego_m": _list(data.get("target_point")),
            "next_target_point_ego_m": _list(data.get("target_point_next")),
            "command_one_hot": _list(data.get("command")),
        },
        "intent": {"anchors": anchors, "valid": valid,
                   "coverage_above_0_5": intent_coverage},
        "candidates": {
            "routes_ego_m": routes,
            "states_ego_2s": _list(data.get("joint_candidate_states")),
            "target_speeds_mps": arm_speeds,
            "score_probabilities": scores,
            "utility": utility,
            "driving_lane_fraction": driving_lane_fraction,
        },
        "selection": {"index": selected, "valid_count": sum(valid),
                      "executed_target_speed_mps": _scalar(closed_loop_prediction.pred_target_speed_scalar)},
        "controller": {
            "route_steer": _scalar(closed_loop_prediction.route_steer),
            "target_speed_throttle": _scalar(closed_loop_prediction.target_speed_throttle),
            "target_speed_brake": _scalar(closed_loop_prediction.target_speed_brake),
            "stuck_detector": int(stuck_detector),
            "force_move_remaining": int(force_move_remaining),
            "commanded": {"steer": float(control.steer), "throttle": float(control.throttle),
                          "brake": float(control.brake)},
        },
    }
