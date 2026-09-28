"""Non-executing local-Path counterfactuals for Bench2Drive diagnostics."""

from __future__ import annotations

import base64
import zlib
from copy import deepcopy

import numpy as np

from lead.tfv6.future_collision import (
    future_collision_label,
    interpolate_route_by_distance,
    speed_profile_distances,
)


DEFAULT_OFFSETS_M = (-1.5, -0.75, 0.75, 1.5)


def encode_intent_probability_map(probability: np.ndarray) -> dict:
    """Compact, lossily quantized BEV map for triggered shadow diagnostics only."""

    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("intent probability map must be finite [H,W]")
    quantized = np.rint(np.clip(probability, 0.0, 1.0) * 255).astype(np.uint8)
    return {
        "shape": list(quantized.shape),
        "encoding": "uint8_zlib_base64",
        "data": base64.b64encode(zlib.compress(quantized.tobytes())).decode("ascii"),
        "fraction_above_0_5": float((probability > 0.5).mean()),
    }


def local_path_variants(
    route: np.ndarray,
    offsets_m: tuple[float, ...] = DEFAULT_OFFSETS_M,
    ramp_m: float = 8.0,
) -> np.ndarray:
    """Smooth ego-anchored lateral offsets of one confidence-selected Path."""

    route = np.asarray(route, dtype=np.float32)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2 or ramp_m <= 0:
        raise ValueError("route must be [N>=2,2] and ramp_m positive")
    path = np.concatenate((np.zeros((1, 2), dtype=np.float32), route))
    segment = np.diff(path, axis=0)
    arc = np.cumsum(np.linalg.norm(segment, axis=1))
    tangent = np.zeros_like(route)
    tangent[0] = segment[0]
    tangent[-1] = segment[-1]
    if len(route) > 2:
        tangent[1:-1] = segment[:-2] + segment[1:-1]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    u = np.clip(arc / float(ramp_m), 0.0, 1.0)
    ramp = u * u * (3.0 - 2.0 * u)
    offsets = np.asarray(offsets_m, dtype=np.float32)
    return route[None] + offsets[:, None, None] * ramp[None, :, None] * normal[None]


def score_local_paths(
    routes: np.ndarray,
    corridor_points: np.ndarray,
    actors,
    lateral_controller,
    *,
    current_speed_mps: float,
    raw_target_speed_mps: float,
    baseline_steer: float,
    offsets_m: tuple[float, ...] = DEFAULT_OFFSETS_M,
    safety_margin_m: float = 0.2,
    max_corridor_mean: float = 0.1,
    max_corridor_point: float = 0.25,
    max_steer_delta: float = 0.25,
    min_effective_steer_delta: float = 0.03,
    min_speed_mps: float = 1.0,
    sensor_agent_steer_correction: bool = False,
) -> dict:
    """Score raw+local paths without mutating the real PID controller."""

    paths = np.asarray(routes, dtype=np.float32)
    costs = np.asarray(corridor_points, dtype=np.float32)
    if paths.ndim != 3 or costs.ndim != 2 or paths.shape[1:] != (costs.shape[1], 2):
        raise ValueError("routes/corridor_points must be [K,N,2]/[K,N]")
    if costs.shape[0] != len(paths) or len(paths) != len(offsets_m) + 1:
        raise ValueError("expected raw plus one path per offset")
    distances = speed_profile_distances(current_speed_mps, raw_target_speed_mps)
    candidates = []
    for index, (route, cost) in enumerate(zip(paths, costs, strict=True)):
        xy, yaw = interpolate_route_by_distance(route, distances, extrapolate=True)
        collision = future_collision_label(
            xy, yaw, actors, safety_margin_m=safety_margin_m,
            include_class_ids=(1, 2),
        )
        steer = deepcopy(lateral_controller).step(
            route,
            float(current_speed_mps),
            0.0,
            0.0,
            sensor_agent_steer_correction=sensor_agent_steer_correction,
        )
        steer_delta = float(steer - baseline_steer)
        candidates.append({
            "index": index,
            "offset_m": 0.0 if index == 0 else float(offsets_m[index - 1]),
            "predicted_collision": bool(collision.collision),
            "predicted_ttc_s": float(collision.ttc_s) if collision.collision else None,
            "predicted_collision_actor_id": int(collision.actor_id) if collision.collision else None,
            "corridor_mean": float(cost.mean()),
            "corridor_max": float(cost.max()),
            "corridor_per_point": cost.astype(float).tolist(),
            "route_xy_m": route.astype(float).tolist(),
            "predicted_ego_xy_m": xy.astype(float).tolist(),
            "predicted_ego_yaw_rad": yaw.astype(float).tolist(),
            "counterfactual_steer": float(steer),
            "steer_delta_from_baseline": steer_delta,
            "pid_steer_effective": bool(
                abs(steer_delta) >= min_effective_steer_delta and abs(steer) < 0.98
            ),
        })
    raw_collision = candidates[0]["predicted_collision"]
    eligible = [
        candidate for candidate in candidates[1:]
        if not candidate["predicted_collision"]
        and candidate["corridor_mean"] <= max_corridor_mean
        and candidate["corridor_max"] <= max_corridor_point
        and abs(candidate["steer_delta_from_baseline"]) <= max_steer_delta
        and candidate["pid_steer_effective"]
    ]
    would_select = None
    if raw_collision and current_speed_mps >= min_speed_mps and eligible:
        would_select = min(eligible, key=lambda item: abs(item["offset_m"]))["index"]
    return {
        "schema_version": 2,
        "shadow_only": True,
        "candidate_evaluated": True,
        "current_speed_mps": float(current_speed_mps),
        "raw_target_speed_mps": float(raw_target_speed_mps),
        "predicted_actor_count": int(actors.num_actors),
        "predicted_actors": [
            {
                "id": int(actors.actor_ids[i]),
                "class_id": int(actors.class_ids[i]),
                "xy_m": actors.positions[i].astype(float).tolist(),
                "yaw_rad": actors.yaws[i].astype(float).tolist(),
                "half_extent_m": actors.extents[i, :2].astype(float).tolist(),
            }
            for i in range(actors.num_actors)
        ],
        "baseline_steer": float(baseline_steer),
        "pid_snapshot_raw_steer": candidates[0]["counterfactual_steer"],
        "raw_predicted_collision": bool(raw_collision),
        "would_select_index": would_select,
        "would_switch": would_select is not None,
        "candidates": candidates,
    }
