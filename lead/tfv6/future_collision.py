"""Time-aligned ego/actor OBB collision labels for B2d."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lead.data_loader.future_actor_cache import FUTURE_TIMES_S, FutureActorFrame


@dataclass(frozen=True)
class FutureCollisionLabel:
    collision: bool
    ttc_s: float
    collision_step: int
    actor_id: int
    actor_class_id: int
    per_step_collision: np.ndarray


def speed_profile_distances(
    current_speed_mps: float,
    target_speed_mps: float,
    *,
    times_s: np.ndarray = FUTURE_TIMES_S,
    max_accel_mps2: float = 1.89,
    max_decel_mps2: float = 4.95,
) -> np.ndarray:
    """Distance travelled while accelerating once toward a target speed."""

    v0 = max(0.0, float(current_speed_mps))
    target = max(0.0, float(target_speed_mps))
    delta = target - v0
    accel = np.clip(
        delta / max(float(times_s[-1]), 1e-6), -max_decel_mps2, max_accel_mps2,
    )
    if abs(accel) < 1e-8:
        return (v0 * times_s).astype(np.float32)
    time_to_target = max(0.0, delta / accel)
    accelerating_time = np.minimum(times_s, time_to_target)
    distance = v0 * accelerating_time + 0.5 * accel * accelerating_time**2
    distance += target * np.maximum(times_s - time_to_target, 0.0)
    return np.maximum.accumulate(distance).astype(np.float32)


def interpolate_route_by_distance(
    route: np.ndarray, distances_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate route centers and tangent yaws at requested arc lengths."""

    route = np.asarray(route, dtype=np.float32).reshape(-1, 2)
    distances_m = np.asarray(distances_m, dtype=np.float32).reshape(-1)
    if len(route) == 0:
        route = np.zeros((1, 2), dtype=np.float32)
    if np.linalg.norm(route[0]) > 1e-4:
        route = np.concatenate([np.zeros((1, 2), dtype=np.float32), route], axis=0)
    segment = np.diff(route, axis=0)
    segment_length = np.linalg.norm(segment, axis=1)
    keep = np.concatenate([[True], segment_length > 1e-5])
    route = route[keep]
    if len(route) == 1:
        return np.repeat(route, len(distances_m), axis=0), np.zeros(len(distances_m), dtype=np.float32)
    segment = np.diff(route, axis=0)
    segment_length = np.linalg.norm(segment, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment_length)])
    clipped = np.clip(distances_m, 0.0, arc[-1])
    x = np.interp(clipped, arc, route[:, 0])
    y = np.interp(clipped, arc, route[:, 1])
    segment_index = np.clip(np.searchsorted(arc, clipped, side="right") - 1, 0, len(segment) - 1)
    tangent = segment[segment_index]
    yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
    return np.stack([x, y], axis=1).astype(np.float32), yaw.astype(np.float32)


def _obb_overlap(
    ego_positions: np.ndarray,
    ego_yaws: np.ndarray,
    ego_half_extent: np.ndarray,
    actor_positions: np.ndarray,
    actor_yaws: np.ndarray,
    actor_half_extents: np.ndarray,
) -> np.ndarray:
    """Vectorized 2-D separating-axis test, returning [A,T]."""

    # Actor-major arrays simplify broadcasting over A,T and four SAT axes.
    delta = actor_positions - ego_positions[None, :, :]
    ego_forward = np.stack([np.cos(ego_yaws), np.sin(ego_yaws)], axis=-1)
    ego_side = np.stack([-np.sin(ego_yaws), np.cos(ego_yaws)], axis=-1)
    actor_forward = np.stack([np.cos(actor_yaws), np.sin(actor_yaws)], axis=-1)
    actor_side = np.stack([-np.sin(actor_yaws), np.cos(actor_yaws)], axis=-1)
    axes = (
        np.broadcast_to(ego_forward, delta.shape),
        np.broadcast_to(ego_side, delta.shape),
        actor_forward,
        actor_side,
    )
    ego_l, ego_w = float(ego_half_extent[0]), float(ego_half_extent[1])
    actor_l = actor_half_extents[:, None, 0]
    actor_w = actor_half_extents[:, None, 1]
    overlap = np.ones(delta.shape[:2], dtype=bool)
    for axis in axes:
        center_projection = np.abs(np.sum(delta * axis, axis=-1))
        ego_radius = ego_l * np.abs(np.sum(ego_forward[None] * axis, axis=-1))
        ego_radius += ego_w * np.abs(np.sum(ego_side[None] * axis, axis=-1))
        actor_radius = actor_l * np.abs(np.sum(actor_forward * axis, axis=-1))
        actor_radius += actor_w * np.abs(np.sum(actor_side * axis, axis=-1))
        overlap &= center_projection <= ego_radius + actor_radius
    return overlap


def future_collision_label(
    ego_positions: np.ndarray,
    ego_yaws: np.ndarray,
    actors: FutureActorFrame,
    *,
    times_s: np.ndarray = FUTURE_TIMES_S,
    safety_margin_m: float = 0.2,
    vertical_margin_m: float = 0.2,
    include_class_ids: tuple[int, ...] | None = None,
) -> FutureCollisionLabel:
    """Return the first time-aligned actor collision and its identity."""

    steps = len(times_s)
    no_collision = FutureCollisionLabel(
        False, float("inf"), -1, -1, 0, np.zeros(steps, dtype=bool),
    )
    if actors.num_actors == 0:
        return no_collision
    ego_positions = np.asarray(ego_positions, dtype=np.float32).reshape(steps, 2)
    ego_yaws = np.asarray(ego_yaws, dtype=np.float32).reshape(steps)
    ego_extent = actors.ego_extent.copy()
    ego_extent[:2] += safety_margin_m
    overlap = _obb_overlap(
        ego_positions, ego_yaws, ego_extent, actors.positions, actors.yaws, actors.extents,
    )
    vertical = np.abs(actors.z) <= (
        float(actors.ego_extent[2]) + actors.extents[:, 2] + vertical_margin_m
    )
    overlap &= actors.valid & vertical[:, None]
    if include_class_ids is not None:
        overlap &= np.isin(actors.class_ids, include_class_ids)[:, None]
    per_step = overlap.any(axis=0)
    if not per_step.any():
        return FutureCollisionLabel(False, float("inf"), -1, -1, 0, per_step)
    step = int(np.flatnonzero(per_step)[0])
    actor_index = int(np.flatnonzero(overlap[:, step])[0])
    return FutureCollisionLabel(
        True,
        float(times_s[step]),
        step,
        int(actors.actor_ids[actor_index]),
        int(actors.class_ids[actor_index]),
        per_step,
    )


def route_future_collision_label(
    route: np.ndarray,
    current_speed_mps: float,
    target_speed_mps: float,
    actors: FutureActorFrame,
    **kwargs,
) -> FutureCollisionLabel:
    distances = speed_profile_distances(current_speed_mps, target_speed_mps)
    positions, yaws = interpolate_route_by_distance(route, distances)
    return future_collision_label(positions, yaws, actors, **kwargs)
