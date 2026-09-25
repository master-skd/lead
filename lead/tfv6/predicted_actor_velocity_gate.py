"""Counterfactual velocity filtering using detected actors and constant velocity."""

from __future__ import annotations

import numpy as np

from lead.data_loader.future_actor_cache import FUTURE_TIMES_S, FutureActorFrame
from lead.tfv6.future_collision import future_collision_label


def extrapolate_detected_actors(
    boxes: np.ndarray,
    *,
    score_threshold: float = 0.3,
    nms_iou_threshold: float | None = None,
    ego_extent: tuple[float, float, float] = (2.45, 0.95, 0.75),
    max_speed_mps: float = 25.0,
) -> FutureActorFrame:
    """Convert CenterNet [x,y,half_l,half_w,yaw,speed,...,class,score]."""

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 9)
    keep = np.isfinite(boxes).all(axis=1)
    keep &= boxes[:, 8] >= score_threshold
    keep &= boxes[:, 2] > 0
    keep &= boxes[:, 3] > 0
    keep &= np.isin(boxes[:, 7].astype(np.int32), (0, 1, 4))
    boxes = boxes[keep]
    if nms_iou_threshold is not None and len(boxes) > 1:
        from lead.inference.inference_utils import non_maximum_suppression

        boxes = non_maximum_suppression([boxes], nms_iou_threshold)
    count = len(boxes)
    steps = len(FUTURE_TIMES_S)
    if count == 0:
        return FutureActorFrame(
            positions=np.empty((0, steps, 2), dtype=np.float32),
            yaws=np.empty((0, steps), dtype=np.float32),
            extents=np.empty((0, 3), dtype=np.float32),
            z=np.empty(0, dtype=np.float32),
            valid=np.empty((0, steps), dtype=bool),
            actor_ids=np.empty(0, dtype=np.int64),
            class_ids=np.empty(0, dtype=np.uint8),
            ego_extent=np.asarray(ego_extent, dtype=np.float32),
        )
    yaw = boxes[:, 4]
    speed = np.clip(boxes[:, 5], 0.0, max_speed_mps)
    direction = np.stack((np.cos(yaw), np.sin(yaw)), axis=1)
    positions = boxes[:, None, :2] + (
        speed[:, None, None] * FUTURE_TIMES_S[None, :, None] * direction[:, None]
    )
    extents = np.stack((boxes[:, 2], boxes[:, 3], np.ones(count)), axis=1)
    actor_class = np.where(boxes[:, 7].astype(np.int32) == 1, 2, 1)
    return FutureActorFrame(
        positions=positions.astype(np.float32),
        yaws=np.broadcast_to(yaw[:, None], (count, steps)).copy(),
        extents=extents.astype(np.float32),
        z=np.zeros(count, dtype=np.float32),
        valid=np.ones((count, steps), dtype=bool),
        actor_ids=np.arange(count, dtype=np.int64),
        class_ids=actor_class.astype(np.uint8),
        ego_extent=np.asarray(ego_extent, dtype=np.float32),
    )


def predicted_candidate_collisions(
    candidate_states: np.ndarray,
    candidate_valid: np.ndarray,
    actors: FutureActorFrame,
    *,
    safety_margin_m: float = 0.2,
) -> np.ndarray:
    """Evaluate each reachable time-sampled ego candidate against actor boxes."""

    states = np.asarray(candidate_states, dtype=np.float32)
    valid = np.asarray(candidate_valid, dtype=bool)
    if states.ndim != 3 or states.shape[1:] != (len(FUTURE_TIMES_S), 6):
        raise ValueError("candidate_states must have shape [M,8,6]")
    if valid.shape != (len(states),):
        raise ValueError("candidate_valid must have shape [M]")
    collisions = np.zeros(len(states), dtype=bool)
    for index in np.flatnonzero(valid):
        yaw = np.arctan2(states[index, :, 2], states[index, :, 3])
        collisions[index] = future_collision_label(
            states[index, :, :2],
            yaw,
            actors,
            safety_margin_m=safety_margin_m,
            include_class_ids=(1, 2),
        ).collision
    return collisions


def select_safe_slowdown(
    candidate_valid: np.ndarray,
    candidate_collision: np.ndarray,
    candidate_velocity: np.ndarray,
    *,
    distance_tolerance_m: float = 0.25,
    current_speed_mps: float | None = None,
    raw_target_speed_mps: float | None = None,
    near_target_tolerance_mps: float = 0.01,
) -> int:
    """Keep raw unless unsafe; take the fastest safe, controller-safe slowdown.

    When current/target speeds are supplied, the actual first-interval PID target
    must not exceed the model's raw target. A two-second distance cap alone does
    not guarantee this for profiles that brake late. A raw brake command is
    always preserved, even if the raw profile is predicted to collide.
    """

    valid = np.asarray(candidate_valid, dtype=bool)
    collision = np.asarray(candidate_collision, dtype=bool)
    velocity = np.asarray(candidate_velocity, dtype=np.float32)
    if (current_speed_mps is None) != (raw_target_speed_mps is None):
        raise ValueError("current and raw target speeds must be provided together")
    if near_target_tolerance_mps < 0:
        raise ValueError("near-target tolerance must be nonnegative")
    if not valid[0] or not collision[0]:
        return 0
    if raw_target_speed_mps is not None and raw_target_speed_mps <= 0.01:
        return 0
    distance = velocity.clip(min=0).sum(axis=1) * float(FUTURE_TIMES_S[0])
    allowed = valid & ~collision & (distance <= distance[0] + distance_tolerance_m)
    if raw_target_speed_mps is not None:
        near_target = np.maximum(2.0 * velocity[:, 0] - current_speed_mps, 0.0)
        allowed &= near_target <= raw_target_speed_mps + near_target_tolerance_mps
    if not allowed.any():
        return 0
    return int(np.where(allowed, distance, -np.inf).argmax())
