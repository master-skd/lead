"""Opt-in, read-only CARLA snapshots for the route 2129 velocity audit."""

from __future__ import annotations

from typing import Any

import torch


def _tensor_list(value: Any) -> list | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim and value.shape[0] == 1:
            value = value[0]
        return value.tolist()
    return value.tolist()


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _xyz(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def build_route2129_capture_record(
    *,
    step: int,
    world: Any,
    ego: Any,
    prediction: Any,
    control: Any,
    sensor_speed_mps: float,
    actor_radius_m: float = 80.0,
) -> dict[str, Any]:
    """Snapshot model output and simulator truth without modifying either.

    Predicted boxes are in the ego-vehicle coordinate system. CARLA actors are
    in world coordinates, with both actor origin and transformed box center.
    Consecutive records can be joined by actor ID to reconstruct actual motion.
    """
    ego_transform = ego.get_transform()
    ego_location = ego_transform.location
    previous_control = ego.get_control()
    actors = []
    for actor in world.get_actors():
        if actor.id == ego.id or not actor.type_id.startswith(("vehicle.", "walker.")):
            continue
        try:
            transform = actor.get_transform()
            location = transform.location
            if location.distance(ego_location) > actor_radius_m:
                continue
            box = actor.bounding_box
            center = transform.transform(box.location)
            actors.append({
                "id": int(actor.id),
                "type_id": str(actor.type_id),
                "location_world_m": _xyz(location),
                "yaw_world_deg": float(transform.rotation.yaw),
                "velocity_world_mps": _xyz(actor.get_velocity()),
                "box_center_world_m": _xyz(center),
                "box_extent_m": _xyz(box.extent),
                "box_yaw_relative_deg": float(box.rotation.yaw),
            })
        except RuntimeError:
            # An actor can disappear while CARLA returns the actor list.
            continue
    actors.sort(key=lambda item: item["id"])
    boxes = prediction.pred_bounding_box_vehicle_system or []
    return {
        "schema_version": 1,
        "step": int(step),
        "carla_frame": int(world.get_snapshot().frame),
        "sensor_speed_mps": float(sensor_speed_mps),
        "ego": {
            "id": int(ego.id),
            "location_world_m": _xyz(ego_location),
            "yaw_world_deg": float(ego_transform.rotation.yaw),
            "velocity_world_mps": _xyz(ego.get_velocity()),
            "box_center_world_m": _xyz(
                ego_transform.transform(ego.bounding_box.location)
            ),
            "box_extent_m": _xyz(ego.bounding_box.extent),
            "box_yaw_relative_deg": float(ego.bounding_box.rotation.yaw),
            "previous_applied_control": {
                "steer": float(previous_control.steer),
                "throttle": float(previous_control.throttle),
                "brake": float(previous_control.brake),
            },
        },
        "commanded_control": {
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
        },
        "controller": {
            "route_steer": float(prediction.route_steer),
            "target_speed_throttle": float(prediction.target_speed_throttle),
            "target_speed_brake": float(prediction.target_speed_brake),
        },
        "raw_target_speed_mps": _scalar(prediction.raw_target_speed_scalar),
        "executed_target_speed_mps": _scalar(prediction.pred_target_speed_scalar),
        "selected_path_ego_m": _tensor_list(prediction.pred_route),
        "selected_trajectory_ego_m": _tensor_list(prediction.pred_trajectory),
        "predicted_boxes_ego": [
            {
                "x_m": float(box.x), "y_m": float(box.y),
                "length_m": float(box.w), "width_m": float(box.h),
                "yaw_rad": float(box.yaw), "velocity_mps": float(box.velocity),
                "brake": float(box.brake), "class": int(box.clazz),
                "score": float(box.score),
            }
            for box in boxes
        ],
        "actors_world": actors,
    }
