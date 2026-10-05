"""Same-pass counterfactual labels for a live multi-arm planner/scorer run.

All privileged futures are used here only to construct targets. The scorer sees
only detached candidate trajectories and detached planner scene tokens.
"""

from __future__ import annotations

import numpy as np
import torch

from lead.data_loader.future_actor_cache import FutureActorFrame
from lead.tfv6.future_collision import future_collision_label
from lead.tfv6.trajectory_scene_scorer import build_trajectory_candidate_states


def compose_route_states(
    routes: torch.Tensor, current_speed: torch.Tensor, target_speed: torch.Tensor,
) -> torch.Tensor:
    """Expand K spatial arms with the planner's single speed into 2 s trajectories."""
    if routes.ndim != 4 or routes.shape[-1] != 2:
        raise ValueError("routes must be [B,K,P,2]")
    batch, arms, points, _ = routes.shape
    now = current_speed.reshape(batch).float().clamp_min(0)
    target = target_speed.reshape(batch).float().clamp_min(0)
    fraction = torch.arange(1, 9, device=routes.device, dtype=torch.float32) / 8
    velocity = now[:, None] + (target - now)[:, None] * fraction[None]
    velocity = velocity[:, None].expand(-1, arms, -1).reshape(batch * arms, 1, 8)
    states = build_trajectory_candidate_states(
        routes.detach().float().reshape(batch * arms, points, 2), velocity,
        now[:, None].expand(-1, arms).reshape(-1),
    )
    return states.reshape(batch, arms, 8, 6).detach()


def _actors_from_batch(data: dict, index: int) -> FutureActorFrame:
    count = int(data["joint_actor_counts"][index])

    def field(name):
        return np.asarray(data[f"joint_actor_{name}"][index, :count].cpu())

    return FutureActorFrame(
        positions=field("positions").astype(np.float32),
        yaws=field("yaws").astype(np.float32),
        extents=field("extents").astype(np.float32),
        z=field("z").astype(np.float32),
        valid=field("valid").astype(bool),
        actor_ids=np.arange(count, dtype=np.int64),
        class_ids=field("class_ids").astype(np.uint8),
        ego_extent=np.asarray(data["joint_actor_ego_extent"][index].cpu(), dtype=np.float32),
    )


def build_same_pass_labels(
    states: torch.Tensor, valid: torch.Tensor, data: dict, config,
) -> dict[str, torch.Tensor]:
    """Compute seven GT targets for *current* generated routes, never stale cache paths."""
    values = states.detach().float().cpu().numpy()
    active = valid.detach().bool().cpu().numpy()
    batch, arms, steps, _ = values.shape
    if steps != 8:
        raise ValueError("joint scorer requires eight 4 Hz trajectory states")
    labels = {name: np.zeros((batch, arms), dtype=np.float32) for name in (
        "collision_free", "ttc", "drivable", "task", "progress", "comfort", "imitation",
    )}
    gt_future = data["joint_ego_future"].cpu().numpy()
    nav_route = data["route"].cpu().numpy()
    corridor = data["visual_intent_label"].cpu().numpy()
    ppm = float(config.pixels_per_meter)
    for b in range(batch):
        actors = _actors_from_batch(data, b)
        map_image = corridor[b, 0]
        nav_angle = np.arctan2(nav_route[b, -1, 1], nav_route[b, -1, 0])
        for k in np.flatnonzero(active[b]):
            candidate = values[b, k]
            xy = candidate[:, :2]
            yaw = np.arctan2(candidate[:, 2], candidate[:, 3])
            collision = future_collision_label(
                xy, yaw, actors, safety_margin_m=0.2,
                include_class_ids=(1, 2),
            )
            labels["collision_free"][b, k] = float(not collision.collision)
            labels["ttc"][b, k] = min(collision.ttc_s / 2, 1.0)
            col = np.rint((xy[:, 0] - config.min_x_meter) * ppm).astype(int)
            row = np.rint((xy[:, 1] - config.min_y_meter) * ppm).astype(int)
            inside = (row >= 0) & (row < map_image.shape[0]) & (col >= 0) & (col < map_image.shape[1])
            road = np.zeros(steps, dtype=np.float32)
            road[inside] = map_image[row[inside], col[inside]]
            labels["drivable"][b, k] = float(np.clip(road.mean(), 0, 1))
            end_angle = np.arctan2(xy[-1, 1], xy[-1, 0])
            angle_error = np.arctan2(np.sin(end_angle - nav_angle), np.cos(end_angle - nav_angle))
            labels["task"][b, k] = float(np.exp(-0.5 * (angle_error / np.deg2rad(35)) ** 2))
            nav_unit = nav_route[b, -1] / max(np.linalg.norm(nav_route[b, -1]), 1e-3)
            expected_distance = max(float(candidate[:, 4].mean()) * 2.0, 0.5)
            progress_m = float(np.dot(xy[-1], nav_unit))
            labels["progress"][b, k] = float(np.clip(progress_m / expected_distance, 0, 1))
            yaw_delta = np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw)))
            yaw_rate = float(np.abs(yaw_delta).mean() / 0.25)
            acceleration = float(np.abs(candidate[:, 5]).mean())
            labels["comfort"][b, k] = float(np.exp(-acceleration / 4.0 - yaw_rate / 1.5))
            ade = np.linalg.norm(xy - gt_future[b], axis=-1).mean()
            labels["imitation"][b, k] = float(np.exp(-ade / 2.0))
    return {name: torch.from_numpy(value).to(states.device) for name, value in labels.items()}
