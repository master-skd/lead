from dataclasses import dataclass
from copy import deepcopy

import jaxtyping as jt
import numpy as np
import torch
from beartype import beartype

from lead.common.pid_controller import LateralPIDController, PIDController, get_throttle
from lead.expert.config_expert import ExpertConfig
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.inference.open_loop_inference import OpenLoopInference, OpenLoopPrediction
from lead.tfv6.tfv6 import Prediction
from lead.tfv6.local_path_shadow import local_path_variants, score_local_paths
from lead.tfv6.collision_cost import corridor_cost_per_point
from lead.tfv6.predicted_actor_velocity_gate import extrapolate_detected_actors
from lead.tfv6.route_safety_rescorer import resolve_route_selection_mode
from lead.tfv6.future_collision import (
    future_collision_label,
    interpolate_route_by_distance,
    speed_profile_distances,
)
from lead.data_loader import carla_dataset_utils
from lead.training.config_training import TrainingConfig

np.set_printoptions(suppress=True)


@dataclass
@beartype
class ClosedLoopPrediction(OpenLoopPrediction):
    steer: float
    throttle: float
    brake: float
    waypoints_steer: float
    waypoints_throttle: float
    waypoints_brake: float
    route_steer: float
    target_speed_throttle: float
    target_speed_brake: float
    route_path_shadow: dict | None


class ClosedLoopInference(OpenLoopInference):
    @beartype
    def __init__(
        self,
        config_training: TrainingConfig,
        config_closed_loop: ClosedLoopConfig,
        config_expert: ExpertConfig,
        model_path: str,
        device: torch.device,
        prefix: str,
    ):
        super().__init__(
            config_training=config_training,
            config_open_loop=config_closed_loop,
            model_path=model_path,
            device=device,
            prefix=prefix,
        )
        self.config_expert = config_expert
        self.config_training = config_training
        self.config_closed_loop = config_closed_loop

        self.lateral_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.turn_kp,
            k_i=self.config_closed_loop.turn_ki,
            k_d=self.config_closed_loop.turn_kd,
            n=self.config_closed_loop.turn_n,
        )
        self.longitudinal_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.speed_kp,
            k_i=self.config_closed_loop.speed_ki,
            k_d=self.config_closed_loop.speed_kd,
            n=self.config_closed_loop.speed_n,
        )
        self.lateral_route_controller = LateralPIDController(self.config_closed_loop)
        if self.config_training.route_local_path_shadow:
            if (
                len(self.nets) != 1
                or resolve_route_selection_mode(self.config_training) != "confidence"
                or not self.config_training.predict_spatial_path
                or not self.config_training.detect_boxes
                or self.config_training.route_predicted_actor_velocity_gate
                or self.config_training.route_velocity_scorer_gate
                or self.config_training.route_speed_safety_gate
                or self.config_training.route_future_safety_gate
            ):
                raise ValueError(
                    "local-Path shadow requires one confidence-route model with boxes "
                    "and no active safety/velocity gate"
                )
        self.longitudinal_target_speed_controller = PIDController(
            k_p=self.config_closed_loop.speed_kp,
            k_i=self.config_closed_loop.speed_ki,
            k_d=self.config_closed_loop.speed_kd,
            n=self.config_closed_loop.speed_n,
        )

        self.step = 4  # Constant so produced images start with 5, not really important

    @jt.jaxtyped(typechecker=beartype)
    def execute_route_and_target_speed(
        self,
        pred_checkpoints: jt.Float[torch.Tensor, "1 num_checkpoints 2"],
        pred_target_speed: jt.Float[torch.Tensor, "1 1"],
        speed: jt.Float[torch.Tensor, "1 1"],
        ego_vehicle_location: float = 0.0,
        ego_vehicle_rotation: float = 0.0,
    ) -> tuple[float, float, float]:
        """Predicts vehicle control with a PID controller. Use route and target speed predictions.

        Args:
            pred_checkpoints: Predicted route checkpoints in ego-vehicle coordinates.
            pred_target_speed: Predicted target speed in m/s.
            speed: Current speed of the vehicle in m/s.
            ego_vehicle_location: Current lateral location of the ego vehicle.
            ego_vehicle_rotation: Current rotation of the ego vehicle.
        Returns:
            steer: Steering command in [-1, 1]
            throttle: Throttle command in [0, 1]
            brake: Brake command in [0, 1]
        """
        pred_checkpoints = pred_checkpoints[0].data.cpu().float().numpy()
        speed = float(speed[0].data.cpu().float().numpy())
        pred_target_speed = float(pred_target_speed[0].data.cpu().float().numpy())

        brake = bool(
            pred_target_speed < 0.01
            or (speed / pred_target_speed) > self.config_closed_loop.brake_ratio,
        )
        steer = self.lateral_route_controller.step(
            pred_checkpoints,
            speed,
            ego_vehicle_location,
            ego_vehicle_rotation,
            sensor_agent_steer_correction=self.config_closed_loop.sensor_agent_steer_correction,
        )
        throttle, brake = get_throttle(
            brake,
            pred_target_speed,
            speed,
            self.config_expert,
        )

        return steer, throttle, float(brake)

    @jt.jaxtyped(typechecker=beartype)
    def execute_waypoints(
        self,
        waypoints: jt.Float[torch.Tensor, "1 num_waypoints 2"],
        velocity: jt.Float[torch.Tensor, "1 1"],
    ) -> tuple[float, float, float]:
        """Predicts vehicle control with a PID controller. Use waypoint predictions.

        Args:
            waypoints: Predicted future waypoints in ego-vehicle coordinates.
            velocity: Current speed of the vehicle in m/s.
        Returns:
            steer: Steering command in [-1, 1]
            throttle: Throttle command in [0, 1]
            brake: Brake command in [0, 1]
        """
        waypoints = waypoints[0].data.cpu().float().numpy()
        speed = velocity[0].data.cpu().float().numpy()

        one_second = int(
            self.config_training.carla_fps // self.config_training.data_save_freq,
        )
        half_second = one_second // 2

        desired_speed = (
            np.linalg.norm(waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0
        )
        delta_speed = np.clip(
            desired_speed - speed,
            0.0,
            self.config_closed_loop.wp_delta_clip,
        )

        brake = (desired_speed < self.config_closed_loop.brake_speed) or (
            (speed / desired_speed) > self.config_closed_loop.brake_ratio
        )
        throttle = self.longitudinal_waypoint_controller.step(delta_speed)
        throttle = throttle if not brake else 0.0

        if (
            self.config_closed_loop.tuned_aim_distance
        ):  # In LB2, we go faster, so we need to choose waypoints farther ahead
            # range [2.4, 10.5] same as in the disentangled rep.
            aim_distance = np.clip(0.975532 * speed + 1.915288, 24, 105) / 10
        else:
            # To replicate the slow TransFuser behaviour we have a different distance
            # inside and outside of intersections (detected by desired_speed)
            if desired_speed < self.config_closed_loop.aim_distance_threshold:
                aim_distance = self.config_closed_loop.aim_distance_slow
            else:
                aim_distance = self.config_closed_loop.aim_distance_fast

        # We follow the waypoint that is at least a certain distance away
        aim_index = waypoints.shape[0] - 1
        for index, predicted_waypoint in enumerate(waypoints):
            if np.linalg.norm(predicted_waypoint) >= aim_distance:
                aim_index = index
                break

        aim = waypoints[aim_index]
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01:
            # When we don't move we don't want the angle error to accumulate in the integral
            angle = 0.0
        if brake:
            angle = 0.0

        steer = self.lateral_waypoint_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)  # Valid steering values are in [-1,1]
        return float(steer), float(throttle), float(brake)

    @beartype
    def ensemble(
        self,
        data: dict[str, torch.Tensor],
        predictions: list[Prediction],
    ) -> ClosedLoopPrediction:
        """
        Args:
            data: Dictionary containing the input data
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            ClosedLoopPrediction: Dataclass containing all predictions
        """
        ego_speed = data["speed"].to(self.device, dtype=torch.float32).unsqueeze(1)
        open_loop_prediction: OpenLoopPrediction = super().ensemble(data, predictions)

        # Convert high-level commands to vehicle controls
        steer = throttle = brake = waypoint_steer = waypoint_throttle = (
            waypoint_brake
        ) = route_steer = target_speed_throttle = target_speed_brake = None
        route_path_shadow = None
        route_controller_snapshot = (
            deepcopy(self.lateral_route_controller)
            if self.config_training.route_local_path_shadow else None
        )

        if open_loop_prediction.pred_route is not None:
            route_steer, target_speed_throttle, target_speed_brake = (
                self.execute_route_and_target_speed(
                    open_loop_prediction.pred_route,
                    open_loop_prediction.pred_target_speed_scalar,
                    ego_speed,
                )
            )
        if route_controller_snapshot is not None:
            prediction = predictions[0]
            if (
                route_steer is None
                or open_loop_prediction.pred_route is None
                or prediction.pred_visual_intent is None
                or prediction.pred_bounding_box is None
            ):
                raise RuntimeError("local-Path shadow is missing route, intent or boxes")
            raw_route = open_loop_prediction.pred_route[0].float().cpu().numpy()
            current_speed = float(ego_speed.reshape(-1)[0].item())
            raw_target_speed = float(
                open_loop_prediction.pred_target_speed_scalar.reshape(-1)[0].item()
            )
            boxes = carla_dataset_utils.bb_image_to_vehicle_system(
                prediction.pred_bounding_box.pred_bounding_box_image_system[0],
                self.config_training.pixels_per_meter,
                self.config_training.min_x_meter,
                self.config_training.min_y_meter,
            )
            actors = extrapolate_detected_actors(
                boxes,
                score_threshold=float(
                    self.config_training.route_predicted_actor_score_threshold
                ),
                nms_iou_threshold=float(
                    self.config_training.route_predicted_actor_nms_iou_threshold
                ),
            )
            distances = speed_profile_distances(current_speed, raw_target_speed)
            raw_xy, raw_yaw = interpolate_route_by_distance(
                raw_route, distances, extrapolate=True
            )
            raw_collision = future_collision_label(
                raw_xy, raw_yaw, actors, safety_margin_m=0.2,
                include_class_ids=(1, 2),
            ).collision
            if not raw_collision or current_speed < 1.0:
                route_path_shadow = {
                    "schema_version": 1,
                    "shadow_only": True,
                    "candidate_evaluated": False,
                    "current_speed_mps": current_speed,
                    "raw_target_speed_mps": raw_target_speed,
                    "predicted_actor_count": int(actors.num_actors),
                    "baseline_steer": float(route_steer),
                    "raw_predicted_collision": bool(raw_collision),
                    "would_select_index": None,
                    "would_switch": False,
                    "candidates": [],
                }
            else:
                variants = local_path_variants(raw_route)
                routes = np.concatenate((raw_route[None], variants), axis=0)
                corridor_points = corridor_cost_per_point(
                    torch.as_tensor(routes[None], device=self.device),
                    torch.sigmoid(prediction.pred_visual_intent.float()),
                    self.config_training,
                    reach_m=float(self.config_training.route_corridor_reach_m),
                )[0].float().cpu().numpy()
                route_path_shadow = score_local_paths(
                    routes,
                    corridor_points,
                    actors,
                    route_controller_snapshot,
                    current_speed_mps=current_speed,
                    raw_target_speed_mps=raw_target_speed,
                    baseline_steer=float(route_steer),
                    sensor_agent_steer_correction=bool(
                        self.config_closed_loop.sensor_agent_steer_correction
                    ),
                )
        if open_loop_prediction.pred_future_waypoints is not None:
            waypoints_steer, waypoints_throttle, waypoints_brake = (
                self.execute_waypoints(
                    open_loop_prediction.pred_future_waypoints,
                    ego_speed,
                )
            )

        # Select which high-level modality we want to use for each control
        if self.config_closed_loop.steer_modality == "route":
            steer = route_steer
        elif self.config_closed_loop.steer_modality == "waypoint":
            steer = waypoints_steer
        else:
            raise ValueError(
                f"Invalid steer_modality: {self.config_closed_loop.steer_modality}",
            )

        if self.config_closed_loop.throttle_modality == "target_speed":
            throttle = target_speed_throttle
        elif self.config_closed_loop.throttle_modality == "waypoint":
            throttle = waypoints_throttle
        else:
            raise ValueError(
                f"Invalid throttle_modality: {self.config_closed_loop.throttle_modality}",
            )

        if self.config_closed_loop.brake_modality == "target_speed":
            brake = target_speed_brake
        elif self.config_closed_loop.brake_modality == "waypoint":
            brake = waypoints_brake
        else:
            raise ValueError(
                f"Invalid brake_modality: {self.config_closed_loop.brake_modality}",
            )

        # Turn off throttle if we brake
        if brake > 0.0:
            throttle = 0.0
            if (
                ego_speed < 0.01
            ):  # When we don't move we don't want the angle error to accumulate in the integral
                steer = 0.0

        # Return all predictions
        return ClosedLoopPrediction(
            **vars(open_loop_prediction),
            steer=steer,
            throttle=throttle,
            brake=brake,
            waypoints_steer=waypoint_steer,
            waypoints_throttle=waypoint_throttle,
            waypoints_brake=waypoint_brake,
            route_steer=route_steer,
            target_speed_throttle=target_speed_throttle,
            target_speed_brake=target_speed_brake,
            route_path_shadow=route_path_shadow,
        )
