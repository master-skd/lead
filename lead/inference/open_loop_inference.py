from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import jaxtyping as jt
import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype

from lead.common.constants import TransfuserBoundingBoxIndex
from lead.data_loader import carla_dataset_utils
from lead.inference import inference_utils
from lead.inference.config_open_loop import OpenLoopConfig
from lead.tfv6.center_net_decoder import PredictedBoundingBox
from lead.tfv6.planning_decoder import decode_two_hot
from lead.tfv6.route_speed_gate import (
    apply_collision_speed_gate,
    collision_risk_to_speed_factor,
)
from lead.tfv6.tfv6 import Prediction
from lead.training.config_training import TrainingConfig
from lead.training.training_utils import create_model

np.set_printoptions(suppress=True)

LOG = logging.getLogger(__name__)


class OpenLoopInference:
    @beartype
    def __init__(
        self,
        config_training: TrainingConfig,
        config_open_loop: OpenLoopConfig,
        model_path: str,
        device: torch.device,
        prefix: str = "model",
    ):
        """
        Open-Loop-Inference constructor.

        Args:
            config_training: Training config object belong to model.
            config_open_loop: Open loop config object.
            model_path: Path to the trained model weights.
            device: Device to run inference on.
            prefix: Prefix of the model weights files to load.
        """
        self.config_training = config_training
        self.config_open_loop = config_open_loop
        self.device = device

        # Loading models
        self.nets: list[torch.nn.Module] = []
        self.model_weight_paths: list[str] = []
        for file in sorted(os.listdir(model_path)):
            if file.startswith(prefix) and file.endswith(".pth"):
                LOG.info(f"Loading model weight from {os.path.join(model_path, file)}")
                net = create_model(self.config_training)
                if self.config_training.sync_batchnorm:
                    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
                state_dict = torch.load(
                    os.path.join(model_path, file),
                    map_location=self.device,
                    weights_only=True,
                )
                net.load_state_dict(
                    state_dict,
                    strict=config_open_loop.strict_weight_load,
                )
                net.cuda(device=self.device).eval()
                self.nets.append(net)
                self.model_weight_paths.append(os.path.abspath(os.path.join(model_path, file)))
        self.route_safety_head = None
        if self.config_training.route_future_safety_gate:
            from lead.tfv6.route_safety_head import load_route_safety_head

            head_path = self.config_training.route_future_safety_head
            checkpoint = torch.load(head_path, map_location=self.device, weights_only=True)
            source = checkpoint.get("source_checkpoint")
            if source is not None and os.path.exists(source):
                source = os.path.abspath(source)
                if not any(os.path.samefile(source, path) for path in self.model_weight_paths):
                    raise ValueError(
                        f"B2d safety head was trained from {source}, but inference loaded "
                        f"{self.model_weight_paths}"
                    )
            self.route_safety_head = load_route_safety_head(checkpoint, self.device).eval()
            self.route_safety_head.requires_grad_(False)
            LOG.info(f"Loaded B2d route safety head from {head_path}")
        self.step = 4  # Constant so produced images start with 5, not really important

    @beartype
    def ensemble_planning_decoder(
        self,
        predictions: list[Prediction],
        data: dict[str, torch.Tensor] | None = None,
    ) -> tuple[
        jt.Float[torch.Tensor, "1 num_waypoints 2"] | None,
        jt.Float[torch.Tensor, "1 num_checkpoints 2"] | None,
        jt.Float[torch.Tensor, " 1 1"] | None,
        jt.Float[torch.Tensor, "1 num_speed_classes"] | None,
        jt.Float[torch.Tensor, "1 num_waypoints"] | None,
        jt.Float[torch.Tensor, " 1"] | None,
        jt.Float[torch.Tensor, "1 1"] | None,
        jt.Float[torch.Tensor, " 1"] | None,
        jt.Float[torch.Tensor, " 1"] | None,
        jt.Float[torch.Tensor, " 1"] | None,
        jt.Float[torch.Tensor, " 1"] | None,
    ]:
        """Ensemble the outputs of the planning decoder from multiple models.

        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_routes: The aggregated route.
            pred_future_waypoints: The aggregated future waypoints.
            pred_target_speed_scalar: The aggregated target speed.
            pred_target_speed_distribution: The aggregated target speed distribution.
        """
        pred_routes = pred_future_waypoints = pred_target_speed_scalar = (
            pred_target_speed_distribution
        ) = pred_future_headings = None
        route_collision_risk = raw_target_speed_scalar = target_speed_factor = None
        route_future_collision_risk = current_speed_factor = future_speed_factor = None

        if self.config_training.use_planning_decoder:
            if self.config_training.predict_target_speed:
                pred_target_speed_logits = torch.stack(
                    [pred.pred_target_speed_distribution[0] for pred in predictions],
                ).mean(dim=0, keepdim=True)  # Average target speed logits.

                pred_target_speed_distribution = F.softmax(
                    pred_target_speed_logits,
                    dim=-1,
                )  # softmax probabilities.
                pred_target_speed_scalar = decode_two_hot(
                    pred_target_speed_distribution,
                    self.config_training.target_speed_classes,
                    self.device,
                ).reshape(1, 1)  # Decode to scalar.
                if (
                    pred_target_speed_distribution[0, 0]
                    > self.config_open_loop.brake_threshold
                ):  # Brake if we are confident enough.
                    pred_target_speed_scalar = pred_target_speed_scalar.new_zeros((1, 1))
                if (
                    self.config_open_loop.lower_target_speed
                ):  # Optionally lower the target speed.
                    pred_target_speed_scalar *= (
                        self.config_open_loop.lower_target_speed_factor
                    )

                if self.config_training.route_speed_safety_gate:
                    risks = [pred.pred_route_collision_risk for pred in predictions]
                    if any(risk is None for risk in risks):
                        raise RuntimeError(
                            "route_speed_safety_gate is enabled but a model did not "
                            "produce pred_route_collision_risk"
                        )
                    # Conservative ensemble: one model seeing danger is sufficient to slow.
                    route_collision_risk = torch.stack(
                        [risk[0].float() for risk in risks],
                    ).amax().reshape(1).to(pred_target_speed_scalar.device)
                    _, current_speed_factor = apply_collision_speed_gate(
                        pred_target_speed_scalar,
                        route_collision_risk,
                        self.config_training,
                    )

                if self.config_training.route_future_safety_gate:
                    if self.route_safety_head is None or data is None or "speed" not in data:
                        raise RuntimeError(
                            "route_future_safety_gate requires a loaded B2d head and input speed"
                        )
                    future_risks = []
                    for prediction in predictions:
                        if (
                            prediction.pred_route_features is None
                            or prediction.pred_route_selected_idx is None
                        ):
                            raise RuntimeError(
                                "route_future_safety_gate requires multimodal route features "
                                "and a selected route index"
                            )
                        features = prediction.pred_route_features[0].float()
                        current_speed = data["speed"].to(features.device).float().reshape(-1)[0]
                        model_target_speed = (
                            prediction.pred_target_speed_scalar.float().reshape(-1)[0]
                        )
                        brake_probability = (
                            prediction.pred_target_speed_distribution.float().softmax(dim=-1)[0, 0]
                        )
                        model_target_speed = torch.where(
                            # Feature extraction used 0.9, so keep the head input exactly
                            # matched to its frozen-feature training distribution.
                            brake_probability > 0.9,
                            torch.zeros_like(model_target_speed),
                            model_target_speed,
                        )
                        arm_risk = torch.sigmoid(
                            self.route_safety_head(features, current_speed, model_target_speed)
                        )
                        selected = prediction.pred_route_selected_idx.reshape(-1)[0].long()
                        future_risks.append(arm_risk[selected])
                    # Same conservative ensemble rule as B2c': one member seeing future
                    # dynamic danger is enough to request a speed cap.
                    route_future_collision_risk = (
                        torch.stack(future_risks).amax().reshape(1)
                    )
                    future_speed_factor = collision_risk_to_speed_factor(
                        route_future_collision_risk,
                        float(self.config_training.route_future_gate_low_threshold),
                        float(self.config_training.route_future_gate_high_threshold),
                        float(self.config_training.route_future_gate_minimum_factor),
                    )

                factors = [
                    factor for factor in (current_speed_factor, future_speed_factor)
                    if factor is not None
                ]
                if factors:
                    raw_target_speed_scalar = pred_target_speed_scalar.clone()
                    target_speed_factor = torch.stack(
                        [factor.reshape(-1)[0] for factor in factors]
                    ).amin().reshape(1).to(pred_target_speed_scalar.device)
                    pred_target_speed_scalar = pred_target_speed_scalar * (
                        target_speed_factor.reshape_as(pred_target_speed_scalar)
                    )

            if self.config_training.predict_temporal_spatial_waypoints:
                pred_future_waypoints = torch.stack(
                    [pred.pred_future_waypoints[0] for pred in predictions],
                ).mean(dim=0, keepdim=True)  # Average waypoints.

            if self.config_training.predict_spatial_path:
                pred_routes = torch.stack(
                    [pred.pred_route[0] for pred in predictions],
                ).mean(dim=0, keepdim=True)  # Average route.

            if (
                self.config_training.use_navsim_data
                and predictions[0].pred_headings is not None
            ):
                pred_future_headings = torch.stack(
                    [pred.pred_headings[0] for pred in predictions],
                ).mean(dim=0, keepdim=True)  # Average headings.

        return (
            pred_routes,
            pred_future_waypoints,
            pred_target_speed_scalar,
            pred_target_speed_distribution,
            pred_future_headings,
            route_collision_risk,
            raw_target_speed_scalar,
            target_speed_factor,
            route_future_collision_risk,
            current_speed_factor,
            future_speed_factor,
        )

    @beartype
    def ensemble_bounding_boxes(
        self,
        predictions: list[Prediction],
    ) -> tuple[list[PredictedBoundingBox], list[PredictedBoundingBox]]:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            List of aggregated bounding boxes in vehicle system.
            List of aggregated bounding boxes in image system.
        """
        pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system = [], []
        if self.config_training.detect_boxes:
            for prediction in predictions:
                pred_bb = prediction.pred_bounding_box.pred_bounding_box_vehicle_system.squeeze().reshape(
                    -1,
                    9,
                )
                if len(pred_bb) > 0:
                    pred_bounding_boxes_vehicle_system.append(pred_bb)

        if len(pred_bounding_boxes_vehicle_system) > 0:
            pred_bounding_boxes_vehicle_system = (
                inference_utils.non_maximum_suppression(
                    pred_bounding_boxes_vehicle_system,
                    float(self.config_training.iou_threshold_nms),
                )
            )

            pred_bounding_boxes_image_system = (
                carla_dataset_utils.bb_vehicle_to_image_system(
                    pred_bounding_boxes_vehicle_system,
                    self.config_training.pixels_per_meter,
                    self.config_training.min_x_meter,
                    self.config_training.min_y_meter,
                )
            )

            pred_bounding_boxes_vehicle_system = [
                PredictedBoundingBox(
                    x=float(bb[TransfuserBoundingBoxIndex.X]),
                    y=float(bb[TransfuserBoundingBoxIndex.Y]),
                    w=float(bb[TransfuserBoundingBoxIndex.W]),
                    h=float(bb[TransfuserBoundingBoxIndex.H]),
                    yaw=float(bb[TransfuserBoundingBoxIndex.YAW]),
                    velocity=float(bb[TransfuserBoundingBoxIndex.VELOCITY]),
                    brake=float(bb[TransfuserBoundingBoxIndex.BRAKE]),
                    clazz=int(bb[TransfuserBoundingBoxIndex.CLASS]),
                    score=float(bb[TransfuserBoundingBoxIndex.SCORE]),
                )
                for bb in pred_bounding_boxes_vehicle_system
            ]

            pred_bounding_boxes_image_system = [
                PredictedBoundingBox(
                    x=float(bb[TransfuserBoundingBoxIndex.X]),
                    y=float(bb[TransfuserBoundingBoxIndex.Y]),
                    w=float(bb[TransfuserBoundingBoxIndex.W]),
                    h=float(bb[TransfuserBoundingBoxIndex.H]),
                    yaw=float(bb[TransfuserBoundingBoxIndex.YAW]),
                    velocity=float(bb[TransfuserBoundingBoxIndex.VELOCITY]),
                    brake=float(bb[TransfuserBoundingBoxIndex.BRAKE]),
                    clazz=int(bb[TransfuserBoundingBoxIndex.CLASS]),
                    score=float(bb[TransfuserBoundingBoxIndex.SCORE]),
                )
                for bb in pred_bounding_boxes_image_system
            ]

        return pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system

    @beartype
    def ensemble_bev_semantic(
        self,
        predictions: list[Prediction],
    ) -> jt.Float[torch.Tensor, "B num_classes bev_height bev_width"] | None:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_bev_semantic: Tensor containing the aggregated BEV semantic map
        """
        if self.config_training.use_bev_semantic:
            pred_bev_semantic = []
            for prediction in predictions:
                pred_bev_semantic.append(prediction.pred_bev_semantic)
            stacked = torch.stack(
                pred_bev_semantic,
                dim=0,
            )  # (num_models, num_batches, num_classes, H, W)
            ch0 = (
                stacked[:, :, 0].min(dim=0).values.unsqueeze(1)
            )  # (num_batches, 1, H, W)
            others = (
                stacked[:, :, 1:].max(dim=0).values
            )  # (num_batches, num_classes-1, H, W)
            return torch.cat([ch0, others], dim=1)  # (num_batches, num_classes, H, W)
        return None

    @beartype
    def ensemble_depth(
        self,
        predictions: list[Prediction],
    ) -> jt.Float[torch.Tensor, "B img_height img_width"] | None:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_depth: Tensor containing the aggregated depth map
        """
        if self.config_training.use_depth:
            pred_depth = []
            for prediction in predictions:
                pred_depth.append(prediction.pred_depth)
            stacked = torch.stack(pred_depth, dim=0)  # (num_models, num_batches, H, W)
            return stacked.mean(dim=0)  # (num_batches, H, W)
        return None

    @beartype
    def ensemble_semantic_segmentation(
        self,
        predictions: list[Prediction],
    ) -> jt.Float[torch.Tensor, "B num_classes img_height img_width"] | None:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            pred_semantic: Tensor containing the aggregated semantic segmentation map
        """
        if self.config_training.use_semantic:
            pred_semantic = []
            for prediction in predictions:
                pred_semantic.append(prediction.pred_semantic)
            stacked = torch.stack(
                pred_semantic,
                dim=0,
            )  # (num_models, num_batches, num_classes, H, W)
            ch0 = (
                stacked[:, :, 0].min(dim=0).values.unsqueeze(1)
            )  # (num_batches, 1, H, W)
            others = (
                stacked[:, :, 1:].max(dim=0).values
            )  # (num_batches, num_classes-1, H, W)
            return torch.cat([ch0, others], dim=1)  # (num_batches, num_classes, H, W)
        return None

    @beartype
    def ensemble(self, _, predictions: list[Prediction]) -> OpenLoopPrediction:
        """
        Args:
            predictions: List of dictionaries containing the predictions of each model
        Returns:
            EnsemblePrediction object containing the aggregated predictions
        """
        # Bounding boxes
        pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system = (
            None,
            None,
        )
        if self.config_training.carla_leaderboard_mode:
            pred_bounding_boxes_vehicle_system, pred_bounding_boxes_image_system = (
                self.ensemble_bounding_boxes(predictions)
            )

        # BEV semantic map
        pred_bev_semantic = None
        if self.config_training.carla_leaderboard_mode:
            pred_bev_semantic = self.ensemble_bev_semantic(predictions)

        # Semantic segmentation
        pred_semantic = None
        if self.config_training.carla_leaderboard_mode:
            pred_semantic = self.ensemble_semantic_segmentation(predictions)

        # Depth
        pred_depth = None
        if self.config_training.carla_leaderboard_mode:
            pred_depth = self.ensemble_depth(predictions)

        # Planning
        (
            pred_route,
            pred_future_waypoints,
            pred_target_speed_scalar,
            pred_target_speed_distribution,
            pred_future_headings,
            route_collision_risk,
            raw_target_speed_scalar,
            target_speed_factor,
            route_future_collision_risk,
            current_speed_factor,
            future_speed_factor,
        ) = self.ensemble_planning_decoder(predictions, _)

        return OpenLoopPrediction(
            pred_future_waypoints=pred_future_waypoints,
            pred_target_speed_scalar=pred_target_speed_scalar,
            pred_target_speed_distribution=pred_target_speed_distribution,
            pred_future_headings=pred_future_headings,
            pred_route=pred_route,
            pred_semantic=pred_semantic,
            pred_depth=pred_depth,
            pred_bev_semantic=pred_bev_semantic,
            pred_bounding_box_vehicle_system=pred_bounding_boxes_vehicle_system,
            pred_bounding_box_image_system=pred_bounding_boxes_image_system,
            pred_radar_predictions=None,
            route_collision_risk=route_collision_risk,
            raw_target_speed_scalar=raw_target_speed_scalar,
            target_speed_factor=target_speed_factor,
            route_future_collision_risk=route_future_collision_risk,
            current_speed_factor=current_speed_factor,
            future_speed_factor=future_speed_factor,
        )

    @beartype
    @torch.inference_mode()
    def forward(self, data: dict[str, torch.Tensor]) -> OpenLoopPrediction:
        """Run inference on the ensemble of models.
        Args:
            data: Dictionary containing the input data for the model

        Returns:
            EnsemblePrediction object containing the aggregated predictions
        """
        self.step += 1
        with torch.amp.autocast(
            device_type="cuda",
            dtype=self.config_training.torch_float_type,
            enabled=self.config_training.use_mixed_precision_training,
        ):
            self.predictions: list[Prediction] = [net(data) for net in self.nets]
        return self.ensemble(data, self.predictions)

    def __getitem__(self, index):
        return self.nets[index]


@jt.jaxtyped(typechecker=beartype)
@dataclass
class OpenLoopPrediction:
    """Raw output predictions from the open loop model."""

    pred_future_waypoints: jt.Float[torch.Tensor, "bs n_waypoints 2"] | None
    pred_future_headings: jt.Float[torch.Tensor, "bs n_waypoints"] | None
    pred_target_speed_scalar: jt.Float[torch.Tensor, "bs 1"] | None
    pred_target_speed_distribution: (
        jt.Float[torch.Tensor, "bs num_speed_classes"] | None
    )
    pred_route: jt.Float[torch.Tensor, "bs n_checkpoints 2"] | None
    pred_semantic: (
        jt.Float[torch.Tensor, "bs num_sem_classes img_height img_width"] | None
    )
    pred_depth: jt.Float[torch.Tensor, "bs img_height img_width"] | None
    pred_bev_semantic: (
        jt.Float[torch.Tensor, "bs num_bev_classes bev_height bev_width"] | None
    )
    pred_bounding_box_vehicle_system: list[PredictedBoundingBox] | None
    pred_bounding_box_image_system: list[PredictedBoundingBox] | None
    pred_radar_predictions: None
    # B2c' post-ensemble speed-gate diagnostics.
    route_collision_risk: torch.Tensor | None
    raw_target_speed_scalar: torch.Tensor | None
    target_speed_factor: torch.Tensor | None
    # B2d' learned future-risk diagnostics; target_speed_factor is the minimum of
    # current_speed_factor and future_speed_factor when both gates are enabled.
    route_future_collision_risk: torch.Tensor | None
    current_speed_factor: torch.Tensor | None
    future_speed_factor: torch.Tensor | None
