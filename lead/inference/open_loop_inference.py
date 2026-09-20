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
from lead.tfv6.route_safety_rescorer import resolve_route_selection_mode
from lead.tfv6.route_speed_gate import (
    apply_collision_speed_gate,
    collision_risk_to_speed_factor,
)
from lead.tfv6.tfv6 import Prediction
from lead.tfv6.velocity_scorer import (
    apply_velocity_scorer_gate,
    compose_route_velocity_trajectory,
    select_velocity_profile,
)
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
                self.model_weight_paths.append(
                    os.path.abspath(os.path.join(model_path, file))
                )
        self.route_safety_head = None
        if self.config_training.route_future_safety_gate:
            from lead.tfv6.route_safety_head import load_route_safety_head

            head_path = self.config_training.route_future_safety_head
            checkpoint = torch.load(
                head_path, map_location=self.device, weights_only=True
            )
            source = checkpoint.get("source_checkpoint")
            if source is not None and os.path.exists(source):
                source = os.path.abspath(source)
                if not any(
                    os.path.samefile(source, path) for path in self.model_weight_paths
                ):
                    raise ValueError(
                        f"B2d safety head was trained from {source}, but inference loaded "
                        f"{self.model_weight_paths}"
                    )
            self.route_safety_head = load_route_safety_head(
                checkpoint, self.device
            ).eval()
            self.route_safety_head.requires_grad_(False)
            LOG.info(f"Loaded B2d route safety head from {head_path}")
        self.velocity_scorer = None
        self.velocity_vocabulary = None
        if self.config_training.route_velocity_scorer_gate:
            selection_mode = self.config_training.route_velocity_selection_mode
            if selection_mode not in {"gate", "profile_select"}:
                raise ValueError(
                    "route_velocity_selection_mode must be 'gate' or 'profile_select'"
                )
            if selection_mode == "profile_select" and (
                resolve_route_selection_mode(self.config_training) != "confidence"
                or not self.config_training.predict_spatial_path
            ):
                raise ValueError(
                    "profile_select requires route_selection_mode='confidence' "
                    "and predict_spatial_path=True"
                )
            if (
                self.config_training.route_speed_safety_gate
                or self.config_training.route_future_safety_gate
            ):
                raise ValueError(
                    "route_velocity_scorer_gate must be evaluated without the B2c/B2d "
                    "speed gates"
                )
            if len(self.nets) != 1:
                raise ValueError(
                    "B3a velocity scorer was trained on one frozen corridor model and "
                    "requires exactly one model weight at inference"
                )
            from lead.tfv6.velocity_scorer import load_velocity_scorer

            scorer_path = self.config_training.route_velocity_scorer_head
            vocabulary_path = self.config_training.route_velocity_vocabulary
            checkpoint = torch.load(
                scorer_path, map_location=self.device, weights_only=True
            )
            source = checkpoint.get("source_checkpoint")
            if source is not None and os.path.exists(source):
                source = os.path.abspath(source)
                if not os.path.samefile(source, self.model_weight_paths[0]):
                    raise ValueError(
                        f"B3a scorer was trained from {source}, but inference loaded "
                        f"{self.model_weight_paths[0]}"
                    )
            source_vocabulary = checkpoint.get("source_vocabulary")
            if source_vocabulary is not None and os.path.exists(source_vocabulary):
                if not os.path.samefile(
                    os.path.abspath(source_vocabulary),
                    os.path.abspath(vocabulary_path),
                ):
                    raise ValueError(
                        "B3a scorer and inference configuration use different velocity "
                        "vocabularies"
                    )
            vocabulary = np.load(vocabulary_path, allow_pickle=False)
            if vocabulary.ndim != 2 or not np.isfinite(vocabulary).all():
                raise ValueError(
                    f"invalid B3a velocity vocabulary shape/content: {vocabulary.shape}"
                )
            profile_steps = int(checkpoint["scorer_config"]["profile_steps"])
            if vocabulary.shape[1] != profile_steps:
                raise ValueError(
                    f"B3a vocabulary has {vocabulary.shape[1]} steps but scorer expects "
                    f"{profile_steps}"
                )
            self.velocity_scorer = (
                load_velocity_scorer(checkpoint, self.device)
                .eval()
                .requires_grad_(False)
            )
            self.velocity_vocabulary = torch.from_numpy(
                vocabulary.astype(np.float32, copy=False)
            ).to(self.device)
            LOG.info(
                "Loaded B3a velocity scorer from %s with vocabulary %s",
                scorer_path,
                vocabulary_path,
            )
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
        jt.Float[torch.Tensor, " 1"] | None,
        jt.Float[torch.Tensor, " 1"] | None,
        jt.Int[torch.Tensor, " 1"] | None,
        jt.Bool[torch.Tensor, " 1"] | None,
        jt.Bool[torch.Tensor, " 1"] | None,
        torch.Tensor | None,
        torch.Tensor | None,
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
        velocity_raw_risk = velocity_selected_risk = velocity_selected_index = None
        velocity_switched = velocity_fallback = None
        velocity_selected_profile = pred_trajectory = None

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
                    pred_target_speed_scalar = pred_target_speed_scalar.new_zeros(
                        (1, 1)
                    )
                if (
                    self.config_open_loop.lower_target_speed
                ):  # Optionally lower the target speed.
                    pred_target_speed_scalar *= (
                        self.config_open_loop.lower_target_speed_factor
                    )

                if self.config_training.route_velocity_scorer_gate:
                    if (
                        self.velocity_scorer is None
                        or self.velocity_vocabulary is None
                        or data is None
                        or "speed" not in data
                    ):
                        raise RuntimeError(
                            "route_velocity_scorer_gate requires its scorer, vocabulary, "
                            "and input speed"
                        )
                    prediction = predictions[0]
                    if (
                        prediction.pred_route_features is None
                        or prediction.pred_route_selected_idx is None
                    ):
                        raise RuntimeError(
                            "route_velocity_scorer_gate requires multimodal route features "
                            "and a selected route index"
                        )
                    selected_arm = prediction.pred_route_selected_idx.reshape(-1)[
                        0
                    ].long()
                    route_feature = prediction.pred_route_features[0, selected_arm][
                        None
                    ]
                    current_speed = (
                        data["speed"]
                        .to(route_feature.device, dtype=torch.float32)
                        .reshape(-1)[:1]
                    )
                    raw_target_speed_scalar = pred_target_speed_scalar.clone()
                    selection_mode = self.config_training.route_velocity_selection_mode
                    if selection_mode not in {"gate", "profile_select"}:
                        raise ValueError(
                            "route_velocity_selection_mode must be 'gate' or 'profile_select'"
                        )
                    if selection_mode == "profile_select" and (
                        resolve_route_selection_mode(self.config_training)
                        != "confidence"
                        or not self.config_training.predict_spatial_path
                    ):
                        raise ValueError(
                            "profile_select requires route_selection_mode='confidence' "
                            "and predict_spatial_path=True"
                        )
                    velocity_selector = (
                        select_velocity_profile
                        if selection_mode == "profile_select"
                        else apply_velocity_scorer_gate
                    )
                    selector_kwargs = dict(
                        safe_threshold=float(
                            self.config_training.route_velocity_safe_threshold
                        ),
                        interval_s=float(
                            self.config_training.route_velocity_profile_interval_s
                        ),
                        max_accel_mps2=float(
                            self.config_training.route_velocity_max_accel_mps2
                        ),
                        max_decel_mps2=float(
                            self.config_training.route_velocity_max_decel_mps2
                        ),
                    )
                    if selection_mode == "gate":
                        selector_kwargs["unsafe_threshold"] = float(
                            self.config_training.route_velocity_unsafe_threshold
                        )
                    velocity_gate = velocity_selector(
                        self.velocity_scorer,
                        route_feature,
                        current_speed,
                        raw_target_speed_scalar,
                        self.velocity_vocabulary,
                        **selector_kwargs,
                    )
                    pred_target_speed_scalar = velocity_gate.target_speed.reshape_as(
                        pred_target_speed_scalar
                    )
                    velocity_raw_risk = velocity_gate.raw_risk
                    velocity_selected_risk = velocity_gate.selected_risk
                    velocity_selected_index = velocity_gate.selected_index
                    velocity_switched = velocity_gate.switched
                    velocity_fallback = velocity_gate.fallback
                    if selection_mode == "profile_select":
                        velocity_selected_profile = velocity_gate.candidate_velocity[
                            0, velocity_gate.selected_index[0]
                        ][None]

                if self.config_training.route_speed_safety_gate:
                    risks = [pred.pred_route_collision_risk for pred in predictions]
                    if any(risk is None for risk in risks):
                        raise RuntimeError(
                            "route_speed_safety_gate is enabled but a model did not "
                            "produce pred_route_collision_risk"
                        )
                    # Conservative ensemble: one model seeing danger is sufficient to slow.
                    route_collision_risk = (
                        torch.stack(
                            [risk[0].float() for risk in risks],
                        )
                        .amax()
                        .reshape(1)
                        .to(pred_target_speed_scalar.device)
                    )
                    _, current_speed_factor = apply_collision_speed_gate(
                        pred_target_speed_scalar,
                        route_collision_risk,
                        self.config_training,
                    )

                if self.config_training.route_future_safety_gate:
                    if (
                        self.route_safety_head is None
                        or data is None
                        or "speed" not in data
                    ):
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
                        current_speed = (
                            data["speed"].to(features.device).float().reshape(-1)[0]
                        )
                        model_target_speed = (
                            prediction.pred_target_speed_scalar.float().reshape(-1)[0]
                        )
                        brake_probability = (
                            prediction.pred_target_speed_distribution.float().softmax(
                                dim=-1
                            )[0, 0]
                        )
                        model_target_speed = torch.where(
                            # Feature extraction used 0.9, so keep the head input exactly
                            # matched to its frozen-feature training distribution.
                            brake_probability > 0.9,
                            torch.zeros_like(model_target_speed),
                            model_target_speed,
                        )
                        arm_risk = torch.sigmoid(
                            self.route_safety_head(
                                features, current_speed, model_target_speed
                            )
                        )
                        selected = prediction.pred_route_selected_idx.reshape(-1)[
                            0
                        ].long()
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
                    factor
                    for factor in (current_speed_factor, future_speed_factor)
                    if factor is not None
                ]
                if factors:
                    raw_target_speed_scalar = pred_target_speed_scalar.clone()
                    target_speed_factor = (
                        torch.stack([factor.reshape(-1)[0] for factor in factors])
                        .amin()
                        .reshape(1)
                        .to(pred_target_speed_scalar.device)
                    )
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
                if velocity_selected_profile is not None:
                    pred_trajectory = compose_route_velocity_trajectory(
                        pred_routes,
                        velocity_selected_profile,
                        float(self.config_training.route_velocity_profile_interval_s),
                    )

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
            velocity_raw_risk,
            velocity_selected_risk,
            velocity_selected_index,
            velocity_switched,
            velocity_fallback,
            velocity_selected_profile,
            pred_trajectory,
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
            velocity_raw_risk,
            velocity_selected_risk,
            velocity_selected_index,
            velocity_switched,
            velocity_fallback,
            velocity_selected_profile,
            pred_trajectory,
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
            velocity_raw_risk=velocity_raw_risk,
            velocity_selected_risk=velocity_selected_risk,
            velocity_selected_index=velocity_selected_index,
            velocity_switched=velocity_switched,
            velocity_fallback=velocity_fallback,
            velocity_selected_profile=velocity_selected_profile,
            pred_trajectory=pred_trajectory,
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
    # B3a frozen-route velocity-profile diagnostics.
    velocity_raw_risk: torch.Tensor | None
    velocity_selected_risk: torch.Tensor | None
    velocity_selected_index: torch.Tensor | None
    velocity_switched: torch.Tensor | None
    velocity_fallback: torch.Tensor | None
    # B3a direct-selection mode: complete selected speed profile and its
    # time-sampled positions on the confidence-selected spatial path.
    velocity_selected_profile: torch.Tensor | None
    pred_trajectory: torch.Tensor | None
