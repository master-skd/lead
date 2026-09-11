from __future__ import annotations

import typing
from dataclasses import dataclass

import jaxtyping as jt
import torch
from beartype import beartype
from torch import nn

from lead.common.constants import SourceDataset
from lead.tfv6.bev_decoder import BEVDecoder
from lead.tfv6.center_net_decoder import (
    CenterNetBoundingBoxPrediction,
    CenterNetDecoder,
)
from lead.tfv6.perspective_decoder import PerspectiveDecoder
from lead.tfv6.planning_decoder import PlanningDecoder
from lead.tfv6.radar_detector import RadarDetector
from lead.tfv6.route_safety_rescorer import (
    highest_confidence_valid,
    rescore_routes_for_safety,
    resolve_route_selection_mode,
)
from lead.tfv6.route_speed_gate import selected_route_collision_risk
from lead.tfv6.transfuser_backbone import TransfuserBackbone
from lead.training.config_training import TrainingConfig


class TFv6(nn.Module):
    @beartype
    def __init__(
        self,
        device: torch.device,
        config: TrainingConfig,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.log = {}

        self.backbone = TransfuserBackbone(self.device, self.config)

        if self.config.use_semantic and self.config.use_carla_data:
            self.semantic_decoder = PerspectiveDecoder(
                config=self.config,
                in_channels=self.backbone.num_image_features,
                out_channels=self.config.num_semantic_classes,
                perspective_upsample_factor=self.backbone.perspective_upsample_factor,
                modality="semantic",
                device=self.device,
                source_data=SourceDataset.CARLA,
            )

        if self.config.use_depth and self.config.use_carla_data:
            self.depth_decoder = PerspectiveDecoder(
                config=self.config,
                in_channels=self.backbone.num_image_features,
                out_channels=1,
                perspective_upsample_factor=self.backbone.perspective_upsample_factor,
                modality="depth",
                device=self.device,
                source_data=SourceDataset.CARLA,
            )

        if self.config.use_bev_semantic:
            if self.config.use_carla_data:
                self.bev_semantic_decoder = BEVDecoder(
                    self.config,
                    self.config.num_bev_semantic_classes,
                    self.device,
                    source_data=SourceDataset.CARLA,
                )

            if self.config.use_navsim_data:
                self.bev_semantic_decoder_navsim = BEVDecoder(
                    self.config,
                    self.config.navsim_num_bev_semantic_classes,
                    self.device,
                    source_data=SourceDataset.NAVSIM,
                )

        if self.config.detect_boxes:
            if self.config.use_carla_data:
                self.center_net_decoder = CenterNetDecoder(
                    self.config.num_bb_classes,
                    self.config,
                    self.device,
                    source_data=SourceDataset.CARLA,
                )
            if self.config.use_navsim_data:
                self.center_net_decoder_navsim = CenterNetDecoder(
                    self.config.navsim_num_bb_classes,
                    self.config,
                    self.device,
                    source_data=SourceDataset.NAVSIM,
                )

        if self.config.radar_detection and self.config.use_carla_data:
            self.radar_detector = RadarDetector(
                bev_input_dim=self.backbone.num_lidar_features,
                config=self.config,
                device=self.device,
            )

        if self.config.use_planning_decoder:
            self.planning_decoder = PlanningDecoder(
                input_bev_channels=self.backbone.num_lidar_features,
                config=self.config,
                device=self.device,
            ).to(self.device)

        if self.config.use_intent_decoder:
            from lead.tfv6.intent_decoder import IntentDecoder

            self.intent_decoder = IntentDecoder(self.config, self.device).to(
                self.device,
            )

        # P5b: frozen VLM intent head. Its output is fed to the planner as the intent
        # condition (external_intent), replacing the TFv6-feature intent head, while
        # backbone+planner are finetuned. Loaded here and kept frozen.
        if self.config.use_vlm_intent:
            import torch as _torch

            from lead.tfv6.vlm_intent_decoder import VLMIntentDecoder

            self.vlm_intent_decoder = VLMIntentDecoder(self.config).to(self.device)
            ckpt = _torch.load(
                self.config.vlm_intent_ckpt,
                map_location=self.device,
                weights_only=True,
            )
            self.vlm_intent_decoder.load_state_dict(ckpt["model"])
            self.vlm_intent_decoder.eval()
            self.vlm_intent_decoder.requires_grad_(False)

    def _anchors_from_blob(self, intent_logits):
        """Closed-loop anchor: extract skeleton anchors from the pred blob and pack to the
        planner's (B, K, 4) = [sin a, cos a, reach/30, valid] embedding input. Matches
        precompute_anchors.py / the dataset packing so train and eval anchors are identical.
        """
        import numpy as np

        from lead.tfv6.anchor_extraction import K_MAX, extract_anchors_from_blob

        cfg = self.config
        ppm = cfg.pixels_per_meter
        row_ego = int((0 - cfg.min_y_meter) * ppm)
        col_ego = int((0 - cfg.min_x_meter) * ppm)
        blob = torch.sigmoid(intent_logits.float()).detach().cpu().numpy()  # (B,1,H,W)
        B = blob.shape[0]
        out = np.zeros((B, K_MAX, 4), dtype=np.float32)
        for b in range(B):
            anchors, _ = extract_anchors_from_blob(blob[b, 0], ppm, row_ego, col_ego)
            valid, ang, reach = anchors[:, 0], anchors[:, 1], anchors[:, 2]
            feat = np.stack([np.sin(ang), np.cos(ang), reach / 30.0, valid], axis=1)
            feat[valid < 0.5] = 0.0
            out[b] = feat
        return torch.from_numpy(out).to(self.device)

    @beartype
    def forward(
        self,
        data: dict[str, typing.Any],
        external_intent: jt.Float[torch.Tensor, "bs 1 ih iw"] | None = None,
    ) -> Prediction:
        """Run the full model.

        Args:
            data: Batched sensor/label dict.
            external_intent: Optional externally-produced BEV intent logits. When
                given, it replaces the internal ``intent_decoder`` output as the
                planner's intent condition (P4b ablation: feed a VLM-produced
                intent while keeping backbone + planner frozen). When ``None`` the
                behaviour is unchanged.

        Returns:
            Model predictions.
        """
        self.log = {}
        pred_route = pred_future_waypoints = pred_target_speed_distribution = (
            pred_target_speed_scalar
        ) = pred_headings = None
        pred_semantic = pred_depth = pred_bounding_box = pred_bev_semantic = None
        pred_bounding_box_navsim = pred_bev_semantic_navsim = None
        pred_visual_intent = None
        pred_route_multimodal = pred_route_conf = pred_route_anchor = None
        pred_route_features = None
        pred_route_selected_idx = pred_route_collision_cost = None
        pred_route_off_corridor_cost = pred_route_safe_mask = None
        pred_route_collision_risk = None
        planner_anchor = None

        # Backbone
        bev_features, image_features = self.backbone(data)

        # Radar detection
        radar_features = radar_predictions = None
        if self.config.use_carla_data and self.config.radar_detection:
            radar_features, radar_predictions = self.radar_detector(bev_features, data)

        # BEV feature grid (shared by intent / detection / bev-semantic heads)
        bev_feature_grid = self.backbone.top_down(bev_features)

        # Visual-intent head: soft BEV intent field. Computed before planning so the
        # control head can be conditioned on it (P2). An externally supplied intent
        # (P4b) overrides the internal head, bypassing the intent_decoder entirely.
        if external_intent is not None:
            pred_visual_intent = external_intent
        elif self.config.use_vlm_intent:
            # P5b: intent from the frozen VLM head (no grad, fp32). The head was
            # trained in fp32; force autocast off so its convs don't see mixed
            # float/bf16 inside the training autocast context. The planner casts the
            # intent to its own dtype internally (planning_decoder detach+to).
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
                pred_visual_intent = self.vlm_intent_decoder(
                    data["vlm_hidden"].to(self.device).float()
                )
        elif self.config.use_intent_decoder:
            pred_visual_intent = self.intent_decoder(bev_feature_grid)

        # Planning / control head
        if self.config.use_planning_decoder:
            planner_radar_features = radar_features
            planner_radar_predictions = radar_predictions
            if not self.config.use_radar_detection or not self.config.use_carla_data:
                planner_radar_features = planner_radar_predictions = None
            planner_intent = (
                pred_visual_intent if self.config.use_control_conditioning else None
            )
            if self.config.multimodal_planner:
                planner_anchor = data.get("anchor")
                if planner_anchor is None and pred_visual_intent is not None:
                    # Closed-loop: no cached anchor -> compute from the (frozen, hence
                    # deterministic) pred blob with the SAME extractor used to build the
                    # training cache, so train/eval anchors are identical.
                    planner_anchor = self._anchors_from_blob(pred_visual_intent)
            (
                pred_route,
                pred_future_waypoints,
                pred_target_speed_distribution,
                pred_target_speed_scalar,
                pred_headings,
            ) = self.planning_decoder(
                bev_features,
                planner_radar_features,
                planner_radar_predictions,
                data,
                log=self.log,
                intent=planner_intent,
                anchor=planner_anchor,
            )

        # Semantic segmentation forward pass
        if self.config.use_carla_data and self.config.use_semantic:
            pred_semantic = self.semantic_decoder(data, image_features, self.log)

        # Depth estimation forward pass
        if self.config.use_carla_data and self.config.use_depth:
            pred_depth = self.depth_decoder(data, image_features, self.log)

        # Bounding box detection forward pass
        if self.config.detect_boxes:
            if self.config.use_carla_data:
                pred_bounding_box = self.center_net_decoder(
                    data,
                    bev_feature_grid,
                    self.log,
                )
            if self.config.use_navsim_data:
                pred_bounding_box_navsim = self.center_net_decoder_navsim(
                    data,
                    bev_feature_grid,
                    self.log,
                )

        # BEV semantic segmentation forward pass
        if self.config.use_bev_semantic:
            if self.config.use_carla_data:
                pred_bev_semantic = self.bev_semantic_decoder(
                    bev_feature_grid,
                    self.log,
                )
            if self.config.use_navsim_data:
                pred_bev_semantic_navsim = self.bev_semantic_decoder_navsim(
                    bev_feature_grid,
                    self.log,
                )

        # B2c late route resolution. It is inference-only and intentionally runs after
        # BEV semantics: collision comes from the predicted current-scene occupancy and
        # corridor support comes from the predicted intent blob, matching closed loop.
        if self.config.multimodal_planner and not self.training:
            pred_route_multimodal = data.get("route_multimodal")
            pred_route_conf = data.get("route_conf")
            pred_route_features = data.get("route_features")
            pred_route_anchor = (
                planner_anchor.to(pred_route_conf.device)
                if planner_anchor is not None and pred_route_conf is not None
                else planner_anchor
            )
            if pred_route_multimodal is not None and pred_route_conf is not None:
                mode = resolve_route_selection_mode(self.config)
                valid = (
                    planner_anchor[..., 3].to(pred_route_conf.device) > 0.5
                    if planner_anchor is not None
                    else torch.ones_like(pred_route_conf, dtype=torch.bool)
                )
                if mode == "slot0":
                    pred_route_selected_idx = torch.zeros_like(
                        pred_route_conf[:, 0], dtype=torch.long,
                    )
                    pred_route = pred_route_multimodal[:, 0]
                elif mode == "confidence":
                    pred_route_selected_idx = highest_confidence_valid(
                        pred_route_conf, valid,
                    )
                    ar = torch.arange(pred_route_conf.shape[0], device=pred_route_conf.device)
                    pred_route = pred_route_multimodal[ar, pred_route_selected_idx]
                else:
                    if pred_bev_semantic is None or pred_visual_intent is None:
                        raise RuntimeError(
                            "route_selection_mode='safety_rescore' requires CARLA BEV "
                            "semantic logits and visual-intent logits"
                        )
                    safety = rescore_routes_for_safety(
                        pred_route_multimodal,
                        pred_route_conf,
                        planner_anchor,
                        pred_bev_semantic,
                        pred_visual_intent,
                        self.config,
                    )
                    pred_route = safety.route
                    pred_route_selected_idx = safety.selection.selected_idx
                    pred_route_collision_cost = safety.collision_cost
                    pred_route_off_corridor_cost = safety.off_corridor_cost
                    pred_route_safe_mask = safety.selection.safe_mask
                    ar = torch.arange(pred_route_conf.shape[0], device=pred_route_conf.device)
                    base = safety.selection.baseline_idx
                    picked = safety.selection.selected_idx
                    self.log["route_safety_switched_rate"] = safety.selection.switched.float().mean()
                    self.log["route_safety_fallback_rate"] = safety.selection.fallback.float().mean()
                    self.log["route_safety_n_safe"] = safety.selection.safe_mask.float().sum(1).mean()
                    self.log["route_safety_collision_before"] = safety.collision_cost[ar, base].mean()
                    self.log["route_safety_collision_after"] = safety.collision_cost[ar, picked].mean()
                    self.log["route_safety_off_corridor_before"] = safety.off_corridor_cost[ar, base].mean()
                    self.log["route_safety_off_corridor_after"] = safety.off_corridor_cost[ar, picked].mean()

        # B2c': route geometry/branch is already resolved above and is never changed here.
        # Only expose its current-scene collision risk; OpenLoopInference applies the speed
        # factor after model ensembling, which is the scalar consumed by the closed-loop PID.
        if self.config.route_speed_safety_gate and not self.training:
            if pred_route is None or pred_bev_semantic is None:
                raise RuntimeError(
                    "route_speed_safety_gate requires a predicted route and CARLA BEV "
                    "semantic logits"
                )
            pred_route_collision_risk = selected_route_collision_risk(
                pred_route,
                pred_bev_semantic,
                self.config,
            )
            self.log["route_speed_collision_risk"] = pred_route_collision_risk.mean()

        # Collect predictions
        return Prediction(
            # Planning prediction
            pred_future_waypoints=pred_future_waypoints,
            pred_target_speed_distribution=pred_target_speed_distribution,
            pred_target_speed_scalar=pred_target_speed_scalar,
            pred_route=pred_route,
            # CARLA perception prediction
            pred_semantic=pred_semantic,
            pred_depth=pred_depth,
            pred_bounding_box=pred_bounding_box,
            pred_bev_semantic=pred_bev_semantic,
            pred_radar_features=radar_features,
            pred_radar_predictions=radar_predictions,
            # NavSim perception prediction
            pred_bounding_box_navsim=pred_bounding_box_navsim,
            pred_bev_semantic_navsim=pred_bev_semantic_navsim,
            pred_headings=pred_headings,
            pred_visual_intent=pred_visual_intent,
            pred_route_multimodal=pred_route_multimodal,
            pred_route_conf=pred_route_conf,
            pred_route_anchor=pred_route_anchor,
            pred_route_features=pred_route_features,
            pred_route_selected_idx=pred_route_selected_idx,
            pred_route_collision_cost=pred_route_collision_cost,
            pred_route_off_corridor_cost=pred_route_off_corridor_cost,
            pred_route_safe_mask=pred_route_safe_mask,
            pred_route_collision_risk=pred_route_collision_risk,
        )

    @beartype
    def compute_loss(
        self,
        predictions: Prediction,
        data: dict[str, typing.Any],
    ) -> tuple[dict, dict]:
        loss = {}
        # Semantic segmentation loss
        if self.config.use_semantic and self.config.use_carla_data:
            self.semantic_decoder.compute_loss(
                predictions.pred_semantic,
                data,
                loss,
                log=self.log,
            )

        # Depth estimation loss
        if self.config.use_depth and self.config.use_carla_data:
            self.depth_decoder.compute_loss(
                predictions.pred_depth,
                data,
                loss,
                log=self.log,
            )

        # BEV semantic segmentation loss
        if self.config.use_bev_semantic:
            if self.config.use_carla_data:
                self.bev_semantic_decoder.compute_loss(
                    predictions.pred_bev_semantic,
                    data,
                    loss,
                    log=self.log,
                )
            if self.config.use_navsim_data:
                self.bev_semantic_decoder_navsim.compute_loss(
                    predictions.pred_bev_semantic_navsim,
                    data,
                    loss,
                    log=self.log,
                )

        # Bounding box detection loss
        if self.config.detect_boxes:
            if self.config.use_carla_data:
                self.center_net_decoder.compute_loss(
                    data=data,
                    bounding_box_features=predictions.pred_bounding_box,
                    losses=loss,
                    log=self.log,
                )
            if self.config.use_navsim_data:
                self.center_net_decoder_navsim.compute_loss(
                    data=data,
                    bounding_box_features=predictions.pred_bounding_box_navsim,
                    losses=loss,
                    log=self.log,
                )

        # Radar detection loss
        if self.config.radar_detection and self.config.use_carla_data:
            self.radar_detector.compute_loss(
                pred=predictions.pred_radar_predictions,
                data=data,
                loss=loss,
                log=self.log,
            )

        # Planning loss
        if self.config.use_planning_decoder:
            self.planning_decoder.compute_loss(
                data=data,
                predictions=predictions,
                loss=loss,
                log=self.log,
            )

        # Visual-intent loss
        if self.config.use_intent_decoder:
            self.intent_decoder.compute_loss(
                predictions.pred_visual_intent,
                data,
                loss,
                log=self.log,
            )

        return loss, self.log


@jt.jaxtyped(typechecker=beartype)
@dataclass
class Prediction:
    """Raw output predictions from the model."""

    # Planning prediction
    pred_future_waypoints: jt.Float[torch.Tensor, "bs n_waypoints 2"] | None
    pred_target_speed_distribution: (
        jt.Float[torch.Tensor, "bs num_speed_classes"] | None
    )
    pred_target_speed_scalar: jt.Float[torch.Tensor, " bs"] | None
    pred_route: jt.Float[torch.Tensor, "bs n_checkpoints 2"] | None

    # CARLA perception prediction
    pred_semantic: (
        jt.Float[torch.Tensor, "bs num_semantic_classes img_height img_width"] | None
    )
    pred_bev_semantic: (
        jt.Float[torch.Tensor, "bs num_bev_classes bev_height bev_width"] | None
    )
    pred_depth: jt.Float[torch.Tensor, "bs img_height img_width"] | None
    pred_bounding_box: CenterNetBoundingBoxPrediction | None
    pred_radar_features: jt.Float[torch.Tensor, "B Q C"] | None
    pred_radar_predictions: jt.Float[torch.Tensor, "B Q 4"] | None

    # NavSim perception prediction
    pred_bounding_box_navsim: CenterNetBoundingBoxPrediction | None
    pred_bev_semantic_navsim: (
        jt.Float[torch.Tensor, "bs num_bev_classes_navsim bev_height bev_width"] | None
    )
    pred_headings: jt.Float[torch.Tensor, "bs n_waypoints"] | None
    pred_visual_intent: (
        jt.Float[torch.Tensor, "bs 1 bev_height bev_width"] | None
    ) = None
    # B2c inference diagnostics (kept optional for all single-route callers).
    pred_route_multimodal: torch.Tensor | None = None
    pred_route_conf: torch.Tensor | None = None
    pred_route_anchor: torch.Tensor | None = None
    pred_route_features: torch.Tensor | None = None
    pred_route_selected_idx: torch.Tensor | None = None
    pred_route_collision_cost: torch.Tensor | None = None
    pred_route_off_corridor_cost: torch.Tensor | None = None
    pred_route_safe_mask: torch.Tensor | None = None
    pred_route_collision_risk: torch.Tensor | None = None
