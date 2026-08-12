import logging
import math

import jaxtyping as jt
import torch
import torch.nn.functional as F
from beartype import beartype
from torch import nn

import lead.common.common_utils as common_utils
from lead.common.constants import RadarLabels
from lead.tfv6 import transfuser_utils as fn
from lead.training.config_training import TrainingConfig

logger = logging.getLogger(__name__)


class PlanningDecoder(nn.Module):
    @beartype
    def __init__(
        self,
        input_bev_channels: int,
        config: TrainingConfig,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.planning_context_encoder = PlanningContextEncoder(
            config=self.config,
            input_bev_channels=input_bev_channels,
            device=self.device,
        )

        # Number of queries: route + waypoints + target_speed (flexible based on config)
        # B2: in multimodal mode the route segment is replicated K times (K arms), each
        # route query gets an anchor embedding added to break symmetry.
        self.mm = self.config.multimodal_planner
        self.K = self.config.multimodal_planner_k if self.mm else 1
        num_queries = 0
        if self.config.predict_spatial_path:
            num_queries += self.config.num_route_points_prediction * self.K
        if self.config.predict_temporal_spatial_waypoints:
            num_queries += self.config.num_way_points_prediction
        if self.config.predict_target_speed:
            num_queries += 1

        self.query = nn.Parameter(
            torch.zeros(
                1,
                num_queries,
                self.config.transfuser_token_dim,
            ),
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                self.config.transfuser_token_dim,
                self.config.transfuser_num_bev_cross_attention_heads,
                activation=nn.GELU(),
                batch_first=True,
            ),
            num_layers=self.config.transfuser_num_bev_cross_attention_layers,
            norm=nn.LayerNorm(self.config.transfuser_token_dim),
        )

        # Only create decoders if needed
        if self.config.predict_spatial_path:
            self.route_decoder = nn.Linear(config.transfuser_token_dim, 2)
            if self.mm:
                # B2: anchor embedding [sin a, cos a, reach/30, valid] -> D, added to each
                # mode's route queries to break symmetry; per-mode confidence head.
                D = self.config.transfuser_token_dim
                self.anchor_embed = nn.Sequential(
                    nn.Linear(4, D), nn.GELU(), nn.Linear(D, D),
                )
                self.conf_decoder = nn.Linear(D, 1)
        if self.config.predict_temporal_spatial_waypoints:
            self.wp_decoder = nn.Linear(config.transfuser_token_dim, 2)
            if self.config.use_navsim_data:
                self.heading_decoder = nn.Linear(config.transfuser_token_dim, 1)
        if self.config.predict_target_speed:
            self.target_speed_decoder = nn.Sequential(
                nn.Linear(
                    self.config.transfuser_token_dim,
                    self.config.transfuser_token_dim,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(
                    self.config.transfuser_token_dim,
                    len(self.config.target_speed_classes),
                ),
            )

        self.tp_normalization_constants = torch.tensor(
            self.config.target_points_normalization_constants,
            device=self.device,
            dtype=self.config.torch_float_type,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.query)

    @beartype
    def forward(
        self,
        bev_features: jt.Float[torch.Tensor, "bs bev_dim height_bev width_bev"],
        radar_features: jt.Float[torch.Tensor, "B Q C"] | None,
        radar_predictions: jt.Float[torch.Tensor, "B Q 4"] | None,
        data: dict,
        log: dict,
        intent: jt.Float[torch.Tensor, "bs 1 ih iw"] | None = None,
        anchor: jt.Float[torch.Tensor, "bs kmax 4"] | None = None,
    ) -> tuple[
        jt.Float[torch.Tensor, "B n_checkpoints 2"] | None,
        jt.Float[torch.Tensor, "B n_waypoints 2"],
        jt.Float[torch.Tensor, "B speed_classes"] | None,
        jt.Float[torch.Tensor, " B"] | None,
        jt.Float[torch.Tensor, "B n_waypoints"] | None,
    ]:
        """
        Args:
            bev_features: BEV features.
            radar_features: Radar features.
            radar_predictions: Radar predictions.
            data: dict
            log: dict
        Returns:
            route: Spatial path.
            waypoints: Spatial and temporal path.
            target_speed: Target speed distribution.
            target_speed_scalar: Target speed in m/s.
            headings: Heading predictions (if using NavSim data).
        """
        self.kv = context_tokens = self.planning_context_encoder(
            bev_features=bev_features,
            radar_logits=radar_features,
            radar_predictions=radar_predictions,
            data=data,
            log=log,
            intent=intent,
        )

        bs = context_tokens.shape[0]

        # Build the decoder input queries. In multimodal mode, add a per-mode anchor
        # embedding to each mode's route queries (before cross-attention) to break the
        # symmetry between the K route arms; otherwise K identical query blocks would
        # collapse to one under WTA. Waypoint/speed queries are untouched.
        query_in = self.query.repeat(bs, 1, 1)  # (B, K*route + wp + speed, D)
        if self.mm and self.config.predict_spatial_path:
            n_route = self.config.num_route_points_prediction
            if anchor is None:
                anchor = torch.zeros(bs, self.K, 4, device=query_in.device, dtype=query_in.dtype)
            # anchor cols already [sin a, cos a, reach/30, valid] (packed in dataset)
            a_embed = self.anchor_embed(anchor.to(query_in.device, query_in.dtype))  # (B, K, D)
            # add each mode's embedding to its n_route query rows
            a_rep = a_embed.unsqueeze(2).expand(bs, self.K, n_route, -1).reshape(bs, self.K * n_route, -1)
            query_in[:, : self.K * n_route] = query_in[:, : self.K * n_route] + a_rep

        queries = self.transformer_decoder(query_in, context_tokens)

        # Split the queries flexibly based on what we're predicting
        query_idx = 0
        route = None
        waypoints = None
        headings = None
        target_speed_dist = None
        target_speed_scalar = None

        if self.config.predict_spatial_path:
            n_route = self.config.num_route_points_prediction
            if self.mm:
                # K modes: decode all K*n_route route queries, reshape to (B,K,n_route,D),
                # cumsum PER MODE along the point axis -> (B,K,n_route,2). Confidence from
                # each mode's mean query. Multimodal outputs go out via `log` (bypass) so
                # the return signature / beartype stay single-route compatible; the main
                # `route` slot holds a placeholder (mode 0) for downstream single-route code.
                route_q = queries[:, query_idx : query_idx + self.K * n_route]
                route_q = route_q.reshape(bs, self.K, n_route, -1)  # (B,K,n_route,D)
                route_all = torch.cumsum(self.route_decoder(route_q), dim=2)  # (B,K,n_route,2)
                conf = self.conf_decoder(route_q.mean(dim=2)).squeeze(-1)  # (B,K)
                # Bypass via `data` (NOT `log`: logger scalar-reduces every log entry and
                # would choke on these tensors). compute_loss reads them from data.
                data["route_multimodal"] = route_all
                data["route_conf"] = conf
                # Which arm goes to the single-route consumers (closed-loop control reads
                # exactly this tensor via Prediction.pred_route).
                #
                # Slot 0 is NOT a safe default in closed loop. The slot order comes from the
                # anchor list, which anchor_extraction sorts by REACH. Training reads
                # carla.Map lane-graph anchors from disk; closed loop re-derives them from the
                # predicted blob, where reach carries ~1.25 m of error -- enough to reorder
                # near-equal-length arms. Measured over 2500 multi-arm frames by comparing the
                # two cached anchor sets: slot 0 points at a DIFFERENT branch (>25 deg, i.e.
                # outside the anchor hinge tolerance) on 74.7% of them, mean angular
                # difference 55.8 deg, p90 108 deg. Where the top-2 arms are within 3 m of
                # each other (20.7% of frames) slot 0 flips on 88.8%. Open-loop eval cannot
                # see any of this: it reads the GT anchors, so the slot order it measures is
                # the training order by construction, which is why ade_arm0 there equals
                # ade_conf to 16 decimals and why that agreement says nothing about closed
                # loop.
                #
                # Selecting by confidence removes the dependence on slot order. It MUST mask
                # padding arms first: they get no gradient from any per-arm loss (all are
                # masked by `valid`), so their route is unconstrained drift, and without
                # route_pad_conf_loss_weight they still score high -- B2-final leaves them at
                # 0.989 against the real arms' 0.414, so a bare argmax steers down one. Use
                # this with a checkpoint trained with term 6 (p5_stepB2_corridor).
                if self.config.route_select_by_conf:
                    valid = (
                        anchor[..., 3] > 0.5
                        if anchor is not None
                        else torch.ones_like(conf, dtype=torch.bool)
                    )  # (B,K)
                    # -inf on padding so argmax can never land there; all-padding rows (no
                    # anchor extracted at all) fall back to slot 0 rather than picking noise.
                    masked = conf.masked_fill(~valid, float("-inf"))
                    pick = torch.where(
                        valid.any(dim=1), masked.argmax(dim=1), torch.zeros_like(conf[:, 0], dtype=torch.long),
                    )  # (B,)
                    route = route_all[torch.arange(bs, device=route_all.device), pick]
                    log["route_arm_picked"] = pick.float().mean()
                    log["route_n_valid_arms"] = valid.float().sum(dim=1).mean()
                else:
                    route = route_all[:, 0]  # placeholder single route
                query_idx += self.K * n_route
            else:
                route_queries = queries[:, query_idx : query_idx + n_route]
                route = torch.cumsum(self.route_decoder(route_queries), 1)
                query_idx += n_route

        if self.config.predict_temporal_spatial_waypoints:
            waypoints_queries = queries[
                :,
                query_idx : query_idx + self.config.num_way_points_prediction,
            ]
            waypoints = torch.cumsum(self.wp_decoder(waypoints_queries), 1)
            if self.config.use_navsim_data:
                headings = torch.cumsum(self.heading_decoder(waypoints_queries), 1)
            query_idx += self.config.num_way_points_prediction

        if self.config.predict_target_speed:
            target_speed_query = queries[:, query_idx]
            target_speed_dist = self.target_speed_decoder(target_speed_query)

            with torch.amp.autocast(device_type="cuda", enabled=False):
                target_speed_softmax = torch.softmax(target_speed_dist.float(), dim=-1)
                target_speed_scalar = decode_two_hot(
                    target_speed_softmax,
                    self.config.target_speed_classes,
                    self.device,
                )

        return (
            route,
            waypoints,
            target_speed_dist,
            target_speed_scalar,
            headings.squeeze(-1) if headings is not None else None,
        )

    def _multimodal_route_loss(self, route_label, data, loss, log):
        """B2 winner-take-all route loss + confidence BCE over K arms.

        log["route_multimodal"]: unused. data["route_multimodal"]: (B,K,n_route,2) ;
        data["route_conf"]: (B,K) logits. data["anchor"]: (B,K,4) with valid flag in col 3.
        Only valid arms compete; the arm closest (mean-L2 to GT) is the winner and gets the
        near-weighted L1. Confidence target: winner=1, other valid=0, padding=0.
        """
        route_all = data["route_multimodal"].float()  # (B,K,n,2)
        conf = data["route_conf"].float()             # (B,K)
        B, K, n, _ = route_all.shape
        gt = route_label.float()                     # (B,n,2)
        valid = data["anchor"].to(self.device).float()[:, :, 3]  # (B,K)

        # per-arm mean-L2 to GT; invalid arms get +inf so they never win
        d = torch.linalg.norm(route_all - gt[:, None], dim=-1).mean(dim=2)  # (B,K)
        d_masked = d + (1.0 - valid) * 1e9
        winner = d_masked.argmin(dim=1)              # (B,)

        # near-weighted L1 on the winner arm
        near = self.config.route_near_points or n
        far_w = self.config.route_far_weight
        w = torch.ones(n, device=self.device)
        w[near:] = far_w
        win_route = route_all[torch.arange(B), winner]  # (B,n,2)
        per_pt = F.l1_loss(win_route, gt, reduction="none").mean(dim=-1)  # (B,n)
        loss["loss_spatial_route"] = (per_pt * w).sum(dim=1).mean() / w.sum()
        loss["loss_spatial_route"] += F.l1_loss(win_route[:, -1], gt[:, -1])  # FDE

        # confidence BCE: winner=1, others 0; mask padding out of the mean
        conf_tgt = torch.zeros_like(conf)
        conf_tgt[torch.arange(B), winner] = 1.0
        bce = F.binary_cross_entropy_with_logits(conf, conf_tgt, reduction="none")  # (B,K)
        loss["loss_route_conf"] = (bce * valid).sum() / valid.sum().clamp(min=1.0)

        # B2 anti-collapse (term 3): push each NON-winner valid arm toward its OWN anchor
        # BRANCH, so the K arms diverge instead of collapsing onto the winner (pure WTA
        # collapses -- cf. DiffusionDrive). We can do this because our anchors are
        # per-scene and multimodal, each pointing at a distinct lane-graph branch (unlike
        # DiffusionDrive's scene-agnostic kmeans anchors, which is why they needed RL).
        #
        # Deliberately LOOSE: an L1 to the anchor endpoint would make the anchor a
        # trajectory label and bind control to intent (the red line -- intent must stay a
        # fuzzy "which way", the planner resolves the actual path late). Instead:
        #   (a) angular hinge -- free inside +-tol_deg of the anchor bearing, so the arm
        #       may curve however it likes as long as it commits to that branch;
        #   (b) reach hinge -- only penalise arms far SHORTER than the anchor's reach
        #       (a near-zero-length arm has no meaningful direction); going further is free.
        anchor = data["anchor"].to(self.device).float()  # (B,K,4)=[sin,cos,reach/30,valid]
        sin_a, cos_a, reach_m = anchor[:, :, 0], anchor[:, :, 1], anchor[:, :, 2] * 30.0
        pred_end = route_all[:, :, -1, :]  # (B,K,2) each arm's endpoint, (x=fwd, y=lat)
        pred_norm = torch.linalg.norm(pred_end, dim=-1)  # (B,K)

        # (a) cosine between the arm's bearing and the anchor's; hinge at cos(tol).
        cos_sim = (pred_end[..., 0] * cos_a + pred_end[..., 1] * sin_a) / pred_norm.clamp(min=1e-3)
        cos_tol = math.cos(math.radians(self.config.route_anchor_tol_deg))
        angle_hinge = F.relu(cos_tol - cos_sim)  # (B,K), 0 when within tolerance

        # (b) one-sided reach hinge, normalised by 30 m so it is scale-comparable to (a).
        min_reach = reach_m * self.config.route_anchor_min_reach_frac
        reach_hinge = F.relu(min_reach - pred_norm) / 30.0  # (B,K)

        nonwin = valid.clone()
        nonwin[torch.arange(B), winner] = 0.0  # exclude winner (it regresses expert)
        arm_pen = angle_hinge + reach_hinge
        loss["loss_route_anchor"] = (arm_pen * nonwin).sum() / nonwin.sum().clamp(min=1.0)

        # B2 term 4: collision cost on ALL valid arms, not just the winner. This is what
        # makes late resolution meaningful in closed loop -- the arm the controller picks
        # is chosen by collision cost, so every arm has to be physically drivable, and an
        # arm that the angular hinge sends into an obstacle must route around it.
        if self.config.use_collision_cost and "bev_semantic" in data:
            from lead.tfv6.collision_cost import collision_cost_per_mode

            per_mode = collision_cost_per_mode(
                route_all,
                data["bev_semantic"].to(self.device, non_blocking=True),
                self.config,
                sigma_m=self.config.collision_sigma_m,
            )  # (B,K)
            loss["loss_route_collision"] = (
                per_mode * valid
            ).sum() / valid.sum().clamp(min=1.0)

        # B2 term 5: keep every valid arm ON a drivable branch. Terms 3-4 cannot do this --
        # the hinges only constrain bearing/reach, and term 4's danger field is built from
        # vehicles/walkers, so an arm crossing onto grass or oncoming lanes costs nothing.
        # Visualising B2-final showed exactly that failure: winner arms tracked the road while
        # every non-winner arm drove off-road, with all four losses still falling.
        # The lane-graph corridor (union of L/S/R arms at lane width) supplies the missing
        # shape while staying weak enough that intent is still "which way", not a path.
        if self.config.use_route_corridor_loss and "visual_intent_label" in data:
            from lead.tfv6.collision_cost import corridor_cost_per_mode

            corr = corridor_cost_per_mode(
                route_all,
                data["visual_intent_label"].to(self.device, non_blocking=True).float(),
                self.config,
                reach_m=self.config.route_corridor_reach_m,
            )  # (B,K)
            loss["loss_route_corridor"] = (
                corr * valid
            ).sum() / valid.sum().clamp(min=1.0)

        # B2 term 6: drive the PADDING arms' confidence to zero. Padding arms get no gradient
        # from terms 1/3/4/5 (all masked by `valid`), so their route output is unconstrained
        # drift -- yet B2-final left them with conf logits ~6.4, indistinguishable from the
        # real arm's 6.375. Any closed-loop selector that ranks by confidence without also
        # checking the valid flag would happily steer down one. Supervising them here makes
        # confidence self-sufficient instead of relying on every consumer to mask correctly.
        pad = 1.0 - valid  # (B,K)
        if self.config.route_pad_conf_loss_weight > 0:
            pad_bce = F.binary_cross_entropy_with_logits(
                conf, torch.zeros_like(conf), reduction="none",
            )  # (B,K)
            loss["loss_route_pad_conf"] = (
                pad_bce * pad
            ).sum() / pad.sum().clamp(min=1.0)

    @beartype
    def compute_loss(self, predictions, data: dict, loss: dict, log: dict):
        # Prepare loss dictionary
        with torch.amp.autocast(device_type="cuda", enabled=False):
            if self.config.predict_temporal_spatial_waypoints:
                waypoints_label = data["future_waypoints"].to(
                    self.device,
                    dtype=self.config.torch_float_type,
                    non_blocking=True,
                )[:, : self.config.num_way_points_prediction]

                loss["loss_spatio_temporal_waypoints"] = F.l1_loss(
                    predictions.pred_future_waypoints.float(),
                    waypoints_label.float(),
                    reduction="none",
                ).mean()

                if self.config.use_navsim_data:
                    heading_label = data["future_yaws"].to(
                        self.device,
                        dtype=self.config.torch_float_type,
                        non_blocking=True,
                    )
                    loss["loss_spatio_temporal_waypoints"] = (
                        loss["loss_spatio_temporal_waypoints"]
                        + F.l1_loss(
                            predictions.pred_headings.float(),
                            heading_label.float(),
                        ).mean()
                    )

            if self.config.predict_target_speed:
                brake_label = data["brake"].to(
                    self.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
                target_speed_distribution = encode_two_hot(
                    data["target_speed"].to(
                        self.device,
                        dtype=self.config.torch_float_type,
                        non_blocking=True,
                    ),
                    self.config.target_speed_classes,
                    brake=brake_label,
                )
                loss["loss_target_speed"] = F.cross_entropy(
                    predictions.pred_target_speed_distribution.float(),
                    target_speed_distribution,
                )

            if self.config.predict_spatial_path:
                route_label = data["route"].to(
                    self.device,
                    dtype=self.config.torch_float_type,
                    non_blocking=True,
                )
                if self.mm and "route_multimodal" in data:
                    # B2 WTA: pick, among VALID arms, the one closest to the expert route;
                    # only the winner gets the (near-weighted) regression gradient so the
                    # other modes aren't dragged toward the single expert path (anti-
                    # collapse). Confidence BCE: winner=1, other valid=0, padding=0.
                    self._multimodal_route_loss(route_label, data, loss, log)
                    continue_single = False
                else:
                    continue_single = True

                if continue_single:
                    # P5 Step A near-weighted single-route loss.
                    near = self.config.route_near_points
                    far_w = self.config.route_far_weight
                    if near is not None and far_w != 1.0:
                        n_pts = route_label.shape[1]
                        w = torch.ones(n_pts, device=self.device, dtype=torch.float32)
                        w[near:] = far_w
                        per_pt = F.l1_loss(
                            predictions.pred_route.float(),
                            route_label.float(),
                            reduction="none",
                        ).mean(dim=-1)
                        loss["loss_spatial_route"] = (per_pt * w).sum(dim=1).mean() / w.sum()
                    else:
                        loss["loss_spatial_route"] = F.l1_loss(
                            predictions.pred_route.float(),
                            route_label.float(),
                        )
                    loss["loss_spatial_route"] += F.l1_loss(
                        predictions.pred_route[:, -1, :].float(),
                        route_label[:, -1, :].float(),
                    )

            # Differentiable collision cost on predicted waypoints (P2.2)
            if self.config.use_collision_cost and (
                predictions.pred_future_waypoints is not None
            ):
                from lead.tfv6.collision_cost import differentiable_collision

                loss["loss_collision"] = differentiable_collision(
                    predictions.pred_future_waypoints.float(),
                    data["bev_semantic"].to(self.device, non_blocking=True),
                    self.config,
                    sigma_m=self.config.collision_sigma_m,
                )

        if (
            "iteration" in data
            and ((data["iteration"] + 1) % self.config.log_scalars_frequency) == 0
        ):
            if self.config.predict_spatial_path:
                route_label = data["route"].to(
                    self.device,
                    dtype=self.config.torch_float_type,
                    non_blocking=True,
                )
                log.update(
                    {
                        "metric/route_ade": common_utils.average_displacement_error(
                            predictions.pred_route,
                            route_label,
                        ),
                        "metric/route_fde": common_utils.final_displacement_error(
                            predictions.pred_route,
                            route_label,
                        ),
                    },
                )

            if self.config.predict_target_speed:
                brake_label = data["brake"].to(
                    self.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
                target_speed_distribution = encode_two_hot(
                    data["target_speed"].to(
                        self.device,
                        dtype=self.config.torch_float_type,
                        non_blocking=True,
                    ),
                    self.config.target_speed_classes,
                    brake=brake_label,
                )
                target_speed_labels = decode_two_hot(
                    target_speed_distribution,
                    self.config.target_speed_classes,
                    self.device,
                )
                log.update(
                    {
                        "metric/target_speed_error": torch.mean(
                            torch.abs(
                                predictions.pred_target_speed_scalar
                                - target_speed_labels,
                            ),
                        ).item(),
                        "metric/target_speed_correlation": torch.corrcoef(
                            torch.stack(
                                [
                                    predictions.pred_target_speed_scalar,
                                    target_speed_labels,
                                ],
                            ),
                        )[0, 1].item(),
                    },
                )

            if self.config.predict_temporal_spatial_waypoints:
                waypoints_label = data["future_waypoints"].to(
                    self.device,
                    dtype=self.config.torch_float_type,
                    non_blocking=True,
                )[:, : self.config.num_way_points_prediction]
                log.update(
                    {
                        "metric/waypoints_ade": common_utils.average_displacement_error(
                            predictions.pred_future_waypoints,
                            waypoints_label,
                        ),
                        "metric/waypoints_fde": common_utils.final_displacement_error(
                            predictions.pred_future_waypoints,
                            waypoints_label,
                        ),
                    },
                )

                if self.config.use_navsim_data:
                    heading_label = data["future_yaws"].to(
                        self.device,
                        dtype=self.config.torch_float_type,
                        non_blocking=True,
                    )
                    log["metric/heading_ade"] = common_utils.average_displacement_error(
                        predictions.pred_headings,
                        heading_label,
                    )


@beartype
def decode_two_hot(
    two_hot_label: jt.Float[torch.Tensor, "B C"],
    class_values: list[float],
    device: torch.device,
) -> jt.Float[torch.Tensor, " B"]:
    """Decode a two-hot encoded tensor into a scalar representation.

    Args:
        two_hot_label: The two-hot encoded tensor. Must be between 0 and 1 and sum to 1 along the last dimension.
        class_values: List of class values (e.g., target_speeds or throttle_classes).
        device: Device to place tensors on.

    Returns:
        The decoded scalar tensor.
    """
    classes = torch.tensor(
        class_values,
        device=device,
        dtype=two_hot_label.dtype,
    ).unsqueeze(0)
    decoded = (two_hot_label * classes).sum(axis=-1)
    return decoded


@beartype
def encode_two_hot(
    scalar_values: jt.Float[torch.Tensor, " B"],
    class_values: list[float],
    brake: jt.Bool[torch.Tensor, " B"],
) -> jt.Float[torch.Tensor, "B C"]:
    """Encode scalar values into two-hot representation with linear interpolation.

    Args:
        scalar_values: Scalar values to encode (e.g., speeds or throttle values).
        class_values: List of class bin values (e.g., [0.0, 4.0, 8.0, ...] for speeds).
        brake: Optional boolean mask. If provided, positions where True will be encoded as class 0.

    Returns:
        Two-hot encoded distribution.
    """
    assert all(scalar_values >= 0.0)
    target_speeds = torch.tensor(
        class_values,
        dtype=scalar_values.dtype,
        device=scalar_values.device,
    )
    labels = torch.zeros(
        len(scalar_values),
        len(target_speeds),
        dtype=scalar_values.dtype,
        device=scalar_values.device,
    )
    labels[brake, 0] = 1.0
    non_brake = ~brake
    scalars = scalar_values[non_brake]
    last_bin = scalars >= target_speeds[-1]
    labels[non_brake & (scalar_values >= target_speeds[-1]), -1] = 1.0

    # Interpolation between bins
    interp_mask = ~last_bin
    if interp_mask.any():
        interp_speeds = scalars[interp_mask]
        upper_idx = torch.searchsorted(target_speeds, interp_speeds, right=False)
        lower_idx = upper_idx - 1

        lower_val = target_speeds[lower_idx]
        upper_val = target_speeds[upper_idx]

        lower_weight = (upper_val - interp_speeds) / (upper_val - lower_val)
        upper_weight = (interp_speeds - lower_val) / (upper_val - lower_val)

        row_idx = torch.where(non_brake)[0][interp_mask]
        labels[row_idx, lower_idx] = lower_weight
        labels[row_idx, upper_idx] = upper_weight

    return labels


class PlanningContextEncoder(nn.Module):
    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        input_bev_channels: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.config: TrainingConfig = config

        self.num_status_tokens = 0

        if self.config.use_velocity:
            self.num_status_tokens += 1
            self.velocity_encoder = nn.Sequential(
                nn.Linear(1, self.config.transfuser_token_dim),
            )
            logger.info("Using velocity encoder.")

        if self.config.use_acceleration:
            self.num_status_tokens += 1
            self.acceleration_encoder = nn.Sequential(
                nn.Linear(1, self.config.transfuser_token_dim),
            )
            logger.info("Using acceleration encoder.")

        if self.config.use_discrete_command:
            self.num_status_tokens += 1
            self.command_encoder = nn.Sequential(
                nn.Linear(
                    self.config.discrete_command_dim,
                    self.config.transfuser_token_dim,
                ),
            )
            logger.info("Using discrete command encoder.")

        if self.config.use_tp:
            self.num_status_tokens += 1
            self.tp_encoder = nn.Linear(2, config.transfuser_token_dim)
            logger.info("Using target point encoder.")

        if self.config.use_previous_tp:
            self.num_status_tokens += 1
            logger.info("Using previous target point encoder.")

        if self.config.use_next_tp:
            self.num_status_tokens += 1
            logger.info("Using next target point encoder.")

        if self.config.use_past_positions:
            self.num_status_tokens += self.config.num_past_samples_used
            logger.info("Using past positions encoder.")
            self.past_positions_encoder = nn.Linear(2, config.transfuser_token_dim)

        if self.config.use_past_speeds:
            self.num_status_tokens += self.config.num_past_samples_used
            logger.info("Using past speeds encoder.")
            self.past_speeds_encoder = nn.Linear(1, config.transfuser_token_dim)

        if (
            self.config.use_radars
            and self.config.radar_detection
            and self.config.use_radar_detection
        ):
            self.num_status_tokens += self.config.num_radar_queries
            self.radar_encoder = nn.Linear(
                self.config.radar_token_dim,
                config.transfuser_token_dim,
            )
            logger.info(
                f"Using radar encoder with {self.config.num_radar_queries} tokens.",
            )

        self.cosine_pos_embeding = PositionEmbeddingSine(
            config,
            self.config.transfuser_token_dim // 2,
            normalize=True,
        )
        self.status_pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_status_tokens, self.config.transfuser_token_dim),
        )

        self.dimension_adapter = nn.Conv2d(
            input_bev_channels,
            self.config.transfuser_token_dim,
            kernel_size=1,
        )
        # P2: encode the soft BEV intent field into planning-context tokens.
        if self.config.use_control_conditioning:
            self.intent_adapter = nn.Conv2d(
                1,
                self.config.transfuser_token_dim,
                kernel_size=1,
            )
        self.reset_parameters()

        self.target_points_normalization_constants = torch.tensor(
            self.config.target_points_normalization_constants,
            device=self.device,
            dtype=self.config.torch_float_type,
        )

    def reset_parameters(self):
        nn.init.uniform_(self.status_pos_embedding)

    @beartype
    def forward(
        self,
        bev_features: jt.Float[torch.Tensor, "B C H W"],
        radar_logits: jt.Float[torch.Tensor, "B Q C"] | None,
        radar_predictions: jt.Float[torch.Tensor, "B Q 4"] | None,
        data: dict,
        log: dict,
        intent: jt.Float[torch.Tensor, "B 1 ih iw"] | None = None,
    ) -> jt.Float[torch.Tensor, "B N D"]:
        """
        Args:
            bev_features: Raw BEV features.
            radar_logits: Radar logits.
            radar_predictions: Radar predictions.
            data: dict
            log: dict
        Returns:
            context_tokens: Output tokens for planning transformer decoder.
        """
        # Load data
        if self.config.use_velocity:
            velocity = (
                data["speed"]
                .reshape(-1, 1)
                .to(self.device, dtype=self.config.torch_float_type)
            )
        if self.config.use_discrete_command:
            command = data["command"].to(
                self.device,
                dtype=self.config.torch_float_type,
            )

        status_tokens = []

        # Encode speed
        if self.config.use_velocity:
            velocity_token = self.velocity_encoder(
                velocity / self.config.max_speed,
            ).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(velocity_token)

        # Encode acceleration
        if self.config.use_acceleration:
            acceleration = (
                data["acceleration"]
                .reshape(-1, 1)
                .to(self.device, dtype=self.config.torch_float_type)
            )
            acceleration_token = self.acceleration_encoder(
                acceleration / self.config.max_acceleration,
            ).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(acceleration_token)

        # Encode command
        if self.config.use_discrete_command:
            command_token = self.command_encoder(command).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(command_token)

        # Encode target point
        if self.config.use_tp:
            target_point = data["target_point"].to(
                self.device,
                dtype=self.config.torch_float_type,
                non_blocking=True,
            )
            target_point = target_point / self.target_points_normalization_constants
            tp_token = self.tp_encoder(target_point).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(tp_token)

        if self.config.use_previous_tp:
            previous_tp = data["target_point_previous"].to(
                self.device,
                dtype=self.config.torch_float_type,
                non_blocking=True,
            )
            previous_tp = previous_tp / self.target_points_normalization_constants
            previous_tp_token = self.tp_encoder(previous_tp).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(previous_tp_token)

        if self.config.use_next_tp:
            next_tp = data["target_point_next"].to(
                self.device,
                dtype=self.config.torch_float_type,
                non_blocking=True,
            )
            next_tp = next_tp / self.target_points_normalization_constants
            next_tp_token = self.tp_encoder(next_tp).reshape(
                -1,
                1,
                self.config.transfuser_token_dim,
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(next_tp_token)

        # Encode radar
        if (
            self.config.use_radars
            and self.config.radar_detection
            and self.config.use_radar_detection
        ):
            radar_token = self.radar_encoder(radar_logits).reshape(
                -1,
                self.config.num_radar_queries,
                self.config.transfuser_token_dim,
            )  # (bs, num_radar_queries, transfuser_token_dim)
            radar_pos_embed = fn.gen_sineembed_for_position(
                fn.unit_normalize_bev_points(
                    radar_predictions[..., [RadarLabels.X, RadarLabels.Y]].reshape(
                        -1,
                        2,
                    ),
                    self.config,
                ),
                self.config.transfuser_token_dim,
            ).reshape(
                radar_token.shape,
            )  # (bs, num_radar_queries, transfuser_token_dim)
            radar_token = (
                radar_token + radar_pos_embed
            )  # (bs, num_radar_queries, transfuser_token_dim)
            status_tokens.append(radar_token)

        # Concatenate status tokens if any
        has_statuses = False
        if len(status_tokens) > 0:
            status_tokens = torch.cat(
                status_tokens,
                dim=1,
            )  # (bs, num_status_tokens, transfuser_token_dim)
            has_statuses = True

        # Process BEV features
        bev_context = self.dimension_adapter(
            bev_features,
        )  # (bs, transfuser_token_dim, height, width)

        # Concatenate and add positional embeddings
        if has_statuses:
            height, width = bev_context.shape[-2:]
            context_tokens = bev_context + self.cosine_pos_embeding(bev_context)
            context_tokens = torch.flatten(
                context_tokens,
                start_dim=2,
            )  # (bs, transfuser_token_dim, height * width)
            context_tokens = torch.permute(
                context_tokens,
                (0, 2, 1),
            )  # (bs, height * width, transfuser_token_dim)

            tokens_to_cat = [context_tokens]

            # P2: intent conditioning -- encode the soft BEV intent field as tokens.
            # Detached so the control loss does not corrupt the intent head.
            if self.config.use_control_conditioning and intent is not None:
                intent_feat = torch.sigmoid(intent.detach().to(bev_context.dtype))
                intent_feat = F.interpolate(
                    intent_feat,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )  # (bs, 1, height, width)
                intent_tokens = self.intent_adapter(intent_feat)
                intent_tokens = intent_tokens + self.cosine_pos_embeding(intent_tokens)
                intent_tokens = torch.flatten(intent_tokens, start_dim=2)
                intent_tokens = torch.permute(
                    intent_tokens,
                    (0, 2, 1),
                )  # (bs, height * width, transfuser_token_dim)
                tokens_to_cat.append(intent_tokens)

            status_tokens = (
                status_tokens + self.status_pos_embedding
            )  # (bs, num_status_tokens, transfuser_token_dim)
            tokens_to_cat.append(status_tokens)
            context_tokens = torch.cat(
                tokens_to_cat,
                dim=1,
            )  # (bs, tokens, transfuser_token_dim)

        return context_tokens


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        config: TrainingConfig,
        num_pos_feats=64,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.config = config
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor: torch.Tensor):
        x = tensor
        bs, _, h, w = x.shape
        not_mask = torch.ones((bs, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * (torch.div(dim_t, 2, rounding_mode="floor")) / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.to(self.config.torch_float_type).contiguous()
