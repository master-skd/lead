"""Inference-only safety gating for B2 multimodal routes.

The planner confidence remains the preference signal. Safety only vetoes an unsafe
confidence winner when another valid arm passes both the collision and corridor gates.
If no arm passes, selection falls back to the confidence winner instead of inventing an
untrained emergency policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lead.tfv6.collision_cost import (
    collision_cost_per_point,
    corridor_cost_per_point,
)
from lead.training.config_training import TrainingConfig

VALID_SELECTION_MODES = {"legacy", "slot0", "confidence", "safety_rescore"}


def resolve_route_selection_mode(config: TrainingConfig) -> str:
    """Resolve the new explicit mode while preserving old checkpoint configs."""
    mode = str(getattr(config, "route_selection_mode", "legacy")).lower()
    if mode not in VALID_SELECTION_MODES:
        raise ValueError(
            f"Unknown route_selection_mode={mode!r}; expected one of "
            f"{sorted(VALID_SELECTION_MODES)}"
        )
    if mode == "legacy":
        return "confidence" if bool(config.route_select_by_conf) else "slot0"
    return mode


@dataclass
class RouteSelection:
    selected_idx: torch.Tensor
    baseline_idx: torch.Tensor
    safe_mask: torch.Tensor
    switched: torch.Tensor
    fallback: torch.Tensor


def highest_confidence_valid(confidence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Highest-confidence valid arm, with deterministic slot-0 all-padding fallback."""
    masked = confidence.masked_fill(~valid, float("-inf"))
    return torch.where(
        valid.any(dim=1),
        masked.argmax(dim=1),
        torch.zeros_like(confidence[:, 0], dtype=torch.long),
    )


def select_indices_from_costs(
    confidence: torch.Tensor,
    valid: torch.Tensor,
    collision_cost: torch.Tensor,
    off_corridor_cost: torch.Tensor,
    collision_threshold: float,
    corridor_threshold: float,
    preserve_safe_baseline: bool = True,
) -> RouteSelection:
    """Select by confidence subject to two independently sweepable safety gates.

    A safe baseline is deliberately retained even if another arm has lower raw cost:
    confidence encodes route preference, while costs are constraints rather than a new
    scalar reward. When the baseline is unsafe, the highest-confidence safe arm wins.
    """
    if not (
        confidence.shape == valid.shape == collision_cost.shape == off_corridor_cost.shape
    ):
        raise ValueError("confidence, valid and both costs must all have shape (B, K)")
    valid = valid.bool()
    finite = torch.isfinite(collision_cost) & torch.isfinite(off_corridor_cost)
    safe = (
        valid
        & finite
        & (collision_cost <= collision_threshold)
        & (off_corridor_cost <= corridor_threshold)
    )
    baseline = highest_confidence_valid(confidence, valid)
    safe_pick = highest_confidence_valid(confidence, safe)
    ar = torch.arange(confidence.shape[0], device=confidence.device)
    baseline_safe = safe[ar, baseline]
    any_safe = safe.any(dim=1)

    if preserve_safe_baseline:
        selected = torch.where(baseline_safe | ~any_safe, baseline, safe_pick)
    else:
        selected = torch.where(any_safe, safe_pick, baseline)
    return RouteSelection(
        selected_idx=selected,
        baseline_idx=baseline,
        safe_mask=safe,
        switched=selected != baseline,
        fallback=~any_safe,
    )


@dataclass
class RouteSafetyResult:
    route: torch.Tensor
    selection: RouteSelection
    collision_cost: torch.Tensor
    off_corridor_cost: torch.Tensor


def rescore_routes_for_safety(
    routes: torch.Tensor,
    confidence: torch.Tensor,
    anchors: torch.Tensor | None,
    bev_semantic_logits: torch.Tensor,
    intent_logits: torch.Tensor,
    config: TrainingConfig,
) -> RouteSafetyResult:
    """Compute current-scene costs and apply the B2c confidence-with-veto policy."""
    valid = (
        anchors[..., 3].to(confidence.device) > 0.5
        if anchors is not None
        else torch.ones_like(confidence, dtype=torch.bool)
    )
    bev_classes = bev_semantic_logits.argmax(dim=1)
    collision_points = collision_cost_per_point(
        routes,
        bev_classes,
        config,
        sigma_m=float(config.collision_sigma_m),
    )
    near = max(1, min(int(config.route_safety_near_points), routes.shape[2]))
    # A single near-horizon contact can be catastrophic, so do not dilute it by averaging.
    collision = collision_points[:, :, :near].amax(dim=2)

    corridor_points = corridor_cost_per_point(
        routes,
        torch.sigmoid(intent_logits),
        config,
        reach_m=float(config.route_corridor_reach_m),
    )
    off_corridor = corridor_points.mean(dim=2)
    selection = select_indices_from_costs(
        confidence,
        valid,
        collision,
        off_corridor,
        collision_threshold=float(config.route_safety_collision_threshold),
        corridor_threshold=float(config.route_safety_corridor_threshold),
    )
    ar = torch.arange(routes.shape[0], device=routes.device)
    return RouteSafetyResult(
        route=routes[ar, selection.selected_idx],
        selection=selection,
        collision_cost=collision,
        off_corridor_cost=off_corridor,
    )
