"""B2c': preserve the selected route branch and reduce speed under collision risk."""

from __future__ import annotations

import torch

from lead.tfv6.collision_cost import collision_cost_per_point
from lead.training.config_training import TrainingConfig


def selected_route_collision_risk(
    route: torch.Tensor,
    bev_semantic_logits: torch.Tensor,
    config: TrainingConfig,
) -> torch.Tensor:
    """Max predicted danger over the near portion of the selected route, shape ``(B,)``."""
    point_cost = collision_cost_per_point(
        route.unsqueeze(1),
        bev_semantic_logits.argmax(dim=1),
        config,
        sigma_m=float(config.collision_sigma_m),
    )[:, 0]
    near = max(1, min(int(config.route_speed_gate_near_points), route.shape[1]))
    return point_cost[:, :near].amax(dim=1)


def collision_risk_to_speed_factor(
    risk: torch.Tensor,
    low_threshold: float,
    high_threshold: float,
    minimum_factor: float = 0.0,
) -> torch.Tensor:
    """Piecewise-linear risk-to-speed factor.

    Risk at or below ``low_threshold`` leaves speed unchanged. Risk at or above
    ``high_threshold`` uses ``minimum_factor``; values between interpolate linearly.
    """
    if not 0.0 <= minimum_factor <= 1.0:
        raise ValueError("minimum_factor must be in [0, 1]")
    if high_threshold <= low_threshold:
        raise ValueError("high_threshold must be greater than low_threshold")
    progress = ((risk - low_threshold) / (high_threshold - low_threshold)).clamp(0.0, 1.0)
    return 1.0 - progress * (1.0 - minimum_factor)


def apply_collision_speed_gate(
    target_speed: torch.Tensor,
    risk: torch.Tensor,
    config: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gated target speed and the applied multiplicative factor."""
    factor = collision_risk_to_speed_factor(
        risk,
        low_threshold=float(config.route_speed_gate_low_threshold),
        high_threshold=float(config.route_speed_gate_high_threshold),
        minimum_factor=float(config.route_speed_gate_minimum_factor),
    )
    return target_speed * factor.reshape_as(target_speed), factor
