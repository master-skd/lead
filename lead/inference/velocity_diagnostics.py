"""JSON-safe diagnostics for B3a velocity-candidate decisions."""

from __future__ import annotations

import numbers
from typing import Any

import torch


def _scalar(value: Any) -> float | int | bool | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"expected scalar tensor, got shape {tuple(value.shape)}")
        return value.detach().cpu().item()
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    raise TypeError(f"unsupported scalar type: {type(value)!r}")


def _single_batch_list(value: torch.Tensor | None) -> list | None:
    if value is None:
        return None
    tensor = value.detach().cpu()
    if tensor.ndim > 0 and tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor.tolist()


def build_velocity_diagnostic_record(
    prediction: Any,
    *,
    step: int,
    current_speed_mps: float,
    safe_threshold: float,
    selection_mode: str,
    final_steer: float,
    final_throttle: float,
    final_brake: float,
    stuck_detector: int,
    force_move_remaining: int,
) -> dict[str, Any] | None:
    """Build one JSON-safe record without changing the planner decision.

    Returns ``None`` when the velocity scorer is disabled. Candidate arrays are
    kept in full so a collision can later be traced to risk filtering versus
    preference ranking without rerunning the model.
    """

    risk_tensor = prediction.velocity_candidate_risk
    if risk_tensor is None:
        return None

    candidate_risk = risk_tensor.detach().float().cpu()
    candidate_valid = prediction.velocity_candidate_valid.detach().bool().cpu()
    candidate_preference = (
        prediction.velocity_candidate_preference.detach().float().cpu()
    )
    candidate_profiles = prediction.velocity_candidate_profiles.detach().float().cpu()
    for name, tensor in (
        ("candidate_risk", candidate_risk),
        ("candidate_valid", candidate_valid),
        ("candidate_preference", candidate_preference),
    ):
        if tensor.ndim != 2 or tensor.shape[0] != 1:
            raise ValueError(f"{name} must have shape [1,M], got {tensor.shape}")
    if candidate_profiles.ndim != 3 or candidate_profiles.shape[0] != 1:
        raise ValueError(
            "candidate_profiles must have shape [1,M,T], got "
            f"{candidate_profiles.shape}"
        )

    risk = candidate_risk[0]
    valid = candidate_valid[0]
    preference = candidate_preference[0]
    profiles = candidate_profiles[0]
    candidate_count = len(risk)
    if not (
        len(valid) == candidate_count
        and len(preference) == candidate_count
        and len(profiles) == candidate_count
    ):
        raise ValueError("velocity diagnostic candidate counts do not match")

    selected_index = int(_scalar(prediction.velocity_selected_index))
    if not 0 <= selected_index < candidate_count:
        raise ValueError(f"selected velocity index {selected_index} is out of range")
    safe = valid & (risk < float(safe_threshold))
    valid_risk = risk[valid]

    return {
        "schema_version": 1,
        "step": int(step),
        "selection_mode": selection_mode,
        "safe_threshold": float(safe_threshold),
        "current_speed_mps": float(current_speed_mps),
        "raw_target_speed_mps": _scalar(prediction.raw_target_speed_scalar),
        "selected_target_speed_mps": _scalar(prediction.pred_target_speed_scalar),
        "selected_index": selected_index,
        "switched": bool(_scalar(prediction.velocity_switched)),
        "fallback": bool(_scalar(prediction.velocity_fallback)),
        "raw_collision_risk": float(_scalar(prediction.velocity_raw_risk)),
        "selected_collision_risk": float(_scalar(prediction.velocity_selected_risk)),
        "selected_preference_logit": float(preference[selected_index]),
        "valid_candidate_count": int(valid.sum()),
        "safe_candidate_count": int(safe.sum()),
        "minimum_valid_collision_risk": float(valid_risk.min()),
        "selected_profile_mps": _single_batch_list(
            prediction.velocity_selected_profile
        ),
        "selected_trajectory_xy_m": _single_batch_list(prediction.pred_trajectory),
        "candidate_profiles_mps": profiles.tolist(),
        "candidate_valid": valid.tolist(),
        "candidate_preference_logits": preference.tolist(),
        "candidate_collision_risk": risk.tolist(),
        "controller": {
            "target_speed_pid_steer": _scalar(prediction.route_steer),
            "target_speed_pid_throttle": _scalar(prediction.target_speed_throttle),
            "target_speed_pid_brake": _scalar(prediction.target_speed_brake),
            "final_steer": float(final_steer),
            "final_throttle": float(final_throttle),
            "final_brake": float(final_brake),
            "stuck_detector": int(stuck_detector),
            "force_move_remaining": int(force_move_remaining),
        },
    }
