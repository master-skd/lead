"""Frozen-route velocity scorer for the P5 B3a residual vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lead.tfv6.future_collision import interpolate_route_by_distance


class VelocityScorer(nn.Module):
    """Score imitation preference and collision risk for velocity candidates."""

    def __init__(
        self,
        route_feature_dim: int = 256,
        profile_steps: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        speed_scale_mps: float = 20.0,
    ) -> None:
        super().__init__()
        self.route_feature_dim = int(route_feature_dim)
        self.profile_steps = int(profile_steps)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.speed_scale_mps = float(speed_scale_mps)
        input_dim = self.route_feature_dim + self.profile_steps + 2
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.preference_head = nn.Linear(self.hidden_dim, 1)
        self.collision_head = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        route_features: torch.Tensor,
        current_speed: torch.Tensor,
        raw_target_speed: torch.Tensor,
        candidate_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return preference and collision logits with shape ``(B,M)``."""

        if route_features.ndim != 2:
            raise ValueError("route_features must have shape [B,D]")
        if candidate_velocity.ndim != 3:
            raise ValueError("candidate_velocity must have shape [B,M,T]")
        if route_features.shape[0] != candidate_velocity.shape[0]:
            raise ValueError("route features and candidates must share the batch size")
        if route_features.shape[1] != self.route_feature_dim:
            raise ValueError("unexpected route feature dimension")
        if candidate_velocity.shape[2] != self.profile_steps:
            raise ValueError("unexpected velocity profile length")

        batch, candidates, _ = candidate_velocity.shape
        current = current_speed.reshape(batch, 1)
        raw_target = raw_target_speed.reshape(batch, 1)
        route = route_features[:, None].expand(-1, candidates, -1)
        residual = (candidate_velocity - current[:, :, None]) / self.speed_scale_mps
        scalars = torch.stack(
            (
                current.expand(-1, candidates),
                raw_target.expand(-1, candidates),
            ),
            dim=-1,
        ).to(route_features.dtype)
        scalars = scalars / self.speed_scale_mps
        fused = torch.cat((route, residual.to(route.dtype), scalars), dim=-1)
        hidden = self.trunk(fused)
        return (
            self.preference_head(hidden).squeeze(-1),
            self.collision_head(hidden).squeeze(-1),
        )

    def checkpoint_config(self) -> dict[str, float | int]:
        return {
            "route_feature_dim": self.route_feature_dim,
            "profile_steps": self.profile_steps,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "speed_scale_mps": self.speed_scale_mps,
        }


def load_velocity_scorer(
    checkpoint: dict, device: torch.device | str = "cpu"
) -> VelocityScorer:
    scorer = VelocityScorer(**checkpoint["scorer_config"])
    scorer.load_state_dict(checkpoint["model"])
    return scorer.to(device)


@dataclass(frozen=True)
class VelocityGateResult:
    """Outputs of the confidence-preserving B3a velocity-profile gate."""

    target_speed: torch.Tensor
    selected_index: torch.Tensor
    raw_risk: torch.Tensor
    selected_risk: torch.Tensor
    triggered: torch.Tensor
    switched: torch.Tensor
    fallback: torch.Tensor
    candidate_velocity: torch.Tensor
    candidate_valid: torch.Tensor
    preference: torch.Tensor
    risk: torch.Tensor


def compose_route_velocity_trajectory(
    route: torch.Tensor,
    velocity_profile: torch.Tensor,
    interval_s: float,
) -> torch.Tensor:
    """Sample a fixed spatial route at distances from interval-average speeds.

    Use the same route interpolation/extrapolation as the counterfactual collision
    labels.  This trajectory is an inference diagnostic; the existing lateral
    controller still follows the spatial route, not these time-sampled points.
    """

    if route.ndim != 3 or route.shape[-1] != 2:
        raise ValueError("route must have shape [B,P,2]")
    if velocity_profile.ndim != 2 or route.shape[0] != velocity_profile.shape[0]:
        raise ValueError("velocity_profile must have shape [B,T]")
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    distances = velocity_profile.clamp_min(0).cumsum(dim=-1) * float(interval_s)
    positions = [
        interpolate_route_by_distance(
            path.detach().float().cpu().numpy(),
            distance.detach().float().cpu().numpy(),
            extrapolate=True,
        )[0]
        for path, distance in zip(route, distances, strict=True)
    ]
    return torch.as_tensor(np.stack(positions), device=route.device, dtype=route.dtype)


@torch.inference_mode()
def select_velocity_profile(
    scorer: VelocityScorer,
    route_features: torch.Tensor,
    current_speed: torch.Tensor,
    raw_target_speed: torch.Tensor,
    residual_vocabulary: torch.Tensor,
    *,
    safe_threshold: float = 0.5,
    interval_s: float = 0.25,
    max_accel_mps2: float = 1.89,
    max_decel_mps2: float = 4.95,
) -> VelocityGateResult:
    """Always select the preferred safe profile on the confidence-selected path.

    Candidate zero (the raw model target) competes on equal terms.  If no valid
    candidate is below the risk threshold, choose the least-risk valid candidate;
    this fallback is reported explicitly.  The longitudinal controller receives
    the speed at the end of the first profile interval, reconstructed from its
    interval-average speed and the measured current speed.
    """

    if not 0.0 < safe_threshold <= 1.0:
        raise ValueError("expected 0 < safe_threshold <= 1")
    current = current_speed.reshape(-1)
    raw_target = raw_target_speed.reshape(-1)
    candidates, valid = build_velocity_candidates(
        current,
        raw_target,
        residual_vocabulary,
        interval_s=interval_s,
        max_accel_mps2=max_accel_mps2,
        max_decel_mps2=max_decel_mps2,
    )
    preference, collision_logit = scorer(
        route_features.to(torch.float16).float(),
        current.to(torch.float16).float(),
        raw_target.to(torch.float16).float(),
        candidates.to(torch.float16).float(),
    )
    risk = torch.sigmoid(collision_logit.float())
    safe = valid & (risk < float(safe_threshold))
    has_safe = safe.any(dim=1)
    preferred = preference.float().masked_fill(~safe, float("-inf")).argmax(dim=1)
    least_risk = risk.masked_fill(~valid, float("inf")).argmin(dim=1)
    selected_index = torch.where(has_safe, preferred, least_risk)
    rows = torch.arange(len(selected_index), device=selected_index.device)
    selected_profile = candidates[rows, selected_index]
    # v_avg=(v_start+v_end)/2 under the first-interval constant-acceleration
    # approximation already used by the vocabulary's reachability check.
    near_target = (2.0 * selected_profile[:, 0] - current).clamp_min(0.0)
    return VelocityGateResult(
        target_speed=near_target.reshape_as(raw_target_speed),
        selected_index=selected_index,
        raw_risk=risk[:, 0],
        selected_risk=risk[rows, selected_index],
        triggered=torch.ones_like(has_safe),
        switched=selected_index != 0,
        fallback=~has_safe,
        candidate_velocity=candidates,
        candidate_valid=valid,
        preference=preference,
        risk=risk,
    )


def _raw_target_velocity_profile(
    current_speed: torch.Tensor,
    target_speed: torch.Tensor,
    profile_steps: int,
    interval_s: float,
    max_accel_mps2: float,
    max_decel_mps2: float,
) -> torch.Tensor:
    """Match ``future_collision.speed_profile_distances`` in torch.

    The scorer was trained with interval-average speeds recovered from that
    distance profile.  Keeping the construction here identical avoids a
    train/closed-loop representation shift for candidate zero.
    """

    if profile_steps < 1:
        raise ValueError("profile_steps must be positive")
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    current = current_speed.reshape(-1).clamp_min(0.0)
    target = target_speed.reshape(-1).clamp_min(0.0)
    if current.shape != target.shape:
        raise ValueError("current and target speeds must share the batch size")
    horizon_s = float(profile_steps) * float(interval_s)
    delta = target - current
    acceleration = (delta / horizon_s).clamp(
        min=-float(max_decel_mps2), max=float(max_accel_mps2)
    )
    moving = acceleration.abs() >= 1e-8
    safe_acceleration = torch.where(moving, acceleration, torch.ones_like(acceleration))
    time_to_target = torch.where(
        moving,
        (delta / safe_acceleration).clamp_min(0.0),
        torch.zeros_like(delta),
    )
    times = torch.arange(
        1,
        profile_steps + 1,
        device=current.device,
        dtype=current.dtype,
    ) * float(interval_s)
    accelerating_time = torch.minimum(times[None], time_to_target[:, None])
    distance = (
        current[:, None] * accelerating_time
        + 0.5 * acceleration[:, None] * accelerating_time.square()
        + target[:, None] * (times[None] - time_to_target[:, None]).clamp_min(0.0)
    )
    distance = torch.where(moving[:, None], distance, current[:, None] * times[None])
    distance = torch.cummax(distance, dim=-1).values
    zeros = torch.zeros_like(distance[:, :1])
    return torch.diff(torch.cat((zeros, distance), dim=-1), dim=-1) / float(interval_s)


def build_velocity_candidates(
    current_speed: torch.Tensor,
    raw_target_speed: torch.Tensor,
    residual_vocabulary: torch.Tensor,
    *,
    interval_s: float = 0.25,
    max_accel_mps2: float = 1.89,
    max_decel_mps2: float = 4.95,
    full_profile_reachability: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build raw + residual-vocabulary profiles exactly as in feature extraction."""

    if residual_vocabulary.ndim != 2:
        raise ValueError("residual vocabulary must have shape [K,T]")
    if not torch.isfinite(residual_vocabulary).all():
        raise ValueError("residual vocabulary must be finite")
    current = current_speed.reshape(-1).clamp_min(0.0)
    raw_target = raw_target_speed.reshape(-1).clamp_min(0.0)
    if current.shape != raw_target.shape:
        raise ValueError("current and raw target speeds must share the batch size")
    vocabulary = residual_vocabulary.to(device=current.device, dtype=current.dtype)
    profile_steps = int(vocabulary.shape[1])
    raw = _raw_target_velocity_profile(
        current,
        raw_target,
        profile_steps,
        interval_s,
        max_accel_mps2,
        max_decel_mps2,
    )
    residual = (current[:, None, None] + vocabulary[None]).clamp_min(0.0)
    first_acceleration = (
        2.0 * (residual[:, :, 0] - current[:, None]) / float(interval_s)
    )
    # The old vocabulary check only constrained the first 250 ms interval.  It
    # therefore admitted profiles with physically impossible jumps later in the
    # horizon.  Consecutive interval-average speeds provide a conservative proxy
    # for acceleration after that first interval; candidate zero is retained as
    # an unconditional fallback because its target-reaching interval can be
    # shorter than the fixed sampling period.
    later_acceleration = torch.diff(residual, dim=-1) / float(interval_s)
    first_valid = (first_acceleration >= -float(max_decel_mps2) - 1e-6) & (
        first_acceleration <= float(max_accel_mps2) + 1e-6
    )
    later_valid = (
        (later_acceleration >= -float(max_decel_mps2) - 1e-6)
        & (later_acceleration <= float(max_accel_mps2) + 1e-6)
    ).all(dim=-1)
    residual_valid = (
        first_valid & later_valid if full_profile_reachability else first_valid
    )
    candidate_velocity = torch.cat((raw[:, None], residual), dim=1)
    candidate_valid = torch.cat(
        (
            torch.ones((len(current), 1), device=current.device, dtype=torch.bool),
            residual_valid,
        ),
        dim=1,
    )
    return candidate_velocity, candidate_valid


@torch.inference_mode()
def apply_velocity_scorer_gate(
    scorer: VelocityScorer,
    route_features: torch.Tensor,
    current_speed: torch.Tensor,
    raw_target_speed: torch.Tensor,
    residual_vocabulary: torch.Tensor,
    *,
    safe_threshold: float = 0.5,
    unsafe_threshold: float = 0.8,
    interval_s: float = 0.25,
    max_accel_mps2: float = 1.89,
    max_decel_mps2: float = 4.95,
) -> VelocityGateResult:
    """Keep the raw speed unless risky, then select a safe preferred profile.

    The spatial route is deliberately not an input/output of this function.  When a
    profile is selected, its terminal speed becomes the scalar target for LEAD's
    existing PID.  Closed-loop replanning repeatedly refreshes that rolling 2 s goal.
    """

    if not 0.0 <= safe_threshold < unsafe_threshold <= 1.0:
        raise ValueError("expected 0 <= safe_threshold < unsafe_threshold <= 1")
    current = current_speed.reshape(-1)
    raw_target = raw_target_speed.reshape(-1)
    candidates, valid = build_velocity_candidates(
        current,
        raw_target,
        residual_vocabulary,
        interval_s=interval_s,
        max_accel_mps2=max_accel_mps2,
        max_decel_mps2=max_decel_mps2,
    )
    # Frozen features were persisted as float16 before scorer training.  Reproduce
    # that quantization online so calibrated collision thresholds see the same input
    # distribution; keep the unquantized candidates for the PID target below.
    scorer_route = route_features.to(torch.float16).float()
    scorer_current = current.to(torch.float16).float()
    scorer_raw_target = raw_target.to(torch.float16).float()
    scorer_candidates = candidates.to(torch.float16).float()
    preference, collision_logit = scorer(
        scorer_route,
        scorer_current,
        scorer_raw_target,
        scorer_candidates,
    )
    risk = torch.sigmoid(collision_logit.float())
    alternative = valid & (risk < float(safe_threshold))
    alternative[:, 0] = False
    has_alternative = alternative.any(dim=1)
    alternative_index = (
        preference.float().masked_fill(~alternative, float("-inf")).argmax(dim=1)
    )
    triggered = risk[:, 0] >= float(unsafe_threshold)
    switched = triggered & has_alternative
    selected_index = torch.where(
        switched, alternative_index, torch.zeros_like(alternative_index)
    )
    rows = torch.arange(len(selected_index), device=selected_index.device)
    selected_risk = risk[rows, selected_index]
    selected_terminal_speed = candidates[rows, selected_index, -1]
    target_speed = torch.where(switched, selected_terminal_speed, raw_target)
    return VelocityGateResult(
        target_speed=target_speed.reshape_as(raw_target_speed),
        selected_index=selected_index,
        raw_risk=risk[:, 0],
        selected_risk=selected_risk,
        triggered=triggered,
        switched=switched,
        fallback=triggered & ~has_alternative,
        candidate_velocity=candidates,
        candidate_valid=valid,
        preference=preference,
        risk=risk,
    )


@dataclass(frozen=True)
class VelocityFeatureArrays:
    route_features: np.ndarray
    current_speed: np.ndarray
    raw_target_speed: np.ndarray
    candidate_velocity: np.ndarray
    candidate_valid: np.ndarray
    collision: np.ndarray
    ttc: np.ndarray
    imitation_error: np.ndarray
    imitation_target: np.ndarray
    raw_route_ade: np.ndarray
    selected_arm: np.ndarray
    multi: np.ndarray
    keys: np.ndarray


_VELOCITY_FEATURE_NAMES = tuple(VelocityFeatureArrays.__dataclass_fields__)


def load_velocity_feature_dir(path: str | Path) -> VelocityFeatureArrays:
    """Load deterministic, non-overlapping velocity-feature shards."""

    shard_paths = sorted(Path(path).glob("velocity_features_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no velocity feature shards found in {path}")
    parts: dict[str, list[np.ndarray]] = {name: [] for name in _VELOCITY_FEATURE_NAMES}
    candidate_shape = None
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            for name in _VELOCITY_FEATURE_NAMES:
                if name not in shard:
                    raise KeyError(f"{shard_path} is missing {name}")
                parts[name].append(shard[name])
            shape = shard["candidate_velocity"].shape[1:]
            if candidate_shape is None:
                candidate_shape = shape
            elif shape != candidate_shape:
                raise ValueError(f"candidate shape mismatch in {shard_path}")
    values = {name: np.concatenate(items, axis=0) for name, items in parts.items()}
    if len(np.unique(values["keys"])) != len(values["keys"]):
        raise ValueError(f"duplicate keys across velocity feature shards in {path}")
    return VelocityFeatureArrays(**values)
