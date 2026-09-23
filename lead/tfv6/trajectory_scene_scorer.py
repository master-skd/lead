"""Scene-conditioned multi-head scoring for Path x Velocity trajectories.

The B3a-v1 scorer only saw a selected-route embedding and eight speed values.
This module instead embeds every time-sampled trajectory and cross-attends it to
the same context memory used by LEAD's planner.  Its heads intentionally predict
separate, interpretable attributes rather than making one WTA score carry
imitation, feasibility and safety at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SCORE_HEADS = (
    "collision_free",
    "ttc",
    "progress",
    "comfort",
    "imitation",
)


def compress_planner_scene_tokens(
    scene_tokens: torch.Tensor,
    *,
    spatial_shape: tuple[int, int] = (10, 12),
    pooled_shape: tuple[int, int] = (4, 5),
    has_intent_tokens: bool = True,
) -> torch.Tensor:
    """Spatially pool planner BEV/intent tokens while retaining status tokens.

    PlanningContextEncoder lays its memory out as BEV tokens, optional intent
    tokens, then a short tail of ego-status/radar tokens.  Pooling only the two
    spatial grids keeps cache size manageable without averaging command/status
    tokens into unrelated spatial cells.
    """

    if scene_tokens.ndim != 3:
        raise ValueError("scene_tokens must have shape [B,N,D]")
    height, width = map(int, spatial_shape)
    spatial_count = height * width
    grid_count = 2 if has_intent_tokens else 1
    required = grid_count * spatial_count
    if scene_tokens.shape[1] < required:
        raise ValueError(
            f"scene memory has {scene_tokens.shape[1]} tokens, expected at least {required}"
        )

    pooled = []
    for grid_index in range(grid_count):
        begin = grid_index * spatial_count
        end = begin + spatial_count
        grid = scene_tokens[:, begin:end].transpose(1, 2)
        grid = grid.reshape(scene_tokens.shape[0], scene_tokens.shape[2], height, width)
        grid = F.adaptive_avg_pool2d(grid, pooled_shape)
        pooled.append(grid.flatten(2).transpose(1, 2))
    tail = scene_tokens[:, required:]
    return torch.cat((*pooled, tail), dim=1)


def build_trajectory_candidate_states(
    route: torch.Tensor,
    candidate_velocity: torch.Tensor,
    current_speed: torch.Tensor,
    *,
    interval_s: float = 0.25,
) -> torch.Tensor:
    """Compose a spatial route and interval speeds into time-sampled states.

    Returns ``[x,y,sin(yaw),cos(yaw),v,a]`` for every candidate.  Interpolation
    is vectorized and extrapolates the last route segment, matching offline
    collision-label construction.
    """

    if route.ndim != 3 or route.shape[-1] != 2:
        raise ValueError("route must have shape [B,P,2]")
    if candidate_velocity.ndim != 3 or candidate_velocity.shape[0] != route.shape[0]:
        raise ValueError("candidate_velocity must have shape [B,M,T]")
    if route.shape[1] < 2:
        raise ValueError("route must contain at least two points")
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    current = current_speed.reshape(-1)
    if len(current) != len(route):
        raise ValueError("current speed and route must share batch size")

    origin = torch.zeros_like(route[:, :1])
    path = torch.cat((origin, route), dim=1)
    segment = torch.diff(path, dim=1)
    segment_length = torch.linalg.vector_norm(segment, dim=-1).clamp_min(1e-6)
    cumulative = torch.cat(
        (torch.zeros_like(segment_length[:, :1]), segment_length.cumsum(dim=1)),
        dim=1,
    )
    distance = candidate_velocity.clamp_min(0).cumsum(dim=-1) * float(interval_s)
    batch, candidates, steps = distance.shape
    flat_distance = distance.reshape(batch, -1).contiguous()
    index = torch.searchsorted(cumulative.contiguous(), flat_distance, right=True) - 1
    index = index.clamp(min=0, max=segment.shape[1] - 1)
    start = torch.gather(
        path,
        1,
        index[..., None].expand(-1, -1, 2),
    )
    direction = torch.gather(
        segment,
        1,
        index[..., None].expand(-1, -1, 2),
    )
    length = torch.gather(segment_length, 1, index)
    start_distance = torch.gather(cumulative, 1, index)
    alpha = (flat_distance - start_distance) / length
    position = (start + alpha[..., None] * direction).reshape(
        batch, candidates, steps, 2
    )
    unit_direction = (direction / length[..., None]).reshape(
        batch, candidates, steps, 2
    )
    speed = candidate_velocity
    first_acceleration = (
        2.0 * (speed[..., :1] - current[:, None, None]) / float(interval_s)
    )
    later_acceleration = torch.diff(speed, dim=-1) / float(interval_s)
    acceleration = torch.cat((first_acceleration, later_acceleration), dim=-1)
    return torch.cat(
        (
            position,
            unit_direction[..., 1:2],  # sin(yaw)
            unit_direction[..., 0:1],  # cos(yaw)
            speed[..., None],
            acceleration[..., None],
        ),
        dim=-1,
    )


class TrajectorySceneScorer(nn.Module):
    """Encode full trajectories, interact with scene memory, predict five scores."""

    def __init__(
        self,
        state_dim: int = 6,
        scene_dim: int = 256,
        hidden_dim: int = 256,
        profile_steps: int = 8,
        num_heads: int = 8,
        temporal_layers: int = 2,
        interaction_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.scene_dim = int(scene_dim)
        self.hidden_dim = int(hidden_dim)
        self.profile_steps = int(profile_steps)
        self.num_heads = int(num_heads)
        self.temporal_layers = int(temporal_layers)
        self.interaction_layers = int(interaction_layers)
        self.dropout = float(dropout)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.temporal_position = nn.Parameter(
            torch.zeros(1, self.profile_steps, self.hidden_dim)
        )
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=self.temporal_layers,
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.scene_projection = (
            nn.Identity()
            if self.scene_dim == self.hidden_dim
            else nn.Linear(self.scene_dim, self.hidden_dim)
        )
        interaction_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.interaction = nn.TransformerDecoder(
            interaction_layer,
            num_layers=self.interaction_layers,
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, 1),
                )
                for name in SCORE_HEADS
            }
        )

    @staticmethod
    def normalize_candidate_states(candidate_states: torch.Tensor) -> torch.Tensor:
        """Normalize ``[x,y,sin(yaw),cos(yaw),v,a]`` without losing geometry."""

        scale = candidate_states.new_tensor((64.0, 40.0, 1.0, 1.0, 20.0, 5.0))
        return candidate_states / scale

    def forward(
        self,
        candidate_states: torch.Tensor,
        scene_tokens: torch.Tensor,
        candidate_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return one logit per head and candidate, each shaped ``[B,M]``."""

        if candidate_states.ndim != 4:
            raise ValueError("candidate_states must have shape [B,M,T,F]")
        if scene_tokens.ndim != 3:
            raise ValueError("scene_tokens must have shape [B,N,D]")
        batch, candidates, steps, features = candidate_states.shape
        if scene_tokens.shape[0] != batch:
            raise ValueError("candidate states and scene tokens must share batch size")
        if steps != self.profile_steps or features != self.state_dim:
            raise ValueError(
                f"expected candidate states [...,{self.profile_steps},{self.state_dim}]"
            )
        if scene_tokens.shape[-1] != self.scene_dim:
            raise ValueError("unexpected scene feature dimension")
        if candidate_valid is not None and candidate_valid.shape != (batch, candidates):
            raise ValueError("candidate_valid must have shape [B,M]")

        state = self.normalize_candidate_states(candidate_states)
        state = self.state_encoder(state.reshape(batch * candidates, steps, features))
        state = self.temporal_encoder(state + self.temporal_position)
        # Average all trajectory instants.  Unlike B3a-v1 this retains the full
        # time evolution before pooling, including acceleration/deceleration.
        query = state.mean(dim=1).reshape(batch, candidates, self.hidden_dim)
        memory = self.scene_projection(scene_tokens)
        invalid = None if candidate_valid is None else ~candidate_valid.bool()
        hidden = self.interaction(query, memory, tgt_key_padding_mask=invalid)
        return {name: head(hidden).squeeze(-1) for name, head in self.heads.items()}

    def checkpoint_config(self) -> dict[str, float | int]:
        return {
            "state_dim": self.state_dim,
            "scene_dim": self.scene_dim,
            "hidden_dim": self.hidden_dim,
            "profile_steps": self.profile_steps,
            "num_heads": self.num_heads,
            "temporal_layers": self.temporal_layers,
            "interaction_layers": self.interaction_layers,
            "dropout": self.dropout,
        }


def trajectory_score_loss(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    candidate_valid: torch.Tensor,
    *,
    head_weights: dict[str, float] | None = None,
    collision_unsafe_weight: float = 1.0,
    collision_pair_weight: float = 0.0,
    ttc_pair_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked BCE plus within-scene safe-over-colliding candidate ranking."""

    valid = candidate_valid.bool()
    if not valid.any():
        raise ValueError("a scorer batch must contain at least one valid candidate")
    weights = {
        "collision_free": 2.0,
        "ttc": 1.0,
        "progress": 1.0,
        "comfort": 0.5,
        "imitation": 0.25,
    }
    if head_weights is not None:
        weights.update(head_weights)
    losses = {}
    total = logits[next(iter(logits))].new_zeros(())
    for name in SCORE_HEADS:
        if name not in logits or name not in labels:
            raise KeyError(f"missing scorer head/label: {name}")
        target = labels[name].to(logits[name].dtype).clamp(0.0, 1.0)
        per_candidate = F.binary_cross_entropy_with_logits(
            logits[name], target, reduction="none"
        )
        if name == "collision_free" and collision_unsafe_weight != 1.0:
            # Factorized velocity lattices are usually >95% collision-free. Give
            # the rare unsafe counterfactuals enough gradient, then calibrate the
            # operating threshold on held-out routes.
            element_weight = torch.where(
                target < 0.5,
                target.new_tensor(float(collision_unsafe_weight)),
                target.new_ones(()),
            )
            loss = (per_candidate[valid] * element_weight[valid]).sum()
            loss = loss / element_weight[valid].sum().clamp_min(1.0)
        else:
            loss = per_candidate[valid].mean()
        losses[name] = loss
        total = total + float(weights[name]) * loss
    if collision_pair_weight or ttc_pair_weight:
        safe = valid & (labels["collision_free"] >= 0.5)
        unsafe = valid & ~safe
        pairs = safe[:, :, None] & unsafe[:, None, :]
        pair_count = pairs.sum(dim=(1, 2))
        mixed = pair_count > 0
        for name, weight in (
            ("collision_free", collision_pair_weight),
            ("ttc", ttc_pair_weight),
        ):
            if not weight:
                continue
            score = logits[name].float()
            difference = score[:, :, None] - score[:, None, :]
            pair_loss = F.softplus(-difference)
            per_frame = (pair_loss * pairs).sum(dim=(1, 2)) / pair_count.clamp_min(1)
            rank_loss = per_frame[mixed].mean() if mixed.any() else score.sum() * 0.0
            losses[f"{name}_rank"] = rank_loss
            total = total + float(weight) * rank_loss
    losses["total"] = total
    return total, losses


@dataclass(frozen=True)
class TrajectoryScoreSelection:
    selected_index: torch.Tensor
    scores: dict[str, torch.Tensor]
    feasible: torch.Tensor
    fallback: torch.Tensor


@torch.inference_mode()
def select_scene_scored_trajectory(
    logits: dict[str, torch.Tensor],
    candidate_valid: torch.Tensor,
    *,
    collision_free_threshold: float = 0.5,
    ttc_threshold: float = 0.5,
) -> TrajectoryScoreSelection:
    """Filter by safety heads, then rank progress/comfort with weak imitation."""

    scores = {name: torch.sigmoid(logits[name].float()) for name in SCORE_HEADS}
    valid = candidate_valid.bool()
    feasible = (
        valid
        & (scores["collision_free"] >= float(collision_free_threshold))
        & (scores["ttc"] >= float(ttc_threshold))
    )
    has_feasible = feasible.any(dim=1)
    utility = (
        0.65 * scores["progress"]
        + 0.25 * scores["comfort"]
        + 0.10 * scores["imitation"]
    )
    preferred = utility.masked_fill(~feasible, float("-inf")).argmax(dim=1)
    safety = 0.7 * scores["collision_free"] + 0.3 * scores["ttc"]
    fallback = safety.masked_fill(~valid, float("-inf")).argmax(dim=1)
    selected = torch.where(has_feasible, preferred, fallback)
    return TrajectoryScoreSelection(
        selected_index=selected,
        scores=scores,
        feasible=feasible,
        fallback=~has_feasible,
    )


def load_trajectory_scene_scorer(
    checkpoint: dict, device: torch.device | str = "cpu"
) -> TrajectorySceneScorer:
    scorer = TrajectorySceneScorer(**checkpoint["scorer_config"])
    scorer.load_state_dict(checkpoint["model"])
    return scorer.to(device)


@dataclass(frozen=True)
class TrajectorySceneFeatureArrays:
    scene_tokens: np.ndarray
    candidate_states: np.ndarray
    candidate_valid: np.ndarray
    label_collision_free: np.ndarray
    label_ttc: np.ndarray
    label_progress: np.ndarray
    label_comfort: np.ndarray
    label_imitation: np.ndarray
    collision: np.ndarray
    imitation_error: np.ndarray
    keys: np.ndarray


_SCENE_FEATURE_NAMES = tuple(TrajectorySceneFeatureArrays.__dataclass_fields__)


def load_trajectory_scene_feature_dir(
    path: str | Path,
) -> TrajectorySceneFeatureArrays:
    """Load deterministic B3a-v2 feature shards and reject mixed schemas."""

    shard_paths = sorted(Path(path).glob("velocity_features_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no trajectory-scene feature shards found in {path}")
    parts: dict[str, list[np.ndarray]] = {name: [] for name in _SCENE_FEATURE_NAMES}
    candidate_shape = scene_shape = None
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            for name in _SCENE_FEATURE_NAMES:
                if name not in shard:
                    raise KeyError(
                        f"{shard_path} is missing B3a-v2 field {name}; "
                        "rebuild the scene feature cache"
                    )
                parts[name].append(shard[name])
            current_candidate_shape = shard["candidate_states"].shape[1:]
            current_scene_shape = shard["scene_tokens"].shape[1:]
            if candidate_shape is None:
                candidate_shape = current_candidate_shape
                scene_shape = current_scene_shape
            elif (
                current_candidate_shape != candidate_shape
                or current_scene_shape != scene_shape
            ):
                raise ValueError(f"feature shape mismatch in {shard_path}")
    values = {name: np.concatenate(items, axis=0) for name, items in parts.items()}
    if len(np.unique(values["keys"])) != len(values["keys"]):
        raise ValueError(f"duplicate keys across trajectory feature shards in {path}")
    return TrajectorySceneFeatureArrays(**values)
