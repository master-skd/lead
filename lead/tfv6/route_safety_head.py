"""Small frozen-feature risk head used by P5 B2d.

The head predicts dynamic collision risk for each valid multimodal route arm.  Future
actor trajectories are privileged training labels only; inference consumes the pooled
planner route token and the two ego-speed scalars available in closed loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class RouteSafetyHead(nn.Module):
    def __init__(
        self,
        route_feature_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        speed_scale_mps: float = 20.0,
    ) -> None:
        super().__init__()
        self.route_feature_dim = int(route_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.speed_scale_mps = float(speed_scale_mps)
        self.net = nn.Sequential(
            nn.LayerNorm(self.route_feature_dim + 2),
            nn.Linear(self.route_feature_dim + 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        route_features: torch.Tensor,
        current_speed: torch.Tensor,
        target_speed: torch.Tensor,
    ) -> torch.Tensor:
        """Return collision-risk logits with shape ``(...,)``.

        ``route_features`` may be ``(N,D)`` or ``(B,K,D)``.  Speeds may omit the
        arm dimension; they are broadcast across K in that case.
        """

        prefix = route_features.shape[:-1]
        current_speed = self._broadcast_speed(current_speed, prefix)
        target_speed = self._broadcast_speed(target_speed, prefix)
        speed = torch.stack((current_speed, target_speed), dim=-1)
        speed = speed.to(route_features.dtype) / self.speed_scale_mps
        return self.net(torch.cat((route_features, speed), dim=-1)).squeeze(-1)

    @staticmethod
    def _broadcast_speed(speed: torch.Tensor, prefix: torch.Size) -> torch.Tensor:
        speed = speed.reshape(speed.shape + (1,) * max(0, len(prefix) - speed.ndim))
        return torch.broadcast_to(speed, prefix)

    def checkpoint_config(self) -> dict[str, float | int]:
        return {
            "route_feature_dim": self.route_feature_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "speed_scale_mps": self.speed_scale_mps,
        }


def load_route_safety_head(checkpoint: dict, device: torch.device | str = "cpu") -> RouteSafetyHead:
    head = RouteSafetyHead(**checkpoint["head_config"])
    head.load_state_dict(checkpoint["model"])
    return head.to(device)


def binary_auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Tie-correct binary AUROC without a scikit-learn dependency."""

    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    positives = int(label.sum())
    negatives = int((~label).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    ranks = np.arange(1, len(score) + 1, dtype=np.float64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1, len(score)]
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        ranks[start:end] = ranks[start:end].mean()
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum = original_ranks[label].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def binary_average_precision(score: np.ndarray, label: np.ndarray) -> float:
    """Average precision at every positive rank (descending score)."""

    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    positives = int(label.sum())
    if positives == 0:
        return float("nan")
    ordered = label[np.argsort(-score, kind="stable")]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / positives)


@dataclass(frozen=True)
class SafetyFeatureArrays:
    route_features: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    collision: np.ndarray
    ttc: np.ndarray
    ade: np.ndarray
    current_speed: np.ndarray
    target_speed: np.ndarray
    keys: np.ndarray


def load_safety_feature_dir(path: str) -> SafetyFeatureArrays:
    """Load and concatenate deterministic B2d feature shards."""

    from pathlib import Path

    shard_paths = sorted(Path(path).glob("safety_features_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no safety feature shards found in {path}")
    names = (
        "route_features", "confidence", "valid", "collision", "ttc", "ade",
        "current_speed", "target_speed", "keys",
    )
    parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            for name in names:
                parts[name].append(shard[name])
    values = {name: np.concatenate(value, axis=0) for name, value in parts.items()}
    if len(np.unique(values["keys"])) != len(values["keys"]):
        raise ValueError(f"duplicate keys across safety feature shards in {path}")
    return SafetyFeatureArrays(**values)
