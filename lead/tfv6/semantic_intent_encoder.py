"""One learned semantic vector per Qwen-Drive route intent."""

from __future__ import annotations

import torch
from torch import nn


class SemanticIntentEncoder(nn.Module):
    """Encode maneuver, obligations, scene rules and navigation to ``[B,K,D]``.

    IDs and multi-hot tensors are produced by ``semantic_intent_slots``.  These
    embeddings encode *content*, not a learned query for each slot index.
    """

    def __init__(self, dim: int, *, maneuver_classes: int = 6, obligations: int = 6):
        super().__init__()
        self.maneuver = nn.Embedding(maneuver_classes, dim, padding_idx=0)
        self.obligations = nn.Linear(obligations, dim, bias=False)
        self.signal = nn.Embedding(5, dim)
        self.rule_flags = nn.ModuleList(nn.Embedding(3, dim) for _ in range(3))
        self.navigation = nn.Linear(6, dim, bias=False)
        self.project = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(
        self,
        maneuver_ids: torch.Tensor,
        obligation_multi_hot: torch.Tensor,
        scene_rule_ids: torch.Tensor,
        navigation_one_hot: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if maneuver_ids.ndim != 2:
            raise ValueError("maneuver_ids must be [B,K]")
        batch, arms = maneuver_ids.shape
        if obligation_multi_hot.shape != (batch, arms, self.obligations.in_features):
            raise ValueError("obligation_multi_hot must be [B,K,O]")
        if scene_rule_ids.shape != (batch, 4):
            raise ValueError("scene_rule_ids must be [B,4]")
        if navigation_one_hot.shape != (batch, 6) or valid.shape != (batch, arms):
            raise ValueError("navigation or valid mask shape mismatch")

        hidden = self.maneuver(maneuver_ids.long())
        hidden = hidden + self.obligations(obligation_multi_hot.to(hidden.dtype))
        scene = self.signal(scene_rule_ids[:, 0].long())
        for offset, embedding in enumerate(self.rule_flags, 1):
            scene = scene + embedding(scene_rule_ids[:, offset].long())
        scene = scene + self.navigation(navigation_one_hot.to(hidden.dtype))
        hidden = hidden + scene[:, None, :]
        return self.project(hidden) * valid.unsqueeze(-1).to(hidden.dtype)
