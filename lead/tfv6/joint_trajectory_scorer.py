"""Scene-conditioned Path x executable-speed scorer (B3b).

The seven heads are deliberately separate: GT-derived scores supervise safety,
road containment, command-compatible path, progress, comfort and imitation.
No GT quantity is an input at inference time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lead.tfv6.trajectory_scene_scorer import TrajectorySceneScorer

JOINT_HEADS = (
    "collision_free", "ttc", "drivable", "task", "progress", "comfort", "imitation"
)


class JointTrajectorySceneScorer(TrajectorySceneScorer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Keep the scene/trajectory interaction used by B3a-v2, but replace the
        # heads so geometry and navigation are explicit rather than hidden in WTA.
        self.heads = torch.nn.ModuleDict({
            name: torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                torch.nn.GELU(), torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hidden_dim, 1),
            ) for name in JOINT_HEADS
        })


def joint_score_loss(logits, labels, valid, *, unsafe_weight=8.0):
    """Masked multi-task loss with extra weight on rare collision negatives."""
    weights = {"collision_free": 2.0, "ttc": 1.0, "drivable": 1.5,
               "task": 1.0, "progress": 0.5, "comfort": 0.25,
               "imitation": 0.5}
    losses = {}
    total = next(iter(logits.values())).sum() * 0.0
    if not valid.any():
        raise ValueError("batch has no valid candidates")
    for name in JOINT_HEADS:
        target = labels[name].float().clamp(0, 1)
        per = F.binary_cross_entropy_with_logits(logits[name], target, reduction="none")
        if name == "collision_free":
            weight = torch.where(target < 0.5, unsafe_weight, 1.0)
            loss = (per * weight)[valid].sum() / weight[valid].sum().clamp_min(1)
        else:
            loss = per[valid].mean()
        losses[name] = loss
        total = total + weights[name] * loss
    safe = valid & (labels["collision_free"] >= 0.5)
    unsafe = valid & ~safe
    pairs = safe[:, :, None] & unsafe[:, None, :]
    if pairs.any():
        score = logits["collision_free"]
        rank = F.softplus(-(score[:, :, None] - score[:, None, :]))
        losses["safety_rank"] = rank[pairs].mean()
        total = total + 0.5 * losses["safety_rank"]
    losses["total"] = total
    return total, losses
