"""Continuous scene tokens -> K latent modes; no corridor or semantic classes."""
import torch
from torch import nn


class HiddenIntentEncoder(nn.Module):
    def __init__(self, hidden_dim, dim, modes=6, heads=4):
        super().__init__()
        self.adapter = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim), nn.LayerNorm(dim))
        # 0/1/2 = front/front-left/front-right; 3 = text. Padding uses -1.
        self.view_embedding = nn.Embedding(4, dim)
        self.navigation = nn.Linear(6, dim, bias=False)
        self.velocity = nn.Linear(1, dim)
        self.mode_queries = nn.Parameter(torch.randn(modes, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, hidden, mask, view_ids, command, speed):
        if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or view_ids.shape != mask.shape:
            raise ValueError('hidden/mask/view_ids must be [B,L,Dq]/[B,L]/[B,L]')
        if hidden.shape[-1] != self.adapter[1].in_features:
            raise ValueError('Qwen hidden dimension differs from trained adapter')
        if mask.dtype != torch.bool or not mask.any(1).all():
            raise ValueError('every sample needs at least one valid Qwen token')
        if command.shape != (hidden.shape[0], 6) or speed.numel() != hidden.shape[0]:
            raise ValueError('invalid navigation/speed batch')
        if ((view_ids[mask] < 0) | (view_ids[mask] > 3)).any():
            raise ValueError('valid tokens require front/left/right/text identities')
        # Mask before projection: even NaN-filled padding must not enter attention.
        hidden = hidden.detach().masked_fill(~mask[..., None], 0)
        if not torch.isfinite(hidden).all():
            raise ValueError('non-finite Qwen features')
        memory = self.adapter(hidden.to(self.adapter[1].weight.dtype))
        memory = memory + self.view_embedding(view_ids.clamp(min=0).long())
        condition = self.navigation(command.to(memory.dtype)) + self.velocity(speed.reshape(-1, 1).to(memory.dtype) / 10)
        query = self.mode_queries[None].expand(hidden.shape[0], -1, -1) + condition[:, None]
        update, _ = self.cross_attention(self.query_norm(query), memory, memory,
                                         key_padding_mask=~mask, need_weights=False)
        latent = query + update
        return self.output_norm(latent + self.ffn(latent))
