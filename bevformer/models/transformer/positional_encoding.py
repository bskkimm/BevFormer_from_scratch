"""Learned factorized 2D positional encoding for the BEV grid."""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedBEVPositionalEncoding(nn.Module):
    def __init__(self, bev_h: int, bev_w: int, embed_dims: int) -> None:
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.row_embed = nn.Embedding(bev_h, embed_dims)
        self.col_embed = nn.Embedding(bev_w, embed_dims)

    def forward(self, batch_size: int) -> torch.Tensor:
        device = self.row_embed.weight.device
        rows = self.row_embed(torch.arange(self.bev_h, device=device))  # [H, C]
        cols = self.col_embed(torch.arange(self.bev_w, device=device))  # [W, C]
        pos = rows[:, None, :] + cols[None, :, :]  # [H, W, C]
        pos = pos.reshape(1, self.bev_h * self.bev_w, -1).expand(batch_size, -1, -1)
        return pos
