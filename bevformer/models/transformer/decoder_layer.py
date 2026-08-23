"""Decoder layer for BEVFormer: self-attn among queries + deformable BEV cross-attn."""

from __future__ import annotations

import torch
import torch.nn as nn

from bevformer.models.transformer.deformable_attention import MultiScaleDeformableAttention


class BEVFormerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        ffn_channels: int = 512,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.cross_attn = MultiScaleDeformableAttention(embed_dims, num_heads, num_levels=1, num_points=num_points)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_channels),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_channels, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points_xy: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        q = query + query_pos
        self_attended, _ = self.self_attn(q, q, query)
        query = self.norm1(query + self_attended)

        num_points = self.cross_attn.num_points
        reference_points = reference_points_xy[:, :, None, None, :].expand(
            reference_points_xy.shape[0], reference_points_xy.shape[1], 1, num_points, 2
        )
        cross = self.cross_attn(query + query_pos, reference_points, value, spatial_shapes)
        query = self.norm2(query + cross)

        ffn_out = self.ffn(query)
        return self.norm3(query + ffn_out)
