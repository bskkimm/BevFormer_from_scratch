"""A single BEVFormer encoder layer: temporal self-attn -> spatial cross-attn -> FFN."""

from __future__ import annotations

import torch
import torch.nn as nn

from bevformer.models.transformer.spatial_cross_attention import SpatialCrossAttention
from bevformer.models.transformer.temporal_self_attention import TemporalSelfAttention


class BEVFormerLayer(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_levels: int,
        num_points_in_pillar: int,
        num_points_temporal: int,
        num_cams: int,
        feedforward_dims: int,
    ) -> None:
        super().__init__()
        self.temporal_self_attn = TemporalSelfAttention(embed_dims, num_heads, num_points_temporal)
        self.spatial_cross_attn = SpatialCrossAttention(
            embed_dims, num_cams, num_levels, num_points_in_pillar, num_heads
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_dims),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_dims, embed_dims),
        )

    def forward(
        self,
        query: torch.Tensor,
        mlvl_feats: list[torch.Tensor],
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        prev_bev: torch.Tensor | None,
        bev_pos: torch.Tensor,
        bev_h: int,
        bev_w: int,
    ) -> torch.Tensor:
        tsa_out = self.temporal_self_attn(query + bev_pos, prev_bev, bev_h, bev_w)
        query = self.norm1(query + tsa_out)

        sca_out = self.spatial_cross_attn(query + bev_pos, mlvl_feats, reference_points_cam, bev_mask)
        query = self.norm2(query + sca_out)

        ffn_out = self.ffn(query)
        query = self.norm3(query + ffn_out)
        return query
