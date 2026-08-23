"""Temporal self-attention: fuse the current BEV query with the previous BEV frame."""

from __future__ import annotations

import torch
import torch.nn as nn

from bevformer.models.transformer.deformable_attention import MultiScaleDeformableAttention
from bevformer.models.transformer.reference_points import get_bev_grid_points_2d


class TemporalSelfAttention(nn.Module):
    """Treats [previous BEV, current BEV] as 2 pseudo-levels of the same
    spatial grid and runs deformable attention with each query's own BEV
    grid location as the shared reference point across both levels."""

    def __init__(self, embed_dims: int, num_heads: int = 8, num_points: int = 4) -> None:
        super().__init__()
        self.deform_attn = MultiScaleDeformableAttention(
            embed_dims, num_heads, num_levels=2, num_points=num_points
        )

    def forward(
        self,
        query: torch.Tensor,
        prev_bev: torch.Tensor | None,
        bev_h: int,
        bev_w: int,
    ) -> torch.Tensor:
        batch, num_query, _ = query.shape
        if prev_bev is None:
            prev_bev = query

        value = torch.cat([prev_bev, query], dim=1)  # levels: [prev, current]
        spatial_shapes = [(bev_h, bev_w), (bev_h, bev_w)]

        grid_points = get_bev_grid_points_2d(bev_h, bev_w).to(device=query.device, dtype=query.dtype)  # [Q, 2]
        num_points = self.deform_attn.num_points
        reference_points = grid_points[None, :, None, None, :].expand(batch, num_query, 2, num_points, 2)

        return self.deform_attn(query, reference_points, value, spatial_shapes)
