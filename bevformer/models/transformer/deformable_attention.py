"""Pure-PyTorch generalized multi-scale deformable attention.

Generalizes the standard Deformable DETR formulation by allowing the
reference point to vary per (level, point) instead of only per level —
this lets one module serve both spatial cross-attention (where each
"point" is a different pillar height, projected to a different image
location) and temporal self-attention (where "levels" are pseudo-levels
for different time steps).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_MASKED_LOGIT = -1e4


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, embed_dims: int, num_heads: int, num_levels: int, num_points: int) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError("embed_dims must be divisible by num_heads")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dims // num_heads

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.sampling_offsets.bias)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: list[tuple[int, int]],
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, Q, C]
            reference_points: [B, Q, num_levels, num_points, 2] normalized [0,1] xy.
            value: [B, S, C] where S = sum(H_l * W_l).
            spatial_shapes: list of (H, W) per level, in the order `value` was flattened.
            point_mask: optional [B, Q, num_levels, num_points] bool, True = invalid/masked.
        Returns:
            [B, Q, C]
        """
        batch, num_query = query.shape[:2]
        num_value = value.shape[1]

        value_proj = self.value_proj(value).view(batch, num_value, self.num_heads, self.head_dim)

        offsets = self.sampling_offsets(query).view(
            batch, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        raw_weights = self.attention_weights(query).view(
            batch, num_query, self.num_heads, self.num_levels * self.num_points
        )
        if point_mask is not None:
            mask = point_mask[:, :, None, :, :].expand(
                batch, num_query, self.num_heads, self.num_levels, self.num_points
            ).reshape(batch, num_query, self.num_heads, self.num_levels * self.num_points)
            raw_weights = raw_weights.masked_fill(mask, _MASKED_LOGIT)
        attention_weights = F.softmax(raw_weights, dim=-1).view(
            batch, num_query, self.num_heads, self.num_levels, self.num_points
        )

        offset_normalizer = torch.tensor(
            [[w, h] for h, w in spatial_shapes], dtype=query.dtype, device=query.device
        )  # [num_levels, 2]
        sampling_locations = (
            reference_points[:, :, None, :, :, :]
            + offsets / offset_normalizer[None, None, None, :, None, :]
        )  # [B, Q, num_heads, num_levels, num_points, 2]

        output = _multi_scale_deformable_attention_core(
            value_proj, spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)


def _multi_scale_deformable_attention_core(
    value: torch.Tensor,
    spatial_shapes: list[tuple[int, int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    batch, _, num_heads, head_dim = value.shape
    _, num_query, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1  # [0,1] -> [-1,1]

    sampled_per_level = []
    for level, (h, w) in enumerate(spatial_shapes):
        # [B, H*W, heads, head_dim] -> [B*heads, head_dim, H, W]
        value_level = (
            value_list[level].flatten(2).transpose(1, 2).reshape(batch * num_heads, head_dim, h, w)
        )
        # [B, Q, heads, points, 2] -> [B*heads, Q, points, 2]
        grid_level = (
            sampling_grids[:, :, :, level].transpose(1, 2).reshape(batch * num_heads, num_query, num_points, 2)
        )
        sampled = F.grid_sample(
            value_level, grid_level, mode="bilinear", padding_mode="zeros", align_corners=False
        )  # [B*heads, head_dim, Q, points]
        sampled_per_level.append(sampled)

    # [B*heads, head_dim, Q, levels, points]
    sampled = torch.stack(sampled_per_level, dim=-2)
    weights = attention_weights.transpose(1, 2).reshape(batch * num_heads, 1, num_query, num_levels, num_points)
    output = (sampled * weights).sum(dim=(-1, -2))  # [B*heads, head_dim, Q]
    output = output.view(batch, num_heads * head_dim, num_query).transpose(1, 2)  # [B, Q, C]
    return output
