"""Spatial cross-attention: sample multi-camera image features into BEV queries."""

from __future__ import annotations

import torch
import torch.nn as nn

from bevformer.models.transformer.deformable_attention import MultiScaleDeformableAttention


class SpatialCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_cams: int,
        num_levels: int,
        num_points_in_pillar: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.deform_attn = MultiScaleDeformableAttention(
            embed_dims, num_heads, num_levels, num_points_in_pillar
        )

    def forward(
        self,
        query: torch.Tensor,
        mlvl_feats: list[torch.Tensor],
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, Q, C]
            mlvl_feats: list of [B, num_cams, C, H, W], one per level.
            reference_points_cam: [num_cams, B, Q, D, 2] normalized [0,1].
            bev_mask: [num_cams, B, Q, D] bool, True = valid projection.
        Returns:
            [B, Q, C]
        """
        batch, num_query, embed_dims = query.shape
        spatial_shapes = [(feat.shape[-2], feat.shape[-1]) for feat in mlvl_feats]

        output_sum = query.new_zeros(batch, num_query, embed_dims)
        weight_sum = query.new_zeros(batch, num_query, 1)

        for cam in range(self.num_cams):
            value = torch.cat(
                [feat[:, cam].flatten(2).transpose(1, 2) for feat in mlvl_feats], dim=1
            )  # [B, S, C]

            ref_points_cam = reference_points_cam[cam]  # [B, Q, D, 2]
            num_points_in_pillar = ref_points_cam.shape[2]
            ref_points_expanded = ref_points_cam[:, :, None, :, :].expand(
                batch, num_query, self.num_levels, num_points_in_pillar, 2
            )

            invalid_mask = ~bev_mask[cam]  # [B, Q, D]
            point_mask = invalid_mask[:, :, None, :].expand(
                batch, num_query, self.num_levels, num_points_in_pillar
            )

            out_cam = self.deform_attn(query, ref_points_expanded, value, spatial_shapes, point_mask=point_mask)
            cam_valid = bev_mask[cam].any(dim=-1).to(query.dtype).unsqueeze(-1)  # [B, Q, 1]

            output_sum = output_sum + out_cam * cam_valid
            weight_sum = weight_sum + cam_valid

        return output_sum / weight_sum.clamp(min=1.0)
