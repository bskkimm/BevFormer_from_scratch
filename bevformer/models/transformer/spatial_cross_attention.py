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
        num_points_per_anchor: int = 2,
    ) -> None:
        """Each of the `num_points_in_pillar` projected pillar anchors gets
        `num_points_per_anchor` learned sampling points (official BEVFormer: 4 x 2 = 8)."""
        super().__init__()
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.num_points_per_anchor = num_points_per_anchor
        self.deform_attn = MultiScaleDeformableAttention(
            embed_dims, num_heads, num_levels, num_points_in_pillar * num_points_per_anchor
        )

    def expand_anchor_points(
        self, ref_points_cam: torch.Tensor, bev_mask_cam: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, Q, D, 2] references and [B, Q, D] validity for one camera ->
        [B, Q, levels, D * P, 2] references and [B, Q, levels, D * P] invalid-masks,
        point k belonging to anchor k % D (MSDeformableAttention3D's layout)."""
        batch, num_query, anchors, _ = ref_points_cam.shape
        points = anchors * self.num_points_per_anchor
        refs = ref_points_cam.repeat(1, 1, self.num_points_per_anchor, 1)
        invalid = (~bev_mask_cam).repeat(1, 1, self.num_points_per_anchor)
        return (
            refs[:, :, None].expand(batch, num_query, self.num_levels, points, 2),
            invalid[:, :, None].expand(batch, num_query, self.num_levels, points),
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

            ref_points_expanded, point_mask = self.expand_anchor_points(reference_points_cam[cam], bev_mask[cam])
            out_cam = self.deform_attn(query, ref_points_expanded, value, spatial_shapes, point_mask=point_mask)
            cam_valid = bev_mask[cam].any(dim=-1).to(query.dtype).unsqueeze(-1)  # [B, Q, 1]

            output_sum = output_sum + out_cam * cam_valid
            weight_sum = weight_sum + cam_valid

        return output_sum / weight_sum.clamp(min=1.0)
