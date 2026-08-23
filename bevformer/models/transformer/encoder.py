"""BEVFormer encoder: stacks encoder layers to produce the BEV feature grid."""

from __future__ import annotations

import torch
import torch.nn as nn

from bevformer.models.transformer.bev_warp import warp_prev_bev
from bevformer.models.transformer.encoder_layer import BEVFormerLayer
from bevformer.models.transformer.point_sampling import project_pillar_points_to_cameras
from bevformer.models.transformer.positional_encoding import LearnedBEVPositionalEncoding
from bevformer.models.transformer.reference_points import get_pillar_reference_points_3d


class BEVFormerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        bev_h: int,
        bev_w: int,
        embed_dims: int,
        pc_range: tuple[float, float, float, float, float, float],
        num_cams: int,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points_in_pillar: int = 4,
        num_points_temporal: int = 4,
        feedforward_dims: int = 512,
    ) -> None:
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar

        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
        self.positional_encoding = LearnedBEVPositionalEncoding(bev_h, bev_w, embed_dims)
        self.layers = nn.ModuleList(
            [
                BEVFormerLayer(
                    embed_dims,
                    num_heads,
                    num_levels,
                    num_points_in_pillar,
                    num_points_temporal,
                    num_cams,
                    feedforward_dims,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        mlvl_feats: list[torch.Tensor],
        img_metas: list[dict],
        prev_bev: torch.Tensor | None = None,
        delta_translation_bev: torch.Tensor | None = None,
        delta_yaw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = mlvl_feats[0].shape[0]
        device = mlvl_feats[0].device

        query = self.bev_embedding.weight.unsqueeze(0).expand(batch, -1, -1).to(device)
        bev_pos = self.positional_encoding(batch).to(device)

        if prev_bev is not None:
            prev_bev = warp_prev_bev(
                prev_bev, self.bev_h, self.bev_w, delta_translation_bev, delta_yaw, self.pc_range
            )

        reference_points_3d = get_pillar_reference_points_3d(
            self.bev_h, self.bev_w, self.pc_range, self.num_points_in_pillar
        ).to(device)
        reference_points_cam, bev_mask = project_pillar_points_to_cameras(
            reference_points_3d, self.pc_range, img_metas
        )
        reference_points_cam = reference_points_cam.to(device)
        bev_mask = bev_mask.to(device)

        for layer in self.layers:
            query = layer(
                query, mlvl_feats, reference_points_cam, bev_mask, prev_bev, bev_pos, self.bev_h, self.bev_w
            )
        return query
