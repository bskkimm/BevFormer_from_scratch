"""BEVFormer object-query decoder over the BEV feature grid."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from bevformer.models.transformer.decoder_layer import BEVFormerDecoderLayer


class BEVFormerDecoder(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_queries: int = 900,
        num_layers: int = 6,
        num_heads: int = 8,
        num_points: int = 4,
        ffn_channels: int = 512,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, embed_dims)
        self.query_pos = nn.Embedding(num_queries, embed_dims)
        self.layers = nn.ModuleList(
            [BEVFormerDecoderLayer(embed_dims, num_heads, num_points, ffn_channels) for _ in range(num_layers)]
        )

    def init_decoder_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        return query, query_pos

    def forward(
        self,
        bev_embed: torch.Tensor,
        bev_h: int,
        bev_w: int,
        init_reference_fn: Callable[[torch.Tensor], torch.Tensor],
        refine_reference_fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Official BEVFormer decoding with box refinement.

        Layer 0 attends at `init_reference_fn(query_pos)`; after each layer the
        reference becomes `refine_reference_fn(layer, hidden, reference)` -- the
        center of that layer's box -- detached before the next layer. Returns hidden
        states [L, B, Q, C], the initial reference [B, Q, 3], and the reference each
        layer attended at [L, B, Q, 3] (which that layer's boxes are relative to).
        """
        batch = bev_embed.shape[0]
        query, query_pos = self.init_decoder_state(batch, bev_embed.device)
        spatial_shapes = [(bev_h, bev_w)]

        init_reference = reference_points = init_reference_fn(query_pos)
        intermediate_states = []
        intermediate_refs = []
        hidden = query
        for layer_idx, layer in enumerate(self.layers):
            intermediate_refs.append(reference_points)
            hidden = layer(hidden, query_pos, reference_points[..., :2], bev_embed, spatial_shapes)
            intermediate_states.append(hidden)
            reference_points = refine_reference_fn(layer_idx, hidden, reference_points).detach()

        return torch.stack(intermediate_states), init_reference, torch.stack(intermediate_refs)
