"""BEVFormer detection head: per-decoder-layer classification and box regression."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from bevformer.models.transformer.reference_points import denormalize_reference_points, inverse_sigmoid


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        *,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BEVFormerHead(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_classes: int = 10,
        box_dim: int = 10,
        num_decoder_layers: int = 6,
        pc_range: tuple[float, float, float, float, float, float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.box_dim = box_dim
        self.num_decoder_layers = num_decoder_layers
        self.pc_range = pc_range
        self.reference_points = nn.Linear(embed_dims, 3)
        self.cls_branches = nn.ModuleList(
            [MLP(embed_dims, embed_dims, num_classes, num_layers=3, use_layernorm=True) for _ in range(num_decoder_layers)]
        )
        self.reg_branches = nn.ModuleList(
            [MLP(embed_dims, embed_dims, box_dim, num_layers=3) for _ in range(num_decoder_layers)]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        prior_prob = 0.01
        cls_bias = -math.log((1 - prior_prob) / prior_prob)

        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0.0)

        for cls_branch in self.cls_branches:
            final_linear = cls_branch.net[-1]
            nn.init.constant_(final_linear.bias, cls_bias)

        for reg_branch in self.reg_branches:
            final_linear = reg_branch.net[-1]
            nn.init.constant_(final_linear.bias, 0.0)

    def _encode_box_predictions(self, reg_output: torch.Tensor, reference_points: torch.Tensor) -> torch.Tensor:
        if self.box_dim != 10:
            raise ValueError(f"Expected official-style box_dim=10, got {self.box_dim}")

        ref_logits = inverse_sigmoid(reference_points)
        center_xy_norm = (reg_output[..., 0:2] + ref_logits[..., 0:2]).sigmoid()
        center_z_norm = (reg_output[..., 4:5] + ref_logits[..., 2:3]).sigmoid()

        center_xyz = denormalize_reference_points(
            torch.cat([center_xy_norm, center_z_norm], dim=-1),
            self.pc_range,
        )

        encoded = reg_output.clone()
        encoded[..., 0:1] = center_xyz[..., 0:1]
        encoded[..., 1:2] = center_xyz[..., 1:2]
        encoded[..., 4:5] = center_xyz[..., 2:3]
        return encoded

    def init_reference_points(self, query_pos: torch.Tensor) -> torch.Tensor:
        return self.reference_points(query_pos).sigmoid()

    def predict_reference_points(self, layer_idx: int, layer_q: torch.Tensor) -> torch.Tensor:
        return self.reference_points(layer_q).sigmoid()

    def regress_boxes(self, layer_idx: int, layer_hs: torch.Tensor) -> torch.Tensor:
        return self.reg_branches[layer_idx](layer_hs)

    def classify(self, layer_idx: int, layer_hs: torch.Tensor) -> torch.Tensor:
        return self.cls_branches[layer_idx](layer_hs)

    def forward_single(
        self,
        layer_idx: int,
        layer_hs: torch.Tensor,
        reference_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reg_output = self.regress_boxes(layer_idx, layer_hs)
        cls_score = self.classify(layer_idx, layer_hs)
        bbox_pred = self._encode_box_predictions(reg_output, reference_points)
        return cls_score, bbox_pred

    def forward(self, hs: torch.Tensor, inter_references: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_scores = []
        bbox_preds = []
        for layer_idx, layer_hs in enumerate(hs):
            reference_points = inter_references[layer_idx]
            cls_score, bbox_pred = self.forward_single(layer_idx, layer_hs, reference_points)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return torch.stack(cls_scores), torch.stack(bbox_preds)
