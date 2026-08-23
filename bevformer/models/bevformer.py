"""Top-level BEVFormer composition module."""

from __future__ import annotations

import torch
import torch.nn as nn


def _quaternion_to_yaw(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))


class BEVFormerModel(nn.Module):
    """Wires backbone, neck, BEV encoder, decoder, and head into one module."""

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        head: nn.Module,
        grid_mask: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.grid_mask = grid_mask

    def extract_bev_features(
        self,
        imgs: torch.Tensor,
        img_metas: list[dict],
        prev_bev: torch.Tensor | None = None,
        delta_translation_bev: torch.Tensor | None = None,
        delta_yaw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.grid_mask is not None and self.training:
            batch, num_cams, channels, height, width = imgs.shape
            flat = imgs.reshape(batch * num_cams, channels, height, width)
            flat = self.grid_mask(flat)
            imgs = flat.reshape(batch, num_cams, channels, height, width)

        features = self.backbone(imgs)
        pyramid = self.neck(features)
        mlvl_feats = list(pyramid.values())
        return self.encoder(
            mlvl_feats,
            img_metas,
            prev_bev=prev_bev,
            delta_translation_bev=delta_translation_bev,
            delta_yaw=delta_yaw,
        )

    def forward(
        self,
        imgs_queue: torch.Tensor,
        img_metas_queue: list[list[dict]],
        can_bus_queue: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            imgs_queue: [B, T, N, 3, H, W].
            img_metas_queue: list of length B, each a list of length T of dicts
                (matches `bevformer.data.collate.collate_fn`'s "img_metas" nesting).
            can_bus_queue: [B, T, 18].
        """
        batch, queue_length = imgs_queue.shape[:2]

        prev_bev = None
        for t in range(queue_length):
            img_metas_t = [img_metas_queue[b][t] for b in range(batch)]
            if t == 0:
                delta_translation_bev = None
                delta_yaw = None
            else:
                delta_translation_bev = can_bus_queue[:, t, 16:18]
                delta_yaw = _quaternion_to_yaw(can_bus_queue[:, t, 3:7]) - _quaternion_to_yaw(
                    can_bus_queue[:, t - 1, 3:7]
                )

            is_last_frame = t == queue_length - 1
            if is_last_frame:
                bev_embed = self.extract_bev_features(
                    imgs_queue[:, t], img_metas_t, prev_bev, delta_translation_bev, delta_yaw
                )
            else:
                with torch.no_grad():
                    prev_bev = self.extract_bev_features(
                        imgs_queue[:, t], img_metas_t, prev_bev, delta_translation_bev, delta_yaw
                    )

        hidden_states, _init_reference, inter_references = self.decoder(
            bev_embed, self.encoder.bev_h, self.encoder.bev_w, reference_point_predictor=self.head.predict_reference_points
        )
        cls_scores, bbox_preds = self.head(hidden_states, inter_references)
        return {"cls_scores": cls_scores, "bbox_preds": bbox_preds, "bev_embed": bev_embed}
