"""Deterministic single-batch overfit regression check.

Trains BEVFormerModel repeatedly on one fixed synthetic batch and reports
the loss trajectory. A healthy implementation should drive the loss down
substantially within a few hundred steps -- this is a fast (CPU-friendly,
no dataset required), strong sanity check that gradients actually flow end
to end and the model is capable of learning, independent of whether the
real nuScenes training run has been attempted yet.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from bevformer.models.backbone.image_backbone import MultiViewImageBackbone
from bevformer.models.bevformer import BEVFormerModel
from bevformer.models.heads.bevformer_head import BEVFormerHead
from bevformer.models.losses.bevformer_loss import BEVFormerLoss
from bevformer.models.neck.fpn import ImageFPN
from bevformer.models.transformer.decoder import BEVFormerDecoder
from bevformer.models.transformer.encoder import BEVFormerEncoder

PC_RANGE = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)


def build_tiny_model(embed_dims: int, num_cams: int, num_classes: int, num_queries: int) -> BEVFormerModel:
    backbone = MultiViewImageBackbone(variant="resnet50", pretrained=False, frozen_stages=-1)
    neck = ImageFPN(in_channels=(512, 1024, 2048), out_channels=embed_dims)
    encoder = BEVFormerEncoder(
        num_layers=1,
        bev_h=4,
        bev_w=4,
        embed_dims=embed_dims,
        pc_range=PC_RANGE,
        num_cams=num_cams,
        num_heads=2,
        num_levels=4,
        num_points_in_pillar=2,
        num_points_temporal=2,
        feedforward_dims=embed_dims * 2,
    )
    decoder = BEVFormerDecoder(
        embed_dims=embed_dims, num_queries=num_queries, num_layers=2, num_heads=2, num_points=2, ffn_channels=embed_dims * 2
    )
    head = BEVFormerHead(embed_dims=embed_dims, num_classes=num_classes, box_dim=10, num_decoder_layers=2, pc_range=PC_RANGE)
    return BEVFormerModel(backbone, neck, encoder, decoder, head)


def build_fixed_batch(batch: int, queue_length: int, num_cams: int, image_size: int) -> dict:
    generator = torch.Generator().manual_seed(0)
    imgs = torch.randn(batch, queue_length, num_cams, 3, image_size, image_size, generator=generator)
    img_metas = [
        [
            {"lidar2img": [np.eye(4, dtype=np.float32) for _ in range(num_cams)], "image_size": (image_size, image_size)}
            for _ in range(queue_length)
        ]
        for _ in range(batch)
    ]
    can_bus = torch.zeros(batch, queue_length, 18)
    gt_boxes_3d = [torch.tensor([[1.0, 2.0, 0.0, 2.0, 4.0, 1.5, 0.1, 0.0, 0.0]]) for _ in range(batch)]
    gt_labels_3d = [torch.tensor([1]) for _ in range(batch)]
    return {
        "imgs": imgs,
        "img_metas": img_metas,
        "can_bus": can_bus,
        "gt_boxes_3d": gt_boxes_3d,
        "gt_labels_3d": gt_labels_3d,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dims", type=int, default=16)
    parser.add_argument("--num-cams", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--queue-length", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    model = build_tiny_model(args.embed_dims, args.num_cams, args.num_classes, args.num_queries)
    criterion = BEVFormerLoss(num_classes=args.num_classes, pc_range=PC_RANGE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batch = build_fixed_batch(1, args.queue_length, args.num_cams, args.image_size)

    model.train()
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["imgs"], batch["img_metas"], batch["can_bus"])
        loss_dict = criterion.loss_by_feat(
            outputs["cls_scores"], outputs["bbox_preds"], batch["gt_boxes_3d"], batch["gt_labels_3d"]
        )
        loss = sum(value for name, value in loss_dict.items() if "loss" in name)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"step={step:4d} loss={losses[-1]:.4f}")

    print(f"\nfirst loss={losses[0]:.4f}  last loss={losses[-1]:.4f}  "
          f"reduction={100 * (1 - losses[-1] / losses[0]):.1f}%")


if __name__ == "__main__":
    main()
