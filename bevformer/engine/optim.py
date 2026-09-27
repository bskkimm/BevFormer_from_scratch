"""Optimizer and learning-rate schedule matching official BEVFormer-Base.

Official config (projects/configs/bevformer/bevformer_base.py):
  optimizer = AdamW(lr=2e-4, weight_decay=0.01, img_backbone lr_mult=0.1)
  lr_config = CosineAnnealing(by_epoch=True), linear warmup over 500 iters
              from warmup_ratio=1/3, min_lr_ratio=1e-3
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr_mult: float = 0.1,
    fused: bool = False,
) -> torch.optim.AdamW:
    """AdamW with a separate (scaled-lr) group for `model.backbone`; frozen params are skipped."""
    backbone_ids = {id(p) for p in model.backbone.parameters()} if hasattr(model, "backbone") else set()
    backbone, other = [], []
    for param in model.parameters():
        if param.requires_grad:
            (backbone if id(param) in backbone_ids else other).append(param)
    groups = [
        {"name": "backbone", "params": backbone, "lr": lr * backbone_lr_mult},
        {"name": "other", "params": other, "lr": lr},
    ]
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay, fused=fused)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return optimizer


class OfficialLrSchedule:
    """mmcv CosineAnnealing (per epoch) with linear warmup (per optimizer update).

    lr = initial_lr * cosine(epoch) * warmup(update), where
      cosine(e)  = min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + cos(pi * e / max_epochs))
      warmup(it) = 1 - (1 - it / warmup_iters) * (1 - warmup_ratio)   for it < warmup_iters, else 1
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        warmup_iters: int = 500,
        warmup_ratio: float = 1.0 / 3,
        min_lr_ratio: float = 1e-3,
    ) -> None:
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio

    def factor(self, epoch: int, update: int) -> float:
        cosine = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * epoch / self.max_epochs))
        if update < self.warmup_iters:
            cosine *= 1 - (1 - update / self.warmup_iters) * (1 - self.warmup_ratio)
        return cosine

    def apply(self, epoch: int, update: int) -> None:
        """Sets every group's lr for the optimizer step with global index `update` in `epoch` (0-based)."""
        factor = self.factor(epoch, update)
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * factor
