"""Batches BevFormerNuScenesDataset samples for a DataLoader."""

from __future__ import annotations

import torch


def collate_fn(batch: list[dict]) -> dict:
    return {
        "imgs": torch.stack([sample["imgs"] for sample in batch], dim=0),
        "can_bus": torch.stack([sample["can_bus"] for sample in batch], dim=0),
        "img_metas": [sample["img_metas"] for sample in batch],
        "gt_boxes_3d": [sample["gt_boxes_3d"] for sample in batch],
        "gt_labels_3d": [sample["gt_labels_3d"] for sample in batch],
    }
