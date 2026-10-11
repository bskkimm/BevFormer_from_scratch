"""Repeatable validation subsets and loss evaluation without optimizer updates."""

from __future__ import annotations

import random
import time
from collections import defaultdict

import torch

from bevformer.engine.trainer import move_batch_to_device


def select_indices(dataset, count: int, seed: int = 0) -> list[int]:
    """Round-robin across shuffled scenes, then frames, with a fixed seed."""
    if count <= 0:
        raise ValueError("sample count must be positive")
    groups = defaultdict(list)
    for index, token in enumerate(dataset.sample_tokens):
        groups[dataset.samples[token]["scene_token"]].append(index)
    rng = random.Random(seed)
    scenes = [groups[key] for key in sorted(groups)]
    rng.shuffle(scenes)
    for scene in scenes:
        rng.shuffle(scene)
    indices = []
    while scenes and len(indices) < count:
        for scene in scenes:
            indices.append(scene.pop())
            if len(indices) == count:
                break
        scenes = [scene for scene in scenes if scene]
    return indices


@torch.no_grad()
def evaluate_loss(model, criterion, dataloader, device, amp_dtype=None):
    was_training = model.training
    model.eval()
    totals = defaultdict(float)
    samples = 0
    try:
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None and device.type == "cuda",
            ):
                outputs = model(batch["imgs"], batch["img_metas"], batch["can_bus"])
            losses = criterion.loss_by_feat(
                outputs["cls_scores"],
                outputs["bbox_preds"],
                batch["gt_boxes_3d"],
                batch["gt_labels_3d"],
            )
            losses["loss"] = sum(
                value for key, value in losses.items() if "loss" in key
            )
            size = batch["imgs"].shape[0]
            for key, value in losses.items():
                if not torch.isfinite(value).all():
                    raise RuntimeError("Encountered non-finite validation loss")
                totals[key] += float(value) * size
            samples += size
            if samples % 16 == 0:
                print(f"Validated {samples} samples", flush=True)
    finally:
        model.train(was_training)
    if not samples:
        raise ValueError("Validation dataset is empty")
    return {
        **{key: value / samples for key, value in totals.items()},
        "samples": samples,
    }


def log_result(client, run_id: str, epoch: int, metrics: dict, elapsed_s: float):
    from mlflow.entities import Metric

    values = {f"val/subset_{key}": float(value) for key, value in metrics.items()}
    values["val/duration_s"] = elapsed_s
    timestamp = int(time.time() * 1000)
    # Client API leaves the training run's lifecycle/status untouched.
    client.log_batch(
        run_id,
        metrics=[Metric(key, value, timestamp, epoch) for key, value in values.items()],
    )
