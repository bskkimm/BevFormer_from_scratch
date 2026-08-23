"""Training loop for BEVFormerModel."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import torch


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "imgs": batch["imgs"].to(device, non_blocking=True),
        "img_metas": batch["img_metas"],
        "can_bus": batch["can_bus"].to(device, non_blocking=True),
        "gt_boxes_3d": [boxes.to(device, non_blocking=True) for boxes in batch["gt_boxes_3d"]],
        "gt_labels_3d": [labels.to(device, non_blocking=True) for labels in batch["gt_labels_3d"]],
    }


def train_one_epoch(
    model,
    criterion,
    dataloader,
    optimizer,
    device: torch.device,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    running: dict[str, float] = defaultdict(float)
    amp_enabled = use_amp and device.type == "cuda"
    num_batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(batch["imgs"], batch["img_metas"], batch["can_bus"])
        loss_dict = criterion.loss_by_feat(
            outputs["cls_scores"], outputs["bbox_preds"], batch["gt_boxes_3d"], batch["gt_labels_3d"]
        )
        loss = sum(value for name, value in loss_dict.items() if "loss" in name)
        if not torch.isfinite(loss):
            raise RuntimeError("Encountered non-finite loss during training.")

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        running["loss"] += float(loss.detach().cpu())
        for name, value in loss_dict.items():
            running[name] += float(value.detach().cpu())
        num_batches += 1

    metrics = {name: total / max(num_batches, 1) for name, total in running.items()}
    metrics["lr"] = float(optimizer.param_groups[0]["lr"])
    return metrics


def fit(
    model,
    criterion,
    dataloader,
    optimizer,
    device: torch.device,
    epochs: int,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
    log_every_epoch: bool = True,
    start_epoch: int = 0,
    epoch_end_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    for epoch in range(start_epoch, start_epoch + epochs):
        metrics = train_one_epoch(
            model, criterion, dataloader, optimizer, device, grad_clip_norm, use_amp, scaler
        )
        metrics["epoch"] = float(epoch + 1)
        history.append(metrics)

        if log_every_epoch:
            summary = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items() if name != "epoch")
            print(f"epoch={epoch + 1}/{start_epoch + epochs} {summary}")

        if epoch_end_callback is not None:
            epoch_end_callback(epoch + 1, metrics)

    return history
