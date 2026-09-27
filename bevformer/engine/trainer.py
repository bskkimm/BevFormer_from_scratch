"""Training loop for BEVFormerModel."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import torch

from bevformer.data.transforms import normalize_images


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Moves a collated batch to `device`; uint8 images are normalized after the transfer."""
    imgs = batch["imgs"].to(device, non_blocking=True)
    if imgs.dtype == torch.uint8:
        imgs = normalize_images(imgs)
    return {
        "imgs": imgs,
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
    amp_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
    accumulation_steps: int = 1,
    epoch: int = 0,
    start_update: int = 0,
    lr_schedule: Callable[[int, int], None] | None = None,
    update_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    """One pass over `dataloader`.

    `amp_dtype` (torch.float16 / torch.bfloat16) enables CUDA autocast; None is fp32.
    Gradients are averaged over `accumulation_steps` micro-batches per optimizer
    update (the last window of an epoch may be shorter); `start_update` is the
    global update count so far. Before each update, `lr_schedule(epoch, update)`
    sets the learning rate; after it, `update_callback` receives the update index,
    the window's mean loss, the lr, and epoch progress.
    """
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    model.train()
    running: dict[str, float] = defaultdict(float)
    amp_enabled = amp_dtype is not None and device.type == "cuda"
    use_scaler = scaler is not None and scaler.is_enabled()
    num_micro_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    update = start_update
    num_batches = 0
    window_loss = 0.0
    window_count = 0

    def optimizer_update() -> None:
        nonlocal update, window_loss, window_count
        if lr_schedule is not None:
            lr_schedule(epoch, update)
        if use_scaler and grad_clip_norm is not None:
            scaler.unscale_(optimizer)
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update_callback is not None:
            update_callback({
                "epoch": epoch,
                "update": update,
                "micro_batches_done": num_batches,
                "num_micro_batches": num_micro_batches,
                "loss": window_loss / max(window_count, 1),
                "lr": float(optimizer.param_groups[-1]["lr"]),
            })
        window_loss, window_count = 0.0, 0

    optimizer.zero_grad(set_to_none=True)
    for index, batch in enumerate(dataloader):
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(batch["imgs"], batch["img_metas"], batch["can_bus"])
        loss_dict = criterion.loss_by_feat(
            outputs["cls_scores"], outputs["bbox_preds"], batch["gt_boxes_3d"], batch["gt_labels_3d"]
        )
        loss = sum(value for name, value in loss_dict.items() if "loss" in name)
        if not torch.isfinite(loss):
            raise RuntimeError("Encountered non-finite loss during training.")

        # Average over the micro-batches of this update window (shorter at epoch end).
        window_start = (index // accumulation_steps) * accumulation_steps
        window_size = accumulation_steps
        if num_micro_batches is not None:
            window_size = min(accumulation_steps, num_micro_batches - window_start)
        scaled = loss / window_size
        (scaler.scale(scaled) if use_scaler else scaled).backward()

        loss_value = float(loss.detach().cpu())
        running["loss"] += loss_value
        for name, value in loss_dict.items():
            running[name] += float(value.detach().cpu())
        num_batches += 1
        window_loss += loss_value
        window_count += 1
        if window_count == window_size:
            optimizer_update()

    if window_count:  # unknown-length iterables can end mid-window
        optimizer_update()

    metrics = {name: total / max(num_batches, 1) for name, total in running.items()}
    metrics["lr"] = float(optimizer.param_groups[-1]["lr"])
    metrics["updates"] = float(update - start_update)
    return metrics


def fit(
    model,
    criterion,
    dataloader,
    optimizer,
    device: torch.device,
    epochs: int,
    grad_clip_norm: float | None = None,
    amp_dtype: torch.dtype | None = None,
    log_every_epoch: bool = True,
    start_epoch: int = 0,
    epoch_end_callback: Callable[[int, dict[str, float]], None] | None = None,
    accumulation_steps: int = 1,
    lr_schedule: Callable[[int, int], None] | None = None,
    update_callback: Callable[[dict[str, Any]], None] | None = None,
    start_update: int = 0,
) -> list[dict[str, float]]:
    """Runs epochs `start_epoch .. start_epoch + epochs - 1`; `epoch_end_callback`
    gets the 1-based epoch number, and metrics["global_update"] the update count."""
    history: list[dict[str, float]] = []
    # Only fp16 needs loss scaling; bf16 has fp32's exponent range.
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16 and device.type == "cuda")
    update = start_update

    for epoch in range(start_epoch, start_epoch + epochs):
        metrics = train_one_epoch(
            model, criterion, dataloader, optimizer, device, grad_clip_norm, amp_dtype, scaler,
            accumulation_steps=accumulation_steps, epoch=epoch, start_update=update,
            lr_schedule=lr_schedule, update_callback=update_callback,
        )
        update += int(metrics["updates"])
        metrics["epoch"] = float(epoch + 1)
        metrics["global_update"] = float(update)
        history.append(metrics)

        if log_every_epoch:
            summary = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items() if name != "epoch")
            print(f"epoch={epoch + 1}/{start_epoch + epochs} {summary}")

        if epoch_end_callback is not None:
            epoch_end_callback(epoch + 1, metrics)

    return history
