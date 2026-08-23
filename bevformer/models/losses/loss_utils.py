"""Loss helpers and box parameterization utilities."""

from __future__ import annotations

import torch


def encode_bbox_targets(boxes: torch.Tensor, pc_range=None) -> torch.Tensor:
    """Encode semantic 9D boxes into the 10D training target.

    Input: `[x, y, z, w, l, h, yaw, vx, vy]`
    Output: `[x, y, log(w), log(l), z, log(h), sin(yaw), cos(yaw), vx, vy]`
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 10))
    if boxes.shape[-1] != 9:
        raise ValueError(f"Expected 9D semantic boxes, got {boxes.shape[-1]}")

    x = boxes[..., 0:1]
    y = boxes[..., 1:2]
    z = boxes[..., 2:3]
    w = boxes[..., 3:4].clamp(min=1e-5).log()
    l = boxes[..., 4:5].clamp(min=1e-5).log()
    h = boxes[..., 5:6].clamp(min=1e-5).log()
    yaw = boxes[..., 6:7]
    vx = boxes[..., 7:8]
    vy = boxes[..., 8:9]
    return torch.cat([x, y, w, l, z, h, yaw.sin(), yaw.cos(), vx, vy], dim=-1)


def wrapped_yaw_difference(pred_yaw: torch.Tensor, target_yaw: torch.Tensor) -> torch.Tensor:
    diff = pred_yaw - target_yaw
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def decode_bbox_predictions(box_preds: torch.Tensor, pc_range=None) -> torch.Tensor:
    """Decode predictions into semantic `[x, y, z, w, l, h, yaw, vx, vy]`."""
    if box_preds.shape[-1] in (7, 9):
        return box_preds
    if box_preds.shape[-1] != 10:
        raise ValueError(f"Expected bbox dim 7, 9, or 10, got {box_preds.shape[-1]}")

    x = box_preds[..., 0:1]
    y = box_preds[..., 1:2]
    z = box_preds[..., 4:5]
    w = box_preds[..., 2:3].exp()
    l = box_preds[..., 3:4].exp()
    h = box_preds[..., 5:6].exp()
    yaw = torch.atan2(box_preds[..., 6:7], box_preds[..., 7:8])
    vx = box_preds[..., 8:9]
    vy = box_preds[..., 9:10]
    return torch.cat([x, y, z, w, l, h, yaw, vx, vy], dim=-1)
