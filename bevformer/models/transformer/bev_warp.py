"""Align a previous BEV feature map to the current ego frame via ego motion."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def warp_prev_bev(
    prev_bev: torch.Tensor,
    bev_h: int,
    bev_w: int,
    delta_translation_bev: torch.Tensor,
    delta_yaw: torch.Tensor,
    pc_range: tuple[float, float, float, float, float, float],
) -> torch.Tensor:
    """
    Args:
        prev_bev: [B, bev_h*bev_w, C], row-major (row=y, col=x).
        delta_translation_bev: [B, 2] ego translation (x, y) in meters,
            current frame minus previous frame.
        delta_yaw: [B] ego yaw change in radians, current minus previous.
        pc_range: [xmin, ymin, zmin, xmax, ymax, zmax].
    Returns:
        [B, bev_h*bev_w, C]: prev_bev resampled into the current ego frame.
    """
    batch, _, embed_dims = prev_bev.shape
    device = prev_bev.device
    feature_map = prev_bev.reshape(batch, bev_h, bev_w, embed_dims).permute(0, 3, 1, 2).float()  # [B, C, H, W]

    span_x = pc_range[3] - pc_range[0]
    span_y = pc_range[4] - pc_range[1]
    tx = (2.0 * delta_translation_bev[:, 0].float() / span_x).view(batch, 1, 1)
    ty = (2.0 * delta_translation_bev[:, 1].float() / span_y).view(batch, 1, 1)
    cos = torch.cos(delta_yaw.float()).view(batch, 1, 1)
    sin = torch.sin(delta_yaw.float()).view(batch, 1, 1)

    # The same grid F.affine_grid(theta, align_corners=False) builds for
    # theta = [[cos, -sin, tx], [sin, cos, ty]], computed elementwise in float32:
    # affine_grid is a matmul, which autocast would run in bf16.
    xs = (2.0 * torch.arange(bev_w, device=device, dtype=torch.float32) + 1.0) / bev_w - 1.0
    ys = (2.0 * torch.arange(bev_h, device=device, dtype=torch.float32) + 1.0) / bev_h - 1.0
    y, x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W] output-pixel centers in [-1, 1]
    grid = torch.stack([cos * x - sin * y + tx, sin * x + cos * y + ty], dim=-1)  # [B, H, W, 2]

    warped = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped.permute(0, 2, 3, 1).reshape(batch, bev_h * bev_w, embed_dims).to(prev_bev.dtype)
