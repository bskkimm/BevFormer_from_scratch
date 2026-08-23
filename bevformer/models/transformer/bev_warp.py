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
    feature_map = prev_bev.reshape(batch, bev_h, bev_w, embed_dims).permute(0, 3, 1, 2)  # [B, C, H, W]

    span_x = pc_range[3] - pc_range[0]
    span_y = pc_range[4] - pc_range[1]
    tx = 2.0 * delta_translation_bev[:, 0] / span_x
    ty = 2.0 * delta_translation_bev[:, 1] / span_y

    cos = torch.cos(delta_yaw)
    sin = torch.sin(delta_yaw)
    theta = torch.zeros(batch, 2, 3, dtype=prev_bev.dtype, device=prev_bev.device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, feature_map.shape, align_corners=False)
    warped = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped.permute(0, 2, 3, 1).reshape(batch, bev_h * bev_w, embed_dims)
