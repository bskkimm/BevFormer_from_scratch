"""BEV grid and pillar reference point generation."""

from __future__ import annotations

import torch


def get_bev_grid_points_2d(bev_h: int, bev_w: int) -> torch.Tensor:
    """Normalized [0,1] xy cell centers of the BEV grid, row-major order."""
    ys, xs = torch.meshgrid(
        (torch.arange(bev_h, dtype=torch.float32) + 0.5) / bev_h,
        (torch.arange(bev_w, dtype=torch.float32) + 0.5) / bev_w,
        indexing="ij",
    )
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)


def get_pillar_reference_points_3d(
    bev_h: int,
    bev_w: int,
    pc_range: tuple[float, float, float, float, float, float],
    num_points_in_pillar: int,
) -> torch.Tensor:
    """Normalized [0,1] xyz points: `num_points_in_pillar` heights per BEV cell."""
    grid_xy = get_bev_grid_points_2d(bev_h, bev_w)  # [Q, 2]
    z_min, z_max = pc_range[2], pc_range[5]
    heights_metric = torch.linspace(z_min, z_max, num_points_in_pillar)
    heights_norm = (heights_metric - z_min) / (z_max - z_min)

    num_query = grid_xy.shape[0]
    xy = grid_xy.unsqueeze(0).expand(num_points_in_pillar, num_query, 2)
    z = heights_norm.view(num_points_in_pillar, 1, 1).expand(num_points_in_pillar, num_query, 1)
    return torch.cat([xy, z], dim=-1)
