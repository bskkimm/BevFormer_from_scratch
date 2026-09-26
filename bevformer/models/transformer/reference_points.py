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


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Computed in float32 with (1 - x) clamped separately: under bf16/fp16
    # autocast, 1 - eps rounds to 1.0, so clamping x alone lets x = 1.0 through
    # and x / (1 - x) overflows to inf with a NaN gradient.
    x = x.float().clamp(min=0.0, max=1.0)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def denormalize_reference_points(
    reference_points: torch.Tensor,
    pc_range: tuple[float, float, float, float, float, float],
) -> torch.Tensor:
    pc_range_t = torch.as_tensor(pc_range, dtype=reference_points.dtype, device=reference_points.device)
    xyz_min = pc_range_t[:3]
    xyz_max = pc_range_t[3:]
    return reference_points * (xyz_max - xyz_min) + xyz_min
