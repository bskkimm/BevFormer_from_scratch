"""Project BEV pillar reference points into camera image planes."""

from __future__ import annotations

import torch


def project_pillar_points_to_cameras(
    reference_points_3d: torch.Tensor,
    pc_range: tuple[float, float, float, float, float, float],
    img_metas: list[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Projects normalized [0,1] pillar points into every camera.

    Args:
        reference_points_3d: [D, Q, 3] normalized xyz.
        pc_range: [xmin, ymin, zmin, xmax, ymax, zmax].
        img_metas: list of length B, each with "lidar2img" (list of num_cam
            4x4 matrices) and "image_size" (height, width).

    Returns:
        reference_points_cam: [num_cam, B, Q, D, 2] normalized [0,1] image xy.
        bev_mask: [num_cam, B, Q, D] bool validity.
    """
    device = reference_points_3d.device
    dtype = torch.float32
    num_points_in_pillar, num_query, _ = reference_points_3d.shape
    batch = len(img_metas)
    num_cam = len(img_metas[0]["lidar2img"])

    xyz_min = torch.tensor(pc_range[:3], dtype=dtype, device=device)
    xyz_max = torch.tensor(pc_range[3:], dtype=dtype, device=device)
    points_metric = reference_points_3d.to(dtype) * (xyz_max - xyz_min) + xyz_min  # [D, Q, 3]
    points_homo = torch.cat([points_metric, torch.ones_like(points_metric[..., :1])], dim=-1)
    points_flat = points_homo.reshape(num_points_in_pillar * num_query, 4)  # [D*Q, 4]

    reference_points_cam = torch.zeros(num_cam, batch, num_query, num_points_in_pillar, 2, dtype=dtype, device=device)
    bev_mask = torch.zeros(num_cam, batch, num_query, num_points_in_pillar, dtype=torch.bool, device=device)

    for b, meta in enumerate(img_metas):
        lidar2img = torch.stack(
            [torch.as_tensor(m, dtype=dtype, device=device) for m in meta["lidar2img"]], dim=0
        )  # [num_cam, 4, 4]
        image_h, image_w = meta["image_size"]

        proj = torch.einsum("cij,pj->cpi", lidar2img, points_flat)  # [num_cam, D*Q, 4]
        depth = proj[..., 2]
        xy = proj[..., :2] / depth.clamp(min=1e-5).unsqueeze(-1)
        norm_x = xy[..., 0] / image_w
        norm_y = xy[..., 1] / image_h
        valid = (
            (depth > 1e-5)
            & (norm_x >= 0.0)
            & (norm_x <= 1.0)
            & (norm_y >= 0.0)
            & (norm_y <= 1.0)
        )

        points_cam = torch.stack([norm_x, norm_y], dim=-1).reshape(
            num_cam, num_points_in_pillar, num_query, 2
        )
        reference_points_cam[:, b] = points_cam.permute(0, 2, 1, 3)
        bev_mask[:, b] = valid.reshape(num_cam, num_points_in_pillar, num_query).permute(0, 2, 1)

    return reference_points_cam, bev_mask
