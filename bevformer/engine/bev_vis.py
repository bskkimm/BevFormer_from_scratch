"""Render learned BEV features as images, for MLflow artifacts during training.

Each figure shows the front camera plus the BEV grid in metric ego coordinates
(nuScenes LIDAR_TOP frame: +x right, +y forward, so forward is up; ego at the origin) with ground-truth boxes (green) and confident
predictions (orange):
  PCA:              top-3 principal components of the C-dim features -> RGB;
                    cells with similar features share a color
  distinctiveness:  distance of each cell's feature from the average cell. (Not the
                    raw norm: the encoder ends in LayerNorm, so every cell's norm is
                    ~sqrt(C) and a norm map would be flat.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from bevformer.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from bevformer.engine.evaluator import decode_predictions
from bevformer.engine.trainer import move_batch_to_device


def pca_rgb(features: torch.Tensor) -> np.ndarray:
    """[H, W, C] -> [H, W, 3] in [0, 1]: top-3 principal components, 1st-99th percentile scaled.

    Each component's sign is fixed (largest-magnitude projection positive) so colors
    stay comparable between epochs.
    """
    height, width, channels = features.shape
    flat = features.reshape(-1, channels).float()
    flat = flat - flat.mean(dim=0, keepdim=True)
    _, _, v = torch.linalg.svd(flat, full_matrices=False)
    proj = flat @ v[:3].T  # [H*W, 3]
    signs = torch.sign(proj.gather(0, proj.abs().argmax(dim=0, keepdim=True)))
    proj = (proj * signs).cpu().numpy()
    lo, hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
    rgb = np.clip((proj - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    return rgb.reshape(height, width, 3)


def box_corners_xy(box: torch.Tensor) -> np.ndarray:
    """Semantic box [x, y, z, w, l, h, yaw, ...] -> [4, 2] BEV corners (length along yaw)."""
    x, y, _, width, length, _, yaw = (float(v) for v in box[:7])
    local = np.array([[length, width], [length, -width], [-length, -width], [-length, width]]) / 2.0
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return local @ rotation.T + np.array([x, y])


def distinctiveness(features: torch.Tensor) -> np.ndarray:
    """[H, W, C] -> [H, W]: L2 distance of each cell's feature from the mean cell feature."""
    flat = features.float()
    return (flat - flat.reshape(-1, flat.shape[-1]).mean(dim=0)).norm(dim=-1).cpu().numpy()


def camera_rgb(image: torch.Tensor) -> np.ndarray:
    """One camera image [3, H, W] (uint8, or ImageNet-normalized float) -> [H, W, 3] in [0, 1]."""
    if image.dtype == torch.uint8:
        array = image.float() / 255.0
    else:
        array = image.float() * torch.as_tensor(IMAGENET_STD).view(3, 1, 1) + torch.as_tensor(IMAGENET_MEAN).view(3, 1, 1)
    return array.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def _draw_boxes(ax, boxes: torch.Tensor | None, color: str) -> None:
    if boxes is None:
        return
    for box in boxes:
        corners = box_corners_xy(box)
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=0.8))
        front = corners[:2].mean(axis=0)
        ax.plot([float(box[0]), front[0]], [float(box[1]), front[1]], color=color, linewidth=0.8)


def render_bev_figure(
    bev_embed: torch.Tensor,
    bev_h: int,
    bev_w: int,
    pc_range: tuple[float, ...],
    out_path: str | Path,
    gt_boxes: torch.Tensor | None = None,
    pred_boxes: torch.Tensor | None = None,
    title: str = "",
    front_camera: np.ndarray | None = None,
) -> Path:
    """`bev_embed`: [bev_h*bev_w, C] (row-major, row = y) for one sample; boxes are semantic [N, 9]."""
    features = bev_embed.detach().float().reshape(bev_h, bev_w, -1).cpu()
    extent = (pc_range[0], pc_range[3], pc_range[1], pc_range[4])

    num_panels = 3 if front_camera is not None else 2
    fig = Figure(figsize=(6 * num_panels, 6), dpi=110)
    if front_camera is not None:
        ax = fig.add_subplot(1, num_panels, 1)
        ax.imshow(front_camera)
        ax.set_title("Front camera (current frame)")
        ax.axis("off")
    panels = [
        ("BEV features (PCA -> RGB)", pca_rgb(features), None),
        ("Feature distinctiveness (distance from mean cell)", distinctiveness(features), "magma"),
    ]
    for index, (name, image, cmap) in enumerate(panels):
        ax = fig.add_subplot(1, num_panels, num_panels - 1 + index)
        shown = ax.imshow(image, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")
        if cmap is not None:
            fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)
        _draw_boxes(ax, gt_boxes, "lime")
        _draw_boxes(ax, pred_boxes, "orange")
        ax.plot(0, 0, marker="^", color="cyan", markersize=6)  # ego vehicle
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("x (m)  ->  right")
        ax.set_ylabel("y (m)  ->  forward")
        ax.set_title(name)
    fig.suptitle(f"{title}   green = ground truth, orange = predictions")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    return out_path


@torch.no_grad()
def visualize_model_bev(
    model,
    batch: dict,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    pc_range: tuple[float, ...],
    out_path: str | Path,
    title: str = "",
    score_threshold: float = 0.3,
) -> Path:
    """Runs `model` in eval mode on the first sample of `batch` and renders its BEV features."""
    was_training = model.training
    model.eval()
    try:
        moved = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
            outputs = model(moved["imgs"], moved["img_metas"], moved["can_bus"])
        boxes, scores, _ = decode_predictions(outputs["cls_scores"][-1][0], outputs["bbox_preds"][-1][0])
        return render_bev_figure(
            outputs["bev_embed"][0],
            model.encoder.bev_h,
            model.encoder.bev_w,
            pc_range,
            out_path,
            gt_boxes=moved["gt_boxes_3d"][0].cpu(),
            pred_boxes=boxes[scores >= score_threshold].cpu(),
            title=title,
            front_camera=camera_rgb(batch["imgs"][0, -1, 0]),
        )
    finally:
        model.train(was_training)
