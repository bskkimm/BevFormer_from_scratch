"""Render learned BEV features as images, for MLflow artifacts during training.

Each figure has one row per fixed sample: the front camera with ground-truth (green)
and confident predicted (orange) 3D boxes projected onto it, plus the BEV grid in
metric ego coordinates (nuScenes LIDAR_TOP frame: +x right, +y forward, so forward
is up; ego at the origin) with the same boxes from above:
  PCA:              top-3 principal components of the C-dim features -> RGB;
                    cells with similar features share a color
  distinctiveness:  distance of each cell's feature from the average cell. (Not the
                    raw norm: the encoder ends in LayerNorm, so every cell's norm is
                    ~sqrt(C) and a norm map would be flat.)
"""

from __future__ import annotations

from dataclasses import dataclass
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


def box_corners_3d(box: torch.Tensor) -> np.ndarray:
    """Semantic box [x, y, z_center, w, l, h, yaw, ...] -> [8, 3] corners: bottom ring, then top ring."""
    corners_xy = box_corners_xy(box)
    z, height = float(box[2]), float(box[5])
    bottom = np.column_stack([corners_xy, np.full(4, z - height / 2.0)])
    top = np.column_stack([corners_xy, np.full(4, z + height / 2.0)])
    return np.concatenate([bottom, top])


_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def project_box_edges(box: torch.Tensor, lidar2img: np.ndarray, min_depth: float = 0.5) -> list[np.ndarray]:
    """A box's 12 edges as [2, 2] pixel segments, or [] if any corner is behind the camera."""
    corners = np.column_stack([box_corners_3d(box), np.ones(8)]) @ np.asarray(lidar2img, dtype=np.float64).T
    depth = corners[:, 2]
    if (depth < min_depth).any():
        return []
    pixels = corners[:, :2] / depth[:, None]
    return [pixels[[a, b]] for a, b in _BOX_EDGES]


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


def _draw_camera_boxes(ax, boxes: torch.Tensor | None, lidar2img: np.ndarray | None, color: str) -> None:
    if boxes is None or lidar2img is None:
        return
    for box in boxes:
        for segment in project_box_edges(box, lidar2img):
            ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=1.0)


def _draw_boxes(ax, boxes: torch.Tensor | None, color: str) -> None:
    if boxes is None:
        return
    for box in boxes:
        corners = box_corners_xy(box)
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=0.8))
        front = corners[:2].mean(axis=0)
        ax.plot([float(box[0]), front[0]], [float(box[1]), front[1]], color=color, linewidth=0.8)


@dataclass
class BevVisSample:
    """One figure row. `bev_embed`: [bev_h*bev_w, C] (row-major, row = y); boxes are
    semantic [N, 9]; `lidar2img` is the front camera's projection, scaled to `front_camera`."""

    bev_embed: torch.Tensor
    gt_boxes: torch.Tensor | None = None
    pred_boxes: torch.Tensor | None = None
    front_camera: np.ndarray | None = None
    lidar2img: np.ndarray | None = None
    name: str = ""


def render_bev_figure(
    samples: list[BevVisSample],
    bev_h: int,
    bev_w: int,
    pc_range: tuple[float, ...],
    out_path: str | Path,
    title: str = "",
) -> Path:
    """One row per sample: [front camera with projected boxes | BEV PCA | distinctiveness]."""
    extent = (pc_range[0], pc_range[3], pc_range[1], pc_range[4])
    row_height = 5.0
    fig = Figure(figsize=(20, row_height * len(samples)), dpi=110, layout="constrained")
    grid = fig.add_gridspec(len(samples), 3, width_ratios=[1.78, 1.0, 1.15])

    for row, sample in enumerate(samples):
        features = sample.bev_embed.detach().float().reshape(bev_h, bev_w, -1).cpu()
        ax = fig.add_subplot(grid[row, 0])
        if sample.front_camera is not None:
            height, width = sample.front_camera.shape[:2]
            ax.imshow(sample.front_camera)
            _draw_camera_boxes(ax, sample.gt_boxes, sample.lidar2img, "lime")
            _draw_camera_boxes(ax, sample.pred_boxes, sample.lidar2img, "orange")
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
        ax.set_title(f"Front camera   {sample.name}".rstrip())
        ax.axis("off")

        panels = [
            ("BEV features (PCA -> RGB)", pca_rgb(features), None),
            ("Feature distinctiveness (distance from mean cell)", distinctiveness(features), "magma"),
        ]
        for column, (name, image, cmap) in enumerate(panels, start=1):
            ax = fig.add_subplot(grid[row, column])
            shown = ax.imshow(image, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")
            if cmap is not None:
                fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)
            _draw_boxes(ax, sample.gt_boxes, "lime")
            _draw_boxes(ax, sample.pred_boxes, "orange")
            ax.plot(0, 0, marker="^", color="cyan", markersize=6)  # ego vehicle
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            if row == 0:
                ax.set_title(name)
    fig.suptitle(f"{title}   green = ground truth, orange = predictions")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    return out_path


@torch.no_grad()
def visualize_model_bev(
    model,
    batches: dict | list[dict],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    pc_range: tuple[float, ...],
    out_path: str | Path,
    title: str = "",
    score_threshold: float = 0.3,
    names: list[str] | None = None,
) -> Path:
    """Runs `model` in eval mode on the first sample of each batch and renders one row per batch."""
    batches = [batches] if isinstance(batches, dict) else batches
    names = names or [""] * len(batches)
    was_training = model.training
    model.eval()
    try:
        samples = []
        for batch, name in zip(batches, names):
            moved = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
                outputs = model(moved["imgs"], moved["img_metas"], moved["can_bus"])
            boxes, scores, _ = decode_predictions(outputs["cls_scores"][-1][0], outputs["bbox_preds"][-1][0])
            samples.append(
                BevVisSample(
                    bev_embed=outputs["bev_embed"][0].cpu(),
                    gt_boxes=moved["gt_boxes_3d"][0].cpu(),
                    pred_boxes=boxes[scores >= score_threshold].cpu(),
                    front_camera=camera_rgb(batch["imgs"][0, -1, 0]),
                    lidar2img=np.asarray(batch["img_metas"][0][-1]["lidar2img"][0]),
                    name=name,
                )
            )
        return render_bev_figure(samples, model.encoder.bev_h, model.encoder.bev_w, pc_range, out_path, title=title)
    finally:
        model.train(was_training)
