"""Lightweight prediction decoding and evaluation.

`evaluate_predictions` here is a sanity metric (greedy center-distance
matching), not the official nuScenes mAP/NDS protocol. Full official
evaluation requires the `nuscenes-devkit`, a submission JSON, and the
complete validation split; that is deferred (see README) as an
integration task orthogonal to the from-scratch model implementation —
`decode_predictions` below is what a future official-eval integration
would build on.
"""

from __future__ import annotations

from typing import Sequence

import torch

from bevformer.models.losses.loss_utils import decode_bbox_predictions

DEFAULT_MAX_NUM = 300
DEFAULT_POST_CENTER_RANGE = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)


def decode_predictions(
    cls_logits: torch.Tensor,
    bbox_preds: torch.Tensor,
    max_num: int = DEFAULT_MAX_NUM,
    post_center_range: Sequence[float] = DEFAULT_POST_CENTER_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NMS-free top-k decode of a single sample's final-layer predictions.

    Args:
        cls_logits: [num_queries, num_classes].
        bbox_preds: [num_queries, 10] encoded boxes.
    Returns:
        boxes: [K, 9] semantic boxes, scores: [K], labels: [K].
    """
    if cls_logits.ndim != 2 or bbox_preds.ndim != 2:
        raise ValueError("Expected unbatched [query, class/box] predictions")
    if len(post_center_range) != 6:
        raise ValueError("post_center_range must contain six values")

    probabilities = cls_logits.float().sigmoid()
    flat_scores = probabilities.reshape(-1)
    count = min(max_num, flat_scores.numel())
    scores, flat_indices = flat_scores.topk(count)
    labels = flat_indices % cls_logits.shape[-1]
    query_indices = torch.div(flat_indices, cls_logits.shape[-1], rounding_mode="floor")
    boxes = decode_bbox_predictions(bbox_preds[query_indices].float())

    lower = boxes.new_tensor(post_center_range[:3])
    upper = boxes.new_tensor(post_center_range[3:])
    keep = ((boxes[:, :3] >= lower) & (boxes[:, :3] <= upper)).all(dim=-1)
    return boxes[keep], scores[keep], labels[keep]


def evaluate_predictions(
    pred_boxes: list[torch.Tensor],
    pred_labels: list[torch.Tensor],
    pred_scores: list[torch.Tensor],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    score_threshold: float = 0.3,
    match_distance: float = 2.0,
) -> dict[str, float]:
    """Greedy same-label center-distance matching, per batch element."""
    total_gt = 0
    total_matched = 0
    center_errors: list[float] = []

    for boxes, labels, scores, gts, gt_lbls in zip(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        keep = scores >= score_threshold
        boxes, labels = boxes[keep], labels[keep]
        used_preds: set[int] = set()
        total_gt += gts.shape[0]

        for gt_idx in range(gts.shape[0]):
            gt_center = gts[gt_idx, :3]
            gt_label = gt_lbls[gt_idx].item()
            best_dist = None
            best_pred_idx = None
            for pred_idx in range(boxes.shape[0]):
                if pred_idx in used_preds or labels[pred_idx].item() != gt_label:
                    continue
                dist = (boxes[pred_idx, :3] - gt_center).norm().item()
                if dist <= match_distance and (best_dist is None or dist < best_dist):
                    best_dist, best_pred_idx = dist, pred_idx
            if best_pred_idx is not None:
                used_preds.add(best_pred_idx)
                total_matched += 1
                center_errors.append(best_dist)

    return {
        "match_rate": total_matched / max(total_gt, 1),
        "mean_center_error": sum(center_errors) / len(center_errors) if center_errors else 0.0,
        "num_gt": float(total_gt),
        "num_matched": float(total_matched),
    }
