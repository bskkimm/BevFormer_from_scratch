"""Hungarian matching for DETR-style 3D detection training."""

from __future__ import annotations

import torch

from bevformer.models.losses.loss_utils import encode_bbox_targets


def _linear_sum_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    device = cost.device
    return (
        torch.as_tensor(row_ind, dtype=torch.long, device=device),
        torch.as_tensor(col_ind, dtype=torch.long, device=device),
    )


def _box_l1_cost_matrix(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    pred = pred_boxes[:, None, :].expand(-1, gt_boxes.shape[0], -1)
    gt = gt_boxes[None, :, :].expand(pred_boxes.shape[0], -1, -1)
    return (pred - gt).abs().sum(dim=-1)


def _focal_class_cost(
    cls_logits: torch.Tensor,
    gt_labels: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    probs = cls_logits.sigmoid().clamp(min=1e-8, max=1 - 1e-8)
    neg_cost = -(1 - probs).log() * (1 - alpha) * probs.pow(gamma)
    pos_cost = -(probs).log() * alpha * (1 - probs).pow(gamma)
    return pos_cost[:, gt_labels] - neg_cost[:, gt_labels]


class HungarianMatcher3D:
    def __init__(
        self,
        num_classes: int,
        pc_range=None,
        cls_weight: float = 2.0,
        bbox_weight: float = 0.25,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def __call__(
        self,
        cls_logits: torch.Tensor,
        box_preds: torch.Tensor,
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        cls_logits = cls_logits.float()
        box_preds = box_preds.float()
        assignments = []
        for batch_idx in range(cls_logits.shape[0]):
            if gt_boxes[batch_idx].numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=cls_logits.device)
                assignments.append((empty, empty))
                continue

            encoded_gt = encode_bbox_targets(gt_boxes[batch_idx].float(), self.pc_range).to(box_preds.dtype)
            cls_cost = _focal_class_cost(
                cls_logits[batch_idx], gt_labels[batch_idx], alpha=self.alpha, gamma=self.gamma
            )
            bbox_cost = _box_l1_cost_matrix(box_preds[batch_idx, :, :8], encoded_gt[:, :8])
            total_cost = self.cls_weight * cls_cost + self.bbox_weight * bbox_cost
            pred_ids, gt_ids = _linear_sum_assignment(total_cost)
            assignments.append((pred_ids, gt_ids))
        return assignments
