"""BEVFormer training loss: Hungarian matching + focal cls loss + L1 box loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bevformer.models.losses.loss_utils import encode_bbox_targets
from bevformer.models.losses.matcher import HungarianMatcher3D


class BEVFormerLoss:
    def __init__(
        self,
        num_classes: int,
        pc_range: tuple[float, float, float, float, float, float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        code_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2),
        matcher_cls_weight: float = 2.0,
        matcher_bbox_weight: float = 0.25,
        loss_cls_weight: float = 2.0,
        loss_bbox_weight: float = 0.25,
        use_auxiliary_losses: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0,
        bg_cls_weight: float = 0.0,
    ) -> None:
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.use_auxiliary_losses = use_auxiliary_losses
        self.code_weights = torch.tensor(code_weights, dtype=torch.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.bg_cls_weight = bg_cls_weight
        self.matcher = HungarianMatcher3D(
            num_classes=num_classes,
            pc_range=pc_range,
            cls_weight=matcher_cls_weight,
            bbox_weight=matcher_bbox_weight,
            alpha=alpha,
            gamma=gamma,
        )

    def _loss_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cls_scores = cls_scores.float()
        bbox_preds = bbox_preds.float()
        batch_size, num_queries, _ = cls_scores.shape
        assignments = self.matcher(cls_scores, bbox_preds, gt_boxes, gt_labels)

        label_indices = torch.full(
            (batch_size, num_queries), fill_value=self.num_classes, dtype=torch.long, device=cls_scores.device
        )
        bbox_targets = torch.zeros_like(bbox_preds)
        bbox_weights = torch.zeros_like(bbox_preds)

        num_pos = 0
        for batch_idx, (pred_ids, gt_ids) in enumerate(assignments):
            if pred_ids.numel() == 0:
                continue
            label_indices[batch_idx, pred_ids] = gt_labels[batch_idx][gt_ids]
            encoded_gt = encode_bbox_targets(gt_boxes[batch_idx].float()[gt_ids], self.pc_range).to(bbox_targets.dtype)
            bbox_targets[batch_idx, pred_ids] = encoded_gt
            bbox_weights[batch_idx, pred_ids] = 1.0
            num_pos += pred_ids.numel()

        num_total = batch_size * num_queries
        num_neg = num_total - num_pos
        normalizer = max(float(num_pos), 1.0)
        cls_avg_factor = max(float(num_pos) + self.bg_cls_weight * float(num_neg), 1.0)

        labels = torch.zeros((batch_size, num_queries, self.num_classes), dtype=cls_scores.dtype, device=cls_scores.device)
        pos_mask = label_indices != self.num_classes
        if pos_mask.any():
            batch_ids, query_ids = pos_mask.nonzero(as_tuple=True)
            labels[batch_ids, query_ids, label_indices[batch_ids, query_ids]] = 1.0

        pred_sigmoid = cls_scores.sigmoid()
        pt = pred_sigmoid * labels + (1 - pred_sigmoid) * (1 - labels)
        focal_weight = (self.alpha * labels + (1 - self.alpha) * (1 - labels)) * (1 - pt).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(cls_scores, labels, reduction="none")
        loss_cls = (bce * focal_weight).sum() / cls_avg_factor * self.loss_cls_weight

        code_weights = self.code_weights.to(bbox_preds.device).view(1, 1, -1)
        abs_diff = (bbox_preds - bbox_targets).abs() * bbox_weights * code_weights
        loss_bbox = abs_diff.sum() / normalizer * self.loss_bbox_weight

        return {"loss_cls": loss_cls, "loss_bbox": loss_bbox}

    def loss_by_feat(
        self,
        all_cls_scores: torch.Tensor,
        all_bbox_preds: torch.Tensor,
        batch_gt_boxes: list[torch.Tensor],
        batch_gt_labels: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.use_auxiliary_losses:
            return self._loss_single(all_cls_scores[-1], all_bbox_preds[-1], batch_gt_boxes, batch_gt_labels)

        losses = [
            self._loss_single(all_cls_scores[layer_idx], all_bbox_preds[layer_idx], batch_gt_boxes, batch_gt_labels)
            for layer_idx in range(all_cls_scores.shape[0])
        ]

        output = {"loss_cls": losses[-1]["loss_cls"], "loss_bbox": losses[-1]["loss_bbox"]}
        for layer_idx in range(len(losses) - 1):
            output[f"d{layer_idx}.loss_cls"] = losses[layer_idx]["loss_cls"]
            output[f"d{layer_idx}.loss_bbox"] = losses[layer_idx]["loss_bbox"]
        return output
