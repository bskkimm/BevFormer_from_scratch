"""Evaluation entry point for BEVFormer.

Runs a checkpoint over a dataloader and reports the lightweight sanity
metrics from `bevformer.engine.evaluator` (see its module docstring for
why this is not the official nuScenes mAP/NDS protocol).
"""

from __future__ import annotations

import argparse

import torch

from bevformer.engine.evaluator import decode_predictions, evaluate_predictions
from bevformer.engine.trainer import move_batch_to_device
from train import add_model_args, build_dataloader, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BEVFormer checkpoint.")
    parser.add_argument("--dataroot", default="~/dataset/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--queue-length", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_model_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataloader = build_dataloader(args)
    model = build_model(args).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_pred_boxes, all_pred_labels, all_pred_scores = [], [], []
    all_gt_boxes, all_gt_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["imgs"], batch["img_metas"], batch["can_bus"])
            cls_scores = outputs["cls_scores"][-1]
            bbox_preds = outputs["bbox_preds"][-1]

            for i in range(cls_scores.shape[0]):
                boxes, scores, labels = decode_predictions(cls_scores[i], bbox_preds[i])
                all_pred_boxes.append(boxes)
                all_pred_labels.append(labels)
                all_pred_scores.append(scores)
                all_gt_boxes.append(batch["gt_boxes_3d"][i])
                all_gt_labels.append(batch["gt_labels_3d"][i])

    metrics = evaluate_predictions(
        all_pred_boxes,
        all_pred_labels,
        all_pred_scores,
        all_gt_boxes,
        all_gt_labels,
        score_threshold=args.score_threshold,
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
