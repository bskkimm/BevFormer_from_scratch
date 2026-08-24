"""Training entry point for BEVFormer."""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from bevformer.data.collate import collate_fn
from bevformer.data.nuscenes_dataset import BevFormerNuScenesDataset
from bevformer.engine.trainer import fit
from bevformer.models.backbone.image_backbone import MultiViewImageBackbone
from bevformer.models.bevformer import BEVFormerModel
from bevformer.models.grid_mask import GridMask
from bevformer.models.heads.bevformer_head import BEVFormerHead
from bevformer.models.losses.bevformer_loss import BEVFormerLoss
from bevformer.models.neck.fpn import ImageFPN
from bevformer.models.transformer.decoder import BEVFormerDecoder
from bevformer.models.transformer.encoder import BEVFormerEncoder

PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
NUM_CAMS = 6


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backbone-variant", default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument("--embed-dims", type=int, default=256)
    parser.add_argument("--bev-h", type=int, default=50)
    parser.add_argument("--bev-w", type=int, default=50)
    parser.add_argument("--num-queries", type=int, default=900)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--num-points-in-pillar", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)


def build_model(args: argparse.Namespace) -> BEVFormerModel:
    backbone = MultiViewImageBackbone(variant=args.backbone_variant, pretrained=True, frozen_stages=1)
    neck = ImageFPN(in_channels=(512, 1024, 2048), out_channels=args.embed_dims)
    encoder = BEVFormerEncoder(
        num_layers=args.num_encoder_layers,
        bev_h=args.bev_h,
        bev_w=args.bev_w,
        embed_dims=args.embed_dims,
        pc_range=PC_RANGE,
        num_cams=NUM_CAMS,
        num_heads=args.num_heads,
        num_levels=4,
        num_points_in_pillar=args.num_points_in_pillar,
        num_points_temporal=args.num_points_in_pillar,
        feedforward_dims=args.embed_dims * 2,
    )
    decoder = BEVFormerDecoder(
        embed_dims=args.embed_dims,
        num_queries=args.num_queries,
        num_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        num_points=args.num_points_in_pillar,
        ffn_channels=args.embed_dims * 2,
    )
    head = BEVFormerHead(
        embed_dims=args.embed_dims,
        num_classes=args.num_classes,
        box_dim=10,
        num_decoder_layers=args.num_decoder_layers,
        pc_range=PC_RANGE,
    )
    grid_mask = GridMask()
    return BEVFormerModel(backbone, neck, encoder, decoder, head, grid_mask=grid_mask)


def build_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset = BevFormerNuScenesDataset(
        dataroot=args.dataroot,
        version=args.version,
        queue_length=args.queue_length,
        image_size=(args.image_height, args.image_width),
        pc_range=PC_RANGE,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BEVFormer.")
    parser.add_argument("--dataroot", default="~/dataset/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--queue-length", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=35.0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-out", default="checkpoints/bevformer.pth")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-experiment", default="bevformer-training")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-run-id", default=None)
    parser.add_argument("--mlflow-log-checkpoints", action="store_true")
    add_model_args(parser)
    return parser.parse_args()


def _mlflow_run_params(args: argparse.Namespace, dataset_size: int) -> dict[str, object]:
    return {
        "dataset_size": dataset_size,
        "dataroot": args.dataroot,
        "version": args.version,
        "queue_length": args.queue_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_clip_norm": args.grad_clip_norm,
        "use_amp": args.use_amp,
        "backbone_variant": args.backbone_variant,
        "embed_dims": args.embed_dims,
        "bev_h": args.bev_h,
        "bev_w": args.bev_w,
        "num_queries": args.num_queries,
        "num_classes": args.num_classes,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "num_points_in_pillar": args.num_points_in_pillar,
        "num_heads": args.num_heads,
    }


def start_mlflow_run(args: argparse.Namespace, dataset_size: int):
    """Starts (or resumes) an MLflow run and logs the run's hyperparameters.

    Returns the `mlflow` module (used as a lightweight run handle by the
    caller, matching DETR3D-from-Scratch's train.py convention) or `None`
    if `--mlflow` was not passed.
    """
    if not args.mlflow:
        return None
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow logging requested, but mlflow is not installed.") from exc

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    if args.mlflow_run_id is not None:
        mlflow.start_run(run_id=args.mlflow_run_id)
    else:
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=args.mlflow_run_name)

    mlflow.log_params(_mlflow_run_params(args, dataset_size))
    return mlflow


def log_mlflow_metrics(mlflow_module, metrics: dict[str, float], *, step: int, prefix: str = "") -> None:
    if mlflow_module is None:
        return
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow_module.log_metric(f"{prefix}{name}", float(value), step=step)


def log_mlflow_artifact(mlflow_module, path: str) -> None:
    if mlflow_module is not None and os.path.exists(path):
        mlflow_module.log_artifact(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataloader = build_dataloader(args)
    model = build_model(args).to(device)
    criterion = BEVFormerLoss(num_classes=args.num_classes, pc_range=PC_RANGE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    mlflow_run = start_mlflow_run(args, dataset_size=len(dataloader.dataset))

    def epoch_end_callback(epoch: int, metrics: dict[str, float]) -> None:
        log_mlflow_metrics(mlflow_run, metrics, step=epoch, prefix="train_")

    try:
        fit(
            model,
            criterion,
            dataloader,
            optimizer,
            device,
            epochs=args.epochs,
            grad_clip_norm=args.grad_clip_norm,
            use_amp=args.use_amp,
            epoch_end_callback=epoch_end_callback,
        )
    except Exception:
        if mlflow_run is not None:
            mlflow_run.end_run(status="FAILED")
        raise

    os.makedirs(os.path.dirname(args.checkpoint_out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_out)
    print(f"Saved checkpoint to {args.checkpoint_out}")

    if mlflow_run is not None:
        if args.mlflow_log_checkpoints:
            log_mlflow_artifact(mlflow_run, args.checkpoint_out)
        mlflow_run.end_run()


if __name__ == "__main__":
    main()
