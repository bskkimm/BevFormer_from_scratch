"""Training entry point for BEVFormer."""

from __future__ import annotations

import argparse
import os

import torch

from bevformer.engine.build import (
    AMP_DTYPES,
    PC_RANGE,
    add_data_args,
    add_model_args,
    add_runtime_args,
    build_dataloader,
    build_model,
    configure_runtime,
)
from bevformer.engine.trainer import fit
from bevformer.models.losses.bevformer_loss import BEVFormerLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BEVFormer.")
    add_data_args(parser)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=35.0)
    add_runtime_args(parser)
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
        "split": args.split,
        "queue_length": args.queue_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_clip_norm": args.grad_clip_norm,
        "amp": args.amp,
        "tf32": args.tf32,
        "cudnn_benchmark": args.cudnn_benchmark,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "image_dtype": args.image_dtype,
        "backbone_variant": args.backbone_variant,
        "dcn": args.dcn,
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
    configure_runtime(args)

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
            amp_dtype=AMP_DTYPES[args.amp],
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
