"""Training entry point for BEVFormer.

Outputs go to --output-dir (default runs/<timestamp>):
  checkpoints/latest.pth     model + optimizer + resume state, rewritten every epoch
  checkpoints/epoch_XX.pth   model weights after epoch XX
  bev_features/epoch_XX.png  learned-BEV visualizations (also MLflow artifacts)
  final.pth                  model weights after the last epoch
Resume an interrupted run with --resume <output-dir>/checkpoints/latest.pth; it
continues from the last completed epoch, in the same MLflow run.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch

from bevformer.data.collate import collate_fn
from bevformer.data.nuscenes_dataset import BevFormerNuScenesDataset
from bevformer.engine.bev_vis import visualize_model_bev
from bevformer.engine.build import (
    AMP_DTYPES,
    PC_RANGE,
    add_data_args,
    add_model_args,
    add_runtime_args,
    build_dataloader,
    build_model,
    configure_runtime,
    prepare_model,
)
from bevformer.engine.checkpoint import load_checkpoint, save_checkpoint
from bevformer.engine.optim import OfficialLrSchedule, build_optimizer
from bevformer.engine.progress import TrainingProgress
from bevformer.engine.trainer import fit
from bevformer.models.losses.bevformer_loss import BEVFormerLoss


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BEVFormer.")
    add_data_args(parser)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=500, help="linear LR warmup length, in optimizer updates")
    parser.add_argument(
        "--accumulation-steps", type=int, default=8,
        help="micro-batches per optimizer update; official BEVFormer uses 8 GPUs x 1 sample",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=35.0)
    add_runtime_args(parser)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None, help="default: runs/<timestamp>")
    parser.add_argument("--resume", default=None, help="resume from a checkpoints/latest.pth")
    parser.add_argument("--log-every-updates", type=int, default=10, help="MLflow step-metric interval")
    parser.add_argument("--vis-every-epochs", type=int, default=2, help="0 disables BEV visualization")
    parser.add_argument("--vis-split", default="val", help="split of the fixed visualization sample")
    parser.add_argument(
        "--vis-sample-indices", "--vis-sample-index", dest="vis_sample_indices", type=int, nargs="+",
        default=[20, 4020, 5999],
        help="fixed visualization samples, one figure row each (val defaults: parking lot, rainy intersection, night traffic)",
    )
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-experiment", default="bevformer-training")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-run-id", default=None)
    parser.add_argument("--mlflow-log-checkpoints", action="store_true")
    add_model_args(parser)
    return parser.parse_args(argv)


def start_mlflow_run(args: argparse.Namespace, dataset_size: int):
    """Starts (or resumes) an MLflow run and logs every CLI argument as a parameter.

    Returns the `mlflow` module (used as a lightweight run handle by the caller,
    matching DETR3D-from-Scratch's train.py convention) or `None` without --mlflow.
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
        params = {key: str(value) for key, value in vars(args).items()}
        params["dataset_size"] = str(dataset_size)
        params["effective_batch_size"] = str(args.batch_size * args.accumulation_steps)
        mlflow.log_params(params)
    return mlflow


def log_mlflow_metrics(mlflow_module, metrics: dict[str, float], *, step: int, prefix: str = "") -> None:
    if mlflow_module is None:
        return
    values = {f"{prefix}{name}": float(value) for name, value in metrics.items() if isinstance(value, (int, float))}
    if values:
        mlflow_module.log_metrics(values, step=step)


def log_mlflow_artifact(mlflow_module, path: str | Path, artifact_path: str | None = None) -> None:
    if mlflow_module is not None and os.path.exists(path):
        mlflow_module.log_artifact(str(path), artifact_path=artifact_path)


def _visualization_batches(args: argparse.Namespace) -> list[dict] | None:
    if args.vis_every_epochs <= 0:
        return None
    dataset = BevFormerNuScenesDataset(
        dataroot=args.dataroot,
        version=args.version,
        queue_length=args.queue_length,
        image_size=(args.image_height, args.image_width),
        pc_range=PC_RANGE,
        image_dtype=args.image_dtype,
        split=args.vis_split,
    )
    return [collate_fn([dataset[min(index, len(dataset) - 1)]]) for index in args.vis_sample_indices]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    configure_runtime(args)
    output_dir = Path(args.output_dir or f"runs/{datetime.now():%Y%m%d-%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = build_dataloader(args)
    vis_batches = _visualization_batches(args)
    model = prepare_model(build_model(args), args, device)
    criterion = BEVFormerLoss(num_classes=args.num_classes, pc_range=PC_RANGE)
    optimizer = build_optimizer(
        model, args.lr, args.weight_decay, args.backbone_lr_mult, fused=args.fused_adamw and device.type == "cuda"
    )
    schedule = OfficialLrSchedule(optimizer, max_epochs=args.epochs, warmup_iters=args.warmup_iters)
    amp_dtype = AMP_DTYPES[args.amp]

    resume_state: dict = {}
    if args.resume:
        resume_state = load_checkpoint(args.resume, model, optimizer, map_location=device)
        args.mlflow_run_id = args.mlflow_run_id or resume_state.get("mlflow_run_id")
    start_epoch = int(resume_state.get("epoch", 0))
    start_update = int(resume_state.get("global_update", 0))
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    mlflow_run = start_mlflow_run(args, dataset_size=len(dataloader.dataset))
    run_id = mlflow_run.active_run().info.run_id if mlflow_run is not None else None
    progress = TrainingProgress(
        total_epochs=args.epochs,
        micro_batches_per_epoch=len(dataloader),
        start_epoch=start_epoch,
        elapsed_offset_s=float(resume_state.get("elapsed_s", 0.0)),
    )

    def visualize(epoch: int) -> None:
        path = visualize_model_bev(
            model, vis_batches, device, amp_dtype, PC_RANGE,
            output_dir / "bev_features" / f"epoch_{epoch:02d}.png", title=f"epoch {epoch}",
            names=[f"({args.vis_split} #{index})" for index in args.vis_sample_indices],
        )
        log_mlflow_artifact(mlflow_run, path, artifact_path="bev_features")

    def update_callback(state: dict) -> None:
        epoch_done = state["micro_batches_done"] == state["num_micro_batches"]
        if state["update"] % args.log_every_updates and not epoch_done:
            return
        snapshot = progress.snapshot(state["epoch"], state["micro_batches_done"])
        metrics = {"train/loss": state["loss"], "train/lr": state["lr"]}
        metrics.update({f"progress/{k}": v for k, v in snapshot.items() if k != "finish_unix_s"})
        log_mlflow_metrics(mlflow_run, metrics, step=state["update"])
        description = progress.description(snapshot, extra_lines=[
            f"**Train loss (last update):** {state['loss']:.3f}",
            f"**Optimizer updates:** {state['update']}",
            f"**Output dir:** `{output_dir.resolve()}`",
        ])
        if mlflow_run is not None:
            mlflow_run.set_tag("mlflow.note.content", description)
        eta = f" eta {snapshot['eta_hours']:.1f}h" if "eta_hours" in snapshot else ""
        print(f"epoch {snapshot['epoch']:.3f}/{args.epochs} update {state['update']} loss {state['loss']:.3f} "
              f"lr {state['lr']:.2e}{eta}", flush=True)

    def epoch_end_callback(epoch: int, metrics: dict[str, float]) -> None:
        log_mlflow_metrics(mlflow_run, metrics, step=epoch, prefix="epoch/")
        state = {"epoch": epoch, "global_update": int(metrics["global_update"]),
                 "elapsed_s": progress.elapsed_s(), "mlflow_run_id": run_id}
        save_checkpoint(output_dir / "checkpoints" / "latest.pth", model, optimizer, **state)
        weights = save_checkpoint(output_dir / "checkpoints" / f"epoch_{epoch:02d}.pth", model, **state)
        if args.mlflow_log_checkpoints:
            log_mlflow_artifact(mlflow_run, weights, artifact_path="checkpoints")
        if vis_batches is not None and epoch % args.vis_every_epochs == 0:
            visualize(epoch)

    try:
        if vis_batches is not None and start_epoch == 0:
            visualize(0)
        fit(
            model,
            criterion,
            dataloader,
            optimizer,
            device,
            epochs=args.epochs - start_epoch,
            grad_clip_norm=args.grad_clip_norm,
            amp_dtype=amp_dtype,
            start_epoch=start_epoch,
            epoch_end_callback=epoch_end_callback,
            accumulation_steps=args.accumulation_steps,
            lr_schedule=schedule.apply,
            update_callback=update_callback,
            start_update=start_update,
        )
    except BaseException as exc:
        if mlflow_run is not None:
            mlflow_run.end_run(status="KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED")
        raise

    final = save_checkpoint(output_dir / "final.pth", model, epoch=args.epochs, mlflow_run_id=run_id)
    print(f"Saved final weights to {final}")
    if mlflow_run is not None:
        mlflow_run.end_run()


if __name__ == "__main__":
    main()
