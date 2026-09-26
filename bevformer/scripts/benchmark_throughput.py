"""Training-throughput benchmark on the real nuScenes data.

Builds exactly the train.py model and DataLoader (bevformer.engine.build)
and times the real training code path (bevformer.engine.trainer):

  loader : DataLoader only                    -> samples/s the CPU side delivers
  step   : one cached batch, repeated          -> GPU step time (forward + loss + backward
                                                  + optimizer, incl. host->device copy)
  e2e    : real DataLoader feeding real steps  -> end-to-end training samples/s

Training is data-bound when `loader` samples/s < `step` samples/s, GPU-bound otherwise.
Example:

  python bevformer/scripts/benchmark_throughput.py --mode e2e --amp bf16 \\
      --image-dtype uint8 --num-workers 16 --pin-memory --persistent-workers
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

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
from bevformer.engine.trainer import train_one_epoch
from bevformer.models.losses.bevformer_loss import BEVFormerLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["loader", "step", "e2e"], required=True)
    parser.add_argument("--warmup", type=int, default=3, help="untimed batches/steps before measuring")
    parser.add_argument("--iters", type=int, default=20, help="timed batches/steps")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true", help="print one machine-readable JSON line")
    add_data_args(parser)
    add_model_args(parser)
    add_runtime_args(parser)
    return parser.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_loader(args: argparse.Namespace) -> dict[str, float]:
    start = time.perf_counter()
    iterator = iter(build_dataloader(args))
    next(iterator)
    startup_s = time.perf_counter() - start
    for _ in range(args.warmup):
        next(iterator)
    start = time.perf_counter()
    for _ in range(args.iters):
        next(iterator)
    elapsed = time.perf_counter() - start
    return {
        "startup_s": startup_s,
        "batch_ms": 1000 * elapsed / args.iters,
        "samples_per_s": args.iters * args.batch_size / elapsed,
    }


def _training_setup(args: argparse.Namespace, device: torch.device):
    model = build_model(args).to(device)
    criterion = BEVFormerLoss(num_classes=args.num_classes, pc_range=PC_RANGE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and device.type == "cuda")
    return model, criterion, optimizer, scaler


def _time_steps(args, device, batches_warmup, batches_timed) -> dict[str, float]:
    model, criterion, optimizer, scaler = _training_setup(args, device)
    step = lambda batches: train_one_epoch(  # noqa: E731
        model, criterion, batches, optimizer, device, 35.0, AMP_DTYPES[args.amp], scaler
    )
    step(batches_warmup)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    metrics = step(batches_timed)
    _sync(device)
    elapsed = time.perf_counter() - start
    result = {
        "step_s": elapsed / args.iters,
        "samples_per_s": args.iters * args.batch_size / elapsed,
        "loss": metrics["loss"],
    }
    if device.type == "cuda":
        result["peak_mem_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    return result


def bench_step(args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    loader = build_dataloader(args)
    batch = next(iter(loader))
    if args.pin_memory:
        batch["imgs"] = batch["imgs"].pin_memory()
    return _time_steps(args, device, [batch] * args.warmup, [batch] * args.iters)


def bench_e2e(args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    iterator = iter(build_dataloader(args))
    return _time_steps(
        args, device, itertools.islice(iterator, args.warmup), itertools.islice(iterator, args.iters)
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    configure_runtime(args)
    if args.mode == "loader":
        result = bench_loader(args)
    elif args.mode == "step":
        result = bench_step(args, device)
    else:
        result = bench_e2e(args, device)

    config = {
        key: getattr(args, key)
        for key in (
            "mode", "batch_size", "num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
            "image_dtype", "amp", "tf32", "cudnn_benchmark", "image_height", "image_width", "queue_length",
            "backbone_variant", "dcn",
        )
    }
    if args.json:
        print(json.dumps({**config, **result}))
    else:
        print(", ".join(f"{k}={v}" for k, v in config.items()))
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in result.items()))


if __name__ == "__main__":
    main()
