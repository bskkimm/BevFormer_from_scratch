import argparse

import torch

from bevformer.engine.build import add_data_args, add_runtime_args, build_dataloader, configure_runtime
from tests.fixtures.build_synthetic_nuscenes import build_synthetic_nuscenes


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    return parser.parse_args(argv)


def test_build_dataloader_applies_performance_flags(tmp_path):
    info = build_synthetic_nuscenes(tmp_path)
    args = _parse([
        "--dataroot", str(info["dataroot"]), "--queue-length", "2", "--image-height", "8", "--image-width", "16",
        "--batch-size", "2", "--num-workers", "2", "--pin-memory", "--persistent-workers",
        "--prefetch-factor", "3", "--image-dtype", "uint8",
    ])
    loader = build_dataloader(args)

    assert loader.num_workers == 2
    assert loader.pin_memory
    assert loader.persistent_workers
    assert loader.prefetch_factor == 3
    batch = next(iter(loader))
    assert batch["imgs"].shape == (2, 2, 6, 3, 8, 16)
    assert batch["imgs"].dtype == torch.uint8


def test_build_dataloader_single_process_ignores_worker_only_flags(tmp_path):
    info = build_synthetic_nuscenes(tmp_path)
    args = _parse([
        "--dataroot", str(info["dataroot"]), "--queue-length", "1", "--image-height", "8", "--image-width", "16",
        "--num-workers", "0", "--persistent-workers", "--prefetch-factor", "4",
    ])
    loader = build_dataloader(args)  # torch rejects these two flags when num_workers == 0
    assert loader.num_workers == 0
    assert next(iter(loader))["imgs"].dtype == torch.uint8  # the CLI default


def test_data_and_runtime_defaults_are_the_measured_throughput_setup():
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    add_runtime_args(parser)
    args = parser.parse_args([])
    assert (args.image_dtype, args.num_workers, args.pin_memory, args.persistent_workers) == ("uint8", 8, True, True)
    assert (args.amp, args.tf32, args.cudnn_benchmark) == ("bf16", True, False)


def test_configure_runtime_sets_backend_flags():
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        configure_runtime(parser.parse_args(["--tf32", "--cudnn-benchmark"]))
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.benchmark
        configure_runtime(parser.parse_args(["--no-tf32"]))
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.benchmark
        # --no-tf32 means stock PyTorch behavior, which keeps TF32 for cuDNN convolutions.
        assert torch.backends.cudnn.allow_tf32 == saved[1]
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = saved
