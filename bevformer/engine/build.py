"""Shared CLI arguments and builders for the model and the nuScenes DataLoader.

Used by train.py, eval.py, and bevformer/scripts/benchmark_throughput.py so
all three construct exactly the same model and data pipeline.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bevformer.data.collate import collate_fn
from bevformer.data.nuscenes_dataset import BevFormerNuScenesDataset
from bevformer.data.nuscenes_splits import SPLITS
from bevformer.models.backbone.image_backbone import MultiViewImageBackbone
from bevformer.models.bevformer import BEVFormerModel
from bevformer.models.grid_mask import GridMask
from bevformer.models.heads.bevformer_head import BEVFormerHead
from bevformer.models.neck.fpn import ImageFPN
from bevformer.models.transformer.decoder import BEVFormerDecoder
from bevformer.models.transformer.encoder import BEVFormerEncoder

PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
NUM_CAMS = 6
AMP_DTYPES = {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--amp", default="bf16", choices=sorted(AMP_DTYPES))
    parser.add_argument(
        "--tf32", action=argparse.BooleanOptionalAction, default=True,
        help="run fp32 matmuls (e.g. inside deformable conv, which autocast keeps in fp32) on TF32 tensor "
        "cores; cuDNN convolutions already use TF32 by PyTorch default",
    )
    parser.add_argument(
        "--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False,
        help="let cuDNN autotune conv algorithms (input sizes are fixed during training)",
    )
    parser.add_argument(
        "--compile-backbone", action=argparse.BooleanOptionalAction, default=True,
        help="torch.compile the image backbone on CUDA (~5%% faster steps after a one-time compile)",
    )
    parser.add_argument(
        "--fused-adamw", action=argparse.BooleanOptionalAction, default=True,
        help="use the fused CUDA AdamW kernel (~2%% faster steps)",
    )


def configure_runtime(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = args.cudnn_benchmark


def prepare_model(model: nn.Module, args: argparse.Namespace, device: torch.device) -> nn.Module:
    """Moves `model` to `device` and, on CUDA with --compile-backbone, compiles the backbone
    in place (nn.Module.compile keeps state_dict keys unchanged, so checkpoints stay portable)."""
    model = model.to(device)
    if args.compile_backbone and device.type == "cuda":
        model.backbone.compile()
    return model


def add_data_args(parser: argparse.ArgumentParser, default_split: str = "train") -> None:
    parser.add_argument("--dataroot", default="~/dataset/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument(
        "--split", default=default_split, choices=list(SPLITS),
        help="official v1.0-trainval scenes: train (28,130 samples), val (6,019), or all",
    )
    parser.add_argument("--queue-length", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=1)
    # Throughput defaults below are the measured-best setup on a 32-CPU / RTX PRO 6000
    # machine (see COMMAND_GUIDE.md "Training throughput").
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--image-dtype", default="uint8", choices=["float32", "uint8"],
        help="uint8 ships raw pixels and normalizes on the GPU (4x less host->device data)",
    )
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2, help="batches prefetched per worker")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backbone-variant", default="resnet101", choices=["resnet50", "resnet101"])
    parser.add_argument("--dcn", default="v2", choices=["v2", "v1", "none"])
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
    backbone = MultiViewImageBackbone(
        variant=args.backbone_variant, pretrained=True, frozen_stages=1, dcn=args.dcn
    )
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


def build_dataloader(args: argparse.Namespace, shuffle: bool = True) -> DataLoader:
    dataset = BevFormerNuScenesDataset(
        dataroot=args.dataroot,
        version=args.version,
        queue_length=args.queue_length,
        image_size=(args.image_height, args.image_width),
        pc_range=PC_RANGE,
        image_dtype=args.image_dtype,
        split=args.split,
    )
    worker_kwargs = {}
    if args.num_workers > 0:  # DataLoader rejects these options without worker processes
        worker_kwargs = {"persistent_workers": args.persistent_workers, "prefetch_factor": args.prefetch_factor}
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        **worker_kwargs,
    )
