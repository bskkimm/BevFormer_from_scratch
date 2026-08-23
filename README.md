# BEVFormer in Pure PyTorch

This repository reimplements BEVFormer from the official paper and reference
implementation without MMDetection, MMDetection3D, or MMCV. Every component —
image backbone, FPN neck, BEV spatiotemporal encoder (spatial cross-attention
and temporal self-attention), detection head, losses, and training/evaluation
loop — is built from scratch using PyTorch and TorchVision.

## Repository Layout

The repository is built in phases, mirroring the structure of
[`DETR3D-from-Scratch`](https://github.com/bskkimm/DETR3D-from-Scratch):

```text
bevformer/
├── data/           # nuScenes temporal-queue dataset, transforms, collate
├── models/
│   ├── backbone/   # image backbone
│   ├── neck/       # FPN
│   ├── transformer/# deformable attention, spatial/temporal attention, BEV encoder
│   ├── heads/      # detection head
│   └── losses/     # Hungarian matcher, loss functions
├── engine/         # trainer, evaluator, hooks
├── scripts/        # sanity checks, benchmarks
└── utils/          # shared helpers

tests/              # unit tests for every phase
docs/superpowers/   # design specs and implementation plans
```

Phase 1 (current) implements the data pipeline: a pure-PyTorch nuScenes
dataset that returns temporal queues of frames (multi-camera images, ego
pose / can_bus deltas, current-frame 3D boxes) matching official BEVFormer's
data contract. Later phases add the image backbone/neck, BEV encoder,
detection head/losses, and training/evaluation engine.

## Setup

Create a Python 3.10+ environment and install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Prepare nuScenes using its standard directory layout (`v1.0-trainval`
metadata tables, `samples/CAM_*` images). This repository expects the
dataset at `~/dataset/nuscenes` by default; pass a different `dataroot` to
`BevFormerNuScenesDataset` to point elsewhere. Dataset files are not
distributed with this repository.

## Testing

```bash
pytest tests/ -v
```

Tests run against a small synthetic nuScenes-style fixture and do not
require the real dataset download.
