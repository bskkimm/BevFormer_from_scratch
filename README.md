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

train.py, eval.py  # training / evaluation CLI entry points
tests/              # unit tests for every phase
```

The implementation was built in five phases:

1. **Data pipeline** — a pure-PyTorch nuScenes dataset returning temporal
   queues of frames (multi-camera images, ego pose / can_bus deltas,
   current-frame 3D boxes), matching official BEVFormer's data contract.
2. **Image backbone + FPN neck** — ResNet with deformable convolutions,
   a 4-level feature pyramid, and the GridMask training augmentation.
3. **BEV spatiotemporal encoder** — a pure-PyTorch multi-scale deformable
   attention core, spatial cross-attention (image → BEV) and temporal
   self-attention (BEV → BEV across frames, aligned by ego motion).
4. **Detection head + losses** — an object-query decoder over the BEV
   grid, Hungarian matching, and focal/L1 losses with auxiliary
   decoder-layer supervision.
5. **Training/evaluation engine** — full model assembly (including
   BEVFormer's no-grad BEV history mechanism), a training loop, and
   `train.py`/`eval.py` CLI entry points.

## Setup

Create a Python 3.10+ environment and install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Prepare nuScenes using its standard directory layout (`v1.0-trainval`
metadata tables, `samples/CAM_*` images). This repository expects the
dataset at `~/dataset/nuscenes` by default; pass a different `dataroot` to
`BevFormerNuScenesDataset` (or `--dataroot` on the CLIs) to point
elsewhere. Dataset files are not distributed with this repository.

## Training and evaluation

```bash
python train.py --dataroot ~/dataset/nuscenes --epochs 24
python eval.py --dataroot ~/dataset/nuscenes --checkpoint checkpoints/bevformer.pth
```

`eval.py` reports lightweight sanity metrics (greedy center-distance
match rate, mean center error) — **not** the official nuScenes mAP/NDS
protocol. Official evaluation requires `nuscenes-devkit`, a submission
JSON, and the full validation split; that integration is intentionally
deferred, since it's orthogonal to finishing the from-scratch model
implementation and can be layered on top of
`bevformer.engine.evaluator.decode_predictions` later without changing
anything else.

## Testing

```bash
pytest tests/ -v
```

Tests run against small synthetic fixtures and do not require network
access or the real dataset download.
