# BEVFormer Command Guide

Canonical commands for this repository. `README.md` covers setup; this guide
is the quick reference for day-to-day use.

## Tests

```bash
pytest tests/ -v
```

Runs against small synthetic fixtures only — no network access or the real
nuScenes dataset required.

## Implementation Sanity Check

Before committing to a full training run, verify gradients flow correctly
end to end by overfitting a tiny model to one fixed synthetic batch:

```bash
python bevformer/scripts/overfit_one_batch.py --steps 200
```

A healthy implementation drives the loss down substantially (observed: an
82.8% reduction over 150 steps on CPU, no dataset required). Useful after any
change to the model architecture, before spending time on a real training run.

## Training

```bash
python train.py \
  --dataroot ~/dataset/nuscenes \
  --version v1.0-trainval \
  --epochs 24 \
  --batch-size 1 \
  --lr 2e-4 \
  --grad-clip-norm 35.0 \
  --checkpoint-out checkpoints/bevformer.pth
```

Training uses the official nuScenes train split (700 scenes, 28,130 samples)
by default; `--split all` adds the val scenes. The throughput-tuned defaults (bf16 autocast, TF32 matmuls, uint8 images, 8
workers, pinned memory) are on without any flags — see "Training Throughput"
below; `--amp none --no-tf32` gives plain fp32. The backbone defaults to
ResNet-101 with DCNv2 in stages 4-5, as in official BEVFormer-Base; use
`--backbone-variant resnet50` and/or `--dcn v1|none` for lighter variants.
Model-size knobs (`--embed-dims`, `--bev-h`, `--bev-w`, `--num-queries`,
`--num-encoder-layers`, `--num-decoder-layers`, ...) default to the sizes in
`train.py`'s `add_model_args`; pass matching values to `eval.py` when
evaluating a checkpoint trained with non-default sizes.

## Training Throughput

Measure on your machine (each mode builds exactly the `train.py` model/loader):

```bash
python bevformer/scripts/benchmark_throughput.py --mode loader   # DataLoader only
python bevformer/scripts/benchmark_throughput.py --mode step     # GPU step, cached batch
python bevformer/scripts/benchmark_throughput.py --mode e2e      # real loader + real steps
```

Training is **data-bound** if `loader` samples/s < `step` samples/s, otherwise
**GPU-bound**. Measured on an RTX PRO 6000 (96 GB), 32 CPUs, local NVMe, with the
default model (ResNet-101 + DCNv2, 1600x900, 4-frame queue, 6 cameras, batch 1):

| | Previous defaults | Current defaults | Gain |
|---|---:|---:|---:|
| End-to-end step (`e2e`) | 0.829 s | **0.512 s** | **1.62x** |
| End-to-end samples/s | 1.21 | **1.95** | |
| GPU-only step (`step`) | 0.785 s | 0.497 s | 1.58x |
| DataLoader samples/s (`loader`) | 4.8 | 21.0 | 4.4x |
| Peak GPU memory | 18.1 GiB | 11.6 GiB | -36% |
| Epoch over the train split (28,130 samples) | ~6.5 h | ~4.0 h | |

Previous defaults: fp32, float32 images, 4 workers, no pinned memory. A cold page
cache (first epoch) measured the same: 0.516 s/step, loader 23.6 samples/s.

**Bottleneck: GPU-bound.** The loader delivers ~10x what the GPU consumes, so the
end-to-end step is within 3% of the GPU-only step.

What each lever did (GPU step unless noted):

| Lever | Effect |
|---|---|
| bf16 autocast | 0.781 -> 0.582 s; memory 18.1 -> 11.6 GiB (fp16 ties, but needs loss scaling) |
| TF32 matmuls | 0.571 -> 0.492 s: torchvision's deformable conv GEMM stays fp32 under autocast and ran on SIMT cores |
| uint8 images + GPU normalization | loader 3.3-3.5x faster at every worker count (4x fewer bytes through shared memory and host->device copies) |
| 8 workers, pinned, persistent | loader 20 samples/s; 16 workers adds ~15% loader headroom but nothing end to end |
| cudnn.benchmark | no measurable change -- left off |
| Batch size 2-8 | no gain (1.96-1.86 vs 2.00 samples/s at batch 1): one sample is already 24 full-size images; choose batch size for optimization, not speed |
| Pre-resized JPEG cache | not needed: at native 1600x900 there is no resize; at 800x450 the loader (21 samples/s) still outpaces even a ResNet-50/no-DCN step (10 samples/s) |

**Mixed-precision correctness.** Under bf16/TF32 the detection head, camera
projection, and BEV ego-motion warp are computed in float32, since bf16 would
shift box centers by ~6 cm (up to 17 cm) and projected pixels by several px.
fp32 and bf16+TF32 training from the same initialization on the same real samples
reach similar losses (80 steps: 52.7 -> 31.0 vs 56.3 -> 25.5).

## Evaluation

```bash
python eval.py \
  --dataroot ~/dataset/nuscenes \
  --checkpoint checkpoints/bevformer.pth
```

Evaluates the official val split (150 scenes, 6,019 samples) by default and
reports lightweight sanity metrics (greedy center-distance match rate, mean
center error) — **not** official nuScenes mAP/NDS. See
`bevformer/engine/evaluator.py`'s module docstring and `README.md` for why.

## MLflow Tracking

The default local tracking backend is SQLite (`mlflow.db` at the repo root,
gitignored):

```bash
python train.py --mlflow --mlflow-experiment bevformer-training ...
```

Start the local UI from the repository root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Resume logging into an existing run with `--mlflow-run-id <run_id>`, or log
the saved checkpoint as an MLflow artifact with `--mlflow-log-checkpoints`.
