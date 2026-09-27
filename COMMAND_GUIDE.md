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
python train.py --mlflow --output-dir runs/bevformer-r101dcn-bev100
```

Defaults follow official BEVFormer-Base except for a 100x100 BEV grid (Base:
200x200, 2.2x slower here):

| Setting | Default |
|---|---|
| Data | official train split (700 scenes, 28,130 samples); `--split all` adds val |
| Model | ResNet-101 + DCNv2 (stages 4-5), FPN, 100x100 BEV, 6 encoder / 6 decoder layers, 900 queries |
| Optimizer | AdamW lr 2e-4, weight decay 0.01, backbone lr x0.1, grad clip 35 |
| Schedule | 24 epochs, cosine per epoch to 1e-3 x lr, linear warmup over 500 updates from 1/3 lr |
| Batch | 1 sample x 8 accumulation steps = effective batch 8 (official: 8 GPUs x 1) |
| Speed | bf16 + TF32, compiled backbone, fused AdamW, uint8 images, 8 workers |

Use `--backbone-variant resnet50`, `--dcn v1|none`, `--bev-h/--bev-w`, or
`--num-encoder-layers` for lighter variants, and pass the same model flags to
`eval.py`.

Outputs (`--output-dir`, default `runs/<timestamp>`):

```text
checkpoints/latest.pth      model + optimizer + resume state, rewritten each epoch
checkpoints/epoch_XX.pth    weights after epoch XX (~230 MB each)
bev_features/epoch_XX.png   BEV feature images, every --vis-every-epochs (default 2) + epoch 0
final.pth                   weights after the last epoch
config.json                 every CLI argument
```

Resume an interrupted run from its last completed epoch, in the same MLflow run:

```bash
python train.py --mlflow --output-dir runs/bevformer-r101dcn-bev100 \
  --resume runs/bevformer-r101dcn-bev100/checkpoints/latest.pth
```

Smoke-test the whole pipeline first with `--subset 64 --epochs 2 --vis-every-epochs 1`
(about 3 minutes).

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

Current default model (100x100 BEV, 6 encoder layers, 8 spatial-cross-attention
points per camera and level as in official BEVFormer, compiled backbone, fused
AdamW): **0.830 s/sample, ~6.5 h per train-split epoch, ~6.5 days for 24 epochs**
(0.742 s with the earlier 4-point spatial cross-attention). The table above was measured on the earlier 50x50 / 3-layer default.

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
| torch.compile(backbone) + fused AdamW | 0.795 -> 0.742 s at 100x100 / 6 layers (compiled bf16 is as close to fp32 as eager bf16); channels_last was slower |
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

From another machine, tunnel the port over SSH and open http://localhost:5000:

```bash
ssh -N -L 5000:127.0.0.1:5000 <training-host>
```

What a training run logs:

| Where in the UI | Content |
|---|---|
| Run description (Overview) | live progress: epoch, % done, elapsed, avg epoch time, ETA and finish time, last loss |
| Metrics `train/loss`, `train/lr` | every `--log-every-updates` (default 10) optimizer updates |
| Metrics `progress/*` | fractional epoch, percent, elapsed / avg-epoch / ETA hours |
| Metrics `epoch/*` | per-epoch mean losses (final and each decoder layer `dN.*`), lr |
| Artifacts `bev_features/` | BEV images every `--vis-every-epochs` epochs: front camera, PCA of the BEV features, feature distinctiveness, GT (green) and predicted (orange) boxes |
| Parameters | every CLI argument plus dataset size and effective batch size |

`--resume` continues the checkpoint's MLflow run automatically; use
`--mlflow-log-checkpoints` to also upload per-epoch weights as artifacts.
