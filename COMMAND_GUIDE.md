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

Resuming restores the weights, optimizer, learning-rate position and MLflow run id.
An interrupted epoch is trained again from its start, so MLflow ends up with two logged
versions of those steps (`plot_training_curves.py` keeps the newest). Pass the same
`--dataroot` spelling when resuming (quote `'~/dataset/nuscenes'` so the shell does not
expand it): `train.py` rewrites `config.json`, and the validation watcher compares it with
its cache.

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
AdamW), measured over a complete 24-epoch run with the validation watcher sharing
the GPU: **0.89 s/sample, 6.9 h per train-split epoch, 166 h (about 6.9 days) of
training in total**. The table above was measured earlier on a 50x50 / 3-layer
default.

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

### Validation loss alongside a running training job

`bevformer/scripts/watch_validation.py` evaluates saved `epoch_*.pth` checkpoints
without restarting training. It logs `val/subset_loss` and loss components to the
existing MLflow run at the checkpoint's epoch number. Choose **Step** as the chart's
horizontal axis. This is a fixed subset estimate, not full-validation loss or official
nuScenes metrics.

Prepare the compact validation cache once, while training is stopped, then start the
watcher (the MLflow run id is stored in `checkpoints/latest.pth` and shown on the
run's page):

```bash
python bevformer/scripts/watch_validation.py --prepare-cache \
  --run-dir runs/<name> --run-id <mlflow run id>
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u bevformer/scripts/watch_validation.py \
  --run-dir runs/<name> --run-id <mlflow run id> --samples 128
```

The cache is about 18 MiB; evaluations load it directly without loading the full raw
nuScenes tables. Rebuild it with training stopped if the dataset code or the subset
settings change. The watcher waits while available system RAM is below 8 GiB
(`--min-free-ram-gib`).

Running training and the watcher as systemd user services keeps their memory budgets
apart. Capping the watcher means it can never take memory away from training:

```bash
systemd-run --user --unit=<name>-validation \
  --property=MemoryMax=6G --property=MemorySwapMax=0 --property=OOMPolicy=kill \
  --property=Restart=on-failure --property=RestartSec=60 \
  --property=StandardOutput=append:$PWD/runs/<name>/validation.log \
  --property=StandardError=append:$PWD/runs/<name>/validation.log \
  --working-directory=$PWD --setenv=OMP_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 \
  $PWD/.venv/bin/python -u bevformer/scripts/watch_validation.py \
  --run-dir runs/<name> --run-id <mlflow run id>
```

Check it with `systemctl --user status <name>-validation`; stop validation and its
evaluator child with `systemctl --user stop <name>-validation`.

The watcher checks every 60 seconds, evaluates the newest pending checkpoint first,
then fills older epochs. It uses the same seeded, scene-spread validation samples for
every checkpoint, batch size 1, no data-loader worker processes, the training image
size and temporal queue, and the training mixed-precision setting. GPU memory is capped
at 25% of the device by default (`--memory-fraction`); each evaluation process exits to
release memory. Training still shares GPU compute during evaluation. `val/duration_s`
records evaluation time including setup.

Only complete checkpoint filenames are consumed (training saves atomically). The
watcher holds a directory lock and skips epochs already in MLflow, making restarts
safe. Changing subset settings for the same metric series is rejected. It exits after
training ends and all saved epochs have been evaluated. Stop the watcher independently
of training using its PID in `<run-dir>/validation.lock`; if stopping during an
evaluation, also stop its evaluator child process.

### Training curves

```bash
python bevformer/scripts/plot_training_curves.py --run-id <mlflow run id> \
  --out docs/assets/training_curves.png
```

Draws the per-update training loss and the per-epoch train versus validation loss. Where a
resumed run logged a step twice, the newest entry is used.

## Demo Videos

```bash
python bevformer/scripts/make_demo_videos.py --run-dir runs/<name>                 # final.pth, every map, 2 min each
python bevformer/scripts/make_demo_videos.py --run-dir runs/<name> \
  --checkpoint checkpoints/epoch_14.pth --maps boston-seaport --seconds 30       # a quick look at one epoch
```

Needs `ffmpeg`. Each video chains validation scenes of one nuScenes map. Every frame
shows the front camera with the predicted 3D boxes projected onto it, next to the BEV
feature-distinctiveness map with ground-truth (green) and predicted (orange) boxes.
nuScenes keyframes are 2 Hz, so playback is real time at 2 fps and 240 frames make two
minutes. All four maps take about 12 minutes on the benchmark GPU. Output goes to
`runs/<name>/videos/` (MP4s, `manifest.json`, `index.html`).

To watch them from another machine, serve the folder and tunnel the port over SSH:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory runs/<name>/videos
ssh -N -L 8765:127.0.0.1:8765 <training-host>      # on your PC, then open http://localhost:8765
```
