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

Add `--use-amp` for mixed precision on CUDA. Model-size knobs
(`--embed-dims`, `--bev-h`, `--bev-w`, `--num-queries`,
`--num-encoder-layers`, `--num-decoder-layers`, ...) default to the sizes in
`train.py`'s `add_model_args`; pass matching values to `eval.py` when
evaluating a checkpoint trained with non-default sizes.

## Evaluation

```bash
python eval.py \
  --dataroot ~/dataset/nuscenes \
  --checkpoint checkpoints/bevformer.pth
```

Reports lightweight sanity metrics (greedy center-distance match rate, mean
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
