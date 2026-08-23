# Phase 1: Repo Scaffold + nuScenes Temporal Data Pipeline

## Goal

Stand up the `BevFormer_from_scratch` repository with a layout mirroring
[`DETR3D-from-Scratch`](https://github.com/bskkimm/DETR3D-from-Scratch), and
implement the first working slice: a pure-PyTorch nuScenes dataset that
returns **temporal queues** of frames, matching official BEVFormer's data
contract. This is Phase 1 of 5; later phases add the image backbone/neck,
BEV spatiotemporal encoder, detection head/losses, and training/eval engine.

The full reimplementation targets a complete, from-scratch, pure-PyTorch
BEVFormer — no MMDetection/MMDetection3D/MMCV — the same approach the
DETR3D reference repo took for DETR3D.

## Non-goals (deferred to later phases)

- Image backbone, FPN neck, BEV encoder, detection head, losses, training
  loop, evaluation. These directories exist only as empty `__init__.py`
  stubs in Phase 1.
- Scene-aware batch sampling logic in `sampler.py` (stubbed, implemented
  when the trainer needs it in Phase 5).

## Repository Layout

```text
BevFormer_from_scratch/
├── bevformer/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── nuscenes_dataset.py   # temporal-queue nuScenes Dataset
│   │   ├── transforms.py         # image resize/normalize/photometric distort
│   │   ├── collate.py            # batches queues into padded tensors
│   │   └── sampler.py            # stub for later scene-aware sampler
│   ├── models/__init__.py
│   ├── engine/__init__.py
│   ├── scripts/__init__.py
│   └── utils/__init__.py
├── tests/
│   └── test_nuscenes_dataset.py
├── docs/superpowers/{specs,plans}/
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .pre-commit-config.yaml
└── README.md
```

## Data Source

Dataset already present locally at `~/dataset/nuscenes`, standard nuScenes
layout: `v1.0-trainval/*.json` metadata tables, `samples/CAM_*` images,
`maps/`. Tests use a small synthetic metadata fixture, not the real dataset.

## Data Contract

`BevFormerNuScenesDataset` (`torch.utils.data.Dataset`), constructed with
`dataroot`, `version` (e.g. `v1.0-trainval`), `queue_length=4` (official
default), `image_size`, `pc_range`.

`__getitem__(idx)` returns a dict:

- `imgs`: `(queue_length, num_cams=6, 3, H, W)` float tensor, normalized
  per-camera using the same `resize_and_normalize_image` approach as
  DETR3D's `transforms.py`.
- `img_metas`: list of length `queue_length`, each entry holding per-camera
  `lidar2img` matrices (built from calibrated_sensor + ego_pose, same SE3
  math as DETR3D's `nuscenes_dataset.py`), scene token, sample token, and
  `prev_bev_exists: bool` (False when the frame is a padding repeat or the
  first frame of a scene).
- `can_bus`: `(queue_length, 18)` float tensor — official BEVFormer ego
  vector: translation (3), rotation quaternion (4), accel (3, zero-filled
  when unavailable in nuScenes ego_pose), rotation rate (3, zero-filled),
  velocity (3), plus 2 scalar slots for patched delta yaw / delta
  translation-in-BEV-plane, computed between consecutive frames in the
  queue (needed for temporal self-attention BEV alignment in Phase 3).
- `gt_boxes_3d`: `Tensor[N, 9]` (cx, cy, cz, w, l, h, yaw, vx, vy) — only
  for the **last** (current) frame of the queue.
- `gt_labels_3d`: `Tensor[N]` class ids for the current frame, using the
  same 10-class nuScenes mapping (`NUSCENES_CLASSES`,
  `OFFICIAL_CATEGORY_MAPPING`) as DETR3D, reused verbatim.

### Queue construction

Frames are sampled backward from `idx`'s sample within the same scene
(`prev_sample_token` chain). If fewer than `queue_length` prior frames
exist before hitting a scene boundary, the queue is left-padded by
repeating the earliest available in-scene frame, and `prev_bev_exists`
is set `False` for each padded position — this matches official
BEVFormer's `NuScenesDataset.prepare_train_data` queue-building behavior
so the encoder knows not to fuse temporal BEV features across scene cuts.

## Transforms

`transforms.py` reuses DETR3D's image normalize/resize/photometric-distort
functions (adapted to accept the BEVFormer `image_size` default), applied
per-frame, per-camera.

## Collate

`collate.py`'s `collate_fn(batch)` stacks `imgs` and `can_bus` along a new
batch dimension (they're fixed-shape per sample), and keeps `img_metas`,
`gt_boxes_3d`, `gt_labels_3d` as `List[...]` per batch element since boxes
are ragged (variable object count per frame).

## Config

No external config framework (no mmcv Config/argparse YAML). Plain Python
module-level constants in `nuscenes_dataset.py`:
`DEFAULT_QUEUE_LENGTH = 4`, `DEFAULT_BEV_H = DEFAULT_BEV_W = 200`,
`DEFAULT_PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)`, matching
official BEVFormer's nuScenes config defaults. Same style as DETR3D.

## Testing

`tests/test_nuscenes_dataset.py` builds a tiny synthetic nuScenes-style
JSON fixture (2-3 scenes, a handful of samples each, small placeholder
images) in a pytest tmp_path fixture and unit-tests:

- Correct queue length and frame ordering (oldest to newest).
- Scene-boundary padding (`prev_bev_exists=False` on padded frames).
- `can_bus` delta computation between consecutive frames.
- `gt_boxes_3d`/`gt_labels_3d` populated only on the last queue frame.
- `collate_fn` output shapes for a small batch.

No real dataset download required for tests; real-dataset loading is
exercised manually against `~/dataset/nuscenes`.

## Top-level files

- `pyproject.toml`: package name `bevformer`, `requires-python = ">=3.10"`,
  same `pytest`/`black`/`ruff` tool config style as DETR3D's.
- `requirements.txt`: `torch`, `torchvision`, `numpy`, `pillow`, `pytest`
  (nuscenes-devkit added in a later phase when evaluation is implemented).
- `.gitignore`: excludes datasets, checkpoints, logs, `__pycache__`, etc.
- `README.md`: project description, pure-PyTorch/from-scratch statement,
  phased repository layout (data → backbone/neck → BEV encoder → head/
  losses → training engine), setup instructions pointing at
  `~/dataset/nuscenes`.
