# Phase 1: Repo Scaffold + nuScenes Temporal Data Pipeline Implementation Plan

> **Execution note:** Executed inline, in-session, by the same agent that
> wrote this plan and the spec — full project context is already loaded,
> so tasks are scoped at file/responsibility granularity rather than
> exhaustive subagent-ready code blocks. Each task still ends in a commit
> with passing tests.

**Goal:** Stand up the `bevformer` package skeleton and a pure-PyTorch
nuScenes dataset that returns temporal queues of frames (images, ego
pose/can_bus deltas, current-frame 3D boxes), matching official BEVFormer's
data contract.

**Architecture:** Mirrors `DETR3D-from-Scratch`'s package layout. Dataset
logic is split into geometry/category utilities, transforms, dataset
class (queue construction), and collate — each independently testable
against a synthetic nuScenes-style JSON fixture (no real dataset needed
for tests; real dataset at `~/dataset/nuscenes` used for manual smoke
checks only).

**Tech Stack:** PyTorch, TorchVision (transforms only), NumPy, Pillow,
pytest. No MMDetection/MMDetection3D/MMCV anywhere in this repo.

**Spec:** `docs/superpowers/specs/2026-08-23-phase1-scaffold-and-data-pipeline-design.md`

## Global Constraints

- Pure PyTorch/TorchVision only — no OpenMMLab dependencies, ever.
- `queue_length = 4` default (official BEVFormer).
- `DEFAULT_PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)`.
- 10-class nuScenes detection taxonomy, same mapping as DETR3D reference.
- Tests must not require the real nuScenes dataset download.
- Only files relevant to this repo are committed — no unrelated/incidental
  files from the working environment.

---

### Task 1: Repository scaffold and packaging

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`,
  `.pre-commit-config.yaml`, `README.md`
- Create: `bevformer/__init__.py`, `bevformer/data/__init__.py`,
  `bevformer/models/__init__.py`, `bevformer/engine/__init__.py`,
  `bevformer/scripts/__init__.py`, `bevformer/utils/__init__.py`
- Create: `tests/__init__.py`

**Deliverable:** `pip install -e .` succeeds and `python -c "import bevformer"`
works.

- [x] Write `pyproject.toml` (package `bevformer`, `requires-python>=3.10`,
  pytest/black/ruff config matching DETR3D's style).
- [x] Write `requirements.txt` (`torch`, `torchvision`, `numpy`, `pillow`,
  `pytest`).
- [x] Write `.gitignore` (datasets, checkpoints, logs, `__pycache__`,
  `*.egg-info`, `.pytest_cache`, venvs).
- [x] Write `.pre-commit-config.yaml` (black + ruff hooks).
- [x] Write `README.md` (project description, pure-PyTorch/from-scratch
  statement, phased layout, setup pointing at `~/dataset/nuscenes`).
- [x] Create empty package dirs with `__init__.py` placeholders.
- [x] Verify: `pip install -e . && python -c "import bevformer"` succeeds.
- [x] Commit: "Scaffold bevformer package structure and tooling config"

---

### Task 2: Geometry and category utilities

**Files:**
- Create: `bevformer/data/nuscenes_geometry.py` (SE3 pose math, quaternion
  conversion — pulled out as its own module so `nuscenes_dataset.py` stays
  focused, unlike DETR3D which inlines these)
- Create: `bevformer/data/nuscenes_categories.py` (`NUSCENES_CLASSES`,
  `CLASS_TO_ID`, `OFFICIAL_CATEGORY_MAPPING`, `category_to_detection_class`)
- Test: `tests/test_nuscenes_geometry.py`, `tests/test_nuscenes_categories.py`

**Interfaces:**
- Produces: `quaternion_to_rotation_matrix(q) -> np.ndarray[3,3]`,
  `pose_to_matrix(rotation, translation) -> np.ndarray[4,4]`,
  `invert_se3(matrix) -> np.ndarray[4,4]`,
  `yaw_from_rotation_matrix(rotation) -> float`,
  `category_to_detection_class(name, official=True) -> str | None`

- [x] Write failing tests: identity quaternion → identity rotation matrix;
  round-trip `invert_se3(pose_to_matrix(...))` composes to identity;
  known category strings map to expected class names, unknown maps to None.
- [x] Run tests, confirm failure (module doesn't exist).
- [x] Implement `nuscenes_geometry.py` and `nuscenes_categories.py`.
- [x] Run tests, confirm pass.
- [x] Commit: "Add nuScenes geometry and category mapping utilities"

---

### Task 3: Image transforms

**Files:**
- Create: `bevformer/data/transforms.py`
- Test: `tests/test_transforms.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `resize_and_normalize_image(image: PIL.Image, image_size=(900,1600)) -> torch.Tensor[3,H,W]`,
  `photometric_distort_bgr(array: np.ndarray) -> np.ndarray`

- [x] Write failing tests: output shape/dtype for a synthetic PIL image;
  normalized values fall in a sane range; photometric distort changes
  pixel values but preserves shape/dtype.
- [x] Run tests, confirm failure.
- [x] Implement `transforms.py` (ported from DETR3D's approach).
- [x] Run tests, confirm pass.
- [x] Commit: "Add image resize/normalize/photometric-distort transforms"

---

### Task 4: nuScenes temporal-queue dataset

**Files:**
- Create: `bevformer/data/nuscenes_dataset.py`
- Test: `tests/test_nuscenes_dataset.py`
- Test fixture: `tests/fixtures/build_synthetic_nuscenes.py` (helper that
  writes a tiny synthetic nuScenes-style JSON metadata set + tiny PNG
  images into a `tmp_path`, used by the test module)

**Interfaces:**
- Consumes: `nuscenes_geometry.*`, `nuscenes_categories.*`,
  `transforms.resize_and_normalize_image`
- Produces: `BevFormerNuScenesDataset(dataroot, version, queue_length=4,
  image_size=(900,1600), pc_range=DEFAULT_PC_RANGE)`, a `torch.utils.data.Dataset`
  whose `__getitem__` returns
  `{"imgs": Tensor[queue_length,6,3,H,W], "img_metas": List[dict] (len queue_length),
  "can_bus": Tensor[queue_length,18], "gt_boxes_3d": Tensor[N,9],
  "gt_labels_3d": Tensor[N]}`. Constants: `DEFAULT_QUEUE_LENGTH=4`,
  `DEFAULT_BEV_H=DEFAULT_BEV_W=200`, `DEFAULT_PC_RANGE`.

- [x] Write the synthetic fixture builder: 2 scenes, one with >=4 samples
  (full queue, no padding) and one with only 2 samples (tests left-padding
  + `prev_bev_exists=False`), tiny (32x32) placeholder JPEGs for the 6
  cameras, calibrated_sensor/ego_pose/sample/scene/category JSON tables.
- [x] Write failing tests using the fixture:
  - queue has correct length and frame order (oldest→newest, current frame
    last)
  - short scene: earliest-frame padding present, `prev_bev_exists=False`
    on padded entries, `True` on real entries after the first
  - `can_bus` shape `(queue_length, 18)` and delta slots (indices 16,17)
    are zero for the first (or padded-repeat) frame, nonzero when ego
    pose actually changed between consecutive frames
  - `gt_boxes_3d`/`gt_labels_3d` correspond only to the last frame's
    annotations, using the expected class ids
  - `img_metas[i]["lidar2img"]` has 6 entries (per camera), each a
    `4x4` matrix
- [x] Run tests, confirm failure.
- [x] Implement `nuscenes_dataset.py`: JSON table loading/indexing, per-scene
  sample chain traversal, queue construction with left-padding, lidar2img
  matrix computation reusing `nuscenes_geometry`, can_bus vector assembly
  (pose translation/quaternion + zero-filled accel/rotation-rate/velocity
  slots + computed delta yaw/translation), box/label extraction for the
  current frame via `category_to_detection_class`.
- [x] Run tests, confirm pass.
- [x] Commit: "Add BevFormerNuScenesDataset with temporal queue construction"

---

### Task 5: Collate function

**Files:**
- Create: `bevformer/data/collate.py`
- Test: `tests/test_collate.py`

**Interfaces:**
- Consumes: dataset sample dicts as produced by Task 4.
- Produces: `collate_fn(batch: List[dict]) -> dict` with `imgs`
  `Tensor[B,queue_length,6,3,H,W]`, `can_bus` `Tensor[B,queue_length,18]`,
  `img_metas: List[List[dict]]` (len B, each len queue_length),
  `gt_boxes_3d: List[Tensor[N_i,9]]`, `gt_labels_3d: List[Tensor[N_i]]`.

- [x] Write failing test: build 2-3 fake sample dicts with matching shapes
  but differing box counts, assert collated shapes/types and that
  `img_metas`/box lists stay ragged per-sample.
- [x] Run test, confirm failure.
- [x] Implement `collate.py`.
- [x] Run test, confirm pass.
- [x] Commit: "Add collate_fn for batching temporal-queue samples"

---

### Task 6: Sampler stub and manual smoke check against real dataset

**Files:**
- Create: `bevformer/data/sampler.py` (stub raising
  `NotImplementedError` with a docstring pointing at Phase 5)
- No new automated test (stub has no behavior to test); manual smoke
  check only.

- [x] Write `sampler.py` stub.
- [x] Manual smoke check: instantiate `BevFormerNuScenesDataset` against
  `~/dataset/nuscenes` (`v1.0-trainval`), fetch one item, print shapes,
  confirm no crash. Not committed as a test — exploratory verification only.
- [x] Run full test suite (`pytest tests/ -v`), confirm all green.
- [x] Commit: "Add sampler stub for Phase 5 scene-aware sampling"

---

## Self-review notes

- Spec coverage: geometry/category utils, transforms, dataset queue+padding
  +can_bus, collate, sampler stub, pyproject/readme/gitignore all covered.
- No placeholders left in deliverables (sampler stub is explicitly a stub
  per spec's non-goals, not a hidden gap).
- Type consistency checked: `BevFormerNuScenesDataset` sample dict keys/
  shapes are identical between Task 4's produces-block and Task 5's
  consumes-block.
