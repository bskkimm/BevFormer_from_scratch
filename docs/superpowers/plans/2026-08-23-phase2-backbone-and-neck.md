# Phase 2: Image Backbone + FPN Neck Implementation Plan

> **Execution note:** Executed inline, in-session, by the same agent that
> wrote this plan and the spec. Tasks scoped at file/responsibility
> granularity. Each task ends in a commit with passing tests.

**Goal:** Multi-camera ResNet+DCN backbone, FPN neck (4 levels), and
GridMask augmentation, all pure PyTorch/TorchVision.

**Spec:** `docs/superpowers/specs/2026-08-23-phase2-backbone-and-neck-design.md`

## Global Constraints

- Pure PyTorch/TorchVision only, no MMDetection/MMCV.
- Tests must not require network access (`pretrained=False`) or the real
  nuScenes dataset.

---

### Task 1: MultiViewImageBackbone

**Files:**
- Create: `bevformer/models/backbone/__init__.py`, `bevformer/models/backbone/image_backbone.py`
- Test: `tests/test_image_backbone.py`

**Produces:** `MultiViewImageBackbone(variant, pretrained, frozen_stages, norm_eval)`,
`forward(images: Tensor[B,N,3,H,W]) -> dict[str, Tensor[B,N,C,H,W]]` keys `stage3/4/5`.

- [ ] Write failing tests (shape/channel checks for resnet50, frozen-stage
  gradient check, deform-conv finite-output check).
- [ ] Implement `image_backbone.py`.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add MultiViewImageBackbone with deformable conv stages"

### Task 2: ImageFPN

**Files:**
- Create: `bevformer/models/neck/__init__.py`, `bevformer/models/neck/fpn.py`
- Test: `tests/test_fpn.py`

**Consumes:** dict with `stage3/4/5` keys of shape `[B,N,C,H,W]` (Task 1's output shape family).
**Produces:** `ImageFPN(in_channels, out_channels, out_names)`,
`forward(features) -> dict[str, Tensor[B,N,out_channels,H_l,W_l]]` for `p3..p6`.

- [ ] Write failing tests (output level count/channels/downsampling ratios).
- [ ] Implement `fpn.py`.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add ImageFPN neck with 4 output levels"

### Task 3: GridMask

**Files:**
- Create: `bevformer/models/grid_mask.py`
- Test: `tests/test_grid_mask.py`

**Produces:** `GridMask(...)`, `forward(images: Tensor) -> Tensor` (same shape).

- [ ] Write failing tests (eval-mode identity, train-mode-with-probability-1
  changes output, shape preserved).
- [ ] Implement `grid_mask.py`.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add GridMask training augmentation"

## Self-review

Spec coverage: backbone, neck, gridmask all covered. No placeholders.
Type consistency: Task 2 consumes exactly Task 1's produced dict shape.
