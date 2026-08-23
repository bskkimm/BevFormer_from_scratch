# Phase 5: Training/Evaluation Engine Implementation Plan

> **Execution note:** Executed inline, in-session, by the same agent that
> wrote this plan and the spec. Each task ends in a commit with passing
> tests.

**Goal:** Full model assembly with temporal history, trainer, lightweight
evaluator, and CLI entry points — the final phase.

**Spec:** `docs/superpowers/specs/2026-08-24-phase5-training-engine-design.md`

## Global Constraints

- Pure PyTorch/TorchVision only, no MMDetection/MMCV/custom CUDA ops.
- Tests must not require network access or the real nuScenes dataset.

---

### Task 1: Fix can_bus translation delta representation

**Files:**
- Modify: `bevformer/data/nuscenes_dataset.py`
- Modify: `tests/test_nuscenes_dataset.py`

- [ ] Update the failing/changed test: check `can_bus[:, 16:18]` are the
  x/y translation delta (nonzero-with-known-sign for the fixture's
  +1m/step ego motion) instead of a scalar magnitude+yaw pair.
- [ ] Update `_build_can_bus` to store x/y delta at indices 16/17,
  drop the separate yaw-delta computation (no longer stored).
- [ ] Run `tests/test_nuscenes_dataset.py`, confirm pass.
- [ ] Commit: "Store can_bus translation delta as x/y instead of magnitude"

### Task 2: Full model assembly with temporal history

**Files:**
- Create: `bevformer/models/bevformer.py`
- Test: `tests/test_bevformer_model.py`

**Consumes:** Phase 2's backbone/neck/GridMask, Phase 3's encoder, Phase 4's
decoder/head, Phase 1's `nuscenes_geometry` (for on-demand yaw delta).
**Produces:** `BEVFormerModel(backbone, neck, encoder, decoder, head, grid_mask=None)`,
`extract_bev_features(imgs, img_metas, prev_bev, delta_translation_bev,
delta_yaw) -> Tensor`, `forward(imgs_queue, img_metas_queue, can_bus_queue) ->
dict[str, Tensor]`.

- [ ] Write failing tests (output shapes for `cls_scores`/`bbox_preds`;
  finite end-to-end gradient on the last frame's backbone parameters).
- [ ] Implement.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add BEVFormerModel with temporal BEV history"

### Task 3: Revise sampler stub

**Files:**
- Modify: `bevformer/data/sampler.py`

- [ ] Replace `SceneAwareSampler` stub with a documented re-export of
  `torch.utils.data.RandomSampler`, explaining why no custom scene-aware
  sampling is needed given self-contained per-sample queues.
- [ ] Commit: "Revise sampler: self-contained queues need no scene-aware logic"

### Task 4: Trainer

**Files:**
- Create: `bevformer/engine/trainer.py`
- Test: `tests/test_trainer.py`

**Consumes:** Task 2's `BEVFormerModel`, Phase 4's `BEVFormerLoss`.
**Produces:** `move_batch_to_device`, `train_one_epoch`, `fit`.

- [ ] Write failing tests (one epoch over synthetic tiny batches returns
  finite loss metrics; a parameter actually changes after the step).
- [ ] Implement.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add training loop (train_one_epoch, fit)"

### Task 5: Evaluator

**Files:**
- Create: `bevformer/engine/evaluator.py`
- Test: `tests/test_evaluator.py`

**Consumes:** Phase 4's `decode_bbox_predictions`.
**Produces:** `decode_predictions`, `evaluate_predictions`.

- [ ] Write failing tests (top-k/range filtering behavior; perfect vs.
  wrong predictions on synthetic data).
- [ ] Implement.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add lightweight prediction decoding and evaluation"

### Task 6: CLI entry points

**Files:**
- Create: `train.py`, `eval.py`

- [ ] Write `train.py` (argparse, builds dataset/model/optimizer, calls `fit`).
- [ ] Write `eval.py` (argparse, loads checkpoint, decodes + evaluates).
- [ ] Verify: `python train.py --help` and `python eval.py --help` run
  without error.
- [ ] Commit: "Add train.py and eval.py CLI entry points"

## Self-review

Spec coverage: can_bus fix, model assembly, sampler revision, trainer,
evaluator, CLI entry points all covered. No placeholders. Type
consistency: Task 4 consumes Task 2's exact `forward` signature and
Phase 4's exact `loss_by_feat` signature.
