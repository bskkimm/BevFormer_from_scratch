# Phase 5: Training/Evaluation Engine

## Goal

Wire everything from Phases 1-4 into a runnable model and training loop:
full model assembly (image -> ... -> predictions) including BEVFormer's
temporal history mechanism (build BEV history over the queue frames under
`torch.no_grad()`, backprop only through the current frame, matching
official BEVFormer training), a trainer, a lightweight evaluator, and
`train.py`/`eval.py` CLI entry points.

## Prerequisite fix: can_bus ego-motion delta representation

Phase 3's `warp_prev_bev` needs a 2D translation delta (`[B, 2]`, x and y
separately) to warp the previous BEV feature map, but Phase 1's
`BevFormerNuScenesDataset._build_can_bus` currently stores only a scalar
translation-delta *magnitude* at index 16 (direction is lost). This is a
real gap between the two phases that only became apparent when wiring
them together here.

Fix: change indices `[16, 17]` of `can_bus` to store the raw x/y
translation delta (ego-plane, meters) between consecutive queue frames,
instead of magnitude. Yaw delta is no longer stored separately in
`can_bus` — Phase 5's model assembly computes it on demand from the
already-present absolute rotation quaternions at `can_bus[:, 3:7]` for
consecutive frames (one call to the existing geometry helpers), so no
18-dim budget is spent on it. `tests/test_nuscenes_dataset.py`'s
`test_can_bus_deltas_zero_when_no_prev_bev_and_nonzero_otherwise` is
updated to match (checks x/y components instead of magnitude).

## Components

### `bevformer/models/bevformer.py`
- `BEVFormerModel(backbone, neck, encoder, decoder, head, grid_mask=None)`:
  - `extract_bev_features(imgs, img_metas, prev_bev=None,
    delta_translation_bev=None, delta_yaw=None) -> bev_embed`: one frame
    through backbone -> (optional GridMask, train only) -> neck -> encoder.
  - `forward(imgs_queue: Tensor[B,T,N,3,H,W], img_metas_queue: list[list[dict]]
    (len T, each len B), can_bus_queue: Tensor[B,T,18]) -> dict`: loops
    frames `0..T-2` under `torch.no_grad()` computing `prev_bev` (each
    step warps the previous result using that frame's ego-motion delta,
    skipping the warp/attention history reset when `img_metas` says the
    frame is a padding repeat), then computes the final frame's
    `bev_embed` with gradients enabled using the accumulated `prev_bev`,
    and runs the decoder + head on it. Returns `{"cls_scores",
    "bbox_preds", "bev_embed"}`.

### `bevformer/data/sampler.py` (revise the Phase 1 stub)
- Since `BevFormerNuScenesDataset.__getitem__` already returns a
  self-contained temporal queue per sample (padded at scene boundaries),
  batches of independently-sampled queues have no cross-sample temporal
  interaction — a plain shuffling sampler is correct and sufficient.
  `SceneAwareSampler` is removed; the module instead re-exports
  `torch.utils.data.RandomSampler` with a short docstring explaining why
  no custom scene-aware logic is needed here (unlike frame-by-frame
  streaming training loops, which this dataset design avoids).

### `bevformer/engine/trainer.py`
- `move_batch_to_device(batch, device) -> dict`
- `train_one_epoch(model, criterion, dataloader, optimizer, device,
  grad_clip_norm=None, use_amp=False, scaler=None) -> dict[str, float]`:
  one pass, AMP + grad-clip support, returns averaged loss metrics.
- `fit(model, criterion, dataloader, optimizer, device, epochs, ...) ->
  list[dict]`: repeats `train_one_epoch`, prints a one-line summary per
  epoch. Deliberately does not port DETR3D's thermal-safety-monitoring
  and fine-grained debug-parameter tracking — that is bespoke hardware
  monitoring for that repo's training runs, not part of this
  reimplementation's scope.

### `bevformer/engine/evaluator.py`
- `decode_predictions(cls_logits, bbox_preds, max_num=300,
  post_center_range=...) -> (boxes, scores, labels)`: NMS-free top-k
  decode (ported from DETR3D's `decode_nuscenes_predictions`, renamed) —
  ranks all (query, class) score pairs, keeps the top `max_num` inside
  `post_center_range`.
- `evaluate_predictions(pred_boxes, pred_labels, pred_scores, gt_boxes,
  gt_labels, score_threshold=0.3) -> dict[str, float]`: a lightweight
  sanity metric (greedy center-distance matching within 2m, per-class
  match rate, mean center L2 error on matches) — **not** the official
  nuScenes mAP/NDS protocol. Full official evaluation requires
  `nuscenes-devkit`, a submission JSON, and the complete validation
  split; wiring that is explicitly deferred (noted in the README) since
  it's an integration/dependency task orthogonal to finishing the
  from-scratch model implementation, and can be added on top of
  `decode_predictions` later without changing anything else.

### `train.py` / `eval.py`
- `train.py`: argparse CLI (`--dataroot`, `--epochs`, `--batch-size`,
  `--lr`, `--device`, model size knobs) building the dataset/dataloader/
  model/optimizer/loss and calling `fit`.
- `eval.py`: argparse CLI loading a checkpoint, running the model over a
  dataloader, decoding predictions with `decode_predictions`, and
  printing `evaluate_predictions` metrics.

## Testing

- `tests/test_bevformer_model.py`: tiny synthetic multi-frame batch
  (`bev_h=bev_w=4`, 2 cameras, `queue_length=3`) through
  `BEVFormerModel.forward` produces correctly-shaped `cls_scores`/
  `bbox_preds`, gradients flow only through the last frame's backbone
  parameters in the sense that the whole call doesn't error and produces
  finite gradients end to end (no NaN from the no-grad history loop
  feeding into the graph incorrectly).
- `tests/test_trainer.py`: one `train_one_epoch` call over 2 tiny
  synthetic batches with a real optimizer step reduces or at least
  computes a finite loss, and parameters actually change.
- `tests/test_evaluator.py`: `decode_predictions` respects `max_num` and
  `post_center_range` filtering; `evaluate_predictions` on a synthetic
  perfect-prediction case reports 100% match rate and ~0 center error,
  and on completely wrong predictions reports 0% match rate without
  crashing.
- `tests/test_nuscenes_dataset.py`: updated per the can_bus fix above.

## Constraints carried over

- Pure PyTorch/TorchVision only, no MMDetection/MMCV/custom CUDA ops.
- Tests must not require network access or the real nuScenes dataset,
  except where already true (the Phase 1 manual real-dataset smoke check
  pattern is not repeated here — CLI scripts are exercised by unit tests
  on synthetic data only).
