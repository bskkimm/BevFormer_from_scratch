# Phase 4: Detection Head + Losses

## Goal

Implement the object-query decoder over the BEV feature grid produced by
Phase 3, the detection head (classification + box regression per decoder
layer), and the training loss (Hungarian matching + focal classification
loss + L1 box loss with auxiliary losses on intermediate decoder layers).
This closes the model's forward path: images -> backbone/neck -> BEV
encoder -> decoder/head -> per-query class + box predictions.

Box parameterization, matching, and loss structure follow the same
official nuScenes 3D detection convention `DETR3D-from-Scratch` already
implements (both trace back to the same mmdetection3d lineage), so this
phase reuses that design directly rather than inventing a new one:

- Semantic box: `[x, y, z, w, l, h, yaw, vx, vy]` (9D).
- Encoded training target / prediction: `[x, y, log(w), log(l), z, log(h),
  sin(yaw), cos(yaw), vx, vy]` (10D).
- Hungarian matching cost: focal-style classification cost + L1 cost on
  the first 8 encoded box dims (excluding velocity, matching DETR3D).
- Loss: sigmoid focal loss for classification (background = no match),
  weighted L1 for boxes, both computed per decoder layer with the final
  layer's loss unweighted and earlier layers as auxiliary losses.

## What's different from DETR3D here

DETR3D's decoder cross-attention projects each query's 3D reference point
into every camera and samples image features directly (`Detr3DCrossAttention`).
BEVFormer's decoder instead runs standard deformable attention over the
single flattened BEV feature grid (`bev_embed` from Phase 3) — there is
only one "camera-less" feature map, keyed by BEV xy position, so
Phase 3's already-built `MultiScaleDeformableAttention` (with
`num_levels=1`) is reused directly as the cross-attention. Everything
else (self-attention among object queries, reference-point-driven box
encoding, per-layer cls/reg branches, matcher, loss) carries over
unchanged from DETR3D's design.

## Components

### `reference_points.py` (extend existing module)
- `inverse_sigmoid(x, eps=1e-5) -> Tensor`
- `denormalize_reference_points(reference_points, pc_range) -> Tensor`:
  maps normalized `[0,1]` xyz back to metric coordinates via `pc_range`.

### `heads/bevformer_head.py`
- `BEVFormerHead(embed_dims, num_classes, box_dim=10, num_decoder_layers, pc_range)`:
  learned `query_embed`/`query_pos` (`nn.Embedding(num_queries, embed_dims)`
  each), a `reference_points` linear predicting normalized 3D xyz from
  `query_pos`, per-layer `cls_branches`/`reg_branches` (3-layer MLPs).
  `_encode_box_predictions` composes the regression output with the
  (inverse-sigmoid'd) reference point for x/y/z exactly as DETR3D does,
  producing the 10D encoded box. Same focal-loss-style bias
  initialization on the final classification layer.

### `transformer/decoder_layer.py` / `transformer/decoder.py`
- `BEVFormerDecoderLayer(embed_dims, num_heads, num_points, ffn_channels)`:
  `nn.MultiheadAttention` self-attention among the `num_queries` object
  queries, then `MultiScaleDeformableAttention(num_levels=1, num_points)`
  as cross-attention against the flattened BEV grid (reference points =
  the layer's current predicted xy, broadcast across `num_points`), then
  an FFN — same post-norm residual structure as DETR3D's decoder layer.
- `BEVFormerDecoder(embed_dims, num_queries, num_layers, num_heads, num_points)`:
  holds `query_embed`/`query_pos`, iterates layers, predicting fresh
  reference points from the current query state before each layer
  (via a caller-supplied `reference_point_predictor`, matching DETR3D's
  `Detr3DTransformer` interface so the head's `predict_reference_points`
  plugs in directly). Returns stacked intermediate hidden states and
  reference points for every layer (needed for auxiliary losses).

### `losses/loss_utils.py`
- `encode_bbox_targets`, `decode_bbox_predictions`, `wrapped_yaw_difference`
  — ported verbatim from DETR3D (box math is detector-agnostic).

### `losses/matcher.py`
- `HungarianMatcher3D(num_classes, pc_range, cls_weight, bbox_weight, alpha, gamma)`
  — ported verbatim from DETR3D (SciPy `linear_sum_assignment`, no gradient).
  Adds a new dependency: `scipy`.

### `losses/bevformer_loss.py`
- `BEVFormerLoss(num_classes, pc_range, code_weights, ...)` — ported from
  `Detr3DLoss`, renamed. `loss_by_feat(all_cls_scores, all_bbox_preds,
  batch_gt_boxes, batch_gt_labels) -> dict[str, Tensor]` with `loss_cls`/
  `loss_bbox` from the final decoder layer plus `d{i}.loss_cls`/
  `d{i}.loss_bbox` auxiliary terms for earlier layers.

## Testing

Shape/behavior-driven tests as in prior phases, tiny synthetic sizes
(`embed_dims=8`, `num_queries=6`, `bev_h=bev_w=4`, `num_classes=3`):
- Head: reference points in `[0,1]`, cls/box output shapes per layer,
  encoded box always 10D.
- Decoder: output hidden states/reference points shapes across layers,
  gradient flow end to end.
- Loss utils: encode/decode round-trip recovers the original semantic
  box (within float tolerance); yaw wrap-around difference is correct
  near +-pi.
- Matcher: with a synthetic near-perfect prediction, the matched pair is
  the correct one; an empty-GT sample produces an empty assignment
  without error.
- Loss: `loss_by_feat` on synthetic multi-layer predictions returns
  finite scalar losses with gradient flowing back into predictions; a
  batch with zero GT boxes anywhere still produces a finite classification
  loss (all-background) and doesn't crash on box loss (zero positives).

## Constraints carried over

- Pure PyTorch/TorchVision only, no MMDetection/MMCV/custom CUDA ops.
  SciPy is added only for CPU-side Hungarian assignment (no gradient
  needed there, matching DETR3D's precedent).
- Tests must not require network access or the real nuScenes dataset.
