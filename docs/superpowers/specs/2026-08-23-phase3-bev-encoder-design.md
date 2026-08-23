# Phase 3: BEV Spatiotemporal Encoder

## Goal

Implement BEVFormer's core contribution: a BEV encoder that turns
multi-camera image features into a unified bird's-eye-view feature grid,
using deformable spatial cross-attention (image → BEV) and deformable
temporal self-attention (BEV → BEV across frames, aligned by ego motion).
Pure PyTorch — deformable attention is implemented with `grid_sample`
rather than a custom CUDA kernel (there is no official CUDA op available
without MMCV, and a pure-PyTorch fallback is the same approach MMCV itself
ships for CPU execution).

## Non-goals (deferred to later phases)

- Detection head, losses (Phase 4).
- Object-query decoder (Phase 4) — this phase only produces the BEV
  feature grid `bev_embed`.
- Exact reproduction of official per-camera dynamic query gathering
  (indexing only the queries a camera can see). This phase instead runs
  every camera over every query and masks/reweights invalid
  projections, which is mathematically equivalent but simpler pure-PyTorch
  code, at the cost of extra (masked-out) compute. Acceptable for
  correctness-focused Phase 3; can be optimized later if profiling shows
  it matters.

## Components

### `reference_points.py`
- `get_bev_grid_points_2d(bev_h, bev_w) -> Tensor[bev_h*bev_w, 2]`: normalized
  `[0,1]` xy cell centers of the BEV grid, row-major (`y` = row, `x` = col).
- `get_pillar_reference_points_3d(bev_h, bev_w, pc_range, num_points_in_pillar) ->
  Tensor[num_points_in_pillar, bev_h*bev_w, 3]`: for every BEV cell, `D`
  points stacked vertically (evenly spaced in z between `pc_range[2]` and
  `pc_range[5]`), normalized to `[0,1]` in xyz.

### `point_sampling.py`
- `project_pillar_points_to_cameras(reference_points_3d, pc_range, img_metas) ->
  (reference_points_cam: Tensor[num_cam, B, Q, D, 2], bev_mask: Tensor[num_cam, B, Q, D])`:
  denormalizes the 3D points using `pc_range`, projects into every camera
  via each `img_meta["lidar2img"]` (reusing the DETR3D-style perspective
  divide + image-shape normalization), producing normalized `[0,1]` image
  coordinates and a validity mask (in front of camera, inside image
  bounds).

### `deformable_attention.py`
- `MultiScaleDeformableAttention(embed_dims, num_heads, num_levels, num_points)`:
  a generalized pure-PyTorch multi-scale deformable attention. Reference
  points have shape `[B, Q, num_levels, num_points, 2]` (allowed to differ
  per level/point, not just broadcast from a single point) — this
  generalization is what lets one module serve both Spatial
  Cross-Attention (`num_points` = pillar height samples, each with its own
  projected 2D location per camera) and Temporal Self-Attention
  (`num_levels=2` pseudo-levels for [previous BEV, current BEV], same
  location, small learned point offsets). Learned per-(head, level,
  point) offsets are added to the reference locations; attention weights
  are a softmax over `(levels * points)`; sampling uses `F.grid_sample`
  per level. Accepts an optional `point_mask: [B, Q, num_levels, num_points]`
  that biases invalid locations' pre-softmax logits strongly negative
  (not `-inf`, to avoid NaN when a row is entirely masked).

### `spatial_cross_attention.py`
- `SpatialCrossAttention(embed_dims, num_cams, num_levels, num_points_in_pillar)`:
  for each camera, flattens that camera's multi-level FPN features into a
  single `value` sequence + `spatial_shapes`, runs
  `MultiScaleDeformableAttention` with the camera's projected pillar
  points (masked per `bev_mask`), and averages the per-camera outputs
  over the cameras that actually see each query (weighted by whether that
  camera has ≥1 valid pillar point for the query; queries seen by zero
  cameras get zero output, which is expected only far outside sensor
  coverage and is not specially handled).

### `temporal_self_attention.py`
- `TemporalSelfAttention(embed_dims, num_heads, num_points)`: treats the
  BEV grid at the current query locations and a (possibly warped)
  previous BEV feature map as 2 pseudo-levels of the same spatial size,
  and runs `MultiScaleDeformableAttention` with the BEV grid's own 2D
  reference points identical across both pseudo-levels. When no previous
  BEV is available (first frame / scene boundary — `prev_bev_exists=False`),
  the previous-BEV pseudo-level is filled with the current query itself
  (mirrors official behavior of duplicating the current BEV when there is
  no history).

### `positional_encoding.py`
- `LearnedBEVPositionalEncoding(bev_h, bev_w, embed_dims)`: learned
  row-embedding + column-embedding tables summed per cell, standard
  factorized 2D learned positional encoding.

### `bev_warp.py`
- `warp_prev_bev(prev_bev: Tensor[B, bev_h*bev_w, C], bev_h, bev_w,
  delta_translation_bev: Tensor[B, 2], delta_yaw: Tensor[B]) -> Tensor[same shape]`:
  aligns a previous BEV feature map to the current ego frame with a 2D
  rigid transform (rotate by `-delta_yaw`, translate by
  `-delta_translation_bev` converted from meters to grid cells via
  `pc_range`), implemented with `F.affine_grid` + `F.grid_sample`
  (bilinear, zero-padded — cells shifted out of the current view are
  discarded, matching official BEVFormer's `feature_map` warp behavior).

### `encoder_layer.py` / `encoder.py`
- `BEVFormerLayer(embed_dims, ...)`: `TemporalSelfAttention` → residual +
  LayerNorm → `SpatialCrossAttention` → residual + LayerNorm → FFN →
  residual + LayerNorm (post-norm, standard transformer layer ordering).
- `BEVFormerEncoder(num_layers, bev_h, bev_w, ...)`: holds the learned BEV
  query embedding and positional encoding; `forward(mlvl_feats, img_metas,
  prev_bev=None, delta_translation_bev=None, delta_yaw=None) -> bev_embed:
  Tensor[B, bev_h*bev_w, C]`. Warps `prev_bev` once (if provided) before
  the layer stack, adds positional encoding to the query at every layer,
  and feeds each layer's output as the next layer's query (iterative
  refinement, matching official design).

## Testing

Every module gets shape/behavior-driven unit tests (consistent with
Phase 2): correct output shapes for tiny synthetic inputs (`bev_h=bev_w=4`,
`embed_dims=8`, 2 cameras, 2 FPN levels), gradient flow (`loss.backward()`
doesn't error and populates `.grad` on learnable parameters), masking
correctness (an all-invalid-camera query produces a finite, not-NaN
output), and `warp_prev_bev` sanity (zero delta reproduces the input;
nonzero translation shifts content in the expected direction on a
synthetic single-hot-pixel input).

## Constraints carried over

- Pure PyTorch/TorchVision only, no MMDetection/MMCV/custom CUDA ops.
- Tests must not require network access or the real nuScenes dataset.
