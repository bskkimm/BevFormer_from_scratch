# Phase 3: BEV Spatiotemporal Encoder Implementation Plan

> **Execution note:** Executed inline, in-session, by the same agent that
> wrote this plan and the spec. Tasks scoped at file/responsibility
> granularity. Each task ends in a commit with passing tests.

**Goal:** BEV encoder producing `bev_embed` from multi-camera FPN features
+ optional previous BEV, using deformable spatial cross-attention and
temporal self-attention.

**Spec:** `docs/superpowers/specs/2026-08-23-phase3-bev-encoder-design.md`

## Global Constraints

- Pure PyTorch/TorchVision only, no MMDetection/MMCV/custom CUDA ops.
- Tests must not require network access or the real nuScenes dataset.
- Tiny synthetic shapes in tests (`bev_h=bev_w=4`, `embed_dims=8`) for speed.

---

### Task 1: BEV reference points + camera projection

**Files:**
- Create: `bevformer/models/transformer/__init__.py`,
  `bevformer/models/transformer/reference_points.py`,
  `bevformer/models/transformer/point_sampling.py`
- Test: `tests/test_bev_reference_points.py`, `tests/test_point_sampling.py`

**Produces:** `get_bev_grid_points_2d(bev_h,bev_w) -> Tensor[bev_h*bev_w,2]`,
`get_pillar_reference_points_3d(bev_h,bev_w,pc_range,num_points_in_pillar) ->
Tensor[D,Q,3]`, `project_pillar_points_to_cameras(ref_points_3d,pc_range,img_metas)
-> (reference_points_cam[num_cam,B,Q,D,2], bev_mask[num_cam,B,Q,D])`.

- [ ] Write failing tests.
- [ ] Implement both modules.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add BEV reference points and camera projection"

### Task 2: Multi-scale deformable attention core

**Files:**
- Create: `bevformer/models/transformer/deformable_attention.py`
- Test: `tests/test_deformable_attention.py`

**Produces:** `MultiScaleDeformableAttention(embed_dims,num_heads,num_levels,num_points)`,
`forward(query[B,Q,C], reference_points[B,Q,num_levels,num_points,2], value[B,S,C],
spatial_shapes:list[(H,W)], point_mask=None) -> Tensor[B,Q,C]`.

- [ ] Write failing tests (output shape, gradient flow, masked row gives
  finite not-NaN output).
- [ ] Implement.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add pure-PyTorch multi-scale deformable attention"

### Task 3: Spatial cross-attention

**Files:**
- Create: `bevformer/models/transformer/spatial_cross_attention.py`
- Test: `tests/test_spatial_cross_attention.py`

**Consumes:** Task 1's `project_pillar_points_to_cameras` output shapes,
Task 2's `MultiScaleDeformableAttention`.
**Produces:** `SpatialCrossAttention(embed_dims,num_cams,num_levels,num_points_in_pillar)`,
`forward(query[B,Q,C], mlvl_feats:list[Tensor[B,N,C,H,W]], reference_points_cam,
bev_mask) -> Tensor[B,Q,C]`.

- [ ] Write failing tests (output shape; query with zero valid cameras
  gives finite zero-ish output, not NaN).
- [ ] Implement.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add spatial cross-attention (image to BEV)"

### Task 4: Temporal self-attention

**Files:**
- Create: `bevformer/models/transformer/temporal_self_attention.py`
- Test: `tests/test_temporal_self_attention.py`

**Consumes:** Task 2's `MultiScaleDeformableAttention`.
**Produces:** `TemporalSelfAttention(embed_dims,num_heads,num_points)`,
`forward(query[B,Q,C], prev_bev:Tensor[B,Q,C]|None, bev_h,bev_w) -> Tensor[B,Q,C]`.

- [ ] Write failing tests (output shape with and without `prev_bev`).
- [ ] Implement.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add temporal self-attention (BEV to BEV)"

### Task 5: Positional encoding + previous-BEV warp

**Files:**
- Create: `bevformer/models/transformer/positional_encoding.py`,
  `bevformer/models/transformer/bev_warp.py`
- Test: `tests/test_bev_positional_encoding.py`, `tests/test_bev_warp.py`

**Produces:** `LearnedBEVPositionalEncoding(bev_h,bev_w,embed_dims)`,
`forward(batch_size) -> Tensor[B,bev_h*bev_w,embed_dims]`;
`warp_prev_bev(prev_bev,bev_h,bev_w,delta_translation_bev,delta_yaw,pc_range) ->
Tensor[same shape]`.

- [ ] Write failing tests (pos encoding output shape and batch
  broadcasting; warp with zero delta reproduces input; warp with nonzero
  translation shifts a synthetic hot-pixel in the expected direction).
- [ ] Implement.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add BEV positional encoding and previous-BEV ego-motion warp"

### Task 6: Encoder layer + encoder stack

**Files:**
- Create: `bevformer/models/transformer/encoder_layer.py`,
  `bevformer/models/transformer/encoder.py`
- Test: `tests/test_bev_encoder.py`

**Consumes:** Tasks 1-5's modules.
**Produces:** `BEVFormerLayer(embed_dims,...)`,
`BEVFormerEncoder(num_layers,bev_h,bev_w,embed_dims,pc_range,...)`,
`forward(mlvl_feats, img_metas, prev_bev=None, delta_translation_bev=None,
delta_yaw=None) -> Tensor[B,bev_h*bev_w,embed_dims]`.

- [ ] Write failing tests (output shape without `prev_bev`; output shape
  and no-crash with `prev_bev` supplied; gradient flow end to end).
- [ ] Implement.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add BEVFormer encoder layer and encoder stack"

## Self-review

Spec coverage: reference points/projection, deformable attention core,
SCA, TSA, positional encoding, prev-BEV warp, encoder layer/stack all
covered. No placeholders. Type consistency: Task 3/4 consume exactly
Task 2's `MultiScaleDeformableAttention` signature; Task 6 consumes
Tasks 1-5's exact produced signatures.
