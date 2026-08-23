# Phase 4: Detection Head + Losses Implementation Plan

> **Execution note:** Executed inline, in-session, by the same agent that
> wrote this plan and the spec. Each task ends in a commit with passing
> tests.

**Goal:** Object-query decoder over `bev_embed`, detection head, Hungarian
matcher, and training loss.

**Spec:** `docs/superpowers/specs/2026-08-23-phase4-detection-head-and-losses-design.md`

## Global Constraints

- Pure PyTorch/TorchVision only, no MMDetection/MMCV/custom CUDA ops.
  SciPy added for CPU Hungarian assignment only.
- Tests must not require network access or the real nuScenes dataset.

---

### Task 1: Reference point helpers

**Files:**
- Modify: `bevformer/models/transformer/reference_points.py`
- Test: `tests/test_reference_point_helpers.py`

**Produces:** `inverse_sigmoid(x, eps=1e-5) -> Tensor`,
`denormalize_reference_points(reference_points, pc_range) -> Tensor`.

- [ ] Write failing tests (round-trip sigmoid/inverse_sigmoid; denormalize
  maps 0/1 corners to pc_range min/max).
- [ ] Implement.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add inverse_sigmoid and denormalize_reference_points helpers"

### Task 2: Detection head

**Files:**
- Create: `bevformer/models/heads/__init__.py`, `bevformer/models/heads/bevformer_head.py`
- Test: `tests/test_bevformer_head.py`

**Consumes:** Task 1's helpers.
**Produces:** `BEVFormerHead(embed_dims, num_classes, box_dim, num_decoder_layers, pc_range)`,
`init_reference_points`, `predict_reference_points`, `regress_boxes`,
`classify`, `forward_single`, `forward(hs, inter_references)`.

- [ ] Write failing tests.
- [ ] Implement (ported from DETR3D's head, box_dim=10 fixed).
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add BEVFormerHead detection head"

### Task 3: Decoder layer + decoder

**Files:**
- Create: `bevformer/models/transformer/decoder_layer.py`,
  `bevformer/models/transformer/decoder.py`
- Test: `tests/test_bevformer_decoder.py`

**Consumes:** Phase 3's `MultiScaleDeformableAttention`, Task 2's head
(`predict_reference_points` as the `reference_point_predictor` callable).
**Produces:** `BEVFormerDecoderLayer(embed_dims,num_heads,num_points,ffn_channels)`,
`BEVFormerDecoder(embed_dims,num_queries,num_layers,num_heads,num_points)`,
`forward(bev_embed[B,bev_h*bev_w,C], bev_h, bev_w, reference_point_predictor) ->
(hidden_states[L,B,Q,C], init_reference, inter_references[L,B,Q,3])`.

- [ ] Write failing tests (output shapes across layers, gradient flow).
- [ ] Implement.
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add BEVFormer decoder with deformable BEV cross-attention"

### Task 4: Loss utilities

**Files:**
- Create: `bevformer/models/losses/__init__.py`, `bevformer/models/losses/loss_utils.py`
- Test: `tests/test_loss_utils.py`

**Produces:** `encode_bbox_targets`, `decode_bbox_predictions`, `wrapped_yaw_difference`.

- [ ] Write failing tests (encode/decode round trip; yaw wrap-around).
- [ ] Implement (ported verbatim from DETR3D).
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add box encoding/decoding loss utilities"

### Task 5: Hungarian matcher

**Files:**
- Modify: `requirements.txt` (add `scipy`)
- Create: `bevformer/models/losses/matcher.py`
- Test: `tests/test_matcher.py`

**Consumes:** Task 4's `encode_bbox_targets`.
**Produces:** `HungarianMatcher3D(num_classes,pc_range,cls_weight,bbox_weight,alpha,gamma)`,
`__call__(cls_logits,box_preds,gt_boxes,gt_labels) -> list[(pred_ids,gt_ids)]`.

- [ ] Add `scipy` to `requirements.txt`, install it.
- [ ] Write failing tests (near-perfect prediction matches correct pair;
  empty GT gives empty assignment).
- [ ] Implement (ported verbatim from DETR3D).
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add Hungarian matcher for 3D box assignment"

### Task 6: BEVFormer loss

**Files:**
- Create: `bevformer/models/losses/bevformer_loss.py`
- Test: `tests/test_bevformer_loss.py`

**Consumes:** Task 4's loss utils, Task 5's matcher.
**Produces:** `BEVFormerLoss(num_classes,pc_range,code_weights,...)`,
`loss_by_feat(all_cls_scores,all_bbox_preds,batch_gt_boxes,batch_gt_labels) -> dict[str,Tensor]`.

- [ ] Write failing tests (finite scalar losses, gradient flow, zero-GT
  batch doesn't crash).
- [ ] Implement (ported from DETR3D's `Detr3DLoss`, renamed).
- [ ] Run tests, confirm pass.
- [ ] Commit: "Add BEVFormerLoss with auxiliary decoder-layer losses"

## Self-review

Spec coverage: reference point helpers, head, decoder, loss utils,
matcher, loss all covered. No placeholders. Type consistency: Task 3
consumes Task 2's exact `predict_reference_points` signature; Task 6
consumes Task 4/5's exact signatures.
