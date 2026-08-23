# Phase 2: Image Backbone + FPN Neck

## Goal

Implement the multi-camera image feature extractor: a pure-PyTorch ResNet
backbone with deformable convolutions (official BEVFormer style) and a
top-down FPN neck producing 4 output levels, plus the GridMask training
augmentation used by official BEVFormer. This is Phase 2 of 5, building on
Phase 1's data pipeline.

## Non-goals (deferred to later phases)

- BEV encoder (spatial/temporal attention), detection head, losses,
  training loop — Phases 3-5.
- Backbone checkpoint initialization from an external pretrained detector
  (e.g. FCOS3D) — not required to unblock later phases; TorchVision
  ImageNet weights are sufficient for now.

## Components

### `bevformer/models/backbone/image_backbone.py`

`MultiViewImageBackbone(variant="resnet50", pretrained=True, frozen_stages=1,
norm_eval=True)`:
- Builds a TorchVision `resnet50` or `resnet101` and splits it into
  `stem`, `stage2..stage5` (C2-C5).
- Replaces the 3x3 conv (`conv2`) of every block in `stage4` and `stage5`
  with a `DeformConv2dPack` (a small conv predicting per-location offsets,
  feeding `torchvision.ops.DeformConv2d`), matching official BEVFormer's
  DCN-in-late-stages backbone. Deform offsets initialize to zero so the
  backbone starts numerically identical to plain ResNet, then learns
  offsets during training.
- `frozen_stages`/`norm_eval` freeze early stages and keep BatchNorm in
  eval mode during training, standard detection-backbone practice (avoids
  BN statistics drifting on small per-GPU batch sizes).
- `forward(images: [B, N, 3, H, W]) -> {"stage3": [B,N,C3,H3,W3],
  "stage4": [...], "stage5": [...]}` — flattens the camera dimension into
  the batch dimension for the 2D conv stack, then reshapes back.

### `bevformer/models/neck/fpn.py`

`ImageFPN(in_channels=(512,1024,2048), out_channels=256,
out_names=("p3","p4","p5","p6"))`:
- 1x1 lateral convs unify channel width from `stage3/4/5`, top-down
  addition with nearest-neighbor upsampling, 3x3 output convs, plus one
  extra stride-2 conv producing a 4th coarser level (`p6`) from `p5`.
- `forward(features: dict[str, Tensor[B,N,C,H,W]]) -> dict[str, Tensor[B,N,out_channels,H_l,W_l]]`
  for `p3..p6`, preserving the multi-view layout.

### `bevformer/models/grid_mask.py`

`GridMask(use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5,
mode=1, probability=0.7)`: random grid-pattern masking augmentation
applied to image tensors during training only (`self.training` gate),
identity at eval time or when the per-call random draw exceeds
`probability`.

## Testing

- `tests/test_image_backbone.py`: forward pass on tiny synthetic
  multi-camera input (small H/W, `pretrained=False` to avoid network
  downloads) produces the right stage channel counts and shapes; deform
  conv stages produce finite outputs; `frozen_stages` actually disables
  gradients on frozen parameters.
- `tests/test_fpn.py`: forward pass on synthetic stage3/4/5 features
  produces 4 output levels with `out_channels` channels and expected
  spatial downsampling ratios relative to the input stage3 map.
- `tests/test_grid_mask.py`: in eval mode, output equals input exactly;
  in train mode with `probability=1.0`, output differs from input at
  least somewhere (mask actually applied); output shape always matches
  input shape.

## Constraints carried over

- Pure PyTorch/TorchVision only.
- No MMDetection/MMCV.
- Tests must not require network access or the real nuScenes dataset
  (`pretrained=False` in backbone tests).
