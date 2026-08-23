"""Batch sampling for BevFormerNuScenesDataset.

Unlike frame-by-frame streaming BEVFormer training loops (which need a
scene-aware sampler to avoid mixing history across scene boundaries
within a batch), `BevFormerNuScenesDataset.__getitem__` already returns a
self-contained temporal queue per sample (padded at scene boundaries, see
`nuscenes_dataset.py`). Batches made of independently-sampled queues have
no cross-sample temporal interaction, so a plain shuffling sampler is
correct and sufficient here — no custom scene-aware logic is needed.
"""

from __future__ import annotations

# Use torch.utils.data.RandomSampler (or DataLoader's default shuffle=True)
# directly — no custom sampler is defined in this module.
