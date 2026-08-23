"""GridMask training augmentation used by official BEVFormer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GridMask(nn.Module):
    def __init__(
        self,
        use_h: bool = True,
        use_w: bool = True,
        rotate: int = 1,
        offset: bool = False,
        ratio: float = 0.5,
        mode: int = 1,
        probability: float = 0.7,
    ) -> None:
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.probability = probability

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.training or np.random.rand() > self.probability:
            return images
        height, width = images.shape[-2:]
        canvas_height = int(1.5 * height)
        canvas_width = int(1.5 * width)
        spacing = np.random.randint(2, height)
        line_width = min(max(int(spacing * self.ratio + 0.5), 1), spacing - 1)
        mask = np.ones((canvas_height, canvas_width), dtype=np.float32)
        start_h = np.random.randint(spacing)
        start_w = np.random.randint(spacing)
        if self.use_h:
            for index in range(canvas_height // spacing):
                start = spacing * index + start_h
                mask[start : min(start + line_width, canvas_height), :] = 0
        if self.use_w:
            for index in range(canvas_width // spacing):
                start = spacing * index + start_w
                mask[:, start : min(start + line_width, canvas_width)] = 0

        rotation = np.random.randint(self.rotate) if self.rotate > 1 else 0
        mask = np.asarray(Image.fromarray(mask).rotate(rotation)).copy()
        top = (canvas_height - height) // 2
        left = (canvas_width - width) // 2
        mask = mask[top : top + height, left : left + width]
        mask_tensor = torch.as_tensor(mask, device=images.device, dtype=images.dtype)
        if self.mode == 1:
            mask_tensor = 1 - mask_tensor
        mask_tensor = mask_tensor.view(1, 1, height, width)
        if self.offset:
            offset_tensor = (
                torch.rand(1, images.shape[-3], height, width, device=images.device, dtype=images.dtype) * 2 - 1
            )
            return images * mask_tensor + offset_tensor * (1 - mask_tensor)
        return images * mask_tensor
