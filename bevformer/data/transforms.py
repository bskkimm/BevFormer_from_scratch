"""Per-camera image resize, normalize, and photometric distortion."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_and_normalize_image(
    image: Image.Image, image_size: tuple[int, int] = (900, 1600)
) -> torch.Tensor:
    height, width = image_size
    if image.size != (width, height):
        image = image.resize((width, height))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def photometric_distort_bgr(
    array: np.ndarray,
    *,
    brightness_delta: float = 32.0,
    contrast_range: tuple[float, float] = (0.5, 1.5),
    saturation_range: tuple[float, float] = (0.5, 1.5),
    hue_delta: float = 18.0,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = array.copy()

    out = out + rng.uniform(-brightness_delta, brightness_delta)

    contrast_factor = rng.uniform(*contrast_range)
    out = out * contrast_factor

    saturation_factor = rng.uniform(*saturation_range)
    gray = out.mean(axis=-1, keepdims=True)
    out = gray + (out - gray) * saturation_factor

    hue_shift = rng.uniform(-hue_delta, hue_delta)
    out = out + hue_shift

    return np.clip(out, 0.0, 255.0).astype(array.dtype)
