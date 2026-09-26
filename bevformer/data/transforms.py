"""Per-camera image resize, normalize, and photometric distortion."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_image_uint8(image: Image.Image, image_size: tuple[int, int] = (900, 1600)) -> torch.Tensor:
    """PIL RGB image -> uint8 [3, H, W] at `image_size` (H, W), not normalized."""
    height, width = image_size
    if image.size != (width, height):
        image = image.resize((width, height))
    return torch.from_numpy(np.ascontiguousarray(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)))


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """uint8 [..., 3, H, W] -> float32 ImageNet-normalized, on whatever device `images` is on."""
    mean = torch.as_tensor(IMAGENET_MEAN, device=images.device).view(3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=images.device).view(3, 1, 1)
    return (images.float() / 255.0 - mean) / std


def resize_and_normalize_image(
    image: Image.Image, image_size: tuple[int, int] = (900, 1600)
) -> torch.Tensor:
    return normalize_images(resize_image_uint8(image, image_size))


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
