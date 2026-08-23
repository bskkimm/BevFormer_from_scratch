import numpy as np
import torch
from PIL import Image

from bevformer.data.transforms import photometric_distort_bgr, resize_and_normalize_image


def _make_image(width=64, height=32, color=(120, 60, 200)):
    array = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(array)


def test_resize_and_normalize_image_shape_and_dtype():
    image = _make_image(width=64, height=32)
    tensor = resize_and_normalize_image(image, image_size=(32, 64))
    assert tensor.shape == (3, 32, 64)
    assert tensor.dtype == torch.float32


def test_resize_and_normalize_image_resizes_mismatched_input():
    image = _make_image(width=100, height=50)
    tensor = resize_and_normalize_image(image, image_size=(20, 40))
    assert tensor.shape == (3, 20, 40)


def test_resize_and_normalize_image_produces_normalized_range():
    image = _make_image(width=16, height=16, color=(255, 255, 255))
    tensor = resize_and_normalize_image(image, image_size=(16, 16))
    # White pixel normalized with ImageNet mean/std should be positive and bounded.
    assert torch.all(tensor > 0)
    assert torch.all(tensor < 5.0)


def test_photometric_distort_bgr_preserves_shape_and_dtype():
    array = np.random.uniform(0, 255, size=(16, 16, 3)).astype(np.float32)
    distorted = photometric_distort_bgr(array)
    assert distorted.shape == array.shape
    assert distorted.dtype == array.dtype


def test_photometric_distort_bgr_changes_values():
    array = np.full((16, 16, 3), 128.0, dtype=np.float32)
    distorted = photometric_distort_bgr(array, seed=0)
    assert not np.allclose(distorted, array)
