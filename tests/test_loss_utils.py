import math

import torch

from bevformer.models.losses.loss_utils import (
    decode_bbox_predictions,
    encode_bbox_targets,
    wrapped_yaw_difference,
)


def test_encode_decode_round_trip_recovers_semantic_box():
    boxes = torch.tensor([[1.0, 2.0, 0.5, 2.0, 4.5, 1.6, 0.3, 1.0, -0.5]])
    encoded = encode_bbox_targets(boxes)
    decoded = decode_bbox_predictions(encoded)
    torch.testing.assert_close(decoded, boxes, atol=1e-4, rtol=1e-4)


def test_encode_empty_boxes_returns_empty_10d():
    boxes = torch.zeros((0, 9))
    encoded = encode_bbox_targets(boxes)
    assert encoded.shape == (0, 10)


def test_wrapped_yaw_difference_handles_wraparound():
    pred = torch.tensor([math.pi - 0.1])
    target = torch.tensor([-math.pi + 0.1])
    diff = wrapped_yaw_difference(pred, target)
    assert abs(diff.item()) < 0.3  # true angular difference is ~0.2, not ~2*pi


def test_wrapped_yaw_difference_zero_for_equal_angles():
    angle = torch.tensor([1.234])
    diff = wrapped_yaw_difference(angle, angle)
    torch.testing.assert_close(diff, torch.zeros_like(diff), atol=1e-6, rtol=1e-6)
