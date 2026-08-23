import torch

from bevformer.models.transformer.reference_points import (
    denormalize_reference_points,
    inverse_sigmoid,
)

PC_RANGE = (-10.0, -20.0, -2.0, 10.0, 20.0, 2.0)


def test_inverse_sigmoid_round_trips_with_sigmoid():
    x = torch.tensor([0.1, 0.5, 0.9])
    recovered = torch.sigmoid(inverse_sigmoid(x))
    torch.testing.assert_close(recovered, x, atol=1e-4, rtol=1e-4)


def test_denormalize_maps_zero_and_one_to_range_bounds():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    denorm = denormalize_reference_points(points, PC_RANGE)
    torch.testing.assert_close(denorm[0], torch.tensor([-10.0, -20.0, -2.0]))
    torch.testing.assert_close(denorm[1], torch.tensor([10.0, 20.0, 2.0]))


def test_denormalize_maps_midpoint_to_range_center():
    points = torch.tensor([[0.5, 0.5, 0.5]])
    denorm = denormalize_reference_points(points, PC_RANGE)
    torch.testing.assert_close(denorm[0], torch.tensor([0.0, 0.0, 0.0]))
