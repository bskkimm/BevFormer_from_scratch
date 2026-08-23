import torch

from bevformer.models.transformer.reference_points import (
    get_bev_grid_points_2d,
    get_pillar_reference_points_3d,
)

PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)


def test_bev_grid_points_2d_shape_and_range():
    points = get_bev_grid_points_2d(bev_h=4, bev_w=4)
    assert points.shape == (16, 2)
    assert torch.all(points >= 0.0) and torch.all(points <= 1.0)


def test_bev_grid_points_2d_row_major_ordering():
    points = get_bev_grid_points_2d(bev_h=2, bev_w=2)
    # Row-major: index 0,1 share the same y (row 0); index 2,3 share row 1.
    assert points[0, 1] == points[1, 1]
    assert points[2, 1] == points[3, 1]
    assert points[0, 1] != points[2, 1]


def test_pillar_reference_points_3d_shape_and_range():
    points = get_pillar_reference_points_3d(
        bev_h=4, bev_w=4, pc_range=PC_RANGE, num_points_in_pillar=3
    )
    assert points.shape == (3, 16, 3)
    assert torch.all(points >= 0.0) and torch.all(points <= 1.0)


def test_pillar_reference_points_3d_xy_matches_bev_grid():
    grid = get_bev_grid_points_2d(bev_h=4, bev_w=4)
    pillar = get_pillar_reference_points_3d(
        bev_h=4, bev_w=4, pc_range=PC_RANGE, num_points_in_pillar=3
    )
    for d in range(3):
        torch.testing.assert_close(pillar[d, :, :2], grid)


def test_pillar_reference_points_3d_heights_are_distinct():
    points = get_pillar_reference_points_3d(
        bev_h=2, bev_w=2, pc_range=PC_RANGE, num_points_in_pillar=3
    )
    heights = points[:, 0, 2]
    assert len(torch.unique(heights)) == 3
