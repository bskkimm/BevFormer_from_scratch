import pytest
import torch

from bevformer.models.transformer.bev_warp import warp_prev_bev

PC_RANGE = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)  # 20m x 20m, 5 cells -> 4m/cell


def _hot_pixel_bev(bev_h, bev_w, row, col):
    grid = torch.zeros(1, bev_h, bev_w, 1)
    grid[0, row, col, 0] = 1.0
    return grid.reshape(1, bev_h * bev_w, 1)


def test_zero_delta_reproduces_input():
    bev_h = bev_w = 5
    prev_bev = torch.randn(1, bev_h * bev_w, 4)
    delta_translation = torch.zeros(1, 2)
    delta_yaw = torch.zeros(1)

    warped = warp_prev_bev(prev_bev, bev_h, bev_w, delta_translation, delta_yaw, PC_RANGE)
    torch.testing.assert_close(warped, prev_bev, atol=1e-4, rtol=1e-4)


def test_translation_shifts_content_opposite_to_ego_motion():
    bev_h = bev_w = 5
    cell_size_x = (PC_RANGE[3] - PC_RANGE[0]) / bev_w  # 4.0 m/cell
    prev_bev = _hot_pixel_bev(bev_h, bev_w, row=2, col=2)

    # Ego moved +1 cell in x; static world content should appear shifted
    # one cell in -x (toward smaller column index) in the warped output.
    delta_translation = torch.tensor([[cell_size_x, 0.0]])
    delta_yaw = torch.zeros(1)

    warped = warp_prev_bev(prev_bev, bev_h, bev_w, delta_translation, delta_yaw, PC_RANGE)
    warped_grid = warped.reshape(1, bev_h, bev_w, 1)

    peak_row, peak_col = (warped_grid[0, :, :, 0] == warped_grid[0].max()).nonzero()[0].tolist()
    assert peak_row == 2
    assert peak_col == 1


def test_output_shape_matches_input():
    bev_h, bev_w, embed_dims = 3, 4, 6
    prev_bev = torch.randn(2, bev_h * bev_w, embed_dims)
    delta_translation = torch.randn(2, 2)
    delta_yaw = torch.randn(2)

    warped = warp_prev_bev(prev_bev, bev_h, bev_w, delta_translation, delta_yaw, PC_RANGE)
    assert warped.shape == prev_bev.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast/TF32 matmul precision is a CUDA concern")
def test_warp_is_float32_exact_under_autocast_and_tf32():
    torch.manual_seed(0)
    prev_bev = torch.randn(2, 50 * 50, 8, device="cuda")
    delta_translation = torch.tensor([[3.7, -1.2], [0.4, 2.9]], device="cuda")
    delta_yaw = torch.tensor([0.05, -0.12], device="cuda")
    exact = warp_prev_bev(prev_bev, 50, 50, delta_translation, delta_yaw, (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0))

    saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            amp = warp_prev_bev(prev_bev, 50, 50, delta_translation, delta_yaw, (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved

    assert amp.dtype == torch.float32
    torch.testing.assert_close(amp, exact, atol=1e-5, rtol=1e-5)


def test_warp_aligns_static_world_points_given_bev_frame_relative_pose():
    # Features = each previous-frame cell's own (x, y) coordinates. Bilinear sampling
    # of a linear function is exact, so after warping, every current cell must hold
    # the previous-frame coordinates of the same world point: R(dyaw) q + t_rel.
    import math

    pc_range = (-10.0, -10.0, -2.0, 10.0, 10.0, 2.0)
    bev = 20  # 1 m cells
    yaw_prev, yaw_curr = 0.3, 0.5
    t_rel = torch.tensor([1.0, 3.0])  # current origin expressed in the previous BEV frame

    centers = torch.arange(bev, dtype=torch.float32) - bev / 2 + 0.5
    ys, xs = torch.meshgrid(centers, centers, indexing="ij")
    prev_bev = torch.stack([xs, ys], dim=-1).reshape(1, bev * bev, 2)

    warped = warp_prev_bev(prev_bev, bev, bev, t_rel[None], torch.tensor([yaw_curr - yaw_prev]), pc_range)
    warped = warped.reshape(bev, bev, 2)

    d = yaw_curr - yaw_prev
    rotation = torch.tensor([[math.cos(d), -math.sin(d)], [math.sin(d), math.cos(d)]])
    expected = (torch.stack([xs, ys], dim=-1) @ rotation.T) + t_rel
    inside = (expected.abs() < bev / 2 - 1).all(dim=-1)  # away from the zero-padded border
    assert inside.sum() > 100
    torch.testing.assert_close(warped[inside], expected[inside], atol=1e-3, rtol=1e-3)
