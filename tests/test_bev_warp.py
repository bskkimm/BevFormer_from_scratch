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
