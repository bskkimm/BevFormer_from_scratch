import torch

from bevformer.models.grid_mask import GridMask


def test_eval_mode_is_identity():
    grid_mask = GridMask(probability=1.0)
    grid_mask.eval()
    images = torch.randn(2, 3, 32, 32)
    out = grid_mask(images)
    assert torch.equal(out, images)


def test_train_mode_with_probability_one_changes_output():
    torch.manual_seed(0)
    grid_mask = GridMask(probability=1.0)
    grid_mask.train()
    images = torch.ones(2, 3, 32, 32)
    out = grid_mask(images)
    assert not torch.equal(out, images)


def test_output_shape_matches_input():
    grid_mask = GridMask(probability=1.0)
    grid_mask.train()
    images = torch.randn(2, 3, 40, 56)
    out = grid_mask(images)
    assert out.shape == images.shape
