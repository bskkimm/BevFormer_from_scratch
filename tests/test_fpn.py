import torch

from bevformer.models.neck.fpn import ImageFPN


def _make_stage_features(batch=2, num_cams=3):
    return {
        "stage3": torch.randn(batch, num_cams, 512, 16, 24),
        "stage4": torch.randn(batch, num_cams, 1024, 8, 12),
        "stage5": torch.randn(batch, num_cams, 2048, 4, 6),
    }


def test_fpn_produces_four_levels_with_out_channels():
    fpn = ImageFPN(in_channels=(512, 1024, 2048), out_channels=256)
    pyramid = fpn(_make_stage_features())

    assert set(pyramid.keys()) == {"p3", "p4", "p5", "p6"}
    for feat in pyramid.values():
        assert feat.shape[2] == 256


def test_fpn_output_shapes_and_downsampling_ratios():
    fpn = ImageFPN(in_channels=(512, 1024, 2048), out_channels=256)
    pyramid = fpn(_make_stage_features())

    batch, num_cams = 2, 3
    assert pyramid["p3"].shape == (batch, num_cams, 256, 16, 24)
    assert pyramid["p4"].shape == (batch, num_cams, 256, 8, 12)
    assert pyramid["p5"].shape == (batch, num_cams, 256, 4, 6)
    assert pyramid["p6"].shape == (batch, num_cams, 256, 2, 3)
