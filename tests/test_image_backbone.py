import pytest
import torch
import torch.nn as nn

from bevformer.models.backbone.image_backbone import (
    DeformConv2dPack,
    ModulatedDeformConv2dPack,
    MultiViewImageBackbone,
)


def test_modulated_deform_conv_matches_plain_conv_at_init():
    torch.manual_seed(0)
    plain = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
    modulated = ModulatedDeformConv2dPack(plain)
    x = torch.randn(2, 8, 10, 12)
    with torch.no_grad():
        torch.testing.assert_close(modulated(x), plain(x), atol=1e-5, rtol=1e-5)


def test_modulated_deform_conv_predicts_offsets_and_masks():
    plain = nn.Conv2d(4, 4, kernel_size=3, padding=1)
    modulated = ModulatedDeformConv2dPack(plain)
    # 2 offset channels (dy, dx) + 1 mask channel per kernel tap.
    assert modulated.conv_offset.out_channels == 3 * 3 * 3


def test_modulated_deform_conv_offset_predictor_receives_gradient():
    torch.manual_seed(0)
    modulated = ModulatedDeformConv2dPack(nn.Conv2d(4, 4, kernel_size=3, padding=1))
    x = torch.randn(1, 4, 8, 8)
    modulated(x).square().sum().backward()
    assert modulated.conv_offset.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("dcn", "expected_type"),
    [("v2", ModulatedDeformConv2dPack), ("v1", DeformConv2dPack), ("none", nn.Conv2d)],
)
def test_dcn_option_selects_late_stage_conv_type(dcn, expected_type):
    backbone = MultiViewImageBackbone(variant="resnet50", pretrained=False, frozen_stages=-1, dcn=dcn)
    for stage in (backbone.stage4, backbone.stage5):
        for block in stage:
            assert type(block.conv2) is expected_type
    for block in backbone.stage3:
        assert type(block.conv2) is nn.Conv2d


def test_unknown_dcn_option_raises():
    with pytest.raises(ValueError):
        MultiViewImageBackbone(variant="resnet50", pretrained=False, dcn="v3")


def test_dcnv2_backbone_equals_plain_resnet_at_init():
    images = torch.randn(1, 1, 3, 64, 64)
    torch.manual_seed(0)
    plain = MultiViewImageBackbone(variant="resnet50", pretrained=False, frozen_stages=-1, dcn="none").eval()
    torch.manual_seed(0)
    dcnv2 = MultiViewImageBackbone(variant="resnet50", pretrained=False, frozen_stages=-1, dcn="v2").eval()
    with torch.no_grad():
        plain_feats = plain(images)
        dcnv2_feats = dcnv2(images)
    for name in plain_feats:
        torch.testing.assert_close(dcnv2_feats[name], plain_feats[name], atol=1e-4, rtol=1e-4)


def test_forward_produces_expected_stage_channels_and_shapes():
    backbone = MultiViewImageBackbone(variant="resnet50", pretrained=False, frozen_stages=-1)
    images = torch.randn(2, 3, 3, 64, 96)  # B=2, N=3 cams, 3x64x96
    features = backbone(images)

    assert set(features.keys()) == {"stage3", "stage4", "stage5"}
    assert features["stage3"].shape == (2, 3, 512, 8, 12)
    assert features["stage4"].shape == (2, 3, 1024, 4, 6)
    assert features["stage5"].shape == (2, 3, 2048, 2, 3)


def test_deformable_stages_produce_finite_output():
    backbone = MultiViewImageBackbone(variant="resnet50", pretrained=False, frozen_stages=-1)
    images = torch.randn(1, 1, 3, 64, 96)
    features = backbone(images)
    for feat in features.values():
        assert torch.isfinite(feat).all()


def test_frozen_stages_disable_gradients_on_stem_and_early_stages():
    backbone = MultiViewImageBackbone(variant="resnet50", pretrained=False, frozen_stages=1)
    for param in backbone.stem.parameters():
        assert not param.requires_grad
    for param in backbone.stage2.parameters():
        assert not param.requires_grad
    # stage3 (index 2) is above frozen_stages=1, so it should remain trainable.
    assert any(param.requires_grad for param in backbone.stage3.parameters())


def test_unfrozen_backbone_has_trainable_stem():
    backbone = MultiViewImageBackbone(variant="resnet50", pretrained=False, frozen_stages=-1)
    assert all(param.requires_grad for param in backbone.stem.parameters())
