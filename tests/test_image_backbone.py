import torch

from bevformer.models.backbone.image_backbone import MultiViewImageBackbone


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
