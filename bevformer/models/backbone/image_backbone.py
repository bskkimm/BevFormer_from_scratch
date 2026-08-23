"""Multi-view image backbone: ResNet with deformable convolutions."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision.ops import DeformConv2d

_WEIGHTS = {
    "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V2),
}


class DeformConv2dPack(nn.Module):
    """Drop-in replacement for a ResNet 3x3 conv using learned offsets."""

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        kernel_h, kernel_w = conv.kernel_size
        offset_channels = 2 * kernel_h * kernel_w
        self.offset_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=offset_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=True,
        )
        self.deform_conv = DeformConv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=kernel_h,
            stride=conv.stride[0],
            padding=conv.padding[0],
            dilation=conv.dilation[0],
            groups=conv.groups,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            self.deform_conv.weight.copy_(conv.weight)
            if conv.bias is not None and self.deform_conv.bias is not None:
                self.deform_conv.bias.copy_(conv.bias)
            nn.init.zeros_(self.offset_conv.weight)
            nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        return self.deform_conv(x, offset)


class MultiViewImageBackbone(nn.Module):
    """Apply a torchvision ResNet backbone to all camera views.

    Input: images [B, N, 3, H, W]
    Output: dict[str, Tensor] with "stage3"/"stage4"/"stage5", each [B, N, C, H_l, W_l]
    """

    def __init__(
        self,
        variant: str = "resnet50",
        pretrained: bool = True,
        frozen_stages: int = 1,
        norm_eval: bool = True,
    ) -> None:
        super().__init__()
        if variant not in _WEIGHTS:
            raise ValueError(f"Unsupported backbone variant: {variant}")
        constructor, weights_enum = _WEIGHTS[variant]
        backbone = constructor(weights=weights_enum if pretrained else None)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stage2 = backbone.layer1
        self.stage3 = backbone.layer2
        self.stage4 = backbone.layer3
        self.stage5 = backbone.layer4
        self._convert_stage_to_deformable(self.stage4)
        self._convert_stage_to_deformable(self.stage5)

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self._freeze_stages()

    @staticmethod
    def _convert_stage_to_deformable(stage: nn.Sequential) -> None:
        for block in stage:
            if hasattr(block, "conv2") and isinstance(block.conv2, nn.Conv2d):
                block.conv2 = DeformConv2dPack(block.conv2)

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
        stages = [self.stage2, self.stage3, self.stage4, self.stage5]
        for idx, stage in enumerate(stages, start=1):
            if self.frozen_stages >= idx:
                stage.eval()
                for param in stage.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True) -> "MultiViewImageBackbone":
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, num_cams, channels, height, width = images.shape
        flat = images.reshape(batch * num_cams, channels, height, width)

        x = self.stem(flat)
        stage2 = self.stage2(x)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)

        features = {"stage3": stage3, "stage4": stage4, "stage5": stage5}
        multi_view = {}
        for name, feat in features.items():
            _, out_channels, out_height, out_width = feat.shape
            multi_view[name] = feat.reshape(batch, num_cams, out_channels, out_height, out_width)
        return multi_view
