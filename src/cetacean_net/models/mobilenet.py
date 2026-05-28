"""MobileNetV2 baselines for single-feature and stacked-feature classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MobileNetV2(nn.Module):
    """MobileNetV2 baseline operating on a single 1-channel feature map.

    The first conv is re-initialised to accept 1 input channel and the
    classifier head is resized to ``num_classes``. Inputs are bilinearly
    resized to 224x224 to match the backbone's expected resolution.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.target_size = (224, 224)

        self.backbone = models.mobilenet_v2(pretrained=False)
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, num_classes)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = F.interpolate(feature, size=self.target_size, mode="bilinear", align_corners=False)
        return self.backbone(feature)


class MobileNetV2Combined(nn.Module):
    """MobileNetV2 baseline that stacks MFCC/WST/MEL as the 3 input channels.

    Uses ImageNet-pretrained weights since the input keeps 3 channels.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.target_size = (224, 224)

        self.backbone = models.mobilenet_v2(pretrained=True)
        self.backbone.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, num_classes)

    def forward(self, mfcc: torch.Tensor, wst: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        mfcc = F.interpolate(mfcc, size=self.target_size, mode="bilinear", align_corners=False)
        wst = F.interpolate(wst, size=self.target_size, mode="bilinear", align_corners=False)
        mel = F.interpolate(mel, size=self.target_size, mode="bilinear", align_corners=False)

        combined = torch.cat([mfcc, wst, mel], dim=1)  # [B, 3, H, W]
        return self.backbone(combined)
