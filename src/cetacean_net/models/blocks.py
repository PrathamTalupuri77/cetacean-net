"""Convolutional building blocks (standard and depthwise-separable residuals)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Standard two-conv residual block with GroupNorm and ReLU."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(4, in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(4, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise conv followed by a pointwise (1x1) conv - the MobileNet motif."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class LightResidualBlock(nn.Module):
    """Residual block built from depthwise-separable convolutions (cheaper)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.dsconv1 = DepthwiseSeparableConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(4, in_channels)
        self.relu = nn.ReLU()
        self.dsconv2 = DepthwiseSeparableConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(4, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.dsconv1(x)))
        out = self.bn2(self.dsconv2(out))
        out = out + residual
        return self.relu(out)
