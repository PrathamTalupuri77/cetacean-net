"""CetaceanNet - the multimodal attention-based classifier and its light variant.

Both models follow the same recipe:

1. Each feature stream (MEL, WST, MFCC) is encoded by a small CNN.
2. The encoded feature maps are reshaped into token sequences and refined with
   self-attention.
3. The MEL stream queries the MFCC and WST streams via cross-attention.
4. The two cross-attended embeddings are combined by a learned modal-fusion gate.
5. A small MLP head produces the class logits.

:class:`CetaceanNet` uses standard convolutions; :class:`CetaceanNetLight`
swaps them for depthwise-separable convolutions for a much smaller parameter
count at similar accuracy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CrossAttentionBlock, ModalFusionAttention, SelfAttentionBlock
from .blocks import DepthwiseSeparableConv2d, LightResidualBlock, ResidualBlock


class _CetaceanNetBase(nn.Module):
    """Shared forward pass and head for the standard and light variants.

    Subclasses must define ``self.mel_conv``, ``self.wst_conv`` and
    ``self.mfcc_conv`` encoders before calling ``super().__init__`` finishers.
    """

    def __init__(self, num_classes: int, dim: int = 128) -> None:
        super().__init__()
        self.dim = dim

        self.mel_attn = SelfAttentionBlock(dim)
        self.wst_attn = SelfAttentionBlock(dim)
        self.mfcc_attn = SelfAttentionBlock(dim)

        self.cross_mfcc = CrossAttentionBlock(query_dim=dim, context_dim=dim)
        self.cross_wst = CrossAttentionBlock(query_dim=dim, context_dim=dim)

        self.fusion_attention = ModalFusionAttention(dim)

        self.classifier = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, mfcc: torch.Tensor, wst: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        # MEL: collapse the frequency axis, treat time frames as tokens.
        x_mel = self.mel_conv(mel)
        x_mel = x_mel.mean(dim=2).transpose(1, 2)  # [B, time, dim]
        x_mel = self.mel_attn(x_mel)

        # WST: flatten the spatial grid into a token sequence.
        x_wst = self.wst_conv(wst)
        x_wst = x_wst.flatten(2).transpose(1, 2)
        x_wst = self.wst_attn(x_wst)

        # MFCC: flatten the spatial grid into a token sequence.
        x_mfcc = self.mfcc_conv(mfcc)
        x_mfcc = x_mfcc.flatten(2).transpose(1, 2)
        x_mfcc = self.mfcc_attn(x_mfcc)

        # MEL attends to the other two modalities.
        x_mel_mfcc = self.cross_mfcc(x_mel, x_mfcc)
        x_mel_wst = self.cross_wst(x_mel, x_wst)

        pooled_mfcc = x_mel_mfcc.mean(dim=1)
        pooled_wst = x_mel_wst.mean(dim=1)

        fused = torch.stack([pooled_mfcc, pooled_wst], dim=1)  # [B, 2, dim]
        out = self.fusion_attention(fused)
        return self.classifier(out)


class CetaceanNet(_CetaceanNetBase):
    """Multimodal attention CNN using standard convolutions."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)

        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4, 128), nn.ReLU(),
            ResidualBlock(128),
        )
        self.wst_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4, 128), nn.ReLU(),
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d((16, 8)),
        )
        self.mfcc_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4, 128), nn.ReLU(),
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d((4, 4)),
        )


class CetaceanNetLight(_CetaceanNetBase):
    """Lightweight CetaceanNet using depthwise-separable convolutions."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)

        self.mel_conv = nn.Sequential(
            DepthwiseSeparableConv2d(1, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64), nn.ReLU(),
            DepthwiseSeparableConv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4, 128), nn.ReLU(),
            LightResidualBlock(128),
        )
        self.wst_conv = nn.Sequential(
            DepthwiseSeparableConv2d(1, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64), nn.ReLU(),
            DepthwiseSeparableConv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4, 128), nn.ReLU(),
            LightResidualBlock(128),
            nn.AdaptiveAvgPool2d((16, 8)),
        )
        self.mfcc_conv = nn.Sequential(
            DepthwiseSeparableConv2d(1, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64), nn.ReLU(),
            DepthwiseSeparableConv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4, 128), nn.ReLU(),
            LightResidualBlock(128),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
