"""
Channel attention modules and position wrappers.

Standalone attention primitives:
  - SE   : Squeeze-and-Excitation
  - ECA  : Efficient Channel Attention (adaptive 1-D conv)
  - CBAM : Convolutional Block Attention (channel + spatial)

Factory:
  - _make_attn(attn_type, channels)  →  nn.Module | nn.Identity

Position wrappers (plug into any encoder–decoder pipeline):
  - InputChannelAttention   — before encoder
  - PostEncoderAttention    — after each encoder stage
  - DecoderOutputAttention  — after decoder, before head
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# Core attention modules
# ====================================================================
class SE(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class ECA(nn.Module):
    """Efficient Channel Attention — adaptive 1-D conv on channel
    descriptors, avoiding dimensionality reduction."""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class CBAM(nn.Module):
    """Convolutional Block Attention Module — sequential channel + spatial."""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        x = x * self.sigmoid(avg_out + max_out)
        avg_s = x.mean(dim=1, keepdim=True)
        max_s, _ = x.max(dim=1, keepdim=True)
        return x * self.spatial(torch.cat([avg_s, max_s], dim=1))


# ====================================================================
# Factory
# ====================================================================
def _make_attn(attn_type, channels, reduction=16):
    """Return an attention module (or ``nn.Identity`` for ``"none"``)."""
    t = attn_type.lower()
    if t == "none":
        return nn.Identity()
    if t == "se":
        return SE(channels, reduction)
    if t == "eca":
        return ECA(channels)
    if t == "cbam":
        return CBAM(channels, reduction)
    raise ValueError(f"Unknown attention type: {attn_type}")


# ====================================================================
# Position wrappers
# ====================================================================
class InputChannelAttention(nn.Module):
    """Apply channel attention to raw input tensor (before encoder)."""

    def __init__(self, in_channels, attn_type="none", reduction=16):
        super().__init__()
        self.attn = _make_attn(attn_type, in_channels, reduction)

    def forward(self, x):
        return self.attn(x)


class PostEncoderAttention(nn.Module):
    """Apply channel attention to each encoder stage output."""

    def __init__(self, channel_list, attn_type="none", reduction=16):
        super().__init__()
        self.attns = nn.ModuleList([
            _make_attn(attn_type, c, reduction) for c in channel_list
        ])

    def forward(self, feats):
        return [self.attns[i](f) for i, f in enumerate(feats)]


class DecoderOutputAttention(nn.Module):
    """Apply channel attention to the fused decoder output."""

    def __init__(self, channels, attn_type="none", reduction=16):
        super().__init__()
        self.attn = _make_attn(attn_type, channels, reduction)

    def forward(self, x):
        return self.attn(x)
