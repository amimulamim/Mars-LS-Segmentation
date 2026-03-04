"""
Multi-scale feature fusion strategies and fusion registry.

Available fusions:
  - FusionLateLogits    — no feature fusion; average logits
  - FusionConcat1x1     — concat + 1×1 projection
  - FusionWeightedSum   — learnable per-stage α/β weights
  - FusionGated         — gated attention fusion
  - FusionFiLM          — Feature-wise Linear Modulation
  - FusionCrossAttention— multi-head cross-attention
  - FusionConcatSE      — concat + SE channel attention
  - FusionConcatECA     — concat + ECA channel attention
  - FusionConcatCBAM    — concat + CBAM channel+spatial attention

Factory:
  build_fusion(name, chs) → nn.Module
"""

import torch
import torch.nn as nn

from .attention import SE, ECA, CBAM


# ====================================================================
# Basic fusions
# ====================================================================
class FusionLateLogits(nn.Module):
    """No feature-level fusion; average logits from each branch."""
    name = "late_logits"

    def forward(self, A, B):
        return A, B


class FusionConcat1x1(nn.Module):
    name = "concat1x1"

    def __init__(self, chs):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(2 * c, c, 1) for c in chs])

    def forward(self, A, B):
        return [self.proj[i](torch.cat([a, b], dim=1))
                for i, (a, b) in enumerate(zip(A, B))]


class FusionWeightedSum(nn.Module):
    name = "weighted_sum"

    def __init__(self, chs):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(len(chs)))
        self.beta  = nn.Parameter(torch.ones(len(chs)))

    def forward(self, A, B):
        return [self.alpha[i] * a + self.beta[i] * b
                for i, (a, b) in enumerate(zip(A, B))]


class FusionGated(nn.Module):
    name = "gated"

    def __init__(self, chs, r=16):
        super().__init__()
        self.gates = nn.ModuleList()
        for c in chs:
            mid = max(c // r, 8)
            self.gates.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(2 * c, mid, 1), nn.ReLU(inplace=True),
                nn.Conv2d(mid, c, 1), nn.Sigmoid(),
            ))

    def forward(self, A, B):
        return [
            g(torch.cat([a, b], 1)) * a + (1 - g(torch.cat([a, b], 1))) * b
            for g, a, b in zip(self.gates, A, B)
        ]


class FusionFiLM(nn.Module):
    name = "film"

    def __init__(self, chs, r=16):
        super().__init__()
        self.film = nn.ModuleList()
        for c in chs:
            mid = max(c // r, 8)
            self.film.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, mid, 1), nn.ReLU(inplace=True),
                nn.Conv2d(mid, 2 * c, 1),
            ))

    def forward(self, A, B):
        out = []
        for i, (a, b) in enumerate(zip(A, B)):
            gb = self.film[i](b)
            gamma, beta = torch.chunk(gb, 2, dim=1)
            out.append((1 + gamma) * a + beta)
        return out


class FusionCrossAttention(nn.Module):
    name = "cross_attn"

    def __init__(self, chs, num_heads=4):
        super().__init__()
        self.proj_q = nn.ModuleList([nn.Conv2d(c, c, 1) for c in chs])
        self.proj_k = nn.ModuleList([nn.Conv2d(c, c, 1) for c in chs])
        self.proj_v = nn.ModuleList([nn.Conv2d(c, c, 1) for c in chs])
        self.attn   = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=num_heads, batch_first=True)
            for c in chs
        ])
        self.out = nn.ModuleList([nn.Conv2d(c, c, 1) for c in chs])

    def forward(self, A, B):
        outs = []
        for i, (a, b) in enumerate(zip(A, B)):
            Bn, C, H, W = a.shape
            q = self.proj_q[i](a).flatten(2).transpose(1, 2)
            k = self.proj_k[i](b).flatten(2).transpose(1, 2)
            v = self.proj_v[i](b).flatten(2).transpose(1, 2)
            y, _ = self.attn[i](q, k, v, need_weights=False)
            y = self.out[i](y.transpose(1, 2).reshape(Bn, C, H, W))
            outs.append(a + y)
        return outs


# ====================================================================
# Attention-enhanced fusions  (ablation variants)
# ====================================================================
class FusionConcatSE(nn.Module):
    """Concat + 1×1 projection + SE channel attention."""
    name = "concat_se"

    def __init__(self, chs, reduction=16):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * c, c, 1, bias=False),
                nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            ) for c in chs
        ])
        self.se = nn.ModuleList([SE(c, reduction) for c in chs])

    def forward(self, A, B):
        return [self.se[i](self.proj[i](torch.cat([a, b], dim=1)))
                for i, (a, b) in enumerate(zip(A, B))]


class FusionConcatECA(nn.Module):
    """Concat + 1×1 projection + ECA channel attention."""
    name = "concat_eca"

    def __init__(self, chs):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * c, c, 1, bias=False),
                nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            ) for c in chs
        ])
        self.eca = nn.ModuleList([ECA(c) for c in chs])

    def forward(self, A, B):
        return [self.eca[i](self.proj[i](torch.cat([a, b], dim=1)))
                for i, (a, b) in enumerate(zip(A, B))]


class FusionConcatCBAM(nn.Module):
    """Concat + 1×1 projection + CBAM channel+spatial attention."""
    name = "concat_cbam"

    def __init__(self, chs, reduction=16):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * c, c, 1, bias=False),
                nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            ) for c in chs
        ])
        self.cbam = nn.ModuleList([CBAM(c, reduction) for c in chs])

    def forward(self, A, B):
        return [self.cbam[i](self.proj[i](torch.cat([a, b], dim=1)))
                for i, (a, b) in enumerate(zip(A, B))]


# ====================================================================
# Registry & factory
# ====================================================================
FUSION_REGISTRY = {
    "late_logits":   FusionLateLogits,
    "concat1x1":     FusionConcat1x1,
    "weighted_sum":  FusionWeightedSum,
    "gated":         FusionGated,
    "film":          FusionFiLM,
    "cross_attn":    FusionCrossAttention,
    "concat_se":     FusionConcatSE,
    "concat_eca":    FusionConcatECA,
    "concat_cbam":   FusionConcatCBAM,
}


def build_fusion(name, chs):
    name = name.lower()
    if name not in FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion '{name}'. Choose from {list(FUSION_REGISTRY)}")
    cls = FUSION_REGISTRY[name]
    if name == "late_logits":
        return cls()
    return cls(chs)
