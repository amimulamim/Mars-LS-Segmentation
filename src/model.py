"""
Dual-Encoder Swin V2 segmentation model with pluggable decoders and fusions.

Architecture
------------
  RGB (3ch) ──► Swin V2 encoder ──┐
                                   ├── Fusion ──► Decoder ──► 1×1 head ──► mask
  AUX (4ch) ──► Swin V2 encoder ──┘

Supported decoders : upernet, segformer_mlp, deeplabv3plus, unetplusplus, fpn
Supported fusions  : late_logits, concat1x1, weighted_sum, gated, film, cross_attn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ═══════════════════════════════════════════════════════════════
# Backbone helpers
# ═══════════════════════════════════════════════════════════════
def adapt_patch_embed_in_chans(model, in_chans_new):
    """Replace patch-embed conv to accept *in_chans_new* channels,
    reusing pretrained RGB weights and mean-initializing extras."""
    pe = model.patch_embed
    old_conv = pe.proj
    old_w = old_conv.weight.data
    embed_dim, old_in, kh, kw = old_w.shape
    assert old_in == 3, f"Expected 3ch pretrained, got {old_in}"
    new_conv = nn.Conv2d(
        in_chans_new, embed_dim,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    with torch.no_grad():
        new_w = torch.zeros(embed_dim, in_chans_new, kh, kw, device=old_w.device)
        new_w[:, :3] = old_w
        if in_chans_new > 3:
            rgb_mean = old_w.mean(dim=1, keepdim=True)
            new_w[:, 3:] = rgb_mean.repeat(1, in_chans_new - 3, 1, 1)
        new_conv.weight.copy_(new_w)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias.data)
    pe.proj = new_conv
    return model


def make_swin_features(encoder_name, pretrained=True, img_size=128):
    enc = timm.create_model(
        encoder_name,
        pretrained=pretrained,
        features_only=True,
        out_indices=(0, 1, 2, 3),
        img_size=img_size,
    )
    if hasattr(enc, "patch_embed"):
        enc.patch_embed.img_size = None
        if hasattr(enc.patch_embed, "strict_img_size"):
            enc.patch_embed.strict_img_size = False
    return enc


def to_nchw(feats, in_chs):
    """Convert (B, H, W, C) → (B, C, H, W) if needed."""
    out = []
    for f, c in zip(feats, in_chs):
        if f.ndim == 4 and f.shape[-1] == c and f.shape[1] != c:
            f = f.permute(0, 3, 1, 2).contiguous()
        out.append(f)
    return out


# ═══════════════════════════════════════════════════════════════
# Common building block
# ═══════════════════════════════════════════════════════════════
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ═══════════════════════════════════════════════════════════════
# Decoders
# ═══════════════════════════════════════════════════════════════

# ─── 1. UPerNet (PPM + FPN) ──────────────────────────────────
class PPM(nn.Module):
    def __init__(self, in_ch, out_ch, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        inter = max(out_ch // len(pool_sizes), 32)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_ch, inter, 1, bias=False),
                nn.BatchNorm2d(inter),
                nn.ReLU(inplace=True),
            ) for ps in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch + inter * len(pool_sizes), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [x] + [
            F.interpolate(s(x), size=(h, w), mode="bilinear", align_corners=False)
            for s in self.stages
        ]
        return self.bottleneck(torch.cat(outs, dim=1))


class UPerNetDecoder(nn.Module):
    def __init__(self, in_channels_list, fpn_channels=256):
        super().__init__()
        self.ppm = PPM(in_channels_list[-1], fpn_channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, 1) for c in in_channels_list[:-1]
        ])
        self.fpn_convs = nn.ModuleList([
            ConvBNReLU(fpn_channels, fpn_channels) for _ in in_channels_list[:-1]
        ])
        self.fuse = nn.Sequential(
            ConvBNReLU(fpn_channels * 4, fpn_channels),
            nn.Dropout2d(0.1),
        )

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        p4 = self.ppm(c4)
        p3 = self.lateral_convs[2](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral_convs[1](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral_convs[0](c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.fpn_convs[2](p3)
        p2 = self.fpn_convs[1](p2)
        p1 = self.fpn_convs[0](p1)
        h, w = p1.shape[-2:]
        return self.fuse(torch.cat([
            p1,
            F.interpolate(p2, size=(h, w), mode="bilinear", align_corners=False),
            F.interpolate(p3, size=(h, w), mode="bilinear", align_corners=False),
            F.interpolate(p4, size=(h, w), mode="bilinear", align_corners=False),
        ], dim=1))


# ─── 2. SegFormer All-MLP ────────────────────────────────────
class SegFormerMLPDecoder(nn.Module):
    def __init__(self, in_channels_list, fpn_channels=256):
        super().__init__()
        self.linear_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fpn_channels, 1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            ) for c in in_channels_list
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_channels * 4, fpn_channels, 1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, feats):
        target = feats[0].shape[-2:]
        outs = []
        for i, f in enumerate(feats):
            x = self.linear_projs[i](f)
            if x.shape[-2:] != target:
                x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
            outs.append(x)
        return self.fuse(torch.cat(outs, dim=1))


# ─── 3. DeepLabV3+ ───────────────────────────────────────────
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):
        super().__init__()
        modules = [nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )]
        for r in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ))
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (2 + len(rates)), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = []
        for conv in self.convs:
            y = conv(x)
            if y.shape[-2:] != (h, w):
                y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(y)
        return self.project(torch.cat(outs, dim=1))


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, in_channels_list, fpn_channels=256):
        super().__init__()
        self.aspp = ASPP(in_channels_list[-1], fpn_channels)
        self.low_proj = nn.Sequential(
            nn.Conv2d(in_channels_list[0], 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            ConvBNReLU(fpn_channels + 48, fpn_channels),
            ConvBNReLU(fpn_channels, fpn_channels),
            nn.Dropout2d(0.1),
        )

    def forward(self, feats):
        c1, _, _, c4 = feats
        x = self.aspp(c4)
        x = F.interpolate(x, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, self.low_proj(c1)], dim=1))


# ─── 4. UNet++ ───────────────────────────────────────────────
class UNetPlusPlusDecoder(nn.Module):
    def __init__(self, in_channels_list, fpn_channels=256):
        super().__init__()
        C = fpn_channels
        self.reduce = nn.ModuleList([
            ConvBNReLU(c, C, k=1, s=1, p=0) for c in in_channels_list
        ])

        def _node(n_in):
            return nn.Sequential(ConvBNReLU(C * n_in, C), ConvBNReLU(C, C))

        self.x01 = _node(2); self.x11 = _node(2); self.x21 = _node(2)
        self.x02 = _node(3); self.x12 = _node(3)
        self.x03 = _node(4)
        self.final = nn.Sequential(ConvBNReLU(C, C), nn.Dropout2d(0.1))

    @staticmethod
    def _up(x, target):
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, feats):
        x00, x10, x20, x30 = [self.reduce[i](f) for i, f in enumerate(feats)]
        x21 = self.x21(torch.cat([x20, self._up(x30, x20)], dim=1))
        x11 = self.x11(torch.cat([x10, self._up(x20, x10)], dim=1))
        x01 = self.x01(torch.cat([x00, self._up(x10, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self._up(x21, x10)], dim=1))
        x02 = self.x02(torch.cat([x00, x01, self._up(x11, x00)], dim=1))
        x03 = self.x03(torch.cat([x00, x01, x02, self._up(x12, x00)], dim=1))
        return self.final(x03)


# ─── 5. Simple FPN ───────────────────────────────────────────
class SimpleFPNDecoder(nn.Module):
    def __init__(self, in_channels_list, fpn_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, 1) for c in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            ConvBNReLU(fpn_channels, fpn_channels) for _ in in_channels_list
        ])
        self.fuse = nn.Sequential(
            ConvBNReLU(fpn_channels * 4, fpn_channels),
            nn.Dropout2d(0.1),
        )

    def forward(self, feats):
        laterals = [self.lateral_convs[i](f) for i, f in enumerate(feats)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:],
                mode="bilinear", align_corners=False)
        outs = [self.output_convs[i](laterals[i]) for i in range(len(laterals))]
        target = outs[0].shape[-2:]
        aligned = [
            F.interpolate(o, size=target, mode="bilinear", align_corners=False)
            if o.shape[-2:] != target else o
            for o in outs
        ]
        return self.fuse(torch.cat(aligned, dim=1))


# ─── Decoder factory ─────────────────────────────────────────
DECODER_REGISTRY = {
    "upernet":       UPerNetDecoder,
    "segformer_mlp": SegFormerMLPDecoder,
    "deeplabv3plus": DeepLabV3PlusDecoder,
    "unetplusplus":  UNetPlusPlusDecoder,
    "fpn":           SimpleFPNDecoder,
}


def build_decoder(name, in_channels_list, fpn_channels=256):
    name = name.lower()
    if name not in DECODER_REGISTRY:
        raise ValueError(f"Unknown decoder '{name}'. Choose from {list(DECODER_REGISTRY)}")
    return DECODER_REGISTRY[name](in_channels_list, fpn_channels)


# ═══════════════════════════════════════════════════════════════
# Fusion strategies
# ═══════════════════════════════════════════════════════════════
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


# ─── Fusion factory ──────────────────────────────────────────
FUSION_REGISTRY = {
    "late_logits":  FusionLateLogits,
    "concat1x1":    FusionConcat1x1,
    "weighted_sum":  FusionWeightedSum,
    "gated":        FusionGated,
    "film":         FusionFiLM,
    "cross_attn":   FusionCrossAttention,
}


def build_fusion(name, chs):
    name = name.lower()
    if name not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion '{name}'. Choose from {list(FUSION_REGISTRY)}")
    cls = FUSION_REGISTRY[name]
    if name == "late_logits":
        return cls()
    return cls(chs)


# ═══════════════════════════════════════════════════════════════
# Dual-Swin V2 Segmentation Model
# ═══════════════════════════════════════════════════════════════
class DualSwinFusionSeg(nn.Module):
    """Two Swin V2 encoders (RGB + AUX) → fusion → decoder → binary mask."""

    def __init__(
        self,
        encoder_name="swinv2_small_window8_256",
        pretrained=True,
        img_size=128,
        fpn_channels=256,
        fusion_name="concat1x1",
        decoder_name="unetplusplus",
    ):
        super().__init__()
        self.enc_rgb = make_swin_features(encoder_name, pretrained=pretrained, img_size=img_size)
        self.enc_aux = make_swin_features(encoder_name, pretrained=pretrained, img_size=img_size)
        adapt_patch_embed_in_chans(self.enc_aux, 4)

        self.chs = self.enc_rgb.feature_info.channels()
        self.fusion  = build_fusion(fusion_name, self.chs)
        self.decoder = build_decoder(decoder_name, self.chs, fpn_channels=fpn_channels)
        self.head    = nn.Conv2d(fpn_channels, 1, kernel_size=1)

        self.img_size     = img_size
        self.fusion_name  = fusion_name
        self.decoder_name = decoder_name

    def _encode_rgb(self, rgb):
        return to_nchw(self.enc_rgb(rgb), self.chs)

    def _encode_aux(self, aux4):
        return to_nchw(self.enc_aux(aux4), self.chs)

    def _decode_to_logits(self, feats):
        x = self.decoder(feats)
        logits = self.head(x)
        return F.interpolate(logits, size=(self.img_size, self.img_size),
                             mode="bilinear", align_corners=False)

    def forward(self, rgb, aux4):
        feats_rgb = self._encode_rgb(rgb)
        feats_aux = self._encode_aux(aux4)
        if isinstance(self.fusion, FusionLateLogits):
            log_rgb = self._decode_to_logits(feats_rgb)
            log_aux = self._decode_to_logits(feats_aux)
            return 0.5 * (log_rgb + log_aux)
        feats = self.fusion(feats_rgb, feats_aux)
        return self._decode_to_logits(feats)
