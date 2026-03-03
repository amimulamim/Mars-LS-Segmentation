"""
Segmentation decoders and decoder registry.

Available decoders:
  - UPerNetDecoder               (PPM + FPN)
  - SegFormerMLPDecoder           (All-MLP)
  - DeepLabV3PlusDecoder          (ASPP + low-level fusion)
  - UNetPlusPlusDecoder           (dense skip connections)
  - SimpleFPNDecoder              (lateral + top-down FPN)
  - HybridSegFormerUNetPPDecoder  (MLP + UNet++ + gated SE fusion + aux heads)

Factory:
  build_decoder(name, in_channels_list, fpn_channels) → nn.Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# Shared building blocks
# ====================================================================
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


class SEBlock(nn.Module):
    """Lightweight SE used inside decoders."""

    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


# ====================================================================
# 1. UPerNet  (PPM + FPN)
# ====================================================================
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
        p3 = self.lateral_convs[2](c3) + F.interpolate(
            p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral_convs[1](c2) + F.interpolate(
            p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral_convs[0](c1) + F.interpolate(
            p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
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


# ====================================================================
# 2. SegFormer All-MLP
# ====================================================================
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
                x = F.interpolate(x, size=target, mode="bilinear",
                                  align_corners=False)
            outs.append(x)
        return self.fuse(torch.cat(outs, dim=1))


# ====================================================================
# 3. DeepLabV3+
# ====================================================================
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
                y = F.interpolate(y, size=(h, w), mode="bilinear",
                                  align_corners=False)
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
        x = F.interpolate(x, size=c1.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.fuse(torch.cat([x, self.low_proj(c1)], dim=1))


# ====================================================================
# 4. UNet++
# ====================================================================
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
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear",
                             align_corners=False)

    def forward(self, feats):
        x00, x10, x20, x30 = [self.reduce[i](f) for i, f in enumerate(feats)]
        x21 = self.x21(torch.cat([x20, self._up(x30, x20)], dim=1))
        x11 = self.x11(torch.cat([x10, self._up(x20, x10)], dim=1))
        x01 = self.x01(torch.cat([x00, self._up(x10, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self._up(x21, x10)], dim=1))
        x02 = self.x02(torch.cat([x00, x01, self._up(x11, x00)], dim=1))
        x03 = self.x03(torch.cat([x00, x01, x02, self._up(x12, x00)], dim=1))
        return self.final(x03)


# ====================================================================
# 5. Simple FPN
# ====================================================================
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


# ====================================================================
# 6. Hybrid SegFormer × UNet++
# ====================================================================
class HybridSegFormerUNetPPDecoder(nn.Module):
    """Combines SegFormer-style MLP projections with UNet++ dense skip
    connections and a gated SE fusion of the two branches.  Also provides
    auxiliary heads on intermediate UNet++ nodes for deep supervision."""

    def __init__(self, in_channels_list, fpn_channels=256):
        super().__init__()
        C = fpn_channels
        self.mlp_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, C, 1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
            ) for c in in_channels_list
        ])
        self.seg_fuse = nn.Sequential(
            nn.Conv2d(C * 4, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        def _node(n_cat):
            return nn.Sequential(ConvBNReLU(C * n_cat, C), ConvBNReLU(C, C))

        self.x01 = _node(2); self.x11 = _node(2); self.x21 = _node(2)
        self.x02 = _node(3); self.x12 = _node(3)
        self.x03 = _node(4)
        self.upp_final = nn.Sequential(ConvBNReLU(C, C), nn.Dropout2d(0.1))

        self.gate_fuse = nn.Sequential(
            nn.Conv2d(2 * C, C, 1, bias=False),
            nn.BatchNorm2d(C), nn.ReLU(inplace=True),
        )
        self.gate_se = SEBlock(C, r=16)

        self.aux_head_01 = nn.Conv2d(C, 1, 1)
        self.aux_head_02 = nn.Conv2d(C, 1, 1)

    @staticmethod
    def _up(x, target):
        return F.interpolate(x, size=target.shape[-2:],
                             mode="bilinear", align_corners=False)

    def forward(self, feats):
        x00 = self.mlp_projs[0](feats[0])
        x10 = self.mlp_projs[1](feats[1])
        x20 = self.mlp_projs[2](feats[2])
        x30 = self.mlp_projs[3](feats[3])

        target_size = x00.shape[-2:]
        seg_global = self.seg_fuse(torch.cat([
            x00,
            F.interpolate(x10, size=target_size, mode="bilinear",
                          align_corners=False),
            F.interpolate(x20, size=target_size, mode="bilinear",
                          align_corners=False),
            F.interpolate(x30, size=target_size, mode="bilinear",
                          align_corners=False),
        ], dim=1))

        x21 = self.x21(torch.cat([x20, self._up(x30, x20)], dim=1))
        x11 = self.x11(torch.cat([x10, self._up(x20, x10)], dim=1))
        x01 = self.x01(torch.cat([x00, self._up(x10, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self._up(x21, x10)], dim=1))
        x02 = self.x02(torch.cat([x00, x01, self._up(x11, x00)], dim=1))
        x03 = self.x03(torch.cat([x00, x01, x02, self._up(x12, x00)], dim=1))
        upp_out = self.upp_final(x03)

        fused = self.gate_se(self.gate_fuse(
            torch.cat([upp_out, seg_global], dim=1)))
        aux_list = [self.aux_head_01(x01), self.aux_head_02(x02)]
        return fused, aux_list


# ====================================================================
# Registry & factory
# ====================================================================
DECODER_REGISTRY = {
    "upernet":                  UPerNetDecoder,
    "segformer_mlp":            SegFormerMLPDecoder,
    "deeplabv3plus":            DeepLabV3PlusDecoder,
    "unetplusplus":             UNetPlusPlusDecoder,
    "fpn":                      SimpleFPNDecoder,
    "hybrid_segformer_unetpp":  HybridSegFormerUNetPPDecoder,
}


def build_decoder(name, in_channels_list, fpn_channels=256):
    name = name.lower()
    if name not in DECODER_REGISTRY:
        raise ValueError(
            f"Unknown decoder '{name}'. Choose from {list(DECODER_REGISTRY)}")
    return DECODER_REGISTRY[name](in_channels_list, fpn_channels)
