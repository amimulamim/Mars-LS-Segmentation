"""
Dual-Encoder Swin V2 segmentation models.

Architecture (default — mid-fusion)
-------------------------------------
  RGB (3ch) --> Swin V2 encoder --+
                                   +-- Fusion --> Decoder --> 1×1 head --> mask
  AUX (4ch) --> Swin V2 encoder --+

Architecture (late fusion)
---------------------------
  RGB (3ch) --> Swin V2 --> Decoder_rgb --> Head_rgb --> logits_rgb -+
                                                                     +-- α·rgb + (1-α)·aux
  AUX (4ch) --> Swin V2 --> Decoder_aux --> Head_aux --> logits_aux -+

Pluggable components live in sibling modules:
  - ``attention.py``  : SE / ECA / CBAM + position wrappers
  - ``decoders.py``   : 6 decoder architectures + registry
  - ``fusions.py``    : 9 fusion strategies + registry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .attention import (
    _make_attn,
    InputChannelAttention,
    PostEncoderAttention,
    DecoderOutputAttention,
)
from .decoders import build_decoder, HybridSegFormerUNetPPDecoder
from .fusions import build_fusion, FusionLateLogits


# ====================================================================
# Backbone helpers
# ====================================================================
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
    """Convert (B, H, W, C) -> (B, C, H, W) if needed."""
    out = []
    for f, c in zip(feats, in_chs):
        if f.ndim == 4 and f.shape[-1] == c and f.shape[1] != c:
            f = f.permute(0, 3, 1, 2).contiguous()
        out.append(f)
    return out


# ====================================================================
# Intra-encoder attention wrapper
# ====================================================================
class SwinWithIntraAttention(nn.Module):
    """Wraps a Swin encoder and injects channel attention after each stage
    via forward hooks."""

    def __init__(self, base_encoder, attn_type="none", reduction=16):
        super().__init__()
        self.base_encoder = base_encoder
        self.attn_type = attn_type.lower()
        self.feature_info = base_encoder.feature_info
        self.chs = base_encoder.feature_info.channels()
        if self.attn_type == "none":
            self.stage_attns = nn.ModuleList([nn.Identity() for _ in self.chs])
        else:
            self.stage_attns = nn.ModuleList([
                _make_attn(attn_type, c, reduction) for c in self.chs
            ])
        self._hook_handles = []
        self._register_hooks()

    def _get_stage_modules(self):
        if hasattr(self.base_encoder, "stages"):
            return list(self.base_encoder.stages)
        if hasattr(self.base_encoder, "layers"):
            return list(self.base_encoder.layers)
        return []

    def _register_hooks(self):
        if self.attn_type == "none":
            return
        for i, stage in enumerate(self._get_stage_modules()):
            handle = stage.register_forward_hook(self._make_hook(i))
            self._hook_handles.append(handle)

    def _make_hook(self, stage_idx):
        def hook(module, input, output):
            if output.ndim == 4:
                c = self.chs[stage_idx]
                if output.shape[-1] == c and output.shape[1] != c:
                    out_nchw = output.permute(0, 3, 1, 2).contiguous()
                    out_attn = self.stage_attns[stage_idx](out_nchw)
                    output = out_attn.permute(0, 2, 3, 1).contiguous()
                else:
                    output = self.stage_attns[stage_idx](output)
            return output
        return hook

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def forward(self, x):
        return self.base_encoder(x)

    def __getattr__(self, name):
        if name in (
            "base_encoder", "attn_type", "chs", "stage_attns",
            "_hook_handles", "feature_info",
        ):
            return super().__getattr__(name)
        return getattr(self.base_encoder, name)


def make_swin_with_intra_attention(
    encoder_name, pretrained=True, img_size=128,
    intra_attn="none", reduction=16,
):
    """Create a Swin encoder optionally wrapped with intra-stage attention."""
    base_enc = make_swin_features(
        encoder_name, pretrained=pretrained, img_size=img_size,
    )
    if intra_attn.lower() == "none":
        return base_enc
    return SwinWithIntraAttention(
        base_enc, attn_type=intra_attn, reduction=reduction,
    )


# ====================================================================
# Mid-Fusion Model  (default)
# ====================================================================
class DualSwinFusionSeg(nn.Module):
    """Two Swin V2 encoders (RGB + AUX) -> fusion -> decoder -> binary mask.

    By default no attention is applied (all set to ``"none"``).  For ablation
    studies set ``input_attn`` / ``post_encoder_attn`` / ``intra_encoder_attn``
    / ``decoder_output_attn`` to ``"se"`` / ``"eca"`` / ``"cbam"``.
    """

    def __init__(
        self,
        encoder_name="swinv2_small_window8_256",
        pretrained=True,
        img_size=128,
        fpn_channels=256,
        fusion_name="concat1x1",
        decoder_name="unetplusplus",
        # Optional attention (default: none — no overhead)
        input_attn="none",
        post_encoder_attn="none",
        intra_encoder_attn="none",
        decoder_output_attn="none",
    ):
        super().__init__()

        # Encoders (optionally with intra-stage attention)
        self.enc_rgb = make_swin_with_intra_attention(
            encoder_name, pretrained=pretrained, img_size=img_size,
            intra_attn=intra_encoder_attn,
        )
        self.enc_aux = make_swin_with_intra_attention(
            encoder_name, pretrained=pretrained, img_size=img_size,
            intra_attn=intra_encoder_attn,
        )
        if intra_encoder_attn.lower() == "none":
            adapt_patch_embed_in_chans(self.enc_aux, 4)
        else:
            adapt_patch_embed_in_chans(self.enc_aux.base_encoder, 4)

        self.chs = self.enc_rgb.feature_info.channels()

        # Attention wrappers
        self.rgb_input_attn = InputChannelAttention(3, attn_type=input_attn)
        self.aux_input_attn = InputChannelAttention(4, attn_type=input_attn)
        self.rgb_post_attn  = PostEncoderAttention(self.chs, attn_type=post_encoder_attn)
        self.aux_post_attn  = PostEncoderAttention(self.chs, attn_type=post_encoder_attn)
        self.dec_output_attn = DecoderOutputAttention(
            fpn_channels, attn_type=decoder_output_attn)

        # Fusion -> Decoder -> Head
        self.fusion  = build_fusion(fusion_name, self.chs)
        self.decoder = build_decoder(decoder_name, self.chs,
                                     fpn_channels=fpn_channels)
        self.head    = nn.Conv2d(fpn_channels, 1, kernel_size=1)

        self.img_size     = img_size
        self.fusion_name  = fusion_name
        self.decoder_name = decoder_name

    def _encode_rgb(self, rgb):
        feats = to_nchw(self.enc_rgb(self.rgb_input_attn(rgb)), self.chs)
        return self.rgb_post_attn(feats)

    def _encode_aux(self, aux4):
        feats = to_nchw(self.enc_aux(self.aux_input_attn(aux4)), self.chs)
        return self.aux_post_attn(feats)

    def _decode_to_logits(self, feats):
        out = self.decoder(feats)
        # HybridSegFormerUNetPPDecoder returns (fused, aux_list)
        if isinstance(out, tuple):
            x, aux_list = out
        else:
            x, aux_list = out, []
        x = self.dec_output_attn(x)
        logits = self.head(x)
        logits = F.interpolate(
            logits, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        aux_logits = [
            F.interpolate(a, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
            for a in aux_list
        ]
        return logits, aux_logits

    def forward(self, rgb, aux4):
        feats_rgb = self._encode_rgb(rgb)
        feats_aux = self._encode_aux(aux4)

        if isinstance(self.fusion, FusionLateLogits):
            log_rgb, _ = self._decode_to_logits(feats_rgb)
            log_aux, _ = self._decode_to_logits(feats_aux)
            return 0.5 * (log_rgb + log_aux)

        feats = self.fusion(feats_rgb, feats_aux)
        logits, aux_logits = self._decode_to_logits(feats)

        if self.training and aux_logits:
            return {"logits": logits, "aux_logits": aux_logits}
        return logits


# ====================================================================
# Late Fusion Model  (ablation variant)
# ====================================================================
class DualSwinLateFusionSeg(nn.Module):
    """Late fusion: each modality has its OWN encoder + decoder + head.
    Final prediction = learnable-weighted blend of both branches\' logits:

        alpha * logits_rgb  +  (1 - alpha) * logits_aux

    During training returns a dict with per-branch outputs for deep
    supervision; during eval returns the fused logits tensor directly.
    """

    def __init__(
        self,
        encoder_name="swinv2_small_window8_256",
        pretrained=True,
        img_size=128,
        fpn_channels=256,
        input_attn="eca",
        intra_encoder_attn="se",
        decoder_output_attn="cbam",
    ):
        super().__init__()

        # --- RGB branch ---
        self.rgb_input_attn = InputChannelAttention(3, attn_type=input_attn)
        self.enc_rgb = make_swin_with_intra_attention(
            encoder_name, pretrained=pretrained, img_size=img_size,
            intra_attn=intra_encoder_attn,
        )
        self.chs = self.enc_rgb.feature_info.channels()
        self.dec_rgb = HybridSegFormerUNetPPDecoder(self.chs, fpn_channels)
        self.dec_rgb_attn = DecoderOutputAttention(
            fpn_channels, attn_type=decoder_output_attn)
        self.head_rgb = nn.Conv2d(fpn_channels, 1, 1)

        # --- AUX branch ---
        self.aux_input_attn = InputChannelAttention(4, attn_type=input_attn)
        self.enc_aux = make_swin_with_intra_attention(
            encoder_name, pretrained=pretrained, img_size=img_size,
            intra_attn=intra_encoder_attn,
        )
        if intra_encoder_attn.lower() == "none":
            adapt_patch_embed_in_chans(self.enc_aux, 4)
        else:
            adapt_patch_embed_in_chans(self.enc_aux.base_encoder, 4)
        self.dec_aux = HybridSegFormerUNetPPDecoder(self.chs, fpn_channels)
        self.dec_aux_attn = DecoderOutputAttention(
            fpn_channels, attn_type=decoder_output_attn)
        self.head_aux = nn.Conv2d(fpn_channels, 1, 1)

        # --- Learnable fusion weight (initialised to 0.5) ---
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        self.img_size = img_size

    def _encode(self, encoder, x):
        return to_nchw(encoder(x), self.chs)

    def _decode_branch(self, decoder, dec_attn, head, feats):
        x, aux_list = decoder(feats)
        x = dec_attn(x)
        logits = head(x)
        logits = F.interpolate(
            logits, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        aux_logits = [
            F.interpolate(a, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
            for a in aux_list
        ]
        return logits, aux_logits

    def forward(self, rgb, aux4):
        rgb_feats = self._encode(self.enc_rgb, self.rgb_input_attn(rgb))
        rgb_logits, rgb_aux = self._decode_branch(
            self.dec_rgb, self.dec_rgb_attn, self.head_rgb, rgb_feats)

        aux_feats = self._encode(self.enc_aux, self.aux_input_attn(aux4))
        aux_logits, aux_aux = self._decode_branch(
            self.dec_aux, self.dec_aux_attn, self.head_aux, aux_feats)

        alpha = torch.sigmoid(self.alpha_logit)
        fused_logits = alpha * rgb_logits + (1 - alpha) * aux_logits

        if self.training:
            return {
                "logits": fused_logits,
                "rgb_logits": rgb_logits,
                "aux_logits_branch": aux_logits,
                "rgb_aux": rgb_aux,
                "aux_aux": aux_aux,
            }
        return fused_logits
