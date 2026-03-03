"""
``src.model`` — Dual-Encoder Swin V2 segmentation model package.

Submodules
----------
- ``attention``  : SE / ECA / CBAM channel-attention modules & position wrappers
- ``decoders``   : 6 decoder architectures + registry
- ``fusions``    : 9 fusion strategies + registry
- ``core``       : Backbone helpers & end-to-end model classes

Public re-exports (so ``from src.model import DualSwinFusionSeg`` keeps working):
"""

# Attention primitives & wrappers
from .attention import (                        # noqa: F401
    SE, ECA, CBAM,
    _make_attn,
    InputChannelAttention,
    PostEncoderAttention,
    DecoderOutputAttention,
)

# Decoders
from .decoders import (                         # noqa: F401
    ConvBNReLU, SEBlock,
    UPerNetDecoder, SegFormerMLPDecoder, DeepLabV3PlusDecoder,
    UNetPlusPlusDecoder, SimpleFPNDecoder, HybridSegFormerUNetPPDecoder,
    DECODER_REGISTRY, build_decoder,
)

# Fusions
from .fusions import (                          # noqa: F401
    FusionLateLogits, FusionConcat1x1, FusionWeightedSum,
    FusionGated, FusionFiLM, FusionCrossAttention,
    FusionConcatSE, FusionConcatECA, FusionConcatCBAM,
    FUSION_REGISTRY, build_fusion,
)

# End-to-end models
from .core import (                             # noqa: F401
    DualSwinFusionSeg,
    DualSwinLateFusionSeg,
    adapt_patch_embed_in_chans,
    make_swin_features,
    make_swin_with_intra_attention,
    to_nchw,
    SwinWithIntraAttention,
)
