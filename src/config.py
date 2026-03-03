"""
Global configuration constants and default hyperparameters.

All channel mappings, band indices, and training defaults live here so that
train.py, infer.py, and the notebooks can import a single source of truth.
"""

import torch

# ──────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────
# Channel / band mapping  (1-indexed, rasterio convention)
# ──────────────────────────────────────────────────────────────
CHANNEL_MAP = {
    "THERMAL": [1],
    "SLOPE":   [2],
    "DEM":     [3],
    "GRAY":    [4],
    "RGB":     [5, 6, 7],
}

CHANNEL_ORDER = ["RGB", "DEM", "SLOPE", "THERMAL", "GRAY"]


def get_band_list(channels):
    """Return flat list of 1-indexed band numbers for the given channel names."""
    bands = []
    for ch in channels:
        bands += CHANNEL_MAP[ch]
    return bands


BAND_INDICES = get_band_list(CHANNEL_ORDER)   # [5, 6, 7, 3, 2, 1, 4]
RGB_BANDS    = CHANNEL_MAP["RGB"]             # [5, 6, 7]
AUX_BANDS    = [b for b in BAND_INDICES if b not in RGB_BANDS]  # [3, 2, 1, 4]

import numpy as np
RGB_STAT_IDX = np.array([BAND_INDICES.index(b) for b in RGB_BANDS])  # [0, 1, 2]
AUX_STAT_IDX = np.array([BAND_INDICES.index(b) for b in AUX_BANDS])  # [3, 4, 5, 6]

# ──────────────────────────────────────────────────────────────
# Default training hyper-parameters (override via CLI / cfg dict)
# ──────────────────────────────────────────────────────────────
DEFAULT_CFG = dict(
    # Architecture
    encoder_name  = "swinv2_small_window8_256",
    decoder_name  = "unetplusplus",
    fusion_name   = "concat1x1",
    fpn_channels  = 256,

    # Data
    img_size      = 128,
    batch_size    = 16,
    num_workers   = 2,

    # Training
    epochs        = 50,
    lr            = 2e-4,
    weight_decay  = 1e-4,
    seed          = 42,
    pretrained    = True,
    amp           = True,
    ema_decay     = 0.995,
    warmup_epochs = 3,
    thresh        = 0.5,

    # K-Fold
    n_folds       = 5,
)
