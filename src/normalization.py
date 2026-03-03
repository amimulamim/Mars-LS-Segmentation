"""
Per-image percentile normalization and channel statistics computation.

v4 key idea: each band is clipped to its OWN image's [P1, P99] percentiles
and rescaled to [0, 1].  This makes features domain-invariant (e.g. DEM
elevation shifts between train and test sets are eliminated).
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Core normalisation
# ──────────────────────────────────────────────────────────────
def normalize_bands_per_image(arr_chw, p_low=1.0, p_high=99.0):
    """Per-image, per-band percentile normalization → [0, 1].

    Parameters
    ----------
    arr_chw : ndarray  (C, H, W)
    p_low, p_high : float  percentile bounds

    Returns
    -------
    ndarray  (C, H, W) float32 in [0, 1]
    """
    C = arr_chw.shape[0]
    out = np.empty_like(arr_chw, dtype=np.float32)
    for c in range(C):
        flat = arr_chw[c].ravel()
        lo = np.percentile(flat, p_low)
        hi = np.percentile(flat, p_high)
        hi = max(hi, lo + 1e-6)
        out[c] = np.clip((arr_chw[c].astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return out


# ──────────────────────────────────────────────────────────────
# Compute global mean / std after per-image norm
# ──────────────────────────────────────────────────────────────
def compute_mean_std_per_image_norm(
    img_paths, band_indices,
    p_low=1.0, p_high=99.0,
    max_files=None, max_pixels=2_000_000,
):
    """Return (means, stds) arrays of shape (C,) computed *after*
    per-image percentile normalization.
    """
    paths = list(img_paths)
    if max_files is not None:
        paths = paths[:max_files]
    C = len(band_indices)
    sums = np.zeros(C, dtype=np.float64)
    sqs  = np.zeros(C, dtype=np.float64)
    n    = 0
    rng  = np.random.default_rng(123)
    for p in tqdm(paths, desc="Compute mean/std (per-image norm)"):
        with rasterio.open(str(p)) as src:
            arr = src.read(band_indices).astype(np.float32)
        arr = normalize_bands_per_image(arr, p_low, p_high)
        flat = arr.reshape(C, -1)
        if flat.shape[1] > 20_000:
            idx = rng.choice(flat.shape[1], size=20_000, replace=False)
            flat = flat[:, idx]
        sums += flat.sum(axis=1)
        sqs  += (flat * flat).sum(axis=1)
        n    += flat.shape[1]
        if n >= max_pixels:
            break
    means = (sums / max(n, 1)).astype(np.float32)
    vars_ = (sqs  / max(n, 1) - means.astype(np.float64) ** 2).clip(min=1e-8)
    stds  = np.sqrt(vars_).astype(np.float32)
    return means, stds


# ──────────────────────────────────────────────────────────────
# Save / load helpers
# ──────────────────────────────────────────────────────────────
def save_norm_stats(path, means, stds, band_indices, rgb_bands, aux_bands,
                    img_size, pos_weight=None, fg_frac=None):
    payload = {
        "normalization": "per_image_percentile",
        "p_low": 1.0,
        "p_high": 99.0,
        "band_indices": band_indices,
        "rgb_bands": rgb_bands,
        "aux_bands": aux_bands,
        "channel_means": means.tolist() if hasattr(means, "tolist") else list(means),
        "channel_stds":  stds.tolist()  if hasattr(stds, "tolist")  else list(stds),
        "img_size": img_size,
    }
    if pos_weight is not None:
        payload["pos_weight"] = pos_weight
    if fg_frac is not None:
        payload["fg_frac"] = fg_frac
    Path(path).write_text(json.dumps(payload, indent=2))
    return payload


def load_norm_stats(path):
    """Return (means, stds) as float32 arrays, plus the full dict."""
    with open(path) as f:
        data = json.load(f)
    means = np.array(data["channel_means"], dtype=np.float32)
    stds  = np.array(data["channel_stds"],  dtype=np.float32)
    return means, stds, data
