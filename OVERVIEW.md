# Concepts & Architecture Deep-Dive

> **This document explains the architectural concepts, attention mechanisms, fusion strategies, loss functions, and design decisions** explored in this project. For setup and running instructions, see [README.md](README.md).

## Table of Contents

1. [Dual-Encoder Swin Transformer V2](#1-dual-encoder-swin-transformer-v2)
2. [Decoders](#2-decoders)
   - [2a. UNet++ (Default)](#2a-unet-default)
   - [2b. Hybrid SegFormer × UNet++ Decoder](#2b-hybrid-segformer--unet-decoder)
   - [2c. Other Decoders](#2c-other-decoders)
3. [Channel Attention Mechanisms](#3-channel-attention-mechanisms)
   - [3a. Squeeze-and-Excitation (SE)](#3a-squeeze-and-excitation-se)
   - [3b. Efficient Channel Attention (ECA)](#3b-efficient-channel-attention-eca)
   - [3c. CBAM (Channel + Spatial)](#3c-cbam-channel--spatial)
4. [Multi-Location Attention](#4-multi-location-attention)
5. [Cross-Modal Encoder Attention](#5-cross-modal-encoder-attention)
6. [Fusion Strategies](#6-fusion-strategies)
7. [Late Fusion Architecture](#7-late-fusion-architecture)
8. [Loss Functions & Training Objectives](#8-loss-functions--training-objectives)
9. [Domain-Invariant Normalization](#9-domain-invariant-normalization)
10. [Training Techniques](#10-training-techniques)
11. [Experiment Version Progression](#11-experiment-version-progression)

---

## 1. Dual-Encoder Swin Transformer V2

**Implementation**: `src/model/core.py` — `DualSwinFusionSeg`

The input is a 7-band GeoTIFF. Rather than forcing all 7 bands through a single encoder, we split by modality:

- **RGB encoder** (3 channels): Standard Swin V2 Small with ImageNet-pretrained weights.
- **AUX encoder** (4 channels): Identical architecture, but the patch embedding `Conv2d` is adapted from 3→4 input channels. RGB-pretrained weights for the first 3 input channels are reused; the 4th is initialized by averaging.

Both encoders produce 4-level hierarchical features. These are fused per level by a configurable fusion module, then decoded to produce the segmentation mask.

**Why Swin V2?**
- Shifted-window attention gives O(n) complexity vs. O(n²) for global attention
- Hierarchical features (like an FPN) at 4 spatial resolutions
- Pretrained on ImageNet-1K — strong transfer to remote sensing

**Patch embedding adaptation** (`adapt_patch_embed_in_chans` in `src/model/core.py`):
```python
# For AUX (4ch): copy first 3ch weights, average them for the 4th
new_weight[:, :3] = old_weight[:, :3]
new_weight[:, 3]  = old_weight[:, :3].mean(dim=1)
```

---

## 2. Decoders

All decoders are registered in `DECODER_REGISTRY` (`src/model/decoders.py`) and selected via the `--decoder_name` CLI argument.

### 2a. UNet++ (Default)

**Implementation**: `src/model/decoders.py` — `UNetPlusPlusDecoder`

The default decoder. Takes fused 4-level features and performs nested dense skip connections.

```
Level 4 ──────────────────────────────────────► Final fuse (1/4 res)
Level 3 ────────────── Skip + Level 4 up ────► Fuse (1/8 res)
Level 2 ──── Skip + ... ────────────────────►
Level 1 ──►
```

Dense connections help propagate fine-grained features across semantic levels, improving boundary delineation for small landslides.

### 2b. Hybrid SegFormer × UNet++ Decoder

**Implementation**: `src/model/decoders.py` — `HybridSegFormerUNetPPDecoder`

A parallel dual-path decoder explored in the mid-fusion experiment:

```
Fused features ───┬──► SegFormer MLP path (global context)  ──►┐
                  │                                              ├ SE gate → merge
                  └──► UNet++ path (local detail)            ──►┘
                                                                  │
                                                               Output
```

- **SegFormer MLP path**: Multi-scale MLP mixing — each level is projected to a common channel dimension, upsampled to 1/4 resolution, and concatenated.
- **UNet++ path**: Standard nested skip connections for fine spatial detail.
- **SE gate**: A squeeze-and-excitation block learns to weight the two branches adaptively.

Selected via `--decoder_name hybrid_segformer_unetpp`.

### 2c. Other Decoders

All in `src/model/decoders.py`:

| Name | CLI key | Description |
|------|---------|-------------|
| UPerNet | `upernet` | Pyramid Pooling Module + FPN lateral connections |
| SegFormer MLP | `segformer_mlp` | Pure MLP mixing across scales |
| DeepLabV3+ | `deeplabv3plus` | ASPP (Atrous Spatial Pyramid Pooling) + skip connection |
| SimpleFPN | `fpn` | Lightweight Feature Pyramid Network |
| Hybrid SegFormer×UNet++ | `hybrid_segformer_unetpp` | Parallel global + local decoder (see §2b) |

---

## 3. Channel Attention Mechanisms

**Implementation**: `src/model/attention.py` — `SE`, `ECA`, `CBAM`

Channel attention recalibrates feature maps by learning per-channel importance weights. Three variants are implemented:

### 3a. Squeeze-and-Excitation (SE)

```
Input ──► GlobalAvgPool ──► FC (C→C/r) ──► ReLU ──► FC (C/r→C) ──► Sigmoid ──► Scale input
```

Standard SE with reduction ratio r=16. Two FC layers create a bottleneck that learns channel interdependencies.

### 3b. Efficient Channel Attention (ECA)

```
Input ──► GlobalAvgPool ──► 1D Conv (kernel=k) ──► Sigmoid ──► Scale input
```

Replaces the FC bottleneck with a single 1D convolution. The kernel size k is adaptively computed from the channel count:
```
k = |log2(C)/γ + b/γ|_odd    (γ=2, b=1)
```

This avoids dimensionality reduction while maintaining local cross-channel interaction.

### 3c. CBAM (Channel + Spatial)

**Channel attention**: Uses both AvgPool and MaxPool → shared FC → sum → sigmoid.

**Spatial attention**: Concatenate channel-wise AvgPool and MaxPool → 7×7 Conv → sigmoid.

```
Input ──► Channel Attention ──► Spatial Attention ──► Output
```

CBAM captures both "what" (channel) and "where" (spatial) information.

**Factory function**: `_make_attn(name, channels)` returns the appropriate module for `"se"`, `"eca"`, or `"cbam"`.

---

## 4. Multi-Location Attention

**Implementation**: `src/model/attention.py` — `InputChannelAttention`, `PostEncoderAttention`, `DecoderOutputAttention`; `src/model/core.py` — `SwinWithIntraAttention`

Attention can be applied at multiple stages of the pipeline. The experiment notebooks explore:

| Location | Module | Purpose |
|----------|--------|---------|
| **Input** | `InputChannelAttention` | Recalibrate raw 7-band channels before encoder input |
| **Intra-encoder** | `SwinWithIntraAttention` | Apply attention after each Swin stage (4 stages) |
| **Post-encoder** | `PostEncoderAttention` | Refine each encoder feature level after extraction |
| **Decoder output** | `DecoderOutputAttention` | Final feature recalibration before the classification head |

Each wrapper accepts an `attn_type` parameter (`"se"`, `"eca"`, `"cbam"`) and constructs the attention module accordingly.

---

## 5. Cross-Modal Encoder Attention

**Implementation**: `src/model/core.py` — inside `SwinWithIntraAttention`

Within the intra-encoder attention variant, cross-modal interaction is injected between the RGB and AUX encoders at each stage:

```
RGB features ──┬──► Self-Attention ──► Updated RGB
               ×
AUX features ──┴──► Self-Attention ──► Updated AUX
```

This cross-modal exchange is implemented via MultiheadAttention where queries come from one modality and keys/values from the other. It allows each encoder to "see" what the other encoder has learned at each hierarchical level, facilitating information sharing before fusion.

---

## 6. Fusion Strategies

**Implementation**: `src/model/fusions.py` — `FUSION_REGISTRY`

After both encoders produce 4-level features, each level is fused. The fusion strategy is selected via `--fusion_name`.

| Strategy | CLI key | How it works |
|----------|---------|-------------|
| **Concat + 1×1 Conv** (default) | `concat1x1` | `[F_rgb; F_aux]` → 1×1 Conv halves channels back |
| **Late Logits** | `late_logits` | No feature fusion; independent decoders → sum logits |
| **Weighted Sum** | `weighted_sum` | Learnable per-level scalar weight (sigmoid-gated) |
| **Gated** | `gated` | Learned gate `σ(W·[F_rgb; F_aux])` blends via element-wise: `g·F_rgb + (1-g)·F_aux` |
| **FiLM** | `film` | Feature-wise Linear Modulation: AUX generates scale γ and shift β for RGB |
| **Cross-Attention** | `cross_attn` | Multi-head attention — RGB as queries, AUX as keys/values (+ residual) |
| **Concat + SE** | `concat_se` | Concat → 1×1 Conv → SE attention |
| **Concat + ECA** | `concat_eca` | Concat → 1×1 Conv → ECA attention |
| **Concat + CBAM** | `concat_cbam` | Concat → 1×1 Conv → CBAM attention |

**Attention-enhanced fusions** (`concat_se`, `concat_eca`, `concat_cbam`) add channel attention after the standard concat+reduce, allowing the network to learn which fused features are most informative at each level.

---

## 7. Late Fusion Architecture

**Implementation**: `src/model/core.py` — `DualSwinLateFusionSeg`

An alternative to mid-fusion where each modality has its **own complete encoder + decoder pipeline**:

```
RGB (3ch) ──► Swin V2 Encoder ──► Decoder ──► Logits_rgb ──┐
                                                             ├──► α·L_rgb + (1-α)·L_aux
AUX (4ch) ──► Swin V2 Encoder ──► Decoder ──► Logits_aux ──┘
                                                             α = sigmoid(learnable)
```

- Each branch produces independent logits at full resolution.
- A learnable scalar α (initialized to 0 → sigmoid(0) = 0.5) blends the two logit maps.
- During training, deep supervision can be applied to intermediate nodes of each branch.

**When to use**: Late fusion is useful when the two modalities have very different statistical properties and forcing early feature sharing may be harmful.

Explored in `experiments/late_fusion_alpha_blend_hybrid_dec/`.

---

## 8. Loss Functions & Training Objectives

**Implementation**: `src/losses.py`

### Default Loss

**Weighted BCE + Dice** (`WeightedBCEDiceLoss`):
```
L = 0.5 × BCE(ŷ, y) + 0.5 × (1 - Dice(ŷ, y))
```

BCE is computed with positive class weight (ratio of background to foreground pixels in training set) to handle class imbalance. The Dice component directly optimizes the overlap metric.

### Experiment Losses

| Loss | Class | Description |
|------|-------|-------------|
| **Boundary Loss** | `boundary_loss()` | Computes distance-weighted loss focusing on pixels near mask boundaries |
| **Lovász Hinge** | `lovasz_hinge()` | Surrogate for IoU optimization — makes IoU differentiable |
| **Hybrid Loss** | `HybridBCEDiceBoundaryLoss` | α·BCE + β·Dice + γ·Boundary + δ·Lovász (all configurable) |
| **Deep Supervision** | `DeepSupervisionLoss` | Weighted sum of main + auxiliary heads from UNet++ intermediate nodes |
| **Late Fusion Deep Supervision** | `LateFusionDeepSupervisionLoss` | Deep supervision applied independently to each branch |

### Metrics

- **IoU** (Intersection over Union) — per-class and mean
- **F1 / Dice** — foreground
- **Precision / Recall** — foreground

---

## 9. Domain-Invariant Normalization

**Implementation**: `src/normalization.py`

Mars orbital imagery has no "natural" data range (unlike 0–255 for photos). Each instrument and band has different value ranges and distributions.

### Two-Stage Normalization

**Stage 1 — Per-image percentile normalization** (at dataset load time):
```python
low, high = np.percentile(band, [1, 99])
band_normalized = (band - low) / (high - low + ε)    # → [0, 1]
```

Clips extreme outliers while preserving relative contrast. Applied per-band, per-image.

**Stage 2 — Global z-score standardization** (after Stage 1):
```python
band_standardized = (band_normalized - global_mean) / global_std
```

Global statistics (`norm_stats_v4.json`) are computed across all training images and reused at inference. This two-stage approach makes the pipeline robust to varying instrument calibrations.

---

## 10. Training Techniques

### Exponential Moving Average (EMA)

**Implementation**: `src/utils.py` — `EMA`

Maintains a shadow copy of model weights updated each step:
```
θ_ema = decay · θ_ema + (1 - decay) · θ_model
```

With warmup: effective decay ramps from 0 to `ema_decay` (0.995) over training. The EMA model is used for validation and inference, producing smoother, more generalizable predictions.

### K-Fold Cross-Validation

**Implementation**: `src/train.py`

Stratified 5-fold CV over the combined train+val set (531 images):
1. Each fold trains for 50 epochs, saving the best checkpoint by validation mIoU
2. Fold predictions on the test set are averaged (ensemble)
3. The ensemble is binarized at threshold (default 0.5) to produce final masks

### Test-Time Augmentation (TTA)

**Implementation**: `src/utils.py` — `predict_with_tta()`

At inference, each image is predicted 4 times:
- Original
- Horizontal flip
- Vertical flip
- Both flips

Predictions are averaged before thresholding, improving robustness to orientation.

### Learning Rate Schedule

Linear warmup (3 epochs) → cosine decay to 0. Implemented via PyTorch's `SequentialLR` combining `LinearLR` and `CosineAnnealingLR`.

### Mixed Precision (AMP)

Enabled by default (`--no_amp` to disable). Uses `torch.cuda.amp` for ~30% memory reduction and ~20% speed improvement.

---

## 11. Experiment Version Progression

The project evolved through multiple architecture experiments. The table below summarizes the progression and approximate validation mIoU improvements:

| Version | Architecture | Key Changes | Approx. mIoU |
|---------|-------------|-------------|---------------|
| v3 | Dual Swin V2 + UNet++ | Baseline dual-encoder, concat1x1 fusion | ~0.81 |
| v4 | v3 + refined training | EMA, longer training, pos_weight tuning | ~0.84 |
| v5 | v4 + merged data | Extended dataset with additional samples | ~0.85 |
| v6 | v5 + hybrid decoder | SegFormer×UNet++ decoder, attention fusion | ~0.86 |
| v7 | v6 + encoder attention | Intra-encoder + cross-modal attention | experimental |
| v8late | v6 + late fusion | Separate encoder-decoder per modality | ~0.85 |

**Submitted model**: v4 — Dual Swin V2 Small + UNet++ + Concat1×1. mIoU = 0.8393 ± 0.0098.

### Notebooks and Experiments

| Notebook / Experiment | Description |
|----------------------|-------------|
| `notebooks/submitted_training.ipynb` | Submitted training pipeline |
| `notebooks/submitted_infer.ipynb` | Submitted inference pipeline |
| `notebooks/dual_swin_unetpp_kfold_training.ipynb` | v4 training — Dual Swin + UNet++ + concat |
| `notebooks/dual_swin_unetpp_kfold_infer.ipynb` | v4 inference |
| `experiments/mid_fusion_concat_eca_hybrid_dec/` | v6–v7 — Hybrid decoder + channel attention |
| `experiments/late_fusion_alpha_blend_hybrid_dec/` | v8late — Late fusion + α-blend |

---

## Source Code Reference

| File | Key Contents |
|------|-------------|
| `src/model/core.py` | `DualSwinFusionSeg` (mid-fusion model), `DualSwinLateFusionSeg` (late fusion model), `SwinWithIntraAttention`, backbone helpers |
| `src/model/attention.py` | `SE`, `ECA`, `CBAM`, `InputChannelAttention`, `PostEncoderAttention`, `DecoderOutputAttention`, `_make_attn()` factory |
| `src/model/decoders.py` | `UNetPlusPlusDecoder`, `UPerNetDecoder`, `SegFormerMLPDecoder`, `DeepLabV3PlusDecoder`, `SimpleFPNDecoder`, `HybridSegFormerUNetPPDecoder`, `DECODER_REGISTRY` |
| `src/model/fusions.py` | `FusionConcat1x1`, `FusionLateLogits`, `FusionWeightedSum`, `FusionGated`, `FusionFiLM`, `FusionCrossAttention`, `FusionConcatSE`, `FusionConcatECA`, `FusionConcatCBAM`, `FUSION_REGISTRY` |
| `src/losses.py` | `WeightedBCEDiceLoss`, `HybridBCEDiceBoundaryLoss`, `DeepSupervisionLoss`, `LateFusionDeepSupervisionLoss`, `boundary_loss()`, `lovasz_hinge()` |
| `src/augmentations.py` | `build_train_transforms()`, `build_train_transforms_strong()`, `build_val_transforms()` |
| `src/normalization.py` | `percentile_normalize()`, `compute_norm_stats()`, `NormStats` |
| `src/train.py` | CLI training: K-Fold CV, EMA, checkpointing, ensemble predictions |
| `src/infer.py` | CLI inference: load checkpoints, TTA, submission zip |
| `src/utils.py` | `EMA`, `seed_everything()`, `predict_with_tta()`, `load_model()` |
| `src/config.py` | `DEFAULT_CFG`, channel maps, band indices |
| `src/dataset.py` | `MarsSegDataset`, `InferenceDataset` |
