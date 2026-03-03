# Mars Landslide Segmentation — Dual Swin V2 with K-Fold Cross-Validation

## Table of Contents

- [Overview](#overview)
- [Architecture at a Glance](#architecture-at-a-glance)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Pre-trained Checkpoints](#pre-trained-checkpoints)
- [Environment Setup](#environment-setup)
- [Training](#training)
- [Inference](#inference)
- [Notebooks](#notebooks)
- [CLI Reference](#cli-reference)
- [Cross-Validation Results](#cross-validation-results)
- [Hardware Requirements](#hardware-requirements)
- [Key Files](#key-files)
- [License](#license)

---

## Overview

This repository implements a complete training and inference pipeline for the **Mars Landslide Segmentation (Mars LS)** challenge. The core method uses a **Dual-Encoder Swin Transformer V2** architecture — two separate Swin V2 Small backbones for RGB and auxiliary modalities (DEM, Slope, Thermal, Grayscale), fused at the feature level and decoded into binary landslide masks.

> For a detailed explanation of all architectural concepts, attention mechanisms, fusion strategies, and experiment progression, see [OVERVIEW.md](OVERVIEW.md).

---

## Architecture at a Glance

```
Input: 7-band GeoTIFF
  │
  ├── RGB (3ch)  ──► Swin V2 Small (ImageNet pretrained) ──► 4-level features
  │                                                              │
  └── AUX (4ch)  ──► Swin V2 Small (adapted patch embed) ──► 4-level features
                                                                 │
                                     Feature Fusion (per level) ◄┘
                                           │
                                        Decoder
                                           │
                                     1×1 Conv Head
                                           │
                                     Binary Mask (128×128)
```

| Component | Default (submitted) | Alternatives |
|-----------|-------------------|--------------|
| **Encoder** | `swinv2_small_window8_256` | Any timm-compatible model |
| **Decoder** | `unetplusplus` | `upernet`, `segformer_mlp`, `deeplabv3plus`, `fpn`, `hybrid_segformer_unetpp` |
| **Fusion** | `concat1x1` | `late_logits`, `weighted_sum`, `gated`, `film`, `cross_attn`, `concat_se`, `concat_eca`, `concat_cbam` |
| **Loss** | 0.5 × BCE + 0.5 × Dice | Hybrid BCE+Dice+Boundary+Lovász (experiments) |
| **Optimizer** | AdamW (lr=2e-4, wd=1e-4) | — |
| **Scheduler** | Linear warmup (3 ep) + cosine decay | — |
| **EMA** | Decay=0.995 with warmup | — |
| **Parameters** | ~100M | — |

The experiment notebooks in `experiments/` explore additional concepts not in the submitted model:

| Concept | Summary | Details |
|---------|---------|---------|
| Hybrid SegFormer×UNet++ decoder | Parallel SegFormer (global context) + UNet++ (local detail) paths, fused via SE gate | [OVERVIEW.md §2b](OVERVIEW.md#2b-hybrid-segformer--unet-decoder) |
| Channel attention (SE / ECA / CBAM) | Per-channel feature recalibration at various pipeline stages | [OVERVIEW.md §3](OVERVIEW.md#3-channel-attention-mechanisms) |
| Multi-location attention | Attention at input, intra-encoder, and decoder-output | [OVERVIEW.md §4](OVERVIEW.md#4-multi-location-attention) |
| Attention-enhanced fusion | SE / ECA / CBAM applied after feature concatenation | [OVERVIEW.md §6](OVERVIEW.md#6-fusion-strategies) |
| Late fusion | Separate encoder+decoder per modality with learnable logit blending | [OVERVIEW.md §7](OVERVIEW.md#7-late-fusion-architecture) |
| Boundary + Lovász loss | Multi-component loss with boundary upweighting and IoU surrogate | [OVERVIEW.md §8](OVERVIEW.md#8-loss-functions--training-objectives) |
| Deep supervision | Auxiliary heads on intermediate UNet++ nodes for gradient flow | [OVERVIEW.md §8](OVERVIEW.md#8-loss-functions--training-objectives) |

### Preprocessing

1. **Per-image percentile normalization**: Each band clipped to P1/P99 → [0, 1] (domain-invariant)
2. **Z-score standardization**: Global mean/std from training set (`norm_stats_v4.json`)
3. **Augmentations** (train): HorizontalFlip, VerticalFlip, RandomRotate90, Affine, GaussianBlur, RandomBrightnessContrast

> More details: [OVERVIEW.md §9](OVERVIEW.md#9-domain-invariant-normalization)

---

## Repository Structure

```
.
├── README.md                       # This file — setup & running guide
├── OVERVIEW.md                     # Architecture concepts deep-dive
├── requirements.txt                # Python dependencies (pip)
├── environment.yml                 # Conda environment specification
│
├── src/                            # All source code (Python package)
│   ├── __init__.py
│   ├── train.py                    # Training entry point (CLI)
│   ├── infer.py                    # Inference entry point (CLI)
│   ├── config.py                   # Constants, channel maps, default hyperparameters
│   ├── normalization.py            # Per-image percentile normalization & stats I/O
│   ├── augmentations.py            # Augmentation pipelines (standard + strong)
│   ├── dataset.py                  # MarsSegDataset (train/val) & InferenceDataset (test)
│   ├── losses.py                   # Loss functions, metrics, pos_weight
│   ├── utils.py                    # EMA, seed, TTA, model loading, submission I/O
│   └── model/                      # Model architecture package
│       ├── __init__.py             # Re-exports all public classes/functions
│       ├── attention.py            # SE, ECA, CBAM + position wrappers
│       ├── decoders.py             # 6 decoder architectures + registry
│       ├── fusions.py              # 9 fusion strategies + registry
│       └── core.py                 # Backbone helpers + DualSwinFusionSeg
│                                   #   + DualSwinLateFusionSeg
│
├── notebooks/                      # Main submission notebooks (Kaggle-ready)
│   ├── submitted_training.ipynb
│   ├── submitted_infer.ipynb
│   ├── dual_swin_unetpp_kfold_training.ipynb
│   └── dual_swin_unetpp_kfold_infer.ipynb
│
├── experiments/                    # Architecture experiments (not submitted)
│   ├── mid_fusion_concat_eca_hybrid_dec/
│   │   ├── training.ipynb          # Hybrid decoder + channel attention + ECA fusion
│   │   └── inference.ipynb
│   └── late_fusion_alpha_blend_hybrid_dec/
│       ├── training.ipynb          # Late fusion + learnable α-blend
│       └── inference.ipynb
│
├── data/                           # Datasets (NOT tracked — see Dataset section)
│   ├── phase1_dataset/
│   └── phase2_dataset/
│
└── trained_model_output/           # Pre-trained weights from HuggingFace
    └── kfold_results_v4/
        ├── norm_stats_v4.json
        ├── kfold_report_v4.json
        └── checkpoints/swinv2_unetplusplus_concat1x1/
```

---

## Dataset

**Mars Landslide Segmentation (Mars LS)** — 7-band GeoTIFF files (128×128 pixels).

| Band | Channel |
|------|---------|
| 1 | Thermal |
| 2 | Slope |
| 3 | DEM (Digital Elevation Model) |
| 4 | Grayscale |
| 5, 6, 7 | RGB |

**Masks**: Binary (0 = background, 1 = landslide).

### Data Splits

| Split | Phase | Images | Masks |
|-------|-------|--------|-------|
| `data/phase1_dataset/train/` | Phase 1 | 465 | 465 |
| `data/phase1_dataset/val/` | Phase 1 | 66 | 66 |
| `data/phase1_dataset/test/` | Phase 1 | 133 | — |
| `data/phase2_dataset/test/` | Phase 2 | 276 | — |

> The `data/` directory is **not tracked by git**. Place the datasets manually.

### Expected Directory Layout

```
data/
├── phase1_dataset/
│   ├── train/
│   │   ├── images/    # .tif files
│   │   └── masks/     # .tif files (same filenames)
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       └── images/
└── phase2_dataset/
    └── test/
        └── images/
```

---

## Pre-trained Checkpoints

Trained model weights (5 folds, ~433 MB each) are hosted on HuggingFace:

> **https://huggingface.co/amimulamim/Mars-LS-Seg_Checkpoints/tree/main/checkpoints/swinv2_unetplusplus_concat1x1**

Download all 5 checkpoint files (`fold1_best.pt` … `fold5_best.pt`) and place them in:
```
trained_model_output/kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1/
```

The pretrained Swin V2 Small backbone is automatically downloaded from HuggingFace/timm on first run.

---

## Environment Setup

**Option A: Conda (recommended)**
```bash
conda env create -f environment.yml
conda activate mars_ls
```

**Option B: pip**
```bash
python -m venv mars_ls_env
source mars_ls_env/bin/activate

# Install PyTorch with CUDA (adjust for your CUDA version)
# See https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

**Option C: Kaggle Notebook**
The notebooks in `notebooks/` can run directly on Kaggle with GPU. Dependencies are installed in-notebook via pip.

---

## Training

### Quick Start

```bash
python -m src.train --data_root data/phase1_dataset
```

This will:
1. Combine train + val images → 531 samples for 5-fold CV
2. Compute per-image normalization stats → `norm_stats_v4.json`
3. Train 5 folds × 50 epochs with EMA + cosine LR
4. Save best checkpoint per fold
5. Generate ensemble test predictions + submission zip
6. Write K-Fold report

### Different Architectures

```bash
# UPerNet decoder + cross-attention fusion
python -m src.train \
    --data_root data/phase1_dataset \
    --decoder_name upernet \
    --fusion_name cross_attn \
    --out_dir output/experiment_upernet_crossattn

# Hybrid decoder + ECA fusion
python -m src.train \
    --data_root data/phase1_dataset \
    --decoder_name hybrid_segformer_unetpp \
    --fusion_name concat_eca \
    --out_dir output/experiment_hybrid_eca

# SegFormer MLP decoder + gated fusion, smaller batch for 8GB GPU
python -m src.train \
    --data_root data/phase1_dataset \
    --decoder_name segformer_mlp \
    --fusion_name gated \
    --batch_size 8 \
    --out_dir output/experiment_segformer_gated
```

### Tuning Hyperparameters

```bash
python -m src.train \
    --data_root data/phase1_dataset \
    --lr 1e-4 --weight_decay 5e-4 \
    --ema_decay 0.999 --warmup_epochs 5 \
    --batch_size 8 --epochs 100 --no_amp
```

### Background Training (tmux)

```bash
tmux new-session -d -s train \
  "conda run --no-capture-output -n mars_ls \
   python -m src.train --data_root data/phase1_dataset \
   2>&1 | tee output/train.log"
tmux attach -t train    # Ctrl+B d to detach
```

**Estimated time**: ~3–5 hours on a single NVIDIA T4/P100 (50 epochs × 5 folds).

---

## Inference

### With Pre-trained Checkpoints

**Phase 1 test set:**
```bash
python -m src.infer \
    --test_dir data/phase1_dataset/test/images \
    --ckpt_dir trained_model_output/kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1 \
    --stats_json trained_model_output/kfold_results_v4/norm_stats_v4.json \
    --out_dir output/inference_output_phase1
```

**Phase 2 test set:**
```bash
python -m src.infer \
    --test_dir data/phase2_dataset/test/images \
    --ckpt_dir trained_model_output/kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1 \
    --stats_json trained_model_output/kfold_results_v4/norm_stats_v4.json \
    --out_dir output/inference_output_phase2
```

### With Freshly Trained Checkpoints

```bash
python -m src.infer \
    --test_dir data/phase1_dataset/test/images \
    --ckpt_dir output/kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1 \
    --stats_json output/kfold_results_v4/norm_stats_v4.json
```

### Non-Default Architecture (must match training)

```bash
python -m src.infer \
    --test_dir data/phase1_dataset/test/images \
    --ckpt_dir output/experiment_upernet_crossattn/checkpoints/swinv2_upernet_cross_attn \
    --stats_json output/experiment_upernet_crossattn/norm_stats_v4.json \
    --decoder_name upernet --fusion_name cross_attn
```

### Options

```bash
--no_tta          # Disable TTA (~2-4× faster)
--thresh 0.5      # Custom binarization threshold
--n_folds 3       # If trained with 3 folds
```

### Output

```
output/inference_output/
├── predictions/        # Individual mask .tif files
└── submission.zip      # Ready-to-submit zip archive
```

---

## Notebooks

**Main submission** (`notebooks/`):
- `submitted_training.ipynb` — submitted training pipeline
- `submitted_infer.ipynb` — submitted inference pipeline
- `dual_swin_unetpp_kfold_training.ipynb` — full training (Dual Swin + UNet++)
- `dual_swin_unetpp_kfold_infer.ipynb` — inference with TTA + ensemble

**Experiments** (`experiments/`):

| Folder | Description |
|--------|-------------|
| `mid_fusion_concat_eca_hybrid_dec/` | Mid-fusion with ECA-enhanced concat + Hybrid SegFormer×UNet++ decoder + multi-location attention |
| `late_fusion_alpha_blend_hybrid_dec/` | Late fusion with learnable α-blend + independent Hybrid decoders per modality |

Each experiment folder contains a `training.ipynb` and `inference.ipynb` pair. Configure paths in the notebook config cells and run all cells sequentially.

---

## CLI Reference

### Training Arguments (`python -m src.train`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_root` | str | **required** | Path to dataset root |
| `--out_dir` | str | `output/kfold_results_v4` | Output directory |
| `--encoder_name` | str | `swinv2_small_window8_256` | timm backbone |
| `--decoder_name` | str | `unetplusplus` | `unetplusplus`, `upernet`, `segformer_mlp`, `deeplabv3plus`, `fpn`, `hybrid_segformer_unetpp` |
| `--fusion_name` | str | `concat1x1` | `concat1x1`, `late_logits`, `weighted_sum`, `gated`, `film`, `cross_attn`, `concat_se`, `concat_eca`, `concat_cbam` |
| `--fpn_channels` | int | `256` | Decoder FPN channels |
| `--img_size` | int | `128` | Input resolution |
| `--epochs` | int | `50` | Epochs per fold |
| `--batch_size` | int | `16` | Batch size |
| `--num_workers` | int | `2` | DataLoader workers |
| `--lr` | float | `2e-4` | Learning rate |
| `--weight_decay` | float | `1e-4` | AdamW weight decay |
| `--seed` | int | `42` | Random seed |
| `--n_folds` | int | `5` | CV folds |
| `--ema_decay` | float | `0.995` | EMA decay rate |
| `--warmup_epochs` | int | `3` | LR warmup epochs |
| `--thresh` | float | `0.5` | Validation threshold |
| `--no_pretrained` | flag | pretrained=**True** | Pass flag to disable ImageNet pretrained weights |
| `--no_amp` | flag | amp=**True** | Pass flag to disable mixed precision |

### Inference Arguments (`python -m src.infer`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--test_dir` | str | **required** | Test image directory |
| `--ckpt_dir` | str | **required** | Checkpoint directory |
| `--stats_json` | str | `None` | Normalization stats JSON |
| `--recompute_from` | str | `None` | Recompute stats from this dataset root |
| `--out_dir` | str | `output/inference_output` | Output directory |
| `--encoder_name` | str | `swinv2_small_window8_256` | Must match training |
| `--decoder_name` | str | `unetplusplus` | Must match training |
| `--fusion_name` | str | `concat1x1` | Must match training |
| `--fpn_channels` | int | `256` | Must match training |
| `--img_size` | int | `128` | Must match training |
| `--orig_size` | int | `128` | Output mask resolution |
| `--batch_size` | int | `8` | Inference batch size |
| `--num_workers` | int | `2` | DataLoader workers |
| `--thresh` | float | `0.51` | Binarization threshold |
| `--n_folds` | int | `5` | Number of fold checkpoints |
| `--no_tta` | flag | tta=**True** | Pass flag to disable test-time augmentation |
| `--seed` | int | `42` | Random seed |

> **Important**: `--encoder_name`, `--decoder_name`, `--fusion_name`, `--fpn_channels`, and `--img_size` **must match** the values used during training.

---

## Cross-Validation Results

Results from 5-fold CV with the default configuration (from `kfold_report_v4.json`):

| Metric | Mean ± Std |
|--------|-----------|
| **mIoU** | 0.8393 ± 0.0098 |
| **IoU (foreground)** | 0.7974 ± 0.0154 |
| **IoU (background)** | 0.8813 ± 0.0054 |
| **F1 (foreground)** | 0.8872 ± 0.0096 |
| **Precision (fg)** | 0.8626 ± 0.0106 |
| **Recall (fg)** | 0.9133 ± 0.0111 |

> For experiment version progression and mIoU across architectures, see [OVERVIEW.md §11](OVERVIEW.md#11-experiment-version-progression).

---

## Hardware Requirements

| | Minimum | Recommended |
|---|---|---|
| **GPU** | NVIDIA GPU, 8 GB VRAM | NVIDIA T4 / P100 / V100 (16 GB) |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 5 GB (code + checkpoints) | 10 GB (with dataset) |
| **CUDA** | 11.8+ | 12.x |

Training with `batch_size=16` requires ~10 GB VRAM. Use `--batch_size 8` for 8 GB GPUs.

---

## Key Files

| File | Description |
|------|-------------|
| `src/train.py` | CLI training — K-Fold CV, checkpointing, ensemble predictions |
| `src/infer.py` | CLI inference — loads checkpoints, TTA, submission zip |
| `src/config.py` | Channel maps, band indices, `DEFAULT_CFG` dict |
| `src/normalization.py` | Per-image percentile normalization, stats save/load |
| `src/augmentations.py` | Augmentation pipelines (standard + strong variant) |
| `src/dataset.py` | `MarsSegDataset` (train/val) and `InferenceDataset` (test) |
| `src/losses.py` | Loss functions (BCE+Dice, Boundary, Lovász, deep supervision), metrics |
| `src/utils.py` | EMA, seed, TTA, model loading, submission writing |
| `src/model/` | **Model architecture package** (see below) |
| `src/model/attention.py` | SE, ECA, CBAM modules + position wrappers |
| `src/model/decoders.py` | 6 decoder architectures + `DECODER_REGISTRY` |
| `src/model/fusions.py` | 9 fusion strategies + `FUSION_REGISTRY` |
| `src/model/core.py` | Backbone helpers, `DualSwinFusionSeg`, `DualSwinLateFusionSeg` |

---

## License

This project was developed for the Mars Landslide Segmentation challenge.
