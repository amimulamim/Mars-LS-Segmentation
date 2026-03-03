# Mars Landslide Segmentation — Dual Swin V2 with K-Fold Cross-Validation

## Overview

This repository contains the complete training and inference pipeline for our Mars Landslide Segmentation challenge submission. The method uses a **Dual-Encoder Swin Transformer V2** architecture with **UNet++ decoder** and **concat1×1 fusion**, trained with **5-fold cross-validation** and **per-image percentile normalization** for domain-invariant feature extraction.

### Key Design Choices
- **Dual-encoder architecture**: Separate Swin V2 Small backbones for RGB (3-channel) and auxiliary (4-channel: DEM, Slope, Thermal, Gray) modalities
- **Per-image percentile normalization (v4)**: Each band is clipped to its own image's P1/P99 percentiles → [0, 1], then z-scored with global mean/std — eliminates domain shift (e.g., DEM elevation differences between train and test sets)
- **5-fold cross-validation** with ensemble inference across all folds
- **4-fold TTA** (test-time augmentation): original + horizontal flip + vertical flip + 90° rotation
- **EMA (Exponential Moving Average)** with warmup for stable training

---

## Repository Structure

```
.
├── README.md                       # This file
├── requirements.txt                # Python dependencies (pip)
├── environment.yml                 # Conda environment specification
├── train.py                        # Training entry point (CLI)
├── infer.py                        # Inference entry point (CLI)
├── src/                            # Modular source code
│   ├── __init__.py
│   ├── config.py                   # Constants, channel maps, default hyperparameters
│   ├── normalization.py            # Per-image percentile normalization & stats
│   ├── augmentations.py            # Albumentations transforms
│   ├── dataset.py                  # Training & inference Dataset classes
│   ├── model.py                    # Encoders, decoders, fusions, DualSwinFusionSeg
│   ├── losses.py                   # BCE+Dice loss, metrics, pos_weight
│   └── utils.py                    # EMA, seed, TTA, submission I/O, model loading
├── notebooks/                      # Original Kaggle notebooks (for reference)
│   ├── dual_swin_unetpp_kfold_training.ipynb  # Full training notebook
│   └── dual_swin_unetpp_kfold_infer.ipynb     # Inference-only notebook
├── data/                           # Datasets (not tracked by git — see below)
│   ├── phase1_dataset/             # Mars LS Phase 1 dataset
│   │   ├── train/
│   │   │   ├── images/             # 465 .tif files
│   │   │   └── masks/              # 465 .tif files
│   │   ├── val/
│   │   │   ├── images/             # 66 .tif files
│   │   │   └── masks/              # 66 .tif files
│   │   └── test/
│   │       └── images/             # 133 .tif files
│   └── phase2_dataset/             # Mars LS Phase 2 test set
│       └── test/
│           └── images/             # 276 .tif files
└── trained_model_output/           # Pre-trained weights & artifacts
    ├── __huggingface_repos__.json
    └── kfold_results_v4/
        ├── norm_stats_v4.json      # Per-image normalization statistics
        ├── kfold_report_v4.json    # Cross-validation metrics report
        ├── kfold_plot_unetplusplus_concat1x1.png
        ├── checkpoints/
        │   └── swinv2_unetplusplus_concat1x1/
        │       ├── fold1_best.pt   # ~433 MB each
        │       ├── fold2_best.pt
        │       ├── fold3_best.pt
        │       ├── fold4_best.pt
        │       └── fold5_best.pt
        └── submissions/
```

---

## Pre-trained Checkpoints

Trained model weights (5 folds, ~433 MB each) are hosted on HuggingFace:

> **https://huggingface.co/amimulamim/Mars-LS-Seg_Checkpoints/tree/main/checkpoints/swinv2_unetplusplus_concat1x1**

Download all 5 checkpoint files (`fold1_best.pt` … `fold5_best.pt`) and place them in:
```
trained_model_output/kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1/
```

The pretrained Swin V2 Small backbone is automatically downloaded from HuggingFace/timm:
- Model: `swinv2_small_window8_256.ms_in1k`
- Source: `timm` library (downloads automatically on first run)

---

## Dataset

The model is trained on the **Mars Landslide Segmentation (Mars LS)** dataset:
- **Format**: 7-band GeoTIFF files (128×128 pixels)
- **Bands** (1-indexed, rasterio order):
  - Band 1: Thermal
  - Band 2: Slope
  - Band 3: DEM (Digital Elevation Model)
  - Band 4: Grayscale
  - Bands 5, 6, 7: RGB
- **Masks**: Binary (0 = background, 1 = landslide)

### Data Splits

| Split | Phase | Images | Masks |
|-------|-------|--------|-------|
| `data/phase1_dataset/train/` | Phase 1 | 465 | 465 |
| `data/phase1_dataset/val/` | Phase 1 | 66 | 66 |
| `data/phase1_dataset/test/` | Phase 1 | 133 | — |
| `data/phase2_dataset/test/` | Phase 2 | 276 | — |

> **Note:** The `data/` directory is **not tracked by git** (too large). Place the datasets manually following the structure above.

### Expected Dataset Directory Structure

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

## Model Architecture

```
Input: 7-band GeoTIFF (RGB[5,6,7] + DEM[3] + Slope[2] + Thermal[1] + Gray[4])
  │
  ├── RGB (3ch) ──► Swin V2 Small (pretrained, ImageNet) ──► 4-level features
  │
  └── AUX (4ch) ──► Swin V2 Small (adapted patch embed)  ──► 4-level features
                                                                 │
                                     concat1x1 Fusion ◄─────────┘
                                           │
                                     UNet++ Decoder
                                           │
                                     1×1 Conv Head
                                           │
                                     Binary Mask (128×128)
```

| Component | Details |
|-----------|---------|
| **Encoder** | `swinv2_small_window8_256` (timm, pretrained on ImageNet-1K) |
| **Decoder** | UNet++ with FPN channels = 256 |
| **Fusion** | concat1×1 — concatenate dual-encoder features, project with 1×1 conv |
| **Loss** | 0.5 × BCE(pos_weight) + 0.5 × Dice |
| **Optimizer** | AdamW (lr=2e-4, weight_decay=1e-4) |
| **Scheduler** | Linear warmup (3 epochs) + cosine decay |
| **EMA** | Decay=0.995 with warmup |
| **Image Size** | 128×128 (native resolution) |
| **Parameters** | ~100M (dual encoder + decoder + fusion) |

---

## Preprocessing / Normalization (v4 — Domain-Invariant)

1. **Per-image percentile normalization**: For each image independently, each band is:
   - Clipped to its own P1 and P99 percentiles
   - Rescaled to [0, 1]

   This eliminates sensitivity to absolute value shifts between domains (e.g., DEM in training has elevations [-1345, 3110] while test has [4275, 7232]).

2. **Z-score standardization**: After per-image normalization, channels are standardized using global mean/std computed over the training set (saved in `norm_stats_v4.json`).

3. **Augmentations** (training only):
   - Geometric: HorizontalFlip, VerticalFlip, RandomRotate90, Affine (translate, scale, rotate)
   - Photometric (RGB only): GaussianBlur, RandomBrightnessContrast

---

## How to Reproduce Results

### 1. Environment Setup

**Option A: pip (recommended)**
```bash
# Create a virtual environment (Python 3.10+)
python -m venv mars_ls_env
source mars_ls_env/bin/activate

# Install PyTorch with CUDA (adjust for your CUDA version)
# See https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

**Option B: Conda**
```bash
conda env create -f environment.yml
conda activate mars_ls
```

**Option C: Kaggle Notebook**
The original notebooks in `notebooks/` can run directly on Kaggle with GPU. Dependencies are installed in-notebook via pip.

### 2. Training from Scratch (Python Scripts)

```bash
python train.py --data_root data/phase1_dataset
```

All hyperparameters can be overridden via CLI flags:
```bash
python train.py \
    --data_root data/phase1_dataset \
    --out_dir kfold_results_v4 \
    --encoder_name swinv2_small_window8_256 \
    --decoder_name unetplusplus \
    --fusion_name concat1x1 \
    --epochs 50 \
    --batch_size 16 \
    --lr 2e-4 \
    --n_folds 5 \
    --seed 42
```

The script will:
1. Load and combine train + val images for K-Fold splitting
2. Compute per-image normalization statistics → `norm_stats_v4.json`
3. Train 5 folds (50 epochs each) with EMA and cosine LR schedule
4. Save best checkpoint per fold → `checkpoints/swinv2_unetplusplus_concat1x1/`
5. Generate ensemble predictions on the test set → `submissions/`
6. Produce a K-Fold report → `kfold_report_v4.json`
7. Save training curves plot

**Estimated training time**: ~3–5 hours on a single NVIDIA T4/P100 GPU (50 epochs × 5 folds).

### 3. Inference with Pre-trained Weights (Python Scripts)

**Phase 1 test set:**
```bash
python infer.py \
    --test_dir data/phase1_dataset/test/images \
    --ckpt_dir trained_model_output/kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1 \
    --stats_json trained_model_output/kfold_results_v4/norm_stats_v4.json \
    --out_dir inference_output_phase1
```

**Phase 2 test set:**
```bash
python infer.py \
    --test_dir data/phase2_dataset/test/images \
    --ckpt_dir trained_model_output/kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1 \
    --stats_json trained_model_output/kfold_results_v4/norm_stats_v4.json \
    --out_dir inference_output_phase2
```

Options:
```bash
# Disable TTA for faster inference:
python infer.py --test_dir data/phase1_dataset/test/images --ckpt_dir ... --stats_json ... --no_tta

# Custom threshold:
python infer.py --test_dir data/phase1_dataset/test/images --ckpt_dir ... --stats_json ... --thresh 0.5
```

The script outputs:
- `inference_output/predictions/` — individual mask TIFs
- `inference_output/submission.zip` — zipped submission

### 4. Training/Inference via Notebooks

The original Kaggle notebooks are preserved in `notebooks/`:
- `notebooks/dual_swin_unetpp_kfold_training.ipynb` — full training pipeline
- `notebooks/dual_swin_unetpp_kfold_infer.ipynb` — inference with TTA + ensemble

Configure paths in the notebook cells and run all cells sequentially.

---

## Cross-Validation Results

Results from 5-fold cross-validation (from `kfold_report_v4.json`):

| Metric | Mean ± Std |
|--------|-----------|
| **mIoU** | 0.8393 ± 0.0098 |
| **IoU (foreground)** | 0.7974 ± 0.0154 |
| **IoU (background)** | 0.8813 ± 0.0054 |
| **F1 (foreground)** | 0.8872 ± 0.0096 |
| **Precision (fg)** | 0.8626 ± 0.0106 |
| **Recall (fg)** | 0.9133 ± 0.0111 |

---

## Hardware Requirements

| | Minimum | Recommended |
|---|---|---|
| **GPU** | NVIDIA GPU with 8 GB VRAM | NVIDIA T4 / P100 / V100 (16 GB) |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 5 GB (code + checkpoints) | 10 GB (with dataset) |
| **CUDA** | 11.8+ | 12.x |

Training with `batch_size=16` requires ~10 GB VRAM. Reduce to 8 if running on 8 GB GPUs.

---

## Key Files Description

| File | Description |
|------|-------------|
| `train.py` | CLI training entry point — K-Fold CV, checkpointing, ensemble |
| `infer.py` | CLI inference entry point — loads checkpoints, TTA, submission zip |
| `src/config.py` | Channel maps, band indices, default hyperparameters |
| `src/normalization.py` | Per-image percentile normalization, stats I/O |
| `src/augmentations.py` | Albumentations augmentation pipelines |
| `src/dataset.py` | `MarsSegDataset` (train/val) and `InferenceDataset` (test) |
| `src/model.py` | `DualSwinFusionSeg` + all decoders (UNet++, UPerNet, etc.) + all fusions |
| `src/losses.py` | `WeightedBCEDiceLoss`, evaluation metrics, pos_weight computation |
| `src/utils.py` | EMA, seed, TTA, model loading, submission writing |
| `notebooks/dual_swin_unetpp_kfold_training.ipynb` | Self-contained training notebook (Dual Swin + UNet++) |
| `notebooks/dual_swin_unetpp_kfold_infer.ipynb` | Self-contained inference notebook (ensemble + TTA) |

---

## License

This code is released for the Mars Landslide Segmentation Challenge. Please cite the challenge dataset and organizers if you use this work.
