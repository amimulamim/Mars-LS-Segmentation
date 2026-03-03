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
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment specification
├── submitted_training.ipynb           # Full training notebook (end-to-end)
├── submitted_infer.ipynb              # Inference-only notebook
└── trained_model_output/              # Pre-trained weights & artifacts
    ├── __huggingface_repos__.json     # Pretrained backbone source
    └── kfold_results_v4/
        ├── norm_stats_v4.json         # Per-image normalization statistics
        ├── kfold_report_v4.json       # Cross-validation metrics report
        ├── kfold_plot_unetplusplus_concat1x1.png
        ├── checkpoints/
        │   └── swinv2_unetplusplus_concat1x1/
        │       ├── fold1_best.pt      # ~433 MB each
        │       ├── fold2_best.pt
        │       ├── fold3_best.pt
        │       ├── fold4_best.pt
        │       └── fold5_best.pt
        └── submissions/
            ├── swinv2_unetplusplus_concat1x1/
            └── swinv2_unetplusplus_concat1x1_kfold_ensemble/
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
- **Split**: Train + Val combined for K-Fold CV; separate test set

### Expected Dataset Directory Structure

```
<data_root>/
├── train/
│   ├── images/    # .tif files
│   └── masks/     # .tif files (same filenames)
├── val/
│   ├── images/
│   └── masks/
└── test/
    └── images/
```

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
Both notebooks are designed to run directly on Kaggle with GPU. Dependencies are installed in-notebook via pip.

### 2. Training from Scratch

Open `submitted_training.ipynb` and configure the following in **Cell 4** (Configuration):

```python
cfg = dict(
    data_root    = "/path/to/mars-landslide",   # ★ CHANGE: path to dataset root
    out_dir      = "kfold_results_v4",           # output directory for checkpoints & results
    encoders     = ["swinv2_small_window8_256"],
    decoders     = ["unetplusplus"],
    fusions      = ["concat1x1"],
    n_folds      = 5,
    img_size     = 128,
    epochs       = 50,
    batch_size   = 16,        # reduce to 8 if OOM
    num_workers  = 2,
    lr           = 2e-4,
    weight_decay = 1e-4,
    seed         = 42,
    pretrained   = True,
    fpn_channels = 256,
    amp          = True,      # mixed precision (requires CUDA)
    ema_decay    = 0.995,
    warmup_epochs= 3,
    thresh       = 0.5,
)
```

Then **run all cells sequentially**. The notebook will:
1. Load and combine train + val images for K-Fold splitting
2. Compute per-image normalization statistics and save to `norm_stats_v4.json`
3. Train 5 folds (each fold trains for 50 epochs)
4. Save best checkpoint per fold to `kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1/`
5. Generate ensemble predictions on the test set
6. Produce a K-Fold cross-validation report (`kfold_report_v4.json`)
7. Visualize per-fold metrics

**Estimated training time**: ~3–5 hours on a single NVIDIA T4/P100 GPU (50 epochs × 5 folds).

### 3. Inference with Pre-trained Weights

Open `submitted_infer.ipynb` and configure the following in **Cell 2** (Configurable Paths):

```python
TEST_DATA_DIR   = "/path/to/test/images"             # ★ CHANGE: folder with test .tif files
CHECKPOINT_DIR  = "trained_model_output/kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1"
STATS_JSON_PATH = "trained_model_output/kfold_results_v4/norm_stats_v4.json"
OUTPUT_DIR      = "v4_inference_output"
```

Then **run all cells sequentially**. The notebook will:
1. Load normalization statistics from `norm_stats_v4.json`
2. Load all 5 fold models
3. Run ensemble inference with 4-fold TTA
4. Output binary mask TIFs and a `submission.zip`

### 4. Pre-trained Weights

Pre-trained model checkpoints (5 folds, ~433 MB each, ~2.1 GB total) are provided in:
```
trained_model_output/kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1/
```

The pretrained Swin V2 Small backbone is automatically downloaded from HuggingFace/timm:
- Model: `swinv2_small_window8_256.ms_in1k`
- Source: `timm` library (downloads automatically on first run)

---

## Cross-Validation Results

Results from 5-fold cross-validation (reported in `kfold_report_v4.json`):

| Metric | Mean ± Std |
|--------|-----------|
| **mIoU** | See `kfold_report_v4.json` |
| **IoU (foreground)** | See `kfold_report_v4.json` |
| **IoU (background)** | See `kfold_report_v4.json` |
| **F1 (foreground)** | See `kfold_report_v4.json` |
| **Precision (fg)** | See `kfold_report_v4.json` |
| **Recall (fg)** | See `kfold_report_v4.json` |

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
| `submitted_training.ipynb` | Complete training pipeline: data loading, preprocessing, model definition, K-Fold training loop, evaluation, and ensemble test prediction |
| `submitted_infer.ipynb` | Inference-only notebook: loads pre-trained checkpoints, applies per-image normalization, runs TTA + ensemble, outputs submission masks |
| `requirements.txt` | Python package dependencies with minimum versions |
| `environment.yml` | Conda environment specification |
| `trained_model_output/kfold_results_v4/norm_stats_v4.json` | Channel-wise mean/std after per-image percentile normalization (needed for inference) |
| `trained_model_output/kfold_results_v4/kfold_report_v4.json` | Detailed per-fold and aggregate cross-validation metrics |
| `trained_model_output/kfold_results_v4/checkpoints/` | Trained model weights (5 folds) |

---

## License

This code is released for the Mars Landslide Segmentation Challenge. Please cite the challenge dataset and organizers if you use this work.
