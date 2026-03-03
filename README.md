# Mars Landslide Segmentation — Dual Swin V2 with K-Fold Cross-Validation

## Overview

This repository implements a complete training and inference pipeline for the **Mars Landslide Segmentation (Mars LS)** challenge. The core method uses a **Dual-Encoder Swin Transformer V2** architecture — two separate Swin V2 Small backbones for RGB and auxiliary modalities (DEM, Slope, Thermal, Grayscale), fused at the feature level and decoded into binary landslide masks.

### Core Concepts & Techniques

| Concept | What it does | Why it helps |
|---------|-------------|--------------|
| **Dual-encoder architecture** | Separate Swin V2 backbones for RGB (3ch) and AUX (4ch: DEM, Slope, Thermal, Gray) | Lets each modality learn specialized features; avoids forcing heterogeneous bands through one encoder |
| **Swin Transformer V2** | Hierarchical vision transformer with shifted windows (timm: `swinv2_small_window8_256`) | Strong ImageNet-pretrained backbone with multi-scale features, better than CNN for small medical/geo datasets |
| **Feature-level fusion** | Merge dual-encoder features before the decoder (default: concat + 1×1 conv) | Richer representation than late (logit-level) fusion; lets decoder reason about both modalities jointly |
| **UNet++ decoder** | Nested skip-connection decoder | Dense skip paths recover fine-grained spatial detail lost in the encoder |
| **Per-image percentile normalization (v4)** | Each band clipped to its own image's P1/P99 → [0,1], then globally z-scored | Eliminates domain shift (e.g., DEM train range [-1345, 3110] vs test [4275, 7232]) |
| **K-Fold cross-validation** | 5-fold CV over combined train+val split | Maximizes training data usage; provides robust metric estimation |
| **Ensemble inference** | Average predictions from all K fold models | Reduces variance; smoother probability maps before thresholding |
| **Test-Time Augmentation (TTA)** | 4-fold: original + H-flip + V-flip + 90° rotation | Exploits symmetry of satellite imagery for ~1-2% IoU gain |
| **EMA (Exponential Moving Average)** | Maintain running average of model weights with warmup | Stabilizes training; often yields better generalization than raw weights |
| **Weighted BCE + Dice loss** | 0.5 × BCE(pos_weight) + 0.5 × Dice | BCE handles pixel-level accuracy; Dice handles class imbalance directly |

### Architectures Explored

The codebase supports **5 decoders** and **6 fusion strategies** that can be mixed and matched via CLI flags:

**Decoders** (plug into `--decoder_name`):

| Decoder | Description |
|---------|-------------|
| `unetplusplus` | UNet++ — nested dense skip connections (default, **submitted**) |
| `upernet` | UPerNet — PPM + FPN-style multi-scale aggregation |
| `segformer_mlp` | SegFormer MLP decoder — lightweight all-MLP design |
| `deeplabv3plus` | DeepLab V3+ — ASPP multi-scale atrous convolutions + low-level skip |
| `fpn` | Simple FPN — feature pyramid with lateral connections |

**Fusion strategies** (plug into `--fusion_name`):

| Fusion | Description |
|--------|-------------|
| `concat1x1` | Concatenate dual features → 1×1 conv projection (default, **submitted**) |
| `late_logits` | Each encoder gets its own decoder; logits averaged at the end |
| `weighted_sum` | Learnable per-level scalar weights to sum dual features |
| `gated` | Sigmoid gating network selects how much of each encoder to use |
| `film` | Feature-wise Linear Modulation — AUX encoder modulates RGB features via scale+shift |
| `cross_attn` | Cross-attention between RGB and AUX feature maps at each level |

**Experiments** (in `experiments/`):
- **Hybrid SegFormer×UNet++ decoder** with **channel attention** (ECA / SE / CBAM) — fuses SegFormer's lightweight MLP aggregation with UNet++'s dense skip connections, plus per-channel recalibration

---

## Repository Structure

```
.
├── README.md                       # This file
├── requirements.txt                # Python dependencies (pip)
├── environment.yml                 # Conda environment specification
├── src/                            # All source code (Python package)
│   ├── __init__.py
│   ├── train.py                    # Training entry point (CLI)
│   ├── infer.py                    # Inference entry point (CLI)
│   ├── config.py                   # Constants, channel maps, default hyperparameters
│   ├── normalization.py            # Per-image percentile normalization & stats I/O
│   ├── augmentations.py            # Albumentations augmentation pipelines
│   ├── dataset.py                  # MarsSegDataset (train/val) & InferenceDataset (test)
│   ├── model.py                    # DualSwinFusionSeg + all decoders & fusions
│   ├── losses.py                   # WeightedBCEDiceLoss, metrics, pos_weight
│   └── utils.py                    # EMA, seed, TTA, model loading, submission I/O
├── notebooks/                      # Main submission notebooks (Kaggle-ready)
│   ├── dual_swin_unetpp_kfold_training.ipynb
│   └── dual_swin_unetpp_kfold_infer.ipynb
├── experiments/                    # Architecture experiments (not submitted)
│   ├── dual_swin_hybrid_sfxunetpp_attn_kfold_training.ipynb
│   └── dual_swin_hybrid_sfxunetpp_attn_kfold_infer.ipynb
├── data/                           # Datasets (NOT tracked — see Dataset section)
│   ├── phase1_dataset/
│   └── phase2_dataset/
├── output/                         # All runtime outputs (gitignored)
│   ├── kfold_results_v4/           # Default training output
│   └── inference_output*/          # Default inference output
└── trained_model_output/           # Pre-trained weights from HuggingFace
    └── kfold_results_v4/
        ├── norm_stats_v4.json
        ├── kfold_report_v4.json
        └── checkpoints/swinv2_unetplusplus_concat1x1/
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

## Dataset

The model is trained on the **Mars Landslide Segmentation (Mars LS)** dataset:
- **Format**: 7-band GeoTIFF files (128×128 pixels)
- **Bands** (1-indexed, rasterio order):

  | Band | Channel |
  |------|---------|
  | 1 | Thermal |
  | 2 | Slope |
  | 3 | DEM (Digital Elevation Model) |
  | 4 | Grayscale |
  | 5, 6, 7 | RGB |

- **Masks**: Binary (0 = background, 1 = landslide)

### Data Splits

| Split | Phase | Images | Masks |
|-------|-------|--------|-------|
| `data/phase1_dataset/train/` | Phase 1 | 465 | 465 |
| `data/phase1_dataset/val/` | Phase 1 | 66 | 66 |
| `data/phase1_dataset/test/` | Phase 1 | 133 | — |
| `data/phase2_dataset/test/` | Phase 2 | 276 | — |

> **Note:** The `data/` directory is **not tracked by git** (too large). Place the datasets manually.

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

| Component | Default | Options |
|-----------|---------|---------|
| **Encoder** | `swinv2_small_window8_256` | Any timm-compatible model |
| **Decoder** | `unetplusplus` (FPN ch=256) | `upernet`, `segformer_mlp`, `deeplabv3plus`, `fpn` |
| **Fusion** | `concat1x1` | `late_logits`, `weighted_sum`, `gated`, `film`, `cross_attn` |
| **Loss** | 0.5 × BCE(pos_weight) + 0.5 × Dice | — |
| **Optimizer** | AdamW (lr=2e-4, wd=1e-4) | — |
| **Scheduler** | Linear warmup (3 epochs) + cosine decay | — |
| **EMA** | Decay=0.995 with warmup | — |
| **Parameters** | ~100M (dual encoder + decoder + fusion) | — |

---

## Preprocessing / Normalization (v4 — Domain-Invariant)

1. **Per-image percentile normalization**: For each image independently, each band is:
   - Clipped to its own P1 and P99 percentiles
   - Rescaled to [0, 1]

   This eliminates sensitivity to absolute value shifts between domains (e.g., DEM in training has elevations [-1345, 3110] while test has [4275, 7232]).

2. **Z-score standardization**: After per-image normalization, channels are standardized using global mean/std computed over the training set (saved in `norm_stats_v4.json`).

3. **Augmentations** (training only):
   - **Geometric**: HorizontalFlip, VerticalFlip, RandomRotate90, Affine (translate, scale, rotate)
   - **Photometric** (RGB only): GaussianBlur, RandomBrightnessContrast

---

## How to Reproduce Results

### 1. Environment Setup

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

### 2. Configuration Reference

Both `src/train.py` and `src/infer.py` are fully configurable via CLI flags. All arguments have sensible defaults defined in `src/config.py`.

#### Training Arguments (`python -m src.train`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_root` | str | **required** | Path to dataset root (must contain `train/`, `val/`, `test/` subdirs) |
| `--out_dir` | str | `output/kfold_results_v4` | Output directory for checkpoints, stats, reports, submissions |
| `--encoder_name` | str | `swinv2_small_window8_256` | timm encoder backbone name |
| `--decoder_name` | str | `unetplusplus` | Decoder head: `unetplusplus`, `upernet`, `segformer_mlp`, `deeplabv3plus`, `fpn` |
| `--fusion_name` | str | `concat1x1` | Fusion strategy: `concat1x1`, `late_logits`, `weighted_sum`, `gated`, `film`, `cross_attn` |
| `--fpn_channels` | int | `256` | Number of channels in decoder FPN layers |
| `--img_size` | int | `128` | Input image resolution (resized if different from native) |
| `--epochs` | int | `50` | Training epochs per fold |
| `--batch_size` | int | `16` | Training batch size |
| `--num_workers` | int | `2` | DataLoader workers |
| `--lr` | float | `2e-4` | Initial learning rate |
| `--weight_decay` | float | `1e-4` | AdamW weight decay |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--n_folds` | int | `5` | Number of cross-validation folds |
| `--ema_decay` | float | `0.995` | EMA decay rate |
| `--warmup_epochs` | int | `3` | LR warmup epochs |
| `--thresh` | float | `0.5` | Binarization threshold for validation metrics |
| `--no_pretrained` | flag | `False` | Disable ImageNet pretrained weights |
| `--no_amp` | flag | `False` | Disable automatic mixed precision |

#### Inference Arguments (`python -m src.infer`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--test_dir` | str | **required** | Directory containing test `.tif` files |
| `--ckpt_dir` | str | **required** | Directory containing `fold{1..K}_best.pt` checkpoints |
| `--stats_json` | str | `None` | Path to `norm_stats_v4.json` (required unless `--recompute_from` is given) |
| `--recompute_from` | str | `None` | If stats_json unavailable, recompute stats from this dataset root |
| `--out_dir` | str | `output/inference_output` | Output directory for predictions and submission zip |
| `--encoder_name` | str | `swinv2_small_window8_256` | Must match the encoder used during training |
| `--decoder_name` | str | `unetplusplus` | Must match the decoder used during training |
| `--fusion_name` | str | `concat1x1` | Must match the fusion used during training |
| `--fpn_channels` | int | `256` | Must match training |
| `--img_size` | int | `128` | Must match training |
| `--orig_size` | int | `128` | Output mask resolution |
| `--batch_size` | int | `8` | Inference batch size |
| `--num_workers` | int | `2` | DataLoader workers |
| `--thresh` | float | `0.51` | Binarization threshold |
| `--n_folds` | int | `5` | Number of fold checkpoints to load |
| `--no_tta` | flag | `False` | Disable test-time augmentation (faster, slightly less accurate) |
| `--seed` | int | `42` | Random seed |

> **Important**: For inference, `--encoder_name`, `--decoder_name`, `--fusion_name`, `--fpn_channels`, and `--img_size` **must match** the values used during training, or the checkpoint will fail to load.

---

### 3. Training

#### Quick start (defaults)
```bash
python -m src.train --data_root data/phase1_dataset
```

This will:
1. Combine train + val images → 531 samples for 5-fold CV
2. Compute per-image normalization stats → `output/kfold_results_v4/norm_stats_v4.json`
3. Train 5 folds × 50 epochs with EMA + cosine LR
4. Save best checkpoint per fold → `output/kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1/`
5. Generate ensemble test predictions → `output/kfold_results_v4/submissions/`
6. Write K-Fold report → `output/kfold_results_v4/kfold_report_v4.json`
7. Save training curves plot

#### Example: different architecture
```bash
# UPerNet decoder + cross-attention fusion
python -m src.train \
    --data_root data/phase1_dataset \
    --decoder_name upernet \
    --fusion_name cross_attn \
    --out_dir output/experiment_upernet_crossattn

# SegFormer MLP decoder + gated fusion, smaller batch for 8GB GPU
python -m src.train \
    --data_root data/phase1_dataset \
    --decoder_name segformer_mlp \
    --fusion_name gated \
    --batch_size 8 \
    --out_dir output/experiment_segformer_gated

# DeepLabV3+ with FiLM fusion, 3-fold CV, more epochs
python -m src.train \
    --data_root data/phase1_dataset \
    --decoder_name deeplabv3plus \
    --fusion_name film \
    --n_folds 3 \
    --epochs 80 \
    --out_dir output/experiment_deeplabv3plus_film

# Simple FPN decoder + late logits fusion (each encoder gets its own decoder)
python -m src.train \
    --data_root data/phase1_dataset \
    --decoder_name fpn \
    --fusion_name late_logits \
    --out_dir output/experiment_fpn_late
```

#### Example: tune training hyperparameters
```bash
python -m src.train \
    --data_root data/phase1_dataset \
    --lr 1e-4 \
    --weight_decay 5e-4 \
    --ema_decay 0.999 \
    --warmup_epochs 5 \
    --batch_size 8 \
    --epochs 100 \
    --no_amp
```

#### Running in background (tmux)
```bash
tmux new-session -d -s train \
  "conda run --no-capture-output -n mars_ls \
   python -m src.train --data_root data/phase1_dataset \
   2>&1 | tee output/train.log"

# Attach to watch progress
tmux attach -t train
# Detach: Ctrl+B  d
```

**Estimated training time**: ~3–5 hours on a single NVIDIA T4/P100 (50 epochs × 5 folds).

---

### 4. Inference

#### With pre-trained HuggingFace checkpoints

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

#### With freshly trained checkpoints (uses output/ paths)
```bash
python -m src.infer \
    --test_dir data/phase1_dataset/test/images \
    --ckpt_dir output/kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1 \
    --stats_json output/kfold_results_v4/norm_stats_v4.json
```

#### With a non-default architecture (must match training)
```bash
# If you trained with upernet + cross_attn:
python -m src.infer \
    --test_dir data/phase1_dataset/test/images \
    --ckpt_dir output/experiment_upernet_crossattn/checkpoints/swinv2_upernet_cross_attn \
    --stats_json output/experiment_upernet_crossattn/norm_stats_v4.json \
    --decoder_name upernet \
    --fusion_name cross_attn
```

#### Inference options
```bash
# Disable TTA for faster inference (~2-4× speedup):
python -m src.infer --test_dir ... --ckpt_dir ... --stats_json ... --no_tta

# Custom binarization threshold:
python -m src.infer --test_dir ... --ckpt_dir ... --stats_json ... --thresh 0.5

# 3-fold model (if trained with --n_folds 3):
python -m src.infer --test_dir ... --ckpt_dir ... --stats_json ... --n_folds 3
```

#### Output
```
output/inference_output/
├── predictions/        # Individual mask .tif files
└── submission.zip      # Ready-to-submit zip archive
```

---

### 5. Training/Inference via Notebooks

**Main submission notebooks** in `notebooks/`:
- `notebooks/dual_swin_unetpp_kfold_training.ipynb` — full training pipeline (Dual Swin + UNet++)
- `notebooks/dual_swin_unetpp_kfold_infer.ipynb` — inference with TTA + ensemble

**Experiment notebooks** in `experiments/`:
- `experiments/dual_swin_hybrid_sfxunetpp_attn_kfold_training.ipynb` — Hybrid SegFormer×UNet++ + channel attention (ECA/SE/CBAM)
- `experiments/dual_swin_hybrid_sfxunetpp_attn_kfold_infer.ipynb` — Hybrid SegFormer×UNet++ inference

Configure paths in the notebook cells and run all cells sequentially.

---

## Cross-Validation Results

Results from 5-fold cross-validation with the default configuration (from `kfold_report_v4.json`):

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
| `src/augmentations.py` | Albumentations augmentation pipelines |
| `src/dataset.py` | `MarsSegDataset` (train/val) and `InferenceDataset` (test) |
| `src/model.py` | `DualSwinFusionSeg` + 5 decoders + 6 fusions |
| `src/losses.py` | `WeightedBCEDiceLoss`, evaluation metrics, pos_weight |
| `src/utils.py` | EMA, seed, TTA, model loading, submission writing |

---

## License

This code is released for the Mars Landslide Segmentation Challenge. Please cite the challenge dataset and organizers if you use this work.
