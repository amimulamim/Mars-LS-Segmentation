#!/usr/bin/env python3
"""
infer.py — Ensemble Inference with TTA for Dual Swin V2 Segmentation.

Usage
-----
    # Phase 1 test set (using pre-trained checkpoints):
    python -m src.infer \
        --test_dir data/phase1_dataset/test/images \
        --ckpt_dir trained_model_output/kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1 \
        --stats_json trained_model_output/kfold_results_v4/norm_stats_v4.json

    # Phase 2 test set:
    python -m src.infer \
        --test_dir data/phase2_dataset/test/images \
        --ckpt_dir trained_model_output/kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1 \
        --stats_json trained_model_output/kfold_results_v4/norm_stats_v4.json

    # After training with src/train.py:
    python -m src.infer \
        --test_dir data/phase1_dataset/test/images \
        --ckpt_dir kfold_results_v4/checkpoints/swinv2_unetplusplus_concat1x1 \
        --stats_json kfold_results_v4/norm_stats_v4.json

    # Disable TTA for faster (but slightly less accurate) inference:
    python -m src.infer --test_dir ... --ckpt_dir ... --stats_json ... --no_tta
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DEVICE, DEFAULT_CFG
from .normalization import load_norm_stats, compute_mean_std_per_image_norm
from .dataset import InferenceDataset
from .utils import (
    set_seed, load_fold_models, ensemble_predict_tta, zip_submission,
)


def parse_args():
    p = argparse.ArgumentParser(description="Mars Landslide Seg — Inference")
    p.add_argument("--test_dir",     type=str, required=True,
                   help="Directory containing test .tif files")
    p.add_argument("--ckpt_dir",     type=str, required=True,
                   help="Directory containing fold{1..K}_best.pt checkpoints")
    p.add_argument("--stats_json",   type=str, default=None,
                   help="Path to norm_stats_v4.json. If omitted, recompute from --recompute_from")
    p.add_argument("--recompute_from", type=str, default=None,
                   help="If stats_json unavailable, recompute stats from this dataset root")
    p.add_argument("--out_dir",      type=str, default="inference_output")
    p.add_argument("--encoder_name", type=str, default=DEFAULT_CFG["encoder_name"])
    p.add_argument("--decoder_name", type=str, default=DEFAULT_CFG["decoder_name"])
    p.add_argument("--fusion_name",  type=str, default=DEFAULT_CFG["fusion_name"])
    p.add_argument("--fpn_channels", type=int, default=DEFAULT_CFG["fpn_channels"])
    p.add_argument("--img_size",     type=int, default=DEFAULT_CFG["img_size"])
    p.add_argument("--orig_size",    type=int, default=128,
                   help="Output mask resolution")
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--num_workers",  type=int, default=2)
    p.add_argument("--thresh",       type=float, default=0.51)
    p.add_argument("--n_folds",      type=int, default=DEFAULT_CFG["n_folds"])
    p.add_argument("--no_tta",       action="store_true",
                   help="Disable test-time augmentation")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"Device: {DEVICE}")

    # ── Load or compute normalization stats ──────────────────
    if args.stats_json and Path(args.stats_json).exists():
        means, stds, stats_data = load_norm_stats(args.stats_json)
        print(f"Loaded stats from {args.stats_json}")
        print(f"  normalization: {stats_data.get('normalization', '?')}")
    elif args.recompute_from:
        from .config import BAND_INDICES
        root = Path(args.recompute_from)
        imgs = sorted(
            list((root / "train" / "images").glob("*.tif")) +
            list((root / "val"   / "images").glob("*.tif"))
        )
        print(f"Recomputing stats from {len(imgs)} images in {root}")
        means, stds = compute_mean_std_per_image_norm(
            [str(p) for p in imgs], BAND_INDICES,
        )
    else:
        raise ValueError(
            "Provide --stats_json or --recompute_from to get normalization stats"
        )
    print(f"  mean = {means}")
    print(f"  std  = {stds}")

    # ── Build test dataset & loader ──────────────────────────
    test_ds = InferenceDataset(
        image_dir=args.test_dir,
        img_size=args.img_size,
        means=means,
        stds=stds,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # ── Load fold models ─────────────────────────────────────
    models = load_fold_models(
        ckpt_dir=args.ckpt_dir,
        num_folds=args.n_folds,
        encoder_name=args.encoder_name,
        img_size=args.img_size,
        fpn_channels=args.fpn_channels,
        fusion_name=args.fusion_name,
        decoder_name=args.decoder_name,
        device=DEVICE,
    )
    assert len(models) > 0, "No fold checkpoints loaded!"
    print(f"\nTest samples: {len(test_ds)} | Batches: {len(test_loader)}")
    print(f"TTA: {'ON (4-fold)' if not args.no_tta else 'OFF'}")

    # ── Run inference ────────────────────────────────────────
    out_dir  = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    ensemble_predict_tta(
        models=models,
        loader=test_loader,
        out_dir=str(pred_dir),
        thresh=args.thresh,
        img_size=args.img_size,
        orig_size=args.orig_size,
        use_tta=not args.no_tta,
        device=DEVICE,
    )

    # ── Zip ──────────────────────────────────────────────────
    zip_path = out_dir / "submission.zip"
    zip_submission(pred_dir, zip_path)
    print(f"\nSubmission ready: {zip_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
