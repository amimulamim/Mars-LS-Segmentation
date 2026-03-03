#!/usr/bin/env python3
"""
train.py — K-Fold Cross-Validation Training for Dual Swin V2 Segmentation.

Usage
-----
    python -m src.train --data_root data/phase1_dataset

    # Override defaults:
    python -m src.train --data_root data/phase1_dataset --epochs 30 --batch_size 8 \
                        --decoder_name unetplusplus --fusion_name concat1x1

All hyperparameters can be overridden via CLI flags (see --help).
The script produces:
  <out_dir>/
    norm_stats_v4.json        — normalization statistics
    kfold_report_v4.json      — per-fold and aggregate metrics
    checkpoints/<tag>/fold{1..K}_best.pt
    submissions/<tag>/        — per-tile predictions
    submissions/<tag>_kfold_ensemble.zip
"""

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    BAND_INDICES, RGB_BANDS, AUX_BANDS, DEVICE, DEFAULT_CFG,
)
from .normalization import (
    compute_mean_std_per_image_norm, save_norm_stats,
)
from .augmentations import build_train_transforms, build_val_transforms
from .dataset import MarsSegDataset
from .model import DualSwinFusionSeg
from .losses import WeightedBCEDiceLoss, compute_metrics, compute_pos_weight
from .utils import (
    set_seed, EMA, make_lr_lambda, ensemble_predict_tiffs, zip_submission,
)


# ──────────────────────────────────────────────────────────────
# Validation loss
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def validate_loss(model, loader, loss_fn, use_amp=True):
    model.eval()
    total = n = 0.0
    for rgb, aux, mask in tqdm(loader, desc="ValLoss", leave=False):
        rgb  = rgb.to(DEVICE, non_blocking=True)
        aux  = aux.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(use_amp and DEVICE == "cuda")):
            logits = model(rgb, aux)
            loss = loss_fn(logits, mask)
        total += loss.item() * rgb.size(0)
        n += rgb.size(0)
    return total / max(n, 1)


# ──────────────────────────────────────────────────────────────
# Single-fold training loop
# ──────────────────────────────────────────────────────────────
def train_one_fold(model, train_loader, val_loader, cfg, loss_fn, fold_num, ckpt_dir):
    """Train one fold; return fold result dict and the best model (eval mode)."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    steps_per_epoch = len(train_loader)
    warmup_iters = cfg["warmup_epochs"] * steps_per_epoch
    total_iters  = cfg["epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup_iters, total_iters),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["amp"] and DEVICE == "cuda"))
    ema    = EMA(model, decay=cfg["ema_decay"], warmup_steps=warmup_iters)

    best_miou  = -1.0
    best_epoch = -1
    best_ckpt  = str(ckpt_dir / f"fold{fold_num}_best.pt")
    epoch_logs = []

    for epoch in range(1, cfg["epochs"] + 1):
        # ── train ────────────────────────────────────────────
        model.train()
        train_loss_sum = n_train = 0
        for rgb, aux, mask in tqdm(train_loader, desc="Train", leave=False):
            rgb  = rgb.to(DEVICE, non_blocking=True)
            aux  = aux.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(cfg["amp"] and DEVICE == "cuda")):
                logits = model(rgb, aux)
                loss = loss_fn(logits, mask)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            ema.update(model)
            bs = rgb.size(0)
            train_loss_sum += loss.item() * bs
            n_train += bs
        train_loss = train_loss_sum / max(n_train, 1)

        # ── validate with RAW weights ────────────────────────
        model.eval()
        raw_val_loss = validate_loss(model, val_loader, loss_fn, use_amp=cfg["amp"])
        raw_metrics  = compute_metrics(model, val_loader, thresh=cfg["thresh"])

        # ── validate with EMA weights ────────────────────────
        ema.apply_shadow(model)
        ema_val_loss = validate_loss(model, val_loader, loss_fn, use_amp=cfg["amp"])
        ema_metrics  = compute_metrics(model, val_loader, thresh=cfg["thresh"])
        ema.restore(model)

        # ── pick the better ──────────────────────────────────
        if ema_metrics["mIoU"] >= raw_metrics["mIoU"]:
            val_loss, metrics, use_ema = ema_val_loss, ema_metrics, True
        else:
            val_loss, metrics, use_ema = raw_val_loss, raw_metrics, False

        log_entry = {
            "epoch": epoch, "train_loss": float(train_loss),
            "val_loss": float(val_loss), "lr": float(optimizer.param_groups[0]["lr"]),
            "used_ema": use_ema,
            "raw_mIoU": float(raw_metrics["mIoU"]),
            "ema_mIoU": float(ema_metrics["mIoU"]),
            **{k: float(metrics[k]) for k in
               ["mIoU", "IoU_fg", "IoU_bg", "F1_fg", "Precision_fg", "Recall_fg"]},
        }
        epoch_logs.append(log_entry)

        tag = "EMA" if use_ema else "RAW"
        print(f"  Epoch {epoch:3d}/{cfg['epochs']} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"mIoU={metrics['mIoU']:.4f} | F1={metrics['F1_fg']:.4f} | "
              f"IoU_fg={metrics['IoU_fg']:.4f} | "
              f"[{tag}] raw={raw_metrics['mIoU']:.4f} ema={ema_metrics['mIoU']:.4f}")

        if metrics["mIoU"] > best_miou:
            best_miou  = metrics["mIoU"]
            best_epoch = epoch
            if use_ema:
                ema.apply_shadow(model)
            torch.save({
                "model": model.state_dict(),
                "fold": fold_num,
                "best_epoch": best_epoch,
                "best_metrics": metrics,
                "used_ema": use_ema,
            }, best_ckpt)
            if use_ema:
                ema.restore(model)

    # reload best
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    result = {
        "fold": fold_num,
        "best_epoch": best_epoch,
        "best_ckpt": best_ckpt,
        "best_metrics": ckpt["best_metrics"],
        "epoch_logs": epoch_logs,
    }
    print(f"  => Fold {fold_num} best mIoU={best_miou:.4f} at epoch {best_epoch}")
    return result, model


# ──────────────────────────────────────────────────────────────
# K-Fold experiment runner
# ──────────────────────────────────────────────────────────────
def run_kfold(cfg, all_img_paths, all_mask_paths, test_img_paths,
              means, stds, pos_weight):
    out_dir  = Path(cfg["out_dir"])
    tag      = f"swinv2_{cfg['decoder_name']}_{cfg['fusion_name']}"
    ckpt_dir = out_dir / "checkpoints" / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    geo_aug, rgb_photo = build_train_transforms(cfg["img_size"])
    val_aug = build_val_transforms(cfg["img_size"])

    kf = KFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    loss_fn = WeightedBCEDiceLoss(pos_weight=pos_weight)

    fold_results = []
    fold_models  = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_img_paths)):
        fold_num = fold_idx + 1
        print(f"\n--- Fold {fold_num}/{cfg['n_folds']} ---")
        print(f"    Train: {len(train_idx)} | Val: {len(val_idx)}")

        train_imgs  = [all_img_paths[i] for i in train_idx]
        train_masks = [all_mask_paths[i] for i in train_idx]
        val_imgs    = [all_img_paths[i] for i in val_idx]
        val_masks   = [all_mask_paths[i] for i in val_idx]

        train_ds = MarsSegDataset(
            train_imgs, train_masks, means, stds,
            geo_aug=geo_aug, rgb_photo_aug=rgb_photo, is_train=True,
        )
        val_ds = MarsSegDataset(
            val_imgs, val_masks, means, stds,
            val_aug=val_aug, is_train=False,
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True,
        )

        model = DualSwinFusionSeg(
            encoder_name=cfg["encoder_name"],
            pretrained=cfg["pretrained"],
            img_size=cfg["img_size"],
            fpn_channels=cfg["fpn_channels"],
            fusion_name=cfg["fusion_name"],
            decoder_name=cfg["decoder_name"],
        ).to(DEVICE)

        result, model = train_one_fold(
            model, train_loader, val_loader, cfg, loss_fn, fold_num, ckpt_dir,
        )
        result["num_train"] = len(train_idx)
        result["num_val"]   = len(val_idx)
        fold_results.append(result)
        fold_models.append(model)

    # ── Aggregate ────────────────────────────────────────────
    metric_keys = ["mIoU", "IoU_fg", "IoU_bg", "F1_fg", "Precision_fg", "Recall_fg"]
    agg = {}
    for k in metric_keys:
        vals = [fr["best_metrics"][k] for fr in fold_results]
        agg[k] = {
            "mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "min": float(np.min(vals)),   "max": float(np.max(vals)),
            "per_fold": vals,
        }

    # ── Ensemble test predictions ────────────────────────────
    test_ds = MarsSegDataset(
        test_img_paths, None, means, stds,
        val_aug=val_aug, is_train=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    sub_dir = out_dir / "submissions" / tag
    ensemble_predict_tiffs(
        fold_models, test_loader, sub_dir,
        thresh=cfg["thresh"], img_size=cfg["img_size"], device=DEVICE,
    )
    zip_path = str(out_dir / "submissions" / f"{tag}_kfold_ensemble.zip")
    zip_submission(sub_dir, zip_path)

    return {
        "encoder": cfg["encoder_name"], "decoder": cfg["decoder_name"],
        "fusion": cfg["fusion_name"], "n_folds": cfg["n_folds"],
        "fold_results": fold_results, "aggregate_metrics": agg,
        "ensemble_submission_zip": zip_path,
        "num_params": sum(p.numel() for p in fold_models[0].parameters()),
    }


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────
def plot_kfold_results(result, out_dir):
    agg = result["aggregate_metrics"]
    tag = f"{result['encoder'].split('_')[0]} / {result['decoder']} / {result['fusion']}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fold_mious = agg["mIoU"]["per_fold"]
    folds_x = list(range(1, len(fold_mious) + 1))

    ax = axes[0]
    ax.bar(folds_x, fold_mious, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axhline(y=agg["mIoU"]["mean"], color="red", linestyle="--", linewidth=2,
               label=f"Mean = {agg['mIoU']['mean']:.4f}")
    ax.fill_between([0.5, len(folds_x) + 0.5],
                    agg["mIoU"]["mean"] - agg["mIoU"]["std"],
                    agg["mIoU"]["mean"] + agg["mIoU"]["std"],
                    alpha=0.15, color="red", label=f"±1 std = {agg['mIoU']['std']:.4f}")
    ax.set_xlabel("Fold"); ax.set_ylabel("mIoU")
    ax.set_title(f"{tag}\nPer-Fold mIoU")
    ax.set_xticks(folds_x); ax.legend()
    ax.set_ylim(0, max(fold_mious) * 1.15)

    ax = axes[1]
    for fr in result["fold_results"]:
        epochs = [e["epoch"] for e in fr["epoch_logs"]]
        mious  = [e["mIoU"]  for e in fr["epoch_logs"]]
        ax.plot(epochs, mious, label=f"Fold {fr['fold']}", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("mIoU")
    ax.set_title(f"{tag}\nValidation mIoU per Epoch")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(out_dir) / f"kfold_plot_{result['decoder']}_{result['fusion']}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved: {plot_path}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Mars Landslide Seg — K-Fold Training")
    p.add_argument("--data_root",    type=str, required=True,
                   help="Path to dataset root (contains train/, val/, test/)")
    p.add_argument("--out_dir",      type=str, default="kfold_results_v4")
    p.add_argument("--encoder_name", type=str, default=DEFAULT_CFG["encoder_name"])
    p.add_argument("--decoder_name", type=str, default=DEFAULT_CFG["decoder_name"])
    p.add_argument("--fusion_name",  type=str, default=DEFAULT_CFG["fusion_name"])
    p.add_argument("--fpn_channels", type=int, default=DEFAULT_CFG["fpn_channels"])
    p.add_argument("--img_size",     type=int, default=DEFAULT_CFG["img_size"])
    p.add_argument("--epochs",       type=int, default=DEFAULT_CFG["epochs"])
    p.add_argument("--batch_size",   type=int, default=DEFAULT_CFG["batch_size"])
    p.add_argument("--num_workers",  type=int, default=DEFAULT_CFG["num_workers"])
    p.add_argument("--lr",           type=float, default=DEFAULT_CFG["lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_CFG["weight_decay"])
    p.add_argument("--seed",         type=int, default=DEFAULT_CFG["seed"])
    p.add_argument("--n_folds",      type=int, default=DEFAULT_CFG["n_folds"])
    p.add_argument("--ema_decay",    type=float, default=DEFAULT_CFG["ema_decay"])
    p.add_argument("--warmup_epochs",type=int, default=DEFAULT_CFG["warmup_epochs"])
    p.add_argument("--thresh",       type=float, default=DEFAULT_CFG["thresh"])
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--no_amp",       action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg = {**DEFAULT_CFG}
    cfg.update(vars(args))
    cfg["pretrained"] = not args.no_pretrained
    cfg["amp"] = not args.no_amp

    set_seed(cfg["seed"])
    print(f"DEVICE: {DEVICE}")
    print(f"Config: {json.dumps({k: v for k, v in cfg.items() if not isinstance(v, (list, np.ndarray))}, indent=2)}")

    data_root = Path(cfg["data_root"])
    out_dir   = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── gather image / mask paths ────────────────────────────
    train_img_dir  = data_root / "train" / "images"
    train_mask_dir = data_root / "train" / "masks"
    val_img_dir    = data_root / "val"   / "images"
    val_mask_dir   = data_root / "val"   / "masks"
    test_img_dir   = data_root / "test"  / "images"

    all_img_paths = sorted(
        list(train_img_dir.glob("*.tif")) + list(val_img_dir.glob("*.tif"))
    )
    all_mask_paths = []
    for img_p in all_img_paths:
        if img_p.parent.parent.name == "train":
            mask_p = train_mask_dir / img_p.name
        else:
            mask_p = val_mask_dir / img_p.name
        assert mask_p.exists(), f"Mask not found: {mask_p}"
        all_mask_paths.append(mask_p)

    test_img_paths = sorted(list(test_img_dir.glob("*.tif")))

    print(f"Total labeled (train+val): {len(all_img_paths)}")
    print(f"Test images: {len(test_img_paths)}")

    # ── normalization stats ──────────────────────────────────
    means, stds = compute_mean_std_per_image_norm(
        [str(p) for p in all_img_paths], BAND_INDICES,
    )
    print(f"Channel means: {means}")
    print(f"Channel stds:  {stds}")

    # ── pos weight ───────────────────────────────────────────
    fg_frac, pos_weight = compute_pos_weight(all_mask_paths)
    print(f"FG frac: {fg_frac:.6f} | pos_weight: {pos_weight:.2f}")

    # ── save norm stats ──────────────────────────────────────
    stats_path = out_dir / "norm_stats_v4.json"
    save_norm_stats(
        stats_path, means, stds, BAND_INDICES, RGB_BANDS, AUX_BANDS,
        cfg["img_size"], pos_weight=pos_weight, fg_frac=fg_frac,
    )
    print(f"Norm stats saved: {stats_path}")

    # ── run K-Fold ───────────────────────────────────────────
    t0 = time.time()
    result = run_kfold(
        cfg, all_img_paths, all_mask_paths, test_img_paths,
        means, stds, pos_weight,
    )
    elapsed = time.time() - t0

    # ── save report ──────────────────────────────────────────
    report = {
        "experiment_id": "DualSwinV2_RGB_AUX4_KFold_v4_PerImageNorm",
        "channels": ["RGB", "DEM", "SLOPE", "THERMAL", "GRAY"],
        "band_indices": BAND_INDICES,
        "rgb_bands": RGB_BANDS,
        "aux_bands": AUX_BANDS,
        "pos_weight": pos_weight,
        "total_labeled_samples": len(all_img_paths),
        "n_folds": cfg["n_folds"],
        "cfg": {k: v for k, v in cfg.items() if not isinstance(v, np.ndarray)},
        "results": [result],
        "elapsed_seconds": elapsed,
    }
    report_path = out_dir / "kfold_report_v4.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    # ── plot ─────────────────────────────────────────────────
    plot_kfold_results(result, out_dir)

    # ── summary ──────────────────────────────────────────────
    agg = result["aggregate_metrics"]
    print("\n" + "=" * 80)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(f"  mIoU:      {agg['mIoU']['mean']:.4f} ± {agg['mIoU']['std']:.4f}")
    print(f"  IoU_fg:    {agg['IoU_fg']['mean']:.4f} ± {agg['IoU_fg']['std']:.4f}")
    print(f"  IoU_bg:    {agg['IoU_bg']['mean']:.4f} ± {agg['IoU_bg']['std']:.4f}")
    print(f"  F1_fg:     {agg['F1_fg']['mean']:.4f} ± {agg['F1_fg']['std']:.4f}")
    print(f"  Precision: {agg['Precision_fg']['mean']:.4f} ± {agg['Precision_fg']['std']:.4f}")
    print(f"  Recall:    {agg['Recall_fg']['mean']:.4f} ± {agg['Recall_fg']['std']:.4f}")
    print(f"  Per-fold:  {agg['mIoU']['per_fold']}")
    print(f"  Submission: {result['ensemble_submission_zip']}")
    print(f"  Time: {elapsed/60:.1f} min")
    print("DONE.")


if __name__ == "__main__":
    main()
