"""
Utility helpers: EMA, seed, submission I/O, model loading.
"""

import math
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
import torch.nn as nn
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ──────────────────────────────────────────────────────────────
# Exponential Moving Average (with warmup)
# ──────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model: nn.Module, decay=0.995, warmup_steps=0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def _get_decay(self):
        if self.warmup_steps > 0 and self.step_count < self.warmup_steps:
            return min(self.decay, 1.0 - 1.0 / (self.step_count + 1))
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self._get_decay()
        self.step_count += 1
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name] = (1.0 - d) * p.data + d * self.shadow[name]

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.data.clone()
            p.data = self.shadow[name].clone()

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data = self.backup[name].clone()
        self.backup = {}


# ──────────────────────────────────────────────────────────────
# LR schedule: linear warmup → cosine decay
# ──────────────────────────────────────────────────────────────
def make_lr_lambda(warmup_iters, total_iters):
    """Return a callable for ``torch.optim.lr_scheduler.LambdaLR``."""
    def lr_lambda(step):
        if step < warmup_iters:
            return max(step / max(warmup_iters, 1), 0.01)
        progress = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ──────────────────────────────────────────────────────────────
# Submission writing
# ──────────────────────────────────────────────────────────────
def write_mask_tiff(mask01, out_path, height=128, width=128):
    """Write a binary mask as a single-band GeoTIFF."""
    if mask01.shape[0] != height or mask01.shape[1] != width:
        mask01 = cv2.resize(mask01.astype(np.uint8), (width, height),
                            interpolation=cv2.INTER_NEAREST)
    with rasterio.open(
        str(out_path), "w", driver="GTiff",
        height=height, width=width, count=1, dtype=rasterio.uint8,
    ) as dst:
        dst.write(mask01.astype(np.uint8), 1)


@torch.no_grad()
def ensemble_predict_tiffs(models, loader, out_dir, thresh=0.5, img_size=128, device="cuda"):
    """Average sigmoid probs from K models, threshold, write mask TIFs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        m.eval()
    written = 0
    for rgb, aux, out_names in tqdm(loader, desc="Ensemble Infer", leave=False):
        rgb = rgb.to(device, non_blocking=True)
        aux = aux.to(device, non_blocking=True)
        avg_probs = None
        for m in models:
            logits = m(rgb, aux)
            probs = torch.sigmoid(logits)
            avg_probs = probs if avg_probs is None else avg_probs + probs
        avg_probs = (avg_probs / len(models)).cpu().numpy()
        for i in range(avg_probs.shape[0]):
            mask01 = (avg_probs[i, 0] > thresh).astype(np.uint8)
            write_mask_tiff(mask01, out_dir / out_names[i], height=img_size, width=img_size)
            written += 1
    print(f"[ensemble_predict_tiffs] Wrote {written} masks → {out_dir}")
    return out_dir


# ──────────────────────────────────────────────────────────────
# TTA (test-time augmentation)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def tta_predict(model, rgb, aux):
    """4-fold TTA: original + hflip + vflip + rot90."""
    def infer(r, a):
        return torch.sigmoid(model(r, a))
    p  = infer(rgb, aux)
    p += infer(torch.flip(rgb, [-1]), torch.flip(aux, [-1])).flip(-1)
    p += infer(torch.flip(rgb, [-2]), torch.flip(aux, [-2])).flip(-2)
    p += infer(torch.rot90(rgb, 1, [-2, -1]),
               torch.rot90(aux, 1, [-2, -1])).rot90(3, [-2, -1])
    return p / 4.0


@torch.no_grad()
def ensemble_predict_tta(models, loader, out_dir, thresh=0.5,
                         img_size=128, orig_size=128, use_tta=True, device="cuda"):
    """Ensemble + optional TTA inference → mask TIFs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        m.eval()
    written = 0
    for rgb, aux, names in tqdm(loader, desc="Inference"):
        rgb = rgb.to(device, non_blocking=True)
        aux = aux.to(device, non_blocking=True)
        prob = torch.zeros(rgb.shape[0], 1, img_size, img_size, device=device)
        for model in models:
            if use_tta:
                prob += tta_predict(model, rgb, aux)
            else:
                prob += torch.sigmoid(model(rgb, aux))
        prob /= len(models)
        masks = (prob[:, 0] > thresh).cpu().numpy().astype(np.uint8)
        for mask, name in zip(masks, names):
            write_mask_tiff(mask, out_dir / name, height=orig_size, width=orig_size)
            written += 1
    print(f"[ensemble_predict_tta] Wrote {written} masks → {out_dir}")
    return out_dir


# ──────────────────────────────────────────────────────────────
# Zip helper
# ──────────────────────────────────────────────────────────────
def zip_submission(pred_dir, zip_path):
    pred_dir = Path(pred_dir)
    tifs = sorted(pred_dir.glob("*.tif")) + sorted(pred_dir.glob("*.TIF"))
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for t in tifs:
            zf.write(t, arcname=t.name)
    print(f"[zip_submission] {len(tifs)} TIFs → {zip_path}")


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────
def load_fold_models(ckpt_dir, num_folds, encoder_name, img_size,
                     fpn_channels, fusion_name, decoder_name, device,
                     model_class="DualSwinFusionSeg", **extra_model_kwargs):
    """Load best checkpoint from each fold, return list of eval-mode models.

    Args:
        model_class: ``"DualSwinFusionSeg"`` (default) or
                     ``"DualSwinLateFusionSeg"`` for late-fusion models.
        extra_model_kwargs: forwarded to the model constructor (e.g.
                            ``input_attn``, ``intra_encoder_attn``, …).
    """
    from .model import DualSwinFusionSeg, DualSwinLateFusionSeg

    MODEL_MAP = {
        "DualSwinFusionSeg":       DualSwinFusionSeg,
        "DualSwinLateFusionSeg":   DualSwinLateFusionSeg,
    }
    cls = MODEL_MAP.get(model_class)
    if cls is None:
        raise ValueError(f"Unknown model_class '{model_class}'. "
                         f"Choose from {list(MODEL_MAP)}")

    ckpt_dir = Path(ckpt_dir)
    models = []
    for fold in range(1, num_folds + 1):
        ckpt_path = ckpt_dir / f"fold{fold}_best.pt"
        if not ckpt_path.exists():
            print(f"  [WARNING] {ckpt_path} not found — skipping fold {fold}")
            continue

        # Build model — mid-fusion needs fusion_name & decoder_name,
        # late-fusion doesn't but ignores them gracefully.
        if cls is DualSwinFusionSeg:
            model = cls(
                encoder_name=encoder_name, pretrained=False,
                img_size=img_size, fpn_channels=fpn_channels,
                fusion_name=fusion_name, decoder_name=decoder_name,
                **extra_model_kwargs,
            ).to(device)
        else:  # DualSwinLateFusionSeg
            model = cls(
                encoder_name=encoder_name, pretrained=False,
                img_size=img_size, fpn_channels=fpn_channels,
                **extra_model_kwargs,
            ).to(device)

        ckpt = torch.load(str(ckpt_path), map_location=device)
        state = ckpt.get("model", ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing:
            print(f"  fold{fold} missing keys: {missing[:5]}")
        if unexpected:
            print(f"  fold{fold} unexpected keys: {unexpected[:5]}")
        model.eval()
        best = ckpt.get("best_metrics", {})
        miou = best.get("mIoU", best.get("miou"))
        info = f" — val_mIoU: {miou:.4f}" if miou else ""
        print(f"  fold{fold} loaded{info}")
        models.append(model)
    print(f"[load_fold_models] {len(models)}/{num_folds} folds loaded.")
    return models
