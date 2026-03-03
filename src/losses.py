"""
Loss functions and evaluation metrics.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import DEVICE


# ──────────────────────────────────────────────────────────────
# Dice loss
# ──────────────────────────────────────────────────────────────
def dice_loss(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


# ──────────────────────────────────────────────────────────────
# Combined BCE + Dice
# ──────────────────────────────────────────────────────────────
class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=DEVICE)
        )

    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * dice_loss(logits, targets)


# ──────────────────────────────────────────────────────────────
# Leaderboard metrics  (TP / FP / FN / TN based)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics(model, loader, thresh=0.5, device=None):
    """Compute mIoU, IoU_fg/bg, F1, precision, recall on a DataLoader.

    Returns a dict of float metrics.
    """
    if device is None:
        device = DEVICE
    model.eval()
    TP = FP = FN = TN = 0.0
    for rgb, aux, mask in tqdm(loader, desc="ValMetric", leave=False):
        rgb  = rgb.to(device, non_blocking=True)
        aux  = aux.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(rgb, aux)
        pred = (torch.sigmoid(logits) > thresh).float()
        TP += (pred * mask).sum().item()
        FP += (pred * (1 - mask)).sum().item()
        FN += ((1 - pred) * mask).sum().item()
        TN += ((1 - pred) * (1 - mask)).sum().item()
    eps = 1e-7
    iou_fg  = TP / (TP + FP + FN + eps)
    iou_bg  = TN / (TN + FP + FN + eps)
    miou    = 0.5 * (iou_fg + iou_bg)
    prec_fg = TP / (TP + FP + eps)
    rec_fg  = TP / (TP + FN + eps)
    f1_fg   = 2 * prec_fg * rec_fg / (prec_fg + rec_fg + eps)
    return {
        "IoU_fg": float(iou_fg), "IoU_bg": float(iou_bg), "mIoU": float(miou),
        "Precision_fg": float(prec_fg), "Recall_fg": float(rec_fg), "F1_fg": float(f1_fg),
        "TP": float(TP), "FP": float(FP), "FN": float(FN), "TN": float(TN),
    }


# ──────────────────────────────────────────────────────────────
# Pos-weight estimation from mask files
# ──────────────────────────────────────────────────────────────
def compute_pos_weight(mask_paths):
    """Return (fg_fraction, pos_weight) from a list of mask file paths."""
    import rasterio
    fg = tot = 0
    for p in tqdm(mask_paths, desc="Compute pos_weight"):
        with rasterio.open(str(p)) as src:
            m = src.read(1)
        m01 = (m > 0).astype("uint8")
        fg  += int(m01.sum())
        tot += int(m01.size)
    frac = fg / tot
    pos_weight = (1.0 - frac) / (frac + 1e-9)
    return float(frac), float(pos_weight)
