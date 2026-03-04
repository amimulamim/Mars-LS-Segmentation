"""
Loss functions and evaluation metrics.

Includes:
  - dice_loss          : basic dice loss
  - WeightedBCEDiceLoss: default 0.5*BCE + 0.5*dice
  - boundary_loss      : boundary-weighted BCE  (ablation)
  - lovasz_hinge       : Lovász-Softmax hinge   (ablation)
  - HybridBCEDiceBoundaryLoss : BCE + Dice + Boundary + Lovász  (ablation)
  - DeepSupervisionLoss       : wraps any base loss with aux-head supervision
  - LateFusionDeepSupervisionLoss : per-branch + fused + aux supervision
  - compute_metrics    : mIoU, F1, precision, recall evaluation
  - compute_pos_weight : foreground fraction + pos-weight from mask files
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Combined BCE + Dice  (default loss)
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
# Boundary-weighted BCE  (ablation component)
# ──────────────────────────────────────────────────────────────
def boundary_loss(logits, targets, kernel_size=3, boundary_weight=3.0):
    """Binary cross-entropy with extra weight along mask boundaries.

    A Laplacian edge detector highlights boundary pixels; those pixels
    receive ``boundary_weight`` times the normal loss contribution.
    """
    with torch.no_grad():
        pad = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, kernel_size,
                            device=targets.device) / (kernel_size ** 2)
        smooth = F.conv2d(targets, kernel, padding=pad)
        edges = ((smooth > 0.01) & (smooth < 0.99)).float()
        weights = 1.0 + edges * (boundary_weight - 1.0)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (bce * weights).mean()


# ──────────────────────────────────────────────────────────────
# Lovász hinge loss  (ablation component)
# ──────────────────────────────────────────────────────────────
def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t. sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovász hinge loss on flattened tensors."""
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge(logits, labels, per_image=True):
    """Lovász-Softmax hinge loss for binary segmentation.

    Args:
        logits: (B, 1, H, W) raw predictions
        labels: (B, 1, H, W) binary ground truth
        per_image: compute per-image then average (recommended)
    """
    if per_image:
        losses = [
            _lovasz_hinge_flat(
                logits[i].reshape(-1), labels[i].reshape(-1)
            )
            for i in range(logits.size(0))
        ]
        return torch.stack(losses).mean()
    return _lovasz_hinge_flat(logits.reshape(-1), labels.reshape(-1))


# ──────────────────────────────────────────────────────────────
# Hybrid BCE + Dice + Boundary + Lovász  (ablation component)
# ──────────────────────────────────────────────────────────────
class HybridBCEDiceBoundaryLoss(nn.Module):
    """Multi-component loss: α·BCE + β·Dice + γ·Boundary + δ·Lovász.

    Defaults reflect the best ablation recipe from the v6 experiments.
    """

    def __init__(self, pos_weight=1.0, alpha=0.3, beta=0.3, gamma=0.2,
                 delta=0.2, boundary_kernel=3, boundary_weight=3.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=DEVICE)
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.bk = boundary_kernel
        self.bw = boundary_weight

    def forward(self, logits, targets):
        loss = self.alpha * self.bce(logits, targets)
        loss += self.beta  * dice_loss(logits, targets)
        loss += self.gamma * boundary_loss(logits, targets, self.bk, self.bw)
        loss += self.delta * lovasz_hinge(logits, targets)
        return loss


# ──────────────────────────────────────────────────────────────
# Deep Supervision wrapper  (ablation component)
# ──────────────────────────────────────────────────────────────
class DeepSupervisionLoss(nn.Module):
    """Wraps a base loss and adds auxiliary-head supervision.

    Expects model output as ``{"logits": ..., "aux_logits": [...]}``.
    Falls back to plain tensor if model is in eval mode.
    """

    def __init__(self, base_loss, aux_weight=0.3):
        super().__init__()
        self.base_loss = base_loss
        self.aux_weight = aux_weight

    def forward(self, output, targets):
        if isinstance(output, dict):
            logits = output["logits"]
            loss = self.base_loss(logits, targets)
            for aux in output.get("aux_logits", []):
                loss += self.aux_weight * self.base_loss(aux, targets)
            return loss
        return self.base_loss(output, targets)


# ──────────────────────────────────────────────────────────────
# Late Fusion Deep Supervision  (ablation component)
# ──────────────────────────────────────────────────────────────
class LateFusionDeepSupervisionLoss(nn.Module):
    """Loss for DualSwinLateFusionSeg.

    Supervises the fused logits, per-branch logits, and per-branch
    auxiliary heads.

    Expects model training output dict with keys:
        ``logits``, ``rgb_logits``, ``aux_logits_branch``,
        ``rgb_aux``, ``aux_aux``.
    """

    def __init__(self, base_loss, branch_weight=0.3, aux_weight=0.3):
        super().__init__()
        self.base_loss = base_loss
        self.branch_weight = branch_weight
        self.aux_weight = aux_weight

    def forward(self, output, targets):
        if isinstance(output, dict):
            loss = self.base_loss(output["logits"], targets)
            loss += self.branch_weight * self.base_loss(output["rgb_logits"], targets)
            loss += self.branch_weight * self.base_loss(output["aux_logits_branch"], targets)
            for aux in output.get("rgb_aux", []):
                loss += self.aux_weight * self.base_loss(aux, targets)
            for aux in output.get("aux_aux", []):
                loss += self.aux_weight * self.base_loss(aux, targets)
            return loss
        return self.base_loss(output, targets)


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
