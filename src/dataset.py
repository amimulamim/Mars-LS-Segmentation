"""
PyTorch Dataset classes for training and inference.

Both datasets apply v4 per-image percentile normalization followed by
channel-wise z-score standardization.
"""

from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .config import BAND_INDICES, RGB_BANDS, AUX_BANDS, RGB_STAT_IDX, AUX_STAT_IDX
from .normalization import normalize_bands_per_image


class MarsSegDataset(Dataset):
    """Training / validation dataset.

    Reads 7-band GeoTIFF images, applies per-image percentile normalization,
    splits into RGB (3ch) and AUX (4ch), z-scores each, and returns tensors.
    """

    def __init__(
        self,
        img_paths,
        mask_paths,
        mean_all,
        std_all,
        geo_aug=None,
        rgb_photo_aug=None,
        val_aug=None,
        is_train=True,
    ):
        self.img_paths  = list(img_paths)
        self.mask_paths = list(mask_paths) if mask_paths is not None else None
        self.mean_all   = np.asarray(mean_all, dtype=np.float32)
        self.std_all    = np.asarray(std_all,  dtype=np.float32)
        self.geo_aug       = geo_aug
        self.rgb_photo_aug = rgb_photo_aug
        self.val_aug       = val_aug
        self.is_train      = is_train

    def __len__(self):
        return len(self.img_paths)

    # ── helpers ──────────────────────────────────────────────
    @staticmethod
    def _standardize(x_chw, mean_all, std_all, stat_idx):
        mean = mean_all[stat_idx][:, None, None]
        std  = std_all[stat_idx][:, None, None]
        return (x_chw - mean) / std

    # ── main ─────────────────────────────────────────────────
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        with rasterio.open(str(img_path)) as src:
            arr = src.read(BAND_INDICES).astype(np.float32)

        # v4: per-image percentile normalization
        arr = normalize_bands_per_image(arr)
        arr_hwc = np.transpose(arr, (1, 2, 0))

        mask = None
        if self.mask_paths is not None:
            with rasterio.open(str(self.mask_paths[idx])) as src:
                mask = src.read(1).astype(np.uint8)
            mask = (mask > 0).astype(np.float32)

        # augmentations
        if self.is_train:
            if self.geo_aug is not None:
                aug = self.geo_aug(image=arr_hwc, mask=mask)
                arr_hwc, mask = aug["image"], aug["mask"]
            if self.rgb_photo_aug is not None:
                rgb = arr_hwc[..., :3]
                aux = arr_hwc[..., 3:]
                rgb = self.rgb_photo_aug(image=rgb)["image"]
                arr_hwc = np.concatenate([rgb, aux], axis=2)
        else:
            if self.val_aug is not None:
                if mask is not None:
                    aug = self.val_aug(image=arr_hwc, mask=mask)
                    arr_hwc, mask = aug["image"], aug["mask"]
                else:
                    arr_hwc = self.val_aug(image=arr_hwc)["image"]

        arr_chw = np.transpose(arr_hwc, (2, 0, 1))
        rgb = self._standardize(arr_chw[:3], self.mean_all, self.std_all, RGB_STAT_IDX)
        aux = self._standardize(arr_chw[3:], self.mean_all, self.std_all, AUX_STAT_IDX)

        rgb_t = torch.from_numpy(rgb).float()
        aux_t = torch.from_numpy(aux).float()

        if mask is not None:
            return rgb_t, aux_t, torch.from_numpy(mask).float().unsqueeze(0)
        else:
            return rgb_t, aux_t, Path(img_path).name


class InferenceDataset(Dataset):
    """Test-time dataset — reads .tif (or .npy fallback), applies per-image
    percentile normalization + z-score standardization, returns (rgb, aux, filename).
    """

    def __init__(self, image_dir, img_size, means, stds):
        image_dir = Path(image_dir)
        self.paths = sorted(image_dir.glob("*.tif")) + sorted(image_dir.glob("*.TIF"))
        if len(self.paths) == 0:
            self.paths = sorted(image_dir.glob("*.npy"))
        self.img_size = img_size
        self.means = np.asarray(means, dtype=np.float32)
        self.stds  = np.asarray(stds,  dtype=np.float32)
        print(f"[InferenceDataset] Found {len(self.paths)} files in {image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        name = p.name

        # read image
        try:
            with rasterio.open(str(p)) as src:
                arr = src.read(BAND_INDICES).astype(np.float32)
        except Exception:
            arr = np.load(str(p)).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[None]

        # v4 per-image percentile normalization
        arr = normalize_bands_per_image(arr)

        # resize if needed
        _, H, W = arr.shape
        if H != self.img_size or W != self.img_size:
            arr = np.stack([
                cv2.resize(arr[c], (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)
                for c in range(arr.shape[0])
            ], axis=0)

        rgb = MarsSegDataset._standardize(arr[:3], self.means, self.stds, RGB_STAT_IDX)
        aux = MarsSegDataset._standardize(arr[3:], self.means, self.stds, AUX_STAT_IDX)
        return torch.from_numpy(rgb).float(), torch.from_numpy(aux).float(), name
