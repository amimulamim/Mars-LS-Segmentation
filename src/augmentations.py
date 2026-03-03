"""
Albumentations-based augmentation pipelines.

Provides two training variants:
  - ``build_train_transforms(img_size)``        — standard (default)
  - ``build_train_transforms_strong(img_size)``  — aggressive (ablation)

The *strong* variant adds elastic/grid distortion and coarse dropout,
wider affine range, and stronger photometric jitter.
"""

import cv2
import albumentations as A


def build_train_transforms(img_size):
    """Return (geo_aug, rgb_photo_aug) for training — standard variant."""
    geo_aug = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.90, 1.10),
            rotate=(-20, 20),
            p=0.5,
            mode=cv2.BORDER_REFLECT_101,
        ),
    ])
    rgb_photo = A.Compose([
        A.GaussianBlur(p=0.15),
        A.RandomBrightnessContrast(p=0.3),
    ])
    return geo_aug, rgb_photo


def build_train_transforms_strong(img_size):
    """Return (geo_aug, rgb_photo_aug) — strong augmentation variant.

    Adds ElasticTransform, GridDistortion, CoarseDropout, and widens
    the affine range.  Best used with longer training schedules or
    smaller datasets to combat overfitting.
    """
    geo_aug = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
            scale=(0.85, 1.15),
            rotate=(-30, 30),
            p=0.6,
            mode=cv2.BORDER_REFLECT_101,
        ),
        A.ElasticTransform(alpha=30, sigma=5, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),
        A.CoarseDropout(
            num_holes_range=(2, 6),
            hole_height_range=(8, 24),
            hole_width_range=(8, 24),
            fill=0,
            p=0.3,
        ),
    ])
    rgb_photo = A.Compose([
        A.GaussianBlur(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                   contrast_limit=0.15, p=0.4),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
    ])
    return geo_aug, rgb_photo


def build_val_transforms(img_size):
    """Return val_aug for validation / inference."""
    return A.Compose([A.Resize(img_size, img_size)])
