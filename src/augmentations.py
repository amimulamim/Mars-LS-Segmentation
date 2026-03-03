"""
Albumentations-based augmentation pipelines.
"""

import cv2
import albumentations as A


def build_train_transforms(img_size):
    """Return (geo_aug, rgb_photo_aug) for training."""
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


def build_val_transforms(img_size):
    """Return val_aug for validation / inference."""
    return A.Compose([A.Resize(img_size, img_size)])
