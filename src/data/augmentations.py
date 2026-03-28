"""
Data Augmentation Pipelines for Lane Segmentation.

Uses Albumentations for efficient, jointly-applied image+mask transforms.
All geometric transforms are applied identically to both image and mask.
Photometric transforms only affect the image.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    img_size: tuple[int, int] = (360, 640),
) -> A.Compose:
    """
    Training augmentation pipeline.

    Includes both geometric and photometric augmentations for robustness
    to varying road conditions, lighting, and viewpoints.

    Args:
        img_size: Target (H, W) after augmentation.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        # --- Geometric Augmentations ---
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.85, 1.15),
            rotate=(-10, 10),
            p=0.5,
        ),
        A.Perspective(
            scale=(0.02, 0.05),
            p=0.3,
        ),
        A.Resize(height=img_size[0], width=img_size[1]),

        # --- Photometric Augmentations (image only) ---
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
            p=0.5,
        ),
        A.GaussianBlur(
            blur_limit=(3, 7),
            p=0.2,
        ),
        A.GaussNoise(
            std_range=(0.02, 0.1),
            p=0.2,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3,
        ),

        # --- Normalization ---
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(
    img_size: tuple[int, int] = (360, 640),
) -> A.Compose:
    """
    Validation/test augmentation pipeline.

    Only resize and normalize — no randomness for deterministic evaluation.

    Args:
        img_size: Target (H, W).

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
