"""
Lane Segmentation Dataset.

Loads image-mask pairs for training and validation.

Expected directory structure:
    dataset/
    ├── images/
    │   ├── train/
    │   │   ├── 000001.jpg
    │   │   ├── 000002.jpg
    │   │   └── ...
    │   └── val/
    │       ├── 000001.jpg
    │       └── ...
    └── masks/
        ├── train/
        │   ├── 000001.png     # Pixel values: 0=bg, 1=dashed, 2=solid
        │   └── ...
        └── val/
            ├── 000001.png
            └── ...

Mask encoding:
    0 = Background (non-lane pixels)
    1 = Dashed lane marking
    2 = Solid lane marking
"""

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LaneSegmentationDataset(Dataset):
    """
    PyTorch Dataset for lane segmentation.

    Loads images and their corresponding segmentation masks, applies
    transforms (augmentations), and returns tensors suitable for training.

    Args:
        image_dir:   Path to the directory containing images.
        mask_dir:    Path to the directory containing segmentation masks.
        transform:   Albumentations transform pipeline (optional).
        img_size:    Target image size (H, W) for resizing.
        num_classes: Number of segmentation classes.
    """

    # Supported image extensions
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Callable | None = None,
        img_size: tuple[int, int] = (360, 640),
        num_classes: int = 3,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.img_size = img_size
        self.num_classes = num_classes

        # Discover image files
        self.image_files = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in self.IMG_EXTENSIONS
        ])

        if len(self.image_files) == 0:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}. "
                f"Supported formats: {self.IMG_EXTENSIONS}"
            )

        # Verify corresponding masks exist
        self.mask_files = []
        for img_file in self.image_files:
            # Try .png mask first, then same extension
            mask_file = self.mask_dir / img_file.with_suffix(".png").name
            if not mask_file.exists():
                mask_file = self.mask_dir / img_file.name
            if not mask_file.exists():
                raise FileNotFoundError(
                    f"Mask not found for image {img_file.name}. "
                    f"Expected at {mask_file}"
                )
            self.mask_files.append(mask_file)

        print(f"[LaneDataset] Loaded {len(self)} image-mask pairs from {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Float tensor (3, H, W) normalized to [0, 1] then ImageNet standardized.
            mask:  Long tensor (H, W) with class indices {0, 1, 2}.
        """
        # Load image (BGR → RGB)
        image = cv2.imread(str(self.image_files[idx]))
        if image is None:
            raise RuntimeError(f"Failed to load image: {self.image_files[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale, single channel)
        mask = cv2.imread(str(self.mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {self.mask_files[idx]}")

        # Resize to target size
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]),
                          interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]),
                         interpolation=cv2.INTER_NEAREST)

        # Clamp mask values to valid range
        mask = np.clip(mask, 0, self.num_classes - 1)

        # Apply augmentations (Albumentations handles image+mask jointly)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensors if not already done by transforms
        if not isinstance(image, torch.Tensor):
            # HWC → CHW, uint8 → float32, normalize to [0, 1]
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return image, mask

    def compute_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights from the dataset.

        Scans all masks and computes pixel counts per class, then returns
        inverse-frequency weights normalized so they sum to num_classes.

        Returns:
            Class weights tensor of shape (num_classes,).
        """
        print("[LaneDataset] Computing class weights (scanning all masks)...")
        class_counts = np.zeros(self.num_classes, dtype=np.int64)

        for mask_file in self.mask_files:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]),
                             interpolation=cv2.INTER_NEAREST)
            mask = np.clip(mask, 0, self.num_classes - 1)

            for c in range(self.num_classes):
                class_counts[c] += np.sum(mask == c)

        # Inverse frequency weighting
        total = class_counts.sum()
        weights = total / (self.num_classes * class_counts + 1e-6)

        # Normalize so weights sum to num_classes
        weights = weights / weights.sum() * self.num_classes

        print(f"[LaneDataset] Class counts: {class_counts}")
        print(f"[LaneDataset] Class weights: {weights}")

        return torch.tensor(weights, dtype=torch.float32)
