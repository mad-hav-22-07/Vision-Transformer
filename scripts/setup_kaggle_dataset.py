"""
Setup script for the "Road Lane Segmentation Train Test Split" Kaggle dataset.

Downloads are manual (requires Kaggle account). This script converts the
color-coded masks from the Kaggle dataset into the 0/1/2 class-index format
required by the ViT-LaneSeg training pipeline.

Kaggle dataset color mapping:
    Background     = [0,   0,   0]    → 0 (background)
    Roads          = [128, 0,   0]    → 0 (we treat road surface as background)
    Lane mark solid  = [0, 128, 0]    → 2 (solid lane)
    Lane mark dashed = [128, 128, 0]  → 1 (dashed lane)

Usage:
    1. Download from: https://www.kaggle.com/datasets/sovitrath/road-lane-segmentation-train-test-split
    2. Unzip to a folder (e.g., C:\\Users\\you\\Downloads\\road-lane-seg\\)
    3. Run:
       python scripts/setup_kaggle_dataset.py --input "C:\\Users\\you\\Downloads\\road-lane-seg"

    This will:
      - Copy images to dataset/images/train/ and dataset/images/val/
      - Convert color masks to class-index masks in dataset/masks/train/ and dataset/masks/val/
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


# =====================================================
# COLOR MAP: Kaggle RGB values → our class indices
# =====================================================
# The Kaggle masks use specific RGB colors per class.
# We map them to integer class IDs for training.
#
#   RGB (0, 0, 0)       → 0  (Background)
#   RGB (128, 0, 0)     → 0  (Road — merged with background since we only care about lanes)
#   RGB (0, 128, 0)     → 2  (Solid lane)
#   RGB (128, 128, 0)   → 1  (Dashed lane)
# =====================================================
COLOR_TO_CLASS = [
    ((0,   0,   0),   0),   # Background
    ((128, 0,   0),   0),   # Road → background
    ((0,   128, 0),   2),   # Solid lane
    ((128, 128, 0),   1),   # Dashed lane
]


def convert_color_mask_to_class_mask(color_mask_path: str) -> np.ndarray:
    """
    Convert a single color-coded mask PNG to a class-index mask.

    Args:
        color_mask_path: Path to the color-coded mask image.

    Returns:
        Single-channel mask (H, W) with values {0, 1, 2}.
    """
    # Load as RGB (Kaggle masks are stored as RGB PNGs)
    mask_bgr = cv2.imread(str(color_mask_path), cv2.IMREAD_COLOR)
    if mask_bgr is None:
        raise FileNotFoundError(f"Cannot read mask: {color_mask_path}")

    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = mask_rgb.shape

    # Start with all-background
    class_mask = np.zeros((h, w), dtype=np.uint8)

    # Map each color to its class index
    for color_rgb, class_id in COLOR_TO_CLASS:
        # Find pixels matching this color (with a small tolerance for JPEG artifacts)
        color_arr = np.array(color_rgb, dtype=np.uint8)
        diff = np.abs(mask_rgb.astype(np.int16) - color_arr.astype(np.int16))
        match = np.all(diff < 30, axis=2)  # tolerance of 30 per channel
        class_mask[match] = class_id

    return class_mask


def find_dataset_structure(input_dir: Path) -> dict:
    """
    Auto-detect the folder structure of the Kaggle download.

    The Kaggle dataset can have various structures depending on how
    it's unzipped. This function searches for image/mask folders.

    Returns:
        Dict with 'train_images', 'train_masks', 'val_images', 'val_masks' paths.
    """
    # Common folder name patterns used in this Kaggle dataset
    possible_structures = [
        # Structure 1: train_images/, train_masks/, val_images/, val_masks/
        {
            "train_images": ["train_images", "train/images"],
            "train_masks": ["train_masks", "train/masks"],
            "val_images": ["val_images", "valid_images", "test_images", "valid/images", "test/images"],
            "val_masks": ["val_masks", "valid_masks", "test_masks", "valid/masks", "test/masks"],
        },
    ]

    result = {}

    # Search recursively for folders containing images
    all_dirs = [d for d in input_dir.rglob("*") if d.is_dir()]
    all_dir_names = {d.name.lower(): d for d in all_dirs}

    # Also check the immediate children
    for child in input_dir.iterdir():
        if child.is_dir():
            all_dir_names[child.name.lower()] = child

    for struct in possible_structures:
        for key, candidates in struct.items():
            for candidate in candidates:
                candidate_lower = candidate.lower().replace("/", "")
                # Check direct path
                direct = input_dir / candidate
                if direct.exists():
                    result[key] = direct
                    break
                # Check by folder name
                if candidate_lower in all_dir_names:
                    result[key] = all_dir_names[candidate_lower]
                    break

    # Fallback: if we can't find the exact structure, look for any folders
    # containing images and masks
    if len(result) < 4:
        print("[Setup] Could not auto-detect standard folder structure.")
        print(f"[Setup] Found directories: {[d.name for d in input_dir.iterdir() if d.is_dir()]}")
        print("[Setup] Searching for image and mask files recursively...")

        # Find all image files
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        all_images = list(input_dir.rglob("*"))
        all_images = [f for f in all_images if f.suffix.lower() in img_extensions]

        if all_images:
            # Group by parent directory
            from collections import defaultdict
            by_parent = defaultdict(list)
            for f in all_images:
                by_parent[f.parent].append(f)

            print(f"[Setup] Found images in {len(by_parent)} directories:")
            for parent, files in sorted(by_parent.items()):
                print(f"  {parent.relative_to(input_dir)}: {len(files)} files")

    return result


def setup_dataset(
    input_dir: str,
    output_dir: str = "dataset",
    max_images: int = 0,
):
    """
    Process the Kaggle dataset and set up the training directory.

    Args:
        input_dir:   Path to unzipped Kaggle dataset.
        output_dir:  Output dataset directory (default: 'dataset').
        max_images:  Max images to use (0 = all).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {input_path}")
        sys.exit(1)

    print(f"[Setup] Input directory: {input_path}")
    print(f"[Setup] Output directory: {output_path}")
    print()

    # Show what's in the input directory
    print("[Setup] Contents of input directory:")
    for item in sorted(input_path.iterdir()):
        if item.is_dir():
            num_files = len(list(item.iterdir()))
            print(f"  📁 {item.name}/ ({num_files} items)")
        else:
            size_kb = item.stat().st_size / 1024
            print(f"  📄 {item.name} ({size_kb:.0f} KB)")
    print()

    # Detect structure
    paths = find_dataset_structure(input_path)

    if len(paths) < 4:
        print("\n[Setup] Could not find all 4 required folders automatically.")
        print("[Setup] Please provide paths manually using:")
        print("  --train_images <path> --train_masks <path> --val_images <path> --val_masks <path>")
        print("\nOr reorganize your download into:")
        print("  your_folder/")
        print("  ├── train_images/")
        print("  ├── train_masks/")
        print("  ├── val_images/ (or valid_images/)")
        print("  └── val_masks/ (or valid_masks/)")
        sys.exit(1)

    print("[Setup] Detected folder structure:")
    for key, path in paths.items():
        num_files = len(list(path.iterdir()))
        print(f"  {key}: {path} ({num_files} files)")
    print()

    # Create output directories
    for subdir in ["images/train", "images/val", "masks/train", "masks/val"]:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)

    # Process training set
    _process_split(
        image_dir=paths["train_images"],
        mask_dir=paths["train_masks"],
        out_image_dir=output_path / "images" / "train",
        out_mask_dir=output_path / "masks" / "train",
        split_name="Training",
        max_images=max_images,
    )

    # Process validation set
    _process_split(
        image_dir=paths["val_images"],
        mask_dir=paths["val_masks"],
        out_image_dir=output_path / "images" / "val",
        out_mask_dir=output_path / "masks" / "val",
        split_name="Validation",
        max_images=0,  # Use all validation images
    )

    print("\n" + "=" * 60)
    print("✅ Dataset setup complete!")
    print("=" * 60)

    # Print stats
    train_count = len(list((output_path / "images" / "train").iterdir()))
    val_count = len(list((output_path / "images" / "val").iterdir()))
    print(f"  Training images:   {train_count}")
    print(f"  Validation images: {val_count}")
    print(f"  Total:             {train_count + val_count}")
    print()
    print("You can now start training with:")
    print("  python -m src.train --config configs/train_config.yaml")


def _process_split(
    image_dir: Path,
    mask_dir: Path,
    out_image_dir: Path,
    out_mask_dir: Path,
    split_name: str,
    max_images: int = 0,
):
    """Process one split (train or val)."""
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # Find all images
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in img_extensions
    ])

    if max_images > 0:
        image_files = image_files[:max_images]

    print(f"[{split_name}] Processing {len(image_files)} images...")

    converted = 0
    skipped = 0
    has_lanes = 0

    for img_file in image_files:
        # Find corresponding mask
        mask_file = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            candidate = mask_dir / (img_file.stem + ext)
            if candidate.exists():
                mask_file = candidate
                break

        if mask_file is None:
            skipped += 1
            continue

        # Convert the color mask to class indices
        try:
            class_mask = convert_color_mask_to_class_mask(mask_file)
        except Exception as e:
            print(f"  WARNING: Failed to convert {mask_file.name}: {e}")
            skipped += 1
            continue

        # Check if this image has any lane pixels at all
        lane_pixels = np.sum(class_mask > 0)
        if lane_pixels == 0:
            # Skip images with no lanes — they don't help the model learn lanes
            skipped += 1
            continue

        has_lanes += 1

        # Copy image to output
        out_img = out_image_dir / img_file.name
        shutil.copy2(img_file, out_img)

        # Save class-index mask as PNG
        out_mask = out_mask_dir / (img_file.stem + ".png")
        cv2.imwrite(str(out_mask), class_mask)

        converted += 1

        if converted % 50 == 0:
            print(f"  [{split_name}] Processed {converted}/{len(image_files)}...")

    print(f"  [{split_name}] Done: {converted} converted, {skipped} skipped (no lanes or missing masks)")
    print(f"  [{split_name}] Images with lane markings: {has_lanes}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup Kaggle Road Lane Segmentation dataset for ViT-LaneSeg",
        epilog="Download from: https://www.kaggle.com/datasets/sovitrath/road-lane-segmentation-train-test-split"
    )
    parser.add_argument("--input", type=str, required=True,
                       help="Path to unzipped Kaggle dataset directory")
    parser.add_argument("--output", type=str, default="dataset",
                       help="Output directory (default: dataset)")
    parser.add_argument("--max_images", type=int, default=0,
                       help="Max training images (0 = all)")

    # Optional manual path overrides
    parser.add_argument("--train_images", type=str, default=None)
    parser.add_argument("--train_masks", type=str, default=None)
    parser.add_argument("--val_images", type=str, default=None)
    parser.add_argument("--val_masks", type=str, default=None)

    args = parser.parse_args()

    setup_dataset(
        input_dir=args.input,
        output_dir=args.output,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
