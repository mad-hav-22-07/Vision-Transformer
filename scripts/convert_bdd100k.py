"""
BDD100K Lane Annotations → Segmentation Masks Converter.

Converts BDD100K lane polyline annotations (JSON) into pixel-level
segmentation masks compatible with this project's training pipeline.

BDD100K lane annotations are stored as poly2d polylines with attributes:
    - laneDirection: "parallel" or "vertical"
    - laneStyle: "solid" or "dashed"
    - laneType: "road curb", "crosswalk", "double white", "double yellow", etc.

This script rasterizes those polylines into masks:
    0 = Background (no lane)
    1 = Dashed lane marking
    2 = Solid lane marking

Usage:
    1. Download BDD100K from https://bdd-data.berkeley.edu/
       - Images: bdd100k_images_100k.zip → extract to bdd100k/images/100k/
       - Labels: bdd100k_labels_release.zip → extract to bdd100k/labels/

    2. Run this script:
       python3 scripts/convert_bdd100k.py \\
           --bdd_images bdd100k/images/100k/train \\
           --bdd_labels bdd100k/labels/lane/masks/train \\
           --output_images dataset/images/train \\
           --output_masks dataset/masks/train

    OR, if you have JSON polyline annotations instead of masks:
       python3 scripts/convert_bdd100k.py \\
           --bdd_images bdd100k/images/100k/train \\
           --bdd_json bdd100k/labels/det_20/det_train.json \\
           --output_images dataset/images/train \\
           --output_masks dataset/masks/train
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


# BDD100K lane style → our class mapping
LANE_STYLE_MAP = {
    "dashed": 1,
    "solid": 2,
}

# Default line thickness for rasterizing polylines (in pixels)
LANE_THICKNESS = 8


def convert_from_json(
    bdd_images_dir: str,
    bdd_json_path: str,
    output_images_dir: str,
    output_masks_dir: str,
    img_size: tuple[int, int] = (720, 1280),
    max_images: int | None = None,
    lane_thickness: int = LANE_THICKNESS,
):
    """
    Convert BDD100K JSON polyline lane annotations to segmentation masks.

    The BDD100K det_train.json / det_val.json files contain frames with
    labels[].poly2d annotations for lane markings.

    Args:
        bdd_images_dir:   Path to BDD100K images (e.g., bdd100k/images/100k/train).
        bdd_json_path:    Path to BDD100K JSON label file.
        output_images_dir: Output directory for images.
        output_masks_dir:  Output directory for masks.
        img_size:         BDD100K image size (H, W). Default is 720x1280.
        max_images:       Limit number of images to convert (None = all).
        lane_thickness:   Line thickness for rasterizing lanes.
    """
    output_img_path = Path(output_images_dir)
    output_mask_path = Path(output_masks_dir)
    output_img_path.mkdir(parents=True, exist_ok=True)
    output_mask_path.mkdir(parents=True, exist_ok=True)

    print(f"[BDD100K] Loading JSON labels from: {bdd_json_path}")
    with open(bdd_json_path, "r") as f:
        bdd_data = json.load(f)

    # Filter frames that have lane annotations
    frames_with_lanes = []
    for frame in bdd_data:
        labels = frame.get("labels", [])
        has_lanes = any(
            "poly2d" in label and label.get("category", "") == "lane"
            for label in labels
        )
        if has_lanes:
            frames_with_lanes.append(frame)

    if max_images:
        frames_with_lanes = frames_with_lanes[:max_images]

    print(f"[BDD100K] Found {len(frames_with_lanes)} frames with lane annotations")

    converted = 0
    skipped = 0

    for frame in tqdm(frames_with_lanes, desc="Converting"):
        image_name = frame["name"]
        image_path = Path(bdd_images_dir) / image_name

        if not image_path.exists():
            skipped += 1
            continue

        # Create blank mask
        H, W = img_size
        mask = np.zeros((H, W), dtype=np.uint8)

        # Rasterize each lane polyline
        labels = frame.get("labels", [])
        for label in labels:
            if label.get("category") != "lane":
                continue

            poly2d_list = label.get("poly2d", [])
            attributes = label.get("attributes", {})
            lane_style = attributes.get("laneStyle", "solid")

            # Get class ID from lane style
            class_id = LANE_STYLE_MAP.get(lane_style, 2)  # Default to solid

            for poly2d in poly2d_list:
                vertices = poly2d.get("vertices", [])
                if len(vertices) < 2:
                    continue

                # Convert to integer pixel coordinates
                pts = np.array(vertices, dtype=np.int32)

                # Draw polyline on mask
                cv2.polylines(
                    mask,
                    [pts],
                    isClosed=False,
                    color=int(class_id),
                    thickness=lane_thickness,
                )

        # Check if mask has any lane pixels
        if mask.max() == 0:
            skipped += 1
            continue

        # Copy image and save mask
        output_idx = f"{converted:06d}"
        dest_image = output_img_path / f"{output_idx}.jpg"
        dest_mask = output_mask_path / f"{output_idx}.png"

        shutil.copy2(str(image_path), str(dest_image))
        cv2.imwrite(str(dest_mask), mask)

        converted += 1

    print(f"\n[BDD100K] Conversion complete!")
    print(f"  Converted: {converted} image-mask pairs")
    print(f"  Skipped:   {skipped} (no image or empty mask)")
    print(f"  Images:    {output_images_dir}")
    print(f"  Masks:     {output_masks_dir}")


def convert_from_bdd_masks(
    bdd_images_dir: str,
    bdd_masks_dir: str,
    output_images_dir: str,
    output_masks_dir: str,
    max_images: int | None = None,
):
    """
    Convert BDD100K pre-rendered lane segmentation masks.

    If you downloaded the lane segmentation subset of BDD100K, the masks
    may already be provided as images. This function remaps the BDD100K
    color-coded masks to our 3-class format.

    BDD100K lane seg masks use:
        - 0 = background
        - Non-zero pixel values represent different lane categories

    We remap to:
        - 0 = background
        - 1 = dashed
        - 2 = solid
    """
    output_img_path = Path(output_images_dir)
    output_mask_path = Path(output_masks_dir)
    output_img_path.mkdir(parents=True, exist_ok=True)
    output_mask_path.mkdir(parents=True, exist_ok=True)

    bdd_img_dir = Path(bdd_images_dir)
    bdd_msk_dir = Path(bdd_masks_dir)

    mask_files = sorted(bdd_msk_dir.glob("*.png"))
    if max_images:
        mask_files = mask_files[:max_images]

    print(f"[BDD100K] Found {len(mask_files)} mask files")

    converted = 0

    for mask_file in tqdm(mask_files, desc="Converting masks"):
        # Find corresponding image
        stem = mask_file.stem
        image_file = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = bdd_img_dir / f"{stem}{ext}"
            if candidate.exists():
                image_file = candidate
                break

        if image_file is None:
            continue

        # Load and remap mask
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # BDD100K masks: remap non-zero to our classes
        # Simple heuristic: any non-zero lane pixel → solid (2)
        # You may need to adjust this based on actual BDD100K encoding
        remapped = np.zeros_like(mask)
        remapped[mask > 0] = 2  # Default: solid lanes

        # Copy image and save remapped mask
        output_idx = f"{converted:06d}"
        shutil.copy2(str(image_file), output_img_path / f"{output_idx}.jpg")
        cv2.imwrite(str(output_mask_path / f"{output_idx}.png"), remapped)

        converted += 1

    print(f"\n[BDD100K] Converted {converted} image-mask pairs")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BDD100K lane annotations to segmentation masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From JSON polyline annotations:
  python3 scripts/convert_bdd100k.py \\
      --bdd_images bdd100k/images/100k/train \\
      --bdd_json bdd100k/labels/lane/polygons/lane_train.json \\
      --output_images dataset/images/train \\
      --output_masks dataset/masks/train

  # From pre-rendered BDD100K lane masks:
  python3 scripts/convert_bdd100k.py \\
      --bdd_images bdd100k/images/100k/train \\
      --bdd_masks bdd100k/labels/lane/masks/train \\
      --output_images dataset/images/train \\
      --output_masks dataset/masks/train

  # Limit to first 1000 images:
  python3 scripts/convert_bdd100k.py \\
      --bdd_images bdd100k/images/100k/train \\
      --bdd_json bdd100k/labels/lane/polygons/lane_train.json \\
      --output_images dataset/images/train \\
      --output_masks dataset/masks/train \\
      --max_images 1000
        """,
    )

    parser.add_argument("--bdd_images", type=str, required=True,
                       help="Path to BDD100K images directory")
    parser.add_argument("--bdd_json", type=str, default=None,
                       help="Path to BDD100K JSON label file (polyline annotations)")
    parser.add_argument("--bdd_masks", type=str, default=None,
                       help="Path to BDD100K pre-rendered lane mask directory")
    parser.add_argument("--output_images", type=str, required=True,
                       help="Output images directory")
    parser.add_argument("--output_masks", type=str, required=True,
                       help="Output masks directory")
    parser.add_argument("--max_images", type=int, default=None,
                       help="Max images to convert (None = all)")
    parser.add_argument("--lane_thickness", type=int, default=LANE_THICKNESS,
                       help=f"Lane line thickness in pixels (default: {LANE_THICKNESS})")

    args = parser.parse_args()

    if args.bdd_json:
        convert_from_json(
            bdd_images_dir=args.bdd_images,
            bdd_json_path=args.bdd_json,
            output_images_dir=args.output_images,
            output_masks_dir=args.output_masks,
            max_images=args.max_images,
            lane_thickness=args.lane_thickness,
        )
    elif args.bdd_masks:
        convert_from_bdd_masks(
            bdd_images_dir=args.bdd_images,
            bdd_masks_dir=args.bdd_masks,
            output_images_dir=args.output_images,
            output_masks_dir=args.output_masks,
            max_images=args.max_images,
        )
    else:
        print("Error: provide either --bdd_json or --bdd_masks")
        exit(1)


if __name__ == "__main__":
    main()
