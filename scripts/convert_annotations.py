"""
Dataset Annotation Converter.

Converts various annotation formats into the segmentation mask format
expected by the training pipeline:
    - Single-channel PNG
    - Pixel value 0 = background
    - Pixel value 1 = dashed lane
    - Pixel value 2 = solid lane

Supported input formats:
    - COCO JSON (with polygon annotations)
    - CVAT XML
    - Polyline text files
    - Color-coded masks (custom colormap)

Usage:
    python scripts/convert_annotations.py --format coco --input annotations.json --image_dir images/ --output masks/
    python scripts/convert_annotations.py --format color_mask --input color_masks/ --output masks/ --colormap colormap.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def convert_coco_to_masks(
    annotation_file: str,
    image_dir: str,
    output_dir: str,
    class_mapping: dict | None = None,
):
    """
    Convert COCO-format polygon annotations to segmentation masks.

    Args:
        annotation_file: Path to COCO JSON file.
        image_dir:       Path to images directory.
        output_dir:      Path to save masks.
        class_mapping:   Dict mapping COCO category_id → mask class index.
                        Default: {1: 1 (dashed), 2: 2 (solid)}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(annotation_file, "r") as f:
        coco = json.load(f)

    # Default class mapping
    if class_mapping is None:
        categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
        print(f"[Convert] Categories found: {categories}")
        class_mapping = {}
        for cat_id, name in categories.items():
            if "dashed" in name.lower():
                class_mapping[cat_id] = 1
            elif "solid" in name.lower():
                class_mapping[cat_id] = 2
            else:
                class_mapping[cat_id] = 0
        print(f"[Convert] Auto-mapping: {class_mapping}")

    # Build image lookup
    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    anns_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    print(f"[Convert] Processing {len(images)} images...")

    for img_id, img_info in images.items():
        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        if img_id in anns_by_image:
            for ann in anns_by_image[img_id]:
                cat_id = ann["category_id"]
                class_idx = class_mapping.get(cat_id, 0)

                if class_idx == 0:
                    continue

                # Draw polygon(s)
                if "segmentation" in ann:
                    for seg in ann["segmentation"]:
                        pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [pts], class_idx)

        # Save mask
        filename = Path(img_info["file_name"]).stem + ".png"
        cv2.imwrite(str(output_path / filename), mask)

    print(f"[Convert] Saved {len(images)} masks to {output_dir}")


def convert_color_masks(
    input_dir: str,
    output_dir: str,
    colormap: dict | None = None,
):
    """
    Convert color-coded masks to class-index masks.

    Args:
        input_dir:  Directory with color-coded mask images.
        output_dir: Directory to save class-index masks.
        colormap:   Dict mapping (R, G, B) → class_index.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if colormap is None:
        # Default colormap (customize as needed)
        colormap = {
            (0, 0, 0): 0,        # Black → background
            (255, 255, 0): 1,    # Yellow → dashed
            (0, 255, 0): 2,      # Green → solid
            (255, 0, 0): 1,      # Red → dashed (alternative)
            (0, 0, 255): 2,      # Blue → solid (alternative)
        }

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    mask_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in exts
    ])

    print(f"[Convert] Converting {len(mask_files)} color masks...")

    for mask_file in mask_files:
        color_mask = cv2.imread(str(mask_file))
        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)

        h, w = color_mask.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)

        for color, class_idx in colormap.items():
            match = np.all(color_mask == color, axis=2)
            class_mask[match] = class_idx

        output_file = output_path / mask_file.with_suffix(".png").name
        cv2.imwrite(str(output_file), class_mask)

    print(f"[Convert] Saved {len(mask_files)} masks to {output_dir}")


def convert_polylines(
    annotation_dir: str,
    image_dir: str,
    output_dir: str,
    line_thickness: int = 5,
):
    """
    Convert polyline annotations to segmentation masks.

    Expects text files with format:
        class_id x1 y1 x2 y2 x3 y3 ...
    One line per lane annotation.

    Args:
        annotation_dir:  Directory with .txt annotation files.
        image_dir:       Directory with images (for resolution).
        output_dir:      Directory to save masks.
        line_thickness:  Thickness of drawn lane lines.
    """
    ann_path = Path(annotation_dir)
    img_path = Path(image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(ann_path.glob("*.txt"))
    print(f"[Convert] Converting {len(txt_files)} polyline annotations...")

    for txt_file in txt_files:
        # Find corresponding image
        stem = txt_file.stem
        image_file = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = img_path / (stem + ext)
            if candidate.exists():
                image_file = candidate
                break

        if image_file is None:
            print(f"  Warning: no image for {txt_file.name}")
            continue

        img = cv2.imread(str(image_file))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                points = np.array(coords).reshape(-1, 2).astype(np.int32)

                cv2.polylines(mask, [points], isClosed=False,
                            color=int(class_id), thickness=line_thickness)

        output_file = output_path / (stem + ".png")
        cv2.imwrite(str(output_file), mask)

    print(f"[Convert] Saved {len(txt_files)} masks to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert annotations to masks")
    parser.add_argument("--format", type=str, required=True,
                       choices=["coco", "color_mask", "polyline"],
                       help="Input annotation format")
    parser.add_argument("--input", type=str, required=True,
                       help="Input file or directory")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="Image directory (required for coco/polyline)")
    parser.add_argument("--output", type=str, default="dataset/masks",
                       help="Output directory for masks")
    parser.add_argument("--line_thickness", type=int, default=5,
                       help="Lane line thickness for polyline conversion")
    args = parser.parse_args()

    if args.format == "coco":
        if not args.image_dir:
            print("Error: --image_dir required for COCO format")
            sys.exit(1)
        convert_coco_to_masks(args.input, args.image_dir, args.output)

    elif args.format == "color_mask":
        convert_color_masks(args.input, args.output)

    elif args.format == "polyline":
        if not args.image_dir:
            print("Error: --image_dir required for polyline format")
            sys.exit(1)
        convert_polylines(args.input, args.image_dir, args.output,
                         line_thickness=args.line_thickness)


if __name__ == "__main__":
    main()
