"""
Quick visual check: renders a few masks in human-visible colors
so you can confirm the conversion worked correctly.

Saves output to dataset/mask_preview/ as colorized PNGs.

Usage:
    python scripts/verify_masks.py
"""

import cv2
import numpy as np
from pathlib import Path


def verify_masks(
    image_dir: str = "dataset/images/train",
    mask_dir: str = "dataset/masks/train",
    output_dir: str = "dataset/mask_preview",
    num_samples: int = 5,
):
    img_dir = Path(image_dir)
    msk_dir = Path(mask_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(img_dir.iterdir())[:num_samples]

    for img_file in image_files:
        # Load original image
        image = cv2.imread(str(img_file))
        if image is None:
            continue

        # Load mask
        mask_file = msk_dir / (img_file.stem + ".png")
        if not mask_file.exists():
            continue
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        # Print raw stats so you can see the values
        unique_vals = np.unique(mask)
        dashed_count = np.sum(mask == 1)
        solid_count = np.sum(mask == 2)
        total = mask.size
        print(f"{img_file.name}:")
        print(f"  Unique pixel values in mask: {unique_vals}")
        print(f"  Dashed (1): {dashed_count} pixels ({dashed_count/total*100:.2f}%)")
        print(f"  Solid  (2): {solid_count} pixels ({solid_count/total*100:.2f}%)")

        # Create a BRIGHT colorized version of the mask for human viewing
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        color_mask[mask == 1] = (0, 255, 255)    # Dashed = bright yellow
        color_mask[mask == 2] = (0, 255, 0)      # Solid  = bright green

        # Resize mask to match image if needed
        if image.shape[:2] != mask.shape:
            color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        # Overlay on the original image
        overlay = image.copy()
        lane_pixels = mask_resized > 0
        overlay[lane_pixels] = cv2.addWeighted(
            image[lane_pixels], 0.5,
            color_mask[lane_pixels], 0.5, 0
        )

        # Add legend text
        cv2.putText(overlay, "Yellow = Dashed lane", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(overlay, "Green  = Solid lane", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Side by side: original | bright mask | overlay
        bright_mask_full = np.zeros_like(image)
        if image.shape[:2] != color_mask.shape[:2]:
            color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]))
        bright_mask_full[:] = color_mask if color_mask.shape == bright_mask_full.shape else 0

        canvas = np.hstack([image, bright_mask_full, overlay])

        out_path = out_dir / f"preview_{img_file.stem}.jpg"
        cv2.imwrite(str(out_path), canvas)
        print(f"  → Saved preview: {out_path}\n")

    print(f"Done! Check the previews in: {out_dir}")
    print("  Left   = Original image")
    print("  Middle = Mask (bright colors)")
    print("  Right  = Overlay on original")


if __name__ == "__main__":
    verify_masks()
