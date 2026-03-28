"""
Visualization script for lane segmentation predictions.

Loads a trained model (or ONNX), runs inference on images, and creates
colorized overlays showing detected lane markings.

Usage:
    python scripts/visualize_predictions.py --checkpoint checkpoints/best_model.pth --image_dir dataset/images/val --output_dir predictions/
    python scripts/visualize_predictions.py --onnx lane_seg.onnx --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import ViTLaneSeg
from src.data.utils import (
    mask_to_color,
    overlay_mask_on_image,
    denormalize_image,
    LANE_CLASSES,
)


def predict_single(
    model: ViTLaneSeg,
    image_path: str,
    img_size: tuple[int, int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run prediction on a single image.

    Returns:
        Tuple of (original_image_bgr, predicted_mask).
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load: {image_path}")

    original = image.copy()

    # Preprocess
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size[1], img_size[0]))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    tensor = tensor.to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize mask to original image size
    h, w = original.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return original, mask


def visualize(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    show: bool = True,
    save_path: str | None = None,
):
    """Create and optionally save/show visualization."""
    # Create overlay
    overlay = overlay_mask_on_image(image, mask, alpha=alpha)

    # Create color mask
    color_mask = mask_to_color(mask)

    # Side-by-side comparison
    h, w = image.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, :w] = image
    canvas[:, w:2*w] = color_mask
    canvas[:, 2*w:] = overlay

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "Prediction", (w + 10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "Overlay", (2*w + 10, 30), font, 0.8, (255, 255, 255), 2)

    # Lane legend
    cv2.putText(canvas, "Dashed", (w + 10, h - 40), font, 0.6, (0, 255, 255), 2)
    cv2.putText(canvas, "Solid", (w + 10, h - 15), font, 0.6, (0, 255, 0), 2)

    # Count lane pixels
    dashed_pixels = np.sum(mask == 1)
    solid_pixels = np.sum(mask == 2)
    total_pixels = mask.size
    cv2.putText(
        canvas,
        f"Dashed: {dashed_pixels/total_pixels*100:.1f}%  Solid: {solid_pixels/total_pixels*100:.1f}%",
        (2*w + 10, h - 15), font, 0.5, (255, 255, 255), 1
    )

    if save_path:
        cv2.imwrite(str(save_path), canvas)
        print(f"Saved: {save_path}")

    if show:
        cv2.imshow("Lane Detection", canvas)
        cv2.waitKey(0)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize Lane Predictions")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay transparency")
    parser.add_argument("--no-show", action="store_true", default=True, help="Don't display images (default: True)")
    parser.add_argument("--show", action="store_true", help="Display images in window (requires GUI)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not args.checkpoint:
        print("Error: --checkpoint required")
        sys.exit(1)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    img_size = tuple(config["model"]["img_size"])

    model = ViTLaneSeg.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Collect images
    if args.image:
        image_paths = [Path(args.image)]
    elif args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = sorted([
            f for f in Path(args.image_dir).iterdir()
            if f.suffix.lower() in exts
        ])
    else:
        print("Error: provide --image or --image_dir")
        sys.exit(1)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(image_paths)} images...")

    for img_path in image_paths:
        print(f"  {img_path.name}...", end=" ")
        original, mask = predict_single(model, img_path, img_size, device)

        save_path = output_dir / f"pred_{img_path.name}"
        visualize(
            original, mask,
            alpha=args.alpha,
            show=args.show,
            save_path=str(save_path),
        )

    print(f"\nDone! {len(image_paths)} predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
