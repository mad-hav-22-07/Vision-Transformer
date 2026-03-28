"""
ONNX Export Script for ViT-LaneSeg.

Converts a trained PyTorch model to ONNX format for TensorRT conversion.
Includes numerical verification against PyTorch output.

Usage:
    python -m src.export_onnx --checkpoint checkpoints/best_model.pth --output lane_seg.onnx
    python -m src.export_onnx --checkpoint checkpoints/best_model.pth --verify
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import ViTLaneSeg


def export_to_onnx(
    model: ViTLaneSeg,
    output_path: str,
    img_size: tuple[int, int] = (360, 640),
    opset_version: int = 17,
    dynamic_batch: bool = True,
    simplify: bool = True,
) -> str:
    """
    Export PyTorch model to ONNX format.

    Args:
        model:         Trained ViTLaneSeg model.
        output_path:   Path to save .onnx file.
        img_size:      Input image size (H, W).
        opset_version: ONNX opset version (17+ recommended for transformers).
        dynamic_batch: Enable dynamic batch dimension.
        simplify:      Run onnx-simplifier to optimize the graph.

    Returns:
        Path to the saved ONNX model.
    """
    model.eval()
    device = next(model.parameters()).device

    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], device=device)

    # Define dynamic axes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch_size"},
            "lane_mask": {0: "batch_size"},
        }

    # Export
    print(f"[Export] Exporting to ONNX: {output_path}")
    print(f"  Input shape:  (batch, 3, {img_size[0]}, {img_size[1]})")
    print(f"  Opset version: {opset_version}")
    print(f"  Dynamic batch: {dynamic_batch}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["lane_mask"],
        dynamic_axes=dynamic_axes,
    )

    print(f"[Export] ONNX model saved to: {output_path}")

    # Validate ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[Export] ONNX model validation: PASSED ✓")
    except ImportError:
        print("[Export] onnx package not installed — skipping validation")
    except Exception as e:
        print(f"[Export] ONNX model validation FAILED: {e}")
        raise

    # Simplify (optional)
    if simplify:
        try:
            import onnxsim
            print("[Export] Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(simplified, output_path)
                print("[Export] Simplification: PASSED ✓")
            else:
                print("[Export] Simplification check failed — keeping original")
        except ImportError:
            print("[Export] onnxsim not installed — skipping simplification")

    return output_path


def verify_onnx(
    model: ViTLaneSeg,
    onnx_path: str,
    img_size: tuple[int, int] = (360, 640),
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Verify ONNX model produces same output as PyTorch model.

    Args:
        model:     PyTorch model.
        onnx_path: Path to ONNX model.
        img_size:  Input size (H, W).
        rtol:      Relative tolerance for comparison.
        atol:      Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance.
    """
    import onnxruntime as ort

    model.eval()
    device = next(model.parameters()).device

    # Generate test input
    test_input = torch.randn(1, 3, img_size[0], img_size[1], device=device)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = model(test_input).cpu().numpy()

    # ONNX Runtime inference
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {"image": test_input.cpu().numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # Compare
    max_diff = np.max(np.abs(pytorch_output - ort_output))
    mean_diff = np.mean(np.abs(pytorch_output - ort_output))

    print(f"\n[Verify] PyTorch vs ONNX Runtime comparison:")
    print(f"  Max absolute diff:  {max_diff:.8f}")
    print(f"  Mean absolute diff: {mean_diff:.8f}")

    match = np.allclose(pytorch_output, ort_output, rtol=rtol, atol=atol)

    if match:
        print(f"  Result: MATCH ✓ (rtol={rtol}, atol={atol})")
    else:
        print(f"  Result: MISMATCH ✗ (rtol={rtol}, atol={atol})")
        print(f"  Max diff exceeds tolerance!")

    return match


def main():
    parser = argparse.ArgumentParser(description="Export ViT-LaneSeg to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="lane_seg.onnx",
                       help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=17,
                       help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true",
                       help="Skip ONNX graph simplification")
    parser.add_argument("--verify", action="store_true",
                       help="Verify ONNX output matches PyTorch")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load checkpoint
    print(f"[Export] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Create model
    model = ViTLaneSeg.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[Export] Model loaded — {model.get_param_count()['total_M']} parameters")

    # Export
    img_size = tuple(config["model"]["img_size"])
    export_to_onnx(
        model,
        args.output,
        img_size=img_size,
        opset_version=args.opset,
        simplify=not args.no_simplify,
    )

    # Verify
    if args.verify:
        verify_onnx(model, args.output, img_size=img_size)


if __name__ == "__main__":
    main()
