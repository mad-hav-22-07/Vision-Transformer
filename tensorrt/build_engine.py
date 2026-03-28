"""
TensorRT Engine Builder.

Converts an ONNX model to a TensorRT engine optimized for the target GPU.
Supports FP16 and INT8 precision modes.

Usage:
    python tensorrt/build_engine.py --onnx lane_seg.onnx --output lane_seg_fp16.engine --fp16
    python tensorrt/build_engine.py --onnx lane_seg.onnx --output lane_seg_int8.engine --int8 --calib_dir dataset/images/train
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("[WARNING] TensorRT/PyCUDA not installed. This script requires:")
    print("  - tensorrt (pip install tensorrt)")
    print("  - pycuda (pip install pycuda)")
    print("  These are typically pre-installed on Jetson Orin via JetPack.")


TRT_LOGGER = trt.Logger(trt.Logger.INFO) if TRT_AVAILABLE else None


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    int8_calibrator=None,
    max_workspace_mb: int = 2048,
    max_batch_size: int = 1,
    min_batch_size: int = 1,
    opt_batch_size: int = 1,
    verbose: bool = False,
) -> str:
    """
    Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path:        Path to input ONNX model.
        engine_path:      Path to save output TRT engine.
        fp16:             Enable FP16 precision.
        int8:             Enable INT8 precision (requires calibrator).
        int8_calibrator:  INT8 calibrator instance.
        max_workspace_mb: Maximum GPU workspace in MB.
        max_batch_size:   Maximum batch size for dynamic shapes.
        min_batch_size:   Minimum batch size for dynamic shapes.
        opt_batch_size:   Optimal batch size for dynamic shapes.
        verbose:          Enable verbose logging.

    Returns:
        Path to the saved engine file.
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not installed. Install via JetPack on Jetson Orin.")

    if verbose:
        TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE

    print(f"[TRT] Building engine from: {onnx_path}")
    print(f"  FP16: {fp16}")
    print(f"  INT8: {int8}")
    print(f"  Workspace: {max_workspace_mb} MB")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print("[TRT] Parsing ONNX model...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Error {i}: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    print(f"[TRT] Network: {network.num_inputs} inputs, {network.num_outputs} outputs")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input  {i}: {inp.name}, shape={inp.shape}, dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output {i}: {out.name}, shape={out.shape}, dtype={out.dtype}")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, max_workspace_mb * (1 << 20)
    )

    # Set precision
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[TRT] FP16 mode enabled")
    elif fp16:
        print("[TRT] WARNING: FP16 not supported on this platform")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if int8_calibrator:
            config.int8_calibrator = int8_calibrator
        print("[TRT] INT8 mode enabled")
    elif int8:
        print("[TRT] WARNING: INT8 not supported on this platform")

    # Dynamic shapes (for variable batch size)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)

        # Replace dynamic dims (-1) with specific values
        min_shape = shape.copy()
        opt_shape = shape.copy()
        max_shape = shape.copy()

        if shape[0] == -1:  # Dynamic batch
            min_shape[0] = min_batch_size
            opt_shape[0] = opt_batch_size
            max_shape[0] = max_batch_size

        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)

    config.add_optimization_profile(profile)

    # Build engine
    print("[TRT] Building engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    engine_size_mb = os.path.getsize(engine_path) / (1 << 20)
    print(f"[TRT] Engine saved to: {engine_path} ({engine_size_mb:.1f} MB)")

    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT Engine")
    parser.add_argument("--onnx", type=str, required=True,
                       help="Path to ONNX model")
    parser.add_argument("--output", type=str, default="lane_seg.engine",
                       help="Output engine path")
    parser.add_argument("--fp16", action="store_true",
                       help="Enable FP16 precision")
    parser.add_argument("--int8", action="store_true",
                       help="Enable INT8 precision")
    parser.add_argument("--calib_dir", type=str, default=None,
                       help="Calibration image dir for INT8")
    parser.add_argument("--workspace", type=int, default=2048,
                       help="Max workspace in MB")
    parser.add_argument("--max_batch", type=int, default=1,
                       help="Max batch size")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    calibrator = None
    if args.int8 and args.calib_dir:
        from calibration import Int8Calibrator
        calibrator = Int8Calibrator(
            image_dir=args.calib_dir,
            cache_file="calibration.cache",
            batch_size=8,
            img_size=(360, 640),
        )

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.output,
        fp16=args.fp16,
        int8=args.int8,
        int8_calibrator=calibrator,
        max_workspace_mb=args.workspace,
        max_batch_size=args.max_batch,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
