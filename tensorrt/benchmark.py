"""
TensorRT Engine Benchmark.

Measures inference latency and throughput for the TensorRT lane
segmentation engine. Compares against PyTorch and ONNX Runtime
baselines when available.

Usage:
    python tensorrt/benchmark.py --engine lane_seg_fp16.engine
    python tensorrt/benchmark.py --engine lane_seg_fp16.engine --onnx lane_seg.onnx --warmup 50 --iterations 500
"""

import argparse
import time
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


def benchmark_tensorrt(
    engine_path: str,
    img_size: tuple[int, int] = (360, 640),
    batch_size: int = 1,
    warmup: int = 50,
    iterations: int = 300,
) -> dict:
    """
    Benchmark TensorRT engine inference speed.

    Returns:
        Dict with latency stats and throughput.
    """
    if not TRT_AVAILABLE:
        print("[Benchmark] TensorRT not available")
        return {}

    print(f"[Benchmark] TensorRT Engine: {engine_path}")

    # Load engine
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    input_shape = (batch_size, 3, img_size[0], img_size[1])
    output_shape = (batch_size, 3, img_size[0], img_size[1])  # num_classes=3

    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()

    # Set input shape for dynamic batch
    context.set_input_shape("image", input_shape)

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # Benchmark
    print(f"  Benchmarking ({iterations} iterations)...")
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)

    results = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies)),
        "fps": float(1000.0 / np.mean(latencies)),
    }

    print(f"\n  TensorRT Results (batch_size={batch_size}):")
    print(f"    Mean latency:   {results['mean_ms']:.2f} ms")
    print(f"    Median latency: {results['median_ms']:.2f} ms")
    print(f"    P95 latency:    {results['p95_ms']:.2f} ms")
    print(f"    P99 latency:    {results['p99_ms']:.2f} ms")
    print(f"    Throughput:     {results['fps']:.1f} FPS")

    return results


def benchmark_onnxruntime(
    onnx_path: str,
    img_size: tuple[int, int] = (360, 640),
    batch_size: int = 1,
    warmup: int = 20,
    iterations: int = 100,
) -> dict:
    """Benchmark ONNX Runtime inference speed."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[Benchmark] onnxruntime not installed — skipping")
        return {}

    print(f"\n[Benchmark] ONNX Runtime: {onnx_path}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)

    dummy_input = np.random.randn(batch_size, 3, img_size[0], img_size[1]).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        session.run(None, {"image": dummy_input})

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {"image": dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)

    results = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "fps": float(1000.0 / np.mean(latencies)),
    }

    print(f"  ONNX Runtime Results:")
    print(f"    Mean latency: {results['mean_ms']:.2f} ms")
    print(f"    Throughput:   {results['fps']:.1f} FPS")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark TRT Engine")
    parser.add_argument("--engine", type=str, required=True, help="TRT engine path")
    parser.add_argument("--onnx", type=str, default=None, help="ONNX model for comparison")
    parser.add_argument("--img_h", type=int, default=360)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()

    img_size = (args.img_h, args.img_w)

    trt_results = benchmark_tensorrt(
        args.engine, img_size, args.batch_size, args.warmup, args.iterations
    )

    if args.onnx:
        ort_results = benchmark_onnxruntime(
            args.onnx, img_size, args.batch_size
        )

        if trt_results and ort_results:
            speedup = ort_results["mean_ms"] / trt_results["mean_ms"]
            print(f"\n  TensorRT speedup over ONNX Runtime: {speedup:.2f}x")


if __name__ == "__main__":
    main()
