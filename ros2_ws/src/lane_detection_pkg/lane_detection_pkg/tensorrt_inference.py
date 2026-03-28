"""
TensorRT Inference Wrapper for Lane Segmentation.

Manages the TensorRT engine lifecycle: loading, memory allocation,
preprocessing, inference, and postprocessing. Designed for real-time
use inside a ROS2 node.

Supports both TensorRT engines and ONNX Runtime as a fallback for
development on machines without TensorRT.
"""

import numpy as np
import cv2
import time

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


class TensorRTInference:
    """
    TensorRT inference engine wrapper.

    Handles:
    - Engine loading and context creation
    - CUDA memory allocation for input/output
    - Image preprocessing (resize, normalize, HWC→CHW)
    - Model inference with CUDA streams
    - Postprocessing (argmax → class mask)

    Falls back to ONNX Runtime if TensorRT is not available.

    Args:
        engine_path:  Path to .engine (TRT) or .onnx (fallback) file.
        img_size:     Expected input size (H, W).
        num_classes:  Number of output classes.
    """

    # ImageNet normalization constants
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        engine_path: str,
        img_size: tuple[int, int] = (360, 640),
        num_classes: int = 3,
    ):
        self.img_size = img_size
        self.num_classes = num_classes
        self.engine_path = engine_path
        self.backend = None

        if engine_path.endswith(".engine") and TRT_AVAILABLE:
            self._init_tensorrt(engine_path)
            self.backend = "tensorrt"
        elif engine_path.endswith(".onnx") and ORT_AVAILABLE:
            self._init_onnxruntime(engine_path)
            self.backend = "onnxruntime"
        elif ORT_AVAILABLE and engine_path.endswith(".engine"):
            # Try to fall back to ONNX
            onnx_path = engine_path.replace(".engine", ".onnx")
            print(f"[Inference] TRT not available, trying ONNX fallback: {onnx_path}")
            self._init_onnxruntime(onnx_path)
            self.backend = "onnxruntime"
        else:
            raise RuntimeError(
                f"Cannot load {engine_path}. Install tensorrt or onnxruntime."
            )

        print(f"[Inference] Backend: {self.backend}")
        print(f"[Inference] Input size: {img_size}")

    def _init_tensorrt(self, engine_path: str):
        """Initialize TensorRT engine and allocate GPU memory."""
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate host and device memory
        input_shape = (1, 3, self.img_size[0], self.img_size[1])
        output_shape = (1, self.num_classes, self.img_size[0], self.img_size[1])

        self.h_input = np.empty(input_shape, dtype=np.float32)
        self.h_output = np.empty(output_shape, dtype=np.float32)

        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        self.context.set_input_shape("image", input_shape)

    def _init_onnxruntime(self, onnx_path: str):
        """Initialize ONNX Runtime session."""
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"[Inference] ONNX Runtime providers: {self.ort_session.get_providers()}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an OpenCV image for model input.

        Args:
            image: BGR image from cv2 (H, W, 3), uint8.

        Returns:
            Preprocessed float32 array (1, 3, H, W).
        """
        # Resize
        img = cv2.resize(image, (self.img_size[1], self.img_size[0]))

        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] then ImageNet standardize
        img = img.astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD

        # HWC → CHW → NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        return img

    def postprocess(self, output: np.ndarray) -> np.ndarray:
        """
        Postprocess model output to segmentation mask.

        Args:
            output: Model output logits (1, num_classes, H, W).

        Returns:
            Class-index mask (H, W) as uint8.
        """
        # Argmax across class dimension
        mask = np.argmax(output[0], axis=0).astype(np.uint8)
        return mask

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Run full inference pipeline: preprocess → inference → postprocess.

        Args:
            image: BGR image from cv2, any size.

        Returns:
            Segmentation mask (H, W) with values {0, 1, 2}.
            The mask is at model input resolution (self.img_size).
        """
        # Preprocess
        input_data = self.preprocess(image)

        if self.backend == "tensorrt":
            return self._infer_trt(input_data)
        else:
            return self._infer_ort(input_data)

    def _infer_trt(self, input_data: np.ndarray) -> np.ndarray:
        """TensorRT inference."""
        np.copyto(self.h_input, input_data)

        # Transfer input to GPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer output back
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.postprocess(self.h_output)

    def _infer_ort(self, input_data: np.ndarray) -> np.ndarray:
        """ONNX Runtime inference."""
        outputs = self.ort_session.run(None, {"image": input_data})
        return self.postprocess(outputs[0])

    def infer_with_timing(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Run inference and return timing.

        Returns:
            Tuple of (mask, inference_time_ms).
        """
        start = time.perf_counter()
        mask = self.infer(image)
        elapsed = (time.perf_counter() - start) * 1000
        return mask, elapsed
