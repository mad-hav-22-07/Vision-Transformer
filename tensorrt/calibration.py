"""
INT8 Calibration for TensorRT.

Provides a calibrator that feeds representative images from the training
set to TensorRT for INT8 quantization. The calibrator generates a
calibration cache that can be reused for subsequent engine builds.

Usage:
    Used internally by build_engine.py when --int8 is specified.
"""

import os
import numpy as np
import cv2
from pathlib import Path

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


class Int8Calibrator(trt.IInt8EntropyCalibrator2 if TRT_AVAILABLE else object):
    """
    INT8 entropy calibrator for TensorRT.

    Reads images from a directory, preprocesses them identically to training,
    and feeds them to TensorRT for calibration.

    Args:
        image_dir:   Path to directory of calibration images.
        cache_file:  Path to save/load calibration cache.
        batch_size:  Calibration batch size.
        img_size:    Input image size (H, W).
        num_images:  Max number of images to use for calibration.
    """

    def __init__(
        self,
        image_dir: str,
        cache_file: str = "calibration.cache",
        batch_size: int = 8,
        img_size: tuple[int, int] = (360, 640),
        num_images: int = 500,
    ):
        if TRT_AVAILABLE:
            super().__init__()

        self.cache_file = cache_file
        self.batch_size = batch_size
        self.img_size = img_size

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Collect image files
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_dir = Path(image_dir)
        self.image_files = sorted([
            f for f in image_dir.iterdir()
            if f.suffix.lower() in img_extensions
        ])[:num_images]

        print(f"[Calibrator] Using {len(self.image_files)} images for calibration")

        self.current_index = 0
        self.batch_count = len(self.image_files) // batch_size

        # Allocate device memory for one batch
        self.device_input = cuda.mem_alloc(
            batch_size * 3 * img_size[0] * img_size[1] * 4  # float32
        )

    def preprocess(self, image_path: str) -> np.ndarray:
        """Preprocess a single image for calibration."""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img = img.astype(np.float32) / 255.0

        # Normalize
        img = (img - self.mean) / self.std

        # HWC → CHW
        img = img.transpose(2, 0, 1)

        return img

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names=None) -> list:
        """Return next batch of calibration data."""
        if self.current_index >= len(self.image_files):
            return None

        # Prepare batch
        batch_images = []
        for i in range(self.batch_size):
            idx = self.current_index + i
            if idx >= len(self.image_files):
                break
            img = self.preprocess(self.image_files[idx])
            batch_images.append(img)

        if not batch_images:
            return None

        self.current_index += self.batch_size

        # Stack and copy to device
        batch = np.stack(batch_images).astype(np.float32)
        batch = np.ascontiguousarray(batch)
        cuda.memcpy_htod(self.device_input, batch)

        batch_num = self.current_index // self.batch_size
        if batch_num % 10 == 0:
            print(f"  [Calibrator] Batch {batch_num}/{self.batch_count}")

        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes | None:
        """Read calibration cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            print(f"[Calibrator] Loading cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes):
        """Write calibration cache to disk."""
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"[Calibrator] Cache saved to: {self.cache_file}")
