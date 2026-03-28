# ViT-LaneSeg: Vision Transformer Lane Detection

A **Vision Transformer (ViT) built from scratch** in PyTorch for semantic segmentation of lane markings. Detects and classifies **dashed** and **solid** lane lines. Designed for deployment on **NVIDIA Jetson Orin** via **TensorRT**.

## Architecture

```
Image (360×640×3)
    │
    ▼
┌──────────────────┐
│ Patch Embedding   │  16×16 patches → 880 tokens × 512-dim
│ + Pos Encoding    │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Transformer       │  12 blocks, 8 heads
│ Encoder           │  Multi-scale skip connections (layers 3, 6, 9)
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Segmentation      │  Progressive upsampling 16× with skip fusion
│ Decoder           │  256 → 128 → 64 → 32 channels
└──────────────────┘
    │
    ▼
Lane Mask (360×640×3 classes)
  0 = Background
  1 = Dashed lane
  2 = Solid lane
```

## Project Structure

```
VIT/
├── configs/train_config.yaml      # Hyperparameters
├── dataset/                       # Your labeled data goes here
│   ├── images/{train,val}/        # RGB images
│   └── masks/{train,val}/         # Segmentation masks (0/1/2 pixel values)
├── src/
│   ├── model/                     # ViT-LaneSeg (from scratch)
│   │   ├── patch_embed.py         # Conv-based patch tokenization
│   │   ├── positional_encoding.py # Learnable 2D positional embeddings
│   │   ├── attention.py           # Multi-Head Self-Attention
│   │   ├── transformer_block.py   # Pre-LN encoder block + DropPath
│   │   ├── encoder.py             # Full encoder with skip extraction
│   │   ├── decoder.py             # Progressive upsampling decoder
│   │   └── vit_lane_seg.py        # Complete model assembly
│   ├── data/                      # Dataset & augmentations
│   ├── losses/                    # Focal + Dice combined loss
│   ├── train.py                   # Training loop with AMP + cosine warmup
│   ├── evaluate.py                # Segmentation metrics (mIoU, F1, etc.)
│   └── export_onnx.py             # PyTorch → ONNX export
├── tensorrt/                      # TensorRT optimization
│   ├── build_engine.py            # ONNX → TensorRT engine
│   ├── calibration.py             # INT8 quantization calibrator
│   └── benchmark.py               # Latency/FPS benchmarks
├── ros2_ws/                       # ROS2 integration
│   └── src/lane_detection_pkg/    # ROS2 node package
├── scripts/                       # Utilities
│   ├── visualize_predictions.py   # Prediction visualization
│   └── convert_annotations.py     # Annotation format converter
└── checkpoints/                   # Saved model weights
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your images and masks in the dataset directory:
```
dataset/images/train/  ← training images (.jpg, .png)
dataset/images/val/    ← validation images
dataset/masks/train/   ← training masks (single-channel PNG, values 0/1/2)
dataset/masks/val/     ← validation masks
```

**Mask format**: Single-channel PNG where pixel `0`=background, `1`=dashed lane, `2`=solid lane.

Need to convert from another format? See:
```bash
python scripts/convert_annotations.py --format coco --input annotations.json --image_dir images/ --output masks/
```

### 3. Train

```bash
python -m src.train --config configs/train_config.yaml
```

Override parameters:
```bash
python -m src.train --config configs/train_config.yaml --epochs 50 --batch_size 4 --lr 5e-5
```

Monitor with TensorBoard:
```bash
tensorboard --logdir runs/
```

### 4. Evaluate

Metrics are computed during training. For standalone evaluation:
```bash
python scripts/visualize_predictions.py --checkpoint checkpoints/best_model.pth --image_dir dataset/images/val --output_dir predictions/ --no-show
```

### 5. Export to ONNX

```bash
python -m src.export_onnx --checkpoint checkpoints/best_model.pth --output lane_seg.onnx --verify
```

### 6. Build TensorRT Engine (on Jetson Orin)

```bash
# FP16 (recommended)
python tensorrt/build_engine.py --onnx lane_seg.onnx --output lane_seg_fp16.engine --fp16

# INT8 (maximum speed)
python tensorrt/build_engine.py --onnx lane_seg.onnx --output lane_seg_int8.engine --int8 --calib_dir dataset/images/train

# Benchmark
python tensorrt/benchmark.py --engine lane_seg_fp16.engine --onnx lane_seg.onnx
```

### 7. ROS2 Deployment

```bash
# Build the ROS2 package
cd ros2_ws
colcon build --packages-select lane_detection_pkg
source install/setup.bash

# Launch
ros2 launch lane_detection_pkg lane_detection_launch.py engine_path:=/path/to/lane_seg_fp16.engine camera_topic:=/camera/image_raw
```

**Published topics:**
| Topic | Type | Description |
|---|---|---|
| `/perception/lane_mask` | `sensor_msgs/Image` | Class-index mask (0/1/2) |
| `/perception/lane_overlay` | `sensor_msgs/Image` | Colorized debug overlay |

## Pipeline Overview

```
Camera → ROS2 Node → TensorRT Engine → Lane Mask → Perception Fusion
                  ↘ YOLOv8 (existing)  → Detections ↗
```

Both the ViT lane detection and your existing YOLOv8 object detection run in parallel, subscribing to the same camera topic.

## Model Configuration

| Parameter | Default | Description |
|---|---|---|
| `img_size` | 360×640 | Input resolution |
| `patch_size` | 16 | Patch size in pixels |
| `embed_dim` | 512 | Embedding dimension |
| `depth` | 12 | Transformer blocks |
| `num_heads` | 8 | Attention heads |
| `num_classes` | 3 | bg + dashed + solid |
| Parameters | ~30M | Total trainable params |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NVIDIA CUDA GPU (training)
- NVIDIA Jetson Orin + JetPack 6.0+ (deployment)
- ROS2 Humble/Iron (ROS integration)
