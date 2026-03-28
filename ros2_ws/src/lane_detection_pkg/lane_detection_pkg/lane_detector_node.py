"""
ROS2 Lane Detection Node.

Subscribes to a camera image topic, runs ViT-LaneSeg inference via
TensorRT (or ONNX Runtime fallback), and publishes:
  - Segmentation mask
  - Debug overlay image
  - Lane info (lane type classification per region)

Topics:
  Subscribed:
    /camera/image_raw  (sensor_msgs/Image)

  Published:
    /perception/lane_mask     (sensor_msgs/Image)  — class-index mask (0/1/2)
    /perception/lane_overlay  (sensor_msgs/Image)  — colorized overlay for debug
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from .tensorrt_inference import TensorRTInference


# Colormap for lane overlay visualization (BGR)
LANE_COLORS = {
    0: (0, 0, 0),        # Background
    1: (0, 255, 255),    # Dashed lane — yellow
    2: (0, 255, 0),      # Solid lane — green
}


class LaneDetectorNode(Node):
    """
    ROS2 node for real-time lane detection.

    Uses ViT-LaneSeg with TensorRT for high-performance inference.
    Subscribes to camera images and publishes segmentation results.
    """

    def __init__(self):
        super().__init__("lane_detector")

        # Declare and read parameters
        self.declare_parameter("engine_path", "lane_seg_fp16.engine")
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("img_height", 360)
        self.declare_parameter("img_width", 640)
        self.declare_parameter("num_classes", 3)
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("overlay_alpha", 0.4)

        engine_path = self.get_parameter("engine_path").value
        camera_topic = self.get_parameter("camera_topic").value
        img_h = self.get_parameter("img_height").value
        img_w = self.get_parameter("img_width").value
        num_classes = self.get_parameter("num_classes").value
        self.publish_overlay = self.get_parameter("publish_overlay").value
        self.overlay_alpha = self.get_parameter("overlay_alpha").value

        # Initialize inference engine
        self.get_logger().info(f"Loading engine: {engine_path}")
        self.engine = TensorRTInference(
            engine_path=engine_path,
            img_size=(img_h, img_w),
            num_classes=num_classes,
        )
        self.get_logger().info(f"Engine loaded, backend: {self.engine.backend}")

        # CV Bridge for ROS Image ↔ OpenCV conversion
        self.bridge = CvBridge()

        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10,  # QoS depth
        )
        self.get_logger().info(f"Subscribed to: {camera_topic}")

        # Publishers
        self.mask_pub = self.create_publisher(
            Image, "/perception/lane_mask", 10
        )

        if self.publish_overlay:
            self.overlay_pub = self.create_publisher(
                Image, "/perception/lane_overlay", 10
            )

        # Stats
        self.frame_count = 0
        self.total_infer_time = 0.0

    def image_callback(self, msg: Image):
        """Process incoming camera image."""
        try:
            # Convert ROS Image → OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Run inference
        mask, infer_ms = self.engine.infer_with_timing(cv_image)

        # Update stats
        self.frame_count += 1
        self.total_infer_time += infer_ms

        if self.frame_count % 30 == 0:
            avg_ms = self.total_infer_time / self.frame_count
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0
            self.get_logger().info(
                f"Frame {self.frame_count}: {infer_ms:.1f}ms "
                f"(avg: {avg_ms:.1f}ms, {fps:.1f} FPS)"
            )

        # Publish segmentation mask
        try:
            # Resize mask to original image size
            orig_h, orig_w = cv_image.shape[:2]
            mask_resized = cv2.resize(
                mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )

            mask_msg = self.bridge.cv2_to_imgmsg(mask_resized, encoding="mono8")
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish mask: {e}")

        # Publish debug overlay
        if self.publish_overlay:
            try:
                overlay = self._create_overlay(cv_image, mask_resized)
                overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
                overlay_msg.header = msg.header
                self.overlay_pub.publish(overlay_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish overlay: {e}")

    def _create_overlay(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Create a colored overlay of the lane mask on the original image."""
        overlay = image.copy()
        color_mask = np.zeros_like(image)

        for class_id, color in LANE_COLORS.items():
            if class_id == 0:
                continue  # Skip background
            color_mask[mask == class_id] = color

        # Blend only where lanes are detected
        lane_pixels = mask > 0
        overlay[lane_pixels] = cv2.addWeighted(
            image[lane_pixels], 1 - self.overlay_alpha,
            color_mask[lane_pixels], self.overlay_alpha,
            0,
        )

        # Add legend
        cv2.putText(overlay, "Dashed", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(overlay, "Solid", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return overlay


def main(args=None):
    rclpy.init(args=args)

    node = LaneDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"Shutting down. Processed {node.frame_count} frames."
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
