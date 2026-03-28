"""
ROS2 Launch File for Lane Detection Pipeline.

Launches the lane detection node with configurable parameters.

Usage:
    ros2 launch lane_detection_pkg lane_detection_launch.py
    ros2 launch lane_detection_pkg lane_detection_launch.py engine_path:=/path/to/engine.engine
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # ===== Launch Arguments =====
        DeclareLaunchArgument(
            'engine_path',
            default_value='lane_seg_fp16.engine',
            description='Path to TensorRT engine or ONNX model'
        ),
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/image_raw',
            description='Camera image topic to subscribe to'
        ),
        DeclareLaunchArgument(
            'img_height',
            default_value='360',
            description='Model input height'
        ),
        DeclareLaunchArgument(
            'img_width',
            default_value='640',
            description='Model input width'
        ),
        DeclareLaunchArgument(
            'publish_overlay',
            default_value='true',
            description='Publish debug overlay image'
        ),

        # ===== Lane Detection Node =====
        Node(
            package='lane_detection_pkg',
            executable='lane_detector',
            name='lane_detector',
            output='screen',
            parameters=[{
                'engine_path': LaunchConfiguration('engine_path'),
                'camera_topic': LaunchConfiguration('camera_topic'),
                'img_height': LaunchConfiguration('img_height'),
                'img_width': LaunchConfiguration('img_width'),
                'publish_overlay': LaunchConfiguration('publish_overlay'),
                'overlay_alpha': 0.4,
                'num_classes': 3,
            }],
            remappings=[
                # Remap if your camera uses a different topic name
                # ('/camera/image_raw', '/your_camera/image'),
            ],
        ),
    ])
