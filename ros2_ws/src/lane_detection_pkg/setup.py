from setuptools import setup

package_name = 'lane_detection_pkg'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/lane_detection_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Abhiyaan',
    maintainer_email='user@abhiyaan.com',
    description='ViT-LaneSeg: Vision Transformer Lane Detection for ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detector = lane_detection_pkg.lane_detector_node:main',
        ],
    },
)
