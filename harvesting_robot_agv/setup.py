import os
from glob import glob

from setuptools import find_packages, setup

package_name = "harvesting_robot_agv"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=[
        "setuptools",
        "bleak",
        "numpy",
        "opencv-python",
    ],
    zip_safe=True,
    maintainer="Francisco Irizar",
    maintainer_email="flirizar@gmail.com",
    description=(
        "ROS 2 Python package for AGV functionality in the berry harvesting "
        "robot system."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            (
                "ble_agv_bridge_node = "
                "harvesting_robot_agv.ble_agv_bridge_node:main"
            ),
            "agv_gui_node = harvesting_robot_agv.agv_gui_node:main",
            "agv_web_control_node = harvesting_robot_agv.agv_web_control_node:main",
            (
                "lidar_360_radar_node = "
                "harvesting_robot_agv.lidar_360_radar_node:main"
            ),
            (
                "agv_position_control_node = "
                "harvesting_robot_agv.agv_position_control_node:main"
            ),
        ],
    },
)
