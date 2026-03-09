from glob import glob
from setuptools import find_packages, setup

package_name = "harvesting_robot"

setup(
    name=package_name,
    version="1.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/models", glob("harvesting_robot/models/*")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/urdf", glob("harvesting_robot/urdf/*")),
    ],
    install_requires=[
        "setuptools",
        "opencv-python",
        "numpy",
        "pyrealsense2",
    ],
    zip_safe=True,
    maintainer="Francisco Irizar",
    maintainer_email="flirizar@gmail.com",
    description=(
        "ROS 2 package containing the core nodes required to run "
        "the berry harvesting system modes."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "master_node = harvesting_robot.master_node:main",
            "mode1_vision_node = harvesting_robot.mode1_vision_node:main",
            "eyeinhand_node = harvesting_robot.eyeinhand_node:main",
            "mode1_trajectory_node = harvesting_robot.mode1_trajectory_node:main",
            "control_node = harvesting_robot.control_node:main",
            "mode2_vision_node = harvesting_robot.mode2_vision_node:main",
            "gripper_node = harvesting_robot.gripper_node:main",
            "camera_node = harvesting_robot.camera_node:main",
        ],
    },
)