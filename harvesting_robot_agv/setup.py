from setuptools import find_packages, setup

package_name = "harvesting_robot_agv"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "bleak",
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
            "ble_agv_bridge_node = harvesting_robot_agv.ble_agv_bridge_node:main",
            "agv_gui_node = harvesting_robot_agv.agv_gui_node:main",
        ],
    },
)
