#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from harvesting_robot_agv.agv_position_control_node import (
    DEFAULT_REFERENCE_SIDE_TOPIC,
    DEFAULT_TARGET_HORIZONTAL_LINE_DISTANCE_M,
    REFERENCE_SIDE_AUTO,
)


DEFAULT_LIDAR_SERIAL_PORT = (
    "/dev/serial/by-id/"
    "usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
)


def generate_launch_description() -> LaunchDescription:
    serial_port = LaunchConfiguration("serial_port")
    serial_baudrate = LaunchConfiguration("serial_baudrate")
    web_host = LaunchConfiguration("web_host")
    web_port = LaunchConfiguration("web_port")
    show_radar_window = LaunchConfiguration("show_radar_window")
    target_horizontal_line_distance_m = LaunchConfiguration(
        "target_horizontal_line_distance_m"
    )
    reference_side = LaunchConfiguration("reference_side")
    reference_side_topic = LaunchConfiguration("reference_side_topic")
    target_side_mode = LaunchConfiguration("target_side_mode")
    target_side_mode_topic = LaunchConfiguration("target_side_mode_topic")
    auto_harvest_enabled = LaunchConfiguration("auto_harvest_enabled")
    auto_harvest_topic = LaunchConfiguration("auto_harvest_topic")
    side_clearance_region_outer_offset_m = LaunchConfiguration(
        "side_clearance_region_outer_offset_m"
    )
    side_clearance_region_forward_m = LaunchConfiguration(
        "side_clearance_region_forward_m"
    )
    side_clearance_detection_min_points = LaunchConfiguration(
        "side_clearance_detection_min_points"
    )
    side_clearance_sticky_timeout_sec = LaunchConfiguration(
        "side_clearance_sticky_timeout_sec"
    )
    side_clearance_sticky_max_motion_m = LaunchConfiguration(
        "side_clearance_sticky_max_motion_m"
    )

    sllidar_launch = os.path.join(
        get_package_share_directory("sllidar_ros2"),
        "launch",
        "sllidar_a1_launch.py",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "serial_port",
                default_value=DEFAULT_LIDAR_SERIAL_PORT,
                description="Serial port for the RPLIDAR USB adapter.",
            ),
            DeclareLaunchArgument(
                "serial_baudrate",
                default_value="115200",
                description="Serial baudrate for the RPLIDAR.",
            ),
            DeclareLaunchArgument(
                "web_host",
                default_value="0.0.0.0",
                description="Host/IP for the mobile web control server.",
            ),
            DeclareLaunchArgument(
                "web_port",
                default_value="8080",
                description="Port for the mobile web control server.",
            ),
            DeclareLaunchArgument(
                "show_radar_window",
                default_value="false",
                description=(
                    "Open the local OpenCV radar window. The web radar image "
                    "is still published when this is false."
                ),
            ),
            DeclareLaunchArgument(
                "target_horizontal_line_distance_m",
                default_value=str(DEFAULT_TARGET_HORIZONTAL_LINE_DISTANCE_M),
                description=(
                    "Horizontal target distance for the radar vertical "
                    "reference lines and AGV position controller."
                ),
            ),
            DeclareLaunchArgument(
                "reference_side",
                default_value=REFERENCE_SIDE_AUTO,
                description=(
                    "AGV bush-line reference side for orientation/position "
                    "control: auto, left, or right."
                ),
            ),
            DeclareLaunchArgument(
                "reference_side_topic",
                default_value=DEFAULT_REFERENCE_SIDE_TOPIC,
                description="Topic used to update the AGV reference side.",
            ),
            DeclareLaunchArgument(
                "target_side_mode",
                default_value="any",
                description=(
                    "Cobot target side accepted by searching mode: any, left, "
                    "or right."
                ),
            ),
            DeclareLaunchArgument(
                "target_side_mode_topic",
                default_value="/searching_mode/target_side",
                description="Topic used to update the cobot target side mode.",
            ),
            DeclareLaunchArgument(
                "auto_harvest_enabled",
                default_value="true",
                description=(
                    "Enable automatic master_node START after target "
                    "confirmation."
                ),
            ),
            DeclareLaunchArgument(
                "auto_harvest_topic",
                default_value="/searching_mode/auto_harvest",
                description="Topic used to enable or disable auto harvesting.",
            ),
            DeclareLaunchArgument(
                "side_clearance_region_outer_offset_m",
                default_value="1.0",
                description=(
                    "Outer lateral edge of the side harvesting regions, in "
                    "meters from the LiDAR center."
                ),
            ),
            DeclareLaunchArgument(
                "side_clearance_region_forward_m",
                default_value="1.4",
                description=(
                    "Forward length of the side harvesting regions from the "
                    "AGV front line, in meters."
                ),
            ),
            DeclareLaunchArgument(
                "side_clearance_detection_min_points",
                default_value="1",
                description=(
                    "Minimum LiDAR points inside the side harvesting regions "
                    "to publish a region detection."
                ),
            ),
            DeclareLaunchArgument(
                "side_clearance_sticky_timeout_sec",
                default_value="0.75",
                description=(
                    "Time before a non-moving side-region detection is "
                    "suppressed while the AGV is moving."
                ),
            ),
            DeclareLaunchArgument(
                "side_clearance_sticky_max_motion_m",
                default_value="0.03",
                description=(
                    "Maximum position change treated as sticky/non-moving for "
                    "side-region detections."
                ),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sllidar_launch),
                launch_arguments={
                    "serial_port": serial_port,
                    "serial_baudrate": serial_baudrate,
                }.items(),
            ),
            Node(
                package="harvesting_robot_agv",
                executable="lidar_360_radar_node",
                name="lidar_360_radar_node",
                output="screen",
                parameters=[
                    {
                        "scan_topic": "/scan",
                        "line_detection_topic": "/agv/line_detections",
                        "radar_image_topic": "/agv/lidar_radar_image",
                        "target_horizontal_line_distance_m": ParameterValue(
                            target_horizontal_line_distance_m,
                            value_type=float,
                        ),
                        "side_clearance_region_outer_offset_m": ParameterValue(
                            side_clearance_region_outer_offset_m,
                            value_type=float,
                        ),
                        "side_clearance_region_forward_m": ParameterValue(
                            side_clearance_region_forward_m,
                            value_type=float,
                        ),
                        "side_clearance_detection_min_points": ParameterValue(
                            side_clearance_detection_min_points,
                            value_type=int,
                        ),
                        "side_clearance_sticky_timeout_sec": ParameterValue(
                            side_clearance_sticky_timeout_sec,
                            value_type=float,
                        ),
                        "side_clearance_sticky_max_motion_m": ParameterValue(
                            side_clearance_sticky_max_motion_m,
                            value_type=float,
                        ),
                        "show_window": ParameterValue(
                            show_radar_window,
                            value_type=bool,
                        ),
                    }
                ],
            ),
            Node(
                package="harvesting_robot_agv",
                executable="agv_position_control_node",
                name="agv_position_control_node",
                output="screen",
                parameters=[
                    {
                        "target_horizontal_line_distance_m": ParameterValue(
                            target_horizontal_line_distance_m,
                            value_type=float,
                        ),
                        "reference_side": reference_side,
                        "reference_side_topic": reference_side_topic,
                    }
                ],
            ),
            Node(
                package="harvesting_robot_agv",
                executable="ble_agv_bridge_node",
                name="ble_agv_bridge_node",
                output="screen",
            ),
            Node(
                package="harvesting_robot_agv",
                executable="agv_web_control_node",
                name="agv_web_control_node",
                output="screen",
                parameters=[
                    {
                        "web_host": web_host,
                        "web_port": ParameterValue(web_port, value_type=int),
                        "radar_image_topic": "/agv/lidar_radar_image",
                        "line_detection_topic": "/agv/line_detections",
                        "reference_side": reference_side,
                        "reference_side_topic": reference_side_topic,
                        "target_side_mode": target_side_mode,
                        "target_side_mode_topic": target_side_mode_topic,
                        "auto_harvest_enabled": ParameterValue(
                            auto_harvest_enabled,
                            value_type=bool,
                        ),
                        "auto_harvest_topic": auto_harvest_topic,
                    }
                ],
            ),
        ]
    )
