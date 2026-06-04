#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


DEFAULT_SEARCH_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]
DEFAULT_SEARCH_JOINT_POSITIONS_DEG = [0.0, -93.0, -147.0, 0.0, -34.0, 154.0]
DEFAULT_TARGET_DISTANCE_REFERENCE_XYZ = [0.0, 0.0, 0.28481]
AGV_CONTROL_MODE_AUTOMATIC = "automatic"
AGV_CONTROL_MODE_MANUAL = "manual"
DEFAULT_LIDAR_SIDE_CLEARANCE_TOPIC = "/agv/line_detections"


class SearchingModeNode(Node):
    """Move the cobot to the searching posture and sweep the first joint."""

    def __init__(self) -> None:
        super().__init__("searching_mode_node")

        self._declare_parameters()
        self._load_parameters()
        self._load_yolo_model()
        self._create_ros_interfaces()

        self.bridge = CvBridge()

        self.is_busy = False
        self.is_oscillating = False
        self.initial_pose_timer = None
        self.oscillation_timer = None
        self.eye_compute_timer = None
        self.final_confirmation_timer = None
        self.final_stop_timer = None
        self.active_oscillation_min_deg = self.oscillation_min_deg
        self.active_oscillation_max_deg = self.oscillation_max_deg
        self.next_oscillation_target_deg = self.oscillation_max_deg
        self.last_commanded_oscillation_joint_deg = None
        self.current_oscillation_joint_deg = None
        self.current_oscillation_joint_time = None
        self.closest_detection_distance_m = None
        self.last_berry_detection_time = None
        self.robot_side = None
        self.target_side = None
        self.is_finalizing_target = False
        self.final_target_confirmed = False
        self.lidar_region_confirmation_active = False
        self.lidar_region_detection_side = None
        self.lidar_region_detection_time = None
        self.agv_control_mode = None
        self.agv_mode_before_stop = None
        self.latest_agv_command = None
        self.agv_command_before_lidar_stop = None
        self.latest_color_image = None
        self.latest_depth_image = None
        self.has_camera_info = False
        self.latest_target_base_distance_m = None
        self.latest_target_base_time = None
        self.fx = None
        self.fy = None
        self.ppx = None
        self.ppy = None
        self.camera_frame_id = "camera_color_optical_frame"
        if self.show_preview:
            try:
                cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
            except Exception as exc:
                self.get_logger().warn(
                    f"Failed to create OpenCV preview window: {exc}"
                )
                self.show_preview = False
        self.status_publisher.publish(String(data="IDLE"))
        self.get_logger().info("searching_mode_node ready")
        self.get_logger().info(f"Command topic: {self.cmd_topic}")
        self.get_logger().info(f"Status topic: {self.status_topic}")
        self.get_logger().info(f"Controller topic: {self.controller_topic}")
        self.get_logger().info(f"Detection result topic: {self.detection_result_topic}")
        self.get_logger().info(f"Annotated image topic: {self.annotated_image_topic}")

    def _declare_parameters(self) -> None:
        self.declare_parameter("cmd_topic", "/searching_mode/cmd")
        self.declare_parameter("status_topic", "/searching_mode/status")
        self.declare_parameter(
            "controller_topic",
            "/joint_trajectory_controller/joint_trajectory",
        )
        self.declare_parameter("search_joint_names", DEFAULT_SEARCH_JOINT_NAMES)
        self.declare_parameter(
            "search_joint_positions_deg",
            DEFAULT_SEARCH_JOINT_POSITIONS_DEG,
        )
        self.declare_parameter("pose_horizon_sec", 5.0)
        self.declare_parameter("oscillation_joint_index", 0)
        self.declare_parameter("oscillation_min_deg", -45.0)
        self.declare_parameter("oscillation_max_deg", 45.0)
        self.declare_parameter("oscillation_horizon_sec", 5.0)
        self.declare_parameter("left_refine_min_deg", -90.0)
        self.declare_parameter("left_refine_max_deg", -45.0)
        self.declare_parameter("right_refine_min_deg", 45.0)
        self.declare_parameter("right_refine_max_deg", 90.0)
        self.declare_parameter("left_final_deg", -90.0)
        self.declare_parameter("right_final_deg", 90.0)
        self.declare_parameter("final_pose_horizon_sec", 2.0)

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("depth_scale_topic", "/camera/depth_scale")
        self.declare_parameter("depth_scale_fallback", 0.001)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("detection_result_topic", "/searching_mode/detection")
        self.declare_parameter(
            "annotated_image_topic",
            "/searching_mode/annotated_image",
        )
        self.declare_parameter("show_preview", False)
        self.declare_parameter("preview_window", "Searching Mode Detections")
        self.declare_parameter("output_point_topic", "/camera_sphere")
        self.declare_parameter("target_base_topic", "/target_base")
        self.declare_parameter("eye_cmd_topic", "/eyeinhand/cmd")
        self.declare_parameter("trigger_eyeinhand_compute", True)
        self.declare_parameter("eye_compute_delay_sec", 0.05)
        self.declare_parameter("detection_period_sec", 0.2)
        self.declare_parameter("berry_lost_timeout_sec", 3.0)
        self.declare_parameter("agv_cmd_topic", "/agv/rpm_cmd")
        self.declare_parameter("agv_control_mode_topic", "/agv/control_mode")
        self.declare_parameter("agv_stop_command", "s")
        self.declare_parameter("agv_manual_command", AGV_CONTROL_MODE_MANUAL)
        self.declare_parameter("agv_automatic_command", AGV_CONTROL_MODE_AUTOMATIC)
        self.declare_parameter("agv_stop_period_sec", 0.05)
        self.declare_parameter("target_stop_distance_m", 1.0)
        self.declare_parameter("final_target_timeout_sec", 3.0)
        self.declare_parameter(
            "lidar_side_clearance_topic",
            DEFAULT_LIDAR_SIDE_CLEARANCE_TOPIC,
        )
        self.declare_parameter("lidar_region_confirmation_timeout_sec", 3.0)
        self.declare_parameter("camera_detection_recent_timeout_sec", 0.6)
        self.declare_parameter(
            "target_distance_reference_xyz",
            DEFAULT_TARGET_DISTANCE_REFERENCE_XYZ,
        )

        share_dir = get_package_share_directory("harvesting_robot")
        models_dir = os.path.join(share_dir, "models")
        self.declare_parameter(
            "model_path",
            os.path.join(models_dir, "best.pt"),
        )
        self.declare_parameter("yolo_device", "0")
        self.declare_parameter(
            "cfg_path",
            os.path.join(models_dir, "yolov4-tiny-custom.cfg"),
        )
        self.declare_parameter(
            "weights_path",
            os.path.join(models_dir, "yolov4-tiny-custom_best.weights"),
        )
        self.declare_parameter(
            "names_path",
            os.path.join(models_dir, "blackberry.names"),
        )
        self.declare_parameter("conf_thresh", 0.6)
        self.declare_parameter("nms_thresh", 0.4)
        self.declare_parameter("target_class_id", 2)
        self.declare_parameter("min_valid_depth_m", 0.10)
        self.declare_parameter("max_valid_depth_m", 2.00)
        self.declare_parameter("min_depth_samples", 10)
        self.declare_parameter("depth_grid_div", 10)
        self.declare_parameter("center_patch_halfwin", 16)
        self.declare_parameter("side_deadband_deg", 2.0)
        self.declare_parameter("negative_joint_side", "right")
        self.declare_parameter("joint_state_max_age_sec", 0.5)

    def _load_parameters(self) -> None:
        self.cmd_topic = str(self.get_parameter("cmd_topic").value).strip()
        self.status_topic = str(self.get_parameter("status_topic").value).strip()
        self.controller_topic = str(
            self.get_parameter("controller_topic").value
        ).strip()
        self.search_joint_names = list(self.get_parameter("search_joint_names").value)
        self.search_joint_positions_deg = [
            float(position_deg)
            for position_deg in self.get_parameter("search_joint_positions_deg").value
        ]
        self.pose_horizon_sec = float(self.get_parameter("pose_horizon_sec").value)
        self.oscillation_joint_index = int(
            self.get_parameter("oscillation_joint_index").value
        )
        self.oscillation_min_deg = float(
            self.get_parameter("oscillation_min_deg").value
        )
        self.oscillation_max_deg = float(
            self.get_parameter("oscillation_max_deg").value
        )
        self.oscillation_horizon_sec = float(
            self.get_parameter("oscillation_horizon_sec").value
        )
        self.left_refine_min_deg = float(
            self.get_parameter("left_refine_min_deg").value
        )
        self.left_refine_max_deg = float(
            self.get_parameter("left_refine_max_deg").value
        )
        self.right_refine_min_deg = float(
            self.get_parameter("right_refine_min_deg").value
        )
        self.right_refine_max_deg = float(
            self.get_parameter("right_refine_max_deg").value
        )
        self.left_final_deg = float(self.get_parameter("left_final_deg").value)
        self.right_final_deg = float(self.get_parameter("right_final_deg").value)
        self.final_pose_horizon_sec = float(
            self.get_parameter("final_pose_horizon_sec").value
        )

        self.color_topic = str(self.get_parameter("color_topic").value).strip()
        self.depth_topic = str(self.get_parameter("depth_topic").value).strip()
        self.camera_info_topic = str(
            self.get_parameter("camera_info_topic").value
        ).strip()
        self.depth_scale_topic = str(
            self.get_parameter("depth_scale_topic").value
        ).strip()
        self.depth_scale_m_per_unit = float(
            self.get_parameter("depth_scale_fallback").value
        )
        self.joint_state_topic = str(
            self.get_parameter("joint_state_topic").value
        ).strip()
        self.detection_result_topic = str(
            self.get_parameter("detection_result_topic").value
        ).strip()
        self.annotated_image_topic = str(
            self.get_parameter("annotated_image_topic").value
        ).strip()
        self.show_preview = bool(self.get_parameter("show_preview").value)
        self.preview_window_name = str(
            self.get_parameter("preview_window").value
        ).strip()
        self.output_point_topic = str(
            self.get_parameter("output_point_topic").value
        ).strip()
        self.target_base_topic = str(
            self.get_parameter("target_base_topic").value
        ).strip()
        self.eye_cmd_topic = str(self.get_parameter("eye_cmd_topic").value).strip()
        self.trigger_eyeinhand_compute = bool(
            self.get_parameter("trigger_eyeinhand_compute").value
        )
        self.eye_compute_delay_sec = float(
            self.get_parameter("eye_compute_delay_sec").value
        )
        self.detection_period_sec = float(
            self.get_parameter("detection_period_sec").value
        )
        self.berry_lost_timeout_sec = float(
            self.get_parameter("berry_lost_timeout_sec").value
        )
        self.agv_cmd_topic = str(self.get_parameter("agv_cmd_topic").value).strip()
        self.agv_control_mode_topic = str(
            self.get_parameter("agv_control_mode_topic").value
        ).strip()
        self.agv_stop_command = str(
            self.get_parameter("agv_stop_command").value
        ).strip()
        self.agv_manual_command = str(
            self.get_parameter("agv_manual_command").value
        ).strip()
        self.agv_automatic_command = str(
            self.get_parameter("agv_automatic_command").value
        ).strip()
        self.agv_stop_period_sec = float(
            self.get_parameter("agv_stop_period_sec").value
        )
        self.target_stop_distance_m = float(
            self.get_parameter("target_stop_distance_m").value
        )
        self.final_target_timeout_sec = float(
            self.get_parameter("final_target_timeout_sec").value
        )
        self.lidar_side_clearance_topic = str(
            self.get_parameter("lidar_side_clearance_topic").value
        ).strip()
        if not self.lidar_side_clearance_topic:
            self.lidar_side_clearance_topic = DEFAULT_LIDAR_SIDE_CLEARANCE_TOPIC
        self.lidar_region_confirmation_timeout_sec = float(
            self.get_parameter("lidar_region_confirmation_timeout_sec").value
        )
        self.camera_detection_recent_timeout_sec = float(
            self.get_parameter("camera_detection_recent_timeout_sec").value
        )
        self.target_distance_reference_xyz = np.array(
            [
                float(value)
                for value in self.get_parameter("target_distance_reference_xyz").value
            ],
            dtype=float,
        )

        self.model_path = os.path.expanduser(
            str(self.get_parameter("model_path").value)
        )
        self.yolo_device = str(self.get_parameter("yolo_device").value).strip()
        self.config_path = os.path.expanduser(str(self.get_parameter("cfg_path").value))
        self.weights_path = os.path.expanduser(
            str(self.get_parameter("weights_path").value)
        )
        self.class_names_path = os.path.expanduser(
            str(self.get_parameter("names_path").value)
        )
        self.confidence_threshold = float(self.get_parameter("conf_thresh").value)
        self.nms_threshold = float(self.get_parameter("nms_thresh").value)
        self.target_class_id = int(self.get_parameter("target_class_id").value)
        self.min_valid_depth_m = float(
            self.get_parameter("min_valid_depth_m").value
        )
        self.max_valid_depth_m = float(
            self.get_parameter("max_valid_depth_m").value
        )
        self.min_depth_samples = int(self.get_parameter("min_depth_samples").value)
        self.depth_grid_div = int(self.get_parameter("depth_grid_div").value)
        self.center_patch_half_window = int(
            self.get_parameter("center_patch_halfwin").value
        )
        self.side_deadband_deg = float(self.get_parameter("side_deadband_deg").value)
        self.negative_joint_side = (
            str(self.get_parameter("negative_joint_side").value).strip().lower()
        )
        if self.negative_joint_side not in ("left", "right"):
            self.get_logger().warn(
                "negative_joint_side must be 'left' or 'right'; using 'right'"
            )
            self.negative_joint_side = "right"
        self.joint_state_max_age_sec = float(
            self.get_parameter("joint_state_max_age_sec").value
        )

    def _load_yolo_model(self) -> None:
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Missing YOLO model file: {self.model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "The searching_mode_node now uses an Ultralytics YOLO .pt model. "
                "Install it with: pip install ultralytics"
            ) from exc

        self.yolo_device = self._resolve_yolo_device(self.yolo_device)
        self.yolo_model = YOLO(self.model_path)
        self.class_names = self._normalize_class_names(
            getattr(self.yolo_model, "names", None)
        )
        if not self.class_names and os.path.isfile(self.class_names_path):
            with open(self.class_names_path, "r", encoding="utf-8") as file:
                self.class_names = [line.strip() for line in file.readlines()]

        self.get_logger().info(
            "Using Ultralytics YOLO model:\n"
            f"  model: {self.model_path}\n"
            f"  device: {self.yolo_device or 'auto'}\n"
            f"  classes: {', '.join(self.class_names) if self.class_names else 'unknown'}"
        )

        if self.class_names and not 0 <= self.target_class_id < len(self.class_names):
            self.get_logger().warn(
                f"target_class_id={self.target_class_id} is outside the "
                f"model class range 0..{len(self.class_names) - 1}"
            )

    def _create_ros_interfaces(self) -> None:
        self.status_publisher = self.create_publisher(String, self.status_topic, 10)
        self.detection_result_publisher = self.create_publisher(
            String,
            self.detection_result_topic,
            10,
        )
        self.annotated_image_publisher = self.create_publisher(
            Image,
            self.annotated_image_topic,
            10,
        )
        self.point_publisher = self.create_publisher(
            PointStamped,
            self.output_point_topic,
            10,
        )
        self.eye_cmd_publisher = self.create_publisher(
            String,
            self.eye_cmd_topic,
            10,
        )
        self.agv_cmd_publisher = self.create_publisher(
            String,
            self.agv_cmd_topic,
            10,
        )
        self.agv_control_mode_publisher = self.create_publisher(
            String,
            self.agv_control_mode_topic,
            10,
        )
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            self.controller_topic,
            10,
        )
        self.create_subscription(String, self.cmd_topic, self._on_command, 10)
        self.create_subscription(
            PointStamped,
            self.target_base_topic,
            self._on_target_base,
            10,
        )
        self.create_subscription(
            String,
            self.agv_control_mode_topic,
            self._on_agv_control_mode,
            10,
        )
        self.create_subscription(
            String,
            self.agv_cmd_topic,
            self._on_agv_command,
            10,
        )
        self.create_subscription(
            String,
            self.lidar_side_clearance_topic,
            self._on_lidar_side_clearance,
            10,
        )
        self.create_subscription(Image, self.color_topic, self._on_color_image, 10)
        self.create_subscription(Image, self.depth_topic, self._on_depth_image, 10)
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._on_camera_info,
            10,
        )
        self.create_subscription(
            Float32,
            self.depth_scale_topic,
            self._on_depth_scale,
            10,
        )
        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._on_joint_state,
            10,
        )
        self.detection_timer = self.create_timer(
            self._positive_seconds(self.detection_period_sec),
            self._on_detection_timer,
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        intrinsic_matrix = list(msg.k)
        if len(intrinsic_matrix) == 9:
            self.fx = float(intrinsic_matrix[0])
            self.fy = float(intrinsic_matrix[4])
            self.ppx = float(intrinsic_matrix[2])
            self.ppy = float(intrinsic_matrix[5])
            self.has_camera_info = True

        if msg.header.frame_id:
            self.camera_frame_id = msg.header.frame_id

    def _on_color_image(self, msg: Image) -> None:
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
        except Exception as exc:
            self.latest_color_image = None
            self.get_logger().warn(f"Failed to convert color image: {exc}")

    def _on_depth_image(self, msg: Image) -> None:
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough",
            )
        except Exception as exc:
            self.latest_depth_image = None
            self.get_logger().warn(f"Failed to convert depth image: {exc}")

    def _on_depth_scale(self, msg: Float32) -> None:
        try:
            depth_scale = float(msg.data)
            if np.isfinite(depth_scale) and depth_scale > 0.0:
                self.depth_scale_m_per_unit = depth_scale
        except Exception:
            pass

    def _on_joint_state(self, msg: JointState) -> None:
        if not 0 <= self.oscillation_joint_index < len(self.search_joint_names):
            return

        joint_name = self.search_joint_names[self.oscillation_joint_index]
        joint_positions = {
            name: position
            for name, position in zip(msg.name, msg.position)
        }
        if joint_name not in joint_positions:
            return

        self.current_oscillation_joint_deg = math.degrees(joint_positions[joint_name])
        self.current_oscillation_joint_time = self.get_clock().now()

    def _on_agv_control_mode(self, msg: String) -> None:
        control_mode = msg.data.strip().lower()
        if control_mode not in (AGV_CONTROL_MODE_MANUAL, AGV_CONTROL_MODE_AUTOMATIC):
            return

        self.agv_control_mode = control_mode

    def _on_agv_command(self, msg: String) -> None:
        self.latest_agv_command = msg.data.strip()

    def _on_lidar_side_clearance(self, msg: String) -> None:
        if not self._should_accept_lidar_region_trigger():
            return

        try:
            values = self._parse_key_value_message(msg.data)
            count = int(values.get("side_clearance_count", "0"))
            if count <= 0:
                return

            sticky = self._parse_bool(values.get("side_clearance_sticky"))
            if sticky:
                return

            side = values.get("side_clearance_side", "none").strip().lower()
            if side not in ("left", "right"):
                return

            nearest_m = self._parse_optional_float(
                values.get("side_clearance_nearest_m")
            )
            x_m = self._parse_optional_float(values.get("side_clearance_x_m"))
            y_m = self._parse_optional_float(values.get("side_clearance_y_m"))

        except ValueError as error:
            self.get_logger().warn(
                f"Ignoring invalid lidar side clearance message: {error}"
            )
            return

        self._handle_lidar_region_detection(
            side=side,
            count=count,
            nearest_m=nearest_m,
            x_m=x_m,
            y_m=y_m,
        )

    def _should_accept_lidar_region_trigger(self) -> bool:
        return (
            self.is_busy
            and not self.is_finalizing_target
            and self.target_side is None
            and not self._has_recent_camera_detection()
        )

    def _on_target_base(self, msg: PointStamped) -> None:
        if (
            not self.is_busy
            or self.target_side is None
        ):
            return

        target_xyz = np.array(
            [msg.point.x, msg.point.y, msg.point.z],
            dtype=float,
        )
        target_distance_m = float(
            np.linalg.norm(target_xyz - self.target_distance_reference_xyz)
        )
        self.latest_target_base_distance_m = target_distance_m
        self.latest_target_base_time = self.get_clock().now()
        self.detection_result_publisher.publish(
            String(
                data=(
                    f"stage={'final_target_base' if self.is_finalizing_target else 'target_base'},"
                    f"side={self.robot_side or 'unknown'},"
                    f"target_side={self.target_side},"
                    f"target_distance_m={target_distance_m:.3f},"
                    f"x={msg.point.x:.3f},"
                    f"y={msg.point.y:.3f},"
                    f"z={msg.point.z:.3f}"
                )
            )
        )

        if target_distance_m > self.target_stop_distance_m:
            return

        if self.is_finalizing_target:
            self._confirm_final_target(target_distance_m)
        else:
            self._handle_target_within_stop_distance(target_distance_m)

    def _on_command(self, msg: String) -> None:
        command = msg.data.strip().upper()
        self.get_logger().info(f"Received command: '{command}'")

        if command in ("START", "SEARCH", "RUN"):
            self._start()
            return

        if command in ("STOP", "ABORT"):
            self._stop(command)
            return

        self.get_logger().warn(f"Ignoring invalid command: '{msg.data}'")

    def _start(self) -> None:
        if self.is_busy:
            self.get_logger().info("START ignored because searching mode is busy.")
            return

        self.is_busy = True
        self.is_oscillating = False
        self.closest_detection_distance_m = None
        self.last_berry_detection_time = None
        self.robot_side = None
        self.target_side = None
        self.is_finalizing_target = False
        self.final_target_confirmed = False
        self.lidar_region_confirmation_active = False
        self.lidar_region_detection_side = None
        self.lidar_region_detection_time = None
        self.agv_mode_before_stop = None
        self.agv_command_before_lidar_stop = None
        self.latest_target_base_distance_m = None
        self.latest_target_base_time = None
        self.active_oscillation_min_deg = self.oscillation_min_deg
        self.active_oscillation_max_deg = self.oscillation_max_deg
        self.status_publisher.publish(String(data="BUSY"))
        if not self._publish_search_pose():
            self._finish("DONE_FAIL")
            return

        self.next_oscillation_target_deg = self.active_oscillation_max_deg
        self.initial_pose_timer = self.create_timer(
            self._positive_seconds(self.pose_horizon_sec),
            self._start_oscillation,
        )
        self.get_logger().info(
            "Searching posture command sent. "
            f"Oscillation starts in {self.pose_horizon_sec:.2f}s"
        )

    def _stop(self, command: str) -> None:
        if not self.is_busy:
            self.status_publisher.publish(String(data="IDLE"))
            return

        status = "DONE_OK" if command == "STOP" else "DONE_FAIL"
        self._finish(status)
        self.get_logger().warn(f"{command} -> {status} / IDLE")

    def _publish_search_pose(self) -> bool:
        if not self._has_valid_configuration():
            return False

        self._publish_joint_positions_deg(
            self.search_joint_positions_deg,
            self.pose_horizon_sec,
        )
        return True

    def _has_valid_configuration(self) -> bool:
        if len(self.search_joint_positions_deg) != len(self.search_joint_names):
            self.get_logger().error(
                "search_joint_positions_deg must have the same length as "
                "search_joint_names"
            )
            return False

        if not self.search_joint_names:
            self.get_logger().error("search_joint_names cannot be empty")
            return False

        if not 0 <= self.oscillation_joint_index < len(self.search_joint_names):
            self.get_logger().error(
                "oscillation_joint_index must point to an entry in search_joint_names"
            )
            return False

        if self.oscillation_min_deg >= self.oscillation_max_deg:
            self.get_logger().error(
                "oscillation_min_deg must be lower than oscillation_max_deg"
            )
            return False

        if self.left_refine_min_deg >= self.left_refine_max_deg:
            self.get_logger().error(
                "left_refine_min_deg must be lower than left_refine_max_deg"
            )
            return False

        if self.right_refine_min_deg >= self.right_refine_max_deg:
            self.get_logger().error(
                "right_refine_min_deg must be lower than right_refine_max_deg"
            )
            return False

        if self.target_distance_reference_xyz.shape != (3,):
            self.get_logger().error(
                "target_distance_reference_xyz must contain exactly 3 values"
            )
            return False

        return True

    def _start_oscillation(self) -> None:
        if not self.is_busy:
            return

        self._destroy_timer("initial_pose_timer")
        self.is_oscillating = True
        self.active_oscillation_min_deg = self.oscillation_min_deg
        self.active_oscillation_max_deg = self.oscillation_max_deg
        self._publish_next_oscillation_target()
        self.oscillation_timer = self.create_timer(
            self._positive_seconds(self.oscillation_horizon_sec),
            self._publish_next_oscillation_target,
        )
        joint_name = self.search_joint_names[self.oscillation_joint_index]
        self.get_logger().info(
            f"Oscillating {joint_name} between "
            f"{self.active_oscillation_min_deg:.1f} and "
            f"{self.active_oscillation_max_deg:.1f} deg"
        )

    def _publish_next_oscillation_target(self) -> None:
        if not self.is_busy or self.is_finalizing_target:
            return

        target_positions_deg = list(self.search_joint_positions_deg)
        target_positions_deg[self.oscillation_joint_index] = (
            self.next_oscillation_target_deg
        )
        self._publish_joint_positions_deg(
            target_positions_deg,
            self.oscillation_horizon_sec,
        )
        if self.next_oscillation_target_deg == self.active_oscillation_max_deg:
            self.next_oscillation_target_deg = self.active_oscillation_min_deg
        else:
            self.next_oscillation_target_deg = self.active_oscillation_max_deg

    def _publish_joint_positions_deg(
        self,
        joint_positions_deg: list[float],
        horizon_sec: float,
    ) -> None:
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.search_joint_names

        point = JointTrajectoryPoint()
        point.positions = [
            math.radians(position_deg)
            for position_deg in joint_positions_deg
        ]
        point.time_from_start = self._duration_from_seconds(
            self._positive_seconds(horizon_sec)
        )
        trajectory_msg.points = [point]

        self.trajectory_publisher.publish(trajectory_msg)
        if 0 <= self.oscillation_joint_index < len(joint_positions_deg):
            self.last_commanded_oscillation_joint_deg = float(
                joint_positions_deg[self.oscillation_joint_index]
            )

    def _on_detection_timer(self) -> None:
        if (
            not self.is_busy
            or (not self.is_oscillating and not self.is_finalizing_target)
            or (self.is_finalizing_target and self.final_target_confirmed)
        ):
            return

        if (
            self.latest_color_image is None
            or self.latest_depth_image is None
            or not self.has_camera_info
        ):
            if self.latest_color_image is not None:
                self._publish_detection_overlay(
                    self.latest_color_image,
                    [],
                    None,
                    "unknown",
                    "unknown",
                    self._get_detection_joint_angle_deg(),
                    "waiting for depth/camera info",
                )
            self._maybe_reset_after_lost_detection()
            return

        joint_angle_deg = self._get_detection_joint_angle_deg()
        side = self._side_from_joint_angle(joint_angle_deg)
        target_side = self._target_side_from_robot_side(side)
        detections = self._detect_berries(
            self.latest_color_image,
            self.latest_depth_image,
        )
        detection = self._closest_detection(detections)
        self._publish_detection_overlay(
            self.latest_color_image,
            detections,
            detection,
            side,
            target_side,
            joint_angle_deg,
            "no valid berry detections" if detection is None else "",
        )
        if detection is None:
            self._maybe_reset_after_lost_detection()
            return

        self.last_berry_detection_time = self.get_clock().now()
        if (
            not self.is_finalizing_target
            and self.target_side is None
            and target_side in ("left", "right")
        ):
            self._start_side_refinement(side, target_side)

        result = (
            f"stage=search,"
            f"side={side},"
            f"target_side={self.target_side or target_side},"
            f"joint_deg={joint_angle_deg:.2f},"
            f"camera_z_m={detection['distance_m']:.3f},"
            f"camera_x_m={detection['camera_x_m']:.3f},"
            f"camera_y_m={detection['camera_y_m']:.3f},"
            f"u={detection['center_u']:.1f},"
            f"v={detection['center_v']:.1f},"
            f"confidence={detection['confidence']:.2f},"
            f"class_id={detection['class_id']}"
        )
        self.detection_result_publisher.publish(String(data=result))

        if self.lidar_region_confirmation_active:
            self._confirm_lidar_region_camera_detection(
                detection,
                side,
                target_side,
                joint_angle_deg,
            )
            return

        if self.target_side is not None:
            self._publish_camera_point(detection)
            self._schedule_eyeinhand_compute()

        if (
            self.closest_detection_distance_m is None
            or detection["distance_m"] < self.closest_detection_distance_m
        ):
            self.closest_detection_distance_m = detection["distance_m"]
            self.get_logger().info(
                "Closest berry during search: "
                f"camera_z={detection['distance_m']:.3f} m, "
                f"side={side}, target_side={self.target_side or target_side}, "
                f"joint={joint_angle_deg:.1f} deg"
            )

    def _start_side_refinement(self, robot_side: str, target_side: str) -> None:
        self.robot_side = robot_side
        self.target_side = target_side
        (
            self.active_oscillation_min_deg,
            self.active_oscillation_max_deg,
        ) = self._refine_range_for_side(target_side)
        self.next_oscillation_target_deg = self.active_oscillation_min_deg
        self._destroy_timer("oscillation_timer")
        self.status_publisher.publish(String(data="REFINING"))
        self._publish_next_oscillation_target()
        self.oscillation_timer = self.create_timer(
            self._positive_seconds(self.oscillation_horizon_sec),
            self._publish_next_oscillation_target,
        )
        self.get_logger().info(
            f"Robot side={robot_side}, berry target_side={target_side}. "
            "Refining oscillation between "
            f"{self.active_oscillation_min_deg:.1f} and "
            f"{self.active_oscillation_max_deg:.1f} deg"
        )

    def _maybe_reset_after_lost_detection(self) -> None:
        if (
            self.target_side is None
            or self.is_finalizing_target
            or self.last_berry_detection_time is None
        ):
            return

        elapsed_sec = (
            self.get_clock().now() - self.last_berry_detection_time
        ).nanoseconds * 1e-9
        if elapsed_sec < self.berry_lost_timeout_sec:
            return

        self._reset_to_full_oscillation(elapsed_sec)

    def _reset_to_full_oscillation(self, elapsed_sec: float) -> None:
        previous_side = self.target_side
        self.robot_side = None
        self.target_side = None
        self.closest_detection_distance_m = None
        self.last_berry_detection_time = None
        self.final_target_confirmed = False
        self.latest_target_base_distance_m = None
        self.latest_target_base_time = None
        self.active_oscillation_min_deg = self.oscillation_min_deg
        self.active_oscillation_max_deg = self.oscillation_max_deg
        self.next_oscillation_target_deg = self.active_oscillation_max_deg
        self._destroy_timer("eye_compute_timer")
        self._destroy_timer("oscillation_timer")
        self.status_publisher.publish(String(data="BUSY"))
        self._publish_next_oscillation_target()
        self.oscillation_timer = self.create_timer(
            self._positive_seconds(self.oscillation_horizon_sec),
            self._publish_next_oscillation_target,
        )
        self.detection_result_publisher.publish(
            String(
                data=(
                    f"stage=reset,"
                    f"previous_side={previous_side},"
                    f"lost_for_sec={elapsed_sec:.2f}"
                )
            )
        )
        self.get_logger().warn(
            f"Berry lost for {elapsed_sec:.2f}s. "
            "Resetting to full oscillation."
        )

    def _refine_range_for_side(self, side: str) -> tuple[float, float]:
        if side == "left":
            return self.left_refine_min_deg, self.left_refine_max_deg

        return self.right_refine_min_deg, self.right_refine_max_deg

    def _final_joint_for_side(self, side: str) -> float:
        if side == "left":
            return self.left_final_deg

        return self.right_final_deg

    def _handle_target_within_stop_distance(self, target_distance_m: float) -> None:
        if self.target_side is None or self.is_finalizing_target:
            return

        self.is_finalizing_target = True
        self.is_oscillating = False
        self.final_target_confirmed = False
        self.lidar_region_confirmation_active = False
        self.lidar_region_detection_side = None
        self.lidar_region_detection_time = None
        self.agv_command_before_lidar_stop = None
        self.agv_mode_before_stop = self.agv_control_mode
        self._destroy_timer("oscillation_timer")
        self._destroy_timer("eye_compute_timer")
        self._publish_agv_stop_sequence()
        self.status_publisher.publish(String(data="HARVESTING"))
        self.final_stop_timer = self.create_timer(
            self._positive_seconds(self.agv_stop_period_sec),
            self._publish_agv_stop_sequence,
        )
        self._publish_final_side_pose()
        self.final_confirmation_timer = self.create_timer(
            self._positive_seconds(self.final_target_timeout_sec),
            self._recover_after_final_target_timeout,
        )
        self.get_logger().info(
            f"Target within {target_distance_m:.3f} m <= "
            f"{self.target_stop_distance_m:.3f} m. "
            f"AGV stop sent and cobot moving to "
            f"{self._final_joint_for_side(self.target_side):.1f} deg. "
            f"Waiting {self.final_target_timeout_sec:.1f}s for confirmation."
        )

    def _handle_lidar_region_detection(
        self,
        side: str,
        count: int,
        nearest_m: float | None,
        x_m: float | None,
        y_m: float | None,
    ) -> None:
        if not self._should_accept_lidar_region_trigger():
            return

        review_joint_deg = self._final_joint_for_side(side)
        self.lidar_region_confirmation_active = True
        self.lidar_region_detection_side = side
        self.lidar_region_detection_time = self.get_clock().now()
        self.is_finalizing_target = True
        self.is_oscillating = False
        self.final_target_confirmed = False
        self.agv_mode_before_stop = self.agv_control_mode
        self.agv_command_before_lidar_stop = self.latest_agv_command
        self._destroy_timer("oscillation_timer")
        self._destroy_timer("eye_compute_timer")
        self._publish_agv_stop_sequence()
        self.status_publisher.publish(String(data="HARVESTING"))
        self.final_stop_timer = self.create_timer(
            self._positive_seconds(self.agv_stop_period_sec),
            self._publish_agv_stop_sequence,
        )
        self._publish_lidar_region_review_pose(review_joint_deg)
        self.final_confirmation_timer = self.create_timer(
            self._positive_seconds(self.lidar_region_confirmation_timeout_sec),
            self._recover_after_lidar_region_false_alarm,
        )
        self.detection_result_publisher.publish(
            String(
                data=(
                    f"stage=lidar_region_trigger,"
                    f"region_side={side},"
                    f"count={count},"
                    f"nearest_m={self._format_optional_float(nearest_m)},"
                    f"x={self._format_optional_float(x_m)},"
                    f"y={self._format_optional_float(y_m)},"
                    f"review_joint_deg={review_joint_deg:.2f}"
                )
            )
        )
        self.get_logger().info(
            "LiDAR side clearance detection while camera has no target. "
            f"side={side}, nearest={self._format_optional_float(nearest_m)} m. "
            f"Stopping AGV and moving cobot to {review_joint_deg:.1f} deg."
        )

    def _publish_lidar_region_review_pose(self, review_joint_deg: float) -> None:
        target_positions_deg = list(self.search_joint_positions_deg)
        target_positions_deg[self.oscillation_joint_index] = review_joint_deg
        self._publish_joint_positions_deg(
            target_positions_deg,
            self.final_pose_horizon_sec,
        )

    def _confirm_lidar_region_camera_detection(
        self,
        detection: dict[str, float | int | str],
        side: str,
        target_side: str,
        joint_angle_deg: float,
    ) -> None:
        if (
            not self.lidar_region_confirmation_active
            or self.final_target_confirmed
        ):
            return

        self.final_target_confirmed = True
        self._destroy_timer("final_confirmation_timer")
        self._publish_agv_stop_sequence()
        self.status_publisher.publish(String(data="HARVESTING"))
        self.detection_result_publisher.publish(
            String(
                data=(
                    f"stage=lidar_region_camera_confirmed,"
                    f"region_side={self.lidar_region_detection_side or 'unknown'},"
                    f"side={side},"
                    f"target_side={target_side},"
                    f"joint_deg={joint_angle_deg:.2f},"
                    f"camera_z_m={float(detection['distance_m']):.3f},"
                    f"camera_x_m={float(detection['camera_x_m']):.3f},"
                    f"camera_y_m={float(detection['camera_y_m']):.3f},"
                    f"confidence={float(detection['confidence']):.2f}"
                )
            )
        )
        self.get_logger().info(
            "LiDAR region trigger confirmed by camera. "
            "Keeping HARVESTING active and AGV stopped."
        )

    def _recover_after_lidar_region_false_alarm(self) -> None:
        self._destroy_timer("final_confirmation_timer")
        if (
            not self.lidar_region_confirmation_active
            or self.final_target_confirmed
        ):
            return

        previous_side = self.lidar_region_detection_side
        previous_mode = self.agv_mode_before_stop
        previous_command = self.agv_command_before_lidar_stop
        self._destroy_timer("final_stop_timer")

        self.lidar_region_confirmation_active = False
        self.lidar_region_detection_side = None
        self.lidar_region_detection_time = None
        self.is_finalizing_target = False
        self.is_oscillating = True
        self.final_target_confirmed = False
        self._reset_to_full_oscillation(self.lidar_region_confirmation_timeout_sec)
        self._restore_agv_after_lidar_false_alarm(previous_mode, previous_command)
        self.detection_result_publisher.publish(
            String(
                data=(
                    f"stage=lidar_region_false_alarm,"
                    f"previous_region_side={previous_side or 'unknown'},"
                    f"restored_mode={previous_mode or 'unknown'},"
                    f"restored_command={previous_command or 'none'}"
                )
            )
        )
        self.agv_mode_before_stop = None
        self.agv_command_before_lidar_stop = None
        self.get_logger().warn(
            "No camera berry found after LiDAR side region trigger. "
            "Treating it as a false alarm and resuming search."
        )

    def _restore_agv_after_lidar_false_alarm(
        self,
        previous_mode: str | None,
        previous_command: str | None,
    ) -> None:
        if previous_mode == AGV_CONTROL_MODE_AUTOMATIC:
            self._publish_agv_automatic()
            return

        self._publish_agv_manual()
        if previous_command:
            self.agv_cmd_publisher.publish(String(data=previous_command))

    def _confirm_final_target(self, target_distance_m: float) -> None:
        if not self.is_finalizing_target or self.final_target_confirmed:
            return

        self.final_target_confirmed = True
        self._destroy_timer("final_confirmation_timer")
        self._publish_agv_stop_sequence()
        self.detection_result_publisher.publish(
            String(
                data=(
                    f"stage=final_confirmed,"
                    f"side={self.robot_side or 'unknown'},"
                    f"target_side={self.target_side},"
                    f"target_distance_m={target_distance_m:.3f}"
                )
            )
        )
        self.get_logger().info(
            f"Final target confirmed at {target_distance_m:.3f} m. DONE_OK"
        )
        self._finish("DONE_OK")

    def _recover_after_final_target_timeout(self) -> None:
        self._destroy_timer("final_confirmation_timer")
        if not self.is_finalizing_target or self.final_target_confirmed:
            return

        previous_robot_side = self.robot_side
        previous_target_side = self.target_side
        should_restore_automatic = (
            self.agv_mode_before_stop == AGV_CONTROL_MODE_AUTOMATIC
        )
        self._destroy_timer("final_stop_timer")
        if should_restore_automatic:
            self._publish_agv_automatic()

        self.is_finalizing_target = False
        self.is_oscillating = True
        self.agv_mode_before_stop = None
        self._reset_to_full_oscillation(self.final_target_timeout_sec)
        self.detection_result_publisher.publish(
            String(
                data=(
                    f"stage=final_timeout,"
                    f"previous_side={previous_robot_side},"
                    f"previous_target_side={previous_target_side},"
                    f"restored_automatic={self._format_bool(should_restore_automatic)}"
                )
            )
        )
        self.get_logger().warn(
            "No transformed berry under "
            f"{self.target_stop_distance_m:.3f} m during final "
            f"{self.final_target_timeout_sec:.1f}s window. "
            "Restarting full oscillation."
        )

    def _publish_agv_stop(self) -> None:
        stop_command = self.agv_stop_command or "s"
        self.agv_cmd_publisher.publish(String(data=stop_command))

    def _publish_agv_manual(self) -> None:
        command = self.agv_manual_command or AGV_CONTROL_MODE_MANUAL
        self.agv_control_mode_publisher.publish(String(data=command))

    def _publish_agv_stop_sequence(self) -> None:
        self._publish_agv_manual()
        self._publish_agv_stop()

    def _publish_agv_automatic(self) -> None:
        command = self.agv_automatic_command or AGV_CONTROL_MODE_AUTOMATIC
        self.agv_control_mode_publisher.publish(String(data=command))

    def _publish_final_side_pose(self) -> None:
        if self.target_side is None:
            return

        target_positions_deg = list(self.search_joint_positions_deg)
        target_positions_deg[self.oscillation_joint_index] = (
            self._final_joint_for_side(self.target_side)
        )
        self._publish_joint_positions_deg(
            target_positions_deg,
            self.final_pose_horizon_sec,
        )

    def _detect_berries(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
    ) -> list[dict[str, float | int | str]]:
        image_height, image_width = color_image.shape[:2]

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(
            rgb_image,
            conf=self.confidence_threshold,
            device=self.yolo_device or None,
            iou=self.nms_threshold,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        yolo_boxes = getattr(result, "boxes", None)
        if yolo_boxes is None or len(yolo_boxes) == 0:
            return []

        xyxy_boxes = self._to_numpy(yolo_boxes.xyxy)
        confidences = self._to_numpy(yolo_boxes.conf)
        class_ids = self._to_numpy(yolo_boxes.cls).astype(int)

        detections = []
        for xyxy, confidence, class_id in zip(xyxy_boxes, confidences, class_ids):
            class_id = int(class_id)
            if class_id != self.target_class_id:
                continue

            confidence = float(confidence)
            if confidence < self.confidence_threshold:
                continue

            x0 = max(0.0, min(float(image_width - 1), float(xyxy[0])))
            y0 = max(0.0, min(float(image_height - 1), float(xyxy[1])))
            x1 = max(0.0, min(float(image_width - 1), float(xyxy[2])))
            y1 = max(0.0, min(float(image_height - 1), float(xyxy[3])))
            if x1 <= x0 or y1 <= y0:
                continue

            x = int(round(x0))
            y = int(round(y0))
            box_width = max(1, int(round(x1 - x0)))
            box_height = max(1, int(round(y1 - y0)))
            center_u = float((x0 + x1) / 2.0)
            center_v = float((y0 + y1) / 2.0)

            depth_m = self._depth_from_bbox(
                depth_image,
                x,
                y,
                box_width,
                box_height,
            )
            if depth_m is None:
                depth_m = self._depth_from_center_patch(
                    depth_image,
                    int(round(center_u)),
                    int(round(center_v)),
                    self.center_patch_half_window,
                )

            if depth_m is None:
                continue

            camera_x_m = (center_u - self.ppx) * depth_m / self.fx
            camera_y_m = (center_v - self.ppy) * depth_m / self.fy
            detections.append(
                {
                    "distance_m": float(depth_m),
                    "camera_x_m": float(camera_x_m),
                    "camera_y_m": float(camera_y_m),
                    "center_u": center_u,
                    "center_v": center_v,
                    "confidence": confidence,
                    "class_id": int(class_id),
                    "class_name": self._class_name(class_id),
                    "bbox_x": int(x),
                    "bbox_y": int(y),
                    "bbox_w": int(box_width),
                    "bbox_h": int(box_height),
                }
            )

        return detections

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()

        return np.asarray(value)

    def _resolve_yolo_device(self, requested_device: str) -> str:
        device = str(requested_device).strip()
        if device.lower() in ("", "none", "auto"):
            return ""

        if device.lower() == "cpu":
            return "cpu"

        try:
            import torch
        except ImportError:
            self.get_logger().warn(
                f"YOLO device '{device}' requested but torch is not installed; using CPU"
            )
            return "cpu"

        if torch.cuda.is_available():
            return device

        self.get_logger().warn(
            f"YOLO device '{device}' requested but CUDA is not available; using CPU"
        )
        return "cpu"

    @staticmethod
    def _normalize_class_names(class_names) -> list[str]:
        if isinstance(class_names, dict):
            return [
                str(class_names[key])
                for key in sorted(class_names, key=lambda item: int(item))
            ]

        if isinstance(class_names, (list, tuple)):
            return [str(class_name) for class_name in class_names]

        return []

    @staticmethod
    def _closest_detection(
        detections: list[dict[str, float | int | str]],
    ) -> dict[str, float | int | str] | None:
        if not detections:
            return None

        return min(detections, key=lambda item: float(item["distance_m"]))

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id] or f"class_{class_id}"

        return f"class_{class_id}"

    def _publish_detection_overlay(
        self,
        color_image: np.ndarray,
        detections: list[dict[str, float | int | str]],
        closest_detection: dict[str, float | int | str] | None,
        side: str,
        target_side: str,
        joint_angle_deg: float,
        status_text: str,
    ) -> None:
        annotated_image = color_image.copy()
        image_height, image_width = annotated_image.shape[:2]

        for detection in detections:
            x = int(detection["bbox_x"])
            y = int(detection["bbox_y"])
            box_width = int(detection["bbox_w"])
            box_height = int(detection["bbox_h"])
            x0 = max(0, min(image_width - 1, x))
            y0 = max(0, min(image_height - 1, y))
            x1 = max(0, min(image_width - 1, x + box_width))
            y1 = max(0, min(image_height - 1, y + box_height))
            if x1 <= x0 or y1 <= y0:
                continue

            is_closest = detection is closest_detection
            color = (0, 255, 255) if is_closest else (0, 180, 0)
            thickness = 3 if is_closest else 2
            cv2.rectangle(annotated_image, (x0, y0), (x1, y1), color, thickness)

            label_lines = [
                f"{detection['class_name']} {float(detection['confidence']):.2f}",
                f"z_cam={float(detection['distance_m']):.2f}m",
                f"side={side} target={target_side}",
            ]
            if is_closest:
                label_lines.append("closest")
                if self.latest_target_base_distance_m is not None:
                    label_lines.append(
                        f"base={self.latest_target_base_distance_m:.2f}m"
                    )
            self._draw_text_lines(
                annotated_image,
                label_lines,
                x0 + 4,
                y0 + 18,
                color,
            )

        header_lines = [
            f"berries={len(detections)} side={side} target_side={self.target_side or target_side}",
            f"joint_1={joint_angle_deg:.1f}deg",
        ]
        if self.latest_target_base_distance_m is not None:
            header_lines.append(
                f"base_distance={self.latest_target_base_distance_m:.2f}m"
            )
        if status_text:
            header_lines.append(status_text)
        self._draw_text_lines(annotated_image, header_lines, 10, 24, (255, 255, 255))

        try:
            image_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = self.camera_frame_id
            self.annotated_image_publisher.publish(image_msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish annotated image: {exc}")

        if self.show_preview:
            try:
                cv2.imshow(self.preview_window_name, annotated_image)
                cv2.waitKey(1)
            except Exception as exc:
                self.get_logger().warn(f"OpenCV preview disabled: {exc}")
                self.show_preview = False

    @staticmethod
    def _draw_text_lines(
        image: np.ndarray,
        lines: list[str],
        x: int,
        first_baseline_y: int,
        text_color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        line_gap = 5
        y = int(first_baseline_y)
        x = max(0, int(x))

        for line in lines:
            text = str(line)
            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                font,
                scale,
                thickness,
            )
            top_left = (x - 2, max(0, y - text_height - 3))
            bottom_right = (
                min(image.shape[1] - 1, x + text_width + 2),
                min(image.shape[0] - 1, y + baseline + 3),
            )
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
            cv2.putText(
                image,
                text,
                (x, y),
                font,
                scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )
            y += text_height + baseline + line_gap

    def _publish_camera_point(self, detection: dict[str, float]) -> None:
        target_msg = PointStamped()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = self.camera_frame_id
        target_msg.point.x = float(detection["camera_x_m"])
        target_msg.point.y = float(detection["camera_y_m"])
        target_msg.point.z = float(detection["distance_m"])
        self.point_publisher.publish(target_msg)

    def _schedule_eyeinhand_compute(self) -> None:
        if not self.trigger_eyeinhand_compute:
            return

        self._destroy_timer("eye_compute_timer")
        self.eye_compute_timer = self.create_timer(
            self._positive_seconds(self.eye_compute_delay_sec),
            self._publish_eyeinhand_compute,
        )

    def _publish_eyeinhand_compute(self) -> None:
        self._destroy_timer("eye_compute_timer")
        self.eye_cmd_publisher.publish(String(data="COMPUTE"))

    def _depth_to_meters(self, depth_value) -> float:
        if (
            isinstance(self.latest_depth_image, np.ndarray)
            and self.latest_depth_image.dtype == np.float32
        ):
            return float(depth_value)

        return float(depth_value) * float(self.depth_scale_m_per_unit)

    def _depth_from_bbox(
        self,
        depth_image: np.ndarray,
        x: int,
        y: int,
        box_width: int,
        box_height: int,
    ) -> float | None:
        image_height, image_width = depth_image.shape[:2]

        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(image_width - 1, int(x + box_width))
        y1 = min(image_height - 1, int(y + box_height))
        if x1 <= x0 or y1 <= y0:
            return None

        step_x = max(1, (x1 - x0) // max(1, self.depth_grid_div))
        step_y = max(1, (y1 - y0) // max(1, self.depth_grid_div))
        valid_depths_m = []

        for row in range(y0, y1, step_y):
            for col in range(x0, x1, step_x):
                scalar_depth_value = float(np.asarray(depth_image[row, col]).item())
                if scalar_depth_value <= 0.0:
                    continue

                depth_m = self._depth_to_meters(scalar_depth_value)
                if self.min_valid_depth_m <= depth_m <= self.max_valid_depth_m:
                    valid_depths_m.append(depth_m)

        if len(valid_depths_m) < self.min_depth_samples:
            return None

        sorted_depths = np.sort(np.array(valid_depths_m, dtype=np.float32))
        keep_count = max(1, int(0.7 * len(sorted_depths)))
        return float(np.median(sorted_depths[:keep_count]))

    def _depth_from_center_patch(
        self,
        depth_image: np.ndarray,
        center_u: int,
        center_v: int,
        half_window: int,
    ) -> float | None:
        image_height, image_width = depth_image.shape[:2]

        x0 = max(0, center_u - half_window)
        x1 = min(image_width - 1, center_u + half_window)
        y0 = max(0, center_v - half_window)
        y1 = min(image_height - 1, center_v + half_window)
        if x1 <= x0 or y1 <= y0:
            return None

        patch = depth_image[y0:y1, x0:x1]
        if patch.size < 1:
            return None

        valid_depths_m = []
        for depth_value in patch.reshape(-1):
            scalar_depth_value = float(np.asarray(depth_value).item())
            if scalar_depth_value <= 0.0:
                continue

            depth_m = self._depth_to_meters(scalar_depth_value)
            if self.min_valid_depth_m <= depth_m <= self.max_valid_depth_m:
                valid_depths_m.append(depth_m)

        if len(valid_depths_m) < max(5, self.min_depth_samples // 2):
            return None

        sorted_depths = np.sort(np.array(valid_depths_m, dtype=np.float32))
        keep_count = max(1, int(0.7 * len(sorted_depths)))
        return float(np.median(sorted_depths[:keep_count]))

    def _get_detection_joint_angle_deg(self) -> float:
        if self._has_fresh_joint_state():
            return float(self.current_oscillation_joint_deg)

        if self.last_commanded_oscillation_joint_deg is not None:
            return float(self.last_commanded_oscillation_joint_deg)

        return 0.0

    def _has_fresh_joint_state(self) -> bool:
        if (
            self.current_oscillation_joint_deg is None
            or self.current_oscillation_joint_time is None
        ):
            return False

        age_sec = (
            self.get_clock().now() - self.current_oscillation_joint_time
        ).nanoseconds * 1e-9
        return age_sec <= max(0.0, self.joint_state_max_age_sec)

    def _side_from_joint_angle(self, joint_angle_deg: float) -> str:
        if joint_angle_deg < -abs(self.side_deadband_deg):
            return self.negative_joint_side

        if joint_angle_deg > abs(self.side_deadband_deg):
            return "left" if self.negative_joint_side == "right" else "right"

        return "center"

    @staticmethod
    def _target_side_from_robot_side(robot_side: str) -> str:
        if robot_side == "left":
            return "right"

        if robot_side == "right":
            return "left"

        return "center"

    def _has_recent_camera_detection(self) -> bool:
        if self.last_berry_detection_time is None:
            return False

        age_sec = (
            self.get_clock().now() - self.last_berry_detection_time
        ).nanoseconds * 1e-9
        return age_sec <= max(0.0, self.camera_detection_recent_timeout_sec)

    @staticmethod
    def _parse_key_value_message(data: str) -> dict[str, str]:
        values = {}

        for item in data.split(","):
            if not item:
                continue

            key, separator, value = item.partition("=")
            if not separator:
                raise ValueError(f"missing '=' in '{item}'")

            values[key.strip()] = value.strip()

        return values

    @staticmethod
    def _parse_optional_float(value: str | None) -> float | None:
        if value is None or value.strip().lower() == "none":
            return None

        return float(value)

    @staticmethod
    def _parse_bool(value: str | None) -> bool:
        if value is None:
            return False

        return value.strip().lower() in ("true", "1", "yes", "on")

    @staticmethod
    def _format_optional_float(value: float | None) -> str:
        if value is None:
            return "none"

        return f"{value:.3f}"

    @staticmethod
    def _format_bool(value: bool) -> str:
        return "true" if value else "false"

    def _finish(self, status: str) -> None:
        self.is_busy = False
        self.is_oscillating = False
        self.is_finalizing_target = False
        self.final_target_confirmed = False
        self.latest_target_base_distance_m = None
        self.latest_target_base_time = None
        self.lidar_region_confirmation_active = False
        self.lidar_region_detection_side = None
        self.lidar_region_detection_time = None
        self.agv_mode_before_stop = None
        self.agv_command_before_lidar_stop = None
        self.robot_side = None
        self.target_side = None
        self._destroy_timer("initial_pose_timer")
        self._destroy_timer("oscillation_timer")
        self._destroy_timer("eye_compute_timer")
        self._destroy_timer("final_confirmation_timer")
        self._destroy_timer("final_stop_timer")
        self.status_publisher.publish(String(data=status))
        self.status_publisher.publish(String(data="IDLE"))

    def _destroy_timer(self, timer_attr_name: str) -> None:
        timer = getattr(self, timer_attr_name)
        if timer is None:
            return

        self.destroy_timer(timer)
        setattr(self, timer_attr_name, None)

    @staticmethod
    def _positive_seconds(seconds: float) -> float:
        return max(float(seconds), 0.001)

    @staticmethod
    def _duration_from_seconds(seconds: float) -> Duration:
        whole_seconds = int(seconds)
        return Duration(
            sec=whole_seconds,
            nanosec=int((seconds - whole_seconds) * 1e9),
        )

    def destroy_node(self) -> None:
        if self.show_preview:
            try:
                cv2.destroyWindow(self.preview_window_name)
            except Exception:
                pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SearchingModeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
