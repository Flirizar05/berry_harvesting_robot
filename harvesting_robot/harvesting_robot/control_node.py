#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    from tf2_geometry_msgs import do_transform_pose_stamped
    from tf2_ros import Buffer, TransformListener

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Build a homogeneous rotation matrix from an axis-angle representation."""
    normalized_axis = axis / max(np.linalg.norm(axis), 1e-12)
    ux, uy, uz = normalized_axis

    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    one_minus_cos = 1.0 - cos_theta

    rotation = np.array([
        [cos_theta + ux * ux * one_minus_cos, ux * uy * one_minus_cos - uz * sin_theta, ux * uz * one_minus_cos + uy * sin_theta],
        [uy * ux * one_minus_cos + uz * sin_theta, cos_theta + uy * uy * one_minus_cos, uy * uz * one_minus_cos - ux * sin_theta],
        [uz * ux * one_minus_cos - uy * sin_theta, uz * uy * one_minus_cos + ux * sin_theta, cos_theta + uz * uz * one_minus_cos],
    ])

    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


def forward_kinematics_and_jacobian(
    joint_positions: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    joint_axes: np.ndarray,
    ee_transform: np.ndarray | None = None,
):
    """Compute TCP position, TCP orientation, and geometric Jacobian."""
    if ee_transform is None:
        ee_transform = np.eye(4)

    num_joints = len(joint_positions)
    total_transform = np.eye(4)
    joint_origins = np.zeros((3, num_joints))
    joint_rotations = np.zeros((3, 3, num_joints))

    for joint_index in range(num_joints):
        transform_i = (
            U[:, :, joint_index]
            @ rotation_from_axis_angle(joint_axes[:, joint_index], joint_positions[joint_index])
            @ V[:, :, joint_index]
        )
        total_transform = total_transform @ transform_i
        joint_rotations[:, :, joint_index] = total_transform[:3, :3]
        joint_origins[:, joint_index] = total_transform[:3, 3]

    tcp_transform = total_transform @ ee_transform
    tcp_position = tcp_transform[:3, 3]
    tcp_rotation = tcp_transform[:3, :3]

    jacobian_linear = np.zeros((3, num_joints))
    jacobian_angular = np.zeros((3, num_joints))

    for joint_index in range(num_joints):
        if joint_index == 0:
            previous_origin = np.zeros(3)
            previous_rotation = np.eye(3)
        else:
            previous_origin = joint_origins[:, joint_index - 1]
            previous_rotation = joint_rotations[:, :, joint_index - 1]

        axis_world = previous_rotation @ U[:3, :3, joint_index] @ joint_axes[:, joint_index]
        jacobian_linear[:, joint_index] = np.cross(axis_world, tcp_position - previous_origin)
        jacobian_angular[:, joint_index] = axis_world

    return tcp_position, tcp_rotation, jacobian_linear, jacobian_angular


def parse_xyz(text: str | None) -> np.ndarray:
    if text is None:
        return np.zeros(3)
    values = [float(value) for value in text.replace(",", " ").split()]
    return np.array((values + [0.0, 0.0, 0.0])[:3])


def parse_rpy(text: str | None) -> np.ndarray:
    if text is None:
        return np.zeros(3)
    values = [float(value) for value in text.replace(",", " ").split()]
    return np.array((values + [0.0, 0.0, 0.0])[:3])


def rpy_to_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    roll_cos, pitch_cos, yaw_cos = math.cos(rpy[0]), math.cos(rpy[1]), math.cos(rpy[2])
    roll_sin, pitch_sin, yaw_sin = math.sin(rpy[0]), math.sin(rpy[1]), math.sin(rpy[2])

    rotation_z = np.array([
        [yaw_cos, -yaw_sin, 0],
        [yaw_sin, yaw_cos, 0],
        [0, 0, 1],
    ])
    rotation_y = np.array([
        [pitch_cos, 0, pitch_sin],
        [0, 1, 0],
        [-pitch_sin, 0, pitch_cos],
    ])
    rotation_x = np.array([
        [1, 0, 0],
        [0, roll_cos, -roll_sin],
        [0, roll_sin, roll_cos],
    ])

    transform = np.eye(4)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = xyz
    return transform


def load_chain_from_urdf(
    urdf_path: str,
    joint_names: list[str],
    ee_link_name: str | None = None,
):
    """Extract chain transforms, joint axes, limits, and fixed EE offset from a URDF."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    namespace = "" if root.tag == "robot" else "{" + root.tag.split("}")[0].strip("{") + "}"
    joints_by_name = {joint.attrib["name"]: joint for joint in root.findall(f"{namespace}joint")}

    num_joints = len(joint_names)
    U = np.zeros((4, 4, num_joints))
    V = np.zeros((4, 4, num_joints))
    joint_axes = np.zeros((3, num_joints))
    joint_min = -np.pi * np.ones(num_joints)
    joint_max = np.pi * np.ones(num_joints)

    fixed_children_by_parent = {}
    joint_child_links = {}

    for joint in root.findall(f"{namespace}joint"):
        parent_element = joint.find(f"{namespace}parent")
        child_element = joint.find(f"{namespace}child")
        if parent_element is None or child_element is None:
            continue

        parent_link = parent_element.attrib["link"]
        child_link = child_element.attrib["link"]
        joint_child_links[joint.attrib["name"]] = child_link

        if joint.attrib.get("type", "") == "fixed":
            fixed_children_by_parent.setdefault(parent_link, []).append(joint)

    for joint_index, joint_name in enumerate(joint_names):
        if joint_name not in joints_by_name:
            raise KeyError(f"Joint '{joint_name}' not found in URDF")

        joint = joints_by_name[joint_name]

        origin_element = joint.find(f"{namespace}origin")
        xyz = parse_xyz(origin_element.attrib.get("xyz")) if origin_element is not None else np.zeros(3)
        rpy = parse_rpy(origin_element.attrib.get("rpy")) if origin_element is not None else np.zeros(3)

        U[:, :, joint_index] = rpy_to_transform(xyz, rpy)
        V[:, :, joint_index] = np.eye(4)

        axis_element = joint.find(f"{namespace}axis")
        axis = (
            parse_xyz(axis_element.attrib["xyz"])
            if axis_element is not None and "xyz" in axis_element.attrib
            else np.array([0, 0, 1], dtype=float)
        )
        joint_axes[:, joint_index] = axis / max(np.linalg.norm(axis), 1e-12)

        limit_element = joint.find(f"{namespace}limit")
        if limit_element is not None and "lower" in limit_element.attrib and "upper" in limit_element.attrib:
            joint_min[joint_index] = float(limit_element.attrib["lower"])
            joint_max[joint_index] = float(limit_element.attrib["upper"])

    ee_transform = np.eye(4)
    if ee_link_name:
        last_joint_name = joint_names[-1]
        if last_joint_name in joint_child_links:
            start_link = joint_child_links[last_joint_name]
            stack = [(start_link, np.eye(4))]
            visited = set()

            while stack:
                link_name, accumulated_transform = stack.pop()
                if link_name in visited:
                    continue
                visited.add(link_name)

                if link_name == ee_link_name:
                    ee_transform = accumulated_transform
                    break

                for joint in fixed_children_by_parent.get(link_name, []):
                    origin_element = joint.find(f"{namespace}origin")
                    xyz = parse_xyz(origin_element.attrib.get("xyz")) if origin_element is not None else np.zeros(3)
                    rpy = parse_rpy(origin_element.attrib.get("rpy")) if origin_element is not None else np.zeros(3)
                    joint_transform = rpy_to_transform(xyz, rpy)
                    child_link = joint.find(f"{namespace}child").attrib["link"]
                    stack.append((child_link, accumulated_transform @ joint_transform))

    return U, V, joint_axes, joint_min, joint_max, ee_transform


class ControlNode(Node):
    def __init__(self) -> None:
        super().__init__("control_node")

        self._set_default_configuration()
        self._declare_parameters()
        self._load_parameters()
        self._initialize_tf()
        self._load_kinematics()
        self._initialize_runtime_state()
        self._create_ros_interfaces()

        self.timer = self.create_timer(self.control_dt, self._control_loop)
        self.status_publisher.publish(String(data="IDLE"))

        self.get_logger().info(
            "control_node ready (position priority + axis alignment). "
            f"dt={self.control_dt:.3f}s horizon={self.command_horizon_sec:.3f}s | "
            f"kp_pos={self.kp_pos:.2f} kp_ori={self.kp_ori:.2f} "
            f"damp_pos={self.damp_pos:.3f} damp_ori={self.damp_ori:.3f} | "
            f"pos_tol={self.position_tolerance_m:.3f} "
            f"settle_cycles={self.settle_cycles} "
            f"done_wp_stale_sec={self.done_waypoint_stale_sec:.2f}"
        )

    def _set_default_configuration(self) -> None:
        package_share = get_package_share_directory("harvesting_robot")
        self.default_urdf_path = os.path.join(package_share, "urdf", "elfin3.urdf")

        self.default_joint_names = [
            "elfin_joint1",
            "elfin_joint2",
            "elfin_joint3",
            "elfin_joint4",
            "elfin_joint5",
            "elfin_joint6",
        ]
        self.default_ee_link = "rg2ft_grasp_point"

        self.default_joint_state_topic = "/joint_states"
        self.default_controller_topic = "/elfin_arm_controller/joint_trajectory"

        self.default_dt = 0.02
        self.default_command_horizon_sec = 0.05

        self.default_kp_pos = 8.0
        self.default_kp_ori = 1.5
        self.default_damp_pos = 0.12
        self.default_damp_ori = 0.12

        self.default_joint_weights = np.array([5.0, 1.5, 1.0, 3.0, 1.0, 6.0], dtype=float)
        self.default_max_joint_step_rad = 0.02

        self.default_base_frame = "elfin_base"
        self.default_use_tf = True and TF_AVAILABLE

        self.default_waypoint_topic = "/trajectory/waypoint"
        self.default_command_topic = "/control/cmd"
        self.default_status_topic = "/control/status"

        self.default_execute_timeout_sec = 120.0
        self.default_waypoint_timeout_sec = 30.0
        self.default_position_tolerance_m = 0.02
        self.default_settle_cycles = 20
        self.default_done_waypoint_stale_sec = 0.30

        self.default_enable_nullspace = True
        self.default_nullspace_gain = 1.0
        self.default_limit_margin_rad = 0.30
        self.default_limit_push_gain = 8.0

        self.default_saturation_eps = 1e-12
        self.default_saturation_hold_cycles = 40

        self.default_log_period_sec = 0.5

        self.local_alignment_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        self.desired_alignment_axis = np.array([0.0, -1.0, 0.0], dtype=float)

    def _declare_parameters(self) -> None:
        self.declare_parameter("urdf_path", self.default_urdf_path)
        self.declare_parameter("joint_names", self.default_joint_names)
        self.declare_parameter("ee_link", self.default_ee_link)
        self.declare_parameter("joint_state_topic", self.default_joint_state_topic)
        self.declare_parameter("controller_topic", self.default_controller_topic)

        self.declare_parameter("dt", self.default_dt)
        self.declare_parameter("command_horizon_sec", self.default_command_horizon_sec)

        self.declare_parameter("kp_pos", self.default_kp_pos)
        self.declare_parameter("kp_ori", self.default_kp_ori)
        self.declare_parameter("damp_pos", self.default_damp_pos)
        self.declare_parameter("damp_ori", self.default_damp_ori)

        self.declare_parameter("joint_weights", self.default_joint_weights.tolist())
        self.declare_parameter("max_joint_step_rad", self.default_max_joint_step_rad)

        self.declare_parameter("base_frame", self.default_base_frame)
        self.declare_parameter("use_tf", bool(self.default_use_tf))

        self.declare_parameter("waypoint_topic", self.default_waypoint_topic)
        self.declare_parameter("cmd_topic", self.default_command_topic)
        self.declare_parameter("status_topic", self.default_status_topic)

        self.declare_parameter("execute_timeout_sec", self.default_execute_timeout_sec)
        self.declare_parameter("waypoint_timeout_sec", self.default_waypoint_timeout_sec)
        self.declare_parameter("pos_tol_m", self.default_position_tolerance_m)
        self.declare_parameter("settle_cycles", self.default_settle_cycles)
        self.declare_parameter("done_wp_stale_sec", self.default_done_waypoint_stale_sec)

        self.declare_parameter("enable_nullspace", bool(self.default_enable_nullspace))
        self.declare_parameter("nullspace_gain", self.default_nullspace_gain)
        self.declare_parameter("limit_margin_rad", self.default_limit_margin_rad)
        self.declare_parameter("limit_push_gain", self.default_limit_push_gain)

        self.declare_parameter("log_period_sec", self.default_log_period_sec)
        self.declare_parameter("sat_eps", self.default_saturation_eps)
        self.declare_parameter("sat_hold_cycles", self.default_saturation_hold_cycles)

    def _load_parameters(self) -> None:
        self.urdf_path = str(self.get_parameter("urdf_path").value)
        self.joint_names = list(self.get_parameter("joint_names").value)
        self.ee_link = self.get_parameter("ee_link").value or None

        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.controller_topic = str(self.get_parameter("controller_topic").value)

        self.control_dt = float(self.get_parameter("dt").value)
        self.command_horizon_sec = float(self.get_parameter("command_horizon_sec").value)

        self.kp_pos = float(self.get_parameter("kp_pos").value)
        self.kp_ori = float(self.get_parameter("kp_ori").value)
        self.damp_pos = float(self.get_parameter("damp_pos").value)
        self.damp_ori = float(self.get_parameter("damp_ori").value)

        self.joint_weights = np.array(self.get_parameter("joint_weights").value, dtype=float)
        if self.joint_weights.size != len(self.joint_names):
            self.get_logger().warn(
                "joint_weights size does not match the number of joints. Falling back to all-ones."
            )
            self.joint_weights = np.ones(len(self.joint_names), dtype=float)

        self.max_joint_step_rad = float(self.get_parameter("max_joint_step_rad").value)

        self.base_frame = str(self.get_parameter("base_frame").value)
        requested_use_tf = bool(self.get_parameter("use_tf").value)
        self.use_tf = requested_use_tf and TF_AVAILABLE

        self.waypoint_topic = str(self.get_parameter("waypoint_topic").value)
        self.command_topic = str(self.get_parameter("cmd_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        self.execute_timeout_sec = float(self.get_parameter("execute_timeout_sec").value)
        self.waypoint_timeout_sec = float(self.get_parameter("waypoint_timeout_sec").value)
        self.position_tolerance_m = float(self.get_parameter("pos_tol_m").value)
        self.settle_cycles = int(self.get_parameter("settle_cycles").value)
        self.done_waypoint_stale_sec = float(self.get_parameter("done_wp_stale_sec").value)

        self.enable_nullspace = bool(self.get_parameter("enable_nullspace").value)
        self.nullspace_gain = float(self.get_parameter("nullspace_gain").value)
        self.limit_margin_rad = float(self.get_parameter("limit_margin_rad").value)
        self.limit_push_gain = float(self.get_parameter("limit_push_gain").value)

        self.log_period_sec = float(self.get_parameter("log_period_sec").value)
        self.saturation_eps = float(self.get_parameter("sat_eps").value)
        self.saturation_hold_cycles = int(self.get_parameter("sat_hold_cycles").value)

    def _initialize_tf(self) -> None:
        if requested := bool(self.get_parameter("use_tf").value):
            if not TF_AVAILABLE:
                self.get_logger().warn(
                    "use_tf=true but tf2_geometry_msgs is not available. Continuing without TF support."
                )

        self.tf_buffer = None
        self.tf_listener = None
        if self.use_tf:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

    def _load_kinematics(self) -> None:
        (
            self.U,
            self.V,
            self.joint_axes,
            self.joint_min,
            self.joint_max,
            self.ee_transform,
        ) = load_chain_from_urdf(
            self.urdf_path,
            self.joint_names,
            self.ee_link,
        )

        self.inverse_weight_matrix = np.diag(1.0 / np.maximum(self.joint_weights, 1e-9))
        self.joint_midpoints = 0.5 * (self.joint_min + self.joint_max)
        self.joint_ranges = np.maximum(self.joint_max - self.joint_min, 1e-6)

    def _initialize_runtime_state(self) -> None:
        self.current_joint_positions = None
        self.current_waypoint = None
        self.last_waypoint_time = None
        self.last_waypoint_frame = None

        self.is_executing = False
        self.execution_start_time = None
        self.settle_count = 0
        self.saturation_count = 0
        self.last_log_time = None
        self.warned_about_waypoint_frame = False

        self.stream_mode = False

    def _create_ros_interfaces(self) -> None:
        self.create_subscription(JointState, self.joint_state_topic, self._on_joint_state, 50)
        self.create_subscription(PoseStamped, self.waypoint_topic, self._on_waypoint, 50)
        self.create_subscription(String, self.command_topic, self._on_command, 10)

        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            self.controller_topic,
            10,
        )
        self.error_norm_publisher = self.create_publisher(
            Float64,
            "/control/control_error_norm",
            10,
        )
        self.status_publisher = self.create_publisher(String, self.status_topic, 10)

    def _on_joint_state(self, msg: JointState) -> None:
        joint_map = {name: position for name, position in zip(msg.name, msg.position)}
        if not all(joint_name in joint_map for joint_name in self.joint_names):
            return

        self.current_joint_positions = np.array(
            [joint_map[joint_name] for joint_name in self.joint_names],
            dtype=float,
        )

    def _on_waypoint(self, msg: PoseStamped) -> None:
        source_frame = (msg.header.frame_id or "").strip()
        self.last_waypoint_frame = source_frame
        self.last_waypoint_time = self.get_clock().now()

        if source_frame == "" or source_frame == self.base_frame or source_frame.endswith("/" + self.base_frame):
            self.current_waypoint = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                dtype=float,
            )
            return

        if self.use_tf:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    source_frame,
                    msg.header.stamp,
                    timeout=rclpy.duration.Duration(seconds=0.1),
                )
                transformed_msg = do_transform_pose_stamped(msg, transform)
                self.current_waypoint = np.array(
                    [
                        transformed_msg.pose.position.x,
                        transformed_msg.pose.position.y,
                        transformed_msg.pose.position.z,
                    ],
                    dtype=float,
                )
                return
            except Exception as exc:
                if not self.warned_about_waypoint_frame:
                    self.get_logger().warn(
                        f"Failed to transform waypoint from '{source_frame}' to '{self.base_frame}': {exc}"
                    )
                    self.warned_about_waypoint_frame = True
                return

        if not self.warned_about_waypoint_frame:
            self.get_logger().warn(
                f"Waypoint frame_id='{source_frame}' does not match base_frame='{self.base_frame}' and TF support is disabled. Ignoring waypoint."
            )
            self.warned_about_waypoint_frame = True

    def _on_command(self, msg: String) -> None:
        command = msg.data.strip().upper()

        if command == "EXECUTE":
            self.stream_mode = False
            self.is_executing = True
            self.execution_start_time = None
            self.settle_count = 0
            self.saturation_count = 0
            self.status_publisher.publish(String(data="BUSY"))
            self.get_logger().info("EXECUTE -> BUSY (mode=ONCE)")
            return

        if command == "EXECUTE_STREAM":
            self.stream_mode = True
            self.is_executing = True
            self.execution_start_time = None
            self.settle_count = 0
            self.saturation_count = 0
            self.status_publisher.publish(String(data="BUSY"))
            self.get_logger().info("EXECUTE_STREAM -> BUSY (mode=STREAM)")
            return

        if command in ("STOP", "ABORT"):
            self.is_executing = False
            self.status_publisher.publish(String(data="DONE_FAIL"))
            self.status_publisher.publish(String(data="IDLE"))
            self.get_logger().warn(f"{command} -> DONE_FAIL / IDLE")

    def _should_log(self, now) -> bool:
        if self.log_period_sec <= 0.0:
            return False

        if self.last_log_time is None:
            self.last_log_time = now
            return True

        elapsed_sec = (now - self.last_log_time).nanoseconds * 1e-9
        if elapsed_sec >= self.log_period_sec:
            self.last_log_time = now
            return True

        return False

    def _limit_avoidance_gradient(self, joint_positions: np.ndarray) -> np.ndarray:
        gradient = 2.0 * (joint_positions - self.joint_midpoints) / (self.joint_ranges ** 2)

        margin = max(self.limit_margin_rad, 1e-6)
        distance_to_min = joint_positions - self.joint_min
        distance_to_max = self.joint_max - joint_positions

        near_min = distance_to_min < margin
        if np.any(near_min):
            gradient[near_min] += -self.limit_push_gain * (margin - distance_to_min[near_min]) / (margin ** 2)

        near_max = distance_to_max < margin
        if np.any(near_max):
            gradient[near_max] += +self.limit_push_gain * (margin - distance_to_max[near_max]) / (margin ** 2)

        return gradient

    def _get_tcp_state(self):
        if self.current_joint_positions is None:
            return None

        return forward_kinematics_and_jacobian(
            self.current_joint_positions,
            self.U,
            self.V,
            self.joint_axes,
            self.ee_transform,
        )

    def _finish_with_failure(self, reason: str) -> None:
        self.is_executing = False
        self.status_publisher.publish(String(data="DONE_FAIL"))
        self.status_publisher.publish(String(data="IDLE"))
        self.get_logger().warn(f"DONE_FAIL: {reason}")

    def _finish_with_success(self, reason: str) -> None:
        self.is_executing = False
        self.status_publisher.publish(String(data="DONE_OK"))
        self.status_publisher.publish(String(data="IDLE"))
        self.get_logger().info(f"DONE_OK: {reason}")

    def _control_loop(self) -> None:
        if not self.is_executing:
            return

        now = self.get_clock().now()

        if (
            self.current_joint_positions is None
            or self.current_waypoint is None
            or self.last_waypoint_time is None
        ):
            return

        if self.execution_start_time is None:
            self.execution_start_time = now

        execution_elapsed_sec = (now - self.execution_start_time).nanoseconds * 1e-9
        if execution_elapsed_sec > self.execute_timeout_sec:
            self._finish_with_failure("execute_timeout")
            return

        waypoint_age_sec = (now - self.last_waypoint_time).nanoseconds * 1e-9
        if waypoint_age_sec > self.waypoint_timeout_sec:
            self._finish_with_failure(f"waypoint_timeout age={waypoint_age_sec:.2f}s")
            return

        tcp_state = self._get_tcp_state()
        if tcp_state is None:
            return

        tcp_position, tcp_rotation, jacobian_linear, jacobian_angular = tcp_state

        position_error = (self.current_waypoint - tcp_position).reshape(3)
        position_error_norm = float(np.linalg.norm(position_error))
        self.error_norm_publisher.publish(Float64(data=position_error_norm))

        if not self.stream_mode:
            if position_error_norm <= self.position_tolerance_m:
                self.settle_count += 1
            else:
                self.settle_count = 0
        else:
            if waypoint_age_sec >= self.done_waypoint_stale_sec and position_error_norm <= self.position_tolerance_m:
                self.settle_count += 1
            else:
                self.settle_count = 0

        if self.settle_count >= self.settle_cycles:
            mode_name = "STREAM" if self.stream_mode else "ONCE"
            self._finish_with_success(
                f"{mode_name} settled pos_err={position_error_norm:.4f} wp_age={waypoint_age_sec:.2f}s"
            )
            return

        current_alignment_axis = tcp_rotation @ self.local_alignment_axis
        desired_alignment_axis = self.desired_alignment_axis
        orientation_error = np.cross(current_alignment_axis, desired_alignment_axis)
        alignment_dot = float(np.dot(current_alignment_axis, desired_alignment_axis))

        num_joints = len(self.joint_names)

        position_task_matrix = jacobian_linear
        position_system = (
            position_task_matrix
            @ self.inverse_weight_matrix
            @ position_task_matrix.T
            + (self.damp_pos ** 2) * np.eye(3)
        )

        try:
            position_rhs = np.linalg.solve(position_system, self.kp_pos * position_error)
        except np.linalg.LinAlgError:
            self._finish_with_failure("singular W-DLS solve (position)")
            return

        dq_position = self.inverse_weight_matrix @ position_task_matrix.T @ position_rhs

        weighted_pseudoinverse_position = (
            self.inverse_weight_matrix
            @ position_task_matrix.T
            @ np.linalg.inv(position_system)
        )
        nullspace_position = np.eye(num_joints) - (
            weighted_pseudoinverse_position @ position_task_matrix
        )

        orientation_task_matrix = jacobian_angular @ nullspace_position
        orientation_system = (
            orientation_task_matrix
            @ self.inverse_weight_matrix
            @ orientation_task_matrix.T
            + (self.damp_ori ** 2) * np.eye(3)
        )

        try:
            orientation_rhs = np.linalg.solve(
                orientation_system,
                self.kp_ori * orientation_error,
            )
        except np.linalg.LinAlgError:
            orientation_rhs = np.zeros(3)

        dq_orientation = (
            nullspace_position
            @ (self.inverse_weight_matrix @ orientation_task_matrix.T @ orientation_rhs)
        )

        dq_command = dq_position + dq_orientation
        nullspace_velocity_norm = 0.0

        if self.enable_nullspace and self.nullspace_gain > 0.0:
            limit_gradient = self._limit_avoidance_gradient(self.current_joint_positions)
            dq_nullspace = nullspace_position @ (-self.nullspace_gain * limit_gradient)
            dq_command += dq_nullspace
            nullspace_velocity_norm = float(np.linalg.norm(dq_nullspace))

        joint_step = self.control_dt * dq_command
        if self.max_joint_step_rad > 0.0:
            joint_step = np.clip(
                joint_step,
                -self.max_joint_step_rad,
                self.max_joint_step_rad,
            )

        raw_command = self.current_joint_positions + joint_step
        clamped_command = np.minimum(np.maximum(raw_command, self.joint_min), self.joint_max)

        saturated = np.abs(clamped_command - raw_command) > self.saturation_eps
        saturated_joint_indices = np.where(saturated)[0].tolist()

        if len(saturated_joint_indices) > 0 and position_error_norm > self.position_tolerance_m:
            self.saturation_count += 1
        else:
            self.saturation_count = 0

        if self.saturation_count >= self.saturation_hold_cycles:
            saturated_joint_names = ",".join(self.joint_names[index] for index in saturated_joint_indices)
            self._finish_with_failure(
                f"joint saturation blocking motion (SAT=[{saturated_joint_names}])"
            )
            return

        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = clamped_command.tolist()
        trajectory_point.time_from_start = Duration(
            sec=0,
            nanosec=int(self.command_horizon_sec * 1e9),
        )
        trajectory_msg.points = [trajectory_point]
        self.trajectory_publisher.publish(trajectory_msg)

        if self._should_log(now):
            source_frame = self.last_waypoint_frame if self.last_waypoint_frame else "''"
            saturated_joint_names = (
                "none"
                if len(saturated_joint_indices) == 0
                else ",".join(self.joint_names[index] for index in saturated_joint_indices)
            )

            abs_joint_step = np.abs(joint_step)
            max_step_index = int(np.argmax(abs_joint_step))
            max_step_value = float(abs_joint_step[max_step_index])
            max_step_joint_name = self.joint_names[max_step_index]

            mode_name = "STREAM" if self.stream_mode else "ONCE"

            self.get_logger().info(
                f"[{mode_name} {source_frame}->{self.base_frame}] "
                f"TCP=({tcp_position[0]:+.3f},{tcp_position[1]:+.3f},{tcp_position[2]:+.3f}) "
                f"WP=({self.current_waypoint[0]:+.3f},{self.current_waypoint[1]:+.3f},{self.current_waypoint[2]:+.3f}) "
                f"|e_p|={position_error_norm:.4f} align={alignment_dot:+.3f} "
                f"|dq_step|={float(np.linalg.norm(joint_step)):.3e} "
                f"maxΔq={max_step_value:.3e}({max_step_joint_name}) "
                f"SAT=[{saturated_joint_names}] "
                f"dq_null={nullspace_velocity_norm:.3e} "
                f"wp_age={waypoint_age_sec:.2f}s settle={self.settle_count}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()