#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, String
from visualization_msgs.msg import Marker


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
    """Compute TCP position and geometric Jacobian for the kinematic chain."""
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

    return tcp_position, jacobian_linear, jacobian_angular


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
    """Extract kinematic chain matrices and limits from the URDF."""
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
            found = False

            while stack and not found:
                link_name, accumulated_transform = stack.pop()
                if link_name in visited:
                    continue
                visited.add(link_name)

                if link_name == ee_link_name:
                    ee_transform = accumulated_transform
                    found = True
                    break

                for joint in fixed_children_by_parent.get(link_name, []):
                    origin_element = joint.find(f"{namespace}origin")
                    xyz = parse_xyz(origin_element.attrib.get("xyz")) if origin_element is not None else np.zeros(3)
                    rpy = parse_rpy(origin_element.attrib.get("rpy")) if origin_element is not None else np.zeros(3)
                    joint_transform = rpy_to_transform(xyz, rpy)
                    child_link = joint.find(f"{namespace}child").attrib["link"]
                    stack.append((child_link, accumulated_transform @ joint_transform))

    return U, V, joint_axes, joint_min, joint_max, ee_transform


def build_sigmoid_path(
    start_point: np.ndarray,
    goal_point: np.ndarray,
    total_time: float,
    dt: float,
    gain_vector: np.ndarray,
) -> np.ndarray:
    """Generate a smooth sigmoid trajectory between two Cartesian points."""
    times = np.arange(0.0, total_time + 1e-9, dt)
    path = []

    for time_value in times:
        sample = start_point + (goal_point - start_point) / (
            1.0 + np.exp(-gain_vector * (time_value - total_time / 2.0))
        )
        path.append(sample)

    return np.array(path, dtype=float)


def clamp(value: float, lower_bound: float, upper_bound: float) -> float:
    return max(lower_bound, min(upper_bound, value))


class Mode1TrajectoryNode(Node):
    def __init__(self) -> None:
        super().__init__("mode1_trajectory_node")

        self._declare_parameters()
        self._load_parameters()

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

        self.current_joint_positions = None
        self.latest_target_msg = None
        self.latest_radius_m = None

        self.path_xyz = None
        self.current_waypoint_index = 0
        self.is_active = False

        self.projected_sphere_center = None
        self.projected_sphere_radius = None
        self.goal_point = None
        self.target_point = None

        self.last_path_marker_pub_time = 0.0

        self._create_ros_interfaces()

        self.status_publisher.publish(String(data="IDLE"))
        self.timer = self.create_timer(1.0 / max(self.loop_rate_hz, 1.0), self._control_loop)

        self.get_logger().info(
            f"mode1_trajectory_node ready. "
            f"projection_distance={self.projection_distance_m:.3f} m, "
            f"clamp_radius={self.clamp_radius}, "
            f"waypoint_tol={self.waypoint_tol_m:.3f} m, "
            f"sphere_tol={self.sphere_touch_tol_m:.3f} m, "
            f"goal_tol={self.goal_tol_m:.3f} m"
        )

    def _declare_parameters(self) -> None:
        package_share = get_package_share_directory("harvesting_robot")
        default_urdf_path = os.path.join(package_share, "urdf", "elfin3.urdf")

        self.declare_parameter("base_frame", "elfin_base")
        self.declare_parameter("target_topic", "/target_base")
        self.declare_parameter("radius_topic", "/sphere_radius")
        self.declare_parameter("joint_state_topic", "/joint_states")

        self.declare_parameter("cmd_topic", "/trajectory/cmd")
        self.declare_parameter("status_topic", "/trajectory/status")
        self.declare_parameter("waypoint_topic", "/trajectory/waypoint")
        self.declare_parameter("path_topic", "/trajectory/path")
        self.declare_parameter("tcp_target_dist_topic", "/trajectory/tcp_target_dist")

        self.declare_parameter("urdf_path", default_urdf_path)
        self.declare_parameter(
            "joint_names",
            [
                "elfin_joint1",
                "elfin_joint2",
                "elfin_joint3",
                "elfin_joint4",
                "elfin_joint5",
                "elfin_joint6",
            ],
        )
        self.declare_parameter("ee_link", "rg2ft_grasp_point")

        self.declare_parameter("path_marker_topic", "/trajectory/path_marker")
        self.declare_parameter("sphere_marker_topic", "/trajectory/sphere_marker")
        self.declare_parameter("goal_marker_topic", "/trajectory/goal_marker")
        self.declare_parameter("target_marker_topic", "/trajectory/target_marker")
        self.declare_parameter("waypoint_marker_topic", "/trajectory/waypoint_marker")
        self.declare_parameter("tcp_marker_topic", "/trajectory/tcp_marker")

        self.declare_parameter("projection_distance_m", 0.10)

        self.declare_parameter("clamp_radius", True)
        self.declare_parameter("min_radius_m", 0.02)
        self.declare_parameter("max_radius_m", 0.25)

        self.declare_parameter("T", 10.0)
        self.declare_parameter("dt_wp", 0.10)
        self.declare_parameter("c", [0.3, 0.7, 1.1])

        self.declare_parameter("waypoint_tol_m", 0.05)
        self.declare_parameter("sphere_touch_tol_m", 0.05)
        self.declare_parameter("goal_tol_m", 0.05)

        self.declare_parameter("loop_rate_hz", 50.0)
        self.declare_parameter("path_pub_period", 1.0)
        self.declare_parameter("path_downsample", 1)

    def _load_parameters(self) -> None:
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.target_topic = str(self.get_parameter("target_topic").value)
        self.radius_topic = str(self.get_parameter("radius_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)

        self.command_topic = str(self.get_parameter("cmd_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.waypoint_topic = str(self.get_parameter("waypoint_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.tcp_target_distance_topic = str(
            self.get_parameter("tcp_target_dist_topic").value
        )

        self.urdf_path = str(self.get_parameter("urdf_path").value)
        self.joint_names = list(self.get_parameter("joint_names").value)
        self.ee_link = self.get_parameter("ee_link").value or None

        self.path_marker_topic = str(self.get_parameter("path_marker_topic").value)
        self.sphere_marker_topic = str(self.get_parameter("sphere_marker_topic").value)
        self.goal_marker_topic = str(self.get_parameter("goal_marker_topic").value)
        self.target_marker_topic = str(self.get_parameter("target_marker_topic").value)
        self.waypoint_marker_topic = str(self.get_parameter("waypoint_marker_topic").value)
        self.tcp_marker_topic = str(self.get_parameter("tcp_marker_topic").value)

        self.projection_distance_m = float(
            self.get_parameter("projection_distance_m").value
        )

        self.clamp_radius = bool(self.get_parameter("clamp_radius").value)
        self.min_radius_m = float(self.get_parameter("min_radius_m").value)
        self.max_radius_m = float(self.get_parameter("max_radius_m").value)

        self.trajectory_total_time = float(self.get_parameter("T").value)
        self.waypoint_dt = float(self.get_parameter("dt_wp").value)
        self.sigmoid_gain_vector = np.array(
            self.get_parameter("c").value,
            dtype=float,
        ).reshape(3)

        self.waypoint_tol_m = float(self.get_parameter("waypoint_tol_m").value)
        self.sphere_touch_tol_m = float(
            self.get_parameter("sphere_touch_tol_m").value
        )
        self.goal_tol_m = float(self.get_parameter("goal_tol_m").value)

        self.loop_rate_hz = float(self.get_parameter("loop_rate_hz").value)
        self.path_pub_period_sec = float(
            self.get_parameter("path_pub_period").value
        )
        self.path_downsample = max(
            1,
            int(self.get_parameter("path_downsample").value),
        )

    def _create_ros_interfaces(self) -> None:
        latched_qos = QoSProfile(depth=1)
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        latched_qos.reliability = ReliabilityPolicy.RELIABLE

        self.create_subscription(JointState, self.joint_state_topic, self._on_joint_state, 50)
        self.create_subscription(PointStamped, self.target_topic, self._on_target, 10)
        self.create_subscription(Float32, self.radius_topic, self._on_radius, 10)
        self.create_subscription(String, self.command_topic, self._on_command, 10)

        self.status_publisher = self.create_publisher(String, self.status_topic, 10)
        self.waypoint_publisher = self.create_publisher(
            PoseStamped,
            self.waypoint_topic,
            10,
        )
        self.path_publisher = self.create_publisher(Path, self.path_topic, latched_qos)

        self.path_marker_publisher = self.create_publisher(
            Marker,
            self.path_marker_topic,
            latched_qos,
        )
        self.sphere_marker_publisher = self.create_publisher(
            Marker,
            self.sphere_marker_topic,
            10,
        )
        self.goal_marker_publisher = self.create_publisher(
            Marker,
            self.goal_marker_topic,
            10,
        )
        self.target_marker_publisher = self.create_publisher(
            Marker,
            self.target_marker_topic,
            10,
        )
        self.waypoint_marker_publisher = self.create_publisher(
            Marker,
            self.waypoint_marker_topic,
            10,
        )
        self.tcp_marker_publisher = self.create_publisher(
            Marker,
            self.tcp_marker_topic,
            10,
        )

        self.tcp_target_distance_publisher = self.create_publisher(
            Float32,
            self.tcp_target_distance_topic,
            10,
        )

    def _on_joint_state(self, msg: JointState) -> None:
        joint_map = {name: position for name, position in zip(msg.name, msg.position)}
        if not all(joint_name in joint_map for joint_name in self.joint_names):
            return

        self.current_joint_positions = np.array(
            [joint_map[joint_name] for joint_name in self.joint_names],
            dtype=float,
        )

    def _on_target(self, msg: PointStamped) -> None:
        self.latest_target_msg = msg

    def _on_radius(self, msg: Float32) -> None:
        self.latest_radius_m = float(msg.data)

    def _on_command(self, msg: String) -> None:
        command = msg.data.strip().upper()

        if command not in ("PLAN", "START"):
            if command in ("STOP", "ABORT"):
                self.is_active = False
                self.status_publisher.publish(String(data="DONE_FAIL"))
                self.status_publisher.publish(String(data="IDLE"))
            return

        if self.is_active:
            return

        self.status_publisher.publish(String(data="BUSY"))

        planned_ok, reason = self._plan_path_once()
        if not planned_ok:
            self.status_publisher.publish(String(data="DONE_FAIL"))
            self.status_publisher.publish(String(data="IDLE"))
            self.get_logger().warn(f"DONE_FAIL: {reason}")
            return

        self._publish_path_message()
        self._publish_static_markers()

        self.is_active = True
        self.get_logger().info("Trajectory planned successfully. Starting waypoint publication.")

    def _get_tcp_position(self) -> np.ndarray | None:
        if self.current_joint_positions is None:
            return None

        tcp_position, _, _ = forward_kinematics_and_jacobian(
            self.current_joint_positions,
            self.U,
            self.V,
            self.joint_axes,
            self.ee_transform,
        )
        return np.array(tcp_position, dtype=float)

    def _plan_path_once(self):
        if self.latest_target_msg is None:
            return False, "no target"
        if self.latest_radius_m is None:
            return False, "no radius"
        if self.current_joint_positions is None:
            return False, "no joint_states yet"

        if (
            self.latest_target_msg.header.frame_id
            and self.latest_target_msg.header.frame_id != self.base_frame
        ):
            return False, f"target frame mismatch: {self.latest_target_msg.header.frame_id}"

        tcp_start = self._get_tcp_position()
        if tcp_start is None:
            return False, "tcp fk not available"

        target_point = np.array(
            [
                self.latest_target_msg.point.x,
                self.latest_target_msg.point.y,
                self.latest_target_msg.point.z,
            ],
            dtype=float,
        )

        direction_vector = target_point - tcp_start
        direction_norm = float(np.linalg.norm(direction_vector))
        if direction_norm < 1e-6:
            return False, "degenerate direction (tcp==target)"

        direction_unit = direction_vector / direction_norm

        raw_radius_m = float(self.latest_radius_m)
        used_radius_m = (
            clamp(raw_radius_m, self.min_radius_m, self.max_radius_m)
            if self.clamp_radius
            else raw_radius_m
        )

        projected_sphere_center = tcp_start + self.projection_distance_m * direction_unit
        if self.projection_distance_m > direction_norm:
            projected_sphere_center = target_point.copy()

        goal_point = projected_sphere_center - used_radius_m * direction_unit

        self.path_xyz = build_sigmoid_path(
            tcp_start,
            goal_point,
            self.trajectory_total_time,
            self.waypoint_dt,
            self.sigmoid_gain_vector,
        )
        self.current_waypoint_index = 0

        self.projected_sphere_center = projected_sphere_center
        self.projected_sphere_radius = used_radius_m
        self.goal_point = goal_point
        self.target_point = target_point

        self.get_logger().info(
            f"PLAN ok: dist={direction_norm:.3f} m, "
            f"projection={self.projection_distance_m:.3f} m, "
            f"radius_raw={raw_radius_m:.3f} m, "
            f"radius_used={used_radius_m:.3f} m, "
            f"clamp_radius={self.clamp_radius}"
        )

        return True, "ok"

    def _finish_success(self, reason: str) -> None:
        self.is_active = False
        self.status_publisher.publish(String(data="DONE_OK"))
        self.status_publisher.publish(String(data="IDLE"))
        self.get_logger().info(f"DONE_OK: {reason}")

    def _control_loop(self) -> None:
        tcp_position = self._get_tcp_position()

        if tcp_position is not None and self.latest_target_msg is not None:
            target_point = np.array(
                [
                    self.latest_target_msg.point.x,
                    self.latest_target_msg.point.y,
                    self.latest_target_msg.point.z,
                ],
                dtype=float,
            )
            tcp_target_distance = float(np.linalg.norm(target_point - tcp_position))
            self.tcp_target_distance_publisher.publish(
                Float32(data=tcp_target_distance)
            )

        if not self.is_active or self.path_xyz is None:
            return

        if tcp_position is None:
            return

        if (
            self.projected_sphere_center is not None
            and self.projected_sphere_radius is not None
        ):
            distance_to_sphere_center = float(
                np.linalg.norm(tcp_position - self.projected_sphere_center)
            )
            if distance_to_sphere_center <= (
                self.projected_sphere_radius + self.sphere_touch_tol_m
            ):
                self._finish_success("Reached projected sphere touch condition.")
                return

        current_waypoint = self.path_xyz[
            min(self.current_waypoint_index, len(self.path_xyz) - 1)
        ]

        waypoint_msg = PoseStamped()
        waypoint_msg.header.frame_id = self.base_frame
        waypoint_msg.header.stamp = self.get_clock().now().to_msg()
        waypoint_msg.pose.position.x = float(current_waypoint[0])
        waypoint_msg.pose.position.y = float(current_waypoint[1])
        waypoint_msg.pose.position.z = float(current_waypoint[2])
        waypoint_msg.pose.orientation.w = 1.0
        self.waypoint_publisher.publish(waypoint_msg)

        self._publish_waypoint_marker(current_waypoint)
        self._publish_tcp_marker(tcp_position)

        if self.current_waypoint_index < (len(self.path_xyz) - 1):
            distance_to_waypoint = float(np.linalg.norm(current_waypoint - tcp_position))
            if distance_to_waypoint <= self.waypoint_tol_m:
                self.current_waypoint_index += 1
        else:
            if self.goal_point is not None:
                distance_to_goal = float(np.linalg.norm(tcp_position - self.goal_point))
                if distance_to_goal <= self.goal_tol_m:
                    self._finish_success("Reached final waypoint.")
                    return

        current_time_sec = self.get_clock().now().nanoseconds / 1e9
        if current_time_sec - self.last_path_marker_pub_time > self.path_pub_period_sec:
            self._publish_path_marker()
            self.last_path_marker_pub_time = current_time_sec

    def _publish_path_message(self) -> None:
        if self.path_xyz is None:
            return

        path_msg = Path()
        path_msg.header.frame_id = self.base_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point_index, point_xyz in enumerate(self.path_xyz):
            if (point_index % self.path_downsample) != 0:
                continue

            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self.base_frame
            pose_msg.header.stamp = path_msg.header.stamp
            pose_msg.pose.position.x = float(point_xyz[0])
            pose_msg.pose.position.y = float(point_xyz[1])
            pose_msg.pose.position.z = float(point_xyz[2])
            pose_msg.pose.orientation.w = 1.0
            path_msg.poses.append(pose_msg)

        self.path_publisher.publish(path_msg)

    def _publish_static_markers(self) -> None:
        self._publish_sphere_marker()
        self._publish_goal_marker()
        self._publish_target_marker()
        self._publish_path_marker()

    def _publish_path_marker(self) -> None:
        if self.path_xyz is None:
            return

        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory_path"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.lifetime = Duration(sec=0, nanosec=0)
        marker.frame_locked = True
        marker.points = []

        for point_index, point_xyz in enumerate(self.path_xyz):
            if (point_index % self.path_downsample) != 0:
                continue
            marker.points.append(
                Point(
                    x=float(point_xyz[0]),
                    y=float(point_xyz[1]),
                    z=float(point_xyz[2]),
                )
            )

        self.path_marker_publisher.publish(marker)

    def _publish_sphere_marker(self) -> None:
        if self.projected_sphere_center is None or self.projected_sphere_radius is None:
            return

        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory_sphere"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(self.projected_sphere_center[0])
        marker.pose.position.y = float(self.projected_sphere_center[1])
        marker.pose.position.z = float(self.projected_sphere_center[2])
        marker.scale.x = 2.0 * float(self.projected_sphere_radius)
        marker.scale.y = 2.0 * float(self.projected_sphere_radius)
        marker.scale.z = 2.0 * float(self.projected_sphere_radius)
        marker.color.a = 0.25
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.sphere_marker_publisher.publish(marker)

    def _publish_goal_marker(self) -> None:
        if self.goal_point is None:
            return

        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(self.goal_point[0])
        marker.pose.position.y = float(self.goal_point[1])
        marker.pose.position.z = float(self.goal_point[2])
        marker.lifetime = Duration(sec=0, nanosec=0)
        self.goal_marker_publisher.publish(marker)

    def _publish_target_marker(self) -> None:
        if self.target_point is None:
            return

        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.scale.y = 0.02
        marker.scale.z = 0.02
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(self.target_point[0])
        marker.pose.position.y = float(self.target_point[1])
        marker.pose.position.z = float(self.target_point[2])
        marker.lifetime = Duration(sec=0, nanosec=0)
        self.target_marker_publisher.publish(marker)

    def _publish_waypoint_marker(self, waypoint: np.ndarray) -> None:
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory_waypoint"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(waypoint[0])
        marker.pose.position.y = float(waypoint[1])
        marker.pose.position.z = float(waypoint[2])
        marker.lifetime = Duration(sec=1, nanosec=0)
        self.waypoint_marker_publisher.publish(marker)

    def _publish_tcp_marker(self, tcp_position: np.ndarray) -> None:
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory_tcp"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(tcp_position[0])
        marker.pose.position.y = float(tcp_position[1])
        marker.pose.position.z = float(tcp_position[2])
        marker.lifetime = Duration(sec=0, nanosec=0)
        self.tcp_marker_publisher.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Mode1TrajectoryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()