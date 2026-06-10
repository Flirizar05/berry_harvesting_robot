#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from dataclasses import dataclass
from statistics import median

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from harvesting_robot_agv.agv_position_control_node import (
    DEFAULT_TARGET_HORIZONTAL_LINE_DISTANCE_M,
)

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


WINDOW_NAME = "RPLIDAR 360 Radar"
DEFAULT_SCAN_TOPIC = "/scan"
DEFAULT_MAX_DISPLAY_RANGE_M = 4.0
DEFAULT_ALERT_DISTANCE_M = 0.4
DEFAULT_SECTOR_WIDTH_DEG = 10.0
DEFAULT_WINDOW_WIDTH_PX = 900
DEFAULT_WINDOW_HEIGHT_PX = 900
DEFAULT_LINE_DETECTION_TOPIC = "/agv/line_detections"
DEFAULT_RADAR_IMAGE_TOPIC = "/agv/lidar_radar_image"
DEFAULT_RADAR_IMAGE_PUBLISH_PERIOD_SEC = 0.10
DEFAULT_SHOW_WINDOW = True
DEFAULT_AGV_CMD_TOPIC = "/agv/rpm_cmd"

BACKGROUND_COLOR = (8, 12, 10)
GRID_COLOR = (58, 82, 70)
TEXT_COLOR = (220, 235, 225)
NORMAL_COLOR = (255, 255, 255)
ALERT_COLOR = (40, 40, 255)
REFERENCE_LINE_DETECTION_COLOR = (80, 220, 120)
HORIZONTAL_REFERENCE_DETECTION_COLOR = (255, 0, 255)
NEAREST_STRAIGHT_LINE_COLOR = (0, 165, 255)
CENTER_COLOR = (255, 255, 255)
REFERENCE_LINE_COLOR = (0, 220, 255)
REFERENCE_LINE_DETECTION_TOLERANCE_M = 0.10
NEAREST_STRAIGHT_LINE_LENGTH_M = 1.0
NEAREST_STRAIGHT_LINE_MIN_POINTS = 8
NEAREST_STRAIGHT_LINE_MAX_POINT_GAP_M = 0.25
NEAREST_STRAIGHT_LINE_MAX_RMSE_M = 0.06
NEAREST_STRAIGHT_LINE_MAX_RESIDUAL_M = 0.12
LIDAR_MOUNT_ROTATION_DEG = 180.0
AGV_COLOR = (255, 80, 0)
AGV_ALPHA = 0.38
AGV_WIDTH_M = 0.77
AGV_LENGTH_M = 1.20
AGV_FRONT_EDGE_BEHIND_LIDAR_M = 0.095
AGV_LIDAR_TO_RIGHT_EDGE_M = 0.395
SIDE_CLEARANCE_REGION_COLOR = (30, 190, 220)
SIDE_CLEARANCE_REGION_ALPHA = 0.24
DEFAULT_SIDE_CLEARANCE_REGION_OUTER_OFFSET_M = 1.0
DEFAULT_SIDE_CLEARANCE_REGION_FORWARD_M = 0.1
DEFAULT_SIDE_CLEARANCE_DETECTION_MIN_POINTS = 1
DEFAULT_SIDE_CLEARANCE_STICKY_TIMEOUT_SEC = 0.75
DEFAULT_SIDE_CLEARANCE_STICKY_MAX_MOTION_M = 0.03
AGV_MOVING_RPM_EPSILON = 1.0


@dataclass(frozen=True)
class StraightLineDetection:
    start_xy_m: tuple[float, float]
    end_xy_m: tuple[float, float]
    closest_xy_m: tuple[float, float]
    distance_m: float
    angle_deg: float
    point_count: int
    rmse_m: float


@dataclass(frozen=True)
class SideClearanceDetection:
    side: str
    xy_m: tuple[float, float]
    distance_m: float
    point_count: int


class Lidar360RadarNode(Node):
    """OpenCV 360-degree radar view for a LaserScan topic."""

    def __init__(self) -> None:
        super().__init__("lidar_360_radar_node")

        self.latest_scan: LaserScan | None = None
        self.nearest_straight_line: StraightLineDetection | None = None
        self.nearest_left_straight_line: StraightLineDetection | None = None
        self.nearest_right_straight_line: StraightLineDetection | None = None
        self.side_clearance_detection: SideClearanceDetection | None = None
        self.side_clearance_sticky_suppressed = False
        self.side_clearance_motion_track: tuple[str, tuple[float, float], float] | None = None
        self.agv_command_is_moving = False
        self.last_radar_image_publish_time = 0.0
        self.is_closing = False

        self._declare_parameters()
        self._load_parameters()
        self._create_ros_interfaces()
        if self.show_window:
            self._create_window()

        self.get_logger().info("lidar_360_radar_node ready")
        self.get_logger().info(f"Subscribing to: {self.scan_topic}")
        self.get_logger().info(
            f"Publishing line detections to: {self.line_detection_topic}"
        )
        self.get_logger().info(
            f"Publishing radar image to: {self.radar_image_topic}"
        )
        self.get_logger().info(f"OpenCV window enabled: {self.show_window}")
        self.get_logger().info(
            f"Display range: {self.max_display_range_m:.2f} m, "
            f"alert distance: {self.alert_distance_m:.2f} m, "
            "target horizontal line distance: "
            f"{self.target_horizontal_line_distance_m:.2f} m"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("scan_topic", DEFAULT_SCAN_TOPIC)
        self.declare_parameter(
            "max_display_range_m",
            DEFAULT_MAX_DISPLAY_RANGE_M,
        )
        self.declare_parameter("alert_distance_m", DEFAULT_ALERT_DISTANCE_M)
        self.declare_parameter("sector_width_deg", DEFAULT_SECTOR_WIDTH_DEG)
        self.declare_parameter("window_width_px", DEFAULT_WINDOW_WIDTH_PX)
        self.declare_parameter("window_height_px", DEFAULT_WINDOW_HEIGHT_PX)
        self.declare_parameter(
            "line_detection_topic",
            DEFAULT_LINE_DETECTION_TOPIC,
        )
        self.declare_parameter("radar_image_topic", DEFAULT_RADAR_IMAGE_TOPIC)
        self.declare_parameter("agv_cmd_topic", DEFAULT_AGV_CMD_TOPIC)
        self.declare_parameter(
            "radar_image_publish_period_sec",
            DEFAULT_RADAR_IMAGE_PUBLISH_PERIOD_SEC,
        )
        self.declare_parameter(
            "target_horizontal_line_distance_m",
            DEFAULT_TARGET_HORIZONTAL_LINE_DISTANCE_M,
        )
        self.declare_parameter(
            "side_clearance_region_outer_offset_m",
            DEFAULT_SIDE_CLEARANCE_REGION_OUTER_OFFSET_M,
        )
        self.declare_parameter(
            "side_clearance_region_forward_m",
            DEFAULT_SIDE_CLEARANCE_REGION_FORWARD_M,
        )
        self.declare_parameter(
            "side_clearance_detection_min_points",
            DEFAULT_SIDE_CLEARANCE_DETECTION_MIN_POINTS,
        )
        self.declare_parameter(
            "side_clearance_sticky_timeout_sec",
            DEFAULT_SIDE_CLEARANCE_STICKY_TIMEOUT_SEC,
        )
        self.declare_parameter(
            "side_clearance_sticky_max_motion_m",
            DEFAULT_SIDE_CLEARANCE_STICKY_MAX_MOTION_M,
        )
        self.declare_parameter("show_window", DEFAULT_SHOW_WINDOW)

    def _load_parameters(self) -> None:
        self.scan_topic = str(self.get_parameter("scan_topic").value).strip()
        if not self.scan_topic:
            self.scan_topic = DEFAULT_SCAN_TOPIC

        self.max_display_range_m = self._positive_float_parameter(
            "max_display_range_m",
            DEFAULT_MAX_DISPLAY_RANGE_M,
            minimum=0.1,
        )
        self.alert_distance_m = self._positive_float_parameter(
            "alert_distance_m",
            DEFAULT_ALERT_DISTANCE_M,
            minimum=0.0,
        )
        self.sector_width_deg = self._positive_float_parameter(
            "sector_width_deg",
            DEFAULT_SECTOR_WIDTH_DEG,
            minimum=0.1,
            maximum=360.0,
        )
        self.window_width_px = self._positive_int_parameter(
            "window_width_px",
            DEFAULT_WINDOW_WIDTH_PX,
            minimum=300,
        )
        self.window_height_px = self._positive_int_parameter(
            "window_height_px",
            DEFAULT_WINDOW_HEIGHT_PX,
            minimum=300,
        )
        self.line_detection_topic = str(
            self.get_parameter("line_detection_topic").value
        ).strip()
        if not self.line_detection_topic:
            self.line_detection_topic = DEFAULT_LINE_DETECTION_TOPIC
        self.radar_image_topic = str(
            self.get_parameter("radar_image_topic").value
        ).strip()
        if not self.radar_image_topic:
            self.radar_image_topic = DEFAULT_RADAR_IMAGE_TOPIC
        self.agv_cmd_topic = str(
            self.get_parameter("agv_cmd_topic").value
        ).strip()
        if not self.agv_cmd_topic:
            self.agv_cmd_topic = DEFAULT_AGV_CMD_TOPIC
        self.radar_image_publish_period_sec = self._positive_float_parameter(
            "radar_image_publish_period_sec",
            DEFAULT_RADAR_IMAGE_PUBLISH_PERIOD_SEC,
            minimum=0.01,
        )
        self.target_horizontal_line_distance_m = self._positive_float_parameter(
            "target_horizontal_line_distance_m",
            DEFAULT_TARGET_HORIZONTAL_LINE_DISTANCE_M,
            minimum=0.0,
        )
        self.side_clearance_region_outer_offset_m = self._positive_float_parameter(
            "side_clearance_region_outer_offset_m",
            DEFAULT_SIDE_CLEARANCE_REGION_OUTER_OFFSET_M,
            minimum=0.0,
        )
        self.side_clearance_region_forward_m = self._positive_float_parameter(
            "side_clearance_region_forward_m",
            DEFAULT_SIDE_CLEARANCE_REGION_FORWARD_M,
            minimum=0.0,
        )
        self.side_clearance_detection_min_points = self._positive_int_parameter(
            "side_clearance_detection_min_points",
            DEFAULT_SIDE_CLEARANCE_DETECTION_MIN_POINTS,
            minimum=1,
        )
        self.side_clearance_sticky_timeout_sec = self._positive_float_parameter(
            "side_clearance_sticky_timeout_sec",
            DEFAULT_SIDE_CLEARANCE_STICKY_TIMEOUT_SEC,
            minimum=0.0,
        )
        self.side_clearance_sticky_max_motion_m = self._positive_float_parameter(
            "side_clearance_sticky_max_motion_m",
            DEFAULT_SIDE_CLEARANCE_STICKY_MAX_MOTION_M,
            minimum=0.0,
        )
        self.show_window = self._bool_parameter(
            "show_window",
            DEFAULT_SHOW_WINDOW,
        )

    def _bool_parameter(self, name: str, default_value: bool) -> bool:
        value = self.get_parameter(name).value

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized_value = value.strip().lower()
            if normalized_value in ("true", "1", "yes", "on"):
                return True
            if normalized_value in ("false", "0", "no", "off"):
                return False

        self.get_logger().warn(
            f"Parameter '{name}' must be boolean; using {default_value}"
        )
        return default_value

    def _positive_float_parameter(
        self,
        name: str,
        default_value: float,
        minimum: float,
        maximum: float | None = None,
    ) -> float:
        value = float(self.get_parameter(name).value)

        if value < minimum:
            message = (
                f"Parameter '{name}' must be >= {minimum}; "
                f"using {default_value}"
            )
            self.get_logger().warn(
                message
            )
            return default_value

        if maximum is not None and value > maximum:
            message = (
                f"Parameter '{name}' must be <= {maximum}; "
                f"using {default_value}"
            )
            self.get_logger().warn(
                message
            )
            return default_value

        return value

    def _positive_int_parameter(
        self,
        name: str,
        default_value: int,
        minimum: int,
    ) -> int:
        value = int(self.get_parameter(name).value)

        if value < minimum:
            message = (
                f"Parameter '{name}' must be >= {minimum}; "
                f"using {default_value}"
            )
            self.get_logger().warn(
                message
            )
            return default_value

        return value

    def _create_ros_interfaces(self) -> None:
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._on_scan_received,
            qos_profile_sensor_data,
        )
        self.agv_cmd_sub = self.create_subscription(
            String,
            self.agv_cmd_topic,
            self._on_agv_command_received,
            10,
        )
        self.line_detection_pub = self.create_publisher(
            String,
            self.line_detection_topic,
            10,
        )
        self.radar_image_pub = self.create_publisher(
            CompressedImage,
            self.radar_image_topic,
            10,
        )

    def _create_window(self) -> None:
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                WINDOW_NAME,
                self.window_width_px,
                self.window_height_px,
            )
        except cv2.error as error:
            raise SystemExit(
                "Could not open the OpenCV radar window.\n"
                "Make sure this node is running in a graphical desktop "
                "session or with X forwarding enabled.\n"
                f"OpenCV error: {error}"
            ) from error

    def _on_scan_received(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        (
            self.nearest_straight_line,
            self.nearest_left_straight_line,
            self.nearest_right_straight_line,
        ) = self._get_nearest_straight_line_detections(msg)
        self.side_clearance_detection = self._get_side_clearance_detection(msg)
        self._publish_line_detections(self.nearest_straight_line)

    def _on_agv_command_received(self, msg: String) -> None:
        self.agv_command_is_moving = self._agv_command_has_motion(msg.data)

    def _publish_line_detections(
        self,
        nearest_straight_line: StraightLineDetection | None,
    ) -> None:
        (
            left_nearest_m,
            left_count,
            right_nearest_m,
            right_count,
        ) = self._get_side_line_horizontal_intersections()
        line_distance_m = self._line_distance(nearest_straight_line)
        line_angle_deg = self._line_angle(nearest_straight_line)
        line_side = self._line_side(nearest_straight_line)
        left_line = self.nearest_left_straight_line
        right_line = self.nearest_right_straight_line
        side_clearance = self.side_clearance_detection

        msg = String()
        msg.data = (
            f"left_count={left_count},"
            f"left_nearest_m={self._format_optional_distance(left_nearest_m)},"
            f"right_count={right_count},"
            f"right_nearest_m={self._format_optional_distance(right_nearest_m)},"
            "bush_line_distance_m="
            f"{self._format_optional_distance(line_distance_m)},"
            "bush_line_angle_deg="
            f"{self._format_optional_angle(line_angle_deg)},"
            f"bush_line_side={line_side},"
            "left_bush_line_distance_m="
            f"{self._format_optional_distance(self._line_distance(left_line))},"
            "left_bush_line_angle_deg="
            f"{self._format_optional_angle(self._line_angle(left_line))},"
            "right_bush_line_distance_m="
            f"{self._format_optional_distance(self._line_distance(right_line))},"
            "right_bush_line_angle_deg="
            f"{self._format_optional_angle(self._line_angle(right_line))},"
            f"side_clearance_count={self._side_clearance_count(side_clearance)},"
            "side_clearance_nearest_m="
            f"{self._format_optional_distance(self._side_clearance_distance(side_clearance))},"
            f"side_clearance_side={self._side_clearance_side(side_clearance)},"
            "side_clearance_x_m="
            f"{self._format_optional_distance(self._side_clearance_x(side_clearance))},"
            "side_clearance_y_m="
            f"{self._format_optional_distance(self._side_clearance_y(side_clearance))},"
            f"side_clearance_sticky={self._format_bool(self.side_clearance_sticky_suppressed)}"
        )
        self.line_detection_pub.publish(msg)

    @staticmethod
    def _get_selected_line_horizontal_intersection(
        nearest_straight_line: StraightLineDetection | None,
    ) -> tuple[float | None, int, float | None, int]:
        if nearest_straight_line is None:
            return None, 0, None, 0

        x_right_m = Lidar360RadarNode._horizontal_intersection_x_right_m(
            nearest_straight_line
        )
        if x_right_m is None:
            return None, 0, None, 0

        if x_right_m < 0.0:
            return abs(x_right_m), 1, None, 0

        if x_right_m > 0.0:
            return None, 0, x_right_m, 1

        return None, 0, None, 0

    def _get_side_line_horizontal_intersections(
        self,
    ) -> tuple[float | None, int, float | None, int]:
        left_nearest_m = self._horizontal_intersection_for_side(
            self.nearest_left_straight_line,
            "left",
        )
        right_nearest_m = self._horizontal_intersection_for_side(
            self.nearest_right_straight_line,
            "right",
        )

        return (
            left_nearest_m,
            1 if left_nearest_m is not None else 0,
            right_nearest_m,
            1 if right_nearest_m is not None else 0,
        )

    @staticmethod
    def _horizontal_intersection_for_side(
        line: StraightLineDetection | None,
        side: str,
    ) -> float | None:
        if line is None:
            return None

        x_right_m = Lidar360RadarNode._horizontal_intersection_x_right_m(line)
        if x_right_m is None:
            return None

        if side == "left" and x_right_m < 0.0:
            return abs(x_right_m)

        if side == "right" and x_right_m > 0.0:
            return x_right_m

        return None

    @staticmethod
    def _horizontal_intersection_x_right_m(
        line: StraightLineDetection,
    ) -> float | None:
        start_x, start_y = line.start_xy_m
        end_x, end_y = line.end_xy_m
        segment_y = end_y - start_y

        if math.isclose(segment_y, 0.0, abs_tol=1e-9):
            if abs(start_y) > REFERENCE_LINE_DETECTION_TOLERANCE_M:
                return None

            if abs(start_x) <= abs(end_x):
                return start_x

            return end_x

        if (start_y <= 0.0 <= end_y) or (end_y <= 0.0 <= start_y):
            ratio = -start_y / segment_y
            return start_x + ratio * (end_x - start_x)

        if abs(start_y) <= REFERENCE_LINE_DETECTION_TOLERANCE_M:
            return start_x

        if abs(end_y) <= REFERENCE_LINE_DETECTION_TOLERANCE_M:
            return end_x

        return None

    def _get_nearest_straight_line_detection(
        self,
        scan: LaserScan,
    ) -> StraightLineDetection | None:
        nearest_line, _, _ = self._get_nearest_straight_line_detections(scan)
        return nearest_line

    def _get_nearest_straight_line_detections(
        self,
        scan: LaserScan,
    ) -> tuple[
        StraightLineDetection | None,
        StraightLineDetection | None,
        StraightLineDetection | None,
    ]:
        best_line = None
        best_score = None
        best_left_line = None
        best_left_score = None
        best_right_line = None
        best_right_score = None

        def consider_line(line: StraightLineDetection) -> None:
            nonlocal best_line
            nonlocal best_score
            nonlocal best_left_line
            nonlocal best_left_score
            nonlocal best_right_line
            nonlocal best_right_score

            best_line, best_score = self._select_nearest_line(
                line,
                best_line,
                best_score,
            )
            side = self._line_side(line)
            if side == "left":
                best_left_line, best_left_score = self._select_nearest_line(
                    line,
                    best_left_line,
                    best_left_score,
                )
            elif side == "right":
                best_right_line, best_right_score = self._select_nearest_line(
                    line,
                    best_right_line,
                    best_right_score,
                )

        for cluster in self._get_scan_point_clusters(scan):
            line = self._fit_straight_line(cluster)
            if line is not None:
                consider_line(line)

            for window in self._iter_one_meter_windows(cluster):
                line = self._fit_straight_line(window)
                if line is None:
                    continue

                expanded_line = self._expand_straight_line_detection(
                    cluster,
                    line,
                )
                if expanded_line is not None:
                    line = expanded_line

                consider_line(line)

        return best_line, best_left_line, best_right_line

    @staticmethod
    def _select_nearest_line(
        candidate: StraightLineDetection,
        best_line: StraightLineDetection | None,
        best_score: tuple[float, float, int] | None,
    ) -> tuple[StraightLineDetection, tuple[float, float, int]]:
        score = (candidate.distance_m, candidate.rmse_m, -candidate.point_count)
        if best_score is None or score < best_score:
            return candidate, score

        return best_line, best_score

    def _get_scan_point_clusters(
        self,
        scan: LaserScan,
    ) -> list[list[tuple[float, float]]]:
        clusters: list[list[tuple[float, float]]] = []
        current_cluster: list[tuple[float, float]] = []
        previous_point: tuple[float, float] | None = None

        def finish_cluster() -> None:
            nonlocal current_cluster, previous_point

            if len(current_cluster) >= NEAREST_STRAIGHT_LINE_MIN_POINTS:
                clusters.append(current_cluster)

            current_cluster = []
            previous_point = None

        for index, distance_m in enumerate(scan.ranges):
            if not self._is_valid_range(scan, distance_m):
                finish_cluster()
                continue

            if distance_m > self.max_display_range_m:
                finish_cluster()
                continue

            angle_rad = self._to_robot_frame_angle(
                scan.angle_min + index * scan.angle_increment
            )
            if self._is_inside_agv_filtered_area(angle_rad, distance_m):
                finish_cluster()
                continue

            point = self._polar_to_robot_xy(angle_rad, distance_m)
            if (
                previous_point is not None
                and self._point_distance(previous_point, point)
                > NEAREST_STRAIGHT_LINE_MAX_POINT_GAP_M
            ):
                finish_cluster()

            current_cluster.append(point)
            previous_point = point

        finish_cluster()
        return self._merge_wrapped_clusters(clusters)

    def _get_side_clearance_detection(
        self,
        scan: LaserScan,
    ) -> SideClearanceDetection | None:
        candidates = []

        for index, distance_m in enumerate(scan.ranges):
            if not self._is_valid_range(scan, distance_m):
                continue

            if distance_m > self.max_display_range_m:
                continue

            angle_rad = self._to_robot_frame_angle(
                scan.angle_min + index * scan.angle_increment
            )
            if self._is_inside_agv_filtered_area(angle_rad, distance_m):
                continue

            x_right_m, y_back_m = self._polar_to_robot_xy(angle_rad, distance_m)
            side = self._side_clearance_region_side_for_point(
                x_right_m,
                y_back_m,
            )
            if side is None:
                continue

            candidates.append((side, x_right_m, y_back_m, float(distance_m)))

        if len(candidates) < self.side_clearance_detection_min_points:
            self.side_clearance_sticky_suppressed = False
            self.side_clearance_motion_track = None
            return None

        side, x_right_m, y_back_m, distance_m = min(
            candidates,
            key=lambda candidate: candidate[3],
        )
        detection = SideClearanceDetection(
            side=side,
            xy_m=(x_right_m, y_back_m),
            distance_m=distance_m,
            point_count=len(candidates),
        )

        if self._side_clearance_detection_is_sticky(detection):
            return None

        self.side_clearance_sticky_suppressed = False
        return detection

    def _side_clearance_detection_is_sticky(
        self,
        detection: SideClearanceDetection,
    ) -> bool:
        if not self.agv_command_is_moving:
            self.side_clearance_motion_track = None
            self.side_clearance_sticky_suppressed = False
            return False

        now = time.monotonic()
        track = self.side_clearance_motion_track
        if track is None or track[0] != detection.side:
            self.side_clearance_motion_track = (detection.side, detection.xy_m, now)
            self.side_clearance_sticky_suppressed = False
            return True

        _, reference_xy_m, track_start_time = track
        motion_m = self._point_distance(reference_xy_m, detection.xy_m)
        if motion_m >= self.side_clearance_sticky_max_motion_m:
            self.side_clearance_motion_track = (detection.side, detection.xy_m, now)
            self.side_clearance_sticky_suppressed = False
            return False

        elapsed_sec = now - track_start_time
        self.side_clearance_sticky_suppressed = (
            elapsed_sec >= self.side_clearance_sticky_timeout_sec
        )
        return True

    @staticmethod
    def _merge_wrapped_clusters(
        clusters: list[list[tuple[float, float]]],
    ) -> list[list[tuple[float, float]]]:
        if len(clusters) < 2:
            return clusters

        first_cluster = clusters[0]
        last_cluster = clusters[-1]
        if (
            Lidar360RadarNode._point_distance(last_cluster[-1], first_cluster[0])
            <= NEAREST_STRAIGHT_LINE_MAX_POINT_GAP_M
        ):
            merged_cluster = last_cluster + first_cluster
            return [merged_cluster] + clusters[1:-1]

        return clusters

    @staticmethod
    def _iter_one_meter_windows(
        cluster: list[tuple[float, float]],
    ) -> list[list[tuple[float, float]]]:
        windows = []
        point_count = len(cluster)

        for start_index in range(
            0,
            point_count - NEAREST_STRAIGHT_LINE_MIN_POINTS + 1,
        ):
            min_end_index = start_index + NEAREST_STRAIGHT_LINE_MIN_POINTS - 1

            for end_index in range(min_end_index, point_count):
                chord_m = Lidar360RadarNode._point_distance(
                    cluster[start_index],
                    cluster[end_index],
                )
                if chord_m >= NEAREST_STRAIGHT_LINE_LENGTH_M:
                    windows.append(cluster[start_index : end_index + 1])
                    break

        return windows

    @staticmethod
    def _fit_straight_line(
        points: list[tuple[float, float]],
    ) -> StraightLineDetection | None:
        if len(points) < NEAREST_STRAIGHT_LINE_MIN_POINTS:
            return None

        point_array = np.asarray(points, dtype=float)
        centroid = point_array.mean(axis=0)
        centered_points = point_array - centroid
        covariance = centered_points.T @ centered_points / len(points)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 0.0:
            return None

        direction = direction / direction_norm
        normal = np.array([-direction[1], direction[0]])
        residuals = np.abs(centered_points @ normal)
        rmse_m = float(np.sqrt(np.mean(residuals * residuals)))
        max_residual_m = float(np.max(residuals))
        if (
            rmse_m > NEAREST_STRAIGHT_LINE_MAX_RMSE_M
            or max_residual_m > NEAREST_STRAIGHT_LINE_MAX_RESIDUAL_M
        ):
            return None

        projections = centered_points @ direction
        min_projection = float(np.min(projections))
        max_projection = float(np.max(projections))
        span_m = max_projection - min_projection
        if span_m < NEAREST_STRAIGHT_LINE_LENGTH_M:
            return None

        start_xy = centroid + min_projection * direction
        end_xy = centroid + max_projection * direction
        closest_xy = Lidar360RadarNode._closest_point_on_segment_to_lidar(
            start_xy,
            end_xy,
        )
        distance_m = float(np.linalg.norm(closest_xy))
        angle_deg = Lidar360RadarNode._line_orientation_deg_from_front(
            float(direction[0]),
            float(direction[1]),
        )

        return StraightLineDetection(
            start_xy_m=(float(start_xy[0]), float(start_xy[1])),
            end_xy_m=(float(end_xy[0]), float(end_xy[1])),
            closest_xy_m=(float(closest_xy[0]), float(closest_xy[1])),
            distance_m=distance_m,
            angle_deg=angle_deg,
            point_count=len(points),
            rmse_m=rmse_m,
        )

    @staticmethod
    def _closest_point_on_segment_to_lidar(start_xy, end_xy):
        segment = end_xy - start_xy
        segment_length_squared = float(segment @ segment)
        if segment_length_squared <= 0.0:
            return start_xy

        ratio = -float(start_xy @ segment) / segment_length_squared
        ratio = min(max(ratio, 0.0), 1.0)
        return start_xy + ratio * segment

    @staticmethod
    def _expand_straight_line_detection(
        cluster: list[tuple[float, float]],
        line: StraightLineDetection,
    ) -> StraightLineDetection | None:
        aligned_points = [
            point
            for point in cluster
            if Lidar360RadarNode._point_is_on_straight_line_detection(
                point,
                line,
                include_segment_bounds=False,
            )
        ]

        if len(aligned_points) <= line.point_count:
            return line

        return Lidar360RadarNode._fit_straight_line(aligned_points)

    def run(self) -> None:
        while rclpy.ok() and not self.is_closing:
            rclpy.spin_once(self, timeout_sec=0.03)
            self._draw_radar()
            if self.show_window:
                self._process_keyboard()
                self._check_window_closed()

    def _draw_radar(self) -> None:
        canvas = np.zeros(
            (self.window_height_px, self.window_width_px, 3),
            dtype=np.uint8,
        )
        canvas[:] = BACKGROUND_COLOR

        center = (self.window_width_px // 2, self.window_height_px // 2)
        radius_px = self._radar_radius_px()
        scale_px_per_m = radius_px / self.max_display_range_m

        self._draw_agv_footprint(canvas, center, scale_px_per_m)
        self._draw_side_clearance_regions(canvas, center, scale_px_per_m)
        self._draw_grid(canvas, center, radius_px)
        self._draw_fixed_reference_line(
            canvas,
            center,
            scale_px_per_m,
            radius_px,
        )
        self._draw_scan_points(canvas, center, scale_px_per_m)
        self._draw_nearest_straight_line(canvas, center, scale_px_per_m)
        self._draw_center(canvas, center)
        self._draw_sector_distances(canvas)

        self._publish_radar_image(canvas)
        if self.show_window:
            cv2.imshow(WINDOW_NAME, canvas)

    def _publish_radar_image(self, canvas) -> None:
        now = time.monotonic()
        if (
            now - self.last_radar_image_publish_time
            < self.radar_image_publish_period_sec
        ):
            return

        self.last_radar_image_publish_time = now
        success, encoded_image = cv2.imencode(
            ".jpg",
            canvas,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if not success:
            self.get_logger().warn("Could not encode radar image as JPEG")
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        if self.latest_scan is not None:
            msg.header.frame_id = self.latest_scan.header.frame_id
        msg.format = "jpeg"
        msg.data = encoded_image.tobytes()
        self.radar_image_pub.publish(msg)

    def _radar_radius_px(self) -> int:
        margin_px = 86
        window_limit_px = min(self.window_width_px, self.window_height_px)
        return max(60, window_limit_px // 2 - margin_px)

    def _draw_grid(
        self,
        canvas,
        center: tuple[int, int],
        radius_px: int,
    ) -> None:
        for ring_index in range(1, 5):
            ring_radius = int(radius_px * ring_index / 4)
            distance_m = self.max_display_range_m * ring_index / 4
            cv2.circle(canvas, center, ring_radius, GRID_COLOR, 1)
            cv2.putText(
                canvas,
                f"{distance_m:.1f} m",
                (center[0] + 8, center[1] - ring_radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                GRID_COLOR,
                1,
                cv2.LINE_AA,
            )

        for angle_deg in range(0, 360, 30):
            angle_rad = math.radians(angle_deg)
            end_point = self._polar_to_pixel(angle_rad, radius_px, center)
            cv2.line(canvas, center, end_point, GRID_COLOR, 1, cv2.LINE_AA)
            label_point = self._polar_to_pixel(
                angle_rad,
                radius_px + 22,
                center,
            )
            self._put_centered_text(
                canvas,
                f"{angle_deg}",
                label_point,
                0.42,
                GRID_COLOR,
            )

        self._draw_cardinal_labels(canvas, center, radius_px)

    def _draw_cardinal_labels(
        self,
        canvas,
        center: tuple[int, int],
        radius_px: int,
    ) -> None:
        labels = [
            ("FRONT 0", 0.0),
            ("LEFT", 90.0),
            ("RIGHT", 270.0),
            ("BACK", 180.0),
        ]

        for label, angle_deg in labels:
            label_point = self._polar_to_pixel(
                math.radians(angle_deg),
                radius_px + 48,
                center,
            )
            self._put_centered_text(
                canvas,
                label,
                label_point,
                0.6,
                TEXT_COLOR,
                2,
            )

    def _draw_agv_footprint(
        self,
        canvas,
        center: tuple[int, int],
        scale_px_per_m: float,
    ) -> None:
        left_edge_m, right_edge_m, front_edge_m, back_edge_m = (
            self._agv_footprint_bounds_m()
        )

        top_left = self._robot_xy_to_pixel(
            left_edge_m,
            front_edge_m,
            center,
            scale_px_per_m,
        )
        bottom_right = self._robot_xy_to_pixel(
            right_edge_m,
            back_edge_m,
            center,
            scale_px_per_m,
        )

        overlay = canvas.copy()
        cv2.rectangle(overlay, top_left, bottom_right, AGV_COLOR, -1)
        cv2.addWeighted(overlay, AGV_ALPHA, canvas, 1.0 - AGV_ALPHA, 0, canvas)
        cv2.rectangle(
            canvas,
            top_left,
            bottom_right,
            AGV_COLOR,
            2,
            cv2.LINE_AA,
        )

    def _draw_side_clearance_regions(
        self,
        canvas,
        center: tuple[int, int],
        scale_px_per_m: float,
    ) -> None:
        region_bounds = self._side_clearance_region_bounds_m()
        if not region_bounds:
            return

        overlay = canvas.copy()
        polygons = []
        for _, x_min_m, x_max_m, y_min_m, y_max_m in region_bounds:
            polygon = np.array(
                [
                    self._robot_xy_to_pixel(
                        x_min_m,
                        y_min_m,
                        center,
                        scale_px_per_m,
                    ),
                    self._robot_xy_to_pixel(
                        x_max_m,
                        y_min_m,
                        center,
                        scale_px_per_m,
                    ),
                    self._robot_xy_to_pixel(
                        x_max_m,
                        y_max_m,
                        center,
                        scale_px_per_m,
                    ),
                    self._robot_xy_to_pixel(
                        x_min_m,
                        y_max_m,
                        center,
                        scale_px_per_m,
                    ),
                ],
                dtype=np.int32,
            )
            polygons.append(polygon)
            cv2.fillPoly(overlay, [polygon], SIDE_CLEARANCE_REGION_COLOR)

        cv2.addWeighted(
            overlay,
            SIDE_CLEARANCE_REGION_ALPHA,
            canvas,
            1.0 - SIDE_CLEARANCE_REGION_ALPHA,
            0,
            canvas,
        )
        for polygon in polygons:
            cv2.polylines(
                canvas,
                [polygon],
                True,
                SIDE_CLEARANCE_REGION_COLOR,
                2,
                cv2.LINE_AA,
            )

    def _draw_fixed_reference_line(
        self,
        canvas,
        center: tuple[int, int],
        scale_px_per_m: float,
        radar_radius_px: int,
    ) -> None:
        if self.target_horizontal_line_distance_m > self.max_display_range_m:
            return

        radius_px = self.target_horizontal_line_distance_m * scale_px_per_m
        left_point = self._polar_to_pixel(
            math.radians(90.0),
            radius_px,
            center,
        )
        right_point = self._polar_to_pixel(
            math.radians(-90.0),
            radius_px,
            center,
        )
        vertical_half_height_px = math.sqrt(
            max(0.0, radar_radius_px * radar_radius_px - radius_px * radius_px)
        )
        vertical_top_y = int(round(center[1] - vertical_half_height_px))
        vertical_bottom_y = int(round(center[1] + vertical_half_height_px))

        for point in (left_point, right_point):
            cv2.line(
                canvas,
                (point[0], vertical_top_y),
                (point[0], vertical_bottom_y),
                REFERENCE_LINE_COLOR,
                1,
                cv2.LINE_AA,
        )

        horizontal_left_point = (center[0] - radar_radius_px, center[1])
        horizontal_right_point = (center[0] + radar_radius_px, center[1])
        self._draw_dotted_line(
            canvas,
            horizontal_left_point,
            horizontal_right_point,
            REFERENCE_LINE_COLOR,
        )
        cv2.circle(
            canvas,
            left_point,
            5,
            REFERENCE_LINE_COLOR,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            right_point,
            5,
            REFERENCE_LINE_COLOR,
            -1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_dotted_line(
        canvas,
        start_point: tuple[int, int],
        end_point: tuple[int, int],
        color: tuple[int, int, int],
        dot_radius: int = 2,
        spacing_px: int = 12,
    ) -> None:
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        length_px = math.hypot(dx, dy)
        dot_count = max(1, int(length_px / spacing_px))

        for index in range(dot_count + 1):
            ratio = index / dot_count
            point = (
                int(round(start_point[0] + dx * ratio)),
                int(round(start_point[1] + dy * ratio)),
            )
            cv2.circle(canvas, point, dot_radius, color, -1, cv2.LINE_AA)

    def _draw_scan_points(
        self,
        canvas,
        center: tuple[int, int],
        scale_px_per_m: float,
    ) -> None:
        scan = self.latest_scan
        if scan is None:
            self._put_centered_text(
                canvas,
                "Waiting for LaserScan on " + self.scan_topic,
                center,
                0.7,
                TEXT_COLOR,
                2,
            )
            return

        for index, distance_m in enumerate(scan.ranges):
            if not self._is_valid_range(scan, distance_m):
                continue

            if distance_m > self.max_display_range_m:
                continue

            angle_rad = self._to_robot_frame_angle(
                scan.angle_min + index * scan.angle_increment
            )
            if self._is_inside_agv_filtered_area(angle_rad, distance_m):
                continue

            radius_px = distance_m * scale_px_per_m
            point = self._polar_to_pixel(angle_rad, radius_px, center)
            point_xy_m = self._polar_to_robot_xy(angle_rad, distance_m)
            is_alert = distance_m <= self.alert_distance_m
            is_on_nearest_bush_line = (
                self.nearest_straight_line is not None
                and self._point_is_on_straight_line_detection(
                    point_xy_m,
                    self.nearest_straight_line,
                )
            )
            is_on_reference_line = self._is_on_reference_vertical_line(
                angle_rad,
                distance_m,
            )
            is_on_horizontal_reference_line = (
                self._is_on_reference_horizontal_line(angle_rad, distance_m)
            )
            if is_alert:
                color = ALERT_COLOR
            elif is_on_nearest_bush_line:
                color = NEAREST_STRAIGHT_LINE_COLOR
            elif is_on_horizontal_reference_line:
                color = HORIZONTAL_REFERENCE_DETECTION_COLOR
            elif is_on_reference_line:
                color = REFERENCE_LINE_DETECTION_COLOR
            else:
                color = NORMAL_COLOR
            point_radius = 4 if is_alert or is_on_nearest_bush_line else 2

            cv2.circle(canvas, point, point_radius, color, -1, cv2.LINE_AA)

    def _draw_nearest_straight_line(
        self,
        canvas,
        center: tuple[int, int],
        scale_px_per_m: float,
    ) -> None:
        line = self.nearest_straight_line
        if line is None:
            return

        start_point = self._robot_xy_to_pixel(
            line.start_xy_m[0],
            line.start_xy_m[1],
            center,
            scale_px_per_m,
        )
        end_point = self._robot_xy_to_pixel(
            line.end_xy_m[0],
            line.end_xy_m[1],
            center,
            scale_px_per_m,
        )
        closest_point = self._robot_xy_to_pixel(
            line.closest_xy_m[0],
            line.closest_xy_m[1],
            center,
            scale_px_per_m,
        )

        cv2.line(
            canvas,
            start_point,
            end_point,
            NEAREST_STRAIGHT_LINE_COLOR,
            4,
            cv2.LINE_AA,
        )
        self._draw_dotted_line(
            canvas,
            center,
            closest_point,
            NEAREST_STRAIGHT_LINE_COLOR,
            dot_radius=2,
            spacing_px=14,
        )
        cv2.circle(
            canvas,
            closest_point,
            6,
            NEAREST_STRAIGHT_LINE_COLOR,
            -1,
            cv2.LINE_AA,
        )

    def _draw_center(self, canvas, center: tuple[int, int]) -> None:
        cv2.circle(canvas, center, 7, CENTER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 13, CENTER_COLOR, 1, cv2.LINE_AA)

    def _draw_sector_distances(self, canvas) -> None:
        scan = self.latest_scan
        distances = {
            "Front": self._sector_median(scan, 0.0),
            "Left": self._sector_median(scan, 90.0),
            "Right": self._sector_median(scan, -90.0),
            "Back": self._sector_median(scan, 180.0),
        }

        x = 18
        y = 32
        cv2.putText(
            canvas,
            f"Topic: {self.scan_topic}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        y += 26
        cv2.putText(
            canvas,
            (
                f"Alert <= {self.alert_distance_m:.2f} m | "
                f"sector {self.sector_width_deg:.1f} deg"
            ),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        y += 34

        for label, value in distances.items():
            text = f"{label}: {self._format_distance(value)}"
            color = TEXT_COLOR
            if value is not None and value <= self.alert_distance_m:
                color = ALERT_COLOR
            cv2.putText(
                canvas,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )
            y += 30

        line = self.nearest_straight_line
        line_color = NEAREST_STRAIGHT_LINE_COLOR if line else TEXT_COLOR
        cv2.putText(
            canvas,
            f"Possible nearest bush line: {self._format_line_distance(line)}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            line_color,
            2 if line else 1,
            cv2.LINE_AA,
        )
        y += 30
        cv2.putText(
            canvas,
            f"Line angle from 0 deg: {self._format_line_angle(line)}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            line_color,
            2 if line else 1,
            cv2.LINE_AA,
        )
        y += 30

        cv2.putText(
            canvas,
            "Press q or ESC to close",
            (x, self.window_height_px - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    def _sector_median(
        self,
        scan: LaserScan | None,
        center_angle_deg: float,
    ) -> float | None:
        if scan is None:
            return None

        center_angle_rad = math.radians(center_angle_deg)
        half_width_rad = math.radians(self.sector_width_deg / 2.0)
        values = []

        for index, distance_m in enumerate(scan.ranges):
            if not self._is_valid_range(scan, distance_m):
                continue

            angle_rad = self._to_robot_frame_angle(
                scan.angle_min + index * scan.angle_increment
            )
            angle_error = self._shortest_angle_error(
                angle_rad,
                center_angle_rad,
            )
            if abs(angle_error) <= half_width_rad:
                values.append(distance_m)

        if not values:
            return None

        return float(median(values))

    @staticmethod
    def _shortest_angle_error(angle_rad: float, reference_rad: float) -> float:
        return math.atan2(
            math.sin(angle_rad - reference_rad),
            math.cos(angle_rad - reference_rad),
        )

    @staticmethod
    def _to_robot_frame_angle(angle_rad: float) -> float:
        return angle_rad + math.radians(LIDAR_MOUNT_ROTATION_DEG)

    @staticmethod
    def _is_valid_range(scan: LaserScan, distance_m: float) -> bool:
        return (
            math.isfinite(distance_m)
            and distance_m >= scan.range_min
            and distance_m <= scan.range_max
        )

    @staticmethod
    def _agv_footprint_bounds_m() -> tuple[float, float, float, float]:
        right_edge_m = AGV_LIDAR_TO_RIGHT_EDGE_M
        left_edge_m = right_edge_m - AGV_WIDTH_M
        front_edge_m = AGV_FRONT_EDGE_BEHIND_LIDAR_M
        back_edge_m = front_edge_m + AGV_LENGTH_M
        return left_edge_m, right_edge_m, front_edge_m, back_edge_m

    @classmethod
    def _agv_point_filter_bounds_m(cls) -> tuple[float, float, float, float]:
        left_edge_m, right_edge_m, front_edge_m, back_edge_m = (
            cls._agv_footprint_bounds_m()
        )
        lidar_to_agv_gap_front_m = 0.0
        return (
            left_edge_m,
            right_edge_m,
            min(lidar_to_agv_gap_front_m, front_edge_m),
            back_edge_m,
        )

    def _side_clearance_region_bounds_m(
        self,
    ) -> list[tuple[str, float, float, float, float]]:
        if self.side_clearance_region_forward_m <= 0.0:
            return []

        outer_offset_m = self.side_clearance_region_outer_offset_m
        left_edge_m, right_edge_m, front_edge_m, _ = self._agv_footprint_bounds_m()
        front_limit_m = front_edge_m - self.side_clearance_region_forward_m
        bounds = []

        x_min_m = -outer_offset_m
        x_max_m = left_edge_m
        if x_max_m > x_min_m:
            bounds.append(("left", x_min_m, x_max_m, front_limit_m, front_edge_m))

        x_min_m = right_edge_m
        x_max_m = outer_offset_m
        if x_max_m > x_min_m:
            bounds.append(("right", x_min_m, x_max_m, front_limit_m, front_edge_m))

        return bounds

    def _side_clearance_region_side_for_point(
        self,
        x_right_m: float,
        y_back_m: float,
    ) -> str | None:
        for side, x_min_m, x_max_m, y_min_m, y_max_m in (
            self._side_clearance_region_bounds_m()
        ):
            if x_min_m <= x_right_m <= x_max_m and y_min_m <= y_back_m <= y_max_m:
                return side

        return None

    @staticmethod
    def _agv_command_has_motion(command: str) -> bool:
        command = str(command).strip().lower()
        if command in ("", "s", "stop"):
            return False

        parts = command.split(",", maxsplit=1)
        if len(parts) != 2:
            return False

        try:
            right_rpm = float(parts[0].strip())
            left_rpm = float(parts[1].strip())
        except ValueError:
            return False

        return (
            abs(right_rpm) > AGV_MOVING_RPM_EPSILON
            or abs(left_rpm) > AGV_MOVING_RPM_EPSILON
        )

    def _is_on_reference_vertical_line(
        self,
        angle_rad: float,
        distance_m: float,
    ) -> bool:
        x_right_m, _ = self._polar_to_robot_xy(angle_rad, distance_m)
        left_line_m = -self.target_horizontal_line_distance_m
        right_line_m = self.target_horizontal_line_distance_m
        return (
            abs(x_right_m - left_line_m) <= REFERENCE_LINE_DETECTION_TOLERANCE_M
            or abs(x_right_m - right_line_m)
            <= REFERENCE_LINE_DETECTION_TOLERANCE_M
        )

    @classmethod
    def _is_inside_agv_filtered_area(
        cls,
        angle_rad: float,
        distance_m: float,
    ) -> bool:
        x_right_m, y_back_m = cls._polar_to_robot_xy(angle_rad, distance_m)
        left_edge_m, right_edge_m, front_edge_m, back_edge_m = (
            cls._agv_point_filter_bounds_m()
        )

        return (
            left_edge_m <= x_right_m <= right_edge_m
            and front_edge_m <= y_back_m <= back_edge_m
        )

    @classmethod
    def _is_on_reference_horizontal_line(
        cls,
        angle_rad: float,
        distance_m: float,
    ) -> bool:
        _, y_back_m = cls._polar_to_robot_xy(angle_rad, distance_m)
        return abs(y_back_m) <= REFERENCE_LINE_DETECTION_TOLERANCE_M

    @staticmethod
    def _polar_to_robot_xy(
        angle_rad: float,
        distance_m: float,
    ) -> tuple[float, float]:
        x_right_m = -math.sin(angle_rad) * distance_m
        y_back_m = -math.cos(angle_rad) * distance_m
        return x_right_m, y_back_m

    @staticmethod
    def _point_distance(
        first_point: tuple[float, float],
        second_point: tuple[float, float],
    ) -> float:
        return math.hypot(
            first_point[0] - second_point[0],
            first_point[1] - second_point[1],
        )

    @staticmethod
    def _line_orientation_deg_from_front(
        x_right_direction: float,
        y_back_direction: float,
    ) -> float:
        if (
            math.isclose(x_right_direction, 0.0)
            and math.isclose(y_back_direction, 0.0)
        ):
            return 0.0

        if y_back_direction > 0.0 or (
            math.isclose(y_back_direction, 0.0) and x_right_direction < 0.0
        ):
            x_right_direction *= -1.0
            y_back_direction *= -1.0

        return math.degrees(math.atan2(x_right_direction, -y_back_direction))

    @staticmethod
    def _point_is_on_straight_line_detection(
        point_xy_m: tuple[float, float],
        line: StraightLineDetection,
        include_segment_bounds: bool = True,
    ) -> bool:
        start_x, start_y = line.start_xy_m
        end_x, end_y = line.end_xy_m
        segment_x = end_x - start_x
        segment_y = end_y - start_y
        segment_length_m = math.hypot(segment_x, segment_y)
        if segment_length_m <= 0.0:
            return False

        direction_x = segment_x / segment_length_m
        direction_y = segment_y / segment_length_m
        relative_x = point_xy_m[0] - start_x
        relative_y = point_xy_m[1] - start_y
        projection_m = relative_x * direction_x + relative_y * direction_y
        residual_m = abs(relative_x * direction_y - relative_y * direction_x)
        if residual_m > NEAREST_STRAIGHT_LINE_MAX_RESIDUAL_M:
            return False

        if not include_segment_bounds:
            return True

        margin_m = NEAREST_STRAIGHT_LINE_MAX_POINT_GAP_M
        return -margin_m <= projection_m <= segment_length_m + margin_m

    @staticmethod
    def _polar_to_pixel(
        angle_rad: float,
        radius_px: float,
        center: tuple[int, int],
    ) -> tuple[int, int]:
        x = center[0] - math.sin(angle_rad) * radius_px
        y = center[1] - math.cos(angle_rad) * radius_px
        return int(round(x)), int(round(y))

    @staticmethod
    def _robot_xy_to_pixel(
        x_right_m: float,
        y_back_m: float,
        center: tuple[int, int],
        scale_px_per_m: float,
    ) -> tuple[int, int]:
        x = center[0] + x_right_m * scale_px_per_m
        y = center[1] + y_back_m * scale_px_per_m
        return int(round(x)), int(round(y))

    @staticmethod
    def _put_centered_text(
        canvas,
        text: str,
        center: tuple[int, int],
        font_scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        origin = (
            int(center[0] - size[0] / 2),
            int(center[1] + size[1] / 2),
        )
        cv2.putText(
            canvas,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _format_distance(distance_m: float | None) -> str:
        if distance_m is None:
            return "--"

        return f"{distance_m:.2f} m"

    @staticmethod
    def _line_distance(line: StraightLineDetection | None) -> float | None:
        if line is None:
            return None

        return line.distance_m

    @staticmethod
    def _line_angle(line: StraightLineDetection | None) -> float | None:
        if line is None:
            return None

        return line.angle_deg

    @staticmethod
    def _line_side(line: StraightLineDetection | None) -> str:
        if line is None:
            return "none"

        closest_x_right_m = line.closest_xy_m[0]
        if closest_x_right_m < 0.0:
            return "left"

        if closest_x_right_m > 0.0:
            return "right"

        return "center"

    @staticmethod
    def _side_clearance_count(
        detection: SideClearanceDetection | None,
    ) -> int:
        if detection is None:
            return 0

        return detection.point_count

    @staticmethod
    def _side_clearance_distance(
        detection: SideClearanceDetection | None,
    ) -> float | None:
        if detection is None:
            return None

        return detection.distance_m

    @staticmethod
    def _side_clearance_side(
        detection: SideClearanceDetection | None,
    ) -> str:
        if detection is None:
            return "none"

        return detection.side

    @staticmethod
    def _side_clearance_x(
        detection: SideClearanceDetection | None,
    ) -> float | None:
        if detection is None:
            return None

        return detection.xy_m[0]

    @staticmethod
    def _side_clearance_y(
        detection: SideClearanceDetection | None,
    ) -> float | None:
        if detection is None:
            return None

        return detection.xy_m[1]

    @staticmethod
    def _format_line_distance(line: StraightLineDetection | None) -> str:
        if line is None:
            return "--"

        return f"{line.distance_m:.2f} m"

    @staticmethod
    def _format_line_angle(line: StraightLineDetection | None) -> str:
        if line is None:
            return "--"

        return f"{line.angle_deg:+.1f} deg"

    @staticmethod
    def _format_optional_distance(distance_m: float | None) -> str:
        if distance_m is None:
            return "none"

        return f"{distance_m:.3f}"

    @staticmethod
    def _format_optional_angle(angle_deg: float | None) -> str:
        if angle_deg is None:
            return "none"

        return f"{angle_deg:.2f}"

    @staticmethod
    def _format_bool(value: bool) -> str:
        return "true" if value else "false"

    def _process_keyboard(self) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.is_closing = True

    def _check_window_closed(self) -> None:
        try:
            visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
            if visible == 0:
                self.is_closing = True
        except cv2.error:
            pass

    @staticmethod
    def close_window() -> None:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass


def _require_opencv() -> None:
    if cv2 is None or np is None:
        raise SystemExit(
            "OpenCV Python bindings are required for lidar_360_radar_node.\n"
            "Install them with: sudo apt install python3-opencv"
        )


def main(args: list[str] | None = None) -> None:
    _require_opencv()
    rclpy.init(args=args)

    node = None

    try:
        node = Lidar360RadarNode()
        node.run()

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.close_window()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
