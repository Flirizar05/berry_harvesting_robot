#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


DEFAULT_COMMAND_TOPIC = "/agv/rpm_cmd"
DEFAULT_LINE_DETECTION_TOPIC = "/agv/line_detections"
DEFAULT_CONTROL_MODE_TOPIC = "/agv/control_mode"
DEFAULT_CONTROL_STATUS_TOPIC = "/agv/position_control_status"
CONTROL_MODE_MANUAL = "manual"
CONTROL_MODE_AUTOMATIC = "automatic"
DEFAULT_CONTROL_PERIOD_SEC = 0.10
DEFAULT_LINE_DETECTION_TIMEOUT_SEC = 0.50
DEFAULT_TARGET_BUSH_LINE_ANGLE_DEG = 0.0
DEFAULT_TARGET_HORIZONTAL_LINE_DISTANCE_M = 2.0
DEFAULT_ANGLE_GAIN_RPM_PER_DEG = 10.0
DEFAULT_DISTANCE_GAIN_RPM_PER_M = 5.0
DEFAULT_MAX_ABS_RPM = 150.0
DEFAULT_MIN_ABS_RPM = 0.0
BASE_RPM = 100.0


class AgvPositionControlNode(Node):
    """Position control node scaffold using detections published by the radar."""

    def __init__(self) -> None:
        super().__init__("agv_position_control_node")

        self.left_line_detection_m: float | None = None
        self.right_line_detection_m: float | None = None
        self.left_line_detection_count = 0
        self.right_line_detection_count = 0
        self.bush_line_distance_m: float | None = None
        self.bush_line_angle_deg: float | None = None
        self.bush_line_side: str | None = None
        self.last_line_detection_time = None
        self.line_detections_are_fresh = False
        self.control_mode = CONTROL_MODE_MANUAL
        self.position_control_active = False
        self.orientation_control_active = False

        self._declare_parameters()
        self._load_parameters()
        self._create_ros_interfaces()

        self.get_logger().info("agv_position_control_node ready")
        self.get_logger().info(f"Command topic: {self.command_topic}")
        self.get_logger().info(
            f"Radar line detection topic: {self.line_detection_topic}"
        )
        self.get_logger().info(
            f"Control mode topic: {self.control_mode_topic}"
        )
        self.get_logger().info(
            f"Control status topic: {self.control_status_topic}"
        )
        self.get_logger().info(
            f"Control period: {self.control_period_sec:.2f} s"
        )
        self.get_logger().info(
            "Bush line controller: "
            f"target={self.target_bush_line_angle_deg:.2f} deg, "
            f"distance_target={self.target_horizontal_line_distance_m:.3f} m, "
            f"angle_gain={self.angle_gain_rpm_per_deg:.3f} rpm/deg, "
            f"distance_gain={self.distance_gain_rpm_per_m:.3f} rpm/m"
        )
        self.get_logger().info(
            "Automatic base RPM: "
            f"min_abs={self.min_abs_rpm:.1f}, "
            f"max_abs={self.max_abs_rpm:.1f}, "
            f"base={BASE_RPM:.1f}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("command_topic", DEFAULT_COMMAND_TOPIC)
        self.declare_parameter(
            "line_detection_topic",
            DEFAULT_LINE_DETECTION_TOPIC,
        )
        self.declare_parameter(
            "control_mode_topic",
            DEFAULT_CONTROL_MODE_TOPIC,
        )
        self.declare_parameter(
            "control_status_topic",
            DEFAULT_CONTROL_STATUS_TOPIC,
        )
        self.declare_parameter("control_period_sec", DEFAULT_CONTROL_PERIOD_SEC)
        self.declare_parameter(
            "line_detection_timeout_sec",
            DEFAULT_LINE_DETECTION_TIMEOUT_SEC,
        )
        self.declare_parameter(
            "target_bush_line_angle_deg",
            DEFAULT_TARGET_BUSH_LINE_ANGLE_DEG,
        )
        self.declare_parameter(
            "target_horizontal_line_distance_m",
            DEFAULT_TARGET_HORIZONTAL_LINE_DISTANCE_M,
        )
        self.declare_parameter(
            "angle_gain_rpm_per_deg",
            DEFAULT_ANGLE_GAIN_RPM_PER_DEG,
        )
        self.declare_parameter(
            "distance_gain_rpm_per_m",
            DEFAULT_DISTANCE_GAIN_RPM_PER_M,
        )
        self.declare_parameter("max_abs_rpm", DEFAULT_MAX_ABS_RPM)
        self.declare_parameter("min_abs_rpm", DEFAULT_MIN_ABS_RPM)

    def _load_parameters(self) -> None:
        self.command_topic = str(self.get_parameter("command_topic").value).strip()
        if not self.command_topic:
            self.command_topic = DEFAULT_COMMAND_TOPIC

        self.line_detection_topic = str(
            self.get_parameter("line_detection_topic").value
        ).strip()
        if not self.line_detection_topic:
            self.line_detection_topic = DEFAULT_LINE_DETECTION_TOPIC

        self.control_mode_topic = str(
            self.get_parameter("control_mode_topic").value
        ).strip()
        if not self.control_mode_topic:
            self.control_mode_topic = DEFAULT_CONTROL_MODE_TOPIC

        self.control_status_topic = str(
            self.get_parameter("control_status_topic").value
        ).strip()
        if not self.control_status_topic:
            self.control_status_topic = DEFAULT_CONTROL_STATUS_TOPIC

        self.control_period_sec = self._positive_float_parameter(
            "control_period_sec",
            DEFAULT_CONTROL_PERIOD_SEC,
            minimum=0.01,
        )
        self.line_detection_timeout_sec = self._positive_float_parameter(
            "line_detection_timeout_sec",
            DEFAULT_LINE_DETECTION_TIMEOUT_SEC,
            minimum=0.01,
        )
        self.target_bush_line_angle_deg = self._float_parameter(
            "target_bush_line_angle_deg",
            DEFAULT_TARGET_BUSH_LINE_ANGLE_DEG,
        )
        self.target_horizontal_line_distance_m = self._positive_float_parameter(
            "target_horizontal_line_distance_m",
            DEFAULT_TARGET_HORIZONTAL_LINE_DISTANCE_M,
            minimum=0.0,
        )
        self.angle_gain_rpm_per_deg = self._float_parameter(
            "angle_gain_rpm_per_deg",
            DEFAULT_ANGLE_GAIN_RPM_PER_DEG,
        )
        self.distance_gain_rpm_per_m = self._float_parameter(
            "distance_gain_rpm_per_m",
            DEFAULT_DISTANCE_GAIN_RPM_PER_M,
        )
        self.max_abs_rpm = self._positive_float_parameter(
            "max_abs_rpm",
            DEFAULT_MAX_ABS_RPM,
            minimum=1.0,
        )
        self.min_abs_rpm = self._positive_float_parameter(
            "min_abs_rpm",
            DEFAULT_MIN_ABS_RPM,
            minimum=0.0,
        )

    def _float_parameter(self, name: str, default_value: float) -> float:
        value = self.get_parameter(name).value

        try:
            return float(value)
        except (TypeError, ValueError):
            self.get_logger().warn(
                f"Parameter '{name}' must be numeric; using {default_value}"
            )
            return default_value

    def _positive_float_parameter(
        self,
        name: str,
        default_value: float,
        minimum: float,
    ) -> float:
        value = float(self.get_parameter(name).value)

        if value < minimum:
            self.get_logger().warn(
                f"Parameter '{name}' must be >= {minimum}; using {default_value}"
            )
            return default_value

        return value

    def _create_ros_interfaces(self) -> None:
        self.command_pub = self.create_publisher(String, self.command_topic, 10)
        self.control_status_pub = self.create_publisher(
            String,
            self.control_status_topic,
            10,
        )
        self.line_detection_sub = self.create_subscription(
            String,
            self.line_detection_topic,
            self._on_line_detection_received,
            10,
        )
        self.control_mode_sub = self.create_subscription(
            String,
            self.control_mode_topic,
            self._on_control_mode_received,
            10,
        )
        self.control_timer = self.create_timer(
            self.control_period_sec,
            self._on_control_timer,
        )

    def _on_line_detection_received(self, msg: String) -> None:
        try:
            values = self._parse_key_value_message(msg.data)
            self.left_line_detection_count = int(values.get("left_count", "0"))
            self.left_line_detection_m = self._parse_optional_distance(
                values.get("left_nearest_m")
            )
            self.right_line_detection_count = int(values.get("right_count", "0"))
            self.right_line_detection_m = self._parse_optional_distance(
                values.get("right_nearest_m")
            )
            self.bush_line_distance_m = self._parse_optional_distance(
                values.get("bush_line_distance_m")
                or values.get("line_1m_distance_m")
            )
            self.bush_line_angle_deg = self._parse_optional_float(
                values.get("bush_line_angle_deg")
                or values.get("line_1m_angle_deg")
            )
            self.bush_line_side = self._parse_optional_side(
                values.get("bush_line_side")
            )
            self.last_line_detection_time = self.get_clock().now()
            self.line_detections_are_fresh = True

        except ValueError as error:
            self.get_logger().warn(
                f"Ignoring invalid line detection message: {error}"
            )

    def _on_control_mode_received(self, msg: String) -> None:
        control_mode = msg.data.strip().lower()
        if control_mode not in (CONTROL_MODE_MANUAL, CONTROL_MODE_AUTOMATIC):
            self.get_logger().warn(
                f"Ignoring invalid control mode: '{msg.data}'"
            )
            return

        if control_mode == self.control_mode:
            return

        self.control_mode = control_mode
        self.get_logger().info(f"Control mode changed to {self.control_mode}")
        if self.control_mode == CONTROL_MODE_MANUAL:
            self._publish_stop_command()

    def _on_control_timer(self) -> None:
        if self.control_mode != CONTROL_MODE_AUTOMATIC:
            self._clear_control_activity()
            self._publish_control_status(0, 0)
            return

        if self._line_detection_data_is_stale():
            self._clear_line_detections()
            self._publish_stop_command()
            self._clear_control_activity()
            self._publish_control_status(0, 0)
            return

        wheel_command = self.control_loop()
        if wheel_command is None:
            self._publish_stop_command()
            self._clear_control_activity()
            self._publish_control_status(0, 0)
            self.get_logger().debug(
                "No bush line angle available; publishing stop command"
            )
            return

        right_rpm, left_rpm = wheel_command
        self._publish_wheel_command(right_rpm, left_rpm)
        self._publish_control_status(right_rpm, left_rpm)
        self.get_logger().debug(
            "Line detections: "
            f"left={self._format_optional_distance(self.left_line_detection_m)} m "
            f"right={self._format_optional_distance(self.right_line_detection_m)} m "
            "bush_line_distance="
            f"{self._format_optional_distance(self.bush_line_distance_m)} m "
            "bush_line_angle="
            f"{self._format_optional_angle(self.bush_line_angle_deg)} deg "
            f"bush_line_side={self.bush_line_side or 'none'} "
            f"control_mode={self.control_mode} "
            f"rpm_cmd={right_rpm},{left_rpm}"
        )

    def _line_detection_data_is_stale(self) -> bool:
        if self.last_line_detection_time is None:
            return True

        age_sec = (
            self.get_clock().now() - self.last_line_detection_time
        ).nanoseconds / 1_000_000_000
        return age_sec > self.line_detection_timeout_sec

    def _clear_line_detections(self) -> None:
        if not self.line_detections_are_fresh:
            return

        self.left_line_detection_m = None
        self.right_line_detection_m = None
        self.left_line_detection_count = 0
        self.right_line_detection_count = 0
        self.bush_line_distance_m = None
        self.bush_line_angle_deg = None
        self.bush_line_side = None
        self.line_detections_are_fresh = False
        self.get_logger().warn("Radar line detections timed out")

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
    def _parse_optional_distance(value: str | None) -> float | None:
        return AgvPositionControlNode._parse_optional_float(value)

    @staticmethod
    def _parse_optional_float(value: str | None) -> float | None:
        if value is None or value.lower() == "none":
            return None

        return float(value)

    @staticmethod
    def _parse_optional_side(value: str | None) -> str | None:
        if value is None:
            return None

        side = value.strip().lower()
        if side in ("left", "right"):
            return side

        return None

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

    def _publish_wheel_command(self, right_rpm: int, left_rpm: int) -> None:
        msg = String()
        msg.data = f"{right_rpm},{left_rpm}"
        self.command_pub.publish(msg)

    def _publish_stop_command(self) -> None:
        self._publish_wheel_command(0, 0)

    def _publish_control_status(self, right_rpm: int, left_rpm: int) -> None:
        msg = String()
        msg.data = (
            f"orientation_active={self._format_bool(self.orientation_control_active)},"
            f"position_active={self._format_bool(self.position_control_active)},"
            f"right_rpm={right_rpm},"
            f"left_rpm={left_rpm}"
        )
        self.control_status_pub.publish(msg)

    def _clear_control_activity(self) -> None:
        self.orientation_control_active = False
        self.position_control_active = False

    def control_loop(self) -> tuple[int, int] | None:
        if self.bush_line_angle_deg is None:
            self._clear_control_activity()
            return None

        theta_ref_deg = self.target_bush_line_angle_deg
        theta_cur_deg = self.bush_line_angle_deg
        angle_error_deg = theta_ref_deg - theta_cur_deg

        angle_correction_rpm = self.angle_gain_rpm_per_deg * angle_error_deg
        (
            distance_correction_rpm,
            self.position_control_active,
        ) = self._distance_correction_rpm()
        self.orientation_control_active = True

        base_rpm = BASE_RPM
        turn_correction_rpm = angle_correction_rpm - distance_correction_rpm

        right_rpm = base_rpm + turn_correction_rpm
        left_rpm = base_rpm - turn_correction_rpm

        return self._limit_wheel_rpm(right_rpm), self._limit_wheel_rpm(left_rpm)

    def _distance_correction_rpm(self) -> tuple[float, bool]:
        horizontal_distance_m = self._horizontal_detection_for_bush_line_side()
        if horizontal_distance_m is None:
            return 0.0, False

        distance_error_m = (
            self.target_horizontal_line_distance_m - horizontal_distance_m
        )
        return self.distance_gain_rpm_per_m * distance_error_m, True

    def _horizontal_detection_for_bush_line_side(self) -> float | None:
        if self.bush_line_side == "left":
            return self.left_line_detection_m

        if self.bush_line_side == "right":
            return self.right_line_detection_m

        return None

    def _clamp_rpm(self, rpm: float) -> int:
        bounded_rpm = max(0.0, min(rpm, self.max_abs_rpm))
        return int(round(bounded_rpm))

    def _limit_wheel_rpm(self, rpm: float) -> int:
        if rpm < self.min_abs_rpm:
            return 0

        bounded_rpm = max(0.0, min(rpm, self.max_abs_rpm))
        return int(round(bounded_rpm))

    @staticmethod
    def _format_bool(value: bool) -> str:
        return "true" if value else "false"


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)

    node = AgvPositionControlNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
