#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from tf2_ros import TransformException


class EyeInHandNode(Node):
    """Transform the latest camera-space target point into the robot base frame."""

    def __init__(self) -> None:
        super().__init__("eyeinhand_node")

        self._declare_parameters()
        self._load_parameters()
        self._create_tf_interfaces()
        self._create_ros_interfaces()

        self.latest_point: PointStamped | None = None
        self.latest_point_seq = 0

        self.is_busy = False
        self.compute_start_time = None
        self.point_seq_at_trigger = 0

        self.status_publisher.publish(String(data="IDLE"))
        self.get_logger().info(
            f"eyeinhand_node ready. "
            f"input='{self.input_point_topic}', "
            f"output='{self.output_point_topic}', "
            f"target_frame='{self.target_frame}', "
            f"offsets(m)=({self.x_offset_m:+.3f},"
            f"{self.y_offset_m:+.3f},"
            f"{self.z_offset_m:+.3f})"
        )

        self.timer = self.create_timer(0.02, self._timer_callback)

    def _declare_parameters(self) -> None:
        self.declare_parameter("input_point_topic", "camera_sphere")
        self.declare_parameter("output_point_topic", "target_base")
        self.declare_parameter("target_frame", "elfin_base")

        self.declare_parameter("tf_timeout_sec", 0.8)
        self.declare_parameter("compute_timeout_sec", 3.0)
        self.declare_parameter("require_fresh_point", False)

        # Offsets are applied after the point has been transformed into target_frame.
        self.declare_parameter("x_offset_m", 0.0)
        self.declare_parameter("y_offset_m", 0.0)
        self.declare_parameter("z_offset_m", 0.11)

    def _load_parameters(self) -> None:
        self.input_point_topic = str(self.get_parameter("input_point_topic").value)
        self.output_point_topic = str(self.get_parameter("output_point_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)

        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.compute_timeout_sec = float(
            self.get_parameter("compute_timeout_sec").value
        )
        self.require_fresh_point = bool(
            self.get_parameter("require_fresh_point").value
        )

        self.x_offset_m = float(self.get_parameter("x_offset_m").value)
        self.y_offset_m = float(self.get_parameter("y_offset_m").value)
        self.z_offset_m = float(self.get_parameter("z_offset_m").value)

    def _create_tf_interfaces(self) -> None:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def _create_ros_interfaces(self) -> None:
        self.point_subscription = self.create_subscription(
            PointStamped,
            self.input_point_topic,
            self._on_point,
            10,
        )

        self.target_publisher = self.create_publisher(
            PointStamped,
            self.output_point_topic,
            10,
        )

        self.command_subscription = self.create_subscription(
            String,
            "/eyeinhand/cmd",
            self._on_command,
            10,
        )

        self.status_publisher = self.create_publisher(
            String,
            "/eyeinhand/status",
            10,
        )

    def _on_point(self, msg: PointStamped) -> None:
        self.latest_point = msg
        self.latest_point_seq += 1

    def _on_command(self, msg: String) -> None:
        command = msg.data.strip().upper()
        if command != "COMPUTE":
            return

        if self.is_busy:
            self.get_logger().info("COMPUTE command ignored because the node is busy.")
            return

        # Reload offsets at trigger time so runtime parameter updates are respected.
        self.x_offset_m = float(self.get_parameter("x_offset_m").value)
        self.y_offset_m = float(self.get_parameter("y_offset_m").value)
        self.z_offset_m = float(self.get_parameter("z_offset_m").value)

        self.is_busy = True
        self.compute_start_time = self.get_clock().now()
        self.point_seq_at_trigger = self.latest_point_seq

        self.status_publisher.publish(String(data="BUSY"))
        self.get_logger().info(
            f"COMPUTE received. "
            f"offsets(m)=({self.x_offset_m:+.3f},"
            f"{self.y_offset_m:+.3f},"
            f"{self.z_offset_m:+.3f})"
        )

    def _timer_callback(self) -> None:
        if not self.is_busy:
            return

        now = self.get_clock().now()
        elapsed_sec = (now - self.compute_start_time).nanoseconds * 1e-9

        if elapsed_sec > self.compute_timeout_sec:
            self._finish(False, "compute_timeout")
            return

        if self.latest_point is None:
            return

        if self.require_fresh_point and self.latest_point_seq <= self.point_seq_at_trigger:
            return

        input_point = self.latest_point
        source_frame = input_point.header.frame_id

        if not source_frame:
            self._finish(False, "empty input frame_id")
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().debug(f"TF lookup failed: {exc}")
            return

        try:
            transformed_point = do_transform_point(input_point, transform)

            transformed_point.point.x += self.x_offset_m
            transformed_point.point.y += self.y_offset_m
            transformed_point.point.z += self.z_offset_m

            transformed_point.header.stamp = self.get_clock().now().to_msg()
            transformed_point.header.frame_id = self.target_frame

            self.target_publisher.publish(transformed_point)

            self._finish(
                True,
                f"{source_frame} -> {self.target_frame} "
                f"xyz=({transformed_point.point.x:.3f},"
                f"{transformed_point.point.y:.3f},"
                f"{transformed_point.point.z:.3f}) "
                f"offset=({self.x_offset_m:+.3f},"
                f"{self.y_offset_m:+.3f},"
                f"{self.z_offset_m:+.3f})",
            )
        except Exception as exc:
            self._finish(False, f"transform error: {exc}")

    def _finish(self, success: bool, reason: str) -> None:
        self.status_publisher.publish(
            String(data="DONE_OK" if success else "DONE_FAIL")
        )
        self.is_busy = False
        self.status_publisher.publish(String(data="IDLE"))

        if success:
            self.get_logger().info(f"DONE_OK ({reason})")
        else:
            self.get_logger().warn(f"DONE_FAIL ({reason})")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EyeInHandNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()