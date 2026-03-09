#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from geometry_msgs.msg import Wrench
from onrobot_rg2ft_msgs.msg import RG2FTCommand
from rclpy.node import Node
from std_msgs.msg import Float32, String


def normalize_command(command: str) -> str:
    """Normalize text commands received from ROS topics."""
    return (command or "").strip().upper()


class GripperNode(Node):
    """Control the RG2FT gripper for calibration, grasp, release, and hold states."""

    def __init__(self) -> None:
        super().__init__("gripper_node")

        self._declare_parameters()
        self._load_parameters()
        self._initialize_state()
        self._create_ros_interfaces()

        self.timer = self.create_timer(self.loop_dt, self._control_loop)

        self._publish_status("IDLE")
        self.get_logger().info(
            f"gripper_node ready. "
            f"cmd={self.command_topic}, "
            f"status={self.status_topic}, "
            f"wrench={self.wrench_topic}, "
            f"force_ref={self.force_reference_n} N, "
            f"min_contact={self.min_contact_force_n} N, "
            f"release_force={self.release_force_n} N, "
            f"release_hold={self.release_hold_sec}s"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("wrench_topic", "/left_wrench")
        self.declare_parameter("cmd_topic", "/gripper/cmd")
        self.declare_parameter("status_topic", "/gripper/status")
        self.declare_parameter("command_topic", "/gripper/command")

        self.declare_parameter("kp", 0.02)
        self.declare_parameter("ki", 0.005)
        self.declare_parameter("kd", 0.001)
        self.declare_parameter("int_max", 200.0)

        self.declare_parameter("force_ref", 20.0)
        self.declare_parameter("force_tol", 2.0)
        self.declare_parameter("stable_cycles", 50)

        self.declare_parameter("use_force_contact", True)
        self.declare_parameter("min_contact_force", 1.0)
        self.declare_parameter("contact_stable_cycles", 25)

        self.declare_parameter("width_open", 1000)
        self.declare_parameter("width_closed", 0)
        self.declare_parameter("width_init", 900)

        self.declare_parameter("max_step_per_cycle", 20.0)
        self.declare_parameter("min_close_step", 3.0)
        self.declare_parameter("closed_margin", 25)
        self.declare_parameter("min_close_delta", 50)

        self.declare_parameter("loop_dt", 0.01)
        self.declare_parameter("calib_timeout_sec", 3.0)
        self.declare_parameter("grasp_timeout_sec", 5.0)

        self.declare_parameter("release_hold_sec", 1.5)
        self.declare_parameter("release_force", 20.0)

        self.declare_parameter("offset_samples", 50)
        self.declare_parameter("calib_open_before", True)

        self.declare_parameter("pub_debug_force", True)
        self.declare_parameter("debug_force_topic", "/gripper/debug_force_con")
        self.declare_parameter("pub_debug_width", True)
        self.declare_parameter("debug_width_topic", "/gripper/debug_width")

    def _load_parameters(self) -> None:
        self.wrench_topic = str(self.get_parameter("wrench_topic").value)
        self.command_topic = str(self.get_parameter("cmd_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.gripper_command_topic = str(self.get_parameter("command_topic").value)

        self.kp = float(self.get_parameter("kp").value)
        self.ki = float(self.get_parameter("ki").value)
        self.kd = float(self.get_parameter("kd").value)
        self.integral_limit = float(self.get_parameter("int_max").value)

        self.force_reference_n = float(self.get_parameter("force_ref").value)
        self.force_tolerance_n = float(self.get_parameter("force_tol").value)
        self.force_stable_cycles_required = int(
            self.get_parameter("stable_cycles").value
        )

        self.use_force_contact = bool(self.get_parameter("use_force_contact").value)
        self.min_contact_force_n = float(
            self.get_parameter("min_contact_force").value
        )
        self.contact_stable_cycles_required = int(
            self.get_parameter("contact_stable_cycles").value
        )

        self.width_open = int(self.get_parameter("width_open").value)
        self.width_closed = int(self.get_parameter("width_closed").value)
        self.current_width = int(self.get_parameter("width_init").value)

        self.max_step_per_cycle = float(
            self.get_parameter("max_step_per_cycle").value
        )
        self.min_close_step = float(self.get_parameter("min_close_step").value)
        self.closed_margin = int(self.get_parameter("closed_margin").value)
        self.min_close_delta = int(self.get_parameter("min_close_delta").value)

        self.loop_dt = float(self.get_parameter("loop_dt").value)
        self.calibration_timeout_sec = float(
            self.get_parameter("calib_timeout_sec").value
        )
        self.grasp_timeout_sec = float(
            self.get_parameter("grasp_timeout_sec").value
        )

        self.release_hold_sec = float(self.get_parameter("release_hold_sec").value)
        self.release_force_n = float(self.get_parameter("release_force").value)

        self.offset_samples_required = int(self.get_parameter("offset_samples").value)
        self.calibrate_open_before = bool(
            self.get_parameter("calib_open_before").value
        )

        self.publish_debug_force = bool(self.get_parameter("pub_debug_force").value)
        self.debug_force_topic = str(self.get_parameter("debug_force_topic").value)
        self.publish_debug_width = bool(self.get_parameter("pub_debug_width").value)
        self.debug_width_topic = str(self.get_parameter("debug_width_topic").value)

    def _initialize_state(self) -> None:
        self.raw_force_n = 0.0
        self.force_offset_n = 0.0
        self.contact_force_n = 0.0

        self.previous_error = 0.0
        self.integral_error = 0.0

        self.offset_samples = []
        self.force_stable_count = 0
        self.contact_stable_count = 0

        self.state = "IDLE"
        self.is_busy = False
        self.state_start_time = None
        self.next_state_after_calibration = "IDLE"

        self.width_at_grasp_start = self.current_width
        self.grasp_logged = False

    def _create_ros_interfaces(self) -> None:
        self.create_subscription(Wrench, self.wrench_topic, self._on_wrench, 10)
        self.create_subscription(String, self.command_topic, self._on_command, 10)

        self.command_publisher = self.create_publisher(
            RG2FTCommand,
            self.gripper_command_topic,
            10,
        )
        self.status_publisher = self.create_publisher(String, self.status_topic, 10)

        self.debug_force_publisher = (
            self.create_publisher(Float32, self.debug_force_topic, 10)
            if self.publish_debug_force
            else None
        )
        self.debug_width_publisher = (
            self.create_publisher(Float32, self.debug_width_topic, 10)
            if self.publish_debug_width
            else None
        )

    def _publish_status(self, status: str) -> None:
        self.status_publisher.publish(String(data=status))

    def _now(self):
        return self.get_clock().now()

    def _elapsed(self) -> float:
        if self.state_start_time is None:
            return 0.0
        return (self._now() - self.state_start_time).nanoseconds * 1e-9

    def _send_gripper_command(self, width: int, force_n: float) -> None:
        clamped_width = int(max(self.width_closed, min(self.width_open, width)))

        command_msg = RG2FTCommand()
        command_msg.target_width = int(clamped_width)
        command_msg.target_force = int(max(0.0, float(force_n)))
        command_msg.control = 1
        self.command_publisher.publish(command_msg)

        if self.debug_width_publisher is not None:
            self.debug_width_publisher.publish(Float32(data=float(clamped_width)))

    def _on_wrench(self, msg: Wrench) -> None:
        self.raw_force_n = abs(float(msg.force.z))
        if self.state == "CALIBRATING":
            self.offset_samples.append(self.raw_force_n)

    def _on_command(self, msg: String) -> None:
        command = normalize_command(msg.data)

        if command == "CALIBRATE":
            if not self.is_busy:
                self._start_calibration(next_state="IDLE")
            return

        if command == "GRASP":
            if self.is_busy:
                return

            if self.force_offset_n == 0.0 and len(self.offset_samples) == 0:
                self._start_calibration(next_state="GRASPING")
            else:
                self._start_grasp()
            return

        if command == "RELEASE":
            self._start_release()
            return

        if command == "STOP":
            self.is_busy = False
            self.state = "IDLE"
            self.force_stable_count = 0
            self.contact_stable_count = 0
            self.integral_error = 0.0
            self.previous_error = 0.0

            self._send_gripper_command(self.width_open, self.release_force_n)
            self._publish_status("DONE_FAIL")
            self._publish_status("IDLE")
            return

    def _start_calibration(self, next_state: str) -> None:
        self.is_busy = True
        self.state = "CALIBRATING"
        self.next_state_after_calibration = next_state
        self.state_start_time = self._now()

        self.offset_samples.clear()
        self.force_stable_count = 0
        self.contact_stable_count = 0
        self.integral_error = 0.0
        self.previous_error = 0.0

        self._publish_status("BUSY")
        self.get_logger().info("CALIBRATING...")

        if self.calibrate_open_before:
            self.current_width = self.width_open
            self._send_gripper_command(self.width_open, self.release_force_n)

    def _finish_calibration(self) -> None:
        self.force_offset_n = sum(self.offset_samples) / max(1, len(self.offset_samples))
        self.get_logger().info(f"CALIBRATED offset={self.force_offset_n:.3f} N")

        if self.next_state_after_calibration == "GRASPING":
            self._start_grasp(already_busy=True)
            return

        self.is_busy = False
        self.state = "IDLE"
        self._publish_status("DONE_OK")
        self._publish_status("IDLE")

    def _start_grasp(self, already_busy: bool = False) -> None:
        self.is_busy = True
        self.state = "GRASPING"
        self.state_start_time = self._now()

        self.force_stable_count = 0
        self.contact_stable_count = 0
        self.integral_error = 0.0
        self.previous_error = 0.0

        self.current_width = int(
            max(self.width_closed, min(self.width_open, self.current_width))
        )
        self.width_at_grasp_start = self.current_width
        self.grasp_logged = False

        if not already_busy:
            self._publish_status("BUSY")

        self.get_logger().info("GRASPING...")

    def _start_release(self) -> None:
        self.is_busy = True
        self.state = "RELEASING"
        self.state_start_time = self._now()

        self._publish_status("BUSY")
        self.get_logger().info("RELEASING...")

    def _finish_release(self) -> None:
        self.is_busy = False
        self.state = "IDLE"

        self._publish_status("DONE_OK")
        self._publish_status("IDLE")
        self.get_logger().info("RELEASED -> IDLE")

    def _finish_grasp_success(self, reason: str) -> None:
        if not self.grasp_logged:
            closed_delta = int(self.width_at_grasp_start - self.current_width)
            self.get_logger().info(
                f"GRASP SUCCESS ({reason}) "
                f"force_con={self.contact_force_n:.2f}N "
                f"width_cmd={self.current_width} "
                f"closed_delta={closed_delta}"
            )
            self.grasp_logged = True

        self.state = "HOLDING"
        self._publish_status("DONE_OK")
        self._publish_status("IDLE")
        self.is_busy = False
        self.get_logger().info(f"HOLDING ({reason})")

    def _finish_failure(self, reason: str) -> None:
        self.is_busy = False
        self.state = "IDLE"
        self._publish_status("DONE_FAIL")
        self._publish_status("IDLE")
        self.get_logger().warn(f"DONE_FAIL ({reason})")

    def _control_loop(self) -> None:
        if self.state == "IDLE":
            return

        if self.state == "CALIBRATING":
            if len(self.offset_samples) >= self.offset_samples_required:
                self._finish_calibration()
                return

            if self._elapsed() > self.calibration_timeout_sec:
                self._finish_failure("calib_timeout")
            return

        if self.state == "RELEASING":
            self.current_width = self.width_open
            self._send_gripper_command(self.width_open, self.release_force_n)

            if self._elapsed() >= self.release_hold_sec:
                self._finish_release()
            return

        if self.state == "HOLDING":
            self._send_gripper_command(self.current_width, self.force_reference_n)
            return

        if self.state == "GRASPING":
            if self._elapsed() > self.grasp_timeout_sec:
                self._finish_failure("grasp_timeout")
                return

            self.contact_force_n = self.raw_force_n - self.force_offset_n
            if self.contact_force_n < 0.0:
                self.contact_force_n = 0.0

            if self.debug_force_publisher is not None:
                self.debug_force_publisher.publish(
                    Float32(data=float(self.contact_force_n))
                )

            if self.use_force_contact:
                if self.contact_force_n >= self.min_contact_force_n:
                    self.contact_stable_count += 1
                else:
                    self.contact_stable_count = 0

                if self.contact_stable_count >= self.contact_stable_cycles_required:
                    self._finish_grasp_success("contact_stable(force)")
                    return

            error = self.force_reference_n - self.contact_force_n
            if error < 0.0:
                error = 0.0

            if abs(error) <= self.force_tolerance_n:
                self.force_stable_count += 1
            else:
                self.force_stable_count = 0

            if self.force_stable_count >= self.force_stable_cycles_required:
                self._finish_grasp_success("force_stable")
                return

            self.integral_error += error * self.loop_dt
            self.integral_error = max(
                -self.integral_limit,
                min(self.integral_limit, self.integral_error),
            )
            derivative_error = (error - self.previous_error) / self.loop_dt

            control_signal = (
                self.kp * error
                + self.ki * self.integral_error
                + self.kd * derivative_error
            )
            control_signal = max(
                -self.max_step_per_cycle,
                min(self.max_step_per_cycle, control_signal),
            )

            close_step = max(self.min_close_step, float(control_signal))

            self.current_width = int(self.current_width - close_step)
            self.current_width = int(
                max(self.width_closed, min(self.width_open, self.current_width))
            )
            self._send_gripper_command(self.current_width, self.force_reference_n)
            self.previous_error = error

            closed_enough = self.current_width <= (self.width_closed + self.closed_margin)
            closed_delta = (
                self.width_at_grasp_start - self.current_width
            ) >= self.min_close_delta

            if closed_enough and closed_delta:
                self._finish_grasp_success("contact_stable(stall_fallback)")
                return

            if closed_enough and not closed_delta:
                self._finish_failure("fully_closed_no_contact")
                return


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GripperNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()