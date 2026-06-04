#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def normalize_command(command: str) -> str:
    """Normalize text commands received from ROS topics."""
    return (command or "").strip().upper()


class GripperNode(Node):
    """Bridge ROS gripper commands to an ESP32 TCP socket."""

    def __init__(self) -> None:
        super().__init__("gripper_node")

        self._declare_parameters()
        self._load_parameters()
        self._initialize_state()
        self._create_ros_interfaces()

        self._publish_status("IDLE")
        self.get_logger().info(
            "gripper_node ready. "
            f"cmd={self.command_topic}, "
            f"status={self.status_topic}, "
            f"esp32={self.esp32_ip}:{self.esp32_port}, "
            f"grasp_cmd={self.grasp_cmd!r}, "
            f"release_cmd={self.release_cmd!r}"
        )

        if self.connect_on_start:
            self._ensure_connected()

    def _declare_parameters(self) -> None:
        self.declare_parameter("cmd_topic", "/gripper/cmd")
        self.declare_parameter("status_topic", "/gripper/status")

        self.declare_parameter("esp32_ip", "192.168.137.43")
        self.declare_parameter("esp32_port", 5000)
        self.declare_parameter("connect_on_start", False)
        self.declare_parameter("connect_timeout_sec", 3.0)
        self.declare_parameter("response_timeout_sec", 3.0)
        self.declare_parameter("recv_buffer_size", 1024)
        self.declare_parameter("wait_for_response", True)
        self.declare_parameter("append_newline", False)
        self.declare_parameter("encoding", "utf-8")

        self.declare_parameter("grasp_cmd", "a")
        self.declare_parameter("release_cmd", "b")
        self.declare_parameter("stop_cmd", "")
        self.declare_parameter("calibrate_cmd", "")
        self.declare_parameter(
            "failure_keywords",
            ["fail", "error", "timeout", "bad", "invalid"],
        )

    def _load_parameters(self) -> None:
        self.command_topic = str(self.get_parameter("cmd_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        self.esp32_ip = str(self.get_parameter("esp32_ip").value)
        self.esp32_port = int(self.get_parameter("esp32_port").value)
        self.connect_on_start = bool(self.get_parameter("connect_on_start").value)
        self.connect_timeout_sec = float(
            self.get_parameter("connect_timeout_sec").value
        )
        self.response_timeout_sec = float(
            self.get_parameter("response_timeout_sec").value
        )
        self.recv_buffer_size = int(self.get_parameter("recv_buffer_size").value)
        self.wait_for_response = bool(self.get_parameter("wait_for_response").value)
        self.append_newline = bool(self.get_parameter("append_newline").value)
        self.encoding = str(self.get_parameter("encoding").value)

        self.grasp_cmd = str(self.get_parameter("grasp_cmd").value)
        self.release_cmd = str(self.get_parameter("release_cmd").value)
        self.stop_cmd = str(self.get_parameter("stop_cmd").value)
        self.calibrate_cmd = str(self.get_parameter("calibrate_cmd").value)
        self.failure_keywords = [
            str(keyword).lower()
            for keyword in self.get_parameter("failure_keywords").value
        ]

    def _initialize_state(self) -> None:
        self.socket_connection: socket.socket | None = None
        self.is_busy = False

    def _create_ros_interfaces(self) -> None:
        self.create_subscription(String, self.command_topic, self._on_command, 10)
        self.status_publisher = self.create_publisher(String, self.status_topic, 10)

    def _publish_status(self, status: str) -> None:
        self.status_publisher.publish(String(data=status))

    def _publish_done(self, status: str) -> None:
        self._publish_status(status)
        self._publish_status("IDLE")

    def _close_socket(self) -> None:
        if self.socket_connection is None:
            return

        try:
            self.socket_connection.close()
        finally:
            self.socket_connection = None

    def _ensure_connected(self) -> bool:
        if self.socket_connection is not None:
            return True

        connection = None
        try:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.settimeout(self.connect_timeout_sec)
            connection.connect((self.esp32_ip, self.esp32_port))
            connection.settimeout(self.response_timeout_sec)
        except OSError as exc:
            if connection is not None:
                connection.close()
            self._close_socket()
            self.get_logger().warning(
                f"Could not connect to ESP32 {self.esp32_ip}:{self.esp32_port}: {exc}"
            )
            return False

        self.socket_connection = connection
        self.get_logger().info(
            f"Connected to ESP32 at {self.esp32_ip}:{self.esp32_port}"
        )
        return True

    def _encode_command(self, command: str) -> bytes:
        text = str(command)
        if self.append_newline and not text.endswith("\n"):
            text += "\n"
        return text.encode(self.encoding)

    def _response_is_success(self, response: str) -> bool:
        if not self.wait_for_response:
            return True

        if not response:
            return False

        response_lower = response.lower()
        return not any(keyword in response_lower for keyword in self.failure_keywords)

    def _send_esp32_command(self, command: str) -> tuple[bool, str, str]:
        if not command:
            return False, "", "empty ESP32 command"

        if not self._ensure_connected():
            return False, "", "ESP32 connection failed"

        payload = self._encode_command(command)
        command_was_sent = False

        try:
            self.socket_connection.sendall(payload)
            command_was_sent = True

            if not self.wait_for_response:
                return True, "", ""

            response = self.socket_connection.recv(self.recv_buffer_size)
            if not response:
                self._close_socket()
                return False, "", "ESP32 closed the connection"

            decoded_response = response.decode(self.encoding, errors="replace").strip()
        except socket.timeout:
            self._close_socket()
            if command_was_sent:
                return False, "", "timed out waiting for ESP32 response"
            return False, "", "timed out sending ESP32 command"
        except OSError as exc:
            self._close_socket()
            return False, "", str(exc)

        if self._response_is_success(decoded_response):
            return True, decoded_response, ""

        return False, decoded_response, "ESP32 response contained a failure keyword"

    def _run_action(self, action_name: str, esp32_command: str, done_status: str) -> None:
        if self.is_busy:
            self.get_logger().warning(f"Ignoring {action_name}; gripper is BUSY")
            return

        self.is_busy = True
        self._publish_status("BUSY")
        self.get_logger().info(
            f"{action_name}: sending {esp32_command!r} to ESP32"
        )

        success, response, error = self._send_esp32_command(esp32_command)

        self.is_busy = False
        if success:
            if response:
                self.get_logger().info(f"{action_name}: ESP32 replied {response!r}")
            else:
                self.get_logger().info(f"{action_name}: command sent")
            self._publish_done(done_status)
            return

        if response:
            self.get_logger().warning(
                f"{action_name}: ESP32 replied {response!r}; {error}"
            )
        else:
            self.get_logger().warning(f"{action_name}: {error}")
        self._publish_done("DONE_FAIL")

    def _on_command(self, msg: String) -> None:
        command = normalize_command(msg.data)
        raw_command = (msg.data or "").strip()

        if command == "CALIBRATE":
            if self.calibrate_cmd:
                self._run_action("CALIBRATE", self.calibrate_cmd, "DONE_OK")
            else:
                self.get_logger().info("CALIBRATE: no hardware command configured")
                self._publish_done("DONE_OK")
            return

        if command in ("GRASP", "CLOSE", "CERRAR"):
            self._run_action("GRASP", self.grasp_cmd, "DONE_OK")
            return

        if command in ("RELEASE", "OPEN", "ABRIR"):
            self._run_action("RELEASE", self.release_cmd, "DONE_OK")
            return

        if command == "STOP":
            stop_command = self.stop_cmd or self.release_cmd
            self._run_action("STOP", stop_command, "DONE_FAIL")
            return

        if command in ("A", "B"):
            self._run_action(f"RAW_{command}", raw_command, "DONE_OK")
            return

        self.get_logger().warning(f"Ignoring unknown gripper command: {raw_command!r}")

    def destroy_node(self) -> bool:
        self._close_socket()
        return super().destroy_node()


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
