#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import re
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any

import rclpy
from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from rclpy.node import Node
from std_msgs.msg import String


DEFAULT_DEVICE_NAME = "ESP32S3_BLE"
DEFAULT_DEVICE_ADDRESS = "48:27:E2:16:D6:61"
NUS_WRITE_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
NUS_READ_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
RPM_COMMAND_PATTERN = re.compile(r"^\s*\d+(\.\d+)?\s*,\s*\d+(\.\d+)?\s*$")
FREE_MODE_COMMAND = "s"
MIN_WHEEL_SPEED = 0.0
MAX_WHEEL_SPEED = 150.0


class BleAgvBridgeNode(Node):
    """Bridge ROS 2 AGV RPM commands to an ESP32-S3 BLE UART service."""

    def __init__(self) -> None:
        super().__init__("ble_agv_bridge_node")

        self.client: BleakClient | None = None
        self.ble_loop = asyncio.new_event_loop()
        self.is_connecting = False
        self.ble_ready = False
        self.last_sent_command: str | None = None
        self.is_shutting_down = False

        self._declare_parameters()
        self._load_parameters()
        self._create_ros_interfaces()
        self._start_ble_loop()

        self.reconnect_timer = self.create_timer(
            self.reconnect_period_sec,
            self._on_reconnect_timer,
        )

        self.get_logger().info("ble_agv_bridge_node ready")
        self.get_logger().info(f"Command topic: {self.command_topic}")
        self.get_logger().info(f"Feedback topic: {self.feedback_topic}")
        self.get_logger().info(f"Log feedback to terminal: {self.log_feedback}")
        self.get_logger().info(
            f"Suppress duplicate commands: {self.suppress_duplicate_commands}"
        )
        self.get_logger().info(f"BLE target address: {self.device_address}")
        self.get_logger().info(f"BLE target name fallback: {self.device_name}")

    def _declare_parameters(self) -> None:
        self.declare_parameter("device_address", DEFAULT_DEVICE_ADDRESS)
        self.declare_parameter("device_name", DEFAULT_DEVICE_NAME)
        self.declare_parameter("write_uuid", NUS_WRITE_UUID)
        self.declare_parameter("read_uuid", NUS_READ_UUID)
        self.declare_parameter("command_topic", "/agv/rpm_cmd")
        self.declare_parameter("feedback_topic", "/agv/ble_feedback")
        self.declare_parameter("log_feedback", True)
        self.declare_parameter("suppress_duplicate_commands", True)
        self.declare_parameter("scan_timeout_sec", 10.0)
        self.declare_parameter("reconnect_period_sec", 3.0)

    def _load_parameters(self) -> None:
        self.device_address = str(self.get_parameter("device_address").value).strip()
        self.device_name = str(self.get_parameter("device_name").value).strip()
        self.write_uuid = str(self.get_parameter("write_uuid").value).strip()
        self.read_uuid = str(self.get_parameter("read_uuid").value).strip()
        self.command_topic = str(self.get_parameter("command_topic").value).strip()
        self.feedback_topic = str(self.get_parameter("feedback_topic").value).strip()
        self.log_feedback = bool(self.get_parameter("log_feedback").value)
        self.suppress_duplicate_commands = bool(
            self.get_parameter("suppress_duplicate_commands").value
        )
        self.scan_timeout_sec = float(self.get_parameter("scan_timeout_sec").value)
        self.reconnect_period_sec = float(
            self.get_parameter("reconnect_period_sec").value
        )

    def _create_ros_interfaces(self) -> None:
        self.command_sub = self.create_subscription(
            String,
            self.command_topic,
            self._on_command_received,
            10,
        )
        self.feedback_pub = self.create_publisher(String, self.feedback_topic, 10)

    def _start_ble_loop(self) -> None:
        self.ble_thread = threading.Thread(
            target=self._run_ble_loop,
            name="ble_agv_bridge_loop",
            daemon=True,
        )
        self.ble_thread.start()

    def _run_ble_loop(self) -> None:
        asyncio.set_event_loop(self.ble_loop)
        self.ble_loop.run_forever()

    def _on_reconnect_timer(self) -> None:
        if self._is_ble_connected() or self.is_connecting:
            return

        self.is_connecting = True
        self._schedule_ble_task(self._connect_ble())

    def _is_ble_connected(self) -> bool:
        return (
            self.client is not None
            and self.client.is_connected
            and self.ble_ready
        )

    def _schedule_ble_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
    ) -> Future[Any]:
        return asyncio.run_coroutine_threadsafe(coroutine, self.ble_loop)

    async def _connect_ble(self) -> None:
        try:
            device = await self._find_ble_device()

            if device is None:
                self.get_logger().warn("BLE device not found")
                return

            self.client = BleakClient(
                device,
                disconnected_callback=self._on_ble_disconnected,
            )
            self.ble_ready = False

            await self.client.connect()

            if not self.client.is_connected:
                self.get_logger().error("Failed to connect to BLE device")
                return

            await self.client.start_notify(self.read_uuid, self._on_ble_feedback)
            self.ble_ready = True
            self.last_sent_command = None

            self.get_logger().info("Connected to ESP32-S3 over BLE")
            self.get_logger().info("BLE notifications enabled")

        except Exception as error:
            self.get_logger().error(f"BLE connection error: {error}")

        finally:
            self.is_connecting = False

    async def _find_ble_device(self) -> BLEDevice | None:
        if self.device_address:
            self.get_logger().info(
                f"Searching for BLE device by address: {self.device_address}"
            )
            return await BleakScanner.find_device_by_address(
                self.device_address,
                timeout=self.scan_timeout_sec,
            )

        self.get_logger().info(f"Searching for BLE device by name: {self.device_name}")
        return await BleakScanner.find_device_by_name(
            self.device_name,
            timeout=self.scan_timeout_sec,
        )

    def _on_ble_disconnected(self, client: BleakClient) -> None:
        self.client = None
        self.ble_ready = False
        self.last_sent_command = None
        self.get_logger().warn("BLE device disconnected")

    def _on_command_received(self, msg: String) -> None:
        raw_command = msg.data.strip()
        command = (
            FREE_MODE_COMMAND
            if raw_command.lower() == FREE_MODE_COMMAND
            else raw_command
        )

        if not self._is_valid_rpm_command(command):
            self.get_logger().warn(
                f"Ignoring invalid AGV command: '{raw_command}'. "
                "Expected 's' or rpm_right,rpm_left with each value "
                "between 0 and 150"
            )
            return

        if not self._is_ble_connected():
            self.get_logger().warn("BLE is not connected; command ignored")
            return

        if self._is_duplicate_command(command):
            self.get_logger().debug(f"Duplicate AGV command suppressed: {command}")
            return

        self.last_sent_command = command
        self._schedule_ble_task(self._send_command(command))

    async def _send_command(self, command: str) -> None:
        if self.client is None:
            return

        try:
            await self.client.write_gatt_char(
                self.write_uuid,
                command.encode("utf-8"),
                response=True,
            )
            self.get_logger().info(f"Sent AGV command: {command}")

        except Exception as error:
            if self.last_sent_command == command:
                self.last_sent_command = None
            self.get_logger().error(f"BLE write error: {error}")

    def _is_duplicate_command(self, command: str) -> bool:
        if command == FREE_MODE_COMMAND:
            return False

        return (
            self.suppress_duplicate_commands
            and self.last_sent_command == command
        )

    def _on_ble_feedback(self, sender: int, data: bytearray) -> None:
        if self.is_shutting_down:
            return

        feedback = data.decode("utf-8", errors="ignore").strip()

        if not feedback:
            return

        try:
            self.feedback_pub.publish(String(data=feedback))
        except Exception as error:
            if not self.is_shutting_down:
                self.get_logger().warn(f"BLE feedback publish warning: {error}")
            return

        if self.log_feedback:
            self.get_logger().info(f"AGV feedback: {feedback}")

    @staticmethod
    def _is_valid_rpm_command(command: str) -> bool:
        if command == FREE_MODE_COMMAND:
            return True

        if RPM_COMMAND_PATTERN.match(command) is None:
            return False

        try:
            right_rpm, left_rpm = (
                float(value.strip()) for value in command.split(",", maxsplit=1)
            )
        except ValueError:
            return False

        return (
            MIN_WHEEL_SPEED <= right_rpm <= MAX_WHEEL_SPEED
            and MIN_WHEEL_SPEED <= left_rpm <= MAX_WHEEL_SPEED
        )

    async def _shutdown_ble(self) -> None:
        if self.client is not None and self.client.is_connected:
            try:
                await self.client.stop_notify(self.read_uuid)
            except Exception:
                pass

            await self.client.disconnect()

        self.client = None

    def destroy_node(self) -> None:
        self.is_shutting_down = True

        if self.ble_loop.is_running():
            future = self._schedule_ble_task(self._shutdown_ble())
            try:
                future.result(timeout=2.0)
            except Exception as error:
                self.get_logger().warn(f"BLE shutdown warning: {error}")

            self.ble_loop.call_soon_threadsafe(self.ble_loop.stop)

        if self.ble_thread.is_alive():
            self.ble_thread.join(timeout=2.0)

        self.ble_loop.close()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)

    node = BleAgvBridgeNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
