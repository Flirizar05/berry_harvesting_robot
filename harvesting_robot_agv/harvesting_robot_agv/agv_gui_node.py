#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


COMMAND_TOPIC = "/agv/rpm_cmd"
CONTROL_MODE_TOPIC = "/agv/control_mode"
CONTROL_STATUS_TOPIC = "/agv/position_control_status"
CONTROL_MODE_MANUAL = "manual"
CONTROL_MODE_AUTOMATIC = "automatic"
FREE_MODE_COMMAND = "s"
MAX_WHEEL_SPEED = 150
MIN_FORWARD_SPEED = 50
MAX_SPEED_PERCENT = 100
PUBLISH_PERIOD_MS = 100
INVERT_TURN_DIRECTION = False
INDICATOR_ON_COLOR = "#2e7d32"
INDICATOR_OFF_COLOR = "#9e9e9e"


class AgvGuiNode(Node):
    """Manual Tkinter control panel for the AGV wheel command topic."""

    def __init__(self) -> None:
        super().__init__("agv_gui_node")

        self.control_mode = CONTROL_MODE_MANUAL
        self.position_control_active = False
        self.orientation_control_active = False
        self.automatic_right_rpm = 0
        self.automatic_left_rpm = 0
        self.free_stop_active = False
        self._declare_parameters()
        self._load_parameters()
        self._create_ros_interfaces()
        self._create_window()
        self._create_widgets()

        self.is_closing = False
        self._update_command_label()
        self.root.after(PUBLISH_PERIOD_MS, self._publish_command)

        self.get_logger().info(
            f"agv_gui_node ready. Publishing to {self.command_topic}"
        )
        self.get_logger().info(
            f"Publishing control mode to {self.control_mode_topic}"
        )
        self.get_logger().info(
            f"Subscribing to control status on {self.control_status_topic}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("command_topic", COMMAND_TOPIC)
        self.declare_parameter("control_mode_topic", CONTROL_MODE_TOPIC)
        self.declare_parameter("control_status_topic", CONTROL_STATUS_TOPIC)
        self.declare_parameter("invert_turn_direction", INVERT_TURN_DIRECTION)

    def _load_parameters(self) -> None:
        self.command_topic = str(self.get_parameter("command_topic").value).strip()
        if not self.command_topic:
            self.command_topic = COMMAND_TOPIC

        self.control_mode_topic = str(
            self.get_parameter("control_mode_topic").value
        ).strip()
        if not self.control_mode_topic:
            self.control_mode_topic = CONTROL_MODE_TOPIC

        self.control_status_topic = str(
            self.get_parameter("control_status_topic").value
        ).strip()
        if not self.control_status_topic:
            self.control_status_topic = CONTROL_STATUS_TOPIC

        self.invert_turn_direction = bool(
            self.get_parameter("invert_turn_direction").value
        )

    def _create_ros_interfaces(self) -> None:
        self.pub_command = self.create_publisher(String, self.command_topic, 10)
        self.pub_control_mode = self.create_publisher(
            String,
            self.control_mode_topic,
            10,
        )
        self.control_status_sub = self.create_subscription(
            String,
            self.control_status_topic,
            self._on_control_status_received,
            10,
        )

    def _create_window(self) -> None:
        self.root = tk.Tk()
        self.root.title("AGV Control")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.speed_var = tk.IntVar(value=0)
        self.turn_var = tk.IntVar(value=0)
        self.right_var = tk.IntVar(value=0)
        self.left_var = tk.IntVar(value=0)
        self.mode_var = tk.StringVar(value="Mode: MANUAL")
        self.command_var = tk.StringVar(value="Command: 0,0   right=0, left=0")

    def _create_widgets(self) -> None:
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.mode_button = tk.Button(
            main_frame,
            textvariable=self.mode_var,
            command=self._toggle_control_mode,
            bg="#2e7d32",
            fg="white",
            activebackground="#1b5e20",
            activeforeground="white",
            width=18,
            height=2,
        )
        self.mode_button.grid(row=0, column=0, columnspan=2, sticky="ew")

        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=1, column=0, columnspan=2, pady=(12, 0), sticky="ew")
        status_frame.columnconfigure(1, weight=1)
        status_frame.columnconfigure(3, weight=1)

        self.orientation_indicator = self._create_indicator(
            status_frame,
            "Orientation",
            row=0,
            column=0,
        )
        self.position_indicator = self._create_indicator(
            status_frame,
            "Position",
            row=0,
            column=2,
        )

        self.notebook = ttk.Notebook(main_frame)
        self.speed_turn_frame = ttk.Frame(self.notebook, padding=12)
        self.independent_frame = ttk.Frame(self.notebook, padding=12)

        self.notebook.add(self.speed_turn_frame, text="Speed + Turn")
        self.notebook.add(self.independent_frame, text="Independent wheels")
        self.notebook.grid(row=2, column=0, columnspan=2, pady=(12, 0), sticky="ew")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_gui_value_changed)

        self._create_labeled_slider(
            parent=self.speed_turn_frame,
            label="Speed (%)",
            variable=self.speed_var,
            from_=0,
            to=MAX_SPEED_PERCENT,
            row=0,
        )
        self._create_labeled_slider(
            parent=self.speed_turn_frame,
            label="Turn",
            variable=self.turn_var,
            from_=-100,
            to=100,
            row=1,
        )
        self._create_labeled_slider(
            parent=self.independent_frame,
            label="Right wheel",
            variable=self.right_var,
            from_=0,
            to=MAX_WHEEL_SPEED,
            row=0,
        )
        self._create_labeled_slider(
            parent=self.independent_frame,
            label="Left wheel",
            variable=self.left_var,
            from_=0,
            to=MAX_WHEEL_SPEED,
            row=1,
        )

        self.command_label = ttk.Label(
            main_frame,
            textvariable=self.command_var,
            font=("TkDefaultFont", 11, "bold"),
        )
        self.command_label.grid(row=3, column=0, columnspan=2, pady=(12, 8), sticky="w")

        self.stop_button = tk.Button(
            main_frame,
            text="STOP",
            command=self._stop_agv,
            bg="#c62828",
            fg="white",
            activebackground="#8e0000",
            activeforeground="white",
            width=18,
            height=2,
        )
        self.stop_button.grid(row=4, column=0, columnspan=2, sticky="ew")
        self._update_mode_button()
        self._update_control_indicators()

    def _create_indicator(
        self,
        parent: ttk.Frame,
        label: str,
        row: int,
        column: int,
    ) -> tk.Canvas:
        indicator = tk.Canvas(
            parent,
            width=18,
            height=18,
            highlightthickness=0,
            bg=self.root.cget("bg"),
        )
        indicator.grid(row=row, column=column, sticky="e", padx=(0, 6))
        indicator.create_oval(
            3,
            3,
            15,
            15,
            fill=INDICATOR_OFF_COLOR,
            outline=INDICATOR_OFF_COLOR,
            tags=("light",),
        )
        ttk.Label(parent, text=label).grid(
            row=row,
            column=column + 1,
            sticky="w",
            padx=(0, 18),
        )
        return indicator

    def _create_labeled_slider(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.IntVar,
        from_: int,
        to: int,
        row: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8))

        slider = tk.Scale(
            parent,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=self._on_gui_value_changed,
            length=300,
            resolution=1,
        )
        slider.grid(row=row, column=1, sticky="ew")

    def _on_gui_value_changed(self, event=None) -> None:
        self.free_stop_active = False
        self._update_command_label()

    def _normalize_wheel_value(self, value: float) -> int:
        wheel_speed = self._clamp_wheel_value(value)

        if wheel_speed == 0:
            return 0

        if wheel_speed < MIN_FORWARD_SPEED:
            return 0

        return wheel_speed

    def _clamp_wheel_value(self, value: float) -> int:
        return int(round(max(0.0, min(float(value), MAX_WHEEL_SPEED))))

    def _get_speed_from_percent(self) -> int:
        speed_percent = int(
            round(max(0.0, min(float(self.speed_var.get()), MAX_SPEED_PERCENT)))
        )

        if speed_percent == 0:
            return 0

        if speed_percent == 1:
            return MIN_FORWARD_SPEED

        speed_range = MAX_WHEEL_SPEED - MIN_FORWARD_SPEED
        return int(
            round(MIN_FORWARD_SPEED + (speed_percent * speed_range / MAX_SPEED_PERCENT))
        )

    def _get_speed_turn_commands(self) -> tuple[int, int]:
        speed = self._get_speed_from_percent()

        if speed == 0:
            return 0, 0

        turn = float(self.turn_var.get())
        if self.invert_turn_direction:
            turn *= -1.0

        turn_ratio = min(abs(turn) / 100.0, 1.0)
        turning_wheel_speed = self._clamp_wheel_value(
            speed * (1.0 - turn_ratio)
        )

        if turn > 0:
            right = turning_wheel_speed
            left = speed
        elif turn < 0:
            right = speed
            left = turning_wheel_speed
        else:
            right = speed
            left = speed

        return right, left

    def _get_independent_wheel_commands(self) -> tuple[int, int]:
        right = self._normalize_wheel_value(self.right_var.get())
        left = self._normalize_wheel_value(self.left_var.get())
        return right, left

    def _get_wheel_commands(self) -> tuple[int, int]:
        if self.notebook.select() == str(self.independent_frame):
            return self._get_independent_wheel_commands()

        return self._get_speed_turn_commands()

    def _update_command_label(self) -> None:
        if self.control_mode == CONTROL_MODE_AUTOMATIC:
            self.command_var.set(
                "Command: "
                f"{self.automatic_right_rpm},{self.automatic_left_rpm}   "
                f"right={self.automatic_right_rpm}, "
                f"left={self.automatic_left_rpm}"
            )
            return

        if self.free_stop_active:
            self.command_var.set(f"Command: {FREE_MODE_COMMAND}   free mode")
            return

        right, left = self._get_wheel_commands()
        self.command_var.set(f"Command: {right},{left}   right={right}, left={left}")

    def _publish_wheel_command(self, right: int, left: int) -> None:
        msg = String()
        msg.data = f"{right},{left}"
        self.pub_command.publish(msg)

    def _publish_free_mode_command(self) -> None:
        msg = String()
        msg.data = FREE_MODE_COMMAND
        self.pub_command.publish(msg)

    def _publish_control_mode(self) -> None:
        msg = String()
        msg.data = self.control_mode
        self.pub_control_mode.publish(msg)

    def _publish_command(self) -> None:
        if self.is_closing:
            return

        rclpy.spin_once(self, timeout_sec=0.0)
        self._publish_control_mode()
        if self.control_mode == CONTROL_MODE_MANUAL:
            if self.free_stop_active:
                self._publish_free_mode_command()
            else:
                right, left = self._get_wheel_commands()
                self._publish_wheel_command(right, left)

        self._update_command_label()
        self.root.after(PUBLISH_PERIOD_MS, self._publish_command)

    def _stop_agv(self) -> None:
        self._set_control_mode(CONTROL_MODE_MANUAL)
        self.speed_var.set(0)
        self.turn_var.set(0)
        self.right_var.set(0)
        self.left_var.set(0)
        self.free_stop_active = True
        self._publish_free_mode_command()
        self._update_command_label()

    def _toggle_control_mode(self) -> None:
        if self.control_mode == CONTROL_MODE_MANUAL:
            self._set_control_mode(CONTROL_MODE_AUTOMATIC)
        else:
            self._set_control_mode(CONTROL_MODE_MANUAL)

    def _set_control_mode(self, control_mode: str) -> None:
        if control_mode not in (CONTROL_MODE_MANUAL, CONTROL_MODE_AUTOMATIC):
            return

        self.control_mode = control_mode
        self.free_stop_active = False
        self._publish_control_mode()
        if control_mode == CONTROL_MODE_AUTOMATIC:
            self._publish_wheel_command(0, 0)
        else:
            self._clear_automatic_status()

        self._update_mode_button()
        self._update_control_indicators()
        self._update_command_label()

    def _update_mode_button(self) -> None:
        if self.control_mode == CONTROL_MODE_AUTOMATIC:
            self.mode_var.set("Mode: AUTOMATIC")
            self.mode_button.configure(
                bg="#1565c0",
                activebackground="#0d47a1",
            )
            return

        self.mode_var.set("Mode: MANUAL")
        self.mode_button.configure(
            bg="#2e7d32",
            activebackground="#1b5e20",
        )

    def _on_control_status_received(self, msg: String) -> None:
        try:
            values = self._parse_key_value_message(msg.data)
            self.orientation_control_active = self._parse_bool(
                values.get("orientation_active")
            )
            self.position_control_active = self._parse_bool(
                values.get("position_active")
            )
            self.automatic_right_rpm = int(values.get("right_rpm", "0"))
            self.automatic_left_rpm = int(values.get("left_rpm", "0"))

        except ValueError as error:
            self.get_logger().warn(
                f"Ignoring invalid control status message: {error}"
            )
            return

        self._update_control_indicators()
        self._update_command_label()

    def _clear_automatic_status(self) -> None:
        self.position_control_active = False
        self.orientation_control_active = False
        self.automatic_right_rpm = 0
        self.automatic_left_rpm = 0

    def _update_control_indicators(self) -> None:
        self._set_indicator(
            self.orientation_indicator,
            self.orientation_control_active,
        )
        self._set_indicator(
            self.position_indicator,
            self.position_control_active,
        )

    @staticmethod
    def _set_indicator(indicator: tk.Canvas, is_active: bool) -> None:
        color = INDICATOR_ON_COLOR if is_active else INDICATOR_OFF_COLOR
        indicator.itemconfigure("light", fill=color, outline=color)

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
    def _parse_bool(value: str | None) -> bool:
        if value is None:
            return False

        normalized_value = value.strip().lower()
        if normalized_value in ("true", "1", "yes", "on"):
            return True

        if normalized_value in ("false", "0", "no", "off"):
            return False

        raise ValueError(f"invalid boolean value '{value}'")

    def _on_close(self) -> None:
        if self.is_closing:
            return

        self.is_closing = True
        self._set_control_mode(CONTROL_MODE_MANUAL)
        self._publish_free_mode_command()
        self.destroy_node()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)

    node = None

    try:
        node = AgvGuiNode()
        node.run()

    except KeyboardInterrupt:
        if node is not None and not node.is_closing:
            node._on_close()

    finally:
        if node is not None and not node.is_closing:
            node._on_close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
