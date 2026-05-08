#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


COMMAND_TOPIC = "/agv/rpm_cmd"
MAX_WHEEL_SPEED = 150
MIN_FORWARD_SPEED = 50
MAX_SPEED_PERCENT = 100
PUBLISH_PERIOD_MS = 100
INVERT_TURN_DIRECTION = False


class AgvGuiNode(Node):
    """Manual Tkinter control panel for the AGV wheel command topic."""

    def __init__(self) -> None:
        super().__init__("agv_gui_node")

        self._declare_parameters()
        self._load_parameters()
        self._create_ros_interfaces()
        self._create_window()
        self._create_widgets()

        self.is_closing = False
        self._update_command_label()
        self.root.after(PUBLISH_PERIOD_MS, self._publish_command)

        self.get_logger().info(f"agv_gui_node ready. Publishing to {self.command_topic}")

    def _declare_parameters(self) -> None:
        self.declare_parameter("command_topic", COMMAND_TOPIC)
        self.declare_parameter("invert_turn_direction", INVERT_TURN_DIRECTION)

    def _load_parameters(self) -> None:
        self.command_topic = str(self.get_parameter("command_topic").value).strip()
        self.invert_turn_direction = bool(
            self.get_parameter("invert_turn_direction").value
        )

    def _create_ros_interfaces(self) -> None:
        self.pub_command = self.create_publisher(String, self.command_topic, 10)

    def _create_window(self) -> None:
        self.root = tk.Tk()
        self.root.title("AGV Control")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.speed_var = tk.IntVar(value=0)
        self.turn_var = tk.IntVar(value=0)
        self.right_var = tk.IntVar(value=0)
        self.left_var = tk.IntVar(value=0)
        self.command_var = tk.StringVar(value="Command: 0,0   right=0, left=0")

    def _create_widgets(self) -> None:
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.notebook = ttk.Notebook(main_frame)
        self.speed_turn_frame = ttk.Frame(self.notebook, padding=12)
        self.independent_frame = ttk.Frame(self.notebook, padding=12)

        self.notebook.add(self.speed_turn_frame, text="Speed + Turn")
        self.notebook.add(self.independent_frame, text="Independent wheels")
        self.notebook.grid(row=0, column=0, columnspan=2, sticky="ew")
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
        self.command_label.grid(row=1, column=0, columnspan=2, pady=(12, 8), sticky="w")

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
        self.stop_button.grid(row=2, column=0, columnspan=2, sticky="ew")

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
        self._update_command_label()

    def _normalize_wheel_value(
        self,
        value: float,
        use_minimum_forward_speed: bool = False,
    ) -> int:
        wheel_speed = int(round(max(0.0, min(float(value), MAX_WHEEL_SPEED))))

        if wheel_speed == 0:
            return 0

        if wheel_speed < MIN_FORWARD_SPEED:
            if use_minimum_forward_speed:
                return MIN_FORWARD_SPEED
            return 0

        return wheel_speed

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
        reduced_speed = speed * (1.0 - turn_ratio)

        if turn_ratio >= 1.0:
            reduced_wheel_speed = 0
        else:
            reduced_wheel_speed = self._normalize_wheel_value(
                reduced_speed,
                use_minimum_forward_speed=True,
            )

        if turn > 0:
            right = reduced_wheel_speed
            left = speed
        elif turn < 0:
            right = speed
            left = reduced_wheel_speed
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
        right, left = self._get_wheel_commands()
        self.command_var.set(f"Command: {right},{left}   right={right}, left={left}")

    def _publish_wheel_command(self, right: int, left: int) -> None:
        msg = String()
        msg.data = f"{right},{left}"
        self.pub_command.publish(msg)

    def _publish_command(self) -> None:
        if self.is_closing:
            return

        right, left = self._get_wheel_commands()
        self._publish_wheel_command(right, left)
        self._update_command_label()
        self.root.after(PUBLISH_PERIOD_MS, self._publish_command)

    def _stop_agv(self) -> None:
        self.speed_var.set(0)
        self.turn_var.set(0)
        self.right_var.set(0)
        self.left_var.set(0)
        self._publish_wheel_command(0, 0)
        self._update_command_label()

    def _on_close(self) -> None:
        if self.is_closing:
            return

        self.is_closing = True
        self._publish_wheel_command(0, 0)
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
