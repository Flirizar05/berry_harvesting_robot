#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class MasterNode(Node):
    """Coordinate the high-level execution flow for Mode 1 and Mode 2."""

    def __init__(self) -> None:
        super().__init__("master_node")

        self._declare_parameters()
        self._load_parameters()
        self._create_publishers()
        self._create_subscribers()
        self._initialize_state()

        self.pub_status.publish(String(data="IDLE"))
        self.get_logger().info("master_node ready. Waiting for /master/cmd START.")

        self.timer = self.create_timer(self.loop_dt, self._control_loop)

    def _declare_parameters(self) -> None:
        self.declare_parameter("loop_dt", 0.02)

        self.declare_parameter("stop_distance_m", 0.25)
        self.declare_parameter("stop_margin_m", 0.01)
        self.declare_parameter("dist_fresh_timeout_sec", 2.0)

        self.declare_parameter("vision_timeout_sec", 6.0)
        self.declare_parameter("eye_timeout_sec", 6.0)
        self.declare_parameter("plan_timeout_sec", 6.0)
        self.declare_parameter("execute_timeout_sec", 180.0)

        self.declare_parameter("do_home_on_start", True)
        self.declare_parameter("home_timeout_sec", 20.0)
        self.declare_parameter("home_pos_tol_rad", 0.05)
        self.declare_parameter("home_settle_cycles", 10)
        self.declare_parameter("home_horizon_sec", 2.0)

        self.declare_parameter(
            "home_joint_names",
            [
                "elfin_joint1",
                "elfin_joint2",
                "elfin_joint3",
                "elfin_joint4",
                "elfin_joint5",
                "elfin_joint6",
            ],
        )
        self.declare_parameter(
            "home_joint_positions_rad",
            [-1.67, -0.52, -1.39, -0.15, 2.39, 2.77],
        )

        self.declare_parameter("master_cmd", "/master/cmd")
        self.declare_parameter("master_status", "/master/status")

        self.declare_parameter("vision_cmd", "/vision/cmd")
        self.declare_parameter("vision_status", "/vision/status")
        self.declare_parameter("eye_cmd", "/eyeinhand/cmd")
        self.declare_parameter("eye_status", "/eyeinhand/status")
        self.declare_parameter("traj_cmd", "/trajectory/cmd")
        self.declare_parameter("traj_status", "/trajectory/status")
        self.declare_parameter("ctrl_cmd", "/control/cmd")
        self.declare_parameter("ctrl_status", "/control/status")

        self.declare_parameter("tcp_target_dist_topic", "/trajectory/tcp_target_dist")
        self.declare_parameter("controller_topic", "/elfin_arm_controller/joint_trajectory")
        self.declare_parameter("joint_state_topic", "/joint_states")

        self.declare_parameter("enable_mode2", True)
        self.declare_parameter("enable_gripper", True)

        self.declare_parameter("pf_cmd", "/potentialfields/cmd")
        self.declare_parameter("pf_status", "/potentialfields/status")
        self.declare_parameter("pf_timeout_sec", 6.0)

        self.declare_parameter("hyrrt_cmd", "/hyrrt/cmd")
        self.declare_parameter("hyrrt_status", "/hyrrt/status")
        self.declare_parameter("hyrrt_plan_timeout_sec", 60.0)
        self.declare_parameter("hyrrt_exec_timeout_sec", 180.0)

        self.declare_parameter("gripper_cmd", "/gripper/cmd")
        self.declare_parameter("gripper_status", "/gripper/status")
        self.declare_parameter("gripper_timeout_sec", 15.0)

        self.declare_parameter("cmd_capture", "CAPTURE")
        self.declare_parameter("cmd_compute", "COMPUTE")
        self.declare_parameter("cmd_plan", "PLAN")
        self.declare_parameter("cmd_execute", "EXECUTE")
        self.declare_parameter("cmd_execute_stream", "EXECUTE_STREAM")
        self.declare_parameter("cmd_grasp", "GRASP")
        self.declare_parameter("cmd_release", "RELEASE")

    def _load_parameters(self) -> None:
        self.loop_dt = float(self.get_parameter("loop_dt").value)

        self.stop_distance_m = float(self.get_parameter("stop_distance_m").value)
        self.stop_margin_m = float(self.get_parameter("stop_margin_m").value)
        self.distance_fresh_timeout_sec = float(
            self.get_parameter("dist_fresh_timeout_sec").value
        )

        self.vision_timeout_sec = float(self.get_parameter("vision_timeout_sec").value)
        self.eye_timeout_sec = float(self.get_parameter("eye_timeout_sec").value)
        self.plan_timeout_sec = float(self.get_parameter("plan_timeout_sec").value)
        self.execute_timeout_sec = float(self.get_parameter("execute_timeout_sec").value)

        self.do_home_on_start = bool(self.get_parameter("do_home_on_start").value)
        self.home_timeout_sec = float(self.get_parameter("home_timeout_sec").value)
        self.home_pos_tol_rad = float(self.get_parameter("home_pos_tol_rad").value)
        self.home_settle_cycles = int(self.get_parameter("home_settle_cycles").value)
        self.home_horizon_sec = float(self.get_parameter("home_horizon_sec").value)

        self.home_joint_names = list(self.get_parameter("home_joint_names").value)
        self.home_joint_positions = np.array(
            self.get_parameter("home_joint_positions_rad").value,
            dtype=float,
        )

        self.master_cmd_topic = str(self.get_parameter("master_cmd").value)
        self.master_status_topic = str(self.get_parameter("master_status").value)

        self.vision_cmd_topic = str(self.get_parameter("vision_cmd").value)
        self.vision_status_topic = str(self.get_parameter("vision_status").value)
        self.eye_cmd_topic = str(self.get_parameter("eye_cmd").value)
        self.eye_status_topic = str(self.get_parameter("eye_status").value)
        self.traj_cmd_topic = str(self.get_parameter("traj_cmd").value)
        self.traj_status_topic = str(self.get_parameter("traj_status").value)
        self.ctrl_cmd_topic = str(self.get_parameter("ctrl_cmd").value)
        self.ctrl_status_topic = str(self.get_parameter("ctrl_status").value)

        self.tcp_target_dist_topic = str(
            self.get_parameter("tcp_target_dist_topic").value
        )
        self.controller_topic = str(self.get_parameter("controller_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)

        self.enable_mode2 = bool(self.get_parameter("enable_mode2").value)
        self.enable_gripper = bool(self.get_parameter("enable_gripper").value)

        self.pf_cmd_topic = str(self.get_parameter("pf_cmd").value)
        self.pf_status_topic = str(self.get_parameter("pf_status").value)
        self.pf_timeout_sec = float(self.get_parameter("pf_timeout_sec").value)

        self.hyrrt_cmd_topic = str(self.get_parameter("hyrrt_cmd").value)
        self.hyrrt_status_topic = str(self.get_parameter("hyrrt_status").value)
        self.hyrrt_plan_timeout_sec = float(
            self.get_parameter("hyrrt_plan_timeout_sec").value
        )
        self.hyrrt_exec_timeout_sec = float(
            self.get_parameter("hyrrt_exec_timeout_sec").value
        )

        self.gripper_cmd_topic = str(self.get_parameter("gripper_cmd").value)
        self.gripper_status_topic = str(self.get_parameter("gripper_status").value)
        self.gripper_timeout_sec = float(
            self.get_parameter("gripper_timeout_sec").value
        )

        self.cmd_capture = str(self.get_parameter("cmd_capture").value)
        self.cmd_compute = str(self.get_parameter("cmd_compute").value)
        self.cmd_plan = str(self.get_parameter("cmd_plan").value)
        self.cmd_execute = str(self.get_parameter("cmd_execute").value)
        self.cmd_execute_stream = str(
            self.get_parameter("cmd_execute_stream").value
        )
        self.cmd_grasp = str(self.get_parameter("cmd_grasp").value)
        self.cmd_release = str(self.get_parameter("cmd_release").value)

    def _create_publishers(self) -> None:
        self.pub_status = self.create_publisher(String, self.master_status_topic, 10)

        self.pub_vision_cmd = self.create_publisher(String, self.vision_cmd_topic, 10)
        self.pub_eye_cmd = self.create_publisher(String, self.eye_cmd_topic, 10)
        self.pub_traj_cmd = self.create_publisher(String, self.traj_cmd_topic, 10)
        self.pub_ctrl_cmd = self.create_publisher(String, self.ctrl_cmd_topic, 10)
        self.pub_home = self.create_publisher(JointTrajectory, self.controller_topic, 10)

        self.pub_pf_cmd = self.create_publisher(String, self.pf_cmd_topic, 10)
        self.pub_hyrrt_cmd = self.create_publisher(String, self.hyrrt_cmd_topic, 10)
        self.pub_gripper_cmd = self.create_publisher(String, self.gripper_cmd_topic, 10)

    def _create_subscribers(self) -> None:
        self.create_subscription(String, self.master_cmd_topic, self._on_master_cmd, 10)

        self.create_subscription(String, self.vision_status_topic, self._on_vision_status, 10)
        self.create_subscription(String, self.eye_status_topic, self._on_eye_status, 10)
        self.create_subscription(String, self.traj_status_topic, self._on_traj_status, 10)
        self.create_subscription(String, self.ctrl_status_topic, self._on_ctrl_status, 10)

        self.create_subscription(Float32, self.tcp_target_dist_topic, self._on_distance, 10)
        self.create_subscription(JointState, self.joint_state_topic, self._on_joint_state, 50)

        self.create_subscription(String, self.pf_status_topic, self._on_pf_status, 10)
        self.create_subscription(String, self.hyrrt_status_topic, self._on_hyrrt_status, 10)
        self.create_subscription(String, self.gripper_status_topic, self._on_gripper_status, 10)

    def _initialize_state(self) -> None:
        self.busy = False
        self.state = "IDLE"
        self.phase_start_time = None

        self.vision_result = None
        self.eye_result = None
        self.traj_result = None
        self.ctrl_result = None

        self.pf_result = None
        self.hyrrt_result = None
        self.gripper_result = None

        self.latest_tcp_target_distance = None
        self.latest_distance_time = None

        self.current_joint_positions = None
        self.home_command_sent = False
        self.home_settle_count = 0

        self.mode2_hyrrt_finished = False
        self.mode2_ctrl_finished = False

    def _now(self):
        return self.get_clock().now()

    def _set_phase(self, phase_name: str) -> None:
        self.state = phase_name
        self.phase_start_time = self._now()

    def _elapsed_phase_time(self) -> float:
        if self.phase_start_time is None:
            return 0.0
        return (self._now() - self.phase_start_time).nanoseconds * 1e-9

    @staticmethod
    def _publish_string_command(publisher, command: str) -> None:
        publisher.publish(String(data=command))

    def _distance_is_fresh(self) -> bool:
        if self.latest_distance_time is None:
            return False
        age_sec = (self._now() - self.latest_distance_time).nanoseconds * 1e-9
        return age_sec <= self.distance_fresh_timeout_sec

    def _send_home_trajectory(self) -> None:
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.home_joint_names

        point = JointTrajectoryPoint()
        point.positions = self.home_joint_positions.tolist()
        point.time_from_start = Duration(
            sec=int(self.home_horizon_sec),
            nanosec=int((self.home_horizon_sec - int(self.home_horizon_sec)) * 1e9),
        )
        trajectory_msg.points = [point]
        self.pub_home.publish(trajectory_msg)

    def _start(self) -> None:
        self.busy = True

        self.vision_result = None
        self.eye_result = None
        self.traj_result = None
        self.ctrl_result = None

        self.pf_result = None
        self.hyrrt_result = None
        self.gripper_result = None

        self.latest_tcp_target_distance = None
        self.latest_distance_time = None

        self.home_command_sent = False
        self.home_settle_count = 0

        self.mode2_hyrrt_finished = False
        self.mode2_ctrl_finished = False

        self.pub_status.publish(String(data="BUSY"))

        if self.do_home_on_start:
            self._set_phase("HOME")
            self.get_logger().info("MASTER START -> HOME")
        else:
            self._set_phase("VISION")
            self.vision_result = None
            self._publish_string_command(self.pub_vision_cmd, self.cmd_capture)
            self.get_logger().info("MASTER START -> VISION: CAPTURE")

    def _finish_ok(self, reason: str) -> None:
        self.busy = False
        self.pub_status.publish(String(data="DONE_OK"))
        self.pub_status.publish(String(data="IDLE"))
        self.get_logger().info(f"MASTER DONE_OK: {reason}")

    def _finish_fail(self, reason: str) -> None:
        self.busy = False
        self.pub_status.publish(String(data="DONE_FAIL"))
        self.pub_status.publish(String(data="IDLE"))
        self.get_logger().warn(f"MASTER DONE_FAIL: {reason}")

    def _abort(self, reason: str) -> None:
        self._publish_string_command(self.pub_traj_cmd, "STOP")
        self._publish_string_command(self.pub_ctrl_cmd, "STOP")
        self._publish_string_command(self.pub_hyrrt_cmd, "STOP")
        self._publish_string_command(self.pub_gripper_cmd, "STOP")
        self._finish_fail(f"ABORT: {reason}")

    def _on_master_cmd(self, msg: String) -> None:
        command = msg.data.strip().upper()

        if command in ("START", "RUN"):
            if not self.busy:
                self._start()
            return

        if command in ("STOP", "ABORT"):
            self._abort("ABORT requested")
            return

        if command in ("RELEASE", "OPEN"):
            self._publish_string_command(self.pub_gripper_cmd, self.cmd_release)

    def _on_vision_status(self, msg: String) -> None:
        status = msg.data.strip().upper()
        if status.startswith("DONE_"):
            self.vision_result = status

    def _on_eye_status(self, msg: String) -> None:
        status = msg.data.strip().upper()
        if status.startswith("DONE_"):
            self.eye_result = status

    def _on_traj_status(self, msg: String) -> None:
        status = msg.data.strip().upper()
        if status.startswith("DONE_"):
            self.traj_result = status

    def _on_ctrl_status(self, msg: String) -> None:
        status = msg.data.strip().upper()
        if status.startswith("DONE_"):
            self.ctrl_result = status

    def _on_pf_status(self, msg: String) -> None:
        status = msg.data.strip().upper()
        if status.startswith("DONE_"):
            self.pf_result = status

    def _on_hyrrt_status(self, msg: String) -> None:
        status = msg.data.strip().upper()
        if status.startswith("DONE_"):
            self.hyrrt_result = status
            if self.state == "HYRRT_STREAM" and status == "DONE_OK":
                self.mode2_hyrrt_finished = True

    def _on_gripper_status(self, msg: String) -> None:
        status = msg.data.strip().upper()
        if status.startswith("DONE_"):
            self.gripper_result = status

    def _on_distance(self, msg: Float32) -> None:
        self.latest_tcp_target_distance = float(msg.data)
        self.latest_distance_time = self._now()

    def _on_joint_state(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return

        joint_map = {name: position for name, position in zip(msg.name, msg.position)}
        if not all(joint_name in joint_map for joint_name in self.home_joint_names):
            return

        self.current_joint_positions = np.array(
            [joint_map[joint_name] for joint_name in self.home_joint_names],
            dtype=float,
        )

    def _control_loop(self) -> None:
        if not self.busy:
            return

        if self.state == "HOME":
            if not self.home_command_sent:
                self._send_home_trajectory()
                self.home_command_sent = True
                self.get_logger().info("HOME command sent.")
                return

            if self.current_joint_positions is None:
                if self._elapsed_phase_time() > self.home_timeout_sec:
                    self._finish_fail("HOME timeout (no joint_states)")
                return

            home_error = float(np.linalg.norm(self.home_joint_positions - self.current_joint_positions))
            if home_error <= self.home_pos_tol_rad:
                self.home_settle_count += 1
            else:
                self.home_settle_count = 0

            if self.home_settle_count >= self.home_settle_cycles:
                self._set_phase("VISION")
                self.vision_result = None
                self._publish_string_command(self.pub_vision_cmd, self.cmd_capture)
                self.get_logger().info("HOME reached -> VISION: CAPTURE")
                return

            if self._elapsed_phase_time() > self.home_timeout_sec:
                self._finish_fail(f"HOME timeout err={home_error:.3f} rad")
            return

        if self.state == "VISION":
            if self.vision_result == "DONE_OK":
                self.vision_result = None
                self._set_phase("EYE")
                self._publish_string_command(self.pub_eye_cmd, self.cmd_compute)
                return

            if self.vision_result == "DONE_FAIL":
                self._finish_fail("Vision failed")
                return

            if self._elapsed_phase_time() > self.vision_timeout_sec:
                self._finish_fail("Vision timeout")
            return

        if self.state == "EYE":
            if self.eye_result == "DONE_OK":
                self.eye_result = None
                self._set_phase("CHECK")
                return

            if self.eye_result == "DONE_FAIL":
                self._finish_fail("Eye-in-hand failed")
                return

            if self._elapsed_phase_time() > self.eye_timeout_sec:
                self._finish_fail("Eye-in-hand timeout")
            return

        if self.state == "CHECK":
            if self.latest_tcp_target_distance is None or not self._distance_is_fresh():
                if self._elapsed_phase_time() > self.distance_fresh_timeout_sec:
                    self._finish_fail("No fresh /trajectory/tcp_target_dist")
                return

            stop_threshold = self.stop_distance_m + self.stop_margin_m
            if float(self.latest_tcp_target_distance) <= stop_threshold:
                if self.enable_mode2:
                    self._publish_string_command(self.pub_traj_cmd, "STOP")
                    self._publish_string_command(self.pub_ctrl_cmd, "STOP")

                    self.get_logger().info(
                        f"MODE1 complete "
                        f"(dist={self.latest_tcp_target_distance:.3f} <= {stop_threshold:.3f}) -> MODE2"
                    )
                    self._set_phase("PF_VISION")
                    self.pf_result = None
                    self._publish_string_command(self.pub_pf_cmd, self.cmd_capture)
                    return

                self._finish_ok(f"Target within {stop_threshold:.3f} m")
                return

            self._set_phase("PLAN")
            self.traj_result = None
            self._publish_string_command(self.pub_traj_cmd, self.cmd_plan)
            return

        if self.state == "PLAN":
            if self.traj_result == "DONE_FAIL":
                self._finish_fail("Trajectory planning failed")
                return

            if self._elapsed_phase_time() > 0.10:
                self._set_phase("EXECUTE")
                self.ctrl_result = None
                self._publish_string_command(self.pub_ctrl_cmd, self.cmd_execute)
                return

            if self._elapsed_phase_time() > self.plan_timeout_sec:
                self._finish_fail("Plan timeout")
            return

        if self.state == "EXECUTE":
            if self.ctrl_result == "DONE_OK":
                self.ctrl_result = None
                self._set_phase("VISION")
                self.vision_result = None
                self._publish_string_command(self.pub_vision_cmd, self.cmd_capture)
                return

            if self.ctrl_result == "DONE_FAIL":
                self._finish_fail("Control execute failed")
                return

            if self._elapsed_phase_time() > self.execute_timeout_sec:
                self._finish_fail("Execute timeout")
            return

        if self.state == "PF_VISION":
            if self.pf_result == "DONE_OK":
                self.pf_result = None
                self._set_phase("EYE2")
                self.eye_result = None
                self._publish_string_command(self.pub_eye_cmd, self.cmd_compute)
                return

            if self.pf_result == "DONE_FAIL":
                self._finish_fail("Potential fields capture failed")
                return

            if self._elapsed_phase_time() > self.pf_timeout_sec:
                self._finish_fail("Potential fields timeout")
            return

        if self.state == "EYE2":
            if self.eye_result == "DONE_OK":
                self.eye_result = None
                self._publish_string_command(self.pub_hyrrt_cmd, "RESET")

                self._set_phase("HYRRT_PLAN")
                self.hyrrt_result = None
                self._publish_string_command(self.pub_hyrrt_cmd, self.cmd_plan)
                return

            if self.eye_result == "DONE_FAIL":
                self._finish_fail("Eye-in-hand (mode2) failed")
                return

            if self._elapsed_phase_time() > self.eye_timeout_sec:
                self._finish_fail("Eye-in-hand (mode2) timeout")
            return

        if self.state == "HYRRT_PLAN":
            if self.hyrrt_result == "DONE_FAIL":
                self._finish_fail("HyRRT planning failed")
                return

            if self.hyrrt_result == "DONE_OK":
                self.hyrrt_result = None
                self._set_phase("HYRRT_STREAM")
                self.ctrl_result = None

                self.mode2_hyrrt_finished = False
                self.mode2_ctrl_finished = False

                self._publish_string_command(
                    self.pub_ctrl_cmd,
                    self.cmd_execute_stream,
                )
                return

            if self._elapsed_phase_time() > self.hyrrt_plan_timeout_sec:
                self._finish_fail("HyRRT planning timeout")
            return

        if self.state == "HYRRT_STREAM":
            if self.ctrl_result == "DONE_OK":
                self.ctrl_result = None
                self.mode2_ctrl_finished = True
                if not self.mode2_hyrrt_finished:
                    self.get_logger().warn(
                        "MODE2: control DONE_OK but HyRRT not finished yet -> waiting."
                    )
                    return

            if self.ctrl_result == "DONE_FAIL":
                self._finish_fail("Control execute (mode2) failed")
                return

            if self.hyrrt_result == "DONE_FAIL":
                self._finish_fail("HyRRT stream failed")
                return

            if self.mode2_hyrrt_finished and self.mode2_ctrl_finished:
                if not self.enable_gripper:
                    self._finish_ok("Mode2 DONE_OK (gripper disabled)")
                    return

                self._set_phase("GRIPPER")
                self.gripper_result = None
                self._publish_string_command(self.pub_gripper_cmd, self.cmd_grasp)
                return

            if self._elapsed_phase_time() > self.hyrrt_exec_timeout_sec:
                self._finish_fail("HyRRT stream timeout")
            return

        if self.state == "GRIPPER":
            if self.gripper_result == "DONE_OK":
                self._finish_ok("Mode2 grasp DONE_OK")
                return

            if self.gripper_result == "DONE_FAIL":
                self._finish_fail("Gripper DONE_FAIL")
                return

            if self._elapsed_phase_time() > self.gripper_timeout_sec:
                self._finish_fail("Gripper timeout")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MasterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()