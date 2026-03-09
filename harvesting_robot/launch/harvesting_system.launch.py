from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory

import os


def pv(name: str, value_type):
    """Return a typed launch configuration parameter."""
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description():
    harvesting_robot_share = get_package_share_directory("harvesting_robot")
    eye_launch_path = os.path.join(
        harvesting_robot_share,
        "launch",
        "eyeinhand.launch.py",
    )
    default_urdf_path = os.path.join(
        harvesting_robot_share,
        "urdf",
        "elfin3.urdf",
    )

    nodes = [
        # ---------------------------------------------------------------------
        # Camera node
        # ---------------------------------------------------------------------
        Node(
            package="harvesting_robot",
            executable="camera_node",
            name="camera_node",
            output="screen",
            parameters=[{
                "depth_width": pv("depth_width", int),
                "depth_height": pv("depth_height", int),
                "color_width": pv("color_width", int),
                "color_height": pv("color_height", int),
                "fps": pv("camera_fps", int),
                "color_topic": pv("camera_color_topic", str),
                "depth_topic": pv("camera_depth_topic", str),
                "color_info_topic": pv("camera_color_info_topic", str),
                "depth_info_topic": pv("camera_depth_info_topic", str),
                "frame_id": pv("camera_frame_id", str),
                "publish_rate_hz": pv("camera_publish_rate", float),
            }],
        ),

        # ---------------------------------------------------------------------
        # Mode 1 vision node
        # ---------------------------------------------------------------------
        Node(
            package="harvesting_robot",
            executable="mode1_vision_node",
            name="mode1_vision_node",
            output="screen",
            parameters=[{
                "cmd_topic": pv("vision_cmd_topic", str),
                "status_topic": pv("vision_status_topic", str),
                "color_topic": pv("camera_color_topic", str),
                "depth_topic": pv("camera_depth_topic", str),
                "camera_info_topic": pv("camera_color_info_topic", str),
                "depth_scale_topic": pv("depth_scale_topic", str),
                "depth_scale_fallback": pv("depth_scale_fallback", float),
                "show_preview": pv("vision_show_preview", bool),
            }],
        ),

        # ---------------------------------------------------------------------
        # Eye-in-hand transform pipeline
        # ---------------------------------------------------------------------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(eye_launch_path)
        ),

        # ---------------------------------------------------------------------
        # Mode 2 vision node
        # ---------------------------------------------------------------------
        Node(
            package="harvesting_robot",
            executable="mode2_vision_node",
            name="mode2_vision_node",
            output="screen",
            parameters=[{
                "cmd_topic": pv("pf_cmd_topic", str),
                "status_topic": pv("pf_status_topic", str),
                "output_point_topic": pv("pf_output_point_topic", str),
                "output_radius_topic": pv("pf_output_radius_topic", str),
                "color_topic": pv("camera_color_topic", str),
                "depth_topic": pv("camera_depth_topic", str),
                "camera_info_topic": pv("camera_color_info_topic", str),
                "depth_scale_topic": pv("depth_scale_topic", str),
                "depth_scale_fallback": pv("depth_scale_fallback", float),
            }],
        ),

        # ---------------------------------------------------------------------
        # Mode 1 trajectory node
        # ---------------------------------------------------------------------
        Node(
            package="harvesting_robot",
            executable="mode1_trajectory_node",
            name="mode1_trajectory_node",
            output="screen",
            parameters=[{
                "base_frame": pv("base_frame", str),
                "target_topic": pv("target_topic", str),
                "radius_topic": pv("radius_topic", str),
                "joint_state_topic": pv("joint_state_topic", str),
                "cmd_topic": pv("traj_cmd_topic", str),
                "status_topic": pv("traj_status_topic", str),
                "waypoint_topic": pv("traj_waypoint_topic", str),
                "path_topic": pv("traj_path_topic", str),
                "tcp_target_dist_topic": pv("tcp_target_dist_topic", str),
                "projection_distance_m": pv("projection_distance_m", float),
            }],
        ),

        # ---------------------------------------------------------------------
        # Mode 2 trajectory node
        # Publishes to the same waypoint topic consumed by control_node.
        # ---------------------------------------------------------------------
        Node(
            package="harvesting_robot_cpp",
            executable="mode2_trajectory_node",
            name="mode2_trajectory_node",
            output="screen",
            parameters=[{
                "cmd_topic": pv("hyrrt_cmd_topic", str),
                "status_topic": pv("hyrrt_status_topic", str),
                "waypoint_topic": pv("traj_waypoint_topic", str),
                "target_topic": pv("hyrrt_target_topic", str),
                "joint_state_topic": pv("joint_state_topic", str),
                "urdf_path": pv("urdf_path", str),
                "base_frame": pv("base_frame", str),
                "ee_link": pv("ee_link", str),
                "waypoint_tol_m": pv("hyrrt_waypoint_tol_m", float),
                "goal_tol_m": pv("hyrrt_goal_tol_m", float),
                "planning_time": pv("hyrrt_planning_time", float),
                "max_cartesian_vel": pv("hyrrt_max_cartesian_vel", float),
                "flow_step": pv("hyrrt_flow_step", float),
                "waypoint_dt": pv("hyrrt_waypoint_dt", float),
            }],
        ),

        # ---------------------------------------------------------------------
        # Control node
        # ---------------------------------------------------------------------
        Node(
            package="harvesting_robot",
            executable="control_node",
            name="control_node",
            output="screen",
            parameters=[{
                "urdf_path": pv("urdf_path", str),
                "ee_link": pv("ee_link", str),
                "joint_state_topic": pv("joint_state_topic", str),
                "controller_topic": pv("controller_topic", str),
                "dt": pv("dt", float),
                "command_horizon_sec": pv("command_horizon_sec", float),
                "kp_pos": pv("kp_pos", float),
                "kp_ori": pv("kp_ori", float),
                "damp_pos": pv("damp_pos", float),
                "damp_ori": pv("damp_ori", float),
                "max_joint_step_rad": pv("max_joint_step_rad", float),
                "pos_tol_m": pv("pos_tol_m", float),
                "settle_cycles": pv("settle_cycles", int),
                "base_frame": pv("base_frame", str),
                "use_tf": pv("use_tf", bool),
                "waypoint_topic": pv("traj_waypoint_topic", str),
                "cmd_topic": pv("ctrl_cmd_topic", str),
                "status_topic": pv("ctrl_status_topic", str),
                "execute_timeout_sec": pv("ctrl_execute_timeout_sec", float),
                "waypoint_timeout_sec": pv("ctrl_waypoint_timeout_sec", float),
                "enable_nullspace": pv("enable_nullspace", bool),
                "nullspace_gain": pv("nullspace_gain", float),
                "limit_margin_rad": pv("limit_margin_rad", float),
                "limit_push_gain": pv("limit_push_gain", float),
            }],
        ),

        # ---------------------------------------------------------------------
        # End-effector and high-level coordinator
        # ---------------------------------------------------------------------
        Node(
            package="harvesting_robot",
            executable="gripper_node",
            name="gripper_node",
            output="screen",
        ),
        Node(
            package="harvesting_robot",
            executable="master_node",
            name="master_node",
            output="screen",
        ),
    ]

    return LaunchDescription([
        # ---------------------------------------------------------------------
        # Frames and robot interfaces
        # ---------------------------------------------------------------------
        DeclareLaunchArgument("base_frame", default_value="elfin_base"),
        DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
        DeclareLaunchArgument(
            "controller_topic",
            default_value="/elfin_arm_controller/joint_trajectory",
        ),

        # ---------------------------------------------------------------------
        # Camera topics and configuration
        # ---------------------------------------------------------------------
        DeclareLaunchArgument("camera_color_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument("camera_depth_topic", default_value="/camera/aligned_depth/image_raw"),
        DeclareLaunchArgument("camera_color_info_topic", default_value="/camera/color/camera_info"),
        DeclareLaunchArgument("camera_depth_info_topic", default_value="/camera/aligned_depth/camera_info"),
        DeclareLaunchArgument("camera_frame_id", default_value="camera_color_optical_frame"),

        DeclareLaunchArgument("depth_scale_topic", default_value="/camera/depth_scale"),
        DeclareLaunchArgument("depth_scale_fallback", default_value="0.001"),
        DeclareLaunchArgument("depth_width", default_value="640"),
        DeclareLaunchArgument("depth_height", default_value="480"),
        DeclareLaunchArgument("color_width", default_value="640"),
        DeclareLaunchArgument("color_height", default_value="480"),
        DeclareLaunchArgument("camera_fps", default_value="30"),
        DeclareLaunchArgument("camera_publish_rate", default_value="30.0"),

        # ---------------------------------------------------------------------
        # Vision
        # ---------------------------------------------------------------------
        DeclareLaunchArgument("vision_cmd_topic", default_value="/vision/cmd"),
        DeclareLaunchArgument("vision_status_topic", default_value="/vision/status"),
        DeclareLaunchArgument("vision_show_preview", default_value="true"),

        # ---------------------------------------------------------------------
        # Mode 1 topics
        # ---------------------------------------------------------------------
        DeclareLaunchArgument("target_topic", default_value="/target_base"),
        DeclareLaunchArgument("radius_topic", default_value="/sphere_radius"),

        DeclareLaunchArgument("traj_cmd_topic", default_value="/trajectory/cmd"),
        DeclareLaunchArgument("traj_status_topic", default_value="/trajectory/status"),
        DeclareLaunchArgument("traj_waypoint_topic", default_value="/trajectory/waypoint"),
        DeclareLaunchArgument("traj_path_topic", default_value="/trajectory/path"),
        DeclareLaunchArgument("tcp_target_dist_topic", default_value="/trajectory/tcp_target_dist"),

        DeclareLaunchArgument("ctrl_cmd_topic", default_value="/control/cmd"),
        DeclareLaunchArgument("ctrl_status_topic", default_value="/control/status"),

        DeclareLaunchArgument("projection_distance_m", default_value="0.15"),

        # ---------------------------------------------------------------------
        # Control
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            "urdf_path",
            default_value=default_urdf_path,
        ),
        DeclareLaunchArgument("ee_link", default_value="rg2ft_grasp_point"),

        DeclareLaunchArgument("dt", default_value="0.02"),
        DeclareLaunchArgument("command_horizon_sec", default_value="0.05"),
        DeclareLaunchArgument("kp_pos", default_value="10.0"),
        DeclareLaunchArgument("kp_ori", default_value="0.5"),
        DeclareLaunchArgument("damp_pos", default_value="0.1"),
        DeclareLaunchArgument("damp_ori", default_value="0.05"),
        DeclareLaunchArgument("max_joint_step_rad", default_value="0.02"),
        DeclareLaunchArgument("pos_tol_m", default_value="0.04"),
        DeclareLaunchArgument("settle_cycles", default_value="20"),
        DeclareLaunchArgument("use_tf", default_value="true"),
        DeclareLaunchArgument("ctrl_execute_timeout_sec", default_value="120.0"),
        DeclareLaunchArgument("ctrl_waypoint_timeout_sec", default_value="30.0"),
        DeclareLaunchArgument("enable_nullspace", default_value="true"),
        DeclareLaunchArgument("nullspace_gain", default_value="1.0"),
        DeclareLaunchArgument("limit_margin_rad", default_value="0.30"),
        DeclareLaunchArgument("limit_push_gain", default_value="8.0"),

        # ---------------------------------------------------------------------
        # Potential fields
        # ---------------------------------------------------------------------
        DeclareLaunchArgument("pf_cmd_topic", default_value="/potentialfields/cmd"),
        DeclareLaunchArgument("pf_status_topic", default_value="/potentialfields/status"),
        DeclareLaunchArgument("pf_output_point_topic", default_value="/camera_sphere"),
        DeclareLaunchArgument("pf_output_radius_topic", default_value="/sphere_radius"),

        # ---------------------------------------------------------------------
        # HyRRT
        # ---------------------------------------------------------------------
        DeclareLaunchArgument("hyrrt_cmd_topic", default_value="/hyrrt/cmd"),
        DeclareLaunchArgument("hyrrt_status_topic", default_value="/hyrrt/status"),
        DeclareLaunchArgument("hyrrt_target_topic", default_value="/target_base"),
        DeclareLaunchArgument("hyrrt_waypoint_tol_m", default_value="0.15"),
        DeclareLaunchArgument("hyrrt_goal_tol_m", default_value="0.02"),
        DeclareLaunchArgument("hyrrt_planning_time", default_value="60.0"),
        DeclareLaunchArgument("hyrrt_max_cartesian_vel", default_value="0.10"),
        DeclareLaunchArgument("hyrrt_flow_step", default_value="0.01"),
        DeclareLaunchArgument("hyrrt_waypoint_dt", default_value="0.01"),

        *nodes,
    ])