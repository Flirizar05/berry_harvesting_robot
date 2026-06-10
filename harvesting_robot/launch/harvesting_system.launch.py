from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
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
        "eyeinhandkinova.launch.py",
    )
    default_urdf_path = os.path.join(
        harvesting_robot_share,
        "urdf",
        "gen3.urdf",
    )
    default_search_model_path = os.path.join(
        harvesting_robot_share,
        "models",
        "best.pt",
    )
    default_obb_pca_model_path = os.path.join(
        harvesting_robot_share,
        "models",
        "yolo_obb_pca_best.pt",
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
            condition=UnlessCondition(LaunchConfiguration("mode1_use_pf_vision")),
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
        # Potential fields vision node used by Mode 1 PF vision and optional Mode 2.
        # ---------------------------------------------------------------------
        Node(
            package="harvesting_robot",
            executable="mode2_vision_node",
            name="mode2_vision_node",
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration("enable_mode2"), "'.lower() == 'true' or '",
                LaunchConfiguration("mode1_use_pf_vision"), "'.lower() == 'true'",
            ])),
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
                "model_path": pv("search_model_path", str),
                "yolo_device": pv("search_yolo_device", str),
                "conf_thresh": pv("search_conf_thresh", float),
                "nms_thresh": pv("search_nms_thresh", float),
                "target_class_id": pv("search_target_class_id", int),
            }],
        ),
        Node(
            package="harvesting_robot",
            executable="obb_pca_vision_node",
            name="obb_pca_vision_node",
            output="screen",
            parameters=[{
                "cmd_topic": pv("obb_pca_cmd_topic", str),
                "status_topic": pv("obb_pca_status_topic", str),
                "result_topic": pv("obb_pca_result_topic", str),
                "annotated_topic": pv("obb_pca_annotated_topic", str),
                "publish_annotated": pv("obb_pca_publish_annotated", bool),
                "color_topic": pv("camera_color_topic", str),
                "depth_topic": pv("camera_depth_topic", str),
                "camera_info_topic": pv("camera_color_info_topic", str),
                "depth_scale_topic": pv("depth_scale_topic", str),
                "depth_scale_fallback": pv("depth_scale_fallback", float),
                "yolo_model_path": pv("obb_pca_yolo_model_path", str),
                "sam_model_path": pv("obb_pca_sam_model_path", str),
                "yolo_device": pv("search_yolo_device", str),
                "conf_thresh": pv("search_conf_thresh", float),
                "nms_thresh": pv("search_nms_thresh", float),
                "target_class_id": pv("search_target_class_id", int),
                "max_detections": pv("obb_pca_max_detections", int),
                "min_valid_depth_m": pv("search_min_valid_depth_m", float),
                "max_valid_depth_m": pv("search_max_valid_depth_m", float),
                "show_result": pv("obb_pca_show_result", bool),
                "result_window_width": pv("obb_pca_result_window_width", int),
                "result_window_height": pv("obb_pca_result_window_height", int),
                "result_window_x": pv("obb_pca_result_window_x", int),
                "result_window_y": pv("obb_pca_result_window_y", int),
                "result_wait_ms": pv("obb_pca_result_wait_ms", int),
                "single_shot": True,
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
                "final_approach_topic": pv("traj_final_approach_topic", str),
                "projection_distance_m": pv("projection_distance_m", float),
                "final_approach_direct_target": pv("final_approach_direct_target", bool),
                "final_approach_goal_tol_m": pv("final_approach_goal_tol_m", float),
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
            condition=IfCondition(LaunchConfiguration("enable_mode2")),
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
                "target_side_mode": pv("target_side_mode", str),
                "target_side_mode_topic": pv("target_side_mode_topic", str),
                "active_target_side_topic": pv("active_target_side_topic", str),
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
            condition=IfCondition(LaunchConfiguration("enable_gripper")),
            output="screen",
            parameters=[{
                "esp32_ip": pv("gripper_esp32_ip", str),
                "esp32_port": pv("gripper_esp32_port", int),
                "grasp_cmd": pv("gripper_grasp_cmd", str),
                "release_cmd": pv("gripper_release_cmd", str),
                "connect_timeout_sec": pv("gripper_connect_timeout_sec", float),
                "response_timeout_sec": pv("gripper_response_timeout_sec", float),
                "wait_for_response": pv("gripper_wait_for_response", bool),
                "append_newline": pv("gripper_append_newline", bool),
            }],
        ),
        Node(
            package="harvesting_robot",
            executable="master_node",
            name="master_node",
            output="screen",
            parameters=[{
                "master_cmd": pv("master_cmd_topic", str),
                "master_status": pv("master_status_topic", str),
                "enable_mode2": pv("enable_mode2", bool),
                "enable_gripper": pv("enable_gripper", bool),
                "mode1_use_pf_vision": pv("mode1_use_pf_vision", bool),
                "mode1_execute_stream": pv("mode1_execute_stream", bool),
                "mode1_final_approach_distance_m": pv("projection_distance_m", float),
                "stop_distance_m": pv("master_stop_distance_m", float),
                "stop_margin_m": pv("master_stop_margin_m", float),
                "obb_pca_cmd": pv("obb_pca_cmd_topic", str),
                "trigger_obb_pca_at_final_vision": pv("trigger_obb_pca_at_final_vision", bool),
                "tcp_target_dist_topic": pv("tcp_target_dist_topic", str),
                "traj_final_approach_topic": pv("traj_final_approach_topic", str),
                "do_home_on_start": pv("do_home_on_start", bool),
                "controller_topic": pv("controller_topic", str),
                "joint_state_topic": pv("joint_state_topic", str),
            }],
        ),
        Node(
            package="harvesting_robot",
            executable="searching_mode_node",
            name="searching_mode_node",
            condition=IfCondition(LaunchConfiguration("enable_searching_mode")),
            output="screen",
            parameters=[{
                "cmd_topic": pv("searching_cmd_topic", str),
                "status_topic": pv("searching_status_topic", str),
                "controller_topic": pv("controller_topic", str),
                "color_topic": pv("camera_color_topic", str),
                "depth_topic": pv("camera_depth_topic", str),
                "camera_info_topic": pv("camera_color_info_topic", str),
                "depth_scale_topic": pv("depth_scale_topic", str),
                "depth_scale_fallback": pv("depth_scale_fallback", float),
                "joint_state_topic": pv("joint_state_topic", str),
                "detection_result_topic": pv("searching_detection_result_topic", str),
                "annotated_image_topic": pv("searching_annotated_image_topic", str),
                "show_preview": pv("searching_show_preview", bool),
                "output_point_topic": pv("pf_output_point_topic", str),
                "target_base_topic": pv("target_topic", str),
                "eye_cmd_topic": pv("eye_cmd_topic", str),
                "lidar_side_clearance_topic": pv("lidar_side_clearance_topic", str),
                "target_side_mode": pv("target_side_mode", str),
                "target_side_mode_topic": pv("target_side_mode_topic", str),
                "active_target_side_topic": pv("active_target_side_topic", str),
                "auto_harvest_enabled": pv("auto_harvest_enabled", bool),
                "auto_harvest_topic": pv("auto_harvest_topic", str),
                "master_cmd_topic": pv("master_cmd_topic", str),
                "master_status_topic": pv("master_status_topic", str),
                "post_harvest_resume_delay_sec": pv("post_harvest_resume_delay_sec", float),
                "model_path": pv("search_model_path", str),
                "yolo_device": pv("search_yolo_device", str),
                "conf_thresh": pv("search_conf_thresh", float),
                "nms_thresh": pv("search_nms_thresh", float),
                "target_class_id": pv("search_target_class_id", int),
                "min_valid_depth_m": pv("search_min_valid_depth_m", float),
                "max_valid_depth_m": pv("search_max_valid_depth_m", float),
                "negative_joint_side": pv("search_negative_joint_side", str),
                "side_deadband_deg": pv("search_side_deadband_deg", float),
                "joint_state_max_age_sec": pv("search_joint_state_max_age_sec", float),
            }],
        ),
    ]

    return LaunchDescription([
        # ---------------------------------------------------------------------
        # Frames and robot interfaces
        # ---------------------------------------------------------------------
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
        DeclareLaunchArgument(
            "controller_topic",
            default_value="/joint_trajectory_controller/joint_trajectory",
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
        DeclareLaunchArgument("vision_show_preview", default_value="false"),

        # ---------------------------------------------------------------------
        # High-level mode switches
        # ---------------------------------------------------------------------
        DeclareLaunchArgument("enable_mode2", default_value="false"),
        DeclareLaunchArgument("enable_gripper", default_value="false"),
        DeclareLaunchArgument("mode1_use_pf_vision", default_value="true"),
        DeclareLaunchArgument("mode1_execute_stream", default_value="true"),
        DeclareLaunchArgument("do_home_on_start", default_value="true"),
        DeclareLaunchArgument("enable_searching_mode", default_value="true"),
        DeclareLaunchArgument("trigger_obb_pca_at_final_vision", default_value="true"),
        DeclareLaunchArgument("master_cmd_topic", default_value="/master/cmd"),
        DeclareLaunchArgument("master_status_topic", default_value="/master/status"),

        # ---------------------------------------------------------------------
        # PF vision model
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            "search_model_path",
            default_value=default_search_model_path,
        ),
        DeclareLaunchArgument("search_yolo_device", default_value="0"),
        DeclareLaunchArgument("search_conf_thresh", default_value="0.6"),
        DeclareLaunchArgument("search_nms_thresh", default_value="0.4"),
        DeclareLaunchArgument("search_target_class_id", default_value="2"),
        DeclareLaunchArgument("search_min_valid_depth_m", default_value="0.10"),
        DeclareLaunchArgument("search_max_valid_depth_m", default_value="2.00"),
        DeclareLaunchArgument("obb_pca_cmd_topic", default_value="/obb_pca/cmd"),
        DeclareLaunchArgument("obb_pca_status_topic", default_value="/obb_pca/status"),
        DeclareLaunchArgument("obb_pca_result_topic", default_value="/obb_pca/result"),
        DeclareLaunchArgument(
            "obb_pca_yolo_model_path",
            default_value=default_obb_pca_model_path,
        ),
        DeclareLaunchArgument(
            "obb_pca_annotated_topic",
            default_value="/obb_pca/annotated_image",
        ),
        DeclareLaunchArgument("obb_pca_publish_annotated", default_value="true"),
        DeclareLaunchArgument("obb_pca_sam_model_path", default_value="mobile_sam.pt"),
        DeclareLaunchArgument("obb_pca_max_detections", default_value="5"),
        DeclareLaunchArgument("obb_pca_show_result", default_value="true"),
        DeclareLaunchArgument("obb_pca_result_window_width", default_value="960"),
        DeclareLaunchArgument("obb_pca_result_window_height", default_value="720"),
        DeclareLaunchArgument("obb_pca_result_window_x", default_value="720"),
        DeclareLaunchArgument("obb_pca_result_window_y", default_value="40"),
        DeclareLaunchArgument("obb_pca_result_wait_ms", default_value="30"),
        DeclareLaunchArgument("search_negative_joint_side", default_value="right"),
        DeclareLaunchArgument("search_side_deadband_deg", default_value="2.0"),
        DeclareLaunchArgument("search_joint_state_max_age_sec", default_value="0.5"),
        DeclareLaunchArgument("searching_cmd_topic", default_value="/searching_mode/cmd"),
        DeclareLaunchArgument("searching_status_topic", default_value="/searching_mode/status"),
        DeclareLaunchArgument(
            "searching_detection_result_topic",
            default_value="/searching_mode/detection",
        ),
        DeclareLaunchArgument(
            "searching_annotated_image_topic",
            default_value="/searching_mode/annotated_image",
        ),
        DeclareLaunchArgument("searching_show_preview", default_value="false"),
        DeclareLaunchArgument(
            "lidar_side_clearance_topic",
            default_value="/agv/line_detections",
        ),
        DeclareLaunchArgument("target_side_mode", default_value="left"),
        DeclareLaunchArgument(
            "target_side_mode_topic",
            default_value="/searching_mode/target_side",
        ),
        DeclareLaunchArgument(
            "active_target_side_topic",
            default_value="/searching_mode/active_target_side",
        ),
        DeclareLaunchArgument("auto_harvest_enabled", default_value="true"),
        DeclareLaunchArgument(
            "auto_harvest_topic",
            default_value="/searching_mode/auto_harvest",
        ),
        DeclareLaunchArgument("post_harvest_resume_delay_sec", default_value="5.0"),

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
        DeclareLaunchArgument("traj_final_approach_topic", default_value="/trajectory/final_approach"),

        DeclareLaunchArgument("ctrl_cmd_topic", default_value="/control/cmd"),
        DeclareLaunchArgument("ctrl_status_topic", default_value="/control/status"),
        DeclareLaunchArgument("eye_cmd_topic", default_value="/eyeinhand/cmd"),

        #DeclareLaunchArgument("projection_distance_m", default_value="0.15"),
        DeclareLaunchArgument("projection_distance_m", default_value="0.25"),
        DeclareLaunchArgument("final_approach_direct_target", default_value="true"),
        DeclareLaunchArgument("final_approach_goal_tol_m", default_value="0.005"),
        DeclareLaunchArgument("master_stop_distance_m", default_value="0.005"),
        DeclareLaunchArgument("master_stop_margin_m", default_value="0.02"),

        # ---------------------------------------------------------------------
        # Control
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            "urdf_path",
            default_value=default_urdf_path,
        ),
        DeclareLaunchArgument("ee_link", default_value="tool_frame"),

        DeclareLaunchArgument("dt", default_value="0.02"),
        DeclareLaunchArgument("command_horizon_sec", default_value="0.05"),
        DeclareLaunchArgument("kp_pos", default_value="8.0"),
        DeclareLaunchArgument("kp_ori", default_value="1.0"),
        DeclareLaunchArgument("damp_pos", default_value="0.5"),
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

        # ---------------------------------------------------------------------
        # ESP32 gripper
        # ---------------------------------------------------------------------
        DeclareLaunchArgument("gripper_esp32_ip", default_value="10.42.0.59"),
        DeclareLaunchArgument("gripper_esp32_port", default_value="5000"),
        DeclareLaunchArgument("gripper_grasp_cmd", default_value="a"),
        DeclareLaunchArgument("gripper_release_cmd", default_value="b"),
        DeclareLaunchArgument("gripper_connect_timeout_sec", default_value="3.0"),
        DeclareLaunchArgument("gripper_response_timeout_sec", default_value="3.0"),
        DeclareLaunchArgument("gripper_wait_for_response", default_value="true"),
        DeclareLaunchArgument("gripper_append_newline", default_value="false"),

        *nodes,
    ])
