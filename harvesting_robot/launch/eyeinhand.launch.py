from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Camera mounting transform parameters: tool0 -> camera_link
    mount_x_arg = DeclareLaunchArgument("cam_x", default_value="-0.03325")
    mount_y_arg = DeclareLaunchArgument("cam_y", default_value="-0.03315")
    mount_z_arg = DeclareLaunchArgument("cam_z", default_value="0.11506")

    mount_roll_arg = DeclareLaunchArgument("cam_roll", default_value="0.0")
    mount_pitch_arg = DeclareLaunchArgument("cam_pitch", default_value="-1.57079632679")
    mount_yaw_arg = DeclareLaunchArgument("cam_yaw", default_value="3.14159265359")

    # Fixed transform from the robot flange reference to the tool frame.
    static_end_to_tool0 = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="elfin_end_to_tool0",
        arguments=[
            "0", "0", "0",
            "0", "0", "0",
            "elfin_end_link",
            "tool0",
        ],
    )

    # Mount transform used to position the camera with respect to tool0.
    static_tool0_to_camera = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="tool0_to_camera_link",
        arguments=[
            LaunchConfiguration("cam_x"),
            LaunchConfiguration("cam_y"),
            LaunchConfiguration("cam_z"),
            LaunchConfiguration("cam_roll"),
            LaunchConfiguration("cam_pitch"),
            LaunchConfiguration("cam_yaw"),
            "tool0",
            "camera_link",
        ],
    )

    # Optical frame transform for the camera image convention.
    static_camera_to_optical = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_link_to_optical",
        arguments=[
            "0", "0", "0",
            "-1.57079632679", "0", "-1.57079632679",
            "camera_link",
            "camera_color_optical_frame",
        ],
    )

    # Transform camera detections into the robot base frame.
    eye_in_hand_node = Node(
        package="harvesting_robot",
        executable="eyeinhand_node",
        name="eyeinhand_node",
        parameters=[{
            "input_point_topic": "/camera_sphere",
            "output_point_topic": "/target_base",
            "target_frame": "elfin_base",
            "tf_timeout_sec": 0.8,
            "compute_timeout_sec": 3.0,
            "require_fresh_point": False,
        }],
    )

    return LaunchDescription([
        mount_x_arg,
        mount_y_arg,
        mount_z_arg,
        mount_roll_arg,
        mount_pitch_arg,
        mount_yaw_arg,
        static_end_to_tool0,
        static_tool0_to_camera,
        static_camera_to_optical,
        eye_in_hand_node,
    ])