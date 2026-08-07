# Berry Harvesting Robot

ROS 2 workspace for a berry harvesting platform built around a Kinova Gen3 arm, an eye-in-hand Intel RealSense camera, vision-based target detection, Cartesian trajectory execution, and an AGV subsystem with RPLIDAR, BLE motor control, and a web control panel.

The recommended robot configuration for this project is the Kinova Gen3 flow. Legacy Elfin-related files may still exist in the repository, but the main launch path uses `gen3.urdf`, `base_link`, `tool_frame`, and `eyeinhandkinova.launch.py`.

## Table of Contents

- [System Overview](#system-overview)
- [Expected Hardware](#expected-hardware)
- [External Dependencies](#external-dependencies)
- [Installation](#installation)
- [Build](#build)
- [Running the System](#running-the-system)
- [Common Use Cases](#common-use-cases)
- [Main ROS Nodes](#main-ros-nodes)
- [Basic ROS Commands](#basic-ros-commands)
- [Important Launch Parameters](#important-launch-parameters)
- [Eye-in-Hand Calibration](#eye-in-hand-calibration)
- [AGV and LiDAR Subsystem](#agv-and-lidar-subsystem)
- [Mode 2 and Custom OMPL](#mode-2-and-custom-ompl)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)

## System Overview

The arm-side harvesting pipeline is:

```text
Intel RealSense
  -> camera_node
  -> vision node
  -> /camera_sphere
  -> eyeinhand_node
  -> /target_base
  -> trajectory node
  -> control_node
  -> /joint_trajectory_controller/joint_trajectory
  -> Kinova Gen3
```

The AGV-side pipeline is:

```text
RPLIDAR A1
  -> sllidar_ros2
  -> /scan
  -> lidar_360_radar_node
  -> /agv/line_detections
  -> agv_position_control_node
  -> /agv/rpm_cmd
  -> ble_agv_bridge_node
  -> ESP32-S3 motor controller
```

The `master_node` coordinates the harvesting cycle for the arm. The `searching_mode_node` can connect the arm, camera, AGV, LiDAR side-clearance detections, and automatic harvesting commands.

## Expected Hardware

- Ubuntu 22.04 development PC.
- ROS 2 Humble.
- Kinova Gen3 arm.
- Kinova Kortex ROS 2 driver.
- Intel RealSense depth camera mounted eye-in-hand.
- RPLIDAR A1 for the AGV.
- ESP32-S3 AGV motor controller using BLE UART.

## External Dependencies

| Component | Purpose | Link |
| --- | --- | --- |
| ROS 2 Humble | Base ROS distribution | https://docs.ros.org/en/humble/Installation.html |
| colcon | ROS 2 workspace build tool | https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html |
| rosdep | ROS dependency resolver | https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosdep.html |
| Kinova ROS 2 Kortex | Kinova Gen3 driver and controllers | https://github.com/Kinovarobotics/ros2_kortex/tree/humble |
| Intel RealSense librealsense | RealSense SDK | https://github.com/realsenseai/librealsense |
| pyrealsense2 | Python RealSense wrapper used by `camera_node` | https://github.com/realsenseai/librealsense/blob/master/wrappers/python/readme.md |
| RealSense ROS wrapper | Optional camera validation through ROS | https://github.com/realsenseai/realsense-ros |
| SLLIDAR ROS 2 | ROS 2 driver for RPLIDAR | https://github.com/Slamtec/sllidar_ros2 |
| Custom OMPL fork | HyRRT and `HybridStateSpace` support for Mode 2 | https://github.com/xu21beve/ompl |
| OMPL documentation | General OMPL reference | https://ompl.kavrakilab.org/core/installation.html |

## Installation

### 1. Install ROS 2 Humble

Follow the official ROS 2 Humble installation guide:

```text
https://docs.ros.org/en/humble/Installation.html
```

Source ROS 2 in each new terminal:

```bash
source /opt/ros/humble/setup.bash
```

Install common development tools:

```bash
sudo apt update
sudo apt install python3-colcon-common-extensions python3-vcstool python3-rosdep git
```

If `rosdep` has not been initialized on the machine:

```bash
sudo rosdep init
rosdep update
```

### 2. Install Kinova Kortex

The recommended setup for this project is Kortex for ROS 2 Humble.

Create a Kortex workspace:

```bash
mkdir -p ~/workspace/ros2_kortex_ws/src
cd ~/workspace/ros2_kortex_ws
```

Clone the Humble branch:

```bash
git clone -b humble --single-branch https://github.com/Kinovarobotics/ros2_kortex.git src/ros2_kortex
```

Import Kortex dependencies:

```bash
vcs import src --skip-existing --input src/ros2_kortex/ros2_kortex.humble.repos
vcs import src --skip-existing --input src/ros2_kortex/ros2_kortex-not-released.humble.repos
```

Install dependencies and build:

```bash
rosdep install --ignore-src --from-paths src -y -r
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --parallel-workers 3
source install/setup.bash
```

Optional but recommended when using MoveIt:

```bash
sudo apt install ros-humble-rmw-cyclonedds-cpp
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

Test the physical Kinova Gen3:

```bash
ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10
```

Change `robot_ip` if your arm uses a different address.

This harvesting project expects the Kinova driver to provide:

```text
/joint_states
/joint_trajectory_controller/joint_trajectory
base_link
end_effector_link
tool_frame
```

### 3. Install RealSense Support

This project uses `pyrealsense2` directly in `camera_node`. The `realsense2_camera` ROS wrapper is optional and is mainly useful for independent camera testing.

Install the Python wrapper:

```bash
python3 -m pip install pyrealsense2
```

Install the rest of the Python packages used by the project:

```bash
python3 -m pip install "opencv-python>=4.6,<4.12" "numpy>=1.23,<2" ultralytics bleak
```

If the camera is not detected, install or repair `librealsense` using the official instructions:

```text
https://github.com/realsenseai/librealsense/blob/master/doc/installation.md
```

Optional ROS wrapper test:

```bash
sudo apt install ros-humble-realsense2-*
ros2 launch realsense2_camera rs_launch.py
```

### 4. Install Custom OMPL for Mode 2

Mode 2 uses the C++ node `mode2_trajectory_node`, which depends on an OMPL fork that includes `HyRRT` and `HybridStateSpace`.

The fork used for this project is:

```text
https://github.com/xu21beve/ompl
```

Install it in the expected workspace location:

```bash
mkdir -p ~/workspace/ompl_ws/src
cd ~/workspace/ompl_ws/src
git clone https://github.com/xu21beve/ompl.git

cd ~/workspace/ompl_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Important: `harvesting_robot_cpp/CMakeLists.txt` may point to absolute OMPL paths under `/home/francisco/workspace/ompl_ws`. If your username or workspace path differs, either install the custom OMPL workspace in an equivalent location or update the OMPL paths in that CMake file.

If you do not need Mode 2, build without the C++ planner:

```bash
colcon build --symlink-install --packages-skip harvesting_robot_cpp
```

Then launch the system with:

```bash
enable_mode2:=false
```

### 5. Create the Main Workspace

Recommended development layout:

```bash
mkdir -p ~/harvesting_robot_ws/src
cd ~/harvesting_robot_ws/src
```

Clone this repository:

```bash
git clone https://github.com/Flirizar05/berry_harvesting_robot.git
```

If using AGV/LiDAR, clone `sllidar_ros2` next to this project:

```bash
git clone https://github.com/Slamtec/sllidar_ros2.git
```

Expected layout:

```text
~/harvesting_robot_ws/
  src/
    berry_harvesting_robot/
      harvesting_robot/
      harvesting_robot_agv/
      harvesting_robot_cpp/
      robot_description/
    sllidar_ros2/
```

### 6. Install ROS Dependencies

From the main workspace:

```bash
cd ~/harvesting_robot_ws

source /opt/ros/humble/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
source ~/workspace/ompl_ws/install/setup.bash

rosdep install --from-paths src --ignore-src -y -r
```

If you are not using Mode 2, omit the `ompl_ws` source line.

For AGV GUI support:

```bash
sudo apt install python3-tk
```

Note: the package may still declare legacy dependencies that are not required for the Kinova-only flow. If `rosdep` cannot resolve an unused legacy dependency, verify whether that package is actually needed for your current setup before installing additional software.

## Build

### Full Build

```bash
cd ~/harvesting_robot_ws

source /opt/ros/humble/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
source ~/workspace/ompl_ws/install/setup.bash

colcon build --symlink-install
source install/setup.bash
```

### Build Without Mode 2

Use this if custom OMPL is not installed:

```bash
cd ~/harvesting_robot_ws

source /opt/ros/humble/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash

colcon build --symlink-install --packages-skip harvesting_robot_cpp
source install/setup.bash
```

### Optional Shell Setup

Add the following to `~/.bashrc` if this is the main development machine:

```bash
source /opt/ros/humble/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
source ~/workspace/ompl_ws/install/setup.bash
source ~/harvesting_robot_ws/install/setup.bash
```

If you do not use Mode 2, do not add the `ompl_ws` line.

## Running the System

### 1. Start Kinova Gen3

Terminal 1:

```bash
source /opt/ros/humble/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash

ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10
```

Verify joint states:

```bash
ros2 topic echo /joint_states
```

### 2. Start the AGV Subsystem

Terminal 2:

```bash
source /opt/ros/humble/setup.bash
source ~/harvesting_robot_ws/install/setup.bash

ros2 launch harvesting_robot_agv agv_web_lidar.launch.py
```

The web control panel is available at:

```text
http://<PC_IP_ADDRESS>:8080
```

If the PC is running an Ubuntu hotspot, the host is commonly:

```text
http://10.42.0.1:8080
```

### 3. Start the Harvesting System

Terminal 3:

```bash
source /opt/ros/humble/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
source ~/workspace/ompl_ws/install/setup.bash
source ~/harvesting_robot_ws/install/setup.bash

ros2 launch harvesting_robot harvesting_system.launch.py
```

Conservative startup for first tests:

```bash
ros2 launch harvesting_robot harvesting_system.launch.py \
  enable_mode2:=false \
  enable_gripper:=false \
  enable_searching_mode:=false \
  enable_obb_pca:=false
```

### 4. Start a Harvesting Cycle

Terminal 4:

```bash
ros2 topic pub --once /master/cmd std_msgs/msg/String "{data: 'START'}"
```

Stop the cycle:

```bash
ros2 topic pub --once /master/cmd std_msgs/msg/String "{data: 'STOP'}"
```

## Common Use Cases

### Camera Only

Use this to verify that the RealSense camera is visible and publishing frames.

```bash
ros2 run harvesting_robot camera_node
```

Expected topics:

```text
/camera/color/image_raw
/camera/aligned_depth/image_raw
/camera/color/camera_info
/camera/aligned_depth/camera_info
/camera/depth_scale
/camera/status
```

Check camera status:

```bash
ros2 topic echo /camera/status
```

### Basic Kinova Harvesting Without AGV or Mode 2

Recommended first full-system test.

```bash
ros2 launch harvesting_robot harvesting_system.launch.py \
  enable_mode2:=false \
  enable_gripper:=false \
  enable_searching_mode:=false \
  enable_obb_pca:=false
```

Start:

```bash
ros2 topic pub --once /master/cmd std_msgs/msg/String "{data: 'START'}"
```

### Kinova With Mode 2 HyRRT

Requires custom OMPL and a successful build of `harvesting_robot_cpp`.

```bash
ros2 launch harvesting_robot harvesting_system.launch.py \
  enable_mode2:=true \
  enable_gripper:=false \
  enable_searching_mode:=false
```

### AGV Manual Web Control

```bash
ros2 launch harvesting_robot_agv agv_web_lidar.launch.py
```

Open:

```text
http://<PC_IP_ADDRESS>:8080
```

Manual command through ROS:

```bash
ros2 topic pub --once /agv/rpm_cmd std_msgs/msg/String "{data: '100,100'}"
ros2 topic pub --once /agv/rpm_cmd std_msgs/msg/String "{data: 's'}"
```

The expected command format is:

```text
right_rpm,left_rpm
```

Each RPM value must be between `0` and `150`. The command `s` stops or releases movement.

### AGV Automatic Mode With Searching Mode

Terminal 1, AGV:

```bash
ros2 launch harvesting_robot_agv agv_web_lidar.launch.py \
  auto_harvest_enabled:=true \
  target_side_mode:=any
```

Terminal 2, arm:

```bash
ros2 launch harvesting_robot harvesting_system.launch.py \
  enable_searching_mode:=true \
  auto_harvest_enabled:=true
```

Start searching:

```bash
ros2 topic pub --once /searching_mode/cmd std_msgs/msg/String "{data: 'START'}"
```

Stop searching:

```bash
ros2 topic pub --once /searching_mode/cmd std_msgs/msg/String "{data: 'STOP'}"
```

### OBB/PCA/SAM Visual Analysis

Requires the OBB YOLO model and SAM model.

```bash
ros2 launch harvesting_robot harvesting_system.launch.py \
  enable_obb_pca:=true \
  obb_pca_show_result:=true
```

Capture manually:

```bash
ros2 topic pub --once /obb_pca/cmd std_msgs/msg/String "{data: 'CAPTURE'}"
```

Reset:

```bash
ros2 topic pub --once /obb_pca/cmd std_msgs/msg/String "{data: 'RESET'}"
```

## Main ROS Nodes

| Node | Package | Purpose |
| --- | --- | --- |
| `camera_node` | `harvesting_robot` | Reads RealSense frames with `pyrealsense2` and publishes color, aligned depth, camera info, depth scale, and an optional point cloud. |
| `mode1_vision_node` | `harvesting_robot` | Detects berries using YOLOv4 through OpenCV DNN and publishes a camera-frame target. |
| `mode2_vision_node` | `harvesting_robot` | Uses an Ultralytics YOLO `.pt` model to publish the target point and radius for potential-fields style targeting. |
| `obb_pca_vision_node` | `harvesting_robot` | Runs one-shot YOLO OBB, SAM segmentation, and 3D PCA visualization. |
| `eyeinhand_node` | `harvesting_robot` | Transforms target points from the camera frame to the Kinova base frame. |
| `mode1_trajectory_node` | `harvesting_robot` | Generates Mode 1 waypoints and path markers toward the transformed target. |
| `mode2_trajectory_node` | `harvesting_robot_cpp` | Uses custom OMPL HyRRT planning for Mode 2. |
| `control_node` | `harvesting_robot` | Converts Cartesian waypoints into Kinova joint trajectory commands. |
| `master_node` | `harvesting_robot` | Coordinates the high-level harvesting sequence. |
| `searching_mode_node` | `harvesting_robot` | Coordinates search posture, camera detection, AGV stop/resume behavior, and optional auto-harvest. |
| `lidar_360_radar_node` | `harvesting_robot_agv` | Processes `/scan`, publishes line detections, side-clearance detections, and a compressed radar image. |
| `agv_position_control_node` | `harvesting_robot_agv` | Produces AGV RPM commands based on detected bush/row line geometry. |
| `ble_agv_bridge_node` | `harvesting_robot_agv` | Sends `/agv/rpm_cmd` to an ESP32-S3 over BLE UART. |
| `agv_web_control_node` | `harvesting_robot_agv` | Hosts the web control panel for manual and automatic AGV operation. |
| `agv_gui_node` | `harvesting_robot_agv` | Local GUI alternative for AGV control. |

## Basic ROS Commands

List active topics:

```bash
ros2 topic list
```

Monitor master status:

```bash
ros2 topic echo /master/status
```

Start harvesting:

```bash
ros2 topic pub --once /master/cmd std_msgs/msg/String "{data: 'START'}"
```

Stop harvesting:

```bash
ros2 topic pub --once /master/cmd std_msgs/msg/String "{data: 'STOP'}"
```

Capture with Mode 1 vision:

```bash
ros2 topic pub --once /vision/cmd std_msgs/msg/String "{data: 'CAPTURE'}"
```

Capture with potential-fields vision:

```bash
ros2 topic pub --once /potentialfields/cmd std_msgs/msg/String "{data: 'CAPTURE'}"
```

Compute eye-in-hand transform:

```bash
ros2 topic pub --once /eyeinhand/cmd std_msgs/msg/String "{data: 'COMPUTE'}"
```

Plan Mode 1 trajectory:

```bash
ros2 topic pub --once /trajectory/cmd std_msgs/msg/String "{data: 'PLAN'}"
```

Execute streamed control:

```bash
ros2 topic pub --once /control/cmd std_msgs/msg/String "{data: 'EXECUTE_STREAM'}"
```

Send manual AGV RPM:

```bash
ros2 topic pub --once /agv/rpm_cmd std_msgs/msg/String "{data: '100,100'}"
```

Stop AGV:

```bash
ros2 topic pub --once /agv/rpm_cmd std_msgs/msg/String "{data: 's'}"
```

Set AGV automatic mode:

```bash
ros2 topic pub --once /agv/control_mode std_msgs/msg/String "{data: 'automatic'}"
```

Set AGV manual mode:

```bash
ros2 topic pub --once /agv/control_mode std_msgs/msg/String "{data: 'manual'}"
```

Set AGV reference side:

```bash
ros2 topic pub --once /agv/reference_side std_msgs/msg/String "{data: 'left'}"
ros2 topic pub --once /agv/reference_side std_msgs/msg/String "{data: 'right'}"
ros2 topic pub --once /agv/reference_side std_msgs/msg/String "{data: 'auto'}"
```

Set harvesting target side:

```bash
ros2 topic pub --once /searching_mode/target_side std_msgs/msg/String "{data: 'left'}"
ros2 topic pub --once /searching_mode/target_side std_msgs/msg/String "{data: 'right'}"
ros2 topic pub --once /searching_mode/target_side std_msgs/msg/String "{data: 'any'}"
```

Enable or disable auto-harvest:

```bash
ros2 topic pub --once /searching_mode/auto_harvest std_msgs/msg/Bool "{data: true}"
ros2 topic pub --once /searching_mode/auto_harvest std_msgs/msg/Bool "{data: false}"
```

## Important Launch Parameters

### `harvesting_system.launch.py`

| Parameter | Default | Purpose |
| --- | --- | --- |
| `enable_mode2` | `true` | Enables Mode 2 HyRRT planning. Requires custom OMPL. |
| `enable_gripper` | `false` | Enables or disables the optional gripper integration. |
| `mode1_use_pf_vision` | `false` | Uses potential-fields YOLO `.pt` vision for Mode 1 instead of `mode1_vision_node`. |
| `do_home_on_start` | `true` | Sends the robot to the configured home pose when a `START` command is received. |
| `enable_searching_mode` | `false` | Starts `searching_mode_node`. |
| `enable_obb_pca` | `false` | Starts `obb_pca_vision_node`. |
| `trigger_obb_pca_at_final_vision` | `false` | Triggers OBB/PCA capture near the final visual stage. |
| `controller_topic` | `/joint_trajectory_controller/joint_trajectory` | Kinova joint trajectory command topic. |
| `joint_state_topic` | `/joint_states` | Joint state input topic. |
| `base_frame` | `base_link` | Kinova base frame. |
| `ee_link` | `tool_frame` | End-effector link used by control and planning. |
| `urdf_path` | `gen3.urdf` | URDF used by trajectory and control nodes. |
| `lock_joint_6` | `true` | Locks joint 6 in the controller. |
| `projection_distance_m` | `0.25` | Mode 1 approach/projection distance. |
| `master_stop_distance_m` | `0.25` | Stop distance used by high-level coordination. |
| `search_yolo_device` | `0` | Ultralytics device. Use `cpu` on machines without GPU support. |
| `search_conf_thresh` | `0.6` | YOLO confidence threshold. |
| `search_target_class_id` | `2` | Target class index used by the detection models. |

Use CPU for YOLO:

```bash
ros2 launch harvesting_robot harvesting_system.launch.py search_yolo_device:=cpu
```

Disable automatic homing:

```bash
ros2 launch harvesting_robot harvesting_system.launch.py do_home_on_start:=false
```

Disable Mode 2:

```bash
ros2 launch harvesting_robot harvesting_system.launch.py enable_mode2:=false
```

### `eyeinhandkinova.launch.py`

| Parameter | Default |
| --- | --- |
| `cam_x` | `-0.04439` |
| `cam_y` | `0.00039` |
| `cam_z` | `0.04472` |
| `cam_roll` | `0.0` |
| `cam_pitch` | `-1.57079632679` |
| `cam_yaw` | `6.0` |
| `eye_x_offset_m` | `-0.044` |
| `eye_y_offset_m` | `0.00` |
| `eye_z_offset_m` | `0.04` |

These values define the camera mount transform and final target offsets.

### `agv_web_lidar.launch.py`

| Parameter | Default | Purpose |
| --- | --- | --- |
| `serial_port` | `/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0` | RPLIDAR serial port. |
| `serial_baudrate` | `115200` | RPLIDAR A1 baudrate. |
| `web_host` | `0.0.0.0` | Web server host. |
| `web_port` | `8080` | Web server port. |
| `show_radar_window` | `false` | Opens a local OpenCV radar window. |
| `target_horizontal_line_distance_m` | `2.0` | Target distance from the detected row/bush line. |
| `reference_side` | `auto` | AGV reference side: `auto`, `left`, or `right`. |
| `target_side_mode` | `any` | Accepted cobot target side: `any`, `left`, or `right`. |
| `auto_harvest_enabled` | `true` | Allows searching mode to trigger harvesting automatically. |

## Eye-in-Hand Calibration

The Kinova eye-in-hand launch publishes the following static transforms:

```text
end_effector_link -> tool0
tool0 -> camera_link
camera_link -> camera_color_optical_frame
```

The `eyeinhand_node` subscribes to:

```text
/camera_sphere
```

and publishes:

```text
/target_base
```

The target frame is:

```text
base_link
```

To tune the camera mount transform at launch time:

```bash
ros2 launch harvesting_robot harvesting_system.launch.py \
  cam_x:=-0.04439 \
  cam_y:=0.00039 \
  cam_z:=0.04472 \
  cam_roll:=0.0 \
  cam_pitch:=-1.57079632679 \
  cam_yaw:=6.0
```

If the robot consistently aims offset from the berry, check:

- `cam_x`, `cam_y`, `cam_z`
- `cam_roll`, `cam_pitch`, `cam_yaw`
- `eye_x_offset_m`, `eye_y_offset_m`, `eye_z_offset_m`
- RealSense camera intrinsics on `/camera/color/camera_info`
- TF availability between `camera_color_optical_frame` and `base_link`

Generate a TF tree:

```bash
ros2 run tf2_tools view_frames
```

Expected frames:

```text
base_link
end_effector_link
tool0
camera_link
camera_color_optical_frame
```

## AGV and LiDAR Subsystem

Start the AGV subsystem:

```bash
ros2 launch harvesting_robot_agv agv_web_lidar.launch.py
```

This launch file starts:

- `sllidar_ros2/sllidar_a1_launch.py`
- `lidar_360_radar_node`
- `agv_position_control_node`
- `ble_agv_bridge_node`
- `agv_web_control_node`

### Test LiDAR Only

```bash
ros2 launch sllidar_ros2 view_sllidar_a1_launch.py
```

If the serial device has permission issues, install the udev rules from the SLLIDAR package:

```bash
cd ~/harvesting_robot_ws/src/sllidar_ros2
source scripts/create_udev_rules.sh
```

Temporary permission test:

```bash
sudo chmod 777 /dev/ttyUSB0
```

Launch with an explicit serial port:

```bash
ros2 launch harvesting_robot_agv agv_web_lidar.launch.py serial_port:=/dev/ttyUSB0
```

### BLE AGV Defaults

`ble_agv_bridge_node` defaults:

```text
device_name: ESP32S3_BLE
device_address: 48:27:E2:16:D6:61
write_uuid: 6E400002-B5A3-F393-E0A9-E50E24DCCA9E
read_uuid: 6E400003-B5A3-F393-E0A9-E50E24DCCA9E
command_topic: /agv/rpm_cmd
feedback_topic: /agv/ble_feedback
```

If the ESP32-S3 MAC address changes, launch with a new `device_address` or configure the node to find the device by name.

## Mode 2 and Custom OMPL

Mode 2 starts:

```text
harvesting_robot_cpp/mode2_trajectory_node
```

It uses:

- Custom OMPL with `HyRRT`
- `HybridStateSpace`
- KDL kinematics
- `gen3.urdf`
- Target topic `/target_base`

Useful Mode 2 parameters:

| Parameter | Purpose |
| --- | --- |
| `enable_mode2` | Enables or disables Mode 2. |
| `hyrrt_planning_time` | Maximum planning time. |
| `hyrrt_max_cartesian_vel` | Maximum Cartesian velocity. |
| `hyrrt_flow_step` | Flow/integration step. |
| `hyrrt_waypoint_dt` | Time step between waypoints. |
| `hyrrt_goal_tol_m` | Goal tolerance. |
| `hyrrt_ws_min_x/y/z` | Workspace lower bounds. |
| `hyrrt_ws_max_x/y/z` | Workspace upper bounds. |
| `hyrrt_clamp_to_workspace` | Clamps targets to the configured workspace. |

Disable Mode 2:

```bash
ros2 launch harvesting_robot harvesting_system.launch.py enable_mode2:=false
```

Enable the optional gripper integration:

```bash
ros2 launch harvesting_robot harvesting_system.launch.py enable_gripper:=true
```

Disable the optional gripper integration:

```bash
ros2 launch harvesting_robot harvesting_system.launch.py enable_gripper:=false
```

## Troubleshooting

### ROS Cannot Find Packages

Source the expected workspaces:

```bash
source /opt/ros/humble/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
source ~/harvesting_robot_ws/install/setup.bash
```

If Mode 2 is enabled:

```bash
source ~/workspace/ompl_ws/install/setup.bash
```

### `/joint_states` Is Missing

Check that Kortex is running:

```bash
ros2 topic list | grep joint_states
```

Start Kortex:

```bash
ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10
```

### Robot Does Not Move

Check:

```bash
ros2 topic echo /joint_states
ros2 topic echo /joint_trajectory_controller/joint_trajectory
ros2 topic echo /control/status
```

Also verify:

- The Kinova arm is not in fault.
- The Kortex driver is connected to the correct robot IP.
- The controller topic matches `controller_topic`.
- The emergency stop is released.

### RealSense Does Not Start

Check Python wrapper:

```bash
python3 -c "import pyrealsense2 as rs; print(rs.__version__)"
```

If this fails, reinstall `pyrealsense2` or follow the `librealsense` installation guide.

### `harvesting_robot_cpp` Fails To Build

Most likely causes:

- Custom OMPL is not built.
- `ompl_ws` was not sourced.
- `harvesting_robot_cpp/CMakeLists.txt` points to paths that do not exist on the current PC.

Build without Mode 2:

```bash
colcon build --symlink-install --packages-skip harvesting_robot_cpp
```

Launch with:

```bash
enable_mode2:=false
```

### LiDAR Does Not Publish `/scan`

Check serial devices:

```bash
ls /dev/ttyUSB*
ls /dev/serial/by-id/
```

Launch with a specific port:

```bash
ros2 launch harvesting_robot_agv agv_web_lidar.launch.py serial_port:=/dev/ttyUSB0
```

### AGV Does Not Respond

Check BLE feedback and commands:

```bash
ros2 topic echo /agv/ble_feedback
ros2 topic echo /agv/rpm_cmd
```

Verify:

- ESP32-S3 is powered.
- BLE address or name matches the launch configuration.
- RPM command format is valid.
- Values are between `0` and `150`, or command is `s`.

### Missing TF Between Camera and Robot

Generate the TF tree:

```bash
ros2 run tf2_tools view_frames
```

Expected frames:

```text
base_link
end_effector_link
tool0
camera_link
camera_color_optical_frame
```

## Development Notes

- Clone this project inside a ROS 2 workspace under `src/`.
- Do not commit generated ROS directories such as `build/`, `install/`, or `log/`.
- Keep model files in `harvesting_robot/models/`.
- If the camera mount changes, update `eyeinhandkinova.launch.py` values.
- If the Kinova controller changes, update `controller_topic` and `joint_state_topic`.
- If AGV operation is not required, use `enable_searching_mode:=false`.
- If custom OMPL is not available, build without `harvesting_robot_cpp` and use `enable_mode2:=false`.
- For first tests on the real arm, start with Mode 2 disabled and searching mode disabled.
- Keep the physical emergency stop accessible during all robot motion tests.

## Repository Structure

```text
berry_harvesting_robot/
  harvesting_robot/
    harvesting_robot/
      camera_node.py
      control_node.py
      eyeinhand_node.py
      master_node.py
      mode1_trajectory_node.py
      mode1_vision_node.py
      mode2_vision_node.py
      obb_pca_vision_node.py
      searching_mode_node.py
    launch/
      harvesting_system.launch.py
      eyeinhandkinova.launch.py
      eyeinhand.launch.py
    models/
    urdf/
      gen3.urdf
      elfin3.urdf
  harvesting_robot_agv/
    harvesting_robot_agv/
      agv_position_control_node.py
      agv_web_control_node.py
      ble_agv_bridge_node.py
      lidar_360_radar_node.py
      agv_gui_node.py
    launch/
      agv_web_lidar.launch.py
  harvesting_robot_cpp/
    src/
      mode2_trajectory_node.cpp
  robot_description/
  docs/
```

## References

- ROS 2 Humble: https://docs.ros.org/en/humble/Installation.html
- Kinova ROS 2 Kortex: https://github.com/Kinovarobotics/ros2_kortex/tree/humble
- Intel RealSense librealsense: https://github.com/realsenseai/librealsense
- pyrealsense2: https://github.com/realsenseai/librealsense/blob/master/wrappers/python/readme.md
- RealSense ROS wrapper: https://github.com/realsenseai/realsense-ros
- SLLIDAR ROS 2: https://github.com/Slamtec/sllidar_ros2
- Custom OMPL fork for HyRRT: https://github.com/xu21beve/ompl
- OMPL documentation: https://ompl.kavrakilab.org/core/installation.html
