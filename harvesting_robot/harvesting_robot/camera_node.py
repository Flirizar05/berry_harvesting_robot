#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import numpy as np
import pyrealsense2 as rs
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32, Header, String


def pack_rgb_float(red: int, green: int, blue: int) -> float:
    """Pack RGB values into the float representation expected by PointCloud2."""
    rgb_uint32 = (int(red) << 16) | (int(green) << 8) | int(blue)
    return np.frombuffer(np.uint32(rgb_uint32).tobytes(), dtype=np.float32)[0]


class RealSenseCameraNode(Node):
    """Publish RealSense color, aligned depth, camera info, and optional debug point clouds."""

    def __init__(self) -> None:
        super().__init__("camera_node")

        self._declare_parameters()
        self._load_parameters()
        self._create_publishers()

        self.bridge = CvBridge()

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.depth,
            self.depth_width,
            self.depth_height,
            rs.format.z16,
            self.fps,
        )
        self.config.enable_stream(
            rs.stream.color,
            self.color_width,
            self.color_height,
            rs.format.bgr8,
            self.fps,
        )

        # Align depth to the color stream so both images share the same pixel reference.
        self.align = rs.align(rs.stream.color)

        self.color_intrinsics = None
        self.depth_scale = None

        self._start_pipeline()
        self.timer = self.create_timer(self.publish_period_sec, self._publish_frame_set)

    def _declare_parameters(self) -> None:
        self.declare_parameter("depth_width", 640)
        self.declare_parameter("depth_height", 480)
        self.declare_parameter("color_width", 640)
        self.declare_parameter("color_height", 480)
        self.declare_parameter("fps", 30)

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth/image_raw")
        self.declare_parameter("color_info_topic", "/camera/color/camera_info")
        self.declare_parameter("depth_info_topic", "/camera/aligned_depth/camera_info")
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("publish_rate_hz", 30.0)

        self.declare_parameter("publish_pointcloud", True)
        self.declare_parameter("pointcloud_topic", "/camera/depth/points")
        self.declare_parameter("pc_stride", 2)
        self.declare_parameter("pc_max_depth_m", 2.5)
        self.declare_parameter("pc_min_depth_m", 0.10)

        self.declare_parameter("depth_scale_topic", "/camera/depth_scale")

    def _load_parameters(self) -> None:
        self.depth_width = int(self.get_parameter("depth_width").value)
        self.depth_height = int(self.get_parameter("depth_height").value)
        self.color_width = int(self.get_parameter("color_width").value)
        self.color_height = int(self.get_parameter("color_height").value)
        self.fps = int(self.get_parameter("fps").value)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.color_info_topic = str(self.get_parameter("color_info_topic").value)
        self.depth_info_topic = str(self.get_parameter("depth_info_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.publish_period_sec = 1.0 / max(1e-3, publish_rate_hz)

        self.publish_pointcloud = bool(self.get_parameter("publish_pointcloud").value)
        self.pointcloud_topic = str(self.get_parameter("pointcloud_topic").value)
        self.pointcloud_stride = max(1, int(self.get_parameter("pc_stride").value))
        self.pointcloud_max_depth_m = float(self.get_parameter("pc_max_depth_m").value)
        self.pointcloud_min_depth_m = float(self.get_parameter("pc_min_depth_m").value)

        self.depth_scale_topic = str(self.get_parameter("depth_scale_topic").value)

    def _create_publishers(self) -> None:
        self.color_publisher = self.create_publisher(Image, self.color_topic, 10)
        self.depth_publisher = self.create_publisher(Image, self.depth_topic, 10)
        self.color_info_publisher = self.create_publisher(
            CameraInfo,
            self.color_info_topic,
            10,
        )
        self.depth_info_publisher = self.create_publisher(
            CameraInfo,
            self.depth_info_topic,
            10,
        )
        self.status_publisher = self.create_publisher(String, "/camera/status", 10)
        self.depth_scale_publisher = self.create_publisher(
            Float32,
            self.depth_scale_topic,
            10,
        )

        # The point cloud is only used for visualization and debugging.
        self.pointcloud_publisher = (
            self.create_publisher(PointCloud2, self.pointcloud_topic, 10)
            if self.publish_pointcloud
            else None
        )

    def _start_pipeline(self) -> None:
        try:
            profile = self.pipeline.start(self.config)
            self.get_logger().info("RealSense pipeline started.")

            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())
            self.get_logger().info(
                f"Depth scale: {self.depth_scale:.9f} m/unit "
                f"(topic: {self.depth_scale_topic})."
            )

            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.color_intrinsics = color_stream.get_intrinsics()
            self.get_logger().info(
                "Color intrinsics loaded: "
                f"fx={self.color_intrinsics.fx:.3f}, "
                f"fy={self.color_intrinsics.fy:.3f}, "
                f"ppx={self.color_intrinsics.ppx:.3f}, "
                f"ppy={self.color_intrinsics.ppy:.3f}, "
                f"width={self.color_intrinsics.width}, "
                f"height={self.color_intrinsics.height}."
            )

            self.status_publisher.publish(String(data="RUNNING"))
        except Exception as exc:
            self.get_logger().error(f"Failed to start RealSense pipeline: {exc}")
            self.status_publisher.publish(String(data="ERROR"))
            raise

    def _build_camera_info(self, intrinsics: rs.intrinsics) -> CameraInfo:
        camera_info = CameraInfo()
        camera_info.header.frame_id = self.frame_id
        camera_info.width = int(intrinsics.width)
        camera_info.height = int(intrinsics.height)

        camera_info.k = [
            float(intrinsics.fx), 0.0, float(intrinsics.ppx),
            0.0, float(intrinsics.fy), float(intrinsics.ppy),
            0.0, 0.0, 1.0,
        ]
        camera_info.p = [
            float(intrinsics.fx), 0.0, float(intrinsics.ppx), 0.0,
            0.0, float(intrinsics.fy), float(intrinsics.ppy), 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        camera_info.distortion_model = "plumb_bob"
        return camera_info

    def _build_colored_pointcloud(
        self,
        depth_image: np.ndarray,
        color_image: np.ndarray,
        stamp_msg,
    ) -> PointCloud2 | None:
        if self.color_intrinsics is None or self.depth_scale is None:
            return None

        fx = float(self.color_intrinsics.fx)
        fy = float(self.color_intrinsics.fy)
        ppx = float(self.color_intrinsics.ppx)
        ppy = float(self.color_intrinsics.ppy)

        image_height, image_width = depth_image.shape[:2]
        points = []

        for v in range(0, image_height, self.pointcloud_stride):
            for u in range(0, image_width, self.pointcloud_stride):
                depth_units = int(depth_image[v, u])
                if depth_units <= 0:
                    continue

                z_m = float(depth_units) * float(self.depth_scale)
                if (
                    z_m < self.pointcloud_min_depth_m
                    or z_m > self.pointcloud_max_depth_m
                ):
                    continue

                # Coordinates are computed in the optical frame convention.
                x_m = (float(u) - ppx) * z_m / fx
                y_m = (float(v) - ppy) * z_m / fy

                blue, green, red = color_image[v, u]
                rgb_float = pack_rgb_float(int(red), int(green), int(blue))
                points.append((float(x_m), float(y_m), float(z_m), float(rgb_float)))

        header = Header()
        header.stamp = stamp_msg
        header.frame_id = self.frame_id

        fields = [
            point_cloud2.PointField(
                name="x",
                offset=0,
                datatype=point_cloud2.PointField.FLOAT32,
                count=1,
            ),
            point_cloud2.PointField(
                name="y",
                offset=4,
                datatype=point_cloud2.PointField.FLOAT32,
                count=1,
            ),
            point_cloud2.PointField(
                name="z",
                offset=8,
                datatype=point_cloud2.PointField.FLOAT32,
                count=1,
            ),
            point_cloud2.PointField(
                name="rgb",
                offset=12,
                datatype=point_cloud2.PointField.FLOAT32,
                count=1,
            ),
        ]
        return point_cloud2.create_cloud(header, fields, points)

    def _publish_frame_set(self) -> None:
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                return

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            stamp_msg = self.get_clock().now().to_msg()

            # Consumers use this topic to convert raw depth units into meters.
            if self.depth_scale is not None:
                self.depth_scale_publisher.publish(
                    Float32(data=float(self.depth_scale))
                )

            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            color_msg.header.stamp = stamp_msg
            color_msg.header.frame_id = self.frame_id
            self.color_publisher.publish(color_msg)

            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
            depth_msg.header.stamp = stamp_msg
            depth_msg.header.frame_id = self.frame_id
            self.depth_publisher.publish(depth_msg)

            if self.color_intrinsics is not None:
                color_info_msg = self._build_camera_info(self.color_intrinsics)
                color_info_msg.header.stamp = stamp_msg
                self.color_info_publisher.publish(color_info_msg)

                depth_info_msg = self._build_camera_info(self.color_intrinsics)
                depth_info_msg.header.stamp = stamp_msg
                self.depth_info_publisher.publish(depth_info_msg)

            if self.pointcloud_publisher is not None and self.publish_pointcloud:
                pointcloud_msg = self._build_colored_pointcloud(
                    depth_image,
                    color_image,
                    stamp_msg,
                )
                if pointcloud_msg is not None:
                    self.pointcloud_publisher.publish(pointcloud_msg)

        except Exception as exc:
            self.get_logger().warn(f"Camera publish loop error: {exc}")
            self.status_publisher.publish(String(data="ERROR"))
            time.sleep(0.2)

    def destroy_node(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealSenseCameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()