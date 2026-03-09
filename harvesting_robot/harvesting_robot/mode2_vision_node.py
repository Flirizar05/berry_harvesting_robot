#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, String


def normalize_command(command: str) -> str:
    """Normalize text commands received from ROS topics."""
    return (command or "").strip().upper()


class Mode2VisionNode(Node):
    """Process shared camera topics and publish a target point for Mode 2."""

    def __init__(self) -> None:
        super().__init__("mode2_vision_node")

        self._declare_parameters()
        self._load_parameters()
        self._validate_model_files()
        self._initialize_visualization()
        self._load_yolo_model()
        self._create_ros_interfaces()

        self.bridge = CvBridge()

        self.latest_color_image = None
        self.latest_depth_image = None
        self.has_color_image = False
        self.has_depth_image = False

        self.fx = None
        self.fy = None
        self.ppx = None
        self.ppy = None
        self.camera_frame_id = "camera_color_optical_frame"
        self.has_camera_info = False

        self.is_busy = False
        self.search_start_time = None

        self.timer = self.create_timer(self.timer_period_sec, self._timer_callback)

        self.status_publisher.publish(String(data="IDLE"))
        self.get_logger().info(
            f"mode2_vision_node ready. "
            f"cmd={self.command_topic}, status={self.status_topic}, "
            f"color={self.color_topic}, depth={self.depth_topic}, info={self.camera_info_topic}"
        )

    def _declare_parameters(self) -> None:
        share_dir = get_package_share_directory("harvesting_robot")
        models_dir = os.path.join(share_dir, "models")

        self.declare_parameter(
            "cfg_path",
            os.path.join(models_dir, "yolov4-tiny-custom.cfg"),
        )
        self.declare_parameter(
            "weights_path",
            os.path.join(models_dir, "yolov4-tiny-custom_best.weights"),
        )
        self.declare_parameter(
            "names_path",
            os.path.join(models_dir, "blackberry.names"),
        )
        self.declare_parameter("conf_thresh", 0.6)
        self.declare_parameter("nms_thresh", 0.4)
        self.declare_parameter("target_class_id", 2)

        self.declare_parameter("cmd_topic", "/potentialfields/cmd")
        self.declare_parameter("status_topic", "/potentialfields/status")
        self.declare_parameter("output_point_topic", "/camera_sphere")
        self.declare_parameter("output_radius_topic", "/sphere_radius")

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")

        self.declare_parameter("depth_scale_topic", "/camera/depth_scale")
        self.declare_parameter("depth_scale_fallback", 0.001)

        self.declare_parameter("search_timeout_sec", 2.0)
        self.declare_parameter("timer_period", 0.02)

        self.declare_parameter("min_valid_depth_m", 0.10)
        self.declare_parameter("max_valid_depth_m", 2.00)
        self.declare_parameter("min_depth_samples", 10)
        self.declare_parameter("depth_grid_div", 10)
        self.declare_parameter("fallback_expand_px", 40)
        self.declare_parameter("center_patch_halfwin", 16)

        self.declare_parameter("show_result", True)
        self.declare_parameter("result_window_name", "PF Capture Result")

        self.declare_parameter("publish_annotated", False)
        self.declare_parameter("annotated_topic", "/potentialfields/annotated")

        self.declare_parameter("publish_legacy_xyzr", False)
        self.declare_parameter("pub_x", "/blackberry/x")
        self.declare_parameter("pub_y", "/blackberry/y")
        self.declare_parameter("pub_z", "/blackberry/z")
        self.declare_parameter("pub_r", "/blackberry/r")

    def _load_parameters(self) -> None:
        self.config_path = os.path.expanduser(
            str(self.get_parameter("cfg_path").value or "")
        )
        self.weights_path = os.path.expanduser(
            str(self.get_parameter("weights_path").value or "")
        )
        self.class_names_path = os.path.expanduser(
            str(self.get_parameter("names_path").value or "")
        )

        self.confidence_threshold = float(self.get_parameter("conf_thresh").value)
        self.nms_threshold = float(self.get_parameter("nms_thresh").value)
        self.target_class_id = int(self.get_parameter("target_class_id").value)

        self.command_topic = str(self.get_parameter("cmd_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.output_point_topic = str(self.get_parameter("output_point_topic").value)
        self.output_radius_topic = str(self.get_parameter("output_radius_topic").value)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)

        self.depth_scale_topic = str(self.get_parameter("depth_scale_topic").value)
        self.depth_scale_m_per_unit = float(
            self.get_parameter("depth_scale_fallback").value
        )

        self.search_timeout_sec = float(
            self.get_parameter("search_timeout_sec").value
        )
        self.timer_period_sec = float(self.get_parameter("timer_period").value)

        self.min_valid_depth_m = float(
            self.get_parameter("min_valid_depth_m").value
        )
        self.max_valid_depth_m = float(
            self.get_parameter("max_valid_depth_m").value
        )
        self.min_depth_samples = int(self.get_parameter("min_depth_samples").value)
        self.depth_grid_div = int(self.get_parameter("depth_grid_div").value)
        self.fallback_expand_px = int(
            self.get_parameter("fallback_expand_px").value
        )
        self.center_patch_half_window = int(
            self.get_parameter("center_patch_halfwin").value
        )

        self.show_result = bool(self.get_parameter("show_result").value)
        self.result_window_name = str(
            self.get_parameter("result_window_name").value
        )

        self.publish_annotated = bool(
            self.get_parameter("publish_annotated").value
        )
        self.annotated_topic = str(self.get_parameter("annotated_topic").value)

        self.publish_legacy_xyzr = bool(
            self.get_parameter("publish_legacy_xyzr").value
        )
        self.legacy_x_topic = str(self.get_parameter("pub_x").value)
        self.legacy_y_topic = str(self.get_parameter("pub_y").value)
        self.legacy_z_topic = str(self.get_parameter("pub_z").value)
        self.legacy_r_topic = str(self.get_parameter("pub_r").value)

    def _validate_model_files(self) -> None:
        required_files = [
            (self.class_names_path, "names"),
            (self.weights_path, "weights"),
            (self.config_path, "cfg"),
        ]

        for file_path, label in required_files:
            if not file_path or not os.path.isfile(file_path):
                self.get_logger().error(f"Missing YOLO {label} file: '{file_path}'")
                raise FileNotFoundError(file_path)

        self.get_logger().info(
            "Using YOLO model files:\n"
            f"  names:   {self.class_names_path}\n"
            f"  weights: {self.weights_path}\n"
            f"  cfg:     {self.config_path}\n"
            f"  depth_scale_topic: {self.depth_scale_topic} "
            f"(fallback={self.depth_scale_m_per_unit})"
        )

    def _initialize_visualization(self) -> None:
        if not self.show_result:
            return

        try:
            cv2.namedWindow(self.result_window_name, cv2.WINDOW_NORMAL)
        except Exception as exc:
            self.get_logger().warn(f"Failed to create OpenCV window: {exc}")
            self.show_result = False

    def _load_yolo_model(self) -> None:
        with open(self.class_names_path, "r", encoding="utf-8") as file:
            self.class_names = [line.strip() for line in file.readlines()]

        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        unconnected_layers = np.array(self.net.getUnconnectedOutLayers()).flatten()
        self.output_layers = [layer_names[int(layer_id) - 1] for layer_id in unconnected_layers]

    def _create_ros_interfaces(self) -> None:
        self.status_publisher = self.create_publisher(String, self.status_topic, 10)
        self.command_subscription = self.create_subscription(
            String,
            self.command_topic,
            self._on_command,
            10,
        )

        self.point_publisher = self.create_publisher(
            PointStamped,
            self.output_point_topic,
            10,
        )
        self.radius_publisher = self.create_publisher(
            Float32,
            self.output_radius_topic,
            10,
        )

        self.color_subscription = self.create_subscription(
            Image,
            self.color_topic,
            self._on_color_image,
            10,
        )
        self.depth_subscription = self.create_subscription(
            Image,
            self.depth_topic,
            self._on_depth_image,
            10,
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._on_camera_info,
            10,
        )
        self.depth_scale_subscription = self.create_subscription(
            Float32,
            self.depth_scale_topic,
            self._on_depth_scale,
            10,
        )

        self.annotated_image_publisher = (
            self.create_publisher(Image, self.annotated_topic, 10)
            if self.publish_annotated
            else None
        )

        if self.publish_legacy_xyzr:
            self.legacy_x_publisher = self.create_publisher(
                String,
                self.legacy_x_topic,
                10,
            )
            self.legacy_y_publisher = self.create_publisher(
                String,
                self.legacy_y_topic,
                10,
            )
            self.legacy_z_publisher = self.create_publisher(
                String,
                self.legacy_z_topic,
                10,
            )
            self.legacy_r_publisher = self.create_publisher(
                String,
                self.legacy_r_topic,
                10,
            )
        else:
            self.legacy_x_publisher = None
            self.legacy_y_publisher = None
            self.legacy_z_publisher = None
            self.legacy_r_publisher = None

    def _on_depth_scale(self, msg: Float32) -> None:
        try:
            depth_scale = float(msg.data)
            if np.isfinite(depth_scale) and depth_scale > 0.0:
                self.depth_scale_m_per_unit = depth_scale
        except Exception:
            pass

    def _on_color_image(self, msg: Image) -> None:
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
            self.has_color_image = True
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert color image: {exc}")

    def _on_depth_image(self, msg: Image) -> None:
        try:
            if msg.encoding in ("16UC1", "mono16"):
                depth_image = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="passthrough",
                )
                self.latest_depth_image = depth_image.astype(np.uint16)
            elif msg.encoding == "32FC1":
                depth_image = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="passthrough",
                )
                self.latest_depth_image = depth_image.astype(np.float32)
            else:
                depth_image = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="passthrough",
                )
                self.latest_depth_image = np.asarray(depth_image)

            self.has_depth_image = True
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert depth image: {exc}")

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if msg.k is not None and len(msg.k) == 9:
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.ppx = float(msg.k[2])
            self.ppy = float(msg.k[5])
            self.has_camera_info = True

        if msg.header.frame_id:
            self.camera_frame_id = msg.header.frame_id

    def _on_command(self, msg: String) -> None:
        command = normalize_command(msg.data)
        if command != "CAPTURE":
            return

        if self.is_busy:
            self.get_logger().info("CAPTURE command ignored because the node is busy.")
            return

        self.is_busy = True
        self.search_start_time = self.get_clock().now()
        self.status_publisher.publish(String(data="BUSY"))
        self.get_logger().info("CAPTURE command received. Entering BUSY state.")

    def _timer_callback(self) -> None:
        if self.show_result:
            try:
                cv2.waitKey(1)
            except Exception:
                pass

        if not self.is_busy:
            return

        if not (
            self.has_color_image
            and self.has_depth_image
            and self.has_camera_info
        ):
            if self._timed_out():
                self._finish_with_failure("No camera data available yet.", None)
            return

        color_image = self.latest_color_image
        depth_image = self.latest_depth_image

        if color_image is None or depth_image is None:
            if self._timed_out():
                self._finish_with_failure("Missing cached frames.", None)
            return

        annotated_image = color_image.copy()
        detection = self._detect_best_target(color_image, annotated_image)

        if detection is None:
            if self._timed_out():
                self._finish_with_failure("No target detected.", annotated_image)
            return

        x, y, box_width, box_height, center_u, center_v, radius_px, score, class_id = detection

        depth_m = self._depth_from_bbox(depth_image, x, y, box_width, box_height)
        if depth_m is None:
            depth_m = self._depth_from_bbox(
                depth_image,
                x - self.fallback_expand_px,
                y - self.fallback_expand_px,
                box_width + 2 * self.fallback_expand_px,
                box_height + 2 * self.fallback_expand_px,
            )
        if depth_m is None:
            depth_m = self._depth_from_center_patch(
                depth_image,
                int(center_u),
                int(center_v),
                self.center_patch_half_window,
            )

        if depth_m is None:
            if self._timed_out():
                self._finish_with_failure("No valid depth found for the target.", annotated_image)
            return

        if not (self.min_valid_depth_m <= depth_m <= self.max_valid_depth_m):
            if self._timed_out():
                self._finish_with_failure(
                    f"Depth out of range: Z={depth_m:.3f} m.",
                    annotated_image,
                )
            return

        target_x_m = (float(center_u) - self.ppx) * depth_m / self.fx
        target_y_m = (float(center_v) - self.ppy) * depth_m / self.fy
        radius_m = float(max(0.0, float(radius_px) * depth_m / float(self.fx)))

        target_msg = PointStamped()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = self.camera_frame_id or "camera_color_optical_frame"
        target_msg.point.x = float(target_x_m)
        target_msg.point.y = float(target_y_m)
        target_msg.point.z = float(depth_m)
        self.point_publisher.publish(target_msg)

        self.radius_publisher.publish(Float32(data=float(radius_m)))

        cv2.rectangle(
            annotated_image,
            (int(x), int(y)),
            (int(x + box_width), int(y + box_height)),
            (255, 0, 255),
            2,
        )
        cv2.circle(
            annotated_image,
            (int(center_u), int(center_v)),
            int(max(2, radius_px)),
            (0, 0, 255),
            3,
        )
        cv2.putText(
            annotated_image,
            f"Target Z={depth_m:.3f}m score={score:.2f} scale={self.depth_scale_m_per_unit:.6f}",
            (int(x), max(0, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        if self.annotated_image_publisher is not None:
            try:
                self.annotated_image_publisher.publish(
                    self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
                )
            except Exception:
                pass

        if self.publish_legacy_xyzr and self.legacy_x_publisher is not None:
            self.legacy_x_publisher.publish(String(data=json.dumps(float(center_u))))
            self.legacy_y_publisher.publish(String(data=json.dumps(float(center_v))))
            self.legacy_z_publisher.publish(String(data=json.dumps(float(depth_m))))
            self.legacy_r_publisher.publish(String(data=json.dumps(float(radius_px))))

        self.get_logger().info(
            f"DONE_OK u={center_u:.1f} v={center_v:.1f} Z={depth_m:.3f} "
            f"XYZ=({target_x_m:.3f},{target_y_m:.3f},{depth_m:.3f}) "
            f"r_px={radius_px:.1f} r_m={radius_m:.3f} "
            f"depth_scale={self.depth_scale_m_per_unit:.9f} "
            f"frame_id={target_msg.header.frame_id}"
        )

        self.status_publisher.publish(String(data="DONE_OK"))
        if self.show_result:
            try:
                cv2.imshow(self.result_window_name, annotated_image)
            except Exception:
                pass

        self.is_busy = False
        self.status_publisher.publish(String(data="IDLE"))

    def _detect_best_target(
        self,
        color_image: np.ndarray,
        annotated_image: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int, float, float, float, float, int]]:
        image_height, image_width = color_image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            color_image,
            1.0 / 255.0,
            (416, 416),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < self.confidence_threshold:
                    continue

                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                box_width = int(detection[2] * image_width)
                box_height = int(detection[3] * image_height)
                top_left_x = int(center_x - box_width / 2)
                top_left_y = int(center_y - box_height / 2)

                boxes.append([top_left_x, top_left_y, box_width, box_height])
                confidences.append(confidence)
                class_ids.append(class_id)

        if not boxes:
            return None

        kept_indexes = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_threshold,
        )
        if kept_indexes is None or len(kept_indexes) == 0:
            return None

        kept_indexes = np.array(kept_indexes).flatten()

        best_target = None
        best_any_detection = None

        for index in kept_indexes:
            x, y, box_width, box_height = boxes[index]
            class_id = class_ids[index]
            confidence = float(confidences[index])

            label = (
                self.class_names[class_id]
                if 0 <= class_id < len(self.class_names)
                else str(class_id)
            )

            box_color = (
                (255, 0, 255)
                if class_id == self.target_class_id
                else (255, 255, 180)
            )

            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(image_width - 1, x + box_width)
            y1 = min(image_height - 1, y + box_height)

            cv2.rectangle(annotated_image, (x0, y0), (x1, y1), box_color, 2)
            cv2.putText(
                annotated_image,
                f"{label} {confidence:.2f}",
                (x0, max(0, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            if best_any_detection is None or confidence > best_any_detection[0]:
                best_any_detection = (confidence, x, y, box_width, box_height, class_id)

            if class_id == self.target_class_id:
                if best_target is None or confidence > best_target[0]:
                    best_target = (confidence, x, y, box_width, box_height, class_id)

        selected_detection = best_target if best_target is not None else best_any_detection
        if selected_detection is None:
            return None

        confidence, x, y, box_width, box_height, class_id = selected_detection
        center_u = float(x + box_width // 2)
        center_v = float(y + box_height // 2)
        radius_px = float(max(box_width, box_height) / 2.0)

        return (
            int(x),
            int(y),
            int(box_width),
            int(box_height),
            center_u,
            center_v,
            radius_px,
            float(confidence),
            int(class_id),
        )

    def _depth_to_meters(self, depth_value) -> float:
        if (
            isinstance(self.latest_depth_image, np.ndarray)
            and self.latest_depth_image.dtype == np.float32
        ):
            return float(depth_value)

        return float(depth_value) * float(self.depth_scale_m_per_unit)

    def _depth_from_bbox(
        self,
        depth_image: np.ndarray,
        x: int,
        y: int,
        box_width: int,
        box_height: int,
    ) -> Optional[float]:
        image_height, image_width = depth_image.shape[:2]

        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(image_width - 1, int(x + box_width))
        y1 = min(image_height - 1, int(y + box_height))

        if x1 <= x0 or y1 <= y0:
            return None

        step_x = max(1, (x1 - x0) // max(1, self.depth_grid_div))
        step_y = max(1, (y1 - y0) // max(1, self.depth_grid_div))

        valid_depths_m = []

        for row in range(y0, y1, step_y):
            for col in range(x0, x1, step_x):
                depth_value = depth_image[row, col]
                if depth_value is None:
                    continue

                scalar_depth_value = (
                    float(np.asarray(depth_value).item())
                    if not np.isscalar(depth_value)
                    else float(depth_value)
                )
                if scalar_depth_value <= 0.0:
                    continue

                depth_m = self._depth_to_meters(scalar_depth_value)
                if self.min_valid_depth_m <= depth_m <= self.max_valid_depth_m:
                    valid_depths_m.append(depth_m)

        if len(valid_depths_m) < self.min_depth_samples:
            return None

        sorted_depths = np.sort(np.array(valid_depths_m, dtype=np.float32))
        keep_count = max(1, int(0.7 * len(sorted_depths)))
        return float(np.median(sorted_depths[:keep_count]))

    def _depth_from_center_patch(
        self,
        depth_image: np.ndarray,
        center_u: int,
        center_v: int,
        half_window: int,
    ) -> Optional[float]:
        image_height, image_width = depth_image.shape[:2]

        x0 = max(0, center_u - half_window)
        x1 = min(image_width - 1, center_u + half_window)
        y0 = max(0, center_v - half_window)
        y1 = min(image_height - 1, center_v + half_window)

        if x1 <= x0 or y1 <= y0:
            return None

        patch = depth_image[y0:y1, x0:x1]
        if patch.size < 1:
            return None

        valid_depths_m = []

        for depth_value in patch.reshape(-1):
            scalar_depth_value = float(depth_value)
            if scalar_depth_value <= 0.0:
                continue

            depth_m = self._depth_to_meters(scalar_depth_value)
            if self.min_valid_depth_m <= depth_m <= self.max_valid_depth_m:
                valid_depths_m.append(depth_m)

        if len(valid_depths_m) < max(5, self.min_depth_samples // 2):
            return None

        sorted_depths = np.sort(np.array(valid_depths_m, dtype=np.float32))
        keep_count = max(1, int(0.7 * len(sorted_depths)))
        return float(np.median(sorted_depths[:keep_count]))

    def _timed_out(self) -> bool:
        if self.search_start_time is None:
            return False

        now = self.get_clock().now()
        elapsed_sec = (now - self.search_start_time).nanoseconds * 1e-9
        return elapsed_sec >= self.search_timeout_sec

    def _finish_with_failure(
        self,
        reason: str,
        annotated_image: Optional[np.ndarray],
    ) -> None:
        self.get_logger().warn(f"DONE_FAIL: {reason}")
        self.status_publisher.publish(String(data="DONE_FAIL"))

        if self.show_result and annotated_image is not None:
            try:
                cv2.imshow(self.result_window_name, annotated_image)
            except Exception:
                pass

        self.is_busy = False
        self.status_publisher.publish(String(data="IDLE"))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Mode2VisionNode()
    try:
        rclpy.spin(node)
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()