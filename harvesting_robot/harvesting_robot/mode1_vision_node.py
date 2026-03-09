#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, String


def simple_dbscan(points, eps, min_samples):
    """Lightweight DBSCAN implementation for small point sets."""
    points_array = np.asarray(points, dtype=float)
    num_points = len(points_array)
    if num_points == 0:
        return np.array([], dtype=int)

    labels = -np.ones(num_points, dtype=int)
    visited = np.zeros(num_points, dtype=bool)
    cluster_id = 0
    eps_squared = eps * eps

    for index in range(num_points):
        if visited[index]:
            continue

        visited[index] = True

        diff = points_array - points_array[index]
        dist_squared = np.sum(diff * diff, axis=1)
        neighbors = np.where(dist_squared <= eps_squared)[0]

        if neighbors.size < min_samples:
            labels[index] = -1
            continue

        labels[neighbors] = cluster_id
        seed_queue = list(neighbors)

        while seed_queue:
            current_index = seed_queue.pop()
            if not visited[current_index]:
                visited[current_index] = True

                diff_current = points_array - points_array[current_index]
                dist_squared_current = np.sum(diff_current * diff_current, axis=1)
                current_neighbors = np.where(dist_squared_current <= eps_squared)[0]

                if current_neighbors.size >= min_samples:
                    for neighbor in current_neighbors:
                        if labels[neighbor] == -1:
                            labels[neighbor] = cluster_id
                        if labels[neighbor] == cluster_id and neighbor not in seed_queue:
                            seed_queue.append(neighbor)

        cluster_id += 1

    return labels


class Mode1VisionNode(Node):
    """Detect berry clusters from color and depth frames and publish the target cluster center."""

    def __init__(self) -> None:
        super().__init__("mode1_vision_node")

        self._declare_parameters()
        self._load_parameters()
        self._load_yolo_model()
        self._create_ros_interfaces()

        self.bridge = CvBridge()

        self.camera_frame_id = "camera_color_optical_frame"

        self.is_busy = False
        self.search_timeout_sec = 2.0
        self.search_start_time = None

        self.latest_color_image = None
        self.latest_depth_image = None

        self.has_camera_info = False
        self.fx = None
        self.fy = None
        self.ppx = None
        self.ppy = None

        if self.show_preview:
            try:
                cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
                cv2.namedWindow(self.result_window_name, cv2.WINDOW_NORMAL)
            except Exception as exc:
                self.get_logger().warn(f"Failed to create OpenCV preview windows: {exc}")
                self.show_preview = False

        self.timer = self.create_timer(0.1, self._timer_callback)
        self.status_publisher.publish(String(data="IDLE"))

    def _declare_parameters(self) -> None:
        self.declare_parameter("cmd_topic", "/vision/cmd")
        self.declare_parameter("status_topic", "/vision/status")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")

        self.declare_parameter("output_point_topic", "/camera_sphere")
        self.declare_parameter("output_radius_topic", "/sphere_radius")

        self.declare_parameter("depth_scale_topic", "/camera/depth_scale")
        self.declare_parameter("depth_scale_fallback", 0.001)

        self.declare_parameter("show_preview", True)
        self.declare_parameter("preview_window", "Camera Preview")
        self.declare_parameter("result_window", "Capture Result")

    def _load_parameters(self) -> None:
        self.command_topic = str(self.get_parameter("cmd_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)

        self.output_point_topic = str(self.get_parameter("output_point_topic").value)
        self.output_radius_topic = str(self.get_parameter("output_radius_topic").value)

        self.depth_scale_topic = str(self.get_parameter("depth_scale_topic").value)
        self.depth_scale_m_per_unit = float(
            self.get_parameter("depth_scale_fallback").value
        )

        self.show_preview = bool(self.get_parameter("show_preview").value)
        self.preview_window_name = str(self.get_parameter("preview_window").value)
        self.result_window_name = str(self.get_parameter("result_window").value)

        self.dbscan_eps = 70
        self.dbscan_min_samples = 2

        self.confidence_threshold = 0.6
        self.nms_score_threshold = 0.5
        self.nms_iou_threshold = 0.4
        self.min_valid_depth_m = 0.10

    def _load_yolo_model(self) -> None:
        share_dir = get_package_share_directory("harvesting_robot")
        models_dir = os.path.join(share_dir, "models")

        class_names_path = os.path.join(models_dir, "blackberry.names")
        weights_path = os.path.join(models_dir, "yolov4-tiny-custom_best.weights")
        config_path = os.path.join(models_dir, "yolov4-tiny-custom.cfg")

        self.get_logger().info(
            "Using YOLO model files:\n"
            f"  {class_names_path}\n"
            f"  {weights_path}\n"
            f"  {config_path}"
        )

        with open(class_names_path, "r", encoding="utf-8") as file:
            self.class_names = [line.strip() for line in file.readlines()]

        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        unconnected_layers = np.array(self.net.getUnconnectedOutLayers()).flatten()
        self.output_layers = [layer_names[int(layer_id) - 1] for layer_id in unconnected_layers]

    def _create_ros_interfaces(self) -> None:
        self.target_publisher = self.create_publisher(
            PointStamped,
            self.output_point_topic,
            10,
        )
        self.radius_publisher = self.create_publisher(
            Float32,
            self.output_radius_topic,
            10,
        )

        self.status_publisher = self.create_publisher(String, self.status_topic, 10)

        self.command_subscription = self.create_subscription(
            String,
            self.command_topic,
            self._on_command,
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

    def _on_depth_scale(self, msg: Float32) -> None:
        try:
            depth_scale = float(msg.data)
            if np.isfinite(depth_scale) and depth_scale > 0.0:
                self.depth_scale_m_per_unit = depth_scale
        except Exception:
            pass

    def _on_camera_info(self, msg: CameraInfo) -> None:
        intrinsic_matrix = list(msg.k)
        if len(intrinsic_matrix) == 9:
            self.fx = float(intrinsic_matrix[0])
            self.fy = float(intrinsic_matrix[4])
            self.ppx = float(intrinsic_matrix[2])
            self.ppy = float(intrinsic_matrix[5])
            self.has_camera_info = True

        if msg.header.frame_id:
            self.camera_frame_id = msg.header.frame_id

    def _on_color_image(self, msg: Image) -> None:
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
        except Exception:
            self.latest_color_image = None

    def _on_depth_image(self, msg: Image) -> None:
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough",
            )
        except Exception:
            self.latest_depth_image = None

    def _on_command(self, msg: String) -> None:
        command = (msg.data or "").strip().upper()
        self.get_logger().info(f"Received command: '{command}'")

        if command != "CAPTURE":
            return

        if self.is_busy:
            self.get_logger().info("CAPTURE command ignored because the node is busy.")
            return

        self.is_busy = True
        self.search_start_time = self.get_clock().now()
        self.status_publisher.publish(String(data="BUSY"))

    def _elapsed_search_time(self) -> float:
        if self.search_start_time is None:
            return 0.0

        now = self.get_clock().now()
        return (now - self.search_start_time).nanoseconds * 1e-9

    def _set_idle(self, success: bool) -> None:
        self.status_publisher.publish(String(data="DONE_OK" if success else "DONE_FAIL"))
        self.is_busy = False
        self.status_publisher.publish(String(data="IDLE"))

    def _run_yolo_inference(self, color_image: np.ndarray):
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

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence <= self.confidence_threshold:
                    continue

                center_x_norm = detection[0]
                center_y_norm = detection[1]
                width_norm = detection[2]
                height_norm = detection[3]

                center_x_px = int(center_x_norm * image_width)
                center_y_px = int(center_y_norm * image_height)
                width_px = int(width_norm * image_width)
                height_px = int(height_norm * image_height)

                top_left_x = int(center_x_px - width_px / 2)
                top_left_y = int(center_y_px - height_px / 2)

                boxes.append([top_left_x, top_left_y, width_px, height_px])
                confidences.append(confidence)
                class_ids.append(class_id)

        if not boxes:
            return [], [], []

        kept_indexes = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.nms_score_threshold,
            self.nms_iou_threshold,
        )

        if len(kept_indexes) == 0:
            return [], [], []

        normalized_indexes = []
        for index in kept_indexes:
            normalized_indexes.append(
                index[0] if isinstance(index, (list, tuple, np.ndarray)) else index
            )

        return normalized_indexes, boxes, confidences, class_ids

    def _extract_cluster_points(
        self,
        kept_indexes,
        boxes,
        confidences,
        class_ids,
        depth_image: np.ndarray,
        annotated_image: np.ndarray,
    ):
        cluster_points = []

        for detection_index in kept_indexes:
            x, y, width_px, height_px = boxes[detection_index]
            confidence = confidences[detection_index]
            class_id = class_ids[detection_index]

            center_x_px = x + width_px // 2
            center_y_px = y + height_px // 2

            box_color = (255, 0, 255) if class_id == 2 else (255, 255, 180)
            cv2.rectangle(
                annotated_image,
                (x, y),
                (x + width_px, y + height_px),
                box_color,
                2,
            )
            cv2.putText(
                annotated_image,
                f"{class_id} {confidence:.2f}",
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            if class_id != 2 or confidence < self.confidence_threshold:
                continue

            if (
                0 <= center_x_px < depth_image.shape[1]
                and 0 <= center_y_px < depth_image.shape[0]
            ):
                depth_m = float(depth_image[center_y_px, center_x_px]) * self.depth_scale_m_per_unit
            else:
                depth_m = 0.0

            if depth_m > self.min_valid_depth_m:
                cluster_points.append([center_x_px, center_y_px, depth_m])

        return cluster_points

    def _select_best_cluster(self, cluster_points, annotated_image: np.ndarray):
        if len(cluster_points) == 0:
            return None

        points = np.array(cluster_points, dtype=float)

        points_for_clustering = points.copy()
        points_for_clustering[:, 2] *= 100.0

        labels = simple_dbscan(
            points_for_clustering,
            self.dbscan_eps,
            self.dbscan_min_samples,
        )

        valid_labels = set(labels.tolist())
        valid_labels.discard(-1)

        best_cluster = None
        best_cluster_size = 0

        for label in valid_labels:
            group_points = points[labels == label]

            center_x_px = float(np.mean(group_points[:, 0]))
            center_y_px = float(np.mean(group_points[:, 1]))
            center_z_m = float(np.median(group_points[:, 2]))

            center_2d = np.array([center_x_px, center_y_px], dtype=float)
            group_points_2d = group_points[:, :2]
            distances_to_center = np.linalg.norm(group_points_2d - center_2d, axis=1)
            cluster_radius_px = int(np.max(distances_to_center)) + 10

            cluster_size = len(group_points)

            cv2.circle(
                annotated_image,
                (int(center_x_px), int(center_y_px)),
                int(cluster_radius_px),
                (255, 0, 255),
                2,
            )
            cv2.putText(
                annotated_image,
                f"G{label}: {cluster_size} berries",
                (int(center_x_px) - 20, int(center_y_px) - cluster_radius_px - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
            )

            if cluster_size > best_cluster_size:
                best_cluster_size = cluster_size
                best_cluster = {
                    "center_x_px": center_x_px,
                    "center_y_px": center_y_px,
                    "center_z_m": center_z_m,
                    "radius_px": cluster_radius_px,
                    "points_2d": group_points_2d,
                }

        return best_cluster

    def _publish_cluster_target(self, cluster_data, annotated_image: np.ndarray) -> bool:
        if cluster_data is None:
            return False

        center_x_px = cluster_data["center_x_px"]
        center_y_px = cluster_data["center_y_px"]
        center_z_m = cluster_data["center_z_m"]
        cluster_radius_px = cluster_data["radius_px"]
        points_2d = cluster_data["points_2d"]

        cv2.circle(
            annotated_image,
            (int(center_x_px), int(center_y_px)),
            int(cluster_radius_px) + 5,
            (0, 0, 255),
            4,
        )
        cv2.putText(
            annotated_image,
            "Target",
            (int(center_x_px), int(center_y_px)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        if center_z_m <= self.min_valid_depth_m:
            return False

        target_x_m = (center_x_px - self.ppx) * center_z_m / self.fx
        target_y_m = (center_y_px - self.ppy) * center_z_m / self.fy

        delta_x_px = points_2d[:, 0] - center_x_px
        delta_y_px = points_2d[:, 1] - center_y_px
        delta_x_m = delta_x_px * center_z_m / self.fx
        delta_y_m = delta_y_px * center_z_m / self.fy
        radius_m = float(np.max(np.sqrt(delta_x_m * delta_x_m + delta_y_m * delta_y_m)))

        target_msg = PointStamped()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = self.camera_frame_id
        target_msg.point.x = float(target_x_m)
        target_msg.point.y = float(target_y_m)
        target_msg.point.z = float(center_z_m)

        self.target_publisher.publish(target_msg)
        self.radius_publisher.publish(Float32(data=radius_m))
        return True

    def _timer_callback(self) -> None:
        if self.show_preview:
            try:
                cv2.waitKey(1)
            except Exception:
                pass

        if self.latest_color_image is None:
            return

        if self.show_preview:
            cv2.imshow(self.preview_window_name, self.latest_color_image)

        if not self.is_busy:
            return

        if self.latest_depth_image is None or not self.has_camera_info:
            if self._elapsed_search_time() > self.search_timeout_sec:
                self._set_idle(success=False)
            return

        color_image = self.latest_color_image
        depth_image = self.latest_depth_image
        annotated_image = color_image.copy()

        kept_indexes, boxes, confidences, class_ids = self._run_yolo_inference(color_image)

        cluster_points = self._extract_cluster_points(
            kept_indexes,
            boxes,
            confidences,
            class_ids,
            depth_image,
            annotated_image,
        )

        best_cluster = self._select_best_cluster(cluster_points, annotated_image)
        published = self._publish_cluster_target(best_cluster, annotated_image)

        if self.show_preview:
            cv2.imshow(self.result_window_name, annotated_image)

        if published:
            self._set_idle(success=True)
        elif self._elapsed_search_time() > self.search_timeout_sec:
            self._set_idle(success=False)

    def destroy_node(self) -> None:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Mode1VisionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()