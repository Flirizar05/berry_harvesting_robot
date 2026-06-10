#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Optional

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, String


def normalize_command(command: str) -> str:
    return (command or "").strip().upper()


class ObbPcaVisionNode(Node):
    """Run one-shot YOLO-OBB + SAM segmentation + 3D PCA visualization."""

    def __init__(self) -> None:
        super().__init__("obb_pca_vision_node")

        self._declare_parameters()
        self._load_parameters()
        self._initialize_runtime_state()
        self._initialize_visualization()
        if self.show_result:
            self.latest_display_image = self._make_status_image(
                "OBB/PCA ready",
                f"Waiting for CAPTURE on {self.command_topic}",
            )
            self._show_result_image(self.latest_display_image)
        self._create_ros_interfaces()

        self.timer = self.create_timer(self.timer_period_sec, self._timer_callback)
        self.status_publisher.publish(String(data="IDLE"))
        self.get_logger().info(
            "obb_pca_vision_node ready. "
            f"cmd={self.command_topic}, status={self.status_topic}, "
            f"result={self.result_topic}, window={self.result_window_name}"
        )

    def _declare_parameters(self) -> None:
        share_dir = get_package_share_directory("harvesting_robot")
        models_dir = os.path.join(share_dir, "models")

        self.declare_parameter("yolo_model_path", os.path.join(models_dir, "best.pt"))
        self.declare_parameter("sam_model_path", "mobile_sam.pt")
        self.declare_parameter("yolo_device", "0")
        self.declare_parameter("conf_thresh", 0.6)
        self.declare_parameter("nms_thresh", 0.4)
        self.declare_parameter("target_class_id", 2)
        self.declare_parameter("max_detections", 5)

        self.declare_parameter("cmd_topic", "/obb_pca/cmd")
        self.declare_parameter("status_topic", "/obb_pca/status")
        self.declare_parameter("result_topic", "/obb_pca/result")
        self.declare_parameter("annotated_topic", "/obb_pca/annotated_image")
        self.declare_parameter("publish_annotated", True)

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("depth_scale_topic", "/camera/depth_scale")
        self.declare_parameter("depth_scale_fallback", 0.001)

        self.declare_parameter("capture_timeout_sec", 5.0)
        self.declare_parameter("timer_period", 0.02)
        self.declare_parameter("single_shot", True)

        self.declare_parameter("min_valid_depth_m", 0.10)
        self.declare_parameter("max_valid_depth_m", 2.00)
        self.declare_parameter("min_pca_points", 50)
        self.declare_parameter("point_stride", 2)
        self.declare_parameter("axis_scale_m", 0.05)

        self.declare_parameter("show_result", True)
        self.declare_parameter("result_window_name", "RGB + YOLO-OBB + SAM + PCA")
        self.declare_parameter("result_window_width", 960)
        self.declare_parameter("result_window_height", 720)
        self.declare_parameter("result_window_x", 720)
        self.declare_parameter("result_window_y", 40)
        self.declare_parameter("result_wait_ms", 30)

    def _load_parameters(self) -> None:
        self.yolo_model_path = os.path.expanduser(
            str(self.get_parameter("yolo_model_path").value or "")
        )
        self.sam_model_path = os.path.expanduser(
            str(self.get_parameter("sam_model_path").value or "mobile_sam.pt")
        )
        self.yolo_device = str(self.get_parameter("yolo_device").value).strip()
        self.confidence_threshold = float(self.get_parameter("conf_thresh").value)
        self.nms_threshold = float(self.get_parameter("nms_thresh").value)
        self.target_class_id = int(self.get_parameter("target_class_id").value)
        self.max_detections = max(1, int(self.get_parameter("max_detections").value))

        self.command_topic = str(self.get_parameter("cmd_topic").value).strip()
        self.status_topic = str(self.get_parameter("status_topic").value).strip()
        self.result_topic = str(self.get_parameter("result_topic").value).strip()
        self.annotated_topic = str(self.get_parameter("annotated_topic").value).strip()
        self.publish_annotated = bool(
            self.get_parameter("publish_annotated").value
        )

        self.color_topic = str(self.get_parameter("color_topic").value).strip()
        self.depth_topic = str(self.get_parameter("depth_topic").value).strip()
        self.camera_info_topic = str(
            self.get_parameter("camera_info_topic").value
        ).strip()
        self.depth_scale_topic = str(
            self.get_parameter("depth_scale_topic").value
        ).strip()
        self.depth_scale_m_per_unit = float(
            self.get_parameter("depth_scale_fallback").value
        )

        self.capture_timeout_sec = float(
            self.get_parameter("capture_timeout_sec").value
        )
        self.timer_period_sec = float(self.get_parameter("timer_period").value)
        self.single_shot = bool(self.get_parameter("single_shot").value)

        self.min_valid_depth_m = float(
            self.get_parameter("min_valid_depth_m").value
        )
        self.max_valid_depth_m = float(
            self.get_parameter("max_valid_depth_m").value
        )
        self.min_pca_points = max(3, int(self.get_parameter("min_pca_points").value))
        self.point_stride = max(1, int(self.get_parameter("point_stride").value))
        self.axis_scale_m = float(self.get_parameter("axis_scale_m").value)

        self.show_result = bool(self.get_parameter("show_result").value)
        self.result_window_name = str(
            self.get_parameter("result_window_name").value
        ).strip()
        self.result_window_width = max(
            320,
            int(self.get_parameter("result_window_width").value),
        )
        self.result_window_height = max(
            240,
            int(self.get_parameter("result_window_height").value),
        )
        self.result_window_x = int(self.get_parameter("result_window_x").value)
        self.result_window_y = int(self.get_parameter("result_window_y").value)
        self.result_wait_ms = max(1, int(self.get_parameter("result_wait_ms").value))

    def _initialize_runtime_state(self) -> None:
        self.bridge = CvBridge()

        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_depth_encoding = ""

        self.fx = None
        self.fy = None
        self.ppx = None
        self.ppy = None
        self.camera_frame_id = "camera_color_optical_frame"
        self.has_camera_info = False

        self.models_loaded = False
        self.yolo_model = None
        self.sam_model = None
        self.class_names = []

        self.is_busy = False
        self.capture_start_time = None
        self.capture_consumed = False
        self.latest_display_image = None
        self.last_display_refresh_time = 0.0

    def _initialize_visualization(self) -> None:
        if not self.show_result:
            return

        try:
            cv2.startWindowThread()
            cv2.namedWindow(self.result_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.result_window_name,
                self.result_window_width,
                self.result_window_height,
            )
            cv2.moveWindow(
                self.result_window_name,
                self.result_window_x,
                self.result_window_y,
            )
        except Exception as exc:
            self.get_logger().warn(f"Failed to create OpenCV window: {exc}")
            self.show_result = False

    def _load_models(self) -> None:
        if self.models_loaded:
            return

        if not self.yolo_model_path or not os.path.isfile(self.yolo_model_path):
            self.get_logger().error(
                f"Missing YOLO model file: '{self.yolo_model_path}'"
            )
            raise FileNotFoundError(self.yolo_model_path)

        try:
            from ultralytics import SAM, YOLO
        except ImportError as exc:
            raise ImportError(
                "obb_pca_vision_node needs the 'ultralytics' package "
                "with YOLO and SAM support."
            ) from exc

        self.yolo_device = self._resolve_yolo_device(self.yolo_device)
        self.yolo_model = YOLO(self.yolo_model_path)

        sam_source = self.sam_model_path or "mobile_sam.pt"
        if os.path.isabs(sam_source) and not os.path.isfile(sam_source):
            self.get_logger().warn(
                f"SAM model path '{sam_source}' does not exist. "
                "Ultralytics will receive the value as-is."
            )
        elif not os.path.isabs(sam_source) and not os.path.isfile(sam_source):
            self.get_logger().warn(
                f"SAM model '{sam_source}' was not found in the current directory. "
                "It must exist locally or be available in the Ultralytics cache."
            )

        self.sam_model = SAM(sam_source)
        self.class_names = self._normalize_class_names(
            getattr(self.yolo_model, "names", None)
        )

        self.get_logger().info(
            "Using YOLO-OBB + SAM models:\n"
            f"  yolo: {self.yolo_model_path}\n"
            f"  sam: {sam_source}\n"
            f"  device: {self.yolo_device or 'auto'}\n"
            f"  target_class_id: {self.target_class_id}"
        )
        self.models_loaded = True

    def _create_ros_interfaces(self) -> None:
        self.status_publisher = self.create_publisher(String, self.status_topic, 10)
        self.result_publisher = self.create_publisher(String, self.result_topic, 10)
        self.annotated_image_publisher = (
            self.create_publisher(Image, self.annotated_topic, 10)
            if self.publish_annotated
            else None
        )

        self.create_subscription(String, self.command_topic, self._on_command, 10)
        self.create_subscription(Image, self.color_topic, self._on_color_image, 10)
        self.create_subscription(Image, self.depth_topic, self._on_depth_image, 10)
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._on_camera_info,
            10,
        )
        self.create_subscription(
            Float32,
            self.depth_scale_topic,
            self._on_depth_scale,
            10,
        )

    def _on_command(self, msg: String) -> None:
        command = normalize_command(msg.data)

        if command == "RESET":
            self.capture_consumed = False
            self.is_busy = False
            self.capture_start_time = None
            self.status_publisher.publish(String(data="IDLE"))
            self.get_logger().info("RESET received. One-shot capture is available.")
            return

        if command != "CAPTURE":
            return

        if self.single_shot and self.capture_consumed:
            self.get_logger().info(
                "CAPTURE ignored because single_shot capture was already consumed."
            )
            return

        if self.is_busy:
            self.get_logger().info("CAPTURE ignored because the node is busy.")
            return

        self.is_busy = True
        self.capture_start_time = self.get_clock().now()
        self.status_publisher.publish(String(data="BUSY"))
        if self.show_result:
            self.latest_display_image = self._make_status_image(
                "OBB/PCA running",
                "Processing final capture...",
            )
            self._show_result_image(self.latest_display_image)
        self.get_logger().info("CAPTURE received. Running one-shot OBB/PCA pipeline.")

    def _on_color_image(self, msg: Image) -> None:
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
        except Exception as exc:
            self.latest_color_image = None
            self.get_logger().warn(f"Failed to convert color image: {exc}")

    def _on_depth_image(self, msg: Image) -> None:
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough",
            )
            self.latest_depth_encoding = msg.encoding or ""
        except Exception as exc:
            self.latest_depth_image = None
            self.latest_depth_encoding = ""
            self.get_logger().warn(f"Failed to convert depth image: {exc}")

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

    def _on_depth_scale(self, msg: Float32) -> None:
        try:
            depth_scale = float(msg.data)
            if np.isfinite(depth_scale) and depth_scale > 0.0:
                self.depth_scale_m_per_unit = depth_scale
        except Exception:
            pass

    def _timer_callback(self) -> None:
        if self.show_result:
            self._refresh_result_window()

        if not self.is_busy:
            return

        if not self._has_required_frame_data():
            if self._timed_out():
                self._finish_capture(
                    False,
                    "No camera frame/depth/intrinsics available.",
                    None,
                    [],
                )
            return

        color_image = self.latest_color_image.copy()
        depth_image = np.asarray(self.latest_depth_image).copy()

        try:
            self._load_models()
            annotated_image, detections = self._run_obb_pca(color_image, depth_image)
        except Exception as exc:
            annotated_image = color_image.copy()
            self._draw_status(annotated_image, "OBB/PCA pipeline error")
            self._draw_text_lines(
                annotated_image,
                self._wrap_text(str(exc), max_chars=72)[:8],
                (12, 58),
            )
            self._finish_capture(False, f"OBB/PCA pipeline error: {exc}", annotated_image, [])
            return

        if not detections:
            self._finish_capture(
                False,
                "No valid YOLO-OBB + SAM mask with enough depth points.",
                annotated_image,
                [],
            )
            return

        self._finish_capture(
            True,
            f"{len(detections)} PCA result(s)",
            annotated_image,
            detections,
        )

    def _has_required_frame_data(self) -> bool:
        return (
            self.latest_color_image is not None
            and self.latest_depth_image is not None
            and self.has_camera_info
            and self.fx is not None
            and self.fy is not None
            and self.ppx is not None
            and self.ppy is not None
        )

    def _run_obb_pca(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
    ) -> tuple[np.ndarray, list[dict]]:
        annotated_image = color_image.copy()
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        image_height, image_width = color_image.shape[:2]

        yolo_results = self.yolo_model(
            rgb_image,
            conf=self.confidence_threshold,
            device=self.yolo_device or None,
            iou=self.nms_threshold,
            verbose=False,
        )

        candidates = []
        for result in yolo_results or []:
            candidates.extend(self._extract_candidates(result, image_width, image_height))

        candidates = sorted(
            candidates,
            key=lambda detection: float(detection["confidence"]),
            reverse=True,
        )

        if not candidates:
            self._draw_status(annotated_image, "No YOLO detections")
            return annotated_image, []

        detections = []
        for candidate in candidates[: self.max_detections]:
            self._draw_candidate_box(annotated_image, candidate, (0, 180, 255), 2)

            mask = self._segment_candidate(rgb_image, candidate, image_width, image_height)
            if mask is None:
                continue

            points_3d = self._points_from_mask(mask, depth_image)
            if points_3d.shape[0] < self.min_pca_points:
                continue

            center, axis = self._compute_pca_axis(points_3d)
            projected_center = self._project_point(center)
            projected_axis_tip = self._project_point(center + axis * self.axis_scale_m)

            detection = {
                "class_id": int(candidate["class_id"]),
                "class_name": self._class_name(int(candidate["class_id"])),
                "confidence": float(candidate["confidence"]),
                "bbox_xyxy": [
                    int(candidate["x0"]),
                    int(candidate["y0"]),
                    int(candidate["x1"]),
                    int(candidate["y1"]),
                ],
                "center_xyz_m": [float(value) for value in center.tolist()],
                "axis_xyz": [float(value) for value in axis.tolist()],
                "point_count": int(points_3d.shape[0]),
                "source": str(candidate["source"]),
            }
            detections.append(detection)

            self._draw_mask_overlay(annotated_image, mask)
            self._draw_candidate_box(annotated_image, candidate, (0, 255, 0), 3)
            self._draw_pca_axis(
                annotated_image,
                projected_center,
                projected_axis_tip,
                detection,
            )

        if detections:
            self._draw_status(annotated_image, f"PCA results: {len(detections)}")
        else:
            self._draw_status(annotated_image, "No valid SAM/PCA result")

        return annotated_image, detections

    def _extract_candidates(
        self,
        result,
        image_width: int,
        image_height: int,
    ) -> list[dict]:
        obb = getattr(result, "obb", None)
        if obb is not None:
            candidates = self._extract_from_prediction(
                obb,
                "obb",
                image_width,
                image_height,
            )
            if candidates:
                return candidates

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        return self._extract_from_prediction(
            boxes,
            "box",
            image_width,
            image_height,
        )

    def _extract_from_prediction(
        self,
        prediction,
        source: str,
        image_width: int,
        image_height: int,
    ) -> list[dict]:
        xyxy = getattr(prediction, "xyxy", None)
        if xyxy is None:
            return []

        xyxy_boxes = self._to_numpy(xyxy)
        if xyxy_boxes.size == 0:
            return []

        confidence_values = getattr(prediction, "conf", None)
        class_values = getattr(prediction, "cls", None)
        confidences = (
            np.ones(len(xyxy_boxes), dtype=np.float32)
            if confidence_values is None
            else self._to_numpy(confidence_values).reshape(-1)
        )
        class_ids = (
            np.zeros(len(xyxy_boxes), dtype=np.int32)
            if class_values is None
            else self._to_numpy(class_values).reshape(-1).astype(int)
        )
        polygons = self._prediction_polygons(prediction)

        candidates = []
        for index, (box, confidence, class_id) in enumerate(
            zip(xyxy_boxes, confidences, class_ids)
        ):
            class_id = int(class_id)
            confidence = float(confidence)
            if confidence < self.confidence_threshold:
                continue
            if self.target_class_id >= 0 and class_id != self.target_class_id:
                continue

            x0 = int(round(max(0.0, min(float(image_width - 1), float(box[0])))))
            y0 = int(round(max(0.0, min(float(image_height - 1), float(box[1])))))
            x1 = int(round(max(0.0, min(float(image_width - 1), float(box[2])))))
            y1 = int(round(max(0.0, min(float(image_height - 1), float(box[3])))))
            if x1 <= x0 or y1 <= y0:
                continue

            polygon = None
            if polygons is not None and index < len(polygons):
                polygon = self._clean_polygon(polygons[index], image_width, image_height)

            candidates.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "confidence": confidence,
                    "class_id": class_id,
                    "source": source,
                    "polygon": polygon,
                }
            )

        return candidates

    def _prediction_polygons(self, prediction) -> Optional[np.ndarray]:
        for attribute_name in ("xyxyxyxy", "xywhr"):
            value = getattr(prediction, attribute_name, None)
            if value is None:
                continue

            array = self._to_numpy(value)
            if array.size > 0:
                return array

        return None

    @staticmethod
    def _clean_polygon(
        polygon: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> Optional[np.ndarray]:
        polygon = np.asarray(polygon, dtype=float)
        if polygon.shape == (4, 2):
            points = polygon
        elif polygon.size == 8:
            points = polygon.reshape(4, 2)
        else:
            return None

        points[:, 0] = np.clip(points[:, 0], 0, image_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, image_height - 1)
        return np.round(points).astype(np.int32)

    def _segment_candidate(
        self,
        rgb_image: np.ndarray,
        candidate: dict,
        image_width: int,
        image_height: int,
    ) -> Optional[np.ndarray]:
        bbox = [
            int(candidate["x0"]),
            int(candidate["y0"]),
            int(candidate["x1"]),
            int(candidate["y1"]),
        ]

        sam_results = self.sam_model(
            rgb_image,
            bboxes=[bbox],
            verbose=False,
        )
        if not sam_results:
            return None

        masks = getattr(sam_results[0], "masks", None)
        if masks is None or getattr(masks, "data", None) is None:
            return None

        mask_data = self._to_numpy(masks.data)
        if mask_data.size == 0:
            return None

        mask = mask_data[0]
        if mask.shape[:2] != (image_height, image_width):
            mask = cv2.resize(
                mask.astype(np.float32),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            )

        return mask > 0.5

    def _points_from_mask(
        self,
        mask: np.ndarray,
        depth_image: np.ndarray,
    ) -> np.ndarray:
        ys, xs = np.where(mask)
        if xs.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        if self.point_stride > 1:
            xs = xs[:: self.point_stride]
            ys = ys[:: self.point_stride]

        depth_values_m = self._depth_values_to_meters(depth_image[ys, xs])
        valid = (
            np.isfinite(depth_values_m)
            & (depth_values_m >= self.min_valid_depth_m)
            & (depth_values_m <= self.max_valid_depth_m)
        )

        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        zs = depth_values_m[valid].astype(np.float32)

        x_m = (xs - float(self.ppx)) * zs / float(self.fx)
        y_m = (ys - float(self.ppy)) * zs / float(self.fy)

        return np.column_stack((x_m, y_m, zs)).astype(np.float32)

    def _depth_values_to_meters(self, depth_values: np.ndarray) -> np.ndarray:
        values = np.asarray(depth_values, dtype=np.float32)
        if self.latest_depth_encoding in ("32FC1", "64FC1"):
            return values

        return values * float(self.depth_scale_m_per_unit)

    @staticmethod
    def _compute_pca_axis(points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.mean(points_3d, axis=0)
        centered_points = points_3d - center
        _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
        axis = vh[0]
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-12:
            axis = axis / axis_norm
        if axis[2] < 0.0:
            axis = -axis
        return center.astype(np.float32), axis.astype(np.float32)

    def _project_point(self, point: np.ndarray) -> Optional[tuple[int, int]]:
        x_m, y_m, z_m = [float(value) for value in point]
        if not np.isfinite(z_m) or z_m <= 1e-6:
            return None

        u = int(round((x_m * float(self.fx) / z_m) + float(self.ppx)))
        v = int(round((y_m * float(self.fy) / z_m) + float(self.ppy)))
        return u, v

    def _draw_candidate_box(
        self,
        image: np.ndarray,
        candidate: dict,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        polygon = candidate.get("polygon")
        if polygon is not None:
            cv2.polylines(image, [polygon], True, color, thickness)
        else:
            cv2.rectangle(
                image,
                (int(candidate["x0"]), int(candidate["y0"])),
                (int(candidate["x1"]), int(candidate["y1"])),
                color,
                thickness,
            )

        label = (
            f"{candidate['source']} {self._class_name(int(candidate['class_id']))} "
            f"{float(candidate['confidence']):.2f}"
        )
        cv2.putText(
            image,
            label,
            (int(candidate["x0"]), max(16, int(candidate["y0"]) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    @staticmethod
    def _draw_mask_overlay(image: np.ndarray, mask: np.ndarray) -> None:
        overlay = image.copy()
        overlay[mask] = (0, 255, 0)
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0.0, dst=image)

    def _draw_pca_axis(
        self,
        image: np.ndarray,
        projected_center: Optional[tuple[int, int]],
        projected_axis_tip: Optional[tuple[int, int]],
        detection: dict,
    ) -> None:
        if projected_center is not None:
            cv2.circle(image, projected_center, 5, (0, 255, 255), -1)

        if projected_center is not None and projected_axis_tip is not None:
            cv2.arrowedLine(
                image,
                projected_center,
                projected_axis_tip,
                (0, 0, 255),
                3,
                tipLength=0.25,
            )

        x0, y0, _, _ = detection["bbox_xyxy"]
        center = detection["center_xyz_m"]
        axis = detection["axis_xyz"]
        text_lines = [
            f"PCA pts={detection['point_count']}",
            f"C=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})m",
            f"A=({axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f})",
        ]
        self._draw_text_lines(image, text_lines, (int(x0), int(y0) + 18))

    @staticmethod
    def _draw_text_lines(
        image: np.ndarray,
        lines: list[str],
        origin: tuple[int, int],
    ) -> None:
        x, y = origin
        for line in lines:
            cv2.putText(
                image,
                line,
                (max(0, x), max(16, y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                image,
                line,
                (max(0, x), max(16, y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 20),
                1,
            )
            y += 17

    @staticmethod
    def _draw_status(image: np.ndarray, text: str) -> None:
        cv2.putText(
            image,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            image,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    def _finish_capture(
        self,
        success: bool,
        reason: str,
        annotated_image: Optional[np.ndarray],
        detections: list[dict],
    ) -> None:
        payload = {
            "success": bool(success),
            "reason": reason,
            "frame_id": self.camera_frame_id,
            "depth_scale_m_per_unit": float(self.depth_scale_m_per_unit),
            "detections": detections,
        }
        self.result_publisher.publish(String(data=json.dumps(payload)))

        status = "DONE_OK" if success else "DONE_FAIL"
        self.status_publisher.publish(String(data=status))
        if success:
            self.get_logger().info(f"{status}: {reason}")
        else:
            self.get_logger().warn(f"{status}: {reason}")

        if annotated_image is not None:
            if self.annotated_image_publisher is not None:
                try:
                    self.annotated_image_publisher.publish(
                        self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
                    )
                except Exception as exc:
                    self.get_logger().warn(f"Failed to publish annotated image: {exc}")

        if self.show_result:
            display_image = annotated_image
            if display_image is None:
                display_image = self._make_status_image(status, reason)
            self.latest_display_image = display_image
            self._show_result_image(display_image)

        self.capture_consumed = True
        self.is_busy = False
        self.capture_start_time = None
        self.status_publisher.publish(String(data="IDLE"))

    def _show_result_image(self, image: np.ndarray) -> None:
        try:
            cv2.imshow(self.result_window_name, image)
            cv2.resizeWindow(
                self.result_window_name,
                self.result_window_width,
                self.result_window_height,
            )
            cv2.moveWindow(
                self.result_window_name,
                self.result_window_x,
                self.result_window_y,
            )
            cv2.waitKey(self.result_wait_ms)
            self.last_display_refresh_time = (
                self.get_clock().now().nanoseconds / 1e9
            )
        except Exception as exc:
            self.get_logger().warn(f"Failed to show result window: {exc}")

    def _refresh_result_window(self) -> None:
        if self.latest_display_image is None:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if now_sec - self.last_display_refresh_time >= 1.0:
            self._show_result_image(self.latest_display_image)
            return

        try:
            cv2.waitKey(1)
        except Exception:
            pass

    def _make_status_image(self, title: str, reason: str) -> np.ndarray:
        image = np.zeros((480, 720, 3), dtype=np.uint8)
        self._draw_status(image, title)
        self._draw_text_lines(
            image,
            self._wrap_text(reason, max_chars=68)[:12],
            (24, 70),
        )
        return image

    @staticmethod
    def _wrap_text(text: str, max_chars: int) -> list[str]:
        words = str(text).replace("\n", " ").split()
        if not words:
            return [""]

        lines = []
        current = ""
        for word in words:
            if not current:
                current = word
                continue

            if len(current) + 1 + len(word) <= max_chars:
                current = f"{current} {word}"
            else:
                lines.append(current)
                current = word

        if current:
            lines.append(current)
        return lines

    def _timed_out(self) -> bool:
        if self.capture_start_time is None:
            return False

        elapsed_sec = (
            self.get_clock().now() - self.capture_start_time
        ).nanoseconds * 1e-9
        return elapsed_sec >= self.capture_timeout_sec

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def _resolve_yolo_device(self, requested_device: str) -> str:
        device = str(requested_device).strip()
        if device.lower() in ("", "none", "auto"):
            return ""

        if device.lower() == "cpu":
            return "cpu"

        try:
            import torch
        except ImportError:
            self.get_logger().warn(
                f"YOLO device '{device}' requested but torch is not installed; using CPU"
            )
            return "cpu"

        if torch.cuda.is_available():
            return device

        self.get_logger().warn(
            f"YOLO device '{device}' requested but CUDA is not available; using CPU"
        )
        return "cpu"

    @staticmethod
    def _normalize_class_names(class_names) -> list[str]:
        if isinstance(class_names, dict):
            return [
                str(class_names[key])
                for key in sorted(class_names, key=lambda item: int(item))
            ]

        if isinstance(class_names, (list, tuple)):
            return [str(class_name) for class_name in class_names]

        return []

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id] or f"class_{class_id}"
        return f"class_{class_id}"


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObbPcaVisionNode()
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
