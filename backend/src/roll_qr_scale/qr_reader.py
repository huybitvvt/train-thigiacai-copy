from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class QRDetection:
    value: str
    points: np.ndarray
    decoder: str = "opencv"


class QRReader:
    """Decode QR labels directly, then localize and zoom difficult distant labels."""

    def __init__(
        self,
        yolo_model: str | Path | None = None,
        confidence: float = 0.25,
        yolo_mode: str = "first",
        yolo_imgsz: int = 960,
    ):
        if yolo_mode not in {"first", "fallback"}:
            raise ValueError("yolo_mode must be 'first' or 'fallback'")
        self.detector = cv2.QRCodeDetector()
        self.confidence = confidence
        self.yolo_mode = yolo_mode
        self.yolo_imgsz = yolo_imgsz
        self.model = None
        if yolo_model:
            from ultralytics import YOLO

            self.model = YOLO(str(yolo_model))

    def decode(self, frame: np.ndarray) -> list[QRDetection]:
        if self.model is not None and self.yolo_mode == "first":
            detections = self._decode_yolo_regions(frame)
            return detections or self._decode_without_yolo(frame)

        detections = self._decode_without_yolo(frame)
        if detections or self.model is None:
            return detections
        return self._decode_yolo_regions(frame)

    def _decode_without_yolo(self, frame: np.ndarray) -> list[QRDetection]:
        detections = self._decode_image(frame)
        if detections:
            return detections
        return self._decode_localized_regions(frame)

    def _decode_localized_regions(self, frame: np.ndarray) -> list[QRDetection]:
        """Find QR-shaped regions, crop them tightly, enlarge them, then decode again."""
        detections = self._decode_regions(frame, self._finder_pattern_regions(frame))
        if detections:
            return detections
        return self._decode_regions(frame, self._opencv_regions(frame))

    def _decode_regions(
        self,
        frame: np.ndarray,
        regions: list[tuple[int, int, int, int, str]],
    ) -> list[QRDetection]:
        height, width = frame.shape[:2]
        found: list[QRDetection] = []
        for left, top, right, bottom, localizer in self._unique_regions(regions):
            left, top = max(0, left), max(0, top)
            right, bottom = min(width, right), min(height, bottom)
            if right <= left or bottom <= top:
                continue
            crop = frame[top:bottom, left:right]
            for item in self._decode_crop(crop):
                shifted = item.points + np.array([left, top], dtype=np.float32)
                found.append(QRDetection(item.value, shifted, f"{localizer}+{item.decoder}"))
        return self._deduplicate(found)

    def _unique_regions(
        self, regions: list[tuple[int, int, int, int, str]]
    ) -> list[tuple[int, int, int, int, str]]:
        unique: list[tuple[int, int, int, int, str]] = []
        ordered = sorted(regions, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]))
        for candidate in ordered:
            if any(self._box_iou(candidate[:4], existing[:4]) >= 0.65 for existing in unique):
                continue
            unique.append(candidate)
            if len(unique) >= 12:
                break
        return unique

    def _opencv_regions(self, frame: np.ndarray) -> list[tuple[int, int, int, int, str]]:
        """Keep OpenCV's location even when its combined detect/decode could not read text."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        regions: list[tuple[int, int, int, int, str]] = []
        try:
            ok, points = self.detector.detect(gray)
        except (cv2.error, ValueError):
            return regions
        if not ok or points is None:
            return regions
        points = np.asarray(points, dtype=np.float32).reshape((4, 2))
        if not np.isfinite(points).all():
            return regions
        x1, y1 = np.floor(points.min(axis=0)).astype(int)
        x2, y2 = np.ceil(points.max(axis=0)).astype(int)
        span = max(x2 - x1, y2 - y1)
        if span < 8:
            return regions
        pad = max(8, int(round(span * 0.25)))
        regions.append((x1 - pad, y1 - pad, x2 + pad, y2 + pad, "opencv-roi"))
        return regions

    def _finder_pattern_regions(
        self, frame: np.ndarray
    ) -> list[tuple[int, int, int, int, str]]:
        """Locate triples of nested square finder patterns in QR labels too small to decode."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        finder_boxes: list[tuple[int, int, int, int, int]] = []
        for binary in (otsu, adaptive):
            contours, hierarchy = cv2.findContours(
                binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            if hierarchy is None:
                continue
            tree = hierarchy[0]
            for index, contour in enumerate(contours):
                x, y, width, height = cv2.boundingRect(contour)
                if width < 5 or height < 5:
                    continue
                ratio = width / height
                if not 0.65 <= ratio <= 1.35:
                    continue
                child = int(tree[index][2])
                depth = 0
                while child >= 0 and depth < 6:
                    depth += 1
                    child = int(tree[child][2])
                if depth < 2:
                    continue
                area = cv2.contourArea(contour)
                if area < 12 or area / (width * height) < 0.45:
                    continue
                finder_boxes.append((x, y, width, height, depth))

        deduplicated: list[tuple[int, int, int, int, int]] = []
        for box in sorted(finder_boxes, key=lambda item: (-item[4], item[2] * item[3])):
            bounds = (box[0], box[1], box[0] + box[2], box[1] + box[3])
            if any(
                self._box_iou(
                    bounds,
                    (item[0], item[1], item[0] + item[2], item[1] + item[3]),
                )
                >= 0.65
                for item in deduplicated
            ):
                continue
            deduplicated.append(box)
            if len(deduplicated) >= 36:
                break

        regions: list[tuple[int, int, int, int, str]] = []
        for triple in combinations(deduplicated, 3):
            sizes = np.array([max(item[2], item[3]) for item in triple], dtype=np.float32)
            if sizes.max() > sizes.min() * 2.2:
                continue
            centers = np.array(
                [[x + width / 2, y + height / 2] for x, y, width, height, _ in triple],
                dtype=np.float32,
            )
            squared = sorted(
                float(np.sum((centers[left] - centers[right]) ** 2))
                for left, right in ((0, 1), (0, 2), (1, 2))
            )
            if squared[0] <= 0 or squared[1] / squared[0] > 2.8:
                continue
            if abs((squared[0] + squared[1]) - squared[2]) / squared[2] > 0.32:
                continue
            x1 = min(item[0] for item in triple)
            y1 = min(item[1] for item in triple)
            x2 = max(item[0] + item[2] for item in triple)
            y2 = max(item[1] + item[3] for item in triple)
            pad = max(6, int(round(float(np.median(sizes)))))
            regions.append((x1 - pad, y1 - pad, x2 + pad, y2 + pad, "finder-roi"))
        return regions

    def _decode_yolo_regions(self, frame: np.ndarray) -> list[QRDetection]:
        """Run a custom QR detector and decode each predicted crop."""
        assert self.model is not None
        result = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.yolo_imgsz,
            verbose=False,
        )[0]
        height, width = frame.shape[:2]
        found: list[QRDetection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        coordinates = boxes.xyxy.cpu().numpy()
        class_values = getattr(boxes, "cls", None)
        class_ids = (
            class_values.cpu().numpy().astype(int).tolist()
            if class_values is not None
            else [0] * len(coordinates)
        )
        for xyxy, class_id in zip(coordinates, class_ids, strict=False):
            if not self._is_qr_class(result, class_id):
                continue
            x1, y1, x2, y2 = (int(value) for value in xyxy)
            pad_x = max(12, (x2 - x1) * 15 // 100)
            pad_y = max(12, (y2 - y1) * 15 // 100)
            left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
            right, bottom = min(width, x2 + pad_x), min(height, y2 + pad_y)
            if right <= left or bottom <= top:
                continue
            crop = frame[top:bottom, left:right]
            for item in self._decode_crop(crop):
                shifted = item.points + np.array([left, top], dtype=np.float32)
                found.append(QRDetection(item.value, shifted, f"yolo+{item.decoder}"))
        return self._deduplicate(found)

    def _decode_crop(self, crop: np.ndarray) -> list[QRDetection]:
        found = self._decode_image(crop)
        if found:
            return found
        longest_side = max(crop.shape[:2])
        if longest_side >= 640:
            return []
        scale = min(8, max(2, int(np.ceil(480 / max(1, longest_side)))))
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return [
            QRDetection(item.value, item.points / scale, f"{item.decoder}@{scale}x")
            for item in self._decode_image(enlarged)
        ]

    @staticmethod
    def _box_iou(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        intersection_width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
        intersection_height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
        intersection = intersection_width * intersection_height
        if not intersection:
            return 0.0
        left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
        right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
        return intersection / max(1, left_area + right_area - intersection)

    def _is_qr_class(self, result: object, class_id: int) -> bool:
        names = getattr(result, "names", None) or getattr(self.model, "names", None)
        if not isinstance(names, (dict, list, tuple)) or len(names) <= 1:
            return True
        try:
            name = names[class_id] if not isinstance(names, dict) else names.get(class_id, "")
        except (IndexError, KeyError, TypeError):
            return False
        normalized = str(name).lower().replace("_", "").replace("-", "").replace(" ", "")
        return "qr" in normalized or "qrcode" in normalized

    def _decode_image(self, image: np.ndarray) -> list[QRDetection]:
        found: list[QRDetection] = []
        try:
            import zxingcpp

            gray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image.ndim == 3
                else np.asarray(image, dtype=np.uint8)
            )
            variants = ((gray, "zxing"),)
            for candidate, decoder in variants:
                barcodes = zxingcpp.read_barcodes(
                    candidate,
                    formats=zxingcpp.BarcodeFormat.QRCode,
                    try_rotate=True,
                    try_downscale=True,
                    try_invert=True,
                )
                for barcode in barcodes:
                    value = barcode.text.strip()
                    if not value:
                        continue
                    position = barcode.position
                    corners = np.array(
                        [
                            [position.top_left.x, position.top_left.y],
                            [position.top_right.x, position.top_right.y],
                            [position.bottom_right.x, position.bottom_right.y],
                            [position.bottom_left.x, position.bottom_left.y],
                        ],
                        dtype=np.float32,
                    )
                    found.append(QRDetection(value, corners, decoder))
                if found:
                    break
            if not found:
                enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
                barcodes = zxingcpp.read_barcodes(
                    enhanced,
                    formats=zxingcpp.BarcodeFormat.QRCode,
                    try_rotate=True,
                    try_downscale=True,
                    try_invert=True,
                )
                for barcode in barcodes:
                    value = barcode.text.strip()
                    if not value:
                        continue
                    position = barcode.position
                    corners = np.array(
                        [
                            [position.top_left.x, position.top_left.y],
                            [position.top_right.x, position.top_right.y],
                            [position.bottom_right.x, position.bottom_right.y],
                            [position.bottom_left.x, position.bottom_left.y],
                        ],
                        dtype=np.float32,
                    )
                    found.append(QRDetection(value, corners, "zxing-clahe"))
        except (ImportError, RuntimeError, TypeError, ValueError):
            # OpenCV remains available when a ZXing wheel is unavailable.
            pass

        if found:
            return self._deduplicate(found)

        try:
            ok, values, points, _ = self.detector.detectAndDecodeMulti(image)
            if ok and points is not None:
                for value, corners in zip(values, points, strict=False):
                    if value.strip():
                        found.append(
                            QRDetection(
                                value.strip(), np.asarray(corners, dtype=np.float32), "opencv"
                            )
                        )
        except (cv2.error, ValueError):
            # Older OpenCV builds can expose a different multi-decode behavior.
            pass

        if found:
            return self._deduplicate(found)

        try:
            value, points, _ = self.detector.detectAndDecode(image)
        except cv2.error:
            return []
        if value.strip() and points is not None:
            found.append(QRDetection(value.strip(), np.asarray(points, dtype=np.float32), "opencv"))
            return found

        try:
            value, points, _ = self.detector.detectAndDecodeCurved(image)
        except (AttributeError, cv2.error):
            return []
        if value.strip() and points is not None:
            found.append(
                QRDetection(value.strip(), np.asarray(points, dtype=np.float32), "opencv-curved")
            )
        return found

    @staticmethod
    def _deduplicate(items: list[QRDetection]) -> list[QRDetection]:
        unique: dict[str, QRDetection] = {}
        for item in items:
            unique.setdefault(item.value, item)
        return list(unique.values())


def draw_qr_detections(frame: np.ndarray, detections: list[QRDetection]) -> None:
    for detection in detections:
        polygon = np.rint(detection.points).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [polygon], True, (0, 220, 0), 3)
        x, y = polygon[0, 0]
        label = detection.value if len(detection.value) <= 48 else detection.value[:45] + "..."
        cv2.putText(frame, label, (int(x), max(25, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
