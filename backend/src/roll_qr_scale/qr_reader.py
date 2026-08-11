from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class QRDetection:
    value: str
    points: np.ndarray
    decoder: str = "opencv"


class QRReader:
    """Localize QR labels with YOLO, then decode their contents with ZXing/OpenCV."""

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
            return detections or self._decode_image(frame)

        detections = self._decode_image(frame)
        if detections or self.model is None:
            return detections
        return self._decode_yolo_regions(frame)

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
        if longest_side >= 480:
            return []
        scale = min(4, max(2, int(np.ceil(320 / max(1, longest_side)))))
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return [
            QRDetection(item.value, item.points / scale, f"{item.decoder}@{scale}x")
            for item in self._decode_image(enlarged)
        ]

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
