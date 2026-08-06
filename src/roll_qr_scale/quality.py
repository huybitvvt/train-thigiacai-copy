from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    width: int
    height: int
    brightness: float
    sharpness: float
    dark_fraction: float
    bright_fraction: float
    issues: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.issues

    def as_dict(self, *, ignore_low_resolution: bool = False) -> dict[str, object]:
        result = asdict(self)
        issues = list(self.issues)
        if ignore_low_resolution:
            issues = [
                issue
                for issue in issues
                if not issue.startswith("Độ phân giải thấp (")
            ]
        result["accepted"] = not issues
        result["issues"] = issues
        result["low_resolution_ignored"] = bool(
            ignore_low_resolution and len(issues) != len(self.issues)
        )
        return result


def assess_frame_quality(
    frame: np.ndarray,
    *,
    min_width: int = 640,
    min_height: int = 480,
    min_brightness: float = 30.0,
    max_brightness: float = 225.0,
    min_sharpness: float = 35.0,
) -> FrameQuality:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Ảnh camera phải có 3 kênh màu")
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_fraction = float(np.mean(gray < 25))
    bright_fraction = float(np.mean(gray > 245))
    issues: list[str] = []
    if width < min_width or height < min_height:
        issues.append(f"Độ phân giải thấp ({width}×{height}); tối thiểu {min_width}×{min_height}")
    if brightness < min_brightness:
        issues.append("Ảnh quá tối; bổ sung đèn hoặc chỉnh phơi sáng")
    elif brightness > max_brightness:
        issues.append("Ảnh quá sáng; giảm đèn hoặc chỉnh phơi sáng")
    if sharpness < min_sharpness:
        issues.append("Ảnh mờ/rung; cố định camera và lấy nét lại")
    return FrameQuality(
        width=width,
        height=height,
        brightness=round(brightness, 2),
        sharpness=round(sharpness, 2),
        dark_fraction=round(dark_fraction, 4),
        bright_fraction=round(bright_fraction, 4),
        issues=tuple(issues),
    )
