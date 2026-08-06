import cv2
import numpy as np

from roll_qr_scale.quality import assess_frame_quality


def test_quality_rejects_dark_low_resolution_blurry_frame() -> None:
    frame = np.full((240, 320, 3), 10, dtype=np.uint8)
    quality = assess_frame_quality(frame)
    assert not quality.accepted
    assert len(quality.issues) == 3


def test_quality_accepts_clear_well_lit_frame() -> None:
    frame = np.full((600, 800, 3), 150, dtype=np.uint8)
    cv2.putText(frame, "20.15", (80, 350), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 15)
    quality = assess_frame_quality(frame)
    assert quality.accepted
    assert quality.width == 800


def test_quality_can_ignore_only_low_resolution_for_cloud_reader() -> None:
    frame = np.full((240, 320, 3), 150, dtype=np.uint8)
    cv2.putText(frame, "7.02", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

    quality = assess_frame_quality(frame)
    payload = quality.as_dict(ignore_low_resolution=True)

    assert quality.accepted is False
    assert payload["accepted"] is True
    assert payload["issues"] == []
    assert payload["low_resolution_ignored"] is True
