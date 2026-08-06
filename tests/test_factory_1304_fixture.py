from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import pytest

from roll_qr_scale.qr_reader import QRReader
from roll_qr_scale.weight_ocr import CameraOCRWeightSource, detect_weight_roi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "factory_scale_13_04_reference.json"
EXPECTED_SHA256 = "2dc2d4dcf1db1262ef9eec8653c720e86fe65b1689de8276e83d6cf487510cf2"


def _load_reference() -> tuple[dict[str, object], Path, object]:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    image_path = METADATA_PATH.parent / str(metadata["image"])
    frame = cv2.imread(str(image_path))
    assert frame is not None, f"Cannot decode immutable fixture: {image_path}"
    return metadata, image_path, frame


def test_factory_1304_reference_integrity() -> None:
    metadata, image_path, frame = _load_reference()
    raw = image_path.read_bytes()

    assert metadata["sha256"] == EXPECTED_SHA256
    assert metadata["byte_length"] == 327487
    assert metadata["width"] == 1086
    assert metadata["height"] == 1448
    assert metadata["expected_qr_code"] == "MT- MN009-3107268353"
    assert metadata["expected_weight"] == 13.04
    assert metadata["unit"] == "kg"
    assert metadata["gross_roi_xyxy"] == [503, 1240, 564, 1274]
    assert "Operator-verified" in str(metadata["verification_note"])
    assert len(raw) == metadata["byte_length"]
    assert hashlib.sha256(raw).hexdigest() == metadata["sha256"]
    assert frame.shape == (metadata["height"], metadata["width"], 3)


def test_factory_1304_reference_qr_decodes_exactly() -> None:
    metadata, _, frame = _load_reference()

    detections = QRReader().decode(frame)

    assert [item.value for item in detections] == [metadata["expected_qr_code"]]


def test_factory_1304_reference_reads_top_gross_row() -> None:
    metadata, _, frame = _load_reference()
    located = detect_weight_roi(frame)
    assert located is not None

    reading = CameraOCRWeightSource(
        located[0],
        unit=str(metadata["unit"]),
        min_confidence=0.60,
        download_enabled=False,
    ).capture(frame)

    assert reading.value == pytest.approx(13.04)
    assert reading.stable
    assert reading.confidence is not None and reading.confidence >= 0.60
    assert "LEDCORE:13.04" in reading.raw
    assert "1304->13.04" in reading.raw
