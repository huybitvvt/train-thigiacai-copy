from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2

from roll_qr_scale.qr_reader import QRReader
from roll_qr_scale.weight_ocr import detect_weight_roi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "factory_scale_7_02_full_reference.json"
EXPECTED_SHA256 = "029b256cb7ac547bf4e938b34ba9be5e90b0957da8e3c7e65daed68a2943935b"
EXPECTED_QR = "MT- MN009-3107263087"


def _load_reference() -> tuple[dict[str, object], Path, object]:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    image_path = METADATA_PATH.parent / str(metadata["image"])
    frame = cv2.imread(str(image_path))
    assert frame is not None, f"Cannot decode immutable fixture: {image_path}"
    return metadata, image_path, frame


def _intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0, right - left) * max(0, bottom - top)


def _box_area(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def _metadata_box(metadata: dict[str, object], key: str) -> tuple[int, int, int, int]:
    values = metadata[key]
    assert isinstance(values, list) and len(values) == 4
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def test_factory_702_reference_integrity() -> None:
    metadata, image_path, frame = _load_reference()
    raw = image_path.read_bytes()

    assert metadata["sha256"] == EXPECTED_SHA256
    assert metadata["byte_length"] == 258127
    assert metadata["width"] == 1086
    assert metadata["height"] == 1448
    assert metadata["expected_qr_code"] == EXPECTED_QR
    assert metadata["expected_weight"] == 7.02
    assert metadata["unit"] == "kg"
    assert metadata["gross_bbox_xyxy"] == [510, 1171, 558, 1208]
    assert metadata["tare_bbox_xyxy"] == [503, 1213, 555, 1248]
    assert metadata["net_bbox_xyxy"] == [507, 1267, 555, 1291]
    assert "Operator-verified" in str(metadata["verification_note"])
    assert len(raw) == metadata["byte_length"]
    assert hashlib.sha256(raw).hexdigest() == metadata["sha256"]
    assert frame.shape == (metadata["height"], metadata["width"], 3)


def test_factory_702_reference_qr_decodes_exactly() -> None:
    metadata, _, frame = _load_reference()

    detections = QRReader().decode(frame)

    assert [item.value for item in detections] == [metadata["expected_qr_code"]]


def test_factory_702_auto_roi_keeps_the_top_gross_row() -> None:
    metadata, _, frame = _load_reference()
    located = detect_weight_roi(frame)

    assert located is not None
    roi, method = located
    detected = roi.pixels(frame)
    gross = _metadata_box(metadata, "gross_bbox_xyxy")
    tare = _metadata_box(metadata, "tare_bbox_xyxy")
    net = _metadata_box(metadata, "net_bbox_xyxy")

    gross_coverage = _intersection_area(detected, gross) / _box_area(gross)
    tare_coverage = _intersection_area(detected, tare) / _box_area(tare)
    net_coverage = _intersection_area(detected, net) / _box_area(net)
    detected_center_y = (detected[1] + detected[3]) / 2

    assert method == "red-led"
    assert gross_coverage >= 0.95
    assert gross[1] <= detected_center_y <= gross[3]
    assert tare_coverage < 0.15
    assert net_coverage == 0
    assert detected[3] < (tare[1] + tare[3]) / 2
