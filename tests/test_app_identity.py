import uuid

import numpy as np

from roll_qr_scale.app import _save_current, build_parser
from roll_qr_scale.scale import WeightReading
from roll_qr_scale.storage import MeasurementStore


def test_parser_exposes_gateway_station_and_camera_identity() -> None:
    args = build_parser().parse_args(
        [
            "--gateway-id",
            "gateway-a",
            "--station-id",
            "station-02",
            "--camera-id",
            "camera-02",
        ]
    )
    assert (args.gateway_id, args.station_id, args.camera_id) == (
        "gateway-a",
        "station-02",
        "camera-02",
    )
    assert build_parser().parse_args(["--device-id", "legacy-gateway"]).gateway_id == (
        "legacy-gateway"
    )


def test_cli_retry_reuses_event_without_duplicate_row(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    reading = WeightReading(9.34, "kg", True, "OCR:9.34")
    event_id = str(uuid.uuid4())
    recent: dict[str, float] = {}

    first = _save_current(
        store,
        frame,
        "ROLL-MULTI-001",
        reading,
        "camera-ocr",
        "camera:zxing",
        False,
        None,
        recent,
        5,
        gateway_id="gateway-a",
        station_id="station-02",
        camera_id="camera-02",
        event_id=event_id,
    )
    retry = _save_current(
        store,
        frame,
        "ROLL-MULTI-001",
        reading,
        "camera-ocr",
        "camera:zxing",
        False,
        None,
        recent,
        5,
        gateway_id="gateway-a",
        station_id="station-02",
        camera_id="camera-02",
        event_id=event_id,
    )
    row = store.connection.execute(
        "SELECT event_id,gateway_id,station_id,camera_id FROM measurements"
    ).fetchone()
    count = store.count()
    store.close()

    assert first.startswith("SAVED #")
    assert "DUPLICATE" in retry
    assert count == 1
    assert tuple(row) == (event_id, "gateway-a", "station-02", "camera-02")
