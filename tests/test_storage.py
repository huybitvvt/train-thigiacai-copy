import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from roll_qr_scale.storage import EventIdConflictError, MeasurementStore


def test_saves_measurement_and_evidence_image(tmp_path) -> None:
    db_path = tmp_path / "measurements.db"
    store = MeasurementStore(db_path, tmp_path / "captures")
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    result = store.save("ROLL-001", 81.25, "kg", frame, "manual")
    store.close()

    assert result.id == 1
    assert result.qr_code == "ROLL-001"
    assert result.weight == 81.25
    assert result.sync_status == "local"
    assert len(result.frame_sha256) == 64
    assert len(result.payload_hash) == 64
    assert len(list((tmp_path / "captures").glob("*.jpg"))) == 1
    assert result.frame_sha256 == hashlib.sha256(Path(result.image_path).read_bytes()).hexdigest()

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT qr_code, weight, unit, weight_source, qr_source FROM measurements"
        ).fetchone()
    assert row == ("ROLL-001", 81.25, "kg", "manual", "camera")


def test_idempotent_save_persists_capture_identity_and_returns_duplicate(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    frame = np.full((32, 48, 3), 17, dtype=np.uint8)
    kwargs = {
        "event_id": "capture-event-001",
        "captured_at": "2026-08-02T05:06:07.123+00:00",
        "gateway_id": "gateway-a",
        "station_id": "station-a",
        "camera_id": "camera-2",
        "analysis_id": "analysis-abc",
    }

    first = store.save_idempotent("ROLL-002", 44.5, "kg", frame, "ocr", **kwargs)
    second = store.save_idempotent("ROLL-002", 44.5, "kg", frame, "ocr", **kwargs)

    assert first.duplicate is False
    assert second.duplicate is True
    assert second.existing == first.measurement
    assert store.count() == 1
    assert len(list((tmp_path / "captures").glob("*.jpg"))) == 1
    assert first.measurement.event_id == "capture-event-001"
    assert first.measurement.captured_at == "2026-08-02T05:06:07.123+00:00"
    assert first.measurement.gateway_id == "gateway-a"
    assert first.measurement.station_id == "station-a"
    assert first.measurement.camera_id == "camera-2"
    assert first.measurement.analysis_id == "analysis-abc"
    store.close()


def test_idempotent_save_rejects_reused_event_id_with_changed_payload(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    common = {
        "event_id": "capture-event-conflict",
        "captured_at": "2026-08-02T05:06:07Z",
        "gateway_id": "gateway-a",
        "station_id": "station-a",
        "camera_id": "camera-1",
        "analysis_id": "analysis-1",
    }
    store.save_idempotent("ROLL-003", 10.0, "kg", frame, "ocr", **common)

    with pytest.raises(EventIdConflictError):
        store.save_idempotent("ROLL-003", 10.1, "kg", frame, "ocr", **common)

    assert store.count() == 1
    assert len(list((tmp_path / "captures").glob("*.jpg"))) == 1
    store.close()


def test_existing_database_is_upgraded_without_losing_rows(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    capture_dir = tmp_path / "captures"
    capture_dir.mkdir()
    image_path = capture_dir / "legacy.jpg"
    image_path.write_bytes(b"legacy-jpeg-evidence")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                qr_code TEXT NOT NULL,
                weight REAL NOT NULL,
                unit TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                image_path TEXT NOT NULL,
                weight_source TEXT NOT NULL,
                sync_status TEXT NOT NULL DEFAULT 'local'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO measurements (
                event_id, qr_code, weight, unit, captured_at, image_path,
                weight_source, sync_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-event",
                "ROLL-LEGACY",
                9.5,
                "kg",
                "2026-08-01T00:00:00Z",
                str(image_path),
                "manual",
                "local",
            ),
        )

    store = MeasurementStore(db_path, capture_dir)
    legacy = store.get("legacy-event")
    assert legacy is not None
    assert legacy.qr_code == "ROLL-LEGACY"
    assert legacy.gateway_id == ""
    assert legacy.frame_sha256 == hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert len(legacy.payload_hash) == 64
    assert store.count() == 1
    store.close()
