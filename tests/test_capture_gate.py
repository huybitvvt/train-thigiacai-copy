import numpy as np

from roll_qr_scale.app import _save_current
from roll_qr_scale.scale import WeightReading
from roll_qr_scale.storage import MeasurementStore


def test_capture_requires_qr_and_stable_weight(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    recent: dict[str, float] = {}

    assert _save_current(
        store, frame, None, WeightReading(10, "kg", True), "serial", "scanner_hid",
        False, None, recent, 5,
    ) == "NOT SAVED: no QR"
    assert _save_current(
        store, frame, "ROLL-1", WeightReading(10, "kg", False), "serial", "scanner_hid",
        False, None, recent, 5,
    ) == "NOT SAVED: weight is unstable"
    store.close()


def test_capture_blocks_same_frame_but_allows_next_frame_for_same_qr(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    reading = WeightReading(125.4, "kg", True, "ST,GS,+125.4kg")
    recent: dict[str, float] = {}

    first = _save_current(
        store, frame, "ROLL-USB-001", reading, "serial", "scanner_hid",
        False, None, recent, 5,
    )
    second = _save_current(
        store, frame, "ROLL-USB-001", reading, "serial", "scanner_hid",
        False, None, recent, 5,
    )
    next_frame = frame.copy()
    next_frame[0, 0] = 1
    third = _save_current(
        store, next_frame, "ROLL-USB-001", reading, "serial", "scanner_hid",
        False, None, recent, 5,
    )
    rows = store.connection.execute(
        "SELECT qr_source, weight_raw, weight_stable FROM measurements"
    ).fetchall()
    store.close()

    assert first.startswith("SAVED #")
    assert second == "NOT SAVED: duplicate frame too soon"
    assert third.startswith("SAVED #")
    assert [tuple(row) for row in rows] == [
        ("scanner_hid", "ST,GS,+125.4kg", 1),
        ("scanner_hid", "ST,GS,+125.4kg", 1),
    ]
