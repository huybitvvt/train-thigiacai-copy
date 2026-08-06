from __future__ import annotations

import json

import numpy as np
import pytest

from roll_qr_scale.storage import EventIdConflictError, MeasurementStore
from tools import stress_test_multicamera as stress


def test_default_stress_distributes_100_events_and_recovers_offline_outbox(
    tmp_path,
) -> None:
    run_dir = tmp_path / "explicit-run"
    parser = stress.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

    args = parser.parse_args(["--run-dir", str(run_dir)])
    assert args.event_count == 100
    assert args.station_count == 3
    assert stress.run(args) == 0

    assert [path.name for path in run_dir.iterdir()] == [stress.REPORT_NAME]
    report = json.loads((run_dir / stress.REPORT_NAME).read_text(encoding="utf-8"))
    assert report["station_counts"] == {
        "station-01": 34,
        "station-02": 33,
        "station-03": 33,
    }
    assert report["row_count"] == 100
    assert report["unique_events"] == 100
    assert report["duplicate_retries"] == 100
    assert report["identity_conflicts_rejected"] == 1
    assert report["cross_identity_mismatches"] == 0
    assert report["offline_attempts"] == 100
    assert report["pending_before_recovery"] == 100
    assert report["synced_during_recovery"] == 100
    assert report["synced_after_recovery"] == 100
    assert report["pending_after_recovery"] == 0
    assert report["elapsed_seconds"] >= 0
    assert report["accepted"] is True


def test_same_event_retry_is_idempotent_and_cross_identity_retry_conflicts(
    tmp_path,
) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    frame = np.full((32, 48, 3), 127, dtype=np.uint8)
    values = {
        "qr_code": "ROLL-IDEMPOTENT-001",
        "weight": 12.5,
        "unit": "kg",
        "frame": frame,
        "weight_source": "test",
        "needs_sync": True,
        "qr_source": "synthetic",
        "weight_raw": "12.500",
        "weight_stable": True,
        "event_id": "event-idempotent-001",
        "captured_at": "2026-01-01T00:00:00.000+00:00",
        "gateway_id": "gateway-01",
        "station_id": "station-01",
        "camera_id": "camera-01",
        "analysis_id": "analysis-001",
    }
    try:
        first = store.save_idempotent(**values)
        retry = store.save_idempotent(**values)

        assert first.duplicate is False
        assert retry.duplicate is True
        assert retry.measurement.id == first.measurement.id
        assert store.count() == 1

        mismatched = dict(values)
        mismatched["station_id"] = "station-02"
        mismatched["camera_id"] = "camera-02"
        with pytest.raises(EventIdConflictError, match="different payload"):
            store.save_idempotent(**mismatched)
        assert store.count() == 1
    finally:
        store.close()
