from __future__ import annotations

import hashlib
import uuid

import numpy as np
import pytest

from roll_qr_scale.station_session import (
    AnalysisBindingMismatch,
    SessionConflictError,
    StationSessionRegistry,
)


def test_analysis_binding_is_immutable_and_backed_by_staged_jpeg(tmp_path) -> None:
    registry = StationSessionRegistry(tmp_path / "staging", ["station-01"])
    registry.configure_camera("station-01", "camera-01")
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    event_id = str(uuid.uuid4())

    binding = registry.stage(
        frame,
        event_id=event_id,
        station_id="station-01",
        camera_id="camera-01",
    )

    assert binding.staged_path.is_file()
    assert hashlib.sha256(binding.staged_path.read_bytes()).hexdigest() == binding.frame_sha256
    assert registry.validate(
        binding.analysis_id,
        event_id=event_id,
        station_id="station-01",
        camera_id="camera-01",
        frame_sha256=binding.frame_sha256,
        require_ready=False,
    ) == binding

    with pytest.raises(AnalysisBindingMismatch, match="không thuộc"):
        registry.stage(
            frame,
            event_id=str(uuid.uuid4()),
            station_id="station-01",
            camera_id="camera-02",
        )


def test_station_does_not_overwrite_unsaved_review_and_releases_after_save(tmp_path) -> None:
    registry = StationSessionRegistry(tmp_path / "staging", ["station-01"])
    registry.configure_camera("station-01", "camera-01")
    frame = np.full((480, 640, 3), 170, dtype=np.uint8)
    first = registry.stage(
        frame,
        event_id=str(uuid.uuid4()),
        station_id="station-01",
        camera_id="camera-01",
    )
    registry.mark_ready(first.analysis_id)

    with pytest.raises(SessionConflictError, match="chưa lưu"):
        registry.stage(
            frame,
            event_id=str(uuid.uuid4()),
            station_id="station-01",
            camera_id="camera-01",
        )

    registry.mark_saved(first.analysis_id)
    second = registry.stage(
        frame.copy(),
        event_id=str(uuid.uuid4()),
        station_id="station-01",
        camera_id="camera-01",
    )
    assert second.analysis_id != first.analysis_id


def test_registry_rejects_duplicate_configured_camera_identity(tmp_path) -> None:
    registry = StationSessionRegistry(tmp_path / "staging", ["station-01", "station-02"])
    registry.configure_camera("station-01", "camera-shared")
    with pytest.raises(ValueError, match="đã được gán"):
        registry.configure_camera("station-02", "camera-shared")


def test_registry_can_restore_active_binding_after_restart(tmp_path) -> None:
    staging = tmp_path / "staging"
    registry = StationSessionRegistry(staging, ["station-01"])
    registry.configure_camera("station-01", "camera-01")
    frame = np.full((480, 640, 3), 190, dtype=np.uint8)
    original = registry.stage(
        frame,
        event_id=str(uuid.uuid4()),
        station_id="station-01",
        camera_id="camera-01",
    )
    ready = registry.mark_ready(original.analysis_id)

    restored_registry = StationSessionRegistry(staging, ["station-01"])
    restored_registry.configure_camera("station-01", "camera-01")
    restored = restored_registry.restore_ready(ready)

    assert restored == ready
    assert restored_registry.statuses()[0]["state"] == "ready"
    assert restored_registry.validate(
        original.analysis_id,
        event_id=original.event_id,
        station_id=original.station_id,
        camera_id=original.camera_id,
        frame_sha256=ready.frame_sha256,
    ) == ready

    tampered = original.staged_path.with_name("tampered.jpg")
    tampered.write_bytes(b"not-a-jpeg")
    with pytest.raises(AnalysisBindingMismatch, match="Hash ảnh"):
        restored_registry.restore_ready(
            original.__class__(
                analysis_id=str(uuid.uuid4()),
                event_id=str(uuid.uuid4()),
                station_id="station-01",
                camera_id="camera-01",
                frame_sha256=ready.frame_sha256,
                staged_path=tampered,
                captured_at=ready.captured_at,
                created_at=ready.created_at,
                state="ready",
            )
        )
