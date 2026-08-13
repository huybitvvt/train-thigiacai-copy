import base64
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
import qrcode

import roll_qr_scale.test_ui as test_ui_module
from roll_qr_scale.scale import WeightReading
from roll_qr_scale.gemini_weight import GeminiWeightSuggestion
from roll_qr_scale.storage import MeasurementStore
from roll_qr_scale.sync import OutboxSyncWorker
from roll_qr_scale.test_ui import TEST_UI_HTML, StationUIService, decode_image
from roll_qr_scale.weight_ocr import NormalizedROI


def make_qr_frame(value: str) -> np.ndarray:
    qr = qrcode.make(value).convert("RGB").resize((360, 360))
    frame = np.full((600, 800, 3), 245, dtype=np.uint8)
    frame[120:480, 220:580] = cv2.cvtColor(np.asarray(qr), cv2.COLOR_RGB2BGR)
    return frame


def image_data_url(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def test_decode_image_accepts_browser_data_url() -> None:
    frame = make_qr_frame("ROLL-WEB-IMAGE")
    decoded = decode_image(image_data_url(frame))
    assert decoded.shape == frame.shape


def test_ui_capture_decodes_qr_and_saves_stable_manual_weight(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(store, None, None, None)
    result = service.capture("", 125.4, "kg", make_qr_frame("ROLL-WEB-001"))
    row = store.connection.execute(
        "SELECT qr_code,weight,unit,weight_source,qr_source,weight_stable,sync_status "
        "FROM measurements"
    ).fetchone()
    store.close()

    assert result["qr_code"] == "ROLL-WEB-001"
    assert result["sync_status"] == "local"
    assert tuple(row) == (
        "ROLL-WEB-001",
        125.4,
        "kg",
        "manual-test-ui",
        "camera:zxing",
        1,
        "local",
    )


def test_ui_save_waits_for_same_event_code_weight_and_image_cloud_ack(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    sent: list[tuple[dict[str, object], bytes]] = []

    def fake_send(url, payload, image_path, token):
        sent.append((dict(payload), Path(image_path).read_bytes()))
        return {
            "ok": True,
            "event_id": payload["event_id"],
            "id": 501,
            "image_url": "https://images.example/evidence.jpg",
            "image_public_id": "roll-captures/event",
        }

    worker = OutboxSyncWorker(
        store,
        "https://example.test/ingest",
        "device-token",
        "gateway-test",
        send=fake_send,
    )
    service = StationUIService(store, worker, None, None)

    result = service.capture(
        "PRODUCT-ENTRY-001",
        7.08,
        "kg",
        make_qr_frame("EVIDENCE-QR"),
    )

    service.close()
    store.close()
    assert result["sync_status"] == "synced"
    assert result["remote_id"] == 501
    assert result["remote_image_url"] == "https://images.example/evidence.jpg"
    assert len(sent) == 1
    assert sent[0][0]["event_id"] == result["event_id"]
    assert sent[0][0]["qr_code"] == "PRODUCT-ENTRY-001"
    assert sent[0][0]["weight"] == pytest.approx(7.08)
    assert sent[0][1].startswith(b"\xff\xd8")


def test_ui_save_reports_cloud_failure_but_keeps_complete_local_event(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")

    def fail_send(*args):
        raise OSError("network unavailable")

    worker = OutboxSyncWorker(
        store,
        "https://offline.test/ingest",
        "device-token",
        "gateway-test",
        send=fail_send,
    )
    service = StationUIService(store, worker, None, None)

    result = service.capture(
        "PRODUCT-OFFLINE-001",
        13.04,
        "kg",
        make_qr_frame("EVIDENCE-OFFLINE"),
    )
    saved = store.get(str(result["event_id"]))

    service.close()
    store.close()
    assert result["sync_status"] == "failed"
    assert "network unavailable" in str(result["sync_error"])
    assert result["pending_count"] == 1
    assert saved is not None
    assert saved.qr_code == "PRODUCT-OFFLINE-001"
    assert saved.weight == pytest.approx(13.04)
    assert Path(saved.image_path).is_file()


def test_ui_save_keeps_both_weights_and_both_images_in_one_event(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(store, None, None, None)
    core_frame = make_qr_frame("CORE-EVIDENCE")
    product_frame = make_qr_frame("PRODUCT-EVIDENCE")

    result = service.capture(
        "PRODUCT-001",
        1.04,
        "kg",
        core_frame,
        product_frame=product_frame,
        product_weight=13.04,
    )
    saved = store.get(str(result["event_id"]))

    service.close()
    store.close()
    assert saved is not None
    assert saved.qr_code == "PRODUCT-001"
    assert saved.weight == pytest.approx(1.04)
    assert saved.product_weight == pytest.approx(13.04)
    assert Path(saved.image_path).is_file()
    assert Path(saved.product_image_path).is_file()
    assert saved.image_path != saved.product_image_path


def test_ui_analyzes_qr_and_camera_weight_together(tmp_path, monkeypatch) -> None:
    class FakeOCRSource:
        def __init__(self, *args, reader=None, **kwargs):
            self._reader = reader or object()

        def capture(self, frame):
            return WeightReading(20.15, "kg", True, "OCR: 20.15@0.96", 0.96)

    monkeypatch.setattr(test_ui_module, "CameraOCRWeightSource", FakeOCRSource)
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(store, None, None, None)

    result = service.analyze(
        make_qr_frame("ROLL-CAMERA-001"),
        "0.4,0.7,0.6,0.9",
        "kg",
    )
    store.close()

    assert result["qr_code"] == "ROLL-CAMERA-001"
    assert result["qr_roi"] is not None
    assert result["weight"] == 20.15
    assert result["confidence"] == 0.96
    assert result["quality_pass"] is True


def test_ui_uses_camera_calibration_and_temporal_burst(tmp_path, monkeypatch) -> None:
    observed = {}

    class FakeOCRSource:
        def __init__(self, roi, *args, reader=None, **kwargs):
            observed["roi"] = roi
            self._reader = reader or object()

        def capture_many(self, frames):
            observed["frames"] = len(frames)
            return WeightReading(
                7.84,
                "kg",
                True,
                "TEMPORAL: agreement=7/9",
                0.88,
            )

    monkeypatch.setattr(test_ui_module, "CameraOCRWeightSource", FakeOCRSource)
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        station_count=1,
        station_ids=["station-01"],
        camera_ids=["camera-01"],
        weight_rois=["0.40,0.70,0.60,0.90"],
    )
    frame = make_qr_frame("ROLL-BURST-001")

    result = service.analyze(
        frame,
        "auto",
        "kg",
        event_id="event-burst-001",
        station_id="station-01",
        camera_id="camera-01",
        weight_frames=[frame.copy(), frame.copy()],
    )
    service.close()
    store.close()

    assert observed["frames"] == 3
    assert observed["roi"].x1 == pytest.approx(0.40)
    assert result["roi_method"] == "camera-calibrated"
    assert result["burst_frames"] == 3
    assert result["weight"] == pytest.approx(7.84)


@pytest.mark.parametrize(
    ("gemini_value", "expected_weight", "human_review"),
    ((7.84, 7.84, False), (1.84, None, True)),
)
def test_hybrid_accepts_only_independent_local_cloud_agreement(
    tmp_path,
    monkeypatch,
    gemini_value,
    expected_weight,
    human_review,
) -> None:
    class FakeOCRSource:
        def __init__(self, *args, reader=None, **kwargs):
            self._reader = reader or object()

        def capture_many(self, frames):
            return WeightReading(None, "kg", False, "LOCAL: strict consensus rejected")

        def candidate_reading(self):
            return WeightReading(
                7.84,
                "kg",
                False,
                "LOCAL CANDIDATE: 7.84kg; votes=2/3",
                0.91,
            )

        def crop(self, frame):
            return frame[10:30, 10:70]

    class FakeGeminiReader:
        def __init__(self):
            self.calls = 0

        def read(self, frames, *, unit):
            self.calls += 1
            assert len(frames) == 3
            return GeminiWeightSuggestion(
                gemini_value,
                unit,
                True,
                True,
                f"GEMINI:{gemini_value}",
                0.2,
            )

        def status(self):
            return {"enabled": True}

        def close(self):
            pass

    monkeypatch.setattr(test_ui_module, "CameraOCRWeightSource", FakeOCRSource)
    gemini = FakeGeminiReader()
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=gemini,
        weight_engine="hybrid",
    )
    frame = make_qr_frame("ROLL-HYBRID-001")

    result = service.analyze(
        frame,
        "0,0,1,1",
        "kg",
        weight_frames=[frame.copy(), frame.copy()],
    )
    service.close()
    store.close()

    assert gemini.calls == 1
    assert result["weight"] == expected_weight
    assert result["requires_human_review"] is human_review
    assert result["recognition_source"] == (
        "paddle-local+gemini" if expected_weight is not None else "none"
    )


def test_hybrid_skips_gemini_when_local_consensus_passes(tmp_path, monkeypatch) -> None:
    class FakeOCRSource:
        def __init__(self, *args, reader=None, **kwargs):
            self._reader = reader or object()

        def capture_many(self, frames):
            return WeightReading(7.84, "kg", True, "LOCAL: accepted", 0.96)

        def candidate_reading(self):
            return WeightReading(7.84, "kg", True, "LOCAL: accepted", 0.96)

    class ForbiddenGeminiReader:
        def read(self, frames, *, unit):
            raise AssertionError("Gemini must not run after local acceptance")

        def status(self):
            return {"enabled": True}

        def close(self):
            pass

    monkeypatch.setattr(test_ui_module, "CameraOCRWeightSource", FakeOCRSource)
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=ForbiddenGeminiReader(),
        weight_engine="hybrid",
    )
    frame = make_qr_frame("ROLL-LOCAL-001")

    result = service.analyze(
        frame,
        "0,0,1,1",
        "kg",
        weight_frames=[frame.copy(), frame.copy()],
    )
    service.close()
    store.close()

    assert result["weight"] == pytest.approx(7.84)
    assert result["gemini_used"] is False
    assert result["recognition_source"] == "paddle-local"


@pytest.mark.parametrize(("gemini_value", "expected_weight"), ((7.84, 7.84), (None, None)))
def test_gemini_primary_reads_same_camera_burst_without_paddle(
    tmp_path,
    monkeypatch,
    gemini_value,
    expected_weight,
) -> None:
    class FakeGeminiReader:
        def __init__(self):
            self.calls = 0

        def read(self, frames, *, unit):
            self.calls += 1
            assert len(frames) == 3
            assert all(frame.shape[:2] == (600, 800) for frame in frames)
            return GeminiWeightSuggestion(
                gemini_value,
                unit,
                gemini_value is not None,
                gemini_value is not None,
                "GEMINI:test",
                0.25,
            )

        def status(self):
            return {"enabled": True}

        def close(self):
            pass

    monkeypatch.setattr(
        test_ui_module.PaddleOCRTextReader,
        "create",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Paddle must not load")),
    )
    gemini = FakeGeminiReader()
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=gemini,
        weight_engine="gemini",
    )
    service.start_ocr_preload()
    frame = make_qr_frame("ROLL-GEMINI-PRIMARY")

    result = service.analyze(
        frame,
        "0,0,0.1,0.1",
        "kg",
        weight_frames=[frame.copy(), frame.copy()],
    )
    service.close()
    store.close()

    assert gemini.calls == 1
    assert result["weight"] == expected_weight
    assert result["recognition_source"] == (
        "gemini-primary" if expected_weight is not None else "none"
    )
    assert result["gemini_used"] is True
    assert result["confidence"] is None
    assert result["gemini_input_tokens"] == 0
    assert result["gemini_output_tokens"] == 0
    assert result["gemini_thinking_tokens"] == 0
    assert result["gemini_total_tokens"] == 0
    assert result["roi"] is None
    assert result["roi_method"] == "gemini-full-frame"
    assert service.status()["weight_engine"] == "gemini"
    assert service.status()["ocr_ready"] is False


def test_codex_can_be_selected_without_replacing_gemini(tmp_path) -> None:
    class FakeGeminiReader:
        model = "gemini-test"

        def read(self, frames, *, unit):
            raise AssertionError("Gemini must not run when Codex is selected")

        def status(self):
            return {"enabled": True, "model": self.model}

        def close(self):
            pass

    class FakeCodexReader:
        def read(self, frames, *, unit):
            return GeminiWeightSuggestion(
                13.04,
                unit,
                True,
                True,
                "CODEX:13.04; auth=ChatGPT",
                0.5,
            )

        def status(self):
            return {
                "enabled": True,
                "installed": True,
                "authenticated": True,
                "available": True,
            }

        def close(self):
            pass

    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    gemini = FakeGeminiReader()
    codex = FakeCodexReader()
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=gemini,
        codex_reader=codex,
        weight_engine="gemini",
    )

    result = service.analyze(
        make_qr_frame("ROLL-CODEX-001"),
        "auto",
        "kg",
        recognition_provider="codex",
    )
    status = service.status()
    service.close()
    store.close()

    assert result["weight"] == pytest.approx(13.04)
    assert result["recognition_provider"] == "codex"
    assert result["recognition_source"] == "codex-primary"
    assert result["codex_used"] is True
    assert result["gemini_used"] is False
    assert status["recognition_providers"]["gemini"]["available"] is True
    assert status["recognition_providers"]["codex"]["available"] is True


def test_gemini_primary_uses_one_full_image_for_file_and_camera(
    tmp_path,
) -> None:
    class FakeGeminiReader:
        def __init__(self):
            self.calls = 0

        def read(self, frames, *, unit):
            self.calls += 1
            assert len(frames) == 1
            assert frames[0].shape[:2] == (600, 800)
            return GeminiWeightSuggestion(7.84, unit, True, True, "GEMINI:test", 0.2)

        def status(self):
            return {"enabled": True}

        def close(self):
            pass

    gemini = FakeGeminiReader()
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=gemini,
        weight_engine="gemini",
    )
    frame = make_qr_frame("ROLL-GEMINI-STILL")

    result = service.analyze(frame, "0,0,0.1,0.1", "kg")
    assert result["weight"] == pytest.approx(7.84)
    assert result["burst_frames"] == 1
    assert "single full-image accepted" in result["weight_raw"]

    camera_result = service.analyze(
        frame,
        "0,0,0.1,0.1",
        "kg",
        require_temporal=True,
    )
    assert camera_result["weight"] == pytest.approx(7.84)
    assert camera_result["burst_frames"] == 1
    assert "single full-image accepted" in camera_result["weight_raw"]

    service.close()
    store.close()
    assert gemini.calls == 2


def test_gemini_allows_successful_low_resolution_full_image(tmp_path) -> None:
    class FakeGeminiReader:
        def read(self, frames, *, unit):
            assert len(frames) == 1
            assert frames[0].shape[:2] == (240, 320)
            return GeminiWeightSuggestion(
                7.02,
                unit,
                True,
                True,
                "GEMINI_FULL:test",
                0.2,
                qr_code="ROLL-LOW-RES",
                qr_readable=True,
            )

        def status(self):
            return {"enabled": True}

        def close(self):
            pass

    frame = np.full((240, 320, 3), 150, dtype=np.uint8)
    cv2.putText(frame, "7.02", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=FakeGeminiReader(),
        weight_engine="gemini",
    )

    result = service.analyze(frame, "auto", "kg")
    saved = service.capture(
        "ROLL-LOW-RES",
        7.02,
        "kg",
        frame,
        vision_confirmed=True,
        weight_raw="GEMINI_FULL:test",
    )

    service.close()
    store.close()
    assert result["qr_code"] == "ROLL-LOW-RES"
    assert result["weight"] == pytest.approx(7.02)
    assert result["quality_pass"] is True
    assert result["quality"]["issues"] == []
    assert result["quality"]["low_resolution_ignored"] is True
    assert saved["qr_code"] == "ROLL-LOW-RES"


def test_gemini_full_frame_supplies_qr_when_local_decoder_misses(tmp_path) -> None:
    class FakeGeminiReader:
        def read(self, frames, *, unit):
            return GeminiWeightSuggestion(
                9.34,
                unit,
                True,
                True,
                "GEMINI_FULL:test",
                0.2,
                qr_code="ROLL-CLOUD-QR",
                qr_readable=True,
            )

        def status(self):
            return {"enabled": True}

        def close(self):
            pass

    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=FakeGeminiReader(),
        weight_engine="gemini",
    )

    result = service.analyze(np.full((600, 800, 3), 180, dtype=np.uint8), "auto", "kg")

    service.close()
    store.close()
    assert result["qr_found"] is True
    assert result["qr_code"] == "ROLL-CLOUD-QR"
    assert result["qr_decoder"] == "gemini-full-frame"
    assert result["weight"] == pytest.approx(9.34)


def test_gemini_full_frame_keeps_local_qr_on_cloud_conflict(tmp_path) -> None:
    class FakeGeminiReader:
        def read(self, frames, *, unit):
            return GeminiWeightSuggestion(
                9.34,
                unit,
                True,
                True,
                "GEMINI_FULL:test",
                0.2,
                qr_code="ROLL-DIFFERENT",
                qr_readable=True,
            )

        def status(self):
            return {"enabled": True}

        def close(self):
            pass

    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=FakeGeminiReader(),
        weight_engine="gemini",
    )

    result = service.analyze(make_qr_frame("ROLL-LOCAL"), "auto", "kg")

    service.close()
    store.close()
    assert result["qr_found"] is True
    assert result["qr_code"] == "ROLL-LOCAL"
    assert result["qr_decoder"].endswith("+gemini-conflict-local-kept")
    assert "kept checksum-validated local QR" in result["weight_raw"]


def test_render_capture_crops_detected_led_before_gemini(tmp_path, monkeypatch) -> None:
    class FakeGeminiReader:
        def read(self, frames, *, unit):
            assert len(frames) == 1
            assert frames[0].shape[0] < 200
            assert frames[0].shape[1] < 300
            return GeminiWeightSuggestion(13.04, unit, True, True, "GEMINI:test", 0.2)

        def status(self):
            return {"enabled": True}

        def close(self):
            pass

    monkeypatch.setattr(
        test_ui_module,
        "detect_weight_roi",
        lambda frame: (NormalizedROI(0.4, 0.7, 0.6, 0.8), "red-led"),
    )
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=FakeGeminiReader(),
        weight_engine="gemini",
    )

    result = service.analyze(
        np.full((600, 800, 3), 180, dtype=np.uint8),
        "auto",
        "kg",
        capture_kind="core",
    )

    service.close()
    store.close()
    assert result["weight"] == pytest.approx(13.04)
    assert result["gemini_crop_applied"] is True
    assert result["gemini_attempts"] == 1
    assert result["gemini_fallback_used"] is False
    assert result["roi_method"] == "gemini-crop-red-led"


def test_core_capture_skips_unrelated_qr_decode(tmp_path, monkeypatch) -> None:
    class FakeGeminiReader:
        def read(self, frames, *, unit):
            return GeminiWeightSuggestion(7.02, unit, True, True, "GEMINI:test", 0.1)

        def status(self):
            return {"enabled": True}

        def close(self):
            pass

    monkeypatch.setattr(test_ui_module, "detect_weight_roi", lambda frame: None)
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=FakeGeminiReader(),
        weight_engine="gemini",
    )
    monkeypatch.setattr(
        service,
        "_decode_qr",
        lambda frame: (_ for _ in ()).throw(AssertionError("core must not decode QR")),
    )

    result = service.analyze(
        np.full((600, 800, 3), 180, dtype=np.uint8),
        "auto",
        "kg",
        capture_kind="core",
    )

    service.close()
    store.close()
    assert result["weight"] == pytest.approx(7.02)
    assert result["qr_found"] is False
    assert result["qr_decoder"] == "not-requested-core-step"


def test_unreadable_gemini_crop_retries_one_full_frame(tmp_path, monkeypatch) -> None:
    class FakeGeminiReader:
        def __init__(self):
            self.shapes = []

        def read(self, frames, *, unit):
            self.shapes.append(frames[0].shape[:2])
            if len(self.shapes) == 1:
                return GeminiWeightSuggestion(
                    None,
                    unit,
                    False,
                    True,
                    "GEMINI_FULL:weight-unreadable",
                    0.2,
                    input_tokens=100,
                    output_tokens=10,
                    total_tokens=110,
                )
            return GeminiWeightSuggestion(
                13.04,
                unit,
                True,
                True,
                "GEMINI_FULL:13.04",
                0.3,
                input_tokens=200,
                output_tokens=20,
                total_tokens=220,
            )

        def status(self):
            return {"enabled": True}

        def close(self):
            pass

    monkeypatch.setattr(
        test_ui_module,
        "detect_weight_roi",
        lambda frame: (NormalizedROI(0.4, 0.7, 0.6, 0.8), "red-led"),
    )
    reader = FakeGeminiReader()
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=reader,
        weight_engine="gemini",
    )

    result = service.analyze(
        np.full((600, 800, 3), 180, dtype=np.uint8),
        "auto",
        "kg",
        capture_kind="core",
    )

    service.close()
    store.close()
    assert reader.shapes[0][0] < 200
    assert reader.shapes[0][1] < 300
    assert reader.shapes[1] == (600, 800)
    assert result["weight"] == pytest.approx(13.04)
    assert result["gemini_attempts"] == 2
    assert result["gemini_fallback_used"] is True
    assert result["gemini_latency_seconds"] == pytest.approx(0.5)
    assert result["gemini_input_tokens"] == 300
    assert result["gemini_output_tokens"] == 30
    assert result["gemini_total_tokens"] == 330
    assert result["roi_method"] == "gemini-crop-red-led+full-frame-retry"
    assert "CROP ATTEMPT" in result["weight_raw"]
    assert "FULL FRAME RETRY" in result["weight_raw"]


def test_gemini_crop_does_not_retry_network_error(tmp_path, monkeypatch) -> None:
    class FakeGeminiReader:
        def __init__(self):
            self.calls = 0

        def read(self, frames, *, unit):
            self.calls += 1
            return GeminiWeightSuggestion(
                None,
                unit,
                False,
                False,
                "GEMINI ERROR: timeout",
                10.0,
            )

        def status(self):
            return {"enabled": True}

        def close(self):
            pass

    monkeypatch.setattr(
        test_ui_module,
        "detect_weight_roi",
        lambda frame: (NormalizedROI(0.4, 0.7, 0.6, 0.8), "red-led"),
    )
    reader = FakeGeminiReader()
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=reader,
        weight_engine="gemini",
    )

    result = service.analyze(
        np.full((600, 800, 3), 180, dtype=np.uint8),
        "auto",
        "kg",
        capture_kind="core",
    )

    service.close()
    store.close()
    assert reader.calls == 1
    assert result["weight_found"] is False
    assert result["gemini_attempts"] == 1
    assert result["gemini_fallback_used"] is False


def test_browser_qr_is_accepted_but_decoder_conflict_requires_manual_code(tmp_path) -> None:
    class FakeGeminiReader:
        def read(self, frames, *, unit):
            return GeminiWeightSuggestion(9.34, unit, True, True, "GEMINI:test", 0.2)

        def status(self):
            return {"enabled": True}

        def close(self):
            pass

    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=FakeGeminiReader(),
        weight_engine="gemini",
    )

    browser_only = service.analyze(
        np.full((600, 800, 3), 180, dtype=np.uint8),
        "auto",
        "kg",
        capture_kind="product",
        client_qr_code="SP-BROWSER-001",
    )
    conflict = service.analyze(
        make_qr_frame("SP-SERVER-001"),
        "auto",
        "kg",
        capture_kind="product",
        client_qr_code="SP-BROWSER-002",
    )

    service.close()
    store.close()
    assert browser_only["qr_code"] == "SP-BROWSER-001"
    assert browser_only["qr_decoder"] == "browser-barcode-detector"
    assert browser_only["qr_conflict"] is False
    assert conflict["qr_found"] is False
    assert conflict["qr_code"] is None
    assert conflict["qr_conflict"] is True


def test_gemini_accurate_profile_uses_accurate_reader(tmp_path) -> None:
    class FakeGeminiReader:
        def __init__(self, model, value):
            self.model = model
            self.value = value
            self.calls = 0

        def read(self, frames, *, unit):
            self.calls += 1
            return GeminiWeightSuggestion(
                self.value,
                unit,
                True,
                True,
                f"GEMINI_FULL:{self.model}",
                0.2,
                qr_code="ROLL-PROFILE",
                qr_readable=True,
            )

        def status(self):
            return {"enabled": True, "model": self.model}

        def close(self):
            pass

    fast = FakeGeminiReader("gemini-3.5-flash-lite", 7.02)
    accurate = FakeGeminiReader("gemini-3.1-pro-preview", 13.04)
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=fast,
        gemini_accurate_reader=accurate,
        weight_engine="gemini",
    )

    result = service.analyze(
        make_qr_frame("ROLL-PROFILE"),
        "auto",
        "kg",
        recognition_profile="accurate",
    )

    status = service.status()
    service.close()
    store.close()
    assert result["weight"] == pytest.approx(13.04)
    assert result["recognition_profile"] == "accurate"
    assert fast.calls == 0
    assert accurate.calls == 1
    assert status["recognition_profiles"]["fast"]["model"] == "gemini-3.5-flash-lite"
    assert status["recognition_profiles"]["accurate"]["model"] == "gemini-3.1-pro-preview"


def test_ui_capture_blocks_same_frame_but_allows_consecutive_new_frames(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(store, None, None, None, duplicate_window=5)
    frame = make_qr_frame("ROLL-WEB-DUPLICATE")
    service.capture("ROLL-WEB-DUPLICATE", 20, "kg", frame)
    with pytest.raises(ValueError, match="khung hình mới"):
        service.capture("ROLL-WEB-DUPLICATE", 20, "kg", frame)
    next_frame = frame.copy()
    next_frame[0, 0] = 0
    result = service.capture("ROLL-WEB-DUPLICATE", 20, "kg", next_frame)
    assert result["event_id"]
    assert store.connection.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 2
    store.close()


def test_ui_rejects_invalid_weight_and_image(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(store, None, None, None)
    with pytest.raises(ValueError, match="không âm"):
        service.capture("ROLL-WEB-BAD", -1, "kg", make_qr_frame("ROLL-WEB-BAD"))
    with pytest.raises(ValueError, match="base64"):
        decode_image("not-base64")
    store.close()


def test_ui_has_capture_controls_without_lookup_panel() -> None:
    for control_id in (
        'id="captureFile"',
        'id="captureFileBtn"',
        'id="analyzeCoreBtn"',
        'id="analyzeProductBtn"',
        'id="roiBox"',
        'id="qrBox"',
        'id="roiValue"',
        'id="syncBadge"',
        'id="geminiBadge"',
        'id="weightModeBadge"',
        'id="factoryBtn"',
    ):
        assert control_id in TEST_UI_HTML
    for removed_control in (
        'id="lookupQr"',
        'id="lookupFile"',
        'id="lookupCameraBtn"',
        '<aside class="card lookup-card">',
        '2. Quét lại QR để tra cứu',
    ):
        assert removed_control not in TEST_UI_HTML
    assert "MỘT CAMERA · QR + CÂN" in TEST_UI_HTML
    assert "ĐÃ LƯU LẦN " in TEST_UI_HTML
    assert "ĐỒNG BỘ SUPABASE: BẬT" in TEST_UI_HTML
    assert "analyzeCurrent()" in TEST_UI_HTML
    assert "SẴN SÀNG CHỤP TIẾP" in TEST_UI_HTML
    assert "prepareNextCapture(data.qr_code)" in TEST_UI_HTML


def test_ui_uses_viet_nhat_red_black_roboto_branding() -> None:
    assert "Trạm cân <span>Ai</span> Việt Nhật IPT" in TEST_UI_HTML
    assert '<img class="brand-mark" src="/logo.jpg" alt="Việt Nhật IPT">' in TEST_UI_HTML
    assert "font-family:Roboto" in TEST_UI_HTML
    assert 'local("Roboto Regular")' in TEST_UI_HTML
    assert "--primary:#d71920" in TEST_UI_HTML
    assert "fonts.googleapis.com" not in TEST_UI_HTML


def test_ui_uses_camera_left_params_right_capture_layout() -> None:
    assert "main{width:100%;margin:18px 0;padding:0 18px 24px;display:block}" in TEST_UI_HTML
    assert "grid-template-columns:minmax(460px,500px) minmax(0,1fr)" in TEST_UI_HTML
    assert "aspect-ratio:1/1" in TEST_UI_HTML
    assert "width:min(100%,480px)" in TEST_UI_HTML
    assert 'class="capture-left"' in TEST_UI_HTML
    assert 'class="capture-right"' in TEST_UI_HTML
    assert '<aside class="card lookup-card">' not in TEST_UI_HTML
    assert "main{width:100%" in TEST_UI_HTML


def test_ui_records_table_shows_bi_and_nvl_weights() -> None:
    assert "Trọng lượng bì" in TEST_UI_HTML
    assert "Trọng lượng NVL" in TEST_UI_HTML
    assert "function biWeightFromRaw(" in TEST_UI_HTML
    assert "function nvlWeight(" in TEST_UI_HTML
    assert "product-core-bi" in TEST_UI_HTML
    assert 'colspan="9"' in TEST_UI_HTML
    assert 'id="sourceShift"' in TEST_UI_HTML
    assert 'HC1 · 06:00–14:00' in TEST_UI_HTML
    assert '12C2 · 18:00–06:00' in TEST_UI_HTML
    assert 'id="sourceMachine"' in TEST_UI_HTML
    assert 'Máy tái chế' in TEST_UI_HTML
    assert 'Máy cách nhiệt' in TEST_UI_HTML
    assert 'id="sourceOrder"' in TEST_UI_HTML
    assert 'id="biWeight"' in TEST_UI_HTML
    assert 'value="0.16"' in TEST_UI_HTML
    assert "Lệnh sản xuất" in TEST_UI_HTML
    assert "SOURCE_PRODUCTION_ORDER=" in TEST_UI_HTML
    assert "BI_WEIGHT=" in TEST_UI_HTML
    assert "production_order:sourceContext.order" in TEST_UI_HTML
    from roll_qr_scale.test_ui import _merge_source_tags
    merged = _merge_source_tags(
        "PRODUCT_WEIGHT=1.2",
        {
            "work_date": "2026-08-08",
            "shift": "HC1",
            "machine": "Máy Bao Bì",
            "production_order": "LSX-01",
            "bi_weight": 0.16,
        },
    )
    assert "SOURCE_SHIFT=HC1" in merged
    assert "SOURCE_MACHINE=Máy Bao Bì" in merged
    assert "SOURCE_PRODUCTION_ORDER=LSX-01" in merged
    assert "BI_WEIGHT=0.16" in merged


def test_remote_product_image_does_not_request_redundant_signed_url() -> None:
    source = Path(test_ui_module.__file__).read_text(encoding="utf-8")
    assert "if not product_url and isinstance(product_path, str) and product_path:" in source


def test_panel_region_configuration_is_persisted_and_validated(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        station_count=1,
        station_ids=["station-01"],
        camera_ids=["camera-01"],
    )
    regions = [
        {"label": "TEMP 1", "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4},
        {"label": "HEAD 1", "x1": 0.5, "y1": 0.2, "x2": 0.7, "y2": 0.4},
    ]

    assert service.save_panel_regions("station-01", regions) == regions
    assert service.panel_regions("station-01") == regions
    with pytest.raises(ValueError, match="Tên vùng"):
        service.save_panel_regions("station-01", [regions[0], {**regions[1], "label": "temp 1"}])
    service.close()
    store.close()


def test_panel_analysis_crops_all_regions_and_calls_gemini_once(tmp_path) -> None:
    class FakePanelReader:
        def __init__(self):
            self.calls = []

        def read_panel_regions(self, regions):
            self.calls.append(regions)
            return {
                "ok": True,
                "readings": [
                    {"label": label, "readable": True, "value": str(index + 1)}
                    for index, (label, _) in enumerate(regions)
                ],
            }

        def status(self):
            return {"enabled": True}

        def close(self):
            pass

    reader = FakePanelReader()
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gemini_reader=reader,
        weight_engine="gemini",
    )
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    result = service.analyze_panel_regions(
        frame,
        [
            {"label": "A", "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4},
            {"label": "B", "x1": 0.5, "y1": 0.5, "x2": 0.9, "y2": 0.8},
        ],
    )

    assert result["readings"][0]["label"] == "A"
    assert len(reader.calls) == 1
    assert reader.calls[0][0][1].shape[:2] == (40, 80)
    assert reader.calls[0][1][1].shape[:2] == (60, 160)
    service.close()
    store.close()


def test_product_capture_uses_detected_qr_as_product_code() -> None:
    assert "session.qr=data.qr_code||''" not in TEST_UI_HTML
    assert "if(isProduct){session.productAnalysis=data" in TEST_UI_HTML
    assert "reliableQr=Boolean(data.qr_found&&!data.qr_conflict&&!qrDecoder.startsWith('gemini'))" in TEST_UI_HTML
    assert "if(reliableQr&&String(data.qr_code||'').trim()&&!String(session.qr||'').trim())session.qr=String(data.qr_code).trim()" in TEST_UI_HTML
    assert "$('analyzeProductBtn').disabled=panelMode||busy||!ready||!coreReady(session)" in TEST_UI_HTML
    assert "function coreCaptured(session)" in TEST_UI_HTML
    assert "function sourceReady(session)" in TEST_UI_HTML
    assert "if(isProduct&&!coreReady(session))" in TEST_UI_HTML
    assert "session._analyzeLock=false;renderControls();status(captureStatus,error.message" in TEST_UI_HTML
    assert "if(eventId){await api('/api/session/discard'" in TEST_UI_HTML
    assert "if(!isProduct&&session.coreAnalysis&&session.analysisId)" in TEST_UI_HTML
    assert 'id="analyzeCoreBtn"' in TEST_UI_HTML
    assert 'id="analyzeProductBtn"' in TEST_UI_HTML
    assert 'id="productWeight"' in TEST_UI_HTML
    assert "analyzeCurrent('core')" in TEST_UI_HTML
    assert "analyzeCurrent('product')" in TEST_UI_HTML
    assert "PRODUCT_WEIGHT=" in TEST_UI_HTML
    assert "function productReady(session)" in TEST_UI_HTML
    assert "ĐÃ CÂN LÕI · CHỜ CÂN SẢN PHẨM" in TEST_UI_HTML
    assert "recognitionProvider.value==='codex'?'Codex':'AI'" in TEST_UI_HTML
    assert "function showPostCaptureSource(session)" in TEST_UI_HTML
    assert "showPostCaptureSource(session);" in TEST_UI_HTML
    assert "showCapturedBlank" not in TEST_UI_HTML
    assert "function cameraVideoConstraints()" in TEST_UI_HTML
    assert "function tuneCameraTrack(track)" in TEST_UI_HTML
    assert "initialCameraPermissionCheck&&stations.some(session=>session.deviceId)" in TEST_UI_HTML
    assert "function decodeClientQr(canvas)" in TEST_UI_HTML
    assert "client_qr_code:clientQr" in TEST_UI_HTML
    assert "CAPTURE_MAX_EDGE=1440" in TEST_UI_HTML
    assert "setInterval(loadRecords,15000)" in TEST_UI_HTML
    assert "appStatus.release||'local'" in TEST_UI_HTML
    assert 'id="panelModeBtn"' in TEST_UI_HTML
    assert 'id="drawPanelRegionBtn"' in TEST_UI_HTML
    assert 'id="scanPanelBtn"' in TEST_UI_HTML
    assert "'/api/panel/analyze'" in TEST_UI_HTML
    assert "'/api/panel/regions'" in TEST_UI_HTML
    assert "function panelPointerDown(" in TEST_UI_HTML
    assert "function scanPanelRegions(" in TEST_UI_HTML
    assert "Mỗi vùng chỉ ôm đúng một hàng số đang sáng" in TEST_UI_HTML
    assert "loadPanelRegions(session).then" in TEST_UI_HTML


def test_ui_enables_local_yolo_model_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert test_ui_module.build_parser().parse_args([]).yolo_model is None
    model = tmp_path / "models" / "qr_demo_synthetic.pt"
    model.parent.mkdir()
    model.write_bytes(b"test")
    args = test_ui_module.build_parser().parse_args([])
    assert args.yolo_model == "models/qr_demo_synthetic.pt"
    assert args.yolo_mode == "fallback"
    assert args.ocr_min_confidence == pytest.approx(0.60)
    assert args.diagnostic_image is None


def test_multistation_defaults_and_html_controls(monkeypatch) -> None:
    monkeypatch.delenv("ROLL_SCALE_GATEWAY_ID", raising=False)
    monkeypatch.delenv("ROLL_SCALE_DEVICE_ID", raising=False)
    monkeypatch.delenv("ROLL_SCALE_WEIGHT_ENGINE", raising=False)
    monkeypatch.delenv("ROLL_SCALE_GEMINI_API_KEY", raising=False)
    args = test_ui_module.build_parser().parse_args([])
    assert args.station_count == 1
    assert args.gateway_id == "gateway-01"
    assert args.station_ids is None
    assert args.camera_ids is None
    assert args.weight_rois is None
    assert args.weight_burst_frames == 5
    assert args.weight_engine == "local"
    assert args.gemini_fallback is False
    assert args.gemini_timeout == pytest.approx(10.0)
    assert args.gemini_model == "gemini-3.5-flash-lite"
    assert args.gemini_accurate_model == "gemini-3.1-pro-preview"
    assert args.gemini_accurate_timeout == pytest.approx(30.0)
    assert args.codex_enabled is True
    assert args.codex_mode == "auto"
    assert args.codex_command == "codex"
    assert args.codex_model == ""
    assert args.codex_timeout == pytest.approx(60.0)
    assert args.auto_advance is True
    for marker in (
        'id="stationGrid"',
        'id="cameraSelect1"',
        'id="autoAdvance"',
        'id="discardBtn"',
        "class CameraSession extends StationSession",
        "navigator.mediaDevices.enumerateDevices()",
        "deviceId:{exact:requested}",
        "localStorage.getItem",
        "crypto.randomUUID",
        "data.event_id!==requestEventId",
        "event.key==='Enter'",
        "['1','2','3']",
        "refreshCameraDevices(true)",
        "ensureCamerasForSelect()",
        "ensureCameraPermission()",
        "session.stream!==stream||session.streamGeneration!==generation",
        "ĐỦ DỮ LIỆU · ",
        "prepareNextCapture('',session)",
        "'awaiting-code'",
        "'awaiting-weight'",
        "function completionReady(session)",
        "function sourceReady(session)",
        "function productReady(session)",
        "ĐÃ CÂN LÕI · CHỜ CÂN SẢN PHẨM",
        'id="analyzeCoreBtn"',
        'id="analyzeProductBtn"',
        'id="productWeight"',
        "analyzeCurrent('product')",
        "Nhanh · Flash-Lite · 10s",
        "Chính xác · Pro · 30s",
        "savedStationIndex=stations.indexOf(session)",
        "captureEditor=event.target===captureQr||event.target===weight||event.target===productWeight",
        "scheduleReconnect(session,session.deviceId)",
        "this.hydratedPending=Boolean(config.event_id)",
        "function pollPendingSessions()",
        "Chọn camera hoặc ảnh thực tế, rồi chụp cân lõi / cân sản phẩm.",
        "session.deviceId&&!session.hasUnsavedReview()",
        "weight_frames:weightFrames",
        "captureWeightBurst(session)",
        "event.key==='Backspace'&&!typing",
        "discardCurrent(false)",
    ):
        assert marker in TEST_UI_HTML


def test_parser_auto_selects_gemini_only_when_key_exists_and_engine_is_omitted(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ROLL_SCALE_WEIGHT_ENGINE", raising=False)
    monkeypatch.setenv("ROLL_SCALE_GEMINI_API_KEY", "configured-test-key")
    assert test_ui_module.build_parser().parse_args([]).weight_engine == "gemini"

    monkeypatch.setenv("ROLL_SCALE_WEIGHT_ENGINE", "local")
    assert test_ui_module.build_parser().parse_args([]).weight_engine == "local"


def test_ui_does_not_offer_fake_gemini_profile_when_backend_is_local() -> None:
    assert 'id="recognitionProfileOption"' in TEST_UI_HTML
    assert 'id="recognitionProvider"' in TEST_UI_HTML
    assert '<option value="gemini">Gemini API</option>' in TEST_UI_HTML
    assert '<option value="codex">Codex · ChatGPT</option>' in TEST_UI_HTML
    assert "$('recognitionProviderOption').hidden=!primary" in TEST_UI_HTML
    assert "recognitionProvider.disabled=!geminiPrimary" in TEST_UI_HTML
    assert "recognition_provider:recognitionProvider.value" in TEST_UI_HTML
    assert "'/api/codex/login'" in TEST_UI_HTML
    assert "'/api/gemini/key'" in TEST_UI_HTML
    assert 'id="geminiKeyBtn"' in TEST_UI_HTML
    assert 'id="geminiApiKeyInput" type="password"' in TEST_UI_HTML
    assert 'id="geminiKeyStatus"' in TEST_UI_HTML
    assert "ĐỔI GEMINI KEY THÀNH CÔNG" in TEST_UI_HTML
    assert "window.open('about:blank'" not in TEST_UI_HTML
    assert "BACKEND ĐANG DÙNG OCR LOCAL" in TEST_UI_HTML


def test_gemini_key_store_uses_supabase_legacy_compatible_action() -> None:
    source = Path(test_ui_module.__file__).read_text(encoding="utf-8")
    assert 'secret_name=f"gemini-api-key:{args.gateway_id}"' in source
    assert 'secret_action="codex-auth"' in source


def test_frozen_ui_does_not_auto_load_workspace_yolo_model(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "models" / "qr_demo_synthetic.pt"
    model.parent.mkdir()
    model.write_bytes(b"test")
    monkeypatch.setattr(test_ui_module.sys, "frozen", True, raising=False)

    assert test_ui_module.build_parser().parse_args([]).yolo_model is None


def test_bound_capture_is_idempotent_and_keeps_analysis_id(tmp_path, monkeypatch) -> None:
    class FakeOCRSource:
        def __init__(self, *args, reader=None, **kwargs):
            self._reader = reader or object()

        def capture(self, frame):
            return WeightReading(20.15, "kg", True, "OCR: 20.15@0.96", 0.96)

    monkeypatch.setattr(test_ui_module, "CameraOCRWeightSource", FakeOCRSource)
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        gateway_id="gateway-test",
        station_count=1,
        station_ids=["station-01"],
        camera_ids=["camera-01"],
    )
    frame = make_qr_frame("ROLL-BOUND-001")
    event_id = str(uuid.uuid4())
    analysis = service.analyze(
        frame,
        "0.4,0.7,0.6,0.9",
        "kg",
        event_id=event_id,
        station_id="station-01",
        camera_id="camera-01",
    )
    kwargs = dict(
        event_id=event_id,
        analysis_id=str(analysis["analysis_id"]),
        station_id="station-01",
        camera_id="camera-01",
        frame_sha256=str(analysis["frame_sha256"]),
    )
    staged_core = service.stage_evidence_step(
        frame,
        event_id=event_id,
        station_id="station-01",
        kind="core",
        weight=20.15,
        unit="kg",
    )
    staged_product = service.stage_evidence_step(
        frame,
        event_id=event_id,
        station_id="station-01",
        kind="product",
        weight=21.15,
        unit="kg",
        qr_code="ROLL-BOUND-001",
    )
    first = service.capture(
        "ROLL-BOUND-001",
        20.15,
        "kg",
        frame,
        True,
        "OCR",
        product_frame=frame,
        product_weight=21.15,
        **kwargs,
    )
    retry = service.capture("ROLL-BOUND-001", 20.15, "kg", frame, True, "OCR", **kwargs)
    row = store.get(event_id)
    service.close()
    store.close()

    assert first["duplicate"] is False
    assert retry["duplicate"] is True
    assert first["event_id"] == retry["event_id"] == event_id
    assert first["analysis_id"] == analysis["analysis_id"]
    assert first["station_id"] == "station-01"
    assert first["camera_id"] == "camera-01"
    assert first["frame_sha256"] == analysis["frame_sha256"]
    assert row is not None and row.captured_at == analysis["captured_at"]
    assert not Path(str(staged_core["image_path"])).exists()
    assert not Path(str(staged_product["image_path"])).exists()
    assert not Path(str(staged_product["metadata_path"])).exists()


def test_discard_session_removes_transient_step_evidence(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    service = StationUIService(
        store,
        None,
        None,
        None,
        station_count=1,
        station_ids=["station-01"],
        camera_ids=["camera-01"],
    )
    event_id = str(uuid.uuid4())
    frame = make_qr_frame("ROLL-DISCARD-001")
    service.sessions.stage(
        frame,
        event_id=event_id,
        station_id="station-01",
        camera_id="camera-01",
    )
    staged = service.stage_evidence_step(
        frame,
        event_id=event_id,
        station_id="station-01",
        kind="core",
        weight=13.04,
        unit="kg",
    )

    assert service.discard_session("station-01", event_id=event_id) is True
    service.close()
    store.close()
    assert not Path(str(staged["image_path"])).exists()
    assert not Path(str(staged["metadata_path"])).exists()


def test_service_rejects_duplicate_logical_camera_ids(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    with pytest.raises(ValueError, match="camera_id values must be unique"):
        StationUIService(
            store,
            None,
            None,
            None,
            station_count=2,
            camera_ids=["camera-same", "camera-same"],
        )
    store.close()
