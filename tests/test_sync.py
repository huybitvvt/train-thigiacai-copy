import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from roll_qr_scale.api_client import post_measurement
from roll_qr_scale.storage import MeasurementStore
from roll_qr_scale.sync import OutboxSyncWorker


def test_outbox_syncs_committed_measurement(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    measurement = store.save(
        "ROLL-OUTBOX-001",
        125.4,
        "kg",
        np.zeros((80, 100, 3), dtype=np.uint8),
        "serial",
        needs_sync=True,
        gateway_id="gateway-row",
        station_id="station-01",
        camera_id="camera-1",
        analysis_id="analysis-1",
    )
    sent: list[tuple] = []

    def fake_send(url, payload, image_path, token):
        sent.append((url, payload, image_path, token))
        return {
            "ok": True,
            "event_id": measurement.event_id,
            "id": 99,
            "image_url": "https://res.cloudinary.com/demo/image/upload/event.jpg",
            "image_public_id": "roll-captures/station-01/event",
        }

    worker = OutboxSyncWorker(
        store,
        "https://example.test/ingest",
        "device-secret",
        "station-01",
        send=fake_send,
    )
    assert worker.sync_once() == 1
    saved = store.get(measurement.event_id)
    store.close()

    assert saved is not None
    assert saved.sync_status == "synced"
    assert saved.remote_id == 99
    assert saved.remote_image_url == "https://res.cloudinary.com/demo/image/upload/event.jpg"
    assert saved.remote_image_public_id == "roll-captures/station-01/event"
    assert sent[0][1]["gateway_id"] == "gateway-row"
    assert sent[0][1]["device_id"] == "gateway-row"
    assert sent[0][1]["station_id"] == "station-01"
    assert sent[0][1]["camera_id"] == "camera-1"
    assert sent[0][1]["analysis_id"] == "analysis-1"
    assert len(sent[0][1]["frame_sha256"]) == 64
    assert len(sent[0][1]["payload_hash"]) == 64
    assert sent[0][1]["qr_code"] == "ROLL-OUTBOX-001"
    assert "image_path" not in sent[0][1]


def test_outbox_accepts_core_image_ack_and_persists_it(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    measurement = store.save(
        "ROLL-CORE-IMAGE-001",
        1.0,
        "kg",
        np.zeros((20, 20, 3), dtype=np.uint8),
        "camera-gemini:test-ui",
        needs_sync=True,
    )

    def fake_send(*args):
        return {
            "ok": True,
            "event_id": measurement.event_id,
            "id": 101,
            "core_image_url": "https://images.example/core.jpg",
            "core_image_public_id": "roll-captures/core-weight/event",
        }

    worker = OutboxSyncWorker(store, "https://example.test", "token", send=fake_send)
    assert worker.sync_once() == 1
    saved = store.get(measurement.event_id)
    store.close()

    assert saved is not None
    assert saved.remote_image_url == "https://images.example/core.jpg"
    assert saved.remote_image_public_id == "roll-captures/core-weight/event"


def test_outbox_keeps_failed_event_for_retry(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    measurement = store.save(
        "ROLL-RETRY-001",
        75,
        "kg",
        np.zeros((40, 40, 3), dtype=np.uint8),
        "manual",
        needs_sync=True,
    )

    def fail(*args):
        raise OSError("network offline")

    worker = OutboxSyncWorker(store, "https://offline.test", "token", "station-01", send=fail)
    assert worker.sync_once() == 0
    saved = store.get(measurement.event_id)
    assert saved is not None
    assert saved.sync_status == "failed"
    assert saved.retry_count == 1
    assert store.pending_count() == 1
    store.close()


@pytest.mark.parametrize(
    "response_factory",
    [
        lambda event_id: {"ok": False, "event_id": event_id},
        lambda event_id: {"ok": True, "event_id": "different-event", "id": 1},
        lambda event_id: {"event_id": event_id},
        lambda event_id: {"ok": True, "event_id": event_id},
        lambda event_id: {"ok": True, "event_id": event_id, "id": 0},
        lambda event_id: {"ok": True, "event_id": event_id, "id": 1},
    ],
)
def test_outbox_does_not_mark_unacknowledged_event_synced(tmp_path, response_factory) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    measurement = store.save(
        "ROLL-ACK-001",
        22,
        "kg",
        np.zeros((20, 20, 3), dtype=np.uint8),
        "manual",
        needs_sync=True,
    )

    def fake_send(*args):
        return response_factory(measurement.event_id)

    worker = OutboxSyncWorker(store, "https://example.test", "token", "legacy-gateway", send=fake_send)
    assert worker.sync_once() == 0
    saved = store.get(measurement.event_id)
    assert saved is not None
    assert saved.sync_status == "failed"
    assert saved.retry_count == 1
    store.close()


def test_outbox_can_allow_legacy_ack_without_image_when_explicitly_configured(tmp_path) -> None:
    store = MeasurementStore(tmp_path / "measurements.db", tmp_path / "captures")
    measurement = store.save(
        "ROLL-LEGACY-ACK",
        3,
        "kg",
        np.zeros((20, 20, 3), dtype=np.uint8),
        "manual",
        needs_sync=True,
    )

    def fake_send(*args):
        return {"ok": True, "event_id": measurement.event_id, "id": "7"}

    worker = OutboxSyncWorker(
        store,
        "https://example.test",
        "token",
        "legacy-gateway",
        send=fake_send,
        require_remote_image=False,
    )
    assert worker.sync_once() == 1
    saved = store.get(measurement.event_id)
    assert saved is not None
    assert saved.sync_status == "synced"
    assert saved.remote_id == 7
    store.close()


def test_http_client_sends_image_and_device_token(tmp_path) -> None:
    received: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["content-length"])
            received["token"] = self.headers["x-device-token"]
            received["body"] = json.loads(self.rfile.read(length))
            response = json.dumps(
                {"ok": True, "event_id": "test-event", "id": 321}
            ).encode()
            self.send_response(201)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    image_path = tmp_path / "capture.jpg"
    image_path.write_bytes(b"\xff\xd8test-jpeg\xff\xd9")
    try:
        response = post_measurement(
            f"http://127.0.0.1:{server.server_port}/ingest",
            {"event_id": "test-event", "qr_code": "ROLL-HTTP-001"},
            image_path,
            "secret-token",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response["id"] == 321
    assert received["token"] == "secret-token"
    assert base64.b64decode(received["body"]["image_base64"]) == image_path.read_bytes()
    assert received["body"]["image_role"] == "core_weight"
