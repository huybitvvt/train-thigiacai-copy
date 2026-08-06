import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from roll_qr_scale.lookup import _format_result, _lookup_from_image
from roll_qr_scale.lookup_client import lookup_roll


def _start_server(status: int, response: dict, received: dict):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            received["path"] = parsed.path
            received["qr"] = urllib.parse.parse_qs(parsed.query).get("qr", [None])[0]
            received["token"] = self.headers.get("x-device-token")
            body = json.dumps(response).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_lookup_client_encodes_qr_and_sends_lookup_token() -> None:
    response = {
        "ok": True,
        "found": True,
        "history_count": 2,
        "measurement": {
            "qr_code": "ROLL A/001",
            "gross_weight": 125.4,
            "tare_weight": 5.4,
            "net_weight": 120,
            "unit": "kg",
            "captured_at": "2026-08-01T15:00:00Z",
        },
    }
    received: dict = {}
    server, thread = _start_server(200, response, received)
    try:
        result = lookup_roll(
            f"http://127.0.0.1:{server.server_port}/lookup-roll",
            "ROLL A/001",
            "lookup-secret",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result == response
    assert received == {
        "path": "/lookup-roll",
        "qr": "ROLL A/001",
        "token": "lookup-secret",
    }
    assert "NET=120 kg" in _format_result(result)
    assert "HISTORY=2" in _format_result(result)


def test_lookup_client_returns_not_found_payload() -> None:
    response = {"ok": False, "found": False, "qr_code": "MISSING-001"}
    received: dict = {}
    server, thread = _start_server(404, response, received)
    try:
        result = lookup_roll(
            f"http://127.0.0.1:{server.server_port}/lookup-roll",
            "MISSING-001",
            "lookup-secret",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result == response
    assert _format_result(result) == "KHÔNG TÌM THẤY QR: MISSING-001"


def test_lookup_client_rejects_invalid_json() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"not-json"
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="invalid JSON"):
            lookup_roll(
                f"http://127.0.0.1:{server.server_port}/lookup-roll",
                "ROLL-001",
                "lookup-secret",
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_lookup_decodes_project_demo_image() -> None:
    assert _lookup_from_image("data/test_frame.png") == "ROLL-DEMO-0001"
