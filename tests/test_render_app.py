from __future__ import annotations

import base64
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from roll_qr_scale.render_app import build_render_argv
from roll_qr_scale.test_ui import build_parser, create_server


def test_render_argv_uses_public_host_port_and_ephemeral_data_root(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PORT", "12345")
    monkeypatch.setenv("ROLL_SCALE_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("ROLL_SCALE_STATION_COUNT", "2")
    monkeypatch.setenv("ROLL_SCALE_STATION_IDS", "scale-a,scale-b")
    monkeypatch.setenv("ROLL_SCALE_CAMERA_IDS", "cam-a,cam-b")

    argv = build_render_argv()

    assert argv[argv.index("--host") + 1] == "0.0.0.0"
    assert argv[argv.index("--port") + 1] == "12345"
    assert argv[argv.index("--db") + 1] == str(tmp_path / "measurements.db")
    assert argv[argv.index("--captures") + 1] == str(tmp_path / "captures")
    assert [argv[index + 1] for index, value in enumerate(argv) if value == "--station-id"] == [
        "scale-a",
        "scale-b",
    ]
    assert [argv[index + 1] for index, value in enumerate(argv) if value == "--camera-id"] == [
        "cam-a",
        "cam-b",
    ]


def test_web_auth_protects_ui_but_leaves_health_check_public(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ROLL_SCALE_WEB_USERNAME", "pilot")
    monkeypatch.setenv("ROLL_SCALE_WEB_PASSWORD", "secret")
    args = build_parser().parse_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--db",
            str(tmp_path / "measurements.db"),
            "--captures",
            str(tmp_path / "captures"),
            "--factory-samples",
            str(tmp_path / "factory_raw"),
            "--api-url",
            "",
            "--api-token",
            "",
            "--lookup-url",
            "",
            "--lookup-token",
            "",
            "--weight-engine",
            "local",
        ]
    )
    server, service = create_server(args)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=2) as response:
            assert response.status == 200
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{base_url}/api/status", timeout=2)
        assert error.value.code == 401

        credentials = base64.b64encode(b"pilot:secret").decode("ascii")
        request = urllib.request.Request(
            f"{base_url}/api/status",
            headers={"Authorization": f"Basic {credentials}"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.status == 200
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        service.store.close()
        thread.join(timeout=2)
