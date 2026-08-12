import json
import subprocess

import numpy as np

from roll_qr_scale.codex_weight import CodexWeightReader


def test_codex_status_reports_missing_cli(monkeypatch) -> None:
    reader = CodexWeightReader("missing-codex-test")
    monkeypatch.setattr(reader, "_executable", lambda: None)

    status = reader.status(refresh=True)

    assert status["installed"] is False
    assert status["authenticated"] is False
    assert status["available"] is False


def test_codex_reader_uses_chatgpt_cli_with_image_and_strict_output(
    monkeypatch,
) -> None:
    reader = CodexWeightReader("codex", timeout_seconds=30)
    calls: list[tuple[list[str], object, float]] = []
    monkeypatch.setattr(
        reader,
        "status",
        lambda **kwargs: {"available": True, "authenticated": True},
    )

    def fake_run(arguments, *, cwd, timeout):
        calls.append((list(arguments), cwd, timeout))
        output_path = cwd / "result.json"
        output_path.write_text(
            json.dumps(
                {
                    "weight_readable": True,
                    "weight_digits": "1304",
                    "all_frames_agree": True,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(reader, "_run", fake_run)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    result = reader.read([frame], unit="kg")

    assert result.value == 13.04
    assert result.readable is True
    assert result.qr_code is None
    assert result.raw.startswith("CODEX:13.04")
    arguments, temp_dir, timeout = calls[0]
    assert arguments[0] == "exec"
    assert "--ignore-user-config" in arguments
    assert "--ignore-rules" in arguments
    assert "--ephemeral" in arguments
    assert "read-only" in arguments
    assert "--image" in arguments
    assert "--output-schema" in arguments
    assert "--" in arguments
    assert timeout == 30
    assert not temp_dir.exists()


def test_codex_reader_rejects_non_fixed_weight(monkeypatch) -> None:
    reader = CodexWeightReader("codex")
    monkeypatch.setattr(reader, "status", lambda **kwargs: {"available": True})

    def fake_run(arguments, *, cwd, timeout):
        (cwd / "result.json").write_text(
            json.dumps(
                {
                    "weight_readable": True,
                    "weight_digits": "13.4",
                    "all_frames_agree": True,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(reader, "_run", fake_run)

    result = reader.read([np.zeros((480, 640, 3), dtype=np.uint8)])

    assert result.value is None
    assert result.readable is False
