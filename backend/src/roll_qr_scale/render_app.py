from __future__ import annotations

import os
from pathlib import Path

from .test_ui import build_parser, run


def _csv_env(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def build_render_argv() -> list[str]:
    data_root = Path(os.environ.get("ROLL_SCALE_DATA_ROOT", "/tmp/tram-can-qr"))
    data_root.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", "10000"))
    station_count = int(os.environ.get("ROLL_SCALE_STATION_COUNT", "1"))
    argv = [
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--db",
        str(data_root / "measurements.db"),
        "--captures",
        str(data_root / "captures"),
        "--factory-samples",
        str(data_root / "factory_raw"),
        "--staging-dir",
        str(data_root / "captures" / ".staging"),
        "--station-count",
        str(station_count),
        "--weight-engine",
        os.environ.get("ROLL_SCALE_WEIGHT_ENGINE", "gemini"),
    ]
    for station_id in _csv_env("ROLL_SCALE_STATION_IDS"):
        argv.extend(("--station-id", station_id))
    for camera_id in _csv_env("ROLL_SCALE_CAMERA_IDS"):
        argv.extend(("--camera-id", camera_id))
    return argv


def main() -> None:
    args = build_parser().parse_args(build_render_argv())
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
