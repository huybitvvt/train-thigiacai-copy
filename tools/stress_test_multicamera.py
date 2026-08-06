from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from roll_qr_scale.storage import EventIdConflictError, MeasurementStore
from roll_qr_scale.sync import OutboxSyncWorker


SUITE = "multicamera_offline_stress"
REPORT_NAME = "multicamera_stress.json"
DEFAULT_EVENT_COUNT = 100
DEFAULT_STATION_COUNT = 3
GATEWAY_ID = "gateway-stress-01"
BASE_CAPTURE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)

Identity = tuple[str, str, str, str]


class FakeOutageRecoverySender:
    """Deterministic in-process sender; it never opens a network connection."""

    def __init__(
        self,
        expected_identities: dict[str, Identity],
        remote_ids: dict[str, int],
    ) -> None:
        self.expected_identities = expected_identities
        self.remote_ids = remote_ids
        self.online = False
        self.outage_attempts = 0
        self.recovery_attempts = 0
        self.cross_identity_mismatches = 0
        self.delivered_event_ids: set[str] = set()

    def recover(self) -> None:
        self.online = True

    def __call__(
        self,
        _url: str,
        payload: dict[str, object],
        _image_path: str,
        _token: str,
    ) -> dict[str, object]:
        if not self.online:
            self.outage_attempts += 1
            raise OSError("simulated offline outage")

        self.recovery_attempts += 1
        event_id = str(payload.get("event_id", ""))
        actual_identity = (
            str(payload.get("gateway_id", "")),
            str(payload.get("station_id", "")),
            str(payload.get("camera_id", "")),
            str(payload.get("analysis_id", "")),
        )
        if self.expected_identities.get(event_id) != actual_identity:
            self.cross_identity_mismatches += 1
            raise ValueError(f"cross-identity payload for {event_id}")

        self.delivered_event_ids.add(event_id)
        return {
            "ok": True,
            "event_id": event_id,
            "id": self.remote_ids[event_id],
            "image_url": f"https://fake.invalid/evidence/{event_id}.jpg",
            "image_public_id": f"offline-stress/{event_id}",
        }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _frame_for(index: int) -> np.ndarray:
    """Return a small, repeatable image with event-specific JPEG content."""

    frame = np.full((32, 48, 3), (index * 29) % 256, dtype=np.uint8)
    frame[index % frame.shape[0], :, 1] = (index * 53 + 17) % 256
    frame[:, index % frame.shape[1], 2] = (index * 71 + 31) % 256
    return frame


def _deterministic_uuid4(value: str) -> str:
    raw = bytearray(hashlib.sha256(value.encode("utf-8")).digest()[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))


def _event_values(index: int, station_count: int) -> dict[str, Any]:
    station_number = index % station_count + 1
    event_id = _deterministic_uuid4(
        f"roll-qr-scale/multicamera/{station_count}/{index}"
    )
    analysis_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"analysis/{event_id}"))
    return {
        "qr_code": f"STRESS-MULTICAMERA-{index + 1:04d}",
        "weight": round(10.0 + index * 0.125, 3),
        "unit": "kg",
        "frame": _frame_for(index),
        "weight_source": "multicamera-stress",
        "needs_sync": True,
        "qr_source": "synthetic:offline",
        "weight_raw": f"STRESS:{10.0 + index * 0.125:.3f}",
        "weight_stable": True,
        "event_id": event_id,
        "captured_at": (BASE_CAPTURE_TIME + timedelta(milliseconds=index)).isoformat(
            timespec="milliseconds"
        ),
        "gateway_id": GATEWAY_ID,
        "station_id": f"station-{station_number:02d}",
        "camera_id": f"camera-{station_number:02d}",
        "analysis_id": analysis_id,
    }


def run_stress(
    event_count: int = DEFAULT_EVENT_COUNT,
    station_count: int = DEFAULT_STATION_COUNT,
) -> dict[str, object]:
    """Exercise local idempotency and outbox recovery without external services."""

    if event_count < 1:
        raise ValueError("event_count must be at least 1")
    if station_count not in (1, 2, 3):
        raise ValueError("station_count must be 1, 2, or 3")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="roll-multicamera-stress-") as temp_dir:
        work_dir = Path(temp_dir)
        store = MeasurementStore(work_dir / "measurements.db", work_dir / "captures")
        try:
            events = [_event_values(index, station_count) for index in range(event_count)]
            expected_identities: dict[str, Identity] = {}
            remote_ids: dict[str, int] = {}
            duplicate_retries = 0

            for index, values in enumerate(events):
                first = store.save_idempotent(**values)
                if first.duplicate:
                    raise AssertionError("first save was unexpectedly classified as a duplicate")

                event_id = str(values["event_id"])
                expected_identities[event_id] = (
                    str(values["gateway_id"]),
                    str(values["station_id"]),
                    str(values["camera_id"]),
                    str(values["analysis_id"]),
                )
                remote_ids[event_id] = index + 1

                identical_retry = store.save_idempotent(**values)
                duplicate_retries += int(identical_retry.duplicate)

            identity_conflicts_rejected = 0
            conflicting_retry = dict(events[0])
            conflicting_retry["station_id"] = "station-conflict"
            conflicting_retry["camera_id"] = "camera-conflict"
            try:
                store.save_idempotent(**conflicting_retry)
            except EventIdConflictError:
                identity_conflicts_rejected = 1

            rows = store.connection.execute(
                "SELECT event_id,gateway_id,station_id,camera_id,analysis_id "
                "FROM measurements ORDER BY id"
            ).fetchall()
            row_count = len(rows)
            unique_events = len({str(row["event_id"]) for row in rows})
            station_counts = {
                f"station-{number:02d}": 0 for number in range(1, station_count + 1)
            }
            local_identity_mismatches = 0
            for row in rows:
                event_id = str(row["event_id"])
                station_id = str(row["station_id"])
                if station_id in station_counts:
                    station_counts[station_id] += 1
                actual_identity = (
                    str(row["gateway_id"]),
                    station_id,
                    str(row["camera_id"]),
                    str(row["analysis_id"]),
                )
                if expected_identities.get(event_id) != actual_identity:
                    local_identity_mismatches += 1

            sender = FakeOutageRecoverySender(expected_identities, remote_ids)
            worker = OutboxSyncWorker(
                store,
                "https://fake.invalid/ingest",
                "offline-test-token",
                GATEWAY_ID,
                send=sender,
            )

            offline_synced = worker.sync_once(
                limit=event_count,
                include_deferred=True,
            )
            pending_before_recovery = store.pending_count()

            sender.recover()
            synced_during_recovery = 0
            while store.pending_count():
                batch_synced = worker.sync_once(limit=17, include_deferred=True)
                synced_during_recovery += batch_synced
                if batch_synced == 0:
                    break

            pending_after_recovery = store.pending_count()
            synced_after_recovery = int(
                store.connection.execute(
                    "SELECT COUNT(*) FROM measurements WHERE sync_status = 'synced'"
                ).fetchone()[0]
            )
            cross_identity_mismatches = (
                local_identity_mismatches + sender.cross_identity_mismatches
            )
            expected_station_counts = {
                f"station-{number:02d}": sum(
                    1 for index in range(event_count) if index % station_count == number - 1
                )
                for number in range(1, station_count + 1)
            }

            accepted = all(
                (
                    row_count == event_count,
                    unique_events == event_count,
                    station_counts == expected_station_counts,
                    duplicate_retries == event_count,
                    identity_conflicts_rejected == 1,
                    cross_identity_mismatches == 0,
                    offline_synced == 0,
                    sender.outage_attempts == event_count,
                    pending_before_recovery == event_count,
                    synced_during_recovery == event_count,
                    synced_after_recovery == event_count,
                    pending_after_recovery == 0,
                    len(sender.delivered_event_ids) == event_count,
                )
            )
            elapsed_seconds = round(time.perf_counter() - started, 6)
            return {
                "suite": SUITE,
                "event_count": event_count,
                "station_count": station_count,
                "station_counts": station_counts,
                "row_count": row_count,
                "unique_events": unique_events,
                "duplicate_retries": duplicate_retries,
                "identity_conflicts_rejected": identity_conflicts_rejected,
                "cross_identity_mismatches": cross_identity_mismatches,
                "offline_attempts": sender.outage_attempts,
                "pending_before_recovery": pending_before_recovery,
                "synced_during_recovery": synced_during_recovery,
                "synced_after_recovery": synced_after_recovery,
                "pending_after_recovery": pending_after_recovery,
                "elapsed_seconds": elapsed_seconds,
                "accepted": accepted,
            }
        finally:
            store.close()


def write_report(run_dir: str | Path, report: dict[str, object]) -> Path:
    destination = Path(run_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / REPORT_NAME
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline deterministic stress test for 1-3 logical camera stations"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Explicit directory that receives multicamera_stress.json",
    )
    parser.add_argument("--event-count", type=_positive_int, default=DEFAULT_EVENT_COUNT)
    parser.add_argument(
        "--station-count",
        type=int,
        choices=(1, 2, 3),
        default=DEFAULT_STATION_COUNT,
    )
    return parser


def run(args: argparse.Namespace) -> int:
    report = run_stress(args.event_count, args.station_count)
    report_path = write_report(args.run_dir, report)
    print(report_path.resolve())
    return 0 if report["accepted"] else 2


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
