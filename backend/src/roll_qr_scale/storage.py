from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Measurement:
    id: int
    event_id: str
    qr_code: str
    weight: float
    unit: str
    captured_at: str
    image_path: str
    weight_source: str
    sync_status: str
    qr_source: str = "camera"
    product_weight: float | None = None
    product_image_path: str = ""
    retry_count: int = 0
    sync_error: str | None = None
    remote_id: int | None = None
    remote_image_url: str | None = None
    remote_image_public_id: str | None = None
    weight_raw: str = ""
    weight_stable: bool = True
    gateway_id: str = ""
    station_id: str = ""
    camera_id: str = ""
    analysis_id: str = ""
    frame_sha256: str = ""
    payload_hash: str = ""

    def api_payload(self, device_id: str = "") -> dict[str, object]:
        payload = asdict(self)
        for local_field in (
            "id",
            "sync_status",
            "retry_count",
            "sync_error",
            "remote_id",
            "remote_image_url",
            "remote_image_public_id",
            "image_path",
            "product_image_path",
        ):
            payload.pop(local_field)

        # Identity belongs to the captured row. ``device_id`` only remains as a
        # fallback for rows produced before gateway identity was persisted.
        effective_gateway_id = self.gateway_id or device_id
        for optional_field in (
            "gateway_id",
            "station_id",
            "camera_id",
            "analysis_id",
            "frame_sha256",
            "payload_hash",
        ):
            if not payload.get(optional_field):
                payload.pop(optional_field, None)
        if effective_gateway_id:
            payload["gateway_id"] = effective_gateway_id
            # Older deployed functions only understand device_id. Sending the
            # same row-derived value keeps those deployments compatible.
            payload["device_id"] = effective_gateway_id
        return payload


@dataclass(frozen=True)
class SaveResult:
    measurement: Measurement
    duplicate: bool

    @property
    def existing(self) -> Measurement:
        """Alias that makes the idempotent outcome explicit to callers."""

        return self.measurement

    def __iter__(self):
        # Allow the concise ``measurement, duplicate = ...`` form as well as
        # named attribute access.
        yield self.measurement
        yield self.duplicate


class EventIdConflictError(ValueError):
    """An event ID was reused for different immutable capture content."""

    def __init__(self, event_id: str):
        super().__init__(f"event_id already exists with different payload: {event_id}")
        self.event_id = event_id


class MeasurementStore:
    """Thread-safe SQLite event store and durable synchronization outbox."""

    def __init__(self, db_path: str | Path, capture_dir: str | Path):
        self.db_path = Path(db_path)
        self.capture_dir = Path(capture_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                qr_code TEXT NOT NULL,
                weight REAL NOT NULL,
                unit TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                image_path TEXT NOT NULL,
                product_image_path TEXT NOT NULL DEFAULT '',
                weight_source TEXT NOT NULL,
                qr_source TEXT NOT NULL DEFAULT 'camera',
                product_weight REAL,
                sync_status TEXT NOT NULL DEFAULT 'local',
                sync_error TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                next_retry_at TEXT,
                last_attempt_at TEXT,
                synced_at TEXT,
                remote_id INTEGER,
                remote_image_url TEXT,
                remote_image_public_id TEXT,
                weight_raw TEXT NOT NULL DEFAULT '',
                weight_stable INTEGER NOT NULL DEFAULT 1,
                gateway_id TEXT NOT NULL DEFAULT '',
                station_id TEXT NOT NULL DEFAULT '',
                camera_id TEXT NOT NULL DEFAULT '',
                analysis_id TEXT NOT NULL DEFAULT '',
                frame_sha256 TEXT NOT NULL DEFAULT '',
                payload_hash TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._migrate_existing_database()
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_measurements_qr ON measurements(qr_code)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_measurements_outbox "
            "ON measurements(sync_status, next_retry_at)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_measurements_station_time "
            "ON measurements(gateway_id, station_id, camera_id, captured_at DESC)"
        )
        self.connection.commit()

    def _migrate_existing_database(self) -> None:
        existing = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(measurements)").fetchall()
        }
        additions = {
            "retry_count": "INTEGER NOT NULL DEFAULT 0",
            "next_retry_at": "TEXT",
            "last_attempt_at": "TEXT",
            "synced_at": "TEXT",
            "remote_id": "INTEGER",
            "remote_image_url": "TEXT",
            "remote_image_public_id": "TEXT",
            "weight_raw": "TEXT NOT NULL DEFAULT ''",
            "weight_stable": "INTEGER NOT NULL DEFAULT 1",
            "qr_source": "TEXT NOT NULL DEFAULT 'camera'",
            "sync_error": "TEXT",
            "gateway_id": "TEXT NOT NULL DEFAULT ''",
            "station_id": "TEXT NOT NULL DEFAULT ''",
            "camera_id": "TEXT NOT NULL DEFAULT ''",
            "analysis_id": "TEXT NOT NULL DEFAULT ''",
            "frame_sha256": "TEXT NOT NULL DEFAULT ''",
            "payload_hash": "TEXT NOT NULL DEFAULT ''",
            "product_image_path": "TEXT NOT NULL DEFAULT ''",
            "product_weight": "REAL",
        }
        for column, definition in additions.items():
            if column not in existing:
                self.connection.execute(
                    f"ALTER TABLE measurements ADD COLUMN {column} {definition}"
                )

        # Backfill structured product weight from captures made before the
        # dedicated column existed. Do not guess rows without this exact tag.
        product_rows = self.connection.execute(
            "SELECT id, weight_raw FROM measurements WHERE product_weight IS NULL"
        ).fetchall()
        for row in product_rows:
            match = re.search(
                r"(?:^|; )PRODUCT_WEIGHT=([0-9]+(?:\.[0-9]+)?)",
                str(row["weight_raw"]),
            )
            if match:
                self.connection.execute(
                    "UPDATE measurements SET product_weight = ? WHERE id = ?",
                    (float(match.group(1)), int(row["id"])),
                )

        # Upgrade prior captures without deleting or rewriting their evidence.
        # Hashes can be reconstructed from the immutable row and on-disk JPEG.
        rows = self.connection.execute(
            "SELECT * FROM measurements WHERE frame_sha256 = '' OR payload_hash = ''"
        ).fetchall()
        for row in rows:
            frame_sha256 = str(row["frame_sha256"])
            if not frame_sha256:
                try:
                    frame_sha256 = hashlib.sha256(
                        Path(str(row["image_path"])).read_bytes()
                    ).hexdigest()
                except OSError:
                    # Missing legacy evidence must not prevent the database from
                    # opening. The empty hash continues to identify it as legacy.
                    frame_sha256 = ""
            payload_hash = str(row["payload_hash"])
            if not payload_hash:
                payload_hash = self._calculate_payload_hash(
                    qr_code=str(row["qr_code"]),
                    weight=float(row["weight"]),
                    unit=str(row["unit"]),
                    captured_at=str(row["captured_at"]),
                    weight_source=str(row["weight_source"]),
                    qr_source=str(row["qr_source"]),
                    weight_raw=str(row["weight_raw"]),
                    weight_stable=bool(row["weight_stable"]),
                    gateway_id=str(row["gateway_id"]),
                    station_id=str(row["station_id"]),
                    camera_id=str(row["camera_id"]),
                    analysis_id=str(row["analysis_id"]),
                    frame_sha256=frame_sha256,
                )
            self.connection.execute(
                "UPDATE measurements SET frame_sha256 = ?, payload_hash = ? WHERE id = ?",
                (frame_sha256, payload_hash, int(row["id"])),
            )

    @staticmethod
    def _calculate_payload_hash(
        *,
        qr_code: str,
        weight: float,
        unit: str,
        captured_at: str,
        weight_source: str,
        qr_source: str,
        weight_raw: str,
        weight_stable: bool,
        gateway_id: str,
        station_id: str,
        camera_id: str,
        analysis_id: str,
        frame_sha256: str,
    ) -> str:
        immutable_payload = {
            "analysis_id": analysis_id,
            "camera_id": camera_id,
            "captured_at": captured_at,
            "frame_sha256": frame_sha256,
            "gateway_id": gateway_id,
            "qr_code": qr_code,
            "qr_source": qr_source,
            "station_id": station_id,
            "unit": unit,
            "weight": float(weight),
            "weight_raw": weight_raw,
            "weight_source": weight_source,
            "weight_stable": bool(weight_stable),
        }
        canonical = json.dumps(
            immutable_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def save(
        self,
        qr_code: str,
        weight: float,
        unit: str,
        frame: np.ndarray,
        weight_source: str,
        needs_sync: bool = False,
        qr_source: str = "camera",
        weight_raw: str = "",
        weight_stable: bool = True,
        *,
        event_id: str | None = None,
        captured_at: str | None = None,
        gateway_id: str = "",
        station_id: str = "",
        camera_id: str = "",
        analysis_id: str = "",
    ) -> Measurement:
        return self.save_idempotent(
            qr_code,
            weight,
            unit,
            frame,
            weight_source,
            needs_sync,
            qr_source,
            weight_raw,
            weight_stable,
            event_id=event_id,
            captured_at=captured_at,
            gateway_id=gateway_id,
            station_id=station_id,
            camera_id=camera_id,
            analysis_id=analysis_id,
        ).measurement

    def save_idempotent(
        self,
        qr_code: str,
        weight: float,
        unit: str,
        frame: np.ndarray,
        weight_source: str,
        needs_sync: bool = False,
        qr_source: str = "camera",
        weight_raw: str = "",
        weight_stable: bool = True,
        *,
        event_id: str | None = None,
        captured_at: str | None = None,
        gateway_id: str = "",
        station_id: str = "",
        camera_id: str = "",
        analysis_id: str = "",
    ) -> SaveResult:
        event_id = event_id or str(uuid.uuid4())
        if captured_at is None:
            # A caller retrying an explicit idempotency key should not have to
            # remember the timestamp that storage assigned on the first call.
            existing_measurement = self.get(event_id)
            if existing_measurement is not None:
                captured_at = existing_measurement.captured_at
        captured_at = captured_at or datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
        weight_raw = weight_raw[:1000]

        encoded_ok, encoded_frame = cv2.imencode(".jpg", frame)
        if not encoded_ok:
            raise OSError("Cannot encode capture image as JPEG")
        jpeg_bytes = encoded_frame.tobytes()
        frame_sha256 = hashlib.sha256(jpeg_bytes).hexdigest()
        payload_hash = self._calculate_payload_hash(
            qr_code=qr_code,
            weight=weight,
            unit=unit,
            captured_at=captured_at,
            weight_source=weight_source,
            qr_source=qr_source,
            weight_raw=weight_raw,
            weight_stable=weight_stable,
            gateway_id=gateway_id,
            station_id=station_id,
            camera_id=camera_id,
            analysis_id=analysis_id,
            frame_sha256=frame_sha256,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        event_token = hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:8]
        image_path = self.capture_dir / f"{timestamp}_{event_token}_{uuid.uuid4().hex[:8]}.jpg"
        try:
            image_path.write_bytes(jpeg_bytes)
        except OSError as exc:
            image_path.unlink(missing_ok=True)
            raise OSError(f"Cannot write capture image: {image_path}") from exc

        sync_status = "pending" if needs_sync else "local"
        try:
            with self._lock:
                existing = self.connection.execute(
                    "SELECT * FROM measurements WHERE event_id = ?", (event_id,)
                ).fetchone()
                if existing is not None:
                    image_path.unlink(missing_ok=True)
                    if str(existing["payload_hash"]) == payload_hash:
                        return SaveResult(self._from_row(existing), duplicate=True)
                    raise EventIdConflictError(event_id)

                cursor = self.connection.execute(
                    """
                    INSERT INTO measurements (
                        event_id, qr_code, weight, unit, captured_at, image_path,
                        weight_source, qr_source, sync_status, weight_raw, weight_stable,
                        gateway_id, station_id, camera_id, analysis_id,
                        frame_sha256, payload_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        qr_code,
                        weight,
                        unit,
                        captured_at,
                        str(image_path.resolve()),
                        weight_source,
                        qr_source,
                        sync_status,
                        weight_raw,
                        int(weight_stable),
                        gateway_id,
                        station_id,
                        camera_id,
                        analysis_id,
                        frame_sha256,
                        payload_hash,
                    ),
                )
                self.connection.commit()
        except sqlite3.IntegrityError as exc:
            # A different store/process may have committed the same event after
            # our pre-insert check. Resolve that race idempotently.
            with self._lock:
                self.connection.rollback()
                existing = self.connection.execute(
                    "SELECT * FROM measurements WHERE event_id = ?", (event_id,)
                ).fetchone()
            image_path.unlink(missing_ok=True)
            if existing is not None and str(existing["payload_hash"]) == payload_hash:
                return SaveResult(self._from_row(existing), duplicate=True)
            if existing is not None:
                raise EventIdConflictError(event_id) from exc
            raise
        except Exception:
            image_path.unlink(missing_ok=True)
            raise

        saved = self.get(event_id)
        if saved is None:  # pragma: no cover - protects against external DB corruption.
            image_path.unlink(missing_ok=True)
            raise RuntimeError(f"Inserted event cannot be read back: {event_id}")
        return SaveResult(saved, duplicate=False)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Measurement:
        return Measurement(
            id=int(row["id"]),
            event_id=str(row["event_id"]),
            qr_code=str(row["qr_code"]),
            weight=float(row["weight"]),
            unit=str(row["unit"]),
            captured_at=str(row["captured_at"]),
            image_path=str(row["image_path"]),
            product_image_path=str(row["product_image_path"]),
            weight_source=str(row["weight_source"]),
            sync_status=str(row["sync_status"]),
            qr_source=str(row["qr_source"]),
            product_weight=(
                float(row["product_weight"])
                if row["product_weight"] is not None
                else None
            ),
            retry_count=int(row["retry_count"]),
            sync_error=str(row["sync_error"]) if row["sync_error"] is not None else None,
            remote_id=int(row["remote_id"]) if row["remote_id"] is not None else None,
            remote_image_url=(
                str(row["remote_image_url"]) if row["remote_image_url"] is not None else None
            ),
            remote_image_public_id=(
                str(row["remote_image_public_id"])
                if row["remote_image_public_id"] is not None
                else None
            ),
            weight_raw=str(row["weight_raw"]),
            weight_stable=bool(row["weight_stable"]),
            gateway_id=str(row["gateway_id"]),
            station_id=str(row["station_id"]),
            camera_id=str(row["camera_id"]),
            analysis_id=str(row["analysis_id"]),
            frame_sha256=str(row["frame_sha256"]),
            payload_hash=str(row["payload_hash"]),
        )

    def get(self, event_id: str) -> Measurement | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM measurements WHERE event_id = ?", (event_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def recent(self, limit: int = 50) -> list[Measurement]:
        safe_limit = max(1, min(int(limit), 200))
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM measurements ORDER BY id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def attach_product_image(self, event_id: str, frame: np.ndarray) -> str:
        """Persist the product-weight evidence beside its core-weight event."""
        encoded_ok, encoded_frame = cv2.imencode(".jpg", frame)
        if not encoded_ok:
            raise OSError("Cannot encode product capture image as JPEG")
        path = self.capture_dir / f"{event_id}_product.jpg"
        path.write_bytes(encoded_frame.tobytes())
        with self._lock:
            cursor = self.connection.execute(
                "UPDATE measurements SET product_image_path = ? WHERE event_id = ?",
                (str(path.resolve()), event_id),
            )
            if cursor.rowcount != 1:
                path.unlink(missing_ok=True)
                raise KeyError(f"Unknown capture event: {event_id}")
            self.connection.commit()
        return str(path.resolve())

    def attach_product_weight(self, event_id: str, product_weight: float) -> None:
        if not np.isfinite(product_weight) or product_weight < 0:
            raise ValueError("Product weight must be a non-negative finite number")
        with self._lock:
            cursor = self.connection.execute(
                "UPDATE measurements SET product_weight = ? WHERE event_id = ?",
                (float(product_weight), event_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown capture event: {event_id}")
            self.connection.commit()

    def pending(
        self,
        limit: int = 20,
        include_deferred: bool = False,
        *,
        include_failed: bool = True,
    ) -> list[Measurement]:
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        retry_clause = "" if include_deferred else "AND (next_retry_at IS NULL OR next_retry_at <= ?)"
        parameters: tuple[object, ...] = (limit,) if include_deferred else (now, limit)
        statuses = "('pending', 'failed')" if include_failed else "('pending')"
        with self._lock:
            rows = self.connection.execute(
                f"""
                SELECT * FROM measurements
                WHERE sync_status IN {statuses}
                  {retry_clause}
                ORDER BY id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def mark_synced(
        self,
        event_id: str,
        remote_id: int | None = None,
        remote_image_url: str | None = None,
        remote_image_public_id: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with self._lock:
            self.connection.execute(
                """
                UPDATE measurements
                SET sync_status = 'synced', sync_error = NULL, next_retry_at = NULL,
                    last_attempt_at = ?, synced_at = ?, remote_id = ?,
                    remote_image_url = ?, remote_image_public_id = ?
                WHERE event_id = ?
                """,
                (
                    now,
                    now,
                    remote_id,
                    remote_image_url,
                    remote_image_public_id,
                    event_id,
                ),
            )
            self.connection.commit()

    def mark_sync_failed(self, event_id: str, error: str) -> None:
        now_dt = datetime.now(timezone.utc)
        with self._lock:
            row = self.connection.execute(
                "SELECT retry_count FROM measurements WHERE event_id = ?", (event_id,)
            ).fetchone()
            retry_count = (int(row["retry_count"]) if row else 0) + 1
            delay_seconds = min(300, 2 ** min(retry_count, 8))
            next_retry = (now_dt + timedelta(seconds=delay_seconds)).isoformat(
                timespec="milliseconds"
            )
            self.connection.execute(
                """
                UPDATE measurements
                SET sync_status = 'failed', sync_error = ?, retry_count = ?,
                    last_attempt_at = ?, next_retry_at = ?
                WHERE event_id = ?
                """,
                (
                    error[:1000],
                    retry_count,
                    now_dt.isoformat(timespec="milliseconds"),
                    next_retry,
                    event_id,
                ),
            )
            self.connection.commit()

    def pending_count(self) -> int:
        with self._lock:
            row = self.connection.execute(
                "SELECT COUNT(*) AS total FROM measurements "
                "WHERE sync_status IN ('pending', 'failed')"
            ).fetchone()
        return int(row["total"])

    def count(self) -> int:
        with self._lock:
            row = self.connection.execute("SELECT COUNT(*) AS total FROM measurements").fetchone()
        return int(row["total"])

    def close(self) -> None:
        with self._lock:
            self.connection.close()
