from __future__ import annotations

import threading
from collections.abc import Callable

from .api_client import post_measurement, validate_ingest_response
from .storage import Measurement, MeasurementStore


SendFunction = Callable[[str, dict[str, object], str, str], dict[str, object]]


def _default_send(
    url: str,
    payload: dict[str, object],
    image_path: str,
    token: str,
) -> dict[str, object]:
    return post_measurement(url, payload, image_path, token)


class OutboxSyncWorker:
    """Upload locally committed events in the background with durable retries."""

    def __init__(
        self,
        store: MeasurementStore,
        api_url: str,
        device_token: str,
        device_id: str = "",
        interval: float = 2.0,
        send: SendFunction = _default_send,
        *,
        gateway_id: str | None = None,
        require_remote_image: bool = True,
    ):
        self.store = store
        self.api_url = api_url
        self.device_token = device_token
        # device_id is retained as a constructor fallback for legacy rows. New
        # captures carry their gateway identity in the outbox row itself.
        self.device_id = gateway_id if gateway_id is not None else device_id
        self.require_remote_image = require_remote_image
        self.interval = interval
        self.send = send
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._sync_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="supabase-outbox", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def notify(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sync_once()
            self._wake.wait(self.interval)
            self._wake.clear()

    def _sync_measurement(self, measurement: Measurement) -> bool:
        try:
            response = self.send(
                self.api_url,
                measurement.api_payload(self.device_id),
                measurement.image_path,
                self.device_token,
            )
            validate_ingest_response(
                response,
                measurement.event_id,
                require_remote_image=self.require_remote_image,
            )
            remote_id = response.get("id")
            remote_image_url = response.get("core_image_url") or response.get("image_url")
            remote_image_public_id = (
                response.get("core_image_public_id") or response.get("image_public_id")
            )
            self.store.mark_synced(
                measurement.event_id,
                int(remote_id) if remote_id is not None else None,
                str(remote_image_url) if remote_image_url else None,
                str(remote_image_public_id) if remote_image_public_id else None,
            )
            return True
        except Exception as exc:
            self.store.mark_sync_failed(measurement.event_id, str(exc))
            return False

    def sync_event(self, event_id: str) -> bool:
        """Synchronize one just-confirmed event before the UI reports cloud success."""

        with self._sync_lock:
            measurement = self.store.get(event_id)
            if measurement is None:
                return False
            if measurement.sync_status == "synced":
                return True
            return self._sync_measurement(measurement)

    def sync_once(self, limit: int = 20, include_deferred: bool = False) -> int:
        with self._sync_lock:
            synced = 0
            for measurement in self.store.pending(limit, include_deferred=include_deferred):
                if self._sync_measurement(measurement):
                    synced += 1
            return synced

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(12.0, self.interval + 1.0))
