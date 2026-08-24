from __future__ import annotations

import threading
import base64
from pathlib import Path
from collections.abc import Callable

from .api_client import post_measurement, validate_ingest_response
from .storage import InventoryCheck, Measurement, MeasurementStore, PhotoDraft


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
            # Failed uploads stay in the durable outbox with exponential
            # backoff. Event IDs make retries idempotent at the cloud boundary.
            self.sync_once(retry_failed=True)
            self._wake.wait(self.interval)
            self._wake.clear()

    def _sync_measurement(self, measurement: Measurement) -> bool:
        try:
            payload = measurement.api_payload(self.device_id)
            if measurement.product_image_path:
                payload["product_image_base64"] = base64.b64encode(
                    Path(measurement.product_image_path).read_bytes()
                ).decode("ascii")
            response = self.send(
                self.api_url,
                payload,
                measurement.image_path,
                self.device_token,
            )
            validate_ingest_response(
                response,
                measurement.event_id,
                require_remote_image=self.require_remote_image,
                require_product_image=bool(measurement.product_image_path),
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

    def _sync_inventory_check(self, check: InventoryCheck) -> bool:
        try:
            response = self.send(
                self.api_url,
                check.api_payload(self.device_id),
                check.image_path,
                self.device_token,
            )
            validate_ingest_response(
                response,
                check.event_id,
                require_remote_image=self.require_remote_image,
            )
            remote_id = response.get("id")
            remote_image_url = response.get("image_url") or response.get("core_image_url")
            remote_image_public_id = (
                response.get("image_public_id") or response.get("core_image_public_id")
            )
            self.store.mark_inventory_check_synced(
                check.event_id,
                int(remote_id) if remote_id is not None else None,
                str(remote_image_url) if remote_image_url else None,
                str(remote_image_public_id) if remote_image_public_id else None,
            )
            return True
        except Exception as exc:
            self.store.mark_inventory_check_failed(check.event_id, str(exc))
            return False

    def _sync_photo_draft(self, draft: PhotoDraft) -> bool:
        try:
            response = self.send(
                self.api_url,
                draft.api_payload(self.device_id),
                draft.image_path,
                self.device_token,
            )
            validate_ingest_response(
                response,
                draft.event_id,
                require_remote_image=self.require_remote_image,
            )
            remote_id = response.get("id")
            remote_image_url = response.get("image_url") or response.get("core_image_url")
            remote_image_public_id = (
                response.get("image_public_id") or response.get("core_image_public_id")
            )
            self.store.mark_photo_draft_synced(
                draft.event_id,
                int(remote_id) if remote_id is not None else None,
                str(remote_image_url) if remote_image_url else None,
                str(remote_image_public_id) if remote_image_public_id else None,
            )
            return True
        except Exception as exc:
            self.store.mark_photo_draft_failed(draft.event_id, str(exc))
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

    def sync_inventory_event(self, event_id: str) -> bool:
        """Synchronize one inventory check while retaining failed rows locally."""

        with self._sync_lock:
            check = self.store.get_inventory_check(event_id)
            if check is None:
                return False
            if check.sync_status == "synced":
                return True
            return self._sync_inventory_check(check)

    def sync_photo_draft_event(self, event_id: str) -> bool:
        """Synchronize a photo-only draft without invoking the weight AI."""

        with self._sync_lock:
            draft = self.store.get_photo_draft(event_id)
            if draft is None:
                return False
            if draft.sync_status == "synced":
                return True
            return self._sync_photo_draft(draft)

    def sync_once(
        self,
        limit: int = 20,
        include_deferred: bool = False,
        *,
        retry_failed: bool = True,
    ) -> int:
        with self._sync_lock:
            synced = 0
            queued: list[Measurement | InventoryCheck | PhotoDraft] = [
                *self.store.pending(
                    limit,
                    include_deferred=include_deferred,
                    include_failed=retry_failed,
                ),
                *self.store.pending_inventory_checks(
                    limit,
                    include_deferred=include_deferred,
                    include_failed=retry_failed,
                ),
                *self.store.pending_photo_drafts(
                    limit,
                    include_deferred=include_deferred,
                    include_failed=retry_failed,
                ),
            ]
            queued.sort(key=lambda item: (item.captured_at, item.id))
            for item in queued[:limit]:
                succeeded = (
                    self._sync_photo_draft(item)
                    if isinstance(item, PhotoDraft)
                    else self._sync_inventory_check(item)
                    if isinstance(item, InventoryCheck)
                    else self._sync_measurement(item)
                )
                if succeeded:
                    synced += 1
            return synced

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(12.0, self.interval + 1.0))
