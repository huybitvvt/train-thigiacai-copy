from __future__ import annotations

import threading
import base64
import time
import urllib.request
from pathlib import Path
from collections.abc import Callable

from .api_client import post_measurement, validate_ingest_response
from .storage import InventoryCheck, Measurement, MeasurementStore


SendFunction = Callable[[str, dict[str, object], str, str], dict[str, object]]
RemoteCheckFunction = Callable[[str], bool]


def _default_send(
    url: str,
    payload: dict[str, object],
    image_path: str,
    token: str,
) -> dict[str, object]:
    return post_measurement(url, payload, image_path, token)


def _default_remote_check(url: str) -> bool:
    request = urllib.request.Request(
        url,
        headers={"Accept": "image/*", "User-Agent": "tram-can-cloud-reconcile/1"},
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=15.0) as response:
        content_type = str(response.headers.get("content-type", "")).lower()
        return 200 <= response.status < 300 and content_type.startswith("image/")


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
        remote_check: RemoteCheckFunction = _default_remote_check,
        maintenance_interval: float = 300.0,
        retention_days: float = 7.0,
        reconcile_recheck_hours: float = 24.0,
    ):
        self.store = store
        self.api_url = api_url
        self.device_token = device_token
        # device_id is retained as a constructor fallback for legacy rows. New
        # captures carry their gateway identity in the outbox row itself.
        self.device_id = gateway_id if gateway_id is not None else device_id
        self.require_remote_image = require_remote_image
        self.remote_check = remote_check
        self.maintenance_interval = max(10.0, float(maintenance_interval))
        self.retention_days = max(0.0, float(retention_days))
        self.reconcile_recheck_hours = max(0.0, float(reconcile_recheck_hours))
        self.interval = interval
        self.send = send
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._sync_lock = threading.Lock()
        self._last_maintenance = 0.0
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
            now = time.monotonic()
            if now - self._last_maintenance >= self.maintenance_interval:
                try:
                    self.maintenance_once()
                except Exception:
                    # Maintenance must never stop the durable outbox loop.
                    pass
                finally:
                    self._last_maintenance = now
            self._wake.wait(self.interval)
            self._wake.clear()

    def _sync_measurement(self, measurement: Measurement) -> bool:
        try:
            payload = measurement.api_payload(self.device_id)
            if measurement.product_image_path:
                payload["product_image_base64"] = base64.b64encode(
                    Path(measurement.product_image_path).read_bytes()
                ).decode("ascii")
            no_image = "IMAGE_SOURCE=NONE" in str(measurement.weight_raw or "")
            response = self.send(
                self.api_url,
                payload,
                measurement.image_path,
                self.device_token,
            )
            validate_ingest_response(
                response,
                measurement.event_id,
                require_remote_image=self.require_remote_image and not no_image,
                require_product_image=bool(measurement.product_image_path) and not no_image,
            )
            remote_id = response.get("id")
            remote_image_url = response.get("core_image_url") or response.get("image_url")
            remote_image_public_id = (
                response.get("core_image_public_id") or response.get("image_public_id")
            )
            remote_product_image_url = response.get("product_image_url")
            remote_product_image_public_id = response.get("product_image_public_id")
            self.store.mark_synced(
                measurement.event_id,
                int(remote_id) if remote_id is not None else None,
                str(remote_image_url) if remote_image_url else None,
                str(remote_image_public_id) if remote_image_public_id else None,
                (
                    str(remote_product_image_url)
                    if remote_product_image_url
                    else None
                ),
                (
                    str(remote_product_image_public_id)
                    if remote_product_image_public_id
                    else None
                ),
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

    def _reconcile_measurement(self, measurement: Measurement) -> bool:
        try:
            current = measurement
            product_expected = bool(
                current.product_image_path
                or current.remote_product_image_url
                or current.remote_product_image_public_id
            )
            core_pair = (current.remote_image_url, current.remote_image_public_id)
            product_pair = (
                current.remote_product_image_url,
                current.remote_product_image_public_id,
            )
            if not all(core_pair) or (product_expected and not all(product_pair)):
                if not Path(current.image_path).is_file():
                    raise RuntimeError("Thiếu định danh ảnh cloud và ảnh local đã mất")
                if not self._sync_measurement(current):
                    raise RuntimeError("Không bổ sung được định danh ảnh cloud")
                current = self.store.get(current.event_id) or current
                core_pair = (current.remote_image_url, current.remote_image_public_id)
                product_pair = (
                    current.remote_product_image_url,
                    current.remote_product_image_public_id,
                )
            if not all(core_pair):
                raise RuntimeError("Supabase thiếu URL/public ID ảnh lõi")
            if product_expected and not all(product_pair):
                raise RuntimeError("Supabase thiếu URL/public ID ảnh sản phẩm")
            if not self.remote_check(str(core_pair[0])):
                raise RuntimeError("Ảnh lõi Cloudinary không tải được")
            if product_expected and not self.remote_check(str(product_pair[0])):
                raise RuntimeError("Ảnh sản phẩm Cloudinary không tải được")
            self.store.mark_cloud_verified(current.event_id)
            return True
        except Exception as exc:
            self.store.mark_cloud_check_failed(measurement.event_id, str(exc))
            return False

    def _reconcile_inventory_check(self, check: InventoryCheck) -> bool:
        try:
            current = check
            if not current.remote_image_url or not current.remote_image_public_id:
                if not Path(current.image_path).is_file():
                    raise RuntimeError("Thiếu định danh ảnh cloud và ảnh local đã mất")
                if not self._sync_inventory_check(current):
                    raise RuntimeError("Không bổ sung được định danh ảnh cloud")
                current = self.store.get_inventory_check(current.event_id) or current
            if not current.remote_image_url or not current.remote_image_public_id:
                raise RuntimeError("Supabase thiếu URL/public ID ảnh kiểm kho")
            if not self.remote_check(current.remote_image_url):
                raise RuntimeError("Ảnh kiểm kho Cloudinary không tải được")
            self.store.mark_cloud_verified(current.event_id, inventory=True)
            return True
        except Exception as exc:
            self.store.mark_cloud_check_failed(check.event_id, str(exc), inventory=True)
            return False

    def reconcile_event(self, event_id: str, *, inventory: bool = False) -> bool:
        with self._sync_lock:
            if inventory:
                check = self.store.get_inventory_check(event_id)
                return bool(check and self._reconcile_inventory_check(check))
            measurement = self.store.get(event_id)
            return bool(measurement and self._reconcile_measurement(measurement))

    def reconcile_once(self, limit: int = 20) -> int:
        with self._sync_lock:
            candidates: list[Measurement | InventoryCheck] = [
                *self.store.reconciliation_candidates(
                    limit,
                    recheck_after_hours=self.reconcile_recheck_hours,
                ),
                *self.store.inventory_reconciliation_candidates(
                    limit,
                    recheck_after_hours=self.reconcile_recheck_hours,
                ),
            ]
            candidates.sort(key=lambda item: (item.captured_at, item.id))
            verified = 0
            for item in candidates[:limit]:
                ok = (
                    self._reconcile_inventory_check(item)
                    if isinstance(item, InventoryCheck)
                    else self._reconcile_measurement(item)
                )
                if ok:
                    verified += 1
            return verified

    def maintenance_once(self, limit: int = 20) -> dict[str, int]:
        verified = self.reconcile_once(limit=limit)
        cleaned = self.store.cleanup_verified_local_images(self.retention_days)
        return {"verified": verified, **cleaned}

    def sync_once(
        self,
        limit: int = 20,
        include_deferred: bool = False,
        *,
        retry_failed: bool = True,
    ) -> int:
        with self._sync_lock:
            synced = 0
            queued: list[Measurement | InventoryCheck] = [
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
            ]
            queued.sort(key=lambda item: (item.captured_at, item.id))
            for item in queued[:limit]:
                succeeded = (
                    self._sync_inventory_check(item)
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
