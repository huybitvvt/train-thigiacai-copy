from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import json
import math
import os
import re
import inspect
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .factory_dataset import FactorySampleStore
from .gemini_weight import (
    DEFAULT_GEMINI_ACCURATE_MODEL,
    DEFAULT_GEMINI_ACCURATE_TIMEOUT_SECONDS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_TIMEOUT_SECONDS,
    GeminiWeightReader,
)
from .gemini_key_manager import GeminiKeyManager
from .codex_weight import DEFAULT_CODEX_TIMEOUT_SECONDS, CodexWeightReader
from .codex_oauth import CodexOAuthClient, EncryptedCodexTokenStore
from .codex_oauth_weight import CodexOAuthWeightReader
from .capture_gate import frame_fingerprint
from .api_client import fetch_supabase_table, persist_product_evidence, sign_storage_image
from .inference_queue import InferenceCoordinator, InferenceQueueFull
from .lookup_client import lookup_roll
from .quality import FrameQuality, assess_frame_quality
from .qr_reader import QRReader
from .scale import WeightReading
from .station_session import (
    AnalysisBindingMismatch,
    AnalysisBindingNotFound,
    SessionConflictError,
    StationSessionRegistry,
    encode_staged_jpeg,
    jpeg_sha256,
)
from .storage import EventIdConflictError, MeasurementStore
from .sync import OutboxSyncWorker
from .weight_ocr import (
    CameraOCRWeightSource,
    NormalizedROI,
    PADDLE_OCR_MODEL_NAME,
    PaddleOCRTextReader,
    detect_weight_roi,
    detect_weight_roi_consensus,
    parse_normalized_roi,
)


MAX_REQUEST_BYTES = 24 * 1024 * 1024
MAX_BURST_FRAMES = 9
DEFAULT_WEIGHT_BURST_FRAMES = 5
UNITS = {"kg", "g", "lb"}
WEIGHT_ENGINES = {"local", "hybrid", "gemini"}
GEMINI_RECOGNITION_PROFILES = {"fast", "accurate"}
AI_RECOGNITION_PROVIDERS = {"gemini", "codex"}


def _supabase_project_url() -> str:
    configured = os.environ.get("ROLL_SCALE_SUPABASE_URL", "").strip().rstrip("/")
    if configured:
        return configured
    api_url = os.environ.get("ROLL_SCALE_API_URL", "").strip()
    marker = "/functions/v1/"
    if marker in api_url:
        return api_url.split(marker, 1)[0].rstrip("/")
    return ""


def _supabase_read_key() -> str:
    return (
        os.environ.get("ROLL_SCALE_SUPABASE_SERVICE_KEY", "").strip()
        or os.environ.get("ROLL_SCALE_SUPABASE_PUBLISHABLE_KEY", "").strip()
    )


def _raw_tag(raw: str, name: str) -> str:
    match = re.search(rf"(?:^|; )\s*{re.escape(name)}=([^;]+)", raw or "")
    return match.group(1).strip() if match else ""


def _item_work_date(captured_at: str, weight_raw: str = "") -> str:
    tagged = _raw_tag(weight_raw, "SOURCE_DATE")
    if tagged:
        return tagged
    text = str(captured_at or "")
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return ""


def _item_source_value(item: dict[str, object], field: str, tag: str) -> str:
    direct = str(item.get(field) or "").strip()
    if direct:
        return direct
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        metadata_value = str(metadata.get(field) or "").strip()
        if metadata_value:
            return metadata_value
        raw = str(metadata.get("weight_raw") or item.get("weight_raw") or "")
    else:
        raw = str(item.get("weight_raw") or "")
    return _raw_tag(raw, tag)


def _item_source_date(item: dict[str, object]) -> str:
    selected = _item_source_value(item, "work_date", "SOURCE_DATE")
    if selected:
        return selected
    metadata = item.get("metadata")
    raw = (
        str(metadata.get("weight_raw") or "")
        if isinstance(metadata, dict)
        else str(item.get("weight_raw") or "")
    )
    return _item_work_date(str(item.get("captured_at") or ""), raw)


def _production_orders_for_date(
    items: list[dict[str, object]], work_date: str
) -> list[str]:
    orders = {
        _item_source_value(item, "production_order", "SOURCE_PRODUCTION_ORDER")
        for item in items
        if _item_source_date(item) == work_date
    }
    return sorted((order for order in orders if order), key=str.casefold)


def _matches_source_filters(
    item: dict[str, object],
    *,
    work_date: str = "",
    shift: str = "",
    machine: str = "",
    production_order: str = "",
) -> bool:
    if work_date:
        item_date = _item_source_date(item)
        if item_date != work_date:
            return False
    if shift:
        item_shift = _item_source_value(item, "shift", "SOURCE_SHIFT")
        if item_shift and item_shift != shift:
            return False
        if not item_shift:
            # Legacy rows without source tags still show for the selected day.
            pass
    if machine:
        item_machine = _item_source_value(item, "machine", "SOURCE_MACHINE")
        if item_machine and item_machine != machine:
            return False
    if production_order:
        item_order = _item_source_value(
            item, "production_order", "SOURCE_PRODUCTION_ORDER"
        )
        if item_order and item_order != production_order:
            return False
    return True


def _merge_source_tags(weight_raw: str, payload: dict[str, object]) -> str:
    """Ensure Ca / Máy / Lệnh sản xuất tags are present for Supabase metadata."""

    raw = str(weight_raw or "").strip()
    extras: list[str] = []
    mapping = (
        ("work_date", "SOURCE_DATE"),
        ("shift", "SOURCE_SHIFT"),
        ("machine", "SOURCE_MACHINE"),
        ("production_order", "SOURCE_PRODUCTION_ORDER"),
    )
    for field, tag in mapping:
        value = re.sub(r"[;\r\n]+", " ", str(payload.get(field, "") or "")).strip()
        if not value or _raw_tag(raw, tag):
            continue
        extras.append(f"{tag}={value[:80]}")
    if not _raw_tag(raw, "BI_WEIGHT"):
        bi_value = payload.get("bi_weight", payload.get("bi"))
        try:
            bi_number = float(bi_value) if bi_value is not None and bi_value != "" else 0.16
        except (TypeError, ValueError):
            bi_number = 0.16
        if bi_number < 0:
            bi_number = 0.16
        extras.append(f"BI_WEIGHT={bi_number:g}")
    if not extras:
        return raw
    prefix = "; ".join(extras)
    return f"{prefix}; {raw}" if raw else prefix


def _persistable_weight_raw(
    weight_raw: str,
    *,
    weight: float,
    vision_confirmed: bool,
) -> str:
    raw = str(weight_raw or "").strip()
    if not vision_confirmed:
        if not raw.startswith("MANUAL:"):
            raw = f"MANUAL:{weight}" + (f"; {raw}" if raw else "")
    return raw[:1000]


def _local_measurement_items(
    store: MeasurementStore,
    limit: int,
    *,
    work_date: str = "",
    shift: str = "",
    machine: str = "",
    production_order: str = "",
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for item in store.recent(max(limit, 200)):
        product_weight = item.product_weight
        if product_weight is None:
            match = re.search(
                r"(?:^|; )PRODUCT_WEIGHT=([0-9]+(?:\.[0-9]+)?)",
                item.weight_raw or "",
            )
            product_weight = float(match.group(1)) if match else None
        has_core = bool(item.image_path and Path(item.image_path).is_file())
        has_product = bool(
            item.product_image_path and Path(item.product_image_path).is_file()
        )
        core_url = (
            f"/api/measurement-image?event_id={urllib.parse.quote(item.event_id)}&kind=core"
            if has_core
            else item.remote_image_url
        )
        product_url = (
            f"/api/measurement-image?event_id={urllib.parse.quote(item.event_id)}&kind=product"
            if has_product
            else None
        )
        payload = {
            "event_id": item.event_id,
            "qr_code": item.qr_code,
            "core_weight": item.weight,
            "product_weight": product_weight,
            "weight_raw": item.weight_raw
            or (
                f"PRODUCT_WEIGHT={product_weight}"
                if product_weight is not None
                else ""
            ),
            "unit": item.unit,
            "captured_at": item.captured_at,
            "sync_status": item.sync_status,
            "sync_error": item.sync_error,
            "core_image_url": core_url,
            "product_image_url": product_url,
            "has_core_image": bool(core_url),
            "has_product_image": bool(product_url),
        }
        if _matches_source_filters(
            payload,
            work_date=work_date,
            shift=shift,
            machine=machine,
            production_order=production_order,
        ):
            items.append(payload)
        if len(items) >= limit:
            break
    return items


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} phải là true hoặc false")


def decode_image(value: str) -> np.ndarray:
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("Ảnh quá lớn")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Ảnh base64 không hợp lệ") from exc
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không giải mã được ảnh")
    return frame


class StationUIService:
    def __init__(
        self,
        store: MeasurementStore,
        sync_worker: OutboxSyncWorker | None,
        lookup_url: str | None,
        lookup_token: str | None,
        duplicate_window: float = 5.0,
        reader: QRReader | None = None,
        ocr_min_confidence: float = 0.60,
        ocr_download: bool = False,
        sample_store: FactorySampleStore | None = None,
        min_frame_width: int = 640,
        min_frame_height: int = 480,
        min_brightness: float = 30.0,
        max_brightness: float = 250.0,
        min_sharpness: float = 35.0,
        *,
        gateway_id: str = "gateway-01",
        station_count: int = 1,
        station_ids: list[str] | tuple[str, ...] | None = None,
        camera_ids: list[str] | tuple[str, ...] | None = None,
        staging_dir: str | Path | None = None,
        diagnostic_image: str | Path | None = None,
        inference_queue_size: int = 8,
        inference_coordinator: InferenceCoordinator | None = None,
        auto_advance: bool = True,
        weight_rois: list[str] | tuple[str, ...] | None = None,
        weight_burst_frames: int = DEFAULT_WEIGHT_BURST_FRAMES,
        gemini_reader: GeminiWeightReader | None = None,
        gemini_accurate_reader: GeminiWeightReader | None = None,
        gemini_key_manager: GeminiKeyManager | None = None,
        codex_reader: CodexWeightReader | CodexOAuthWeightReader | None = None,
        weight_engine: str = "local",
    ):
        if not 1 <= int(station_count) <= 3:
            raise ValueError("station_count phải từ 1 đến 3")
        configured_station_ids = list(station_ids or ())
        if configured_station_ids and len(configured_station_ids) != int(station_count):
            raise ValueError("Số station_id phải bằng station_count")
        if not configured_station_ids:
            configured_station_ids = [f"station-{index:02d}" for index in range(1, station_count + 1)]
        configured_camera_ids = list(camera_ids or ())
        if configured_camera_ids and len(configured_camera_ids) != int(station_count):
            raise ValueError("Số camera_id phải bằng station_count")
        if not configured_camera_ids:
            configured_camera_ids = [f"camera-{index:02d}" for index in range(1, station_count + 1)]
        if len(set(configured_camera_ids)) != len(configured_camera_ids):
            raise ValueError("camera_id values must be unique")
        configured_weight_rois = list(weight_rois or ())
        if configured_weight_rois and len(configured_weight_rois) != int(station_count):
            raise ValueError("Số weight ROI phải bằng station_count")
        if not 1 <= int(weight_burst_frames) <= MAX_BURST_FRAMES:
            raise ValueError(f"weight_burst_frames phải từ 1 đến {MAX_BURST_FRAMES}")
        if weight_engine not in WEIGHT_ENGINES:
            raise ValueError("weight_engine phải là local, hybrid hoặc gemini")
        if weight_engine in {"hybrid", "gemini"} and gemini_reader is None:
            raise ValueError(f"weight_engine={weight_engine} cần Gemini reader")
        parsed_weight_rois = [parse_normalized_roi(value) for value in configured_weight_rois]
        self.store = store
        self.sync_worker = sync_worker
        self.lookup_url = lookup_url
        self.lookup_token = lookup_token
        self.duplicate_window = duplicate_window
        self.reader = reader or QRReader()
        self.ocr_min_confidence = ocr_min_confidence
        self.ocr_download = ocr_download
        self.sample_store = sample_store
        self.gateway_id = gateway_id
        self.auto_advance = bool(auto_advance)
        self.weight_burst_frames = int(weight_burst_frames)
        self.gemini_reader = gemini_reader
        self.gemini_accurate_reader = gemini_accurate_reader
        self.gemini_key_manager = gemini_key_manager
        self._retired_gemini_readers: list[GeminiWeightReader] = []
        self.codex_reader = codex_reader
        self.weight_engine = weight_engine
        self.weight_rois: dict[str, NormalizedROI] = {
            station_id: roi
            for station_id, roi in zip(configured_station_ids, parsed_weight_rois)
        }
        self.station_count = int(station_count)
        self.station_configs = [
            {
                "index": index,
                "station_id": station_id,
                "camera_id": configured_camera_ids[index - 1],
                "weight_roi": self._roi_text(self.weight_rois.get(station_id)),
            }
            for index, station_id in enumerate(configured_station_ids, start=1)
        ]
        default_staging_dir = Path(getattr(store, "capture_dir", "data/captures")) / ".staging"
        self.sessions = StationSessionRegistry(
            staging_dir or default_staging_dir,
            configured_station_ids,
        )
        for station_id, camera_id in zip(configured_station_ids, configured_camera_ids):
            self.sessions.configure_camera(station_id, camera_id)
        self.inference = inference_coordinator or InferenceCoordinator(inference_queue_size)
        self._owns_inference = inference_coordinator is None
        self.diagnostic_image = Path(diagnostic_image) if diagnostic_image else None
        self.quality_settings = {
            "min_width": min_frame_width,
            "min_height": min_frame_height,
            "min_brightness": min_brightness,
            "max_brightness": max_brightness,
            "min_sharpness": min_sharpness,
        }
        self._ocr_reader: object | None = None
        self._ocr_lock = threading.Lock()
        self._ocr_preload_started = False
        self._ocr_preload_error: str | None = None
        self._recent: dict[str, float] = {}
        self._lock = threading.Lock()

    def start_ocr_preload(self) -> None:
        if self.weight_engine == "gemini":
            return
        if self._ocr_preload_started:
            return
        self._ocr_preload_started = True
        threading.Thread(
            target=self._preload_ocr,
            name="paddleocr-preload",
            daemon=True,
        ).start()

    def _preload_ocr(self) -> None:
        try:
            with self._ocr_lock:
                if self._ocr_reader is None:
                    self._ocr_reader = PaddleOCRTextReader.create(
                        download_enabled=self.ocr_download,
                        gpu=False,
                    )
            self._ocr_preload_error = None
        except RuntimeError as exc:
            self._ocr_preload_error = str(exc)

    def close(self) -> None:
        if self._owns_inference:
            self.inference.close()
        if self.gemini_reader is not None:
            self.gemini_reader.close()
        if (
            self.gemini_accurate_reader is not None
            and self.gemini_accurate_reader is not self.gemini_reader
        ):
            self.gemini_accurate_reader.close()
        for reader in self._retired_gemini_readers:
            reader.close()
        if self.codex_reader is not None:
            self.codex_reader.close()

    @staticmethod
    def _reader_model(reader: GeminiWeightReader | None) -> str | None:
        if reader is None:
            return None
        model = getattr(reader, "model", None)
        if model:
            return str(model)
        status = reader.status()
        value = status.get("model") if isinstance(status, dict) else None
        return str(value) if value else None

    def _gemini_reader_for(self, profile: str) -> GeminiWeightReader:
        if profile not in GEMINI_RECOGNITION_PROFILES:
            raise ValueError("Chế độ nhận diện phải là fast hoặc accurate")
        if profile == "accurate":
            if self.gemini_accurate_reader is None:
                raise ValueError("Chế độ Chuẩn chưa được cấu hình")
            return self.gemini_accurate_reader
        if self.gemini_reader is None:
            raise RuntimeError("Gemini primary chưa được cấu hình")
        return self.gemini_reader

    def replace_gemini_key(self, api_key: str) -> dict[str, object]:
        if self.gemini_key_manager is None:
            raise ValueError("Chức năng đổi Gemini key chưa được cấu hình")
        fast, accurate = self.gemini_key_manager.replace(api_key)
        with self._lock:
            old_fast = self.gemini_reader
            old_accurate = self.gemini_accurate_reader
            self.gemini_reader = fast
            self.gemini_accurate_reader = accurate
        if old_fast is not None:
            self._retired_gemini_readers.append(old_fast)
        if old_accurate is not None and old_accurate is not old_fast:
            self._retired_gemini_readers.append(old_accurate)
        return {
            "ok": True,
            "changed": True,
            "stored_encrypted": True,
            "key_id": self.gemini_key_manager.key_id(api_key),
            "message": "Đã kiểm tra, mã hóa và áp dụng Gemini API key mới",
        }

    @staticmethod
    def _roi_text(roi: NormalizedROI | None) -> str:
        if roi is None:
            return ""
        return f"{roi.x1:.4f},{roi.y1:.4f},{roi.x2:.4f},{roi.y2:.4f}"

    @staticmethod
    def _gemini_crop(frame: np.ndarray, roi: NormalizedROI) -> np.ndarray:
        left, top, right, bottom = roi.pixels(frame)
        width = max(1, right - left)
        height = max(1, bottom - top)
        pad_x = max(4, round(width * 0.12))
        pad_y = max(4, round(height * 0.18))
        frame_height, frame_width = frame.shape[:2]
        return frame[
            max(0, top - pad_y) : min(frame_height, bottom + pad_y),
            max(0, left - pad_x) : min(frame_width, right + pad_x),
        ]

    @staticmethod
    def _distant_weight_roi(frame: np.ndarray) -> tuple[NormalizedROI, str] | None:
        """Find a tiny red LED row in a portrait factory overview.

        The normal detector is deliberately strict for OCR. This fallback is
        only used to create a contextual zoom inset for AI and cloud evidence.
        """

        if frame.ndim != 3 or frame.shape[2] < 3:
            return None
        height, width = frame.shape[:2]
        blue, green, red = cv2.split(frame[:, :, :3])
        red_i16 = red.astype(np.int16)
        other = np.maximum(green, blue).astype(np.int16)
        mask = ((red_i16 >= 120) & (red_i16 - other >= 45)).astype(np.uint8)
        mask[: round(height * 0.30), :] = 0
        joined = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for index in range(1, count):
            left, top, box_width, box_height, _ = (int(value) for value in stats[index])
            if box_width < 5 or box_height < 5:
                continue
            if box_width > width * 0.08 or box_height > height * 0.035:
                continue
            aspect = box_width / max(1, box_height)
            if not 0.8 <= aspect <= 12:
                continue
            region = mask[top : top + box_height, left : left + box_width]
            red_pixels = int(np.count_nonzero(region))
            density = red_pixels / max(1, box_width * box_height)
            if red_pixels < 8 or density < 0.08:
                continue
            center_bias = (top + (box_height / 2)) / max(1, height)
            score = red_pixels * (1 + min(aspect, 4) * 0.12) + center_bias * 8
            candidates.append(
                (score, (left, top, left + box_width, top + box_height))
            )
        if not candidates:
            return None
        _, (left, top, right, bottom) = max(candidates, key=lambda item: item[0])
        return NormalizedROI(
            left / width,
            top / height,
            right / width,
            bottom / height,
        ), "distant-red-led"

    @staticmethod
    def _scale_context_crop(frame: np.ndarray, roi: NormalizedROI) -> np.ndarray:
        """Include the indicator, scale platform, and weighed object in the zoom."""

        left, top, right, bottom = roi.pixels(frame)
        frame_height, frame_width = frame.shape[:2]
        roi_width = max(1, right - left)
        roi_height = max(1, bottom - top)
        crop_width = min(
            frame_width,
            max(round(frame_width * 0.40), roi_width * 10),
        )
        crop_height = min(
            frame_height,
            max(round(frame_height * 0.24), roi_height * 12),
        )
        center_x = (left + right) / 2
        # Keep more room above the LED row for the platform and weighed item.
        crop_left = round(center_x - (crop_width / 2))
        crop_top = round(((top + bottom) / 2) - (crop_height * 0.64))
        crop_left = min(max(0, crop_left), frame_width - crop_width)
        crop_top = min(max(0, crop_top), frame_height - crop_height)
        return frame[
            crop_top : crop_top + crop_height,
            crop_left : crop_left + crop_width,
        ]

    @classmethod
    def _zoomed_evidence(
        cls,
        frame: np.ndarray,
        roi: NormalizedROI,
    ) -> tuple[np.ndarray, NormalizedROI]:
        """Keep the full scene and append a large, clearly marked scale-display inset."""

        crop = cls._scale_context_crop(frame, roi)
        crop_height, crop_width = crop.shape[:2]
        frame_height, frame_width = frame.shape[:2]
        if crop_height < 1 or crop_width < 1:
            return frame, roi

        margin = max(12, round(min(frame_height, frame_width) * 0.012))
        label_height = max(30, round(frame_height * 0.04))
        portrait = frame_height > frame_width * 1.15
        target_width = max(1, frame_width - (2 * margin))
        target_height = (
            max(120, frame_height - label_height - (3 * margin))
            if portrait
            else max(120, round(frame_height * 0.34))
        )
        scale = min(target_width / crop_width, target_height / crop_height)
        zoom_width = max(1, min(target_width, round(crop_width * scale)))
        zoom_height = max(1, min(target_height, round(crop_height * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        zoomed = cv2.resize(crop, (zoom_width, zoom_height), interpolation=interpolation)

        if portrait:
            composite = np.full(
                (frame_height, frame_width * 2, 3),
                24,
                dtype=frame.dtype,
            )
            zoom_left = frame_width + ((frame_width - zoom_width) // 2)
            zoom_top = label_height + margin + (
                (frame_height - label_height - (2 * margin) - zoom_height) // 2
            )
            label_origin = (frame_width + margin, label_height)
        else:
            panel_height = label_height + zoom_height + (2 * margin)
            composite = np.full(
                (frame_height + panel_height, frame_width, 3),
                24,
                dtype=frame.dtype,
            )
            zoom_left = (frame_width - zoom_width) // 2
            zoom_top = frame_height + label_height + margin
            label_origin = (margin, frame_height + label_height)
        composite[:frame_height, :frame_width] = frame
        composite[
            zoom_top : zoom_top + zoom_height,
            zoom_left : zoom_left + zoom_width,
        ] = zoomed
        cv2.rectangle(
            composite,
            (max(0, zoom_left - 2), max(0, zoom_top - 2)),
            (
                min(composite.shape[1] - 1, zoom_left + zoom_width + 1),
                min(composite.shape[0] - 1, zoom_top + zoom_height + 1),
            ),
            (230, 230, 230),
            2,
        )
        cv2.putText(
            composite,
            "ZOOM MAN HINH CAN",
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.55, frame_width / 1800),
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        composite_height = composite.shape[0]
        composite_width = composite.shape[1]
        zoom_roi = NormalizedROI(
            zoom_left / composite_width,
            zoom_top / composite_height,
            (zoom_left + zoom_width) / composite_width,
            (zoom_top + zoom_height) / composite_height,
        )
        return composite, zoom_roi

    @staticmethod
    def _canonical_evidence(frame: np.ndarray) -> tuple[np.ndarray, str]:
        encoded = encode_staged_jpeg(frame, quality=90)
        canonical = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if canonical is None:
            raise OSError("Không chuẩn hóa được ảnh zoom cân")
        data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
        return canonical, data_url

    def status(self) -> dict[str, object]:
        station_states = {item["station_id"]: item for item in self.sessions.statuses()}
        stations: list[dict[str, object]] = []
        for config in self.station_configs:
            station = dict(config)
            station.update(station_states.get(str(config["station_id"]), {}))
            station["diagnostic_path"] = str(
                self.diagnostic_path_for(str(config["station_id"])) or ""
            )
            stations.append(station)
        return {
            "gateway_id": self.gateway_id,
            "station_count": self.station_count,
            "auto_advance": self.auto_advance,
            "weight_burst_frames": self.weight_burst_frames,
            "weight_engine": self.weight_engine,
            "ocr_ready": self._ocr_reader is not None,
            "ocr_preload_error": self._ocr_preload_error,
            "gemini": (
                {
                    **self.gemini_reader.status(),
                    **(self.gemini_key_manager.status() if self.gemini_key_manager else {}),
                }
                if self.gemini_reader is not None
                else {"enabled": False}
            ),
            "codex": (
                self.codex_reader.status()
                if self.codex_reader is not None
                else {
                    "enabled": False,
                    "installed": False,
                    "authenticated": False,
                    "available": False,
                }
            ),
            "recognition_providers": {
                "default": "gemini",
                "gemini": {"available": self.gemini_reader is not None},
                "codex": {
                    "available": bool(
                        self.codex_reader is not None
                        and self.codex_reader.status().get("available")
                    )
                },
            },
            "recognition_profiles": {
                "default": "fast",
                "fast": {
                    "enabled": self.gemini_reader is not None,
                    "model": self._reader_model(self.gemini_reader),
                },
                "accurate": {
                    "enabled": self.gemini_accurate_reader is not None,
                    "model": self._reader_model(self.gemini_accurate_reader),
                },
            },
            "stations": stations,
            "inference": self.inference.status().as_dict(),
        }

    def stage_evidence_step(
        self,
        frame: np.ndarray,
        *,
        event_id: str,
        station_id: str,
        kind: str,
        weight: float,
        unit: str,
        qr_code: str = "",
    ) -> dict[str, object]:
        """Durably save each accepted weighing step before final confirmation."""
        if kind not in {"core", "product"}:
            raise ValueError("capture_kind phải là core hoặc product")
        if station_id not in {str(item["station_id"]) for item in self.station_configs}:
            raise ValueError("station_id không hợp lệ")
        uuid.UUID(event_id)
        folder = self.sessions.staging_dir / station_id
        folder.mkdir(parents=True, exist_ok=True)
        image_path = folder / f"{event_id}_{kind}.jpg"
        metadata_path = folder / f"{event_id}_steps.json"
        encoded_ok, encoded = cv2.imencode(".jpg", frame)
        if not encoded_ok:
            raise OSError("Không mã hóa được ảnh bằng chứng")
        with self._lock:
            image_path.write_bytes(encoded.tobytes())
            metadata: dict[str, object] = {"event_id": event_id, "station_id": station_id}
            if metadata_path.is_file():
                try:
                    loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        metadata.update(loaded)
                except (OSError, json.JSONDecodeError):
                    pass
            metadata[kind] = {
                "image_path": str(image_path.resolve()),
                "weight": float(weight),
                "unit": unit,
                "qr_code": qr_code.strip(),
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return {"image_path": str(image_path.resolve()), "metadata_path": str(metadata_path.resolve())}

    def analyze_panel_regions(
        self,
        frame: np.ndarray,
        regions: object,
        *,
        recognition_profile: str = "fast",
    ) -> dict[str, object]:
        if self.gemini_reader is None:
            raise ValueError("Gemini chưa được bật trên backend")
        if not isinstance(regions, list) or not 1 <= len(regions) <= 24:
            raise ValueError("Cần cấu hình từ 1 đến 24 vùng chỉ số")
        cropped: list[tuple[str, np.ndarray]] = []
        normalized: list[dict[str, object]] = []
        labels: set[str] = set()
        for item in regions:
            if not isinstance(item, dict):
                raise ValueError("Vùng chỉ số không hợp lệ")
            label = str(item.get("label", "")).strip()
            if not label or len(label) > 80 or label.casefold() in labels:
                raise ValueError("Tên vùng chỉ số trống, trùng hoặc quá dài")
            try:
                roi = NormalizedROI(
                    float(item["x1"]),
                    float(item["y1"]),
                    float(item["x2"]),
                    float(item["y2"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Tọa độ vùng {label} không hợp lệ") from exc
            if not (0 <= roi.x1 < roi.x2 <= 1 and 0 <= roi.y1 < roi.y2 <= 1):
                raise ValueError(f"Tọa độ vùng {label} nằm ngoài ảnh")
            left, top, right, bottom = roi.pixels(frame)
            if right - left < 8 or bottom - top < 8:
                raise ValueError(f"Vùng {label} quá nhỏ")
            labels.add(label.casefold())
            cropped.append((label, frame[top:bottom, left:right]))
            normalized.append(
                {
                    "label": label,
                    "x1": roi.x1,
                    "y1": roi.y1,
                    "x2": roi.x2,
                    "y2": roi.y2,
                }
            )
        reader = self._gemini_reader_for(recognition_profile)
        result = reader.read_panel_regions(cropped)
        return {**result, "regions": normalized}

    def panel_regions(self, station_id: str) -> list[dict[str, object]]:
        station_id = str(station_id).strip()
        if station_id not in {str(item["station_id"]) for item in self.station_configs}:
            raise ValueError("station_id không hợp lệ")
        path = self.sessions.staging_dir.parent / "panel-regions.json"
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        items = stored.get(station_id, []) if isinstance(stored, dict) else []
        return items if isinstance(items, list) else []

    def save_panel_regions(self, station_id: str, regions: object) -> list[dict[str, object]]:
        station_id = str(station_id).strip()
        if station_id not in {str(item["station_id"]) for item in self.station_configs}:
            raise ValueError("station_id không hợp lệ")
        if not isinstance(regions, list) or len(regions) > 24:
            raise ValueError("Danh sách vùng chỉ số không hợp lệ")
        normalized: list[dict[str, object]] = []
        labels: set[str] = set()
        for item in regions:
            if not isinstance(item, dict):
                raise ValueError("Vùng chỉ số không hợp lệ")
            label = str(item.get("label", "")).strip()
            if not label or len(label) > 80 or label.casefold() in labels:
                raise ValueError("Tên vùng chỉ số trống, trùng hoặc quá dài")
            try:
                x1, y1, x2, y2 = (
                    float(item["x1"]),
                    float(item["y1"]),
                    float(item["x2"]),
                    float(item["y2"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Tọa độ vùng {label} không hợp lệ") from exc
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"Tọa độ vùng {label} nằm ngoài ảnh")
            if x2 - x1 < 0.005 or y2 - y1 < 0.005:
                raise ValueError(f"Vùng {label} quá nhỏ")
            labels.add(label.casefold())
            normalized.append(
                {"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            )
        path = self.sessions.staging_dir.parent / "panel-regions.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stored = {}
            if not isinstance(stored, dict):
                stored = {}
            stored[station_id] = normalized
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(path)
        return normalized

    def _cleanup_evidence_steps(self, station_id: str, event_id: str) -> None:
        """Remove transient two-step evidence after save or operator discard."""
        station_id = str(station_id).strip()
        if station_id not in {str(item["station_id"]) for item in self.station_configs}:
            return
        try:
            uuid.UUID(str(event_id))
        except (ValueError, TypeError, AttributeError):
            return
        folder = self.sessions.staging_dir / station_id
        for suffix in ("_core.jpg", "_product.jpg", "_steps.json"):
            try:
                (folder / f"{event_id}{suffix}").unlink(missing_ok=True)
            except OSError:
                pass

    def discard_session(self, station_id: str, *, event_id: str | None = None) -> bool:
        discarded = self.sessions.discard(station_id, event_id=event_id)
        if event_id:
            self._cleanup_evidence_steps(station_id, event_id)
        return discarded

    def diagnostic_path_for(self, station_id: str) -> Path | None:
        if self.diagnostic_image is None:
            return None
        text = str(self.diagnostic_image)
        if "{station_id}" in text:
            return Path(text.format(station_id=station_id))
        if self.station_count == 1:
            return self.diagnostic_image
        return self.diagnostic_image.with_name(
            f"{self.diagnostic_image.stem}_{station_id}{self.diagnostic_image.suffix or '.jpg'}"
        )

    def _write_diagnostic(self, station_id: str, frame: np.ndarray) -> None:
        path = self.diagnostic_path_for(station_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError("Không ghi được ảnh chẩn đoán OCR")

    def _decode_qr(self, frame: np.ndarray) -> dict[str, object]:
        detections = self.reader.decode(frame)
        if not detections:
            return {"ok": True, "found": False}
        detection = detections[0]
        height, width = frame.shape[:2]
        points = np.asarray(detection.points, dtype=np.float32).reshape((-1, 2))
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        qr_roi = (
            f"{max(0.0, x1 / width):.4f},{max(0.0, y1 / height):.4f},"
            f"{min(1.0, x2 / width):.4f},{min(1.0, y2 / height):.4f}"
        )
        return {
            "ok": True,
            "found": True,
            "qr_code": detection.value,
            "decoder": detection.decoder,
            "qr_roi": qr_roi,
        }

    def decode_qr(self, frame: np.ndarray) -> dict[str, object]:
        return self.inference.run(self._decode_qr, frame)

    def assess_quality(self, frame: np.ndarray) -> FrameQuality:
        return assess_frame_quality(frame, **self.quality_settings)

    def quality_result(self, quality: FrameQuality) -> tuple[dict[str, object], bool]:
        payload = quality.as_dict(ignore_low_resolution=self.weight_engine == "gemini")
        return payload, bool(payload["accepted"])

    def _analyze_frame(
        self,
        frame: np.ndarray,
        roi_text: str,
        unit: str,
        weight_frames: list[np.ndarray] | None = None,
        roi_method_override: str | None = None,
        recognition_profile: str = "fast",
        recognition_provider: str = "gemini",
        capture_kind: str = "",
        client_qr_code: str = "",
    ) -> dict[str, object]:
        if unit not in UNITS:
            raise ValueError("Đơn vị không hợp lệ")
        if recognition_profile not in GEMINI_RECOGNITION_PROFILES:
            raise ValueError("Chế độ nhận diện phải là fast hoặc accurate")
        recognition_provider = recognition_provider.strip().lower()
        if recognition_provider not in AI_RECOGNITION_PROVIDERS:
            raise ValueError("Bộ AI nhận diện phải là gemini hoặc codex")
        if capture_kind not in {"", "core", "product"}:
            raise ValueError("capture_kind phải là core hoặc product")
        quality = self.assess_quality(frame)
        quality_payload, quality_pass = self.quality_result(quality)
        # QR belongs to the product step. Avoid spending CPU decoding unrelated
        # pixels while the operator is capturing the core weight.
        decoded = (
            {
                "ok": True,
                "found": False,
                "qr_code": None,
                "decoder": "not-requested-core-step",
                "qr_roi": None,
            }
            if capture_kind == "core"
            else self._decode_qr(frame)
        )
        client_qr = client_qr_code.strip()
        if len(client_qr) > 512 or any(ord(character) < 32 for character in client_qr):
            raise ValueError("Mã QR từ trình duyệt không hợp lệ")
        qr_conflict = False
        if client_qr:
            local_qr = str(decoded.get("qr_code") or "").strip()
            if local_qr and local_qr != client_qr:
                qr_conflict = True
                decoded = {
                    "ok": True,
                    "found": False,
                    "qr_code": None,
                    "decoder": "dedicated-decoder-conflict",
                    "qr_roi": decoded.get("qr_roi"),
                }
            elif local_qr:
                decoded["decoder"] = f"{decoded.get('decoder', 'local')}+browser-confirmed"
            else:
                decoded = {
                    "ok": True,
                    "found": True,
                    "qr_code": client_qr,
                    "decoder": "browser-barcode-detector",
                    "qr_roi": None,
                }
        frames = [frame, *(weight_frames or [])]
        auto_roi = roi_text.strip().lower() in {"", "auto"}
        roi: NormalizedROI | None
        crop_retry_method: str | None = None
        if self.weight_engine == "gemini":
            provider_prefix = recognition_provider
            optimized_capture = capture_kind in {"core", "product"}
            if not optimized_capture:
                roi = None
                roi_method = f"{provider_prefix}-full-frame"
            elif auto_roi:
                located = detect_weight_roi(frame)
                if located is None:
                    roi = None
                else:
                    roi, detected_method = located
                    crop_retry_method = detected_method
                roi_method = f"{provider_prefix}-full-frame"
            else:
                roi = parse_normalized_roi(roi_text)
                crop_retry_method = roi_method_override or "manual"
                roi_method = (
                    f"{provider_prefix}-full-frame+{roi_method_override}"
                    if roi_method_override and roi_method_override.startswith("zoom-")
                    else f"{provider_prefix}-full-frame"
                )
        elif auto_roi:
            located = (
                detect_weight_roi_consensus(frames)
                if len(frames) > 1
                else detect_weight_roi(frame)
            )
            if located is None:
                return {
                    "ok": True,
                    "qr_found": bool(decoded.get("found")),
                    "qr_code": decoded.get("qr_code"),
                    "qr_decoder": decoded.get("decoder"),
                    "qr_roi": decoded.get("qr_roi"),
                    "weight_found": False,
                    "weight": None,
                    "unit": unit,
                    "confidence": None,
                    "weight_raw": "AUTO ROI: không tìm thấy vùng LED đỏ",
                    "recognition_source": "none",
                    "local_candidate": None,
                    "gemini_used": False,
                    "gemini_suggestion": None,
                    "gemini_latency_seconds": None,
                    "gemini_input_tokens": None,
                    "gemini_output_tokens": None,
                    "gemini_thinking_tokens": None,
                    "gemini_total_tokens": None,
                    "requires_human_review": False,
                    "roi": None,
                    "roi_method": "not-found",
                    "quality": quality_payload,
                    "quality_pass": quality_pass,
                }
            roi, roi_method = located
        else:
            roi = parse_normalized_roi(roi_text)
            roi_method = roi_method_override or "manual"

        local_candidate: WeightReading | None = None
        gemini_used = self.weight_engine == "gemini"
        codex_used = False
        gemini_suggestion: float | None = None
        gemini_latency_seconds: float | None = None
        gemini_input_tokens: int | None = None
        gemini_output_tokens: int | None = None
        gemini_thinking_tokens: int | None = None
        gemini_total_tokens: int | None = None
        gemini_attempts = 0
        gemini_fallback_used = False
        ai_crop_applied = False
        requires_human_review = False
        if self.weight_engine == "gemini":
            if recognition_provider == "codex":
                if self.codex_reader is None:
                    raise ValueError("Codex chưa được bật trên máy backend")
                codex_status = self.codex_reader.status()
                if not codex_status.get("available"):
                    raise ValueError(str(codex_status.get("message") or "Codex chưa đăng nhập"))
                selected_ai_reader = self.codex_reader
                gemini_used = False
                codex_used = True
            else:
                selected_ai_reader = self._gemini_reader_for(recognition_profile)
            provider_label = "CODEX" if codex_used else "GEMINI"
            provider_source = "codex-primary" if codex_used else "gemini-primary"
            single_image_request = len(frames) == 1
            ai_frames = [frame] if single_image_request else frames
            if len(ai_frames) == 2:
                reading = WeightReading(
                    None,
                    unit,
                    False,
                    f"{provider_label} PRIMARY: cần ít nhất 3 frame camera mới",
                )
                recognition_source = "none"
            else:
                suggestion = selected_ai_reader.read(ai_frames, unit=unit)
                gemini_attempts = 1
                suggestion_raw = suggestion.raw
                suggestions = [suggestion]
                crop_is_available = bool(
                    capture_kind in {"core", "product"} and roi is not None
                )
                # Full-frame is the normal path: Gemini can locate the small scale
                # display from its factory context. Only retry a focused LED crop
                # when a valid AI response explicitly says the weight is unreadable.
                # Network/API errors are not retried here.
                if (
                    crop_is_available
                    and suggestion.value is None
                    and not suggestion.raw.startswith(f"{provider_label} ERROR:")
                ):
                    cropped_frames = [self._gemini_crop(item, roi) for item in ai_frames]
                    fallback = selected_ai_reader.read(cropped_frames, unit=unit)
                    suggestions.append(fallback)
                    gemini_attempts = 2
                    gemini_fallback_used = True
                    ai_crop_applied = True
                    suggestion_raw = (
                        f"FULL FRAME ATTEMPT: {suggestion.raw}; "
                        f"CROP RETRY: {fallback.raw}"
                    )
                    suggestion = fallback
                    roi_method = (
                        f"{provider_prefix}-full-frame+crop-"
                        f"{crop_retry_method or 'detected'}-retry"
                    )
                gemini_suggestion = suggestion.value
                gemini_latency_seconds = sum(item.latency_seconds for item in suggestions)
                gemini_input_tokens = sum(item.input_tokens for item in suggestions)
                gemini_output_tokens = sum(item.output_tokens for item in suggestions)
                gemini_thinking_tokens = sum(item.thinking_tokens for item in suggestions)
                gemini_total_tokens = sum(item.total_tokens for item in suggestions)
                local_qr = str(decoded.get("qr_code") or "").strip()
                gemini_qr = str(suggestion.qr_code or "").strip()
                qr_note = ""
                if qr_conflict:
                    qr_note = "; QR: browser/backend conflict; manual code required"
                elif local_qr and gemini_qr and local_qr != gemini_qr:
                    qr_note = "; QR: local/Gemini conflict; kept checksum-validated local QR"
                    decoded["decoder"] = (
                        f"{decoded.get('decoder', 'local')}+gemini-conflict-local-kept"
                    )
                elif local_qr:
                    if gemini_qr == local_qr:
                        decoded["decoder"] = f"{decoded.get('decoder', 'local')}+gemini"
                elif gemini_qr:
                    decoded = {
                        "ok": True,
                        "found": True,
                        "qr_code": gemini_qr,
                        "decoder": "gemini-full-frame",
                        "qr_roi": None,
                    }
                if suggestion.value is None:
                    reading = WeightReading(
                        None,
                        unit,
                        False,
                        f"{suggestion_raw}{qr_note}; {provider_label} PRIMARY: rejected",
                    )
                    recognition_source = "none"
                else:
                    reading = WeightReading(
                        suggestion.value,
                        suggestion.unit,
                        True,
                        (
                            f"{suggestion_raw}{qr_note}; {provider_label} PRIMARY: "
                            + (
                                "single focused-crop retry accepted"
                                if ai_crop_applied
                                else "single full-image accepted"
                            )
                            if single_image_request
                            else f"{suggestion_raw}{qr_note}; {provider_label} PRIMARY: 3-frame schema accepted"
                        ),
                    )
                    recognition_source = provider_source
        else:
            with self._ocr_lock:
                weight_source = CameraOCRWeightSource(
                    roi,
                    unit=unit,
                    min_confidence=self.ocr_min_confidence,
                    download_enabled=self.ocr_download,
                    reader=self._ocr_reader,
                )
                reading = (
                    weight_source.capture_many(frames)
                    if len(frames) > 1
                    else weight_source.capture(frame)
                )
                self._ocr_reader = weight_source._reader
            candidate_reader = getattr(weight_source, "candidate_reading", None)
            local_candidate = (
                candidate_reader()
                if callable(candidate_reader)
                else (reading if reading.value is not None else None)
            )
            recognition_source = "paddle-local" if reading.value is not None else "none"
        if (
            self.weight_engine == "hybrid"
            and reading.value is None
            and self.gemini_reader is not None
            and len(frames) >= 3
        ):
            gemini_used = True
            selected_gemini_reader = self._gemini_reader_for(recognition_profile)
            suggestion = selected_gemini_reader.read(
                frames,
                unit=unit,
            )
            gemini_suggestion = suggestion.value
            gemini_latency_seconds = suggestion.latency_seconds
            gemini_input_tokens = suggestion.input_tokens
            gemini_output_tokens = suggestion.output_tokens
            gemini_thinking_tokens = suggestion.thinking_tokens
            gemini_total_tokens = suggestion.total_tokens
            local_cloud_agree = (
                suggestion.value is not None
                and local_candidate is not None
                and local_candidate.value is not None
                and local_candidate.unit == suggestion.unit
                and round(local_candidate.value, 6) == round(suggestion.value, 6)
            )
            if local_cloud_agree:
                confidence = local_candidate.confidence
                reading = WeightReading(
                    suggestion.value,
                    suggestion.unit,
                    True,
                    (
                        f"{reading.raw}; {local_candidate.raw}; {suggestion.raw}; "
                        "HYBRID: local majority confirmed by Gemini"
                    ),
                    confidence,
                )
                recognition_source = "paddle-local+gemini"
            else:
                requires_human_review = suggestion.value is not None
                reading = WeightReading(
                    None,
                    unit,
                    False,
                    f"{reading.raw}; {suggestion.raw}; HYBRID: no independent agreement",
                )
        return {
            "ok": True,
            "qr_found": bool(decoded.get("found")),
            "qr_code": decoded.get("qr_code"),
            "qr_decoder": decoded.get("decoder"),
            "qr_roi": decoded.get("qr_roi"),
            "qr_conflict": qr_conflict,
            "weight_found": reading.value is not None,
            "weight": reading.value,
            "unit": reading.unit,
            "confidence": reading.confidence,
            "weight_raw": reading.raw,
            "recognition_source": recognition_source,
            "recognition_profile": recognition_profile,
            "recognition_provider": recognition_provider,
            "local_candidate": (
                local_candidate.value
                if local_candidate is not None
                else None
            ),
            "gemini_used": gemini_used,
            "codex_used": codex_used,
            "gemini_suggestion": gemini_suggestion,
            "gemini_latency_seconds": gemini_latency_seconds,
            "gemini_input_tokens": gemini_input_tokens,
            "gemini_output_tokens": gemini_output_tokens,
            "gemini_thinking_tokens": gemini_thinking_tokens,
            "gemini_total_tokens": gemini_total_tokens,
            "gemini_attempts": gemini_attempts,
            "gemini_fallback_used": gemini_fallback_used,
            "ai_latency_seconds": gemini_latency_seconds,
            "ai_attempts": gemini_attempts,
            "ai_fallback_used": gemini_fallback_used,
            "requires_human_review": requires_human_review,
            "roi": self._roi_text(roi) if roi is not None else None,
            "roi_method": roi_method,
            "gemini_crop_applied": ai_crop_applied,
            "ai_crop_applied": ai_crop_applied,
            "burst_frames": len(frames),
            "quality": quality_payload,
            "quality_pass": quality_pass,
        }

    def analyze(
        self,
        frame: np.ndarray,
        roi_text: str,
        unit: str,
        *,
        event_id: str | None = None,
        station_id: str | None = None,
        camera_id: str | None = None,
        weight_frames: list[np.ndarray] | None = None,
        require_temporal: bool = False,
        recognition_profile: str = "fast",
        recognition_provider: str = "gemini",
        capture_kind: str = "",
        client_qr_code: str = "",
        context_station_id: str | None = None,
    ) -> dict[str, object]:
        additional_frames = list(weight_frames or [])
        if len(additional_frames) >= MAX_BURST_FRAMES:
            raise ValueError(f"Burst chỉ được tối đa {MAX_BURST_FRAMES} frame kể cả ảnh chính")
        expected_aspect = frame.shape[1] / max(1, frame.shape[0])
        for extra in additional_frames:
            if extra.ndim != 3 or extra.shape[2] < 3:
                raise ValueError("Burst chứa frame không hợp lệ")
            aspect = extra.shape[1] / max(1, extra.shape[0])
            if abs(aspect - expected_aspect) > 0.03:
                raise ValueError("Các frame trong burst không cùng tỷ lệ ảnh")
        if (
            require_temporal
            and self.weight_engine != "gemini"
            and self.weight_burst_frames >= 3
            and len(additional_frames) < 2
        ):
            raise ValueError("Camera không lấy đủ tối thiểu 3 frame LED; hãy chụp lại")
        identities = (event_id, station_id, camera_id)
        configured_roi = self.weight_rois.get(context_station_id or station_id or "")
        if configured_roi is None and not any(identities) and self.station_count == 1:
            configured_roi = self.weight_rois.get(str(self.station_configs[0]["station_id"]))
        auto_roi_requested = roi_text.strip().lower() in {"", "auto"}
        roi_method_override = None
        if auto_roi_requested and configured_roi is not None:
            roi_text = self._roi_text(configured_roi)
            roi_method_override = "camera-calibrated"
        evidence_image: str | None = None
        evidence_zoom_applied = False
        evidence_zoom_method: str | None = None
        if self.weight_engine == "gemini" and capture_kind in {"core", "product"}:
            located: tuple[NormalizedROI, str] | None
            if configured_roi is not None and auto_roi_requested:
                located = configured_roi, "camera-calibrated"
            elif not auto_roi_requested:
                located = parse_normalized_roi(roi_text), "manual"
            else:
                located = detect_weight_roi(frame) or self._distant_weight_roi(frame)
            if located is not None:
                source_roi, evidence_zoom_method = located
                composite, zoom_roi = self._zoomed_evidence(frame, source_roi)
                frame, evidence_image = self._canonical_evidence(composite)
                roi_text = self._roi_text(zoom_roi)
                roi_method_override = f"zoom-{evidence_zoom_method}"
                evidence_zoom_applied = True
        evidence_metadata = {
            "evidence_image": evidence_image,
            "evidence_zoom_applied": evidence_zoom_applied,
            "evidence_zoom_method": evidence_zoom_method,
        }
        if not any(identities):
            result = self.inference.run(
                self._analyze_frame,
                frame,
                roi_text,
                unit,
                additional_frames,
                roi_method_override,
                recognition_profile,
                recognition_provider,
                capture_kind,
                client_qr_code,
            )
            return {**result, **evidence_metadata}
        if not all(identities):
            raise ValueError("Cần đủ event_id, station_id và camera_id")
        assert event_id is not None and station_id is not None and camera_id is not None
        binding = self.sessions.stage(
            frame,
            event_id=event_id,
            station_id=station_id,
            camera_id=camera_id,
        )
        try:
            self._write_diagnostic(station_id, frame)
            result = self.inference.run(
                self._analyze_frame,
                frame,
                roi_text,
                unit,
                additional_frames,
                roi_method_override,
                recognition_profile,
                recognition_provider,
                capture_kind,
                client_qr_code,
            )
            binding = self.sessions.mark_ready(binding.analysis_id)
        except Exception as exc:
            try:
                self.sessions.mark_failed(binding.analysis_id, exc)
            except SessionConflictError:
                pass
            raise
        return {**result, **binding.as_dict(), **evidence_metadata}

    def capture(
        self,
        qr_code: str,
        weight: float,
        unit: str,
        frame: np.ndarray,
        vision_confirmed: bool = False,
        weight_raw: str = "",
        *,
        product_frame: np.ndarray | None = None,
        product_weight: float | None = None,
        event_id: str | None = None,
        analysis_id: str | None = None,
        station_id: str | None = None,
        camera_id: str | None = None,
        frame_sha256: str | None = None,
    ) -> dict[str, object]:
        qr_code = qr_code.strip()
        if not qr_code:
            decoded = self.decode_qr(frame)
            if not decoded.get("found"):
                raise ValueError("Chưa có QR; hãy quét mã hoặc đưa QR vào camera")
            qr_code = str(decoded["qr_code"])
            qr_source = f"camera:{decoded['decoder']}"
        else:
            qr_source = "test-ui:input"
        if len(qr_code) > 512:
            raise ValueError("QR dài quá 512 ký tự")
        if unit not in UNITS:
            raise ValueError("Đơn vị không hợp lệ")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Số cân phải là số không âm")
        quality = self.assess_quality(frame)
        quality_payload, quality_pass = self.quality_result(quality)
        if not quality_pass:
            raise ValueError(
                "Ảnh chưa đạt chất lượng: "
                + "; ".join(str(issue) for issue in quality_payload["issues"])
            )

        identity_values = (event_id, analysis_id, station_id, camera_id)
        bound_capture = any(identity_values) or bool(frame_sha256)
        if bound_capture and not all(identity_values):
            raise ValueError("Cần đủ event_id, analysis_id, station_id và camera_id")
        computed_frame_sha = jpeg_sha256(encode_staged_jpeg(frame))
        if frame_sha256 and frame_sha256.lower() != computed_frame_sha:
            raise AnalysisBindingMismatch("frame_sha256 không khớp ảnh gửi để lưu")

        captured_at: str | None = None
        if bound_capture:
            assert event_id and analysis_id and station_id and camera_id
            try:
                binding = self.sessions.validate(
                    analysis_id,
                    event_id=event_id,
                    station_id=station_id,
                    camera_id=camera_id,
                    frame_sha256=computed_frame_sha,
                )
                captured_at = binding.captured_at
            except AnalysisBindingNotFound:
                # Completed bindings are retained for a TTL, but a valid retry
                # must remain idempotent even after that TTL or a process restart.
                existing = self.store.get(event_id)
                if existing is None:
                    raise
                expected = (
                    getattr(existing, "analysis_id", ""),
                    getattr(existing, "station_id", ""),
                    getattr(existing, "camera_id", ""),
                    getattr(existing, "frame_sha256", ""),
                )
                supplied = (analysis_id, station_id, camera_id, computed_frame_sha)
                if supplied != expected:
                    raise AnalysisBindingMismatch(
                        "Danh tính lần lưu lại không khớp bản ghi cục bộ"
                    )
                captured_at = existing.captured_at

        with self._lock:
            capture_key = frame_fingerprint(frame)
            if not bound_capture:
                now = time.monotonic()
                if now - self._recent.get(capture_key, float("-inf")) < self.duplicate_window:
                    raise ValueError("Ảnh này vừa được lưu; hãy chụp khung hình mới")
            measurement, duplicate = self._save_idempotent(
                qr_code=qr_code,
                weight=weight,
                unit=unit,
                frame=frame,
                weight_source=(
                    "camera-gemini:test-ui"
                    if vision_confirmed and "GEMINI" in weight_raw
                    else "camera-ocr:test-ui"
                    if vision_confirmed
                    else "manual-test-ui"
                ),
                needs_sync=self.sync_worker is not None,
                qr_source=qr_source,
                weight_raw=_persistable_weight_raw(
                    weight_raw,
                    weight=weight,
                    vision_confirmed=vision_confirmed,
                ),
                weight_stable=True,
                event_id=event_id,
                captured_at=captured_at,
                gateway_id=self.gateway_id if bound_capture else "",
                station_id=station_id or "",
                camera_id=camera_id or "",
                analysis_id=analysis_id or "",
            )
            if not bound_capture:
                self._recent[capture_key] = time.monotonic()

        if bound_capture:
            self.sessions.mark_saved(str(analysis_id))
        if product_weight is not None:
            self.store.attach_product_weight(measurement.event_id, product_weight)
        if product_frame is not None:
            self.store.attach_product_image(measurement.event_id, product_frame)
            measurement = self.store.get(measurement.event_id) or measurement
        if bound_capture and station_id:
            self._cleanup_evidence_steps(station_id, measurement.event_id)
        # Commit locally first, then synchronously confirm this exact event so
        # one operator click sends the product code, core weight and evidence
        # image together. A failed cloud attempt remains durable in the outbox.
        if self.sync_worker is not None:
            cloud_confirmed = self.sync_worker.sync_event(measurement.event_id)
            if (
                not cloud_confirmed
                and product_weight is not None
                and measurement.product_image_path
            ):
                supabase_url = os.environ.get("ROLL_SCALE_SUPABASE_URL", "").strip()
                service_key = os.environ.get("ROLL_SCALE_SUPABASE_SERVICE_KEY", "").strip()
                if supabase_url and service_key:
                    try:
                        remote = persist_product_evidence(
                            supabase_url,
                            service_key,
                            event_id=measurement.event_id,
                            gateway_id=self.gateway_id,
                            image_path=measurement.product_image_path,
                            product_weight=product_weight,
                        )
                        core_url = remote.get("core_image_url") or remote.get("image_url")
                        core_public_id = remote.get("core_image_public_id") or remote.get("image_public_id")
                        self.store.mark_synced(
                            measurement.event_id,
                            int(remote["id"]) if remote.get("id") is not None else None,
                            str(core_url) if core_url else None,
                            str(core_public_id) if core_public_id else None,
                        )
                        cloud_confirmed = True
                    except Exception:
                        cloud_confirmed = False
        saved = self.store.get(measurement.event_id)
        current = saved or measurement
        return {
            "ok": True,
            "id": current.id,
            "event_id": current.event_id,
            "analysis_id": getattr(current, "analysis_id", analysis_id or ""),
            "gateway_id": getattr(current, "gateway_id", self.gateway_id if bound_capture else ""),
            "station_id": getattr(current, "station_id", station_id or ""),
            "camera_id": getattr(current, "camera_id", camera_id or ""),
            "frame_sha256": getattr(current, "frame_sha256", computed_frame_sha),
            "duplicate": duplicate,
            "qr_code": qr_code,
            "weight": weight,
            "unit": unit,
            "sync_status": current.sync_status,
            "remote_id": current.remote_id,
            "remote_image_url": current.remote_image_url,
            "remote_image_public_id": getattr(current, "remote_image_public_id", None),
            "sync_error": current.sync_error,
            "pending_count": self.store.pending_count(),
        }

    def _save_idempotent(self, **kwargs: object) -> tuple[object, bool]:
        """Isolate old/new MeasurementStore return and keyword contracts."""

        method = getattr(self.store, "save_idempotent", None)
        idempotent = callable(method)
        if not idempotent:
            method = self.store.save
        assert callable(method)
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            call_kwargs = kwargs
        else:
            accepts_arbitrary = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            call_kwargs = kwargs if accepts_arbitrary else {
                key: value for key, value in kwargs.items() if key in signature.parameters
            }
        result = method(**call_kwargs)
        if hasattr(result, "measurement") and hasattr(result, "duplicate"):
            return result.measurement, bool(result.duplicate)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], bool(result[1])
        return result, False

    def save_factory_sample(
        self,
        frame: np.ndarray,
        qr_roi: str,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        if self.sample_store is None:
            raise RuntimeError("Chưa cấu hình thư mục thu ảnh xưởng")
        metadata = dict(metadata)
        metadata["quality"] = self.assess_quality(frame).as_dict()
        return self.sample_store.save(frame, metadata, qr_roi)

    def lookup(self, qr_code: str) -> dict[str, object]:
        qr_code = qr_code.strip()
        if not qr_code:
            raise ValueError("Thiếu mã QR")
        if not self.lookup_url or not self.lookup_token:
            raise RuntimeError("Chưa cấu hình Supabase lookup")
        return lookup_roll(self.lookup_url, qr_code, self.lookup_token)


def _frontend_index_path() -> Path:
    configured = os.environ.get("ROLL_SCALE_FRONTEND_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve() / "index.html"
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "assets" / "frontend" / "index.html"
    return Path(__file__).resolve().parents[3] / "frontend" / "index.html"


def load_frontend_html() -> str:
    index_path = _frontend_index_path()
    try:
        return index_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Không đọc được frontend: {index_path}") from exc


TEST_UI_HTML = load_frontend_html()


def _default_weight_engine() -> str:
    configured = os.environ.get("ROLL_SCALE_WEIGHT_ENGINE", "").strip().lower()
    if configured:
        return configured
    return (
        "gemini"
        if os.environ.get("ROLL_SCALE_GEMINI_API_KEY", "").strip()
        else "local"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local web UI for testing QR + scale + Supabase")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--db", default="data/measurements.db")
    parser.add_argument("--captures", default="data/captures")
    parser.add_argument("--demo-image", default="data/warehouse_scale_demo.png")
    parser.add_argument("--logo-image", default="data/viet_nhat_ipt_logo.jpg")
    parser.add_argument("--duplicate-window", type=float, default=5.0)
    default_yolo_model = os.environ.get("ROLL_SCALE_YOLO_MODEL")
    if (
        not default_yolo_model
        and not getattr(sys, "frozen", False)
        and Path("models/qr_demo_synthetic.pt").is_file()
    ):
        default_yolo_model = "models/qr_demo_synthetic.pt"
    parser.add_argument("--yolo-model", default=default_yolo_model, help="Custom QR detector best.pt")
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=960)
    parser.add_argument("--yolo-mode", choices=("first", "fallback"), default="fallback")
    parser.add_argument("--ocr-min-confidence", type=float, default=0.60)
    parser.add_argument("--ocr-download", action="store_true")
    parser.add_argument(
        "--weight-engine",
        choices=tuple(sorted(WEIGHT_ENGINES)),
        default=_default_weight_engine(),
        help="Bộ đọc cân: local, hybrid hoặc gemini (Gemini primary)",
    )
    parser.add_argument(
        "--gemini-fallback",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("ROLL_SCALE_GEMINI_ENABLED", False),
        help="Gọi Gemini bằng một ảnh toàn khung để đọc QR và cân",
    )
    parser.add_argument(
        "--gemini-model",
        default=os.environ.get("ROLL_SCALE_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    )
    parser.add_argument(
        "--gemini-accurate-model",
        default=os.environ.get(
            "ROLL_SCALE_GEMINI_ACCURATE_MODEL",
            DEFAULT_GEMINI_ACCURATE_MODEL,
        ),
    )
    parser.add_argument(
        "--gemini-timeout",
        type=float,
        default=float(
            os.environ.get(
                "ROLL_SCALE_GEMINI_TIMEOUT",
                str(DEFAULT_GEMINI_TIMEOUT_SECONDS),
            )
        ),
    )
    parser.add_argument(
        "--gemini-accurate-timeout",
        type=float,
        default=float(
            os.environ.get(
                "ROLL_SCALE_GEMINI_ACCURATE_TIMEOUT",
                str(DEFAULT_GEMINI_ACCURATE_TIMEOUT_SECONDS),
            )
        ),
    )
    parser.add_argument(
        "--codex-enabled",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("ROLL_SCALE_CODEX_ENABLED", True),
        help="Cho phép chọn Codex/ChatGPT mà không thay thế Gemini API",
    )
    parser.add_argument(
        "--codex-mode",
        choices=("auto", "cli", "oauth"),
        default=os.environ.get("ROLL_SCALE_CODEX_MODE", "auto").strip().lower(),
        help="auto dùng OAuth trên Render và Codex CLI ở máy local",
    )
    parser.add_argument(
        "--codex-command",
        default=os.environ.get("ROLL_SCALE_CODEX_COMMAND", "codex"),
    )
    parser.add_argument(
        "--codex-model",
        default=os.environ.get("ROLL_SCALE_CODEX_MODEL", ""),
        help="Để trống để Codex dùng model mặc định của tài khoản ChatGPT",
    )
    parser.add_argument(
        "--codex-timeout",
        type=float,
        default=float(
            os.environ.get(
                "ROLL_SCALE_CODEX_TIMEOUT",
                str(DEFAULT_CODEX_TIMEOUT_SECONDS),
            )
        ),
    )
    parser.add_argument(
        "--diagnostic-image",
        default=os.environ.get("ROLL_SCALE_DIAGNOSTIC_IMAGE"),
        help="Ghi đè ảnh phân tích gần nhất để hiệu chỉnh OCR tại xưởng",
    )
    parser.add_argument("--factory-samples", default="dataset/factory_raw")
    parser.add_argument("--staging-dir", help="Thư mục JPEG tạm đã khóa theo analysis_id")
    parser.add_argument("--inference-queue-size", type=int, default=8)
    parser.add_argument("--station-count", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--station-id",
        action="append",
        dest="station_ids",
        help="ID trạm theo thứ tự; lặp lại 1-3 lần (mặc định station-01...)",
    )
    parser.add_argument(
        "--camera-id",
        action="append",
        dest="camera_ids",
        help="ID camera cấu hình theo trạm; lặp lại 1-3 lần (mặc định camera-01...)",
    )
    parser.add_argument(
        "--weight-roi",
        action="append",
        dest="weight_rois",
        help=(
            "ROI hàng gross x1,y1,x2,y2 theo từng trạm; lặp lại đúng station-count lần. "
            "Bỏ qua để tự dò LED."
        ),
    )
    parser.add_argument(
        "--weight-burst-frames",
        type=int,
        choices=range(1, MAX_BURST_FRAMES + 1),
        default=DEFAULT_WEIGHT_BURST_FRAMES,
        help="Tổng số frame cho local/hybrid; chế độ Gemini luôn chụp đúng 1 ảnh",
    )
    parser.add_argument(
        "--auto-advance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sau khi lưu SQLite thành công, chọn trạm kế tiếp",
    )
    parser.add_argument("--min-frame-width", type=int, default=640)
    parser.add_argument("--min-frame-height", type=int, default=480)
    parser.add_argument("--min-brightness", type=float, default=30.0)
    parser.add_argument("--max-brightness", type=float, default=250.0)
    parser.add_argument("--min-sharpness", type=float, default=35.0)
    parser.add_argument("--api-url", default=os.environ.get("ROLL_SCALE_API_URL"))
    parser.add_argument("--api-token", default=os.environ.get("ROLL_SCALE_DEVICE_TOKEN"))
    parser.add_argument("--lookup-url", default=os.environ.get("ROLL_SCALE_LOOKUP_URL"))
    parser.add_argument("--lookup-token", default=os.environ.get("ROLL_SCALE_LOOKUP_TOKEN"))
    parser.add_argument(
        "--gateway-id",
        "--device-id",
        dest="gateway_id",
        default=os.environ.get("ROLL_SCALE_GATEWAY_ID")
        or os.environ.get("ROLL_SCALE_DEVICE_ID", "gateway-01"),
        help="ID gateway; --device-id được giữ làm bí danh tương thích",
    )
    return parser


def create_server(args: argparse.Namespace) -> tuple[ThreadingHTTPServer, StationUIService]:
    if bool(args.api_url) != bool(args.api_token):
        raise ValueError("Cần đủ URL và token của API ghi")
    if bool(args.lookup_url) != bool(args.lookup_token):
        raise ValueError("Cần đủ URL và token của API tra cứu")
    web_username = os.environ.get("ROLL_SCALE_WEB_USERNAME", "").strip()
    web_password = os.environ.get("ROLL_SCALE_WEB_PASSWORD", "")
    if bool(web_username) != bool(web_password):
        raise ValueError("Cần đủ ROLL_SCALE_WEB_USERNAME và ROLL_SCALE_WEB_PASSWORD")
    station_ids = getattr(args, "station_ids", None)
    camera_ids = getattr(args, "camera_ids", None)
    weight_rois = getattr(args, "weight_rois", None)
    if station_ids and len(station_ids) != args.station_count:
        raise ValueError("Số --station-id phải bằng --station-count")
    if camera_ids and len(camera_ids) != args.station_count:
        raise ValueError("Số --camera-id phải bằng --station-count")
    if camera_ids and len(set(camera_ids)) != len(camera_ids):
        raise ValueError("camera_id values must be unique")
    if weight_rois and len(weight_rois) != args.station_count:
        raise ValueError("Số --weight-roi phải bằng --station-count")
    weight_engine = args.weight_engine
    if args.gemini_fallback and weight_engine == "local":
        weight_engine = "hybrid"
    gemini_reader = None
    gemini_accurate_reader = None
    gemini_key_manager = None
    if weight_engine in {"hybrid", "gemini"}:
        gemini_key_store = EncryptedCodexTokenStore(
            args.api_url,
            args.api_token,
            secret_name=f"gemini-api-key:{args.gateway_id}",
            # Keep the legacy action name so already-deployed Supabase Functions
            # can store any encrypted secret by its distinct name. New Functions
            # accept this action too.
            secret_action="codex-auth",
            encryption_key=os.environ.get("ROLL_SCALE_GEMINI_KEY_ENCRYPTION_KEY", "")
            or os.environ.get("ROLL_SCALE_CODEX_TOKEN_KEY", ""),
        )
        gemini_key_manager = GeminiKeyManager(
            gemini_key_store,
            fast_model=args.gemini_model,
            accurate_model=args.gemini_accurate_model,
            fast_timeout=args.gemini_timeout,
            accurate_timeout=args.gemini_accurate_timeout,
            initial_key=os.environ.get("ROLL_SCALE_GEMINI_API_KEY", ""),
        )
        gemini_api_key = gemini_key_manager.load_key()
        if not gemini_api_key:
            raise ValueError(
                f"weight_engine={weight_engine} cần Gemini API key trong biến môi trường "
                "hoặc kho key mã hóa Supabase"
            )
        gemini_reader = GeminiWeightReader(
            gemini_api_key,
            model=args.gemini_model,
            timeout_seconds=args.gemini_timeout,
            thinking_level="minimal",
            max_image_edge=1600,
            jpeg_quality=90,
            media_resolution="high",
            include_qr=False,
        )
        gemini_accurate_reader = GeminiWeightReader(
            gemini_api_key,
            model=args.gemini_accurate_model,
            timeout_seconds=args.gemini_accurate_timeout,
            thinking_level="medium",
            max_image_edge=1600,
            jpeg_quality=90,
            media_resolution="high",
            include_qr=False,
        )
    codex_reader: CodexWeightReader | CodexOAuthWeightReader | None = None
    if args.codex_enabled:
        codex_mode = args.codex_mode
        if codex_mode == "auto":
            codex_mode = "oauth" if os.environ.get("RENDER", "").strip() else "cli"
        if codex_mode == "oauth":
            token_store = EncryptedCodexTokenStore(
                args.api_url,
                args.api_token,
                secret_name=f"codex-oauth:{args.gateway_id}",
                encryption_key=os.environ.get("ROLL_SCALE_CODEX_TOKEN_KEY", ""),
            )
            codex_reader = CodexOAuthWeightReader(
                CodexOAuthClient(token_store, timeout_seconds=min(args.codex_timeout, 30.0)),
                model=args.codex_model,
                timeout_seconds=args.codex_timeout,
                client_version=os.environ.get("ROLL_SCALE_CODEX_CLIENT_VERSION", "0.124.0"),
            )
        else:
            codex_reader = CodexWeightReader(
                args.codex_command,
                model=args.codex_model,
                timeout_seconds=args.codex_timeout,
            )
    store = MeasurementStore(args.db, args.captures)
    worker = None
    if args.api_url:
        worker = OutboxSyncWorker(store, args.api_url, args.api_token, args.gateway_id)
        worker.start()
    service = StationUIService(
        store,
        worker,
        args.lookup_url,
        args.lookup_token,
        args.duplicate_window,
        QRReader(args.yolo_model, args.yolo_confidence, args.yolo_mode, args.yolo_imgsz),
        args.ocr_min_confidence,
        args.ocr_download,
        FactorySampleStore(args.factory_samples),
        args.min_frame_width,
        args.min_frame_height,
        args.min_brightness,
        args.max_brightness,
        args.min_sharpness,
        gateway_id=args.gateway_id,
        station_count=args.station_count,
        station_ids=station_ids,
        camera_ids=camera_ids,
        staging_dir=args.staging_dir,
        diagnostic_image=args.diagnostic_image,
        inference_queue_size=args.inference_queue_size,
        auto_advance=args.auto_advance,
        weight_rois=weight_rois,
        weight_burst_frames=args.weight_burst_frames,
        gemini_reader=gemini_reader,
        gemini_accurate_reader=gemini_accurate_reader,
        gemini_key_manager=gemini_key_manager,
        codex_reader=codex_reader,
        weight_engine=weight_engine,
    )
    demo_path = Path(args.demo_image)
    logo_path = Path(args.logo_image)

    class Handler(BaseHTTPRequestHandler):
        def is_authorized(self) -> bool:
            if not web_username:
                return True
            authorization = self.headers.get("authorization", "")
            if not authorization.startswith("Basic "):
                return False
            try:
                decoded = base64.b64decode(
                    authorization.removeprefix("Basic "), validate=True
                ).decode("utf-8")
                username, password = decoded.split(":", 1)
            except (binascii.Error, UnicodeDecodeError, ValueError):
                return False
            return hmac.compare_digest(username, web_username) and hmac.compare_digest(
                password, web_password
            )

        def require_authorization(self) -> bool:
            if self.is_authorized():
                return True
            body = json.dumps(
                {"ok": False, "error": "authentication_required"}
            ).encode("utf-8")
            self.send_response(401)
            self.send_header("www-authenticate", 'Basic realm="Tram Can QR Pilot"')
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.send_header("x-content-type-options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
            return False

        def send_bytes(self, status_code: int, content_type: str, body: bytes) -> None:
            self.send_response(status_code)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.send_header("x-content-type-options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status_code: int, payload: dict[str, object]) -> None:
            self.send_bytes(
                status_code,
                "application/json; charset=utf-8",
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            )

        def read_json(self) -> dict[str, object]:
            length = int(self.headers.get("content-length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request rỗng hoặc quá lớn")
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("JSON không hợp lệ") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON phải là object")
            return payload

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/health":
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "release": os.environ.get("RENDER_GIT_COMMIT", "local")[:12],
                    },
                )
                return
            if not self.require_authorization():
                return
            if parsed.path == "/":
                self.send_bytes(200, "text/html; charset=utf-8", TEST_UI_HTML.encode("utf-8"))
                return
            if parsed.path == "/demo.jpg":
                if not demo_path.is_file():
                    self.send_json(404, {"ok": False, "error": "demo_image_missing"})
                    return
                content_type = "image/png" if demo_path.suffix.lower() == ".png" else "image/jpeg"
                self.send_bytes(200, content_type, demo_path.read_bytes())
                return
            if parsed.path == "/logo.jpg":
                if not logo_path.is_file():
                    self.send_json(404, {"ok": False, "error": "logo_image_missing"})
                    return
                content_type = "image/png" if logo_path.suffix.lower() == ".png" else "image/jpeg"
                self.send_bytes(200, content_type, logo_path.read_bytes())
                return
            if parsed.path == "/api/measurement-image":
                query = urllib.parse.parse_qs(parsed.query)
                event_id = query.get("event_id", [""])[0]
                kind = query.get("kind", [""])[0]
                measurement = store.get(event_id)
                if measurement is None or kind not in {"core", "product"}:
                    self.send_json(404, {"ok": False, "error": "image_not_found"})
                    return
                image_path = Path(
                    measurement.image_path
                    if kind == "core"
                    else measurement.product_image_path
                )
                if not image_path.is_file():
                    self.send_json(404, {"ok": False, "error": "image_not_found"})
                    return
                self.send_bytes(200, "image/jpeg", image_path.read_bytes())
                return
            if parsed.path == "/api/status":
                identity_status = service.status()
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "pending_count": store.pending_count(),
                        "yolo_enabled": service.reader.model is not None,
                         "yolo_mode": service.reader.yolo_mode,
                        "ocr_enabled": service.weight_engine != "gemini",
                        "ocr_engine": (
                            PADDLE_OCR_MODEL_NAME
                            if service.weight_engine != "gemini"
                            else None
                        ),
                        "ocr_min_confidence": service.ocr_min_confidence,
                        "sync_enabled": service.sync_worker is not None,
                        "image_provider": "cloudinary" if service.sync_worker is not None else "local",
                        "release": os.environ.get("RENDER_GIT_COMMIT", "local")[:12],
                        "quality_settings": service.quality_settings,
                        **identity_status,
                    },
                )
                return
            if parsed.path == "/api/production-orders":
                query = urllib.parse.parse_qs(parsed.query)
                work_date = str(query.get("work_date", [""])[0]).strip()
                try:
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", work_date):
                        raise ValueError
                    time.strptime(work_date, "%Y-%m-%d")
                except ValueError:
                    self.send_json(
                        422,
                        {
                            "ok": False,
                            "error": "invalid_input",
                            "message": "Ngày chọn LSX không hợp lệ",
                        },
                    )
                    return
                rows = [
                    {
                        "captured_at": item.captured_at,
                        "weight_raw": item.weight_raw,
                    }
                    for item in store.recent(200)
                ]
                source = "local"
                fallback_error = ""
                supabase_url = _supabase_project_url()
                publishable_key = _supabase_read_key()
                if supabase_url and publishable_key:
                    try:
                        rows = [
                            *fetch_supabase_table(
                                supabase_url,
                                publishable_key,
                                limit=200,
                            ),
                            *rows,
                        ]
                        source = "can_tu_dong"
                    except Exception as exc:
                        fallback_error = str(exc)
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "source": source,
                        "work_date": work_date,
                        "orders": _production_orders_for_date(rows, work_date),
                        "fallback_error": fallback_error or None,
                    },
                )
                return
            if parsed.path == "/api/measurements":
                query = urllib.parse.parse_qs(parsed.query)
                try:
                    limit = int(query.get("limit", ["50"])[0])
                except ValueError:
                    limit = 50
                work_date = str(query.get("work_date", [""])[0]).strip()
                shift = str(query.get("shift", [""])[0]).strip()
                machine = str(query.get("machine", [""])[0]).strip()
                production_order = str(query.get("production_order", [""])[0]).strip()
                supabase_url = _supabase_project_url()
                publishable_key = _supabase_read_key()
                if not supabase_url or not publishable_key:
                    self.send_json(
                        200,
                        {
                            "ok": True,
                            "source": "local",
                            "work_date": work_date,
                            "shift": shift,
                            "machine": machine,
                            "production_order": production_order,
                            "items": _local_measurement_items(
                                store,
                                limit,
                                work_date=work_date,
                                shift=shift,
                                machine=machine,
                                production_order=production_order,
                            ),
                        },
                    )
                    return
                try:
                    remote_items = fetch_supabase_table(
                        supabase_url,
                        publishable_key,
                        limit=max(limit, 200),
                    )
                except Exception as exc:
                    # Keep the operator UI usable when REST keys are missing/wrong.
                    self.send_json(
                        200,
                        {
                            "ok": True,
                            "source": "local",
                            "fallback_error": str(exc),
                            "work_date": work_date,
                            "shift": shift,
                            "machine": machine,
                            "production_order": production_order,
                            "items": _local_measurement_items(
                                store,
                                limit,
                                work_date=work_date,
                                shift=shift,
                                machine=machine,
                                production_order=production_order,
                            ),
                        },
                    )
                    return
                items = []
                for item in remote_items:
                    core_url = item.get("core_image_url") or item.get("image_url")
                    product_url = item.get("product_image_url")
                    product_path = item.get("product_image_path")
                    if not product_url and isinstance(product_path, str) and product_path:
                        try:
                            product_url = sign_storage_image(
                                supabase_url,
                                publishable_key,
                                product_path,
                            )
                        except Exception:
                            pass
                    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                    raw_weight = str(metadata.get("weight_raw", ""))
                    product_weight_value = item.get("product_weight")
                    if product_weight_value is None:
                        match = re.search(
                            r"(?:^|; )PRODUCT_WEIGHT=([0-9]+(?:\.[0-9]+)?)",
                            raw_weight,
                        )
                        product_weight_value = float(match.group(1)) if match else item.get("weight")
                    core_weight_value = metadata.get("core_weight")
                    if core_weight_value is None:
                        tare_value = item.get("tare_weight")
                        core_weight_value = tare_value if tare_value not in (None, 0, 0.0) else item.get("weight")
                    payload = {
                        "event_id": item.get("event_id", ""),
                        "qr_code": item.get("qr_code", ""),
                        "core_weight": core_weight_value,
                        "product_weight": product_weight_value,
                        "tare_weight": item.get("tare_weight"),
                        "net_weight": item.get("net_weight"),
                        "weight_raw": (
                            raw_weight
                            if raw_weight
                            else (
                                f"PRODUCT_WEIGHT={product_weight_value}"
                                if product_weight_value is not None
                                else ""
                            )
                        ),
                        "unit": item.get("unit", "kg"),
                        "captured_at": item.get("captured_at", ""),
                        "work_date": _item_source_date(item),
                        "shift": _item_source_value(item, "shift", "SOURCE_SHIFT"),
                        "machine": _item_source_value(item, "machine", "SOURCE_MACHINE"),
                        "production_order": _item_source_value(
                            item,
                            "production_order",
                            "SOURCE_PRODUCTION_ORDER",
                        ),
                        "sync_status": "synced",
                        "sync_error": None,
                        "core_image_url": core_url,
                        "product_image_url": product_url,
                        "has_core_image": bool(core_url),
                        "has_product_image": bool(product_url),
                    }
                    if _matches_source_filters(
                        payload,
                        work_date=work_date,
                        shift=shift,
                        machine=machine,
                        production_order=production_order,
                    ):
                        items.append(payload)
                    if len(items) >= limit:
                        break
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "source": "can_tu_dong",
                        "work_date": work_date,
                        "shift": shift,
                        "machine": machine,
                        "production_order": production_order,
                        "items": items,
                    },
                )
                return
            if parsed.path == "/api/lookup":
                qr_code = urllib.parse.parse_qs(parsed.query).get("qr", [""])[0]
                try:
                    result = service.lookup(qr_code)
                    self.send_json(200 if result.get("found") else 404, result)
                except ValueError as exc:
                    self.send_json(422, {"ok": False, "error": "invalid_input", "message": str(exc)})
                except Exception as exc:
                    self.send_json(502, {"ok": False, "error": "lookup_failed", "message": str(exc)})
                return
            if parsed.path == "/api/panel/regions":
                station_id = urllib.parse.parse_qs(parsed.query).get("station_id", [""])[0]
                try:
                    self.send_json(
                        200,
                        {"ok": True, "regions": service.panel_regions(station_id)},
                    )
                except ValueError as exc:
                    self.send_json(
                        422,
                        {"ok": False, "error": "invalid_input", "message": str(exc)},
                    )
                return
            self.send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            if not self.require_authorization():
                return
            try:
                payload = self.read_json()
                if self.path == "/api/session/discard":
                    discarded = service.discard_session(
                        str(payload.get("station_id", "")),
                        event_id=str(payload["event_id"]) if payload.get("event_id") else None,
                    )
                    self.send_json(200, {"ok": True, "discarded": discarded})
                    return
                if self.path == "/api/codex/login":
                    if service.codex_reader is None:
                        raise ValueError("Codex chưa được bật trên máy backend")
                    self.send_json(200, service.codex_reader.start_device_login())
                    return
                if self.path == "/api/codex/login/poll":
                    if service.codex_reader is None:
                        raise ValueError("Codex chưa được bật trên máy backend")
                    poll_login = getattr(service.codex_reader, "poll_device_login", None)
                    if poll_login is None:
                        raise ValueError("Chế độ Codex local không dùng đăng nhập web")
                    self.send_json(200, poll_login(str(payload.get("session_id", ""))))
                    return
                if self.path == "/api/gemini/key":
                    self.send_json(200, service.replace_gemini_key(str(payload.get("api_key", ""))))
                    return
                if self.path == "/api/panel/regions":
                    self.send_json(
                        200,
                        {
                            "ok": True,
                            "regions": service.save_panel_regions(
                                str(payload.get("station_id", "")),
                                payload.get("regions"),
                            ),
                        },
                    )
                    return
                frame = decode_image(str(payload.get("image", "")))
                if self.path == "/api/panel/analyze":
                    self.send_json(
                        200,
                        service.analyze_panel_regions(
                            frame,
                            payload.get("regions"),
                            recognition_profile=str(payload.get("recognition_profile", "fast")),
                        ),
                    )
                    return
                product_frame = (
                    decode_image(str(payload.get("product_image", "")))
                    if payload.get("product_image")
                    else None
                )
                if self.path == "/api/decode":
                    self.send_json(200, service.decode_qr(frame))
                    return
                if self.path == "/api/analyze":
                    capture_kind = str(payload.get("capture_kind", "")).strip().lower()
                    if capture_kind and capture_kind not in {"core", "product"}:
                        raise ValueError("capture_kind phải là core hoặc product")
                    encoded_weight_frames = payload.get("weight_frames", [])
                    if not isinstance(encoded_weight_frames, list):
                        raise ValueError("weight_frames phải là danh sách")
                    if len(encoded_weight_frames) >= MAX_BURST_FRAMES:
                        raise ValueError(
                            f"Burst chỉ được tối đa {MAX_BURST_FRAMES} frame kể cả ảnh chính"
                        )
                    weight_frames = []
                    for encoded_frame in encoded_weight_frames:
                        if not isinstance(encoded_frame, str):
                            raise ValueError("Burst chứa ảnh không hợp lệ")
                        weight_frames.append(decode_image(encoded_frame))
                    bind_core = capture_kind != "product"
                    result = service.analyze(
                        frame,
                        str(payload.get("roi", "")),
                        str(payload.get("unit", "kg")),
                        event_id=str(payload["event_id"]) if bind_core and payload.get("event_id") else None,
                        station_id=str(payload["station_id"]) if bind_core and payload.get("station_id") else None,
                        camera_id=str(payload["camera_id"]) if bind_core and payload.get("camera_id") else None,
                        weight_frames=weight_frames,
                        require_temporal=bool(payload.get("camera_capture", False)),
                        recognition_profile=str(payload.get("recognition_profile", "fast")),
                        recognition_provider=str(payload.get("recognition_provider", "gemini")),
                        capture_kind=capture_kind,
                        client_qr_code=str(payload.get("client_qr_code", "")),
                        context_station_id=str(payload.get("station_id", "")) or None,
                    )
                    if (
                        capture_kind
                        and result.get("weight_found")
                        and result.get("quality_pass")
                    ):
                        evidence_frame = (
                            decode_image(str(result["evidence_image"]))
                            if result.get("evidence_image")
                            else frame
                        )
                        staged = service.stage_evidence_step(
                            evidence_frame,
                            event_id=str(payload.get("event_id", "")),
                            station_id=str(payload.get("station_id", "")),
                            kind=capture_kind,
                            weight=float(result["weight"]),
                            unit=str(result.get("unit", payload.get("unit", "kg"))),
                            qr_code=str(result.get("qr_code", "")),
                        )
                        result["step_saved"] = True
                        result["step_image_path"] = staged["image_path"]
                    self.send_json(200, result)
                    return
                if self.path == "/api/factory-sample":
                    metadata = {
                        "predicted_qr_code": str(payload.get("predicted_qr_code", ""))[:512],
                        "predicted_weight": payload.get("predicted_weight"),
                        "expected_qr_code": str(payload.get("expected_qr_code", ""))[:512],
                        "expected_weight": payload.get("expected_weight"),
                        "unit": str(payload.get("unit", "kg")),
                        "recognition_ok": bool(payload.get("recognition_ok", False)),
                        "qr_decoder": str(payload.get("qr_decoder", ""))[:100],
                        "ocr_confidence": payload.get("ocr_confidence"),
                    }
                    result = service.save_factory_sample(
                        frame,
                        str(payload.get("qr_roi", "")),
                        metadata,
                    )
                    self.send_json(201, result)
                    return
                if self.path == "/api/capture":
                    try:
                        weight = float(payload.get("weight", ""))
                    except (TypeError, ValueError) as exc:
                        raise ValueError("Số cân không hợp lệ") from exc
                    weight_raw = _merge_source_tags(str(payload.get("weight_raw", "")), payload)
                    if not _raw_tag(weight_raw, "SOURCE_PRODUCTION_ORDER"):
                        raise ValueError("Thiếu Lệnh sản xuất")
                    product_weight_value = payload.get("product_weight")
                    if product_weight_value is None:
                        product_match = re.search(
                            r"(?:^|; )PRODUCT_WEIGHT=([0-9]+(?:\.[0-9]+)?)",
                            weight_raw,
                        )
                        product_weight_value = product_match.group(1) if product_match else None
                    result = service.capture(
                        str(payload.get("qr_code", "")),
                        weight,
                        str(payload.get("unit", "kg")),
                        frame,
                        bool(payload.get("vision_confirmed", False)),
                        weight_raw,
                        product_frame=product_frame,
                        product_weight=float(product_weight_value)
                        if product_weight_value is not None
                        else None,
                        event_id=str(payload["event_id"]) if payload.get("event_id") else None,
                        analysis_id=str(payload["analysis_id"])
                        if payload.get("analysis_id")
                        else None,
                        station_id=str(payload["station_id"])
                        if payload.get("station_id")
                        else None,
                        camera_id=str(payload["camera_id"])
                        if payload.get("camera_id")
                        else None,
                        frame_sha256=str(payload["frame_sha256"])
                        if payload.get("frame_sha256")
                        else None,
                    )
                    self.send_json(201, result)
                    return
                self.send_json(404, {"ok": False, "error": "not_found"})
            except (SessionConflictError, AnalysisBindingMismatch, EventIdConflictError) as exc:
                self.send_json(409, {"ok": False, "error": "session_conflict", "message": str(exc)})
            except InferenceQueueFull as exc:
                self.send_json(503, {"ok": False, "error": "inference_queue_full", "message": str(exc)})
            except ValueError as exc:
                self.send_json(422, {"ok": False, "error": "invalid_input", "message": str(exc)})
            except Exception as exc:
                self.send_json(500, {"ok": False, "error": "server_error", "message": str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            return None

    return ThreadingHTTPServer((args.host, args.port), Handler), service


def run(args: argparse.Namespace) -> int:
    server, service = create_server(args)
    service.start_ocr_preload()
    print(f"Test UI: http://{args.host}:{server.server_port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
        if service.sync_worker is not None:
            service.sync_worker.stop()
        service.close()
        service.store.close()
    return 0


def main() -> None:
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
