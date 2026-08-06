from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import json
import math
import os
import inspect
import sys
import threading
import time
import urllib.parse
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
from .capture_gate import frame_fingerprint
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

    @staticmethod
    def _roi_text(roi: NormalizedROI | None) -> str:
        if roi is None:
            return ""
        return f"{roi.x1:.4f},{roi.y1:.4f},{roi.x2:.4f},{roi.y2:.4f}"

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
                self.gemini_reader.status()
                if self.gemini_reader is not None
                else {"enabled": False}
            ),
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
    ) -> dict[str, object]:
        if unit not in UNITS:
            raise ValueError("Đơn vị không hợp lệ")
        if recognition_profile not in GEMINI_RECOGNITION_PROFILES:
            raise ValueError("Chế độ nhận diện phải là fast hoặc accurate")
        quality = self.assess_quality(frame)
        quality_payload, quality_pass = self.quality_result(quality)
        decoded = self._decode_qr(frame)
        frames = [frame, *(weight_frames or [])]
        auto_roi = roi_text.strip().lower() in {"", "auto"}
        roi: NormalizedROI | None
        if self.weight_engine == "gemini":
            # Gemini primary receives full frames. ROI detection must never
            # block or misdirect cloud recognition; the yellow box is hidden.
            roi = None
            roi_method = "gemini-full-frame"
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
        gemini_suggestion: float | None = None
        gemini_latency_seconds: float | None = None
        gemini_input_tokens: int | None = None
        gemini_output_tokens: int | None = None
        gemini_thinking_tokens: int | None = None
        gemini_total_tokens: int | None = None
        requires_human_review = False
        if self.weight_engine == "gemini":
            selected_gemini_reader = self._gemini_reader_for(recognition_profile)
            single_image_request = len(frames) == 1
            gemini_frames = [frame] if single_image_request else frames
            if len(gemini_frames) == 2:
                reading = WeightReading(
                    None,
                    unit,
                    False,
                    "GEMINI PRIMARY: cần ít nhất 3 frame camera mới",
                )
                recognition_source = "none"
            else:
                suggestion = selected_gemini_reader.read(gemini_frames, unit=unit)
                gemini_suggestion = suggestion.value
                gemini_latency_seconds = suggestion.latency_seconds
                gemini_input_tokens = suggestion.input_tokens
                gemini_output_tokens = suggestion.output_tokens
                gemini_thinking_tokens = suggestion.thinking_tokens
                gemini_total_tokens = suggestion.total_tokens
                local_qr = str(decoded.get("qr_code") or "").strip()
                gemini_qr = str(suggestion.qr_code or "").strip()
                qr_note = ""
                if local_qr and gemini_qr and local_qr != gemini_qr:
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
                        f"{suggestion.raw}{qr_note}; GEMINI PRIMARY: rejected",
                    )
                    recognition_source = "none"
                else:
                    reading = WeightReading(
                        suggestion.value,
                        suggestion.unit,
                        True,
                        (
                            f"{suggestion.raw}{qr_note}; GEMINI PRIMARY: single full-image accepted"
                            if single_image_request
                            else f"{suggestion.raw}{qr_note}; GEMINI PRIMARY: 3-frame schema accepted"
                        ),
                    )
                    recognition_source = "gemini-primary"
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
            "weight_found": reading.value is not None,
            "weight": reading.value,
            "unit": reading.unit,
            "confidence": reading.confidence,
            "weight_raw": reading.raw,
            "recognition_source": recognition_source,
            "recognition_profile": recognition_profile,
            "local_candidate": (
                local_candidate.value
                if local_candidate is not None
                else None
            ),
            "gemini_used": gemini_used,
            "gemini_suggestion": gemini_suggestion,
            "gemini_latency_seconds": gemini_latency_seconds,
            "gemini_input_tokens": gemini_input_tokens,
            "gemini_output_tokens": gemini_output_tokens,
            "gemini_thinking_tokens": gemini_thinking_tokens,
            "gemini_total_tokens": gemini_total_tokens,
            "requires_human_review": requires_human_review,
            "roi": self._roi_text(roi) if roi is not None else None,
            "roi_method": roi_method,
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
        configured_roi = self.weight_rois.get(station_id or "")
        if configured_roi is None and not any(identities) and self.station_count == 1:
            configured_roi = self.weight_rois.get(str(self.station_configs[0]["station_id"]))
        roi_method_override = None
        if roi_text.strip().lower() in {"", "auto"} and configured_roi is not None:
            roi_text = self._roi_text(configured_roi)
            roi_method_override = "camera-calibrated"
        if not any(identities):
            return self.inference.run(
                self._analyze_frame,
                frame,
                roi_text,
                unit,
                additional_frames,
                roi_method_override,
                recognition_profile,
            )
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
            )
            binding = self.sessions.mark_ready(binding.analysis_id)
        except Exception as exc:
            try:
                self.sessions.mark_failed(binding.analysis_id, exc)
            except SessionConflictError:
                pass
            raise
        return {**result, **binding.as_dict()}

    def capture(
        self,
        qr_code: str,
        weight: float,
        unit: str,
        frame: np.ndarray,
        vision_confirmed: bool = False,
        weight_raw: str = "",
        *,
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
                weight_raw=weight_raw[:500] if vision_confirmed else f"MANUAL:{weight}",
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
        # Commit locally first, then synchronously confirm this exact event so
        # one operator click sends the product code, core weight and evidence
        # image together. A failed cloud attempt remains durable in the outbox.
        if self.sync_worker is not None:
            cloud_confirmed = self.sync_worker.sync_event(measurement.event_id)
            if not cloud_confirmed:
                self.sync_worker.notify()
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


TEST_UI_HTML = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Việt Nhật IPT — Trạm cân QR</title>
<style>
@font-face{font-family:Roboto;src:local("Roboto"),local("Roboto Regular");font-style:normal;font-weight:400;font-display:swap}
:root{--ink:#151517;--muted:#666a73;--line:#d9dadd;--primary:#d71920;--primary-dark:#a90f15;--blue:#d71920;--green:#08783e;--red:#bd1e2d;--amber:#9a6200;--bg:#f1f2f4;--surface:#fff;--black:#0d0d0f}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#eceef1 0,#f6f6f7 360px);color:var(--ink);font:15px/1.45 Roboto,"Segoe UI",Arial,sans-serif}
.top{position:sticky;top:0;z-index:50;background:linear-gradient(105deg,#09090a 0,#17171a 62%,#0c0c0e 100%);color:#fff;padding:10px 24px;border-bottom:4px solid var(--primary);box-shadow:0 7px 24px #0004}.top-inner{width:100%;margin:auto;display:flex;align-items:center;justify-content:space-between;gap:22px}.brand{display:flex;align-items:center;min-width:max-content}.brand-mark{display:block;width:240px;height:auto;background:#fff;border-radius:9px;box-shadow:0 5px 18px #0006}.badges{display:flex;justify-content:flex-end;gap:7px;flex-wrap:wrap}.mode{display:inline-flex;align-items:center;gap:6px;background:#242428;color:#e7e7ea;border:1px solid #3b3b40;padding:6px 10px;border-radius:999px;font-weight:700;font-size:11px;letter-spacing:.045em}.mode.ai{background:#2b1215;color:#ffd9db;border-color:#6e2026}.mode.ai:before{content:"";width:7px;height:7px;border-radius:50%;background:#f1262d;box-shadow:0 0 0 3px #d7192033}.mode.off{background:#242428;color:#a9a9b0;border-color:#3b3b40}
main{width:100%;margin:18px 0;padding:0 18px 24px;display:block}.card{min-width:0;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 8px 28px #15151712}.card:first-child{min-height:calc(100vh - 126px)}.card h2{display:flex;align-items:center;gap:9px;font-size:18px;margin:0 0 17px;font-weight:800}.card h2:before{content:"";display:block;width:4px;height:22px;border-radius:3px;background:var(--primary)}.video-box{position:relative;aspect-ratio:16/9;background:#0b0b0d;border-radius:10px;overflow:hidden;margin-bottom:14px;box-shadow:inset 0 0 0 1px #ffffff14}
video,#preview,.station-card img{width:100%;height:100%;object-fit:contain;display:none}.placeholder{position:absolute;inset:0;display:grid;place-items:center;color:#9eacbd;text-align:center;padding:24px}
#roiBox,#qrBox,.roi-overlay,.qr-overlay{position:absolute;display:none;box-shadow:0 0 0 1px #000;pointer-events:none;z-index:3}#roiBox,.roi-overlay{border:3px solid #ffd400;background:#ffd40022}#qrBox,.qr-overlay{border:3px solid #00d36f;background:#00d36f22}#roiLabel,#qrLabel,.roi-overlay span,.qr-overlay span{position:absolute;left:0;top:-27px;padding:2px 7px;font-size:12px;font-weight:900;white-space:nowrap}#roiLabel,.roi-overlay span{background:#ffd400;color:#302600}#qrLabel,.qr-overlay span{background:#00d36f;color:#043c22}
.toolbar,.row{display:flex;gap:10px;flex-wrap:wrap}.toolbar{margin-bottom:15px}.field{margin:12px 0}.field label{display:block;color:var(--muted);font-size:13px;font-weight:700;margin-bottom:6px}
input,select,button{font:inherit;border-radius:8px}input,select{border:1px solid #b8bac0;padding:11px 12px;background:#fff;color:var(--ink)}input:focus,select:focus{outline:3px solid #d7192026;border-color:var(--primary)}
.grow{flex:1;min-width:180px}#captureQr{width:100%}.weight{font-size:28px;font-weight:800;width:240px}button{border:1px solid #d1d2d6;padding:10px 14px;font-weight:700;cursor:pointer;background:#f0f0f2;color:var(--ink);transition:background .15s,border-color .15s,transform .15s,box-shadow .15s}button:hover:not(:disabled){background:#e4e4e7;border-color:#b9bac0;transform:translateY(-1px)}button.primary{border-color:var(--primary);background:var(--primary);color:#fff;box-shadow:0 4px 12px #d7192033}button.primary:hover:not(:disabled),button.save:hover:not(:disabled){background:var(--primary-dark);border-color:var(--primary-dark)}button.save{width:100%;font-size:18px;border-color:var(--primary);background:linear-gradient(135deg,#e31a22,#bd1118);color:#fff;padding:14px;margin-top:8px;box-shadow:0 5px 16px #bd111833}button:disabled{opacity:.5;cursor:not-allowed;transform:none}
.status{margin-top:14px;padding:12px;border-radius:9px;background:#eef3f8;white-space:pre-wrap}.status.ok{background:#e5f5ec;color:#075a30}.status.bad{background:#fdebed;color:#8d1420}.status.warn{background:#fff3d9;color:#694300}
.lookup-form{display:flex;gap:8px}.lookup-form input{min-width:0;flex:1}.big-weight{font-size:38px;line-height:1.1;color:var(--primary);font-weight:900;margin:14px 0 6px}.details{color:var(--muted)}.evidence{width:100%;border-radius:9px;margin-top:14px;border:1px solid var(--line)}
.lookup-tools{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.lookup-camera{display:none;aspect-ratio:4/3;background:#0d1521;border-radius:9px;overflow:hidden;margin-top:10px}.lookup-camera video{display:block;width:100%;height:100%;object-fit:contain}
.flow{margin:18px 0 0;padding-left:20px;color:var(--muted)}.flow li::marker{color:var(--primary);font-weight:800}.kbd{border:1px solid #aeb0b6;border-bottom-width:3px;border-radius:5px;padding:1px 6px;background:#fff;color:#1c1c1f;font-weight:700}.roi-help{margin:-5px 0 12px;color:var(--muted);font-size:13px}.roi-help code{color:#6b5000;background:#fff4bd;padding:2px 5px;border-radius:4px}.result-label{font-size:13px;font-weight:800;color:var(--muted);margin:0 0 7px}.step{display:inline-grid;place-items:center;width:22px;height:22px;border-radius:50%;background:var(--primary);color:#fff;margin-right:5px}canvas{display:none}
.station-tools{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 12px}.station-tools .identity{margin-left:auto;color:var(--muted);font-size:12px}.station-grid{display:grid;grid-template-columns:repeat(var(--station-count,1),minmax(0,1fr));gap:12px;margin-bottom:14px}.station-card{border:2px solid var(--line);border-radius:12px;padding:10px;min-width:0;background:#f8f8f9;cursor:pointer;transition:border-color .15s,box-shadow .15s,background .15s}.station-card:hover{border-color:#b9bac0}.station-card.selected{border-color:var(--primary);background:#fffafa;box-shadow:0 0 0 3px #d719201f}.station-head{display:flex;gap:8px;align-items:center;margin-bottom:8px}.station-name{font-weight:900}.station-camera{min-width:0;flex:1;padding:7px}.station-card .video-box{margin-bottom:8px}.station-state{font-size:12px;color:var(--muted);overflow-wrap:anywhere}.station-state.ready{color:var(--green);font-weight:800}.station-state.bad{color:var(--red)}.station-card:not(.selected) #roiBox,.station-card:not(.selected) #qrBox{display:none!important}.auto-option{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:13px}.auto-option input{width:auto;accent-color:var(--primary)}.discard{background:#fdebed;color:var(--red);border-color:#f1c9cd}
.profile-option{display:flex;align-items:center;gap:7px;padding:0 10px;border:1px solid var(--line);border-radius:8px;background:#fafafa;color:var(--muted);font-size:13px;font-weight:700}.profile-option select{border:0;padding:9px 4px;background:transparent;font-weight:800;color:var(--ink)}
@media(max-width:1100px){.card:first-child{min-height:0}.top{position:static}.top-inner{align-items:flex-start;flex-direction:column}.badges{justify-content:flex-start}.weight{width:100%}}
@media(max-width:1050px){.station-grid{grid-template-columns:1fr}}
@media(max-width:520px){.top{padding:10px 14px}.brand-mark{width:205px}.mode{font-size:10px;padding:5px 8px}main{padding-inline:10px}.card{padding:15px}}
</style></head><body>
<header class="top"><div class="top-inner"><div class="brand"><img class="brand-mark" src="/logo.jpg" alt="Việt Nhật IPT"></div><div class="badges"><span id="weightModeBadge" class="mode ai">MỘT CAMERA · QR + CÂN</span><span id="ocrBadge" class="mode ai">CÂN: ĐANG KIỂM TRA</span><span id="geminiBadge" class="mode off">GEMINI: TẮT</span><span id="yoloBadge" class="mode off">YOLO: ĐANG KIỂM TRA</span><span id="syncBadge" class="mode off">CLOUD: ĐANG KIỂM TRA</span></div></div></header>
<main>
<section class="card"><h2>1. Ghi nhận lần cân</h2>
 <div class="result-label"><span class="step">3</span>Ảnh camera / ảnh bằng chứng</div>
 <div class="station-tools"><button id="refreshCamerasBtn">Làm mới camera</button><button id="openAllBtn">Mở camera đã gán</button><label class="auto-option"><input id="autoAdvance" type="checkbox">Tự chọn trạm kế tiếp sau khi lưu</label><span id="gatewayIdentity" class="identity">gateway: --</span></div>
 <div id="stationGrid" class="station-grid" style="--station-count:1">
  <article id="stationCard1" class="station-card selected" data-station-index="0" tabindex="0"><div class="station-head"><span id="stationName1" class="station-name">Trạm 1 <span class="kbd">1</span></span><select id="cameraSelect1" class="station-camera" aria-label="Camera trạm 1"><option value="">Chọn camera chính xác…</option></select></div>
   <div id="videoBox" class="video-box"><div id="placeholder" class="placeholder">Gán camera hoặc chọn ảnh có QR + màn hình cân</div><video id="video" playsinline muted></video><img id="preview" alt="Ảnh bằng chứng"><div id="qrBox"><span id="qrLabel">MÃ QR</span></div><div id="roiBox"><span id="roiLabel">VÙNG SỐ CÂN</span></div></div>
   <canvas id="canvas"></canvas><div id="stationStatus1" class="station-state">Chưa gán camera</div>
  </article>
 </div>
 <div class="toolbar"><button id="cameraBtn">Mở camera đã chọn</button><button id="captureFileBtn">Chọn ảnh thực tế</button><input id="captureFile" type="file" accept="image/*" hidden><button id="demoBtn">Dùng ảnh demo kho</button><label id="recognitionProfileOption" class="profile-option" for="recognitionProfile" hidden>Gemini<select id="recognitionProfile"><option value="fast">Nhanh · Flash-Lite · 10s</option><option value="accurate">Chính xác · Pro · 30s</option></select></label><button id="analyzeBtn" class="primary">Chụp cân lõi <span class="kbd">Space</span></button><button id="discardBtn" class="discard" disabled>Bỏ lần đang xem <span class="kbd">Backspace</span></button><button id="factoryBtn" disabled>Lưu mẫu đã kiểm tra</button></div>
 <div class="roi-help">Gemini nhận một ảnh toàn khung để đọc cân lõi. Đọc xong, ảnh và số cân được giữ chờ mã nhập SP: <code id="roiValue">TOÀN ẢNH</code></div>
 <div class="row"><div class="field grow"><label for="weight"><span class="step">1</span>Khối lượng lõi — AI tự đọc, có thể sửa trước khi lưu</label><input class="weight" id="weight" type="number" min="0" step="0.001" placeholder="--"></div><div class="field"><label for="unit">Đơn vị</label><select id="unit"><option>kg</option><option>g</option><option>lb</option></select></div></div>
 <div class="field"><label for="captureQr"><span class="step">2</span>Mã nhập SP — quét hoặc nhập sau khi cân lõi</label><input class="grow" id="captureQr" autocomplete="off" autofocus placeholder="Nhập mã SP để hoàn tất"></div>
 <button id="saveBtn" class="save" disabled>Xác nhận mã SP, lưu và đồng bộ <span class="kbd">Enter</span></button>
 <div id="captureStatus" class="status">Đang chờ dữ liệu…</div>
</section>
</main>
<script id="legacyScript" type="text/plain">
const $=id=>document.getElementById(id),video=$('video'),preview=$('preview'),canvas=$('canvas'),videoBox=$('videoBox'),roiBox=$('roiBox'),qrBox=$('qrBox');
const captureQr=$('captureQr'),weight=$('weight'),unit=$('unit'),captureStatus=$('captureStatus'),recognitionProfile=$('recognitionProfile');
const lookupQr=$('lookupQr'),lookupStatus=$('lookupStatus'),lookupResult=$('lookupResult');
const lookupVideo=$('lookupVideo'),lookupCanvas=$('lookupCanvas');
let stream=null,lookupStream=null,lookupDetector=null,lookupScanBusy=false;
let roi=null,qrRoi=null,capturedImage=null,analyzedWeight=null,analyzedRaw='',lastAnalysis=null,captureCount=0;
function status(el,text,type=''){el.textContent=text;el.className='status '+type}
function visibleSource(){return preview.style.display==='block'?preview:video}
function sourceSize(source){return source===video?[source.videoWidth,source.videoHeight]:[source.naturalWidth,source.naturalHeight]}
function mediaGeometry(){const source=visibleSource(),[sw,sh]=sourceSize(source),box=videoBox.getBoundingClientRect();if(!sw||!sh)return null;const scale=Math.min(box.width/sw,box.height/sh),width=sw*scale,height=sh*scale;return{box,left:(box.width-width)/2,top:(box.height-height)/2,width,height}}
function roiText(){return roi?[roi.x1,roi.y1,roi.x2,roi.y2].map(v=>v.toFixed(4)).join(','):'auto'}
function parseBox(text){const values=String(text||'').split(',').map(Number);return values.length===4&&values.every(Number.isFinite)?{x1:values[0],y1:values[1],x2:values[2],y2:values[3]}:null}
function positionBox(element,value,g){if(!g||!value){element.style.display='none';return}element.style.display='block';element.style.left=(g.left+value.x1*g.width)+'px';element.style.top=(g.top+value.y1*g.height)+'px';element.style.width=((value.x2-value.x1)*g.width)+'px';element.style.height=((value.y2-value.y1)*g.height)+'px'}
function updateBoxes(){const g=mediaGeometry();positionBox(roiBox,roi,g);positionBox(qrBox,qrRoi,g);if(!roi)$('roiValue').textContent='TỰ ĐỘNG'}
function applyDetectedRoi(text,method){roi=parseBox(text);$('roiValue').textContent=roi?'TỰ ĐỘNG · '+method:'KHÔNG TÌM THẤY';updateBoxes()}
function applyQrRoi(text){qrRoi=parseBox(text);updateBoxes()}
function showVideo(){video.style.display='block';preview.style.display='none';$('placeholder').style.display='none';requestAnimationFrame(updateBoxes)}
function showPreview(){video.style.display='none';preview.style.display='block';$('placeholder').style.display='none';requestAnimationFrame(updateBoxes)}
function resetResult(){roi=null;qrRoi=null;capturedImage=null;analyzedWeight=null;analyzedRaw='';lastAnalysis=null;captureQr.value='';weight.value='';$('saveBtn').disabled=true;$('factoryBtn').disabled=true;updateBoxes()}
function prepareNextCapture(qrCode){roi=null;qrRoi=null;capturedImage=null;analyzedWeight=null;analyzedRaw='';lastAnalysis=null;captureQr.value=qrCode;weight.value='';$('saveBtn').disabled=true;$('factoryBtn').disabled=true;if(stream)showVideo();updateBoxes()}
async function openCamera(){try{stopLookupCamera();if(stream)stream.getTracks().forEach(t=>t.stop());stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1920},height:{ideal:1080}},audio:false});video.srcObject=stream;await video.play();resetResult();showVideo();status(captureStatus,'Camera sẵn sàng. Đặt QR và màn hình cân rõ trong ảnh rồi nhấn Space.','ok')}catch(e){status(captureStatus,'Không mở được camera: '+e.message,'bad')}}
function loadDemo(){if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;resetResult();preview.onload=()=>{showPreview();status(captureStatus,'Đã nạp ảnh demo kho. Đang tự đọc QR + cân…','warn');analyzeCurrent()};preview.src='/demo.jpg?t='+Date.now()}
function fileData(file){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error('Không đọc được tệp ảnh'));reader.readAsDataURL(file)})}
async function loadCaptureFile(file){if(!file)return;try{if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;const data=await fileData(file);resetResult();preview.onload=()=>{showPreview();status(captureStatus,'Đã chọn '+file.name+'. Đang tự đọc QR + cân…','warn');analyzeCurrent()};preview.src=data}catch(e){status(captureStatus,e.message,'bad')}}
function drawCurrent(){const source=stream&&video.videoWidth?video:preview,[w,h]=sourceSize(source);if(!w||!h)throw new Error('Chưa có ảnh hoặc camera');canvas.width=w;canvas.height=h;canvas.getContext('2d').drawImage(source,0,0,w,h);return canvas.toDataURL('image/jpeg',.94)}
async function api(path,options){const response=await fetch(path,options);let data;try{data=await response.json()}catch{throw new Error('Server trả dữ liệu không hợp lệ')}if(!response.ok)throw new Error(data.message||data.error||('HTTP '+response.status));return data}
async function loadStatus(){try{const data=await api('/api/status');const yolo=$('yoloBadge');yolo.textContent=data.yolo_enabled?'YOLO: BẬT · '+data.yolo_mode.toUpperCase():'YOLO: TẮT · KHÔNG CẦN CHO QR RÕ';yolo.className='mode '+(data.yolo_enabled?'ai':'off');$('ocrBadge').textContent='OCR: '+(data.ocr_engine||'LOCAL')+' · BATCH 3';const gemini=$('geminiBadge'),geminiOn=Boolean(data.gemini&&data.gemini.enabled);gemini.textContent=geminiOn?'GEMINI FALLBACK: BẬT':'GEMINI FALLBACK: TẮT';gemini.className='mode '+(geminiOn?'ai':'off');const sync=$('syncBadge');sync.textContent=data.sync_enabled?'ĐỒNG BỘ SUPABASE: BẬT':'CLOUD: CHỈ LƯU OFFLINE';sync.className='mode '+(data.sync_enabled?'ai':'off')}catch{}}
async function analyzeCurrent(){if($('analyzeBtn').disabled)return;try{$('analyzeBtn').disabled=true;$('saveBtn').disabled=true;status(captureStatus,'Đang tự tìm QR, màn hình cân và kiểm tra chất lượng ảnh…','warn');capturedImage=drawCurrent();preview.onload=()=>showPreview();preview.src=capturedImage;const data=await api('/api/analyze',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({image:capturedImage,roi:'auto',unit:unit.value})});lastAnalysis=data;$('factoryBtn').disabled=false;applyDetectedRoi(data.roi,data.roi_method);applyQrRoi(data.qr_roi);captureQr.value=data.qr_code||'';weight.value=data.weight_found?data.weight:'';analyzedWeight=data.weight;analyzedRaw=data.weight_raw||'';const confidence=data.confidence==null?'--':Math.round(data.confidence*100)+'%';const issues=(data.quality&&data.quality.issues)||[];if(data.qr_found&&data.weight_found&&data.quality_pass){$('saveBtn').disabled=false;status(captureStatus,'ĐÃ TỰ NHẬN DIỆN · ẢNH ĐẠT\nQR: '+data.qr_code+' ('+data.qr_decoder+')\nCÂN: '+data.weight+' '+data.unit+' · tin cậy '+confidence+'\nKiểm tra rồi nhấn Enter để lưu.','ok')}else if(data.qr_found&&data.weight_found){status(captureStatus,'ĐÃ ĐỌC ĐƯỢC NHƯNG KHÔNG CHO LƯU\n'+issues.join('\n')+'\nChụp lại để có ảnh bằng chứng đạt chất lượng.','bad')}else{const missing=[!data.qr_found?'QR':'',!data.weight_found?'số cân':''].filter(Boolean).join(' và ');const quality=issues.length?'\nChất lượng: '+issues.join('; '):'';status(captureStatus,'CHƯA ĐỌC ĐƯỢC '+missing+'. Đưa QR/màn hình cân rõ hơn rồi thử lại.'+quality+'\nChi tiết: '+data.weight_raw,'bad')}}catch(e){lastAnalysis=null;status(captureStatus,'NHẬN DIỆN LỖI: '+e.message,'bad')}finally{$('analyzeBtn').disabled=false}}
async function saveFactorySample(){if(!capturedImage||!lastAnalysis)return;const expectedWeight=Number(weight.value);if(!captureQr.value.trim()||!Number.isFinite(expectedWeight)){status(captureStatus,'Hãy kiểm tra và điền đúng QR + số cân trước khi lưu mẫu.','bad');return}try{$('factoryBtn').disabled=true;status(captureStatus,'Đang lưu ảnh xưởng, kết quả máy và giá trị nhân sự đã kiểm tra…','warn');const data=await api('/api/factory-sample',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({image:capturedImage,qr_roi:lastAnalysis.qr_roi||'',predicted_qr_code:lastAnalysis.qr_code||'',predicted_weight:lastAnalysis.weight,expected_qr_code:captureQr.value.trim(),expected_weight:expectedWeight,unit:unit.value,recognition_ok:Boolean(lastAnalysis.qr_found&&lastAnalysis.weight_found),qr_decoder:lastAnalysis.qr_decoder||'',ocr_confidence:lastAnalysis.confidence})});status(captureStatus,'ĐÃ LƯU MẪU XƯỞNG '+data.sample_id+'\nĐã ghi riêng kết quả máy và giá trị nhân sự kiểm tra.\n'+(data.auto_labeled?'Có nhãn QR tự động — vẫn phải duyệt bounding box trước khi train.':'Chưa có nhãn QR — cần gán nhãn thủ công.'),'ok')}catch(e){status(captureStatus,'KHÔNG LƯU ĐƯỢC MẪU: '+e.message,'bad')}finally{$('factoryBtn').disabled=false}}
async function saveCapture(){if($('saveBtn').disabled||!capturedImage)return;const value=Number(weight.value);if(!captureQr.value.trim()){status(captureStatus,'Mã QR đang trống.','bad');return}if(!Number.isFinite(value)||value<0){status(captureStatus,'Số cân không hợp lệ.','bad');return}try{$('saveBtn').disabled=true;status(captureStatus,'Đang lưu ảnh + QR + số cân bằng cùng một ID…','warn');const data=await api('/api/capture',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({qr_code:captureQr.value,weight:value,unit:unit.value,image:capturedImage,vision_confirmed:value===Number(analyzedWeight),weight_raw:analyzedRaw+'; HUMAN_CONFIRMED='+value})});lookupQr.value=data.qr_code;captureCount+=1;let cloud='Đã lưu offline · đang chờ đồng bộ cloud';if(data.sync_status==='synced'&&data.remote_image_url)cloud='Ảnh đã lên Cloudinary · dữ liệu đã lên Supabase';else if(data.sync_status==='synced')cloud='Dữ liệu đã lên Supabase · Cloudinary chưa được xác nhận';prepareNextCapture(data.qr_code);status(captureStatus,'ĐÃ LƯU LẦN '+captureCount+'\nID: '+data.event_id+'\nQR: '+data.qr_code+'\nCÂN: '+data.weight+' '+data.unit+'\n'+cloud+'\nSẴN SÀNG CHỤP TIẾP · Nhấn Space.','ok')}catch(e){status(captureStatus,'KHÔNG LƯU: '+e.message,'bad');$('saveBtn').disabled=false}}
async function lookup(){const qr=lookupQr.value.trim();if(!qr)return;try{status(lookupStatus,'Đang tra cứu…','warn');lookupResult.replaceChildren();const data=await api('/api/lookup?qr='+encodeURIComponent(qr));const m=data.measurement;status(lookupStatus,'Tìm thấy '+data.history_count+' lần cân.','ok');const title=document.createElement('div');title.textContent='QR: '+m.qr_code;const big=document.createElement('div');big.className='big-weight';big.textContent='NET '+m.net_weight+' '+m.unit;const details=document.createElement('div');details.className='details';details.textContent='Gross '+m.gross_weight+' '+m.unit+' · Tare '+m.tare_weight+' '+m.unit+' · '+new Date(m.captured_at).toLocaleString('vi-VN');lookupResult.append(title,big,details);if(m.image_url){const img=document.createElement('img');img.className='evidence';img.alt='Ảnh bằng chứng';img.src=m.image_url;lookupResult.appendChild(img)}}catch(e){status(lookupStatus,'Không tra cứu được: '+e.message,'bad')}}
async function decodeLookupImage(image,label='ảnh'){try{status(lookupStatus,'Đang đọc QR từ '+label+'…','warn');const data=await api('/api/decode',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({image})});if(!data.found)throw new Error('Không tìm thấy QR trong '+label);lookupQr.value=data.qr_code;status(lookupStatus,'Đã đọc '+data.qr_code+' ('+data.decoder+'). Đang tra cứu…','ok');await lookup();return true}catch(e){status(lookupStatus,e.message,'bad');return false}}
async function loadLookupFile(file){if(!file)return;try{await decodeLookupImage(await fileData(file),'ảnh '+file.name)}catch(e){status(lookupStatus,e.message,'bad')}}
function stopLookupCamera(){if(lookupStream)lookupStream.getTracks().forEach(t=>t.stop());lookupStream=null;lookupVideo.srcObject=null;$('lookupCameraBox').style.display='none';$('lookupStopBtn').hidden=true;$('lookupCameraBtn').hidden=false}
async function startLookupCamera(){try{if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}lookupStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});lookupVideo.srcObject=lookupStream;await lookupVideo.play();$('lookupCameraBox').style.display='block';$('lookupStopBtn').hidden=false;$('lookupCameraBtn').hidden=true;status(lookupStatus,'Đưa QR vào camera. Hệ thống đang quét…','warn');lookupCameraLoop()}catch(e){status(lookupStatus,'Không mở được camera: '+e.message,'bad')}}
async function lookupCameraLoop(){if(!lookupStream||lookupScanBusy)return;lookupScanBusy=true;try{let value='';if('BarcodeDetector'in window){lookupDetector=lookupDetector||new BarcodeDetector({formats:['qr_code']});const codes=await lookupDetector.detect(lookupVideo);if(codes.length)value=codes[0].rawValue}if(value){lookupQr.value=value;stopLookupCamera();await lookup();return}if(lookupVideo.videoWidth){const max=960,scale=Math.min(1,max/lookupVideo.videoWidth);lookupCanvas.width=Math.round(lookupVideo.videoWidth*scale);lookupCanvas.height=Math.round(lookupVideo.videoHeight*scale);lookupCanvas.getContext('2d').drawImage(lookupVideo,0,0,lookupCanvas.width,lookupCanvas.height);const found=await decodeLookupImage(lookupCanvas.toDataURL('image/jpeg',.78),'camera');if(found){stopLookupCamera();return}}}catch{}finally{lookupScanBusy=false}if(lookupStream)setTimeout(lookupCameraLoop,700)}
$('cameraBtn').onclick=openCamera;$('captureFileBtn').onclick=()=>$('captureFile').click();$('captureFile').onchange=e=>loadCaptureFile(e.target.files[0]);$('demoBtn').onclick=loadDemo;$('analyzeBtn').onclick=analyzeCurrent;$('factoryBtn').onclick=saveFactorySample;$('saveBtn').onclick=saveCapture;$('lookupBtn').onclick=lookup;
$('lookupCameraBtn').onclick=startLookupCamera;$('lookupStopBtn').onclick=stopLookupCamera;$('lookupFileBtn').onclick=()=>$('lookupFile').click();$('lookupFile').onchange=e=>loadLookupFile(e.target.files[0]);
lookupQr.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();e.stopPropagation();lookup()}});
document.addEventListener('keydown',e=>{if(e.code==='Space'&&!e.repeat&&!e.ctrlKey&&!e.altKey&&!e.metaKey){e.preventDefault();analyzeCurrent();return}if(e.key==='Enter'&&!e.repeat&&e.target!==lookupQr){e.preventDefault();saveCapture()}});
window.addEventListener('resize',updateBoxes);window.addEventListener('beforeunload',()=>{if(stream)stream.getTracks().forEach(t=>t.stop());stopLookupCamera()});loadStatus();loadDemo();
</script>
<script>
/* Multi-station controller. The disabled script above is retained as a compact
   record of the original one-camera workflow and its stable DOM/API names. */
const $=id=>document.getElementById(id);
const captureQr=$('captureQr'),weight=$('weight'),unit=$('unit'),captureStatus=$('captureStatus'),recognitionProfile=$('recognitionProfile');
const CAMERA_MAP_PREFIX='rollQrScale.cameraMap.v1:';
const RECOGNITION_PROFILE_KEY='rollQrScale.recognitionProfile.v1';
let appStatus=null,stations=[],selectedIndex=0,cameraDevices=[],captureCount=0;
try{const savedProfile=localStorage.getItem(RECOGNITION_PROFILE_KEY);if(['fast','accurate'].includes(savedProfile))recognitionProfile.value=savedProfile}catch{}

function status(el,text,type=''){el.textContent=text;el.className='status '+type}
async function api(path,options){const response=await fetch(path,options);let data;try{data=await response.json()}catch{throw new Error('Server trả dữ liệu không hợp lệ')}if(!response.ok)throw new Error(data.message||data.error||('HTTP '+response.status));return data}
function newEventId(){if(window.crypto&&typeof crypto.randomUUID==='function')return crypto.randomUUID();return'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g,c=>{const r=Math.random()*16|0,v=c==='x'?r:(r&3|8);return v.toString(16)})}
function mappingKey(){return CAMERA_MAP_PREFIX+(appStatus&&appStatus.gateway_id||'gateway-01')}
function readMappings(){try{const value=JSON.parse(localStorage.getItem(mappingKey())||'{}');return value&&typeof value==='object'?value:{}}catch{return{}}}
function writeMappings(value){localStorage.setItem(mappingKey(),JSON.stringify(value))}
function current(){return stations[selectedIndex]}

class StationSession{
 constructor(config,card){this.config=config;this.stationId=config.station_id;this.cameraId=config.camera_id;this.card=card;this.video=card.querySelector('video');this.preview=card.querySelector('img');this.canvas=card.querySelector('canvas');this.box=card.querySelector('.video-box');this.placeholder=card.querySelector('.placeholder');this.qrBox=card.querySelector('#qrBox,.qr-overlay');this.roiBox=card.querySelector('#roiBox,.roi-overlay');this.cameraSelect=card.querySelector('select');this.stateElement=card.querySelector('.station-state');this.stream=null;this.streamGeneration=0;this.reconnectTimer=null;this.deviceId='';this.state='idle';this.eventId=null;this.analysisId=null;this.frameSha256=null;this.capturedImage=null;this.lastAnalysis=null;this.qr='';this.weight='';this.unit='kg';this.analyzedWeight=null;this.analyzedRaw='';this.configuredRoi=parseBox(config.weight_roi);this.roi=this.configuredRoi;this.qrRoi=null;this.captureCount=0;this.hydratedPending=Boolean(config.event_id);if(this.hydratedPending){this.state=config.state||'review';this.eventId=config.event_id;this.analysisId=config.analysis_id||null;this.frameSha256=config.frame_sha256||null}}
 hasUnsavedReview(){return['analyzing','review','awaiting-code','ready','saving','error'].includes(this.state)&&Boolean(this.eventId)}
 setState(value,message,type=''){this.state=value;this.stateElement.textContent=message||value;this.stateElement.className='station-state '+type;renderControls()}
 stop(){clearTimeout(this.reconnectTimer);this.reconnectTimer=null;this.streamGeneration+=1;const oldStream=this.stream;this.stream=null;this.video.srcObject=null;if(oldStream)oldStream.getTracks().forEach(track=>track.stop())}
}
class CameraSession extends StationSession{}

function createStationCard(config,index){
 if(index===0){const card=$('stationCard1');$('stationName1').firstChild.textContent='Trạm 1 · '+config.station_id+' ';return card}
 const n=index+1,card=document.createElement('article');card.id='stationCard'+n;card.className='station-card';card.dataset.stationIndex=String(index);card.tabIndex=0;
 card.innerHTML='<div class="station-head"><span class="station-name"></span><select class="station-camera" aria-label="Camera"><option value="">Chọn camera chính xác…</option></select></div><div class="video-box"><div class="placeholder">Gán camera hoặc chọn ảnh</div><video playsinline muted></video><img alt="Ảnh bằng chứng"><div class="qr-overlay"><span>MÃ QR</span></div><div class="roi-overlay"><span>VÙNG SỐ CÂN</span></div></div><canvas></canvas><div class="station-state">Chưa gán camera</div>';
 const name=card.querySelector('.station-name');name.textContent='Trạm '+n+' · '+config.station_id+' ';const key=document.createElement('span');key.className='kbd';key.textContent=String(n);name.appendChild(key);$('stationGrid').appendChild(card);return card
}
function buildStations(configs){stations.forEach(item=>item.stop());stations=[];$('stationGrid').querySelectorAll('.station-card:not(#stationCard1)').forEach(card=>card.remove());$('stationGrid').style.setProperty('--station-count',String(configs.length));configs.forEach((config,index)=>{const session=new CameraSession(config,createStationCard(config,index));session.card.addEventListener('click',event=>{if(event.target!==session.cameraSelect)selectStation(index)});session.card.addEventListener('focus',()=>selectStation(index));session.cameraSelect.addEventListener('change',()=>mapCamera(session,session.cameraSelect.value));stations.push(session);if(session.hydratedPending){const waiting=session.state==='analyzing';session.stateElement.textContent=waiting?'PHÂN TÍCH ĐANG CHẠY · chờ cập nhật':'PHIÊN CHƯA LƯU TỪ TRƯỚC · bấm Bỏ lần đang xem';session.stateElement.className='station-state '+(waiting?'':'bad')}});selectedIndex=0;applyMappings();selectStation(0)}
function persistEditor(session){if(!session)return;session.qr=captureQr.value;session.weight=weight.value;session.unit=unit.value}
function selectStation(index){if(index<0||index>=stations.length)return;persistEditor(current());selectedIndex=index;stations.forEach((session,i)=>session.card.classList.toggle('selected',i===index));const session=current();captureQr.value=session.qr||'';weight.value=session.weight==null?'':session.weight;unit.value=session.unit||'kg';updateBoxes(session);renderControls();if(session.hydratedPending){const message=session.state==='analyzing'?'Phiên '+session.eventId+' vẫn đang phân tích; đang chờ trạng thái backend.':'Có phiên '+session.eventId+' từ trước nhưng trình duyệt không còn ảnh/form xem lại. Bấm Bỏ lần đang xem để tiếp tục.';status(captureStatus,message,'warn')}else if(session.state==='awaiting-code')status(captureStatus,'ĐÃ CÂN LÕI: '+session.weight+' '+session.unit+'\nẢnh và số cân đang chờ MÃ NHẬP SP.','warn');else if(session.state==='ready')status(captureStatus,'ĐÃ ĐỦ CÂN LÕI + MÃ NHẬP SP. Nhấn Enter để hoàn tất.','ok');else status(captureStatus,'Đã chọn '+session.stationId+' / '+session.cameraId+'. Space chỉ chụp trạm này.','ok')}
function coreReady(session){return Boolean(session&&session.capturedImage&&session.lastAnalysis&&session.lastAnalysis.weight_found&&session.lastAnalysis.quality_pass&&Number.isFinite(Number(session.weight))&&Number(session.weight)>=0)}
function completionReady(session){return coreReady(session)&&Boolean(String(session.qr||'').trim())}
function renderControls(){const session=current();if(!session)return;const geminiPrimary=Boolean(appStatus&&appStatus.weight_engine==='gemini');$('saveBtn').disabled=session.state!=='ready'||!completionReady(session);$('discardBtn').disabled=!session.hasUnsavedReview()||session.state==='saving'||session.state==='analyzing';$('factoryBtn').disabled=!session.lastAnalysis;$('analyzeBtn').disabled=session.state==='analyzing'||session.state==='saving';recognitionProfile.disabled=!geminiPrimary||session.state==='analyzing'||session.state==='saving'}
function refreshCompletionState(session=current()){if(!session||!session.eventId||!['awaiting-code','ready'].includes(session.state)){renderControls();return}const ready=completionReady(session);session.state=ready?'ready':'awaiting-code';session.stateElement.textContent=ready?'ĐỦ DỮ LIỆU · '+session.qr+' · '+session.weight+' '+session.unit:'ĐÃ CÂN LÕI · CHỜ MÃ NHẬP SP';session.stateElement.className='station-state '+(ready?'ready':'');renderControls();if(session===current())status(captureStatus,ready?'ĐÃ ĐỦ CÂN LÕI + MÃ NHẬP SP.\nNhấn Enter hoặc nút đỏ để lưu một lần và đưa vào hàng đồng bộ Supabase.':'ĐÃ CÂN LÕI: '+session.weight+' '+session.unit+'\nĐang giữ ảnh và kết quả. Quét hoặc nhập mã SP để hoàn tất.',''+(ready?'ok':'warn'))}

function applyMappings(){const mappings=readMappings();stations.forEach(session=>{session.deviceId=String(mappings[session.stationId]||'')});populateCameraSelectors()}
function cameraLabel(device,index){return device.label||('Camera '+(index+1)+' · '+device.deviceId.slice(0,8))}
function populateCameraSelectors(){stations.forEach(session=>{const selected=session.deviceId;session.cameraSelect.replaceChildren(new Option('Chọn camera chính xác…',''));cameraDevices.forEach((device,index)=>session.cameraSelect.add(new Option(cameraLabel(device,index),device.deviceId)));if(selected&&!cameraDevices.some(device=>device.deviceId===selected))session.cameraSelect.add(new Option('Camera đã gán (đang mất kết nối)',selected));session.cameraSelect.value=selected})}
async function refreshCameraDevices(requestPermission=false){try{if(requestPermission){const permissionStream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});permissionStream.getTracks().forEach(track=>track.stop())}const devices=await navigator.mediaDevices.enumerateDevices();cameraDevices=devices.filter(device=>device.kind==='videoinput'&&device.deviceId);populateCameraSelectors();return cameraDevices}catch(error){status(captureStatus,'Không liệt kê được camera: '+error.message,'bad');return[]}}
async function mapCamera(session,deviceId){if(session.hasUnsavedReview()){session.cameraSelect.value=session.deviceId;status(captureStatus,'Không đổi camera khi còn lần cân chưa lưu. Hãy lưu hoặc bấm Bỏ lần đang xem.','bad');return}if(deviceId&&stations.some(other=>other!==session&&other.deviceId===deviceId)){session.cameraSelect.value=session.deviceId;status(captureStatus,'Camera này đã được gán cho trạm khác.','bad');return}session.stop();session.deviceId=deviceId;const mappings=readMappings();if(deviceId)mappings[session.stationId]=deviceId;else delete mappings[session.stationId];writeMappings(mappings);session.setState(deviceId?'disconnected':'idle',deviceId?'Đã gán · chưa kết nối':'Chưa gán camera');if(deviceId)await openStationCamera(session)}
async function openStationCamera(session){if(!session.deviceId){await refreshCameraDevices(true);session.setState('idle','Hãy chọn camera chính xác trong danh sách','bad');return}if(session.hasUnsavedReview()){status(captureStatus,'Không mở lại camera khi còn lần cân chưa lưu.','bad');return}session.stop();const generation=session.streamGeneration;try{const requested=session.deviceId;const stream=await navigator.mediaDevices.getUserMedia({video:{deviceId:{exact:requested},width:{ideal:1920},height:{ideal:1080},frameRate:{ideal:25,max:30}},audio:false});if(session.streamGeneration!==generation||session.deviceId!==requested){stream.getTracks().forEach(item=>item.stop());return}const track=stream.getVideoTracks()[0],actual=track&&track.getSettings().deviceId;if(actual&&actual!==requested){stream.getTracks().forEach(item=>item.stop());throw new Error('Trình duyệt trả camera khác camera đã gán')}session.stream=stream;session.video.srcObject=stream;await session.video.play();track.addEventListener('ended',()=>{if(session.stream!==stream||session.streamGeneration!==generation)return;session.stream=null;session.video.srcObject=null;scheduleReconnect(session,requested)},{once:true});showVideo(session);session.setState('live','LIVE · '+session.stationId+' / '+session.cameraId,'ready')}catch(error){if(session.streamGeneration!==generation)return;const failedStream=session.stream;session.stream=null;session.video.srcObject=null;if(failedStream)failedStream.getTracks().forEach(item=>item.stop());session.setState('disconnected','MẤT KẾT NỐI · chỉ thử lại đúng camera đã gán','bad');status(captureStatus,'Không mở được '+session.stationId+': '+error.message,'bad');scheduleReconnect(session,session.deviceId)}}
function scheduleReconnect(session,expectedDeviceId){clearTimeout(session.reconnectTimer);if(!expectedDeviceId||session.deviceId!==expectedDeviceId)return;session.reconnectTimer=setTimeout(async()=>{await refreshCameraDevices();if(session.deviceId===expectedDeviceId)openStationCamera(session)},1800)}
async function openAllMapped(){await refreshCameraDevices(true);for(const session of stations)if(session.deviceId&&!session.hasUnsavedReview())await openStationCamera(session)}

function sourceSize(source,session){return source===session.video?[source.videoWidth,source.videoHeight]:[source.naturalWidth,source.naturalHeight]}
function showVideo(session){session.video.style.display='block';session.preview.style.display='none';session.placeholder.style.display='none';requestAnimationFrame(()=>updateBoxes(session))}
function showPreview(session){session.video.style.display='none';session.preview.style.display='block';session.placeholder.style.display='none';requestAnimationFrame(()=>updateBoxes(session))}
function visibleSource(session){return session.preview.style.display==='block'?session.preview:session.video}
function drawSession(session,quality=.94){const source=session.stream&&session.video.videoWidth&&session.preview.style.display!=='block'?session.video:visibleSource(session),[w,h]=sourceSize(source,session);if(!w||!h)throw new Error('Trạm đã chọn chưa có khung hình');session.canvas.width=w;session.canvas.height=h;session.canvas.getContext('2d').drawImage(source,0,0,w,h);return session.canvas.toDataURL('image/jpeg',quality)}
function waitForVideoFrame(video){return new Promise(resolve=>{let finished=false,timer;const done=()=>{if(finished)return;finished=true;clearTimeout(timer);resolve()};timer=setTimeout(done,120);if(typeof video.requestVideoFrameCallback==='function')video.requestVideoFrameCallback(done);else requestAnimationFrame(()=>requestAnimationFrame(done))})}
async function captureWeightBurst(session){if(appStatus&&appStatus.weight_engine==='gemini')return[];const total=Math.max(1,Math.min(9,Number(appStatus&&appStatus.weight_burst_frames||1)));if(total<=1||!session.stream||!session.video.videoWidth)return[];const frames=[];for(let index=1;index<total;index++){await waitForVideoFrame(session.video);if(!session.stream)break;frames.push(drawSession(session,.78))}return frames}
function parseBox(text){const values=String(text||'').split(',').map(Number);return values.length===4&&values.every(Number.isFinite)?{x1:values[0],y1:values[1],x2:values[2],y2:values[3]}:null}
function positionBox(element,value,geometry){if(!element||!geometry||!value){if(element)element.style.display='none';return}element.style.display='block';element.style.left=(geometry.left+value.x1*geometry.width)+'px';element.style.top=(geometry.top+value.y1*geometry.height)+'px';element.style.width=((value.x2-value.x1)*geometry.width)+'px';element.style.height=((value.y2-value.y1)*geometry.height)+'px'}
function updateBoxes(session=current()){if(!session)return;const source=visibleSource(session),[sw,sh]=sourceSize(source,session),rect=session.box.getBoundingClientRect();if(!sw||!sh){positionBox(session.roiBox,null,null);positionBox(session.qrBox,null,null);return}const scale=Math.min(rect.width/sw,rect.height/sh),width=sw*scale,height=sh*scale,geometry={left:(rect.width-width)/2,top:(rect.height-height)/2,width,height};positionBox(session.roiBox,session.roi,geometry);positionBox(session.qrBox,session.qrRoi,geometry);if(session===current())$('roiValue').textContent=session.configuredRoi?'CỐ ĐỊNH · '+session.config.weight_roi:session.roi?'TỰ ĐỘNG · '+((session.lastAnalysis&&session.lastAnalysis.roi_method)||'')+' · '+((session.lastAnalysis&&session.lastAnalysis.roi)||''):'TỰ ĐỘNG'}
function clearReview(session){session.eventId=null;session.analysisId=null;session.frameSha256=null;session.capturedImage=null;session.lastAnalysis=null;session.qr='';session.weight='';session.analyzedWeight=null;session.analyzedRaw='';session.roi=session.configuredRoi;session.qrRoi=null;session.hydratedPending=false;if(session.stream)showVideo(session);session.setState(session.stream?'live':'idle',session.stream?'LIVE · sẵn sàng chụp':'Đang chờ ảnh/camera')}
function prepareNextCapture(qrCode,session=current()){clearReview(session);session.qr=qrCode||'';if(session===current()){captureQr.value=session.qr;weight.value='';updateBoxes(session)}}

async function analyzeCurrent(){
 const session=current();
 if(!session||session.state==='analyzing'||session.state==='saving')return;
 if(session.hasUnsavedReview()){status(captureStatus,'Trạm này còn ảnh đang xem lại. Enter để lưu hoặc bấm Bỏ lần đang xem; không ghi đè im lặng.','bad');return}
 let image;
 try{image=drawSession(session)}catch(error){status(captureStatus,error.message,'bad');return}
 const requestEventId=newEventId();
 session.eventId=requestEventId;session.capturedImage=image;session.state='analyzing';
 if(session===current())status(captureStatus,'Đang lấy burst LED của '+session.stationId+'…','warn');
 let weightFrames=[];
 try{weightFrames=await captureWeightBurst(session)}catch(error){weightFrames=[]}
 session.preview.onload=()=>showPreview(session);session.preview.src=image;
 session.setState('analyzing','ĐANG PHÂN TÍCH · '+requestEventId,'');
 if(session===current())status(captureStatus,(appStatus&&appStatus.weight_engine==='gemini'?'Đang gửi một ảnh toàn khung để Gemini đọc cân lõi…':'Đang biểu quyết '+(weightFrames.length+1)+' frame local qua hàng đợi FIFO…'),'warn');
 try{
  const data=await api('/api/analyze',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({image,weight_frames:weightFrames,camera_capture:Boolean(session.stream),roi:'auto',unit:session.unit,event_id:requestEventId,station_id:session.stationId,camera_id:session.cameraId,recognition_profile:recognitionProfile.value})});
  if(session.eventId!==requestEventId||data.event_id!==requestEventId)return;
  session.analysisId=data.analysis_id;session.frameSha256=data.frame_sha256;session.lastAnalysis=data;session.roi=parseBox(data.roi)||session.configuredRoi;session.qrRoi=parseBox(data.qr_roi);session.qr='';session.weight=data.weight_found?String(data.weight):'';session.analyzedWeight=data.weight;session.analyzedRaw=data.weight_raw||'';
  const coreAccepted=Boolean(data.weight_found&&data.quality_pass),ready=Boolean(coreAccepted&&session.qr.trim());
  session.setState(ready?'ready':coreAccepted?'awaiting-code':'review',ready?'ĐỦ DỮ LIỆU · '+session.qr+' · '+session.weight+' '+data.unit:coreAccepted?'ĐÃ CÂN LÕI · CHỜ MÃ NHẬP SP':'CẦN KIỂM TRA / BỎ ẢNH',''+(ready?'ready':coreAccepted?'':'bad'));
  if(session===current()){
   captureQr.value=session.qr;weight.value=session.weight;unit.value=data.unit||session.unit;updateBoxes(session);
   const issues=(data.quality&&data.quality.issues)||[],burst=data.burst_frames||1;
     if(ready){$('saveBtn').disabled=false;status(captureStatus,'ĐÃ ĐỦ DỮ LIỆU · '+(data.recognition_source==='gemini-primary'?'GEMINI':'LOCAL')+' · '+burst+' ẢNH\nMÃ NHẬP SP: '+session.qr+'\nCÂN LÕI: '+session.weight+' '+data.unit+'\nNhấn Enter để lưu và đồng bộ một lần.','ok')}
  else if(coreAccepted){status(captureStatus,'ĐÃ CÂN LÕI: '+session.weight+' '+data.unit+'\nẢnh và số cân đang được giữ chờ. Quét hoặc nhập MÃ NHẬP SP để hoàn tất.','warn');captureQr.focus();captureQr.select()}
  else{const localNotice=appStatus&&appStatus.weight_engine!=='gemini'?'\nBACKEND ĐANG DÙNG OCR LOCAL; ảnh này chưa được gửi Gemini. Bật ROLL_SCALE_WEIGHT_ENGINE=gemini rồi khởi động lại.':'';status(captureStatus,'CHƯA ĐỌC ĐƯỢC CÂN LÕI. '+issues.join('; ')+'\n'+(data.weight_raw||'Không đủ kết quả nhận diện')+localNotice+'\nNhấn Backspace để bỏ ngay lần này.','bad')}
  }
 }catch(error){if(session.eventId!==requestEventId)return;session.setState('error','NHẬN DIỆN LỖI · cần bỏ ảnh','bad');if(session===current())status(captureStatus,'NHẬN DIỆN LỖI: '+error.message,'bad')}
}
async function discardCurrent(requireConfirmation=true){const session=current();if(!session||!session.hasUnsavedReview()||session.state==='saving'||session.state==='analyzing')return;if(requireConfirmation&&!confirm('Bỏ ảnh chưa lưu của '+session.stationId+'?'))return;const eventId=session.eventId;try{await api('/api/session/discard',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({station_id:session.stationId,event_id:eventId})});if(session.eventId===eventId){clearReview(session);captureQr.value='';weight.value='';status(captureStatus,'Đã bỏ ảnh chưa lưu của '+session.stationId+'.','warn')}}catch(error){status(captureStatus,'Không bỏ được ảnh: '+error.message,'bad')}}
async function saveCapture(){const session=current();if(!session||session.state!=='ready'||!completionReady(session))return;persistEditor(session);const value=Number(session.weight),productCode=session.qr.trim();if(!productCode){status(captureStatus,'Mã nhập SP đang trống.','bad');return}if(!Number.isFinite(value)||value<0){status(captureStatus,'Số cân lõi không hợp lệ.','bad');return}const requestEventId=session.eventId,savedStationIndex=stations.indexOf(session);session.setState('saving','ĐANG GỬI TRỌN BỘ · '+requestEventId,'');status(captureStatus,'Đang gửi cùng một event: mã SP + cân lõi + ẢNH TL LÕI lên Supabase…','warn');try{const data=await api('/api/capture',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({qr_code:productCode,weight:value,unit:session.unit,image:session.capturedImage,vision_confirmed:value===Number(session.analyzedWeight),weight_raw:session.analyzedRaw+'; PRODUCT_ENTRY_CODE='+productCode+'; HUMAN_CONFIRMED='+value,event_id:requestEventId,analysis_id:session.analysisId,station_id:session.stationId,camera_id:session.cameraId,frame_sha256:session.frameSha256})});if(session.eventId!==requestEventId||data.event_id!==requestEventId)return;const stillSelected=session===current();captureCount+=1;session.captureCount+=1;session.setState('saved','ĐÃ LƯU · '+data.event_id,'ready');prepareNextCapture('',session);if(stillSelected){const cloudEnabled=Boolean(appStatus&&appStatus.sync_enabled),cloudSynced=!cloudEnabled||data.sync_status==='synced',cloud=cloudEnabled?(cloudSynced?'SUPABASE ĐÃ XÁC NHẬN đủ mã SP + cân lõi + ẢNH TL LÕI.':'ĐÃ LƯU LOCAL; Supabase chưa xác nhận và sẽ tự thử lại. '+(data.sync_error||'')):'Chưa cấu hình Supabase; hiện chỉ lưu cục bộ.';status(captureStatus,'ĐÃ HOÀN TẤT LẦN '+captureCount+'\nID: '+data.event_id+'\nMÃ NHẬP SP: '+data.qr_code+'\nCÂN LÕI: '+data.weight+' '+data.unit+'\n'+cloud,cloudSynced?'ok':'warn');if($('autoAdvance').checked&&stations.length>1){const next=(savedStationIndex+1)%stations.length;selectStation(next)}}}catch(error){if(session.eventId!==requestEventId)return;session.setState('ready','SẴN SÀNG THỬ LƯU LẠI · '+requestEventId,'bad');if(session===current())status(captureStatus,'KHÔNG LƯU: '+error.message,'bad')}}

function fileData(file){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error('Không đọc được tệp ảnh'));reader.readAsDataURL(file)})}
async function loadCaptureFile(file){if(!file)return;const session=current();if(session.hasUnsavedReview()){status(captureStatus,'Hãy lưu hoặc bỏ ảnh đang xem trước khi chọn tệp khác.','bad');return}try{const data=await fileData(file);session.preview.onload=()=>{showPreview(session);session.setState('idle','ẢNH TỆP · '+file.name);if(session===current())analyzeCurrent()};session.preview.src=data}catch(error){status(captureStatus,error.message,'bad')}}
function loadDemo(){const session=current();if(!session||session.hasUnsavedReview())return;session.preview.onload=()=>{showPreview(session);session.setState('idle','ẢNH DEMO KHO');if(session===current()){status(captureStatus,'Đã nạp ảnh demo kho. Đang tự đọc QR + cân…','warn');analyzeCurrent()}};session.preview.src='/demo.jpg?t='+Date.now()}
async function saveFactorySample(){const session=current();if(!session||!session.capturedImage||!session.lastAnalysis)return;persistEditor(session);const expectedWeight=Number(session.weight);if(!Number.isFinite(expectedWeight)){status(captureStatus,'Hãy điền đúng số cân trước khi lưu mẫu.','bad');return}try{const data=await api('/api/factory-sample',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({image:session.capturedImage,qr_roi:session.lastAnalysis.qr_roi||'',predicted_qr_code:session.lastAnalysis.qr_code||'',predicted_weight:session.lastAnalysis.weight,expected_qr_code:session.qr.trim(),expected_weight:expectedWeight,unit:session.unit,recognition_ok:Boolean(session.lastAnalysis.weight_found),qr_decoder:session.lastAnalysis.qr_decoder||'',ocr_confidence:session.lastAnalysis.confidence,station_id:session.stationId,camera_id:session.cameraId,event_id:session.eventId})});status(captureStatus,'ĐÃ LƯU MẪU XƯỞNG '+data.sample_id,'ok')}catch(error){status(captureStatus,'KHÔNG LƯU ĐƯỢC MẪU: '+error.message,'bad')}}

let pendingPollTimer=null;
function pollPendingSessions(){clearTimeout(pendingPollTimer);if(!stations.some(session=>session.hydratedPending&&session.state==='analyzing'))return;pendingPollTimer=setTimeout(async()=>{try{const latest=await api('/api/status');for(const session of stations.filter(item=>item.hydratedPending)){const remote=(latest.stations||[]).find(item=>item.station_id===session.stationId);if(!remote||!remote.event_id){clearReview(session);continue}if(remote.event_id!==session.eventId)continue;session.state=remote.state||session.state;session.analysisId=remote.analysis_id||session.analysisId;session.frameSha256=remote.frame_sha256||session.frameSha256;if(session.state!=='analyzing'){session.stateElement.textContent='PHIÊN CHƯA LƯU TỪ TRƯỚC · bấm Bỏ lần đang xem';session.stateElement.className='station-state bad';if(session===current())status(captureStatus,'Phiên '+session.eventId+' đã '+session.state+'. Bấm Bỏ lần đang xem để tiếp tục.','warn')}}renderControls()}catch{}pollPendingSessions()},1500)}
async function loadStatus(){try{appStatus=await api('/api/status');buildStations(appStatus.stations||[{index:1,station_id:'station-01',camera_id:'camera-01'}]);$('gatewayIdentity').textContent='gateway: '+appStatus.gateway_id+' · '+appStatus.station_count+' trạm';$('autoAdvance').checked=Boolean(appStatus.auto_advance);const engine=appStatus.weight_engine||'local',primary=engine==='gemini';$('recognitionProfileOption').hidden=!primary;recognitionProfile.disabled=!primary;$('weightModeBadge').textContent='CÂN LÕI → MÃ SP → '+(appStatus.sync_enabled?'SUPABASE':'LƯU LOCAL');const yolo=$('yoloBadge');yolo.textContent=appStatus.yolo_enabled?'QR LOCAL: BẬT · '+appStatus.yolo_mode.toUpperCase():'QR LOCAL: TẮT';yolo.className='mode '+(appStatus.yolo_enabled?'ai':'off');const ocr=$('ocrBadge');ocr.textContent=primary?'PADDLE: KHÔNG CHẠY':'OCR LOCAL: ĐANG CHẠY';ocr.className='mode '+(primary?'off':'ai');const gemini=$('geminiBadge'),geminiOn=Boolean(appStatus.gemini&&appStatus.gemini.enabled);gemini.textContent=primary?'GEMINI: FLASH-LITE / PRO':geminiOn?'GEMINI FALLBACK: BẬT':'GEMINI: CHƯA BẬT';gemini.className='mode '+(geminiOn?'ai':'off');const sync=$('syncBadge');sync.textContent=appStatus.sync_enabled?'ĐỒNG BỘ SUPABASE: BẬT':'CLOUD: CHỈ LƯU OFFLINE';sync.className='mode '+(appStatus.sync_enabled?'ai':'off');$('saveBtn').childNodes[0].nodeValue=appStatus.sync_enabled?'Xác nhận mã SP và gửi Supabase ':'Xác nhận mã SP và lưu cục bộ ';await refreshCameraDevices(false);const mapped=stations.filter(session=>session.deviceId&&!session.hasUnsavedReview());for(const session of mapped)await openStationCamera(session);const pending=stations.filter(session=>session.hasUnsavedReview());if(pending.length){status(captureStatus,'Có '+pending.length+' phiên backend chưa kết thúc/lưu. Chọn từng trạm để chờ hoặc bỏ phiên.','warn');pollPendingSessions()}else if(!mapped.length&&stations.length===1)loadDemo();else if(!mapped.length)status(captureStatus,'Chọn camera cho từng trạm, rồi bấm Mở camera đã gán.','warn')}catch(error){status(captureStatus,'Không tải được cấu hình trạm: '+error.message,'bad')}}

$('cameraBtn').onclick=()=>openStationCamera(current());$('refreshCamerasBtn').onclick=refreshCameraDevices;$('openAllBtn').onclick=openAllMapped;$('captureFileBtn').onclick=()=>$('captureFile').click();$('captureFile').onchange=event=>loadCaptureFile(event.target.files[0]);$('demoBtn').onclick=loadDemo;$('analyzeBtn').onclick=analyzeCurrent;$('discardBtn').onclick=discardCurrent;$('factoryBtn').onclick=saveFactorySample;$('saveBtn').onclick=saveCapture;captureQr.oninput=()=>{const session=current();if(session){session.qr=captureQr.value;refreshCompletionState(session)}};weight.oninput=()=>{const session=current();if(session){session.weight=weight.value;refreshCompletionState(session)}};unit.onchange=()=>{if(current())current().unit=unit.value};recognitionProfile.onchange=()=>{try{localStorage.setItem(RECOGNITION_PROFILE_KEY,recognitionProfile.value)}catch{}};
document.addEventListener('keydown',event=>{if(event.repeat||event.ctrlKey||event.altKey||event.metaKey)return;const typing=['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName),captureEditor=event.target===captureQr||event.target===weight;if(['1','2','3'].includes(event.key)&&!typing){const index=Number(event.key)-1;if(index<stations.length){event.preventDefault();selectStation(index)}return}if(event.key==='Backspace'&&!typing){event.preventDefault();discardCurrent(false);return}if(typing&&!captureEditor)return;if(event.code==='Space'){event.preventDefault();analyzeCurrent();return}if(event.key==='Enter'){event.preventDefault();saveCapture()}});
navigator.mediaDevices&&navigator.mediaDevices.addEventListener&&navigator.mediaDevices.addEventListener('devicechange',async()=>{await refreshCameraDevices();stations.forEach(session=>{if(session.deviceId&&!session.stream)scheduleReconnect(session,session.deviceId)})});window.addEventListener('resize',()=>stations.forEach(updateBoxes));window.addEventListener('beforeunload',()=>stations.forEach(session=>session.stop()));loadStatus();
</script></body></html>"""


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
    if weight_engine in {"hybrid", "gemini"}:
        gemini_api_key = os.environ.get("ROLL_SCALE_GEMINI_API_KEY", "").strip()
        if not gemini_api_key:
            raise ValueError(
                f"weight_engine={weight_engine} cần ROLL_SCALE_GEMINI_API_KEY "
                "trong biến môi trường"
            )
        gemini_reader = GeminiWeightReader(
            gemini_api_key,
            model=args.gemini_model,
            timeout_seconds=args.gemini_timeout,
            thinking_level="minimal",
        )
        gemini_accurate_reader = GeminiWeightReader(
            gemini_api_key,
            model=args.gemini_accurate_model,
            timeout_seconds=args.gemini_accurate_timeout,
            thinking_level="medium",
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
                self.send_json(200, {"ok": True})
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
                        "quality_settings": service.quality_settings,
                        **identity_status,
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
            self.send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            if not self.require_authorization():
                return
            try:
                payload = self.read_json()
                if self.path == "/api/session/discard":
                    discarded = service.sessions.discard(
                        str(payload.get("station_id", "")),
                        event_id=str(payload["event_id"]) if payload.get("event_id") else None,
                    )
                    self.send_json(200, {"ok": True, "discarded": discarded})
                    return
                frame = decode_image(str(payload.get("image", "")))
                if self.path == "/api/decode":
                    self.send_json(200, service.decode_qr(frame))
                    return
                if self.path == "/api/analyze":
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
                    result = service.analyze(
                        frame,
                        str(payload.get("roi", "")),
                        str(payload.get("unit", "kg")),
                        event_id=str(payload["event_id"]) if payload.get("event_id") else None,
                        station_id=str(payload["station_id"]) if payload.get("station_id") else None,
                        camera_id=str(payload["camera_id"]) if payload.get("camera_id") else None,
                        weight_frames=weight_frames,
                        require_temporal=bool(payload.get("camera_capture", False)),
                        recognition_profile=str(payload.get("recognition_profile", "fast")),
                    )
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
                    result = service.capture(
                        str(payload.get("qr_code", "")),
                        weight,
                        str(payload.get("unit", "kg")),
                        frame,
                        bool(payload.get("vision_confirmed", False)),
                        str(payload.get("weight_raw", "")),
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
