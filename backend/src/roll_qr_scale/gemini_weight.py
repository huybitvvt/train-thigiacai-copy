from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_ACCURATE_MODEL = "gemini-3.1-pro-preview"
DEFAULT_GEMINI_TIMEOUT_SECONDS = 10.0
DEFAULT_GEMINI_ACCURATE_TIMEOUT_SECONDS = 30.0
DEFAULT_GEMINI_MAX_IMAGE_EDGE = 1280
DEFAULT_GEMINI_JPEG_QUALITY = 86
DEFAULT_GEMINI_MEDIA_RESOLUTION = "medium"
_FIXED_WEIGHT = re.compile(r"^(?:0|[1-9]\d{0,3})\.\d{2}$")


class _GeminiScalePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight_readable: bool
    weight_digits: str | None
    qr_readable: bool
    qr_code: str | None
    all_frames_agree: bool


class _GeminiPanelPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readable: bool
    value: str | None


_GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "weight_readable": {"type": "boolean"},
        "weight_digits": {"type": "string", "nullable": True},
        "qr_readable": {"type": "boolean"},
        "qr_code": {"type": "string", "nullable": True},
        "all_frames_agree": {"type": "boolean"},
    },
    "required": [
        "weight_readable",
        "weight_digits",
        "qr_readable",
        "qr_code",
        "all_frames_agree",
    ],
}

_GEMINI_PANEL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "readable": {"type": "boolean"},
        "value": {"type": "string", "nullable": True},
    },
    "required": ["readable", "value"],
}


@dataclass(frozen=True)
class GeminiWeightSuggestion:
    value: float | None
    unit: str
    readable: bool
    all_frames_agree: bool
    raw: str
    latency_seconds: float
    qr_code: str | None = None
    qr_readable: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0


class GeminiWeightReader:
    """Read QR content and scale weight from full factory-camera frames."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout_seconds: float = DEFAULT_GEMINI_TIMEOUT_SECONDS,
        thinking_level: str = "minimal",
        max_image_edge: int = DEFAULT_GEMINI_MAX_IMAGE_EDGE,
        jpeg_quality: int = DEFAULT_GEMINI_JPEG_QUALITY,
        media_resolution: str = DEFAULT_GEMINI_MEDIA_RESOLUTION,
        include_qr: bool = True,
        client: object | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Gemini API key is required")
        if not model.strip():
            raise ValueError("Gemini model is required")
        if not 10.0 <= timeout_seconds <= 30.0:
            raise ValueError("Gemini timeout must be between 10 and 30 seconds")
        if thinking_level not in {"minimal", "low", "medium", "high"}:
            raise ValueError("Gemini thinking level is invalid")
        if not 512 <= int(max_image_edge) <= 2048:
            raise ValueError("Gemini max image edge must be between 512 and 2048")
        if not 70 <= int(jpeg_quality) <= 95:
            raise ValueError("Gemini JPEG quality must be between 70 and 95")
        if media_resolution not in {"low", "medium", "high"}:
            raise ValueError("Gemini media resolution must be low, medium or high")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout_seconds = float(timeout_seconds)
        self.thinking_level = thinking_level
        self.max_image_edge = int(max_image_edge)
        self.jpeg_quality = int(jpeg_quality)
        self.media_resolution = media_resolution
        self.include_qr = bool(include_qr)
        if client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise RuntimeError(
                    "Gemini fallback requires google-genai==2.16.0"
                ) from exc
            client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=round(self.timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        self.client = client
        self._lock = threading.Lock()
        self._requests = 0
        self._successes = 0
        self._failures = 0
        self._last_latency_seconds: float | None = None
        self._last_error: str | None = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._thinking_tokens = 0

    @staticmethod
    def _sample_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
        usable = [frame for frame in frames if isinstance(frame, np.ndarray) and frame.size]
        if len(usable) == 1:
            return usable
        if len(usable) < 3:
            return []
        if len(usable) == 3:
            return usable
        middle = len(usable) // 2
        return [usable[0], usable[middle], usable[-1]]

    def _jpeg(self, image: np.ndarray) -> bytes:
        if image.ndim == 2:
            prepared = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] >= 3:
            prepared = image[:, :, :3]
        else:
            raise ValueError("Gemini camera image is invalid")
        height, width = prepared.shape[:2]
        longest = max(height, width)
        if longest > self.max_image_edge:
            scale = self.max_image_edge / longest
            prepared = cv2.resize(
                prepared,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            height, width = prepared.shape[:2]
        if height < 256:
            scale = min(
                256 / max(1, height),
                self.max_image_edge / max(1, width, height),
            )
            if scale > 1:
                prepared = cv2.resize(
                    prepared,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_CUBIC,
                )
        ok, encoded = cv2.imencode(
            ".jpg",
            prepared,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise ValueError("Cannot encode Gemini ROI image")
        return encoded.tobytes()

    @staticmethod
    def _panel_evidence(image: np.ndarray) -> np.ndarray:
        """Place the crop beside a view containing only actively lit LED colors."""
        prepared = (
            image[:, :, :3]
            if image.ndim == 3
            else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        )
        blue, green, red = cv2.split(prepared)
        red_i16, green_i16, blue_i16 = (
            red.astype(np.int16),
            green.astype(np.int16),
            blue.astype(np.int16),
        )
        active_red = (red_i16 >= 105) & (
            red_i16 - np.maximum(green_i16, blue_i16) >= 28
        )
        active_green = (green_i16 >= 80) & (
            green_i16 - np.maximum(red_i16, blue_i16) >= 20
        )
        active = (active_red | active_green).astype(np.uint8) * 255
        isolated = np.zeros_like(prepared)
        isolated[active > 0] = prepared[active > 0]
        divider = np.full((prepared.shape[0], 3, 3), 255, dtype=np.uint8)
        return np.concatenate((prepared, divider, isolated), axis=1)

    @staticmethod
    def _payload(response: object) -> _GeminiScalePayload:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, _GeminiScalePayload):
            return parsed
        if isinstance(parsed, dict):
            return _GeminiScalePayload.model_validate(parsed)
        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            raise ValueError("Gemini returned an empty response")
        return _GeminiScalePayload.model_validate(json.loads(text))

    def _safe_error(self, exc: Exception) -> str:
        message = str(exc).replace(self.api_key, "[redacted]").strip()
        return f"{type(exc).__name__}: {message}"[:240]

    def read(
        self,
        frames: list[np.ndarray],
        *,
        unit: str = "kg",
    ) -> GeminiWeightSuggestion:
        sampled = self._sample_frames(frames)
        if len(sampled) not in {1, 3}:
            return GeminiWeightSuggestion(
                None,
                unit,
                False,
                False,
                "GEMINI FULL: need 1 still image or 3 camera frames",
                0.0,
            )
        started = time.perf_counter()
        with self._lock:
            self._requests += 1
        try:
            from google.genai import types

            image_description = (
                "one factory-camera image or a focused display crop"
                if len(sampled) == 1
                else "three chronological images from the same factory camera"
            )
            qr_instruction = (
                "Also find the product QR label and decode its encoded content exactly "
                "into qr_code; do not substitute nearby printed text. "
                if self.include_qr
                else "Do not read or infer a product code; set qr_readable=false and qr_code=null. "
            )
            prompt = (
                f"These are {image_description}. Inspect the entire supplied image. "
                "Find the electronic scale display and read "
                "only the illuminated digit glyphs in its top gross-weight row, left to "
                "right. Return weight_digits without a decimal point. A small round "
                "decimal LED is punctuation, never a zero digit. The scale always has "
                "two decimal places: 7.02 means weight_digits=\"702\" and 13.04 means "
                "weight_digits=\"1304\". "
                + qr_instruction
                + "Treat the lower tare/net rows, keypad, labels, dates and other "
                "numbers as irrelevant. If an item cannot be read exactly, set its "
                "readable flag false and its value null. For three images, set "
                "all_frames_agree=false if any visible weight or QR content conflicts. "
                "For one image, set all_frames_agree=true. Never guess missing segments "
                "or QR content."
            )
            contents: list[object] = [prompt]
            contents.extend(
                types.Part.from_bytes(
                    data=self._jpeg(frame),
                    mime_type="image/jpeg",
                    media_resolution=f"MEDIA_RESOLUTION_{self.media_resolution.upper()}",
                )
                for frame in sampled
            )
            thinking_config = (
                types.ThinkingConfig(thinking_level=self.thinking_level)
                if self.model.startswith("gemini-3")
                else types.ThinkingConfig(thinking_budget=0)
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=112,
                    response_mime_type="application/json",
                    response_schema=_GEMINI_RESPONSE_SCHEMA,
                    thinking_config=thinking_config,
                ),
            )
            payload = self._payload(response)
            latency = time.perf_counter() - started
            usage = getattr(response, "usage_metadata", None)
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            thinking_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
            total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
            digits = (payload.weight_digits or "").strip()
            reading = (
                f"{digits[:-2]}.{digits[-2:]}"
                if digits.isdigit() and 3 <= len(digits) <= 6
                else ""
            )
            agreement = len(sampled) == 1 or bool(payload.all_frames_agree)
            valid = (
                payload.weight_readable
                and agreement
                and _FIXED_WEIGHT.fullmatch(reading) is not None
            )
            value = float(reading) if valid else None
            if value is not None and (not math.isfinite(value) or value < 0):
                value = None
                valid = False
            qr_code = (payload.qr_code or "").strip()
            qr_valid = (
                self.include_qr
                and payload.qr_readable
                and agreement
                and 1 <= len(qr_code) <= 512
                and all(ord(character) >= 32 for character in qr_code)
            )
            if not qr_valid:
                qr_code = ""
            raw = (
                f"GEMINI_FULL:{reading if valid else 'weight-unreadable'}@{self.model}; "
                f"qr={'readable' if qr_valid else 'unreadable'}; agree={agreement}; "
                f"tokens={input_tokens}+{output_tokens}+{thinking_tokens}"
            )
            with self._lock:
                self._last_latency_seconds = latency
                self._last_error = None
                self._input_tokens += input_tokens
                self._output_tokens += output_tokens
                self._thinking_tokens += thinking_tokens
                if valid:
                    self._successes += 1
                else:
                    self._failures += 1
            return GeminiWeightSuggestion(
                value,
                unit,
                valid,
                agreement,
                raw,
                latency,
                qr_code=qr_code or None,
                qr_readable=qr_valid,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                total_tokens=total_tokens,
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            error = self._safe_error(exc)
            with self._lock:
                self._failures += 1
                self._last_latency_seconds = latency
                self._last_error = error
            return GeminiWeightSuggestion(
                None,
                unit,
                False,
                False,
                f"GEMINI ERROR: {error}",
                latency,
            )

    def read_panel_regions(
        self,
        regions: list[tuple[str, np.ndarray]],
    ) -> dict[str, object]:
        """Read independently configured controller displays in one Gemini request."""
        if not 1 <= len(regions) <= 24:
            raise ValueError("Panel scan requires between 1 and 24 configured regions")
        labels: list[str] = []
        label_keys: set[str] = set()
        for label, image in regions:
            clean_label = str(label).strip()
            if not clean_label or len(clean_label) > 80:
                raise ValueError("Panel region label is invalid")
            if clean_label.casefold() in label_keys:
                raise ValueError("Panel region labels must be unique")
            if not isinstance(image, np.ndarray) or image.size == 0:
                raise ValueError(f"Panel region {clean_label} is empty")
            labels.append(clean_label)
            label_keys.add(clean_label.casefold())

        started = time.perf_counter()
        with self._lock:
            self._requests += 1
        try:
            from google.genai import types

            region_keys = [f"region_{index:02d}" for index in range(1, len(labels) + 1)]
            schema = {
                "type": "object",
                "properties": {
                    key: _GEMINI_PANEL_RESPONSE_SCHEMA for key in region_keys
                },
                "required": region_keys,
            }
            prompt = (
                "Each following image is one separately cropped industrial digital "
                "controller value row. The images appear in this exact order: "
                + ", ".join(
                    f"{key}={json.dumps(label, ensure_ascii=False)}"
                    for key, label in zip(region_keys, labels, strict=True)
                )
                + ". Each image has the original crop on the left and, after a white "
                "divider, an active-color isolation on the right. For each named "
                "display, transcribe only digit segments that are "
                "actively illuminated red or green. Ignore every gray/unlit ghost "
                "segment, printed label, setpoint row, status lamp, button and keypad. "
                "Preserve the visible decimal point and minus sign exactly; never add "
                "a unit or infer a decimal position. Each crop must contain exactly one "
                "active numeric row; if it contains multiple active numeric rows or any "
                "active digit cannot be read exactly, set readable=false and value=null. "
                "Never guess. Return one object for every supplied region key."
            )
            contents: list[object] = [prompt]
            contents.extend(
                types.Part.from_bytes(
                    data=self._jpeg(self._panel_evidence(image)),
                    mime_type="image/jpeg",
                    media_resolution=f"MEDIA_RESOLUTION_{self.media_resolution.upper()}",
                )
                for _, image in regions
            )
            thinking_config = (
                types.ThinkingConfig(thinking_level=self.thinking_level)
                if self.model.startswith("gemini-3")
                else types.ThinkingConfig(thinking_budget=0)
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=max(160, len(regions) * 48),
                    response_mime_type="application/json",
                    response_schema=schema,
                    thinking_config=thinking_config,
                ),
            )
            parsed = getattr(response, "parsed", None)
            if not isinstance(parsed, dict):
                text = str(getattr(response, "text", "") or "").strip()
                parsed = json.loads(text) if text else None
            if not isinstance(parsed, dict):
                raise ValueError("Gemini returned an invalid panel response")
            readings: list[dict[str, object]] = []
            for key, label in zip(region_keys, labels, strict=True):
                try:
                    payload = _GeminiPanelPayload.model_validate(parsed.get(key))
                except Exception:
                    readings.append(
                        {"label": label, "readable": False, "value": None}
                    )
                    continue
                value = (payload.value or "").strip()
                valid = bool(
                    payload.readable
                    and re.fullmatch(r"-?\d{1,7}(?:[.,]\d{1,4})?", value)
                )
                readings.append(
                    {
                        "label": label,
                        "readable": valid,
                        "value": value.replace(",", ".") if valid else None,
                    }
                )
            latency = time.perf_counter() - started
            usage = getattr(response, "usage_metadata", None)
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            thinking_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
            total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
            with self._lock:
                self._last_latency_seconds = latency
                self._last_error = None
                self._input_tokens += input_tokens
                self._output_tokens += output_tokens
                self._thinking_tokens += thinking_tokens
                if all(item["readable"] for item in readings):
                    self._successes += 1
                else:
                    self._failures += 1
            return {
                "ok": True,
                "readings": readings,
                "latency_seconds": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "total_tokens": total_tokens,
                "model": self.model,
            }
        except Exception as exc:
            latency = time.perf_counter() - started
            error = self._safe_error(exc)
            with self._lock:
                self._failures += 1
                self._last_latency_seconds = latency
                self._last_error = error
            raise RuntimeError(error) from exc

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": True,
                "model": self.model,
                "thinking_level": self.thinking_level,
                "media_resolution": self.media_resolution,
                "max_image_edge": self.max_image_edge,
                "jpeg_quality": self.jpeg_quality,
                "include_qr": self.include_qr,
                "timeout_seconds": self.timeout_seconds,
                "requests": self._requests,
                "successes": self._successes,
                "failures": self._failures,
                "last_latency_seconds": self._last_latency_seconds,
                "last_error": self._last_error,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "thinking_tokens": self._thinking_tokens,
            }

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
