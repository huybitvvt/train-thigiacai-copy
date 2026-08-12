from __future__ import annotations

import base64
import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np

from .codex_oauth import CodexOAuthClient, CodexOAuthError
from .gemini_weight import GeminiWeightSuggestion


CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_CODEX_OAUTH_MODEL = "gpt-5.5"
DEFAULT_CODEX_CLIENT_VERSION = "0.124.0"
_FIXED_WEIGHT = re.compile(r"^(?:0|[1-9]\d{0,3})\.\d{2}$")


class CodexOAuthWeightReader:
    """Read a fixed two-decimal scale display through ChatGPT device OAuth."""

    def __init__(
        self,
        oauth: CodexOAuthClient,
        *,
        model: str = DEFAULT_CODEX_OAUTH_MODEL,
        timeout_seconds: float = 60.0,
        max_image_edge: int = 1280,
        jpeg_quality: int = 86,
        client_version: str = DEFAULT_CODEX_CLIENT_VERSION,
    ) -> None:
        if not 10.0 <= float(timeout_seconds) <= 180.0:
            raise ValueError("Codex timeout must be between 10 and 180 seconds")
        self.oauth = oauth
        self.model = model.strip() or DEFAULT_CODEX_OAUTH_MODEL
        self.timeout_seconds = float(timeout_seconds)
        self.max_image_edge = int(max_image_edge)
        self.jpeg_quality = int(jpeg_quality)
        self.client_version = client_version.strip() or DEFAULT_CODEX_CLIENT_VERSION
        self._lock = threading.Lock()
        self._requests = 0
        self._successes = 0
        self._failures = 0
        self._last_latency_seconds: float | None = None
        self._last_error: str | None = None

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
            raise ValueError("Codex camera image is invalid")
        height, width = prepared.shape[:2]
        longest = max(height, width)
        if longest > self.max_image_edge:
            scale = self.max_image_edge / longest
            prepared = cv2.resize(
                prepared,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            prepared,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise ValueError("Cannot encode Codex ROI image")
        return encoded.tobytes()

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        return f"{type(exc).__name__}: {str(exc).strip()}"[:300]

    def status(self, *, refresh: bool = False) -> dict[str, object]:
        del refresh
        result = {
            "enabled": True,
            "installed": True,
            "model": self.model,
            **self.oauth.status(),
        }
        with self._lock:
            result.update(
                {
                    "requests": self._requests,
                    "successes": self._successes,
                    "failures": self._failures,
                    "last_latency_seconds": self._last_latency_seconds,
                    "last_error": self._last_error,
                }
            )
        return result

    def start_device_login(self) -> dict[str, object]:
        return self.oauth.start_device_login()

    def poll_device_login(self, session_id: str) -> dict[str, object]:
        return self.oauth.poll_device_login(session_id)

    @staticmethod
    def _extract_text(event: dict[str, object]) -> str:
        response = event.get("response")
        if not isinstance(response, dict):
            return ""
        chunks: list[str] = []
        output = response.get("output")
        if not isinstance(output, list):
            return ""
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(str(part["text"]))
        return "".join(chunks)

    def _request(self, frames: list[np.ndarray]) -> str:
        access_token, account_id = self.oauth.access()
        content: list[dict[str, object]] = [
            {
                "type": "input_text",
                "text": (
                    "Read only the numeric LED scale display in these images. "
                    "The scale always has exactly two decimal places. "
                    "Return ONLY one JSON object: "
                    '{"weight_readable":true,"weight_digits":"1304",'
                    '"all_frames_agree":true}. '
                    "weight_digits must contain digits only with the decimal separator removed. "
                    "Use null when unreadable. Do not read QR codes or product labels."
                ),
            }
        ]
        for frame in frames:
            encoded = base64.b64encode(self._jpeg(frame)).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded}",
                }
            )
        request_body = {
            "model": self.model,
            "instructions": (
                "You are a deterministic industrial scale OCR. "
                "Follow the requested JSON shape exactly and output no prose."
            ),
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "low"},
            "stream": True,
            "store": False,
        }
        session_id = str(uuid.uuid4())
        request = urllib.request.Request(
            CODEX_RESPONSES_URL,
            data=json.dumps(request_body, separators=(",", ":")).encode("utf-8"),
            headers={
                "authorization": f"Bearer {access_token}",
                "chatgpt-account-id": account_id,
                "content-type": "application/json",
                "accept": "text/event-stream",
                "openai-beta": "responses=experimental",
                "originator": "codex_cli_rs",
                "session_id": str(uuid.uuid4()),
                "version": self.client_version,
                "user-agent": f"codex_cli_rs/{self.client_version}",
            },
            method="POST",
        )
        deltas: list[str] = []
        completed_text = ""
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("type") == "response.output_text.delta" and isinstance(event.get("delta"), str):
                        deltas.append(str(event["delta"]))
                    elif event.get("type") == "response.output_item.done":
                        item = event.get("item")
                        if isinstance(item, dict) and item.get("type") == "message":
                            parts = item.get("content")
                            if isinstance(parts, list):
                                completed_text = "".join(
                                    str(part.get("text") or "")
                                    for part in parts
                                    if isinstance(part, dict) and part.get("type") == "output_text"
                                )
                    elif event.get("type") == "response.completed":
                        completed_text = self._extract_text(event)
                    elif event.get("type") in {"response.failed", "error"}:
                        detail = event.get("error") or event
                        raise CodexOAuthError(f"Codex response failed: {str(detail)[:300]}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise CodexOAuthError(f"Codex HTTP {exc.code}: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise CodexOAuthError(f"Không kết nối được Codex: {exc.reason}") from exc
        text = completed_text or "".join(deltas)
        if not text.strip():
            raise CodexOAuthError("Codex không trả về kết quả")
        return text.strip()

    @staticmethod
    def _json_object(text: str) -> dict[str, object]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        decoder = json.JSONDecoder()
        for index, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
        raise ValueError("Codex response is not a JSON object")

    def read(self, roi_images: list[np.ndarray], *, unit: str = "kg") -> GeminiWeightSuggestion:
        started = time.perf_counter()
        with self._lock:
            self._requests += 1
        try:
            sampled = self._sample_frames(roi_images)
            if not sampled:
                raise ValueError("Codex cần 1 hoặc ít nhất 3 frame hợp lệ")
            payload = self._json_object(self._request(sampled))
            readable = bool(payload.get("weight_readable"))
            raw_digits = str(payload.get("weight_digits") or "").strip()
            digits = re.sub(r"\D", "", raw_digits)
            reading = f"{digits[:-2]}.{digits[-2:]}" if 3 <= len(digits) <= 6 else ""
            agreement = len(sampled) == 1 or bool(payload.get("all_frames_agree"))
            valid = readable and agreement and _FIXED_WEIGHT.fullmatch(reading) is not None
            value = float(reading) if valid else None
            if value is not None and (not math.isfinite(value) or value < 0):
                value = None
                valid = False
            latency = time.perf_counter() - started
            raw = f"CODEX-OAUTH:{reading if valid else 'weight-unreadable'}; agree={agreement}"
            with self._lock:
                self._last_latency_seconds = latency
                self._last_error = None
                self._successes += int(valid)
                self._failures += int(not valid)
            return GeminiWeightSuggestion(value, unit, valid, agreement, raw, latency)
        except Exception as exc:
            latency = time.perf_counter() - started
            error = self._safe_error(exc)
            with self._lock:
                self._failures += 1
                self._last_latency_seconds = latency
                self._last_error = error
            return GeminiWeightSuggestion(None, unit, False, False, f"CODEX ERROR: {error}", latency)

    def close(self) -> None:
        return None
