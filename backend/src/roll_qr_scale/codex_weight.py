from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .gemini_weight import GeminiWeightSuggestion


DEFAULT_CODEX_TIMEOUT_SECONDS = 60.0
DEFAULT_CODEX_MAX_IMAGE_EDGE = 1280
DEFAULT_CODEX_JPEG_QUALITY = 86
_FIXED_WEIGHT = re.compile(r"^(?:0|[1-9]\d{0,3})\.\d{2}$")
_CODEX_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "weight_readable": {"type": "boolean"},
        "weight_digits": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        },
        "all_frames_agree": {"type": "boolean"},
    },
    "required": ["weight_readable", "weight_digits", "all_frames_agree"],
    "additionalProperties": False,
}


class CodexWeightReader:
    """Read a scale display through the locally installed, ChatGPT-authenticated Codex CLI."""

    def __init__(
        self,
        command: str = "codex",
        *,
        model: str = "",
        timeout_seconds: float = DEFAULT_CODEX_TIMEOUT_SECONDS,
        max_image_edge: int = DEFAULT_CODEX_MAX_IMAGE_EDGE,
        jpeg_quality: int = DEFAULT_CODEX_JPEG_QUALITY,
    ) -> None:
        configured_command = command.strip()
        if not configured_command:
            raise ValueError("Codex command is required")
        if not 10.0 <= float(timeout_seconds) <= 180.0:
            raise ValueError("Codex timeout must be between 10 and 180 seconds")
        if not 512 <= int(max_image_edge) <= 2048:
            raise ValueError("Codex max image edge must be between 512 and 2048")
        if not 70 <= int(jpeg_quality) <= 95:
            raise ValueError("Codex JPEG quality must be between 70 and 95")
        self.command = configured_command
        self.model = model.strip()
        self.timeout_seconds = float(timeout_seconds)
        self.max_image_edge = int(max_image_edge)
        self.jpeg_quality = int(jpeg_quality)
        self._lock = threading.Lock()
        self._requests = 0
        self._successes = 0
        self._failures = 0
        self._last_latency_seconds: float | None = None
        self._last_error: str | None = None
        self._status_cache: tuple[float, dict[str, object]] | None = None
        self._login_process: subprocess.Popen[bytes] | None = None

    def _executable(self) -> str | None:
        candidate = Path(self.command).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        return shutil.which(self.command)

    @staticmethod
    def _command_for(executable: str, arguments: list[str]) -> list[str]:
        suffix = Path(executable).suffix.lower()
        if os.name == "nt" and suffix in {".cmd", ".bat"}:
            return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", executable, *arguments]
        if os.name == "nt" and suffix == ".ps1":
            return [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                executable,
                *arguments,
            ]
        return [executable, *arguments]

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

    def _run(self, arguments: list[str], *, cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
        executable = self._executable()
        if executable is None:
            raise FileNotFoundError("Không tìm thấy Codex CLI trên máy backend")
        return subprocess.run(
            self._command_for(executable, arguments),
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )

    def status(self, *, refresh: bool = False) -> dict[str, object]:
        with self._lock:
            if (
                not refresh
                and self._status_cache is not None
                and time.monotonic() - self._status_cache[0] < 5.0
            ):
                return dict(self._status_cache[1])
        executable = self._executable()
        if executable is None:
            result: dict[str, object] = {
                "enabled": True,
                "installed": False,
                "authenticated": False,
                "available": False,
                "model": self.model or None,
                "message": "Chưa cài Codex CLI trên máy backend",
            }
        else:
            try:
                completed = self._run(
                    ["login", "status"],
                    cwd=Path(tempfile.gettempdir()),
                    timeout=min(10.0, self.timeout_seconds),
                )
                output = f"{completed.stdout}\n{completed.stderr}".strip()
                authenticated = completed.returncode == 0 and "logged in" in output.lower()
                result = {
                    "enabled": True,
                    "installed": True,
                    "authenticated": authenticated,
                    "available": authenticated,
                    "model": self.model or None,
                    "message": (
                        output[:240]
                        if output
                        else ("Đã đăng nhập Codex" if authenticated else "Codex chưa đăng nhập")
                    ),
                }
            except Exception as exc:
                result = {
                    "enabled": True,
                    "installed": True,
                    "authenticated": False,
                    "available": False,
                    "model": self.model or None,
                    "message": self._safe_error(exc),
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
            self._status_cache = (time.monotonic(), dict(result))
        return result

    def start_device_login(self) -> dict[str, object]:
        executable = self._executable()
        if executable is None:
            raise RuntimeError("Chưa cài Codex CLI trên máy backend")
        current = self.status(refresh=True)
        if current.get("authenticated"):
            return {"started": False, "authenticated": True, "message": "Codex đã đăng nhập"}
        if os.name != "nt":
            return {
                "started": False,
                "authenticated": False,
                "message": "Chạy `codex login --device-auth` trong terminal của máy backend",
            }
        with self._lock:
            if self._login_process is not None and self._login_process.poll() is None:
                return {
                    "started": False,
                    "authenticated": False,
                    "message": "Cửa sổ đăng nhập Codex đang mở",
                }
            creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            self._login_process = subprocess.Popen(
                self._command_for(executable, ["login", "--device-auth"]),
                cwd=tempfile.gettempdir(),
                creationflags=creation_flags,
            )
            self._status_cache = None
        return {
            "started": True,
            "authenticated": False,
            "message": "Đã mở cửa sổ đăng nhập Codex; nhập mã thiết bị rồi quay lại kiểm tra",
        }

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
                "CODEX: need 1 still image or 3 camera frames",
                0.0,
            )
        started = time.perf_counter()
        with self._lock:
            self._requests += 1
        try:
            if not self.status().get("available"):
                raise RuntimeError("Codex CLI chưa đăng nhập bằng ChatGPT trên máy backend")
            with tempfile.TemporaryDirectory(prefix="roll-scale-codex-") as temp_name:
                temp_dir = Path(temp_name)
                image_paths: list[Path] = []
                for index, frame in enumerate(sampled, start=1):
                    path = temp_dir / f"scale-{index}.jpg"
                    path.write_bytes(self._jpeg(frame))
                    image_paths.append(path)
                schema_path = temp_dir / "response-schema.json"
                output_path = temp_dir / "result.json"
                schema_path.write_text(
                    json.dumps(_CODEX_RESPONSE_SCHEMA, ensure_ascii=False),
                    encoding="utf-8",
                )
                prompt = (
                    "Chỉ nhận diện số cân trong ảnh, không chạy lệnh và không đọc file khác. "
                    "Tìm màn hình LED đỏ của cân và chỉ đọc hàng số gross trên cùng. "
                    "Bỏ qua QR, nhãn, bàn phím, hàng tare/net và mọi số khác. "
                    "Trả weight_digits không có dấu thập phân; cân luôn có đúng hai số lẻ: "
                    "7.02 thành 702, 13.04 thành 1304. Chấm LED nhỏ là dấu thập phân, "
                    "không phải số 0. Không đoán nét bị mất. Nếu không chắc, đặt "
                    "weight_readable=false và weight_digits=null. Với ba ảnh, "
                    "all_frames_agree chỉ true khi cả ba hiển thị cùng một số."
                )
                arguments = [
                    "exec",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--color",
                    "never",
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                ]
                if self.model:
                    arguments.extend(("--model", self.model))
                for path in image_paths:
                    arguments.extend(("--image", str(path)))
                # `--image` accepts one or more paths, so terminate option parsing
                # before the positional prompt or the CLI treats the prompt as a file.
                arguments.extend(("--", prompt))
                completed = self._run(arguments, cwd=temp_dir, timeout=self.timeout_seconds)
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout or "Codex exec failed").strip()
                    raise RuntimeError(detail[-500:])
                if not output_path.is_file():
                    raise RuntimeError("Codex không tạo kết quả JSON")
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            readable = bool(payload.get("weight_readable"))
            digits = str(payload.get("weight_digits") or "").strip()
            reading = (
                f"{digits[:-2]}.{digits[-2:]}"
                if digits.isdigit() and 3 <= len(digits) <= 6
                else ""
            )
            agreement = len(sampled) == 1 or bool(payload.get("all_frames_agree"))
            valid = readable and agreement and _FIXED_WEIGHT.fullmatch(reading) is not None
            value = float(reading) if valid else None
            if value is not None and (not math.isfinite(value) or value < 0):
                value = None
                valid = False
            latency = time.perf_counter() - started
            raw = (
                f"CODEX:{reading if valid else 'weight-unreadable'}; "
                f"agree={agreement}; auth=ChatGPT"
            )
            with self._lock:
                self._last_latency_seconds = latency
                self._last_error = None
                if valid:
                    self._successes += 1
                else:
                    self._failures += 1
                self._status_cache = None
            return GeminiWeightSuggestion(
                value,
                unit,
                valid,
                agreement,
                raw,
                latency,
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            error = self._safe_error(exc)
            with self._lock:
                self._failures += 1
                self._last_latency_seconds = latency
                self._last_error = error
                self._status_cache = None
            return GeminiWeightSuggestion(
                None,
                unit,
                False,
                False,
                f"CODEX ERROR: {error}",
                latency,
            )

    def close(self) -> None:
        return None
