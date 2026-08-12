from __future__ import annotations

import hashlib
import threading
from typing import Callable

from .codex_oauth import EncryptedCodexTokenStore
from .gemini_weight import GeminiWeightReader


class GeminiKeyManager:
    """Validate, encrypt and hot-swap the two Gemini readers as one unit."""

    def __init__(
        self,
        store: EncryptedCodexTokenStore,
        *,
        fast_model: str,
        accurate_model: str,
        fast_timeout: float,
        accurate_timeout: float,
        initial_key: str,
        reader_factory: Callable[..., GeminiWeightReader] = GeminiWeightReader,
    ) -> None:
        self.store = store
        self.fast_model = fast_model
        self.accurate_model = accurate_model
        self.fast_timeout = float(fast_timeout)
        self.accurate_timeout = float(accurate_timeout)
        self.initial_key = initial_key.strip()
        self.reader_factory = reader_factory
        self._lock = threading.RLock()
        self._source = "environment" if self.initial_key else "none"
        self._key_id = self.key_id(self.initial_key) if self.initial_key else ""
        self._last_error = ""

    @staticmethod
    def key_id(api_key: str) -> str:
        """Return a safe short identifier without exposing any part of the key."""
        value = api_key.strip()
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10] if value else ""

    def load_key(self) -> str:
        if self.store.configured:
            try:
                saved = self.store.read()
                if saved and str(saved.get("api_key") or "").strip():
                    value = str(saved["api_key"]).strip()
                    self._source = "supabase-encrypted"
                    self._key_id = self.key_id(value)
                    self._last_error = ""
                    return value
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {str(exc).strip()}"[:300]
        self._source = "environment" if self.initial_key else "none"
        self._key_id = self.key_id(self.initial_key)
        return self.initial_key

    def create_readers(self, api_key: str) -> tuple[GeminiWeightReader, GeminiWeightReader]:
        fast = self.reader_factory(
            api_key,
            model=self.fast_model,
            timeout_seconds=self.fast_timeout,
            thinking_level="minimal",
            max_image_edge=1280,
            jpeg_quality=86,
            media_resolution="medium",
            include_qr=False,
        )
        try:
            accurate = self.reader_factory(
                api_key,
                model=self.accurate_model,
                timeout_seconds=self.accurate_timeout,
                thinking_level="medium",
                max_image_edge=1600,
                jpeg_quality=90,
                media_resolution="high",
                include_qr=False,
            )
        except Exception:
            fast.close()
            raise
        return fast, accurate

    @staticmethod
    def _validate_format(api_key: str) -> str:
        value = api_key.strip()
        if not 20 <= len(value) <= 256 or any(character.isspace() for character in value):
            raise ValueError("Gemini API key không đúng định dạng")
        return value

    def validate(self, api_key: str) -> None:
        value = self._validate_format(api_key)
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Thiếu google-genai trên backend") from exc
        client = genai.Client(
            api_key=value,
            http_options=types.HttpOptions(timeout=10000),
        )
        try:
            client.models.get(model=self.fast_model)
        except Exception as exc:
            raise ValueError(f"Gemini từ chối key hoặc model: {str(exc)[:240]}") from exc
        finally:
            client.close()

    def replace(self, api_key: str) -> tuple[GeminiWeightReader, GeminiWeightReader]:
        value = self._validate_format(api_key)
        self.validate(value)
        readers = self.create_readers(value)
        try:
            self.store.write({"api_key": value, "provider": "gemini"})
        except Exception:
            readers[0].close()
            readers[1].close()
            raise
        with self._lock:
            self._source = "supabase-encrypted"
            self._key_id = self.key_id(value)
            self._last_error = ""
        return readers

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "change_enabled": self.store.configured,
                "source": self._source,
                "key_id": self._key_id or None,
                "stored_encrypted": self._source == "supabase-encrypted",
                "last_error": self._last_error or None,
                "message": self.store.config_error if not self.store.configured else "Sẵn sàng đổi Gemini API key",
            }
