from __future__ import annotations

import pytest

from roll_qr_scale.gemini_key_manager import GeminiKeyManager
from roll_qr_scale.codex_oauth import EncryptedCodexTokenStore


class MemoryStore:
    configured = True
    config_error = ""

    def __init__(self, value=None, error: Exception | None = None):
        self.value = value
        self.error = error

    def read(self):
        if self.error:
            raise self.error
        return self.value

    def write(self, value):
        if self.error:
            raise self.error
        self.value = value


class FakeReader:
    def __init__(self, api_key, **kwargs):
        self.api_key = api_key
        self.kwargs = kwargs
        self.closed = False

    def close(self):
        self.closed = True


def manager(store, initial="environment-key-value"):
    return GeminiKeyManager(
        store,
        fast_model="fast-model",
        flash37_model="flash37-model",
        accurate_model="accurate-model",
        fast_timeout=10,
        flash37_timeout=30,
        accurate_timeout=30,
        initial_key=initial,
        reader_factory=FakeReader,
    )


def test_load_prefers_encrypted_supabase_key() -> None:
    current = manager(MemoryStore({"api_key": "stored-key-value-123456"}))
    assert current.load_key() == "stored-key-value-123456"
    assert current.status()["source"] == "supabase-encrypted"
    assert current.status()["key_id"] == current.key_id("stored-key-value-123456")


def test_load_falls_back_to_environment_when_supabase_is_temporarily_down() -> None:
    current = manager(MemoryStore(error=RuntimeError("network down")))
    assert current.load_key() == "environment-key-value"
    assert current.status()["source"] == "environment"
    assert "network down" in str(current.status()["last_error"])


def test_replace_validates_before_persisting_and_creates_all_readers(monkeypatch) -> None:
    store = MemoryStore()
    current = manager(store)
    monkeypatch.setattr(current, "validate", lambda key: None)
    fast, flash37, accurate = current.replace("new-key-value-123456789")
    assert store.value == {"api_key": "new-key-value-123456789", "provider": "gemini"}
    assert fast.kwargs["model"] == "fast-model"
    assert flash37.kwargs["model"] == "flash37-model"
    assert flash37.kwargs["thinking_level"] == "low"
    assert accurate.kwargs["model"] == "accurate-model"
    assert current.status()["stored_encrypted"] is True
    assert current.status()["key_id"] == current.key_id("new-key-value-123456789")


def test_replace_keeps_new_readers_out_when_store_fails(monkeypatch) -> None:
    current = manager(MemoryStore(error=RuntimeError("write failed")))
    monkeypatch.setattr(current, "validate", lambda key: None)
    created = []

    def create(api_key, **kwargs):
        reader = FakeReader(api_key, **kwargs)
        created.append(reader)
        return reader

    current.reader_factory = create
    with pytest.raises(RuntimeError, match="write failed"):
        current.replace("new-key-value-123456789")
    assert len(created) == 3
    assert all(item.closed for item in created)


def test_gemini_store_uses_legacy_compatible_encrypted_secret_action() -> None:
    store = EncryptedCodexTokenStore(
        "https://example.invalid/ingest",
        "device-token",
        secret_name="gemini-api-key:gateway-01",
        secret_action="codex-auth",
    )
    assert store.secret_action == "codex-auth"
