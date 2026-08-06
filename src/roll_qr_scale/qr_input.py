from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class QRValue:
    value: str
    source: str
    seen_at: float


class HIDQRInput:
    """Collect characters emitted by a USB scanner in keyboard-wedge mode."""

    name = "scanner_hid"

    def __init__(self, min_length: int = 3, character_timeout: float = 0.5):
        self.min_length = min_length
        self.character_timeout = character_timeout
        self.buffer = ""
        self.current: QRValue | None = None
        self._last_character_at = 0.0

    def handle_key(self, key: int, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if key in (10, 13, 9):  # Scanner suffix: Enter or Tab.
            if len(self.buffer) >= self.min_length:
                self.current = QRValue(self.buffer, self.name, now)
            self.buffer = ""
            return True
        if key in (8, 127):
            self.buffer = self.buffer[:-1]
            return True
        if 33 <= key <= 126:  # Space is reserved for committing the measurement.
            if self.buffer and now - self._last_character_at > self.character_timeout:
                self.buffer = ""
            self.buffer += chr(key)
            self._last_character_at = now
            return True
        return False

    def reading(self) -> QRValue | None:
        return self.current

    def clear(self) -> None:
        self.buffer = ""
        self.current = None

    def close(self) -> None:
        return None


class SerialQRInput:
    """Read line-delimited QR values from a scanner's virtual COM port."""

    name = "scanner_serial"

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        min_length: int = 3,
        encoding: str = "utf-8",
    ):
        import serial

        self.min_length = min_length
        self.encoding = encoding
        self._current: QRValue | None = None
        self._error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=0.25)
        self._thread = threading.Thread(target=self._read_loop, name="qr-serial", daemon=True)
        self._thread.start()

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self._serial.readline()
                if not payload:
                    continue
                value = payload.decode(self.encoding, errors="strict").strip()
                if len(value) < self.min_length:
                    continue
                with self._lock:
                    self._current = QRValue(value, self.name, time.monotonic())
                    self._error = None
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                self._stop.wait(0.25)

    def reading(self) -> QRValue | None:
        with self._lock:
            return self._current

    def clear(self) -> None:
        with self._lock:
            self._current = None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._serial.close()

