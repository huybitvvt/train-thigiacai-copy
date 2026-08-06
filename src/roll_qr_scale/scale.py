from __future__ import annotations

import re
import statistics
import threading
from collections import deque
from dataclasses import dataclass


WEIGHT_PATTERN = re.compile(
    r"(?P<value>[-+]?\d+(?:[.,]\d+)?)\s*(?P<unit>kg|kgs|g|lb|lbs)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WeightReading:
    value: float | None
    unit: str
    stable: bool
    raw: str = ""
    confidence: float | None = None


def parse_weight_line(line: str, default_unit: str = "kg") -> tuple[float, str] | None:
    """Parse common streaming scale lines such as 'ST,GS,+ 12.340 kg'."""
    matches = list(WEIGHT_PATTERN.finditer(line))
    if not matches:
        return None
    match = matches[-1]
    try:
        value = float(match.group("value").replace(",", "."))
    except ValueError:
        return None
    unit = (match.group("unit") or default_unit).lower()
    if unit == "kgs":
        unit = "kg"
    elif unit == "lbs":
        unit = "lb"
    return value, unit


class ManualWeightSource:
    name = "manual"

    def __init__(self, initial: str = "", unit: str = "kg"):
        self.text = initial
        self.unit = unit

    def reading(self) -> WeightReading:
        try:
            value = float(self.text.replace(",", ".")) if self.text not in {"", "-", ".", "-."} else None
        except ValueError:
            value = None
        return WeightReading(value=value, unit=self.unit, stable=value is not None, raw=self.text)

    def handle_key(self, key: int) -> bool:
        """Update the manual entry. Returns True when the key was consumed."""
        if ord("0") <= key <= ord("9"):
            self.text += chr(key)
            return True
        if key in (ord("."), ord(",")) and "." not in self.text and "," not in self.text:
            self.text += "."
            return True
        if key == ord("-") and not self.text:
            self.text = "-"
            return True
        if key in (8, 127):
            self.text = self.text[:-1]
            return True
        if key in (ord("c"), ord("C")):
            self.text = ""
            return True
        return False

    def close(self) -> None:
        return None


class SerialWeightSource:
    name = "serial"

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        unit: str = "kg",
        stable_samples: int = 3,
        tolerance: float = 0.02,
    ):
        import serial

        self.unit = unit
        self.stable_samples = stable_samples
        self.tolerance = tolerance
        self._samples: deque[float] = deque(maxlen=max(stable_samples, 10))
        self._last_raw = ""
        self._last_unit = unit
        self._error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=0.25)
        self._thread = threading.Thread(target=self._read_loop, name="scale-serial", daemon=True)
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
                line = payload.decode("ascii", errors="ignore").strip()
                parsed = parse_weight_line(line, self.unit)
                if parsed is None:
                    continue
                value, unit = parsed
                with self._lock:
                    self._samples.append(value)
                    self._last_unit = unit
                    self._last_raw = line
                    self._error = None
            except Exception as exc:  # Device disconnection must be visible in the UI.
                with self._lock:
                    self._error = str(exc)
                self._stop.wait(0.25)

    def reading(self) -> WeightReading:
        with self._lock:
            samples = list(self._samples)
            unit = self._last_unit
            raw = self._last_raw
        if not samples:
            return WeightReading(value=None, unit=unit, stable=False, raw=raw)
        window = samples[-self.stable_samples :]
        stable = len(window) >= self.stable_samples and max(window) - min(window) <= self.tolerance
        return WeightReading(value=statistics.median(window), unit=unit, stable=stable, raw=raw)

    def handle_key(self, key: int) -> bool:
        return False

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._serial.close()
