from __future__ import annotations

import json
import os
import re
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .scale import WeightReading


OCR_ALLOWLIST = "0123456789.,-kKgGlLbBsS"
PADDLE_OCR_MODEL_NAME = "PP-OCRv6_medium_rec"
PADDLE_OCR_TRUST_CONFIDENCE = 0.82
SEVEN_SEGMENT_OVERRIDE_CONFIDENCE = 0.95
BRIGHT_LED_MIN_CONFIDENCE = 0.50
# A geometry-only LEDCORE vote is allowed to participate at the specialized
# minimum, but it is authoritative only above this stricter confidence.  Lower
# confidence geometry must be corroborated by another preprocessing branch.
BRIGHT_LED_TRUST_CONFIDENCE = 0.75
TEMPORAL_MIN_AGREEMENT = 0.60
TEMPORAL_MIN_VALID_AGREEMENT = 0.75
TEMPORAL_MIN_CONSECUTIVE = 3
MIN_LED_DIGIT_HEIGHT = 16
MAX_TEMPORAL_OCR_FRAMES = 5
PADDLE_BATCH_TEMPORAL_FRAMES = 3
OCR_WEIGHT_PATTERN = re.compile(
    r"^\s*(?P<value>[-+]?\d+(?:[.,]\d+)?)\s*(?P<unit>kg|kgs|g|lb|lbs)?\s*$",
    re.IGNORECASE,
)

SEVEN_SEGMENT_PATTERNS = {
    "0": frozenset("abcedf"),
    "1": frozenset("bc"),
    "2": frozenset("abged"),
    "3": frozenset("abgcd"),
    "4": frozenset("fgbc"),
    "5": frozenset("afgcd"),
    "6": frozenset("afgecd"),
    "7": frozenset("abc"),
    "8": frozenset("abcedfg"),
    "9": frozenset("abfgcd"),
}


def _bright_led_core_mask(
    crop: np.ndarray,
    red_threshold: int = 210,
    green_threshold: int = 80,
) -> np.ndarray:
    """Keep the yellow-white LED core while rejecting the surrounding red glow."""
    if crop.ndim != 3 or crop.shape[2] < 3:
        return np.zeros(crop.shape[:2], dtype=np.uint8)
    blue, green, red = cv2.split(crop[:, :, :3])
    return (
        (red >= red_threshold)
        & (green >= green_threshold)
        & (blue < 180)
    ).astype(np.uint8) * 255


def _red_led_digit_height(crop: np.ndarray) -> int:
    """Estimate physical LED glyph height before any artificial upscaling."""

    if crop.ndim != 3 or crop.shape[2] < 3 or crop.size == 0:
        return 0
    bright_core = _bright_led_core_mask(crop) > 0
    core_rows = np.flatnonzero(np.any(bright_core, axis=1))
    if core_rows.size:
        return int(core_rows[-1] - core_rows[0] + 1)
    blue, green, red = cv2.split(crop[:, :, :3])
    red_i16 = red.astype(np.int16)
    other = np.maximum(green, blue).astype(np.int16)
    mask = (red_i16 >= 120) & (red_i16 - other >= 45)
    rows = np.flatnonzero(np.any(mask, axis=1))
    return int(rows[-1] - rows[0] + 1) if rows.size else 0


def _restore_fixed_scale_decimal(text: str, crop: np.ndarray) -> str:
    """Restore the configured two-decimal layout used by the factory scale."""
    stripped = text.strip()
    normalized = stripped.replace(",", ".")
    if normalized.count(".") > 1:
        return text
    digits = normalized.replace(".", "")
    if not digits.isdigit() or len(digits) < 3:
        return text
    core = _bright_led_core_mask(crop)
    core_fraction = float(np.mean(core > 0)) if core.size else 0.0
    if not 0.005 <= core_fraction <= 0.35:
        return text
    restored = digits[:-2] + "." + digits[-2:]
    return text if normalized == restored else restored


def _correct_bright_led_nine_confusions(text: str, crop: np.ndarray) -> str:
    """Recover a 9 when OCR misses its upper-left LED segment."""
    stripped = text.strip()
    if not stripped.isdigit():
        return text
    core = _bright_led_core_mask(crop) > 0
    if not np.any(core):
        return text
    digit_runs = [
        run
        for run in _runs(np.any(core, axis=0))
        if run[1] - run[0] >= 3
    ]
    if len(digit_runs) != len(stripped):
        return text

    corrected = list(stripped)
    for index, (character, (left, right)) in enumerate(zip(stripped, digit_runs, strict=True)):
        if character not in {"2", "3"}:
            continue
        digit = core[:, left:right]
        active_rows = np.flatnonzero(np.any(digit, axis=1))
        if active_rows.size == 0:
            continue
        digit = digit[active_rows[0] : active_rows[-1] + 1]
        normalized = cv2.resize(
            digit.astype(np.uint8),
            (60, 100),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        # 9 differs from the common OCR guesses 2/3 by a strong upper-left
        # vertical segment. Require substantial coverage before correcting it.
        upper_left = normalized[12:49, 0:18]
        upper_right = normalized[12:49, 42:60]
        left_score = float(np.mean(upper_left))
        right_score = float(np.mean(upper_right))
        if left_score >= 0.60 and left_score >= right_score * 0.70:
            corrected[index] = "9"
    return "".join(corrected)


def parse_ocr_weight(text: str, default_unit: str) -> tuple[float, str] | None:
    match = OCR_WEIGHT_PATTERN.fullmatch(text)
    if match is None:
        return None
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


def _restore_led_decimal(text: str, crop: np.ndarray) -> str:
    """Restore a decimal point that OCR dropped from a red seven-segment display."""
    stripped = text.strip()
    if not stripped.isdigit() or len(stripped) < 2 or crop.ndim != 3:
        return text
    blue, green, red = cv2.split(crop[:, :, :3])
    red_i16 = red.astype(np.int16)
    other = np.maximum(green, blue).astype(np.int16)
    mask = ((red_i16 >= 100) & (red_i16 - other >= 30)).astype(np.uint8) * 255
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 2:
        return text

    components = [
        {
            "x": int(stats[index, cv2.CC_STAT_LEFT]),
            "y": int(stats[index, cv2.CC_STAT_TOP]),
            "width": int(stats[index, cv2.CC_STAT_WIDTH]),
            "height": int(stats[index, cv2.CC_STAT_HEIGHT]),
            "area": int(stats[index, cv2.CC_STAT_AREA]),
            "center_x": float(centroids[index, 0]),
            "strength": int(np.sum((red_i16 - other)[labels == index])),
        }
        for index in range(1, count)
    ]
    maximum_height = max(component["height"] for component in components)
    glyphs = [
        component
        for component in components
        if component["height"] >= maximum_height * 0.65
    ]
    if len(glyphs) != len(stripped):
        return text

    content_bottom = max(component["y"] + component["height"] for component in glyphs)
    dots = [
        component
        for component in components
        if component not in glyphs
        and component["height"] <= max(5, maximum_height * 0.45)
        and component["width"] <= max(5, maximum_height * 0.45)
        and 0.45 <= component["width"] / max(1, component["height"]) <= 2.2
        and component["y"] + component["height"] >= content_bottom - max(2, maximum_height * 0.2)
    ]
    if not dots:
        return text
    # Reflections can create a second faint red point. Prefer the component
    # carrying the strongest red LED energy, not simply the rightmost dot.
    dot = max(dots, key=lambda component: (component["strength"], component["area"]))
    insertion = sum(glyph["center_x"] < dot["center_x"] for glyph in glyphs)
    if not 1 <= insertion < len(stripped):
        return text
    return stripped[:insertion] + "." + stripped[insertion:]


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(active.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _classify_bright_core_digit(digit: np.ndarray) -> tuple[str, float] | None:
    """Match one isolated bright LED glyph against the seven-segment alphabet."""
    active_rows = np.flatnonzero(np.any(digit, axis=1))
    if active_rows.size == 0:
        return None
    digit = digit[active_rows[0] : active_rows[-1] + 1]
    # A leading ``1`` often contains only the two right-hand vertical bars.
    # At the small scale of a factory photo those bars are much narrower than
    # the other glyphs and the generic segment score can otherwise call them
    # a damaged ``2``/``3``.  The seven-segment layout makes this a safe,
    # geometry-only decision.
    aspect = digit.shape[1] / max(1, digit.shape[0])
    if aspect < 0.34:
        return "1", 0.90
    normalized = cv2.resize(
        digit.astype(np.uint8),
        (60, 100),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    segment_regions = {
        "a": (0.16, 0.00, 0.84, 0.18),
        "b": (0.70, 0.12, 1.00, 0.49),
        "c": (0.70, 0.51, 1.00, 0.88),
        "d": (0.16, 0.82, 0.84, 1.00),
        "e": (0.00, 0.51, 0.30, 0.88),
        "f": (0.00, 0.12, 0.30, 0.49),
        "g": (0.16, 0.41, 0.84, 0.59),
    }
    scores: dict[str, float] = {}
    for name, (x1, y1, x2, y2) in segment_regions.items():
        region = normalized[
            round(y1 * 100) : max(round(y1 * 100) + 1, round(y2 * 100)),
            round(x1 * 60) : max(round(x1 * 60) + 1, round(x2 * 60)),
        ]
        scores[name] = float(np.mean(region))
    peak = max(scores.values())
    if peak <= 0:
        return None
    normalized_scores = {
        name: min(1.0, value / peak) for name, value in scores.items()
    }
    matches: list[tuple[float, str]] = []
    for number, enabled in SEVEN_SEGMENT_PATTERNS.items():
        error = float(np.mean([
            1.0 - value if name in enabled else value
            for name, value in normalized_scores.items()
        ]))
        matches.append((error, number))
    error, number = min(matches)
    return number, max(0.0, 1.0 - error)


def _correct_bright_led_zero_eight_confusions(text: str, crop: np.ndarray) -> str:
    """Correct only the physical ``0``/``8`` ambiguity in OCR output.

    Reflections on a red LED can fill the middle bar of a ``0`` in the green
    preprocessing branches, making a generic OCR model report ``8``. Conversely, a weak
    middle bar can make an ``8`` look like ``0``.  We use the independent
    seven-segment core at several thresholds and change a character only when
    a clear majority of thresholds agrees on the opposite member of the pair.
    This deliberately does not rewrite other digits, so a bad geometry guess
    cannot replace an otherwise valid OCR result.
    """
    if crop.ndim != 3 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return text
    digits = "".join(character for character in text.strip() if character.isdigit())
    if len(digits) < 2:
        return text

    votes: list[list[tuple[str, float]]] = [[] for _ in digits]
    minimum_run_width = max(3, round(crop.shape[0] * 0.08))
    for red_threshold in range(180, 241, 5):
        green_threshold = max(50, red_threshold - 130)
        core = _bright_led_core_mask(crop, red_threshold, green_threshold) > 0
        digit_runs = [
            run
            for run in _runs(np.any(core, axis=0))
            if run[1] - run[0] >= minimum_run_width
        ]
        if len(digit_runs) != len(digits):
            continue
        matches = [
            _classify_bright_core_digit(core[:, left:right])
            for left, right in digit_runs
        ]
        if any(match is None for match in matches):
            continue
        for index, match in enumerate(matches):
            assert match is not None
            number, confidence = match
            if number in {"0", "8"} and confidence >= 0.70:
                votes[index].append((number, confidence))

    corrected_digits = list(digits)
    for index, observed in enumerate(digits):
        if observed not in {"0", "8"} or not votes[index]:
            continue
        opposite = "0" if observed == "8" else "8"
        opposite_votes = [item for item in votes[index] if item[0] == opposite]
        if len(opposite_votes) < 3:
            continue
        if len(opposite_votes) / len(votes[index]) < 0.65:
            continue
        if float(np.mean([item[1] for item in opposite_votes])) < 0.74:
            continue
        corrected_digits[index] = opposite

    if corrected_digits == list(digits):
        return text
    iterator = iter(corrected_digits)
    return "".join(
        next(iterator) if character.isdigit() else character
        for character in text.strip()
    )


def _decode_bright_core_digits(crop: np.ndarray) -> tuple[str, float] | None:
    """Decode a fixed LED row by consensus across several core thresholds.

    Saturated seven-segment photos often lose one thin stroke at a single
    threshold. Requiring the same glyph sequence at multiple thresholds makes
    the geometry decoder useful without trusting one brittle binary mask.
    """
    if crop.ndim != 3 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None
    minimum_run_width = max(3, round(crop.shape[0] * 0.08))
    votes: dict[str, list[float]] = defaultdict(list)
    total_votes = 0
    for red_threshold in range(255, 204, -5):
        green_threshold = max(75, red_threshold - 130)
        core = _bright_led_core_mask(
            crop,
            red_threshold,
            green_threshold,
        ) > 0
        digit_runs = [
            run
            for run in _runs(np.any(core, axis=0))
            if run[1] - run[0] >= minimum_run_width
        ]
        if not 3 <= len(digit_runs) <= 6:
            continue
        matches = [
            _classify_bright_core_digit(core[:, left:right])
            for left, right in digit_runs
        ]
        if any(match is None or match[1] < 0.58 for match in matches):
            continue
        text = "".join(match[0] for match in matches if match is not None)
        text = _correct_bright_led_nine_confusions(text, crop)
        quality = float(np.mean([
            match[1] for match in matches if match is not None
        ]))
        votes[text].append(quality)
        total_votes += 1
    if total_votes == 0:
        return None
    text, qualities = max(
        votes.items(),
        key=lambda item: (len(item[1]), float(np.mean(item[1]))),
    )
    agreement = len(qualities) / total_votes
    mean_quality = float(np.mean(qualities))
    if len(qualities) < 2 or agreement < 0.65 or mean_quality < 0.58:
        return None
    confidence = min(0.99, mean_quality * (0.80 + 0.20 * agreement))
    return text, confidence


def _correct_bright_led_confusions(
    text: str,
    crop: np.ndarray,
    decoded_digits: str | None = None,
) -> str:
    """Disambiguate OCR text with multi-threshold LED geometry."""
    stripped = text.strip()
    if not stripped.isdigit():
        return text
    corrected = _correct_bright_led_nine_confusions(stripped, crop)
    if decoded_digits is None:
        decoded = _decode_bright_core_digits(crop)
        decoded_digits = decoded[0] if decoded is not None else None
    if decoded_digits is None or len(decoded_digits) != len(corrected):
        return corrected
    mismatches = [
        (observed, expected)
        for expected, observed in zip(decoded_digits, corrected, strict=True)
        if expected != observed
    ]
    # One missing/merged segment is the failure mode seen on small factory LED
    # displays. Larger disagreements remain uncorrected instead of guessing.
    known_segment_losses = {
        ("2", "7"),
        ("3", "7"),
        ("8", "0"),
        ("7", "2"),
    }
    return (
        decoded_digits
        if not mismatches or (len(mismatches) == 1 and mismatches[0] in known_segment_losses)
        else corrected
    )


def _decode_seven_segment(crop: np.ndarray) -> tuple[str, float] | None:
    """Decode a red seven-segment row and its physical decimal LED."""
    if crop.ndim != 3 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None
    blue, green, red = cv2.split(crop[:, :, :3])
    red_i16 = red.astype(np.int16)
    other = np.maximum(green, blue).astype(np.int16)
    strength = red_i16 - other
    mask = ((red_i16 >= 100) & (strength >= 30)).astype(np.uint8) * 255
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 2:
        return None

    components = []
    for index in range(1, count):
        components.append(
            {
                "label": index,
                "x": int(stats[index, cv2.CC_STAT_LEFT]),
                "y": int(stats[index, cv2.CC_STAT_TOP]),
                "width": int(stats[index, cv2.CC_STAT_WIDTH]),
                "height": int(stats[index, cv2.CC_STAT_HEIGHT]),
                "area": int(stats[index, cv2.CC_STAT_AREA]),
                "center_x": float(centroids[index, 0]),
                "strength": int(np.sum(strength[labels == index])),
            }
        )
    maximum_height = max(component["height"] for component in components)
    dots = [
        component
        for component in components
        if component["height"] <= max(5, maximum_height * 0.45)
        and component["width"] <= max(5, maximum_height * 0.45)
        and 0.45 <= component["width"] / max(1, component["height"]) <= 2.2
        and component["y"] >= crop.shape[0] * 0.45
    ]
    decimal = max(dots, key=lambda item: (item["strength"], item["area"])) if dots else None
    digit_mask = mask.copy()
    for dot in dots:
        digit_mask[labels == dot["label"]] = 0

    active_columns = np.any(digit_mask > 0, axis=0).astype(np.uint8)
    close_width = max(1, round(crop.shape[0] * 0.06))
    if close_width > 1:
        active_columns = cv2.morphologyEx(
            active_columns.reshape(1, -1),
            cv2.MORPH_CLOSE,
            np.ones((1, close_width), dtype=np.uint8),
        ).reshape(-1)
    digit_runs = [
        run for run in _runs(active_columns)
        if run[1] - run[0] >= max(2, round(crop.shape[0] * 0.12))
    ]
    if not 1 <= len(digit_runs) <= 8:
        return None

    decoded: list[str] = []
    confidences: list[float] = []
    segment_regions = {
        "a": (0.16, 0.00, 0.84, 0.18),
        "b": (0.70, 0.12, 1.00, 0.49),
        "c": (0.70, 0.51, 1.00, 0.88),
        "d": (0.16, 0.82, 0.84, 1.00),
        "e": (0.00, 0.51, 0.30, 0.88),
        "f": (0.00, 0.12, 0.30, 0.49),
        "g": (0.16, 0.41, 0.84, 0.59),
    }
    for left, right in digit_runs:
        digit = digit_mask[:, left:right]
        rows = np.flatnonzero(np.any(digit > 0, axis=1))
        if rows.size == 0:
            return None
        digit = digit[rows[0] : rows[-1] + 1]
        aspect = digit.shape[1] / max(1, digit.shape[0])
        if aspect < 0.34:
            decoded.append("1")
            confidences.append(0.90)
            continue
        normalized = cv2.resize(digit, (60, 100), interpolation=cv2.INTER_NEAREST) > 0
        scores: dict[str, float] = {}
        for name, (x1, y1, x2, y2) in segment_regions.items():
            region = normalized[
                round(y1 * 100) : max(round(y1 * 100) + 1, round(y2 * 100)),
                round(x1 * 60) : max(round(x1 * 60) + 1, round(x2 * 60)),
            ]
            scores[name] = float(np.mean(region))
        peak = max(scores.values())
        if peak <= 0:
            return None
        normalized_scores = {name: min(1.0, value / peak) for name, value in scores.items()}
        matches = []
        for number, enabled in SEVEN_SEGMENT_PATTERNS.items():
            error = float(np.mean([
                1.0 - value if name in enabled else value
                for name, value in normalized_scores.items()
            ]))
            matches.append((error, number))
        error, number = min(matches)
        confidence = max(0.0, 1.0 - error)
        if confidence < 0.58:
            return None
        decoded.append(number)
        confidences.append(confidence)

    decimal_index = None
    if decimal is not None:
        decimal_index = sum(
            (left + right) / 2 < decimal["center_x"]
            for left, right in digit_runs
        )
        if not 1 <= decimal_index < len(decoded):
            decimal_index = None
    text = "".join(decoded)
    if decimal_index is not None:
        text = text[:decimal_index] + "." + text[decimal_index:]
    return text, min(confidences)


@dataclass(frozen=True)
class NormalizedROI:
    """A frame-relative rectangle expressed as x1,y1,x2,y2 in the 0..1 range."""

    x1: float
    y1: float
    x2: float
    y2: float

    def pixels(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        left = max(0, min(width - 1, round(self.x1 * width)))
        top = max(0, min(height - 1, round(self.y1 * height)))
        right = max(left + 1, min(width, round(self.x2 * width)))
        bottom = max(top + 1, min(height, round(self.y2 * height)))
        return left, top, right, bottom


def parse_normalized_roi(value: str) -> NormalizedROI:
    try:
        values = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("weight ROI must contain four decimal numbers: x1,y1,x2,y2") from exc
    if len(values) != 4:
        raise ValueError("weight ROI must contain four decimal numbers: x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("weight ROI values must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return NormalizedROI(x1, y1, x2, y2)


def _top_row_from_led_stack(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the first row of a vertically stacked gross/tare/net LED display."""
    height, width = mask.shape
    minimum_row_pixels = max(2, round(width * 0.001))
    active_rows = (np.count_nonzero(mask, axis=1) >= minimum_row_pixels).astype(np.uint8)
    active_rows[: int(height * 0.40)] = 0
    # Only bridge gaps inside one seven-segment row. A larger kernel merges the
    # three closely spaced gross/tare/net rows into one vertical rectangle.
    close_height = max(2, round(height * 0.003))
    active_rows = cv2.morphologyEx(
        active_rows.reshape(-1, 1),
        cv2.MORPH_CLOSE,
        np.ones((close_height, 1), dtype=np.uint8),
    ).reshape(-1)

    rows: list[tuple[tuple[int, int, int, int], int]] = []
    for top, bottom in _runs(active_rows):
        band_height = bottom - top
        if band_height < max(4, round(height * 0.004)) or band_height > height * 0.10:
            continue
        ys, xs = np.nonzero(mask[top:bottom])
        if xs.size == 0:
            continue
        left, right = int(xs.min()), int(xs.max()) + 1
        box_width = right - left
        if box_width < max(12, round(width * 0.010)):
            continue
        red_pixels = int(xs.size)
        rows.append(((left, top, right, bottom), red_pixels))
    if len(rows) < 2:
        return None

    stacks: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for box, red_pixels in rows:
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        center_x = (box[0] + box[2]) / 2
        below = []
        for other, other_pixels in rows:
            if other[1] <= box[1]:
                continue
            other_width = other[2] - other[0]
            other_height = other[3] - other[1]
            other_center_x = (other[0] + other[2]) / 2
            if (
                other[1] - box[3] <= max(box_height, other_height) * 5.5
                and abs(other_center_x - center_x) <= max(box_width, other_width) * 0.55
                and 0.40 <= other_width / max(1, box_width) <= 2.5
            ):
                below.append(other_pixels)
        if below:
            stacks.append((len(below), red_pixels + sum(below), box))
    if not stacks:
        return None
    return max(stacks, key=lambda item: (item[0], item[1]))[2]


def _split_vertical_stack_to_top_row(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Recover the top LED row when contour morphology merged the full stack."""
    left, top, right, bottom = box
    box_width, box_height = right - left, bottom - top
    if box_height <= box_width * 0.75:
        return box
    band_counts = np.count_nonzero(mask[top:bottom], axis=1)
    active = (band_counts >= 2).astype(np.uint8)
    active = cv2.morphologyEx(
        active.reshape(-1, 1),
        cv2.MORPH_CLOSE,
        np.ones((max(2, round(mask.shape[0] * 0.003)), 1), dtype=np.uint8),
    ).reshape(-1)
    bands = [run for run in _runs(active) if run[1] - run[0] >= 3]
    if len(bands) < 2:
        return box
    relative_top, relative_bottom = bands[0]
    row_top, row_bottom = top + relative_top, top + relative_bottom
    _, xs = np.nonzero(mask[row_top:row_bottom])
    if xs.size == 0:
        return box
    return int(xs.min()), row_top, int(xs.max()) + 1, row_bottom


def detect_weight_roi(frame: np.ndarray) -> tuple[NormalizedROI, str] | None:
    """Locate a red LED weight readout without running an object detector."""
    if frame.ndim != 3 or frame.shape[2] < 3:
        return None
    height, width = frame.shape[:2]
    blue, green, red = cv2.split(frame[:, :, :3])
    red_i16 = red.astype(np.int16)
    other = np.maximum(green, blue).astype(np.int16)
    mask = ((red_i16 >= 120) & (red_i16 - other >= 45)).astype(np.uint8) * 255
    mask[: int(height * 0.40), :] = 0

    # A strong mask removes the red halo that otherwise connects gross, tare
    # and net into one tall component in real factory photos.
    stacked_row = None
    stack_mask = mask
    first_stack: tuple[tuple[int, int, int, int], np.ndarray] | None = None
    for red_threshold, dominance_threshold in (
        (150, 65),
        (160, 70),
        (170, 80),
        (180, 90),
        (190, 100),
        (200, 110),
        (210, 120),
        (220, 130),
    ):
        candidate_mask = (
            (red_i16 >= red_threshold)
            & (red_i16 - other >= dominance_threshold)
        ).astype(np.uint8) * 255
        candidate_mask[: int(height * 0.40), :] = 0
        candidate = _top_row_from_led_stack(candidate_mask)
        if candidate is None:
            continue
        if first_stack is None:
            first_stack = (candidate, candidate_mask)
        candidate_width = candidate[2] - candidate[0]
        candidate_height = candidate[3] - candidate[1]
        if candidate_height <= candidate_width * 0.90:
            stacked_row = candidate
            stack_mask = candidate_mask
            break
    if stacked_row is None and first_stack is not None:
        stacked_row, stack_mask = first_stack
    if stacked_row is None:
        stacked_row = _top_row_from_led_stack(mask)
        stack_mask = mask
    if stacked_row is not None:
        left, top, right, bottom = _split_vertical_stack_to_top_row(
            stack_mask,
            stacked_row,
        )
        pad_x = max(3, round((right - left) * 0.06))
        pad_y = max(3, round((bottom - top) * 0.22))
        left, top = max(0, left - pad_x), max(0, top - pad_y)
        right, bottom = min(width, right + pad_x), min(height, bottom + pad_y)
        return NormalizedROI(
            left / width,
            top / height,
            right / width,
            bottom / height,
        ), "red-led"

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(5, width // 120), max(3, height // 140)),
    )
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    joined = cv2.dilate(
        joined,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(3, width // 420), max(3, height // 320)),
        ),
        iterations=1,
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Portrait factory photos make the scale display small relative to the full
    # frame. Keep the lower bound tied to digit height, not the frame width.
    minimum_height = max(6, round(height * 0.008))
    components: list[tuple[int, int, int, int]] = []
    for contour in contours:
        left, top, box_width, box_height = cv2.boundingRect(contour)
        if box_height < minimum_height or box_width < 3:
            continue
        if box_height > height * 0.20 or box_width > width * 0.45:
            continue
        components.append((left, top, left + box_width, top + box_height))
    if not components:
        return None

    candidates: dict[tuple[int, int, int, int], int] = {}
    for seed in components:
        seed_height = seed[3] - seed[1]
        seed_center_y = (seed[1] + seed[3]) / 2
        aligned = [
            box
            for box in components
            if abs((box[1] + box[3]) / 2 - seed_center_y)
            <= max(seed_height, box[3] - box[1]) * 0.65
            and 0.45 <= (box[3] - box[1]) / seed_height <= 2.2
            and max(0, max(seed[0], box[0]) - min(seed[2], box[2]))
            <= max(seed_height, box[3] - box[1]) * 1.2
        ]
        left = min(box[0] for box in aligned)
        top = min(box[1] for box in aligned)
        right = max(box[2] for box in aligned)
        bottom = max(box[3] for box in aligned)
        box_width, box_height = right - left, bottom - top
        aspect = box_width / max(1, box_height)
        if not (1.0 <= aspect <= 16 and box_width >= max(14, width * 0.015)):
            continue
        red_pixels = int(cv2.countNonZero(mask[top:bottom, left:right]))
        candidates[(left, top, right, bottom)] = max(
            red_pixels,
            candidates.get((left, top, right, bottom), 0),
        )
    if not candidates:
        return None

    # Industrial indicators commonly show gross/tare/net on vertically stacked
    # rows. The gross weight is the first row; selecting by pixel count alone can
    # incorrectly return a lower "0.00" row.
    stacks: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for box, red_pixels in candidates.items():
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        center_x = (box[0] + box[2]) / 2
        rows_below = []
        for other, other_pixels in candidates.items():
            if other == box or other[1] <= box[1]:
                continue
            other_width = other[2] - other[0]
            other_height = other[3] - other[1]
            other_center_x = (other[0] + other[2]) / 2
            vertical_gap = other[1] - box[3]
            if (
                -max(box_height, other_height) * 0.35
                <= vertical_gap
                <= max(box_height, other_height) * 4.5
                and abs(other_center_x - center_x) <= max(box_width, other_width) * 0.45
                and 0.45 <= other_width / max(1, box_width) <= 2.2
                and 0.45 <= other_height / max(1, box_height) <= 2.2
            ):
                rows_below.append(other_pixels)
        if rows_below:
            stacks.append((len(rows_below), red_pixels + sum(rows_below), box))

    if stacks:
        best_box = max(stacks, key=lambda item: (item[0], item[1]))[2]
    else:
        best_box = max(candidates.items(), key=lambda item: item[1])[0]

    left, top, right, bottom = _split_vertical_stack_to_top_row(mask, best_box)
    pad_x = max(3, round((right - left) * 0.035))
    pad_y = max(3, round((bottom - top) * 0.16))
    left, top = max(0, left - pad_x), max(0, top - pad_y)
    right, bottom = min(width, right + pad_x), min(height, bottom + pad_y)
    return NormalizedROI(left / width, top / height, right / width, bottom / height), "red-led"


def detect_weight_roi_consensus(
    frames: list[np.ndarray] | tuple[np.ndarray, ...],
) -> tuple[NormalizedROI, str] | None:
    """Locate the same gross-weight row across a short camera burst.

    A median box is taken only from the largest spatially consistent cluster.
    This prevents one reflected red patch or one lower tare/net row from
    moving the OCR crop for the whole burst.
    """

    detections = [
        found[0]
        for frame in frames
        if isinstance(frame, np.ndarray)
        and frame.size
        and (found := detect_weight_roi(frame)) is not None
    ]
    if not detections:
        return None
    if len(detections) == 1:
        return detections[0], "red-led"

    def compatible(first: NormalizedROI, second: NormalizedROI) -> bool:
        first_width, second_width = first.x2 - first.x1, second.x2 - second.x1
        first_height, second_height = first.y2 - first.y1, second.y2 - second.y1
        dx = abs((first.x1 + first.x2 - second.x1 - second.x2) / 2)
        dy = abs((first.y1 + first.y2 - second.y1 - second.y2) / 2)
        width_ratio = first_width / max(second_width, 1e-6)
        height_ratio = first_height / max(second_height, 1e-6)
        return (
            dx <= max(0.02, max(first_width, second_width) * 0.55)
            and dy <= max(0.015, max(first_height, second_height) * 0.80)
            and 0.45 <= width_ratio <= 2.20
            and 0.45 <= height_ratio <= 2.20
        )

    clusters = [
        [candidate for candidate in detections if compatible(seed, candidate)]
        for seed in detections
    ]
    cluster = max(
        clusters,
        key=lambda items: (
            len(items),
            -float(np.std([(item.y1 + item.y2) / 2 for item in items])),
        ),
    )
    coordinates = np.asarray(
        [[item.x1, item.y1, item.x2, item.y2] for item in cluster],
        dtype=np.float64,
    )
    x1, y1, x2, y2 = np.median(coordinates, axis=0).tolist()
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    return (
        NormalizedROI(float(x1), float(y1), float(x2), float(y2)),
        f"red-led-temporal({len(cluster)}/{len(frames)})",
    )


def _paddle_model_dir() -> Path | None:
    configured = os.environ.get("ROLL_SCALE_PADDLE_MODEL_DIR", "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_dir():
            raise RuntimeError(
                f"PaddleOCR model directory does not exist: {path}"
            )
        return path

    cache_root = os.environ.get("PADDLE_PDX_CACHE_HOME", "").strip()
    candidates = []
    if cache_root:
        root = Path(cache_root).expanduser()
        candidates.extend((
            root / "official_models" / PADDLE_OCR_MODEL_NAME,
            root / PADDLE_OCR_MODEL_NAME,
        ))
    candidates.append(
        Path.home() / ".paddlex" / "official_models" / PADDLE_OCR_MODEL_NAME
    )
    return next((path.resolve() for path in candidates if path.is_dir()), None)


class PaddleOCRTextReader:
    """Expose PaddleOCR recognition-only inference through the old reader API."""

    def __init__(self, model: object):
        self.model = model

    @classmethod
    def create(
        cls,
        *,
        download_enabled: bool,
        gpu: bool,
    ) -> "PaddleOCRTextReader":
        model_dir = _paddle_model_dir()
        if model_dir is None and not download_enabled:
            raise RuntimeError(
                f"PaddleOCR model {PADDLE_OCR_MODEL_NAME} is missing. "
                "Connect once and run with --ocr-download, then run offline."
            )
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed; run: python -m pip install "
                "paddlepaddle==3.3.1 paddleocr==3.7.0"
            ) from exc

        try:
            model = TextRecognition(
                model_name=PADDLE_OCR_MODEL_NAME,
                model_dir=str(model_dir) if model_dir is not None else None,
                device="gpu" if gpu else "cpu",
                enable_hpi=False,
                enable_mkldnn=not gpu,
                cpu_threads=max(1, min(4, os.cpu_count() or 1)),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot initialize local PaddleOCR model {PADDLE_OCR_MODEL_NAME}: {exc}"
            ) from exc
        return cls(model)

    @staticmethod
    def _payload(result: object) -> Mapping[str, object] | None:
        payload: object = result
        if not isinstance(payload, Mapping):
            payload = getattr(result, "json", None)
            if callable(payload):
                payload = payload()
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return None
        if not isinstance(payload, Mapping):
            return None
        nested = payload.get("res")
        return nested if isinstance(nested, Mapping) else payload

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray | None:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] >= 3:
            return image[:, :, :3]
        return None

    @classmethod
    def _recognized_result(
        cls,
        prepared: np.ndarray,
        result: object,
    ) -> list[tuple[object, str, float]]:
        payload = cls._payload(result)
        if payload is None:
            return []
        text = str(payload.get("rec_text", "")).strip()
        try:
            score = float(payload.get("rec_score", 0.0))
        except (TypeError, ValueError):
            return []
        if not text or not math.isfinite(score):
            return []
        height, width = prepared.shape[:2]
        box = [[0, 0], [width, 0], [width, height], [0, height]]
        return [(box, text, max(0.0, min(1.0, score)))]

    def recognize_batch(
        self,
        images: list[np.ndarray] | tuple[np.ndarray, ...],
        **_: object,
    ) -> list[list[tuple[object, str, float]]]:
        prepared = [self._prepare(image) for image in images]
        if any(image is None for image in prepared):
            return [[] for _ in images]
        valid = [image for image in prepared if image is not None]
        try:
            results = list(self.model.predict(input=valid, batch_size=len(valid)))
        except Exception as exc:
            raise RuntimeError(f"PaddleOCR recognition failed: {exc}") from exc
        output = [
            self._recognized_result(image, result)
            for image, result in zip(valid, results)
        ]
        output.extend([[] for _ in range(len(valid) - len(output))])
        return output

    def recognize(self, image: np.ndarray, **kwargs: object) -> list[tuple[object, str, float]]:
        return self.recognize_batch([image], **kwargs)[0]


class CameraOCRWeightSource:
    """Read a scale display with local PaddleOCR plus LED-specific decoders."""

    name = "camera-ocr"

    def __init__(
        self,
        roi: NormalizedROI,
        unit: str = "kg",
        min_confidence: float = 0.60,
        download_enabled: bool = False,
        gpu: bool = False,
        reader: object | None = None,
    ):
        if not 0 <= min_confidence <= 1:
            raise ValueError("OCR minimum confidence must be between 0 and 1")
        self.roi = roi
        self.unit = unit
        self.min_confidence = min_confidence
        self.download_enabled = download_enabled
        self.gpu = gpu
        self._reader = reader
        self._last = WeightReading(None, unit, False, "")
        self._temporal_candidate: WeightReading | None = None

    def _get_reader(self) -> object:
        if self._reader is None:
            self._reader = PaddleOCRTextReader.create(
                download_enabled=self.download_enabled,
                gpu=self.gpu,
            )
        return self._reader

    def crop(self, frame: np.ndarray) -> np.ndarray:
        left, top, right, bottom = self.roi.pixels(frame)
        return frame[top:bottom, left:right]

    @staticmethod
    def _preprocess(
        crop: np.ndarray,
        core_thresholds: tuple[int, int] = (210, 80),
    ) -> np.ndarray:
        if crop.ndim == 3:
            bright_core = _bright_led_core_mask(crop, *core_thresholds)
            core_fraction = float(np.mean(bright_core > 0))
            if 0.005 <= core_fraction <= 0.35:
                scale = min(8.0, max(2.0, 320 / max(1, bright_core.shape[0])))
                gray = cv2.resize(
                    bright_core,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                border = max(8, round(scale * 5))
                return cv2.copyMakeBorder(
                    gray,
                    border,
                    border,
                    border,
                    border,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            blue, green, red = cv2.split(crop[:, :, :3])
            red_i16 = red.astype(np.int16)
            other = np.maximum(green, blue).astype(np.int16)
            led_mask = ((red_i16 >= 120) & (red_i16 - other >= 45)).astype(np.uint8) * 255
            led_fraction = float(np.mean(led_mask > 0))
            if 0.002 <= led_fraction <= 0.45:
                gray = cv2.resize(
                    led_mask,
                    None,
                    fx=min(8.0, max(2.0, 160 / max(1, led_mask.shape[0]))),
                    fy=min(8.0, max(2.0, 160 / max(1, led_mask.shape[0]))),
                    interpolation=cv2.INTER_CUBIC,
                )
                scale = gray.shape[0] / max(1, led_mask.shape[0])
                border = max(8, round(scale * 4))
                return cv2.copyMakeBorder(
                    gray,
                    border,
                    border,
                    border,
                    border,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            else:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
        target_height = max(96, gray.shape[0])
        scale = min(6.0, max(1.0, target_height / max(1, gray.shape[0])))
        if scale > 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    @staticmethod
    def _preprocess_green_otsu(
        crop: np.ndarray,
        *,
        target_height: int,
    ) -> np.ndarray | None:
        """Isolate the white/yellow LED core without joining its red halo.

        Small factory displays can occupy only about fifty pixels in a portrait
        frame.  A fixed RGB threshold then either drops thin segments or merges
        adjacent glyphs through the red glow.  The green channel carries the
        saturated segment core while largely rejecting that glow.  Otsu keeps
        the branch exposure-adaptive; two resize scales are voted independently
        by ``capture`` instead of trusting one interpolation result.
        """

        if crop.ndim != 3 or crop.shape[2] < 3 or crop.size == 0:
            return None
        green = crop[:, :, 1]
        if int(green.max()) - int(green.min()) < 12:
            return None
        _, mask = cv2.threshold(
            green,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        active_fraction = float(np.mean(mask > 0))
        if not 0.005 <= active_fraction <= 0.55:
            return None
        scale = min(8.0, max(1.0, target_height / max(1, mask.shape[0])))
        resized = cv2.resize(
            mask,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        border = max(8, round(resized.shape[0] * 0.125))
        return cv2.copyMakeBorder(
            resized,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=0,
        )

    @staticmethod
    def _preprocess_channel_clahe(
        crop: np.ndarray,
        *,
        channel: str,
    ) -> np.ndarray | None:
        """Preserve soft LED edges that binary masks erase on blurred photos."""

        if crop.ndim != 3 or crop.shape[2] < 3 or crop.size == 0:
            return None
        if channel == "gray":
            image = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
        elif channel == "green":
            image = crop[:, :, 1]
        else:
            raise ValueError("unsupported CLAHE channel")
        if int(image.max()) - int(image.min()) < 12:
            return None
        enhanced = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(4, 4),
        ).apply(image)
        return cv2.resize(
            enhanced,
            None,
            fx=4.0,
            fy=4.0,
            interpolation=cv2.INTER_CUBIC,
        )

    def capture(
        self,
        frame: np.ndarray,
        *,
        _recognized_paddle_results: list[tuple[object, str, float]] | None = None,
    ) -> WeightReading:
        self._temporal_candidate = None
        crop = self.crop(frame)
        if crop.size == 0:
            self._last = WeightReading(None, self.unit, False, "OCR: empty ROI")
            return self._last

        primary_core = _bright_led_core_mask(crop)
        primary_core_fraction = float(np.mean(primary_core > 0))
        fixed_led_evidence = 0.005 <= primary_core_fraction <= 0.35
        blue, green, red = cv2.split(crop[:, :, :3])
        red_i16 = red.astype(np.int16)
        other = np.maximum(green, blue).astype(np.int16)
        red_led_fraction = float(np.mean((red_i16 >= 120) & (red_i16 - other >= 45)))
        red_led_evidence = 0.002 <= red_led_fraction <= 0.75
        digit_height = _red_led_digit_height(crop)
        if red_led_evidence and digit_height < MIN_LED_DIGIT_HEIGHT:
            self._last = WeightReading(
                None,
                self.unit,
                False,
                (
                    f"OCR: LED height {digit_height}px below safe minimum "
                    f"{MIN_LED_DIGIT_HEIGHT}px"
                ),
            )
            return self._last
        bright_core_result = (
            _decode_bright_core_digits(crop) if fixed_led_evidence else None
        )
        decoded_digits = bright_core_result[0] if bright_core_result is not None else None
        reader = self._get_reader()
        if isinstance(reader, PaddleOCRTextReader):
            # PP-OCRv6 medium is trained for industrial digital displays and
            # performs best on the original colour ROI. The older binary
            # branches erase the leading 7 on the small factory fixtures.
            variants = [("paddle-raw", crop)]
        else:
            variants = [("main", self._preprocess(crop))]
        if fixed_led_evidence and not isinstance(reader, PaddleOCRTextReader):
            secondary_core = _bright_led_core_mask(crop, 215, 85)
            secondary_fraction = float(np.mean(secondary_core > 0))
            if 0.005 <= secondary_fraction <= 0.35:
                variants.append(("core215", self._preprocess(crop, (215, 85))))
            # Multi-scale Otsu branches recover tiny saturated segments that
            # disappear in one fixed core mask.  They become one trusted vote
            # only when a strict majority agrees.
            for target_height in (128, 160, 224):
                green_otsu = self._preprocess_green_otsu(
                    crop,
                    target_height=target_height,
                )
                if green_otsu is not None:
                    variants.append((f"green{target_height}", green_otsu))
        if red_led_evidence and not isinstance(reader, PaddleOCRTextReader):
            # Binary LED masks are strong on crisp frames but can turn a
            # bloomed 7.84 into 7.04.  Grayscale and green CLAHE retain the
            # soft gaps independently.  Neither branch is trusted alone; the
            # existing fixed-layout grouping requires their agreement.
            for channel in ("gray", "green"):
                channel_clahe = self._preprocess_channel_clahe(
                    crop,
                    channel=channel,
                )
                if channel_clahe is not None:
                    variants.append((f"cla-{channel}", channel_clahe))

        recognized: list[tuple[str, object]] = []
        recognize_batch = getattr(reader, "recognize_batch", None)
        if _recognized_paddle_results is not None:
            if not isinstance(reader, PaddleOCRTextReader):
                raise ValueError("precomputed Paddle results require PaddleOCRTextReader")
            recognized.extend(
                ("paddle-raw", result)
                for result in _recognized_paddle_results
            )
        elif callable(recognize_batch):
            batches = recognize_batch(
                [image for _, image in variants],
                allowlist=OCR_ALLOWLIST,
                detail=1,
                paragraph=False,
                reformat=False,
            )
            for (variant_name, _), results in zip(variants, batches):
                recognized.extend((variant_name, result) for result in results)
        else:
            for variant_name, image in variants:
                height, width = image.shape[:2]
                for result in reader.recognize(
                    image,
                    horizontal_list=[[0, width, 0, height]],
                    free_list=[],
                    allowlist=OCR_ALLOWLIST,
                    detail=1,
                    paragraph=False,
                    reformat=False,
                ):
                    recognized.append((variant_name, result))

        candidates: list[tuple[float, int, str, float, str, bool, bool]] = []
        green_votes: list[tuple[float, int, str, float, str, bool, bool]] = []
        explicit_decimal_keys: set[tuple[float, str]] = set()
        raw_parts: list[str] = []
        if bright_core_result is not None:
            bright_text, bright_confidence = bright_core_result
            restored_bright_text = _restore_fixed_scale_decimal(bright_text, crop)
            parsed_bright = parse_ocr_weight(restored_bright_text, self.unit)
            raw_parts.append(
                f"LEDCORE:{restored_bright_text}@{bright_confidence:.3f}"
            )
            if (
                parsed_bright is not None
                and bright_confidence >= BRIGHT_LED_MIN_CONFIDENCE
            ):
                bright_value, bright_unit = parsed_bright
                candidates.append(
                    (
                        bright_confidence,
                        sum(character.isdigit() for character in restored_bright_text),
                        restored_bright_text,
                        bright_value,
                        bright_unit,
                        True,
                        bright_confidence >= BRIGHT_LED_TRUST_CONFIDENCE,
                    )
                )
        segment_candidate: tuple[float, int, str, float, str] | None = None
        segment_result = _decode_seven_segment(crop)
        if segment_result is not None:
            segment_text, segment_confidence = segment_result
            parsed_segment = parse_ocr_weight(segment_text, self.unit)
            raw_parts.append(f"7SEG:{segment_text}@{segment_confidence:.3f}")
            if parsed_segment is not None and segment_confidence >= self.min_confidence:
                segment_value, segment_unit = parsed_segment
                segment_candidate = (
                    segment_confidence,
                    sum(character.isdigit() for character in segment_text),
                    segment_text,
                    segment_value,
                    segment_unit,
                )
        for variant_name, result in recognized:
            if not isinstance(result, (list, tuple)) or len(result) < 3:
                continue
            text = str(result[1]).strip()
            try:
                confidence = float(result[2])
            except (TypeError, ValueError):
                continue
            # Green/Otsu and CLAHE votes independently check LEDCORE geometry.
            # Do not feed a complete LEDCORE string back into them.  The only
            # exception is the narrowly scoped physical 0/8 ambiguity: a
            # multi-threshold segment vote can safely remove a reflected middle
            # bar without allowing one bad geometry guess to rewrite the row.
            corrected_text = (
                text
                if variant_name.startswith("green") or variant_name.startswith("cla-")
                else _correct_bright_led_confusions(
                    text,
                    crop,
                    decoded_digits,
                )
            )
            corrected_text = _correct_bright_led_zero_eight_confusions(
                corrected_text,
                crop,
            )
            restored_text = _restore_led_decimal(corrected_text, crop)
            if restored_text == corrected_text:
                restored_text = _restore_fixed_scale_decimal(corrected_text, crop)
            if text:
                display_text = (
                    f"{text}->{restored_text}" if restored_text != text else text
                )
                prefix = (
                    f"{variant_name}:"
                    if len(variants) > 1 or variant_name == "paddle-raw"
                    else ""
                )
                raw_parts.append(f"{prefix}{display_text}@{confidence:.3f}")
            parsed = parse_ocr_weight(restored_text, self.unit)
            fixed_led_layout = (
                restored_text != corrected_text
                and restored_text == _restore_fixed_scale_decimal(corrected_text, crop)
            )
            digit_count = sum(character.isdigit() for character in restored_text)
            led_layout_candidate = (
                red_led_evidence
                and digit_count >= 3
                and (fixed_led_layout or "." in restored_text or "," in restored_text)
            )
            required_confidence = (
                self.min_confidence
                if variant_name.startswith("green")
                else min(self.min_confidence, BRIGHT_LED_MIN_CONFIDENCE)
                if fixed_led_evidence and led_layout_candidate
                else self.min_confidence
            )
            if (
                parsed is None
                or confidence < required_confidence
                or (red_led_evidence and not led_layout_candidate)
            ):
                continue
            value, unit = parsed
            if (
                ("." in text or "," in text)
                and confidence >= self.min_confidence
            ):
                explicit_decimal_keys.add((round(value, 6), unit))
            candidate = (
                confidence,
                digit_count,
                restored_text,
                value,
                unit,
                led_layout_candidate,
                (
                    variant_name == "paddle-raw"
                    and confidence >= PADDLE_OCR_TRUST_CONFIDENCE
                ),
            )
            if variant_name.startswith("green"):
                green_votes.append(candidate)
            else:
                candidates.append(candidate)

        if green_votes:
            green_groups: dict[
                tuple[float, str],
                list[tuple[float, int, str, float, str, bool, bool]],
            ] = defaultdict(list)
            for item in green_votes:
                green_groups[(round(item[3], 6), item[4])].append(item)
            green_group = max(
                green_groups.values(),
                key=lambda group: (len(group), max(item[0] for item in group)),
            )
            if len(green_group) >= 2 and len(green_group) * 2 > len(green_votes):
                green_selected = max(green_group, key=lambda item: item[0])
                green_confidence = float(np.mean([item[0] for item in green_group]))
                candidates.append(
                    (
                        green_confidence,
                        green_selected[1],
                        green_selected[2],
                        green_selected[3],
                        green_selected[4],
                        True,
                        True,
                    )
                )
                raw_parts.append(
                    f"GREENCONS:{green_selected[2]}@{green_confidence:.3f}"
                    f"({len(green_group)}/{len(green_votes)})"
                )

        raw = "OCR: " + ("; ".join(raw_parts) if raw_parts else "no text")
        # The handcrafted decoder is useful when OCR returns nothing, but a
        # marginal seven-segment match must not overwrite readable OCR text.
        if candidates:
            fixed_candidates = [item for item in candidates if item[5]]
            if fixed_candidates:
                grouped: dict[tuple[float, str], list[tuple[float, int, str, float, str, bool, bool]]] = defaultdict(list)
                for item in fixed_candidates:
                    grouped[(round(item[3], 6), item[4])].append(item)
                trusted_keys = {
                    key
                    for key, group in grouped.items()
                    if any(item[6] for item in group)
                }
                if len(trusted_keys) > 1:
                    self._last = WeightReading(
                        None,
                        self.unit,
                        False,
                        raw + "; trusted fixed-layout conflict",
                    )
                    return self._last
                if trusted_keys:
                    trusted_key = next(iter(trusted_keys))
                    # Two agreeing OCR branches are meaningful contrary
                    # evidence even when a geometry decoder is individually
                    # confident.  Fail closed instead of choosing either side.
                    if any(
                        key != trusted_key and len(group) >= 2
                        for key, group in grouped.items()
                    ):
                        self._last = WeightReading(
                            None,
                            self.unit,
                            False,
                            raw + "; confirmed fixed-layout conflict",
                        )
                        return self._last
                    winning_group = grouped[trusted_key]
                else:
                    winning_key, winning_group = max(
                        grouped.items(),
                        key=lambda entry: (
                            len(entry[1]),
                            max(item[0] for item in entry[1]),
                        ),
                    )
                    # A lone inferred fixed-layout reading is unsafe: blur can
                    # turn 9 into 3 while keeping deceptively high shape
                    # confidence.  OCR that directly saw the decimal in the
                    # source text has independent layout evidence and may
                    # stand alone after passing the normal confidence gate.
                    if (
                        len(winning_group) < 2
                        and winning_key not in explicit_decimal_keys
                    ):
                        self._last = WeightReading(
                            None,
                            self.unit,
                            False,
                            raw + "; unconfirmed fixed-layout reading",
                        )
                        return self._last
                selected = max(
                    winning_group,
                    key=lambda item: (item[0], item[1]),
                )
            else:
                selected = max(
                    candidates,
                    key=lambda item: (item[0], item[1]),
                )
            confidence, _, text, value, unit, _, _ = selected
        elif (
            segment_candidate is not None
            and segment_candidate[0] >= SEVEN_SEGMENT_OVERRIDE_CONFIDENCE
        ):
            confidence, _, text, value, unit = segment_candidate
        else:
            self._last = WeightReading(None, self.unit, False, raw)
            return self._last
        self._last = WeightReading(
            value,
            unit,
            True,
            f"{raw}; selected={text}; confidence={confidence:.3f}",
            confidence,
        )
        self._temporal_candidate = self._last
        return self._last

    def capture_many(
        self,
        frames: list[np.ndarray] | tuple[np.ndarray, ...],
        *,
        min_agreement: float = TEMPORAL_MIN_AGREEMENT,
    ) -> WeightReading:
        """Read a short camera burst and fail closed on unstable digits.

        Scale LEDs are commonly multiplexed. A single exposure can therefore
        lose one segment even when the camera and scale are stationary. The
        Paddle path samples first/middle/last and recognizes all three in one
        batch. Legacy injected readers retain the percentile-fusion path used
        by deterministic decoder tests. Original frames must agree on the
        exact numeric value before a reading is returned.
        """

        usable = [frame for frame in frames if isinstance(frame, np.ndarray) and frame.size]
        if not usable:
            self._last = WeightReading(None, self.unit, False, "TEMPORAL: no frames")
            return self._last
        if len(usable) == 1:
            return self.capture(usable[0])
        if len(usable) < 3:
            self._last = WeightReading(
                None,
                self.unit,
                False,
                "TEMPORAL: need at least 3 distinct camera frames",
            )
            return self._last
        if not 0.5 <= min_agreement <= 1.0:
            raise ValueError("temporal minimum agreement must be between 0.5 and 1")

        reader = self._get_reader()
        paddle_batch = isinstance(reader, PaddleOCRTextReader)
        sample_limit = (
            PADDLE_BATCH_TEMPORAL_FRAMES
            if paddle_batch
            else MAX_TEMPORAL_OCR_FRAMES
        )
        if len(usable) <= sample_limit:
            sampled = usable
        else:
            sample_indices = sorted({
                round(index)
                for index in np.linspace(
                    0,
                    len(usable) - 1,
                    sample_limit,
                )
            })
            sampled = [usable[index] for index in sample_indices]
        if paddle_batch:
            crops = [self.crop(frame) for frame in sampled]
            batches = reader.recognize_batch(
                crops,
                allowlist=OCR_ALLOWLIST,
                detail=1,
                paragraph=False,
                reformat=False,
            )
            frame_readings = [
                self.capture(frame, _recognized_paddle_results=results)
                for frame, results in zip(sampled, batches)
            ]
            while len(frame_readings) < len(sampled):
                frame_readings.append(
                    WeightReading(None, self.unit, False, "PADDLE BATCH: missing result")
                )
            fused_reading = WeightReading(
                None,
                self.unit,
                False,
                "TEMPORAL FUSION: skipped for Paddle batch",
            )
        else:
            frame_readings = [self.capture(frame) for frame in sampled]
            fused_reading = self._capture_temporal_fusion(usable)
        self._temporal_candidate = None
        valid = [reading for reading in frame_readings if reading.value is not None]
        groups: dict[tuple[float, str], list[tuple[int, WeightReading]]] = defaultdict(list)
        for index, reading in enumerate(frame_readings):
            if reading.value is None:
                continue
            groups[(round(reading.value, 6), reading.unit)].append((index, reading))

        compact_votes = ",".join(
            f"{key[0]:g}{key[1]}x{len(group)}"
            for key, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
        ) or "none"
        fused_text = (
            "skipped"
            if paddle_batch
            else "none"
            if fused_reading.value is None
            else f"{fused_reading.value:g}{fused_reading.unit}"
        )
        prefix = (
            f"TEMPORAL: engine={'paddle-batch' if paddle_batch else 'legacy'}; "
            f"frames={len(usable)}; sampled={len(sampled)}; "
            f"valid={len(valid)}; "
            f"votes={compact_votes}; fused={fused_text}"
        )
        if not groups:
            self._last = WeightReading(None, self.unit, False, prefix)
            return self._last

        winning_key, winning_group = max(
            groups.items(),
            key=lambda item: (
                len(item[1]),
                float(np.mean([
                    pair[1].confidence or 0.0 for pair in item[1]
                ])),
            ),
        )
        winning_count = len(winning_group)
        runner_up_count = max(
            (len(group) for key, group in groups.items() if key != winning_key),
            default=0,
        )
        required_total = max(3, math.ceil(len(sampled) * min_agreement))
        required_total = min(len(sampled), required_total)
        valid_agreement = winning_count / max(1, len(valid))
        winning_indices = {index for index, _ in winning_group}
        longest_run = 0
        current_run = 0
        for index in range(len(sampled)):
            if index in winning_indices:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0
        required_run = min(TEMPORAL_MIN_CONSECUTIVE, required_total)
        fused_conflict = (
            fused_reading.value is not None
            and (round(fused_reading.value, 6), fused_reading.unit) != winning_key
        )
        candidate_confidences = [
            reading.confidence
            for _, reading in winning_group
            if reading.confidence is not None
        ]
        candidate_confidence = (
            float(np.mean(candidate_confidences))
            if candidate_confidences
            else 0.0
        )
        candidate_value = winning_group[0][1].value
        if (
            candidate_value is not None
            and winning_count >= 2
            and winning_count > runner_up_count
        ):
            self._temporal_candidate = WeightReading(
                candidate_value,
                winning_key[1],
                False,
                (
                    f"LOCAL CANDIDATE: {candidate_value:g}{winning_key[1]}; "
                    f"votes={winning_count}/{len(sampled)}"
                ),
                candidate_confidence,
            )
        if (
            winning_count < required_total
            or valid_agreement < TEMPORAL_MIN_VALID_AGREEMENT
            or longest_run < required_run
            or fused_conflict
        ):
            reasons = []
            if winning_count < required_total:
                reasons.append(f"need {required_total} matching frames")
            if valid_agreement < TEMPORAL_MIN_VALID_AGREEMENT:
                reasons.append("conflicting valid frames")
            if longest_run < required_run:
                reasons.append(f"need {required_run} consecutive frames")
            if fused_conflict:
                reasons.append("fused LED conflict")
            self._last = WeightReading(
                None,
                self.unit,
                False,
                prefix + "; rejected=" + ", ".join(reasons),
            )
            return self._last

        confidences = [
            reading.confidence
            for _, reading in winning_group
            if reading.confidence is not None
        ]
        mean_confidence = float(np.mean(confidences)) if confidences else 0.0
        total_agreement = winning_count / len(sampled)
        confidence = min(mean_confidence, 0.50 + 0.50 * total_agreement)
        value = winning_group[0][1].value
        assert value is not None
        self._last = WeightReading(
            value,
            winning_key[1],
            True,
            (
                prefix
                + f"; selected={value:g}; agreement={winning_count}/{len(sampled)}"
                + f"; consecutive={longest_run}; confidence={confidence:.3f}"
            ),
            confidence,
        )
        self._temporal_candidate = self._last
        return self._last

    def candidate_reading(self) -> WeightReading | None:
        """Return a unique repeated local candidate for cloud cross-checking."""

        return self._temporal_candidate

    def _capture_temporal_fusion(self, frames: list[np.ndarray]) -> WeightReading:
        """Decode a percentile fusion of aligned, normalized LED crops."""

        crops = [self.crop(frame) for frame in frames]
        crops = [crop for crop in crops if crop.size]
        if not crops:
            return WeightReading(None, self.unit, False, "TEMPORAL FUSION: empty ROI")
        target_height, target_width = crops[0].shape[:2]
        normalized = [
            crop
            if crop.shape[:2] == (target_height, target_width)
            else cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_AREA)
            for crop in crops
        ]
        # The 75th percentile retains a segment illuminated in at least a
        # quarter of the burst while rejecting one-frame sensor noise.  All
        # channels are fused together so the red/yellow LED colour relation is
        # preserved for the existing specialized decoders.
        fused_crop = np.percentile(
            np.stack(normalized, axis=0).astype(np.float32),
            75,
            axis=0,
        ).astype(np.uint8)
        fused_frame = frames[0].copy()
        left, top, right, bottom = self.roi.pixels(fused_frame)
        if fused_crop.shape[:2] != (bottom - top, right - left):
            fused_crop = cv2.resize(
                fused_crop,
                (right - left, bottom - top),
                interpolation=cv2.INTER_AREA,
            )
        fused_frame[top:bottom, left:right] = fused_crop
        return self.capture(fused_frame)

    def reading(self) -> WeightReading:
        return self._last

    def reset(self) -> None:
        self._last = WeightReading(None, self.unit, False, "")
        self._temporal_candidate = None

    def handle_key(self, key: int) -> bool:
        return False

    def draw_roi(self, frame: np.ndarray) -> None:
        left, top, right, bottom = self.roi.pixels(frame)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 255), 2)
        cv2.putText(
            frame,
            "WEIGHT OCR ROI",
            (left + 6, max(24, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )

    def close(self) -> None:
        return None
