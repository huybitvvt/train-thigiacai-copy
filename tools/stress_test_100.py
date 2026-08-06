from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import qrcode

from roll_qr_scale.api_client import post_measurement
from roll_qr_scale.lookup_client import lookup_roll
from roll_qr_scale.quality import assess_frame_quality
from roll_qr_scale.qr_reader import QRReader
from roll_qr_scale.storage import Measurement, MeasurementStore
from roll_qr_scale.sync import OutboxSyncWorker
from roll_qr_scale.weight_ocr import CameraOCRWeightSource, detect_weight_roi


SUITE = "STRESS_100"
CASE_COUNT = 100
OCR_MIN_CONFIDENCE = 0.60
WEIGHT_TOLERANCE = 0.001
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs" / "stress_100"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")
REQUIRED_UPLOAD_ENV = (
    "ROLL_SCALE_API_URL",
    "ROLL_SCALE_DEVICE_TOKEN",
    "ROLL_SCALE_LOOKUP_URL",
    "ROLL_SCALE_LOOKUP_TOKEN",
)


@dataclass(frozen=True)
class BaseCase:
    name: str
    relative_path: str
    expected_weight: float
    qr_center_x: float
    qr_center_y: float
    qr_size_fraction: float = 0.24


BASES = (
    BaseCase(
        "warehouse_scale_demo",
        "data/warehouse_scale_demo.png",
        20.15,
        0.333,
        0.200,
    ),
    BaseCase(
        "factory_capture_20260802_175736",
        "data/captures/20260802_175736_868757_65389c95.jpg",
        7.02,
        0.510,
        0.460,
    ),
    BaseCase(
        "factory_scale_7_02_full_reference",
        # Immutable full-frame factory photo confirmed by the operator as
        # 7.02 kg. This exposure previously produced a low-confidence 2.82.
        "data/factory_scale_7_02_full_reference.jpg",
        7.02,
        0.510,
        0.456,
    ),
    BaseCase(
        "factory_scale_9_34_reference",
        # Never use the UI-overwritten factory_scale_diagnostic.jpg here.
        # Acceptance stays on the immutable, human-verified 9.34 kg reference.
        "data/factory_scale_9_34_reference.jpg",
        9.34,
        0.510,
        0.450,
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _validated_run_id(value: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "run_id must be 1-48 characters and contain only letters, digits, '.', '_', or '-'"
        )
    return value


def _device_id(run_id: str) -> str:
    device_id = f"stress-100-{_validated_run_id(run_id)}"
    if len(device_id) > 64:
        raise ValueError("derived stress device_id exceeds 64 characters")
    return device_id


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _qr_image(value: str, target_size: int) -> np.ndarray:
    code = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=8,
        border=4,
    )
    code.add_data(value)
    code.make(fit=True)
    rgb = np.asarray(
        code.make_image(fill_color="black", back_color="white").convert("RGB")
    )
    return cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST,
    )


def _overlay_unique_qr(frame: np.ndarray, value: str, base: BaseCase) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    size = max(180, round(min(height, width) * base.qr_size_fraction))
    size = min(size, height, width)
    left = round(width * base.qr_center_x - size / 2)
    top = round(height * base.qr_center_y - size / 2)
    left = min(max(0, left), width - size)
    top = min(max(0, top), height - size)
    frame[top : top + size, left : left + size] = _qr_image(value, size)
    return left, top, left + size, top + size


def _augmentation_parameters(case_index: int) -> dict[str, float | int]:
    """Return the fixed combined-stress envelope used by the default gate."""
    return {
        "contrast": (0.990, 0.995, 1.005, 1.010)[case_index % 4],
        "brightness_delta": (-2, -1, 1, 2)[(case_index // 4) % 4],
        "blur_sigma": (0.20, 0.25, 0.30, 0.35)[(case_index // 16) % 4],
        "scale": (0.985, 0.995, 1.005, 1.015)[(case_index // 64) % 4],
        "jpeg_quality": (97, 98, 99, 96)[(case_index // 8) % 4],
    }


def _augment_frame(
    frame: np.ndarray,
    parameters: dict[str, float | int],
) -> tuple[np.ndarray, bytes]:
    contrast = float(parameters["contrast"])
    brightness = int(parameters["brightness_delta"])
    adjusted = np.clip(
        frame.astype(np.float32) * contrast + brightness,
        0,
        255,
    ).astype(np.uint8)
    adjusted = cv2.GaussianBlur(
        adjusted,
        (3, 3),
        float(parameters["blur_sigma"]),
    )
    height, width = adjusted.shape[:2]
    scale = float(parameters["scale"])
    interpolation = cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_AREA
    adjusted = cv2.resize(
        adjusted,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=interpolation,
    )
    ok, encoded = cv2.imencode(
        ".jpg",
        adjusted,
        [cv2.IMWRITE_JPEG_QUALITY, int(parameters["jpeg_quality"])],
    )
    if not ok:
        raise OSError("OpenCV could not encode an augmented stress frame")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise OSError("OpenCV could not decode an augmented stress frame")
    return decoded, encoded.tobytes()


def _roi_as_dict(located: object) -> dict[str, object] | None:
    if located is None:
        return None
    roi, method = located  # type: ignore[misc]
    return {
        "x1": float(roi.x1),
        "y1": float(roi.y1),
        "x2": float(roi.x2),
        "y2": float(roi.y2),
        "method": str(method),
    }


def _synthetic_proof_frame(qr_value: str, weight: float) -> np.ndarray:
    """Create visibly synthetic, non-sensitive evidence safe for a public image host."""
    frame = np.full((720, 960, 3), 220, dtype=np.uint8)
    for y in range(0, frame.shape[0], 40):
        cv2.line(frame, (0, y), (frame.shape[1] - 1, y), (205, 205, 205), 1)
    cv2.putText(
        frame,
        "SYNTHETIC STRESS PROOF",
        (165, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "NOT A FACTORY CAPTURE",
        (230, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (50, 50, 50),
        2,
        cv2.LINE_AA,
    )
    frame[105:405, 330:630] = _qr_image(qr_value, 300)
    cv2.rectangle(frame, (170, 490), (790, 690), (16, 18, 20), -1)
    cv2.rectangle(frame, (170, 490), (790, 690), (80, 85, 90), 5)

    text = f"{weight:.2f}"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 3.0
    thickness = 6
    (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    left = (frame.shape[1] - text_width) // 2
    baseline = 620
    glow = np.zeros_like(frame)
    cv2.putText(
        glow,
        text,
        (left, baseline),
        font,
        font_scale,
        (20, 20, 255),
        thickness + 7,
        cv2.LINE_AA,
    )
    glow = cv2.GaussianBlur(glow, (0, 0), 5)
    cv2.addWeighted(frame, 1.0, glow, 0.45, 0, frame)
    cv2.putText(
        frame,
        text,
        (left, baseline),
        font,
        font_scale,
        (30, 45, 255),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "kg",
        (700, 650),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 45, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _read_weight(
    frame: np.ndarray,
    reader: object | None,
) -> tuple[object | None, object | None, dict[str, object] | None]:
    located = detect_weight_roi(frame)
    if located is None:
        return None, reader, None
    source = CameraOCRWeightSource(
        located[0],
        unit="kg",
        min_confidence=OCR_MIN_CONFIDENCE,
        download_enabled=False,
        reader=reader,
    )
    try:
        reading = source.capture(frame)
    finally:
        reader = source._reader
    return reading, reader, _roi_as_dict(located)


def _recognition_result(
    frame: np.ndarray,
    expected_qr: str,
    expected_weight: float,
    qr_reader: QRReader,
    ocr_reader: object | None,
) -> tuple[dict[str, object], object | None]:
    quality = assess_frame_quality(frame)
    detections = qr_reader.decode(frame)
    reading, ocr_reader, roi = _read_weight(frame, ocr_reader)
    qr_values = [item.value for item in detections]
    predicted_qr = qr_values[0] if qr_values else None
    predicted_weight = getattr(reading, "value", None) if reading is not None else None
    ocr_confidence = (
        getattr(reading, "confidence", None) if reading is not None else None
    )
    weight_stable = bool(getattr(reading, "stable", False)) if reading is not None else False
    weight_confident = (
        ocr_confidence is not None
        and math.isfinite(float(ocr_confidence))
        and float(ocr_confidence) >= OCR_MIN_CONFIDENCE
    )
    qr_exact = predicted_qr == expected_qr and len(qr_values) == 1
    weight_exact = (
        predicted_weight is not None
        and math.isfinite(float(predicted_weight))
        and abs(float(predicted_weight) - expected_weight) <= WEIGHT_TOLERANCE
    )
    weight_gate_passed = (
        predicted_weight is not None and weight_stable and weight_confident
    )
    production_accepted = (
        quality.accepted and predicted_qr is not None and weight_gate_passed
    )
    exact = qr_exact and weight_exact
    return (
        {
            "quality": quality.as_dict(),
            "predicted_qr_code": predicted_qr,
            "all_predicted_qr_codes": qr_values,
            "qr_decoder": detections[0].decoder if detections else None,
            "predicted_weight": float(predicted_weight) if predicted_weight is not None else None,
            "ocr_confidence": ocr_confidence,
            "ocr_raw": getattr(reading, "raw", "") if reading is not None else "",
            "weight_stable": weight_stable,
            "weight_gate_passed": weight_gate_passed,
            "weight_roi": roi,
            "qr_exact": qr_exact,
            "weight_exact": weight_exact,
            "production_accepted": production_accepted,
            "wrong_accepted": production_accepted and not exact,
            "passed": quality.accepted and exact and weight_gate_passed,
        },
        ocr_reader,
    )


def _relative_display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def run_local(args: argparse.Namespace) -> int:
    run_id = _validated_run_id(args.run_id or _default_run_id())
    runs_root = Path(args.runs_root).resolve()
    data_root = Path(args.data_root).resolve()
    run_dir = runs_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"stress run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    local_cases_dir = run_dir / "local_cases"
    local_cases_dir.mkdir()
    store = MeasurementStore(run_dir / "outbox.db", run_dir / "synthetic_captures")
    report_path = run_dir / "local_report.json"

    report: dict[str, Any] = {
        "suite": SUITE,
        "phase": "local",
        "run_id": run_id,
        "created_at": _utc_now(),
        "accepted": False,
        "classification": "deterministic augmented smoke/stress; not held-out real acceptance",
        "security": {
            "local_case_artifacts": "factory-derived; local diagnostic use only; never uploaded",
            "cloud_artifact_kind": "synthetic",
            "cloud_artifacts_are_raw_factory_frames": False,
        },
        "criteria": {
            "exact_case_count": CASE_COUNT,
            "all_cases_must_pass": True,
            "zero_wrong_accepted": True,
            "exact_qr": True,
            "weight_tolerance": WEIGHT_TOLERANCE,
            "ocr_min_confidence": OCR_MIN_CONFIDENCE,
            "staged_pending_rows_required": CASE_COUNT,
        },
        "augmentation": {
            "combined": True,
            "deterministic": True,
            "operations": ["contrast", "brightness", "gaussian_blur", "scale", "jpeg"],
            "held_out_real_acceptance": False,
        },
        "bases": [
            {
                "name": base.name,
                "path": base.relative_path,
                "expected_weight": base.expected_weight,
                "scheduled_cases": sum(1 for index in range(CASE_COUNT) if BASES[index % len(BASES)] == base),
            }
            for base in BASES
        ],
        "cases": [],
        "staged_events": [],
    }

    qr_reader = QRReader(args.model, yolo_mode="fallback", yolo_imgsz=640)
    ocr_reader: object | None = None
    try:
        for zero_index in range(CASE_COUNT):
            case_number = zero_index + 1
            base = BASES[zero_index % len(BASES)]
            expected_qr = f"STRESS100-{run_id}-{case_number:03d}"
            base_path = data_root / base.relative_path
            case_path = local_cases_dir / f"{case_number:03d}.jpg"
            parameters = _augmentation_parameters(zero_index)
            case: dict[str, Any] = {
                "index": case_number,
                "base": base.name,
                "base_path": base.relative_path,
                "expected_qr_code": expected_qr,
                "expected_weight": base.expected_weight,
                "unit": "kg",
                "augmentation": parameters,
                "augmented": True,
                "held_out_real_acceptance": False,
                "local_case_path": _relative_display(case_path, run_dir),
                "local_case_upload_allowed": False,
                "passed": False,
            }
            try:
                frame = cv2.imread(str(base_path))
                if frame is None:
                    raise ValueError(f"cannot read base image: {base.relative_path}")
                qr_box = _overlay_unique_qr(frame, expected_qr, base)
                augmented, encoded = _augment_frame(frame, parameters)
                case_path.write_bytes(encoded)
                case["qr_overlay_box_before_scale"] = list(qr_box)
                recognition, ocr_reader = _recognition_result(
                    augmented,
                    expected_qr,
                    base.expected_weight,
                    qr_reader,
                    ocr_reader,
                )
                case.update(recognition)
                case["recognition_passed"] = bool(recognition["passed"])

                if recognition["passed"]:
                    proof = _synthetic_proof_frame(expected_qr, base.expected_weight)
                    proof_result, ocr_reader = _recognition_result(
                        proof,
                        expected_qr,
                        base.expected_weight,
                        qr_reader,
                        ocr_reader,
                    )
                    case["synthetic_proof"] = {
                        "cloud_artifact_kind": "synthetic",
                        **proof_result,
                    }
                    if proof_result["passed"]:
                        marker = (
                            f"{SUITE};cloud_artifact_kind=synthetic;case={case_number:03d};"
                            f"local={recognition['ocr_raw']};proof={proof_result['ocr_raw']}"
                        )
                        saved = store.save(
                            expected_qr,
                            base.expected_weight,
                            "kg",
                            proof,
                            f"{SUITE}:camera-ocr",
                            needs_sync=True,
                            qr_source=f"{SUITE}:{proof_result['qr_decoder']}",
                            weight_raw=marker,
                            weight_stable=True,
                        )
                        staged = {
                            "index": case_number,
                            "event_id": saved.event_id,
                            "qr_code": saved.qr_code,
                            "weight": saved.weight,
                            "unit": saved.unit,
                            "image_path": _relative_display(Path(saved.image_path), run_dir),
                            "cloud_artifact_kind": "synthetic",
                            "needs_sync": True,
                        }
                        case["staged_event"] = staged
                        report["staged_events"].append(staged)
                        case["passed"] = True
                    else:
                        case["passed"] = False
                        case["error"] = "synthetic proof did not pass production readers"
            except Exception as exc:
                case["error"] = f"{type(exc).__name__}: {exc}"
                case["passed"] = False
            report["cases"].append(case)

        passed = sum(bool(case["passed"]) for case in report["cases"])
        wrong_accepted = sum(bool(case.get("wrong_accepted")) for case in report["cases"])
        recognition_passed = sum(
            bool(case.get("recognition_passed")) for case in report["cases"]
        )
        pending_rows = store.pending_count()
        stored_rows = store.count()
        report.update(
            {
                "completed_at": _utc_now(),
                "total": len(report["cases"]),
                "recognition_passed": recognition_passed,
                "passed": passed,
                "failed": len(report["cases"]) - passed,
                "wrong_accepted": wrong_accepted,
                "stored_rows": stored_rows,
                "pending_rows": pending_rows,
                "accepted": (
                    len(report["cases"]) == CASE_COUNT
                    and passed == CASE_COUNT
                    and wrong_accepted == 0
                    and stored_rows == CASE_COUNT
                    and pending_rows == CASE_COUNT
                ),
            }
        )
    finally:
        store.close()
        _write_json(report_path, report)

    print(
        f"Local stress: {report.get('passed', 0)}/{report.get('total', CASE_COUNT)} passed; "
        f"wrong_accepted={report.get('wrong_accepted', 0)}; "
        f"pending={report.get('pending_rows', 0)}; report={report_path}"
    )
    return 0 if report["accepted"] else 2


def _required_upload_config() -> dict[str, str]:
    missing = [name for name in REQUIRED_UPLOAD_ENV if not os.environ.get(name)]
    if missing:
        raise ValueError("missing required environment variables: " + ", ".join(missing))
    return {name: os.environ[name] for name in REQUIRED_UPLOAD_ENV}


def _rows_by_status(store: MeasurementStore) -> dict[str, int]:
    rows = store.connection.execute(
        "SELECT sync_status, COUNT(*) AS total FROM measurements GROUP BY sync_status"
    ).fetchall()
    return {str(row["sync_status"]): int(row["total"]) for row in rows}


def _all_measurements(store: MeasurementStore) -> list[Measurement]:
    rows = store.connection.execute("SELECT * FROM measurements ORDER BY id").fetchall()
    return [store._from_row(row) for row in rows]


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _preflight_upload(
    report: dict[str, Any],
    run_id: str,
    run_dir: Path,
    store: MeasurementStore,
) -> list[Measurement]:
    if report.get("suite") != SUITE or report.get("phase") != "local":
        raise ValueError("local_report.json is not a STRESS_100 local report")
    if report.get("run_id") != run_id:
        raise ValueError("local report run_id does not match the requested run")
    if report.get("accepted") is not True:
        raise ValueError("local report is not accepted; upload is fail-closed")
    if (
        report.get("total") != CASE_COUNT
        or report.get("passed") != CASE_COUNT
        or report.get("wrong_accepted") != 0
    ):
        raise ValueError("local report does not prove 100/100 with zero wrong accepts")
    statuses = _rows_by_status(store)
    if statuses != {"pending": CASE_COUNT}:
        raise ValueError(f"outbox must contain exactly 100 pending rows; found statuses={statuses}")
    measurements = _all_measurements(store)
    if len(measurements) != CASE_COUNT:
        raise ValueError("outbox row count is not exactly 100")
    if len({item.event_id for item in measurements}) != CASE_COUNT:
        raise ValueError("outbox event_id values are not unique")
    if len({item.qr_code for item in measurements}) != CASE_COUNT:
        raise ValueError("outbox QR values are not unique")

    staged = report.get("staged_events")
    if not isinstance(staged, list) or len(staged) != CASE_COUNT:
        raise ValueError("local report does not list exactly 100 staged events")
    report_events = {str(item.get("event_id")) for item in staged if isinstance(item, dict)}
    if report_events != {item.event_id for item in measurements}:
        raise ValueError("local report event IDs do not match the outbox")

    capture_dir = run_dir / "synthetic_captures"
    for item in measurements:
        image_path = Path(item.image_path)
        if not _is_within(image_path, capture_dir) or not image_path.is_file():
            raise ValueError("outbox contains a missing or non-synthetic-capture image path")
        if not item.weight_source.startswith(f"{SUITE}:"):
            raise ValueError("outbox contains an unmarked weight source")
        if not item.qr_source.startswith(f"{SUITE}:"):
            raise ValueError("outbox contains an unmarked QR source")
        if f"{SUITE};cloud_artifact_kind=synthetic" not in item.weight_raw:
            raise ValueError("outbox contains an artifact without the synthetic STRESS_100 marker")
    return measurements


def _cloudinary_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "").lower() == "res.cloudinary.com"


def _head_url(url: str, timeout: float) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "image/*"},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            content_type = response.headers.get("content-type", "")
        return {
            "ok": 200 <= status < 400 and content_type.lower().startswith("image/"),
            "status": status,
            "content_type": content_type,
        }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": int(exc.code), "error": "HTTPError"}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def _lookup_check(
    measurement: Measurement,
    lookup_url: str,
    lookup_token: str,
    timeout: float,
) -> dict[str, object]:
    try:
        result = lookup_roll(lookup_url, measurement.qr_code, lookup_token, timeout=timeout)
        remote = result.get("measurement")
        if not result.get("found") or not isinstance(remote, dict):
            return {"ok": False, "error": "not_found"}
        checks = {
            "event_id": remote.get("event_id") == measurement.event_id,
            "qr_code": remote.get("qr_code") == measurement.qr_code,
            "weight": (
                remote.get("weight") is not None
                and abs(float(remote["weight"]) - measurement.weight) <= WEIGHT_TOLERANCE
            ),
            "unit": remote.get("unit") == measurement.unit,
            "image_url": remote.get("image_url") == measurement.remote_image_url,
            "image_public_id": remote.get("image_public_id")
            == measurement.remote_image_public_id,
        }
        return {"ok": all(checks.values()), "checks": checks}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def _parallel_checks(
    measurements: list[Measurement],
    lookup_url: str,
    lookup_token: str,
    timeout: float,
    workers: int,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    lookups: dict[str, dict[str, object]] = {}
    heads: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        lookup_futures = {
            executor.submit(
                _lookup_check,
                item,
                lookup_url,
                lookup_token,
                timeout,
            ): item.event_id
            for item in measurements
        }
        head_futures = {
            executor.submit(_head_url, str(item.remote_image_url), timeout): item.event_id
            for item in measurements
        }
        for future in as_completed(lookup_futures):
            lookups[lookup_futures[future]] = future.result()
        for future in as_completed(head_futures):
            heads[head_futures[future]] = future.result()
    return lookups, heads


def run_upload(args: argparse.Namespace) -> int:
    run_id = _validated_run_id(args.run_id)
    runs_root = Path(args.runs_root).resolve()
    run_dir = runs_root / run_id
    cloud_report_path = run_dir / "cloud_report.json"
    cloud_report: dict[str, Any] = {
        "suite": SUITE,
        "phase": "upload",
        "run_id": run_id,
        "created_at": _utc_now(),
        "accepted": False,
        "device_id": _device_id(run_id),
        "cloud_artifact_kind": "synthetic",
        "raw_factory_frames_uploaded": False,
    }
    store: MeasurementStore | None = None
    try:
        report_path = run_dir / "local_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"missing accepted local report: {report_path}")
        local_report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(local_report, dict):
            raise ValueError("local_report.json must contain a JSON object")
        store = MeasurementStore(run_dir / "outbox.db", run_dir / "synthetic_captures")
        pending = _preflight_upload(local_report, run_id, run_dir, store)
        config = _required_upload_config()

        def send(
            url: str,
            payload: dict[str, object],
            image_path: str,
            token: str,
        ) -> dict[str, object]:
            return post_measurement(url, payload, image_path, token, timeout=args.timeout)

        worker = OutboxSyncWorker(
            store,
            config["ROLL_SCALE_API_URL"],
            config["ROLL_SCALE_DEVICE_TOKEN"],
            _device_id(run_id),
            send=send,
        )
        canary_synced = worker.sync_once(limit=1)
        canary = store.get(pending[0].event_id)
        if canary_synced != 1 or canary is None or canary.sync_status != "synced":
            raise RuntimeError("cloud canary did not synchronize")
        canary_lookup = _lookup_check(
            canary,
            config["ROLL_SCALE_LOOKUP_URL"],
            config["ROLL_SCALE_LOOKUP_TOKEN"],
            args.timeout,
        )
        canary_head = _head_url(str(canary.remote_image_url), args.timeout)
        canary_ok = (
            canary.remote_id is not None
            and bool(canary.remote_image_url)
            and _cloudinary_url(str(canary.remote_image_url))
            and bool(canary.remote_image_public_id)
            and bool(canary_lookup.get("ok"))
            and bool(canary_head.get("ok"))
        )
        cloud_report["canary"] = {
            "event_id": canary.event_id,
            "synced": canary_synced == 1,
            "lookup": canary_lookup,
            "head": canary_head,
            "accepted": canary_ok,
        }
        if not canary_ok:
            raise RuntimeError("cloud canary verification failed; remaining 99 not sent")

        remaining_synced = worker.sync_once(limit=CASE_COUNT - 1)
        synced_now = canary_synced + remaining_synced
        synced = [store.get(item.event_id) for item in pending]
        measurements = [item for item in synced if item is not None]
        statuses = _rows_by_status(store)
        cloud_report["sync"] = {
            "attempted": CASE_COUNT,
            "synced_now": synced_now,
            "statuses": statuses,
        }
        if len(measurements) != CASE_COUNT or statuses != {"synced": CASE_COUNT}:
            raise RuntimeError("not all 100 outbox rows synchronized")
        if len({item.event_id for item in measurements}) != CASE_COUNT:
            raise RuntimeError("synchronized event IDs are not unique")
        if len({item.remote_id for item in measurements}) != CASE_COUNT or any(
            item.remote_id is None for item in measurements
        ):
            raise RuntimeError("remote measurement IDs are missing or not unique")
        if len({item.remote_image_url for item in measurements}) != CASE_COUNT or any(
            not item.remote_image_url or not _cloudinary_url(item.remote_image_url)
            for item in measurements
        ):
            raise RuntimeError("Cloudinary image URLs are missing, invalid, or not unique")
        if len({item.remote_image_public_id for item in measurements}) != CASE_COUNT or any(
            not item.remote_image_public_id for item in measurements
        ):
            raise RuntimeError("Cloudinary public IDs are missing or not unique")

        lookups, heads = _parallel_checks(
            measurements,
            config["ROLL_SCALE_LOOKUP_URL"],
            config["ROLL_SCALE_LOOKUP_TOKEN"],
            args.timeout,
            args.workers,
        )
        lookup_passed = sum(bool(item.get("ok")) for item in lookups.values())
        head_passed = sum(bool(item.get("ok")) for item in heads.values())
        cloud_report["lookup"] = {
            "checked": len(lookups),
            "passed": lookup_passed,
            "results_by_event_id": lookups,
        }
        cloud_report["head"] = {
            "checked": len(heads),
            "passed": head_passed,
            "results_by_event_id": heads,
        }

        first = measurements[0]
        duplicate = post_measurement(
            config["ROLL_SCALE_API_URL"],
            first.api_payload(_device_id(run_id)),
            first.image_path,
            config["ROLL_SCALE_DEVICE_TOKEN"],
            timeout=args.timeout,
        )
        duplicate_ok = (
            duplicate.get("duplicate") is True
            and str(duplicate.get("event_id")) == first.event_id
            and int(duplicate.get("id")) == first.remote_id
            and duplicate.get("image_url") == first.remote_image_url
            and duplicate.get("image_public_id") == first.remote_image_public_id
        )
        cloud_report["duplicate_repost"] = {
            "event_id": first.event_id,
            "duplicate": duplicate.get("duplicate") is True,
            "matched_original": duplicate_ok,
        }
        cloud_report["events"] = [
            {
                "event_id": item.event_id,
                "qr_code": item.qr_code,
                "weight": item.weight,
                "unit": item.unit,
                "remote_id": item.remote_id,
                "image_url": item.remote_image_url,
                "image_public_id": item.remote_image_public_id,
                "cloud_artifact_kind": "synthetic",
            }
            for item in measurements
        ]
        cloud_report["accepted"] = (
            lookup_passed == CASE_COUNT and head_passed == CASE_COUNT and duplicate_ok
        )
        if not cloud_report["accepted"]:
            raise RuntimeError("cloud verification did not pass all required checks")
    except Exception as exc:
        cloud_report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cloud_report["completed_at"] = _utc_now()
        if store is not None:
            store.close()
        _write_json(cloud_report_path, cloud_report)

    print(
        f"Cloud stress: accepted={cloud_report['accepted']}; report={cloud_report_path}"
    )
    return 0 if cloud_report["accepted"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-phase, fail-closed STRESS_100 recognition and cloud verification harness"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser(
        "local",
        help="run exactly 100 deterministic augmented recognition cases without network access",
    )
    local.add_argument("--run-id", help="safe run identifier used in all 100 unique QR values")
    local.add_argument("--data-root", default=str(PROJECT_ROOT))
    local.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    local.add_argument("--model", help="optional custom QR YOLO model; direct decode remains first")
    local.set_defaults(handler=run_local)

    upload = subparsers.add_parser(
        "upload",
        help="upload and verify an already accepted 100/100 local run",
    )
    upload.add_argument("--run-id", required=True)
    upload.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    upload.add_argument("--timeout", type=float, default=20.0)
    upload.add_argument("--workers", type=int, default=8)
    upload.set_defaults(handler=run_upload)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "timeout", 1.0) <= 0:
        raise SystemExit("--timeout must be positive")
    if getattr(args, "workers", 1) < 1 or getattr(args, "workers", 1) > 32:
        raise SystemExit("--workers must be between 1 and 32")
    try:
        result = int(args.handler(args))
    except KeyboardInterrupt:
        result = 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        result = 2
    raise SystemExit(result)


if __name__ == "__main__":
    main()
