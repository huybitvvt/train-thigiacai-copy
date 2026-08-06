from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roll_qr_scale.qr_reader import QRReader
from tools import stress_test_100 as stress


def test_case_schedule_and_combined_augmentation_are_exact_and_deterministic() -> None:
    scheduled = [stress.BASES[index % len(stress.BASES)].name for index in range(100)]
    assert len(scheduled) == 100
    assert scheduled.count("warehouse_scale_demo") == 25
    assert scheduled.count("factory_capture_20260802_175736") == 25
    assert scheduled.count("factory_scale_7_02_full_reference") == 25
    assert scheduled.count("factory_scale_9_34_reference") == 25

    first = stress._augmentation_parameters(0)
    assert first == stress._augmentation_parameters(0)
    assert set(first) == {
        "contrast",
        "brightness_delta",
        "blur_sigma",
        "scale",
        "jpeg_quality",
    }
    frame = np.full((480, 640, 3), 127, dtype=np.uint8)
    augmented_a, encoded_a = stress._augment_frame(frame, first)
    augmented_b, encoded_b = stress._augment_frame(frame, first)
    assert encoded_a == encoded_b
    assert np.array_equal(augmented_a, augmented_b)


def test_synthetic_cloud_proof_is_visibly_distinct_and_decodable() -> None:
    qr_value = "STRESS100-unit-test-001"
    frame = stress._synthetic_proof_frame(qr_value, 9.34)
    quality = stress.assess_frame_quality(frame)
    detections = QRReader().decode(frame)

    assert frame.shape == (720, 960, 3)
    assert quality.accepted
    assert [item.value for item in detections] == [qr_value]
    assert stress.detect_weight_roi(frame) is not None


@pytest.mark.parametrize(
    "run_id",
    ["../escape", "contains space", "", "x" * 49, "-starts-with-dash"],
)
def test_run_id_rejects_unsafe_values(run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        stress._validated_run_id(run_id)


def test_upload_is_fail_closed_before_network_for_rejected_local_report(tmp_path) -> None:
    run_id = "unit-rejected"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "local_report.json").write_text(
        json.dumps(
            {
                "suite": stress.SUITE,
                "phase": "local",
                "run_id": run_id,
                "accepted": False,
                "total": 100,
                "passed": 99,
                "wrong_accepted": 1,
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        run_id=run_id,
        runs_root=str(tmp_path),
        timeout=0.1,
        workers=1,
    )

    assert stress.run_upload(args) == 2
    cloud_report = json.loads((run_dir / "cloud_report.json").read_text(encoding="utf-8"))
    assert cloud_report["accepted"] is False
    assert "upload is fail-closed" in cloud_report["error"]
    assert cloud_report["raw_factory_frames_uploaded"] is False


def test_local_cli_exposes_only_fixed_gate_not_tolerance_overrides() -> None:
    parser = stress.build_parser()
    args = parser.parse_args(["local", "--run-id", "unit-safe"])
    assert args.handler is stress.run_local
    assert not hasattr(args, "weight_tolerance")
    assert not hasattr(args, "ocr_min_confidence")


@pytest.mark.parametrize(
    ("stable", "confidence"),
    [(False, 0.99), (True, 0.59), (True, None)],
)
def test_recognition_gate_rejects_unstable_or_low_confidence_weight(
    monkeypatch,
    stable: bool,
    confidence: float | None,
) -> None:
    qr_value = "STRESS100-gate-test"
    frame = stress._synthetic_proof_frame(qr_value, 7.02)
    reading = SimpleNamespace(
        value=7.02,
        confidence=confidence,
        stable=stable,
        raw="unit-test",
    )
    monkeypatch.setattr(
        stress,
        "_read_weight",
        lambda frame, reader: (reading, reader, (0, 0, 10, 10)),
    )

    result, _ = stress._recognition_result(
        frame,
        qr_value,
        7.02,
        QRReader(),
        None,
    )

    assert result["weight_exact"] is True
    assert result["weight_gate_passed"] is False
    assert result["production_accepted"] is False
    assert result["passed"] is False
