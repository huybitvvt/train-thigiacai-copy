from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2

from roll_qr_scale.quality import assess_frame_quality
from roll_qr_scale.qr_reader import QRReader
from roll_qr_scale.weight_ocr import CameraOCRWeightSource, detect_weight_roi


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed acceptance test on a factory-only, held-out image set"
    )
    parser.add_argument("--dataset", default="dataset/factory_acceptance")
    parser.add_argument("--model", help="Custom QR YOLO best.pt; direct decoding remains first")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--weight-tolerance", type=float, default=0.001)
    parser.add_argument("--ocr-min-confidence", type=float, default=0.60)
    parser.add_argument("--output", default="runs/factory_acceptance.json")
    args = parser.parse_args()
    if args.min_samples < 1 or args.weight_tolerance < 0:
        raise ValueError("Ngưỡng nghiệm thu không hợp lệ")

    root = Path(args.dataset)
    images: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        images.extend((root / "images").glob(f"*{extension}"))
    if len(images) < args.min_samples:
        raise RuntimeError(
            f"Tập nghiệm thu riêng mới có {len(images)} ảnh; cần tối thiểu "
            f"{args.min_samples} ảnh không được dùng để train"
        )

    reader = QRReader(args.model, yolo_mode="fallback", yolo_imgsz=640)
    ocr_reader: object | None = None
    cases: list[dict[str, object]] = []
    for image_path in sorted(images):
        metadata_path = root / "metadata" / f"{image_path.stem}.json"
        case: dict[str, object] = {"sample_id": image_path.stem, "passed": False}
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_qr = str(metadata["expected_qr_code"]).strip()
            expected_weight = float(metadata["expected_weight"])
            unit = str(metadata.get("unit", "kg"))
            if not expected_qr or not math.isfinite(expected_weight):
                raise ValueError("ground truth rỗng/không hợp lệ")
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError("không đọc được ảnh")

            quality = assess_frame_quality(frame)
            qr_items = reader.decode(frame)
            predicted_qr = qr_items[0].value if qr_items else None
            located = detect_weight_roi(frame)
            predicted_weight = None
            confidence = None
            if located is not None:
                source = CameraOCRWeightSource(
                    located[0],
                    unit=unit,
                    min_confidence=args.ocr_min_confidence,
                    reader=ocr_reader,
                )
                reading = source.capture(frame)
                ocr_reader = source._reader
                predicted_weight = reading.value
                confidence = reading.confidence

            qr_ok = predicted_qr == expected_qr
            weight_ok = (
                predicted_weight is not None
                and abs(float(predicted_weight) - expected_weight) <= args.weight_tolerance
            )
            case.update(
                {
                    "expected_qr_code": expected_qr,
                    "predicted_qr_code": predicted_qr,
                    "expected_weight": expected_weight,
                    "predicted_weight": predicted_weight,
                    "ocr_confidence": confidence,
                    "quality_pass": quality.accepted,
                    "quality_issues": list(quality.issues),
                    "qr_exact": qr_ok,
                    "weight_exact": weight_ok,
                    "passed": quality.accepted and qr_ok and weight_ok,
                }
            )
        except Exception as exc:
            case["error"] = str(exc)
        cases.append(case)

    passed = sum(bool(case["passed"]) for case in cases)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(root.resolve()),
        "model": str(Path(args.model).resolve()) if args.model else None,
        "total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "pass_rate": passed / len(cases),
        "accepted": passed == len(cases),
        "criteria": {
            "all_cases_must_pass": True,
            "weight_tolerance": args.weight_tolerance,
            "min_samples": args.min_samples,
            "dataset_must_be_held_out": True,
        },
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Factory acceptance: {passed}/{len(cases)} passed; report={output.resolve()}"
    )
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
