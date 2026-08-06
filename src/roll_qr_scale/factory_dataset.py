from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


class FactorySampleStore:
    """Persist real camera frames and auto-label decoded QR boxes for review."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.label_dir = self.root / "labels"
        self.metadata_dir = self.root / "metadata"
        for directory in (self.image_dir, self.label_dir, self.metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        frame: np.ndarray,
        metadata: dict[str, object],
        qr_roi: str | None = None,
    ) -> dict[str, object]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        sample_id = f"factory_{timestamp}_{uuid.uuid4().hex[:8]}"
        image_path = self.image_dir / f"{sample_id}.jpg"
        if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Không lưu được ảnh mẫu: {image_path}")

        label_path: Path | None = None
        normalized = self._parse_roi(qr_roi)
        if normalized is not None:
            x1, y1, x2, y2 = normalized
            label_path = self.label_dir / f"{sample_id}.txt"
            label_path.write_text(
                f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                f"{x2 - x1:.6f} {y2 - y1:.6f}\n",
                encoding="utf-8",
            )

        metadata_path = self.metadata_dir / f"{sample_id}.json"
        document = {
            "sample_id": sample_id,
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "image": str(image_path.resolve()),
            "label": str(label_path.resolve()) if label_path else None,
            "auto_labeled": label_path is not None,
            **metadata,
        }
        metadata_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "ok": True,
            "sample_id": sample_id,
            "image_path": str(image_path.resolve()),
            "label_path": str(label_path.resolve()) if label_path else None,
            "auto_labeled": label_path is not None,
        }

    @staticmethod
    def _parse_roi(value: str | None) -> tuple[float, float, float, float] | None:
        if not value:
            return None
        try:
            x1, y1, x2, y2 = (float(part) for part in value.split(","))
        except (TypeError, ValueError):
            return None
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            return None
        return x1, y1, x2, y2
