import json

import numpy as np

from roll_qr_scale.factory_dataset import FactorySampleStore


def test_factory_sample_saves_image_metadata_and_yolo_label(tmp_path) -> None:
    store = FactorySampleStore(tmp_path / "factory")
    result = store.save(
        np.full((480, 640, 3), 120, dtype=np.uint8),
        {"qr_code": "ROLL-FACTORY-001", "weight": 20.15},
        "0.1,0.2,0.3,0.4",
    )
    assert result["auto_labeled"] is True
    label = next((tmp_path / "factory" / "labels").glob("*.txt")).read_text().strip()
    assert label == "0 0.200000 0.300000 0.200000 0.200000"
    metadata = json.loads(next((tmp_path / "factory" / "metadata").glob("*.json")).read_text())
    assert metadata["qr_code"] == "ROLL-FACTORY-001"


def test_factory_sample_keeps_unlabeled_failure(tmp_path) -> None:
    store = FactorySampleStore(tmp_path / "factory")
    result = store.save(np.zeros((480, 640, 3), dtype=np.uint8), {"recognition_ok": False})
    assert result["auto_labeled"] is False
    assert not list((tmp_path / "factory" / "labels").glob("*.txt"))
