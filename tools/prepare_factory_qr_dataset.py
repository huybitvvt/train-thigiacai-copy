from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def copy_pair(image: Path, label: Path, root: Path, split: str, prefix: str) -> None:
    output_name = f"{prefix}_{image.stem}"
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_dir / f"{output_name}{image.suffix.lower()}")
    shutil.copy2(label, label_dir / f"{output_name}.txt")


def session_id(factory_root: Path, image: Path) -> str:
    metadata_path = factory_root / "metadata" / f"{image.stem}.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            explicit = str(metadata.get("session_id", "")).strip()
            if explicit:
                return explicit
            captured_at = str(metadata.get("captured_at", ""))
            if len(captured_at) >= 10:
                return captured_at[:10]
        except (OSError, ValueError, TypeError):
            pass
    parts = image.stem.split("_")
    return parts[1][:8] if len(parts) > 1 and len(parts[1]) >= 8 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a mixed synthetic + reviewed factory QR dataset"
    )
    parser.add_argument("--factory", default="dataset/factory_raw")
    parser.add_argument("--synthetic", default="dataset/qr")
    parser.add_argument("--output", default="dataset/qr_factory")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    if not 0.1 <= args.val_ratio <= 0.5:
        raise ValueError("--val-ratio phải nằm trong khoảng 0.1 đến 0.5")

    factory_root = Path(args.factory)
    pairs: list[tuple[Path, Path]] = []
    for extension in IMAGE_EXTENSIONS:
        for image in (factory_root / "images").glob(f"*{extension}"):
            label = factory_root / "labels" / f"{image.stem}.txt"
            if label.is_file():
                pairs.append((image, label))
    if len(pairs) < 20:
        raise RuntimeError(
            f"Mới có {len(pairs)} ảnh xưởng đã gán nhãn; cần tối thiểu 20 để chạy thử, "
            "khuyến nghị 300+ trước production"
        )

    sessions: dict[str, list[tuple[Path, Path]]] = {}
    for pair in pairs:
        sessions.setdefault(session_id(factory_root, pair[0]), []).append(pair)
    if len(sessions) < 2:
        raise RuntimeError(
            "Cần ảnh từ ít nhất 2 ngày/ca (metadata session_id) để tách validation "
            "mà không rò rỉ các frame gần giống nhau"
        )
    rng = random.Random(args.seed)
    session_names = list(sessions)
    rng.shuffle(session_names)
    target_val_count = max(1, round(len(pairs) * args.val_ratio))
    val_sessions: set[str] = set()
    selected = 0
    for name in session_names:
        if selected >= target_val_count and val_sessions:
            break
        val_sessions.add(name)
        selected += len(sessions[name])
    if len(val_sessions) == len(sessions):
        val_sessions.remove(session_names[-1])
    output = Path(args.output)
    for image, label in pairs:
        split = "val" if session_id(factory_root, image) in val_sessions else "train"
        copy_pair(image, label, output, split, "factory")

    synthetic = Path(args.synthetic)
    synthetic_count = 0
    for split in ("train", "val"):
        for extension in IMAGE_EXTENSIONS:
            for image in (synthetic / "images" / split).glob(f"*{extension}"):
                label = synthetic / "labels" / split / f"{image.stem}.txt"
                if label.is_file():
                    copy_pair(image, label, output, split, "synthetic")
                    synthetic_count += 1
    print(
        f"Prepared {len(pairs)} factory images from {len(sessions)} sessions "
        f"(validation sessions: {', '.join(sorted(val_sessions))}) + "
        f"{synthetic_count} synthetic images under {output.resolve()}"
    )


if __name__ == "__main__":
    main()
