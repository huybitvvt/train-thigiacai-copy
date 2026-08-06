from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import cv2
import numpy as np
import qrcode


def make_background(size: int, rng: random.Random) -> np.ndarray:
    base = np.empty((size, size, 3), dtype=np.uint8)
    color_a = np.array([rng.randint(120, 235) for _ in range(3)], dtype=np.float32)
    color_b = np.array([rng.randint(100, 225) for _ in range(3)], dtype=np.float32)
    gradient = np.linspace(0, 1, size, dtype=np.float32)[:, None]
    column = color_a * (1 - gradient) + color_b * gradient
    base[:] = column[:, None, :]

    # Large ellipses and bands approximate cylindrical rolls and warehouse clutter.
    for _ in range(rng.randint(2, 7)):
        center = (rng.randrange(size), rng.randrange(size))
        axes = (rng.randint(size // 10, size // 2), rng.randint(size // 12, size // 2))
        color = tuple(rng.randint(80, 245) for _ in range(3))
        cv2.ellipse(base, center, axes, rng.randrange(180), 0, 360, color, rng.randint(2, 20))
    for _ in range(rng.randint(3, 10)):
        y = rng.randrange(size)
        cv2.line(
            base,
            (0, y),
            (size, min(size - 1, y + rng.randint(-35, 35))),
            tuple(rng.randint(70, 220) for _ in range(3)),
            rng.randint(1, 8),
        )
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(2, 10), base.shape)
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def make_qr(value: str) -> np.ndarray:
    code = qrcode.QRCode(version=None, box_size=8, border=4)
    code.add_data(value)
    code.make(fit=True)
    return cv2.cvtColor(np.asarray(code.make_image().convert("RGB")), cv2.COLOR_RGB2BGR)


def paste_perspective_qr(
    frame: np.ndarray,
    qr: np.ndarray,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    side = rng.randint(max(42, width // 14), max(60, width * 5 // 12))
    qr = cv2.resize(qr, (side, side), interpolation=cv2.INTER_NEAREST)
    max_x = max(1, width - side - 1)
    max_y = max(1, height - side - 1)
    x, y = rng.randint(0, max_x), rng.randint(0, max_y)
    jitter = max(2, int(side * rng.uniform(0.02, 0.13)))
    source = np.float32([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]])
    target = np.float32(
        [
            [x + rng.randint(-jitter, jitter), y + rng.randint(-jitter, jitter)],
            [x + side + rng.randint(-jitter, jitter), y + rng.randint(-jitter, jitter)],
            [x + side + rng.randint(-jitter, jitter), y + side + rng.randint(-jitter, jitter)],
            [x + rng.randint(-jitter, jitter), y + side + rng.randint(-jitter, jitter)],
        ]
    )
    target[:, 0] = np.clip(target[:, 0], 0, width - 1)
    target[:, 1] = np.clip(target[:, 1], 0, height - 1)
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(qr, matrix, (width, height), borderValue=(0, 0, 0))
    mask_source = np.full((side, side), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(mask_source, matrix, (width, height), borderValue=0)
    frame[mask > 0] = warped[mask > 0]
    x1, y1 = np.floor(target.min(axis=0)).astype(int)
    x2, y2 = np.ceil(target.max(axis=0)).astype(int)
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def make_value(index: int, rng: random.Random) -> str:
    suffix = "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
    return f"ROLL-SYNTH-{index:05d}-{suffix}"


def write_split(root: Path, split: str, count: int, size: int, seed: int) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for index in range(count):
        frame = make_background(size, rng)
        labels: list[str] = []
        # Negative images teach the detector that warehouse patterns are not QR labels.
        if rng.random() >= 0.12:
            value = make_value(index, rng)
            x1, y1, x2, y2 = paste_perspective_qr(frame, make_qr(value), rng)
            center_x = ((x1 + x2) / 2) / size
            center_y = ((y1 + y2) / 2) / size
            box_width = (x2 - x1) / size
            box_height = (y2 - y1) / size
            labels.append(f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
        if rng.random() < 0.35:
            frame = cv2.GaussianBlur(frame, (3, 3), rng.uniform(0.2, 1.0))
        name = f"synth_{index:05d}"
        if not cv2.imwrite(str(image_dir / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(72, 96)]):
            raise OSError(f"Cannot write generated image {name}")
        (label_dir / f"{name}.txt").write_text("\n".join(labels), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic one-class YOLO QR dataset")
    parser.add_argument("--output", default="dataset/qr")
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--val", type=int, default=60)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.train < 1 or args.val < 1 or args.size < 160:
        raise ValueError("train/val must be positive and size must be at least 160")
    root = Path(args.output)
    write_split(root, "train", args.train, args.size, args.seed)
    write_split(root, "val", args.val, args.size, args.seed + 1)
    print(f"Generated {args.train} train + {args.val} val images under {root.resolve()}")
    print("Synthetic data is for pipeline validation; retrain with real factory images before production.")


if __name__ == "__main__":
    main()
