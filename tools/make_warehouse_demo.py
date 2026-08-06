from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import qrcode


QR_VALUE = "ROLL-WAREHOUSE-002015"
WEIGHT_TEXT = "20.15"


def _qr_image(value: str, size: int) -> np.ndarray:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=8,
        border=2,
    )
    qr.add_data(value)
    qr.make(fit=True)
    rgb = np.asarray(qr.make_image(fill_color="black", back_color="white").convert("RGB"))
    return cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    )


def _draw_led_text(frame: np.ndarray, display: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = display
    cv2.rectangle(frame, (left, top), (right, bottom), (3, 3, 4), -1)

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 2.05
    thickness = 4
    (text_width, text_height), baseline = cv2.getTextSize(WEIGHT_TEXT, font, scale, thickness)
    text_left = left + max(8, (right - left - text_width - 30) // 2)
    text_baseline = top + (bottom - top + text_height) // 2 - 2

    glow = np.zeros_like(frame)
    cv2.putText(
        glow,
        WEIGHT_TEXT,
        (text_left, text_baseline),
        font,
        scale,
        (20, 20, 255),
        thickness + 5,
        cv2.LINE_AA,
    )
    glow = cv2.GaussianBlur(glow, (0, 0), 6)
    cv2.addWeighted(frame, 1, glow, 0.55, 0, frame)
    cv2.putText(
        frame,
        WEIGHT_TEXT,
        (text_left, text_baseline),
        font,
        scale,
        (35, 55, 255),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "kg",
        (right - 32, bottom - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (35, 55, 255),
        1,
        cv2.LINE_AA,
    )
    return (
        max(left, text_left - 5),
        max(top, text_baseline - text_height - baseline - 5),
        min(right, text_left + text_width + 5),
        min(bottom, text_baseline + baseline + 5),
    )


def main() -> None:
    base_path = Path("data/warehouse_scale_demo_base.png")
    output_path = Path("data/warehouse_scale_demo.png")
    frame = cv2.imread(str(base_path))
    if frame is None:
        raise RuntimeError(f"Cannot read generated base image: {base_path}")

    height, width = frame.shape[:2]

    # The base was generated with these deliberately blank, front-facing regions.
    scale_x, scale_y = width / 1680, height / 945

    def scaled_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        left, top, right, bottom = box
        return (
            round(left * scale_x),
            round(top * scale_y),
            round(right * scale_x),
            round(bottom * scale_y),
        )

    label = scaled_box((501, 130, 612, 243))
    display = scaled_box((688, 704, 958, 800))

    qr_size = min(label[2] - label[0], label[3] - label[1]) - 8
    qr_left = label[0] + (label[2] - label[0] - qr_size) // 2
    qr_top = label[1] + (label[3] - label[1] - qr_size) // 2
    frame[qr_top : qr_top + qr_size, qr_left : qr_left + qr_size] = _qr_image(
        QR_VALUE, qr_size
    )
    digit_box = _draw_led_text(frame, display)

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Cannot write {output_path}")

    roi = (
        digit_box[0] / width,
        digit_box[1] / height,
        digit_box[2] / width,
        digit_box[3] / height,
    )
    print(f"Created {output_path}")
    print(f"QR={QR_VALUE}")
    print(f"WEIGHT={WEIGHT_TEXT} kg")
    print("ROI=" + ",".join(f"{value:.4f}" for value in roi))


if __name__ == "__main__":
    main()
