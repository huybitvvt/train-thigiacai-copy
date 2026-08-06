from pathlib import Path

import cv2
import numpy as np
import qrcode


def main() -> None:
    output = Path("data/test_frame.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    value = "ROLL-DEMO-0001"
    qr_image = qrcode.make(value).convert("RGB").resize((320, 320))
    frame = np.full((720, 1280, 3), 238, dtype=np.uint8)
    frame[70:390, 480:800] = cv2.cvtColor(np.asarray(qr_image), cv2.COLOR_RGB2BGR)
    cv2.putText(frame, "QR + SCALE OCR DEMO", (390, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 3)
    cv2.rectangle(frame, (335, 430), (945, 650), (18, 22, 25), -1)
    cv2.rectangle(frame, (335, 430), (945, 650), (80, 85, 90), 5)
    cv2.putText(frame, "125.4 kg", (395, 575), cv2.FONT_HERSHEY_DUPLEX, 2.8, (245, 245, 245), 7)
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Cannot write {output}")
    print(f"Created {output} with QR={value} and weight=125.4 kg")


if __name__ == "__main__":
    main()
