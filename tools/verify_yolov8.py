from pathlib import Path

import numpy as np
from ultralytics import YOLO


def main() -> None:
    model_path = Path("yolov8n.pt")
    model = YOLO(str(model_path))
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    result = model.predict(frame, imgsz=640, verbose=False)[0]
    print(f"YOLOv8 OK: model={model_path} detections={len(result.boxes)} device={result.speed}")


if __name__ == "__main__":
    main()

