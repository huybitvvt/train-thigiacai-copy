import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a one-class YOLOv8 QR localizer")
    parser.add_argument("--data", default="config/qr_dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="For example 0 for CUDA or cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="yolov8n")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path("runs/qr").resolve()),
        name=args.name,
        exist_ok=True,
        patience=10,
    )


if __name__ == "__main__":
    main()
