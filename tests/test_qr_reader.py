import cv2
import numpy as np
import qrcode

from roll_qr_scale.qr_reader import QRReader


def make_qr_frame(value: str) -> np.ndarray:
    qr_image = qrcode.make(value).convert("RGB").resize((360, 360))
    qr_bgr = cv2.cvtColor(np.asarray(qr_image), cv2.COLOR_RGB2BGR)
    frame = np.full((600, 900, 3), 245, dtype=np.uint8)
    frame[120:480, 270:630] = qr_bgr
    return frame


def test_decodes_qr_from_frame() -> None:
    value = "ROLL-2026-000123"
    detections = QRReader().decode(make_qr_frame(value))
    assert [item.value for item in detections] == [value]
    assert detections[0].points.shape == (4, 2)
    assert detections[0].decoder in {"zxing", "opencv", "opencv-curved"}


def test_decodes_small_low_contrast_qr_after_grayscale_preprocessing() -> None:
    value = "SP-2026-000123"
    qr_image = np.asarray(qrcode.make(value).convert("RGB"))
    qr_bgr = cv2.resize(
        cv2.cvtColor(qr_image, cv2.COLOR_RGB2BGR),
        (64, 64),
        interpolation=cv2.INTER_AREA,
    )
    qr_bgr = np.clip(128 + (qr_bgr.astype(np.float32) - 128) * 0.35, 0, 255).astype(
        np.uint8
    )
    qr_bgr = cv2.GaussianBlur(qr_bgr, (3, 3), 0.7)
    frame = np.full((720, 1280, 3), 205, dtype=np.uint8)
    frame[300:364, 600:664] = qr_bgr

    detections = QRReader().decode(frame)

    assert [item.value for item in detections] == [value]
    assert detections[0].decoder.startswith("zxing")


def test_decoder_first_success_does_not_call_yolo_fallback() -> None:
    class ModelMustNotRun:
        def predict(self, *args, **kwargs):
            raise AssertionError("YOLO fallback ran even though OpenCV decoded the QR")

    reader = QRReader(yolo_mode="fallback")
    reader.model = ModelMustNotRun()
    assert reader.decode(make_qr_frame("ROLL-OPENCV-FIRST"))[0].value == "ROLL-OPENCV-FIRST"


def test_yolo_localizer_runs_only_after_opencv_failure(monkeypatch) -> None:
    class ArrayLike:
        def cpu(self):
            return self

        def numpy(self):
            return np.array([[100, 120, 300, 320]], dtype=np.float32)

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def predict(self, *args, **kwargs):
            self.calls += 1
            boxes = type("Boxes", (), {"xyxy": ArrayLike()})()
            return [type("Result", (), {"boxes": boxes})()]

    reader = QRReader(yolo_mode="fallback")
    model = FakeModel()
    reader.model = model
    calls = 0

    def decode_full_then_crop(image):
        nonlocal calls
        calls += 1
        if calls == 1:
            return []
        return [
            type(
                "Detection",
                (),
                    {
                        "value": "ROLL-YOLO-FALLBACK",
                        "points": np.zeros((4, 2), dtype=np.float32),
                        "decoder": "zxing",
                    },
            )()
        ]

    monkeypatch.setattr(reader, "_decode_image", decode_full_then_crop)
    detections = reader.decode(np.zeros((480, 640, 3), dtype=np.uint8))

    assert model.calls == 1
    assert detections[0].value == "ROLL-YOLO-FALLBACK"
    # The crop has 15% horizontal/vertical padding around the fake YOLO box.
    assert np.array_equal(detections[0].points[0], np.array([70, 90], dtype=np.float32))


def test_yolo_first_localizes_before_decoding() -> None:
    class ArrayLike:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.value)

    class FakeModel:
        names = {0: "qr"}

        def __init__(self):
            self.kwargs = None

        def predict(self, *args, **kwargs):
            self.kwargs = kwargs
            boxes = type(
                "Boxes",
                (),
                {
                    "xyxy": ArrayLike([[270, 120, 630, 480]]),
                    "cls": ArrayLike([0]),
                },
            )()
            return [type("Result", (), {"boxes": boxes, "names": self.names})()]

    reader = QRReader(yolo_mode="first", yolo_imgsz=640)
    model = FakeModel()
    reader.model = model
    detections = reader.decode(make_qr_frame("ROLL-YOLO-FIRST"))

    assert detections[0].value == "ROLL-YOLO-FIRST"
    assert detections[0].decoder.startswith("yolo+")
    assert model.kwargs["imgsz"] == 640
