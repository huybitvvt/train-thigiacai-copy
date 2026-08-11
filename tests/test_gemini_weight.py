from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roll_qr_scale.gemini_weight import GeminiWeightReader


class FakeModels:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            parsed=self.payload,
            text="",
            usage_metadata=SimpleNamespace(
                prompt_token_count=1234,
                candidates_token_count=37,
                thoughts_token_count=19,
                total_token_count=1290,
            ),
        )


class FakeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.models = FakeModels(payload)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_gemini_reader_sends_three_sampled_full_images_and_returns_qr() -> None:
    client = FakeClient({
        "weight_readable": True,
        "weight_digits": "784",
        "qr_readable": True,
        "qr_code": "ROLL-784",
        "all_frames_agree": True,
    })
    reader = GeminiWeightReader("secret-key", client=client)
    frames = [
        np.full((20, 60, 3), index, dtype=np.uint8)
        for index in range(5)
    ]

    result = reader.read(frames)

    assert result.value == pytest.approx(7.84)
    assert result.readable
    assert result.qr_code == "ROLL-784"
    assert result.qr_readable
    assert result.input_tokens == 1234
    assert result.output_tokens == 37
    assert result.thinking_tokens == 19
    assert result.total_tokens == 1290
    assert len(client.models.calls) == 1
    assert len(client.models.calls[0]["contents"]) == 4  # prompt + 3 JPEG ROIs
    assert reader.status()["successes"] == 1
    assert reader.status()["input_tokens"] == 1234
    assert reader.status()["output_tokens"] == 37
    assert reader.status()["thinking_tokens"] == 19


@pytest.mark.parametrize(
    "payload",
    (
        {
            "weight_readable": False,
            "weight_digits": None,
            "qr_readable": False,
            "qr_code": None,
            "all_frames_agree": False,
        },
        {
            "weight_readable": True,
            "weight_digits": "78",
            "qr_readable": False,
            "qr_code": None,
            "all_frames_agree": True,
        },
        {
            "weight_readable": True,
            "weight_digits": "784",
            "qr_readable": False,
            "qr_code": None,
            "all_frames_agree": False,
        },
    ),
)
def test_gemini_reader_fails_closed_on_unreadable_or_invalid_payload(payload) -> None:
    reader = GeminiWeightReader("secret-key", client=FakeClient(payload))

    result = reader.read([
        np.zeros((20, 60, 3), dtype=np.uint8) for _ in range(3)
    ])

    assert result.value is None
    assert not result.readable
    assert reader.status()["failures"] == 1


def test_gemini_reader_accepts_one_full_still_image() -> None:
    client = FakeClient({
        "weight_readable": True,
        "weight_digits": "1304",
        "qr_readable": True,
        "qr_code": "ROLL-STILL",
        "all_frames_agree": True,
    })
    reader = GeminiWeightReader("secret-key", client=client)

    result = reader.read([np.zeros((480, 640, 3), dtype=np.uint8)])

    assert result.value == pytest.approx(13.04)
    assert result.qr_code == "ROLL-STILL"
    assert len(client.models.calls[0]["contents"]) == 2  # prompt + full still image


def test_gemini_reader_redacts_api_key_from_errors() -> None:
    class FailingModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("request failed for secret-key")

    reader = GeminiWeightReader(
        "secret-key",
        client=SimpleNamespace(models=FailingModels()),
    )

    result = reader.read([
        np.zeros((20, 60, 3), dtype=np.uint8) for _ in range(3)
    ])

    assert result.value is None
    assert "secret-key" not in result.raw
    assert "[redacted]" in result.raw


def test_fast_reader_limits_image_size_and_does_not_trust_ai_qr() -> None:
    client = FakeClient({
        "weight_readable": True,
        "weight_digits": "1304",
        "qr_readable": True,
        "qr_code": "HALLUCINATED-CODE",
        "all_frames_agree": True,
    })
    reader = GeminiWeightReader(
        "secret-key",
        client=client,
        max_image_edge=640,
        jpeg_quality=80,
        media_resolution="medium",
        include_qr=False,
    )

    result = reader.read([np.zeros((1080, 1920, 3), dtype=np.uint8)])

    image_part = client.models.calls[0]["contents"][1]
    encoded = np.frombuffer(image_part.inline_data.data, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) == 640
    assert result.value == pytest.approx(13.04)
    assert result.qr_code is None
    assert result.qr_readable is False
    assert reader.status()["media_resolution"] == "medium"
    assert reader.status()["include_qr"] is False
