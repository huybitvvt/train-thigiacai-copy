from pathlib import Path

import cv2
import numpy as np
import pytest

from roll_qr_scale.app import build_parser as build_app_parser
from roll_qr_scale.weight_ocr import (
    CameraOCRWeightSource,
    NormalizedROI,
    PaddleOCRTextReader,
    _correct_bright_led_confusions,
    _decode_seven_segment,
    _restore_fixed_scale_decimal,
    _restore_led_decimal,
    detect_weight_roi,
    detect_weight_roi_consensus,
    parse_normalized_roi,
    parse_ocr_weight,
)
from roll_qr_scale.scale import WeightReading


class FakeReader:
    def __init__(self, results):
        self.results = results
        self.last_shape = None

    def recognize(self, image, **kwargs):
        self.last_shape = image.shape
        assert kwargs["allowlist"].startswith("0123456789")
        assert kwargs["horizontal_list"] == [[0, image.shape[1], 0, image.shape[0]]]
        return self.results


class SequenceReader(FakeReader):
    def __init__(self, result_sets):
        super().__init__([])
        self.result_sets = list(result_sets)

    def recognize(self, image, **kwargs):
        self.last_shape = image.shape
        assert kwargs["allowlist"].startswith("0123456789")
        assert kwargs["horizontal_list"] == [[0, image.shape[1], 0, image.shape[0]]]
        return self.result_sets.pop(0)


def test_paddle_reader_adapts_recognition_only_result() -> None:
    class FakePaddleModel:
        def predict(self, *, input, batch_size):
            assert len(input) == 1
            assert input[0].shape == (20, 60, 3)
            assert batch_size == 1
            return [{"res": {"rec_text": "7.84", "rec_score": 0.932}}]

    reader = PaddleOCRTextReader(FakePaddleModel())

    results = reader.recognize(np.zeros((20, 60), dtype=np.uint8))

    assert len(results) == 1
    assert results[0][1] == "7.84"
    assert results[0][2] == pytest.approx(0.932)


def test_paddle_temporal_capture_uses_one_three_frame_batch() -> None:
    class FakePaddleModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, *, input, batch_size):
            self.calls += 1
            assert len(input) == batch_size == 3
            return [
                {"res": {"rec_text": "7.84", "rec_score": 0.96}}
                for _ in input
            ]

    model = FakePaddleModel()
    source = CameraOCRWeightSource(
        parse_normalized_roi("0,0,1,1"),
        reader=PaddleOCRTextReader(model),
    )

    reading = source.capture_many([
        np.zeros((40, 80, 3), dtype=np.uint8) for _ in range(9)
    ])

    assert reading.value == pytest.approx(7.84)
    assert reading.stable
    assert model.calls == 1
    assert "engine=paddle-batch" in reading.raw
    assert "frames=9; sampled=3" in reading.raw
    assert "fused=skipped" in reading.raw


def test_paddle_temporal_capture_exposes_unique_rejected_candidate() -> None:
    class FakePaddleModel:
        def predict(self, *, input, batch_size):
            assert len(input) == batch_size == 3
            return [
                {"res": {"rec_text": text, "rec_score": 0.96}}
                for text in ("7.84", "7.84", "1.84")
            ]

    source = CameraOCRWeightSource(
        parse_normalized_roi("0,0,1,1"),
        reader=PaddleOCRTextReader(FakePaddleModel()),
    )

    reading = source.capture_many([
        np.zeros((40, 80, 3), dtype=np.uint8) for _ in range(5)
    ])
    candidate = source.candidate_reading()

    assert reading.value is None
    assert candidate is not None
    assert candidate.value == pytest.approx(7.84)
    assert "votes=2/3" in candidate.raw


def test_production_camera_ocr_default_matches_test_station_threshold() -> None:
    source = CameraOCRWeightSource(parse_normalized_roi("0,0,1,1"), reader=FakeReader([]))

    assert source.min_confidence == pytest.approx(0.60)
    assert build_app_parser().parse_args([]).ocr_min_confidence == pytest.approx(0.60)


def test_parse_normalized_roi_and_pixel_crop() -> None:
    roi = parse_normalized_roi("0.25,0.5,0.75,1")
    assert roi.pixels(np.zeros((200, 400, 3), dtype=np.uint8)) == (100, 100, 300, 200)


@pytest.mark.parametrize(
    "value",
    ("0,0,1", "-0.1,0,1,1", "0.8,0,0.2,1", "0,0,1.1,1", "x,0,1,1"),
)
def test_rejects_invalid_normalized_roi(value: str) -> None:
    with pytest.raises(ValueError, match="ROI|roi"):
        parse_normalized_roi(value)


def test_ocr_parser_requires_the_whole_result_to_be_a_weight() -> None:
    assert parse_ocr_weight("125,40 kg", "kg") == (125.4, "kg")
    assert parse_ocr_weight("L2s4k", "kg") is None
    assert parse_ocr_weight("ID 001 125.4 kg", "kg") is None


def test_restores_decimal_point_dropped_from_red_led_digits() -> None:
    crop = np.full((36, 100, 3), 20, dtype=np.uint8)
    cv2.putText(
        crop,
        "70.2",
        (4, 29),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (10, 20, 255),
        2,
        cv2.LINE_AA,
    )

    assert _restore_led_decimal("702", crop) == "70.2"
    assert _restore_led_decimal("70.2", crop) == "70.2"


def test_prefers_bright_real_decimal_over_faint_reflection() -> None:
    crop = np.full((42, 110, 3), 20, dtype=np.uint8)
    cv2.putText(
        crop,
        "7.02",
        (4, 33),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (10, 20, 255),
        2,
        cv2.LINE_AA,
    )
    # A weak reflection between 0 and 2 must not move the decimal point.
    cv2.circle(crop, (58, 32), 2, (30, 40, 125), -1, cv2.LINE_AA)

    assert _restore_led_decimal("702", crop) == "7.02"


def test_restores_two_decimal_factory_scale_layout_from_bright_core() -> None:
    crop = np.zeros((40, 80, 3), dtype=np.uint8)
    cv2.putText(
        crop,
        "702",
        (3, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (60, 180, 255),
        2,
        cv2.LINE_AA,
    )

    assert _restore_fixed_scale_decimal("702", crop) == "7.02"
    assert _restore_fixed_scale_decimal("70.2", crop) == "7.02"
    assert _restore_fixed_scale_decimal("20.15", crop) == "20.15"


def test_corrects_bright_seven_segment_nine_misread_as_three() -> None:
    crop = np.zeros((44, 90, 3), dtype=np.uint8)
    cv2.putText(
        crop,
        "934",
        (2, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (60, 180, 255),
        2,
        cv2.LINE_AA,
    )

    assert _correct_bright_led_confusions("334", crop) == "934"
    assert _correct_bright_led_confusions("234", crop) == "934"
    assert _correct_bright_led_confusions("934", crop) == "934"


def test_corrects_bright_seven_segment_seven_misread_as_two() -> None:
    crop = np.zeros((44, 90, 3), dtype=np.uint8)
    cv2.putText(
        crop,
        "702",
        (2, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (60, 180, 255),
        2,
        cv2.LINE_AA,
    )

    assert _correct_bright_led_confusions("202", crop) == "702"


def test_decodes_small_red_segment_but_abstains_on_conflicting_integer_ocr() -> None:
    patterns = {
        "0": "abcedf",
        "2": "abged",
        "7": "abc",
    }
    crop = np.full((32, 62, 3), 15, dtype=np.uint8)
    segment_lines = {
        "a": ((3, 2), (9, 2)),
        "b": ((10, 3), (10, 13)),
        "c": ((10, 17), (10, 27)),
        "d": ((3, 28), (9, 28)),
        "e": ((2, 17), (2, 27)),
        "f": ((2, 3), (2, 13)),
        "g": ((3, 15), (9, 15)),
    }
    for index, number in enumerate("702"):
        offset = 3 + index * 18
        for segment in patterns[number]:
            start, end = segment_lines[segment]
            cv2.line(
                crop,
                (start[0] + offset, start[1]),
                (end[0] + offset, end[1]),
                (10, 20, 255),
                2,
                cv2.LINE_AA,
            )
    cv2.circle(crop, (18, 27), 1, (10, 20, 255), -1, cv2.LINE_AA)
    cv2.circle(crop, (36, 27), 1, (30, 40, 125), -1, cv2.LINE_AA)

    result = _decode_seven_segment(crop)

    assert result is not None
    text, confidence = result
    assert text == "7.02"
    assert confidence >= 0.72

    reader = FakeReader([([[0, 0], [10, 0], [10, 10], [0, 10]], "1", 0.99)])
    reading = CameraOCRWeightSource(
        parse_normalized_roi("0,0,1,1"),
        reader=reader,
        min_confidence=0.7,
    ).capture(crop)
    assert reading.value is None
    assert not reading.stable
    assert "7SEG:7.02" in reading.raw


def test_unconfirmed_ledcore_geometry_does_not_accept_wrong_weight(monkeypatch) -> None:
    import roll_qr_scale.weight_ocr as weight_ocr_module

    monkeypatch.setattr(weight_ocr_module, "_decode_seven_segment", lambda crop: None)
    monkeypatch.setattr(
        weight_ocr_module,
        "_decode_bright_core_digits",
        lambda crop: ("334", 0.743),
    )
    crop = np.zeros((44, 90, 3), dtype=np.uint8)
    cv2.putText(
        crop,
        "934",
        (2, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (60, 180, 255),
        2,
        cv2.LINE_AA,
    )

    reading = CameraOCRWeightSource(
        parse_normalized_roi("0,0,1,1"),
        reader=FakeReader([]),
        min_confidence=0.60,
    ).capture(crop)

    assert reading.value is None
    assert not reading.stable
    assert "LEDCORE:3.34@0.743" in reading.raw
    assert "unconfirmed fixed-layout reading" in reading.raw


def test_marginal_seven_segment_guess_does_not_override_ocr(monkeypatch) -> None:
    import roll_qr_scale.weight_ocr as weight_ocr_module

    monkeypatch.setattr(
        weight_ocr_module,
        "_decode_seven_segment",
        lambda crop: ("88.56", 0.701),
    )
    reader = FakeReader(
        [([[0, 0], [10, 0], [10, 10], [0, 10]], "20.15", 0.627)]
    )
    reading = CameraOCRWeightSource(
        parse_normalized_roi("0,0,1,1"),
        reader=reader,
        min_confidence=0.60,
    ).capture(np.zeros((40, 120, 3), dtype=np.uint8))

    assert reading.value == 20.15
    assert reading.confidence == pytest.approx(0.627)
    assert "7SEG:88.56@0.701" in reading.raw


def test_accepts_clear_fixed_layout_led_at_specialized_confidence(monkeypatch) -> None:
    import roll_qr_scale.weight_ocr as weight_ocr_module

    monkeypatch.setattr(weight_ocr_module, "_decode_seven_segment", lambda crop: None)
    crop = np.zeros((40, 80, 3), dtype=np.uint8)
    cv2.putText(
        crop,
        "702",
        (3, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (60, 180, 255),
        2,
        cv2.LINE_AA,
    )
    reader = FakeReader(
        [([[0, 0], [10, 0], [10, 10], [0, 10]], "702", 0.516)]
    )

    reading = CameraOCRWeightSource(
        parse_normalized_roi("0,0,1,1"),
        reader=reader,
        min_confidence=0.60,
    ).capture(crop)

    assert reading.value == 7.02
    assert reading.confidence is not None and reading.confidence >= 0.516


def test_fixed_led_consensus_beats_one_higher_confidence_conflict(monkeypatch) -> None:
    import roll_qr_scale.weight_ocr as weight_ocr_module

    monkeypatch.setattr(weight_ocr_module, "_decode_seven_segment", lambda crop: None)
    monkeypatch.setattr(
        weight_ocr_module,
        "_decode_bright_core_digits",
        lambda crop: ("934", 0.767),
    )
    crop = np.zeros((44, 90, 3), dtype=np.uint8)
    cv2.putText(
        crop,
        "934",
        (2, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (60, 180, 255),
        2,
        cv2.LINE_AA,
    )
    box = [[0, 0], [10, 0], [10, 10], [0, 10]]
    reader = SequenceReader([
        [(box, "934", 0.631)],
        [(box, "134", 0.819)],
        [(box, "934", 0.731)],
        [(box, "934", 0.742)],
        [(box, "934", 0.756)],
        [(box, "934", 0.711)],
        [(box, "934", 0.722)],
    ])

    reading = CameraOCRWeightSource(
        parse_normalized_roi("0,0,1,1"),
        reader=reader,
        min_confidence=0.60,
    ).capture(crop)

    assert reading.value == 9.34
    assert "LEDCORE:9.34" in reading.raw


def test_green_consensus_is_not_rewritten_by_wrong_ledcore_geometry(monkeypatch) -> None:
    import roll_qr_scale.weight_ocr as weight_ocr_module

    monkeypatch.setattr(weight_ocr_module, "_decode_seven_segment", lambda crop: None)
    monkeypatch.setattr(
        weight_ocr_module,
        "_decode_bright_core_digits",
        lambda crop: ("202", 0.707),
    )
    crop = np.zeros((44, 90, 3), dtype=np.uint8)
    cv2.putText(
        crop,
        "702",
        (2, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (60, 180, 255),
        2,
        cv2.LINE_AA,
    )
    box = [[0, 0], [10, 0], [10, 10], [0, 10]]
    reader = SequenceReader(
        [
            [(box, "204", 0.237)],
            [(box, "782", 0.140)],
            [(box, "702", 0.984)],
                [(box, "702", 0.902)],
                [(box, "702", 0.983)],
                [(box, "702", 0.811)],
                [(box, "702", 0.822)],
        ]
    )

    reading = CameraOCRWeightSource(
        parse_normalized_roi("0,0,1,1"),
        reader=reader,
        min_confidence=0.60,
    ).capture(crop)

    assert reading.value == pytest.approx(7.02)
    assert reading.stable
    assert "GREENCONS:7.02" in reading.raw
    assert "green128:702->7.02" in reading.raw


def test_reads_human_verified_full_factory_702_reference() -> None:
    reference = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "factory_scale_7_02_full_reference.jpg"
    )
    frame = cv2.imread(str(reference))
    assert frame is not None
    located = detect_weight_roi(frame)
    assert located is not None

    reading = CameraOCRWeightSource(
        located[0],
        unit="kg",
        min_confidence=0.60,
        download_enabled=False,
    ).capture(frame)

    assert reading.value == pytest.approx(7.02)
    assert reading.stable
    assert reading.confidence is not None and reading.confidence >= 0.60
    assert "70.2->7.02" in reading.raw


def test_reads_blurred_station_01_784_with_paddleocr_v6() -> None:
    reference = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "factory_scale_diagnostic_station-01.jpg"
    )
    frame = cv2.imread(str(reference))
    assert frame is not None
    located = detect_weight_roi(frame)
    assert located is not None

    reading = CameraOCRWeightSource(
        located[0],
        unit="kg",
        min_confidence=0.60,
        download_enabled=False,
    ).capture(frame)

    assert reading.value == pytest.approx(7.84)
    assert reading.stable
    assert reading.confidence is not None and reading.confidence >= 0.70
    assert "784->7.84" in reading.raw


def test_auto_detects_red_led_weight_region() -> None:
    frame = np.full((400, 800, 3), 40, dtype=np.uint8)
    cv2.putText(
        frame,
        "20.15",
        (285, 335),
        cv2.FONT_HERSHEY_DUPLEX,
        2.2,
        (20, 30, 255),
        5,
        cv2.LINE_AA,
    )

    found = detect_weight_roi(frame)

    assert found is not None
    roi, method = found
    assert method == "red-led"
    assert 0.30 < roi.x1 < 0.45
    assert 0.55 < roi.x2 < 0.80
    assert 0.65 < roi.y1 < 0.90
    assert 0.75 < roi.y2 < 0.95


def test_auto_detects_small_top_gross_row_in_portrait_factory_photo() -> None:
    frame = np.full((1440, 1080, 3), 35, dtype=np.uint8)
    for text, y in (("7.02", 1170), ("0.00", 1212), ("0.00", 1254)):
        cv2.putText(
            frame,
            text,
            (505, y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.62,
            (10, 20, 255),
            2,
            cv2.LINE_AA,
        )

    found = detect_weight_roi(frame)

    assert found is not None
    roi, method = found
    assert method == "red-led"
    assert 0.44 < roi.x1 < 0.52
    assert 0.50 < roi.x2 < 0.58
    assert 0.77 < roi.y1 < 0.83
    assert 0.80 < roi.y2 < 0.85


def test_led_stack_projection_ignores_brighter_lower_zero_rows() -> None:
    frame = np.full((1440, 1080, 3), 35, dtype=np.uint8)
    cv2.putText(
        frame, "7.02", (505, 1170), cv2.FONT_HERSHEY_DUPLEX,
        0.62, (20, 30, 190), 2, cv2.LINE_AA,
    )
    for text, y in (("0.00", 1212), ("0.00", 1254)):
        cv2.putText(
            frame, text, (505, y), cv2.FONT_HERSHEY_DUPLEX,
            0.62, (10, 20, 255), 3, cv2.LINE_AA,
        )

    found = detect_weight_roi(frame)

    assert found is not None
    roi, _ = found
    assert roi.y2 < 0.84


def test_close_led_rows_are_split_before_selecting_gross() -> None:
    frame = np.full((1440, 1080, 3), 35, dtype=np.uint8)
    for text, y in (("7.02", 1170), ("0.00", 1196), ("0.00", 1222)):
        cv2.putText(
            frame, text, (505, y), cv2.FONT_HERSHEY_DUPLEX,
            0.62, (10, 20, 255), 2, cv2.LINE_AA,
        )

    found = detect_weight_roi(frame)

    assert found is not None
    roi, _ = found
    assert roi.y2 < 0.83


def test_camera_ocr_reads_weight_from_configured_roi() -> None:
    reader = FakeReader(
        [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "noise", 0.99),
            ([[0, 0], [80, 0], [80, 30], [0, 30]], "125.40 kg", 0.93),
        ]
    )
    source = CameraOCRWeightSource(
        parse_normalized_roi("0.5,0.5,1,1"),
        reader=reader,
        min_confidence=0.5,
    )

    reading = source.capture(np.full((200, 400, 3), 240, dtype=np.uint8))

    assert reading.value == 125.4
    assert reading.unit == "kg"
    assert reading.stable
    assert reading.confidence == pytest.approx(0.93)
    assert "125.40 kg" in reading.raw
    assert reader.last_shape == (100, 200)


def test_camera_ocr_rejects_low_confidence_number() -> None:
    reader = FakeReader([([[0, 0], [10, 0], [10, 10], [0, 10]], "88.2", 0.2)])
    source = CameraOCRWeightSource(
        parse_normalized_roi("0,0,1,1"),
        reader=reader,
        min_confidence=0.5,
    )

    reading = source.capture(np.zeros((100, 200, 3), dtype=np.uint8))

    assert reading.value is None
    assert not reading.stable
    assert "88.2@0.200" in reading.raw


def test_camera_ocr_rejects_red_led_with_too_few_source_pixels() -> None:
    crop = np.zeros((18, 60, 3), dtype=np.uint8)
    cv2.putText(
        crop,
        "784",
        (2, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 80, 255),
        1,
        cv2.LINE_AA,
    )
    source = CameraOCRWeightSource(
        parse_normalized_roi("0,0,1,1"),
        reader=FakeReader([
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "7.84", 0.99)
        ]),
    )

    reading = source.capture(crop)

    assert reading.value is None
    assert not reading.stable
    assert "below safe minimum" in reading.raw


def test_temporal_capture_accepts_only_repeated_exact_weight(monkeypatch) -> None:
    source = CameraOCRWeightSource(
        parse_normalized_roi("0,0,1,1"),
        reader=FakeReader([]),
    )
    outputs = iter([
        WeightReading(7.84, "kg", True, "frame", 0.91),
        WeightReading(7.84, "kg", True, "frame", 0.89),
        WeightReading(7.84, "kg", True, "frame", 0.90),
        WeightReading(1.84, "kg", True, "bad", 0.73),
        WeightReading(7.84, "kg", True, "frame", 0.92),
        WeightReading(7.84, "kg", True, "fused", 0.93),
    ])
    monkeypatch.setattr(source, "capture", lambda frame: next(outputs))

    reading = source.capture_many([
        np.zeros((40, 80, 3), dtype=np.uint8) for _ in range(8)
    ])

    assert reading.value == pytest.approx(7.84)
    assert reading.stable
    assert "agreement=4/5" in reading.raw
    assert "fused=7.84kg" in reading.raw


def test_temporal_capture_rejects_two_frame_short_burst() -> None:
    source = CameraOCRWeightSource(
        parse_normalized_roi("0,0,1,1"),
        reader=FakeReader([]),
    )

    reading = source.capture_many([
        np.zeros((40, 80, 3), dtype=np.uint8),
        np.zeros((40, 80, 3), dtype=np.uint8),
    ])

    assert reading.value is None
    assert "at least 3" in reading.raw


def test_temporal_capture_rejects_conflicting_or_changing_display(monkeypatch) -> None:
    source = CameraOCRWeightSource(
        parse_normalized_roi("0,0,1,1"),
        reader=FakeReader([]),
    )
    outputs = iter([
        WeightReading(value, "kg", True, "frame", 0.90)
        for value in (7.82, 7.83, 7.84, 7.84, 7.85, 7.84)
    ])
    monkeypatch.setattr(source, "capture", lambda frame: next(outputs))

    reading = source.capture_many([
        np.zeros((40, 80, 3), dtype=np.uint8) for _ in range(8)
    ])

    assert reading.value is None
    assert not reading.stable
    assert "rejected=" in reading.raw


def test_temporal_roi_uses_largest_consistent_cluster(monkeypatch) -> None:
    import roll_qr_scale.weight_ocr as weight_ocr_module

    values = iter([
        (NormalizedROI(0.45, 0.80, 0.55, 0.84), "red-led"),
        (NormalizedROI(0.452, 0.801, 0.552, 0.841), "red-led"),
        (NormalizedROI(0.10, 0.60, 0.20, 0.65), "red-led"),
        (NormalizedROI(0.448, 0.799, 0.548, 0.839), "red-led"),
    ])
    monkeypatch.setattr(weight_ocr_module, "detect_weight_roi", lambda frame: next(values))

    result = detect_weight_roi_consensus([
        np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)
    ])

    assert result is not None
    roi, method = result
    assert roi.x1 == pytest.approx(0.45)
    assert roi.y1 == pytest.approx(0.80)
    assert method == "red-led-temporal(3/4)"
