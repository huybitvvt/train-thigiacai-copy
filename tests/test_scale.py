import pytest

from roll_qr_scale.scale import ManualWeightSource, parse_weight_line


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("ST,GS,+ 12.340 kg", (12.34, "kg")),
        ("US,NT,-0.50kg", (-0.5, "kg")),
        ("weight=145,75 KGS", (145.75, "kg")),
        (" 2500 g", (2500.0, "g")),
    ],
)
def test_parse_common_scale_lines(line: str, expected: tuple[float, str]) -> None:
    assert parse_weight_line(line) == expected


def test_manual_keyboard_entry() -> None:
    source = ManualWeightSource(unit="kg")
    for key in map(ord, "123.45"):
        assert source.handle_key(key)
    reading = source.reading()
    assert reading.value == 123.45
    assert reading.stable

