from roll_qr_scale.qr_input import HIDQRInput


def test_hid_scanner_collects_code_until_enter() -> None:
    scanner = HIDQRInput(min_length=3)
    for index, character in enumerate("ROLL-USB-001"):
        assert scanner.handle_key(ord(character), now=1.0 + index * 0.01)
    assert scanner.reading() is None
    assert scanner.handle_key(13, now=1.2)

    reading = scanner.reading()
    assert reading is not None
    assert reading.value == "ROLL-USB-001"
    assert reading.source == "scanner_hid"


def test_hid_scanner_timeout_discards_partial_code() -> None:
    scanner = HIDQRInput(min_length=3, character_timeout=0.2)
    scanner.handle_key(ord("O"), now=1.0)
    scanner.handle_key(ord("L"), now=1.1)
    scanner.handle_key(ord("D"), now=2.0)
    scanner.handle_key(ord("1"), now=2.01)
    scanner.handle_key(ord("2"), now=2.02)
    scanner.handle_key(13, now=2.03)
    assert scanner.reading().value == "D12"


def test_space_is_reserved_for_commit() -> None:
    scanner = HIDQRInput()
    assert scanner.handle_key(32, now=1.0) is False

