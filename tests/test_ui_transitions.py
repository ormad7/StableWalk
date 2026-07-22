"""Unit tests for UI transition helpers (pure logic, no Tk required)."""

from stablewalk.ui.tk.ui_transitions import (
    extract_leading_number,
    format_interpolated_number,
)


def test_extract_leading_number_with_unit_suffix():
    value, prefix, suffix = extract_leading_number("72 steps/min")
    assert value == 72.0
    assert prefix == ""
    assert suffix == " steps/min"


def test_extract_leading_number_with_prefix_and_percent():
    value, prefix, suffix = extract_leading_number("Stance: 62%")
    assert value == 62.0
    assert prefix == "Stance: "
    assert suffix == "%"


def test_extract_leading_number_missing():
    value, prefix, suffix = extract_leading_number("—")
    assert value is None
    assert prefix == "—"
    assert suffix == ""


def test_format_interpolated_number_respects_decimals():
    text = format_interpolated_number("", 1.234, " s", decimals=2)
    assert text == "1.23 s"

    text_int = format_interpolated_number("", 62.4, "%", decimals=0)
    assert text_int == "62%"
