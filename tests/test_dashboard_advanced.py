"""Tests for Advanced & Export engineering dashboard."""

from __future__ import annotations

from stablewalk.ui.tk.dashboard_advanced import (
    ENGINEERING_STATUS_LABEL,
    ENGINEERING_STATUS_SYMBOL,
    _status_display,
)


def test_engineering_status_labels_match_spec() -> None:
    # Text-first lab chrome (color strip carries meaning; no glyph prefix).
    assert ENGINEERING_STATUS_SYMBOL["completed"] == ""
    assert ENGINEERING_STATUS_SYMBOL["partial"] == ""
    assert ENGINEERING_STATUS_SYMBOL["unavailable"] == ""
    assert ENGINEERING_STATUS_LABEL["completed"] == "Completed"
    assert ENGINEERING_STATUS_LABEL["partial"] == "Partial"
    assert ENGINEERING_STATUS_LABEL["unavailable"] == "Missing"


def test_status_display_formats_badge() -> None:
    assert _status_display("completed") == "Completed"
    assert _status_display("partial") == "Partial"
    assert _status_display("unavailable") == "Missing"
