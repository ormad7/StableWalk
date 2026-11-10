"""Tests for user-facing error formatting."""

from __future__ import annotations

from stablewalk.ui.tk.user_errors import format_user_error


def test_format_user_error_first_line_only() -> None:
    assert format_user_error("Boom\nTraceback (most recent call last):") == "Boom"


def test_format_user_error_skips_traceback_header() -> None:
    text = "Traceback (most recent call last):\n  File \"x.py\"\nValueError: bad"
    assert format_user_error(text) == "ValueError: bad"


def test_format_user_error_fallback() -> None:
    assert "wrong" in format_user_error(None).lower() or "detail" in format_user_error(None).lower()
