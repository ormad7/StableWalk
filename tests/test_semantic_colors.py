"""Canonical semantic chart colors."""

from __future__ import annotations

from stablewalk.ui import colors as C


def test_semantic_side_colors() -> None:
    assert C.SIDE_LEFT == "#22c55e"
    assert C.SIDE_RIGHT == "#ef4444"
    assert C.CRITICAL == C.SIDE_RIGHT


def test_semantic_metric_and_com_colors() -> None:
    assert C.METRIC_GLOBAL == "#f0f4f8"
    assert C.COM == "#4dabf7"
    assert C.PLAYHEAD == C.METRIC_GLOBAL


def test_semantic_stability_warning_critical() -> None:
    assert C.STABILITY == "#ffd43b"
    assert C.WARNING == "#ffc857"
    assert C.CRITICAL == "#ef4444"
    assert C.STABILITY_STABLE == C.STABILITY
    assert C.STABILITY_REDUCED == C.WARNING
    assert C.STABILITY_UNSTABLE == C.CRITICAL


def test_chrome_accent_distinct_from_limb_sides() -> None:
    """Instrument accent must not collide with L/R limb green/red."""
    assert C.ACCENT == "#5b9fd4"
    assert C.ACCENT != C.SIDE_LEFT
    assert C.ACCENT != C.SIDE_RIGHT
    assert C.ACCENT_ALT == C.COM
    assert C.PANEL == "#171e2a"
    assert C.BORDER == "#2f3b4d"
