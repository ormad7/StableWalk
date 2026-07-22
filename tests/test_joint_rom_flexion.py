"""Joint ROM uses flexion convention and robust spans."""

from __future__ import annotations

from stablewalk.analysis.biomechanical.joint_rom import (
    _robust_span,
    _to_flexion_series,
)


def test_interior_to_flexion_series_for_knee() -> None:
    # Interior 180° → flexion 0°; interior 120° → flexion 60°.
    series = _to_flexion_series([180.0, 120.0, 150.0], joint="knee")
    assert series == [0.0, 60.0, 30.0]


def test_robust_span_rejects_single_spike() -> None:
    values = [10.0, 12.0, 11.0, 13.0, 12.5, 11.5, 160.0]  # spike
    lo, hi, rom = _robust_span(values)
    assert rom < 50.0
    assert hi < 30.0


def test_impossible_knee_flexion_dropped() -> None:
    # Interior 0° → flexion 180° — outside physiological walking window.
    series = _to_flexion_series([0.0, 180.0, 140.0], joint="knee")
    assert series[0] is None
    assert series[1] == 0.0
    assert series[2] == 40.0
