"""Contextual unavailable labels — no bare ``Unavailable`` placeholders."""

from __future__ import annotations

from stablewalk.ui.unavailable_labels import (
    MEASUREMENT_FOR_RECORDING,
    METRIC_FOR_RECORDING,
    MISSING_SESSION_A,
    MISSING_SESSION_B,
    REQUIRES_ANALYZED_VIDEO,
    REQUIRES_GAIT_CYCLES,
    WAITING_COMPARISON,
    compare_reason_label,
    compare_unavailable_label,
    foot_clearance_unavailable_label,
    opensim_unavailable_label,
)


def test_compare_unavailable_is_contextual() -> None:
    assert (
        compare_unavailable_label(
            "Cadence", left_available=False, right_available=False
        )
        == REQUIRES_GAIT_CYCLES
    )
    assert (
        compare_unavailable_label(
            "Speed", left_available=False, right_available=True
        )
        == MISSING_SESSION_A
    )
    assert (
        compare_unavailable_label(
            "Speed", left_available=True, right_available=False
        )
        == MISSING_SESSION_B
    )
    assert (
        compare_unavailable_label(
            "Cadence",
            left_available=True,
            right_available=False,
            right_session_ready=True,
        )
        == REQUIRES_GAIT_CYCLES
    )
    assert (
        compare_unavailable_label(
            "Custom", left_available=False, right_available=False
        )
        == WAITING_COMPARISON
    )
    assert "Unavailable" not in compare_unavailable_label("Cadence")


def test_compare_reason_avoids_bare_unavailable() -> None:
    reason = compare_reason_label(
        "Speed",
        left_available=False,
        right_available=True,
        reason_default="Walking speed unavailable — scale/view not reportable.",
    )
    assert reason.startswith(MISSING_SESSION_A)
    assert "Unavailable in Session" not in reason


def test_foot_and_opensim_contextual_labels() -> None:
    assert foot_clearance_unavailable_label("") == MEASUREMENT_FOR_RECORDING
    assert "Floor" in foot_clearance_unavailable_label("floor estimate invalid")
    assert opensim_unavailable_label(sdk=False, has_session=False) == REQUIRES_ANALYZED_VIDEO
    assert "SDK" in opensim_unavailable_label(sdk=False, has_session=True)
    assert METRIC_FOR_RECORDING != "Unavailable"
    assert WAITING_COMPARISON != "Unavailable"
