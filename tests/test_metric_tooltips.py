"""Tests for professional metric tooltips."""

from __future__ import annotations

from stablewalk.ui.dashboard_interpretability import METRIC_HELP
from stablewalk.ui.metric_tooltips import (
    METRIC_TOOLTIPS,
    combine_metric_tooltip,
    get_metric_tooltip,
    metric_help_body,
)


REQUIRED_KEYS = (
    "com",
    "bos",
    "grf",
    "rom",
    "cadence",
    "symmetry",
    "stability",
    "heel_strike",
    "toe_off",
    "foot_clearance",
    "joint_angle",
    "pipeline_status",
)


def test_required_metric_tooltips_present() -> None:
    for key in REQUIRED_KEYS:
        assert key in METRIC_TOOLTIPS
        text = get_metric_tooltip(key)
        assert text is not None
        assert "Meaning:" in text
        assert "Calculation:" in text
        assert "Units:" in text
        assert "Normal:" in text
        assert "Clinical:" in text


def test_aliases_share_canonical_copy() -> None:
    assert get_metric_tooltip("vgrf") == get_metric_tooltip("grf")
    assert get_metric_tooltip("joint_rom") == get_metric_tooltip("rom")
    assert get_metric_tooltip("stability_margin") == get_metric_tooltip("stability")


def test_combine_metric_tooltip() -> None:
    base = get_metric_tooltip("cadence")
    combined = combine_metric_tooltip(base, "112 steps/min")
    assert combined is not None
    assert "112 steps/min" in combined
    assert "Meaning:" in combined


def test_metric_help_synced_for_dashboard_keys() -> None:
    for key in (
        "knee_motion",
        "foot_clearance",
        "gait_cycle",
        "movement_stability",
        "gait_quality",
        "analysis_confidence",
        "joint_movement_3d",
    ):
        assert key in METRIC_HELP
        assert METRIC_HELP[key] == metric_help_body(key)
        assert "Units:" in METRIC_HELP[key]
