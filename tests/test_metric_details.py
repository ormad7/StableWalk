"""Tests for Metric Details catalog (formulas from implemented code only)."""

from __future__ import annotations

from stablewalk.ui.metric_details import (
    METRIC_DETAILS,
    get_metric_details,
    list_metric_details,
    resolve_metric_key,
)


def test_catalog_covers_core_metrics() -> None:
    required = {
        "movement_stability",
        "gait_coordination",
        "gait_quality",
        "cadence",
        "walking_speed",
        "symmetry",
        "stability_margin",
        "com",
        "vgrf",
        "rom",
        "heel_strike",
        "trajectory_path",
        "biomechanics_timeline",
    }
    assert required.issubset(METRIC_DETAILS.keys())


def test_aliases_resolve() -> None:
    assert resolve_metric_key("stability") == "stability_margin"
    assert resolve_metric_key("grf") == "vgrf"
    assert resolve_metric_key("knee_flexion") == "rom"
    assert resolve_metric_key("biomechanics") == "biomechanics_timeline"
    assert get_metric_details("walking_speed_m_s") is not None


def test_each_entry_has_required_sections() -> None:
    for md in list_metric_details():
        assert md.title
        assert md.mathematical_definition
        assert md.coordinate_system
        assert md.units
        assert md.assumptions
        assert md.limitations
        assert md.confidence_estimation
        assert md.source
        assert ".py" in md.source


def test_definitions_reference_code_symbols() -> None:
    """Spot-check that documented formulas mention symbols present in source modules."""
    cadence = get_metric_details("cadence")
    assert cadence is not None
    assert "60" in cadence.mathematical_definition
    assert "step_time" in cadence.mathematical_definition

    margin = get_metric_details("stability_margin")
    assert margin is not None
    assert "0.04" in margin.mathematical_definition

    vgrf = get_metric_details("vgrf")
    assert vgrf is not None
    assert "a_z" in vgrf.mathematical_definition or "acc" in vgrf.mathematical_definition.lower()

    gq = get_metric_details("gait_quality")
    assert gq is not None
    assert "0.22" in gq.mathematical_definition
