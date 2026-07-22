"""Tests for StableWalk analysis summary builder."""

from __future__ import annotations

from stablewalk.analysis.analysis_summary import build_analysis_summary


def test_empty_summary_shows_not_available():
    summary = build_analysis_summary(
        source="",
        biomechanical=None,
        estimated_vgrf=None,
        contact=None,
        cycles=None,
    )
    assert summary.overall_gait_quality is not None
    assert summary.overall_gait_quality.available is False
    assert summary.overall_gait_quality.value == "Not available"
    assert len(summary.gait_events) == 4
    assert all(not e.detected for e in summary.gait_events)


def test_summary_to_dict_has_terminology():
    summary = build_analysis_summary(
        source="demo",
        biomechanical=None,
        estimated_vgrf=None,
        contact=None,
        cycles=None,
    )
    payload = summary.to_dict()
    assert payload["source"] == "demo"
    assert "measured" in payload["terminology"]
    assert payload["overall_gait_quality"]["value"] == "Not available"
    assert "scientific_interpretation" in payload
    assert payload["scientific_interpretation"]
