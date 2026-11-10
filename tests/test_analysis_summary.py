"""Tests for StableWalk analysis summary builder."""

from __future__ import annotations

from types import SimpleNamespace

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


def test_summary_cadence_falls_back_to_cycle_timing():
    ba = SimpleNamespace(
        gait_metrics=SimpleNamespace(
            cadence=SimpleNamespace(value=None, note="Insufficient direct cadence"),
            walking_speed=None,
        ),
        gait_quality=None,
        symmetry=None,
        stability_margin=None,
        center_of_mass=None,
        joint_rom=None,
        video_quality=None,
    )
    cycles = SimpleNamespace(
        metrics=SimpleNamespace(
            cadence_steps_per_min=108.0,
            contact_confidence=0.8,
            gait_cycle_count=2,
        )
    )

    summary = build_analysis_summary(
        source="cycle-only",
        biomechanical=ba,
        estimated_vgrf=None,
        contact=None,
        cycles=cycles,
    )

    assert summary.cadence.available is True
    assert summary.cadence.value == "108 steps/min"
    assert "gait-cycle timing" in summary.cadence.reason


def test_unreportable_com_bos_is_not_exported_as_stability_kpi():
    ba = SimpleNamespace(
        gait_metrics=None,
        gait_quality=None,
        symmetry=None,
        stability_margin=SimpleNamespace(
            stable_pct=5.83,
            reportable=False,
            experimental_note=(
                "Monocular COM–BoS is experimental and is not a physical stability score."
            ),
            per_frame=[],
            warnings=[],
        ),
        center_of_mass=None,
        joint_rom=None,
        video_quality=None,
    )
    summary = build_analysis_summary(
        source="normal_gait.mp4",
        biomechanical=ba,
        estimated_vgrf=None,
        contact=None,
        cycles=None,
    )
    assert summary.stability_margin is not None
    assert summary.stability_margin.available is False
    assert summary.stability_margin.value == "Not available"
    assert "5.83" not in summary.stability_margin.value
    assert "reduced" not in summary.stability_margin.value.lower()
    blob = (summary.scientific_interpretation or "").lower()
    assert "5.83" not in blob
    assert "markedly reduced stability" not in blob
    payload = summary.to_dict()
    assert payload["stability_margin"]["available"] is False
    assert payload["stability_margin"]["value"] == "Not available"
