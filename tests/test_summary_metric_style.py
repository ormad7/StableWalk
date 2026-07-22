"""Tests for Results Summary metric color interpretation."""

from __future__ import annotations

from stablewalk.analysis.analysis_summary import SummaryField
from stablewalk.ui.scientific_labels import TIER_DERIVED, TIER_ESTIMATED
from stablewalk.ui.summary_metric_style import (
    format_status_line,
    interpret_gait_event,
    interpret_summary_metric,
    metric_visual_style,
)
from stablewalk.ui.theme import DANGER, ORANGE, SUCCESS


def test_gait_quality_color_bands() -> None:
    assert interpret_summary_metric("gait_quality", SummaryField("", "72/100", TIER_DERIVED, True)) == "normal"
    assert interpret_summary_metric("gait_quality", SummaryField("", "52/100", TIER_DERIVED, True)) == "borderline"
    assert interpret_summary_metric("gait_quality", SummaryField("", "42/100", TIER_DERIVED, True)) == "abnormal"


def test_symmetry_and_stability_interpretation() -> None:
    assert interpret_summary_metric("symmetry", SummaryField("", "78%", TIER_DERIVED, True)) == "normal"
    assert interpret_summary_metric("symmetry", SummaryField("", "58%", TIER_DERIVED, True)) == "borderline"
    assert interpret_summary_metric("symmetry", SummaryField("", "41%", TIER_DERIVED, True)) == "abnormal"
    assert (
        interpret_summary_metric(
            "stability_margin",
            SummaryField("", "Stable (82% stable frames)", TIER_DERIVED, True),
        )
        == "normal"
    )
    assert (
        interpret_summary_metric(
            "stability_margin",
            SummaryField("", "Reduced Stability (31% stable frames)", TIER_DERIVED, True),
        )
        == "borderline"
    )
    assert (
        interpret_summary_metric(
            "stability_margin",
            SummaryField("", "Unstable (18% stable frames)", TIER_DERIVED, True),
        )
        == "abnormal"
    )


def test_biomechanics_text_fields() -> None:
    assert (
        interpret_summary_metric(
            "center_of_mass",
            SummaryField("", "Normal oscillation", TIER_ESTIMATED, True),
        )
        == "normal"
    )
    assert (
        interpret_summary_metric(
            "vgrf",
            SummaryField("", "Normal estimated loading pattern", TIER_ESTIMATED, True),
        )
        == "normal"
    )
    assert (
        interpret_summary_metric(
            "center_of_mass",
            SummaryField("", "Reduced vertical oscillation", TIER_ESTIMATED, True),
        )
        == "borderline"
    )


def test_confidence_scores_use_green_yellow_red() -> None:
    assert interpret_summary_metric("video_quality", SummaryField("", "80/100", TIER_DERIVED, True)) == "normal"
    assert interpret_summary_metric("video_quality", SummaryField("", "61/100", TIER_DERIVED, True)) == "borderline"
    assert interpret_summary_metric("video_quality", SummaryField("", "44/100", TIER_DERIVED, True)) == "abnormal"


def test_cadence_bands_accept_monocular_demo_rates() -> None:
    assert (
        interpret_summary_metric("cadence", SummaryField("", "63 spm", TIER_DERIVED, True))
        == "normal"
    )
    assert (
        interpret_summary_metric("cadence", SummaryField("", "100 spm", TIER_DERIVED, True))
        == "normal"
    )
    assert (
        interpret_summary_metric("cadence", SummaryField("", "40 spm", TIER_DERIVED, True))
        == "abnormal"
    )


def test_frontal_view_softens_unstable_stability() -> None:
    field = SummaryField("", "Unstable (0% stable frames)", TIER_DERIVED, True)
    assert interpret_summary_metric("stability_margin", field) == "abnormal"
    assert (
        interpret_summary_metric("stability_margin", field, view_type="FRONTAL")
        == "borderline"
    )


def test_gait_event_detected_is_normal() -> None:
    assert interpret_gait_event(detected=True) == "normal"
    assert interpret_gait_event(detected=False) == "abnormal"


def test_status_line_includes_tier() -> None:
    line = format_status_line("normal", TIER_DERIVED)
    assert "Normal" in line
    assert "derived" in line


def test_visual_style_uses_theme_colors() -> None:
    assert metric_visual_style("normal").value_fg == SUCCESS
    assert metric_visual_style("borderline").value_fg == ORANGE
    assert metric_visual_style("abnormal").value_fg == DANGER
