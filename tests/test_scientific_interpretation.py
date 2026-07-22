"""Tests for non-clinical scientific interpretation text."""

from __future__ import annotations

from stablewalk.analysis.analysis_summary import (
    AnalysisSummary,
    GaitEventStatus,
    SummaryField,
    build_analysis_summary,
)
from stablewalk.analysis.scientific_interpretation import (
    build_scientific_interpretation,
    build_scientific_interpretation_sentences,
)
from stablewalk.ui.scientific_labels import (
    LABEL_ANALYSIS_CONFIDENCE,
    LABEL_CADENCE,
    LABEL_COM_FULL,
    LABEL_GAIT_QUALITY,
    LABEL_JOINT_ROM_SUMMARY,
    LABEL_STABILITY_MARGIN,
    LABEL_VIDEO_QUALITY,
    LABEL_VIRTUAL_GRF_PANEL,
    LABEL_WALKING_SPEED,
    TIER_DERIVED,
    TIER_ESTIMATED,
)


def _rich_summary() -> AnalysisSummary:
    return AnalysisSummary(
        source="walk_stream.mp4",
        overall_gait_quality=SummaryField(
            "Composite Gait Quality (derived)",
            "42/100",
            TIER_DERIVED,
            True,
            "Mixed cycle consistency and clearance.",
        ),
        symmetry=SummaryField("Symmetry (derived)", "58%", TIER_DERIVED, True, ""),
        stability_margin=SummaryField(
            LABEL_STABILITY_MARGIN,
            "Reduced Stability (31% stable frames)",
            TIER_DERIVED,
            True,
            "COM–base-of-support margin classification",
        ),
        center_of_mass=SummaryField(
            LABEL_COM_FULL,
            "Reduced vertical oscillation",
            TIER_ESTIMATED,
            True,
            "Vertical COM range ≈ 1.8 cm (body-normalized).",
        ),
        estimated_virtual_grf=SummaryField(
            LABEL_VIRTUAL_GRF_PANEL,
            "Normal estimated loading pattern",
            TIER_ESTIMATED,
            True,
            "Peak total ≈ 1.12 BW (estimated, not force-plate).",
        ),
        cadence=SummaryField(LABEL_CADENCE, "108 steps/min", "calculated", True, ""),
        walking_speed=SummaryField(
            LABEL_WALKING_SPEED,
            "1.15 m/s (estimated)",
            TIER_ESTIMATED,
            True,
            "",
        ),
        gait_events=[
            GaitEventStatus("Heel Strike", True, "6 pulse(s) detected"),
            GaitEventStatus("Toe Off", True, "5 pulse(s) detected"),
            GaitEventStatus("Double Support", True, "12 frame(s)"),
        ],
        video_quality=SummaryField(LABEL_VIDEO_QUALITY, "72/100", TIER_DERIVED, True, ""),
        analysis_confidence=SummaryField(
            LABEL_ANALYSIS_CONFIDENCE,
            "Moderate",
            TIER_DERIVED,
            True,
            "Composite confidence 61%",
        ),
    )


def test_interpretation_has_five_to_eight_sentences():
    sentences = build_scientific_interpretation_sentences(_rich_summary())
    assert 5 <= len(sentences) <= 8


def test_interpretation_mentions_stability_and_asymmetry():
    text = build_scientific_interpretation(_rich_summary())
    lower = text.lower()
    assert "stability" in lower or "support boundary" in lower
    assert "asymmetry" in lower or "symmetry" in lower


def test_interpretation_avoids_clinical_diagnosis_language():
    text = build_scientific_interpretation(_rich_summary()).lower()
    banned = ("diagnose", "disease", "pathology", "disorder", "patient has")
    assert not any(word in text for word in banned)
    assert "clinical diagnosis" in text or "not" in text


def test_build_analysis_summary_includes_interpretation():
    summary = build_analysis_summary(
        source="",
        biomechanical=None,
        estimated_vgrf=None,
        contact=None,
        cycles=None,
    )
    assert summary.scientific_interpretation
    payload = summary.to_dict()
    assert "scientific_interpretation" in payload
    assert payload["scientific_interpretation"]


def test_report_export_includes_interpretation_section():
    from stablewalk.io.analysis_summary_export import format_analysis_summary_report

    summary = _rich_summary()
    summary.scientific_interpretation = build_scientific_interpretation(summary)
    report = format_analysis_summary_report(summary)
    assert "Scientific Interpretation" in report
    assert "clinical diagnosis" in report.lower()
