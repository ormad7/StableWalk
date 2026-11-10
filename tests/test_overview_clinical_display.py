"""Display-only helpers for the Overview clinical summary redesign."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.ui.overview_clinical_summary import (
    INSUFFICIENT_EVIDENCE,
    NOT_AVAILABLE,
    build_data_quality_warning,
    build_overview_interpretation,
    build_overview_key_findings_texts,
    card_provenance,
    extract_overview_session_badge,
)


def test_card_provenance_tags() -> None:
    assert card_provenance("stability_state") == "Derived"
    assert card_provenance("cadence") == "Estimated"
    assert card_provenance("gait_cycles") == "Measured"


def test_interpretation_empty_and_loading() -> None:
    empty = build_overview_interpretation(
        usable_cycles=None,
        detected_cycles=None,
        cadence_spm=None,
        symmetry_pct=None,
        stable_pct=None,
        stability_margin_m=None,
        analysis_confidence_level=None,
        gait_quality_score=None,
        has_session=False,
    )
    assert "Load and analyze" in empty

    loading = build_overview_interpretation(
        usable_cycles=None,
        detected_cycles=None,
        cadence_spm=None,
        symmetry_pct=None,
        stable_pct=None,
        stability_margin_m=None,
        analysis_confidence_level=None,
        gait_quality_score=None,
        has_session=True,
        analyzing=True,
    )
    assert "progress" in loading.lower()


def test_key_findings_capped_at_three() -> None:
    texts = build_overview_key_findings_texts(
        usable_cycles=3,
        detected_cycles=3,
        cadence_spm=110.0,
        symmetry_pct=80.0,
        stable_pct=75.0,
        stability_margin_m=0.05,
        analysis_confidence_level="HIGH",
        gait_quality_score=72.0,
        max_findings=3,
    )
    assert len(texts) <= 3
    assert all("Pose-derived estimation" not in t for t in texts)


def test_data_quality_warning_only_when_insufficient() -> None:
    assert (
        build_data_quality_warning(
            analysis_confidence_level="HIGH",
            confidence_pct=80.0,
            usable_cycles=3,
            detected_cycles=3,
            tracking_score=80.0,
        )
        is None
    )
    warn = build_data_quality_warning(
        analysis_confidence_level="LOW",
        confidence_pct=30.0,
        usable_cycles=0,
        detected_cycles=2,
        tracking_score=40.0,
    )
    assert warn is not None
    assert "Data quality" in warn


def test_session_badge_includes_timestamp_and_avoids_zero_frames() -> None:
    gui = SimpleNamespace(
        _session_display_src="",
        _current_source="",
        _run_metadata=None,
        _selected_pose_backend=lambda: "mediapipe",
        skeleton_player=None,
        pose_indices=[],
        _biomech_analysis=None,
        _stability_result=None,
        _last_stability_result=None,
        _overview_analysis_timestamp=None,
    )
    badge = extract_overview_session_badge(gui)
    assert badge["video_name"] == "No video loaded"
    assert badge["analyzed_frames"] == NOT_AVAILABLE
    assert badge["analysis_timestamp"] == NOT_AVAILABLE
    assert INSUFFICIENT_EVIDENCE  # constant imported for clinical vocabulary


def test_overview_kpi_detail_uses_session_values_only() -> None:
    from stablewalk.ui.overview_clinical_summary import (
        build_overview_kpi_detail,
        resolve_overview_kpi_key,
    )

    assert resolve_overview_kpi_key("stability_margin") == "stability_state"
    assert resolve_overview_kpi_key("gait_quality") == "gait_quality"

    empty = SimpleNamespace(
        _biomech_analysis=None,
        _gait_cycle=None,
        _stability_result=None,
        _last_stability_result=None,
        _resolved_gait_cycle_count=None,
    )
    missing = build_overview_kpi_detail(empty, "cadence")
    assert missing["title"] == "Cadence"
    assert NOT_AVAILABLE in missing["body"]
    assert "60 / mean step time" in missing["body"]

    ba = SimpleNamespace(
        gait_quality=SimpleNamespace(
            score=72.0,
            confidence=0.8,
            dominant_factors=["symmetry"],
            explanation="Pose-derived composite from available components.",
        ),
        symmetry=SimpleNamespace(
            overall_symmetry_pct=SimpleNamespace(value=88.0, confidence=0.9),
            stance_symmetry=SimpleNamespace(value=0.94),
            swing_symmetry=None,
            knee_rom_symmetry=None,
        ),
        gait_metrics=SimpleNamespace(
            cadence=SimpleNamespace(value=110.0, confidence=0.75),
        ),
        stability_margin=SimpleNamespace(
            stable_pct=40.0,
            mean_margin_m=-0.026,
            min_margin_m=-0.099,
            max_margin_m=0.057,
            valid_frame_ratio=0.65,
            confidence=0.85,
        ),
        video_quality=None,
    )
    gui = SimpleNamespace(
        _biomech_analysis=ba,
        _gait_cycle=SimpleNamespace(
            metrics=SimpleNamespace(step_time_s=0.55, cadence_steps_per_min=110.0),
        ),
        _stability_result=None,
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (2, 3),
    )

    quality = build_overview_kpi_detail(gui, "gait_quality")
    assert "72/100" in quality["body"]
    assert "symmetry" in quality["body"]
    assert "Not a clinical diagnosis" in quality["body"]
    assert "Formula:" in quality["body"]
    assert "Threshold source:" in quality["body"]
    assert "heuristic" in quality["body"].lower()
    assert "Normal / Abnormal / Fast Walking" in quality["body"]

    stability = build_overview_kpi_detail(gui, "stability_margin")
    assert NOT_AVAILABLE in stability["body"]
    assert "40%" not in stability["body"]
    assert "unvalidated" in stability["body"].lower() or "n/a" in stability["body"].lower()
    assert "0.04" in stability["body"] or "classification" in stability["body"].lower()

    symmetry = build_overview_kpi_detail(gui, "symmetry")
    assert "88%" in symmetry["body"]
    assert "94%" in symmetry["body"]
    assert "Contributing components" in symmetry["body"]
    assert "Formula:" in symmetry["body"]
    assert "Missing inputs:" in symmetry["body"]
    assert "Threshold source:" in symmetry["body"]
    assert "heuristic" in symmetry["body"].lower()

    cadence = build_overview_kpi_detail(gui, "cadence")
    assert "110" in cadence["body"]
    assert "0.55 s" in cadence["body"]
    assert "2 / 3" in cadence["body"]
