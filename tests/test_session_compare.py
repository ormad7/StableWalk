"""Tests for Comparison Mode session metrics and diffs."""

from __future__ import annotations

from stablewalk.analysis.session_compare import (
    CompareMetrics,
    build_comparison_interpretation,
    compare_session_metrics,
    extract_compare_metrics,
    joint_angle_difference_heatmap,
)
from stablewalk.ui.tk.session_cache import (
    AnalyzedSessionSnapshot,
    SessionCompareCache,
    sessions_are_identical,
    try_build_demo_snapshot,
)


def test_compare_metrics_diff_tones() -> None:
    left = CompareMetrics(
        label="Normal Gait",
        session_key="normal",
        gait_quality=82.0,
        cadence_spm=118.0,
        walking_speed_m_s=1.28,
        symmetry_pct=95.0,
        stability_label="High",
        stability_margin_m=0.05,
        com_excursion_m=0.04,
        knee_rom_deg=62.0,
        hip_rom_deg=40.0,
        ankle_rom_deg=28.0,
        step_length_m=0.65,
        duration_s=4.0,
        virtual_grf_peak_bw=1.2,
        usable_gait_cycles=6,
    )
    right = CompareMetrics(
        label="Abnormal Gait",
        session_key="abnormal",
        gait_quality=61.0,
        cadence_spm=83.0,
        walking_speed_m_s=0.74,
        symmetry_pct=71.0,
        stability_label="Low",
        stability_margin_m=0.01,
        com_excursion_m=0.11,
        knee_rom_deg=40.0,
        hip_rom_deg=28.0,
        ankle_rom_deg=18.0,
        step_length_m=0.42,
        duration_s=5.5,
        virtual_grf_peak_bw=0.9,
        usable_gait_cycles=2,
    )
    result = compare_session_metrics(left, right)
    by_name = {d.name: d for d in result.diffs}
    assert by_name["Cadence difference"].tone_right == "worse"
    assert by_name["ROM difference"].tone_right == "worse"
    assert by_name["Step length difference"].tone_right == "worse"
    assert by_name["Stability difference"].tone_right == "worse"
    assert by_name["Virtual GRF difference"].tone_right == "worse"
    assert by_name["Timeline difference"].delta is not None
    assert by_name["Walking speed"].tone_right == "worse"
    assert by_name["Symmetry"].tone_right == "worse"
    assert by_name["COM excursion"].tone_right == "worse"
    assert by_name["Gait quality"].tone_right == "worse"
    assert by_name["Usable gait cycles"].tone_right == "worse"
    assert "lower walking speed" in result.interpretation.lower()
    assert "symmetry" in result.interpretation.lower()
    assert "LABORATORY" in result.lab_report_summary
    assert (
        "Step length" in result.lab_report_summary
        or "step length" in result.lab_report_summary.lower()
    )
    assert "Knee ROM" in left.display_map()
    assert "Step Length" in left.display_map()


def test_interpretation_relative_to_reference() -> None:
    left = CompareMetrics(label="healthy gait", walking_speed_m_s=1.2, symmetry_pct=90.0)
    right = CompareMetrics(label="abnormal gait", walking_speed_m_s=0.9, symmetry_pct=72.0)
    result = compare_session_metrics(left, right)
    text = build_comparison_interpretation(left, right, result.diffs)
    assert text.startswith("Compared with healthy gait")
    assert "abnormal gait" in text


def test_session_cache_defaults() -> None:
    cache = SessionCompareCache()
    assert cache.left_key == "normal"
    assert cache.right_key == "abnormal"
    snap = AnalyzedSessionSnapshot(key="normal", label="Normal Gait", n_frames=10)
    cache.pin(snap)
    assert cache.left() is snap
    assert cache.right() is None


def test_refresh_does_not_alias_b_to_a_when_abnormal_missing() -> None:
    """Missing Session B must not silently become Session A."""
    cache = SessionCompareCache()
    cache.pin(
        AnalyzedSessionSnapshot(
            key="normal",
            label="Normal Gait",
            n_frames=12,
            metrics=CompareMetrics(label="Normal", cadence_spm=110.0),
        )
    )
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    assert cache.left() is not None
    assert cache.right() is None
    assert cache.right_key == "abnormal"
    assert cache.left_key != cache.right_key


def test_sessions_are_identical_detection() -> None:
    snap = AnalyzedSessionSnapshot(key="normal", label="Normal", n_frames=10)
    assert sessions_are_identical(snap, snap)
    assert sessions_are_identical(None, None, left_key="normal", right_key="normal")
    other = AnalyzedSessionSnapshot(key="abnormal", label="Abnormal", n_frames=10)
    assert not sessions_are_identical(snap, other)


def test_normal_vs_abnormal_from_disk_nonzero() -> None:
    """Independent Normal and Abnormal pose caches must produce non-zero diffs."""
    normal = try_build_demo_snapshot("normal")
    abnormal = try_build_demo_snapshot("abnormal")
    if normal is None or abnormal is None:
        import pytest

        pytest.skip("normal_gait / abnormal_gait poses not available")

    assert normal is not abnormal
    assert normal.key == "normal"
    assert abnormal.key == "abnormal"
    assert normal.sequence is not None
    assert abnormal.sequence is not None
    assert normal.sequence is not abnormal.sequence

    result = compare_session_metrics(normal.metrics, abnormal.metrics)
    nonzero = [
        d for d in result.diffs if d.delta is not None and abs(float(d.delta)) > 1e-6
    ]
    assert nonzero, (
        "Expected non-zero Normal vs Abnormal differences; "
        f"got {[d.display for d in result.diffs]}"
    )


def test_cache_pins_independent_sessions() -> None:
    cache = SessionCompareCache()
    a = AnalyzedSessionSnapshot(
        key="normal",
        label="Normal",
        n_frames=20,
        metrics=CompareMetrics(label="Normal", cadence_spm=120.0, knee_rom_deg=60.0),
    )
    b = AnalyzedSessionSnapshot(
        key="abnormal",
        label="Abnormal",
        n_frames=18,
        metrics=CompareMetrics(label="Abnormal", cadence_spm=80.0, knee_rom_deg=40.0),
    )
    cache.pin(a)
    cache.pin(b)
    cache.set_slot("left", "normal")
    cache.set_slot("right", "abnormal")
    assert cache.left() is a
    assert cache.right() is b
    assert cache.left() is not cache.right()
    result = compare_session_metrics(cache.left().metrics, cache.right().metrics)
    cadence = next(d for d in result.diffs if d.name == "Cadence difference")
    assert cadence.delta is not None
    assert abs(float(cadence.delta)) > 1e-6


def test_extract_compare_metrics_empty() -> None:
    m = extract_compare_metrics(label="Empty", session_key="x")
    assert m.label == "Empty"
    assert m.gait_quality is None
    assert m.display_map()["Gait Quality"] == "—"


def test_joint_angle_difference_heatmap_empty() -> None:
    grid, labels = joint_angle_difference_heatmap(None, None)
    assert len(labels) == 6
    assert grid.shape[0] == 6
    assert not __import__("numpy").isfinite(grid).any()
