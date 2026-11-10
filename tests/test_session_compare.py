"""Tests for Comparison Mode session metrics and diffs."""

from __future__ import annotations

from stablewalk.analysis.session_compare import (
    CompareMetrics,
    build_comparison_interpretation,
    compare_session_metrics,
    extract_compare_metrics,
    joint_angle_difference_heatmap,
)
from stablewalk.ui.tk.compare_state import SESSION_B_NEXT_ACTION
from stablewalk.ui.tk.session_cache import (
    AnalyzedSessionSnapshot,
    SessionCompareCache,
    demo_snapshot_is_compare_ready,
    load_or_analyze_demo_session,
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
        stability_label="Acceptable",
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
    # Primary gait metrics: higher Session B is improvement (abnormal vs normal → worse).
    assert by_name["Cadence difference"].tone_right == "worse"
    assert by_name["ROM difference"].tone_right == "worse"
    assert by_name["Walking speed"].tone_right == "worse"
    # Still context-neutral (no clinical better/worse).
    assert by_name["Step length difference"].tone_right == "neutral"
    assert by_name["Virtual GRF difference"].tone_right == "neutral"
    assert by_name["Timeline difference"].delta is not None
    # Directionality-defined metrics may use better/worse.
    assert by_name["Stability difference"].tone_right == "worse"
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


def test_sessions_with_different_keys_but_same_recording_are_identical(tmp_path) -> None:
    recording = tmp_path / "walk.mp4"
    left = AnalyzedSessionSnapshot(
        key="analysis-a",
        label="A",
        source=str(recording),
        n_frames=10,
    )
    right = AnalyzedSessionSnapshot(
        key="analysis-b",
        label="B",
        video_path=str(recording.parent / "." / recording.name),
        n_frames=8,
    )

    assert sessions_are_identical(left, right)


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


def test_load_or_analyze_demo_session_is_compare_ready() -> None:
    """Cached demo poses must yield a Compare-ready Session B without faking metrics."""
    snap, err = load_or_analyze_demo_session("abnormal")
    if snap is None:
        import pytest

        pytest.skip(f"abnormal demo analysis unavailable: {err}")
    assert err is None
    assert snap.key == "abnormal"
    assert snap.sequence is not None
    assert int(snap.n_frames) > 0
    assert demo_snapshot_is_compare_ready(snap)
    assert snap.metrics.knee_rom_deg is not None
    assert snap.metrics.stability_margin_m is not None
    assert snap.video_path or snap.frame_paths


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
    assert m.display_map()["Gait Quality"] == "N/A"


def test_display_map_uses_na_for_missing_metrics() -> None:
    m = CompareMetrics(label="Partial", knee_rom_deg=55.0)
    dm = m.display_map()
    assert dm["Knee ROM"] == "55°"
    assert dm["Gait Quality"] == "N/A"
    assert dm["Cadence"] == "N/A"
    assert dm["Speed"] == "N/A"


def test_compare_metrics_reuses_analyzed_cycle_cadence() -> None:
    from types import SimpleNamespace

    cycles = SimpleNamespace(
        metrics=SimpleNamespace(
            cadence_steps_per_min=108.0,
            gait_cycle_count=3,
            detected_cycle_count=4,
        )
    )

    metrics = extract_compare_metrics(label="Session B", cycles=cycles)

    assert metrics.cadence_spm == 108.0
    assert metrics.display_map()["Cadence"] == "108 steps/min"


def test_compare_metrics_reuses_available_summary_cadence() -> None:
    from types import SimpleNamespace

    summary = SimpleNamespace(
        overall_gait_quality=None,
        cadence=SimpleNamespace(available=True, value="96 steps/min"),
        symmetry=None,
        stability_margin=None,
    )

    metrics = extract_compare_metrics(label="Session B", summary=summary)

    assert metrics.cadence_spm == 96.0


def test_compare_metrics_keeps_zero_symmetry_when_analysis_reports_it() -> None:
    from types import SimpleNamespace

    class FalseyMetric:
        value = 0.0

        def __bool__(self) -> bool:
            return False

    biomechanical = SimpleNamespace(
        gait_quality=None,
        gait_metrics=None,
        symmetry=SimpleNamespace(overall_symmetry_pct=FalseyMetric()),
        stability_margin=None,
        center_of_mass=None,
    )

    metrics = extract_compare_metrics(
        label="Session B",
        biomechanical=biomechanical,
    )

    assert metrics.symmetry_pct == 0.0


def test_compare_metric_display_units_match_difference_sources() -> None:
    metrics = CompareMetrics(
        cadence_spm=102.0,
        walking_speed_m_s=1.15,
        symmetry_pct=88.0,
        stability_margin_m=0.032,
        knee_rom_deg=54.0,
    )
    display = metrics.display_map()

    assert display["Cadence"] == "102 steps/min"
    assert display["Speed"] == "1.15 m/s"
    assert display["Symmetry"] == "88%"
    assert display["Mean COM–BoS Margin"] == "0.032 m"
    assert display["Knee ROM"] == "54°"


def test_compare_metrics_ready_requires_primary_kpi() -> None:
    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    incomplete = AnalyzedSessionSnapshot(
        key="abnormal",
        label="Abnormal",
        n_frames=20,
        frame_paths=[],
        metrics=CompareMetrics(label="Abnormal", knee_rom_deg=40.0),
    )
    # ready property needs sequence/frames/skeletons
    incomplete_ready = AnalyzedSessionSnapshot(
        key="abnormal",
        label="Abnormal",
        n_frames=20,
        skeletons={0: object()},  # type: ignore[dict-item]
        metrics=CompareMetrics(label="Abnormal", knee_rom_deg=40.0, duration_s=2.0),
    )
    complete = AnalyzedSessionSnapshot(
        key="abnormal",
        label="Abnormal",
        n_frames=20,
        skeletons={0: object()},  # type: ignore[dict-item]
        metrics=CompareMetrics(
            label="Abnormal",
            cadence_spm=80.0,
            knee_rom_deg=40.0,
            duration_s=2.0,
        ),
    )
    assert not ComparisonModeController._metrics_ready(None)
    assert not ComparisonModeController._metrics_ready(incomplete)
    assert not ComparisonModeController._metrics_ready(incomplete_ready)
    assert ComparisonModeController._metrics_ready(complete)


def test_sessions_are_independent_rejects_shared_path_and_metrics() -> None:
    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController
    import numpy as np

    shared_path = np.zeros((5, 3), dtype=float)
    shared_metrics = CompareMetrics(label="Shared", cadence_spm=100.0)
    a = AnalyzedSessionSnapshot(
        key="normal",
        label="Normal",
        n_frames=10,
        skeletons={0: object()},  # type: ignore[dict-item]
        metrics=CompareMetrics(label="Normal", cadence_spm=110.0),
        path_xyz=shared_path,
    )
    b_alias = AnalyzedSessionSnapshot(
        key="abnormal",
        label="Abnormal",
        n_frames=10,
        skeletons={0: object()},  # type: ignore[dict-item]
        metrics=shared_metrics,
        path_xyz=shared_path,
    )
    b_ok = AnalyzedSessionSnapshot(
        key="abnormal",
        label="Abnormal",
        n_frames=12,
        skeletons={0: object()},  # type: ignore[dict-item]
        metrics=CompareMetrics(label="Abnormal", cadence_spm=80.0),
        path_xyz=np.ones((5, 3), dtype=float),
    )
    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    assert not ctrl._sessions_are_independent(
        a, a, left_key="normal", right_key="abnormal"
    )
    assert not ctrl._sessions_are_independent(
        a, b_alias, left_key="normal", right_key="abnormal"
    )
    assert not ctrl._sessions_are_independent(
        a, b_ok, left_key="normal", right_key="normal"
    )
    assert ctrl._sessions_are_independent(
        a, b_ok, left_key="normal", right_key="abnormal"
    )


def test_both_sessions_chart_ready_loading_and_incomplete_messages() -> None:
    """Trajectories stay locked until both independent sessions are valid."""
    from types import SimpleNamespace
    import numpy as np

    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    cache = SessionCompareCache()
    a = AnalyzedSessionSnapshot(
        key="normal",
        label="Normal",
        n_frames=20,
        skeletons={0: object()},  # type: ignore[dict-item]
        metrics=CompareMetrics(label="Normal", cadence_spm=110.0, knee_rom_deg=60.0),
        path_xyz=np.zeros((8, 3), dtype=float),
    )
    cache.pin(a)
    cache.set_slot("left", "normal")
    cache.set_slot("right", "abnormal")

    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    ctrl.gui = SimpleNamespace(_compare_session_cache=cache)
    ctrl._loading_keys = {"abnormal"}
    ctrl._failed_snapshot_keys = set()
    ctrl._playing = False
    ctrl.btn_play = None

    ready, message = ctrl._both_sessions_chart_ready()
    assert ready is False
    assert message == "Loading Session B..."

    ctrl._loading_keys.clear()
    b_incomplete = AnalyzedSessionSnapshot(
        key="abnormal",
        label="Abnormal",
        n_frames=18,
        skeletons={0: object()},  # type: ignore[dict-item]
        metrics=CompareMetrics(label="Abnormal", knee_rom_deg=40.0),
        path_xyz=np.ones((8, 3), dtype=float),
    )
    cache.pin(b_incomplete)
    ready, message = ctrl._both_sessions_chart_ready()
    assert ready is False
    assert message == f"Session B analysis is incomplete.\n{SESSION_B_NEXT_ACTION}"

    b_ok = AnalyzedSessionSnapshot(
        key="abnormal",
        label="Abnormal",
        n_frames=18,
        skeletons={0: object()},  # type: ignore[dict-item]
        metrics=CompareMetrics(
            label="Abnormal", cadence_spm=80.0, knee_rom_deg=40.0
        ),
        path_xyz=np.ones((8, 3), dtype=float),
    )
    cache.pin(b_ok)
    ready, message = ctrl._both_sessions_chart_ready()
    assert ready is True
    assert message == ""
    assert cache.left() is not cache.right()
    assert cache.left().metrics is not cache.right().metrics
    assert cache.left().path_xyz is not cache.right().path_xyz


def test_joint_angle_difference_heatmap_empty() -> None:
    grid, labels = joint_angle_difference_heatmap(None, None)
    assert len(labels) == 6
    assert grid.shape[0] == 6
    assert not __import__("numpy").isfinite(grid).any()
