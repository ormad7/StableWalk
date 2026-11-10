"""Unit tests for Performance Diagnostics collector and budgets."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.ui.tk.perf_diagnostics import (
    CHART_REFRESH_BUDGET_MS,
    PERF_SECTIONS,
    UI_CALLBACK_BUDGET_MS,
    count_analysis_workers,
    count_playback_timers,
    format_perf_diagnostics_report,
    get_perf_state,
    percentile,
    set_perf_diagnostics,
    timed_section,
)


def test_percentile_p95() -> None:
    samples = [float(i) for i in range(1, 101)]
    assert percentile(samples, 95) == 95.0
    assert percentile([10.0], 95) == 10.0
    assert percentile([], 95) is None


def test_timed_section_records_avg_max_p95() -> None:
    gui = SimpleNamespace(_perf_diagnostics=False)
    set_perf_diagnostics(gui, True)
    with timed_section(gui, "decode"):
        pass
    with timed_section(gui, "frame_decode"):  # alias
        pass
    state = get_perf_state(gui)
    bucket = state.sections["decode"]
    assert bucket.count >= 2
    assert bucket.average() is not None
    assert bucket.maximum() is not None
    assert bucket.p95() is not None
    report = format_perf_diagnostics_report(gui)
    assert "PERFORMANCE DIAGNOSTICS" in report
    assert "decode:" in report
    assert "avg=" in report
    assert "p95=" in report


def test_budget_warning_on_slow_section() -> None:
    gui = SimpleNamespace(_perf_diagnostics=True)
    set_perf_diagnostics(gui, True)
    with timed_section(gui, "video", budget_ms=0.0001, warn_label="video (playback UI)"):
        # force a measurable duration
        _ = sum(range(5000))
    state = get_perf_state(gui)
    assert state.budget_warnings
    assert any("budget" in w.lower() for w in state.budget_warnings)


def test_count_playback_timers_single() -> None:
    gui = SimpleNamespace(
        _after_id="job-1",
        _comparison_mode=SimpleNamespace(_play_job=None),
    )
    assert count_playback_timers(gui) == 1
    gui._comparison_mode._play_job = "cmp"
    assert count_playback_timers(gui) == 2


def test_count_analysis_workers_mutex_fields() -> None:
    gui = SimpleNamespace(
        _pipeline_callback=None,
        _analysis_compute_active=True,
        _analysis_worker=None,
        _session_save_in_progress=False,
        _opensim_worker=None,
        _comparison_mode=None,
    )
    assert count_analysis_workers(gui) == 1


def test_budget_constants() -> None:
    assert UI_CALLBACK_BUDGET_MS == 16.0
    assert CHART_REFRESH_BUDGET_MS == 50.0


def test_perf_sections_cover_render_surfaces() -> None:
    for key in (
        "decode",
        "video",
        "skeleton",
        "trajectory",
        "3d",
        "mpl_draw",
        "tk_paint",
        "chart",
        "tab_switch",
        "startup",
    ):
        assert key in PERF_SECTIONS


def test_qt_opengl_aliases_map_to_tk_backends() -> None:
    gui = SimpleNamespace(_perf_diagnostics=True)
    set_perf_diagnostics(gui, True)
    with timed_section(gui, "qt_paint"):
        pass
    with timed_section(gui, "opengl"):
        pass
    with timed_section(gui, "trajectory_tip"):
        pass
    state = get_perf_state(gui)
    assert state.sections["tk_paint"].count >= 1
    assert state.sections["3d"].count >= 1
    assert state.sections["trajectory"].count >= 1
    report = format_perf_diagnostics_report(gui)
    assert "Tkinter + Matplotlib Agg" in report
    assert "paint signature" in report.lower()
