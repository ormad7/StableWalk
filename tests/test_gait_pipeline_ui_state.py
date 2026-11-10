"""Gait-pipeline UI state: moments copy, progress flags, Results headline."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from stablewalk.analysis.gait_asymmetry_moments import GaitAsymmetryMomentReport
from stablewalk.analysis.gait_pipeline_trace import log_stage
from stablewalk.ui.tk.analyze_progress import (
    AnalyzeProgressSnapshot,
    analyze_progress_active,
)
from stablewalk.ui.tk.asymmetric_moments_panel import (
    _FAILED_COPY,
    _INSUFFICIENT_COPY,
    _PENDING_COPY,
    _SYMMETRIC_COPY,
    empty_state_copy,
)
from stablewalk.ui.tk.dashboard_results_pipeline import update_results_analysis_pipeline


def test_moments_pending_copy_before_analysis() -> None:
    gui = SimpleNamespace(
        _gait_asymmetry_moments=None,
        _session_analysis=SimpleNamespace(ready=False),
        _gait_cycle=None,
        _biomech_analysis=None,
    )
    assert empty_state_copy(gui) == _PENDING_COPY


def test_moments_insufficient_copy_after_analysis_without_inventing() -> None:
    report = GaitAsymmetryMomentReport(
        available=False,
        cycle_duration_s=None,
        half_cycle_s=None,
        notes=[
            "Right-limb samples are compared with left-limb samples half a gait cycle later.",
            "Descriptions are measured movement differences, not medical diagnoses.",
            "Detected gait cycles are required; none were provided.",
        ],
    )
    gui = SimpleNamespace(
        _gait_asymmetry_moments=report,
        _session_analysis=SimpleNamespace(ready=True),
        _gait_cycle=object(),
        _biomech_analysis=object(),
    )
    text = empty_state_copy(gui, report)
    assert text != _PENDING_COPY
    assert "Analyze a walk" not in text
    assert "none were provided" in text.lower() or text == _INSUFFICIENT_COPY


def test_moments_insufficient_when_cache_ready_but_report_missing() -> None:
    gui = SimpleNamespace(
        _gait_asymmetry_moments=None,
        _session_analysis=SimpleNamespace(ready=True),
        _gait_cycle=object(),
        _biomech_analysis=None,
    )
    assert empty_state_copy(gui) == _INSUFFICIENT_COPY


def test_moments_symmetric_copy_when_available_without_peaks() -> None:
    report = GaitAsymmetryMomentReport(
        available=True,
        cycle_duration_s=1.0,
        half_cycle_s=0.5,
        outcome="symmetric_gait",
        notes=["Symmetric gait: no left–right difference exceeded the documented score threshold (28) for 0.08 s."],
    )
    gui = SimpleNamespace(
        _gait_asymmetry_moments=report,
        _session_analysis=SimpleNamespace(ready=True),
        _gait_cycle=object(),
        _biomech_analysis=object(),
    )
    text = empty_state_copy(gui, report)
    assert text == _SYMMETRIC_COPY or "symmetric gait" in text.lower()
    assert "insufficient" not in text.lower()
    assert "failed" not in text.lower()


def test_moments_failed_copy_is_distinct() -> None:
    report = GaitAsymmetryMomentReport(
        available=False,
        cycle_duration_s=None,
        half_cycle_s=None,
        outcome="analysis_failed",
        notes=["Asymmetry-moment detection failed: RuntimeError."],
    )
    gui = SimpleNamespace(
        _gait_asymmetry_moments=report,
        _session_analysis=SimpleNamespace(ready=True),
        _gait_cycle=object(),
        _biomech_analysis=object(),
    )
    text = empty_state_copy(gui, report)
    assert "failed" in text.lower()
    assert text != _INSUFFICIENT_COPY
    assert text != _SYMMETRIC_COPY
    assert text != _PENDING_COPY
    assert text == "Asymmetry-moment detection failed: RuntimeError." or text == _FAILED_COPY


def test_completed_progress_overlay_is_not_active_analysis() -> None:
    gui = SimpleNamespace(
        _analyze_progress_active=True,
        _analyze_progress_snapshot=AnalyzeProgressSnapshot(stage_key="completed"),
    )
    assert analyze_progress_active(gui) is False
    gui._analyze_progress_snapshot = AnalyzeProgressSnapshot(stage_key="biomechanics")
    assert analyze_progress_active(gui) is True


def test_results_pipeline_force_updates_while_playing() -> None:
    from unittest.mock import patch

    host = MagicMock()
    host.winfo_children.return_value = [object()]
    gui = SimpleNamespace(
        playing=True,
        sequence=None,
        gait_motion=None,
        _analysis_summary_cache=None,
        _analysis_compute_active=False,
        _analysis_running=False,
        _opensim_auto_pipeline_busy=False,
        _pipeline_status_context_cache=None,
        _results_pipeline_host=host,
        _results_pipeline_summary=None,
        _results_pipeline_wrap_cols=1,
        _paint_signatures={},
        smooth_motion=SimpleNamespace(get=lambda: True),
        _load_rts_report_for_session=lambda: None,
    )
    update_results_analysis_pipeline(gui, None)
    assert not hasattr(gui, "_results_pipeline_headline_kind")
    with patch(
        "stablewalk.ui.tk.playback_render_policy.paint_signature_changed",
        return_value=False,
    ):
        update_results_analysis_pipeline(gui, None, force=True)
    assert gui._results_pipeline_headline_kind == "waiting"


def test_log_stage_accepts_pipeline_statuses(caplog) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="stablewalk.gait_pipeline")
    log_stage("gait_cycle", "insufficient", cycles=0)
    assert "gait-pipeline gait_cycle: insufficient" in caplog.text
    assert "cycles=0" in caplog.text
