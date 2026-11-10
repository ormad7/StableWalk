"""Analyzed span vs cumulative valid duration from original timestamps."""

from __future__ import annotations

from stablewalk.analysis.timeline_duration import (
    INTERVAL_COUNTING_CONVENTION,
    compute_timeline_duration_metrics,
    cycle_interval_pose_evidence,
)


def test_interval_convention_uses_original_timestamps_not_fps() -> None:
    times = [0.0, 0.10, 0.20, 1.00, 1.10]
    valid = [True, True, True, True, True]
    metrics = compute_timeline_duration_metrics(times, valid, fps=10.0)
    # 0.00–0.20 is one run; 1.00–1.10 is a second run (gap > 2.5/fps).
    assert abs(metrics.analyzed_span_s - 1.10) < 1e-9
    assert abs(metrics.cumulative_valid_duration_s - (0.20 + 0.10)) < 1e-9
    assert abs(metrics.longest_valid_segment_s - 0.20) < 1e-9
    assert metrics.n_valid_runs == 2
    assert abs(metrics.coverage - (0.30 / 1.10)) < 1e-9
    assert "original source timestamps" in INTERVAL_COUNTING_CONVENTION
    assert "n−1" in INTERVAL_COUNTING_CONVENTION or "n-1" in INTERVAL_COUNTING_CONVENTION


def test_single_valid_frame_contributes_zero_seconds() -> None:
    metrics = compute_timeline_duration_metrics([0.4], [True], fps=25.0)
    assert metrics.analyzed_span_s == 0.0
    assert metrics.cumulative_valid_duration_s == 0.0
    assert metrics.longest_valid_segment_s == 0.0
    assert metrics.n_valid_frames == 1


def test_invalid_frames_break_runs() -> None:
    times = [0.0, 0.04, 0.08, 0.12, 0.16]
    valid = [True, True, False, True, True]
    metrics = compute_timeline_duration_metrics(times, valid, fps=25.0)
    assert abs(metrics.cumulative_valid_duration_s - (0.04 + 0.04)) < 1e-9
    assert metrics.n_valid_runs == 2


def test_cycle_interval_reports_unsupported_internal_gap() -> None:
    times = [i * 0.1 for i in range(11)]
    valid = [i < 3 or i > 7 for i in range(11)]
    evidence = cycle_interval_pose_evidence(0.0, 1.0, times, valid, fps=10.0)
    assert evidence.n_unsupported_gaps >= 1
    assert evidence.longest_internal_gap_s > 0.25
    assert evidence.valid_pose_coverage < 0.8


def test_cycle_interval_connected_poses_have_no_gap() -> None:
    times = [i * 0.1 for i in range(11)]
    valid = [True] * 11
    evidence = cycle_interval_pose_evidence(0.0, 1.0, times, valid, fps=10.0)
    assert evidence.n_unsupported_gaps == 0
    assert evidence.longest_internal_gap_s == 0.0
    assert evidence.valid_pose_coverage == 1.0
