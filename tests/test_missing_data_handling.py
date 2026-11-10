"""Missing-data handling: gap breaks, COM placeholders, exploratory cadence."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from stablewalk.analysis.analysis_summary import _cycle_evidence_field, _field_from_evidence
from stablewalk.analysis.biomechanical.com_estimation import (
    analyze_center_of_mass,
    estimate_frame_com,
)
from stablewalk.analysis.gait_cycle_analysis import (
    DetectedGaitCycle,
    GaitTemporalMetrics,
    cadence_is_exploratory,
    summary_cadence_spm,
)
from stablewalk.analysis.joint_path_trace import series_with_gap_breaks
from stablewalk.analysis.quality_layers import (
    cycle_detection_evidence,
    pose_coverage_evidence,
)
from stablewalk.analysis.result_validity import ResultValidityReport
from stablewalk.models.gait_motion import (
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)


def test_series_with_gap_breaks_inserts_nan_and_keeps_zeros() -> None:
    fps = 30.0
    times = np.array([0.0, 1 / fps, 2 / fps, 1.0, 1.0 + 1 / fps], dtype=float)
    values = np.array([0.0, 0.1, 0.2, 0.0, 0.05], dtype=float)
    t_out, y_out = series_with_gap_breaks(times, values, fps=fps)
    assert np.isnan(y_out).any()
    assert 0.0 in [float(v) for v in y_out if np.isfinite(v)]
    nan_at = int(np.flatnonzero(np.isnan(y_out))[0])
    assert math.isclose(float(t_out[nan_at - 1]), float(times[2]), abs_tol=1e-9)
    assert math.isclose(float(t_out[nan_at + 1]), float(times[3]), abs_tol=1e-9)


def test_gap_break_separates_finite_clusters() -> None:
    fps = 30.0
    times = np.array([0.0, 1 / fps, 2 / fps, 1.0, 1.0 + 1 / fps], dtype=float)
    values = np.array([0.0, 0.1, 0.2, 0.0, 0.05], dtype=float)
    _t_out, y_out = series_with_gap_breaks(times, values, fps=fps)
    nan_idx = np.flatnonzero(np.isnan(y_out))
    assert nan_idx.size >= 1
    split = int(nan_idx[0])
    assert math.isclose(float(y_out[split - 1]), 0.2, abs_tol=1e-9)
    assert math.isclose(float(y_out[split + 1]), 0.0, abs_tol=1e-9)


def test_missing_com_is_nan_placeholder_not_origin() -> None:
    snap = SkeletonSnapshot(frame_index=0, time_s=0.0, joints={})
    pos, conf = estimate_frame_com(snap)
    assert conf == 0.0
    assert all(math.isnan(float(v)) for v in pos)


def test_valid_zero_com_is_kept() -> None:
    origin = Vec3(0.0, 0.0, 0.0)
    joints = {
        name: JointSample(joint_id=name, position=origin)
        for name in (
            "head",
            "neck",
            "spine",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_toe",
            "right_toe",
            "left_shoulder",
            "right_shoulder",
            "left_wrist",
            "right_wrist",
        )
    }
    snap = SkeletonSnapshot(frame_index=0, time_s=0.0, joints=joints)
    pos, conf = estimate_frame_com(snap)
    assert conf > 0.05
    assert all(math.isfinite(float(v)) for v in pos)
    assert all(abs(float(v)) < 1e-12 for v in pos)


def _com_snap(frame: int, time_s: float, y: float | None) -> SkeletonSnapshot:
    if y is None:
        return SkeletonSnapshot(frame_index=frame, time_s=time_s, joints={})
    hip = Vec3(0.0, float(y), 0.0)
    joints = {
        "left_hip": JointSample(joint_id="left_hip", position=hip),
        "right_hip": JointSample(joint_id="right_hip", position=hip),
    }
    return SkeletonSnapshot(frame_index=frame, time_s=time_s, joints=joints)


def test_com_derivatives_do_not_span_tracking_gaps() -> None:
    fps = 30.0
    snaps = []
    for i in range(3):
        snaps.append(_com_snap(i, i / fps, 1.0 + 0.01 * i))
    for i in range(3, 20):
        snaps.append(_com_snap(i, i / fps, None))
    for i in range(20, 23):
        snaps.append(_com_snap(i, i / fps, 1.4 + 0.01 * (i - 20)))
    rec = GaitMotionRecording(source="gap.mp4", fps=fps, snapshots=snaps)
    com = analyze_center_of_mass(rec)
    assert len(com.per_frame) == 23
    gap = com.per_frame[10]
    assert gap.confidence == 0.0
    assert all(math.isnan(float(v)) for v in gap.position)
    assert all(math.isnan(float(v)) for v in gap.velocity)
    first_after = com.per_frame[20]
    assert first_after.confidence > 0.05
    # Finite-diff must not use the last sample before the hole (frame 2).
    assert all(math.isnan(float(v)) or abs(float(v)) < 5.0 for v in first_after.velocity)
    vy = first_after.velocity[1]
    # One-sided diff inside the second cluster is ~0.01 / (1/30) = 0.3 m/s, not
    # (1.4 - 1.02) / (18/30) ≈ 0.63 m/s from spanning the hole.
    if math.isfinite(vy):
        assert abs(vy) < 0.5


def test_pose_coverage_labels_detected_and_usable() -> None:
    ev = pose_coverage_evidence(n_frames=300, n_detected=130, n_usable=110)
    assert ev.n_detected == 130
    assert ev.extras.get("n_usable") == 110
    assert "130 detected" in ev.explanation
    assert "110 usable" in ev.explanation
    field = _field_from_evidence("Pose coverage", ev)
    assert "130 detected" in field.value
    assert "110 usable" in field.value
    assert "/ 300" in field.value


def test_cycle_evidence_labels_candidates_and_accepted() -> None:
    ev = cycle_detection_evidence(
        n_cycles=2,
        n_left=2,
        n_right=0,
        n_candidates=5,
        n_accepted=2,
    )
    assert "5 candidates" in ev.explanation
    assert "2 accepted" in ev.explanation
    report = ResultValidityReport(
        pose_coverage=pose_coverage_evidence(n_frames=10, n_detected=10),
        contact_left=pose_coverage_evidence(n_frames=10, n_detected=10),
        contact_right=pose_coverage_evidence(n_frames=10, n_detected=10),
        contact_coverage=pose_coverage_evidence(n_frames=10, n_detected=10),
        spectral=pose_coverage_evidence(n_frames=10, n_detected=10),
        biomechanical=pose_coverage_evidence(n_frames=10, n_detected=10),
        cycle_detection=ev,
    )
    field = _cycle_evidence_field(report)
    assert field.value == "2 L / 0 R accepted · 5 candidates"


def test_low_confidence_and_unilateral_cadence_withheld() -> None:
    low = GaitTemporalMetrics(
        cadence_steps_per_min=108.0,
        metrics_reliable=False,
        confidence_tier="LOW_CONFIDENCE",
        reliability_reason="contact too low",
    )
    assert cadence_is_exploratory(low)
    assert summary_cadence_spm(low) is None

    cycles = SimpleNamespace(
        cycles=[
            DetectedGaitCycle(0, 0, 10, 0.0, 1.0, 1.0, "left"),
            DetectedGaitCycle(1, 10, 20, 1.0, 2.0, 1.0, "left"),
        ]
    )
    unilateral = GaitTemporalMetrics(
        cadence_steps_per_min=100.0,
        metrics_reliable=True,
        confidence_tier="MEDIUM",
    )
    assert cadence_is_exploratory(unilateral, cycles)
    assert summary_cadence_spm(unilateral, cycles) is None

    bilateral = SimpleNamespace(
        cycles=[
            DetectedGaitCycle(0, 0, 10, 0.0, 1.0, 1.0, "left"),
            DetectedGaitCycle(1, 10, 20, 1.0, 2.0, 1.0, "right"),
            DetectedGaitCycle(2, 20, 30, 2.0, 3.0, 1.0, "left"),
            DetectedGaitCycle(3, 30, 40, 3.0, 4.0, 1.0, "right"),
        ]
    )
    ok = GaitTemporalMetrics(
        cadence_steps_per_min=110.0,
        metrics_reliable=True,
        confidence_tier="HIGH",
    )
    assert not cadence_is_exploratory(ok, bilateral)
    assert summary_cadence_spm(ok, bilateral) == 110.0

    empty = GaitTemporalMetrics(
        cadence_steps_per_min=None,
        metrics_reliable=False,
        confidence_tier="LOW_CONFIDENCE",
    )
    assert not cadence_is_exploratory(empty)
    assert summary_cadence_spm(empty) is None
