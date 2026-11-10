"""Task 4 — metric integrity: derivatives, cadence availability, demo-clip scores.

Does not retune Gait Quality to a historical GUI snapshot.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from stablewalk.analysis.biomechanical.gait_quality_score import compute_gait_quality_score
from stablewalk.analysis.gait_cycle_analysis import (
    DetectedGaitCycle,
    FootLandmarkSample,
    FrameContactState,
    GaitEvent,
    annotate_event_confidence,
    compute_temporal_metrics,
)
from stablewalk.analysis.kinematic_derivatives import (
    differentiate_chain,
    finite_diff,
    savgol_smooth,
)
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.biomechanical.orchestrator import run_biomechanical_analysis
from stablewalk.analysis.analysis_summary import build_analysis_summary
from stablewalk.ui.viewers.joint_motion_analysis_chart import build_joint_motion_series
from stablewalk import config


def _foot(clearance: float = 0.01, vis: float = 0.9) -> FootLandmarkSample:
    return FootLandmarkSample(
        heel_clearance_m=clearance,
        toe_clearance_m=clearance + 0.004,
        ankle_clearance_m=clearance + 0.008,
        foot_clearance_m=clearance,
        heel_velocity_m_s=0.0,
        toe_velocity_m_s=0.0,
        ankle_velocity_m_s=0.0,
        visibility=vis,
    )


def _two_cycles(*, fps: float = 25.0):
    n = 100
    times = [i / fps for i in range(n)]
    per_frame = [
        FrameContactState(
            frame_index=i,
            time_s=times[i],
            left_contact=1,
            right_contact=1,
            phase="DOUBLE_SUPPORT",
            left=_foot(),
            right=_foot(),
        )
        for i in range(n)
    ]
    events = [
        GaitEvent("left_heel_strike", 5, times[5], "left"),
        GaitEvent("right_heel_strike", 20, times[20], "right"),
        GaitEvent("left_heel_strike", 30, times[30], "left"),
        GaitEvent("right_heel_strike", 45, times[45], "right"),
        GaitEvent("left_heel_strike", 55, times[55], "left"),
        GaitEvent("right_heel_strike", 70, times[70], "right"),
    ]
    cycles = [
        DetectedGaitCycle(0, 5, 30, times[5], times[30], times[30] - times[5], "left"),
        DetectedGaitCycle(1, 30, 55, times[30], times[55], times[55] - times[30], "left"),
    ]
    return per_frame, events, cycles, fps


def test_low_confidence_still_reports_cadence_from_two_cycles() -> None:
    per_frame, events, cycles, fps = _two_cycles()
    metrics = compute_temporal_metrics(
        per_frame,
        events,
        cycles,
        fps=fps,
        contact_confidence=0.30,
        confidence_tier="LOW_CONFIDENCE",
    )
    assert metrics.cadence_steps_per_min is not None
    assert metrics.stride_time_s == pytest.approx(1.0, abs=1e-9)
    assert metrics.cadence_steps_per_min == pytest.approx(120.0, abs=1e-6)
    assert metrics.metrics_reliable is False
    assert "reduced reliability" in metrics.reliability_reason.lower()


def test_one_cycle_cadence_is_unavailable_not_zero() -> None:
    per_frame, events, cycles, fps = _two_cycles()
    metrics = compute_temporal_metrics(
        per_frame,
        events,
        cycles[:1],
        fps=fps,
        contact_confidence=1.0,
        confidence_tier="HIGH",
    )
    assert metrics.cadence_steps_per_min is None
    assert metrics.metrics_reliable is False


def test_event_confidence_is_foot_visibility() -> None:
    per_frame, events, _cycles, _fps = _two_cycles()
    per_frame[5] = FrameContactState(
        frame_index=5,
        time_s=per_frame[5].time_s,
        left_contact=1,
        right_contact=0,
        phase="LEFT_STANCE",
        left=_foot(vis=0.62),
        right=_foot(vis=0.91),
    )
    annotated = annotate_event_confidence(events, per_frame)
    left_hs = next(e for e in annotated if e.frame_index == 5)
    assert left_hs.confidence == pytest.approx(0.62)


def test_analytic_cubic_jerk_units_and_fps() -> None:
    j_ang = 48.0
    for fps in (25.0, 50.0, 100.0):
        t = np.arange(0.40, 1.40 + 0.5 / fps, 1.0 / fps)
        angle = (1.0 / 6.0) * j_ang * (t ** 3)
        chain = differentiate_chain(angle, t, order=3, fps=fps)
        finite = np.isfinite(chain[3])
        assert finite.sum() >= 8
        np.testing.assert_allclose(chain[3][finite], j_ang, rtol=0.05, atol=1.5)


def test_one_frame_noise_is_not_physiological_jerk() -> None:
    rng = np.random.default_rng(4)
    fps = 25.0
    t = np.arange(0.0, 4.0, 1.0 / fps)
    clean = 20.0 * np.sin(2.0 * np.pi * 1.0 * t)
    noisy = clean + rng.normal(0.0, 1.0, size=t.size)
    raw = noisy.copy()
    for _ in range(3):
        raw = finite_diff(raw, t, fps=fps)
    smoothed = differentiate_chain(noisy, t, order=3, fps=fps)[3]
    raw_peak = float(np.nanmax(np.abs(raw[12:-12])))
    sm_fin = smoothed[12:-12]
    sm_fin = sm_fin[np.isfinite(sm_fin)]
    sm_peak = float(np.nanmax(np.abs(sm_fin)))
    assert raw_peak > 5_000.0
    assert sm_peak < raw_peak * 0.35


def test_savgol_keeps_sustained_rom_step() -> None:
    y = np.concatenate([np.full(12, 10.0), np.full(12, 55.0)])
    out = savgol_smooth(y, window_samples=7, polyorder=3)
    assert out[4] == pytest.approx(10.0, abs=1.5)
    assert out[-5] == pytest.approx(55.0, abs=1.5)
    assert out[14] > 30.0


def test_gap_is_not_differenced() -> None:
    t = np.array([0.00, 0.04, 0.08, 0.20, 0.24], dtype=float)
    y = np.array([0.0, 1.0, 2.0, 20.0, 21.0], dtype=float)
    d = finite_diff(y, t, fps=25.0)
    assert np.isnan(d[2]) or abs(d[2]) < 40.0
    assert np.isnan(d[3]) or abs(float(d[3])) < 40.0


def _load_demo(stem: str):
    path = config.POSES_DIR / f"{stem}_poses.json"
    if not path.is_file():
        pytest.skip(f"{stem}_poses.json not available")
    seq = load_pose_sequence(path)
    rec = pose_sequence_to_gait_motion(seq)
    return seq, rec


def test_normal_gait_session_metrics_are_repeatable() -> None:
    seq, rec = _load_demo("normal_gait")
    first = run_biomechanical_analysis(rec, seq)
    second = run_biomechanical_analysis(rec, seq)
    assert first.gait_quality is not None and first.gait_quality.score is not None
    assert first.gait_quality.score == pytest.approx(second.gait_quality.score, abs=1e-9)
    assert first.cycles.metrics.cadence_steps_per_min == pytest.approx(
        second.cycles.metrics.cadence_steps_per_min, abs=1e-12
    )
    events_a = [(e.event_type, round(e.time_s, 4), e.side) for e in first.cycles.events]
    events_b = [(e.event_type, round(e.time_s, 4), e.side) for e in second.cycles.events]
    assert events_a == events_b
    # Historical GUI 84 and the later 26 observation are not targets.
    assert 60.0 <= first.gait_quality.score <= 90.0
    summary = build_analysis_summary(
        source="normal_gait.mp4",
        biomechanical=first,
        estimated_vgrf=None,
        contact=first.contact,
        cycles=first.cycles,
    )
    assert summary.analysis_confidence.value != "High"
    assert first.cycles.metrics.gait_cycle_count == 2


def test_abnormal_gait_differs_from_normal_without_score_tuning() -> None:
    n_seq, n_rec = _load_demo("normal_gait")
    a_seq, a_rec = _load_demo("abnormal_gait")
    normal = run_biomechanical_analysis(n_rec, n_seq)
    abnormal = run_biomechanical_analysis(a_rec, a_seq)
    assert normal.gait_quality.score is not None
    assert abnormal.gait_quality.score is not None
    assert abs(float(normal.gait_quality.score) - float(abnormal.gait_quality.score)) > 1.0
    n_cad = normal.cycles.metrics.cadence_steps_per_min
    a_cad = abnormal.cycles.metrics.cadence_steps_per_min
    assert n_cad is not None and a_cad is not None
    assert abs(n_cad - a_cad) > 1.0


def test_normal_knee_angular_jerk_below_raw_third_difference() -> None:
    from stablewalk.ui.dof_position_table import angle_value_for_item

    _seq, rec = _load_demo("normal_gait")
    series = build_joint_motion_series(rec, "right_knee")
    assert series is not None
    raw_angle = np.full(rec.frame_count, np.nan, dtype=float)
    times = np.empty(rec.frame_count, dtype=float)
    for i in range(rec.frame_count):
        snap = rec.snapshot_at(i)
        times[i] = float(snap.time_s) if snap is not None else i / rec.fps
        ang = angle_value_for_item("right_knee", snap) if snap is not None else None
        if ang is not None:
            raw_angle[i] = float(ang)
    raw = raw_angle.copy()
    for _ in range(3):
        raw = finite_diff(raw, times, fps=float(rec.fps))
    raw_fin = raw[12:-12]
    raw_peak = float(np.nanmax(np.abs(raw_fin[np.isfinite(raw_fin)])))
    sm = series.angular_jerk_deg_s3[12:-12]
    sm_peak = float(np.nanmax(np.abs(sm[np.isfinite(sm)])))
    assert raw_peak > 10_000.0
    assert sm_peak < raw_peak * 0.35


def test_gait_quality_unavailable_stays_none() -> None:
    gq = compute_gait_quality_score()
    assert gq.available is False
    assert gq.score is None


def test_replay_does_not_use_undefined_result_name() -> None:
    """Regression: Overview session-view used an unbound `result` name."""
    text = (
        config.PROJECT_ROOT / "stablewalk" / "ui" / "tk" / "app.py"
    ).read_text(encoding="utf-8")
    block = text.split("def _update_overview_clinical_summary", 1)[1].split(
        "def ", 1
    )[0]
    assert "stability_result" in block
    assert "gait_summary=getattr(result," not in block
