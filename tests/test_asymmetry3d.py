"""Tests for dynamic 3D reconstruction adapter and gait asymmetry (req 4)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from stablewalk.analysis.asymmetry3d.events import (
    _cycle_matches_heel_strike,
    cycles_aligned_to_heel_strike,
    cycles_from_heel_strikes,
    summarize_gait_events,
)
from stablewalk.analysis.asymmetry3d.export import export_asymmetry_csv, export_asymmetry_json
from stablewalk.analysis.asymmetry3d.indices import (
    ALIGNMENT_EVENT,
    absolute_symmetry_index,
    aligned_pearson,
    aligned_rmse,
    asymmetry_index,
    circular_phase_difference_pp,
    robinson_si,
)
from stablewalk.analysis.gait_feature_analysis import symmetry_index as si_norm
from stablewalk.analysis.asymmetry3d.kinematics import (
    PROVENANCE_INFERRED,
    build_kinematics,
    rom_percentile,
)
from stablewalk.analysis.asymmetry3d.laterality import assess_laterality
from stablewalk.analysis.asymmetry3d.quality import (
    UnusualInterval,
    _segment_length_check,
    resolve_unusual_interval_target,
)
from stablewalk.analysis.asymmetry3d.report import analyze_asymmetry3d
from stablewalk.analysis.gait_cycle_analysis import (
    DetectedGaitCycle,
    GaitCycleAnalysisResult,
    GaitEvent,
    GaitTemporalMetrics,
)
from stablewalk.analysis.gait_view_analysis import GaitViewEstimate, GaitViewType
from stablewalk.models.gait_motion import (
    DofSample,
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.models.joint_registry import DOF_IDS, JOINT_IDS, JOINT_PARENTS, ROOT_JOINT_ID


def _pos(jid: str, t: float, *, heel_amp: float = 0.06, freq: float = 1.0) -> Vec3:
    side = 1.0 if jid.startswith("right") else -1.0
    x = 0.10 * side
    y = 0.0
    z = 0.0
    phase = 2.0 * math.pi * freq * t
    if "right" in jid:
        phase += math.pi
    if jid.endswith("_hip"):
        y = 0.0
    elif jid.endswith("_knee"):
        y = -0.245
        z = 0.02 * math.sin(phase)
    elif jid.endswith("_ankle"):
        y = -0.491
        z = 0.04 * math.sin(phase)
    elif jid.endswith("_heel") or jid.endswith("_toe") or jid.endswith("_foot"):
        swing = max(0.0, math.sin(phase))
        y = -0.52 + heel_amp * swing
        z = 0.05 * math.sin(phase)
    elif "shoulder" in jid:
        y = 0.30
    elif "elbow" in jid:
        y = 0.12
        x = 0.18 * side
    elif "wrist" in jid:
        y = -0.02
        x = 0.22 * side
    elif jid in ("head", "neck"):
        y = 0.43
    elif jid == "spine":
        y = 0.30
    elif jid == ROOT_JOINT_ID:
        y = 0.0
        x = 0.0
    return Vec3(x, y, z)


def _walking_recording(
    *,
    fps: float = 50.0,
    n_strides: int = 6,
    cycle_s: float = 1.0,
    left_amp: float = 40.0,
    right_amp: float = 40.0,
    drop_every: int | None = None,
    heel_amp: float = 0.06,
    right_waveform_shift_pct: float = 0.0,
) -> tuple[GaitMotionRecording, GaitCycleAnalysisResult]:
    n = int(round(n_strides * cycle_s * fps))
    snaps: list[SkeletonSnapshot] = []
    freq = 1.0 / cycle_s
    for i in range(n):
        t = i / fps
        dropped = bool(drop_every and i % drop_every == 0)
        vis = 0.12 if dropped else 0.95
        detected = not dropped
        left_phase = 2.0 * math.pi * freq * t
        right_phase = (
            2.0 * math.pi * freq * (t - 0.5 * cycle_s)
            - 2.0 * math.pi * (right_waveform_shift_pct / 100.0)
        )
        left_knee = 20.0 + left_amp * 0.5 * (1.0 - math.cos(left_phase))
        right_knee = 20.0 + right_amp * 0.5 * (1.0 - math.cos(right_phase))
        joints = {}
        if detected:
            for jid in (*JOINT_IDS, ROOT_JOINT_ID):
                joints[jid] = JointSample(
                    joint_id=jid,
                    position=_pos(jid, t, heel_amp=heel_amp, freq=freq),
                    parent_id=JOINT_PARENTS.get(jid),
                )
        dofs = {}
        for dof in DOF_IDS:
            angle = 30.0
            if dof == "left_knee_flexion":
                angle = left_knee
            elif dof == "right_knee_flexion":
                angle = right_knee
            elif dof == "left_hip_flexion":
                angle = 10.0 + 0.35 * (left_knee - 20.0)
            elif dof == "right_hip_flexion":
                angle = 10.0 + 0.35 * (right_knee - 20.0)
            elif dof == "left_ankle_flexion":
                angle = 85.0 + 0.15 * (left_knee - 20.0)
            elif dof == "right_ankle_flexion":
                angle = 85.0 + 0.15 * (right_knee - 20.0)
            if dropped:
                angle = None
            dofs[dof] = DofSample(dof_id=dof, angle_deg=angle)
        snaps.append(
            SkeletonSnapshot(
                frame_index=i,
                time_s=t,
                joints=joints,
                dofs=dofs,
                metadata={
                    "detected": detected,
                    "landmark_visibility": {
                        jid: vis for jid in (*JOINT_IDS, ROOT_JOINT_ID)
                    },
                },
            )
        )

    events: list[GaitEvent] = []
    cycles: list[DetectedGaitCycle] = []
    idx = 0
    for k in range(n_strides):
        t0 = k * cycle_s
        t1 = t0 + cycle_s
        f0 = int(round(t0 * fps))
        f1 = min(int(round(t1 * fps)), n - 1)
        to_t = t0 + 0.60 * cycle_s
        to_f = int(round(to_t * fps))
        events.append(GaitEvent("left_heel_strike", f0, t0, "left"))
        events.append(GaitEvent("left_toe_off", to_f, to_t, "left"))
        cycles.append(DetectedGaitCycle(idx, f0, f1, t0, t1, cycle_s, "left"))
        idx += 1
        rt0 = t0 + 0.5 * cycle_s
        rt1 = rt0 + cycle_s
        if rt1 * fps >= n:
            break
        rf0 = int(round(rt0 * fps))
        rf1 = min(int(round(rt1 * fps)), n - 1)
        rto = rt0 + 0.60 * cycle_s
        events.append(GaitEvent("right_heel_strike", rf0, rt0, "right"))
        events.append(GaitEvent("right_toe_off", int(round(rto * fps)), rto, "right"))
        cycles.append(DetectedGaitCycle(idx, rf0, rf1, rt0, rt1, cycle_s, "right"))
        idx += 1

    recording = GaitMotionRecording(
        source="synthetic_walk",
        fps=fps,
        snapshots=snaps,
        source_kind="mock",
        coordinate_system="sw_canonical_y_up",
    )
    result = GaitCycleAnalysisResult(
        events=events,
        cycles=cycles,
        metrics=GaitTemporalMetrics(
            contact_confidence=0.82,
            metrics_reliable=True,
            gait_cycle_count=len(cycles),
        ),
        fps=fps,
    )
    return recording, result


def test_robinson_and_asi_definitions() -> None:
    assert robinson_si(10.0, 10.0) == pytest.approx(0.0)
    assert absolute_symmetry_index(10.0, 10.0) == pytest.approx(0.0)
    assert si_norm(10.0, 10.0) == pytest.approx(1.0)
    assert robinson_si(12.0, 8.0) == pytest.approx(40.0)
    assert absolute_symmetry_index(12.0, 8.0) == pytest.approx(40.0)
    assert robinson_si(8.0, 12.0) == pytest.approx(-40.0)
    assert robinson_si(0.0, 0.0) is None
    assert si_norm(0.0, 0.0) is None


def test_ai_asi_and_normalized_similarity_are_distinct() -> None:
    assert asymmetry_index(12.0, 8.0) == pytest.approx(robinson_si(12.0, 8.0))
    assert asymmetry_index(8.0, 12.0) == pytest.approx(-40.0)
    assert absolute_symmetry_index(8.0, 12.0) == pytest.approx(40.0)
    assert si_norm(12.0, 8.0) == pytest.approx(0.8)
    assert abs(float(asymmetry_index(12.0, 8.0))) != pytest.approx(float(si_norm(12.0, 8.0)))
    assert absolute_symmetry_index(12.0, 8.0) != pytest.approx(float(si_norm(12.0, 8.0)))
    assert asymmetry_index(None, 8.0) is None
    assert absolute_symmetry_index(12.0, None) is None
    assert si_norm(-4.0, 8.0) is None


def test_identical_curves_are_perfect_symmetry() -> None:
    grid = np.linspace(0.0, 2.0 * math.pi, 101)
    curve = 25.0 + 15.0 * (1.0 - np.cos(grid))
    assert aligned_rmse(curve, curve) == pytest.approx(0.0)
    assert aligned_pearson(curve, curve) == pytest.approx(1.0)
    rom = float(np.nanmax(curve) - np.nanmin(curve))
    assert robinson_si(rom, rom) == pytest.approx(0.0)
    assert absolute_symmetry_index(rom, rom) == pytest.approx(0.0)
    assert si_norm(rom, rom) == pytest.approx(1.0)


def test_circular_peak_timing_wraps_cycle_boundary() -> None:
    assert circular_phase_difference_pp(50.0, 40.0) == pytest.approx(10.0)
    assert circular_phase_difference_pp(98.0, 2.0) == pytest.approx(-4.0)
    assert circular_phase_difference_pp(2.0, 98.0) == pytest.approx(4.0)
    assert circular_phase_difference_pp(0.0, 100.0) == pytest.approx(0.0)
    assert circular_phase_difference_pp(None, 40.0) is None


def test_symmetric_gait_indices_near_zero() -> None:
    rec, cycles = _walking_recording(left_amp=40.0, right_amp=40.0)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    assert report.available
    assert report.kinematics.provenance == PROVENANCE_INFERRED
    knee = report.scalar_by_name("knee_flexion_rom")
    assert knee is not None and knee.left is not None and knee.right is not None
    assert abs(knee.left - 40.0) < 8.0
    assert abs(knee.right - 40.0) < 8.0
    assert knee.si_percent is not None and abs(knee.si_percent) < 12.0
    assert knee.asi_percent is not None and knee.asi_percent < 12.0
    pair = report.trajectory_by_name("knee_flexion")
    assert pair is not None and pair.aligned
    assert (pair.rmse or 99.0) < 8.0
    step_metric = report.scalar_by_name("step_length")
    assert step_metric is not None
    assert step_metric.left is None
    assert step_metric.reliable is False


def test_asymmetric_gait_detected() -> None:
    rec, cycles = _walking_recording(left_amp=30.0, right_amp=60.0)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    knee = report.scalar_by_name("knee_flexion_rom")
    assert knee is not None and knee.left and knee.right
    assert knee.right > knee.left + 10.0
    assert abs(knee.asi_percent or 0.0) > 20.0
    pair = report.trajectory_by_name("knee_flexion")
    assert pair is not None and pair.aligned
    assert (pair.rmse or 0.0) > 8.0
    peak = report.scalar_by_name("knee_flexion_peak")
    assert peak is not None and peak.left is not None and peak.right is not None
    assert peak.right > peak.left + 5.0
    assert pair.peak_difference is not None
    timing = report.scalar_by_name("knee_flexion_peak_phase")
    assert timing is not None
    assert timing.units == "% gait cycle"
    assert timing.si_percent is None
    assert timing.asi_percent is None
    assert timing.si_norm is None
    assert timing.difference_units == "percentage points"


def test_different_walking_speeds_phase_align() -> None:
    slow_rec, slow_c = _walking_recording(cycle_s=1.20, n_strides=5, fps=50.0)
    fast_rec, fast_c = _walking_recording(cycle_s=0.80, n_strides=6, fps=50.0)
    slow = analyze_asymmetry3d(slow_rec, cycles=slow_c)
    fast = analyze_asymmetry3d(fast_rec, cycles=fast_c)
    s_pair = slow.trajectory_by_name("knee_flexion")
    f_pair = fast.trajectory_by_name("knee_flexion")
    assert s_pair and f_pair and s_pair.aligned and f_pair.aligned
    err = aligned_rmse(s_pair.left_mean, f_pair.left_mean)
    assert err is not None and err < 6.0
    stride_s = slow.scalar_by_name("stride_duration")
    stride_f = fast.scalar_by_name("stride_duration")
    assert stride_s and stride_f
    assert (stride_s.left or 0) > (stride_f.left or 9)


def test_missing_joints_do_not_fabricate_values() -> None:
    rec, cycles = _walking_recording(drop_every=3, n_strides=5)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    # Short gaps are interpolated; dropped frames remain low-confidence.
    assert float(np.mean(report.kinematics.low_confidence)) > 0.2
    for row in report.scalars:
        if row.left is not None:
            assert math.isfinite(row.left)
        if row.right is not None:
            assert math.isfinite(row.right)


def test_unaligned_frames_are_not_compared() -> None:
    rec, _cycles = _walking_recording(n_strides=4)
    empty = GaitCycleAnalysisResult(
        fps=rec.fps,
        metrics=GaitTemporalMetrics(contact_confidence=0.0),
    )
    report = analyze_asymmetry3d(rec, cycles=empty)
    pair = report.trajectory_by_name("knee_flexion")
    assert pair is not None
    assert pair.aligned is False
    assert pair.rmse is None
    assert "Unaligned" in pair.note or "unaligned" in pair.note.lower()


def test_frontal_view_marks_spatial_unreliable() -> None:
    rec, cycles = _walking_recording()
    frontal = GaitViewEstimate(GaitViewType.FRONTAL, 0.88)
    sag = GaitViewEstimate(GaitViewType.SAGITTAL_LEFT, 0.88)
    front_events = summarize_gait_events(rec, cycles, view_estimate=frontal)
    sag_events = summarize_gait_events(rec, cycles, view_estimate=sag)
    assert front_events.spatial_reliable is False
    assert "Frontal" in " ".join(front_events.warnings)
    assert sag_events.events_reliable is True
    unknown = GaitViewEstimate(GaitViewType.UNKNOWN, 0.05)
    bad = summarize_gait_events(rec, cycles, view_estimate=unknown)
    assert bad.camera_unsuitable is True


def test_gait_events_split_stance_swing_and_timing() -> None:
    rec, cycles = _walking_recording(cycle_s=1.0, n_strides=6)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    ev = report.events
    assert len(ev.heel_strikes) >= 4
    assert len(ev.toe_offs) >= 4
    assert ev.left_cycles and ev.right_cycles
    stance = report.scalar_by_name("stance_duration")
    swing = report.scalar_by_name("swing_duration")
    step = report.scalar_by_name("step_duration")
    stride = report.scalar_by_name("stride_duration")
    assert stance and stance.left and abs(stance.left - 0.60) < 0.08
    assert swing and swing.left and abs(swing.left - 0.40) < 0.08
    assert step and step.left and abs(step.left - 0.50) < 0.08
    assert stride and stride.left and abs(stride.left - 1.00) < 0.08


def test_inferred_vs_calibrated_provenance() -> None:
    rec, cycles = _walking_recording()
    inferred = build_kinematics(rec)
    metric = build_kinematics(rec, calibrated_metric=True)
    assert inferred.provenance == PROVENANCE_INFERRED
    assert inferred.units == "body_normalized_m"
    assert metric.provenance == "calibrated_metric"
    report = analyze_asymmetry3d(rec, cycles=cycles, calibrated_metric=True)
    step = report.scalar_by_name("step_length")
    assert step is not None
    # Still view-gated; synthetic has unknown view so spatial may be limited,
    # but the metric row is no longer blanked for provenance.
    assert step.units == "m"


def test_export_json_and_csv(tmp_path: Path) -> None:
    rec, cycles = _walking_recording(n_strides=4)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    js = export_asymmetry_json(report, tmp_path / "a.json")
    csv_path = export_asymmetry_csv(report, tmp_path / "a.csv")
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["kinematics"]["provenance"] == PROVENANCE_INFERRED
    assert "robinson_si" in payload["index_definitions"]
    text = csv_path.read_text(encoding="utf-8")
    assert "# scalars" in text
    assert "phase_aligned_trajectories" in text


def test_reconstruction_note_is_monocular() -> None:
    rec, cycles = _walking_recording(n_strides=3)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    blob = report.reconstruction_assumptions.lower()
    assert "monocular" in blob or "simplified 3d" in blob
    assert "not" in blob


def test_central_diff_matches_analytic_sine_velocity_and_acceleration() -> None:
    """Finite-difference ω, α on a known sinusoid (Erik validation step 8)."""
    from stablewalk.analysis.asymmetry3d.kinematics import _central_diff

    fps = 200.0
    dt = 1.0 / fps
    t = np.arange(0.0, 2.0, dt)
    freq = 1.5
    amp = 12.0
    y = amp * np.sin(2.0 * math.pi * freq * t)
    omega = _central_diff(y, dt)
    alpha = _central_diff(omega, dt)
    omega_true = amp * 2.0 * math.pi * freq * np.cos(2.0 * math.pi * freq * t)
    alpha_true = -amp * (2.0 * math.pi * freq) ** 2 * np.sin(2.0 * math.pi * freq * t)
    sl = slice(4, -4)
    np.testing.assert_allclose(omega[sl], omega_true[sl], rtol=0.03, atol=0.8)
    np.testing.assert_allclose(alpha[sl], alpha_true[sl], rtol=0.08, atol=8.0)

    y_gap = y.copy()
    y_gap[40] = np.nan
    d_gap = _central_diff(y_gap, dt)
    assert np.isnan(d_gap[39])
    assert np.isnan(d_gap[41])


def test_build_kinematics_omega_from_known_knee_flexion() -> None:
    fps = 100.0
    rec, _cycles = _walking_recording(fps=fps, n_strides=4, left_amp=30.0, right_amp=30.0)
    kin = build_kinematics(rec)
    idx = list(kin.dof_ids).index("left_knee_flexion")
    ang = kin.angle_deg[:, idx]
    omega = kin.omega_deg_s[:, idx]
    finite = np.isfinite(ang) & np.isfinite(omega)
    assert finite.sum() > 20
    # Peak |ω| for ~30° at 1 Hz is on the order of 2π·30 ≈ 188 °/s.
    peak = float(np.nanmax(np.abs(omega)))
    assert 40.0 < peak < 500.0
    assert kin.provenance == PROVENANCE_INFERRED


def test_identical_session_exposes_formula_cycles_and_alignment() -> None:
    rec, cycles = _walking_recording(left_amp=40.0, right_amp=40.0)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    knee = report.scalar_by_name("knee_flexion_rom")
    assert knee is not None
    assert knee.n_left_cycles >= 1 and knee.n_right_cycles >= 1
    assert knee.alignment_event == ALIGNMENT_EVENT
    assert "200" in knee.formula and "min(L, R)" in knee.formula
    assert knee.si_percent is not None and abs(knee.si_percent) < 12.0
    assert knee.asi_percent is not None and knee.asi_percent < 12.0
    assert knee.si_norm is not None and knee.si_norm > 0.90
    assert report.events.alignment_verified is True
    pair = report.trajectory_by_name("knee_flexion")
    assert pair is not None and pair.aligned
    assert pair.alignment_event == ALIGNMENT_EVENT
    amp = report.scalar_by_name("traj_amp_knee_flexion")
    assert amp is not None
    assert amp.si_percent is not None and abs(amp.si_percent) < 8.0
    assert amp.si_norm is not None and amp.si_norm > 0.95
    timing = report.scalar_by_name("knee_flexion_peak_phase")
    assert timing is not None
    assert timing.comparison_kind == "timing"
    assert timing.si_percent is None and timing.si_norm is None
    assert timing.difference is not None
    assert abs(timing.difference) < 6.0
    assert report.index_definitions["overall_vs_knee"]


def test_known_phase_shift_reported_in_percentage_points() -> None:
    rec, cycles = _walking_recording(
        left_amp=40.0, right_amp=40.0, right_waveform_shift_pct=20.0
    )
    report = analyze_asymmetry3d(rec, cycles=cycles)
    pair = report.trajectory_by_name("knee_flexion")
    assert pair is not None and pair.aligned
    assert pair.peak_phase_left_pct is not None
    assert pair.peak_phase_right_pct is not None
    assert pair.peak_phase_left_pct == pytest.approx(50.0, abs=3.0)
    assert pair.peak_phase_right_pct == pytest.approx(70.0, abs=3.0)
    assert pair.timing_difference_pct == pytest.approx(-20.0, abs=4.0)
    timing = report.scalar_by_name("knee_flexion_peak_phase")
    assert timing is not None
    assert timing.units == "% gait cycle"
    assert timing.difference_units == "percentage points"
    assert timing.difference == pytest.approx(-20.0, abs=4.0)
    assert timing.si_percent is None
    assert timing.quality != "insufficient"


def test_unaligned_cycles_do_not_receive_reassuring_scores() -> None:
    rec, _cycles = _walking_recording(n_strides=4)
    empty = GaitCycleAnalysisResult(
        fps=rec.fps,
        metrics=GaitTemporalMetrics(contact_confidence=0.0),
    )
    report = analyze_asymmetry3d(rec, cycles=empty)
    knee = report.scalar_by_name("knee_flexion_rom")
    assert knee is not None
    assert knee.quality == "insufficient"
    assert knee.si_percent is None
    assert knee.asi_percent is None
    assert knee.si_norm is None
    timing = report.scalar_by_name("knee_flexion_peak_phase")
    assert timing is not None
    assert timing.quality == "insufficient"
    assert timing.difference is None
    assert timing.si_norm is None


def test_thigh_and_shank_rom_are_phase_local() -> None:
    rec, cycles = _walking_recording(n_strides=5)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    thigh = report.scalar_by_name("thigh_rom")
    shank = report.scalar_by_name("shank_rom")
    assert thigh is not None and shank is not None
    assert thigh.left is not None and thigh.right is not None
    assert shank.left is not None and shank.right is not None
    assert thigh.n_left_cycles >= 1 and thigh.n_right_cycles >= 1
    assert "unaligned" in thigh.note.lower()
    pair = report.trajectory_by_name("thigh_tilt")
    assert pair is not None and pair.aligned
    stride = report.scalar_by_name("stride_duration")
    assert stride is not None
    assert "gait cycle" in stride.display.lower()
    step = report.scalar_by_name("step_duration")
    assert step is not None
    assert "step timing" in step.display.lower()


def test_quality_checks_and_ground_clearance_gate() -> None:
    rec, cycles = _walking_recording(n_strides=5)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    names = {c.name for c in report.quality.checks}
    assert names >= {
        "segment_lengths",
        "lr_identity",
        "temporal_continuity",
        "ground_reference",
        "hka_tracking",
    }
    lr = next(c for c in report.quality.checks if c.name == "lr_identity")
    assert lr.status == "pass"
    ground = report.scalar_by_name("foot_clearance_ground")
    assert ground is not None
    if not report.quality.ground_reliable:
        assert ground.quality == "unavailable"
        assert ground.left is None
    assert report.to_dict()["clinical"] is False
    assert "not a laboratory" in " ".join(c.detail.lower() for c in report.quality.checks)


def test_asymmetric_gait_flags_unusual_phase_matched_sections() -> None:
    rec, cycles = _walking_recording(left_amp=30.0, right_amp=60.0, n_strides=6)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    assert report.quality.unusual_intervals or report.quality.unusual_phase_bands
    for interval in report.quality.unusual_intervals:
        assert interval.end_s >= interval.start_s
        assert interval.channel == "knee_flexion"
        assert "not a clinical" in interval.note.lower()


def _scale_recording(rec: GaitMotionRecording, scale: float) -> None:
    for snap in rec.snapshots:
        for jid, sample in list(snap.joints.items()):
            pos = sample.position
            snap.joints[jid] = JointSample(
                joint_id=sample.joint_id,
                position=Vec3(pos.x * scale, pos.y * scale, pos.z * scale),
                parent_id=sample.parent_id,
                angle_deg=sample.angle_deg,
            )


def test_segment_lengths_use_observed_stature_not_unit_height_table() -> None:
    rec, cycles = _walking_recording(n_strides=5)
    _scale_recording(rec, 3.0)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    kin = report.kinematics
    assert kin.stature is not None and kin.stature > 1.5
    seg = next(c for c in report.quality.checks if c.name == "segment_lengths")
    assert seg.status in {"pass", "warn"}
    assert seg.status != "fail"
    thigh = next(
        i for i, pair in enumerate(kin.bone_pairs) if pair == ("left_hip", "left_knee")
    )
    med = float(np.nanmedian(kin.bone_length[:, thigh]))
    exp = float(kin.bone_length_expected[thigh])
    assert med == pytest.approx(0.245 * 3.0, rel=0.15)
    assert exp == pytest.approx(0.245 * kin.stature, rel=0.20)


def test_segment_length_fail_is_temporal_or_homology_not_foreshorten() -> None:
    rec, _cycles = _walking_recording(n_strides=4)
    kin = build_kinematics(rec)
    check = _segment_length_check(kin)
    assert check.status in {"pass", "warn"}

    noisy = build_kinematics(rec)
    noisy.bone_length = noisy.bone_length.copy()
    noisy.bone_length[:, 0] = np.linspace(0.05, 0.90, noisy.n_frames)
    assert _segment_length_check(noisy).status == "fail"

    split = build_kinematics(rec)
    split.bone_length = split.bone_length.copy()
    left = next(i for i, pair in enumerate(split.bone_pairs) if pair == ("left_hip", "left_knee"))
    right = next(i for i, pair in enumerate(split.bone_pairs) if pair == ("right_hip", "right_knee"))
    split.bone_length[:, left] = 0.40
    split.bone_length[:, right] = 0.12
    assert _segment_length_check(split).status == "fail"

    table = build_kinematics(rec)
    table.bone_length_expected = np.full_like(table.bone_length_expected, 0.01)
    advisory = _segment_length_check(table)
    assert advisory.status in {"pass", "warn"}
    assert advisory.status != "fail"
    assert "advisory" in advisory.detail.lower() or "foreshorten" in advisory.detail.lower()


def test_facing_camera_left_on_plus_x_is_not_a_swap() -> None:
    n = 20
    left = np.column_stack(
        [np.full(n, 0.12), np.zeros(n), np.zeros(n)]
    )
    right = np.column_stack(
        [np.full(n, -0.12), np.zeros(n), np.zeros(n)]
    )
    head = np.column_stack(
        [np.zeros(n), np.full(n, 0.43), np.full(n, 0.14)]
    )
    pelvis = np.zeros((n, 3))
    facing = assess_laterality(
        left,
        right,
        left_knee=left + np.array([0.0, -0.24, 0.0]),
        right_knee=right + np.array([0.0, -0.24, 0.0]),
        head=head,
        pelvis=pelvis,
    )
    assert facing.facing == "camera"
    assert facing.status == "pass"
    assert facing.swapped is False

    rec, cycles = _walking_recording(n_strides=4)
    for snap in rec.snapshots:
        for jid, sample in list(snap.joints.items()):
            pos = sample.position
            snap.joints[jid] = JointSample(
                joint_id=sample.joint_id,
                position=Vec3(-pos.x, pos.y, pos.z),
                parent_id=sample.parent_id,
                angle_deg=sample.angle_deg,
            )
        head_s = snap.joints.get("head")
        if head_s is not None:
            snap.joints["head"] = JointSample(
                joint_id="head",
                position=Vec3(head_s.position.x, head_s.position.y, 0.15),
                parent_id=head_s.parent_id,
            )
    report = analyze_asymmetry3d(rec, cycles=cycles)
    lr = next(c for c in report.quality.checks if c.name == "lr_identity")
    assert lr.status == "pass"


def test_true_laterality_swap_still_fails() -> None:
    n = 20
    left = np.column_stack([np.full(n, 0.12), np.zeros(n), np.zeros(n)])
    right = np.column_stack([np.full(n, -0.12), np.zeros(n), np.zeros(n)])
    head = np.column_stack([np.zeros(n), np.full(n, 0.43), np.full(n, -0.14)])
    swapped = assess_laterality(
        left,
        right,
        left_knee=left + np.array([0.0, -0.24, 0.0]),
        right_knee=right + np.array([0.0, -0.24, 0.0]),
        head=head,
        pelvis=np.zeros((n, 3)),
    )
    assert swapped.facing == "away"
    assert swapped.status == "fail"
    assert swapped.swapped is True

    inconsistent = assess_laterality(
        left,
        right,
        left_knee=right + np.array([0.0, -0.24, 0.0]),
        right_knee=left + np.array([0.0, -0.24, 0.0]),
        pelvis=np.zeros((n, 3)),
    )
    assert inconsistent.status == "fail"


def test_ground_reference_keeps_monocular_uncertainty() -> None:
    rec, cycles = _walking_recording(n_strides=5)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    ground = next(c for c in report.quality.checks if c.name == "ground_reference")
    assert ground.status in {"pass", "warn", "unavailable"}
    assert "not a laboratory" in ground.detail.lower()
    if ground.status == "warn":
        assert report.quality.ground_reliable is False
        clearance = report.scalar_by_name("foot_clearance_ground")
        assert clearance is not None
        assert clearance.quality == "unavailable"


def test_heel_strike_alignment_uses_frame_when_clock_drifts() -> None:
    rec, cycles = _walking_recording(n_strides=4)
    shifted = [
        GaitEvent(event.event_type, event.frame_index, event.time_s + 0.22, event.side)
        for event in cycles.events
    ]
    cycles.events = shifted
    report = analyze_asymmetry3d(rec, cycles=cycles)
    assert report.events.left_cycles
    assert report.events.right_cycles
    knee = report.scalar_by_name("knee_flexion_rom")
    assert knee is not None
    assert knee.n_left_cycles >= 1 and knee.n_right_cycles >= 1


def test_detect_if_missing_runs_existing_cycle_detector() -> None:
    rec, _cycles = _walking_recording(n_strides=5)
    silent = summarize_gait_events(rec, None, detect_if_missing=False)
    assert any("no gait-cycle result supplied" in w.lower() for w in silent.warnings)
    detected = summarize_gait_events(rec, None, detect_if_missing=True)
    assert not any("no gait-cycle result supplied" in w.lower() for w in detected.warnings)


def test_rom_percentile_is_p95_minus_p5() -> None:
    y = np.linspace(0.0, 100.0, 101)
    assert rom_percentile(y) == pytest.approx(90.0)
    assert rom_percentile(np.array([1.0, np.nan, 2.0])) is None


def test_rom_uses_ipsilateral_hs_samples() -> None:
    rec, cycles = _walking_recording(left_amp=40.0, right_amp=40.0, n_strides=4)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    for name in (
        "knee_flexion_rom",
        "hip_flexion_rom",
        "ankle_flexion_rom",
        "thigh_rom",
        "shank_rom",
    ):
        row = report.scalar_by_name(name)
        assert row is not None, name
        assert row.left is not None and row.right is not None
        assert row.n_left_cycles >= 1 and row.n_right_cycles >= 1
        assert "P95" in (row.formula or "") or "P5" in (row.note or "")
        assert row.si_percent is not None
        assert row.asi_percent is not None
        assert row.si_norm is not None


def test_cycles_require_ipsilateral_heel_strike() -> None:
    rec, cycles = _walking_recording(n_strides=3)
    hs = [e for e in cycles.events if e.event_type == "left_heel_strike"]
    built = cycles_from_heel_strikes(list(cycles.events), "left")
    assert built
    assert all(c.boundary_side == "left" for c in built)
    assert abs(built[0].start_time_s - hs[0].time_s) < 1e-9

    stray = DetectedGaitCycle(
        cycle_index=99,
        start_frame=3,
        end_frame=20,
        start_time_s=0.37,
        end_time_s=1.37,
        duration_s=1.0,
        boundary_side="left",
    )
    assert _cycle_matches_heel_strike(stray, hs, fps=rec.fps) is False
    kept = cycles_aligned_to_heel_strike(
        [stray, built[0]], list(cycles.events), "left", fps=rec.fps
    )
    assert all(abs(c.start_time_s - stray.start_time_s) > 1e-6 for c in kept)

    report = analyze_asymmetry3d(rec, cycles=cycles)
    assert report.events.alignment_event == ALIGNMENT_EVENT
    assert report.events.alignment_verified is True
    assert report.events.left_cycles and report.events.right_cycles


def test_unusual_interval_resolves_to_recording_frame() -> None:
    rec, cycles = _walking_recording(left_amp=30.0, right_amp=60.0, n_strides=6)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    intervals = list(report.quality.unusual_intervals)
    if not intervals:
        interval = UnusualInterval(
            start_s=1.20,
            end_s=1.40,
            phase_start_pct=20.0,
            phase_end_pct=35.0,
            channel="knee_flexion",
            peak_delta=18.0,
            side="left",
            start_frame=int(round(1.20 * rec.fps)),
        )
    else:
        interval = intervals[0]
        assert interval.start_frame is not None
    frame, time_s = resolve_unusual_interval_target(interval, rec)
    assert frame is not None
    snap = rec.snapshot_at(int(frame))
    assert snap is not None
    assert abs(float(snap.time_s) - float(time_s)) < 1.5 / rec.fps
    assert abs(float(snap.time_s) - float(interval.start_s)) < 2.0 / rec.fps


def test_ground_warning_does_not_invent_calibration() -> None:
    rec, cycles = _walking_recording(n_strides=5)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    ground = next(c for c in report.quality.checks if c.name == "ground_reference")
    detail = ground.detail.lower()
    assert "force-plate" in detail or "laboratory" in detail
    assert "calibration" not in detail or "no physical calibration" in detail
    if ground.status == "warn":
        assert report.quality.ground_reliable is False
        assert "withheld" in detail
        clearance = report.scalar_by_name("foot_clearance_ground")
        assert clearance is not None
        assert clearance.quality == "unavailable"
        assert clearance.left is None
