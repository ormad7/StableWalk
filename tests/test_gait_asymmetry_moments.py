"""Half-cycle gait asymmetry moments: localization, merge, confidence, wording."""

from __future__ import annotations

import math

from stablewalk.analysis.gait_asymmetry_moments import (
    MIN_DURATION_S,
    SCORE_THRESHOLD,
    detect_gait_asymmetry_moments,
)
from stablewalk.analysis.gait_cycle_analysis import (
    DetectedGaitCycle,
    GaitCycleAnalysisResult,
    GaitEvent,
    GaitTemporalMetrics,
)
from stablewalk.models.gait_motion import (
    DofSample,
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.models.joint_registry import DOF_IDS, JOINT_IDS, JOINT_PARENTS, ROOT_JOINT_ID


def _pos(jid: str, t: float, *, freq: float = 1.0) -> Vec3:
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
    elif jid.endswith("_ankle"):
        y = -0.491
        z = 0.04 * math.sin(phase)
    elif jid.endswith("_heel") or jid.endswith("_toe") or jid.endswith("_foot"):
        swing = max(0.0, math.sin(phase))
        y = -0.52 + 0.06 * swing
        z = 0.05 * math.sin(phase)
    elif "shoulder" in jid:
        y = 0.30
    elif jid in ("head", "neck"):
        y = 0.43
    elif jid == "spine":
        y = 0.30
    elif jid == ROOT_JOINT_ID:
        y = 0.0
        x = 0.02 * math.sin(2.0 * math.pi * freq * t)
    return Vec3(x, y, z)


def _recording(
    *,
    fps: float = 50.0,
    n_strides: int = 6,
    cycle_s: float = 1.0,
    knee_spike_s: float | None = None,
    spike_width_s: float = 0.12,
    spike_deg: float = 40.0,
    second_spike_s: float | None = None,
    low_vis_s: float | None = None,
) -> tuple[GaitMotionRecording, GaitCycleAnalysisResult]:
    n = int(round(n_strides * cycle_s * fps))
    snaps: list[SkeletonSnapshot] = []
    freq = 1.0 / cycle_s
    for i in range(n):
        t = i / fps
        vis = 0.2 if (low_vis_s is not None and abs(t - low_vis_s) <= 0.08) else 0.95
        left_phase = 2.0 * math.pi * freq * t
        right_phase = 2.0 * math.pi * freq * (t - 0.5 * cycle_s)
        left_knee = 20.0 + 20.0 * (1.0 - math.cos(left_phase))
        right_knee = 20.0 + 20.0 * (1.0 - math.cos(right_phase))
        if knee_spike_s is not None and abs(t - knee_spike_s) <= spike_width_s:
            right_knee += spike_deg
        if second_spike_s is not None and abs(t - second_spike_s) <= spike_width_s:
            right_knee += spike_deg
        joints = {
            jid: JointSample(joint_id=jid, position=_pos(jid, t, freq=freq), parent_id=JOINT_PARENTS.get(jid))
            for jid in (*JOINT_IDS, ROOT_JOINT_ID)
        }
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
            dofs[dof] = DofSample(dof_id=dof, angle_deg=angle)
        snaps.append(
            SkeletonSnapshot(
                frame_index=i,
                time_s=t,
                joints=joints,
                dofs=dofs,
                metadata={
                    "detected": True,
                    "landmark_visibility": {jid: vis for jid in (*JOINT_IDS, ROOT_JOINT_ID)},
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
        events.append(GaitEvent("left_heel_strike", f0, t0, "left", 0.9))
        events.append(GaitEvent("left_toe_off", int(round((t0 + 0.6 * cycle_s) * fps)), t0 + 0.6 * cycle_s, "left", 0.9))
        cycles.append(DetectedGaitCycle(idx, f0, f1, t0, t1, cycle_s, "left"))
        idx += 1
        rt0 = t0 + 0.5 * cycle_s
        rt1 = rt0 + cycle_s
        if rt1 * fps >= n:
            continue
        rf0 = int(round(rt0 * fps))
        rf1 = min(int(round(rt1 * fps)), n - 1)
        events.append(GaitEvent("right_heel_strike", rf0, rt0, "right", 0.9))
        events.append(
            GaitEvent(
                "right_toe_off",
                int(round((rt0 + 0.6 * cycle_s) * fps)),
                rt0 + 0.6 * cycle_s,
                "right",
                0.9,
            )
        )
        cycles.append(DetectedGaitCycle(idx, rf0, rf1, rt0, rt1, cycle_s, "right"))
        idx += 1

    result = GaitCycleAnalysisResult(
        per_frame=[],
        events=events,
        cycles=cycles,
        metrics=GaitTemporalMetrics(stride_time_s=cycle_s, gait_cycle_count=len(cycles)),
        fps=fps,
    )
    rec = GaitMotionRecording(
        source="synthetic_asymmetry_moments",
        fps=fps,
        snapshots=snaps,
        source_kind="mock",
        coordinate_system="sw_canonical_y_up",
    )
    return rec, result


def test_symmetric_walking_has_no_meaningful_moments() -> None:
    rec, cycles = _recording()
    report = detect_gait_asymmetry_moments(rec, cycles)
    assert report.available is True
    assert report.outcome == "symmetric_gait"
    assert report.half_cycle_s == 0.5
    used_scores = [s.score for s in report.per_frame if s.used and s.score is not None]
    assert used_scores
    assert max(used_scores) < SCORE_THRESHOLD
    assert report.moments == []
    names = {row.name for row in report.joint_comparisons}
    assert {"hip_angle", "knee_angle", "ankle_angle"} <= names
    knee = next(row for row in report.joint_comparisons if row.name == "knee_angle")
    assert knee.rom_left is not None and knee.rom_right is not None
    assert abs(knee.rom_left - knee.rom_right) < 2.0
    assert knee.si_norm is not None and knee.si_norm > 0.9
    assert knee.si_percent is not None and abs(knee.si_percent) < 8.0
    assert knee.asi_percent is not None and knee.asi_percent < 8.0
    assert knee.peak_difference is not None and abs(knee.peak_difference) < 2.0
    assert knee.timing_difference_pct is not None and abs(knee.timing_difference_pct) < 8.0
    assert knee.timing_difference_units == "percentage points"
    assert knee.quality == "derived"
    assert knee.n_left_cycles >= 1 and knee.n_right_cycles >= 1
    used = [s for s in report.per_frame if s.used]
    assert used
    assert all(s.phase_pct is not None for s in used)
    assert all(s.match_phase_pct == s.phase_pct for s in used)


def test_knee_spike_is_top_moment_with_explanation() -> None:
    rec, cycles = _recording(knee_spike_s=2.4, spike_width_s=0.10, spike_deg=45.0)
    report = detect_gait_asymmetry_moments(rec, cycles)
    assert report.available is True
    assert len(report.moments) >= 1
    assert len(report.moments) <= 3
    top = report.moments[0]
    assert abs(top.time_s - 2.4) <= 0.12
    assert "right_knee" in top.involved_joints or "left_knee" in top.involved_joints
    names = [c.measurement for c in top.contributing_measurements]
    assert "knee_angle" in names
    assert top.severity in {"mild", "moderate", "marked"}
    text = top.explanation.lower()
    assert "knee" in text
    assert "frame" in text
    banned = ("diagnos", "disease", "patholog", "disorder", "injury", "trendelenburg", "limp")
    assert not any(word in text for word in banned)
    assert report.to_dict()["clinical"] is False
    assert report.outcome == "moments"
    top_knee = next(
        (c for c in top.contributing_measurements if c.measurement == "knee_angle"),
        None,
    )
    assert top_knee is not None
    assert top_knee.left_value is not None and top_knee.right_value is not None
    assert top.confidence >= 0.4
    assert top.duration_s >= MIN_DURATION_S


def test_adjacent_detections_are_merged() -> None:
    rec, cycles = _recording(
        knee_spike_s=2.40,
        second_spike_s=2.50,
        spike_width_s=0.06,
        spike_deg=40.0,
    )
    report = detect_gait_asymmetry_moments(rec, cycles)
    near = [m for m in report.moments if abs(m.time_s - 2.45) <= 0.25]
    assert len(near) == 1


def test_separated_spikes_remain_distinct() -> None:
    rec, cycles = _recording(
        knee_spike_s=1.8,
        second_spike_s=3.6,
        spike_width_s=0.08,
        spike_deg=42.0,
    )
    report = detect_gait_asymmetry_moments(rec, cycles)
    assert len(report.moments) >= 2
    times = sorted(m.time_s for m in report.moments[:2])
    assert abs(times[1] - times[0]) >= 1.0


def test_low_confidence_frames_are_ignored() -> None:
    rec, cycles = _recording(knee_spike_s=2.4, spike_width_s=0.10, spike_deg=45.0, low_vis_s=2.4)
    report = detect_gait_asymmetry_moments(rec, cycles)
    around = [s for s in report.per_frame if abs(s.time_s - 2.4) <= 0.06]
    assert around
    assert all(s.used is False for s in around)
    assert all(abs(m.time_s - 2.4) > 0.08 for m in report.moments)


def test_missing_cycles_are_unavailable() -> None:
    rec, _cycles = _recording()
    report = detect_gait_asymmetry_moments(rec, None)
    assert report.available is False
    assert report.outcome == "insufficient_data"
    assert report.moments == []
    assert report.joint_comparisons == []
    assert any("cycle" in n.lower() for n in report.notes)


def test_single_frame_spike_is_below_minimum_duration() -> None:
    rec, cycles = _recording(knee_spike_s=2.4, spike_width_s=0.0, spike_deg=45.0)
    report = detect_gait_asymmetry_moments(rec, cycles)
    near = [m for m in report.moments if abs(m.time_s - 2.4) <= 0.04]
    assert near == []


def test_one_sided_detector_cycles_are_filled_from_heel_strikes() -> None:
    rec, cycles = _recording()
    left_only = [c for c in cycles.cycles if c.boundary_side == "left"]
    assert left_only
    cycles.cycles = left_only
    report = detect_gait_asymmetry_moments(rec, cycles)
    assert report.n_left_cycles >= 1
    assert report.n_right_cycles >= 1
    assert report.available is True
