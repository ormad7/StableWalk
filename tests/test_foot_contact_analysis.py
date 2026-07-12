"""Tests for robust foot-contact and estimated vGRF analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stablewalk.analysis.estimated_vgrf_analysis import METHOD_NAME, analyze_estimated_vgrf
from stablewalk.analysis.foot_contact_analysis import (
    FootContactAnalysisResult,
    FootContactFrame,
    FootLandmarkSample,
    apply_probability_hysteresis,
    clearance_contact_score,
    combine_contact_probability,
    detect_event_pulses,
    horizontal_speeds_m_s,
    macro_phase_from_contacts,
    run_foot_substate_machine,
    velocity_contact_score,
)
from stablewalk.analysis.gait_cycle_analysis import (
    FrameContactState,
    GaitCycleAnalysisResult,
    GaitTemporalMetrics,
)
from stablewalk.io.contact_mask_export import export_contact_mask_npz
from stablewalk.io.foot_contact_export import export_foot_contact_artifacts
from stablewalk.io.virtual_grf_export import export_virtual_grf_npz
from stablewalk.models.gait_motion import GaitMotionRecording, JointSample, SkeletonSnapshot, Vec3
from stablewalk.real_to_sim.contact_sync_reward import (
    contact_timing_sync_reward,
    mismatch_frames,
)


def _sample(
    clearance: float,
    vel: float = 0.05,
    *,
    visibility: float = 0.9,
) -> FootLandmarkSample:
    return FootLandmarkSample(
        heel_clearance_m=clearance,
        toe_clearance_m=clearance + 0.004,
        ankle_clearance_m=clearance + 0.01,
        foot_clearance_m=clearance,
        heel_velocity_m_s=vel,
        toe_velocity_m_s=vel,
        ankle_velocity_m_s=vel,
        visibility=visibility,
    )


def _snap(i: int, t: float, *, foot_y: float = 0.0) -> SkeletonSnapshot:
    joints = {
        "left_hip": JointSample("left_hip", Vec3(-0.1, 0.9, 0.0)),
        "right_hip": JointSample("right_hip", Vec3(0.1, 0.9, 0.0)),
        "left_ankle": JointSample("left_ankle", Vec3(-0.1, foot_y, 0.0)),
        "right_ankle": JointSample("right_ankle", Vec3(0.1, foot_y + 0.02, 0.0)),
        "left_heel": JointSample("left_heel", Vec3(-0.1, foot_y - 0.01, -0.02)),
        "right_heel": JointSample("right_heel", Vec3(0.1, foot_y + 0.01, -0.02)),
        "left_toe": JointSample("left_toe", Vec3(-0.08, foot_y, 0.04)),
        "right_toe": JointSample("right_toe", Vec3(0.08, foot_y + 0.03, 0.04)),
    }
    return SkeletonSnapshot(frame_index=i, time_s=t, joints=joints, dofs={})


def _recording(n: int = 30) -> GaitMotionRecording:
    snaps = [_snap(i, i / 30.0, foot_y=0.02 * (i % 2)) for i in range(n)]
    return GaitMotionRecording(source="test", fps=30.0, snapshots=snaps)


def _cycles_alternating(n: int = 20) -> GaitCycleAnalysisResult:
    per_frame = []
    for i in range(n):
        left = 1 if (i // 5) % 2 == 0 else 0
        right = 1 - left
        per_frame.append(
            FrameContactState(
                frame_index=i,
                time_s=i / 30.0,
                left_contact=left,
                right_contact=right,
                phase="LEFT_STANCE" if left else "RIGHT_STANCE",
                left=_sample(0.02 if left else 0.10, visibility=0.85),
                right=_sample(0.10 if left else 0.02, visibility=0.85),
            )
        )
    return GaitCycleAnalysisResult(
        per_frame=per_frame,
        metrics=GaitTemporalMetrics(contact_confidence=0.75),
        fps=30.0,
    )


def test_clearance_score_hysteresis() -> None:
    low = clearance_contact_score(0.02, entry_m=0.04, exit_m=0.10, in_contact=False)
    high = clearance_contact_score(0.12, entry_m=0.04, exit_m=0.10, in_contact=False)
    assert low is not None and high is not None
    assert low > high
    mid_in = clearance_contact_score(0.06, entry_m=0.04, exit_m=0.10, in_contact=True)
    mid_out = clearance_contact_score(0.06, entry_m=0.04, exit_m=0.10, in_contact=False)
    assert mid_in is not None and mid_out is not None
    assert mid_in >= mid_out


def test_velocity_contact_score_prefers_slow_feet() -> None:
    slow = velocity_contact_score(0.05, 0.04, entry_max_m_s=0.30, exit_min_m_s=0.45, in_contact=False)
    fast = velocity_contact_score(0.60, 0.55, entry_max_m_s=0.30, exit_min_m_s=0.45, in_contact=False)
    assert slow is not None and fast is not None
    assert slow > fast


def test_combine_probability_respects_low_visibility() -> None:
    high_vis = combine_contact_probability(
        heel_score=0.9, toe_score=0.85, vel_score=0.8, visibility=0.9
    )
    low_vis = combine_contact_probability(
        heel_score=0.9, toe_score=0.85, vel_score=0.8, visibility=0.2
    )
    assert high_vis > low_vis


def test_probability_hysteresis_prevents_flicker() -> None:
    noisy = [0.55, 0.60, 0.56, 0.50, 0.59, 0.45, 0.30]
    states = apply_probability_hysteresis(noisy, enter=0.58, exit_threshold=0.42)
    assert states[0] == 0
    assert 1 in states
    entered = False
    flips = 0
    prev = states[0]
    for s in states[1:]:
        if s != prev:
            flips += 1
        if s == 1:
            entered = True
        prev = s
    assert entered
    assert flips <= 3


def test_detect_heel_strike_and_toe_off() -> None:
    contact = [0, 0, 1, 1, 1, 0, 0]
    hs, to, events, hs_p, to_p = detect_event_pulses(contact, "left")
    assert len(hs) == 1
    assert len(to) == 1
    assert hs_p[2] == 1
    assert to_p[5] == 1
    assert any(e.event_type == "left_heel_strike" for e in events)


def test_macro_phase_double_support() -> None:
    assert macro_phase_from_contacts(1, 1) == "double_support"
    assert macro_phase_from_contacts(1, 0) == "stance"
    assert macro_phase_from_contacts(0, 0) == "swing"


def test_foot_substate_machine_progression() -> None:
    contact = [0, 0, 1, 1, 1, 1, 1, 0, 0]
    samples = [_sample(0.12 if c == 0 else 0.02) for c in contact]
    states = run_foot_substate_machine(
        contact,
        samples,
        heel_strike_frames=[2],
        toe_off_frames=[7],
        fps=30.0,
    )
    assert states[0] == "swing"
    assert states[2] == "heel_strike"
    assert states[4] in ("foot_flat", "mid_stance", "heel_strike")
    assert states[7] == "toe_off"
    assert states[8] == "swing"


def test_horizontal_speed_from_trajectory() -> None:
    xs = [0.0, 0.1, 0.2, 0.3]
    zs = [0.0, 0.0, 0.0, 0.0]
    times = [0.0, 0.1, 0.2, 0.3]
    speeds = horizontal_speeds_m_s(xs, zs, times)
    assert speeds[1] == pytest.approx(1.0, rel=0.01)


def test_contact_timing_sync_and_mismatch() -> None:
    ref = np.array([1, 1, 0, 0, 1], dtype=np.int8)
    cmp = np.array([1, 0, 0, 0, 1], dtype=np.int8)
    reward = contact_timing_sync_reward(ref, cmp)
    assert reward[0] == 1.0
    assert reward[1] == 0.0
    mm = mismatch_frames(ref, cmp)
    assert mm[1]
    assert int(mm.sum()) == 1


def test_export_contact_mask_npz(tmp_path: Path) -> None:
    frames = [
        FootContactFrame(
            frame_index=0,
            time_s=0.0,
            left_contact_probability=0.9,
            right_contact_probability=0.1,
            left_contact_binary=1,
            right_contact_binary=0,
            left_heel_strike=1,
            right_heel_strike=0,
            left_toe_off=0,
            right_toe_off=0,
            left_foot_substate="heel_strike",
            right_foot_substate="swing",
            macro_phase="stance",
            left_confidence=0.9,
            right_confidence=0.8,
        )
    ]
    contact = FootContactAnalysisResult(per_frame=frames, fps=30.0)
    out = tmp_path / "contact_mask.npz"
    export_contact_mask_npz(contact, out, run_name="test_run")
    data = np.load(out)
    assert data["left_contact_probability"][0] == pytest.approx(0.9)
    assert int(data["left_heel_strike"][0]) == 1
    assert data["double_support_phase"].shape == (1,)


def test_vgrf_normalization_body_weight(tmp_path: Path) -> None:
    n = 10
    frames = []
    for i in range(n):
        left_c = 1 if i < 6 else 0
        frames.append(
            FootContactFrame(
                frame_index=i,
                time_s=i / 30.0,
                left_contact_probability=0.9 if left_c else 0.1,
                right_contact_probability=0.1,
                left_contact_binary=left_c,
                right_contact_binary=0,
                left_heel_strike=1 if i == 0 else 0,
                right_heel_strike=0,
                left_toe_off=1 if i == 6 else 0,
                right_toe_off=0,
                left_foot_substate="mid_stance" if left_c else "swing",
                right_foot_substate="swing",
                macro_phase="stance" if left_c else "swing",
                left_confidence=0.85,
                right_confidence=0.85,
            )
        )
    contact = FootContactAnalysisResult(per_frame=frames, fps=30.0)
    rec = _recording(n)
    vgrf = analyze_estimated_vgrf(rec, contact, body_mass_kg=70.0)
    assert vgrf.available
    assert vgrf.method_name == METHOD_NAME
    assert len(vgrf.left_vgrf_bw) == n
    bw = 70.0 * 9.81
    assert np.allclose(vgrf.left_vgrf_bw, vgrf.left_vgrf_vertical / bw)
    assert vgrf.metrics.peak_force_bw == pytest.approx(vgrf.metrics.peak_force_n / bw, rel=0.01)

    out = tmp_path / "virtual_grf.npz"
    export_virtual_grf_npz(vgrf, out)
    loaded = np.load(out)
    assert str(loaded["method_name"]) == METHOD_NAME


def test_short_contact_gap_smoothed_by_hysteresis() -> None:
    probs = [0.1, 0.65, 0.62, 0.1, 0.1]
    states = apply_probability_hysteresis(probs, enter=0.58, exit_threshold=0.42)
    assert states[1] == 1


def test_low_confidence_reduces_combined_probability() -> None:
    s_low = combine_contact_probability(
        heel_score=0.8,
        toe_score=0.75,
        vel_score=0.7,
        visibility=0.15,
    )
    s_ok = combine_contact_probability(
        heel_score=0.8,
        toe_score=0.75,
        vel_score=0.7,
        visibility=0.85,
    )
    assert s_ok > s_low


def test_foot_contact_export_pipeline(tmp_path: Path) -> None:
    rec = _recording(24)
    cycles = _cycles_alternating(24)
    result = export_foot_contact_artifacts(
        rec,
        tmp_path / "run_a",
        run_name="run_a",
        cycles=cycles,
    )
    assert result.contact_mask_path.is_file()
    assert result.virtual_grf_path.is_file()
    assert result.contact_sync_path is not None
    sync = np.load(result.contact_sync_path)
    assert "mean_sync_score" in sync
    assert "mismatch_frames" in sync
