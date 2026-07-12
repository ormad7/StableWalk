"""Unit tests for anthropometrically normalized gait features."""

from __future__ import annotations

import numpy as np
import pytest

from stablewalk.analysis.gait_cycle_analysis import (
    DetectedGaitCycle,
    FootLandmarkSample,
    FrameContactState,
    GaitCycleAnalysisResult,
    GaitEvent,
    GaitPhaseName,
)
from stablewalk.analysis.gait_feature_analysis import (
    FeatureNormalization,
    analyze_cycle_consistency,
    analyze_gait_features,
    estimate_body_segment_dimensions,
    gait_cycle_percent_grid,
    resample_cycle_trajectory,
    symmetry_index,
)
from stablewalk.models.gait_motion import GaitMotionRecording, JointSample, SkeletonSnapshot, Vec3


def _snap(i: int, t: float) -> SkeletonSnapshot:
    y_base = 0.9 - (i % 20) * 0.01
    joints = {
        "pelvis": JointSample("pelvis", Vec3(0.0, y_base + 0.5, 0.0), None),
        "left_hip": JointSample("left_hip", Vec3(-0.12, y_base + 0.5, 0.0), "pelvis"),
        "right_hip": JointSample("right_hip", Vec3(0.12, y_base + 0.5, 0.0), "pelvis"),
        "left_shoulder": JointSample("left_shoulder", Vec3(-0.15, y_base + 1.4, 0.0), "spine"),
        "right_shoulder": JointSample("right_shoulder", Vec3(0.15, y_base + 1.4, 0.0), "spine"),
        "left_knee": JointSample("left_knee", Vec3(-0.12, y_base + 0.25, 0.02), "left_hip"),
        "right_knee": JointSample("right_knee", Vec3(0.12, y_base + 0.25, 0.02), "right_hip"),
        "left_ankle": JointSample("left_ankle", Vec3(-0.11, y_base, 0.0), "left_knee"),
        "right_ankle": JointSample("right_ankle", Vec3(0.11, y_base, 0.0), "right_knee"),
        "left_heel": JointSample("left_heel", Vec3(-0.12, y_base - 0.02, -0.03), "left_ankle"),
        "right_heel": JointSample("right_heel", Vec3(0.12, y_base - 0.02, -0.03), "right_ankle"),
        "left_toe": JointSample("left_toe", Vec3(-0.08, y_base - 0.01, 0.05), "left_heel"),
        "right_toe": JointSample("right_toe", Vec3(0.08, y_base - 0.01, 0.05), "right_heel"),
    }
    return SkeletonSnapshot(frame_index=i, time_s=t, joints=joints, dofs={})


def _recording(n: int = 40) -> GaitMotionRecording:
    snaps = [_snap(i, i / 30.0) for i in range(n)]
    return GaitMotionRecording(source="test", fps=30.0, snapshots=snaps)


def _foot_sample(clearance: float = 0.05) -> FootLandmarkSample:
    return FootLandmarkSample(
        heel_clearance_m=clearance,
        toe_clearance_m=clearance,
        ankle_clearance_m=clearance,
        foot_clearance_m=clearance,
        heel_velocity_m_s=0.05,
        toe_velocity_m_s=0.05,
        ankle_velocity_m_s=0.05,
        visibility=0.9,
    )


def _cycles(n: int = 40) -> GaitCycleAnalysisResult:
    per_frame = []
    for i in range(n):
        phase: GaitPhaseName = "LEFT_STANCE" if (i // 10) % 2 == 0 else "RIGHT_STANCE"
        per_frame.append(
            FrameContactState(
                frame_index=i,
                time_s=i / 30.0,
                left_contact=1 if phase == "LEFT_STANCE" else 0,
                right_contact=1 if phase == "RIGHT_STANCE" else 0,
                phase=phase,
                left=_foot_sample(0.02 if phase == "LEFT_STANCE" else 0.08),
                right=_foot_sample(0.08 if phase == "LEFT_STANCE" else 0.02),
            )
        )
    events = [
        GaitEvent("left_heel_strike", 0, 0.0, "left"),
        GaitEvent("left_heel_strike", 20, 20 / 30.0, "left"),
        GaitEvent("right_heel_strike", 10, 10 / 30.0, "right"),
    ]
    cycles = [
        DetectedGaitCycle(0, 0, 19, 0.0, 19 / 30.0, 19 / 30.0),
        DetectedGaitCycle(1, 20, 39, 20 / 30.0, 39 / 30.0, 19 / 30.0),
    ]
    return GaitCycleAnalysisResult(
        per_frame=per_frame,
        events=events,
        cycles=cycles,
    )


def test_symmetry_index_stable_near_zero():
    assert symmetry_index(0.0, 0.0) is None
    assert symmetry_index(1.0, 1.0) == pytest.approx(1.0)
    assert symmetry_index(0.8, 1.2) == pytest.approx(0.8, rel=0.01)


def test_estimate_body_dimensions_uses_multiple_frames():
    rec = _recording(30)
    dims = estimate_body_segment_dimensions(rec)
    assert dims.frame_count_used >= 8
    assert dims.hip_width > 0.2
    assert dims.leg_length_average > dims.thigh_length_left * 0.5


def test_resample_cycle_has_101_samples():
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    values = [0.0, 1.0, 0.5, 1.2, 0.2]
    arr = resample_cycle_trajectory(times, values, t_start=0.0, t_end=2.0, n_samples=101)
    assert arr is not None
    assert len(arr) == 101


def test_gait_cycle_percent_grid():
    pct = gait_cycle_percent_grid(101)
    assert len(pct) == 101
    assert pct[0] == 0.0
    assert pct[-1] == 100.0


def test_analyze_gait_features_normalization_metadata():
    rec = _recording(40)
    cycles = _cycles(40)
    result = analyze_gait_features(rec, cycles)
    assert result.features.normalization["normalized_pelvis_sway"] == FeatureNormalization.BODY_NORMALIZED
    assert result.dimensions.leg_length_average > 0


def test_cycle_consistency_produces_trajectories():
    rec = _recording(40)
    cycles = _cycles(40)
    cc = analyze_cycle_consistency(rec, cycles)
    assert cc.cycle_count == 2
    assert "left_knee_angle" in cc.trajectories or "pelvis_vertical" in cc.trajectories
    if cc.trajectories:
        traj = next(iter(cc.trajectories.values()))
        assert len(traj.percent) == 101
        assert traj.normalization == FeatureNormalization.GAIT_CYCLE_NORMALIZED
