"""Tests for virtual GRF estimator architecture."""

from __future__ import annotations

import numpy as np

from stablewalk.analysis.gait_cycle_analysis import (
    FootLandmarkSample,
    FrameContactState,
    GaitCycleAnalysisResult,
    GaitTemporalMetrics,
)
from stablewalk.analysis.virtual_grf import (
    SCIENTIFIC_DISCLAIMER,
    LearnedForceEstimator,
    PhysicsSimulationForceEstimator,
    UnavailableForceEstimator,
    VirtualForceMethod,
    build_virtual_force_input,
    estimate_virtual_grf,
)
from stablewalk.models.gait_motion import GaitMotionRecording, JointSample, SkeletonSnapshot, Vec3


def _snap(frame: int, t: float) -> SkeletonSnapshot:
    joints = {
        "left_hip": JointSample("left_hip", Vec3(-0.1, 0.9, 0.0)),
        "right_hip": JointSample("right_hip", Vec3(0.1, 0.9, 0.0)),
        "left_knee": JointSample("left_knee", Vec3(-0.1, 0.5, 0.05)),
        "right_knee": JointSample("right_knee", Vec3(0.1, 0.5, 0.05)),
    }
    return SkeletonSnapshot(frame_index=frame, time_s=t, joints=joints, dofs={})


def _recording(n: int = 10) -> GaitMotionRecording:
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


def _cycles(recording: GaitMotionRecording) -> GaitCycleAnalysisResult:
    per_frame = [
        FrameContactState(
            frame_index=s.frame_index,
            time_s=s.time_s,
            left_contact=i % 2,
            right_contact=1 - (i % 2),
            phase="LEFT_STANCE" if i % 2 else "RIGHT_STANCE",
            left=_foot_sample(0.02 if i % 2 else 0.08),
            right=_foot_sample(0.08 if i % 2 else 0.02),
        )
        for i, s in enumerate(recording.snapshots)
    ]
    return GaitCycleAnalysisResult(
        per_frame=per_frame,
        metrics=GaitTemporalMetrics(contact_confidence=0.8),
        fps=recording.fps,
    )


def test_unavailable_estimator_disclaimer() -> None:
    rec = _recording()
    cycles = _cycles(rec)
    result = estimate_virtual_grf(rec, cycles, estimator=UnavailableForceEstimator())
    assert result.available is False
    assert result.method == VirtualForceMethod.UNAVAILABLE
    assert "not configured" in result.notes[0].lower()
    assert "measured" in result.scientific_disclaimer.lower()
    assert SCIENTIFIC_DISCLAIMER in result.scientific_disclaimer


def test_build_virtual_force_input_contract() -> None:
    rec = _recording()
    cycles = _cycles(rec)
    data = build_virtual_force_input(rec, cycles, body_mass_kg=72.0)
    assert len(data.timestamps) == len(rec.snapshots)
    assert data.fps == 30.0
    assert data.body_mass_kg == 72.0
    assert len(data.left_contact_mask) == len(rec.snapshots)
    assert "left_hip" in data.joint_positions
    assert np.allclose(data.root_positions[0], (0.0, 0.9, 0.0))


def test_physics_and_learned_placeholders_not_available() -> None:
    rec = _recording()
    cycles = _cycles(rec)
    data = build_virtual_force_input(rec, cycles)
    for Est in (PhysicsSimulationForceEstimator, LearnedForceEstimator):
        result = Est().estimate(data)
        assert result.available is False
        assert result.confidence == 0.0
