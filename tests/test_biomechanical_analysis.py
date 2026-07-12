"""Tests for advanced biomechanical analysis module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stablewalk.analysis.biomechanical.advanced_gait_metrics import analyze_advanced_gait_metrics
from stablewalk.analysis.biomechanical.base_of_support import analyze_base_of_support
from stablewalk.analysis.biomechanical.com_estimation import (
    analyze_center_of_mass,
    estimate_frame_com,
)
from stablewalk.analysis.biomechanical.gait_quality_score import compute_gait_quality_score
from stablewalk.analysis.biomechanical.orchestrator import run_biomechanical_analysis
from stablewalk.analysis.biomechanical.stability_margin import analyze_stability_margin
from stablewalk.analysis.biomechanical.symmetry_metrics import analyze_symmetry
from stablewalk.analysis.biomechanical.video_quality import assess_video_quality
from stablewalk.analysis.foot_contact_analysis import FootContactFrame, FootContactAnalysisResult
from stablewalk.analysis.gait_cycle_analysis import (
    FrameContactState,
    FootLandmarkSample,
    GaitCycleAnalysisResult,
    GaitTemporalMetrics,
)
from stablewalk.io.biomechanical_export import export_biomechanical_artifacts
from stablewalk.models.gait_motion import GaitMotionRecording, JointSample, SkeletonSnapshot, Vec3
from stablewalk.models.pose_data import Keypoint, PoseFrame, PoseSequence


def _snap(i: int, t: float) -> SkeletonSnapshot:
    y = 0.9
    joints = {
        "left_hip": JointSample("left_hip", Vec3(-0.1, y, 0.0)),
        "right_hip": JointSample("right_hip", Vec3(0.1, y, 0.0)),
        "spine": JointSample("spine", Vec3(0.0, y + 0.3, 0.0)),
        "neck": JointSample("neck", Vec3(0.0, y + 0.5, 0.0)),
        "head": JointSample("head", Vec3(0.0, y + 0.65, 0.0)),
        "left_shoulder": JointSample("left_shoulder", Vec3(-0.15, y + 0.55, 0.0)),
        "right_shoulder": JointSample("right_shoulder", Vec3(0.15, y + 0.55, 0.0)),
        "left_elbow": JointSample("left_elbow", Vec3(-0.18, y + 0.35, 0.0)),
        "right_elbow": JointSample("right_elbow", Vec3(0.18, y + 0.35, 0.0)),
        "left_wrist": JointSample("left_wrist", Vec3(-0.2, y + 0.2, 0.0)),
        "right_wrist": JointSample("right_wrist", Vec3(0.2, y + 0.2, 0.0)),
        "left_knee": JointSample("left_knee", Vec3(-0.1, y - 0.35, 0.02)),
        "right_knee": JointSample("right_knee", Vec3(0.1, y - 0.35, 0.02)),
        "left_ankle": JointSample("left_ankle", Vec3(-0.1, y - 0.7, 0.0)),
        "right_ankle": JointSample("right_ankle", Vec3(0.1, y - 0.68, 0.0)),
        "left_heel": JointSample("left_heel", Vec3(-0.1, y - 0.72, -0.03)),
        "right_heel": JointSample("right_heel", Vec3(0.1, y - 0.70, -0.03)),
        "left_toe": JointSample("left_toe", Vec3(-0.08, y - 0.69, 0.05)),
        "right_toe": JointSample("right_toe", Vec3(0.08, y - 0.67, 0.05)),
    }
    return SkeletonSnapshot(frame_index=i, time_s=t, joints=joints, dofs={})


def _recording(n: int = 20) -> GaitMotionRecording:
    return GaitMotionRecording(
        source="test",
        fps=30.0,
        snapshots=[_snap(i, i / 30.0) for i in range(n)],
    )


def _foot_sample(c: float = 0.02) -> FootLandmarkSample:
    return FootLandmarkSample(c, c, c, c, 0.05, 0.05, 0.05, 0.9)


def _contact(n: int = 20) -> FootContactAnalysisResult:
    frames = []
    for i in range(n):
        left = 1 if i % 10 < 5 else 0
        right = 1 - left if i % 10 >= 5 else (1 if left == 0 else 0)
        if i % 10 == 4:
            left, right = 1, 1
        frames.append(
            FootContactFrame(
                frame_index=i,
                time_s=i / 30.0,
                left_contact_probability=0.9 if left else 0.1,
                right_contact_probability=0.9 if right else 0.1,
                left_contact_binary=left,
                right_contact_binary=right,
                left_heel_strike=0,
                right_heel_strike=0,
                left_toe_off=0,
                right_toe_off=0,
                left_foot_substate="mid_stance" if left else "swing",
                right_foot_substate="mid_stance" if right else "swing",
                macro_phase="double_support" if left and right else ("stance" if left or right else "swing"),
                left_confidence=0.85,
                right_confidence=0.85,
            )
        )
    return FootContactAnalysisResult(
        per_frame=frames,
        metrics=GaitTemporalMetrics(contact_confidence=0.8),
        fps=30.0,
    )


def _cycles(n: int = 20) -> GaitCycleAnalysisResult:
    per_frame = [
        FrameContactState(
            i, i / 30.0, i % 2, 1 - (i % 2), "LEFT_STANCE", _foot_sample(), _foot_sample()
        )
        for i in range(n)
    ]
    return GaitCycleAnalysisResult(
        per_frame=per_frame,
        metrics=GaitTemporalMetrics(
            contact_confidence=0.8,
            cadence_steps_per_min=110.0,
            gait_cycle_consistency=0.85,
        ),
        fps=30.0,
    )


def _sequence(n: int = 20) -> PoseSequence:
    frames = []
    for i in range(n):
        kps = [
            Keypoint("left_hip", 0.4, 0.5, 0.9, 0.9),
            Keypoint("right_hip", 0.6, 0.5, 0.9, 0.9),
            Keypoint("left_ankle", 0.42, 0.85, 0.9, 0.85),
            Keypoint("right_ankle", 0.58, 0.84, 0.9, 0.85),
            Keypoint("nose", 0.5, 0.15, 0.9, 0.9),
        ]
        frames.append(
            PoseFrame(
                frame_index=i,
                image_path=f"frame_{i:04d}.jpg",
                timestamp_s=i / 30.0,
                detected=True,
                keypoints=kps,
            )
        )
    return PoseSequence(source_video="test.mp4", fps=30.0, frames=frames)


def test_estimate_frame_com() -> None:
    snap = _snap(0, 0.0)
    pos, conf = estimate_frame_com(snap)
    assert conf > 0.5
    assert pos[1] > 0.0


def test_com_velocity_acceleration() -> None:
    rec = _recording(15)
    contact = _contact(15)
    com = analyze_center_of_mass(rec, contact)
    assert len(com.per_frame) == 15
    assert com.positions.shape == (15, 3)
    assert com.per_frame[5].velocity[0] is not None


def test_base_of_support_double_support() -> None:
    rec = _recording(10)
    contact = _contact(10)
    bos = analyze_base_of_support(rec, contact)
    types = {f.support_type for f in bos.per_frame}
    assert "double_support" in types or "left_stance" in types


def test_stability_margin_states() -> None:
    rec = _recording(12)
    contact = _contact(12)
    com = analyze_center_of_mass(rec, contact)
    bos = analyze_base_of_support(rec, contact)
    sm = analyze_stability_margin(com, bos)
    assert sm.per_frame
    states = {f.stability_state for f in sm.per_frame}
    assert states <= {"Stable", "Reduced Stability", "Unstable"}


def test_symmetry_analysis() -> None:
    sym = analyze_symmetry(_cycles(), None, _contact())
    assert sym.overall_symmetry_pct is not None


def test_gait_quality_score_range() -> None:
    contact = _contact()
    cycles = _cycles()
    gq = compute_gait_quality_score(cycles=cycles, contact=contact)
    assert 0.0 <= gq.score <= 100.0
    assert gq.explanation


def test_video_quality_assessment() -> None:
    vq = assess_video_quality(_sequence())
    assert 0.0 <= vq.overall_quality_score <= 100.0


def test_run_biomechanical_analysis_orchestrator() -> None:
    rec = _recording(18)
    seq = _sequence(18)
    result = run_biomechanical_analysis(rec, seq, cycles=_cycles(18), contact=_contact(18))
    assert result.center_of_mass is not None
    assert result.gait_quality is not None
    assert result.video_quality is not None


def test_export_biomechanical_artifacts(tmp_path: Path) -> None:
    rec = _recording(12)
    result = run_biomechanical_analysis(rec, _sequence(12), contact=_contact(12))
    exports = export_biomechanical_artifacts(result, tmp_path / "run")
    assert exports.center_of_mass_path.is_file()
    assert exports.base_of_support_path.is_file()
    assert exports.video_quality_path.is_file()
    assert exports.biomechanical_report_path.is_file()
    com = np.load(exports.center_of_mass_path)
    assert com["kind"] == "estimated"
