"""Tests for automatic gait view estimation and reliability profiles."""

from __future__ import annotations

from stablewalk.analysis.gait_view_analysis import (
    GaitViewType,
    assess_cross_video_comparability,
    build_view_reliability_profile,
    estimate_gait_view,
)
from stablewalk.models.pose_data import Keypoint, PoseFrame


def _frame(
    *,
    lh_x: float,
    rh_x: float,
    ls_x: float,
    rs_x: float,
    lh_z: float = 0.0,
    rh_z: float = 0.0,
    frame_index: int = 0,
) -> PoseFrame:
    keypoints = [
        Keypoint("left_hip", lh_x, 0.6, lh_z, 0.9),
        Keypoint("right_hip", rh_x, 0.6, rh_z, 0.9),
        Keypoint("left_shoulder", ls_x, 0.35, lh_z, 0.9),
        Keypoint("right_shoulder", rs_x, 0.35, rh_z, 0.9),
        Keypoint("left_knee", lh_x, 0.75, lh_z, 0.9),
        Keypoint("right_knee", rh_x, 0.75, rh_z, 0.9),
        Keypoint("nose", 0.5, 0.2, 0.0, 0.9),
        Keypoint("left_ankle", lh_x, 0.9, lh_z, 0.9),
        Keypoint("right_ankle", rh_x, 0.9, rh_z, 0.9),
    ]
    return PoseFrame(
        frame_index=frame_index,
        image_path=f"frame_{frame_index:04d}.jpg",
        timestamp_s=frame_index / 30.0,
        detected=True,
        keypoints=keypoints,
    )


def test_frontal_view_wide_shoulders():
    frames = [
        _frame(lh_x=0.42, rh_x=0.58, ls_x=0.38, rs_x=0.62, frame_index=i)
        for i in range(12)
    ]
    est = estimate_gait_view(frames)
    assert est.view_type == GaitViewType.FRONTAL
    assert est.view_confidence > 0.3


def test_sagittal_view_narrow_width():
    frames = [
        _frame(lh_x=0.49, rh_x=0.51, ls_x=0.495, rs_x=0.505, lh_z=-0.05, rh_z=0.05, frame_index=i)
        for i in range(12)
    ]
    est = estimate_gait_view(frames)
    assert est.view_type in (GaitViewType.SAGITTAL_LEFT, GaitViewType.SAGITTAL_RIGHT, GaitViewType.OBLIQUE)


def test_sagittal_low_pelvis_lateral_reliability():
    profile = build_view_reliability_profile(GaitViewType.SAGITTAL_RIGHT)
    assert profile.domain_coefficient("pelvis_stability") < profile.domain_coefficient("foot_clearance")
    assert profile.metric_tier("pelvis_lateral_sway") == "LOW"
    assert profile.metric_tier("knee_flexion") == "HIGH"


def test_cross_video_comparability_mixed_views():
    result = assess_cross_video_comparability(
        [
            {"view_type": "FRONTAL", "view_confidence": 0.8, "valid_pose_frame_pct": 95, "gait_cycles": 4, "duration_s": 10, "fps": 30},
            {"view_type": "SAGITTAL_RIGHT", "view_confidence": 0.85, "valid_pose_frame_pct": 98, "gait_cycles": 5, "duration_s": 12, "fps": 30},
        ]
    )
    assert result.level in ("MODERATE", "LOW")
    assert result.warning is not None
