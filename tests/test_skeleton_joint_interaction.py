"""Tests for skeleton joint hover tooltip and pick hitbox scaling."""

from __future__ import annotations

from stablewalk.models.gait_motion import DofSample, JointSample, SkeletonSnapshot, Vec3
from stablewalk.ui.viewers import gait_skeleton_renderer as gsr
from stablewalk.ui.viewers.skeleton_joint_tooltip import format_skeleton_joint_tooltip


def test_pick_hitbox_scale_is_enlarged_for_comfort() -> None:
    assert gsr._PICK_HITBOX_SCALE >= 1.5


def test_dimmed_alpha_when_selection_active() -> None:
    assert 0.2 <= gsr._DIMMED_JOINT_ALPHA <= 0.45
    assert 0.2 <= gsr._DIMMED_BONE_ALPHA <= 0.45


def test_format_skeleton_joint_tooltip_includes_kinematics() -> None:
    snap = SkeletonSnapshot(
        frame_index=12,
        time_s=0.4,
        joints={
            "right_knee": JointSample(
                joint_id="right_knee",
                position=Vec3(0.1, 0.55, -0.02),
                angle_deg=42.5,
                velocity=0.12,
            )
        },
        dofs={
            "right_knee_flexion": DofSample(
                dof_id="right_knee_flexion",
                angle_deg=42.5,
                velocity_deg_s=18.2,
                joint_id="right_knee",
            )
        },
    )
    text = format_skeleton_joint_tooltip("right_knee", snap)
    assert "Right Knee" in text
    assert "x=0.100" in text
    assert "Angle:" in text
    assert "Velocity:" in text
