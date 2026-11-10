"""Pelvis fallback when selected DOF has no drawable 3D path."""

from __future__ import annotations

import unittest

from matplotlib.figure import Figure

from stablewalk.models.gait_motion import (
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.models.joint_registry import ROOT_JOINT_ID
from stablewalk.ui.viewers.dof_trajectory_3d import (
    draw_single_dof_trajectory_3d,
    resolve_trajectory_draw_target,
    trajectory_path_is_drawable,
)


def _snap(frame: int, *, knee: Vec3 | None, pelvis: Vec3) -> SkeletonSnapshot:
    joints = {
        ROOT_JOINT_ID: JointSample(ROOT_JOINT_ID, pelvis),
    }
    if knee is not None:
        joints["right_knee"] = JointSample("right_knee", knee)
    return SkeletonSnapshot(frame_index=frame, time_s=frame / 25.0, joints=joints)


def _recording_with_pelvis_only() -> GaitMotionRecording:
    snaps = [
        _snap(i, knee=None, pelvis=Vec3(0.02 * i, 0.9, 0.0))
        for i in range(8)
    ]
    return GaitMotionRecording(source="test", fps=25.0, snapshots=snaps)


def _recording_with_knee() -> GaitMotionRecording:
    snaps = [
        _snap(
            i,
            knee=Vec3(0.05 * (i % 3), 0.45 + 0.02 * (i % 2), 0.01 * i),
            pelvis=Vec3(0.02 * i, 0.9, 0.0),
        )
        for i in range(8)
    ]
    return GaitMotionRecording(source="test", fps=25.0, snapshots=snaps)


class TrajectoryPelvisFallbackTests(unittest.TestCase):
    def test_selected_dof_drawable_keeps_joint(self) -> None:
        rec = _recording_with_knee()
        joint_id, mode, fallback = resolve_trajectory_draw_target(
            rec,
            "right_knee",
            end_frame_float=7.0,
            coord_mode="Root-relative",
        )
        self.assertEqual(joint_id, "right_knee")
        self.assertEqual(mode, "Root-relative")
        self.assertFalse(fallback)

    def test_missing_dof_falls_back_to_pelvis(self) -> None:
        rec = _recording_with_pelvis_only()
        self.assertFalse(
            trajectory_path_is_drawable(
                rec, "right_knee", 7.0, coord_mode="Root-relative"
            )
        )
        joint_id, mode, fallback = resolve_trajectory_draw_target(
            rec,
            "right_knee",
            end_frame_float=7.0,
            coord_mode="Root-relative",
        )
        self.assertEqual(joint_id, ROOT_JOINT_ID)
        self.assertTrue(fallback)
        # Global keeps the walking translation visible.
        self.assertEqual(mode, "Global")

    def test_draw_never_returns_empty_scene_for_pelvis_only(self) -> None:
        rec = _recording_with_pelvis_only()
        fig = Figure(figsize=(3.0, 3.0))
        ax = fig.add_subplot(111, projection="3d")
        ax._stablewalk_overview_dock = True
        ok, status = draw_single_dof_trajectory_3d(
            ax,
            rec,
            "right_knee",
            end_frame_float=7.0,
            clear=True,
            coord_mode="Root-relative",
        )
        self.assertTrue(ok)
        self.assertTrue(bool(getattr(ax, "_stablewalk_traj_fallback_pelvis", False)))
        self.assertGreaterEqual(len(ax.lines) + len(ax.collections), 1)
        # Axes limits must enclose finite data (not a collapsed / empty cube).
        xl, yl, zl = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        self.assertGreater(xl[1] - xl[0], 1e-6)
        self.assertGreater(yl[1] - yl[0], 1e-6)
        self.assertGreater(zl[1] - zl[0], 1e-6)
        self.assertIsInstance(status, str)

    def test_no_motion_still_sets_up_axes(self) -> None:
        fig = Figure(figsize=(3.0, 3.0))
        ax = fig.add_subplot(111, projection="3d")
        ok, _status = draw_single_dof_trajectory_3d(
            ax,
            None,
            "right_knee",
            end_frame_float=0.0,
            clear=True,
        )
        self.assertFalse(ok)
        # Floor / cube scaffolding must exist even without a path.
        self.assertTrue(hasattr(ax, "get_xlim"))
        xl = ax.get_xlim()
        self.assertNotEqual(xl[0], xl[1])


if __name__ == "__main__":
    unittest.main()
