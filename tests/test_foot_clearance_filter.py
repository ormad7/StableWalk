"""Tests for robust foot clearance filtering."""

from __future__ import annotations

import unittest

from stablewalk.analysis.foot_clearance_filter import (
    MAX_PLAUSIBLE_CLEARANCE_M,
    build_filtered_foot_clearance_series,
)
from stablewalk.analysis.ground_reference import (
    GroundReferencePlane,
    _ground_plane_cache,
    estimate_ground_plane,
)
from stablewalk.models.gait_motion import (
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)


def _stance_snap(frame: int, heel_y: float, *, side: str = "left") -> SkeletonSnapshot:
    joints = {
        f"{side}_hip": JointSample(f"{side}_hip", Vec3(-0.1, 0.5, 0.0)),
        f"{side}_ankle": JointSample(f"{side}_ankle", Vec3(-0.1, heel_y + 0.03, 0.0)),
        f"{side}_heel": JointSample(f"{side}_heel", Vec3(-0.1, heel_y, 0.0)),
        f"{side}_toe": JointSample(f"{side}_toe", Vec3(-0.1, heel_y + 0.01, 0.0)),
        "head": JointSample("head", Vec3(0.0, 0.95, 0.0)),
    }
    return SkeletonSnapshot(
        frame_index=frame,
        time_s=float(frame) * 0.033,
        joints=joints,
        dofs={},
        metadata={"landmark_visibility": {jid: 0.9 for jid in joints}},
    )


class FootClearanceFilterTests(unittest.TestCase):
    def test_outlier_swing_rejected(self) -> None:
        _ground_plane_cache.clear()
        snapshots = [
            _stance_snap(0, 0.08),
            _stance_snap(1, 0.08),
            _stance_snap(2, 0.22),
            SkeletonSnapshot(
                frame_index=3,
                time_s=0.1,
                joints={
                    "left_hip": JointSample("left_hip", Vec3(-0.1, 0.5, 0.0)),
                    "left_ankle": JointSample("left_ankle", Vec3(-0.1, 0.5, 0.0)),
                    "left_heel": JointSample("left_heel", Vec3(-0.1, 0.5, 0.0)),
                    "left_toe": JointSample("left_toe", Vec3(-0.1, 0.5, 0.0)),
                    "head": JointSample("head", Vec3(0.0, 0.95, 0.0)),
                },
                dofs={},
                metadata={
                    "landmark_visibility": {
                        "left_heel": 0.9,
                        "left_toe": 0.9,
                        "left_ankle": 0.9,
                    }
                },
            ),
        ]
        recording = GaitMotionRecording(source="test.mp4", fps=30.0, snapshots=snapshots)
        plane = estimate_ground_plane(recording, 3.0)
        assert plane is not None
        series = build_filtered_foot_clearance_series(recording, plane, "left")
        outlier = series.samples[3]
        self.assertFalse(outlier.is_valid)
        self.assertIn(
            outlier.reject_reason,
            ("clearance_exceeds_plausible_max", "heel_toe_above_ankle", "temporal_outlier"),
        )

    def test_heel_toe_min_not_ankle(self) -> None:
        plane = GroundReferencePlane(floor_y=0.0, scale_mode="body_normalized")
        snap = SkeletonSnapshot(
            frame_index=0,
            time_s=0.0,
            joints={
                "left_ankle": JointSample("left_ankle", Vec3(0.0, 0.15, 0.0)),
                "left_heel": JointSample("left_heel", Vec3(0.0, 0.05, 0.0)),
                "left_toe": JointSample("left_toe", Vec3(0.0, 0.08, 0.0)),
            },
            dofs={},
            metadata={"landmark_visibility": {"left_heel": 0.9, "left_toe": 0.9}},
        )
        series = build_filtered_foot_clearance_series(
            GaitMotionRecording(source="t", fps=30.0, snapshots=[snap]),
            plane,
            "left",
        )
        sample = series.samples[0]
        self.assertAlmostEqual(sample.raw_clearance_m or 0.0, 0.05, places=3)

    def test_min_swing_from_valid_samples_only(self) -> None:
        plane = GroundReferencePlane(floor_y=0.0, scale_mode="body_normalized")
        snapshots = [
            _stance_snap(0, 0.00),
            _stance_snap(1, 0.06),
            _stance_snap(2, 0.11),
            _stance_snap(3, 0.04),
        ]
        series = build_filtered_foot_clearance_series(
            GaitMotionRecording(source="t", fps=30.0, snapshots=snapshots),
            plane,
            "left",
        )
        stats = series.swing_stats
        assert stats is not None
        self.assertIsNotNone(stats.min_swing_m)
        self.assertIsNotNone(stats.max_swing_m)
        assert stats.min_swing_m is not None and stats.max_swing_m is not None
        self.assertLessEqual(stats.min_swing_m, stats.max_swing_m)
        self.assertAlmostEqual(stats.min_swing_m, 0.04, places=3)
        self.assertAlmostEqual(stats.max_swing_m, 0.11, places=3)

    def test_plausible_max_constant(self) -> None:
        self.assertEqual(MAX_PLAUSIBLE_CLEARANCE_M, 0.25)


if __name__ == "__main__":
    unittest.main()
