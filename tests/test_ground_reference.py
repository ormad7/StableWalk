"""Unit tests for ground reference plane and foot ground-distance metrics."""

from __future__ import annotations

import unittest

from stablewalk.analysis.ground_reference import (
    NEAR_GROUND_THRESHOLD_M,
    ON_GROUND_THRESHOLD_M,
    GroundReferencePlane,
    estimate_ground_plane,
    foot_clearance_m,
    foot_contact_state,
    ground_distance_m,
    vertical_coordinate,
)
from stablewalk.models.gait_motion import (
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)


def _foot_snapshot(frame_index: int, toe_y: float) -> SkeletonSnapshot:
    joints = {
        "left_toe": JointSample(
            joint_id="left_toe",
            position=Vec3(0.0, toe_y, 0.1),
        ),
    }
    return SkeletonSnapshot(
        frame_index=frame_index,
        time_s=float(frame_index) * 0.033,
        joints=joints,
        dofs={},
    )


class GroundReferenceTests(unittest.TestCase):
    def test_ground_distance_uses_vertical_axis_only(self) -> None:
        plane = GroundReferencePlane(floor_y=0.0)
        point = Vec3(0.5, 0.12, -0.3)
        self.assertAlmostEqual(vertical_coordinate(point), 0.12)
        self.assertAlmostEqual(ground_distance_m(point, plane), 0.12)

    def test_contact_state_thresholds(self) -> None:
        self.assertEqual(foot_contact_state(None), "—")
        self.assertEqual(foot_contact_state(0.0), "On Ground")
        self.assertEqual(
            foot_contact_state(ON_GROUND_THRESHOLD_M - 0.001),
            "On Ground",
        )
        self.assertEqual(
            foot_contact_state(ON_GROUND_THRESHOLD_M),
            "Near Ground",
        )
        self.assertEqual(
            foot_contact_state(NEAR_GROUND_THRESHOLD_M - 0.001),
            "Near Ground",
        )
        self.assertEqual(
            foot_contact_state(NEAR_GROUND_THRESHOLD_M),
            "In Air",
        )
        self.assertEqual(foot_contact_state(0.15), "In Air")

    def test_foot_clearance_clamps_to_zero(self) -> None:
        plane = GroundReferencePlane(floor_y=0.05)
        point = Vec3(0.0, 0.04, 0.0)
        self.assertAlmostEqual(ground_distance_m(point, plane), -0.01)
        self.assertAlmostEqual(foot_clearance_m(point, plane), 0.0)
        above = Vec3(0.0, 0.12, 0.0)
        self.assertAlmostEqual(foot_clearance_m(above, plane), 0.07)

    def test_estimate_ground_plane_from_foot_heights(self) -> None:
        recording = GaitMotionRecording(
            source="test.mp4",
            fps=30.0,
            snapshots=[
                _foot_snapshot(0, toe_y=0.02),
                _foot_snapshot(1, toe_y=0.18),
            ],
        )
        plane = estimate_ground_plane(recording, end_frame_float=1.0)
        self.assertIsNotNone(plane)
        assert plane is not None
        self.assertAlmostEqual(plane.floor_y, 0.02, delta=0.015)
        stance_dist = ground_distance_m(Vec3(0.0, 0.02, 0.0), plane)
        swing_dist = ground_distance_m(Vec3(0.0, 0.18, 0.0), plane)
        self.assertIsNotNone(stance_dist)
        self.assertIsNotNone(swing_dist)
        assert stance_dist is not None and swing_dist is not None
        self.assertLess(stance_dist, swing_dist)
        self.assertEqual(foot_contact_state(stance_dist), "On Ground")
        self.assertEqual(foot_contact_state(swing_dist), "In Air")

    def test_outlier_foot_height_does_not_deflate_floor(self) -> None:
        """A single bad landmark must not push floor_y far below real contact."""
        from stablewalk.models.joint_registry import ROOT_JOINT_ID

        def _snap(frame: int, toe_y: float, *, outlier: bool = False) -> SkeletonSnapshot:
            joints = {
                ROOT_JOINT_ID: JointSample(
                    joint_id=ROOT_JOINT_ID,
                    position=Vec3(0.0, 0.55, 0.0),
                ),
                "left_hip": JointSample(
                    joint_id="left_hip",
                    position=Vec3(-0.08, 0.55, 0.0),
                ),
                "right_hip": JointSample(
                    joint_id="right_hip",
                    position=Vec3(0.08, 0.55, 0.0),
                ),
                "right_toe": JointSample(
                    joint_id="right_toe",
                    position=Vec3(0.05, toe_y, 0.12),
                ),
                "right_heel": JointSample(
                    joint_id="right_heel",
                    position=Vec3(0.05, toe_y + 0.01, 0.08),
                ),
                "right_ankle": JointSample(
                    joint_id="right_ankle",
                    position=Vec3(0.05, toe_y + 0.03, 0.06),
                ),
            }
            if outlier:
                joints["left_toe"] = JointSample(
                    joint_id="left_toe",
                    position=Vec3(-0.05, -3.0, 0.1),
                )
            return SkeletonSnapshot(
                frame_index=frame,
                time_s=float(frame) * 0.033,
                joints=joints,
                dofs={},
                metadata={"gait_phase": {"right": "stance", "left": "swing"}},
            )

        recording = GaitMotionRecording(
            source="test.mp4",
            fps=30.0,
            snapshots=[
                _snap(0, toe_y=0.08),
                _snap(1, toe_y=0.10),
                _snap(2, toe_y=0.09, outlier=True),
            ],
        )
        plane = estimate_ground_plane(recording, end_frame_float=2.0)
        self.assertIsNotNone(plane)
        assert plane is not None
        heel = Vec3(0.05, 0.11, 0.08)
        dist = ground_distance_m(heel, plane)
        self.assertIsNotNone(dist)
        assert dist is not None
        self.assertLess(dist, 0.15, msg=f"unrealistic clearance {dist:.3f} m")
        self.assertGreater(dist, -0.05)

    def test_sanity_flag_for_extreme_clearance(self) -> None:
        from stablewalk.analysis.ground_reference import (
            CALIBRATION_CHECK_LABEL,
            clearance_sanity_flag,
            compute_foot_clearance_reading,
            compute_session_foot_clearance_stats,
            format_clearance_cm,
        )

        plane = GroundReferencePlane(floor_y=-2.5)
        heel = Vec3(0.0, 0.12, 0.0)
        reading = compute_foot_clearance_reading(heel, plane)
        self.assertTrue(reading.sanity_flag)
        self.assertTrue(clearance_sanity_flag(reading.foot_clearance_m))
        self.assertEqual(reading.contact_state, CALIBRATION_CHECK_LABEL)
        self.assertAlmostEqual(reading.foot_clearance_m or 0.0, 2.62, places=2)

        stats = compute_session_foot_clearance_stats([heel], plane)
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertTrue(stats.calibration_check_needed)
        self.assertEqual(
            format_clearance_cm(stats.current.foot_clearance_m, calibration_check=True),
            CALIBRATION_CHECK_LABEL,
        )

    def test_session_clearance_min_current_max_consistent(self) -> None:
        from stablewalk.analysis.ground_reference import (
            _ground_plane_cache,
            compute_session_foot_clearance_stats,
        )
        from stablewalk.models.joint_registry import ROOT_JOINT_ID

        _ground_plane_cache.clear()

        def _snap(frame: int, heel_y: float) -> SkeletonSnapshot:
            return SkeletonSnapshot(
                frame_index=frame,
                time_s=float(frame) * 0.033,
                joints={
                    ROOT_JOINT_ID: JointSample(
                        joint_id=ROOT_JOINT_ID,
                        position=Vec3(0.0, 0.55, 0.0),
                    ),
                    "right_heel": JointSample(
                        joint_id="right_heel",
                        position=Vec3(0.05, heel_y, 0.08),
                    ),
                    "right_ankle": JointSample(
                        joint_id="right_ankle",
                        position=Vec3(0.05, heel_y + 0.03, 0.06),
                    ),
                    "right_toe": JointSample(
                        joint_id="right_toe",
                        position=Vec3(0.05, heel_y + 0.01, 0.12),
                    ),
                },
                dofs={},
                metadata={"gait_phase": {"right": "stance", "left": "swing"}},
            )

        recording = GaitMotionRecording(
            source="walk.mp4",
            fps=30.0,
            snapshots=[
                _snap(0, heel_y=0.08),
                _snap(1, heel_y=0.10),
                _snap(2, heel_y=0.18),
                _snap(3, heel_y=0.09),
            ],
        )
        plane = estimate_ground_plane(recording, end_frame_float=3.0)
        self.assertIsNotNone(plane)

        positions = []
        for i in range(4):
            snap = recording.snapshot_at(i)
            assert snap is not None
            heel = snap.joints.get("right_heel")
            assert heel is not None
            positions.append(heel.position)
        stats = compute_session_foot_clearance_stats(positions, plane)
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertFalse(stats.calibration_check_needed)
        self.assertIsNotNone(stats.min_clearance_m)
        self.assertIsNotNone(stats.max_clearance_m)
        self.assertIsNotNone(stats.current.foot_clearance_m)
        assert stats.min_clearance_m is not None
        assert stats.max_clearance_m is not None
        assert stats.current.foot_clearance_m is not None
        self.assertLessEqual(stats.min_clearance_m, stats.current.foot_clearance_m)
        self.assertLessEqual(stats.current.foot_clearance_m, stats.max_clearance_m)
        self.assertLessEqual(stats.min_clearance_m, stats.avg_clearance_m or 0.0)
        self.assertLessEqual(stats.avg_clearance_m or 0.0, stats.max_clearance_m)
        self.assertLess(stats.max_clearance_m, 0.35)

    def test_foot_clearance_display_parts(self) -> None:
        from stablewalk.analysis.ground_reference import (
            CALIBRATION_CHECK_LABEL,
            format_foot_clearance_display,
        )

        normal = format_foot_clearance_display(0.124, scale_mode="body_normalized")
        self.assertEqual(normal.value_cm, "12.4 cm")
        self.assertEqual(normal.quality_label, "Estimated body-scale")
        self.assertIn("estimated", normal.full_line)

        relative = format_foot_clearance_display(0.05, scale_mode="unknown")
        self.assertEqual(relative.quality_label, "Relative scale")

        bad = format_foot_clearance_display(2.5, calibration_check=True)
        self.assertEqual(bad.value_cm, "—")
        self.assertEqual(bad.quality_label, CALIBRATION_CHECK_LABEL)
        self.assertTrue(bad.calibration_check)


if __name__ == "__main__":
    unittest.main()
