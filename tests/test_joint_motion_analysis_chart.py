"""Tests for Joint Motion Analysis series and CSV export."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stablewalk.models.gait_motion import (
    DofSample,
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.ui.viewers.joint_motion_analysis_chart import (
    JOINT_MOTION_METRICS,
    MOTION_WORKSPACE_METRICS,
    build_joint_motion_bundle,
    build_joint_motion_series,
    compute_joint_motion_stats,
    compute_metric_ribbon_stats,
    write_joint_motion_csv,
)


def _hka_joints(
    joint_id: str,
    flexion_deg: float,
    frame: int,
) -> dict[str, JointSample]:
    """Hip–knee–ankle geometry whose 3D flexion equals ``flexion_deg``."""
    prefix = "left" if joint_id.startswith("left") else "right"
    dx = 0.1 + 0.01 * frame
    dz = 0.2 + 0.005 * frame
    knee = Vec3(dx, 0.8, dz)
    hip = Vec3(dx, 1.2, dz)
    rad = math.radians(float(flexion_deg))
    shank = 0.4
    ankle = Vec3(dx + shank * math.sin(rad), 0.8 - shank * math.cos(rad), dz)
    interior = 180.0 - float(flexion_deg)
    return {
        f"{prefix}_hip": JointSample(joint_id=f"{prefix}_hip", position=hip),
        f"{prefix}_knee": JointSample(
            joint_id=f"{prefix}_knee",
            position=knee,
            angle_deg=interior,
        ),
        f"{prefix}_ankle": JointSample(joint_id=f"{prefix}_ankle", position=ankle),
    }


def _snap(
    frame: int,
    *,
    t: float,
    angle: float,
    omega: float,
    joint_id: str = "right_knee",
    dof_id: str = "right_knee_flexion",
) -> SkeletonSnapshot:
    interior = 180.0 - float(angle)
    return SkeletonSnapshot(
        frame_index=frame,
        time_s=t,
        joints=_hka_joints(joint_id, angle, frame),
        dofs={
            dof_id: DofSample(
                dof_id=dof_id,
                angle_deg=interior,
                velocity_deg_s=omega,
                joint_id=joint_id,
            )
        },
    )


class JointMotionAnalysisChartTests(unittest.TestCase):
    def setUp(self) -> None:
        snaps = []
        for frame, t, a, w in (
            (0, 0.0, 10.0, 20.0),
            (1, 0.1, 12.0, 40.0),
            (2, 0.2, 15.0, 30.0),
        ):
            right = _snap(frame, t=t, angle=a, omega=w)
            left = _snap(
                frame,
                t=t,
                angle=a + 5.0,
                omega=w - 5.0,
                joint_id="left_knee",
                dof_id="left_knee_flexion",
            )
            snaps.append(
                SkeletonSnapshot(
                    frame_index=frame,
                    time_s=t,
                    joints={**right.joints, **left.joints},
                    dofs={**right.dofs, **left.dofs},
                )
            )
        self.recording = GaitMotionRecording(source="test", fps=10.0, snapshots=snaps)

    def test_six_metric_panels_defined(self) -> None:
        self.assertEqual(len(JOINT_MOTION_METRICS), 6)
        ids = [m[0] for m in JOINT_MOTION_METRICS]
        self.assertEqual(ids, ["angle", "omega", "alpha", "x", "y", "z"])

    def test_motion_workspace_seven_panels(self) -> None:
        self.assertEqual(len(MOTION_WORKSPACE_METRICS), 7)
        ids = [m[0] for m in MOTION_WORKSPACE_METRICS]
        self.assertEqual(
            ids,
            ["angle", "omega", "alpha", "position", "speed", "accel", "jerk"],
        )

    def test_series_includes_acceleration_and_jerk(self) -> None:
        series = build_joint_motion_series(self.recording, "right_knee")
        assert series is not None
        self.assertEqual(len(series.times_s), 3)
        self.assertTrue(np.isfinite(series.angle_deg).all())
        self.assertTrue(np.isfinite(series.omega_deg_s).all())
        # Mid-point α from (30-20)/(0.2-0.0) = 50
        self.assertAlmostEqual(float(series.alpha_deg_s2[1]), 50.0, places=3)
        self.assertTrue(np.isfinite(series.speed_m_s).any())
        self.assertTrue(np.isfinite(series.accel_m_s2).any())
        self.assertTrue(np.isfinite(series.jerk_m_s3).any())
        self.assertTrue(np.isfinite(series.angular_jerk_deg_s3).any())
        self.assertIn("jerk_m_s3", series.__dataclass_fields__)
        self.assertIn("angular_jerk_deg_s3", series.__dataclass_fields__)

    def test_stats_strip_values(self) -> None:
        bundle = build_joint_motion_bundle(self.recording, {"right_knee"})
        stats = compute_joint_motion_stats(bundle)
        self.assertEqual(len(stats), 1)
        self.assertIsNotNone(stats[0].rom_deg)
        self.assertGreater(float(stats[0].rom_deg or 0), 0.0)

    def test_metric_ribbon_stats(self) -> None:
        bundle = build_joint_motion_bundle(self.recording, {"right_knee"})
        ribbon = compute_metric_ribbon_stats(
            bundle, "angle", time_s=0.1, unit="°"
        )
        self.assertAlmostEqual(float(ribbon.minimum or 0), 10.0, places=3)
        self.assertAlmostEqual(float(ribbon.maximum or 0), 15.0, places=3)
        self.assertAlmostEqual(float(ribbon.rom or 0), 5.0, places=3)
        self.assertIsNotNone(ribbon.mean)
        self.assertIsNotNone(ribbon.std)
        self.assertAlmostEqual(float(ribbon.current or 0), 12.0, places=3)
        display = ribbon.as_display()
        for key in ("minimum", "maximum", "mean", "rom", "std", "current"):
            self.assertIn(key, display)
            self.assertNotEqual(display[key], "—")

    def test_bundle_multi_joint_colors_differ(self) -> None:
        bundle = build_joint_motion_bundle(
            self.recording, {"right_knee", "left_knee"}
        )
        self.assertEqual(len(bundle.series), 2)
        colors = {s.color for s in bundle.series}
        self.assertEqual(len(colors), 2)

    def test_csv_export_rows(self) -> None:
        bundle = build_joint_motion_bundle(self.recording, {"right_knee"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            write_joint_motion_csv(bundle, path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["joint_id"], "right_knee")
            self.assertIn("angular_acceleration_deg_s2", rows[0])
            self.assertIn("jerk_m_s3", rows[0])
            self.assertIn("angular_jerk_deg_s3", rows[0])
            self.assertIn("raw_pose_angle_deg", rows[0])
            self.assertAlmostEqual(float(rows[0]["angle_deg"]), 10.0, places=1)
            self.assertAlmostEqual(float(rows[0]["raw_pose_angle_deg"]), 170.0, places=1)

    def test_primary_knee_series_is_flexion_not_raw_interior(self) -> None:
        series = build_joint_motion_series(self.recording, "right_knee")
        assert series is not None
        self.assertAlmostEqual(float(series.angle_deg[0]), 10.0, places=1)
        self.assertIsNotNone(series.raw_pose_angle_deg)
        self.assertAlmostEqual(float(series.raw_pose_angle_deg[0]), 170.0, places=1)

    def test_missing_hka_does_not_invent_knee_flexion(self) -> None:
        snap = SkeletonSnapshot(
            frame_index=0,
            time_s=0.0,
            joints={
                "right_knee": JointSample(
                    joint_id="right_knee",
                    position=Vec3(0.1, 0.8, 0.2),
                    angle_deg=170.0,
                )
            },
            dofs={
                "right_knee_flexion": DofSample(
                    dof_id="right_knee_flexion",
                    angle_deg=170.0,
                    joint_id="right_knee",
                )
            },
        )
        rec = GaitMotionRecording(source="incomplete", fps=10.0, snapshots=[snap])
        series = build_joint_motion_series(rec, "right_knee")
        assert series is not None
        self.assertFalse(np.isfinite(series.angle_deg).any())
        self.assertTrue(np.isfinite(series.raw_pose_angle_deg).any())
        self.assertAlmostEqual(float(series.raw_pose_angle_deg[0]), 170.0, places=3)

    def test_motion_angle_convention_marks_knee_flexion(self) -> None:
        from stablewalk.ui.viewers.joint_motion_analysis_chart import (
            KNEE_ANGLE_CONVENTION_SHORT,
            motion_angle_convention,
        )

        short, tip = motion_angle_convention(("right_knee",))
        self.assertEqual(short, KNEE_ANGLE_CONVENTION_SHORT)
        blob = (short + " " + tip).lower()
        self.assertIn("knee flexion", blob)
        self.assertNotIn("raw pose joint angle", blob)
        self.assertIn("180", tip)
        self.assertIn("deg", blob)
        other_short, other_tip = motion_angle_convention(("right_hip",))
        self.assertIn("interior", other_short.lower())
        self.assertNotEqual(other_short, short)
        self.assertIn("interior", other_tip.lower())


if __name__ == "__main__":
    unittest.main()
