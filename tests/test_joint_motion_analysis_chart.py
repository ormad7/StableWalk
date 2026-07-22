"""Tests for Joint Motion Analysis series and CSV export."""

from __future__ import annotations

import csv
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
    build_joint_motion_bundle,
    build_joint_motion_series,
    write_joint_motion_csv,
)


def _snap(
    frame: int,
    *,
    t: float,
    angle: float,
    omega: float,
    joint_id: str = "right_knee",
    dof_id: str = "right_knee_flexion",
) -> SkeletonSnapshot:
    return SkeletonSnapshot(
        frame_index=frame,
        time_s=t,
        joints={
            joint_id: JointSample(
                joint_id=joint_id,
                position=Vec3(0.1 + 0.01 * frame, 0.8, 0.2 + 0.005 * frame),
                angle_deg=angle,
            )
        },
        dofs={
            dof_id: DofSample(
                dof_id=dof_id,
                angle_deg=angle,
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

    def test_series_includes_acceleration(self) -> None:
        series = build_joint_motion_series(self.recording, "right_knee")
        assert series is not None
        self.assertEqual(len(series.times_s), 3)
        self.assertTrue(np.isfinite(series.angle_deg).all())
        self.assertTrue(np.isfinite(series.omega_deg_s).all())
        # Mid-point α from (30-20)/(0.2-0.0) = 50
        self.assertAlmostEqual(float(series.alpha_deg_s2[1]), 50.0, places=3)

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


if __name__ == "__main__":
    unittest.main()
