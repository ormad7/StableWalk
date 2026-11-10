"""Unit tests for live Joint Information snapshot formatting."""

from __future__ import annotations

import math
import unittest

from stablewalk.models.gait_motion import (
    DofSample,
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.ui.joint_information import (
    JOINT_INFO_FIELD_ORDER,
    JOINT_INFO_TITLES,
    build_joint_information,
    empty_joint_information,
)


def _snap(
    frame: int,
    *,
    time_s: float,
    angle: float,
    omega: float,
    pos: tuple[float, float, float] = (0.1, 0.8, 0.2),
) -> SkeletonSnapshot:
    x, y, z = pos
    rad = math.radians(float(angle))
    thigh, shank = 0.04, 0.04
    hip = Vec3(x, y + thigh, z)
    ankle = Vec3(x + shank * math.sin(rad), y - shank * math.cos(rad), z)
    return SkeletonSnapshot(
        frame_index=frame,
        time_s=time_s,
        joints={
            "right_hip": JointSample(joint_id="right_hip", position=hip),
            "right_knee": JointSample(
                joint_id="right_knee",
                position=Vec3(x, y, z),
                angle_deg=180.0 - float(angle),
                velocity=0.05,
            ),
            "right_ankle": JointSample(joint_id="right_ankle", position=ankle),
        },
        dofs={
            "right_knee_flexion": DofSample(
                dof_id="right_knee_flexion",
                angle_deg=180.0 - float(angle),
                velocity_deg_s=omega,
                joint_id="right_knee",
            )
        },
    )


class JointInformationTests(unittest.TestCase):
    def test_field_catalog_covers_required_metrics(self) -> None:
        required = {
            "Joint",
            "X (cm, root-rel.)",
            "Y (cm, root-rel.)",
            "Z (cm, root-rel.)",
            "Current angle (est., °)",
            "ω (°/s, derived)",
            "α (°/s², derived)",
            "Session ROM (°)",
            "Frame",
            "Time",
            "Tracking confidence",
            "Contact state",
            "Phase (derived)",
            "Clearance (est., cm)",
        }
        self.assertEqual(set(JOINT_INFO_TITLES.values()), required)
        self.assertEqual(len(JOINT_INFO_FIELD_ORDER), 11)

    def test_empty_snapshot_uses_dashes(self) -> None:
        info = empty_joint_information()
        for value in info.as_field_map().values():
            self.assertEqual(value, "—")

    def test_build_live_fields_and_angular_acceleration(self) -> None:
        snaps = [
            _snap(0, time_s=0.0, angle=10.0, omega=20.0),
            _snap(1, time_s=0.1, angle=12.0, omega=40.0),
        ]
        recording = GaitMotionRecording(
            source="test",
            fps=10.0,
            snapshots=snaps,
        )
        info = build_joint_information(
            "right_knee",
            snaps[1],
            recording=recording,
            sequence=None,
            gait_phase="Mid-stance",
            end_frame_float=1.0,
        )
        self.assertEqual(info.joint_name.lower(), "right knee")
        self.assertEqual(info.frame, "2")
        self.assertEqual(info.time, "0.10 s")
        self.assertEqual(info.x, "10.0 cm")
        self.assertEqual(info.angle, "12.0°")
        self.assertEqual(info.angular_velocity, "40.0 °/s")
        # Δω/Δt = (40 - 20) / 0.1 = 200 °/s²
        self.assertEqual(info.angular_acceleration, "200.0 °/s²")
        self.assertEqual(info.gait_phase, "Mid-stance")
        self.assertTrue(info.foot_clearance.startswith("N/A"))
        self.assertIn("foot", info.foot_clearance.lower())
        self.assertIn("–", info.rom)

    def test_no_selection_returns_empty(self) -> None:
        info = build_joint_information(None, None)
        self.assertEqual(info, empty_joint_information())


if __name__ == "__main__":
    unittest.main()
