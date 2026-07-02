"""Print joint coordinates and angles for detected frames."""

from __future__ import annotations

import logging

from stablewalk.gait_dof import GAIT_VELOCITY_JOINTS
from stablewalk.models.pose_data import JointAngles, Keypoint, PoseFrame

logger = logging.getLogger(__name__)

CORE_JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
)

GAIT_ANGLES = (
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
)


def print_frame_report(frame: PoseFrame) -> None:
    """Log key joint coordinates and gait angles for one frame."""
    if not frame.detected or not frame.keypoints:
        return

    by_name = {kp.name: kp for kp in frame.keypoints}
    logger.info("--- Frame %d ---", frame.frame_index)
    logger.info("Joint coordinates (normalized x, y; visibility):")
    for name in CORE_JOINTS:
        kp = by_name.get(name)
        if kp and kp.visibility >= 0.5:
            logger.info(
                "  %s: (%.3f, %.3f) vis=%.2f",
                name,
                kp.x,
                kp.y,
                kp.visibility,
            )

    if frame.joint_angles:
        angles: JointAngles = frame.joint_angles
        logger.info("Joint angles (degrees) — gait DOF:")
        for name in GAIT_ANGLES:
            value = getattr(angles, name)
            if value is not None:
                logger.info("  %s: %.1f°", name, value)
        logger.info(
            "  Limb angles: %d/6  |  Gait DOF: %d/14+  |  Velocities: %d joints",
            sum(1 for n in GAIT_ANGLES if getattr(angles, n) is not None),
            angles.gait_dof_count(),
            len(frame.velocities),
        )
        if frame.velocity_scalar:
            logger.info("  Key joint speed (|v|):")
            for name in GAIT_VELOCITY_JOINTS:
                speed = frame.velocity_scalar.get(name)
                if speed is not None:
                    logger.info("    %s: %.5f", name, speed)
        if frame.gait_events:
            logger.info("  Gait events: %s", ", ".join(frame.gait_events))


def print_sequence_summary(frames: list[PoseFrame]) -> None:
    detected = [f for f in frames if f.detected]
    logger.info(
        "Summary: %d / %d frames with valid pose",
        len(detected),
        len(frames),
    )
