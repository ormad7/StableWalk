"""
Robotic humanoid model: revolute joints driven by gait DOF angles.

Interior angles from pose estimation are converted to mechanical joint rotations.
Forward kinematics places link endpoints in 3D (hip-centered, Y = forward walk).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from stablewalk.pose.dof import GAIT_ANGLE_FIELDS
from stablewalk.models.pose_data import JointAngles

# Mechanical joint names (revolute axes)
ROBOT_JOINT_NAMES: tuple[str, ...] = (
    "pelvis_yaw",
    "torso_pitch",
    "neck_pitch",
    "head_pitch",
    "left_shoulder_pitch",
    "left_elbow_flex",
    "right_shoulder_pitch",
    "right_elbow_flex",
    "left_hip_pitch",
    "left_knee_flex",
    "left_ankle_pitch",
    "right_hip_pitch",
    "right_knee_flex",
    "right_ankle_pitch",
)

ROBOT_SEGMENT_NAMES: tuple[tuple[str, str], ...] = (
    ("pelvis", "torso_top"),
    ("torso_top", "neck"),
    ("neck", "head"),
    ("torso_top", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("torso_top", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
)


@dataclass
class RobotConfig:
    """Link lengths in meters (normalized humanoid proportions)."""

    hip_half_width: float = 0.10
    thigh_length: float = 0.26
    shank_length: float = 0.26
    foot_length: float = 0.08
    torso_length: float = 0.28
    neck_length: float = 0.06
    head_length: float = 0.12
    upper_arm: float = 0.17
    forearm: float = 0.15
    shoulder_offset_y: float = 0.08  # shoulders slightly forward of pelvis


@dataclass
class RobotJointState:
    """Revolute joint angles in radians (0 ≈ neutral / extended)."""

    pelvis_yaw: float = 0.0
    torso_pitch: float = 0.0
    neck_pitch: float = 0.0
    head_pitch: float = 0.0
    left_shoulder_pitch: float = 0.0
    left_elbow_flex: float = 0.0
    right_shoulder_pitch: float = 0.0
    right_elbow_flex: float = 0.0
    left_hip_pitch: float = 0.0
    left_knee_flex: float = 0.0
    left_ankle_pitch: float = 0.0
    right_hip_pitch: float = 0.0
    right_knee_flex: float = 0.0
    right_ankle_pitch: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in ROBOT_JOINT_NAMES}

    def active_dof_count(self) -> int:
        return len(ROBOT_JOINT_NAMES)


@dataclass
class RobotGeometry:
    """Link endpoint positions in robot base frame."""

    points: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    segments: tuple[tuple[str, str], ...] = ROBOT_SEGMENT_NAMES

    def get(self, name: str) -> tuple[float, float, float] | None:
        return self.points.get(name)


def _deg(val: float | None, default: float = 0.0) -> float:
    return float(val) if val is not None else default


def interior_to_flexion(interior_deg: float | None, default: float = 0.0) -> float:
    """
    Convert interior joint angle (180° = straight) to flexion rotation (0 = straight).

    Positive flexion = bending (knee/elbow/hip flexion convention).
    """
    if interior_deg is None:
        return default
    return math.radians(180.0 - interior_deg)


def joint_angles_to_robot_state(angles: JointAngles | None) -> RobotJointState:
    """Map gait DOF (degrees, interior angles) → robotic revolute joint state (rad)."""
    if angles is None:
        return RobotJointState()

    pelvis_deg = _deg(angles.pelvis_rotation)
    torso_deg = _deg(angles.torso_tilt)

    return RobotJointState(
        pelvis_yaw=math.radians(pelvis_deg),
        torso_pitch=math.radians(torso_deg),
        neck_pitch=interior_to_flexion(angles.neck),
        head_pitch=interior_to_flexion(angles.head_neck),
        left_shoulder_pitch=interior_to_flexion(angles.left_shoulder),
        left_elbow_flex=interior_to_flexion(angles.left_elbow),
        right_shoulder_pitch=interior_to_flexion(angles.right_shoulder),
        right_elbow_flex=interior_to_flexion(angles.right_elbow),
        left_hip_pitch=interior_to_flexion(angles.left_hip),
        left_knee_flex=interior_to_flexion(angles.left_knee),
        left_ankle_pitch=interior_to_flexion(
            angles.left_ankle_flexion or angles.left_ankle
        ),
        right_hip_pitch=interior_to_flexion(angles.right_hip),
        right_knee_flex=interior_to_flexion(angles.right_knee),
        right_ankle_pitch=interior_to_flexion(
            angles.right_ankle_flexion or angles.right_ankle
        ),
    )


def _rot_y(angle: float) -> tuple[tuple[float, float, float], ...]:
    c, s = math.cos(angle), math.sin(angle)
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def _mat_vec(m: tuple, v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _add(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def forward_kinematics(
    joints: RobotJointState,
    cfg: RobotConfig | None = None,
) -> RobotGeometry:
    """
    Compute 3D positions of all robot link endpoints.

    Frame: pelvis at origin; +Y forward (walk); +Z up; +X = left.
    """
    cfg = cfg or RobotConfig()
    pts: dict[str, tuple[float, float, float]] = {}

    pelvis = (0.0, 0.0, 0.0)
    pts["pelvis"] = pelvis

    # Torso (pitch + pelvis yaw)
    ry = _rot_y(joints.pelvis_yaw)
    rx = ((1, 0, 0), (0, math.cos(joints.torso_pitch), -math.sin(joints.torso_pitch)),
          (0, math.sin(joints.torso_pitch), math.cos(joints.torso_pitch)))
    torso_vec = _mat_vec(ry, _mat_vec(rx, (0, cfg.shoulder_offset_y, cfg.torso_length)))
    torso_top = _add(pelvis, torso_vec)
    pts["torso_top"] = torso_top

    # Neck & head
    neck_base = torso_top
    rx_n = ((1, 0, 0), (0, math.cos(joints.neck_pitch), -math.sin(joints.neck_pitch)),
            (0, math.sin(joints.neck_pitch), math.cos(joints.neck_pitch)))
    neck = _add(neck_base, _mat_vec(rx_n, (0, 0, cfg.neck_length)))
    pts["neck"] = neck
    rx_h = ((1, 0, 0), (0, math.cos(joints.head_pitch), -math.sin(joints.head_pitch)),
            (0, math.sin(joints.head_pitch), math.cos(joints.head_pitch)))
    head = _add(neck, _mat_vec(rx_h, (0, 0, cfg.head_length)))
    pts["head"] = head

    def arm(side: str, sign: float, q_shoulder: float, q_elbow: float) -> None:
        shoulder = _add(torso_top, (sign * cfg.hip_half_width * 0.85, 0, 0))
        pts[f"{side}_shoulder"] = shoulder
        rx_s = ((1, 0, 0), (0, math.cos(q_shoulder), -math.sin(q_shoulder)),
                (0, math.sin(q_shoulder), math.cos(q_shoulder)))
        elbow = _add(shoulder, _mat_vec(rx_s, (0, 0, -cfg.upper_arm)))
        pts[f"{side}_elbow"] = elbow
        rx_e = ((1, 0, 0),
                (0, math.cos(q_shoulder + q_elbow), -math.sin(q_shoulder + q_elbow)),
                (0, math.sin(q_shoulder + q_elbow), math.cos(q_shoulder + q_elbow)))
        wrist = _add(shoulder, _mat_vec(rx_e, (0, 0, -(cfg.upper_arm + cfg.forearm))))
        pts[f"{side}_wrist"] = wrist

    arm("left", 1.0, joints.left_shoulder_pitch, joints.left_elbow_flex)
    arm("right", -1.0, joints.right_shoulder_pitch, joints.right_elbow_flex)

    def leg(side: str, sign: float, q_hip: float, q_knee: float, q_ankle: float) -> None:
        hip = (sign * cfg.hip_half_width, 0.0, 0.0)
        pts[f"{side}_hip"] = hip

        def rot_x(a: float) -> tuple:
            c, s = math.cos(a), math.sin(a)
            return ((1, 0, 0), (0, c, -s), (0, s, c))

        # Sagittal plane (Y-Z): hip/knee/ankle pitch about X; +Y = forward step
        rx1 = rot_x(-q_hip)
        knee = _add(hip, _mat_vec(rx1, (0, 0, -cfg.thigh_length)))
        pts[f"{side}_knee"] = knee
        rx2 = rot_x(-(q_hip + q_knee))
        ankle = _add(hip, _mat_vec(rx2, (0, 0, -(cfg.thigh_length + cfg.shank_length))))
        pts[f"{side}_ankle"] = ankle
        rx3 = rot_x(-(q_hip + q_knee + q_ankle))
        foot = _add(
            hip,
            _mat_vec(
                rx3,
                (0, cfg.foot_length, -(cfg.thigh_length + cfg.shank_length)),
            ),
        )
        pts[f"{side}_foot"] = foot

    leg("left", 1.0, joints.left_hip_pitch, joints.left_knee_flex, joints.left_ankle_pitch)
    leg("right", -1.0, joints.right_hip_pitch, joints.right_knee_flex, joints.right_ankle_pitch)

    return RobotGeometry(points=pts)
