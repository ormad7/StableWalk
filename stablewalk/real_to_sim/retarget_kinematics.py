"""
Kinematics helpers for human → humanoid retargeting.

Computes joint flexion angles from 3D positions, applies segment-specific scaling,
and enforces robot joint limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stablewalk.models.joint_registry import ROOT_JOINT_ID
from stablewalk.real_to_sim.robot_config import RobotJointDefinition, RobotSkeletonConfig


@dataclass
class ScaleFactors:
    """Per-body-region scale factors (not one global multiplier)."""

    leg: float = 1.0
    arm: float = 1.0
    torso: float = 1.0
    root_translation: float = 1.0
    foot: float = 1.0

    def for_segment(self, segment: str | None) -> float:
        if segment == "thigh" or segment == "shank":
            return self.leg
        if segment in ("upper_arm", "forearm"):
            return self.arm
        if segment == "torso":
            return self.torso
        if segment == "foot":
            return self.foot
        return self.root_translation

    def to_dict(self) -> dict[str, float]:
        return {
            "leg": self.leg,
            "arm": self.arm,
            "torso": self.torso,
            "root_translation": self.root_translation,
            "foot": self.foot,
        }


@dataclass
class JointLimitStats:
    """How often joint limits were applied during retargeting."""

    clamp_count: int = 0
    total_samples: int = 0
    per_joint_clamps: dict[str, int] = field(default_factory=dict)

    @property
    def clamp_fraction(self) -> float:
        if self.total_samples <= 0:
            return 0.0
        return self.clamp_count / self.total_samples

    def to_dict(self) -> dict[str, Any]:
        return {
            "clamp_count": self.clamp_count,
            "total_samples": self.total_samples,
            "clamp_fraction": self.clamp_fraction,
            "per_joint_clamps": dict(self.per_joint_clamps),
        }


def compute_scale_factors(
    human_dims: dict[str, Any],
    robot: RobotSkeletonConfig,
) -> ScaleFactors:
    """Derive regional scale factors from human body dimensions vs robot reference."""
    human_leg = float(human_dims.get("leg_length_average_m", 0.9) or 0.9)
    human_arm = float(human_dims.get("arm_length_average_m", 0.55) or 0.55)
    ref_leg = max(robot.scale_reference_leg_length_m, 0.4)
    robot_arm = robot.segment_lengths_m.get("upper_arm", 0.28) + robot.segment_lengths_m.get(
        "forearm", 0.25
    )
    leg_ratio = ref_leg / max(human_leg, 0.4)
    arm_ratio = robot_arm / max(human_arm, 0.35)
    return ScaleFactors(
        leg=leg_ratio,
        arm=arm_ratio,
        torso=leg_ratio,
        root_translation=leg_ratio,
        foot=leg_ratio,
    )


def transform_position(
    pos: np.ndarray,
    *,
    root: np.ndarray,
    scale: float,
    axis_flip: np.ndarray | None = None,
) -> np.ndarray:
    """Scale offset from root and optionally flip axes human→robot."""
    offset = pos - root
    if axis_flip is not None:
        offset = offset * axis_flip
    return root + offset * scale


def bone_flexion_angle(
    parent: np.ndarray,
    joint: np.ndarray,
    child: np.ndarray,
) -> float:
    """Flexion angle (radians) from three joint positions."""
    v1 = joint - parent
    v2 = child - joint
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    v1 = v1 / n1
    v2 = v2 / n2
    cos_a = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.pi - np.arccos(cos_a))


def clamp_joint_angle(
    angle_rad: float,
    limits: tuple[float, float] | None,
) -> tuple[float, bool]:
    """Return clamped angle and whether clamping occurred."""
    if limits is None or not np.isfinite(angle_rad):
        return angle_rad, False
    lo, hi = limits
    clamped = float(np.clip(angle_rad, lo, hi))
    return clamped, clamped != angle_rad


def extract_human_joint_positions(
    motion_joints: np.ndarray,
    joint_ids: list[str],
    frame_index: int,
) -> dict[str, np.ndarray]:
    """Get human joint positions for one frame from canonical array."""
    positions: dict[str, np.ndarray] = {}
    id_to_idx = {jid: i for i, jid in enumerate(joint_ids)}
    for jid, idx in id_to_idx.items():
        pos = motion_joints[frame_index, idx]
        if np.all(np.isfinite(pos)):
            positions[jid] = np.asarray(pos, dtype=np.float64)
    if ROOT_JOINT_ID not in positions:
        lh = positions.get("left_hip")
        rh = positions.get("right_hip")
        if lh is not None and rh is not None:
            positions[ROOT_JOINT_ID] = (lh + rh) * 0.5
    return positions


def estimate_root_orientation_from_motion(
    root_positions: np.ndarray,
    frame_index: int,
) -> np.ndarray:
    """Estimate root quaternion wxyz from root displacement (same as AMP export)."""
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if frame_index < 1:
        return quat
    dx = root_positions[frame_index, 0] - root_positions[frame_index - 1, 0]
    dy = root_positions[frame_index, 1] - root_positions[frame_index - 1, 1]
    dz = root_positions[frame_index, 2] - root_positions[frame_index - 1, 2]
    horiz = np.hypot(dx, dz)
    if horiz < 1e-8 and abs(dy) < 1e-8:
        return quat
    yaw = float(np.arctan2(dz, dx))
    pitch = float(np.arctan2(dy, horiz))
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    return np.array([cy * cp, cy * sp, sy * cp, sy * sp], dtype=np.float64)


def compute_robot_dof_angles_frame(
    human_pos: dict[str, np.ndarray],
    robot: RobotSkeletonConfig,
    scales: ScaleFactors,
    limit_stats: JointLimitStats,
) -> dict[str, float]:
    """Map human 3D pose to robot revolute DOF angles for one frame."""
    dof_angles: dict[str, float] = {}
    for joint_def in robot.revolute_joints:
        hj = joint_def.human_joint
        hc = joint_def.human_child
        if not hj or not hc:
            continue
        parent_hj = robot.joint_by_name.get(joint_def.parent or "")
        parent_key = parent_hj.human_joint if parent_hj and parent_hj.human_joint else ROOT_JOINT_ID

        p = human_pos.get(parent_key)
        j = human_pos.get(hj)
        c = human_pos.get(hc)
        if p is None or j is None or c is None:
            continue

        angle = bone_flexion_angle(p, j, c)
        if not np.isfinite(angle):
            continue

        clamped, was_clamped = clamp_joint_angle(angle, joint_def.limits_rad)
        dof_angles[joint_def.name] = clamped
        limit_stats.total_samples += 1
        if was_clamped:
            limit_stats.clamp_count += 1
            limit_stats.per_joint_clamps[joint_def.name] = (
                limit_stats.per_joint_clamps.get(joint_def.name, 0) + 1
            )
    return dof_angles


def compute_target_joint_positions_frame(
    human_pos: dict[str, np.ndarray],
    robot: RobotSkeletonConfig,
    scales: ScaleFactors,
    root_scaled: np.ndarray,
    axis_flip: np.ndarray,
) -> dict[str, np.ndarray]:
    """Place robot link positions using segment-scaled offsets from root."""
    root_human = human_pos.get(ROOT_JOINT_ID)
    if root_human is None:
        return {}

    out: dict[str, np.ndarray] = {robot.root_joint: root_scaled.copy()}
    for joint_def in robot.joints:
        if joint_def.joint_type == "free":
            continue
        hj = joint_def.human_joint
        if not hj or hj not in human_pos:
            continue
        seg_scale = scales.for_segment(joint_def.segment)
        out[joint_def.name] = transform_position(
            human_pos[hj],
            root=root_human,
            scale=seg_scale,
            axis_flip=axis_flip,
        )
        out[joint_def.name] = out[joint_def.name] - root_human + root_scaled

    return out


def mapping_coverage_report(
    robot: RobotSkeletonConfig,
    human_joint_ids: list[str],
) -> tuple[float, list[str], list[str]]:
    """
    Return (coverage fraction, mapped human joints, unmapped robot revolute joints).
    """
    human_set = set(human_joint_ids)
    mapped_human: list[str] = []
    unmapped_robot: list[str] = []

    for joint_def in robot.revolute_joints:
        hj = joint_def.human_joint
        hc = joint_def.human_child
        if hj and hj in human_set and hc and hc in human_set:
            mapped_human.append(hj)
        else:
            missing = []
            if hj and hj not in human_set:
                missing.append(hj)
            if hc and hc not in human_set:
                missing.append(hc)
            unmapped_robot.append(
                f"{joint_def.name} (missing human: {', '.join(missing) or '?'})"
            )

    total = len(robot.revolute_joints) or 1
    coverage = len(mapped_human) / total
    return coverage, mapped_human, unmapped_robot


__all__ = [
    "ScaleFactors",
    "JointLimitStats",
    "compute_scale_factors",
    "transform_position",
    "bone_flexion_angle",
    "clamp_joint_angle",
    "extract_human_joint_positions",
    "estimate_root_orientation_from_motion",
    "compute_robot_dof_angles_frame",
    "compute_target_joint_positions_frame",
    "mapping_coverage_report",
]
