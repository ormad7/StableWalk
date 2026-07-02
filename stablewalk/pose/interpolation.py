"""
Smooth motion between pose frames using linear interpolation and velocity hints.
"""

from __future__ import annotations

import math
from dataclasses import fields

from stablewalk.models.pose_data import JointAngles
from stablewalk.simulation.robotic_model import (
    RobotConfig,
    RobotGeometry,
    RobotJointState,
    forward_kinematics,
    joint_angles_to_robot_state,
)
from stablewalk.pose.skeleton_3d import Joint3D, Skeleton3D, SKELETON_3D_JOINTS


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interpolate_skeleton_3d(a: Skeleton3D, b: Skeleton3D, alpha: float) -> Skeleton3D:
    """Blend two 3D skeletons (same joint set)."""
    alpha = max(0.0, min(1.0, alpha))
    names = set(a.joints) | set(b.joints)
    joints: dict[str, Joint3D] = {}
    for name in names:
        ja, jb = a.joints.get(name), b.joints.get(name)
        if ja and jb:
            joints[name] = Joint3D(
                name=name,
                x=_lerp(ja.x, jb.x, alpha),
                y=_lerp(ja.y, jb.y, alpha),
                z=_lerp(ja.z, jb.z, alpha),
                visibility=_lerp(ja.visibility, jb.visibility, alpha),
            )
        elif ja:
            joints[name] = ja
        elif jb:
            joints[name] = jb
    scale = _lerp(a.scale, b.scale, alpha)
    return Skeleton3D(joints=joints, scale=scale)


def normalize_skeleton_height(skeleton: Skeleton3D, target_height: float = 1.0) -> Skeleton3D:
    """Scale skeleton so vertical span (head to feet) matches target height."""
    if not skeleton.joints:
        return skeleton
    ys = [j.y for j in skeleton.joints.values()]
    height = max(ys) - min(ys)
    if height < 1e-6:
        return skeleton
    factor = target_height / height
    joints = {
        name: Joint3D(
            name=j.name,
            x=j.x * factor,
            y=j.y * factor,
            z=j.z * factor,
            visibility=j.visibility,
        )
        for name, j in skeleton.joints.items()
    }
    return Skeleton3D(joints=joints, scale=skeleton.scale * factor)


def interpolate_joint_angles(
    a: JointAngles | None,
    b: JointAngles | None,
    alpha: float,
) -> JointAngles | None:
    """Linear interpolation of all angle fields (degrees)."""
    if a is None:
        return b
    if b is None:
        return a
    alpha = max(0.0, min(1.0, alpha))
    out: dict[str, float | None] = {}
    for f in fields(JointAngles):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if va is not None and vb is not None:
            out[f.name] = _lerp(va, vb, alpha)
        elif va is not None:
            out[f.name] = va
        else:
            out[f.name] = vb
    return JointAngles(**out)


def velocity_adjusted_angles(
    prev: JointAngles | None,
    curr: JointAngles | None,
    alpha: float,
    dt: float,
    velocity_scalar: dict[str, float] | None = None,
) -> JointAngles | None:
    """
    Interpolate angles with optional velocity-based extrapolation for smoother motion.

    Uses hip/knee angular rate estimated from neighbor frames when velocities are present.
    """
    base = interpolate_joint_angles(prev, curr, alpha)
    if base is None or not velocity_scalar or dt <= 0:
        return base

    # Angular velocity from consecutive frames (degrees per second)
    nudge_scale = 2.0 * alpha * (1.0 - alpha)
    if prev and curr and dt > 0:
        for f in fields(JointAngles):
            pv, cv = getattr(prev, f.name), getattr(curr, f.name)
            if pv is not None and cv is not None:
                rate = (cv - pv) / dt
                blended = getattr(base, f.name)
                if blended is not None:
                    setattr(base, f.name, blended + rate * dt * nudge_scale * 0.15)

    # Landmark speed nudge for legs
    mapping = {
        "left_hip": "left_hip",
        "right_hip": "right_hip",
        "left_knee": "left_knee",
        "right_knee": "right_knee",
        "left_ankle": "left_ankle_flexion",
        "right_ankle": "right_ankle_flexion",
    }
    for vel_name, angle_name in mapping.items():
        speed = velocity_scalar.get(vel_name)
        if speed is None:
            continue
        val = getattr(base, angle_name, None)
        if val is not None:
            delta = math.degrees(speed) * nudge_scale * 0.35
            setattr(base, angle_name, val + delta)
    return base


def interpolate_robot_geometry(
    geom_a: RobotGeometry,
    geom_b: RobotGeometry,
    alpha: float,
) -> RobotGeometry:
    """Blend robot link endpoint positions."""
    alpha = max(0.0, min(1.0, alpha))
    names = set(geom_a.points) | set(geom_b.points)
    points: dict[str, tuple[float, float, float]] = {}
    for name in names:
        pa, pb = geom_a.points.get(name), geom_b.points.get(name)
        if pa and pb:
            points[name] = (
                _lerp(pa[0], pb[0], alpha),
                _lerp(pa[1], pb[1], alpha),
                _lerp(pa[2], pb[2], alpha),
            )
        elif pa:
            points[name] = pa
        elif pb:
            points[name] = pb
    return RobotGeometry(points=points)


def robot_geometry_from_angles(
    angles: JointAngles | None,
    cfg: RobotConfig | None = None,
) -> RobotGeometry:
    state = joint_angles_to_robot_state(angles)
    return forward_kinematics(state, cfg)


def smooth_skeleton_temporal(
    skeletons: list[Skeleton3D],
    window: int = 3,
) -> list[Skeleton3D]:
    """
    Moving-average smoothing over a sequence of skeletons (reduces jitter).
    """
    if window < 2 or len(skeletons) < 2:
        return skeletons
    half = window // 2
    out: list[Skeleton3D] = []
    for i, skel in enumerate(skeletons):
        lo = max(0, i - half)
        hi = min(len(skeletons), i + half + 1)
        chunk = skeletons[lo:hi]
        names = set(skel.joints)
        for other in chunk:
            names |= set(other.joints)
        joints: dict[str, Joint3D] = {}
        n = len(chunk)
        for name in names:
            xs, ys, zs, vs = [], [], [], []
            for c in chunk:
                j = c.joints.get(name)
                if j:
                    xs.append(j.x)
                    ys.append(j.y)
                    zs.append(j.z)
                    vs.append(j.visibility)
            if xs:
                joints[name] = Joint3D(
                    name=name,
                    x=sum(xs) / len(xs),
                    y=sum(ys) / len(ys),
                    z=sum(zs) / len(zs),
                    visibility=sum(vs) / len(vs),
                )
        out.append(Skeleton3D(joints=joints, scale=skel.scale))
    return out
