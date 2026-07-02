"""
Kinematics utilities: joint angles and limb geometry from keypoints.

Used by pose estimation (Step 2) and later by visualization / simulation (Steps 3–7).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping

import numpy as np

from stablewalk.pose.dof import GAIT_ANGLE_FIELDS
from stablewalk.models.pose_data import JointAngles, Keypoint, PoseFrame

if TYPE_CHECKING:
    pass


def _kp_map(keypoints: list[Keypoint]) -> dict[str, Keypoint]:
    return {kp.name: kp for kp in keypoints}


def _visible(kp: Keypoint | None, min_visibility: float = 0.5) -> bool:
    return kp is not None and kp.visibility >= min_visibility


def angle_at_joint(
    proximal: Keypoint | None,
    joint: Keypoint | None,
    distal: Keypoint | None,
    min_visibility: float = 0.5,
) -> float | None:
    """
    Interior angle at `joint` formed by segments proximal→joint and distal→joint.

    Returns angle in degrees [0, 180], or None if landmarks are not reliable.
    """
    if not all(_visible(k, min_visibility) for k in (proximal, joint, distal)):
        return None

    # 2D vectors in image plane (x right, y down)
    v1 = np.array([proximal.x - joint.x, proximal.y - joint.y], dtype=float)
    v2 = np.array([distal.x - joint.x, distal.y - joint.y], dtype=float)

    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return None

    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_angle)))


def _segment_angle_from_horizontal(
    a: Keypoint | None,
    b: Keypoint | None,
    min_visibility: float = 0.5,
) -> float | None:
    """Angle of segment a→b relative to horizontal axis (degrees)."""
    if not _visible(a, min_visibility) or not _visible(b, min_visibility):
        return None
    dx = b.x - a.x
    dy = b.y - a.y
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return None
    return float(math.degrees(math.atan2(dy, dx)))


def _segment_angle_from_vertical(
    a: Keypoint | None,
    b: Keypoint | None,
    min_visibility: float = 0.5,
) -> float | None:
    """Tilt of segment a→b from vertical (0° = upright, 90° = horizontal)."""
    if not _visible(a, min_visibility) or not _visible(b, min_visibility):
        return None
    dx = b.x - a.x
    dy = b.y - a.y
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return None
    # Vertical in image coords is (0, 1); angle between segment and vertical
    seg_angle = math.degrees(math.atan2(dy, dx))
    vertical_angle = 90.0
    diff = abs(seg_angle - vertical_angle)
    return float(min(diff, 180.0 - diff))


def _pelvis_rotation(kp: dict[str, Keypoint], min_visibility: float) -> float | None:
    """Pelvis line (hip to hip) angle from horizontal."""
    return _segment_angle_from_horizontal(
        kp.get("left_hip"), kp.get("right_hip"), min_visibility
    )


def _midpoint_keypoint(
    kp: dict[str, Keypoint],
    a: str,
    b: str,
    name: str = "mid",
) -> Keypoint | None:
    la, rb = kp.get(a), kp.get(b)
    if la is None or rb is None:
        return None
    return Keypoint(
        name=name,
        x=(la.x + rb.x) / 2,
        y=(la.y + rb.y) / 2,
        z=(la.z + rb.z) / 2,
        visibility=min(la.visibility, rb.visibility),
    )


def compute_joint_angles(
    keypoints: list[Keypoint],
    min_visibility: float = 0.5,
) -> JointAngles:
    """
    Compute 14+ DOF joint angles from MediaPipe-style landmark names.
    """
    kp = _kp_map(keypoints)

    def ang(a: str, b: str, c: str) -> float | None:
        return angle_at_joint(kp.get(a), kp.get(b), kp.get(c), min_visibility)

    left_ankle_flex = ang("left_knee", "left_ankle", "left_foot_index")
    right_ankle_flex = ang("right_knee", "right_ankle", "right_foot_index")
    mid_shoulder = _midpoint_keypoint(kp, "left_shoulder", "right_shoulder", "mid_shoulder")

    return JointAngles(
        left_elbow=ang("left_shoulder", "left_elbow", "left_wrist"),
        right_elbow=ang("right_shoulder", "right_elbow", "right_wrist"),
        left_knee=ang("left_hip", "left_knee", "left_ankle"),
        right_knee=ang("right_hip", "right_knee", "right_ankle"),
        left_hip=ang("left_shoulder", "left_hip", "left_knee"),
        right_hip=ang("right_shoulder", "right_hip", "right_knee"),
        left_shoulder=ang("left_hip", "left_shoulder", "left_elbow"),
        right_shoulder=ang("right_hip", "right_shoulder", "right_elbow"),
        left_ankle=left_ankle_flex,
        right_ankle=right_ankle_flex,
        left_ankle_flexion=left_ankle_flex,
        right_ankle_flexion=right_ankle_flex,
        neck=ang("left_shoulder", "nose", "right_shoulder"),
        head_neck=angle_at_joint(mid_shoulder, kp.get("nose"), kp.get("mid_hip"), min_visibility),
        torso_tilt=_segment_angle_from_vertical(kp.get("mid_hip"), kp.get("nose"), min_visibility),
        pelvis_rotation=_pelvis_rotation(kp, min_visibility),
        spine=ang("nose", "mid_hip", "left_hip"),
        left_wrist=ang("left_elbow", "left_wrist", "left_index"),
        right_wrist=ang("right_elbow", "right_wrist", "right_index"),
    )


def pose_bounding_spans(
    keypoints: list[Keypoint],
    min_visibility: float = 0.5,
) -> tuple[float, float, int]:
    """
    Return (x_span, y_span, visible_count) in normalized image coordinates.
    """
    visible = [kp for kp in keypoints if kp.visibility >= min_visibility]
    if len(visible) < 2:
        return 0.0, 0.0, len(visible)
    xs = [max(0.0, min(1.0, kp.x)) for kp in visible]
    ys = [max(0.0, min(1.0, kp.y)) for kp in visible]
    return max(xs) - min(xs), max(ys) - min(ys), len(visible)


def is_quality_pose(
    keypoints: list[Keypoint],
    *,
    min_visibility: float = 0.5,
    min_x_span: float = 0.18,
    min_y_span: float = 0.35,
    min_landmarks: int = 18,
) -> bool:
    """True when landmarks cover enough of the frame for a usable skeleton."""
    from stablewalk.pose.validation import is_plausible_human_pose

    if not is_plausible_human_pose(keypoints, min_visibility):
        return False
    x_span, y_span, count = pose_bounding_spans(keypoints, min_visibility)
    return count >= min_landmarks and x_span >= min_x_span and y_span >= min_y_span


def limb_length(
    keypoints: Mapping[str, Keypoint],
    start: str,
    end: str,
    min_visibility: float = 0.5,
) -> float | None:
    """Euclidean length between two landmarks in normalized image coordinates."""
    a, b = keypoints.get(start), keypoints.get(end)
    if not _visible(a, min_visibility) or not _visible(b, min_visibility):
        return None
    return float(math.hypot(b.x - a.x, b.y - a.y))


def compute_joint_velocities(
    previous: list[Keypoint],
    current: list[Keypoint],
    delta_time: float,
    *,
    min_visibility: float = 0.5,
) -> dict[str, dict[str, float]]:
    """
    Per-joint velocity in normalized image coordinates per second.

    velocity = (current_position - previous_position) / delta_time
    """
    if delta_time <= 0:
        return {}

    prev_map = _kp_map(previous)
    curr_map = _kp_map(current)
    velocities: dict[str, dict[str, float]] = {}

    for name, curr_kp in curr_map.items():
        if name == "mid_hip":
            continue
        prev_kp = prev_map.get(name)
        if not _visible(prev_kp, min_visibility) or not _visible(curr_kp, min_visibility):
            continue

        vx = (curr_kp.x - prev_kp.x) / delta_time
        vy = (curr_kp.y - prev_kp.y) / delta_time
        velocities[name] = {
            "vx": float(vx),
            "vy": float(vy),
            "speed": float(math.hypot(vx, vy)),
        }

    return velocities


def velocities_to_scalar_map(
    velocities: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Export-friendly speed magnitude per joint."""
    return {
        name: float(v["speed"])
        for name, v in velocities.items()
        if "speed" in v
    }


def dof_angular_velocities(
    prev: PoseFrame,
    curr: PoseFrame,
    fps: float,
) -> dict[str, float]:
    """Angular velocity (degrees per second) for each gait DoF angle."""
    if not prev.joint_angles or not curr.joint_angles:
        return {}
    delta_frames = max(curr.frame_index - prev.frame_index, 1)
    dt = delta_frames / max(fps, 1e-6)
    out: dict[str, float] = {}
    for name in GAIT_ANGLE_FIELDS:
        a0 = getattr(prev.joint_angles, name, None)
        a1 = getattr(curr.joint_angles, name, None)
        if a0 is not None and a1 is not None:
            out[name] = float((a1 - a0) / dt)
    return out


def velocity_between_frames(
    prev: PoseFrame,
    curr: PoseFrame,
    fps: float,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Velocity from two detected frames using their frame indices."""
    if not prev.keypoints or not curr.keypoints:
        return {}, {}
    delta = max(curr.frame_index - prev.frame_index, 1)
    dt = delta / max(fps, 1e-6)
    velocities = compute_joint_velocities(prev.keypoints, curr.keypoints, dt)
    return velocities, velocities_to_scalar_map(velocities)


def attach_sequence_velocities(frames: list, fps: float) -> None:
    """
    Fill PoseFrame.velocities and velocity_scalar in-place.

    Uses actual frame-index spacing: dt = (i - i_prev) / fps so skipped frames
    do not underestimate speed.
    """
    from stablewalk.models.pose_data import PoseFrame  # noqa: F401

    prev_detected: PoseFrame | None = None

    for frame in frames:
        if not frame.detected or not frame.keypoints:
            frame.velocities = {}
            frame.velocity_scalar = {}
            continue

        if prev_detected and prev_detected.keypoints:
            delta_frames = frame.frame_index - prev_detected.frame_index
            dt = delta_frames / max(fps, 1e-6)
            frame.velocities = compute_joint_velocities(
                prev_detected.keypoints,
                frame.keypoints,
                dt,
            )
        else:
            frame.velocities = {}

        frame.velocity_scalar = velocities_to_scalar_map(frame.velocities)
        prev_detected = frame
