"""
3D skeleton model: 2D pose → hip-centered (x, y, z) with heuristic depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from stablewalk.coordinates.coordinate_map import (
    DEPTH_LAYERS,
    TARGET_SKELETON_HEIGHT,
    depth_from_mediapipe_landmark,
    mediapipe_to_canonical,
)
from stablewalk.pose.coordinates import hip_center
from stablewalk.models.pose_data import Keypoint

# Primary joints for 3D gait visualization
SKELETON_3D_JOINTS: tuple[str, ...] = (
    "nose",
    "mid_shoulder",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "mid_hip",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Bone connections (pairs of joint names)
SKELETON_3D_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("mid_hip", "left_hip"),
    ("mid_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("mid_hip", "mid_shoulder"),
    ("mid_shoulder", "left_shoulder"),
    ("mid_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("mid_shoulder", "nose"),
)

# Ideal segment lengths as fraction of body height (anthropometric averages)
LIMB_RATIOS: dict[tuple[str, str], float] = {
    ("mid_shoulder", "left_elbow"): 0.186,
    ("left_elbow", "left_wrist"): 0.146,
    ("mid_shoulder", "right_elbow"): 0.186,
    ("right_elbow", "right_wrist"): 0.146,
    ("left_hip", "left_knee"): 0.245,
    ("left_knee", "left_ankle"): 0.246,
    ("right_hip", "right_knee"): 0.245,
    ("right_knee", "right_ankle"): 0.246,
    ("mid_hip", "mid_shoulder"): 0.30,
    ("mid_shoulder", "nose"): 0.13,
}


@dataclass
class Joint3D:
    """Single joint in hip-centered 3D space."""

    name: str
    x: float
    y: float
    z: float
    visibility: float = 1.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "visibility": self.visibility}


@dataclass
class Skeleton3D:
    """Hip-centered 3D skeleton for one frame."""

    joints: dict[str, Joint3D] = field(default_factory=dict)
    scale: float = 1.0

    def get(self, name: str) -> Joint3D | None:
        return self.joints.get(name)

    def max_extent(self) -> float:
        if not self.joints:
            return 0.0
        return max(
            math.sqrt(j.x * j.x + j.y * j.y + j.z * j.z)
            for j in self.joints.values()
        )

    def copy(self) -> Skeleton3D:
        return Skeleton3D(
            joints={
                name: Joint3D(
                    name=j.name,
                    x=j.x,
                    y=j.y,
                    z=j.z,
                    visibility=j.visibility,
                )
                for name, j in self.joints.items()
            },
            scale=self.scale,
        )

    def to_export_dict(self) -> dict[str, Any]:
        return {
            "coordinate_system": "sw_canonical_y_up",
            "scale": self.scale,
            "joints": {name: j.to_dict() for name, j in self.joints.items()},
            "connections": list(SKELETON_3D_CONNECTIONS),
        }


def _rotation_matrix(from_vec: "np.ndarray", to_vec: "np.ndarray") -> list[list[float]]:
    """3x3 rotation mapping unit vector from_vec -> to_vec."""
    import numpy as np

    a = from_vec.astype(float)
    b = to_vec.astype(float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3).tolist()
    a /= na
    b /= nb
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-8:
        if c < 0:
            return np.diag([1.0, -1.0, 1.0]).tolist()
        return np.eye(3).tolist()
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )
    return (np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))).tolist()


def align_skeleton_upright(skeleton: Skeleton3D) -> Skeleton3D:
    """
    Rotate hip-centered skeleton so torso points +Y (up) for consistent display.

    Does not translate; root remains at mid_hip after rotation.
    """
    import numpy as np

    j = skeleton.joints
    if "mid_hip" not in j:
        return skeleton
    hip = np.array([j["mid_hip"].x, j["mid_hip"].y, j["mid_hip"].z])
    top = None
    for name in ("nose", "mid_shoulder"):
        if name in j:
            top = np.array([j[name].x, j[name].y, j[name].z])
            break
    if top is None:
        return skeleton
    spine = top - hip
    if np.linalg.norm(spine) < 1e-8:
        return skeleton
    return transform_skeleton(skeleton, _rotation_matrix(spine, np.array([0.0, 1.0, 0.0])))


def transform_skeleton(skeleton: Skeleton3D, matrix: list[list[float]] | None = None) -> Skeleton3D:
    """Apply 3×3 rotation/reflection matrix to all joints (hip-centered)."""
    if not skeleton.joints or matrix is None:
        return skeleton
    m = matrix
    out = skeleton.copy()
    for name, j in out.joints.items():
        x = m[0][0] * j.x + m[0][1] * j.y + m[0][2] * j.z
        y = m[1][0] * j.x + m[1][1] * j.y + m[1][2] * j.z
        z = m[2][0] * j.x + m[2][1] * j.y + m[2][2] * j.z
        j.x, j.y, j.z = float(x), float(y), float(z)
    return out


def _mid_shoulder(kp_map: dict[str, Keypoint]) -> Keypoint | None:
    left = kp_map.get("left_shoulder")
    right = kp_map.get("right_shoulder")
    if not left or not right:
        return None
    return Keypoint(
        name="mid_shoulder",
        x=(left.x + right.x) / 2,
        y=(left.y + right.y) / 2,
        z=(left.z + right.z) / 2,
        visibility=min(left.visibility, right.visibility),
    )


def estimate_joint_depth(name: str, kp: Keypoint) -> float:
    """Heuristic depth: fixed body layer + MediaPipe relative z."""
    return depth_from_mediapipe_landmark(name, float(kp.z))


def reconstruct_skeleton_3d(
    keypoints: list[Keypoint],
    *,
    scale: float | None = None,
) -> Skeleton3D:
    """
    Convert 2D normalized keypoints to hip-centered 3D skeleton.

    - Origin: mid_hip (0, 0, 0)
    - x: horizontal (image x, centered)
    - y: vertical up (negative of image y)
    - z: estimated depth
    """
    kp_map = {kp.name: kp for kp in keypoints}
    center = hip_center(keypoints)
    cx, cy = center if center else (0.5, 0.5)

    # Ensure mid_hip exists
    if "mid_hip" not in kp_map:
        kp_map["mid_hip"] = Keypoint(name="mid_hip", x=cx, y=cy, z=0.0, visibility=1.0)

    mid_sh = _mid_shoulder(kp_map)
    if mid_sh:
        kp_map["mid_shoulder"] = mid_sh

    joints: dict[str, Joint3D] = {}

    for name in SKELETON_3D_JOINTS:
        kp = kp_map.get(name)
        if kp is None or kp.visibility < 0.3:
            continue

        if name == "mid_hip":
            joints[name] = Joint3D(name=name, x=0.0, y=0.0, z=0.0, visibility=kp.visibility)
            continue

        x, y, z = mediapipe_to_canonical(
            kp.x,
            kp.y,
            kp.z,
            cx=cx,
            cy=cy,
            joint_name=name,
            scale=1.0,
        )
        joints[name] = Joint3D(name=name, x=x, y=y, z=z, visibility=kp.visibility)

    skeleton = Skeleton3D(joints=joints)

    # Normalize by body height (nose to lowest ankle) for consistent proportions
    body_height = _estimate_body_height(joints)
    if body_height > 1e-6:
        height_scale = TARGET_SKELETON_HEIGHT / body_height
    else:
        height_scale = 1.0

    if scale is None:
        skeleton.scale = height_scale
    else:
        skeleton.scale = scale * height_scale

    for joint in skeleton.joints.values():
        joint.x *= skeleton.scale
        joint.y *= skeleton.scale
        joint.z *= skeleton.scale

    _apply_limb_proportions(skeleton)
    return skeleton


def _apply_limb_proportions(skeleton: Skeleton3D) -> None:
    """Rescale arm/leg bones toward anthropometric length ratios."""
    if "mid_hip" not in skeleton.joints:
        return
    height = _estimate_body_height(skeleton.joints)
    if height < 1e-6:
        return

    for (parent_name, child_name), ratio in LIMB_RATIOS.items():
        parent = skeleton.joints.get(parent_name)
        child = skeleton.joints.get(child_name)
        if not parent or not child:
            continue
        dx = child.x - parent.x
        dy = child.y - parent.y
        dz = child.z - parent.z
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        target = ratio * height
        if length < 1e-8 or target < 1e-8:
            continue
        factor = target / length
        child.x = parent.x + dx * factor
        child.y = parent.y + dy * factor
        child.z = parent.z + dz * factor


def _estimate_body_height(joints: dict[str, Joint3D]) -> float:
    """Vertical span used for height normalization."""
    if not joints:
        return 0.0
    ys = [j.y for j in joints.values()]
    ankles = [
        joints[n].y
        for n in ("left_ankle", "right_ankle")
        if n in joints
    ]
    if "nose" in joints and ankles:
        return joints["nose"].y - min(ankles)
    return max(ys) - min(ys) if ys else 0.0


def _unscaled_extent(keypoints: list[Keypoint]) -> float:
    """Max radius before uniform scaling."""
    center = hip_center(keypoints) or (0.5, 0.5)
    cx, cy = center
    kp_map = {kp.name: kp for kp in keypoints}
    if mid_sh := _mid_shoulder(kp_map):
        kp_map["mid_shoulder"] = mid_sh

    max_extent = 0.0
    for name in SKELETON_3D_JOINTS:
        if name == "mid_hip":
            continue
        kp = kp_map.get(name)
        if not kp or kp.visibility < 0.3:
            continue
        x = float(kp.x - cx)
        y = float(-(kp.y - cy))
        z = estimate_joint_depth(name, kp)
        max_extent = max(max_extent, math.sqrt(x * x + y * y + z * z))
    return max_extent


def sequence_skeleton_scale(frames_keypoints: list[list[Keypoint]]) -> float:
    """Uniform scale so all frames fit a consistent 3D view."""
    max_extent = max((_unscaled_extent(kps) for kps in frames_keypoints if kps), default=0.0)
    if max_extent < 1e-6:
        return 1.0
    return TARGET_SKELETON_HEIGHT / max_extent


def skeleton_has_valid_shape(
    skeleton: Skeleton3D,
    *,
    min_extent: float = 0.22,
    min_height: float = 0.32,
    min_joints: int = 6,
) -> bool:
    """False when stored JSON coords collapse to a dot (common with old exports)."""
    if len(skeleton.joints) < min_joints:
        return False
    ys = [j.y for j in skeleton.joints.values()]
    height = max(ys) - min(ys) if ys else 0.0
    return skeleton.max_extent() >= min_extent and height >= min_height


def skeleton_from_frame_data(
    keypoints: list[Keypoint],
    skeleton_3d_data: dict[str, Any] | None,
    uniform_scale: float,
) -> Skeleton3D:
    """Build Skeleton3D from keypoints, reusing stored data if present."""
    raw_joints = None
    if skeleton_3d_data:
        raw_joints = skeleton_3d_data.get("joints") or skeleton_3d_data.get("body")

    if raw_joints and isinstance(raw_joints, dict):
        joints = {}
        for name, coords in raw_joints.items():
            if isinstance(coords, dict) and "x" in coords:
                joints[name] = Joint3D(
                    name=name,
                    x=float(coords["x"]),
                    y=float(coords["y"]),
                    z=float(coords.get("z", 0.0)),
                    visibility=float(coords.get("visibility", 1.0)),
                )
        if joints:
            stored = Skeleton3D(
                joints=joints,
                scale=float(skeleton_3d_data.get("scale", uniform_scale)),
            )
            if skeleton_has_valid_shape(stored) or not keypoints:
                return stored

    return reconstruct_skeleton_3d(keypoints, scale=uniform_scale)
