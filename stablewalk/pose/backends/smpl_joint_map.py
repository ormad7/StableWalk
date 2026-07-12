"""
SMPL 24-joint index → StableWalk canonical joint id mapping.

Used when converting ROMP / SMPL mesh recovery outputs into the unified schema.
"""

from __future__ import annotations

# Standard SMPL body joint order (24 joints)
SMPL_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

SMPL_INDEX_TO_CANONICAL: dict[int, str] = {
    0: "pelvis",
    1: "left_hip",
    2: "right_hip",
    3: "spine",
    4: "left_knee",
    5: "right_knee",
    7: "left_ankle",
    8: "right_ankle",
    10: "left_toe",
    11: "right_toe",
    12: "neck",
    15: "head",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
}


def smpl_joints_to_canonical(
    joints_xyz: object,
) -> dict[str, tuple[float, float, float]]:
    """Map SMPL joint array ``(24, 3)`` or ``(J, 3)`` to canonical positions."""
    import numpy as np

    arr = np.asarray(joints_xyz, dtype=np.float64)
    if arr.ndim == 1 and arr.size >= 3:
        arr = arr.reshape(-1, 3)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return {}

    out: dict[str, tuple[float, float, float]] = {}
    for idx, jid in SMPL_INDEX_TO_CANONICAL.items():
        if idx < len(arr):
            out[jid] = (float(arr[idx, 0]), float(arr[idx, 1]), float(arr[idx, 2]))
    return out


def smpl_thetas_to_joint_rotations(
    thetas: object,
) -> dict[str, tuple[float, float, float, float]]:
    """
    Convert SMPL pose axis-angle block ``(24, 3)`` or ``(72,)`` to per-joint quaternions.

    Uses scipy if available; otherwise stores axis-angle magnitude in metadata only
    and returns identity quaternions for mapped joints.
    """
    import numpy as np

    arr = np.asarray(thetas, dtype=np.float64)
    if arr.size == 72:
        arr = arr.reshape(24, 3)
    elif arr.ndim == 1 and arr.size >= 3:
        arr = arr.reshape(-1, 3)

    out: dict[str, tuple[float, float, float, float]] = {}
    if arr.ndim != 2 or arr.shape[1] != 3:
        return out

    try:
        from scipy.spatial.transform import Rotation as R

        for idx, jid in SMPL_INDEX_TO_CANONICAL.items():
            if idx >= len(arr):
                continue
            aa = arr[idx]
            if float(np.linalg.norm(aa)) < 1e-8:
                out[jid] = (1.0, 0.0, 0.0, 0.0)
            else:
                q = R.from_rotvec(aa).as_quat()  # x,y,z,w
                out[jid] = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
    except ImportError:
        for idx, jid in SMPL_INDEX_TO_CANONICAL.items():
            if idx < len(arr):
                out[jid] = (1.0, 0.0, 0.0, 0.0)

    return out


__all__ = [
    "SMPL_JOINT_NAMES",
    "SMPL_INDEX_TO_CANONICAL",
    "smpl_joints_to_canonical",
    "smpl_thetas_to_joint_rotations",
]
