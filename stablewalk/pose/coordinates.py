"""
Hip-centered coordinates and 3D-ready skeleton layout (z placeholder).

Prepares 2D pose data for future 3D reconstruction without rendering 3D yet.
"""

from __future__ import annotations

from typing import Any

from stablewalk.models.pose_data import Keypoint

HIP_CENTER_NAME = "mid_hip"


def hip_center(keypoints: list[Keypoint]) -> tuple[float, float] | None:
    """Return (x, y) of pelvis center in normalized image coordinates."""
    by_name = {kp.name: kp for kp in keypoints}
    mid = by_name.get(HIP_CENTER_NAME)
    if mid and mid.visibility >= 0.3:
        return mid.x, mid.y
    left = by_name.get("left_hip")
    right = by_name.get("right_hip")
    if left and right and left.visibility >= 0.3 and right.visibility >= 0.3:
        return (left.x + right.x) / 2, (left.y + right.y) / 2
    return None


def extract_positions(keypoints: list[Keypoint]) -> dict[str, dict[str, float]]:
    """Explicit labeled (x, y) for every landmark."""
    return {
        kp.name: {"x": float(kp.x), "y": float(kp.y)}
        for kp in keypoints
        if kp.name != HIP_CENTER_NAME
    }


def extract_positions_xy(keypoints: list[Keypoint]) -> dict[str, list[float]]:
    """Compact [x, y] pairs for export."""
    return {
        kp.name: [round(float(kp.x), 6), round(float(kp.y), 6)]
        for kp in keypoints
        if kp.name != HIP_CENTER_NAME
    }


def normalize_to_hip_center(
    keypoints: list[Keypoint],
) -> dict[str, dict[str, float]]:
    """Hip-centered positions with estimated z (delegates to 3D model)."""
    from stablewalk.pose.skeleton_3d import reconstruct_skeleton_3d

    skel = reconstruct_skeleton_3d(keypoints, scale=1.0)
    return {name: j.to_dict() for name, j in skel.joints.items()}


def build_skeleton_3d_ready(
    keypoints: list[Keypoint],
    *,
    include_image_coords: bool = True,
    uniform_scale: float | None = None,
) -> dict[str, Any]:
    """
    Structured skeleton for 3D visualization and export.

    Includes hip-centered (x, y, z) joints and optional raw image coords.
    """
    from stablewalk.pose.skeleton_3d import reconstruct_skeleton_3d

    center = hip_center(keypoints)
    cx, cy = center if center else (0.5, 0.5)
    skel = reconstruct_skeleton_3d(keypoints, scale=uniform_scale)

    result: dict[str, Any] = {
        **skel.to_export_dict(),
        "hip_center": {"x": cx, "y": cy},
    }

    if include_image_coords:
        result["image"] = {
            kp.name: {
                "x": float(kp.x),
                "y": float(kp.y),
                "visibility": float(kp.visibility),
            }
            for kp in keypoints
        }
    return result
