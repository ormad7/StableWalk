"""Joint angle helpers — 3D depth must contribute for frontal gait ROM."""

from __future__ import annotations

import math

from stablewalk.models.pose_data import Keypoint
from stablewalk.pose.kinematics import angle_at_joint


def _kp(name: str, x: float, y: float, z: float = 0.0) -> Keypoint:
    return Keypoint(name=name, x=x, y=y, z=z, visibility=1.0)


def test_frontal_flexion_needs_depth_for_knee_rom() -> None:
    """In the image plane the knee looks almost straight; depth shows flexion."""
    hip = _kp("hip", 0.0, 0.0, 0.0)
    knee = _kp("knee", 0.0, 0.4, 0.0)
    # Distal ankle shifted mostly in z (toward camera) — classic frontal walk.
    ankle_flexed = _kp("ankle", 0.02, 0.8, 0.25)
    ankle_ext = _kp("ankle", 0.0, 0.8, 0.0)

    flexed = angle_at_joint(hip, knee, ankle_flexed)
    extended = angle_at_joint(hip, knee, ankle_ext)
    assert flexed is not None and extended is not None
    # Interior angle drops when the shank moves forward in depth.
    assert abs(extended - flexed) > 15.0


def test_pure_2d_still_works_when_z_is_flat() -> None:
    hip = _kp("hip", 0.0, 0.0, 0.0)
    knee = _kp("knee", 0.0, 1.0, 0.0)
    ankle = _kp("ankle", 1.0, 1.0, 0.0)
    ang = angle_at_joint(hip, knee, ankle)
    assert ang is not None
    assert abs(ang - 90.0) < 1e-6


def test_collinear_3d_is_180() -> None:
    a = _kp("a", 0.0, 0.0, 0.0)
    b = _kp("b", 0.0, 1.0, 0.0)
    c = _kp("c", 0.0, 2.0, 0.5)
    # Make c collinear in 3D along y with slight z still on the ray from b through
    # a synthetic distal that stays nearly 180 when nearly aligned.
    c = _kp("c", 0.0, 2.0, 0.0)
    ang = angle_at_joint(a, b, c)
    assert ang is not None
    assert abs(ang - 180.0) < 1e-6
    assert math.isfinite(ang)
