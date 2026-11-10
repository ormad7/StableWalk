"""Joint angle helpers — 3D depth must contribute for frontal gait ROM."""

from __future__ import annotations

import math

from stablewalk.models.pose_data import Keypoint
from stablewalk.pose.kinematics import (
    KNEE_FLEXION_CONVENTION,
    KNEE_FLEXION_UNITS,
    angle_at_joint,
    interior_angle_3d,
    knee_flexion_from_hka,
)


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


def test_knee_flexion_is_180_minus_internal_3d_angle() -> None:
    hip = _kp("hip", 0.0, 0.0, 0.0)
    knee = _kp("knee", 0.0, 1.0, 0.0)
    straight = _kp("ankle", 0.0, 2.0, 0.0)
    flexed = _kp("ankle", 1.0, 1.0, 0.0)

    assert KNEE_FLEXION_UNITS == "deg"
    assert "180" in KNEE_FLEXION_CONVENTION
    assert "deg" in KNEE_FLEXION_CONVENTION.lower()

    interior_ext = interior_angle_3d(hip, knee, straight)
    flex_ext = knee_flexion_from_hka(hip, knee, straight)
    assert interior_ext is not None and flex_ext is not None
    assert abs(interior_ext - 180.0) < 1e-6
    assert abs(flex_ext - 0.0) < 1e-6
    assert abs(flex_ext - (180.0 - interior_ext)) < 1e-9

    interior_90 = interior_angle_3d(hip, knee, flexed)
    flex_90 = knee_flexion_from_hka(hip, knee, flexed)
    assert interior_90 is not None and flex_90 is not None
    assert abs(interior_90 - 90.0) < 1e-6
    assert abs(flex_90 - 90.0) < 1e-6


def test_knee_flexion_uses_3d_depth() -> None:
    hip = _kp("hip", 0.0, 0.0, 0.0)
    knee = _kp("knee", 0.0, 1.0, 0.0)
    ankle = _kp("ankle", 0.0, 1.0, 1.0)
    flex = knee_flexion_from_hka(hip, knee, ankle)
    assert flex is not None
    assert abs(flex - 90.0) < 1e-6


def test_knee_flexion_omits_missing_or_low_confidence() -> None:
    hip = _kp("hip", 0.0, 0.0, 0.0)
    knee = _kp("knee", 0.0, 1.0, 0.0)
    ankle = _kp("ankle", 1.0, 1.0, 0.0)
    low = Keypoint(name="ankle", x=1.0, y=1.0, z=0.0, visibility=0.1)
    assert knee_flexion_from_hka(hip, knee, None) is None
    assert knee_flexion_from_hka(hip, None, ankle) is None
    assert knee_flexion_from_hka(None, knee, ankle) is None
    assert knee_flexion_from_hka(hip, knee, low) is None
    assert interior_angle_3d(hip, knee, low) is None
