"""Tests for centralized coordinate conversions and anatomical checks."""

from __future__ import annotations

import pytest

from stablewalk.coordinates.coordinate_map import (
    CANONICAL_VERTICAL_AXIS,
    check_anatomical_ordering,
    mediapipe_to_canonical,
    mediapipe_to_opensim_trc_m,
    mediapipe_to_opensim_trc_mm,
    canonical_to_visualization_oblique,
    estimate_mediapipe_to_meter_scale,
)


class _Pt:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


def test_mediapipe_to_canonical_flips_y_and_centers_hip():
    x, y, z = mediapipe_to_canonical(0.6, 0.7, -0.1, cx=0.5, cy=0.5, joint_name="left_knee", scale=1.0)
    assert x == pytest.approx(0.1)
    assert y == pytest.approx(-0.2)  # below hip in image → negative canonical Y
    assert z == pytest.approx(-0.06 + 0.02)  # layer -0.06, mp z -0.1 → +0.02 depth term


def test_mediapipe_to_opensim_trc_matches_legacy_formula():
    scale = 1.7
    x, y, z = 0.4, 0.8, 0.05
    xm, ym, zm = mediapipe_to_opensim_trc_m(x, y, z, scale)
    assert xm == pytest.approx(x * scale)
    assert ym == pytest.approx((1.0 - y) * scale)
    assert zm == pytest.approx(-z * scale)

    xmm, ymm, zmm = mediapipe_to_opensim_trc_mm(x, y, z, scale)
    assert (xmm, ymm, zmm) == pytest.approx((xm * 1000, ym * 1000, zm * 1000))


def test_estimate_scale_from_vertical_span():
    scale = estimate_mediapipe_to_meter_scale([0.2, 0.9], subject_height_m=1.7)
    assert scale == pytest.approx(1.7 / 0.7)


def test_anatomical_ordering_passes_upright_walking_pose():
    joints = {
        "pelvis": _Pt(0, 0, 0),
        "head": _Pt(0, 0.5, 0),
        "left_knee": _Pt(0.1, -0.25, 0),
        "left_ankle": _Pt(0.1, -0.5, 0),
        "left_heel": _Pt(0.1, -0.52, 0),
        "left_toe": _Pt(0.12, -0.51, 0.05),
    }
    result = check_anatomical_ordering(joints)
    assert result.passed


def test_anatomical_ordering_fails_inverted_y():
    joints = {
        "pelvis": _Pt(0, 0.5, 0),
        "head": _Pt(0, 0, 0),
        "left_knee": _Pt(0, 0.3, 0),
        "left_ankle": _Pt(0, 0.4, 0),
    }
    result = check_anatomical_ordering(joints)
    assert not result.passed
    assert any("head" in f for f in result.failures)


def test_canonical_vertical_axis_is_y():
    assert CANONICAL_VERTICAL_AXIS == "y"


def test_oblique_projection_uses_shear():
    ox, oy = canonical_to_visualization_oblique(0.1, 0.2, 0.5)
    assert ox == pytest.approx(0.1 + 0.22 * 0.5)
    assert oy == pytest.approx(0.2)
