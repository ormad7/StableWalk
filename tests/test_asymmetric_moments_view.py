"""Asymmetric-moment cards and 3D highlight helpers (no second renderer)."""

from __future__ import annotations

import inspect

from stablewalk.analysis.gait_asymmetry_moments import (
    AsymmetryContribution,
    AsymmetryMoment,
    contralateral_joint_id,
    difference_joint_pairs,
)
from stablewalk.ui.asymmetric_moments_view import (
    ASYMMETRY_HIGHLIGHT_BGR,
    ASYMMETRY_HIGHLIGHT_COLOR,
    display_joint_id,
    moment_card_payload,
    moment_difference_pairs,
    moment_highlight_joints,
    moment_measurement_labels,
    moment_measurement_lines,
)
from stablewalk.ui.theme import DANGER
from stablewalk.ui.viewers.gait_skeleton_renderer import draw_gait_skeleton


def _moment() -> AsymmetryMoment:
    contrib = AsymmetryContribution(
        measurement="knee_angle",
        joints=("right_knee", "left_knee"),
        difference=18.4,
        units="deg",
        score=62.0,
        left_value=38.2,
        right_value=56.6,
        phase_pct=90.0,
    )
    return AsymmetryMoment(
        frame_index=120,
        time_s=2.40,
        match_time_s=2.90,
        involved_joints=["right_knee", "left_knee"],
        contributing_measurements=[contrib],
        severity="moderate",
        score=64.0,
        confidence=0.9,
        explanation="At 2.40 s (frame 120), knee flexion differed by 18.4°.",
    )


def test_contralateral_and_difference_pairs_are_right_vs_left() -> None:
    assert contralateral_joint_id("right_knee") == "left_knee"
    assert contralateral_joint_id("pelvis") == "pelvis"
    pairs = difference_joint_pairs(_moment())
    assert pairs == [("right_knee", "left_knee")]


def test_display_joint_ids_follow_skeleton_mode() -> None:
    assert display_joint_id("pelvis", "gait") == "pelvis"
    assert display_joint_id("pelvis", "gait_16") == "pelvis_center"
    assert display_joint_id("right_knee", "gait_16") == "right_knee"


def test_card_payload_has_required_fields() -> None:
    payload = moment_card_payload(_moment(), 0)
    assert "2.40 s" in payload["timestamp_label"]
    assert "frame 120" in payload["timestamp_label"]
    assert payload["score_label"] == "64"
    assert payload["confidence_label"] == "0.90"
    assert "knee" in payload["joints_label"]
    assert "38.2" in payload["values_label"]
    assert "56.6" in payload["values_label"]
    assert "18.4" in payload["explanation"]
    assert payload["frame_index"] == 120
    lines = moment_measurement_lines(_moment())
    assert lines
    assert "knee" in lines[0].lower()
    assert "L 38.2" in lines[0]


def test_highlight_joints_are_red_palette_not_gold() -> None:
    assert ASYMMETRY_HIGHLIGHT_COLOR == DANGER
    assert ASYMMETRY_HIGHLIGHT_BGR == (85, 85, 217)
    joints = moment_highlight_joints(_moment(), "gait")
    assert "right_knee" in joints
    assert "left_knee" in joints
    # Full limb wire, not a single-joint trajectory.
    assert "right_hip" in joints
    assert "right_ankle" in joints
    labels = moment_measurement_labels(_moment(), "gait")
    assert "right_knee" in labels
    pairs = moment_difference_pairs(_moment(), "gait_16")
    assert pairs == [("right_knee", "left_knee")]


def test_moments_section_is_mounted_on_the_skeleton_3d_view() -> None:
    from pathlib import Path

    layout = Path("stablewalk/ui/tk/dashboard_layout.py").read_text(encoding="utf-8")
    marker = "build_asymmetric_moments_section("
    assert marker in layout
    chunk = layout.split(marker, 1)[1][:280]
    assert "skel_frame" in chunk
    assert "overview_traj_panel" not in chunk


def test_existing_3d_renderer_gains_moment_kwargs_without_a_new_draw_function() -> None:
    params = inspect.signature(draw_gait_skeleton).parameters
    for name in (
        "highlight_joints",
        "highlight_color",
        "dim_unhighlighted",
        "difference_match_snapshot",
        "difference_joint_pairs",
        "difference_arrow_color",
        "motion_arrows",
    ):
        assert name in params
    # Default DOF highlight still dims unselected joints.
    assert params["dim_unhighlighted"].default is True
    assert params["highlight_color"].default is None
