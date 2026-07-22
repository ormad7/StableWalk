"""Tests for General vs Foot analysis mode metric splits."""

from __future__ import annotations

from stablewalk.ui.selected_point_analysis import (
    analysis_mode_for_item,
    is_foot_analysis_point,
    metrics_for_analysis_panel,
    panel_movement_metrics_for_panel,
)
from stablewalk.ui.presentation_demo import generate_presentation_recording


def test_foot_points_use_foot_mode() -> None:
    for item_id in (
        "left_heel",
        "right_toe",
        "left_ankle",
        "right_ankle",
    ):
        assert is_foot_analysis_point(item_id)
        assert analysis_mode_for_item(item_id) == "foot"


def test_hip_uses_general_mode() -> None:
    assert not is_foot_analysis_point("right_hip")
    assert analysis_mode_for_item("right_hip") == "general"


def test_general_panel_includes_derived_not_foot_card() -> None:
    recording = generate_presentation_recording()
    snap = recording.snapshot_at(0)
    assert snap is not None
    from stablewalk.ui.selected_point_analysis import build_selected_point_analysis

    analysis = build_selected_point_analysis(
        "right_hip",
        snap,
        recording,
        0.0,
    )
    panel = metrics_for_analysis_panel(
        "right_hip",
        snap,
        analysis,
        recording=recording,
        end_frame_float=0.0,
    )
    assert panel.mode == "general"
    assert panel.foot_card is None
    assert len(panel.identity) == 3
    assert panel.identity[0][0] == "Selected Point"
    assert len(panel.kinematics) == 5
    assert panel.kinematics[3][0] == "Velocity (m/s)"
    assert panel.kinematics[4][0] == "Acceleration (m/s²)"
    assert len(panel.derived) == 3
    titles = [t for t, _ in panel.derived]
    assert titles == [
        "Path Length",
        "Delta from Start",
        "Vertical Position",
    ]


def test_foot_panel_includes_kinematics_derived_and_foot_card() -> None:
    recording = generate_presentation_recording()
    snap = recording.snapshot_at(0)
    assert snap is not None
    from stablewalk.ui.selected_point_analysis import build_selected_point_analysis

    analysis = build_selected_point_analysis(
        "left_heel",
        snap,
        recording,
        0.0,
    )
    panel = metrics_for_analysis_panel(
        "left_heel",
        snap,
        analysis,
        recording=recording,
        end_frame_float=0.0,
    )
    assert panel.mode == "foot"
    assert panel.foot_card is not None
    assert panel.identity[0][0] == "Selected Point"
    assert len(panel.kinematics) == 5
    assert panel.kinematics[3][0] == "Velocity (m/s)"
    assert panel.kinematics[4][0] == "Acceleration (m/s²)"
    assert len(panel.derived) == 4
    titles = [t for t, _ in panel.derived]
    assert titles == [
        "Current distance from ground",
        "Contact status",
        "Path since playback start",
        "Current vertical position",
    ]
    assert panel.foot_card.ground_note
    assert panel.foot_card.clearance_value_cm.endswith(" cm") or panel.foot_card.clearance_value_cm == "—"


def test_graph_caption_one_line_per_mode() -> None:
    from stablewalk.ui.selected_point_analysis import graph_caption_for_panel
    from stablewalk.ui.theme import DOF_ANALYSIS_GRAPH_CAPTION_GENERAL

    general = graph_caption_for_panel("right_hip")
    foot = graph_caption_for_panel("left_heel")
    assert general == DOF_ANALYSIS_GRAPH_CAPTION_GENERAL
    assert general  # non-empty general caption
    assert foot == ""


def test_foot_point_has_no_general_derived_metrics() -> None:
    recording = generate_presentation_recording()
    snap = recording.snapshot_at(0)
    assert snap is not None
    from stablewalk.ui.selected_point_analysis import build_selected_point_analysis

    analysis = build_selected_point_analysis(
        "right_heel",
        snap,
        recording,
        0.0,
    )
    derived = panel_movement_metrics_for_panel(
        "right_heel",
        snap,
        analysis,
        recording,
        0.0,
    )
    assert derived == ()


def test_position_table_columns_by_point_type() -> None:
    from stablewalk.ui.dof_position_table import (
        DOF_TABLE_COLUMNS,
        DOF_TABLE_GENERAL_COLUMNS,
        table_display_columns_for_item,
    )

    assert table_display_columns_for_item("right_hip") == DOF_TABLE_GENERAL_COLUMNS
    assert table_display_columns_for_item("left_heel") == DOF_TABLE_COLUMNS
    assert "foot_clearance" not in DOF_TABLE_GENERAL_COLUMNS
    assert "angle" not in DOF_TABLE_COLUMNS
