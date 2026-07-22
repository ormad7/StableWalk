"""Tests for stable per-joint color mapping."""

from __future__ import annotations

from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, label_for_item
from stablewalk.ui.joint_colors import (
    JOINT_COLORS,
    item_id_for_joint_label,
    joint_color,
    joint_row_tag,
    tags_for_joint_row,
)


def test_every_gui_joint_has_unique_color() -> None:
    assert len(JOINT_COLORS) == len(GUI_DOF_ITEM_IDS)
    colors = list(JOINT_COLORS.values())
    assert len(set(colors)) == len(colors)


def test_joint_color_is_stable() -> None:
    assert joint_color("left_knee") == joint_color("left_knee")
    assert joint_color("left_knee") != joint_color("right_knee")


def test_item_id_for_joint_label_roundtrip() -> None:
    for item_id in GUI_DOF_ITEM_IDS:
        label = label_for_item(item_id)
        assert item_id_for_joint_label(label) == item_id


def test_joint_row_tags() -> None:
    assert joint_row_tag("left_hip") == "joint_left_hip"
    assert joint_row_tag("left_hip", active=True) == "joint_active_left_hip"
    assert tags_for_joint_row("right_knee") == ("joint_right_knee",)
    assert tags_for_joint_row("right_knee", active=True) == ("joint_active_right_knee",)
