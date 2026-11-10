"""Selected-joint accent stays consistent across Overview surfaces."""

from __future__ import annotations

from stablewalk.pose.skeleton import DOF_COLOR
from stablewalk.ui.theme import (
    DOF_TRAJ_DOT_COLOR,
    SELECTED_JOINT_ACCENT,
    SELECTED_JOINT_ACCENT_BGR,
)
from stablewalk.ui.viewers.dof_trajectory_3d import _CURRENT_DOT_COLOR
from stablewalk.ui.viewers.gait_skeleton_renderer import (
    _COLOR_SELECT_JOINT,
    _COLOR_SELECT_RING,
)


def test_overview_joint_accent_is_one_color() -> None:
    assert SELECTED_JOINT_ACCENT == DOF_TRAJ_DOT_COLOR == "#ffd666"
    assert SELECTED_JOINT_ACCENT_BGR == (102, 214, 255)
    assert tuple(DOF_COLOR) == SELECTED_JOINT_ACCENT_BGR
    assert _CURRENT_DOT_COLOR == SELECTED_JOINT_ACCENT
    assert _COLOR_SELECT_JOINT == SELECTED_JOINT_ACCENT
    assert _COLOR_SELECT_RING == SELECTED_JOINT_ACCENT
