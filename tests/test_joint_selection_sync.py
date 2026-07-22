"""Selected-joint panel sync and highlight focus."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from stablewalk.ui.dof_selection import DofSelectionState
from stablewalk.ui.tk.app import StableWalkGUI


def test_resolve_highlight_uses_selected_anchors_only() -> None:
    app = SimpleNamespace(
        highlight_dof=SimpleNamespace(get=lambda: True),
        selection=DofSelectionState(),
    )
    app.selection.select_only("right_knee")
    # monkey-patch helper used by _resolve_highlight_joints via real method
    joints = StableWalkGUI._resolve_highlight_joints(app)
    assert joints == {"right_knee"}

    app.selection.activate_item("left_hip")
    joints2 = StableWalkGUI._resolve_highlight_joints(app)
    assert joints2 == {"right_knee", "left_hip"}


def test_resolve_highlight_none_when_toggle_off() -> None:
    app = SimpleNamespace(
        highlight_dof=SimpleNamespace(get=lambda: False),
        selection=DofSelectionState(),
    )
    app.selection.select_only("right_ankle")
    assert StableWalkGUI._resolve_highlight_joints(app) is None


def test_sync_panels_for_selected_joint_calls_core_refreshers() -> None:
    app = MagicMock()
    app.selection = DofSelectionState()
    app.selection.select_only("right_hip")
    app._overview_traj_dock_visible = True
    app.sequence = None
    app.lbl_dof_table_hint = None
    app._dof_table_history = SimpleNamespace(rows=[])

    StableWalkGUI._sync_panels_for_selected_joint(app, force_draw=True)

    app._refresh_realtime_analysis.assert_called()
    app._refresh_selected_dof_trajectory_3d.assert_called()
    app._update_interactive_skeleton.assert_called()
    app._refresh_overview_trajectory_dock.assert_called()
