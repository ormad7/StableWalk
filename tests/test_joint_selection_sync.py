"""Selected-joint panel sync and highlight focus."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from stablewalk.ui.dof_selection import DofSelectionState
from stablewalk.ui.tk.app import StableWalkGUI


def test_resolve_highlight_uses_selected_local_kinematic_chains() -> None:
    app = SimpleNamespace(
        highlight_dof=SimpleNamespace(get=lambda: True),
        selection=DofSelectionState(),
    )
    app.selection.select_only("right_knee")
    joints = StableWalkGUI._resolve_highlight_joints(app)
    assert joints == {"right_hip", "right_knee", "right_ankle"}

    app.selection.activate_item("left_hip")
    joints2 = StableWalkGUI._resolve_highlight_joints(app)
    assert {"right_hip", "right_knee", "right_ankle", "left_hip"} <= joints2


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


def test_checkbox_joint_change_uses_lightweight_sync_during_playback() -> None:
    app = SimpleNamespace(
        _dof_list_syncing=False,
        _dof_checkbox_vars={"right_knee": SimpleNamespace(get=lambda: True)},
        selection=DofSelectionState(),
        playing=True,
        _notify_dof_selection_changed=MagicMock(),
    )

    StableWalkGUI._on_dof_checkbox_changed(app, "right_knee")

    app._notify_dof_selection_changed.assert_called_once_with(lightweight=True)


def test_lightweight_joint_sync_invalidates_video_overlay_without_rebuilding_motion() -> None:
    app = MagicMock()
    app._session_restoring = True
    app.playing = True
    app.selection = DofSelectionState()
    app.selection.select_only("right_knee")
    app._dof_table_history = SimpleNamespace(
        _last_frame_by_item={},
        clear=MagicMock(),
    )
    app._session_collector = SimpleNamespace(clear=MagicMock())
    app._last_video_list_pos = 5
    app._video_photo_key = ("old",)

    StableWalkGUI._notify_dof_selection_changed(app, lightweight=True)

    assert app._last_video_list_pos == -1
    assert app._video_photo_key is None
    app._sync_panels_for_skeleton_pick.assert_called_once_with()
    app.refresh_motion_analysis_tab.assert_not_called()
    app._mark_analysis_panels_dirty.assert_called_once_with(
        "motion", "joint_motion", "knee"
    )
