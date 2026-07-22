"""Overview interactive joint inspection (skeleton click → 3D path)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from stablewalk.ui.dof_selection import DofSelectionState
from stablewalk.ui.theme import EMPTY_OVERVIEW_JOINT_INSPECT
from stablewalk.ui.tk.app import StableWalkGUI


def test_empty_overview_joint_inspect_copy() -> None:
    assert "Click a joint" in EMPTY_OVERVIEW_JOINT_INSPECT
    assert "3D trajectory" in EMPTY_OVERVIEW_JOINT_INSPECT


def test_pick_event_is_compare_detects_control_key() -> None:
    ctrl = SimpleNamespace(mouseevent=SimpleNamespace(key="control", guiEvent=None))
    plain = SimpleNamespace(mouseevent=SimpleNamespace(key=None, guiEvent=None))
    assert StableWalkGUI._pick_event_is_compare(ctrl) is True
    assert StableWalkGUI._pick_event_is_compare(plain) is False


def test_pick_event_is_compare_detects_tk_control_mask() -> None:
    gui = SimpleNamespace(state=0x4)
    event = SimpleNamespace(mouseevent=SimpleNamespace(key=None, guiEvent=gui))
    assert StableWalkGUI._pick_event_is_compare(event) is True


def test_overview_title_single_and_compare() -> None:
    app = SimpleNamespace(
        selection=DofSelectionState(),
        _active_demo_gait=SimpleNamespace(button_label="Normal"),
    )
    app.selection.select_only("right_hip")
    title = StableWalkGUI._overview_joint_inspect_title(app)
    assert "Normal" in title
    assert "Right Hip" in title
    assert "3D Path" in title

    app.selection.activate_item("left_knee")
    title2 = StableWalkGUI._overview_joint_inspect_title(app)
    assert "vs" in title2
    assert "Left Knee" in title2 or "Right Hip" in title2


def test_focus_joint_select_only_then_compare(monkeypatch) -> None:
    """Plain click replaces; compare=True adds without clearing."""
    app = MagicMock()
    app.selection = DofSelectionState()
    app._notify_dof_selection_changed = MagicMock()
    app.show_joint_3d_panel = MagicMock()
    app.update_joint_3d_graph = MagicMock()
    app._schedule_dof_traj_reflow = MagicMock()
    app._ensure_playback_continues_after_pick = MagicMock()
    app._analysis_motion_recording = MagicMock(
        return_value=SimpleNamespace(frame_count=40)
    )
    app.skeleton_player = SimpleNamespace(state=SimpleNamespace(frame_float=3.0))
    app.status = MagicMock()
    app.playing = False
    app._overview_traj_dock_visible = False
    app._apply_overview_joint_motion_expanded = MagicMock()

    monkeypatch.setattr(
        "stablewalk.ui.tk.dashboard_notebook.select_dashboard_tab",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "stablewalk.ui.tk.dashboard_overview_view_mode.apply_overview_view_mode",
        MagicMock(),
    )
    monkeypatch.setattr(
        "stablewalk.ui.tk.dashboard_overview_view_mode.current_overview_view_mode",
        lambda *_a, **_k: "side_by_side",
    )

    StableWalkGUI._focus_joint_trajectory_from_skeleton(app, "right_knee", compare=False)
    assert app.selection.selected == {"right_knee"}
    app.show_joint_3d_panel.assert_called()
    app.update_joint_3d_graph.assert_called_with("right_knee")

    StableWalkGUI._focus_joint_trajectory_from_skeleton(app, "left_hip", compare=False)
    assert app.selection.selected == {"left_hip"}

    StableWalkGUI._focus_joint_trajectory_from_skeleton(app, "right_ankle", compare=True)
    assert app.selection.selected == {"left_hip", "right_ankle"}
