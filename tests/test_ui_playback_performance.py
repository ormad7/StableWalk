"""Focused guards for low-cost playback and cached static UI data."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_playback_does_not_reconstruct_unused_panel_skeleton() -> None:
    from stablewalk.ui.tk.app import StableWalkGUI

    frame = SimpleNamespace(
        frame_index=0,
        detected=True,
        keypoints=[object()],
        image_path="unused.png",
    )
    gui = SimpleNamespace(
        sequence=SimpleNamespace(frames=[frame]),
        pose_indices=[0],
        current_pos=0,
        playing=True,
        _last_video_list_pos=0,
        _pb_heavy=False,
        _pb_ui_tick=True,
        _scrubbing=False,
        _playback_sync_immediate=True,
        _panel_tick=0,
        skeleton_player=object(),
        show_robot_panel=SimpleNamespace(get=lambda: False),
        _group_visible=lambda name: name == "skeleton",
        _resolve_highlight_joints=lambda: set(),
        _get_skeleton=MagicMock(),
        _update_interactive_skeleton=MagicMock(),
    )

    StableWalkGUI._show_pose_at(gui, 0.0)

    gui._get_skeleton.assert_not_called()
    gui._update_interactive_skeleton.assert_called_once()


def test_analysis_summary_reuses_same_session_playhead_cache() -> None:
    from stablewalk.ui.tk.app import StableWalkGUI

    cached = object()
    session_key = ("run", "poses.json", 11, 12)
    gui = SimpleNamespace(
        sequence=None,
        pose_indices=[],
        current_pos=0,
        _active_run_name="run",
        _poses_path="poses.json",
        _gait_cycle=SimpleNamespace(),
        _biomech_analysis=SimpleNamespace(),
        _analysis_summary_cache=cached,
        _analysis_summary_session_key=None,
        _analysis_summary_playhead_key=(4, 0.2),
        playing=False,
        _build_analysis_summary=MagicMock(),
        _refresh_pipeline_status_panel=MagicMock(),
    )
    session_key = (
        gui._active_run_name,
        gui._poses_path,
        id(gui._gait_cycle),
        id(gui._biomech_analysis),
    )
    gui._analysis_summary_session_key = session_key

    with patch(
        "stablewalk.ui.tk.dashboard_summary.update_results_summary_panel"
    ) as update:
        StableWalkGUI._update_analysis_summary_panel(
            gui, frame_index=4, timestamp_s=0.2
        )

    gui._build_analysis_summary.assert_not_called()
    update.assert_called_once_with(gui, cached)


def test_hidden_panel_dirty_set_is_not_rewritten_each_play_tick() -> None:
    from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW
    from stablewalk.ui.tk.playback_active_tab import mark_hidden_tab_panels_dirty

    class CountingSet(set):
        def __init__(self) -> None:
            super().__init__()
            self.update_calls = 0

        def update(self, *others) -> None:
            self.update_calls += 1
            super().update(*others)

    dirty = CountingSet()
    gui = SimpleNamespace(playing=True, _analysis_panel_dirty=dirty)
    with patch(
        "stablewalk.ui.tk.dashboard_tab_lifecycle.active_tab_id",
        return_value=TAB_OVERVIEW,
    ):
        mark_hidden_tab_panels_dirty(gui)
        mark_hidden_tab_panels_dirty(gui)

    assert dirty.update_calls == 1


def test_playback_on_nonvisual_tab_updates_transport_only() -> None:
    from stablewalk.ui.tk.app import StableWalkGUI

    gui = SimpleNamespace(
        playing=True,
        skeleton_player=object(),
        _sync_all_guard=False,
        _visible_panel_groups=lambda: frozenset(),
        _sync_transport_labels=MagicMock(),
    )
    with patch(
        "stablewalk.ui.tk.playback_active_tab.mark_hidden_tab_panels_dirty"
    ):
        StableWalkGUI._sync_all_to_frame(gui, force_draw=False)

    gui._sync_transport_labels.assert_called_once()
    assert gui._sync_all_guard is False


def test_motion_tab_reflow_reuses_contact_chart_artists() -> None:
    from stablewalk.ui.tk.dashboard_notebook import reflow_tab_canvases

    motion = object()

    class _Notebook:
        def select(self) -> str:
            return "motion"

        def nametowidget(self, _name: str):
            return motion

    update_contact = MagicMock()
    gui = SimpleNamespace(
        _dashboard_notebook=_Notebook(),
        _tab_overview=object(),
        _tab_motion=motion,
        _tab_biomechanics=object(),
        _overview_traj_dock_visible=False,
        _ensure_motion_tab_charts=MagicMock(),
        _update_chart=MagicMock(),
        _update_contact_gait_chart=update_contact,
        _ensure_motion_traj_chart=MagicMock(),
    )

    reflow_tab_canvases(gui)

    update_contact.assert_called_once_with()
