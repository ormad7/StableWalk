"""Regression: skeleton joint click opens / updates the Overview 3D path graph."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from stablewalk.ui.dof_selection import (
    DofSelectionState,
    normalize_gui_dof_id,
)
from stablewalk.ui.tk.app import StableWalkGUI
from stablewalk.ui.tk.dashboard_overview_panes import layout_panels_for_view_mode
from stablewalk.ui.tk.dashboard_overview_view_mode import (
    VIEW_MODE_OVERLAY,
    VIEW_MODE_RECONSTRUCTION_FULL,
    VIEW_MODE_SIDE_BY_SIDE,
    VIEW_MODE_SKELETON_ONLY,
    VIEW_MODE_VIDEO_ONLY,
)


SUPPORTED_JOINTS = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_shoulder",
    "right_shoulder",
)


def test_normalize_gui_dof_id_canonicalizes_labels() -> None:
    assert normalize_gui_dof_id("Right Hip") == "right_hip"
    assert normalize_gui_dof_id("RIGHT_HIP") == "right_hip"
    assert normalize_gui_dof_id("right hip") == "right_hip"
    assert normalize_gui_dof_id("right-hip") == "right_hip"
    assert normalize_gui_dof_id("not_a_joint") is None


def test_supported_joints_normalize() -> None:
    for joint_id in SUPPORTED_JOINTS:
        assert normalize_gui_dof_id(joint_id) == joint_id
        spaced = joint_id.replace("_", " ").title()
        assert normalize_gui_dof_id(spaced) == joint_id


def _mock_panel(name: str, grids: dict) -> MagicMock:
    widget = MagicMock(name=name)
    widget.grid = MagicMock(
        side_effect=lambda **kw: grids.setdefault(name, []).append(dict(kw))
    )
    widget.grid_remove = MagicMock()
    return widget


def test_side_by_side_places_3d_path_in_column_2() -> None:
    grids: dict = {}
    gui = SimpleNamespace(
        _section_visual=MagicMock(),
        _primary_viz_content_row=0,
        _overview_summary_row=1,
        # Even if previously hidden, Side-by-Side always maps the path column.
        _overview_traj_dock_visible=False,
        video_frame=_mock_panel("video", grids),
        skel_frame=_mock_panel("skel", grids),
        sidebar=_mock_panel("sidebar", grids),
        overview_traj_panel=_mock_panel("traj", grids),
        _schedule_overview_media_refit=None,
        _refresh_overview_trajectory_dock=MagicMock(),
    )
    gui._section_visual.columnconfigure = MagicMock()
    gui._section_visual.rowconfigure = MagicMock()

    layout_panels_for_view_mode(gui, VIEW_MODE_SIDE_BY_SIDE)

    assert gui._overview_traj_dock_visible is True
    assert grids["traj"][-1]["row"] == 0
    assert grids["traj"][-1]["column"] == 2
    assert grids["video"][-1]["column"] == 0
    assert grids["skel"][-1]["column"] == 1


def test_three_panel_layout_grids_children_not_paned() -> None:
    from stablewalk.ui.tk.dashboard_overview_panes import apply_overview_three_panel_layout

    grids: dict = {}
    gui = SimpleNamespace(
        _section_visual=MagicMock(),
        _primary_viz_content_row=0,
        video_frame=_mock_panel("video", grids),
        skel_frame=_mock_panel("skel", grids),
        overview_traj_panel=_mock_panel("traj", grids),
        sidebar=_mock_panel("sidebar", grids),
        _overview_viz_split=None,
    )
    gui._section_visual.columnconfigure = MagicMock()
    gui._section_visual.rowconfigure = MagicMock()

    assert apply_overview_three_panel_layout(gui) is True
    assert grids["video"][-1]["column"] == 0
    assert grids["skel"][-1]["column"] == 1
    assert grids["traj"][-1]["column"] == 2
    assert grids["video"][-1]["sticky"] == "nsew"


def test_three_panel_layout_does_not_regrid_when_already_mounted() -> None:
    from stablewalk.ui.tk.dashboard_overview_panes import apply_overview_three_panel_layout

    section = MagicMock()
    grids: dict = {}
    video = _mock_panel("video", grids)
    skel = _mock_panel("skel", grids)
    traj = _mock_panel("traj", grids)
    for widget, col in ((video, 0), (skel, 1), (traj, 2)):
        widget.winfo_manager = MagicMock(return_value="grid")
        widget.grid_info = MagicMock(return_value={"column": col, "in": section})
        widget.master = section
    gui = SimpleNamespace(
        _section_visual=section,
        _primary_viz_content_row=0,
        video_frame=video,
        skel_frame=skel,
        overview_traj_panel=traj,
        playing=True,
    )
    assert apply_overview_three_panel_layout(gui) is True
    video.grid.assert_not_called()
    skel.grid.assert_not_called()
    traj.grid.assert_not_called()


def test_playback_layout_does_not_remount_mounted_panels() -> None:
    from stablewalk.ui.tk.dashboard_overview_panes import layout_panels_for_view_mode

    section = MagicMock()
    grids: dict = {}
    video = _mock_panel("video", grids)
    skel = _mock_panel("skel", grids)
    traj = _mock_panel("traj", grids)
    for widget, col in ((video, 0), (skel, 1), (traj, 2)):
        widget.winfo_manager = MagicMock(return_value="grid")
        widget.grid_info = MagicMock(return_value={"column": col, "in": section})
        widget.master = section
    gui = SimpleNamespace(
        _section_visual=section,
        _primary_viz_content_row=0,
        video_frame=video,
        skel_frame=skel,
        overview_traj_panel=traj,
        sidebar=_mock_panel("sidebar", grids),
        playing=True,
        _overview_focus_mode=VIEW_MODE_VIDEO_ONLY,
        _refresh_overview_trajectory_dock=MagicMock(),
        _schedule_overview_media_refit=MagicMock(),
    )
    layout_panels_for_view_mode(gui, VIEW_MODE_VIDEO_ONLY)
    video.grid.assert_not_called()
    gui._refresh_overview_trajectory_dock.assert_not_called()
    gui._schedule_overview_media_refit.assert_not_called()


def test_resize_reflow_does_not_relayout_panels_while_playing() -> None:
    from stablewalk.ui.tk.overview_responsive import reflow_overview_viz_panels

    gui = SimpleNamespace(playing=True, overview_traj_panel=MagicMock())
    assert reflow_overview_viz_panels(gui, width=1600) == "three_col"
    gui.overview_traj_panel.rowconfigure.assert_not_called()


def test_focused_modes_hide_unused_viewers() -> None:
    """Video / Overlay / Skeleton focus one viewer; Split keeps all three."""
    for mode, shown, hidden in (
        (VIEW_MODE_VIDEO_ONLY, "video", ("skel", "traj")),
        (VIEW_MODE_OVERLAY, "video", ("skel", "traj")),
        (VIEW_MODE_SKELETON_ONLY, "skel", ("video", "traj")),
    ):
        grids: dict = {}
        gui = SimpleNamespace(
            _section_visual=MagicMock(),
            _primary_viz_content_row=0,
            _overview_summary_row=1,
            _overview_traj_dock_visible=True,
            video_frame=_mock_panel("video", grids),
            skel_frame=_mock_panel("skel", grids),
            sidebar=_mock_panel("sidebar", grids),
            overview_traj_panel=_mock_panel("traj", grids),
            _overview_viz_split=None,
            _schedule_overview_media_refit=None,
            _refresh_overview_trajectory_dock=MagicMock(),
        )
        gui._section_visual.columnconfigure = MagicMock()
        gui._section_visual.rowconfigure = MagicMock()

        layout_panels_for_view_mode(gui, mode)

        assert grids[shown][-1]["columnspan"] == 3
        for name in hidden:
            getattr(gui, {"skel": "skel_frame", "traj": "overview_traj_panel", "video": "video_frame"}[name]).grid_remove.assert_called()


def test_reconstruction_full_keeps_path_in_holder() -> None:
    """3D Full must map the path panel inside its build-time holder, not reparent it."""
    import tkinter as tk
    from tkinter import ttk

    from stablewalk.ui.tk.analysis_split import create_h_split, split_add

    try:
        root = tk.Tk()
    except tk.TclError:  # pragma: no cover - headless CI without display
        return
    try:
        root.geometry("900x500+40+40")
        section = tk.Frame(root, width=900, height=500)
        section.grid(row=0, column=0, sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        section.columnconfigure(0, weight=1)
        section.rowconfigure(0, weight=1)
        paned = create_h_split(section)
        pane_v = tk.Frame(paned)
        pane_s = tk.Frame(paned)
        pane_t = tk.Frame(paned)
        pane_t.rowconfigure(0, weight=1)
        pane_t.columnconfigure(0, weight=1)
        video = ttk.LabelFrame(pane_v, text="Video")
        skel = ttk.LabelFrame(pane_s, text="Skeleton")
        traj = ttk.LabelFrame(pane_t, text="3D Path")
        video.grid(row=0, column=0, sticky="nsew")
        skel.grid(row=0, column=0, sticky="nsew")
        traj.grid(row=0, column=0, sticky="nsew")
        split_add(paned, pane_v)
        split_add(paned, pane_s)
        split_add(paned, pane_t)
        paned.grid(row=0, column=0, columnspan=3, sticky="nsew")
        gui = SimpleNamespace(
            root=None,
            _section_visual=section,
            _primary_viz_content_row=0,
            video_frame=video,
            skel_frame=skel,
            sidebar=None,
            overview_traj_panel=traj,
            _overview_viz_split=paned,
            _overview_pane_path=pane_t,
            _overview_pane_video=pane_v,
            _overview_pane_skeleton=pane_s,
            _overview_phase_strip=None,
            _foot_clearance_detail_host=None,
            playing=False,
            _refresh_overview_trajectory_dock=MagicMock(),
            _schedule_overview_media_refit=MagicMock(),
        )
        root.update()
        layout_panels_for_view_mode(gui, VIEW_MODE_RECONSTRUCTION_FULL)
        root.update()
        assert gui._overview_traj_dock_visible is True
        assert str(traj.winfo_manager()) == "grid"
        assert str(traj.winfo_parent()) == str(pane_t)
        assert len(paned.panes()) == 1
        assert str(paned.winfo_manager()) == "grid"
        assert bool(traj.winfo_ismapped())
        assert not bool(video.winfo_ismapped())
        assert not bool(skel.winfo_ismapped())
        gui._refresh_overview_trajectory_dock.assert_called()
    finally:
        root.destroy()


def test_side_by_side_keeps_all_three_panels() -> None:
    grids: dict = {}
    gui = SimpleNamespace(
        _section_visual=MagicMock(),
        _primary_viz_content_row=0,
        _overview_summary_row=1,
        _overview_traj_dock_visible=True,
        video_frame=_mock_panel("video", grids),
        skel_frame=_mock_panel("skel", grids),
        sidebar=_mock_panel("sidebar", grids),
        overview_traj_panel=_mock_panel("traj", grids),
        _overview_viz_split=None,
        _schedule_overview_media_refit=None,
        _refresh_overview_trajectory_dock=MagicMock(),
    )
    gui._section_visual.columnconfigure = MagicMock()
    gui._section_visual.rowconfigure = MagicMock()

    layout_panels_for_view_mode(gui, VIEW_MODE_SIDE_BY_SIDE)

    assert grids["video"][-1]["column"] == 0
    assert grids["skel"][-1]["column"] == 1
    assert grids["traj"][-1]["column"] == 2


def test_restore_playback_position_does_not_trip_slider_pause() -> None:
    """Layout restore must set frame_var under the sync guard (no play pause)."""
    from stablewalk.ui.tk.dashboard_overview_view_mode import _restore_playback_position

    calls: list[bool] = []

    class _Var:
        def __init__(self) -> None:
            self._v = 0

        def get(self) -> int:
            return self._v

        def set(self, value: int) -> None:
            self._v = int(value)
            # Mimic Scale command: pause if unguarded.
            if not getattr(gui, "_sync_all_guard", False):
                gui.playing = False
            calls.append(bool(getattr(gui, "_sync_all_guard", False)))

    player = SimpleNamespace(
        frame_count=100,
        state=SimpleNamespace(frame_float=17.4, frame_index=17),
        go_to=MagicMock(
            side_effect=lambda pos: setattr(
                player.state, "frame_float", float(pos)
            )
            or setattr(player.state, "frame_index", int(pos)),
        ),
    )
    gui = SimpleNamespace(
        skeleton_player=player,
        _playback_pos=0.0,
        current_pos=0,
        frame_var=_Var(),
        playing=True,
        _sync_all_guard=False,
        _sync_transport_labels=MagicMock(),
    )
    _restore_playback_position(gui, 17.4)
    assert gui.playing is True
    assert calls and all(calls)
    assert gui.frame_var.get() == 17


def test_focus_while_playing_skips_heavy_panel_remount(monkeypatch) -> None:
    """Playing + Side-by-Side: refresh path, but do not remount Overview layout."""
    app = MagicMock()
    app.selection = DofSelectionState()
    app.selection.select_only("right_hip")
    app._notify_dof_selection_changed = MagicMock()
    app.show_joint_3d_panel = MagicMock()
    app.update_joint_3d_graph = MagicMock()
    app._schedule_dof_traj_reflow = MagicMock()
    app._ensure_playback_continues_after_pick = MagicMock()
    app._expand_overview_joint_dof_analysis = MagicMock()
    app._update_overview_playback_hud = MagicMock()
    app._refresh_overview_joint_information = MagicMock()
    app._overview_traj_dock_visible = True
    app._apply_overview_joint_motion_expanded = MagicMock()
    app.skeleton_player = SimpleNamespace(state=SimpleNamespace(frame_float=12.0))
    app.status = MagicMock()
    app.playing = True

    monkeypatch.setattr(
        "stablewalk.ui.tk.dashboard_notebook.select_dashboard_tab",
        lambda *_a, **_k: None,
    )
    apply_mode = MagicMock()
    monkeypatch.setattr(
        "stablewalk.ui.tk.dashboard_overview_view_mode.apply_overview_view_mode",
        apply_mode,
    )
    monkeypatch.setattr(
        "stablewalk.ui.tk.dashboard_overview_view_mode.current_overview_view_mode",
        lambda *_a, **_k: VIEW_MODE_SIDE_BY_SIDE,
    )

    StableWalkGUI._focus_joint_trajectory_from_skeleton(app, "right_knee")
    assert app.selection.selected == {"right_knee"}
    app._notify_dof_selection_changed.assert_called_with(lightweight=True)
    # Path stays visible and refreshes; heavy view-mode remount is skipped.
    app.show_joint_3d_panel.assert_called_once()
    app.update_joint_3d_graph.assert_called_with("right_knee")
    apply_mode.assert_not_called()
    app._ensure_playback_continues_after_pick.assert_called_with(was_playing=True)


def test_focus_joint_calls_show_and_update_graph(monkeypatch) -> None:
    app = MagicMock()
    app.selection = DofSelectionState()
    app._notify_dof_selection_changed = MagicMock()
    app.show_joint_3d_panel = MagicMock()
    app.update_joint_3d_graph = MagicMock()
    app._schedule_dof_traj_reflow = MagicMock()
    app._ensure_playback_continues_after_pick = MagicMock()
    app._expand_overview_joint_dof_analysis = MagicMock()
    app._overview_traj_dock_visible = False
    app._overview_joint_motion_expanded = True
    app._apply_overview_joint_motion_expanded = MagicMock()
    app._analysis_motion_recording = MagicMock(
        return_value=SimpleNamespace(frame_count=120)
    )
    app.skeleton_player = SimpleNamespace(state=SimpleNamespace(frame_float=12.0))
    app.status = MagicMock()
    app.playing = False

    monkeypatch.setattr(
        "stablewalk.ui.tk.dashboard_notebook.select_dashboard_tab",
        lambda *_a, **_k: None,
    )
    apply_mode = MagicMock()
    monkeypatch.setattr(
        "stablewalk.ui.tk.dashboard_overview_view_mode.apply_overview_view_mode",
        apply_mode,
    )
    monkeypatch.setattr(
        "stablewalk.ui.tk.dashboard_overview_view_mode.current_overview_view_mode",
        lambda *_a, **_k: VIEW_MODE_SKELETON_ONLY,
    )

    StableWalkGUI._focus_joint_trajectory_from_skeleton(
        app, "Right Hip", compare=False
    )
    assert app.selection.selected == {"right_hip"}
    assert app.selection.last_selected == "right_hip"
    assert app._overview_traj_dock_visible is True
    app._apply_overview_joint_motion_expanded.assert_called_with(False)
    apply_mode.assert_called()
    assert apply_mode.call_args.args[1] == VIEW_MODE_SIDE_BY_SIDE
    app.show_joint_3d_panel.assert_called_once()
    app.update_joint_3d_graph.assert_called_with("right_hip")
    status_text = app.status.configure.call_args.kwargs.get("text", "")
    assert "DOF analysis: Right Hip" in status_text
    assert "3D path" in status_text
    app._expand_overview_joint_dof_analysis.assert_called_with("right_hip")


@pytest.mark.parametrize(
    "label,item_id",
    [
        ("Right Hip", "right_hip"),
        ("Right Knee", "right_knee"),
        ("Right Ankle", "right_ankle"),
        ("Left Hip", "left_hip"),
        ("Left Knee", "left_knee"),
    ],
)
def test_focus_requested_joints_one_click(monkeypatch, label: str, item_id: str) -> None:
    """User-requested joints: one click selects + updates Overview 3D path."""
    app = MagicMock()
    app.selection = DofSelectionState()
    app._notify_dof_selection_changed = MagicMock()
    app.show_joint_3d_panel = MagicMock()
    app.update_joint_3d_graph = MagicMock()
    app._schedule_dof_traj_reflow = MagicMock()
    app._ensure_playback_continues_after_pick = MagicMock()
    app._expand_overview_joint_dof_analysis = MagicMock()
    app._overview_traj_dock_visible = False
    app._apply_overview_joint_motion_expanded = MagicMock()
    app._analysis_motion_recording = MagicMock(
        return_value=SimpleNamespace(frame_count=60)
    )
    app.skeleton_player = SimpleNamespace(state=SimpleNamespace(frame_float=5.0))
    app.status = MagicMock()
    app.playing = False
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
        lambda *_a, **_k: VIEW_MODE_SKELETON_ONLY,
    )

    StableWalkGUI._focus_joint_trajectory_from_skeleton(app, label)
    assert app.selection.selected == {item_id}
    app.update_joint_3d_graph.assert_called_with(item_id)
    app._apply_overview_joint_motion_expanded.assert_called_with(False)
    title = StableWalkGUI._overview_joint_inspect_title(
        SimpleNamespace(
            selection=app.selection,
            _active_demo_gait=None,
        )
    )
    assert label in title
    assert "3D Path" in title


def test_focus_switches_joint_updates_graph(monkeypatch) -> None:
    app = MagicMock()
    app.selection = DofSelectionState()
    app._notify_dof_selection_changed = MagicMock()
    app.show_joint_3d_panel = MagicMock()
    app.update_joint_3d_graph = MagicMock()
    app._schedule_dof_traj_reflow = MagicMock()
    app._ensure_playback_continues_after_pick = MagicMock()
    app._expand_overview_joint_dof_analysis = MagicMock()
    app._apply_overview_joint_motion_expanded = MagicMock()
    app._analysis_motion_recording = MagicMock(
        return_value=SimpleNamespace(frame_count=80)
    )
    app.skeleton_player = SimpleNamespace(state=SimpleNamespace(frame_float=0.0))
    app.status = MagicMock()
    app.playing = False
    app._overview_traj_dock_visible = False
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
        lambda *_a, **_k: VIEW_MODE_SIDE_BY_SIDE,
    )

    StableWalkGUI._focus_joint_trajectory_from_skeleton(app, "right_hip")
    StableWalkGUI._focus_joint_trajectory_from_skeleton(app, "right_knee")
    assert app.selection.selected == {"right_knee"}
    assert app.update_joint_3d_graph.call_args_list[-1].args[0] == "right_knee"
    title = StableWalkGUI._overview_joint_inspect_title(
        SimpleNamespace(
            selection=app.selection,
            _active_demo_gait=SimpleNamespace(display_name="Normal", button_label="Normal"),
        )
    )
    assert "Right Knee" in title
    assert "3D Path" in title
    # Joint Graphs must stay collapsed across successive clicks.
    assert app._apply_overview_joint_motion_expanded.call_count >= 2
    for call in app._apply_overview_joint_motion_expanded.call_args_list:
        assert call.args[0] is False


def test_normal_gait_right_hip_has_xyz_trajectory() -> None:
    from stablewalk import config
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.io.pose_loader import load_pose_sequence
    from stablewalk.pose.enrichment import enrich_pose_sequence
    from stablewalk.ui.dof_selection import anchor_joint_for_item
    from stablewalk.ui.viewers.dof_trajectory_3d import _joint_path_with_times

    poses = config.POSES_DIR / "normal_gait_poses.json"
    if not poses.is_file():
        pytest.skip("normal_gait_poses.json not available")

    sequence = load_pose_sequence(poses)
    enrich_pose_sequence(sequence)
    recording = pose_sequence_to_gait_motion(sequence)
    assert recording is not None
    assert recording.frame_count > 0

    joint_id = anchor_joint_for_item("right_hip")
    assert joint_id == "right_hip"
    path = _joint_path_with_times(
        recording,
        joint_id,
        float(recording.frame_count - 1),
        coord_mode="ROOT-RELATIVE",
        motion_series=None,
    )
    assert len(path) > 10
    xs = [p[0].x for p in path]
    ys = [p[0].y for p in path]
    zs = [p[0].z for p in path]
    assert len(xs) == len(ys) == len(zs) == len(path)


def test_overview_title_contains_right_hip_after_select() -> None:
    app = SimpleNamespace(
        selection=DofSelectionState(),
        _active_demo_gait=SimpleNamespace(display_name="Normal", button_label="Normal"),
    )
    app.selection.select_only("right_hip")
    title = StableWalkGUI._overview_joint_inspect_title(app)
    assert "Right Hip" in title
    assert "Normal" in title
