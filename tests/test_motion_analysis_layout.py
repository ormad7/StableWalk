"""Motion Analysis contact/vGRF layout minsizes and figure fill helpers."""

from __future__ import annotations

from types import SimpleNamespace


def test_contact_gait_min_height_keeps_three_plots_readable() -> None:
    from stablewalk.ui.tk.dashboard_layout import (
        CONTACT_GAIT_CHART_MIN_H,
        MOTION_TOP_COLUMN_WEIGHTS,
        _MOTION_GRAPH_ROW_MINSIZE,
        _MOTION_KNEE_PANEL_MIN_W,
        _MOTION_METRICS_PANEL_MIN_W,
        _MOTION_TOP_ROW_MIN_H,
        _MOTION_TRAJ_PANEL_MIN_W,
    )

    assert CONTACT_GAIT_CHART_MIN_H >= 360
    assert _MOTION_TOP_ROW_MIN_H >= 360
    assert _MOTION_GRAPH_ROW_MINSIZE >= 240
    # Contact strip must be tall enough for contact + phase + Estimated Virtual GRF.
    assert CONTACT_GAIT_CHART_MIN_H >= 280
    # Two-column Motion workspace: knee | selected-joint 3D path.
    assert MOTION_TOP_COLUMN_WEIGHTS == (1, 1)
    assert sum(MOTION_TOP_COLUMN_WEIGHTS) == 2
    assert _MOTION_KNEE_PANEL_MIN_W >= 240
    assert _MOTION_TRAJ_PANEL_MIN_W >= 200
    assert _MOTION_METRICS_PANEL_MIN_W >= 180


def test_motion_workspace_graph_is_visual_focus() -> None:
    from stablewalk.ui.tk.dashboard_motion_workspace import (
        _MOTION_WORKSPACE_FIGSIZE,
        _MOTION_WORKSPACE_GRAPH_MIN_H,
    )

    # figsize height ~35% above the previous 4.8 in baseline.
    assert _MOTION_WORKSPACE_FIGSIZE[1] >= 6.4
    assert _MOTION_WORKSPACE_GRAPH_MIN_H >= 360


def test_motion_graph_owns_seventy_five_percent_of_height() -> None:
    from stablewalk.ui.tk.dashboard_motion_workspace import (
        _MOTION_GRAPH_HEIGHT_SHARE,
        _MOTION_PATH_WIDTH_SHARE,
        _MOTION_STACK_BELOW_W,
        _MOTION_STATS_COLLAPSED_H,
        _MOTION_STATS_MAX_H,
    )

    assert _MOTION_GRAPH_HEIGHT_SHARE >= 0.75
    assert _MOTION_STACK_BELOW_W >= 1100
    assert _MOTION_PATH_WIDTH_SHARE <= 0.24
    # Empty status row is shorter than the populated strip so the graph grows.
    assert _MOTION_STATS_COLLAPSED_H < _MOTION_STATS_MAX_H
    assert _MOTION_STATS_COLLAPSED_H <= 28
    assert 96 <= _MOTION_STATS_MAX_H <= 168


def test_motion_chrome_stacks_before_tools_overlap_tabs() -> None:
    from stablewalk.ui.tk.dashboard_motion_workspace import (
        _MOTION_CHROME_STACK_BELOW_W,
        _MOTION_WORKSPACE_GRAPH_MIN_H,
    )

    assert 720 <= _MOTION_CHROME_STACK_BELOW_W <= 1100
    assert _MOTION_WORKSPACE_GRAPH_MIN_H >= 360


def test_live_kinematics_wraps_before_cells_become_cramped() -> None:
    from stablewalk.ui.tk.dashboard_motion_workspace import (
        _motion_stats_column_count,
    )

    assert _motion_stats_column_count(500, 9) == 2
    assert _motion_stats_column_count(760, 9) == 4
    assert _motion_stats_column_count(1100, 9) == 5
    assert _motion_stats_column_count(1400, 9) == 9


def test_reflow_motion_middle_keeps_graph_on_host_when_path_hidden() -> None:
    import tkinter as tk

    from stablewalk.ui.tk.analysis_split import create_h_split
    from stablewalk.ui.tk.dashboard_motion_workspace import _reflow_motion_middle

    try:
        root = tk.Tk()
    except tk.TclError:  # pragma: no cover - headless CI without display
        return
    root.withdraw()
    try:
        host = tk.Frame(root, width=900, height=400)
        host.grid_propagate(False)
        host.grid(row=0, column=0, sticky="nsew")
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)
        paned = create_h_split(host)
        graphs = tk.Frame(host, width=400, height=300)
        path = tk.Frame(host, width=200, height=300)
        graphs.grid(row=0, column=0, sticky="nsew")
        gui = SimpleNamespace(
            _motion_workspace_middle=host,
            _motion_middle_split=paned,
            _motion_graphs_panel=graphs,
            _motion_path_slot=path,
            _motion_middle_mode=None,
            skeleton_player=None,
            selection=None,
        )
        root.update_idletasks()
        _reflow_motion_middle(gui)
        root.update_idletasks()
        assert gui._motion_middle_mode == "graph-only"
        assert str(graphs.grid_info().get("in")) == str(host)
        assert graphs.winfo_manager() == "grid"
        assert int(graphs.winfo_width()) >= 200
        assert int(graphs.winfo_height()) >= 160
        assert not path.winfo_ismapped()
        assert not paned.winfo_ismapped()
    finally:
        root.destroy()


def test_motion_path_only_uses_space_when_session_and_joint_are_ready() -> None:
    from stablewalk.ui.tk.dashboard_motion_workspace import (
        _motion_path_content_available,
    )

    gui = SimpleNamespace(skeleton_player=None, selection=None)
    assert _motion_path_content_available(gui) is False

    gui.skeleton_player = SimpleNamespace(frame_count=120)
    gui.selection = SimpleNamespace(selected=())
    assert _motion_path_content_available(gui) is False

    gui.selection = SimpleNamespace(selected=("right_knee_flexion",))
    assert _motion_path_content_available(gui) is True


def test_graph_row_tracks_seventy_five_percent_of_workspace_height() -> None:
    import tkinter as tk

    from stablewalk.ui.tk.dashboard_motion_workspace import (
        _MOTION_GRAPH_CANVAS_ROW,
        _MOTION_LOWER_COMFORT_PX,
        _MOTION_WORKSPACE_GRAPH_MIN_H,
        _install_motion_graph_height_share,
    )

    try:
        root = tk.Tk()
    except tk.TclError:  # pragma: no cover - headless CI without display
        return
    root.withdraw()
    try:
        host = tk.Frame(root, width=1200, height=1000)
        host.grid_propagate(False)
        host.grid(row=0, column=0)
        toolbar = tk.Frame(host, height=48)
        # Collapsed Live Kinematics status row (no values).
        stats = tk.Frame(host, height=22)
        chips = tk.Frame(host, height=74)
        graphs = tk.Frame(host)
        graph_host = tk.Frame(graphs)
        gui = SimpleNamespace(
            _motion_workspace_toolbar=toolbar,
            _motion_stats_panel=stats,
            _motion_joint_row=chips,
            _motion_graphs_panel=graphs,
            motion_graph_host=graph_host,
        )
        _install_motion_graph_height_share(gui, host)
        root.update_idletasks()
        gui._sync_motion_graph_height_share()
        assert host.grid_rowconfigure(1)["minsize"] == 750

        # Short workspaces leave slack for Compare Joints / live data instead
        # of inflating the graph to the entire leftover height.
        host.configure(height=400)
        root.update_idletasks()
        gui._sync_motion_graph_height_share()
        reserved = (
            int(toolbar.cget("height") or toolbar.winfo_reqheight() or 0)
            + int(stats.cget("height") or stats.winfo_reqheight() or 0)
            + int(chips.cget("height") or chips.winfo_reqheight() or 0)
        )
        available = max(80, 400 - reserved - _MOTION_LOWER_COMFORT_PX)
        graph_min = int(host.grid_rowconfigure(1)["minsize"])
        assert graph_min <= max(160, available - 24)
        assert graph_min >= 160
        assert int(host.grid_rowconfigure(3)["minsize"] or 0) >= 52
        assert int(graphs.grid_rowconfigure(_MOTION_GRAPH_CANVAS_ROW)["minsize"]) < (
            _MOTION_WORKSPACE_GRAPH_MIN_H
        )
        assert int(graph_host.grid_rowconfigure(0)["minsize"]) == int(
            graphs.grid_rowconfigure(_MOTION_GRAPH_CANVAS_ROW)["minsize"]
        )
    finally:
        root.destroy()


def test_live_kinematics_keeps_requested_fields_visible() -> None:
    import tkinter as tk

    from stablewalk.ui.tk.dashboard_motion_workspace import (
        _MOTION_STATS_COLLAPSED_H,
        _MOTION_STATS_FIELDS,
        _MOTION_STATS_MAX_H,
        sync_motion_live_kinematics,
    )
    from stablewalk.ui.theme import ELEVATED

    try:
        root = tk.Tk()
    except tk.TclError:  # pragma: no cover - headless CI without display
        return
    root.withdraw()
    try:
        gui = SimpleNamespace()
        row = tk.Frame(root)
        gui._motion_stats_row = row
        gui._motion_stats_cells = {}
        gui._motion_stats_value_labels = {}
        for key, _title in _MOTION_STATS_FIELDS:
            cell = tk.Frame(row, bg=ELEVATED)
            cell.grid(row=0, column=0)
            cell.grid_remove()
            gui._motion_stats_cells[key] = cell
            gui._motion_stats_value_labels[key] = tk.Label(cell, text="")
        gui._motion_stats_empty_hint = tk.Label(row, text="No live values")
        gui._motion_stats_empty_hint.grid(row=0, column=0)

        panel = tk.Frame(root, height=_MOTION_STATS_COLLAPSED_H)
        panel.grid_propagate(False)
        gui._motion_stats_panel = panel
        gui._motion_workspace_host = tk.Frame(root)
        gui._motion_stats_collapsed = True
        sync_calls: list[int] = []

        def _sync() -> None:
            sync_calls.append(int(panel.cget("height")))

        gui._sync_motion_graph_height_share = _sync

        visible = sync_motion_live_kinematics(
            gui,
            {
                "joint": "Left knee",
                "angle": "12.3°",
                "velocity": "0.42 m/s",
            },
        )
        assert visible == len(_MOTION_STATS_FIELDS)
        assert gui._motion_stats_cells["angle"].winfo_manager() == "grid"
        assert gui._motion_stats_value_labels["x"].cget("text") == "Unavailable"
        assert gui._motion_stats_value_labels["frame"].cget("text") == "Unavailable"
        # Requested cells pack into consecutive columns, leaving no gaps.
        positions = [
            (
                gui._motion_stats_cells[key].grid_info()["row"],
                gui._motion_stats_cells[key].grid_info()["column"],
            )
            for key in ("joint", "angle", "velocity")
        ]
        assert [(int(r), int(c)) for r, c in positions] == [(0, 0), (0, 1), (1, 0)]
        assert gui._motion_stats_empty_hint.winfo_manager() == ""
        assert int(panel.cget("height")) == _MOTION_STATS_MAX_H
        assert gui._motion_stats_collapsed is False
        # Live statistics stay close to the graph without competing for slack.
        assert int(gui._motion_workspace_host.grid_rowconfigure(2)["weight"]) == 0
        assert sync_calls == [_MOTION_STATS_MAX_H]

        assert sync_motion_live_kinematics(gui, {}) == 0
        assert gui._motion_stats_empty_hint.winfo_manager() == "grid"
        assert gui._motion_stats_cells["angle"].winfo_manager() == ""
        assert int(panel.cget("height")) == _MOTION_STATS_COLLAPSED_H
        assert gui._motion_stats_collapsed is True
    finally:
        root.destroy()


def test_fit_figure_to_host_expands_width() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from stablewalk.ui.tk.dashboard_layout import _fit_figure_to_host

    fig = Figure(figsize=(4.0, 2.0), dpi=100)
    drawn: list[bool] = []
    root = SimpleNamespace(playing=False)

    class _Canvas:
        def get_tk_widget(self):
            return SimpleNamespace(
                winfo_width=lambda: 100,
                winfo_height=lambda: 80,
                winfo_toplevel=lambda: root,
                configure=lambda **_kwargs: None,
            )

        def resize(self, event) -> None:
            dpi = float(fig.get_dpi() or 100.0)
            fig.set_size_inches(event.width / dpi, event.height / dpi, forward=True)

        def draw_idle(self) -> None:
            drawn.append(True)

    host = SimpleNamespace(
        update_idletasks=lambda: None,
        winfo_width=lambda: 900,
        winfo_height=lambda: 400,
        winfo_toplevel=lambda: root,
    )
    canvas = _Canvas()
    ok = _fit_figure_to_host(canvas, fig, host=host, min_px=80)  # type: ignore[arg-type]
    assert ok
    w_in, h_in = fig.get_size_inches()
    dpi = float(fig.get_dpi() or 100.0)
    assert w_in * dpi >= 880
    assert h_in * dpi >= 380
    assert drawn


def test_fit_figure_to_host_shrinks_width() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from stablewalk.ui.tk.dashboard_layout import _fit_figure_to_host

    fig = Figure(figsize=(9.2, 6.5), dpi=110)
    root = SimpleNamespace(playing=False)

    class _Canvas:
        def get_tk_widget(self):
            return SimpleNamespace(
                winfo_width=lambda: 1012,
                winfo_height=lambda: 715,
                winfo_toplevel=lambda: root,
                configure=lambda **_kwargs: None,
            )

        def resize(self, event) -> None:
            dpi = float(fig.get_dpi() or 110.0)
            fig.set_size_inches(event.width / dpi, event.height / dpi, forward=True)

        def draw_idle(self) -> None:
            return None

    host = SimpleNamespace(
        update_idletasks=lambda: None,
        winfo_width=lambda: 640,
        winfo_height=lambda: 400,
        winfo_toplevel=lambda: root,
    )
    ok = _fit_figure_to_host(_Canvas(), fig, host=host, min_px=80)  # type: ignore[arg-type]
    assert ok
    w_in, h_in = fig.get_size_inches()
    dpi = float(fig.get_dpi() or 110.0)
    assert w_in * dpi <= 650
    assert h_in * dpi <= 410
