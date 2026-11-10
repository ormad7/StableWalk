"""Motion Analysis workspace — synchronized multi-graph motion dashboard.

Presentation layout only. Playback and kinematics reuse existing analysis paths.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, GUI_DOF_LABELS
from stablewalk.ui.theme import (
    ELEVATED,
    FONT_KPI_SUPPORTING,
    FONT_KPI_TITLE,
    FONT_UI_SM,
    MUTED,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    TEXT_SECONDARY,
    create_tooltip,
    tk_labelframe_title,
)
from stablewalk.ui.viewers.joint_motion_analysis_chart import (
    MOTION_ANALYSIS_TABS,
    motion_angle_convention,
)

# Graph figure / host floor — tall enough that axis labels stay readable
# after tools + metric tabs; live numerics live in a separate panel below.
_MOTION_WORKSPACE_FIGSIZE = (9.2, 6.8)
_MOTION_WORKSPACE_GRAPH_MIN_H = 380
# One chrome row (metric tabs + tools). Statistics live in the Live data panel.
_MOTION_GRAPH_CHROME_RESERVE_H = 40
_MOTION_GRAPH_CANVAS_ROW = 1
_MOTION_PATH_MIN_W = 280
_MOTION_PATH_WIDTH_SHARE = 0.18
# Share of the page / workspace height reserved for the kinematics graph row.
_MOTION_GRAPH_HEIGHT_SHARE = 0.75
# Small buffer below the graph; compact lower rows retain their natural height.
_MOTION_LOWER_COMFORT_PX = 8
# Below this width the optional path collapses so the kinematics graph keeps
# the full row instead of creating a cramped or empty right-hand panel.
# Stay graph-only through typical 1920-wide layouts; the path split is
# reserved for extra-wide windows.
_MOTION_STACK_BELOW_W = 2400
# Live data panel: ribbon + kinematics cells. Empty collapses so the graph grows.
# 148 px fits two kinematics rows at 125% DPI; three-row wraps grow toward the cap.
_MOTION_STATS_MAX_H = 148
_MOTION_STATS_COLLAPSED_H = 24
_MOTION_STATS_HARD_CAP_H = 188
_MOTION_STATS_RIBBON_H = 22
_MOTION_STATS_CELL_ROW_H = 36
# Short hosts keep Live data compact so the kinematics graph is not covered.
_MOTION_STATS_SHORT_HOST_H = 88
_MOTION_GRAPH_SHORT_SHARE = 0.72
_MOTION_SHORT_WINDOW_H = 900
# Compare Joints row floors — combobox + chips must stay fully visible.
_MOTION_COMPARE_UNSTACKED_H = 52
_MOTION_COMPARE_STACKED_H = 88
# Metric tabs and Graph/Reset/Export tools share one row until this width.
_MOTION_CHROME_STACK_BELOW_W = 860

# Live Kinematics cells, in display order.
_MOTION_STATS_FIELDS: tuple[tuple[str, str], ...] = (
    ("joint", "Joint"),
    ("angle", "Knee Flexion"),
    ("velocity", "Angular velocity"),
    ("acceleration", "Angular acceleration"),
    ("jerk", "Linear jerk"),
    ("angular_jerk", "Angular jerk"),
    ("x", "X"),
    ("y", "Y"),
    ("z", "Z"),
    ("frame", "Frame"),
    ("time", "Time"),
)

_MOTION_STATS_EMPTY = ("", "—", "-", "N/A", "n/a", "None")


def _is_blank_metric(text: Any) -> bool:
    """True when a live-kinematics value carries no measurement."""
    return str(text).strip() in _MOTION_STATS_EMPTY


def _motion_workspace_is_short(gui: Any | None, host_h: int = 0) -> bool:
    """True on 768p-class windows where Live data must stay compact."""
    h = int(host_h or 0)
    if 360 <= h < 720:
        return True
    root = getattr(gui, "root", None) if gui is not None else None
    if root is not None:
        try:
            rh = int(root.winfo_height() or 0)
            if 480 <= rh < int(_MOTION_SHORT_WINDOW_H):
                return True
        except tk.TclError:
            pass
    return False


def _motion_stats_column_count(width: int, count: int) -> int:
    """Responsive Live Kinematics columns without squeezing all fields into one row."""
    count = max(1, int(count))
    width = max(0, int(width))
    if width < 560:
        return min(2, count)
    if width < 900:
        return min(4, count)
    if width < 1280:
        return min(5, count)
    return count


def _motion_stats_expanded_height(
    *, width: int, n_fields: int, host_h: int = 0, gui: Any | None = None
) -> int:
    """Expanded Live data height: two rows at MAX_H; extra rows grow toward the cap.

    Short 768p hosts keep a compact cap so the kinematics graph stays readable.
    """
    n_fields = max(1, int(n_fields))
    width = max(0, int(width))
    floor = int(_MOTION_STATS_MAX_H)
    short_host = _motion_workspace_is_short(gui, host_h)
    if short_host:
        floor = min(floor, int(_MOTION_STATS_SHORT_HOST_H))
    if width < 80:
        return floor
    cols = _motion_stats_column_count(width, n_fields)
    rows = max(1, (n_fields + cols - 1) // cols)
    needed = int(_MOTION_STATS_RIBBON_H + rows * _MOTION_STATS_CELL_ROW_H + 8)
    if short_host:
        needed = int(rows * _MOTION_STATS_CELL_ROW_H + 8)
    cap = int(_MOTION_STATS_HARD_CAP_H)
    if short_host:
        cap = min(cap, int(_MOTION_STATS_SHORT_HOST_H))
    if rows <= 2:
        return floor
    return max(floor, min(cap, needed))


def refresh_motion_angle_convention(gui: Any) -> None:
    """Update the Motion Analysis knee/joint angle convention label."""
    lbl = getattr(gui, "lbl_motion_angle_convention", None)
    if lbl is None:
        return
    selection = getattr(gui, "selection", None)
    active = getattr(selection, "active_item_id", None)
    selected = (active,) if active else (
        getattr(selection, "selected", None) or ()
    )
    short, tip = motion_angle_convention(selected)
    metric = str(getattr(gui, "_motion_active_metric", None) or "angle")
    if metric != "angle":
        short = ""
    try:
        lbl.configure(text=short)
    except tk.TclError:
        return
    gui._motion_angle_convention_tip = tip
    create_tooltip(lbl, tip, wraplength=380)
    title = getattr(gui, "lbl_motion_stats_angle_title", None)
    if title is not None:
        create_tooltip(title, tip, wraplength=380)


def _apply_motion_stats_panel_mode(gui: Any, *, populated: bool) -> None:
    """Collapse Live Kinematics to a status row when empty; expand when values exist."""
    stats = getattr(gui, "_motion_stats_panel", None)
    host = getattr(gui, "_motion_workspace_host", None)
    if stats is None:
        return
    collapsed = not populated
    if collapsed == bool(getattr(gui, "_motion_stats_collapsed", None)):
        _reflow_motion_stats_cells(gui)
        return
    gui._motion_stats_collapsed = collapsed
    height = _MOTION_STATS_COLLAPSED_H if collapsed else _MOTION_STATS_MAX_H
    if not collapsed:
        try:
            stats_w = int(stats.winfo_width() or 0)
        except tk.TclError:
            stats_w = 0
        host_h = 0
        if host is not None:
            try:
                host_h = int(host.winfo_height() or 0)
            except tk.TclError:
                host_h = 0
        height = _motion_stats_expanded_height(
            width=stats_w,
            n_fields=len(_MOTION_STATS_FIELDS),
            host_h=host_h,
            gui=gui,
        )
    ribbon = getattr(gui, "_motion_graph_stats_ribbon", None)
    try:
        host_h = 0
        if host is not None:
            try:
                host_h = int(host.winfo_height() or 0)
            except tk.TclError:
                host_h = 0
        short_host = _motion_workspace_is_short(gui, host_h)
        if ribbon is not None:
            if collapsed or short_host:
                ribbon.grid_remove()
            else:
                ribbon.grid()
        # Lock height so Live data cannot grow past Compare Joints on 768p.
        stats.grid_propagate(False)
        stats.configure(height=height)
        if host is not None:
            host.rowconfigure(2, weight=0, minsize=height)
    except tk.TclError:
        pass
    _reflow_motion_stats_cells(gui)
    sync = getattr(gui, "_sync_motion_graph_height_share", None)
    if callable(sync):
        try:
            # Force recompute so freed height returns to the graph immediately.
            gui._motion_graph_row_minsize = None
            sync()
        except Exception:
            pass
    _schedule_motion_figure_fit(gui)


def sync_motion_live_kinematics(gui: Any, values: dict[str, Any]) -> int:
    """Fill every Live Kinematics field, marking genuine gaps unavailable."""
    cells = getattr(gui, "_motion_stats_cells", None) or {}
    labels = getattr(gui, "_motion_stats_value_labels", None) or {}
    if not cells:
        _apply_motion_stats_panel_mode(gui, populated=False)
        return 0

    if not values:
        for cell in cells.values():
            try:
                cell.grid_remove()
            except tk.TclError:
                pass
        gui._motion_stats_layout = {
            key: None for key, _title in _MOTION_STATS_FIELDS
        }
        hint = getattr(gui, "_motion_stats_empty_hint", None)
        if hint is not None:
            try:
                hint.grid(row=0, column=0, sticky="w")
            except tk.TclError:
                pass
        _apply_motion_stats_panel_mode(gui, populated=False)
        return 0

    placed: dict[str, int | None] = getattr(gui, "_motion_stats_layout", None) or {}
    layout: dict[str, int | None] = {}
    visible = 0
    for key, _title in _MOTION_STATS_FIELDS:
        cell = cells.get(key)
        if cell is None:
            continue
        text = values.get(key, "Unavailable")
        try:
            if _is_blank_metric(text):
                text = "Unavailable"
            lbl = labels.get(key)
            if lbl is not None:
                lbl.configure(text=str(text))
            # Re-grid only when the collapse pass moved this cell.
            if placed.get(key) != visible:
                cell.grid(
                    row=0,
                    column=visible,
                    sticky="w",
                    padx=(0 if visible == 0 else PAD_MD, 0),
                )
        except tk.TclError:
            layout[key] = placed.get(key)
            continue
        layout[key] = visible
        visible += 1

    gui._motion_stats_layout = layout

    hint = getattr(gui, "_motion_stats_empty_hint", None)
    if hint is not None:
        try:
            if visible:
                hint.grid_remove()
            else:
                hint.grid(row=0, column=0, sticky="w")
        except tk.TclError:
            pass

    _apply_motion_stats_panel_mode(gui, populated=True)
    _reflow_motion_stats_cells(gui)
    return visible


def _reflow_motion_stats_cells(gui: Any) -> None:
    """Wrap live-kinematics cells so values stay readable on a narrower window."""
    if bool(getattr(gui, "_motion_stats_collapsed", False)):
        return
    if bool(getattr(gui, "_motion_stats_reflow_active", False)):
        return
    gui._motion_stats_reflow_active = True
    try:
        _reflow_motion_stats_cells_impl(gui)
    finally:
        gui._motion_stats_reflow_active = False


def _reflow_motion_stats_cells_impl(gui: Any) -> None:
    """Inner Live-data wrap; caller holds the re-entry lock."""
    row = getattr(gui, "_motion_stats_row", None)
    cells = getattr(gui, "_motion_stats_cells", None) or {}
    layout = getattr(gui, "_motion_stats_layout", None) or {}
    if row is None or not cells:
        return
    visible = [
        cells[key]
        for key, _title in _MOTION_STATS_FIELDS
        if layout.get(key) is not None and cells.get(key) is not None
    ]
    n = len(visible)
    if n == 0:
        return
    try:
        width = int(row.winfo_width() or 0)
        if width < 40:
            stats = getattr(gui, "_motion_stats_panel", None)
            width = int(stats.winfo_width() or 0) if stats is not None else 0
    except tk.TclError:
        return
    cols = _motion_stats_column_count(width, n)
    flag = (cols, n)
    wrap_bucket = max(0, width // 24)
    flag = (cols, n, wrap_bucket)
    if flag != getattr(gui, "_motion_stats_wrap", None):
        gui._motion_stats_wrap = flag
        previous_cols = int(getattr(gui, "_motion_stats_configured_cols", 0) or 0)
        gui._motion_stats_configured_cols = cols
        for column in range(max(previous_cols, n)):
            try:
                row.columnconfigure(
                    column,
                    weight=1 if column < cols else 0,
                    uniform="motion-live" if column < cols else "",
                )
            except tk.TclError:
                pass
        for index, cell in enumerate(visible):
            r, c = divmod(index, cols)
            try:
                cell.grid(
                    row=r,
                    column=c,
                    sticky="ew",
                    padx=(0 if c == 0 else PAD_MD, 0),
                    pady=(0 if r == 0 else PAD_XS, 0),
                )
            except tk.TclError:
                continue
    cell_wrap = max(56, (width // max(1, cols)) - 12)
    labels = getattr(gui, "_motion_stats_value_labels", None) or {}
    for key, _title in _MOTION_STATS_FIELDS:
        value = labels.get(key)
        if value is None:
            continue
        try:
            value.configure(wraplength=cell_wrap, justify=tk.LEFT)
        except tk.TclError:
            continue
    stats = getattr(gui, "_motion_stats_panel", None)
    host = getattr(gui, "_motion_workspace_host", None)
    if stats is None:
        return
    host_h = 0
    if host is not None:
        try:
            host_h = int(host.winfo_height() or 0)
        except tk.TclError:
            host_h = 0
    height = _motion_stats_expanded_height(
        width=width, n_fields=n, host_h=host_h, gui=gui
    )
    try:
        current = int(stats.cget("height") or 0)
    except (tk.TclError, TypeError, ValueError):
        current = 0
    if height != current:
        try:
            stats.configure(height=height)
            if host is not None:
                host.rowconfigure(2, weight=0, minsize=height)
        except tk.TclError:
            pass
        sync = getattr(gui, "_sync_motion_graph_height_share", None)
        if callable(sync):
            try:
                gui._motion_graph_row_minsize = None
                sync()
            except Exception:
                pass
        _schedule_motion_figure_fit(gui)


def build_motion_analysis_workspace(gui: Any, tab_motion: tk.Misc) -> tk.Frame:
    """Install the professional Motion Analysis workspace into ``tab_motion``.

    Layout:
      row 0 — page header (caller)
      host row 0 — Selected Joint
      host row 1 — tall kinematics graph (focus) | optional 3D path
      host row 2 — Live data panel (ribbon + kinematics; not over the plot)
      host row 3 — Compare Joints chips
    """
    from stablewalk.ui.design_system.components import EmptyState, QuietPanel
    from stablewalk.ui.viewers.chart_interactions import (
        attach_chart_interactions,
        build_chart_tools_bar,
    )

    # Content host under the page header.
    host = ttk.Frame(tab_motion)
    host.grid(row=1, column=0, sticky="nsew")
    tab_motion.rowconfigure(1, weight=1)
    tab_motion.columnconfigure(0, weight=1)
    host.columnconfigure(0, weight=1)
    host.rowconfigure(0, weight=0)  # toolbar
    # Preferred graph floor is applied by height-share after the host is mapped.
    # A 380 px Tk minsize here clips Compare Joints on 768p.
    host.rowconfigure(1, weight=1, minsize=160)
    host.rowconfigure(2, weight=0)  # compact live kinematics
    host.rowconfigure(3, weight=0, minsize=40)  # Compare Joints — never collapse
    gui._section_kinematic = host
    gui._motion_workspace_host = host

    # ── Top: timeline + joint + play ──────────────────────────────────────
    top = tk.Frame(host, bg=ELEVATED, highlightthickness=0)
    top.grid(row=0, column=0, sticky="ew", padx=(PAD_XS, 0), pady=(0, PAD_SM))
    gui._motion_workspace_toolbar = top

    play_row = tk.Frame(top, bg=ELEVATED, highlightthickness=0)
    play_row.grid(row=0, column=0, sticky="w", padx=(PAD_SM, PAD_MD), pady=PAD_SM)
    gui._motion_workspace_play_row = play_row

    # Playback lives on the global footer. Keep a hidden timeline for sync
    # (same IntVar) so graph scrub still follows the playhead.
    gui.btn_motion_play = None
    play_row.grid_remove()
    if not hasattr(gui, "frame_var") or gui.frame_var is None:
        gui.frame_var = tk.IntVar(value=0)
    gui.motion_timeline = ttk.Scale(
        top,
        from_=0,
        to=1,
        orient=tk.HORIZONTAL,
        variable=gui.frame_var,
        command=getattr(gui, "_on_slider", None),
        style="Transport.Horizontal.TScale",
    )
    gui.motion_timeline.grid_remove()
    # Playhead lives on the global transport. Keep hidden labels so existing
    # playback updates still have a target; do not duplicate them here.
    top.columnconfigure(0, weight=1)
    top.columnconfigure(1, weight=0)
    top.columnconfigure(2, weight=0)

    readout = tk.Frame(top, bg=ELEVATED, highlightthickness=0)
    gui._motion_workspace_readout = readout
    gui.lbl_motion_time = tk.Label(
        readout,
        text="0.00 s",
        bg=ELEVATED,
        fg=TEXT_SECONDARY,
        font=FONT_KPI_SUPPORTING,
        anchor="w",
    )
    gui.lbl_motion_time.grid(row=0, column=0, sticky="w")
    gui.lbl_motion_frame = tk.Label(
        readout,
        text="Frame —",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_motion_frame.grid(
        row=0,
        column=1,
        sticky="w",
        padx=(PAD_MD, 0),
    )

    joint_row = tk.Frame(
        top,
        bg=ELEVATED,
        highlightthickness=0,
    )
    joint_row.grid(
        row=0,
        column=0,
        sticky="ew",
        padx=(PAD_SM, PAD_SM),
        pady=PAD_XS,
    )
    joint_row.columnconfigure(0, weight=0)
    joint_row.columnconfigure(1, weight=0)
    joint_row.columnconfigure(2, weight=1)
    gui._motion_workspace_joint_controls = joint_row
    gui.lbl_motion_selected_joint = tk.Label(
        joint_row,
        text="Selected Joint",
        bg=ELEVATED,
        fg=TEXT_SECONDARY,
        font=FONT_KPI_TITLE,
        anchor="w",
        justify=tk.LEFT,
        wraplength=0,
        padx=PAD_SM,
        pady=2,
    )
    gui.lbl_motion_selected_joint.grid(
        row=0, column=0, sticky="w", padx=(PAD_SM, PAD_XS), pady=2
    )

    if not hasattr(gui, "var_overview_joint_pick") or gui.var_overview_joint_pick is None:
        gui.var_overview_joint_pick = tk.StringVar(value="Select joint…")
    joint_labels = ["Select joint…"] + [
        GUI_DOF_LABELS.get(item_id, item_id) for item_id in GUI_DOF_ITEM_IDS
    ]
    from stablewalk.ui.tk.layout_scale import combobox_char_width

    gui.cmb_motion_joint = ttk.Combobox(
        joint_row,
        textvariable=gui.var_overview_joint_pick,
        values=joint_labels,
        state="readonly",
        width=combobox_char_width(joint_labels, minimum=18, maximum=36),
        takefocus=False,
    )
    gui.cmb_motion_joint.grid(
        row=0, column=1, sticky="w", padx=(0, PAD_SM), pady=2
    )
    create_tooltip(
        gui.cmb_motion_joint,
        "Active joint for live kinematics and the graph focus. "
        "Choosing a joint here replaces the current selection. "
        "Use Compare Joints below to overlay additional series.",
        wraplength=360,
    )
    short_conv, tip_conv = motion_angle_convention()
    gui.lbl_motion_angle_convention = tk.Label(
        joint_row,
        text=short_conv,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=0,
        padx=PAD_XS,
        pady=2,
    )
    gui.lbl_motion_angle_convention.grid(
        row=0, column=2, sticky="w", padx=(PAD_SM, PAD_SM), pady=2
    )
    create_tooltip(gui.lbl_motion_angle_convention, tip_conv, wraplength=380)
    gui._motion_angle_convention_tip = tip_conv

    def _on_joint_chosen(_event: object = None) -> None:
        choose = getattr(gui, "_on_overview_joint_pick_chosen", None)
        if callable(choose):
            choose()
        _reflow_motion_middle(gui)

    gui.cmb_motion_joint.bind("<<ComboboxSelected>>", _on_joint_chosen)

    # ── Middle: graphs (visual focus) | 3D path ───────────────────────────
    from stablewalk.ui.tk.analysis_split import create_h_split

    middle_host = ttk.Frame(host)
    middle_host.grid(row=1, column=0, sticky="nsew")
    middle_host.columnconfigure(0, weight=1)
    middle_host.rowconfigure(0, weight=1)
    middle = create_h_split(middle_host)
    gui._motion_workspace_middle = middle_host
    gui._motion_middle_split = middle

    graphs_panel = ttk.Frame(middle_host)
    graphs_panel.columnconfigure(0, weight=1)
    graphs_panel.rowconfigure(0, weight=0)  # metric tabs + tools
    graphs_panel.rowconfigure(
        _MOTION_GRAPH_CANVAS_ROW, weight=1, minsize=160
    )
    gui._motion_graphs_panel = graphs_panel

    chrome = tk.Frame(graphs_panel, bg=PANEL, highlightthickness=0)
    chrome.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
    chrome.columnconfigure(0, weight=1)
    chrome.columnconfigure(1, weight=0)
    gui._motion_graph_chrome = chrome

    # Metric tabs — only the active graph is created / painted.
    metric_nb = ttk.Notebook(chrome)
    metric_nb.grid(row=0, column=0, sticky="ew")
    gui._motion_metric_notebook = metric_nb

    tools_host = tk.Frame(chrome, bg=PANEL, highlightthickness=0)
    tools_host.grid(row=0, column=1, sticky="e", padx=(PAD_SM, 0))
    tools_host.columnconfigure(0, weight=1)
    gui._motion_graph_tools_host = tools_host
    gui._motion_metric_panels: dict[str, dict[str, Any]] = {}
    gui._motion_active_metric = MOTION_ANALYSIS_TABS[0][0]

    gui.motion_graph_host = tk.Frame(graphs_panel, bg=PANEL, highlightthickness=0)
    gui.motion_graph_host.grid(row=_MOTION_GRAPH_CANVAS_ROW, column=0, sticky="nsew")
    gui.motion_graph_host.columnconfigure(0, weight=1)
    gui.motion_graph_host.rowconfigure(0, weight=1, minsize=160)
    # Host size comes from the allocated row, not the figure's preferred inches.
    # Otherwise a 9.2" canvas blocks shrinking and clips the right-hand chrome.
    gui.motion_graph_host.grid_propagate(False)
    gui._motion_graph_clip_inner = gui.motion_graph_host

    gui.lbl_motion_graph_empty = EmptyState(
        gui.motion_graph_host,
        "Load and analyze a walking video, then select a joint.\n"
        "Angle / Velocity / Acceleration / Position / Jerk share one timeline.",
        bg=PANEL,
    )
    gui.lbl_motion_graph_empty.grid(row=0, column=0, sticky="nsew")

    gui.fig_motion_workspace = None
    gui.canvas_motion_workspace = None
    gui._motion_workspace_chart_ready = False

    def _seek_from_chart(time_s: float) -> None:
        seek = getattr(gui, "_seek_motion_workspace_time", None)
        if callable(seek):
            seek(float(time_s))

    def _hover_seek_enabled() -> bool:
        return not bool(getattr(gui, "playing", False))

    def _activate_metric_panel(metric_id: str) -> dict[str, Any] | None:
        """Show one metric canvas; create its figure lazily on first visit."""
        panels = getattr(gui, "_motion_metric_panels", None)
        if panels is None:
            gui._motion_metric_panels = {}
            panels = gui._motion_metric_panels
        host_w = getattr(gui, "_motion_graph_clip_inner", None) or getattr(
            gui, "motion_graph_host", None
        )
        if host_w is None:
            return None

        # Hide every other metric canvas.
        for mid, panel in panels.items():
            canvas = panel.get("canvas")
            if canvas is None:
                continue
            try:
                if mid == metric_id:
                    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
                else:
                    canvas.get_tk_widget().grid_remove()
            except tk.TclError:
                pass

        panel = panels.get(metric_id)
        if panel is None or panel.get("canvas") is None:
            from stablewalk.ui.viewers.chart_style import apply_chart_panel_style

            fig = Figure(figsize=_MOTION_WORKSPACE_FIGSIZE, dpi=110, facecolor=PANEL)
            ax = fig.add_subplot(1, 1, 1)
            # Style immediately so a deferred first draw never leaves a white Axes.
            apply_chart_panel_style(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            canvas = FigureCanvasTkAgg(fig, master=host_w)
            widget = canvas.get_tk_widget()
            widget.configure(bg=PANEL, highlightthickness=0)
            widget.grid(row=0, column=0, sticky="nsew")
            from stablewalk.ui.tk.clip_viewport import ensure_widget_fills_host

            ensure_widget_fills_host(widget)
            try:
                canvas.draw_idle()
            except Exception:
                pass
            attach_chart_interactions(
                fig,
                canvas,
                on_seek_time=_seek_from_chart,
                hover_seek=True,
                hover_seek_enabled=_hover_seek_enabled,
                sync_crosshair=True,
                enable_measure=True,
            )
            panel = {
                "metric_id": metric_id,
                "fig": fig,
                "canvas": canvas,
                "render_key": None,
            }
            panels[metric_id] = panel
        else:
            try:
                panel["canvas"].get_tk_widget().grid(row=0, column=0, sticky="nsew")
            except tk.TclError:
                pass

        gui._motion_active_metric = metric_id
        gui.fig_motion_workspace = panel["fig"]
        gui.canvas_motion_workspace = panel["canvas"]
        gui._motion_workspace_chart_ready = True
        # A lazily-created figure initially carries its preferred inch size.
        # Fit it after Tk has assigned the flexible graph host its real pixels.
        _schedule_motion_figure_fit(gui)
        refresh_motion_angle_convention(gui)
        # Keep EmptyState until the first successful series paint (see app.py).
        if panel.get("render_key") is not None:
            try:
                gui.lbl_motion_graph_empty.grid_remove()
            except tk.TclError:
                pass
        return panel

    def _ensure_motion_workspace_chart() -> bool:
        metric_id = getattr(gui, "_motion_active_metric", None) or MOTION_ANALYSIS_TABS[0][0]
        return _activate_metric_panel(str(metric_id)) is not None

    def _on_metric_tab_changed(_event: object = None) -> None:
        try:
            tab_text = str(metric_nb.tab(metric_nb.select(), "text"))
        except tk.TclError:
            return
        metric_id = None
        for mid, title, _ylabel in MOTION_ANALYSIS_TABS:
            if title == tab_text:
                metric_id = mid
                break
        if not metric_id:
            return
        if metric_id == getattr(gui, "_motion_active_metric", None) and getattr(
            gui, "canvas_motion_workspace", None
        ) is not None:
            return
        _activate_metric_panel(metric_id)
        # Redraw active tab from the shared bundle — never recompute series.
        # First-paint exception in app.py draws even during playback when
        # render_key is unset (avoids a blank/white Axes on tab switch).
        refresh = getattr(gui, "_update_joint_motion_analysis_chart", None)
        if callable(refresh):
            try:
                refresh(force_rebuild=False)
            except Exception:
                pass
        try:
            from stablewalk.ui.tk.ui_motion import mark_viz_reveal, reveal_host

            mark_viz_reveal(gui, kind="graph")
            host = getattr(gui, "knee_chart_host", None) or getattr(
                gui, "_motion_graphs_panel", None
            )
            reveal_host(gui, host, key=f"motion-metric:{metric_id}")
        except Exception:
            pass

    for _metric_id, title, _ylabel in MOTION_ANALYSIS_TABS:
        page = ttk.Frame(metric_nb)
        metric_nb.add(page, text=title)
    metric_nb.bind("<<NotebookTabChanged>>", _on_metric_tab_changed)

    gui._ensure_motion_workspace_chart = _ensure_motion_workspace_chart
    gui._activate_motion_metric_panel = _activate_metric_panel

    tools = build_chart_tools_bar(
        gui,
        tools_host,
        fig_attr="fig_motion_workspace",
        canvas_attr="canvas_motion_workspace",
        export_name="motion_workspace",
        enable_measure=True,
    )
    from stablewalk.ui.tk.tab_grid import grid_ew, stretch_columns

    stretch_columns(tools_host)
    grid_ew(tools)
    gui.motion_workspace_tools = tools

    def _export_csv() -> None:
        export = getattr(gui, "_export_joint_motion_csv", None)
        if export is not None:
            export()

    btn_csv = ttk.Button(
        tools, text="CSV", width=5, command=_export_csv, takefocus=False
    )
    btn_csv.pack(side=tk.LEFT, padx=(PAD_XS, 0), pady=PAD_XS // 2)
    create_tooltip(btn_csv, "Export synchronized motion series as CSV")

    # Right: 3D path panel — caller reparents traj_panel here after build.
    path_slot = ttk.Frame(middle_host)
    path_slot.columnconfigure(0, weight=1)
    path_slot.rowconfigure(0, weight=1)
    gui._motion_path_slot = path_slot

    # Start graph-only on the middle host. The optional path is added by the
    # responsive reflow only when a session and selected joint can populate it.
    # Do not parent the graph into the hidden split — forget() leaves the
    # widget's master as the PanedWindow, so a later grid() would stay unmapped.
    graphs_panel.grid(row=0, column=0, sticky="nsew")

    # ── Bottom: Live data — ribbon + kinematics; never overlays the plot ──
    stats = QuietPanel(host, bg=ELEVATED)
    stats.grid(row=2, column=0, sticky="ew", pady=(PAD_XS, 0))
    stats.columnconfigure(0, weight=1)
    stats.rowconfigure(0, weight=0)
    stats.rowconfigure(1, weight=0)
    stats.rowconfigure(2, weight=0)
    stats.grid_propagate(False)
    # Start collapsed: no values until a joint/frame provides kinematics.
    stats.configure(height=_MOTION_STATS_COLLAPSED_H)
    gui._motion_stats_panel = stats
    gui._motion_stats_collapsed = True

    header = tk.Frame(stats, bg=ELEVATED, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(1, weight=1)
    tk.Label(
        header,
        text="Live data",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_KPI_TITLE,
        anchor="w",
    ).grid(row=0, column=0, sticky="nw", padx=(PAD_SM, PAD_MD), pady=(PAD_XS, 0))

    ribbon = tk.Frame(stats, bg=ELEVATED, highlightthickness=0)
    ribbon.grid(row=1, column=0, sticky="ew", padx=(PAD_SM, PAD_SM), pady=(0, PAD_XS))
    gui._motion_graph_stats_ribbon = ribbon
    gui._motion_graph_stats_labels: dict[str, tk.Label] = {}
    ribbon_cells: list[tk.Frame] = []
    for col, (key, title) in enumerate(
        (
            ("minimum", "Minimum"),
            ("maximum", "Maximum"),
            ("mean", "Mean"),
            ("rom", "ROM"),
            ("std", "Std Dev"),
            ("current", "Current"),
        )
    ):
        ribbon.columnconfigure(col, weight=1)
        cell = tk.Frame(ribbon, bg=ELEVATED, highlightthickness=0)
        cell.grid(
            row=0,
            column=col,
            sticky="nsew",
            padx=(PAD_SM if col else 0, 0),
            pady=(2, 2),
        )
        ribbon_cells.append(cell)
        tk.Label(
            cell,
            text=title,
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_KPI_TITLE,
            anchor="w",
        ).pack(anchor="w")
        value = tk.Label(
            cell,
            text="—",
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_KPI_SUPPORTING,
            anchor="w",
        )
        value.pack(anchor="w")
        gui._motion_graph_stats_labels[key] = value
    gui._motion_graph_stats_cells = ribbon_cells
    ribbon.grid_remove()

    stats_row = tk.Frame(stats, bg=ELEVATED, highlightthickness=0)
    stats_row.grid(row=2, column=0, sticky="ew", padx=(PAD_SM, PAD_SM), pady=(0, PAD_XS))
    gui._motion_stats_row = stats_row
    gui._motion_stats_value_labels: dict[str, tk.Label] = {}
    gui._motion_stats_cells: dict[str, tk.Frame] = {}
    gui._motion_stats_layout: dict[str, int | None] = {}

    for col, (key, title) in enumerate(_MOTION_STATS_FIELDS):
        cell = tk.Frame(stats_row, bg=ELEVATED, highlightthickness=0)
        cell.grid(row=0, column=col, sticky="w", padx=(0 if col == 0 else PAD_MD, 0))
        title_lbl = tk.Label(
            cell,
            text=title,
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_KPI_TITLE,
            anchor="w",
        )
        title_lbl.pack(anchor="w")
        if key == "angle":
            gui.lbl_motion_stats_angle_title = title_lbl
            _, angle_tip = motion_angle_convention()
            create_tooltip(title_lbl, angle_tip, wraplength=380)
        value = tk.Label(
            cell,
            text="",
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_UI_SM,
            anchor="w",
            justify=tk.LEFT,
        )
        value.pack(anchor="w")
        gui._motion_stats_value_labels[key] = value
        gui._motion_stats_cells[key] = cell
        cell.grid_remove()

    gui._motion_stats_empty_hint = tk.Label(
        stats_row,
        text="No live values",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui._motion_stats_empty_hint.grid(row=0, column=0, sticky="w")
    stats_row.bind(
        "<Configure>",
        lambda _e: _reflow_motion_stats_cells(gui),
        add="+",
    )

    # Compare Joints chips footer (existing builder fills this).
    # Natural height — a locked 40 px strip clipped the combobox when stacked.
    joint_footer = ttk.Frame(host)
    joint_footer.grid(row=3, column=0, sticky="ew", pady=(PAD_XS, 0))
    joint_footer.columnconfigure(0, weight=1)
    host.rowconfigure(3, weight=0, minsize=40)
    gui._motion_joint_row = joint_footer

    _install_motion_graph_height_share(gui, host)
    _install_motion_workspace_reflow(gui)

    gui._joint_motion_bundle = None
    gui._joint_motion_bundle_key = None
    return host


def _install_motion_graph_height_share(gui: Any, host: tk.Misc) -> None:
    """Keep the kinematics graph at ~75% of the workspace height on resize."""

    def _reserved_height(widget: tk.Misc | None) -> int:
        if widget is None:
            return 0
        try:
            # Prefer the locked strip height when Live Kinematics is collapsed /
            # expanded with grid_propagate(False); fall back to requested size.
            configured = int(widget.cget("height") or 0)
            req = int(widget.winfo_reqheight() or 0)
            mapped = 0
            try:
                mapped = int(widget.winfo_height() or 0)
            except tk.TclError:
                mapped = 0
            if widget is getattr(gui, "_motion_joint_row", None):
                stacked = bool(getattr(gui, "_motion_compare_stacked", False))
                floor = (
                    _MOTION_COMPARE_STACKED_H
                    if stacked
                    else _MOTION_COMPARE_UNSTACKED_H
                )
                return max(req, mapped, int(floor))
            if widget is getattr(gui, "_motion_stats_panel", None) and not bool(
                getattr(gui, "_motion_stats_collapsed", True)
            ):
                return max(configured, req, mapped)
            if configured > 0:
                return configured
            return max(req, mapped)
        except (tk.TclError, TypeError, ValueError):
            try:
                return int(widget.winfo_reqheight() or 0)
            except tk.TclError:
                return 0

    def _apply(_event: object | None = None) -> None:
        try:
            height = int(host.winfo_height() or 0)
        except tk.TclError:
            return
        if height < 80:
            return
        reserved = 0
        for attr in (
            "_motion_workspace_toolbar",
            "_motion_stats_panel",
            "_motion_joint_row",
        ):
            reserved += _reserved_height(getattr(gui, attr, None))
        available = max(80, height - reserved - _MOTION_LOWER_COMFORT_PX)
        share = _MOTION_GRAPH_HEIGHT_SHARE
        if _motion_workspace_is_short(gui, height):
            share = _MOTION_GRAPH_SHORT_SHARE
        target = min(int(height * share), available)
        preferred = min(_MOTION_WORKSPACE_GRAPH_MIN_H, available)
        if not _motion_workspace_is_short(gui, height):
            target = max(preferred, target)
        else:
            # Do not inflate to the 380 px floor on short windows — that
            # overflows Compare Joints. weight=1 still gives leftover to the graph.
            slack = available - 24 if available > 200 else available
            target = max(80, min(target, slack, available))
        try:
            host.rowconfigure(
                2,
                weight=0,
                minsize=_reserved_height(getattr(gui, "_motion_stats_panel", None)),
            )
            host.rowconfigure(
                3,
                weight=0,
                minsize=_reserved_height(getattr(gui, "_motion_joint_row", None)),
            )
        except tk.TclError:
            pass
        if target == getattr(gui, "_motion_graph_row_minsize", None):
            return
        gui._motion_graph_row_minsize = target
        try:
            host.rowconfigure(1, weight=1, minsize=target)
            graph_floor = max(
                120,
                min(
                    _MOTION_WORKSPACE_GRAPH_MIN_H,
                    target - _MOTION_GRAPH_CHROME_RESERVE_H,
                ),
            )
            graphs_panel = getattr(gui, "_motion_graphs_panel", None)
            if graphs_panel is not None:
                graphs_panel.rowconfigure(
                    _MOTION_GRAPH_CANVAS_ROW, weight=1, minsize=graph_floor
                )
            graph_host = getattr(gui, "motion_graph_host", None)
            if graph_host is not None:
                graph_host.rowconfigure(0, weight=1, minsize=graph_floor)
            clip_inner = getattr(gui, "_motion_graph_clip_inner", None)
            if clip_inner is not None:
                clip_inner.rowconfigure(0, weight=1, minsize=graph_floor)
        except tk.TclError:
            pass

    host.bind("<Configure>", _apply, add="+")
    gui._sync_motion_graph_height_share = _apply


def _motion_path_content_available(gui: Any) -> bool:
    """Whether the optional path panel has enough state to show real content."""
    player = getattr(gui, "skeleton_player", None)
    if player is None or int(getattr(player, "frame_count", 0) or 0) <= 0:
        return False
    selected = getattr(getattr(gui, "selection", None), "selected", None) or ()
    return bool(selected)


def _reflow_motion_middle(gui: Any) -> None:
    """Give the graph the full row unless a useful path fits beside it."""
    from stablewalk.ui.tk.analysis_split import (
        MIN_W_CHART,
        MIN_W_PATH,
        split_add,
        split_clear,
        split_has_children,
        split_mapped,
    )

    host = getattr(gui, "_motion_workspace_middle", None)
    paned = getattr(gui, "_motion_middle_split", None)
    graphs = getattr(gui, "_motion_graphs_panel", None)
    path = getattr(gui, "_motion_path_slot", None)
    if host is None or graphs is None or path is None:
        return
    try:
        width = int(host.winfo_width() or 0)
        height = int(host.winfo_height() or 0)
    except tk.TclError:
        return
    if width < 80:
        return
    show_path = (
        width >= _MOTION_STACK_BELOW_W
        and _motion_path_content_available(gui)
    )
    mode = "split" if show_path else "graph-only"
    prev_mode = getattr(gui, "_motion_middle_mode", None)
    prev_w = int(getattr(gui, "_motion_middle_width", 0) or 0)
    gui._motion_middle_width = int(width)
    if mode == prev_mode and (
        (not show_path and not split_mapped(paned))
        or (show_path and split_has_children(paned, 2))
    ):
        # Keep the graph/path ratio as the window grows. Otherwise leftover
        # pixels dump into the right pane and the kinematics graph looks empty.
        if show_path and paned is not None and abs(int(width) - prev_w) >= 12:
            target = max(MIN_W_CHART, int(width * (1.0 - _MOTION_PATH_WIDTH_SHARE)))
            try:
                paned.sash_place(0, target, 0)
            except tk.TclError:
                pass
        return
    gui._motion_middle_mode = mode
    gui._motion_middle_stacked = not show_path  # compatibility for diagnostics
    try:
        if not show_path:
            split_clear(paned)
            if paned is not None:
                paned.grid_remove()
            host.columnconfigure(0, weight=1)
            host.columnconfigure(1, weight=0)
            graph_floor = min(
                _MOTION_WORKSPACE_GRAPH_MIN_H,
                max(120, height),
            )
            host.rowconfigure(0, weight=1, minsize=graph_floor)
            host.rowconfigure(1, weight=0, minsize=0)
            graphs.grid(
                row=0,
                column=0,
                sticky="nsew",
                padx=0,
                pady=0,
                in_=host,
            )
            path.grid_remove()
        else:
            try:
                graphs.grid_forget()
            except tk.TclError:
                pass
            try:
                path.grid_forget()
            except tk.TclError:
                pass
            host.columnconfigure(0, weight=1)
            middle_floor = min(
                _MOTION_WORKSPACE_GRAPH_MIN_H,
                max(160, height),
            )
            host.rowconfigure(0, weight=1, minsize=middle_floor)
            host.rowconfigure(1, weight=0, minsize=0)
            if paned is not None:
                paned.grid(row=0, column=0, sticky="nsew")
                split_add(paned, graphs, minsize=MIN_W_CHART, padx=2)
                split_add(
                    paned,
                    path,
                    minsize=max(MIN_W_PATH, _MOTION_PATH_MIN_W),
                    padx=2,
                )
                # Establish a graph-first 3:1 split only when entering split
                # mode; users can still drag the sash afterward.
                target = max(MIN_W_CHART, int(width * (1.0 - _MOTION_PATH_WIDTH_SHARE)))
                try:
                    paned.sash_place(0, target, 0)
                except tk.TclError:
                    pass
    except tk.TclError:
        pass


def _reflow_motion_ribbon(gui: Any) -> None:
    """Wrap the stats ribbon so min/max/mean labels are not crushed."""
    ribbon = getattr(gui, "_motion_graph_stats_ribbon", None)
    cells = getattr(gui, "_motion_graph_stats_cells", None) or []
    if ribbon is None or not cells:
        return
    try:
        width = int(ribbon.winfo_width() or 0)
    except tk.TclError:
        return
    n = len(cells)
    cols = n
    if width >= 80:
        if width < 520:
            cols = min(2, n)
        elif width < 760:
            cols = min(3, n)
    if cols == getattr(gui, "_motion_ribbon_wrap", None):
        return
    gui._motion_ribbon_wrap = cols
    try:
        for c in range(n):
            ribbon.columnconfigure(c, weight=1 if c < cols else 0)
        for index, cell in enumerate(cells):
            r, c = divmod(index, cols)
            cell.grid(
                row=r,
                column=c,
                sticky="nsew",
                padx=(PAD_SM if c else PAD_MD, 0),
                pady=(2 if r == 0 else PAD_XS, 2),
            )
    except tk.TclError:
        pass


def _schedule_motion_figure_fit(gui: Any) -> None:
    """Debounced chart fit so the active figure tracks the graph host."""
    root = getattr(gui, "root", None)
    pending = getattr(gui, "_motion_figure_fit_after", None)
    if pending is not None and root is not None:
        try:
            root.after_cancel(pending)
        except (tk.TclError, ValueError):
            pass

    def _fit() -> None:
        gui._motion_figure_fit_after = None
        canvas = getattr(gui, "canvas_motion_workspace", None)
        fig = getattr(gui, "fig_motion_workspace", None)
        host = getattr(gui, "motion_graph_host", None)
        if canvas is None or fig is None or host is None:
            return
        try:
            from stablewalk.ui.tk.dashboard_layout import _fit_figure_to_host
            from stablewalk.ui.viewers.joint_motion_analysis_chart import (
                motion_metric_chart_margins,
            )

            _fit_figure_to_host(
                canvas,
                fig,
                host=host,
                margins=motion_metric_chart_margins,
                min_px=80,
            )
        except Exception:
            pass

    if root is not None:
        try:
            gui._motion_figure_fit_after = root.after(40, _fit)
            return
        except tk.TclError:
            pass
    _fit()


def _reflow_motion_graph_chrome(gui: Any) -> None:
    """Stack metric tabs above tools before they overlap on a narrow graph."""
    chrome = getattr(gui, "_motion_graph_chrome", None)
    metric_nb = getattr(gui, "_motion_metric_notebook", None)
    tools = getattr(gui, "_motion_graph_tools_host", None)
    if chrome is None or metric_nb is None or tools is None:
        return
    try:
        width = int(chrome.winfo_width() or 0)
    except tk.TclError:
        return
    if width < 40:
        return
    stacked = width < _MOTION_CHROME_STACK_BELOW_W
    if stacked == getattr(gui, "_motion_chrome_stacked", None):
        return
    gui._motion_chrome_stacked = stacked
    try:
        if stacked:
            metric_nb.grid(row=0, column=0, columnspan=2, sticky="ew")
            tools.grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="e",
                padx=0,
                pady=(PAD_XS, 0),
            )
            chrome.columnconfigure(1, weight=0)
        else:
            metric_nb.grid(row=0, column=0, columnspan=1, sticky="ew")
            tools.grid(row=0, column=1, columnspan=1, sticky="e", padx=(PAD_SM, 0), pady=0)
            chrome.columnconfigure(1, weight=0)
    except tk.TclError:
        gui._motion_chrome_stacked = None


def _install_motion_workspace_reflow(gui: Any) -> None:
    """Bind chart / path / ribbon / live-kinematics reflow to window resize."""
    middle = getattr(gui, "_motion_workspace_middle", None)
    ribbon = getattr(gui, "_motion_graph_stats_ribbon", None)
    graph_host = getattr(gui, "motion_graph_host", None)
    toolbar = getattr(gui, "_motion_workspace_toolbar", None)

    def _on_middle(_event: object | None = None) -> None:
        _reflow_motion_middle(gui)
        _schedule_motion_figure_fit(gui)

    def _on_ribbon(_event: object | None = None) -> None:
        _reflow_motion_ribbon(gui)

    def _on_graph(_event: object | None = None) -> None:
        _schedule_motion_figure_fit(gui)

    def _on_graph_chrome(_event: object | None = None) -> None:
        _reflow_motion_graph_chrome(gui)

    def _on_toolbar(_event: object | None = None) -> None:
        controls = getattr(gui, "_motion_workspace_joint_controls", None)
        convention = getattr(gui, "lbl_motion_angle_convention", None)
        if toolbar is None or controls is None:
            return
        try:
            width = int(toolbar.winfo_width() or 0)
            toolbar.columnconfigure(0, weight=1)
            toolbar.columnconfigure(1, weight=0)
            toolbar.columnconfigure(2, weight=0)
            controls.grid(
                row=0,
                column=0,
                columnspan=1,
                sticky="ew",
                padx=(PAD_SM, PAD_SM),
                pady=PAD_XS,
            )
            combo = getattr(gui, "cmb_motion_joint", None)
            if combo is not None and width > 80:
                # Character width: keep long DOF names from pushing the
                # convention label off the toolbar.
                chars = max(14, min(28, max(12, (width - 240) // 9)))
                combo.configure(width=chars)
            if convention is not None:
                if 0 < width < 760:
                    convention.grid(
                        row=1,
                        column=0,
                        columnspan=3,
                        sticky="w",
                        padx=(PAD_SM, PAD_SM),
                        pady=(0, 2),
                    )
                else:
                    convention.grid(
                        row=0,
                        column=2,
                        sticky="w",
                        padx=(PAD_SM, PAD_SM),
                        pady=2,
                    )
        except tk.TclError:
            pass

    if middle is not None:
        middle.bind("<Configure>", _on_middle, add="+")
    if ribbon is not None:
        ribbon.bind("<Configure>", _on_ribbon, add="+")
    if graph_host is not None:
        graph_host.bind("<Configure>", _on_graph, add="+")
    if toolbar is not None:
        toolbar.bind("<Configure>", _on_toolbar, add="+")
    chrome = getattr(gui, "_motion_graph_chrome", None)
    if chrome is not None:
        chrome.bind("<Configure>", _on_graph_chrome, add="+")
    gui._motion_middle_stacked = None
    gui._motion_middle_mode = None
    gui._motion_middle_width = None
    gui._reflow_motion_workspace = lambda: _reflow_motion_middle(gui)
    gui._motion_ribbon_wrap = None


def mount_motion_path_panel(gui: Any, traj_panel: tk.Misc) -> None:
    """Place the existing 3D Joint Movement panel into the workspace path slot."""
    slot = getattr(gui, "_motion_path_slot", None)
    if slot is None or traj_panel is None:
        return
    try:
        traj_panel.grid_forget()
    except tk.TclError:
        pass
    try:
        # Re-parent into the Motion workspace right column.
        traj_panel.grid(row=0, column=0, sticky="nsew", in_=slot)
        slot.columnconfigure(0, weight=1)
        slot.rowconfigure(0, weight=1)
    except tk.TclError:
        pass


__all__ = [
    "_MOTION_GRAPH_CANVAS_ROW",
    "_MOTION_GRAPH_HEIGHT_SHARE",
    "_MOTION_PATH_MIN_W",
    "_MOTION_PATH_WIDTH_SHARE",
    "_MOTION_COMPARE_STACKED_H",
    "_MOTION_COMPARE_UNSTACKED_H",
    "_MOTION_STATS_COLLAPSED_H",
    "_MOTION_STATS_FIELDS",
    "_MOTION_STATS_HARD_CAP_H",
    "_MOTION_STATS_MAX_H",
    "_MOTION_WORKSPACE_FIGSIZE",
    "_MOTION_WORKSPACE_GRAPH_MIN_H",
    "_motion_path_content_available",
    "_motion_stats_column_count",
    "build_motion_analysis_workspace",
    "mount_motion_path_panel",
    "sync_motion_live_kinematics",
]
