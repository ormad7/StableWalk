"""
Professional dashboard grid for the StableWalk gait analysis GUI.

Organizes visualization, analytics tables, and the DOF control sidebar into
clear, presentation-ready sections with consistent spacing and empty states.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from stablewalk.ui.dof_position_table import DOF_TABLE_MODE_DEFAULT
from stablewalk.ui.joint_data_table import (
    JOINT_DATA_TABLE_COLUMNS,
    JOINT_DATA_TABLE_HEADINGS,
    JOINT_DATA_TABLE_WIDTHS,
    JOINT_FILTER_ALL,
)
from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, GUI_DOF_LABELS
from stablewalk.ui.dashboard_interpretability import METRIC_HELP
from stablewalk.ui.scientific_labels import (
    LABEL_CADENCE_SHORT,
    LABEL_CONTACT_VGRF_SECTION,
    LABEL_VIRTUAL_GRF_PANEL,
)
from stablewalk.ui.tk.metric_help import add_metric_help_icon
from stablewalk.ui.theme import (
    ACCENT,
    ACCENT_ALT,
    BG,
    BORDER,
    ELEVATED,
    EMPTY_OVERVIEW_JOINT_INSPECT,
    EMPTY_SELECT_DOF_TABLE,
    EMPTY_SELECT_DOF_CHART,
    EMPTY_VIDEO_TEXT,
    FONT_MONO_SM,
    FONT_METRIC,
    FONT_METRIC_TITLE,
    FONT_METRIC_VALUE,
    FONT_METRIC_VALUE_ACCENT,
    FONT_PANEL_HEADER,
    FONT_UI,
    FONT_UI_SM,
    FONT_UI_XS,
    INFO,
    MUTED,
    DASHBOARD_CARD_PAD,
    DASHBOARD_GUTTER,
    DASHBOARD_MAIN_PAD,
    DASHBOARD_SIDE_PAD,
    DASHBOARD_VIZ_CARD_PAD,
    OVERVIEW_COL_GUTTER,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    PANEL_HOVER,
    REFRESH_INTERVAL_CHOICES,
    TEXT,
    TEXT_SECONDARY,
    WARNING,
    REFRESH_INTERVAL_DEFAULT,
    DOF_TRAJ_DOT_COLOR,
    DOF_TRAJ_START_COLOR,
    OPENSIM_PANEL_MIN_HEIGHT,
    OPENSIM_STATUS_SCROLL_HEIGHT,
    SELECTION_BG,
    SIDEBAR_MIN_WIDTH,
    SIDEBAR_WIDTH,
    SURFACE,
    ORANGE,
    configure_demo_overlay,
    configure_video_placeholder,
    create_tooltip,
)
from stablewalk.ui.colors import SIDE_LEFT, SIDE_RIGHT
from stablewalk.ui.tk.clip_viewport import install_clipped_viewport
from stablewalk.ui.tk.dashboard_sections import (
    SEC1_TRAJ_VIZ_ROW_MINSIZE as _OVERVIEW_TRAJ_VIZ_ROW_MINSIZE,
    SEC1_VIZ_ROW_MINSIZE as _OVERVIEW_VIZ_ROW_MINSIZE,
)
from stablewalk.ui.viewers.dof_trajectory_3d import setup_single_dof_trajectory_axes
from stablewalk.ui.viewers.gait_skeleton_renderer import (
    DEFAULT_SKELETON_DISPLAY_MODE,
    LABEL_TO_SKELETON_MODE,
    MODE_TO_SKELETON_LABEL,
    SKELETON_MODE_LABELS,
    SKELETON_PANEL_MARGINS,
    _layout_skeleton_figure,
    relayout_skeleton_viewport,
    setup_skeleton_axes,
)

# Placeholder shown in the "Add a point" dropdown
ADD_POINT_PLACEHOLDER = "Add joint\u2026"

# Primary visual area: Original Video + 3D Reconstruction (50/50).
# Summary docks below — weights mirror dashboard_sections.SEC1_*.
_TOP_VIDEO_WEIGHT = 50
_TOP_SKELETON_WEIGHT = 50
_SIDEBAR_WEIGHT = 0
_BOTTOM_ANALYSIS_WEIGHT = 1
_BOTTOM_TABLE_WEIGHT = 0
_TABLE_PANEL_WIDTH = 0
_TABLE_PANEL_WIDTH_EXPANDED = 520
_DASH_VIZ_ROW_WEIGHT = 58
_DASH_ANALYTICS_ROW_WEIGHT = 42
_DASH_VIZ_ROW_MINSIZE = 0
_DASH_ANALYTICS_ROW_MINSIZE = 0
_SIDEBAR_MIN_WIDTH = SIDEBAR_MIN_WIDTH
_OVERVIEW_TRAJ_CANVAS_MINSIZE = 240
_OVERVIEW_COL_GUTTER = OVERVIEW_COL_GUTTER
_ANALYTICS_ANALYSIS_MINSIZE = 260
_ANALYTICS_KNEE_MINSIZE = 220
_ANALYTICS_TABLE_MINSIZE = 0
_ANALYSIS_GRAPH_INNER_PAD = (2, 2, 2, 2)
_OVERVIEW_TRAJ_GRAPH_INNER_PAD = (2, 2, 2, 2)
_ANALYSIS_INFO_STRIP_PAD = (2, 3)
_ANALYSIS_PANEL_PAD = DASHBOARD_CARD_PAD
_ANALYSIS_GRAPH_BOTTOM_PAD = 0
_ANALYSIS_PANEL_PLAYBACK_GAP = 8
_ANALYSIS_LEFT_PANEL_WIDTH = 204
# Width of the explanation panel placed to the right of the 3D cube.
_ANALYSIS_EXPLAIN_WIDTH = 176
_ANALYSIS_GRAPH_CANVAS_STICKY = "nsew"
_ANALYSIS_GRAPH_HEADER_ROW = 0
_ANALYSIS_GRAPH_CANVAS_ROW = 1
_ANALYSIS_GRAPH_CAPTION_ROW = 2
_ANALYSIS_BODY_ROW_WEIGHT = 1
_TRAJ_FIG_HEIGHT_SCALE = 1.0
# The 3D graph row expands by weight; keep only a small floor so it never forces
# its panel to overflow (which previously pushed the plot over the chart below).
_MOTION_GRAPH_ROW_MINSIZE = 250
# Foot contact + gait phase + Estimated Virtual GRF — three stacked plots.
CONTACT_GAIT_CHART_MIN_H = 420
_MOTION_TOP_ROW_MIN_H = 480
_MOTION_KNEE_PANEL_MIN_W = 300
_MOTION_TRAJ_PANEL_MIN_W = 460
_MOTION_METRICS_PANEL_MIN_W = 190
MOTION_TOP_COLUMN_WEIGHTS = (3, 5, 2)
_OVERVIEW_TRAJ_DOCK_MINSIZE = 300
_OVERVIEW_TRAJ_RESIZE_DEBOUNCE_MS = 60
_FIGURE_RESIZE_DEBOUNCE_MS = 50
_TABLE_PANEL_PAD = DASHBOARD_CARD_PAD
_SKELETON_PANEL_PAD = DASHBOARD_VIZ_CARD_PAD
_SKELETON_CANVAS_PAD = (0, 0, 0, 0)
_SKELETON_HEADER_PAD = (0, 0)
_ANALYTICS_SECONDARY_SLOTS = 4
_ANALYTICS_ADVANCED_COORD_SLOTS = 4

_BODY_CHECKLIST_HEIGHT = 1
_ANALYTICS_SUMMARY_SLOTS = 4
_TABLE_TREE_HEIGHT = 12
# Readable column widths: core columns fit without horizontal scrolling.
_TABLE_COL_WIDTHS_COMPACT: dict[str, int] = {
    "time": 58,
    "frame": 52,
    "dof": 112,
    "x": 60,
    "y": 60,
    "z": 60,
    "speed": 64,
    "foot_clearance": 92,
    "contact_status": 100,
}


def _install_motion_analysis_scroll(gui, parent: ttk.Frame) -> ttk.Frame:
    """
    Vertical scroll for Motion Analysis graphs (knee / path / contact+vGRF).

    Joint Selection stays outside this shell so it remains a compact footer.
    Uses local ``_motion_scroll_*`` attrs — not the Advanced-tab scroll.
    """
    from stablewalk.ui.theme import BG

    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)

    outer = ttk.Frame(parent)
    outer.grid(row=0, column=0, sticky="nsew")
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(0, weight=1)

    canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, borderwidth=0, bd=0)
    vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    canvas.configure(yscrollcommand=vsb.set)

    scroll_content = ttk.Frame(canvas)
    inner_id = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
    scroll_content.columnconfigure(0, weight=1)
    scroll_content.rowconfigure(0, weight=1)

    sections = ttk.Frame(scroll_content)
    sections.grid(row=0, column=0, sticky="nsew")
    sections.columnconfigure(0, weight=1)
    sections.rowconfigure(0, weight=1)

    spacer = ttk.Frame(scroll_content, height=PAD_SM)
    spacer.grid(row=1, column=0, sticky="ew")
    spacer.grid_propagate(False)

    gui._motion_scroll_outer = outer
    gui._motion_scroll_canvas = canvas
    gui._motion_scrollbar = vsb
    gui._motion_scroll_inner = scroll_content
    gui._motion_scroll_window_id = inner_id
    gui._motion_scroll_sections = sections

    def _sync(_event: object | None = None) -> None:
        try:
            canvas.update_idletasks()
            bbox = canvas.bbox("all")
            if bbox is not None:
                canvas.configure(scrollregion=bbox)
            cw = max(int(canvas.winfo_width()), 1)
            ch = max(int(canvas.winfo_height()), 1)
            # Measure natural graph requirements independently from the current
            # canvas-window height; using scroll_content.winfo_reqheight() here
            # creates a growth feedback loop when weighted rows are expanded.
            requested_h = max(
                int(
                    getattr(
                        gui,
                        "_motion_content_min_height",
                        sections.winfo_reqheight(),
                    )
                )
                + int(spacer.winfo_reqheight()),
                1,
            )
            content_h = max(ch, requested_h)
            canvas.itemconfigure(inner_id, width=cw, height=content_h)
            needs_scroll = requested_h > ch + 2
            if needs_scroll:
                if not vsb.winfo_ismapped():
                    vsb.grid(row=0, column=1, sticky="ns")
            else:
                if vsb.winfo_ismapped():
                    vsb.grid_remove()
                canvas.yview_moveto(0.0)
        except tk.TclError:
            pass

    gui._sync_motion_scroll = _sync
    scroll_content.bind("<Configure>", _sync, add="+")
    canvas.bind("<Configure>", _sync, add="+")

    def _wheel(event: tk.Event) -> str | None:
        if getattr(event, "delta", None):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"
        return None

    def _bind_wheel(widget: tk.Misc) -> None:
        widget.bind("<MouseWheel>", _wheel, add="+")
        try:
            for child in widget.winfo_children():
                _bind_wheel(child)
        except tk.TclError:
            pass

    gui._motion_bind_wheel = _bind_wheel
    for widget in (canvas, scroll_content, sections):
        widget.bind("<MouseWheel>", _wheel, add="+")
    canvas.bind("<Enter>", lambda _e: canvas.focus_set(), add="+")
    return sections


def _hide_trajectory_debug_placeholder(gui) -> None:
    """Hide the layout debug label once the trajectory canvas is confirmed visible."""
    lbl = getattr(gui, "lbl_traj_graph_debug", None)
    if lbl is None:
        return
    try:
        lbl.grid_remove()
    except tk.TclError:
        pass


def _draw_trajectory_startup_test(gui) -> None:
    """
    Draw a hardcoded 3D path at GUI init to prove the Motion Analysis graph area
    is visible before any session or joint data is loaded.
    """
    if not hasattr(gui, "fig_dof_traj") or not hasattr(gui, "canvas_dof_traj"):
        return
    from stablewalk.coordinates.coordinate_map import axis_labels_canonical

    ax = gui.ax_dof_traj
    ax.cla()
    if hasattr(ax, "_stablewalk_traj_artists"):
        del ax._stablewalk_traj_artists
    if hasattr(ax, "_stablewalk_stable_viewport"):
        del ax._stablewalk_stable_viewport
    ax._stablewalk_plot_legend = None

    labels = axis_labels_canonical()
    ax.set_facecolor(PANEL)
    gui.fig_dof_traj.patch.set_facecolor(PANEL)
    ax.set_xlabel(labels["x"], color=TEXT, fontsize=8)
    ax.set_ylabel(labels["y"], color=TEXT, fontsize=8)
    ax.set_zlabel(labels["z"], color=TEXT, fontsize=8)

    x = [0.0, 0.02, 0.04, 0.03]
    y = [0.0, 0.03, 0.06, 0.09]
    z = [0.0, 0.01, 0.03, 0.05]
    ax.plot(x, y, z, color=ACCENT, linewidth=2.0, label="Path")
    ax.scatter(x[0], y[0], z[0], color=DOF_TRAJ_START_COLOR, s=48, label="Start", depthshade=False)
    ax.scatter(x[-1], y[-1], z[-1], color=DOF_TRAJ_DOT_COLOR, s=48, label="Now", depthshade=False)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax.view_init(elev=20.0, azim=-60.0)

    from stablewalk.ui.viewers.dof_trajectory_3d import relayout_single_dof_viewport

    relayout_single_dof_viewport(ax)
    _ensure_trajectory_canvas_gridded(gui.canvas_dof_traj)
    _fit_trajectory_figure(
        gui.canvas_dof_traj,
        gui.fig_dof_traj,
        ax,
        graph_host=getattr(gui, "dof_analysis_graph_canvas_host", None),
    )
    gui.canvas_dof_traj.draw()
    gui._traj_startup_test_drawn = True
    _hide_trajectory_debug_placeholder(gui)


def _fit_figure_to_host(
    canvas: FigureCanvasTkAgg,
    fig: Figure,
    *,
    host: tk.Misc | None = None,
    margins: dict[str, float] | None = None,
    min_px: int = 100,
) -> bool:
    """Expand a matplotlib figure to fill its host — no empty side bands."""
    # Re-entrancy guard: Configure → resize → Configure must not recurse.
    if getattr(canvas, "_stablewalk_fitting", False):
        return False
    setattr(canvas, "_stablewalk_fitting", True)
    try:
        widget = canvas.get_tk_widget()
        target = host if host is not None else widget.master
        if target is None:
            target = widget
        try:
            target.update_idletasks()
            width = int(target.winfo_width())
            height = int(target.winfo_height())
        except tk.TclError:
            return False
        if width < min_px or height < min_px:
            try:
                width = max(width, int(widget.winfo_width()))
                height = max(height, int(widget.winfo_height()))
            except tk.TclError:
                return False
        if width < min_px or height < min_px:
            return False
        dpi = float(fig.get_dpi() or 100.0)
        try:
            cur_w, cur_h = fig.get_size_inches()
            if abs(cur_w * dpi - width) < 2.0 and abs(cur_h * dpi - height) < 2.0:
                return True
        except Exception:
            pass
        try:
            fig.set_size_inches(width / dpi, height / dpi, forward=False)
        except Exception:
            return False
        # Prefer TkAgg resize without pinning widget width/height — configuring
        # the widget fires <Configure> and can recurse into this fit path.
        try:
            canvas.resize(type("E", (), {"width": int(width), "height": int(height)})())
        except (TypeError, AttributeError, tk.TclError):
            try:
                fig.set_size_inches(width / dpi, height / dpi, forward=True)
            except Exception:
                return False
        if margins is not None:
            try:
                fig.subplots_adjust(**margins)
            except Exception:
                pass
        try:
            canvas.draw_idle()
        except Exception:
            pass
        return True
    finally:
        setattr(canvas, "_stablewalk_fitting", False)


def _bind_figure_resize(
    canvas: FigureCanvasTkAgg,
    fig: Figure,
    *,
    margins: dict[str, float] | None = None,
    min_px: int = 100,
) -> None:
    """Resize a matplotlib figure to fill its host (keeps plots usable)."""
    widget = canvas.get_tk_widget()
    host = widget.master
    state: dict[str, object] = {"job": None}

    def _apply() -> None:
        state["job"] = None
        _fit_figure_to_host(
            canvas, fig, host=host, margins=margins, min_px=min_px
        )

    def _on_resize(event: tk.Event | object | None = None) -> None:
        # Ignore Configure events fired by our own canvas resize.
        if getattr(canvas, "_stablewalk_fitting", False):
            return
        # When bound on the canvas widget, only react to real size changes.
        if event is not None and getattr(event, "widget", None) is widget:
            w = int(getattr(event, "width", 0) or 0)
            h = int(getattr(event, "height", 0) or 0)
            if w < min_px or h < min_px:
                return
        job = state.get("job")
        if job is not None:
            try:
                widget.after_cancel(job)  # type: ignore[arg-type]
            except Exception:
                pass
        try:
            state["job"] = widget.after(_FIGURE_RESIZE_DEBOUNCE_MS, _apply)
        except Exception:
            _apply()

    # Host-only binding avoids the widget.configure → Configure feedback loop.
    if host is not None:
        try:
            host.bind("<Configure>", _on_resize, add="+")
        except tk.TclError:
            widget.bind("<Configure>", _on_resize, add="+")
    else:
        widget.bind("<Configure>", _on_resize, add="+")


def _bind_skeleton_figure_resize(
    canvas: FigureCanvasTkAgg,
    fig: Figure,
    ax,
    *,
    min_px: int = 100,
    pad: tuple[int, int, int, int] = _SKELETON_CANVAS_PAD,
) -> None:
    """Resize the skeleton figure and preserve equal-aspect layout."""
    state: dict[str, object | None] = {"job": None, "event": None}

    def _fit(event: tk.Event | object) -> None:
        widget = canvas.get_tk_widget()
        host = widget.master
        width = int(getattr(event, "width", 0) or 0)
        height = int(getattr(event, "height", 0) or 0)
        if host is not None:
            host.update_idletasks()
            host_w = host.winfo_width()
            host_h = host.winfo_height()
            if host_w >= min_px:
                width = host_w
            if host_h >= min_px:
                height = host_h
        pad_l, pad_t, pad_r, pad_b = pad
        width = max(min_px, width - pad_l - pad_r)
        height = max(min_px, height - pad_t - pad_b)
        if width < min_px or height < min_px:
            return
        try:
            canvas.resize(type("E", (), {"width": width, "height": height})())
        except (TypeError, AttributeError, tk.TclError):
            dpi = fig.get_dpi()
            fig.set_size_inches(width / dpi, height / dpi, forward=True)
        relayout_skeleton_viewport(ax)
        canvas.draw_idle()

    def _on_resize(event: tk.Event) -> None:
        state["event"] = event
        widget = canvas.get_tk_widget()
        previous = state["job"]
        if previous is not None:
            try:
                widget.after_cancel(previous)  # type: ignore[arg-type]
            except (tk.TclError, ValueError):
                pass

        def _apply() -> None:
            state["job"] = None
            pending = state["event"]
            if pending is not None:
                _fit(pending)

        state["job"] = widget.after(_OVERVIEW_TRAJ_RESIZE_DEBOUNCE_MS, _apply)

    widget = canvas.get_tk_widget()
    widget.configure(bg=PANEL, highlightthickness=0)
    widget.bind("<Configure>", _on_resize)

    host = widget.master
    if host is not None:
        host.bind("<Configure>", _on_resize)


def _fit_trajectory_figure(
    canvas: FigureCanvasTkAgg,
    fig: Figure,
    ax,
    *,
    min_px: int = 40,
    graph_host: tk.Misc | None = None,
) -> bool:
    """Resize the 3D figure to exactly fill the graph host (no empty bands)."""
    if getattr(canvas, "_stablewalk_fitting", False):
        return False
    setattr(canvas, "_stablewalk_fitting", True)
    try:
        return _fit_trajectory_figure_impl(
            canvas, fig, ax, min_px=min_px, graph_host=graph_host
        )
    finally:
        setattr(canvas, "_stablewalk_fitting", False)


def _fit_trajectory_figure_impl(
    canvas: FigureCanvasTkAgg,
    fig: Figure,
    ax,
    *,
    min_px: int = 40,
    graph_host: tk.Misc | None = None,
) -> bool:
    """Implementation of trajectory figure fit (caller holds re-entrancy lock)."""
    widget = canvas.get_tk_widget()
    root = widget.winfo_toplevel()
    root.update_idletasks()
    widget.update_idletasks()

    host = graph_host if graph_host is not None else widget.master
    overview_dock = bool(getattr(ax, "_stablewalk_overview_dock", False))
    if overview_dock:
        pad_l, pad_t, pad_r, pad_b = _OVERVIEW_TRAJ_GRAPH_INNER_PAD
    else:
        pad_l, pad_t, pad_r, pad_b = _ANALYSIS_GRAPH_INNER_PAD
    host_w = 0
    host_h = 0

    if host is not None:
        host.update_idletasks()
        host_w = int(host.winfo_width())
        host_h = int(host.winfo_height())

    if host_w < min_px or host_h < min_px:
        host_w = max(host_w, int(widget.winfo_width()))
        host_h = max(host_h, int(widget.winfo_height()))

    if host_w < min_px or host_h < min_px:
        return False

    width = max(min_px, host_w - pad_l - pad_r)
    height = max(min_px, host_h - pad_t - pad_b)

    if _TRAJ_FIG_HEIGHT_SCALE < 1.0:
        height = max(min_px, int(height * _TRAJ_FIG_HEIGHT_SCALE))

    # Never grow a cached figure beyond the live host (avoids first-paint clipping
    # from the tall default Overview figsize).
    dpi = float(fig.get_dpi()) or 100.0
    try:
        cur_w, cur_h = fig.get_size_inches()
        if abs(cur_w * dpi - width) < 2.0 and abs(cur_h * dpi - height) < 2.0:
            return True
    except Exception:
        pass
    fig.set_size_inches(width / dpi, height / dpi, forward=False)

    # Let matplotlib map the widget's pixel size to the figure: under Tk display
    # scaling (Windows 125%/150% DPI) it divides by the device-pixel-ratio so
    # the Agg buffer still renders at the widget's true pixel size and fills it.
    try:
        canvas.resize(type("E", (), {"width": int(width), "height": int(height)})())
    except (TypeError, AttributeError, tk.TclError):
        fig.set_size_inches(width / dpi, height / dpi, forward=True)

    # Pin size only when it actually changed — avoids Configure feedback loops.
    try:
        cur_tw = int(widget.winfo_width())
        cur_th = int(widget.winfo_height())
        if abs(cur_tw - width) >= 2 or abs(cur_th - height) >= 2:
            widget.configure(
                width=int(width),
                height=int(height),
                bg=PANEL,
                highlightthickness=0,
            )
        else:
            widget.configure(bg=PANEL, highlightthickness=0)
    except tk.TclError:
        try:
            widget.configure(bg=PANEL, highlightthickness=0)
        except tk.TclError:
            pass

    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)

    from stablewalk.ui.viewers.dof_trajectory_3d import relayout_single_dof_viewport

    relayout_single_dof_viewport(ax)
    return True


def _ensure_trajectory_canvas_gridded(
    canvas: FigureCanvasTkAgg,
    *,
    overview_dock: bool = False,
) -> None:
    """Grid the trajectory canvas once — never re-pack during resize."""
    widget = canvas.get_tk_widget()
    pad_l, pad_t, pad_r, pad_b = (
        _OVERVIEW_TRAJ_GRAPH_INNER_PAD if overview_dock else _ANALYSIS_GRAPH_INNER_PAD
    )
    try:
        if not widget.grid_info():
            widget.grid(
                row=0,
                column=0,
                sticky=_ANALYSIS_GRAPH_CANVAS_STICKY,
                padx=(pad_l, pad_r),
                pady=(pad_t, pad_b),
            )
        widget.lift()
    except tk.TclError:
        widget.grid(
            row=0,
            column=0,
            sticky=_ANALYSIS_GRAPH_CANVAS_STICKY,
            padx=(pad_l, pad_r),
            pady=(pad_t, pad_b),
        )
        try:
            widget.lift()
        except tk.TclError:
            pass


def _bind_trajectory_figure_resize(
    canvas: FigureCanvasTkAgg,
    fig: Figure,
    ax,
    *,
    min_px: int = 40,
    graph_host: tk.Misc | None = None,
    extra_hosts: tuple[tk.Misc, ...] | None = None,
    debounce_ms: int | None = None,
) -> None:
    """Resize the selected-point 3D figure when the canvas or host changes size."""
    overview_dock = bool(getattr(ax, "_stablewalk_overview_dock", False))
    delay = (
        _OVERVIEW_TRAJ_RESIZE_DEBOUNCE_MS
        if debounce_ms is None
        else max(int(debounce_ms), 0)
    )
    state = {"after_id": None}

    def _reflow(_event: object | None = None) -> None:
        state["after_id"] = None
        if _fit_trajectory_figure(
            canvas,
            fig,
            ax,
            min_px=min_px,
            graph_host=graph_host,
        ):
            _ensure_trajectory_canvas_gridded(canvas, overview_dock=overview_dock)
            canvas.draw_idle()

    def _on_resize(event: tk.Event) -> None:
        width = int(getattr(event, "width", 0) or 0)
        height = int(getattr(event, "height", 0) or 0)
        # Host Configure events can report a single axis during mid-layout; only
        # ignore clearly empty sizes. The fit helper re-reads host winfo_*.
        if width < 2 and height < 2:
            return
        if delay <= 0:
            _reflow()
            return
        widget = canvas.get_tk_widget()
        root = widget.winfo_toplevel()
        prev = state["after_id"]
        if prev is not None:
            try:
                root.after_cancel(prev)
            except (tk.TclError, ValueError):
                pass
        state["after_id"] = root.after(delay, _reflow)

    widget = canvas.get_tk_widget()
    widget.configure(bg=PANEL, highlightthickness=0)
    widget.bind("<Configure>", _on_resize)

    host = graph_host if graph_host is not None else widget.master
    if host is not None and host is not widget:
        host.bind("<Configure>", _on_resize)
    inner = getattr(host, "master", None)
    if inner is not None and inner is not host and inner is not widget:
        inner.bind("<Configure>", _on_resize)
    for extra in extra_hosts or ():
        if extra is not None and extra is not host and extra is not widget:
            extra.bind("<Configure>", _on_resize)


def _metric_grid_cell(
    parent: tk.Misc,
    *,
    row: int,
    column: int,
    title: str,
    compact: bool = False,
    accent: bool = False,
) -> tuple[tk.Label, tk.Label]:
    """One titled value cell inside a metrics grid (roomy, readable cards)."""
    pad = 2 if compact else 2
    cell_pad = (0, 0) if compact else (1, 1)
    title_pad = (1, 0) if compact else (2, 0)
    value_pad = (0, 1) if compact else (0, 3)
    inner_x = 4 if compact else 8
    grid_kwargs = dict(row=row, column=column, sticky="nsew", padx=pad, pady=cell_pad)

    if accent and not compact:
        shell = tk.Frame(parent, bg=ORANGE, highlightthickness=0)
        shell.grid(**grid_kwargs)
        cell = tk.Frame(shell, bg=ELEVATED, highlightthickness=0)
        cell.pack(fill=tk.BOTH, expand=True, padx=(3, 0))
    else:
        cell = tk.Frame(
            parent,
            bg=PANEL if compact else ELEVATED,
            highlightthickness=0 if compact else 1,
            highlightbackground=BORDER,
            highlightcolor=BORDER,
        )
        cell.grid(**grid_kwargs)

    cell_bg = PANEL if compact else ELEVATED
    title_lbl = tk.Label(
        cell,
        text=title,
        bg=cell_bg,
        fg=MUTED,
        font=FONT_METRIC_TITLE if not compact else FONT_UI_XS,
        anchor="w",
    )
    title_lbl.pack(fill=tk.X, padx=inner_x, pady=title_pad)
    value_lbl = tk.Label(
        cell,
        text="—",
        bg=cell_bg,
        fg=TEXT,
        font=FONT_METRIC_VALUE,
        anchor="w",
    )
    value_lbl.pack(fill=tk.X, padx=inner_x, pady=value_pad)
    return title_lbl, value_lbl


def _metric_slot_host(title_lbl: tk.Label, grid_parent: tk.Misc) -> tk.Misc:
    """Outer grid cell for a metric slot (handles accent shell wrapper)."""
    cell = title_lbl.master
    if cell.master is grid_parent:
        return cell
    return cell.master


def _foot_card_metric(
    parent: tk.Misc,
    *,
    row: int,
    column: int,
    title: str,
    compact: bool = False,
) -> tuple[tk.Label, tk.Label]:
    """One secondary metric inside the Foot Analysis card."""
    cell = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)
    cell.grid(row=row, column=column, sticky="nsew", padx=3, pady=2)
    title_font = FONT_UI_XS
    value_font = FONT_UI_XS if compact else FONT_METRIC
    title_lbl = tk.Label(
        cell,
        text=title,
        bg=ELEVATED,
        fg=MUTED,
        font=title_font,
        anchor="w",
    )
    title_lbl.pack(fill=tk.X, padx=5, pady=(4, 0))
    value_lbl = tk.Label(
        cell,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=value_font,
        anchor="w",
    )
    value_lbl.pack(fill=tk.X, padx=5, pady=(0, 4))
    return title_lbl, value_lbl


def _foot_card_row(
    parent: tk.Misc,
    *,
    row: int,
    label: str,
    compact_value: bool = False,
) -> tk.Label:
    """One label + value pair inside the Foot Analysis card."""
    cell = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)
    cell.grid(row=row, column=0, sticky="ew", padx=8, pady=(0, 4))
    tk.Label(
        cell,
        text=label,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(anchor="w")
    value_lbl = tk.Label(
        cell,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_XS if compact_value else FONT_METRIC,
        anchor="w",
        wraplength=_ANALYSIS_LEFT_PANEL_WIDTH - 20,
        justify=tk.LEFT,
    )
    value_lbl.pack(anchor="w", pady=(1, 0))
    return value_lbl


def _foot_card_inline_pair(
    parent: tk.Misc,
    *,
    row: int,
    left_label: str,
    right_label: str,
) -> tuple[tk.Label, tk.Label]:
    """Frame and Time on one row."""
    row_frame = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)
    row_frame.grid(row=row, column=0, sticky="ew", padx=8, pady=(0, 8))
    row_frame.columnconfigure(0, weight=1)
    row_frame.columnconfigure(1, weight=1)

    def _cell(col: int, title: str) -> tk.Label:
        cell = tk.Frame(row_frame, bg=ELEVATED, highlightthickness=0)
        cell.grid(row=0, column=col, sticky="ew")
        tk.Label(
            cell,
            text=title,
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        ).pack(anchor="w")
        lbl = tk.Label(
            cell,
            text="—",
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_UI_XS,
            anchor="w",
        )
        lbl.pack(anchor="w", pady=(1, 0))
        return lbl

    return _cell(0, left_label), _cell(1, right_label)


def _build_ground_clearance_strip(gui, parent: tk.Misc) -> tk.Frame:
    """Compact bilateral foot readout under the skeleton view."""
    from stablewalk.ui.theme import FONT_METRIC_VALUE, FONT_UI_XS

    strip = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    gui.ground_clearance_strip = strip

    inner = tk.Frame(strip, bg=PANEL, highlightthickness=0)
    inner.pack(fill=tk.X, padx=2, pady=2)

    tk.Label(inner, text="L", bg=PANEL, fg=SIDE_LEFT, font=FONT_UI_XS).pack(
        side=tk.LEFT, padx=(0, PAD_XS // 2)
    )
    gui.lbl_ground_clearance_left = tk.Label(
        inner,
        text="\u2014",
        bg=PANEL,
        fg=SIDE_LEFT,
        font=FONT_METRIC_VALUE,
        anchor="w",
    )
    gui.lbl_ground_clearance_left.pack(side=tk.LEFT, padx=(0, PAD_MD))

    tk.Label(inner, text="R", bg=PANEL, fg=SIDE_RIGHT, font=FONT_UI_XS).pack(
        side=tk.LEFT, padx=(0, PAD_XS // 2)
    )
    gui.lbl_ground_clearance_right = tk.Label(
        inner,
        text="\u2014",
        bg=PANEL,
        fg=SIDE_RIGHT,
        font=FONT_METRIC_VALUE,
        anchor="w",
    )
    gui.lbl_ground_clearance_right.pack(side=tk.LEFT)

    # Phase / scale kept for logic and tooltips — not shown in the main strip.
    gui.lbl_ground_clearance_phase = tk.Label(strip, text="", bg=PANEL)
    gui.lbl_ground_clearance_scale = tk.Label(strip, text="", bg=PANEL)

    strip.grid_remove()
    return strip


def _build_foot_analysis_card(gui, parent: tk.Misc) -> tk.Frame:
    """Session min/max/avg clearance — current clearance lives in the metrics row."""
    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.columnconfigure(0, weight=1)

    card = tk.Frame(
        host,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    card.grid(row=0, column=0, sticky="ew")
    card.columnconfigure(0, weight=1)
    gui.dof_analysis_foot_card = card

    tk.Label(
        card,
        text="Session clearance",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_METRIC,
        anchor="w",
    ).grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

    clearance_grid = tk.Frame(card, bg=ELEVATED)
    clearance_grid.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 2))
    for col in range(3):
        clearance_grid.columnconfigure(col, weight=1)

    _, gui.foot_card_min_lbl = _foot_card_metric(
        clearance_grid, row=0, column=0, title="Min", compact=True
    )
    _, gui.foot_card_max_lbl = _foot_card_metric(
        clearance_grid, row=0, column=1, title="Max", compact=True
    )
    _, gui.foot_card_avg_lbl = _foot_card_metric(
        clearance_grid, row=0, column=2, title="Average", compact=True
    )

    gui.lbl_foot_ground_note = tk.Label(
        card,
        text="",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=_ANALYSIS_LEFT_PANEL_WIDTH - 16,
    )
    gui.lbl_foot_ground_note.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 8))

    gui.dof_analysis_foot_host = host
    gui.lbl_foot_clearance_hero = None
    gui.lbl_foot_clearance_quality = None
    gui.lbl_foot_clearance_definition = None
    gui.lbl_foot_distance_ground = None
    gui.lbl_foot_contact_status = None
    gui.foot_card_point_lbl = None
    gui.lbl_foot_analysis_explanation = gui.lbl_foot_ground_note
    gui.lbl_dof_analysis_foot_title = None
    gui.foot_analysis_secondary_slots = []
    gui.lbl_foot_card_speed = None
    gui.lbl_foot_card_position = None
    gui.foot_card_frame_lbl = None
    gui.foot_card_time_lbl = None
    host.grid_remove()
    return host


def _build_metric_grid(
    parent: tk.Misc,
    *,
    columns: int,
    slot_titles: tuple[str, ...],
    row: int = 0,
    compact: bool = False,
    accent_first: bool = False,
    accent_index: int | None = None,
) -> list[tuple[tk.Label, tk.Label]]:
    """Create a row of metric cells and return (title, value) label pairs."""
    accent_col = accent_index if accent_index is not None else (0 if accent_first else -1)
    for col in range(columns):
        weight = 2 if col == 0 and accent_first else 1
        parent.columnconfigure(col, weight=weight, uniform="metric")
    slots: list[tuple[tk.Label, tk.Label]] = []
    for col, title in enumerate(slot_titles):
        slots.append(
            _metric_grid_cell(
                parent,
                row=row,
                column=col,
                title=title,
                compact=compact,
                accent=accent_col >= 0 and col == accent_col,
            )
        )
    return slots


def _card_title(label: str) -> str:
    """Consistent LabelFrame title spacing."""
    return f" {label.strip()} "


def _sidebar_rule(parent: tk.Misc) -> None:
    tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, pady=(2, 2))


def _analysis_sidebar_heading(parent: tk.Misc, text: str, *, first: bool = False) -> None:
    tk.Label(
        parent,
        text=text,
        bg=PANEL,
        fg=ACCENT_ALT,
        font=FONT_UI_XS,
        anchor=tk.W,
    ).pack(fill=tk.X, pady=(0 if first else PAD_XS, 2))


def _analysis_sidebar_line(
    parent: tk.Misc,
    text: str,
    *,
    color: str | None = None,
    wrap: int = 160,
) -> tk.Label:
    lbl = tk.Label(
        parent,
        text=text,
        bg=PANEL,
        fg=color or TEXT_SECONDARY,
        font=FONT_UI_XS,
        justify=tk.LEFT,
        anchor=tk.W,
        wraplength=wrap,
    )
    lbl.pack(fill=tk.X, pady=(0, 0))
    return lbl


def _analysis_sidebar_metric(
    parent: tk.Misc,
    title: str,
    *,
    wrap: int = 160,
) -> tuple[tk.Label, tk.Label]:
    row = tk.Frame(parent, bg=PANEL)
    row.pack(fill=tk.X, pady=(0, 1))
    title_lbl = tk.Label(
        row,
        text=title,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor=tk.W,
    )
    title_lbl.pack(anchor=tk.W)
    value_lbl = tk.Label(
        row,
        text="—",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_XS,
        justify=tk.LEFT,
        anchor=tk.W,
        wraplength=wrap,
    )
    value_lbl.pack(anchor=tk.W, fill=tk.X)
    return title_lbl, value_lbl


def _compact_interp_label(parent: tk.Misc, gui, attr: str) -> tk.Label:
    lbl = tk.Label(
        parent,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=280,
    )
    setattr(gui, attr, lbl)
    return lbl


def _metric_title_row(parent: tk.Misc, title: str, help_key: str) -> tk.Frame:
    row = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    tk.Label(row, text=title, bg=PANEL, fg=MUTED, font=FONT_UI_XS, anchor="w").pack(
        side=tk.LEFT
    )
    add_metric_help_icon(row, title, METRIC_HELP[help_key])
    return row


def _analysis_sidebar_section(parent: tk.Misc, text: str, *, first: bool = False) -> None:
    tk.Label(
        parent,
        text=text,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor=tk.W,
    ).pack(fill=tk.X, pady=(0 if first else 4, 1))


def _build_compact_graph_legend(parent: tk.Misc) -> list[tk.Label]:
    """Inline color legend: green start, cyan path, red current, blue end."""
    from stablewalk.ui.theme import (
        DOF_TRAJ_DOT_COLOR,
        DOF_TRAJ_END_COLOR,
        DOF_TRAJ_PATH_COLOR,
        DOF_TRAJ_START_COLOR,
        FONT_CHART_LEGEND,
    )

    try:
        bg = parent.cget("bg")
    except tk.TclError:
        bg = ELEVATED

    items = (
        (DOF_TRAJ_START_COLOR, "Start"),
        (DOF_TRAJ_PATH_COLOR, "Path"),
        (DOF_TRAJ_DOT_COLOR, "Current"),
        (DOF_TRAJ_END_COLOR, "End"),
    )
    labels: list[tk.Label] = []
    for index, (color, name) in enumerate(items):
        if index:
            tk.Label(parent, text="·", bg=bg, fg=MUTED, font=FONT_UI_SM).pack(
                side=tk.LEFT, padx=(6, 6)
            )
        chip = tk.Frame(parent, bg=bg)
        chip.pack(side=tk.LEFT)
        tk.Label(chip, text="●", bg=bg, fg=color, font=FONT_UI_SM).pack(side=tk.LEFT)
        lbl = tk.Label(chip, text=name, bg=bg, fg=TEXT_SECONDARY, font=FONT_CHART_LEGEND)
        lbl.pack(side=tk.LEFT, padx=(1, 0))
        labels.append(lbl)
    return labels


def _build_overview_joint_information_panel(gui, parent: tk.Misc) -> None:
    """Collapsible live Joint Information grid under the 3D path."""
    from stablewalk.ui.joint_information import JOINT_INFO_FIELD_ORDER, JOINT_INFO_TITLES

    panel = ttk.LabelFrame(
        parent,
        text=_card_title("Joint Information"),
        style="Card.TLabelframe",
        padding=(PAD_XS, PAD_XS),
    )
    panel.grid(row=3, column=0, sticky="ew", pady=(PAD_XS, 0))
    panel.columnconfigure(0, weight=1)
    gui.overview_joint_info_panel = panel

    gui._overview_joint_info_expanded = False
    toggle = ttk.Button(
        panel,
        text="▶ Details",
        style="Compact.TButton",
        width=10,
        takefocus=False,
    )
    toggle.pack(anchor="e")

    body = tk.Frame(panel, bg=PANEL, highlightthickness=0)
    gui._overview_joint_info_body = body

    grid = tk.Frame(body, bg=PANEL, highlightthickness=0)
    grid.pack(fill=tk.X, expand=False)
    gui.overview_joint_info_grid = grid

    columns = 4
    for col in range(columns):
        grid.columnconfigure(col, weight=1, uniform="joint_info")

    value_labels: dict[str, tk.Label] = {}
    for index, field_id in enumerate(JOINT_INFO_FIELD_ORDER):
        row, col = divmod(index, columns)
        _title_lbl, value_lbl = _metric_grid_cell(
            grid,
            row=row,
            column=col,
            title=JOINT_INFO_TITLES[field_id],
            compact=True,
        )
        value_labels[field_id] = value_lbl
    gui.overview_joint_info_values = value_labels

    gui.lbl_overview_joint_info_hint = tk.Label(
        body,
        text="Select a joint on the skeleton — values update with video playback.",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
    )
    gui.lbl_overview_joint_info_hint.pack(fill=tk.X, pady=(4, 0))

    def _toggle_details() -> None:
        expanded = not bool(getattr(gui, "_overview_joint_info_expanded", False))
        gui._overview_joint_info_expanded = expanded
        if expanded:
            body.pack(fill=tk.X, expand=False, pady=(PAD_XS, 0))
            toggle.configure(text="▼ Details")
        else:
            body.pack_forget()
            toggle.configure(text="▶ Details")
        # The canvas owns all remaining row height; refit after the row changes.
        try:
            parent.update_idletasks()
            fit = getattr(gui, "_render_overview_traj_canvas", None)
            if callable(fit):
                fit(force=True)
        except (tk.TclError, TypeError):
            pass

    toggle.configure(command=_toggle_details)
    gui.btn_overview_joint_info_toggle = toggle


def _build_overview_joint_motion_analysis_panel(gui, parent: tk.Misc) -> None:
    """Collapsible Joint Graphs strip — never replaces Video | Skeleton | 3D Path.

    Starts collapsed so the three Overview visualizations own the viewport.
    Expand via ▶ Joint Graphs to reveal the six synchronized motion plots.
    """
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    from stablewalk.ui.tk.dashboard_sections import apply_overview_joint_motion_row_weight
    from stablewalk.ui.viewers.chart_interactions import (
        attach_chart_interactions,
        build_chart_tools_bar,
    )

    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.grid(row=2, column=0, sticky="nsew", pady=(PAD_XS, 0))
    host.columnconfigure(0, weight=1)
    host.rowconfigure(1, weight=1)
    gui.overview_joint_motion_panel = host
    # Collapsed by default — Video | Skeleton | 3D Path keep the height budget.
    gui._overview_joint_motion_expanded = False

    header = tk.Frame(host, bg=PANEL, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)

    tk.Label(
        header,
        text="Joint Graphs",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).grid(row=0, column=0, sticky="w")

    body = ttk.LabelFrame(
        host,
        text=_card_title("Joint Motion Analysis"),
        style="Card.TLabelframe",
        padding=(PAD_XS, PAD_XS),
    )
    body.columnconfigure(0, weight=1)
    body.rowconfigure(1, weight=1)
    gui._overview_joint_motion_body = body

    tools_host = tk.Frame(body, bg=PANEL, highlightthickness=0)
    tools_host.grid(row=0, column=0, sticky="ew", pady=(0, 2))
    tools_host.columnconfigure(0, weight=1)

    # Compact figure — only shown when the strip is expanded.
    gui.fig_joint_motion = Figure(figsize=(10.0, 3.2), dpi=100, facecolor=PANEL)
    gui.fig_joint_motion.subplots(2, 3, sharex=True)
    gui.canvas_joint_motion = FigureCanvasTkAgg(gui.fig_joint_motion, master=body)
    canvas_widget = gui.canvas_joint_motion.get_tk_widget()
    canvas_widget.configure(bg=PANEL, highlightthickness=0)
    canvas_widget.grid(row=1, column=0, sticky="nsew")

    attach_chart_interactions(gui.fig_joint_motion, gui.canvas_joint_motion)

    tools = build_chart_tools_bar(
        gui,
        tools_host,
        fig_attr="fig_joint_motion",
        canvas_attr="canvas_joint_motion",
        export_name="joint_motion_analysis",
    )
    tools.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _export_csv() -> None:
        export = getattr(gui, "_export_joint_motion_csv", None)
        if export is not None:
            export()

    btn_csv = ttk.Button(
        tools, text="CSV", width=5, command=_export_csv, takefocus=False
    )
    btn_csv.pack(side=tk.LEFT, padx=(2, PAD_XS), pady=2)
    create_tooltip(btn_csv, "Export joint motion series as CSV")

    gui.joint_motion_tools = tools
    gui._joint_motion_bundle = None
    gui._joint_motion_bundle_key = None

    gui.lbl_joint_motion_hint = tk.Label(
        body,
        text=(
            "Select joints on the skeleton or Motion checklist — "
            "six synchronized plots update with video playback."
        ),
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_joint_motion_hint.grid(row=2, column=0, sticky="ew", pady=(2, 0))

    def _apply_expanded(expanded: bool) -> None:
        gui._overview_joint_motion_expanded = expanded
        if expanded:
            body.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
        else:
            body.grid_remove()
        apply_overview_joint_motion_row_weight(parent, expanded=expanded, row=2)
        btn = getattr(gui, "_btn_overview_joint_motion_toggle", None)
        if btn is not None:
            btn.configure(
                text="▼ Joint Graphs" if expanded else "▶ Joint Graphs"
            )
        try:
            parent.update_idletasks()
        except tk.TclError:
            pass
        schedule = getattr(gui, "_schedule_overview_media_refit", None)
        if callable(schedule):
            try:
                schedule()
            except Exception:
                pass
        # Redraw charts when revealing so axes match the new canvas size.
        if expanded:
            update = getattr(gui, "_update_joint_motion_analysis_chart", None)
            if callable(update):
                try:
                    update()
                except Exception:
                    pass

    def _toggle() -> None:
        _apply_expanded(not bool(getattr(gui, "_overview_joint_motion_expanded", False)))

    toggle = ttk.Button(
        header,
        text="▶ Joint Graphs",
        style="Compact.TButton",
        width=14,
        command=_toggle,
        takefocus=False,
    )
    toggle.grid(row=0, column=1, sticky="e", padx=(PAD_XS, 0))
    gui._btn_overview_joint_motion_toggle = toggle
    gui._toggle_overview_joint_motion = _toggle
    gui._apply_overview_joint_motion_expanded = _apply_expanded

    # Start collapsed — do not steal height from Video | Skeleton | 3D Path.
    body.grid_remove()
    apply_overview_joint_motion_row_weight(parent, expanded=False, row=2)


def _build_analysis_graph_explainer(gui, parent: tk.Misc) -> tk.Frame:
    """Compact side panel beside the 3D trajectory cube."""
    from stablewalk.ui.theme import (
        DOF_TRAJ_DOT_COLOR,
        DOF_TRAJ_END_COLOR,
        DOF_TRAJ_PATH_COLOR,
        DOF_TRAJ_START_COLOR,
    )

    container = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    container.pack(fill=tk.X, expand=False, pady=(2, 0))
    container.columnconfigure(0, weight=1)

    panel = tk.Frame(container, bg=PANEL, highlightthickness=0)
    panel.grid(row=0, column=0, sticky="new")
    panel.columnconfigure(0, weight=1)

    floor_card = tk.Frame(panel, bg=ELEVATED, highlightthickness=0)
    floor_card.grid(row=0, column=0, sticky="ew", pady=(0, 4))
    gui.dof_graph_floor_card = floor_card
    tk.Label(
        floor_card,
        text="Floor distance",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(fill=tk.X, padx=6, pady=(4, 0))
    gui.lbl_dof_graph_floor_value = tk.Label(
        floor_card,
        text="\u2014",
        bg=ELEVATED,
        fg=ACCENT,
        font=FONT_METRIC_VALUE,
        anchor="w",
        justify=tk.LEFT,
        wraplength=_ANALYSIS_EXPLAIN_WIDTH - 16,
    )
    gui.lbl_dof_graph_floor_value.pack(fill=tk.X, padx=6, pady=(0, 4))
    gui.lbl_dof_graph_floor_note = tk.Label(floor_card, text="", bg=ELEVATED)
    gui.lbl_dof_graph_floor_range = tk.Label(floor_card, text="", bg=ELEVATED)

    overlay_row = tk.Frame(panel, bg=PANEL, highlightthickness=0)
    overlay_row.grid(row=1, column=0, sticky="ew", pady=(0, 2))
    gui.var_foot_clearance_graph = tk.StringVar(value="Off")
    gui.cmb_foot_clearance_graph = ttk.Combobox(
        overlay_row,
        textvariable=gui.var_foot_clearance_graph,
        values=("Off", "Left foot", "Right foot"),
        state="readonly",
        width=12,
    )
    gui.cmb_foot_clearance_graph.pack(side=tk.LEFT, fill=tk.X, expand=True)
    gui.cmb_foot_clearance_graph.bind(
        "<<ComboboxSelected>>",
        lambda _e: gui._refresh_foot_clearance_graph_hint(),
    )
    gui.lbl_foot_clearance_graph_range = tk.Label(panel, text="", bg=PANEL)

    legend = tk.Frame(panel, bg=PANEL, highlightthickness=0)
    legend.grid(row=2, column=0, sticky="ew", pady=(2, 0))
    legend_items = (
        (DOF_TRAJ_START_COLOR, "Start"),
        (DOF_TRAJ_PATH_COLOR, "Path"),
        (DOF_TRAJ_DOT_COLOR, "Current"),
        (DOF_TRAJ_END_COLOR, "End"),
    )
    gui.dof_graph_explain_legend_labels = []
    for index, (color, name) in enumerate(legend_items):
        if index:
            tk.Label(legend, text="\u00b7", bg=PANEL, fg=MUTED, font=FONT_UI_XS).pack(
                side=tk.LEFT, padx=2
            )
        chip = tk.Frame(legend, bg=PANEL, highlightthickness=0)
        chip.pack(side=tk.LEFT)
        tk.Label(chip, text="\u25cf", bg=PANEL, fg=color, font=FONT_UI_XS).pack(
            side=tk.LEFT, padx=(0, 2)
        )
        lbl = tk.Label(chip, text=name, bg=PANEL, fg=TEXT_SECONDARY, font=FONT_UI_XS)
        lbl.pack(side=tk.LEFT)
        gui.dof_graph_explain_legend_labels.append(lbl)

    gui.lbl_dof_graph_explain_body = tk.Label(
        panel,
        text="",
        bg=PANEL,
        fg=TEXT_SECONDARY,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=_ANALYSIS_EXPLAIN_WIDTH - 8,
    )
    gui.lbl_dof_graph_explain_body.grid(row=3, column=0, sticky="ew", pady=(4, 0))

    gui.lbl_dof_traj_interp_compact = _compact_interp_label(
        panel, gui, "lbl_dof_traj_interp_compact"
    )
    gui.lbl_dof_traj_interp_compact.grid(row=4, column=0, sticky="ew", pady=(2, 0))

    from stablewalk.ui.theme import DOF_ANALYSIS_GRAPH_CAPTION_GENERAL

    gui.lbl_dof_analysis_graph_caption = tk.Label(
        panel,
        text=DOF_ANALYSIS_GRAPH_CAPTION_GENERAL,
        bg=PANEL,
        fg=ACCENT_ALT,
        font=FONT_UI_XS,
        anchor="w",
    )

    gui.dof_analysis_graph_explainer = panel
    return panel


def _build_analysis_graph_chrome(gui, parent: tk.Misc) -> tk.Frame:
    """Reserved graph chrome host (legend lives in the summary bar)."""
    chrome = tk.Frame(parent, bg=PANEL)
    gui.dof_graph_legend_labels = getattr(gui, "dof_graph_legend_labels", [])
    gui.lbl_dof_graph_annotation = None
    gui.lbl_dof_graph_ground = None
    gui.dof_graph_chrome = chrome
    gui.lbl_dof_analysis_graph_caption = None
    return chrome


def _build_analysis_graph_caption(gui, parent: tk.Misc) -> tk.Label:
    """Short header shown above the 3D trajectory canvas ("3D point movement")."""
    from stablewalk.ui.theme import DOF_ANALYSIS_GRAPH_CAPTION_GENERAL

    lbl = tk.Label(
        parent,
        text=DOF_ANALYSIS_GRAPH_CAPTION_GENERAL,
        bg=PANEL,
        fg=ACCENT_ALT,
        font=FONT_PANEL_HEADER,
        justify=tk.LEFT,
        anchor=tk.W,
    )
    gui.lbl_dof_analysis_graph_caption = lbl
    lbl.grid(row=0, column=0, sticky="w")
    lbl.grid_remove()
    return lbl


def _build_analysis_sidebar(gui, parent: tk.Misc) -> tk.Frame:
    """Left column: color legend for the 3D movement graph."""
    from stablewalk.ui.selected_point_analysis import legend_lines_for_panel
    from stablewalk.ui.theme import DOF_ANALYSIS_LEGEND_TITLE

    sidebar = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    sidebar.columnconfigure(0, weight=1)

    gui.dof_analysis_legend_labels: list[tk.Label] = []

    _analysis_sidebar_section(sidebar, DOF_ANALYSIS_LEGEND_TITLE, first=True)
    legend_host = tk.Frame(sidebar, bg=PANEL)
    legend_host.pack(fill=tk.X, anchor=tk.NW, padx=2, pady=(0, 0))
    for line, accent in legend_lines_for_panel():
        lbl = _analysis_sidebar_line(legend_host, line, color=accent, wrap=200)
        gui.dof_analysis_legend_labels.append(lbl)

    def _sync_sidebar_wrap(_event: object | None = None) -> None:
        width = sidebar.winfo_width()
        if width < 48:
            return
        line_wrap = max(96, width - 8)
        for lbl in getattr(gui, "dof_analysis_legend_labels", []):
            try:
                lbl.configure(wraplength=line_wrap)
            except tk.TclError:
                pass

    sidebar.bind("<Configure>", _sync_sidebar_wrap)
    gui._sync_analysis_sidebar_wrap = _sync_sidebar_wrap

    gui.dof_analysis_sidebar = sidebar
    gui.dof_analysis_legend_section = legend_host
    gui.dof_analysis_movement_slots = []
    return sidebar


def _build_analysis_metrics_panel(
    gui,
    parent: tk.Misc,
    *,
    path_metrics_parent: tk.Misc | None = None,
) -> tk.Frame:
    """Primary metrics row + collapsible advanced measurements."""
    from stablewalk.ui.theme import (
        DOF_ANALYSIS_MODE_FOOT,
        DOF_ANALYSIS_MODE_FOOT_HINT,
        DOF_ANALYSIS_MODE_GENERAL,
        DOF_ANALYSIS_MODE_GENERAL_HINT,
    )

    section = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    section.columnconfigure(0, weight=1)

    mode_holder = tk.Frame(section, bg=PANEL, highlightthickness=0)
    gui.lbl_dof_analysis_mode = tk.Label(mode_holder, text="", bg=PANEL, fg=ACCENT)
    gui.lbl_dof_analysis_mode_hint = tk.Label(mode_holder, text="", bg=PANEL, fg=MUTED)
    gui._analysis_mode_labels = (
        DOF_ANALYSIS_MODE_GENERAL,
        DOF_ANALYSIS_MODE_GENERAL_HINT,
        DOF_ANALYSIS_MODE_FOOT,
        DOF_ANALYSIS_MODE_FOOT_HINT,
    )

    summary_bar = tk.Frame(section, bg=PANEL, highlightthickness=0)
    summary_bar.grid(row=0, column=0, sticky="ew", pady=(0, 2))
    summary_bar.columnconfigure(0, weight=1)

    metrics_host = tk.Frame(summary_bar, bg=PANEL, highlightthickness=0)
    metrics_host.grid(row=0, column=0, sticky="ew")
    metrics_host.columnconfigure(0, weight=1)

    metrics_parent = path_metrics_parent if path_metrics_parent is not None else metrics_host
    primary_host = tk.Frame(metrics_parent, bg=PANEL, highlightthickness=0)
    primary_host.grid(row=0, column=0, sticky="ew")
    gui.dof_analysis_summary_slots = _build_metric_grid(
        primary_host,
        columns=_ANALYTICS_SUMMARY_SLOTS,
        row=0,
        slot_titles=(
            "Selected point",
            "Time",
            "Speed",
            "Path length",
        ),
        compact=True,
        accent_first=True,
    )
    gui.dof_analysis_point_value_lbl = gui.dof_analysis_summary_slots[0][1]
    gui.dof_analysis_point_value_lbl.configure(
        wraplength=150,
        justify=tk.LEFT,
        anchor="w",
    )
    gui.dof_analysis_identity_slots = gui.dof_analysis_summary_slots[:2]
    gui.dof_analysis_kinematics_slots = gui.dof_analysis_summary_slots[2:]
    gui.dof_analysis_summary_hosts = [
        _metric_slot_host(title_lbl, primary_host)
        for title_lbl, _ in gui.dof_analysis_summary_slots
    ]
    gui.dof_analysis_primary_metrics_host = primary_host

    toggle_row = tk.Frame(metrics_host, bg=PANEL, highlightthickness=0)
    toggle_row.grid(row=1, column=0, sticky="ew", pady=(2, 0))
    gui.btn_toggle_joint_advanced = ttk.Button(
        toggle_row,
        text="Detailed Joint Data \u25be",
        style="Compact.TButton",
        command=gui._toggle_joint_advanced_data,
    )
    gui.btn_toggle_joint_advanced.pack(side=tk.LEFT)
    gui.btn_collected_data = ttk.Button(
        toggle_row,
        text="Data (0)",
        style="Compact.TButton",
        command=gui._open_collected_data_dialog,
    )
    gui.btn_collected_data.pack(side=tk.RIGHT)
    gui._joint_advanced_visible = False

    advanced_host = tk.Frame(metrics_host, bg=PANEL, highlightthickness=0)
    advanced_host.grid(row=2, column=0, sticky="ew", pady=(2, 0))
    advanced_host.grid_remove()
    gui.dof_analysis_advanced_host = advanced_host

    gui.dof_analysis_advanced_coord_slots = _build_metric_grid(
        advanced_host,
        columns=_ANALYTICS_ADVANCED_COORD_SLOTS,
        row=0,
        slot_titles=("Frame", "X (m)", "Y (m)", "Z (m)"),
        compact=True,
    )
    gui.dof_analysis_advanced_coord_hosts = [
        _metric_slot_host(title_lbl, advanced_host)
        for title_lbl, _ in gui.dof_analysis_advanced_coord_slots
    ]

    secondary_host = tk.Frame(advanced_host, bg=PANEL, highlightthickness=0)
    secondary_host.grid(row=1, column=0, sticky="ew", pady=(PAD_XS, 0))
    gui.dof_analysis_secondary_host = secondary_host
    gui.dof_analysis_derived_row = secondary_host
    gui.dof_analysis_derived_slots = _build_metric_grid(
        secondary_host,
        columns=_ANALYTICS_SECONDARY_SLOTS,
        row=0,
        slot_titles=(
            "Path Length",
            "Delta from Start",
            "Vertical Position",
            "",
        ),
        compact=True,
    )
    gui.dof_analysis_secondary_hosts = [
        _metric_slot_host(title_lbl, secondary_host)
        for title_lbl, _ in gui.dof_analysis_derived_slots
    ]

    gui.dof_graph_legend_labels = getattr(gui, "dof_graph_legend_labels", [])
    gui.btn_export_analysis = getattr(gui, "btn_export_analysis", None)
    gui.lbl_dof_analysis_movement_title = tk.Label(mode_holder, text="", bg=PANEL)

    gui.dof_analysis_metric_slots = (
        gui.dof_analysis_summary_slots
        + gui.dof_analysis_advanced_coord_slots
        + gui.dof_analysis_derived_slots
    )
    gui.dof_analysis_metrics_section = section
    gui.dof_analysis_header = section
    gui.dof_analysis_summary = summary_bar
    gui.dof_analysis_base_summary = summary_bar
    gui.dof_analysis_base_slots = gui.dof_analysis_kinematics_slots
    gui.dof_analysis_context_slot = None
    gui.dof_analysis_joint_summary = None
    gui.dof_analysis_joint_slots = []
    gui.dof_analysis_slots = gui.dof_analysis_metric_slots
    gui.dof_analysis_status = None
    gui.dof_analysis_export_row = None
    gui.dof_analysis_details_row = None
    gui.lbl_dof_analysis_clarification = None
    gui.lbl_dof_analysis_guide = None
    gui.lbl_dof_analysis_red_marker_clarification = None
    gui.lbl_dof_analysis_interpretation = None
    gui.lbl_dof_analysis_metrics_link = None
    gui.lbl_dof_analysis_panel_summary = None
    gui.lbl_dof_analysis_movement_summary = None
    gui.dof_analysis_graph_pad = _ANALYSIS_GRAPH_INNER_PAD

    return section


def _build_analysis_derived_row(gui, parent: tk.Misc) -> tk.Frame:
    """Legacy hook — secondary metrics row is built inside the summary panel."""
    return getattr(gui, "dof_analysis_secondary_host", parent)


def _analysis_metric_cell(
    parent: tk.Misc,
    *,
    row: int,
    column: int,
    title: str,
    gui,
    attr: str,
) -> None:
    """One titled value cell in the selected-point analysis summary grid."""
    cell = tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    cell.grid(row=row, column=column, sticky="nsew", padx=2, pady=2)
    title_lbl = tk.Label(
        cell,
        text=title,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    title_lbl.pack(fill=tk.X, padx=PAD_XS, pady=(PAD_XS, 0))
    value_lbl = tk.Label(
        cell,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_MONO_SM,
        anchor="w",
    )
    value_lbl.pack(fill=tk.X, padx=PAD_XS, pady=(0, PAD_XS))
    if not hasattr(gui, "dof_analysis_slots"):
        gui.dof_analysis_slots = []
    gui.dof_analysis_slots.append((title_lbl, value_lbl))
    setattr(gui, attr, value_lbl)


# Saved inspection points (Movement Checkpoints)
CHECKPOINT_COLUMNS: tuple[str, ...] = (
    "idx",
    "frame",
    "time",
    "dof",
    "joint",
    "x",
    "y",
    "z",
    "velocity",
    "angle",
)
CHECKPOINT_HEADINGS: dict[str, str] = {
    "idx": "#",
    "frame": "Frame",
    "time": "Time (s)",
    "dof": "Point",
    "joint": "Joint",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "velocity": "Velocity",
    "angle": "Angle",
}
CHECKPOINT_WIDTHS: dict[str, int] = {
    "idx": 34,
    "frame": 52,
    "time": 60,
    "dof": 88,
    "joint": 88,
    "x": 58,
    "y": 58,
    "z": 58,
    "velocity": 72,
    "angle": 58,
}


def _make_data_tree(
    parent: ttk.Frame,
    columns: tuple[str, ...],
    headings: dict[str, str],
    *,
    col_widths: dict[str, int] | None = None,
    height: int = 6,
    text_cols: frozenset[str] = frozenset({"dof", "joint", "direction"}),
    style: str = "Large.Treeview",
) -> ttk.Treeview:
    """Scrollable data table with horizontal and vertical bars."""
    host = ttk.Frame(parent)
    host.pack(fill=tk.BOTH, expand=True)
    host.columnconfigure(0, weight=1)
    host.rowconfigure(0, weight=1)

    tree = ttk.Treeview(
        host,
        columns=columns,
        show="headings",
        height=height,
        style=style,
    )
    widths = col_widths or {}
    for col in columns:
        tree.heading(col, text=headings.get(col, col))
        w = widths.get(col, 72)
        anchor = tk.W if col in text_cols else tk.E
        stretch = col in {"joint", "contact_state", "dof", "contact_status"}
        tree.column(col, width=w, anchor=anchor, minwidth=44, stretch=stretch)
    tree._sw_column_widths = dict(widths)  # type: ignore[attr-defined]
    tree.grid(row=0, column=0, sticky="nsew")

    vsb = ttk.Scrollbar(host, orient=tk.VERTICAL, command=tree.yview)
    hsb = ttk.Scrollbar(host, orient=tk.HORIZONTAL, command=tree.xview)
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    tree.tag_configure("even", background=PANEL)
    tree.tag_configure("odd", background=ELEVATED)
    tree.tag_configure("selected", background=SELECTION_BG, foreground=TEXT)
    tree.tag_configure("low_confidence", background="#3d1f28", foreground=TEXT)
    tree._sw_host = host  # type: ignore[attr-defined]
    return tree


# Combined width budget shared by the video and skeleton columns; the sidebar
# keeps its own fixed weight so only these two reflow with the clip.
_TOP_SPAN_WEIGHT = _TOP_VIDEO_WEIGHT + _TOP_SKELETON_WEIGHT
_TOP_VIDEO_WEIGHT_MIN = 30
_TOP_VIDEO_WEIGHT_MAX = _TOP_SPAN_WEIGHT - 28


def apply_top_row_aspect(gui, aspect: float | None) -> None:
    """Record video aspect only — Overview view mode owns column weights.

    Column mutation for portrait clips previously narrowed the video panel and
    made clips look side-cropped. Contain-fit letterboxing handles aspect
    inside the fixed 50/50 Side-by-Side columns.
    """
    gui._top_video_weight = _TOP_VIDEO_WEIGHT
    if aspect is not None:
        gui._recorded_video_aspect = float(aspect)


def build_dashboard_layout(gui) -> None:
    """Build the main dashboard grid and attach widgets to ``gui``."""
    from stablewalk.ui.tk.dashboard_responsive import (
        finalize_responsive_dashboard,
        install_responsive_shell,
    )
    from stablewalk.ui.tk.dashboard_sections import (
        SECTION_KINEMATIC_TITLE,
        SECTION_VISUAL_TITLE,
        SEC1_SKELETON_WEIGHT,
        SEC1_TRAJ_PATH_WEIGHT,
        SEC1_TRAJ_PANEL_MINWIDTH,
        SEC1_TRAJ_VIZ_ROW_MINSIZE,
        SEC1_VIDEO_WEIGHT,
        SEC1_VIZ_ROW_WEIGHT,
        bind_panel_word_wrap,
        build_data_export_section,
        build_gait_metrics_section,
        build_real_to_sim_section,
        build_gait_summary_cards,
        build_overview_metrics_row,
    )

    # Hidden host for widgets kept for logic/menu but not shown on the dashboard.
    if not hasattr(gui, "_sidebar_hidden"):
        gui._sidebar_hidden = ttk.Frame(gui.root)

    main = ttk.Frame(gui.root, padding=DASHBOARD_MAIN_PAD)
    main.pack(fill=tk.BOTH, expand=True)
    gui._dashboard_main = main
    main.columnconfigure(0, weight=1)
    main.rowconfigure(0, weight=1)

    (
        tab_overview,
        tab_motion,
        tab_biomechanics,
        tab_results_summary,
        tab_compare,
        tab_advanced_content,
    ) = install_responsive_shell(gui, main)
    gui._dashboard_body = tab_overview
    gui._top_video_weight = _TOP_VIDEO_WEIGHT

    # ── Tab 1: Overview — professional visualization workspace ──────────────
    # Row 0: View Mode · Row 1: Video|Skeleton|Path · Row 2: Joint Graphs
    # (collapsed) · Row 3: Gait info (collapsed).
    from stablewalk.ui.tk.dashboard_overview_view_mode import (
        build_overview_view_mode_selector,
    )
    from stablewalk.ui.tk.dashboard_sections import apply_overview_joint_motion_row_weight

    tab_overview.columnconfigure(0, weight=1)
    tab_overview.rowconfigure(0, weight=0)  # view-mode toolbar
    tab_overview.rowconfigure(1, weight=SEC1_VIZ_ROW_WEIGHT)  # ~75%+ visuals
    apply_overview_joint_motion_row_weight(tab_overview, expanded=False, row=2)
    tab_overview.rowconfigure(3, weight=0)  # gait info starts collapsed

    view_mode_bar = build_overview_view_mode_selector(gui, tab_overview)
    view_mode_bar.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))

    section1 = ttk.Frame(tab_overview)
    section1.grid(row=1, column=0, sticky="nsew")
    section1.columnconfigure(0, weight=SEC1_VIDEO_WEIGHT, uniform="sec1", minsize=220)
    section1.columnconfigure(1, weight=SEC1_SKELETON_WEIGHT, uniform="sec1", minsize=220)
    section1.columnconfigure(
        2, weight=SEC1_TRAJ_PATH_WEIGHT, uniform="sec1", minsize=SEC1_TRAJ_PANEL_MINWIDTH
    )
    section1.rowconfigure(0, weight=1, minsize=SEC1_TRAJ_VIZ_ROW_MINSIZE)
    section1.rowconfigure(1, weight=0)  # unused summary strip
    gui._section_visual = section1
    gui._primary_viz_host = section1
    gui._primary_viz_content_row = 0
    gui._overview_summary_row = 1
    gui._viz_tabs_active = False
    gui._overview_summary_expanded = True

    _build_overview_joint_motion_analysis_panel(gui, tab_overview)

    overview_metrics = build_overview_metrics_row(gui, tab_overview)
    overview_metrics.grid(row=3, column=0, sticky="nsew", pady=(PAD_XS, 0))

    overview_traj_panel = ttk.LabelFrame(
        section1,
        text=_card_title("3D Joint Path"),
        style="Card.TLabelframe",
        padding=(PAD_XS, PAD_XS),
    )
    overview_traj_panel.grid(
        row=0,
        column=2,
        sticky="nsew",
        padx=(_OVERVIEW_COL_GUTTER, 0),
    )
    overview_traj_panel.columnconfigure(0, weight=1)
    # Title (LabelFrame text) → compact camera toolbar → canvas fills rest → slim footer.
    overview_traj_panel.rowconfigure(0, weight=0)  # camera toolbar
    overview_traj_panel.rowconfigure(
        1, weight=1, minsize=_OVERVIEW_TRAJ_CANVAS_MINSIZE
    )  # canvas
    overview_traj_panel.rowconfigure(2, weight=0)  # compact caption footer
    overview_traj_panel.rowconfigure(3, weight=0)  # live Joint Information
    gui.overview_traj_panel = overview_traj_panel
    gui.lbl_overview_traj_title = None

    from stablewalk.ui.tk.trajectory_camera_toolbar import build_trajectory_camera_toolbar

    gui.overview_traj_camera_bar = build_trajectory_camera_toolbar(
        gui,
        overview_traj_panel,
        ax_attr="ax_dof_traj_overview",
        canvas_attr="canvas_dof_traj_overview",
        compact=True,
    )
    gui.overview_traj_camera_bar.grid(row=0, column=0, sticky="ew", pady=(0, 2))

    gui.overview_traj_canvas_host = tk.Frame(
        overview_traj_panel, bg=PANEL, highlightthickness=0
    )
    gui.overview_traj_canvas_host.grid(row=1, column=0, sticky="nsew")
    gui.overview_traj_canvas_host.columnconfigure(0, weight=1)
    gui.overview_traj_canvas_host.rowconfigure(0, weight=1)

    # Compact readout under the graph (outside the Matplotlib canvas).
    footer = tk.Frame(overview_traj_panel, bg=PANEL, highlightthickness=0)
    footer.grid(row=2, column=0, sticky="ew", pady=(2, 0))
    gui.overview_traj_footer = footer

    gui.lbl_overview_traj_info = tk.Label(
        footer,
        text="Select DOF: click a joint on the 3D gait figure to inspect its path.",
        bg=PANEL,
        fg=MUTED,
        font=FONT_MONO_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=380,
    )
    gui.lbl_overview_traj_info.pack(fill=tk.X, anchor="w")

    legend_row = tk.Frame(footer, bg=PANEL, highlightthickness=0)
    legend_row.pack(fill=tk.X, anchor="w", pady=(3, 0))
    gui.overview_traj_legend_labels = _build_compact_graph_legend(legend_row)
    # Legacy binding target (hidden) — chip legend above is the visible key.
    gui.lbl_overview_traj_legend = tk.Label(footer, text="", bg=PANEL)
    gui.lbl_overview_traj_metrics = tk.Label(footer, text="", bg=PANEL)
    gui.lbl_overview_traj_detail = tk.Label(footer, text="", bg=PANEL)
    gui.lbl_overview_traj_video = tk.Label(footer, text="", bg=PANEL)
    gui.lbl_overview_category_note = tk.Label(footer, text="", bg=PANEL)
    gui.lbl_overview_traj_motion = None
    gui.overview_traj_mount = gui.overview_traj_canvas_host
    # Path column is always mapped in Side-by-Side (empty until a joint is picked).
    gui._overview_traj_dock_visible = True

    _build_overview_joint_information_panel(gui, overview_traj_panel)

    viz_tab_bar = ttk.Frame(section1)
    viz_tab_bar.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, PAD_XS))
    viz_tab_bar.grid_remove()
    gui._viz_tab_bar = viz_tab_bar
    gui._viz_tab_var = tk.StringVar(value="video")

    def _on_viz_tab_selected() -> None:
        from stablewalk.ui.tk.dashboard_responsive import apply_viz_tab_visibility

        apply_viz_tab_visibility(gui)

    for tab_label, tab_value in (("Video", "video"), ("3D Reconstruction", "3d")):
        ttk.Radiobutton(
            viz_tab_bar,
            text=tab_label,
            variable=gui._viz_tab_var,
            value=tab_value,
            command=_on_viz_tab_selected,
        ).pack(side=tk.LEFT, padx=(0, 4))

    video_frame = ttk.LabelFrame(
        section1,
        text=_card_title("Original Video"),
        style="Card.TLabelframe",
        padding=DASHBOARD_VIZ_CARD_PAD,
    )
    video_frame.grid(
        row=0,
        column=0,
        sticky="nsew",
        padx=(0, _OVERVIEW_COL_GUTTER),
    )
    gui.video_frame = video_frame
    video_frame.columnconfigure(0, weight=1)
    video_frame.rowconfigure(0, weight=0)  # demo title
    video_frame.rowconfigure(1, weight=0)  # demo meta
    video_frame.rowconfigure(2, weight=0)  # overlay controls
    video_frame.rowconfigure(3, weight=1)  # video viewport

    from stablewalk.ui.tk.overlay_mode_controls import build_overlay_mode_controls

    def _on_overlay_controls_changed() -> None:
        show = getattr(gui, "_show_pose_at", None)
        if show is None:
            return
        pos = getattr(gui, "_playback_pos", None)
        if pos is None and getattr(gui, "skeleton_player", None) is not None:
            pos = float(gui.skeleton_player.state.frame_float)
        try:
            show(pos if pos is not None else getattr(gui, "current_pos", 0), force_draw=True)
        except Exception:
            pass

    build_overlay_mode_controls(
        gui,
        video_frame,
        row=2,
        on_change=_on_overlay_controls_changed,
    )

    gui._video_clip_canvas, gui.video_display_host, gui._video_clip_window_id = (
        install_clipped_viewport(
            video_frame,
            bg=PANEL,
            row=3,
            column=0,
            sticky="nsew",
        )
    )
    gui.video_display_host.columnconfigure(0, weight=1)
    gui.video_display_host.rowconfigure(0, weight=1)

    gui.lbl_demo_analysis_title = tk.Label(
        video_frame,
        text="",
        bg=ELEVATED,
        fg=ACCENT,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        padx=4,
        pady=2,
    )
    gui.lbl_demo_analysis_title.grid(row=0, column=0, sticky="ew")
    gui.lbl_demo_analysis_title.grid_remove()

    demo_meta_row = tk.Frame(video_frame, bg=ELEVATED, highlightthickness=0)
    demo_meta_row.grid(row=1, column=0, sticky="ew")
    demo_meta_row.columnconfigure(0, weight=1)
    demo_meta_row.grid_remove()
    gui._demo_meta_row = demo_meta_row

    gui.lbl_demo_source_attribution = tk.Label(
        demo_meta_row,
        text="",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        padx=4,
        pady=0,
    )
    gui.lbl_demo_source_attribution.grid(row=0, column=0, sticky="ew")

    gui.btn_demo_video_info = ttk.Button(
        demo_meta_row,
        text="i",
        style="Compact.TButton",
        width=2,
        command=gui._show_demo_video_details,
    )
    gui.btn_demo_video_info.grid(row=0, column=1, sticky="e", padx=(2, 0))
    gui.btn_demo_video_info.grid_remove()

    gui.video_label = tk.Label(gui.video_display_host, text=EMPTY_VIDEO_TEXT)
    configure_video_placeholder(gui.video_label)
    gui.video_label.configure(anchor=tk.CENTER)
    gui.video_label.grid(row=0, column=0, sticky="nsew")

    gui._demo_overlay = tk.Label(gui.video_display_host, text="")
    configure_demo_overlay(gui._demo_overlay)
    gui._demo_overlay.grid(row=0, column=0, sticky="n", pady=(4, 0))
    gui._demo_overlay.grid_remove()

    # Floating, semi-transparent playback info bar overlaid on the video. It is
    # kept hidden until a session loads and is refreshed live during playback.
    from stablewalk.ui.tk.dashboard_sections import build_overview_playback_hud

    build_overview_playback_hud(gui, gui.video_display_host)
    gui._overview_hud_place_kwargs = dict(
        relx=0.5, rely=1.0, y=-8, anchor="s", relwidth=0.96
    )
    gui._overview_hud_shown = False
    gui._overview_playback_hud.place_forget()

    skel_frame = ttk.LabelFrame(
        section1,
        text=_card_title("3D Gait Reconstruction"),
        style="Card.TLabelframe",
        padding=_SKELETON_PANEL_PAD,
    )
    skel_frame.grid(
        row=0,
        column=1,
        sticky="nsew",
        padx=(_OVERVIEW_COL_GUTTER, _OVERVIEW_COL_GUTTER),
    )
    skel_frame.columnconfigure(0, weight=1)
    skel_frame.rowconfigure(0, weight=0)  # overlay control bar
    skel_frame.rowconfigure(1, weight=1)  # visualization (max space)
    skel_frame.rowconfigure(2, weight=0)  # foot clearance detail
    gui.skel_frame = skel_frame

    # ── Overlay control bar (grouped, icon-tagged) ───────────────────────────
    # Sits ABOVE the viewport and never overlaps the skeleton. Overlays are
    # grouped into Motion / Biomechanics / Environment; the display-mode selector
    # rides on the right so the whole bar stays on one line for max viz space.
    from stablewalk.ui.tk.skeleton_overlay_toolbar import build_overlay_control_bar

    overlay_bar, overlay_right_slot = build_overlay_control_bar(gui, skel_frame)
    overlay_bar.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))

    gui.skeleton_display_mode = tk.StringVar(
        value=MODE_TO_SKELETON_LABEL[DEFAULT_SKELETON_DISPLAY_MODE]
    )
    # Maximize packs first (RIGHT) so it stays at the far right of the ribbon.
    from stablewalk.ui.tk.dashboard_panel_maximize import install_maximize_button

    install_maximize_button(
        gui, overlay_right_slot, "skeleton", pack_side=tk.RIGHT, padx=(6, 0)
    )
    gui.cmb_skeleton_mode = ttk.Combobox(
        overlay_right_slot,
        textvariable=gui.skeleton_display_mode,
        values=SKELETON_MODE_LABELS,
        state="readonly",
        width=14,
    )
    gui.cmb_skeleton_mode.pack(side=tk.RIGHT, padx=(0, 4))
    gui.cmb_skeleton_mode.bind("<<ComboboxSelected>>", gui._on_skeleton_display_mode)
    gui.lbl_skeleton_status = None

    # Pure visualization host — no side rail, no overlapping chrome. The
    # canvas fills every remaining pixel under the overlay ribbon.
    gui.skel_canvas_host = tk.Frame(skel_frame, bg=PANEL, highlightthickness=0)
    gui.skel_canvas_host.grid(row=1, column=0, sticky="nsew")
    gui.skel_canvas_host.columnconfigure(0, weight=1)
    gui.skel_canvas_host.rowconfigure(0, weight=1)

    # Single-line foot-clearance strip under the viewport (never on the canvas).
    clearance_badges = tk.Frame(skel_frame, bg=PANEL, highlightthickness=0)
    _metric_title_row(clearance_badges, "Clearance", "foot_clearance").pack(
        side=tk.LEFT, padx=(0, 6)
    )
    tk.Label(clearance_badges, text="L", bg=PANEL, fg=SIDE_LEFT, font=FONT_UI_XS).pack(
        side=tk.LEFT, padx=(0, PAD_XS // 2)
    )
    gui.lbl_ground_clearance_left = tk.Label(
        clearance_badges,
        text="\u2014",
        bg=PANEL,
        fg=SIDE_LEFT,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_ground_clearance_left.pack(side=tk.LEFT, padx=(0, PAD_SM))
    tk.Label(clearance_badges, text="R", bg=PANEL, fg=SIDE_RIGHT, font=FONT_UI_XS).pack(
        side=tk.LEFT, padx=(0, PAD_XS // 2)
    )
    gui.lbl_ground_clearance_right = tk.Label(
        clearance_badges,
        text="\u2014",
        bg=PANEL,
        fg=SIDE_RIGHT,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_ground_clearance_right.pack(side=tk.LEFT)
    # State labels kept for data binding but not shown on the compact strip
    # (avoids "L — R — L: — R: —" duplication under the skeleton).
    gui.lbl_ground_clearance_left_state = tk.Label(
        clearance_badges,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_ground_clearance_right_state = tk.Label(
        clearance_badges,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.ground_clearance_strip = clearance_badges
    gui.lbl_ground_clearance_phase = tk.Label(clearance_badges, text="", bg=PANEL)
    gui.lbl_ground_clearance_scale = tk.Label(clearance_badges, text="", bg=PANEL)
    gui.lbl_clearance_interp_compact = _compact_interp_label(
        clearance_badges, gui, "lbl_clearance_interp_compact"
    )
    gui.lbl_clearance_interp_compact.pack_forget()
    clearance_badges.grid(row=2, column=0, sticky="ew", pady=(1, 0))
    clearance_badges.grid_remove()

    from stablewalk.ui.tk.dashboard_sections import build_foot_clearance_detail_panel

    # Build detail widgets for data binding / Details dialog, but parent them to a
    # hidden host so they never steal Overview visualization height.
    gui._foot_clearance_hidden_host = tk.Frame(gui._sidebar_hidden, bg=PANEL)
    gui._foot_clearance_detail_host = build_foot_clearance_detail_panel(
        gui, gui._foot_clearance_hidden_host
    )
    skel_frame.rowconfigure(2, weight=0)

    gui.fig_3d = Figure(figsize=(8.4, 7.2), dpi=100, facecolor=PANEL)
    gui.ax_3d = gui.fig_3d.add_subplot(111)
    setup_skeleton_axes(gui.ax_3d)
    _layout_skeleton_figure(gui.ax_3d)
    gui._skel_clip_canvas, gui._skel_plot_host, gui._skel_clip_window_id = (
        install_clipped_viewport(
            gui.skel_canvas_host,
            bg=PANEL,
            row=0,
            column=0,
            sticky="nsew",
            padx=(_SKELETON_CANVAS_PAD[0], _SKELETON_CANVAS_PAD[2]),
            pady=(_SKELETON_CANVAS_PAD[1], _SKELETON_CANVAS_PAD[3]),
        )
    )
    gui._skel_plot_host.columnconfigure(0, weight=1)
    gui._skel_plot_host.rowconfigure(0, weight=1)
    gui.canvas_3d = FigureCanvasTkAgg(gui.fig_3d, master=gui._skel_plot_host)
    skel_canvas = gui.canvas_3d.get_tk_widget()
    skel_canvas.configure(bg=PANEL, highlightthickness=0)
    skel_canvas.grid(row=0, column=0, sticky="nsew")
    _bind_skeleton_figure_resize(gui.canvas_3d, gui.fig_3d, gui.ax_3d)
    gui.canvas_3d.mpl_connect("pick_event", gui._on_skeleton_pick)
    gui.canvas_3d.mpl_connect("button_press_event", gui._on_skeleton_button_press)
    gui.canvas_3d.mpl_connect("motion_notify_event", gui._on_skeleton_motion)

    # ── Tab 2: Motion Analysis (knee | joint path | joint selection | data) ─
    section2 = ttk.Frame(tab_motion)
    section2.grid(row=0, column=0, sticky="nsew")
    tab_motion.rowconfigure(0, weight=1)
    tab_motion.columnconfigure(0, weight=1)
    gui._section_kinematic = section2

    from stablewalk.ui.tk.dashboard_biomechanics import build_biomechanics_tab

    build_biomechanics_tab(gui, tab_biomechanics)
    gui._section_biomechanics = tab_biomechanics

    from stablewalk.ui.tk.dashboard_summary import build_results_summary_tab

    build_results_summary_tab(gui, tab_results_summary)
    gui._section_results_summary = tab_results_summary

    from stablewalk.ui.tk.dashboard_compare import build_comparison_tab

    build_comparison_tab(gui, tab_compare)
    gui._section_compare = tab_compare

    # Scrollable analysis workspace: every graph keeps a readable minimum size.
    # Joint Selection stays outside the scroll shell as a compact fixed footer.
    section2.columnconfigure(0, weight=1)
    section2.rowconfigure(0, weight=1)
    section2.rowconfigure(1, weight=0)

    motion_joint_row = ttk.Frame(section2)
    motion_joint_row.grid(row=1, column=0, sticky="ew", pady=(PAD_XS, 0))
    motion_joint_row.columnconfigure(0, weight=1)
    gui._motion_joint_row = motion_joint_row

    motion_sections = _install_motion_analysis_scroll(gui, section2)
    bottom_row = ttk.Frame(motion_sections)
    bottom_row.grid(row=0, column=0, sticky="nsew")
    gui._motion_content_min_height = (
        _MOTION_TOP_ROW_MIN_H + CONTACT_GAIT_CHART_MIN_H + PAD_XS
    )
    bottom_row.configure(height=gui._motion_content_min_height)
    bottom_row.grid_propagate(False)
    gui._analysis_scroll_row = 1
    gui._analysis_scroll_outer = gui._motion_scroll_outer
    gui._analysis_scroll_canvas = gui._motion_scroll_canvas
    gui._analysis_scrollbar = gui._motion_scrollbar
    gui._analysis_scroll_inner = gui._motion_scroll_inner
    gui._analysis_scroll_window_id = gui._motion_scroll_window_id
    # Exact 30 / 50 / 20 top-row allocation on medium and maximized windows.
    bottom_row.columnconfigure(
        0,
        weight=MOTION_TOP_COLUMN_WEIGHTS[0],
        minsize=_MOTION_KNEE_PANEL_MIN_W,
        uniform="motion_top",
    )
    bottom_row.columnconfigure(
        1,
        weight=MOTION_TOP_COLUMN_WEIGHTS[1],
        minsize=_MOTION_TRAJ_PANEL_MIN_W,
        uniform="motion_top",
    )
    bottom_row.columnconfigure(
        2,
        weight=MOTION_TOP_COLUMN_WEIGHTS[2],
        minsize=_MOTION_METRICS_PANEL_MIN_W,
        uniform="motion_top",
    )
    bottom_row.rowconfigure(0, weight=3, minsize=_MOTION_TOP_ROW_MIN_H)
    bottom_row.rowconfigure(1, weight=2, minsize=CONTACT_GAIT_CHART_MIN_H)
    bottom_row.rowconfigure(2, weight=0, minsize=0)
    gui._dashboard_bottom_row = bottom_row
    gui.dashboard = bottom_row

    from stablewalk.ui.tk.dashboard_motion_metrics import build_motion_temporal_metrics_panel

    motion_metrics_panel = build_motion_temporal_metrics_panel(gui, bottom_row)
    motion_metrics_panel.grid(
        row=0,
        column=2,
        sticky="nsew",
        padx=(DASHBOARD_GUTTER, 0),
    )

    knee_panel = ttk.LabelFrame(
        bottom_row,
        text=_card_title("Knee Motion Analysis"),
        style="Card.TLabelframe",
        padding=(PAD_XS, PAD_XS),
    )
    knee_panel.grid(row=0, column=0, sticky="nsew", padx=(0, DASHBOARD_GUTTER))
    gui.knee_panel = knee_panel
    knee_panel.columnconfigure(0, weight=1)
    knee_panel.rowconfigure(0, weight=0)
    knee_panel.rowconfigure(1, weight=1)
    knee_panel.rowconfigure(2, weight=0)
    knee_panel.rowconfigure(3, weight=0)

    chart_toolbar = tk.Frame(knee_panel, bg=PANEL, highlightthickness=0)
    chart_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
    tk.Label(chart_toolbar, text="Mode", bg=PANEL, fg=MUTED, font=FONT_UI_SM).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    gui.var_knee_chart_axis = tk.StringVar(value="Video Time")
    gui.cmb_knee_chart_axis = ttk.Combobox(
        chart_toolbar,
        textvariable=gui.var_knee_chart_axis,
        values=("Video Time", "Gait Cycle %"),
        state="readonly",
        width=14,
    )
    gui.cmb_knee_chart_axis.pack(side=tk.LEFT)
    gui.cmb_knee_chart_axis.bind("<<ComboboxSelected>>", lambda _e: gui._on_knee_chart_mode_changed())

    gui._knee_source_toolbar = tk.Frame(chart_toolbar, bg=PANEL, highlightthickness=0)
    tk.Label(
        gui._knee_source_toolbar,
        text="Angle source",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
    ).pack(side=tk.LEFT, padx=(12, 6))
    gui.var_knee_angle_source = tk.StringVar(value="Auto")
    gui.cmb_knee_angle_source = ttk.Combobox(
        gui._knee_source_toolbar,
        textvariable=gui.var_knee_angle_source,
        values=("Auto", "Pose-derived", "OpenSim IK"),
        state="readonly",
        width=14,
    )
    gui.cmb_knee_angle_source.pack(side=tk.LEFT)
    gui.cmb_knee_angle_source.bind(
        "<<ComboboxSelected>>", lambda _e: gui._on_knee_angle_source_changed()
    )

    gui._knee_source_toolbar.pack_forget()
    gui.lbl_knee_angle_source = None

    # Direct chart host — nested clip-viewport inside Motion can stay 1×1 unmapped.
    gui.knee_chart_host = tk.Frame(knee_panel, bg=PANEL, highlightthickness=0)
    gui.knee_chart_host.grid(row=1, column=0, sticky="nsew")
    gui.knee_chart_host.columnconfigure(0, weight=1)
    gui.knee_chart_host.rowconfigure(0, weight=1)
    gui._knee_clip_canvas = None
    gui._knee_clip_window_id = None

    # Compact requested height; grid weights expand this in large windows.
    gui.fig = Figure(figsize=(5.8, 2.0), dpi=100, facecolor=PANEL)
    gui.ax_chart = gui.fig.add_subplot(111)
    gui.ax_chart.set_facecolor(PANEL)
    gui.chart_canvas = FigureCanvasTkAgg(gui.fig, master=gui.knee_chart_host)
    gui._chart_widget = gui.chart_canvas.get_tk_widget()
    gui._chart_widget.configure(bg=PANEL, highlightthickness=0)
    gui._chart_widget.grid(row=0, column=0, sticky="nsew")
    _bind_figure_resize(gui.chart_canvas, gui.fig, min_px=80)
    from stablewalk.ui.viewers.chart_interactions import (
        attach_chart_interactions,
        build_chart_tools_bar,
    )

    attach_chart_interactions(gui.fig, gui.chart_canvas)
    gui._init_chart()

    knee_tools = build_chart_tools_bar(
        gui, knee_panel, fig_attr="fig", canvas_attr="chart_canvas", export_name="knee_flexion"
    )
    knee_tools.grid(row=2, column=0, sticky="ew", pady=(4, 0))

    knee_legend_row = tk.Frame(knee_panel, bg=PANEL, highlightthickness=0)
    knee_legend_row.grid(row=3, column=0, sticky="ew", pady=(4, 0))
    knee_legend_row.grid_remove()

    knee_summary_row = tk.Frame(knee_panel, bg=PANEL, highlightthickness=0)
    knee_summary_row.grid(row=4, column=0, sticky="ew", pady=(6, 0))
    knee_summary_row.columnconfigure(0, weight=1)
    gui.lbl_knee_summary_compact = tk.Label(
        knee_summary_row,
        text="L ROM: —  R ROM: —  Asymmetry: —",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
    )
    gui.lbl_knee_summary_compact.grid(row=0, column=0, sticky="ew")
    gui.lbl_knee_interp_compact = gui.lbl_knee_summary_compact
    gui.lbl_knee_convention = None
    gui.lbl_knee_summary_title = None
    gui.btn_knee_details = ttk.Button(
        knee_summary_row,
        text="Details",
        style="Compact.TButton",
        command=gui._show_knee_details,
    )
    gui.btn_knee_details.grid(row=0, column=1, sticky="e", padx=(8, 0))

    contact_gait_panel = ttk.LabelFrame(
        bottom_row,
        text=_card_title(LABEL_CONTACT_VGRF_SECTION),
        style="Card.TLabelframe",
        padding=_ANALYSIS_PANEL_PAD,
    )
    contact_gait_panel.grid(
        row=1,
        column=0,
        columnspan=3,
        sticky="nsew",
        padx=(0, 0),
        pady=(PAD_XS, 0),
    )
    gui.contact_gait_panel = contact_gait_panel
    bottom_row.rowconfigure(1, weight=2, minsize=CONTACT_GAIT_CHART_MIN_H)
    contact_gait_panel.columnconfigure(0, weight=1)
    contact_gait_panel.rowconfigure(0, weight=0)
    contact_gait_panel.rowconfigure(1, weight=1)

    gui.fig_contact_gait = Figure(figsize=(10.0, 2.35), dpi=100, facecolor=PANEL)
    gui.ax_contact_gait = None
    from stablewalk.ui.viewers.chart_interactions import (
        attach_chart_interactions,
        build_chart_tools_bar,
    )

    gui.contact_gait_tools = build_chart_tools_bar(
        gui,
        contact_gait_panel,
        fig_attr="fig_contact_gait",
        canvas_attr="canvas_contact_gait",
        export_name="contact_vgrf",
    )
    gui.contact_gait_tools.grid(row=0, column=0, sticky="ew", pady=(0, 2))

    gui.contact_gait_chart_host = tk.Frame(
        contact_gait_panel,
        bg=PANEL,
        highlightthickness=0,
    )
    gui.contact_gait_chart_host.grid(row=1, column=0, sticky="nsew")
    gui.contact_gait_chart_host.columnconfigure(0, weight=1)
    gui.contact_gait_chart_host.rowconfigure(0, weight=1)

    gui.canvas_contact_gait = FigureCanvasTkAgg(
        gui.fig_contact_gait, master=gui.contact_gait_chart_host
    )
    cg_widget = gui.canvas_contact_gait.get_tk_widget()
    cg_widget.configure(bg=PANEL, highlightthickness=0)
    cg_widget.grid(row=0, column=0, sticky="nsew")
    _bind_figure_resize(gui.canvas_contact_gait, gui.fig_contact_gait, min_px=80)
    attach_chart_interactions(gui.fig_contact_gait, gui.canvas_contact_gait)

    def _fit_contact_gait_canvas() -> None:
        _fit_figure_to_host(
            gui.canvas_contact_gait,
            gui.fig_contact_gait,
            host=gui.contact_gait_chart_host,
            min_px=80,
        )

    gui._fit_contact_gait_canvas = _fit_contact_gait_canvas
    _contact_fit_job: dict[str, object] = {"id": None}

    def _on_contact_panel_configure(_event: object | None = None) -> None:
        job = _contact_fit_job.get("id")
        if job is not None:
            try:
                gui.root.after_cancel(job)  # type: ignore[arg-type]
            except Exception:
                pass
        try:
            _contact_fit_job["id"] = gui.root.after(
                32, gui._fit_contact_gait_canvas
            )
        except Exception:
            gui._fit_contact_gait_canvas()

    contact_gait_panel.bind("<Configure>", _on_contact_panel_configure, add="+")

    traj_panel = ttk.LabelFrame(
        bottom_row,
        text=_card_title("Selected Joint 3D Movement"),
        style="Card.TLabelframe",
        padding=_ANALYSIS_PANEL_PAD,
    )
    traj_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 0))
    gui.traj_panel = traj_panel
    traj_panel.columnconfigure(0, weight=1)
    traj_panel.rowconfigure(0, weight=1)
    traj_panel.rowconfigure(1, weight=0)

    gui.traj_panel_header_host = tk.Frame(traj_panel, bg=PANEL, highlightthickness=0)
    gui.traj_panel_header_host.grid(row=0, column=0, sticky="ew")
    gui.traj_panel_header_host.columnconfigure(0, weight=1)
    gui.traj_panel_header_host.grid_remove()

    gui.traj_panel_stack = tk.Frame(traj_panel, bg=PANEL, highlightthickness=0)
    gui.traj_panel_stack.grid(row=0, column=0, sticky="nsew")
    gui.traj_panel_stack.columnconfigure(0, weight=1)
    gui.traj_panel_stack.rowconfigure(0, weight=1)

    gui.dof_analysis_empty_host = tk.Frame(gui.traj_panel_stack, bg=PANEL)
    gui.dof_analysis_empty_host.grid(row=0, column=0, sticky="nsew")
    gui.dof_analysis_empty_host.columnconfigure(0, weight=1)
    gui.dof_analysis_empty_host.rowconfigure(0, weight=0)

    gui.lbl_dof_analysis_empty = ttk.Label(
        gui.dof_analysis_empty_host,
        text=EMPTY_SELECT_DOF_CHART,
        style="EmptyState.TLabel",
    )
    gui.lbl_dof_analysis_empty.grid(row=0, column=0)

    gui.dof_analysis_body = tk.Frame(gui.traj_panel_stack, bg=PANEL, highlightthickness=0)
    gui.dof_analysis_body.grid(row=0, column=0, sticky="nsew")
    gui.dof_analysis_body.columnconfigure(0, weight=1)
    gui.dof_analysis_body.rowconfigure(0, weight=1)

    gui.dof_analysis_left = tk.Frame(
        gui.dof_analysis_body, bg=PANEL, highlightthickness=0,
    )
    gui.dof_analysis_left.grid(row=0, column=0, sticky="new", padx=(0, 6))
    gui.dof_analysis_left.grid_remove()
    _build_foot_analysis_card(gui, gui.dof_analysis_left).grid(row=0, column=0, sticky="new")
    gui.dof_analysis_foot_row = None
    gui.dof_analysis_foot_slots = []

    gui.dof_analysis_graph_section = tk.Frame(
        gui.dof_analysis_body, bg=PANEL, highlightthickness=0
    )
    gui.dof_analysis_graph_section.grid(row=0, column=0, columnspan=2, sticky="nsew")
    gui.dof_analysis_graph_section.columnconfigure(0, weight=1)
    gui.dof_analysis_graph_section.rowconfigure(0, weight=0)
    gui.dof_analysis_graph_section.rowconfigure(1, weight=0)
    gui.dof_analysis_graph_section.rowconfigure(2, weight=0)
    gui.dof_analysis_graph_section.rowconfigure(3, weight=1, minsize=_MOTION_GRAPH_ROW_MINSIZE)
    gui.dof_analysis_graph_section.rowconfigure(4, weight=0)
    gui.dof_analysis_graph_section.rowconfigure(5, weight=0)
    gui.dof_analysis_graph_section.rowconfigure(6, weight=0)
    gui.dof_analysis_graph_section.rowconfigure(7, weight=0)

    status_row = tk.Frame(gui.dof_analysis_graph_section, bg=PANEL, highlightthickness=0)
    status_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
    status_row.columnconfigure(1, weight=1)

    def _status_label(row: int, caption: str, attr: str) -> tk.Label:
        tk.Label(
            status_row,
            text=caption,
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=1)
        lbl = tk.Label(
            status_row,
            text="—",
            bg=PANEL,
            fg=TEXT,
            font=FONT_UI_SM,
            anchor="w",
        )
        lbl.grid(row=row, column=1, sticky="w", pady=1)
        setattr(gui, attr, lbl)
        return lbl

    _status_label(0, "Selected Joint:", "lbl_selected_joint_value")
    _status_label(1, "Coordinate Mode:", "lbl_dof_coord_mode_value")
    gui.lbl_dof_coord_mode_value.configure(text="ROOT-RELATIVE")
    _status_label(2, "Trajectory Mode:", "lbl_dof_traj_mode_value")
    gui.lbl_dof_traj_mode_value.configure(text="CURRENT PROGRESS")
    _status_label(3, "View:", "lbl_dof_view_mode_value")
    gui.lbl_dof_view_mode_value.configure(text="3D")
    status_row.grid_remove()

    joint_header = tk.Frame(gui.dof_analysis_graph_section, bg=PANEL, highlightthickness=0)
    joint_header.grid(row=1, column=0, sticky="ew", pady=(0, 2))
    joint_header.columnconfigure(0, weight=1)

    gui.lbl_joint_movement_title = tk.Label(
        joint_header,
        text="Select a joint to view its 3D movement path",
        bg=PANEL,
        fg=TEXT,
        font=FONT_PANEL_HEADER,
        anchor="w",
    )
    gui.lbl_joint_movement_title.grid(row=0, column=0, sticky="w")
    gui.lbl_dof_coord_mode = None
    # The LabelFrame is already titled "Selected Joint 3D Movement"; hide the
    # duplicate inner heading so the 3D graph reclaims that vertical band.
    joint_header.grid_remove()

    joint_controls = tk.Frame(gui.dof_analysis_graph_section, bg=PANEL, highlightthickness=0)
    joint_controls.grid(row=2, column=0, sticky="ew", pady=(0, 0))
    joint_controls.columnconfigure(0, weight=1)

    controls_left = tk.Frame(joint_controls, bg=PANEL, highlightthickness=0)
    controls_left.grid(row=0, column=0, sticky="w")

    gui.var_dof_traj_display = tk.StringVar(value="CURRENT PROGRESS")
    tk.Label(controls_left, text="Trajectory Display", bg=PANEL, fg=MUTED, font=FONT_UI_SM).pack(
        side=tk.LEFT, padx=(0, 4)
    )
    gui.cmb_dof_traj_display = ttk.Combobox(
        controls_left,
        textvariable=gui.var_dof_traj_display,
        values=("CURRENT PROGRESS", "FULL TRAJECTORY"),
        state="readonly",
        width=13,
    )
    gui.cmb_dof_traj_display.pack(side=tk.LEFT, padx=(0, 10))
    gui.cmb_dof_traj_display.bind(
        "<<ComboboxSelected>>", lambda _e: gui._on_dof_traj_display_changed()
    )

    gui.var_dof_coord_mode = tk.StringVar(value="ROOT-RELATIVE")
    tk.Label(controls_left, text="Coord", bg=PANEL, fg=MUTED, font=FONT_UI_SM).pack(
        side=tk.LEFT, padx=(0, 4)
    )
    gui.cmb_dof_coord_mode = ttk.Combobox(
        controls_left,
        textvariable=gui.var_dof_coord_mode,
        values=("ROOT-RELATIVE", "GLOBAL"),
        state="readonly",
        width=10,
    )
    gui.cmb_dof_coord_mode.pack(side=tk.LEFT, padx=(0, 10))
    gui.cmb_dof_coord_mode.bind(
        "<<ComboboxSelected>>", lambda _e: gui._on_dof_coord_mode_changed()
    )

    gui.var_dof_projection = tk.StringVar(value="3D")
    tk.Label(controls_left, text="View", bg=PANEL, fg=MUTED, font=FONT_UI_SM).pack(
        side=tk.LEFT, padx=(0, 4)
    )
    gui.cmb_dof_projection = ttk.Combobox(
        controls_left,
        textvariable=gui.var_dof_projection,
        values=("3D", "Frontal Plane", "Sagittal Plane"),
        state="readonly",
        width=10,
    )
    gui.cmb_dof_projection.pack(side=tk.LEFT)
    gui.cmb_dof_projection.bind(
        "<<ComboboxSelected>>", lambda _e: gui._on_dof_projection_changed()
    )

    from stablewalk.ui.tk.trajectory_camera_toolbar import build_trajectory_camera_toolbar

    # Camera tools share the controls row (right side) — never on the plot itself.
    joint_controls.columnconfigure(1, weight=0)
    gui.dof_traj_camera_bar = build_trajectory_camera_toolbar(
        gui,
        joint_controls,
        ax_attr="ax_dof_traj",
        canvas_attr="canvas_dof_traj",
        compact=True,
        show_rotate=False,
    )
    gui.dof_traj_camera_bar.grid(row=0, column=1, sticky="e", padx=(8, 0))

    _build_analysis_graph_chrome(gui, gui.dof_analysis_graph_section)
    gui.dof_graph_chrome.grid_remove()

    gui.dof_analysis_graph_frame = tk.Frame(
        gui.dof_analysis_graph_section,
        bg=PANEL,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    gui.dof_analysis_graph_frame.grid(row=3, column=0, sticky="nsew", pady=(3, 0))
    gui.dof_analysis_graph_frame.columnconfigure(0, weight=1)
    gui.dof_analysis_graph_frame.rowconfigure(0, weight=1, minsize=_MOTION_GRAPH_ROW_MINSIZE)

    gui.lbl_traj_graph_debug = None

    gui.dof_analysis_graph_summary_slots = []
    gui.dof_analysis_graph_summary_row = None
    gui.dof_analysis_graph_header = None
    gui.dof_analysis_export_row = None
    gui.dof_graph_legend_labels = []

    # Direct canvas host — no clip-viewport wrapper (Tk Canvas windows can stay 1×1
    # until the Motion tab is mapped, which hid the matplotlib widget in practice).
    gui.dof_analysis_graph_inner = tk.Frame(
        gui.dof_analysis_graph_frame, bg=PANEL, highlightthickness=0
    )
    gui.dof_analysis_graph_inner.grid(row=0, column=0, sticky="nsew")
    gui.dof_analysis_graph_inner.columnconfigure(0, weight=1)
    gui.dof_analysis_graph_inner.rowconfigure(0, weight=1)

    gui.dof_analysis_graph_canvas_host = gui.dof_analysis_graph_inner
    gui._traj_clip_canvas = None
    gui._traj_clip_window_id = None

    gui.dof_traj_live_info_frame = tk.Frame(
        gui.dof_analysis_graph_section,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    gui.dof_traj_live_info_frame.grid(
        row=4, column=0, sticky="ew", pady=(3, 0)
    )

    def _trajectory_info_field(
        caption: str, attr: str, initial: str = "—", *, numeric: bool = False
    ) -> None:
        field = tk.Frame(gui.dof_traj_live_info_frame, bg=ELEVATED)
        field.pack(side=tk.LEFT, padx=(8, 10), pady=3)
        tk.Label(
            field,
            text=f"{caption} ",
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
        ).pack(side=tk.LEFT)
        value = tk.Label(
            field,
            text=initial,
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_MONO_SM if numeric else FONT_UI_SM,
        )
        value.pack(side=tk.LEFT)
        setattr(gui, attr, value)

    _trajectory_info_field("Joint", "lbl_traj_info_joint")
    _trajectory_info_field("Frame", "lbl_traj_info_frame", numeric=True)
    _trajectory_info_field("Time", "lbl_traj_info_time", numeric=True)
    _trajectory_info_field("X", "lbl_traj_info_x", numeric=True)
    _trajectory_info_field("Y", "lbl_traj_info_y", numeric=True)
    _trajectory_info_field("Z", "lbl_traj_info_z", numeric=True)
    tk.Label(
        gui.dof_traj_live_info_frame,
        text="root-relative",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
    ).pack(side=tk.RIGHT, padx=(0, 8), pady=3)

    gui.dof_analysis_path_metrics_frame = tk.Frame(
        gui.dof_analysis_graph_section, bg=PANEL, highlightthickness=0
    )
    gui.dof_analysis_path_metrics_frame.grid(row=5, column=0, sticky="ew", pady=(3, 0))
    gui.dof_analysis_path_metrics_frame.columnconfigure(0, weight=1)
    gui.dof_analysis_path_metrics_frame.columnconfigure(1, weight=1)
    gui.dof_analysis_path_metrics_frame.columnconfigure(2, weight=1)
    gui.dof_analysis_path_metrics_frame.columnconfigure(3, weight=1)

    gui.lbl_traj_travel = tk.Label(
        gui.dof_analysis_path_metrics_frame,
        text="Travel: —",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_traj_travel.grid(row=0, column=0, sticky="w")
    gui.lbl_traj_smoothness = tk.Label(
        gui.dof_analysis_path_metrics_frame,
        text="Smoothness: —",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_traj_smoothness.grid(row=0, column=1, sticky="w")
    gui.lbl_traj_max_deviation = tk.Label(
        gui.dof_analysis_path_metrics_frame,
        text="Maximum deviation: —",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_traj_max_deviation.grid(row=0, column=2, sticky="w")

    gui.lbl_traj_samples = tk.Label(
        gui.dof_analysis_path_metrics_frame,
        text="Samples: —",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_traj_samples.grid(row=0, column=3, sticky="w")

    gui.dof_analysis_header = _build_analysis_metrics_panel(
        gui,
        gui.traj_panel_header_host,
        path_metrics_parent=gui.dof_analysis_path_metrics_frame,
    )
    gui.dof_analysis_header.grid(row=0, column=0, sticky="ew")
    gui.dof_analysis_header.grid_remove()
    primary_metrics = getattr(gui, "dof_analysis_primary_metrics_host", None)
    if primary_metrics is not None:
        primary_metrics.grid_remove()
    advanced_host = getattr(gui, "dof_analysis_advanced_host", None)
    if advanced_host is not None:
        advanced_host.grid_remove()
    toggle_btn = getattr(gui, "btn_toggle_joint_advanced", None)
    if toggle_btn is not None:
        toggle_parent = toggle_btn.master
        if toggle_parent is not None:
            toggle_parent.grid_remove()

    gui.dof_analysis_interp_frame = tk.Frame(
        gui.dof_analysis_graph_section, bg=PANEL, highlightthickness=0
    )
    traj_legend_row = tk.Frame(gui.dof_analysis_graph_section, bg=PANEL, highlightthickness=0)
    traj_legend_row.grid(row=6, column=0, sticky="ew", pady=(2, 0))
    gui.var_dof_show_body_reference = tk.BooleanVar(value=False)
    gui.chk_dof_show_body_reference = ttk.Checkbutton(
        traj_legend_row,
        text="Show body reference",
        variable=gui.var_dof_show_body_reference,
        command=gui._on_dof_body_reference_changed,
        takefocus=False,
    )
    gui.chk_dof_show_body_reference.pack(side=tk.RIGHT)
    gui.dof_traj_legend_labels = _build_compact_graph_legend(traj_legend_row)

    gui.dof_analysis_interp_frame.grid(row=7, column=0, sticky="ew", pady=(2, 0))
    gui.dof_analysis_interp_frame.columnconfigure(0, weight=1)

    gui.lbl_joint_path_summary = tk.Label(
        gui.dof_analysis_interp_frame,
        text="Select a joint to view its movement path.",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=920,
    )
    gui.lbl_joint_path_summary.grid(row=0, column=0, sticky="ew")

    gui.lbl_motion_traj_metrics = tk.Label(
        gui.dof_analysis_interp_frame,
        text="",
        bg=PANEL,
        fg=ACCENT,
        font=FONT_METRIC,
        anchor="w",
        justify=tk.LEFT,
        wraplength=920,
    )
    gui.lbl_motion_traj_metrics.grid(row=1, column=0, sticky="ew", pady=(1, 0))
    # Travel is already reported in the path-metrics strip above the summary, so
    # keep this secondary ROM/span breakdown out of the graph column to give the
    # 3D plot the vertical space instead.
    gui.lbl_motion_traj_metrics.grid_remove()

    gui.lbl_traj_confidence = tk.Label(
        gui.dof_analysis_interp_frame,
        text="Trajectory Confidence: —",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_traj_confidence.grid(row=2, column=0, sticky="w", pady=(1, 0))
    gui.lbl_dof_graph_explain_body = gui.lbl_joint_path_summary
    gui.lbl_dof_traj_interp_compact = None
    gui.dof_analysis_graph_explainer = gui.dof_analysis_interp_frame

    gui.dof_analysis_sidebar = None
    gui.dof_analysis_legend_section = None
    gui.dof_analysis_legend_labels = gui.dof_graph_legend_labels

    gui.lbl_dof_traj_empty = gui.lbl_dof_analysis_empty
    gui.dof_analysis_legend_body = None
    gui.lbl_dof_analysis_tracked = None
    gui.lbl_dof_analysis_direction = None

    gui.fig_dof_traj = Figure(figsize=(6.5, 2.0), dpi=100, facecolor=PANEL)
    gui.ax_dof_traj = gui.fig_dof_traj.add_subplot(111, projection="3d")
    gui.ax_dof_traj._stablewalk_motion_dock = True
    gui.ax_dof_traj._stablewalk_overview_cm_ticks = True
    setup_single_dof_trajectory_axes(gui.ax_dof_traj)
    from stablewalk.ui.viewers.dof_trajectory_3d import relayout_single_dof_viewport

    relayout_single_dof_viewport(gui.ax_dof_traj)
    gui.canvas_dof_traj = FigureCanvasTkAgg(
        gui.fig_dof_traj, master=gui.dof_analysis_graph_canvas_host
    )
    traj_canvas = gui.canvas_dof_traj.get_tk_widget()
    traj_canvas.configure(bg=PANEL, highlightthickness=0)
    _ensure_trajectory_canvas_gridded(gui.canvas_dof_traj)
    from stablewalk.ui.tk.trajectory_camera_toolbar import bind_trajectory_camera_canvas

    bind_trajectory_camera_canvas(
        gui, canvas_attr="canvas_dof_traj", ax_attr="ax_dof_traj"
    )

    # Start near-square; _fit_trajectory_figure immediately maps to host pixels.
    gui.fig_dof_traj_overview = Figure(figsize=(4.0, 4.0), dpi=100, facecolor=PANEL)
    gui.ax_dof_traj_overview = gui.fig_dof_traj_overview.add_subplot(111, projection="3d")
    gui.ax_dof_traj_overview._stablewalk_overview_dock = True
    gui.ax_dof_traj_overview._stablewalk_overview_cm_ticks = True
    gui.ax_dof_traj_overview._stablewalk_overview_use_progress_viewport = False
    setup_single_dof_trajectory_axes(gui.ax_dof_traj_overview)
    gui.ax_dof_traj_overview.text2D(
        0.5,
        0.5,
        EMPTY_OVERVIEW_JOINT_INSPECT,
        transform=gui.ax_dof_traj_overview.transAxes,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=10,
    )
    relayout_single_dof_viewport(gui.ax_dof_traj_overview)
    gui.canvas_dof_traj_overview = FigureCanvasTkAgg(
        gui.fig_dof_traj_overview, master=gui.overview_traj_canvas_host
    )
    overview_traj_canvas = gui.canvas_dof_traj_overview.get_tk_widget()
    overview_traj_canvas.configure(bg=PANEL, highlightthickness=0)
    _ensure_trajectory_canvas_gridded(
        gui.canvas_dof_traj_overview, overview_dock=True
    )
    bind_trajectory_camera_canvas(
        gui,
        canvas_attr="canvas_dof_traj_overview",
        ax_attr="ax_dof_traj_overview",
    )
    _bind_trajectory_figure_resize(
        gui.canvas_dof_traj_overview,
        gui.fig_dof_traj_overview,
        gui.ax_dof_traj_overview,
        graph_host=gui.overview_traj_canvas_host,
        extra_hosts=(
            gui.overview_traj_panel,
            gui.overview_traj_canvas_host,
            getattr(gui, "overview_traj_footer", None),
        ),
    )

    gui.dof_analysis_empty_host.grid_remove()
    _bind_trajectory_figure_resize(
        gui.canvas_dof_traj,
        gui.fig_dof_traj,
        gui.ax_dof_traj,
        graph_host=gui.dof_analysis_graph_canvas_host,
        extra_hosts=(
            gui.dof_analysis_graph_inner,
            gui.dof_analysis_graph_section,
            gui.dof_analysis_body,
            gui.traj_panel,
            getattr(gui, "overview_traj_mount", None),
            getattr(gui, "overview_traj_panel", None),
        ),
    )

    def _on_traj_panel_resize(_event: object = None) -> None:
        after_id = getattr(gui, "_traj_resize_after", None)
        if after_id is not None:
            try:
                gui.root.after_cancel(after_id)
            except tk.TclError:
                pass

        def _reflow() -> None:
            gui._traj_resize_after = None
            overview_panel = getattr(gui, "overview_traj_panel", None)
            if overview_panel is not None:
                wrap = max(180, min(360, overview_panel.winfo_width() - 16))
                for attr in (
                    "lbl_overview_traj_legend",
                    "lbl_overview_traj_metrics",
                    "lbl_overview_traj_detail",
                    "lbl_overview_traj_video",
                    "lbl_overview_category_note",
                ):
                    lbl = getattr(gui, attr, None)
                    if lbl is not None:
                        try:
                            lbl.configure(wraplength=wrap)
                        except tk.TclError:
                            pass
            _ensure_trajectory_canvas_gridded(gui.canvas_dof_traj)
            if _fit_trajectory_figure(
                gui.canvas_dof_traj,
                gui.fig_dof_traj,
                gui.ax_dof_traj,
                graph_host=gui.dof_analysis_graph_canvas_host,
            ):
                gui.canvas_dof_traj.draw_idle()
            overview_canvas = getattr(gui, "canvas_dof_traj_overview", None)
            if overview_canvas is not None and getattr(
                gui, "_overview_traj_dock_visible", False
            ):
                _ensure_trajectory_canvas_gridded(
                    overview_canvas, overview_dock=True
                )
                if _fit_trajectory_figure(
                    overview_canvas,
                    gui.fig_dof_traj_overview,
                    gui.ax_dof_traj_overview,
                    graph_host=gui.overview_traj_canvas_host,
                ):
                    overview_canvas.draw_idle()
            sync = getattr(gui, "_sync_dashboard_scroll", None)
            if sync is not None:
                sync()

        gui._traj_resize_after = gui.root.after(
            _OVERVIEW_TRAJ_RESIZE_DEBOUNCE_MS, _reflow
        )

    def _on_viz_row_resize(_event: object = None) -> None:
        fit = getattr(gui, "_fit_skeleton_canvas", None)
        if fit is not None:
            fit()

    traj_panel.bind("<Configure>", _on_traj_panel_resize)
    gui.dof_analysis_graph_frame.bind("<Configure>", _on_traj_panel_resize)
    gui.dof_analysis_graph_inner.bind("<Configure>", _on_traj_panel_resize)
    overview_traj = getattr(gui, "overview_traj_panel", None)
    if overview_traj is not None:
        overview_traj.bind("<Configure>", _on_traj_panel_resize)
        overview_mount = getattr(gui, "overview_traj_mount", None)
        if overview_mount is not None:
            overview_mount.bind("<Configure>", _on_traj_panel_resize)
    gui.skel_canvas_host.bind("<Configure>", _on_viz_row_resize)
    if hasattr(gui, "knee_panel"):
        gui.knee_panel.bind("<Configure>", lambda _e: gui._update_chart())

    gui.lbl_dof_traj_path = gui.dof_analysis_graph_section

    motion_tools = ttk.Frame(tab_advanced_content)
    motion_tools.columnconfigure(0, weight=1)
    gui._motion_tools_row = motion_tools

    section3 = build_gait_metrics_section(gui, tab_advanced_content)
    section3.grid(row=2, column=0, sticky="ew", pady=(0, PAD_XS))

    from stablewalk.ui.tk.dashboard_pipeline_status import build_pipeline_status_panel

    advanced_pipeline = build_pipeline_status_panel(
        gui,
        tab_advanced_content,
        host_attr="_pipeline_status_host_advanced",
        section_attr="_section_pipeline_status_advanced",
    )
    advanced_pipeline.grid(row=1, column=0, sticky="ew", pady=(0, PAD_XS))
    gui._section_pipeline_status_advanced = advanced_pipeline
    gui._pipeline_panel_host = advanced_pipeline

    real_to_sim_section = build_real_to_sim_section(gui, tab_advanced_content)
    real_to_sim_section.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))

    advanced_info = ttk.LabelFrame(
        tab_advanced_content,
        text="  Analysis Evidence  ",
        style="Card.TLabelframe",
        padding=(PAD_XS, PAD_XS),
    )
    advanced_info.grid(row=3, column=0, sticky="ew", pady=(0, PAD_XS))
    advanced_info.columnconfigure(0, weight=1)
    gui.lbl_advanced_temporal = ttk.Label(
        advanced_info,
        text="Temporal Symmetry: —",
        style="Card.TLabel",
    )
    gui.lbl_advanced_temporal.grid(row=0, column=0, sticky="w")
    gui.lbl_advanced_pelvis = ttk.Label(
        advanced_info,
        text="Pelvis Stability: —",
        style="Card.TLabel",
    )
    gui.lbl_advanced_pelvis.grid(row=1, column=0, sticky="w", pady=(2, 0))
    gui.lbl_advanced_evidence = tk.Label(
        advanced_info,
        text="Evidence: —",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
        wraplength=720,
        justify=tk.LEFT,
    )
    gui.lbl_advanced_evidence.grid(row=2, column=0, sticky="w", pady=(2, 0))
    gui._section_advanced_info = advanced_info

    motion_tools.grid(row=4, column=0, sticky="ew", pady=(0, PAD_XS))

    data_row = build_data_export_section(gui, tab_advanced_content)
    data_row.grid(row=5, column=0, sticky="ew", pady=(0, PAD_XS))

    from stablewalk.ui.tk.generated_files_panel import build_generated_files_section

    generated_files = build_generated_files_section(gui, tab_advanced_content)
    generated_files.grid(row=6, column=0, sticky="nsew", pady=(0, PAD_XS))
    tab_advanced_content.rowconfigure(6, weight=0)

    table_panel = ttk.LabelFrame(
        motion_tools,
        text="  Detailed Joint Data  ",
        style="Card.TLabelframe",
        padding=(PAD_XS, PAD_XS),
    )
    table_panel.grid(row=0, column=0, sticky="nsew")
    gui.table_panel = table_panel
    gui._table_data_expanded = True
    gui._collected_data_bar = None

    gui.dof_table_display_mode = tk.StringVar(value=DOF_TABLE_MODE_DEFAULT)
    gui.lbl_table_summary = ttk.Label(
        table_panel,
        text="Data (0 samples)",
        style="Card.TLabel",
    )
    gui.lbl_table_summary.grid(row=0, column=0, sticky="w", pady=(0, PAD_XS))

    table_body = ttk.Frame(table_panel)
    table_body.grid(row=1, column=0, sticky="nsew")
    gui.table_body = table_body
    table_panel.rowconfigure(1, weight=1)
    table_panel.columnconfigure(0, weight=1)
    table_body.columnconfigure(0, weight=1)
    table_body.rowconfigure(0, weight=1)

    gui.lbl_table_empty = ttk.Label(
        table_body,
        text=EMPTY_SELECT_DOF_TABLE,
        style="EmptyState.TLabel",
    )
    gui.lbl_table_empty.pack(fill=tk.BOTH, expand=True)

    gui.dof_pos_tree = _make_data_tree(
        table_body,
        JOINT_DATA_TABLE_COLUMNS,
        JOINT_DATA_TABLE_HEADINGS,
        col_widths=JOINT_DATA_TABLE_WIDTHS,
        height=_TABLE_TREE_HEIGHT,
        text_cols=frozenset({"joint", "contact_state"}),
        style="Large.Treeview",
    )
    gui.dof_pos_tree.bind("<<TreeviewSelect>>", gui._on_dof_pos_tree_select)

    # Movement Checkpoints — hidden from dashboard; logic kept for optional reuse
    gui._checkpoint_hidden = ttk.Frame(gui.root)
    gui.btn_add_checkpoint = ttk.Button(
        gui._checkpoint_hidden,
        text="Add point (current frame)",
        command=gui._add_checkpoint,
    )
    ttk.Button(
        gui._checkpoint_hidden,
        text="Clear points",
        command=gui._clear_checkpoints,
    )
    gui.lbl_checkpoint_status = ttk.Label(
        gui._checkpoint_hidden,
        text="No points added yet",
    )
    gui.lbl_checkpoint_empty = ttk.Label(
        gui._checkpoint_hidden,
        text="No checkpoints yet — select body points and click 'Add point'.",
    )
    gui.checkpoint_tree = _make_data_tree(
        gui._checkpoint_hidden,
        CHECKPOINT_COLUMNS,
        CHECKPOINT_HEADINGS,
        col_widths=CHECKPOINT_WIDTHS,
        height=6,
        text_cols=frozenset({"dof", "joint"}),
    )

    # ── Section 1 summary dock BELOW video|skeleton (Gait Analysis Summary) ─
    from stablewalk.ui.tk.dashboard_responsive import _install_sidebar_scroll

    sidebar = ttk.Frame(section1, style="Card.TFrame", padding=DASHBOARD_VIZ_CARD_PAD)
    sidebar.grid(
        row=1,
        column=0,
        columnspan=2,
        sticky="ew",
        padx=0,
        pady=(PAD_XS, 0),
    )
    sidebar_inner = _install_sidebar_scroll(gui, sidebar)
    gui.sidebar = sidebar
    gui._overview_summary_host = sidebar

    _side_wrap = max(80, _SIDEBAR_MIN_WIDTH - 24)

    def _sync_sidebar_wrap(_event: object | None = None) -> None:
        width = sidebar.winfo_width()
        if width < 40:
            return
        wrap = max(72, width - 20)
        for attr in (
            "lbl_stab_headline",
            "lbl_stab_metrics",
            "lbl_stab_reason",
            "lbl_opensim_compact_ready",
        ):
            lbl = getattr(gui, attr, None)
            if lbl is not None and hasattr(lbl, "configure"):
                try:
                    lbl.configure(wraplength=wrap)
                except tk.TclError:
                    pass

    sidebar.bind("<Configure>", _sync_sidebar_wrap)
    gui._sync_utility_sidebar_wrap = _sync_sidebar_wrap

    if gui._sidebar_hidden.master is None:
        gui._sidebar_hidden = ttk.Frame(gui.root)
    gui.lbl_session_status = ttk.Label(
        gui._sidebar_hidden,
        text="No session loaded",
        style="SideMuted.TLabel",
        wraplength=_side_wrap,
    )
    gui.lbl_summary = gui.lbl_session_status
    # The visible sampling selector now lives in the playback controls bar and
    # owns ``gui.refresh_var``; only create a fallback var if it doesn't exist.
    if not hasattr(gui, "refresh_var"):
        gui.refresh_var = tk.StringVar(value=REFRESH_INTERVAL_DEFAULT)
    refresh_combo = ttk.Combobox(
        gui._sidebar_hidden,
        textvariable=gui.refresh_var,
        values=REFRESH_INTERVAL_CHOICES,
        state="readonly",
        width=8,
    )
    refresh_combo.bind("<<ComboboxSelected>>", gui._on_refresh_interval)

    dof_panel = ttk.LabelFrame(
        motion_joint_row,
        text="  Joint Selection  ",
        style="Card.TLabelframe",
        padding=(PAD_XS, 2),
    )
    dof_panel.grid(row=0, column=0, sticky="ew")
    gui._sidebar_dof_panel = dof_panel

    gui.dof_chips_frame = tk.Frame(dof_panel, bg=PANEL, highlightthickness=0)
    gui.dof_chips_frame.pack(fill=tk.X, pady=(0, 1))

    btn_row = ttk.Frame(dof_panel)
    btn_row.pack(fill=tk.X, pady=(0, 0))
    ttk.Button(
        btn_row,
        text="Clear all",
        style="Compact.TButton",
        command=gui._clear_dof_selection,
    ).pack(side=tk.LEFT)

    # Full checklist kept off-screen for selection state; not shown in sidebar.
    if not hasattr(gui, "_dof_checklist_hidden"):
        gui._dof_checklist_hidden = ttk.Frame(gui._sidebar_hidden)
    list_frame = ttk.Frame(gui._dof_checklist_hidden, style="Card.TFrame", padding=2)
    list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
    list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    gui.dof_checkbox_canvas = tk.Canvas(
        list_frame,
        bg=ELEVATED,
        highlightthickness=0,
        borderwidth=0,
        height=_BODY_CHECKLIST_HEIGHT,
    )
    gui.dof_checkbox_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    list_scroll.config(command=gui.dof_checkbox_canvas.yview)

    gui.dof_checkboxes_inner = tk.Frame(gui.dof_checkbox_canvas, bg=ELEVATED)
    gui._dof_checkbox_window = gui.dof_checkbox_canvas.create_window(
        (0, 0),
        window=gui.dof_checkboxes_inner,
        anchor="nw",
    )
    gui.dof_checkbox_canvas.configure(yscrollcommand=list_scroll.set)

    gui._dof_checkbox_vars: dict[str, tk.BooleanVar] = {}
    gui._dof_checkbox_rows: dict[str, tk.Frame] = {}
    gui._dof_checkbox_name_labels: dict[str, tk.Label] = {}

    def _dof_checkbox_scroll(_event: object) -> None:
        gui.dof_checkbox_canvas.yview_scroll(int(-1 * (_event.delta / 120)), "units")

    def _dof_inner_configure(_event: object) -> None:
        gui.dof_checkbox_canvas.configure(scrollregion=gui.dof_checkbox_canvas.bbox("all"))
        gui.dof_checkbox_canvas.itemconfigure(
            gui._dof_checkbox_window,
            width=gui.dof_checkbox_canvas.winfo_width(),
        )

    gui.dof_checkboxes_inner.bind("<Configure>", _dof_inner_configure)
    gui.dof_checkbox_canvas.bind("<Configure>", _dof_inner_configure)
    gui.dof_checkbox_canvas.bind("<MouseWheel>", _dof_checkbox_scroll)

    for item_id in GUI_DOF_ITEM_IDS:
        var = tk.BooleanVar(value=False)
        gui._dof_checkbox_vars[item_id] = var
        row_bg = ELEVATED
        row = tk.Frame(gui.dof_checkboxes_inner, bg=row_bg, padx=3, pady=3)
        gui._dof_checkbox_rows[item_id] = row
        tk.Checkbutton(
            row,
            text="",
            variable=var,
            command=lambda i=item_id: gui._on_dof_checkbox_changed(i),
            bg=row_bg,
            fg=TEXT,
            selectcolor=PANEL,
            activebackground=row_bg,
            activeforeground=TEXT,
            anchor="w",
            relief=tk.FLAT,
            highlightthickness=0,
            padx=1,
        ).pack(side=tk.LEFT)
        name_lbl = tk.Label(
            row,
            text=GUI_DOF_LABELS[item_id],
            bg=row_bg,
            fg=TEXT,
            anchor="w",
            font=FONT_UI_SM,
            cursor="hand2",
        )
        name_lbl.pack(fill=tk.X, side=tk.LEFT, expand=True)
        name_lbl.bind(
            "<Button-1>",
            lambda _event, i=item_id: gui._on_dof_item_focus(i),
        )
        gui._dof_checkbox_name_labels[item_id] = name_lbl
        row.pack(fill=tk.X, pady=(0, 1))

    stability_panel = ttk.Frame(
        sidebar_inner,
        style="Card.TFrame",
        padding=DASHBOARD_SIDE_PAD,
    )
    gui._sidebar_stability_panel = stability_panel

    build_gait_summary_cards(gui, stability_panel)

    # Legacy widgets retained for details dialog helpers and validation scripts (hidden).
    legacy_host = gui._sidebar_hidden
    gui.lbl_stab_title = ttk.Label(legacy_host, text="", style="SideMuted.TLabel")
    gui.lbl_stab_score = tk.Label(legacy_host, text="—", bg=PANEL, fg=MUTED)
    gui.lbl_stab_score_suffix = tk.Label(legacy_host, text="", bg=PANEL, fg=MUTED)
    gui.lbl_stab_partial_estimate = tk.Label(legacy_host, text="", bg=PANEL, fg=MUTED)
    gui.lbl_stab_partial_label = tk.Label(legacy_host, text="", bg=PANEL, fg=MUTED)
    gui.lbl_gait_explanation = tk.Label(legacy_host, text="", bg=PANEL, fg=MUTED)
    gui.lbl_stab_legacy = tk.Label(legacy_host, text="", bg=PANEL, fg=MUTED)
    gui.lbl_stab_category = tk.Label(
        legacy_host,
        text="No walk analyzed yet",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_stab_headline = gui.lbl_stab_category
    gui.lbl_stab_confidence_badge = tk.Label(legacy_host, text="", bg=PANEL, fg=MUTED)
    gui.lbl_stab_view = tk.Label(legacy_host, text="", bg=PANEL, fg=MUTED)
    gui.lbl_stab_analysis_confidence = tk.Label(legacy_host, text="", bg=PANEL, fg=MUTED)

    stab_compact = tk.Frame(legacy_host, bg=PANEL, highlightthickness=0)
    stab_compact.columnconfigure(0, weight=1)
    stab_compact.columnconfigure(1, weight=1)

    gui._walk_summary_slots: list[tuple[tk.Label, tk.Label, tk.Label | None]] = []
    for index, title in enumerate(("Temporal", "Spatial", "Pelvis", "Cycles")):
        row, col = divmod(index, 2)
        cell = tk.Frame(stab_compact, bg=PANEL, highlightthickness=0)
        cell.grid(row=row, column=col, sticky="ew", padx=(0, 6), pady=1)
        title_lbl = tk.Label(
            cell,
            text=title,
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        )
        title_lbl.pack(side=tk.LEFT)
        value_lbl = tk.Label(
            cell,
            text="—",
            bg=PANEL,
            fg=TEXT,
            font=FONT_METRIC,
            anchor="w",
        )
        value_lbl.pack(side=tk.LEFT, padx=(4, 0))
        interp_lbl: tk.Label | None = None
        if index in (0, 2):
            interp_lbl = tk.Label(
                stab_compact,
                text="",
                bg=PANEL,
                fg=MUTED,
                font=FONT_UI_XS,
                anchor="w",
                justify=tk.LEFT,
                wraplength=120,
            )
            interp_lbl.grid(row=row + 2, column=col, sticky="ew", padx=(0, 6), pady=(0, 2))
        gui._walk_summary_slots.append((title_lbl, value_lbl, interp_lbl))
        if index in (1, 3):
            cell.grid_remove()
            if interp_lbl is not None:
                interp_lbl.grid_remove()

    gui.lbl_stab_metrics = ttk.Label(legacy_host, text="")
    gui.lbl_stab_steps = ttk.Label(legacy_host, text="")
    steps_detail_row = ttk.Frame(legacy_host)
    gui.lbl_stab_step_confidence = ttk.Label(steps_detail_row, text="")
    gui.btn_stab_steps_details = ttk.Button(
        steps_detail_row,
        text="Details",
        style="Compact.TButton",
        command=gui._show_gait_summary_details,
    )
    gui.lbl_stab_reason = ttk.Label(legacy_host, text="")

    gait_cycle_panel = ttk.Frame(
        sidebar_inner,
        style="Card.TFrame",
        padding=DASHBOARD_SIDE_PAD,
    )
    gui._sidebar_gait_cycle_panel = gait_cycle_panel

    gait_header = tk.Frame(gait_cycle_panel, bg=PANEL, highlightthickness=0)
    gait_header.pack(anchor=tk.W, pady=(0, 2), fill=tk.X)
    _metric_title_row(gait_header, "Gait Cycle", "gait_cycle").pack(side=tk.LEFT)

    gui.lbl_gait_cycle_phase = tk.Label(
        gait_cycle_panel,
        text="Phase: —",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_gait_cycle_phase.pack(anchor=tk.W, pady=(0, 2))

    contact_row = tk.Frame(gait_cycle_panel, bg=PANEL, highlightthickness=0)
    contact_row.pack(fill=tk.X, pady=(0, 2))
    contact_row.columnconfigure(0, weight=1)
    contact_row.columnconfigure(1, weight=1)

    gui.lbl_gait_cycle_left_contact = tk.Label(
        contact_row,
        text="Left: —",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_gait_cycle_left_contact.grid(row=0, column=0, sticky="w")
    gui.lbl_gait_cycle_right_contact = tk.Label(
        contact_row,
        text="Right: —",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_gait_cycle_right_contact.grid(row=0, column=1, sticky="w")

    cycle_metrics = tk.Frame(gait_cycle_panel, bg=PANEL, highlightthickness=0)
    cycle_metrics.pack(fill=tk.X, pady=(0, 0))
    cycle_metrics.columnconfigure(0, weight=1)
    cycle_metrics.columnconfigure(1, weight=1)

    gui._gait_cycle_metric_slots: list[tuple[tk.Label, tk.Label]] = []
    for index, title in enumerate(
        (LABEL_CADENCE_SHORT, "Stance L/R sym", "Swing L/R sym", "DS %")
    ):
        row, col = divmod(index, 2)
        cell = tk.Frame(cycle_metrics, bg=PANEL, highlightthickness=0)
        cell.grid(row=row, column=col, sticky="ew", padx=(0, 6), pady=1)
        title_lbl = tk.Label(
            cell,
            text=title,
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        )
        title_lbl.pack(side=tk.LEFT)
        value_lbl = tk.Label(
            cell,
            text="—",
            bg=PANEL,
            fg=TEXT,
            font=FONT_METRIC,
            anchor="w",
        )
        value_lbl.pack(side=tk.LEFT, padx=(4, 0))
        gui._gait_cycle_metric_slots.append((title_lbl, value_lbl))

    gui.lbl_gait_cycle_confidence = tk.Label(
        gait_cycle_panel,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_gait_cycle_confidence.pack(anchor=tk.W, pady=(2, 0))

    gui.lbl_gait_cycle_interp_compact = _compact_interp_label(
        gait_cycle_panel, gui, "lbl_gait_cycle_interp_compact"
    )
    gui.lbl_gait_cycle_interp_compact.pack(anchor=tk.W, pady=(2, 0))

    # ── Physics force estimation (vGRF research placeholder) ───────────────
    physics_force_panel = ttk.Frame(
        sidebar_inner,
        style="Card.TFrame",
        padding=DASHBOARD_SIDE_PAD,
    )
    gui._sidebar_physics_force_panel = physics_force_panel

    ttk.Label(
        physics_force_panel,
        text=LABEL_VIRTUAL_GRF_PANEL,
        style="SideMuted.TLabel",
    ).pack(anchor=tk.W, pady=(0, 2))
    from stablewalk.ui.metric_tooltips import attach_metric_tooltip

    gui.lbl_physics_force_status = tk.Label(
        physics_force_panel,
        text="Status: Not configured",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_physics_force_status.pack(anchor=tk.W, pady=(0, 1))

    gui.lbl_physics_force_method = tk.Label(
        physics_force_panel,
        text="Method: None",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_physics_force_method.pack(anchor=tk.W, pady=(0, 1))

    gui.lbl_physics_force_note = tk.Label(
        physics_force_panel,
        text="Contact mask ≠ force data",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=_side_wrap,
    )
    gui.lbl_physics_force_note.pack(anchor=tk.W, pady=(0, 0))
    for _w in (
        physics_force_panel,
        gui.lbl_physics_force_status,
        gui.lbl_physics_force_method,
        gui.lbl_physics_force_note,
    ):
        attach_metric_tooltip(_w, "grf")

    gui.lbl_real_to_sim_style = tk.Label(
        physics_force_panel,
        text="Gait style: —",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=_side_wrap,
    )
    gui.lbl_real_to_sim_style.pack(anchor=tk.W, pady=(4, 0))

    comparison_panel = ttk.Frame(
        gui._sidebar_hidden,
        style="Card.TFrame",
        padding=DASHBOARD_SIDE_PAD,
    )
    gui._sidebar_comparison_panel = comparison_panel
    gui._comparison_expanded = False

    gui.btn_toggle_comparison = ttk.Button(
        sidebar_inner,
        text="Compare",
        style="ExportAccent.TButton",
        command=gui._open_comparison_mode,
    )
    gui.btn_compare_gait = gui.btn_toggle_comparison

    gui.comparison_body = ttk.Frame(comparison_panel)
    gui.comparison_body.pack(fill=tk.X)
    gui.comparison_body.pack_forget()

    cmp_tools = ttk.Frame(gui.comparison_body)
    cmp_tools.pack(fill=tk.X, pady=(0, PAD_XS))
    ttk.Button(
        cmp_tools,
        text="Clear",
        style="Compact.TButton",
        command=gui._clear_demo_comparison,
    ).pack(side=tk.RIGHT)

    cmp_body = ttk.Frame(gui.comparison_body)
    cmp_body.pack(fill=tk.X)
    cmp_body.columnconfigure(0, weight=1)
    cmp_body.rowconfigure(0, weight=1)

    gui.demo_comparison_tree = _make_data_tree(
        cmp_body,
        ("demo_type", "joint", "max_angle", "avg_velocity"),
        {
            "demo_type": "Demo",
            "joint": "Joint",
            "max_angle": "Max °",
            "avg_velocity": "Avg v",
        },
        col_widths={
            "demo_type": 72,
            "joint": 64,
            "max_angle": 48,
            "avg_velocity": 52,
        },
        height=4,
        text_cols=frozenset({"demo_type", "joint"}),
        style="Compact.Treeview",
    )

    # ── OpenSim panel: dot status + export; details on demand ─────────────
    opensim_panel = ttk.Frame(
        sidebar_inner,
        style="Card.TFrame",
        padding=DASHBOARD_SIDE_PAD,
    )
    gui._sidebar_opensim_panel = opensim_panel

    opensim_actions = ttk.Frame(opensim_panel)
    opensim_actions.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=(2, 0))

    # Experimental IK actions stay in the menu — the sidebar keeps export + status only.
    gui.btn_opensim_run_demo_ik = ttk.Button(
        gui._sidebar_hidden,
        text="Demo IK",
        style="Compact.TButton",
        command=gui._run_opensim_demo_ik,
        state="disabled",
    )

    gui.btn_opensim_run_ik = ttk.Button(
        gui._sidebar_hidden,
        text="Run IK",
        style="Compact.TButton",
        command=gui._run_stablewalk_ik_experimental,
        state="disabled",
    )

    fmt_row = ttk.Frame(gui._sidebar_hidden)
    gui.opensim_motion_fmt = tk.StringVar(value=".mot")
    ttk.Combobox(
        fmt_row,
        textvariable=gui.opensim_motion_fmt,
        values=(".mot", ".csv"),
        state="readonly",
        width=8,
    ).pack(fill=tk.X)

    compact_status = ttk.Frame(opensim_panel)
    compact_status.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
    compact_status.columnconfigure(1, weight=1)

    gui.lbl_opensim_status_dot = tk.Label(
        compact_status,
        text="\u25cf",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI,
    )
    gui.lbl_opensim_status_dot.grid(row=0, column=0, sticky="w", padx=(0, 4))

    gui.lbl_opensim_compact_ready = ttk.Label(
        compact_status,
        text="OpenSim",
        style="SideAccent.TLabel",
        cursor="hand2",
    )
    gui.lbl_opensim_compact_ready.grid(row=0, column=1, sticky="w")
    gui.lbl_opensim_compact_ready.bind("<Button-1>", lambda _e: gui._open_opensim_details_dialog())

    gui.lbl_opensim_compact_mode = ttk.Label(compact_status, text="", style="SideMuted.TLabel")
    gui.lbl_opensim_compact_model = ttk.Label(compact_status, text="", style="SideMuted.TLabel")
    gui.lbl_opensim_compact_export = ttk.Label(compact_status, text="", style="SideMuted.TLabel")

    gui.btn_opensim_toggle_details = ttk.Button(
        gui._sidebar_hidden,
        text="Details",
        style="Compact.TButton",
        command=gui._open_opensim_details_dialog,
    )
    gui._opensim_details_visible = False

    gui._opensim_details_frame = ttk.Frame(gui._sidebar_hidden)

    scroll_host = ttk.Frame(gui._opensim_details_frame)
    scroll_host.pack(fill=tk.BOTH, expand=True)
    scroll_host.columnconfigure(0, weight=1)
    scroll_host.rowconfigure(0, weight=1)

    gui._opensim_scrollbar = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL)
    gui._opensim_scrollbar.grid(row=0, column=1, sticky="ns")

    gui._opensim_canvas = tk.Canvas(
        scroll_host,
        bg=PANEL,
        highlightthickness=0,
        borderwidth=0,
        height=OPENSIM_STATUS_SCROLL_HEIGHT,
    )
    gui._opensim_canvas.grid(row=0, column=0, sticky="nsew")
    gui._opensim_canvas.configure(yscrollcommand=gui._opensim_scrollbar.set)
    gui._opensim_scrollbar.configure(command=gui._opensim_canvas.yview)

    opensim_inner = ttk.Frame(gui._opensim_canvas)
    gui._opensim_canvas_window = gui._opensim_canvas.create_window(
        (0, 0),
        window=opensim_inner,
        anchor="nw",
    )

    def _opensim_scroll_region(_event: object | None = None) -> None:
        gui._opensim_canvas.configure(scrollregion=gui._opensim_canvas.bbox("all"))
        gui._opensim_canvas.itemconfigure(
            gui._opensim_canvas_window,
            width=max(gui._opensim_canvas.winfo_width(), max(sidebar.winfo_width(), _SIDEBAR_MIN_WIDTH) - 52),
        )

    def _opensim_mousewheel(event: tk.Event) -> None:
        if event.delta:
            gui._opensim_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    opensim_inner.bind("<Configure>", _opensim_scroll_region)
    gui._opensim_canvas.bind("<Configure>", _opensim_scroll_region)
    for widget in (opensim_inner, gui._opensim_canvas, scroll_host):
        widget.bind(
            "<Enter>",
            lambda _e: gui._opensim_canvas.bind_all("<MouseWheel>", _opensim_mousewheel),
        )
        widget.bind("<Leave>", lambda _e: gui._opensim_canvas.unbind_all("<MouseWheel>"))

    def _section(title: str, *, top_pad: int = 0) -> None:
        if top_pad:
            ttk.Frame(opensim_inner, height=top_pad).pack()
        ttk.Label(opensim_inner, text=title, style="SideAccent.TLabel").pack(
            anchor=tk.W, pady=(0, 3)
        )

    _section("OpenSim Status")
    gui.lbl_opensim_demo_mode = ttk.Label(
        opensim_inner,
        text="Presentation Demo Mode",
        style="SideAccent.TLabel",
        wraplength=_side_wrap,
    )
    gui.lbl_opensim_demo_subtitle = ttk.Label(
        opensim_inner,
        text="Synthetic demo — no video export required",
        style="SideHint.TLabel",
        wraplength=_side_wrap,
    )
    gui.lbl_opensim_demo_note = tk.Label(
        opensim_inner,
        text="",
        bg=SURFACE,
        fg=INFO,
        font=FONT_UI_XS,
        wraplength=_side_wrap,
        justify=tk.LEFT,
        anchor=tk.W,
    )

    gui.lbl_opensim_sdk = ttk.Label(
        opensim_inner, text="SDK: Not installed", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_sdk.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_mode = ttk.Label(
        opensim_inner, text="Mode: Export only", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_mode.pack(anchor=tk.W, pady=(0, 3))

    def _opensim_path_status_label(initial: str) -> tk.Label:
        return tk.Label(
            opensim_inner,
            text=initial,
            bg=SURFACE,
            fg=MUTED,
            font=FONT_UI_XS,
            wraplength=_side_wrap,
            justify=tk.LEFT,
            anchor=tk.W,
        )

    gui.lbl_opensim_pipeline_model = _opensim_path_status_label(
        "IK pipeline model: —"
    )
    gui.lbl_opensim_pipeline_model.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_suggested_model = _opensim_path_status_label("Suggested model: —")
    gui.lbl_opensim_suggested_model.pack(anchor=tk.W, pady=(0, 2))
    from stablewalk.ui.tk.sidebar_display import (
        LOADED_MODEL_NONE_HINT,
        SUGGESTED_MODEL_NOTE,
    )

    gui.lbl_opensim_suggested_note = ttk.Label(
        opensim_inner,
        text=SUGGESTED_MODEL_NOTE,
        style="SideHint.TLabel",
        wraplength=_side_wrap,
    )
    gui.lbl_opensim_suggested_note.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_loaded_model = _opensim_path_status_label(
        f"Loaded model: {LOADED_MODEL_NONE_HINT}"
    )
    gui.lbl_opensim_loaded_model.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_demo_ik = ttk.Label(
        opensim_inner, text="Demo IK: —", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_demo_ik.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_stablewalk_ik = ttk.Label(
        opensim_inner, text="StableWalk IK: Not ready", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_stablewalk_ik.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_ik = gui.lbl_opensim_stablewalk_ik

    gui.lbl_opensim_marker_mapping = ttk.Label(
        opensim_inner, text="Mapping: —", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_marker_mapping.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_reliability = tk.Label(
        opensim_inner,
        text="Reliability: Experimental",
        bg=SURFACE,
        fg=ORANGE,
        font=FONT_UI_XS,
        wraplength=_side_wrap,
        justify=tk.LEFT,
        anchor=tk.W,
    )
    gui.lbl_opensim_reliability.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_warning = gui.lbl_opensim_reliability
    gui.lbl_opensim_coverage = ttk.Label(
        opensim_inner, text="Coverage: —", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_coverage.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_markers_count = ttk.Label(
        opensim_inner, text="Markers: —", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_markers_count.pack(anchor=tk.W, pady=(0, PAD_XS))

    ttk.Separator(opensim_inner, orient=tk.HORIZONTAL).pack(
        fill=tk.X, pady=(PAD_XS, PAD_XS)
    )
    _section("Export Status")
    gui.lbl_opensim_trc = ttk.Label(
        opensim_inner, text="TRC: Missing", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_trc.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_mot = ttk.Label(
        opensim_inner, text="MOT: Missing", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_mot.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_json = ttk.Label(
        opensim_inner, text="JSON: Missing", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_json.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_stablewalk_trc = ttk.Label(
        opensim_inner, text="Mapped TRC: Missing", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_stablewalk_trc.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_mapped_trc = gui.lbl_opensim_stablewalk_trc

    gui.lbl_opensim_last_export = ttk.Label(
        opensim_inner,
        text="Last Export: Not yet exported",
        style="SideHint.TLabel",
    )
    gui.lbl_opensim_last_export.pack(anchor=tk.W, pady=(0, 2))
    gui.lbl_opensim_last_ik = ttk.Label(
        opensim_inner,
        text="Last IK: Not yet run",
        style="SideHint.TLabel",
    )
    gui.lbl_opensim_last_ik.pack(anchor=tk.W, pady=(0, PAD_XS))

    ttk.Separator(opensim_inner, orient=tk.HORIZONTAL).pack(
        fill=tk.X, pady=(PAD_XS, PAD_XS)
    )
    _section("Details")
    gui.btn_opensim_mapping_report = ttk.Button(
        opensim_inner,
        text="View Mapping Report",
        style="Secondary.TButton",
        command=gui._show_marker_mapping_report,
    )
    gui.btn_opensim_mapping_report.pack(fill=tk.X, pady=(0, PAD_XS))
    gui.btn_opensim_open_output = ttk.Button(
        opensim_inner,
        text="Open Output Folder",
        style="Secondary.TButton",
        command=gui._open_opensim_folder,
    )
    gui.btn_opensim_open_output.pack(fill=tk.X, pady=(0, PAD_XS))
    gui.btn_opensim_view_folder = gui.btn_opensim_open_output
    gui.btn_opensim_folder = gui.btn_opensim_open_output
    gui.btn_opensim_view_log = ttk.Button(
        opensim_inner,
        text="View OpenSim Log",
        style="Secondary.TButton",
        command=gui._show_opensim_log,
    )
    gui.btn_opensim_view_log.pack(fill=tk.X, pady=(0, PAD_XS))
    gui.btn_opensim_details = gui.btn_opensim_view_log

    ttk.Separator(opensim_inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, PAD_XS))
    from stablewalk.ui.tk.sidebar_display import (
        OPENSIM_MODEL_DROPDOWN_HINT,
        OPENSIM_SUGGESTED_MODEL_SECTION,
    )

    ttk.Label(
        opensim_inner,
        text=OPENSIM_SUGGESTED_MODEL_SECTION,
        style="SideMuted.TLabel",
    ).pack(anchor=tk.W, pady=(0, 2))
    gui.btn_opensim_select_model = ttk.Button(
        opensim_inner,
        text="Load .osim Model",
        style="Secondary.TButton",
        command=gui._select_opensim_model,
    )
    gui.btn_opensim_select_model.pack(fill=tk.X, pady=(0, PAD_XS))

    local_row = ttk.Frame(opensim_inner)
    local_row.pack(fill=tk.X, pady=(0, PAD_XS))
    gui.opensim_model_var = tk.StringVar(value="")
    gui.opensim_model_combo = ttk.Combobox(
        local_row,
        textvariable=gui.opensim_model_var,
        state="readonly",
    )
    gui.opensim_model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
    gui.opensim_model_combo.bind(
        "<<ComboboxSelected>>", gui._on_opensim_suggested_model_combo
    )
    gui.btn_opensim_local_model = ttk.Button(
        local_row,
        text="Select",
        style="Secondary.TButton",
        width=8,
        command=gui._select_discovered_opensim_model,
        state="disabled",
    )
    gui.btn_opensim_local_model.pack(side=tk.LEFT, padx=(PAD_XS, 0))
    gui.lbl_opensim_model_dropdown_hint = ttk.Label(
        opensim_inner,
        text=OPENSIM_MODEL_DROPDOWN_HINT,
        style="SideHint.TLabel",
        wraplength=_side_wrap,
    )
    gui.lbl_opensim_model_dropdown_hint.pack(anchor=tk.W, pady=(0, PAD_XS))

    gui.lbl_opensim_export_status = gui.lbl_opensim_trc
    gui.lbl_opensim_files = gui.lbl_opensim_trc

    # Point Details — kept for logic but hidden from the main dashboard.
    gui._point_details_hidden = ttk.Frame(gui.root)
    details = ttk.Frame(gui._point_details_hidden)

    gui.lbl_dof_summary = ttk.Label(
        details,
        text="No joint selected",
        style="SideAccent.TLabel",
        wraplength=SIDEBAR_WIDTH - 28,
    )
    gui.lbl_dof_summary.pack(anchor=tk.W, pady=(0, PAD_XS))

    gui.lbl_details_empty = ttk.Label(
        details,
        text="Select points to see live metrics.",
        style="SideEmpty.TLabel",
        wraplength=_side_wrap,
    )
    gui.lbl_details_empty.pack(anchor=tk.W, pady=(PAD_XS, 0))

    gui.dof_details_frame = ttk.Frame(details)
    gui.dof_details_frame.pack(fill=tk.X, pady=(PAD_XS, 0))

    gui.lbl_compare = ttk.Label(
        gui._sidebar_hidden,
        text="",
        style="SideMuted.TLabel",
        wraplength=SIDEBAR_WIDTH - 28,
    )

    gui.btn_opensim_export = gui.btn_opensim_export_data

    bind_panel_word_wrap(
        gui,
        sidebar,
        (
            "lbl_summary_ms_explain",
            "lbl_summary_gq_explain",
            "lbl_summary_ac_explain",
            "lbl_summary_demo_headline",
            "lbl_summary_demo_compare",
            "lbl_opensim_compact_ready",
        ),
    )

    gui.dof_pos_tree._sw_host.pack_forget()  # type: ignore[attr-defined]

    bind_panel_word_wrap(
        gui,
        data_row,
        ("lbl_export_output_folder",),
    )

    finalize_responsive_dashboard(gui)

    # Apply the remembered professional view mode now that all panels exist.
    from stablewalk.ui.tk.dashboard_overview_view_mode import apply_overview_view_mode

    try:
        apply_overview_view_mode(gui, animate=False, persist=False)
    except Exception:
        pass

    from stablewalk.ui.tk.dashboard_panel_maximize import wire_all_panel_maximize_buttons

    try:
        wire_all_panel_maximize_buttons(gui)
    except Exception:
        pass

    _schedule_dashboard_reflow(gui)


def _schedule_dashboard_reflow(gui) -> None:
    """Reflow matplotlib canvases and sidebar wrap after the first layout pass."""

    def _reflow() -> None:
        for attr in ("_sync_analysis_sidebar_wrap", "_sync_utility_sidebar_wrap"):
            sync_wrap = getattr(gui, attr, None)
            if sync_wrap is not None:
                sync_wrap()
        sync_scroll = getattr(gui, "_sync_tab_advanced_scroll", None) or getattr(
            gui, "_sync_dashboard_scroll", None
        )
        if sync_scroll is not None:
            sync_scroll()
        bind_wheel = getattr(gui, "_motion_bind_wheel", None)
        sections = getattr(gui, "_motion_scroll_sections", None)
        if callable(bind_wheel) and sections is not None:
            try:
                bind_wheel(sections)
            except Exception:
                pass
        sync_motion = getattr(gui, "_sync_motion_scroll", None)
        if sync_motion is not None:
            try:
                sync_motion()
            except Exception:
                pass
        from stablewalk.ui.tk.dashboard_notebook import reflow_tab_canvases

        reflow_tab_canvases(gui)
        if hasattr(gui, "sidebar"):
            try:
                gui.sidebar.update_idletasks()
            except tk.TclError:
                pass
        if hasattr(gui, "_fit_skeleton_canvas"):
            gui._fit_skeleton_canvas()
        if hasattr(gui, "_fit_dof_traj_canvas"):
            gui._fit_dof_traj_canvas()
        fit_contact = getattr(gui, "_fit_contact_gait_canvas", None)
        if callable(fit_contact):
            try:
                fit_contact()
            except Exception:
                pass
        fit_biomech = getattr(gui, "_fit_biomech_canvas", None)
        if callable(fit_biomech):
            try:
                fit_biomech()
            except Exception:
                pass
        from stablewalk.ui.tk.gui_layout_debug import log_gui_layout_audit_if_enabled

        log_gui_layout_audit_if_enabled(gui, context="dashboard_reflow")

    gui.root.after_idle(_reflow)
    gui.root.after(120, _reflow)
