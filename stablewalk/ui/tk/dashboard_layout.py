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

from stablewalk.ui.tk.layout_playback import resolve_gui, should_defer_layout
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
    UNITS_COORD_LEGEND,
)
from stablewalk.ui.tk.metric_help import add_metric_help_icon
from stablewalk.ui.theme import (
    ACCENT,
    ACCENT_ALT,
    BG,
    BORDER,
    BORDER_SUBTLE,
    ELEVATED,
    EMPTY_OVERVIEW_JOINT_INSPECT,
    EMPTY_SELECT_DOF_TABLE,
    EMPTY_SELECT_DOF_CHART,
    EMPTY_VIDEO_TEXT,
    FONT_CAPTION,
    FONT_MONO,
    FONT_MONO_SM,
    FONT_KPI_SUPPORTING,
    FONT_KPI_TITLE,
    SUCCESS_SOFT,
    FONT_METRIC,
    FONT_METRIC_TITLE,
    FONT_METRIC_VALUE,
    FONT_METRIC_VALUE_ACCENT,
    FONT_PANEL_HEADER,
    FONT_SECTION,
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
    OVERVIEW_ROW_GAP,
    OVERVIEW_VIZ_TITLE_PADX,
    OVERVIEW_VIZ_TITLE_PADY,
    OVERVIEW_VIZ_TITLE_SPACER_PX,
    OVERVIEW_VIZ_TOOLBAR_MIN_H,
    OVERVIEW_VIZ_TOOLBAR_PADX,
    OVERVIEW_VIZ_TOOLBAR_PADY,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    PANEL_HOVER,
    REFRESH_INTERVAL_CHOICES,
    TEXT,
    TEXT_SECONDARY,
    TREE_LOW_CONFIDENCE_BG,
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
    create_elevated_card,
    disclosure_label,
    format_labelframe_title,
    set_wraplength_safe,
    tk_labelframe_title,
)
from stablewalk.ui.colors import SIDE_LEFT, SIDE_RIGHT
from stablewalk.ui.tk.analysis_split import MIN_H_CHART, MIN_W_CHART, MIN_W_PATH
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

# Primary visual area: Original Video + 3D Reconstruction (45/55).
# Summary docks below — weights mirror dashboard_sections.SEC1_*.
_TOP_VIDEO_WEIGHT = 45
_TOP_SKELETON_WEIGHT = 55
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
# Inner media viewports can yield on short windows; the outer visualization row
# retains the larger soft floor and expands normally with available space.
_OVERVIEW_MEDIA_VIEWPORT_MIN_H = 120
_OVERVIEW_TRAJ_CANVAS_MINSIZE = _OVERVIEW_MEDIA_VIEWPORT_MIN_H
_OVERVIEW_PATH_VIEW_MINWIDTH = 0
# Wide default: 3D Path ~80% | Coordinate Inspector ~20%. Inspector width is
# floored in reflow so labels stay readable; leftover width goes to the cube.
_OVERVIEW_PATH_VIEW_WEIGHT = 80
_OVERVIEW_COORD_INSPECTOR_WEIGHT = 20
_OVERVIEW_COORD_INSPECTOR_MINWIDTH = 0
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
# Motion Analysis graph row — usable floor, extra height from weight=1.
_MOTION_GRAPH_ROW_MINSIZE = MIN_H_CHART
# Foot contact + gait phase + Estimated Virtual GRF — three stacked plots.
CONTACT_GAIT_CHART_MIN_H = 400
_MOTION_TOP_ROW_MIN_H = 400
_MOTION_KNEE_PANEL_MIN_W = MIN_W_CHART
_MOTION_TRAJ_PANEL_MIN_W = MIN_W_PATH
# 50 / 50 knee kinematics | selected-joint path (temporal metrics live on Biomech).
MOTION_TOP_COLUMN_WEIGHTS = (1, 1)
_MOTION_METRICS_PANEL_MIN_W = 190  # fallback only if metrics remain on Motion

_OVERVIEW_TRAJ_DOCK_MINSIZE = 260
from stablewalk.ui.tk.resize_coalesce import FIGURE_RESIZE_SETTLE_MS

_OVERVIEW_TRAJ_RESIZE_DEBOUNCE_MS = FIGURE_RESIZE_SETTLE_MS
_OVERVIEW_SKEL_RESIZE_DEBOUNCE_MS = FIGURE_RESIZE_SETTLE_MS
_FIGURE_RESIZE_DEBOUNCE_MS = FIGURE_RESIZE_SETTLE_MS
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

    Joint chips (Compare Joints) stay outside this shell so they remain a compact footer.
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
    vsb.grid_remove()
    canvas.configure(yscrollcommand=vsb.set)

    scroll_content = ttk.Frame(canvas)
    inner_id = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
    scroll_content.columnconfigure(0, weight=1)
    scroll_content.rowconfigure(0, weight=1)

    sections = ttk.Frame(scroll_content)
    sections.grid(row=0, column=0, sticky="nsew", pady=(0, PAD_SM))
    sections.columnconfigure(0, weight=1)
    sections.rowconfigure(0, weight=1)

    # No forever-mapped spacer row — bottom pad comes from sections pady above.
    spacer = None

    gui._motion_scroll_outer = outer
    gui._motion_scroll_canvas = canvas
    gui._motion_scrollbar = vsb
    gui._motion_scroll_inner = scroll_content
    gui._motion_scroll_window_id = inner_id
    gui._motion_scroll_sections = sections

    def _sync(_event: object | None = None) -> None:
        try:
            if should_defer_layout(gui):
                return
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
                + int(PAD_SM),
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
        if not vsb.winfo_ismapped() or not getattr(event, "delta", None):
            return None
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

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
    """Empty laboratory axes at GUI init — no invented path data."""
    if getattr(gui, "canvas_dof_traj", None) is None or getattr(gui, "ax_dof_traj", None) is None:
        return

    ax = gui.ax_dof_traj
    ax.cla()
    if hasattr(ax, "_stablewalk_traj_artists"):
        del ax._stablewalk_traj_artists
    if hasattr(ax, "_stablewalk_stable_viewport"):
        del ax._stablewalk_stable_viewport
    ax._stablewalk_plot_legend = None

    ax.set_facecolor(PANEL)
    gui.fig_dof_traj.patch.set_facecolor(PANEL)
    ax.set_xlabel("X · Lat (cm)", color=TEXT, fontsize=8)
    ax.set_ylabel("Y · Up (cm)", color=TEXT, fontsize=8)
    ax.set_zlabel("Z · Fwd (cm)", color=TEXT, fontsize=8)
    ax.set_xlim(-10.0, 10.0)
    ax.set_ylim(-10.0, 10.0)
    ax.set_zlim(-10.0, 10.0)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=20.0, azim=-60.0)
    ax.text2D(
        0.5,
        0.5,
        EMPTY_SELECT_DOF_CHART,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=9,
    )

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
        if should_defer_layout(widget=widget):
            return False
        try:
            mapped = getattr(widget, "winfo_ismapped", None)
            if callable(mapped) and not bool(mapped()):
                return False
        except tk.TclError:
            return False
        target = host if host is not None else widget.master
        if target is None:
            target = widget
        # Prefer cached host size while avoiding forced idle-task storms.
        try:
            width = int(target.winfo_width())
            height = int(target.winfo_height())
        except tk.TclError:
            return False
        if width < min_px or height < min_px:
            from stablewalk.ui.tk.resize_coalesce import resize_drag_active

            gui = resolve_gui(widget)
            if gui is not None and resize_drag_active(gui):
                return False
            try:
                target.update_idletasks()
                width = int(target.winfo_width())
                height = int(target.winfo_height())
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
        # Prefer TkAgg resize. Pin the Tk widget to the host pixels while the
        # fitting guard is set so a tall natural figure cannot paint over
        # sibling panels (Windows Frames do not clip children).
        try:
            canvas.resize(type("E", (), {"width": int(width), "height": int(height)})())
        except (TypeError, AttributeError, tk.TclError):
            try:
                fig.set_size_inches(width / dpi, height / dpi, forward=True)
            except Exception:
                return False
        try:
            widget.configure(width=int(width), height=int(height))
        except tk.TclError:
            pass
        if margins is not None:
            try:
                resolved = margins(fig) if callable(margins) else margins
                if resolved:
                    fig.subplots_adjust(**resolved)
            except Exception:
                pass
        from stablewalk.ui.tk.redraw_policy import full_tab_redraw_blocked

        gui = resolve_gui(widget)
        if gui is not None and full_tab_redraw_blocked(gui):
            return True
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
        if should_defer_layout(widget=widget):
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
    """Resize the skeleton figure and preserve equal-aspect layout.

    Host binding only — the gated ``_on_viz_row_resize`` path also fits the
    skeleton; binding the canvas widget itself caused duplicate Configure work.
    """
    state: dict[str, object | None] = {"job": None, "event": None}

    def _fit(event: tk.Event | object) -> None:
        widget = canvas.get_tk_widget()
        if should_defer_layout(widget=widget):
            return
        host = widget.master
        width = int(getattr(event, "width", 0) or 0)
        height = int(getattr(event, "height", 0) or 0)
        if host is not None:
            try:
                host_w = int(host.winfo_width())
                host_h = int(host.winfo_height())
            except tk.TclError:
                host_w = host_h = 0
            if host_w < min_px or host_h < min_px:
                try:
                    host.update_idletasks()
                    host_w = int(host.winfo_width())
                    host_h = int(host.winfo_height())
                except tk.TclError:
                    return
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
        try:
            widget.configure(width=int(width), height=int(height))
        except tk.TclError:
            pass
        relayout_skeleton_viewport(ax)
        canvas.draw_idle()

    def _on_resize(event: tk.Event) -> None:
        if should_defer_layout(widget=canvas.get_tk_widget()):
            return
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

        state["job"] = widget.after(_OVERVIEW_SKEL_RESIZE_DEBOUNCE_MS, _apply)

    widget = canvas.get_tk_widget()
    widget.configure(bg=PANEL, highlightthickness=0)
    # Host-only — avoids canvas.resize → Configure feedback on the widget.
    host = widget.master
    if host is not None:
        host.bind("<Configure>", _on_resize, add="+")
    else:
        widget.bind("<Configure>", _on_resize, add="+")


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
    if should_defer_layout(widget=widget):
        return False
    host = graph_host if graph_host is not None else widget.master
    overview_dock = bool(getattr(ax, "_stablewalk_overview_dock", False))
    if overview_dock:
        pad_l, pad_t, pad_r, pad_b = _OVERVIEW_TRAJ_GRAPH_INNER_PAD
    else:
        pad_l, pad_t, pad_r, pad_b = _ANALYSIS_GRAPH_INNER_PAD
    host_w = 0
    host_h = 0

    if host is not None:
        try:
            host_w = int(host.winfo_width())
            host_h = int(host.winfo_height())
        except tk.TclError:
            host_w = host_h = 0
        if host_w < min_px or host_h < min_px:
            try:
                host.update_idletasks()
                host_w = int(host.winfo_width())
                host_h = int(host.winfo_height())
            except tk.TclError:
                return False

    if host_w < min_px or host_h < min_px:
        # Do not fall back to the figure's natural HWND size — that paints
        # the 3D path through the gait-phase strip on Windows.
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

    # Fixed-size policy: pin once when the host size changes, not every paint.
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
    canvas: FigureCanvasTkAgg | None,
    *,
    overview_dock: bool = False,
) -> None:
    """Grid the trajectory canvas once — never re-pack during resize."""
    if canvas is None:
        return
    widget = canvas.get_tk_widget()
    pad_l, pad_t, pad_r, pad_b = (
        _OVERVIEW_TRAJ_GRAPH_INNER_PAD if overview_dock else _ANALYSIS_GRAPH_INNER_PAD
    )
    from stablewalk.ui.tk.clip_viewport import ensure_widget_fills_host

    try:
        if not widget.grid_info():
            widget.grid(
                row=0,
                column=0,
                sticky=_ANALYSIS_GRAPH_CANVAS_STICKY,
                padx=(pad_l, pad_r),
                pady=(pad_t, pad_b),
            )
        else:
            widget.grid_configure(sticky=_ANALYSIS_GRAPH_CANVAS_STICKY)
        ensure_widget_fills_host(widget)
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
            ensure_widget_fills_host(widget)
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
    state = {"after_id": None, "last_size": None}

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
        if should_defer_layout(widget=canvas.get_tk_widget()):
            return
        width = int(getattr(event, "width", 0) or 0)
        height = int(getattr(event, "height", 0) or 0)
        # Host Configure events can report a single axis during mid-layout; only
        # ignore clearly empty sizes. The fit helper re-reads host winfo_*.
        if width < 2 and height < 2:
            return
        last = state.get("last_size")
        if isinstance(last, tuple) and abs(width - last[0]) < 2 and abs(height - last[1]) < 2:
            return
        state["last_size"] = (width, height)
        if delay <= 0:
            _reflow()
            return
        try:
            widget = canvas.get_tk_widget()
            root = widget.winfo_toplevel()
        except tk.TclError:
            return
        prev = state["after_id"]
        if prev is not None:
            try:
                root.after_cancel(prev)
            except (tk.TclError, ValueError):
                pass
        try:
            state["after_id"] = root.after(delay, _reflow)
        except tk.TclError:
            state["after_id"] = None

    widget = canvas.get_tk_widget()
    widget.configure(bg=PANEL, highlightthickness=0)
    # Host-only Configure — canvas widget Configure fires on every pinned resize.
    host = graph_host if graph_host is not None else widget.master
    if host is not None and host is not widget:
        host.bind("<Configure>", _on_resize, add="+")
    else:
        widget.bind("<Configure>", _on_resize, add="+")
    # One parent hop only — avoid binding every ancestor (reflow storms).
    parent = getattr(host, "master", None) if host is not None else None
    if parent is not None and parent is not host and parent is not widget:
        parent.bind("<Configure>", _on_resize, add="+")
    for extra in extra_hosts or ():
        if (
            extra is not None
            and extra is not host
            and extra is not widget
            and extra is not parent
        ):
            extra.bind("<Configure>", _on_resize, add="+")


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
    pad = 3 if compact else 2
    cell_pad = (1, 2) if compact else (1, 1)
    title_pad = (2, 1) if compact else (2, 0)
    value_pad = (0, 3) if compact else (0, 3)
    inner_x = 6 if compact else 8
    grid_kwargs = dict(row=row, column=column, sticky="nsew", padx=pad, pady=cell_pad)

    if accent and not compact:
        shell = tk.Frame(parent, bg=ACCENT, highlightthickness=0)
        shell.grid(**grid_kwargs)
        cell = tk.Frame(shell, bg=ELEVATED, highlightthickness=0)
        cell.pack(fill=tk.BOTH, expand=True, padx=(2, 0))
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
    cell.grid(row=row, column=column, sticky="nsew", padx=PAD_XS, pady=PAD_XS)
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
    title_lbl.pack(fill=tk.X, padx=PAD_SM, pady=(PAD_XS, 0))
    value_lbl = tk.Label(
        cell,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=value_font,
        anchor="w",
    )
    value_lbl.pack(fill=tk.X, padx=PAD_SM, pady=(0, PAD_XS))
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
    cell.grid(row=row, column=0, sticky="ew", padx=PAD_SM, pady=(0, PAD_XS))
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
    value_lbl.pack(anchor="w", pady=(PAD_XS // 4, 0))
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
    row_frame.grid(row=row, column=0, sticky="ew", padx=PAD_SM, pady=(0, PAD_SM))
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
        lbl.pack(anchor="w", pady=(PAD_XS // 4, 0))
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

    card = create_elevated_card(host)
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
    return format_labelframe_title(label)


def _viz_labelframe_title(parent: tk.Misc, text: str) -> tk.Frame:
    """Overview panel caption with a left spacer so Windows cannot clip glyphs.

    ttk.LabelFrame places ``labelwidget`` at x=0. The first pixels are often
    sheared off; an empty spacer absorbs that clip so Video, Skeleton, and
    3D Path titles sit on the same baseline with matching padding.
    """
    host = tk.Frame(parent, bg=PANEL, highlightthickness=0, bd=0)
    spacer = tk.Frame(
        host,
        bg=PANEL,
        width=OVERVIEW_VIZ_TITLE_SPACER_PX,
        height=1,
        highlightthickness=0,
        bd=0,
    )
    spacer.pack(side=tk.LEFT)
    spacer.pack_propagate(False)
    lbl = tk_labelframe_title(host, text)
    try:
        lbl.configure(
            padx=OVERVIEW_VIZ_TITLE_PADX,
            pady=OVERVIEW_VIZ_TITLE_PADY,
            anchor="w",
        )
    except tk.TclError:
        pass
    lbl.pack(side=tk.LEFT)
    host.title_label = lbl  # type: ignore[attr-defined]
    return host


def _sidebar_rule(parent: tk.Misc) -> None:
    tk.Frame(parent, bg=BORDER_SUBTLE, height=1, highlightthickness=0).pack(
        fill=tk.X, pady=(PAD_SM, PAD_SM)
    )


def _analysis_sidebar_heading(parent: tk.Misc, text: str, *, first: bool = False) -> None:
    tk.Label(
        parent,
        text=text,
        bg=PANEL,
        fg=ACCENT_ALT,
        font=FONT_UI_XS,
        anchor=tk.W,
    ).pack(fill=tk.X, pady=(0 if first else PAD_MD, PAD_XS))


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
    add_metric_help_icon(
        row, title, METRIC_HELP.get(help_key, title), metric_key=help_key
    )
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


def _install_coord_inspector_scroll(
    parent: tk.Misc,
) -> tuple[tk.Frame, tk.Frame]:
    """Coordinate Inspector host without a nested scrollbar.

    Advanced inspector rows stay collapsed. The Overview pane already
    scrolls when needed, so a second Canvas+scrollbar nested two viewports.
    """
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)

    inner = tk.Frame(
        parent,
        bg=PANEL,
        highlightthickness=0,
        padx=PAD_SM,
        pady=PAD_XS,
    )
    inner.grid(row=0, column=0, sticky="nsew")
    inner._coord_inspector_window_id = None  # type: ignore[attr-defined]
    inner._coord_inspector_scrollbar = None  # type: ignore[attr-defined]
    return inner, inner


def _build_overview_traj_readout(
    parent: tk.Misc,
    gui: object | None = None,
) -> dict[str, tk.Label]:
    """Compact 3D Coordinate Inspector — Joint/Frame/Time/XYZ, Details for the rest."""
    from stablewalk.ui.overview_coordinate_inspector import (
        COORDINATE_INSPECTOR_DETAILS_KEYS,
        COORDINATE_INSPECTOR_PRIMARY_KEYS,
        COORDINATE_INSPECTOR_TITLES,
        INSPECTOR_UNAVAILABLE_DISPLAY,
    )
    from stablewalk.ui.theme import GLYPH_COLLAPSE, GLYPH_EXPAND

    # Family accents used by live playback updates (labels stay in `values`).
    family_colors = {
        "identity": TEXT,
        "position": ACCENT_ALT,
        "motion": WARNING,
        "quality": SUCCESS_SOFT,
    }

    font_title = FONT_SECTION
    font_label = FONT_UI_XS
    font_value = FONT_MONO

    card_pad_x = PAD_SM
    card_pad_y = max(2, PAD_XS // 2)
    row_pad_y = 1
    label_col_w = max(
        (
            len(COORDINATE_INSPECTOR_TITLES.get(k, k))
            for k in (*COORDINATE_INSPECTOR_PRIMARY_KEYS, *COORDINATE_INSPECTOR_DETAILS_KEYS)
        ),
        default=10,
    )
    # Single-line field names ("Acceleration"); extra width is allocated to
    # the inspector rail so wraplength cannot clip the first glyphs.
    label_col_minsize = max(110, min(132, int(label_col_w * 7.2) + PAD_SM))

    shell = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    shell.pack(fill=tk.X, expand=False, padx=(PAD_XS, PAD_XS), pady=(0, PAD_XS))

    header = tk.Frame(shell, bg=PANEL, highlightthickness=0)
    header.pack(fill=tk.X, expand=False, pady=(0, 2))
    title_lbl = tk.Label(
        header,
        text="Coordinate Inspector",
        bg=PANEL,
        fg=TEXT,
        font=font_title,
        anchor="w",
        justify=tk.LEFT,
        wraplength=200,
    )
    title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor="w")
    pop_lbl = tk.Label(
        header,
        text="Open",
        bg=PANEL,
        fg=ACCENT_ALT,
        font=font_label,
        cursor="hand2",
        anchor="e",
    )
    pop_lbl.pack(side=tk.RIGHT)
    if gui is not None:
        pop_lbl.bind(
            "<Button-1>",
            lambda _e: gui._open_coordinate_inspector_window(),  # type: ignore[attr-defined]
        )
        create_tooltip(
            pop_lbl,
            "Open the 3D Coordinate Inspector in a window with a joint picker and frame slider.",
        )

    stack = tk.Frame(shell, bg=PANEL, highlightthickness=0)
    stack.pack(fill=tk.X, expand=False)
    stack.columnconfigure(0, weight=1)

    values: dict[str, tk.Label] = {}
    name_labels: list[tk.Label] = []
    wrap_values: list[tk.Label] = []
    wrap_bodies: list[tk.Frame] = []

    card = tk.Frame(stack, bg=PANEL, highlightthickness=0)
    card.grid(row=0, column=0, sticky="ew")
    card.columnconfigure(0, weight=1)

    def _add_rows(body: tk.Frame, keys: tuple[str, ...]) -> None:
        body.columnconfigure(0, weight=0, minsize=label_col_minsize)
        body.columnconfigure(1, weight=1)
        wrap_bodies.append(body)
        for row_i, key in enumerate(keys):
            name_lbl = tk.Label(
                body,
                text=COORDINATE_INSPECTOR_TITLES.get(key, key),
                bg=PANEL,
                fg=MUTED,
                font=font_label,
                anchor="w",
                justify=tk.LEFT,
                wraplength=0,
            )
            name_lbl.grid(
                row=row_i,
                column=0,
                sticky="nw",
                pady=(row_pad_y, row_pad_y),
                padx=(PAD_XS, PAD_SM),
            )
            name_labels.append(name_lbl)
            value_lbl = tk.Label(
                body,
                text=INSPECTOR_UNAVAILABLE_DISPLAY,
                bg=PANEL,
                fg=MUTED,
                font=font_value,
                anchor="ne",
                justify=tk.RIGHT,
                wraplength=120,
            )
            value_lbl.grid(
                row=row_i,
                column=1,
                sticky="ne",
                pady=(row_pad_y, row_pad_y),
            )
            values[key] = value_lbl
            wrap_values.append(value_lbl)

    primary = tk.Frame(card, bg=PANEL, highlightthickness=0)
    primary.grid(
        row=0,
        column=0,
        sticky="ew",
        padx=card_pad_x,
        pady=(card_pad_y, 0),
    )
    _add_rows(primary, COORDINATE_INSPECTOR_PRIMARY_KEYS)

    details_toggle = tk.Label(
        card,
        text=f"{GLYPH_EXPAND} More",
        bg=PANEL,
        fg=MUTED,
        font=font_label,
        anchor="w",
        cursor="hand2",
        padx=PAD_XS,
    )
    details_toggle.grid(
        row=1,
        column=0,
        sticky="w",
        padx=card_pad_x,
        pady=(2, card_pad_y),
    )

    details = tk.Frame(card, bg=PANEL, highlightthickness=0)
    details.grid(
        row=2,
        column=0,
        sticky="ew",
        padx=card_pad_x,
        pady=(0, card_pad_y),
    )
    _add_rows(details, COORDINATE_INSPECTOR_DETAILS_KEYS)
    details.grid_remove()
    details_state = {"expanded": False}

    def _toggle_details(_event: object | None = None) -> None:
        details_state["expanded"] = not details_state["expanded"]
        expanded = details_state["expanded"]
        try:
            details_toggle.configure(
                text=f"{GLYPH_COLLAPSE} More" if expanded else f"{GLYPH_EXPAND} More"
            )
            if expanded:
                details.grid()
            else:
                details.grid_remove()
        except tk.TclError:
            return
        _sync_inspector_wrap()

    details_toggle.bind("<Button-1>", _toggle_details)

    def _sync_inspector_wrap(_event: object | None = None) -> None:
        try:
            width = int(shell.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 40:
            return
        inner_w = max(120, width - 4)
        title_wrap = max(110, inner_w - 8)
        set_wraplength_safe(title_lbl, title_wrap)
        label_w = max(110, min(label_col_minsize, max(110, inner_w * 2 // 5)))
        value_w = max(72, inner_w - label_w - PAD_SM - 12)
        for body in wrap_bodies:
            try:
                body.columnconfigure(0, minsize=label_w)
            except tk.TclError:
                pass
        for lbl in name_labels:
            set_wraplength_safe(lbl, 0)
        for lbl in wrap_values:
            set_wraplength_safe(lbl, value_w)

    shell.bind("<Configure>", _sync_inspector_wrap, add="+")
    parent._coord_inspector_family_colors = family_colors  # type: ignore[attr-defined]
    parent._coord_inspector_wrap_sync = _sync_inspector_wrap  # type: ignore[attr-defined]
    bind_wheel = getattr(parent, "_bind_coord_inspector_wheel", None)
    if callable(bind_wheel):
        bind_wheel(shell)
        bind_wheel(details)
    return values


def _build_overview_path_extents(parent: tk.Misc) -> dict[str, tk.Label]:
    """Live Bounding Box + X/Y/Z ranges under the Path viewport."""
    from stablewalk.ui.overview_coordinate_inspector import (
        PATH_EXTENT_KEYS,
        PATH_EXTENT_TITLES,
    )

    font_label = ("Segoe UI", 9)
    font_value = ("Cascadia Mono", 10)
    try:
        _probe = tk.Label(parent, font=font_value)
        _probe.destroy()
    except tk.TclError:
        font_value = (FONT_MONO[0], 10) if isinstance(FONT_MONO, tuple) else FONT_MONO

    from stablewalk.ui.tk.tab_grid import grid_ew, stretch_columns

    shell = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    stretch_columns(parent)
    grid_ew(shell, row=0, pady=(0, PAD_XS))

    card = tk.Frame(
        shell,
        bg=ELEVATED,
        highlightthickness=0,
    )
    card.pack(fill=tk.X)
    card.columnconfigure(1, weight=1)

    tk.Frame(card, bg=ACCENT_ALT, height=2, highlightthickness=0).grid(
        row=0, column=0, columnspan=2, sticky="ew"
    )
    tk.Label(
        card,
        text="Path Extents",
        bg=ELEVATED,
        fg=ACCENT_ALT,
        font=FONT_CAPTION,
        anchor="w",
    ).grid(row=1, column=0, columnspan=2, sticky="w", padx=PAD_SM, pady=(PAD_SM, PAD_XS))

    values: dict[str, tk.Label] = {}
    for i, key in enumerate(PATH_EXTENT_KEYS):
        r = i + 2
        tk.Label(
            card,
            text=PATH_EXTENT_TITLES.get(key, key),
            bg=ELEVATED,
            fg=MUTED,
            font=font_label,
            anchor="w",
        ).grid(row=r, column=0, sticky="w", padx=(PAD_SM, PAD_SM), pady=(3, 3))
        value_lbl = tk.Label(
            card,
            text="N/A",
            bg=ELEVATED,
            fg=TEXT,
            font=font_value,
            anchor="e",
            justify=tk.RIGHT,
        )
        value_lbl.grid(row=r, column=1, sticky="ew", padx=(0, PAD_SM), pady=(3, 3))
        values[key] = value_lbl
    return values


def _build_compact_graph_legend(parent: tk.Misc) -> list[tk.Label]:
    """Inline color legend: green start, path, amber current, red end."""
    from stablewalk.ui.theme import (
        DOF_TRAJ_DOT_COLOR,
        DOF_TRAJ_END_COLOR,
        DOF_TRAJ_PATH_COLOR,
        DOF_TRAJ_START_COLOR,
        FONT_CHART_LEGEND,
        GLYPH_LEGEND_DOT,
    )

    try:
        bg = parent.cget("bg")
    except tk.TclError:
        bg = ELEVATED

    items = (
        (DOF_TRAJ_START_COLOR, "Start"),
        (DOF_TRAJ_PATH_COLOR, "Path (time)"),
        (DOF_TRAJ_DOT_COLOR, "Current"),
        (DOF_TRAJ_END_COLOR, "End"),
    )
    labels: list[tk.Label] = []
    for index, (color, name) in enumerate(items):
        if index:
            tk.Label(parent, text="·", bg=bg, fg=MUTED, font=FONT_UI_XS).pack(
                side=tk.LEFT, padx=(PAD_SM, PAD_SM)
            )
        chip = tk.Frame(parent, bg=bg)
        chip.pack(side=tk.LEFT)
        tk.Label(chip, text=GLYPH_LEGEND_DOT, bg=bg, fg=color, font=FONT_UI_SM).pack(
            side=tk.LEFT
        )
        lbl = tk.Label(chip, text=name, bg=bg, fg=TEXT_SECONDARY, font=FONT_CHART_LEGEND)
        lbl.pack(side=tk.LEFT, padx=(PAD_XS // 2, 0))
        labels.append(lbl)
    return labels


def set_overview_path_stats_expanded(gui, expanded: bool) -> None:
    """Show or hide advanced extents / caption / legend. Compact stats stay visible."""
    from stablewalk.ui.theme import GLYPH_COLLAPSE, GLYPH_EXPAND

    body = getattr(gui, "_overview_path_stats_body", None)
    glyph = getattr(gui, "_overview_path_stats_glyph", None)
    gui._overview_path_stats_expanded = bool(expanded)
    if body is None:
        return
    try:
        if expanded:
            body.grid(row=2, column=0, sticky="ew", pady=(PAD_XS, 0))
            if glyph is not None:
                glyph.configure(text=GLYPH_COLLAPSE)
        else:
            body.grid_remove()
            if glyph is not None:
                glyph.configure(text=GLYPH_EXPAND)
    except tk.TclError:
        return
    try:
        parent = getattr(gui, "overview_traj_panel", None)
        if parent is not None:
            parent.update_idletasks()
    except tk.TclError:
        pass
    if not bool(getattr(gui, "playing", False)):
        schedule = getattr(gui, "_schedule_overview_media_refit", None)
        if callable(schedule):
            try:
                schedule()
            except Exception:
                pass


def _build_overview_path_metrics_strip(gui, parent: tk.Misc) -> tk.Frame:
    """Always-visible Path length, Displacement, and Position (XYZ)."""
    card = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)
    card.columnconfigure(1, weight=1)
    rows = (
        (
            "path_length",
            "Path length",
            "Gap-aware arc length of the selected joint: sum of 3D segments "
            "between valid consecutive samples (dt ≤ 1.5/fps). Long tracking "
            "gaps are not connected. Stature-scaled estimate unless calibrated. "
            "Root-relative length is not overground or treadmill travel.",
        ),
        (
            "displacement",
            "Displacement",
            "Endpoint displacement: Euclidean first-valid sample to the current "
            "sample in this coordinate frame. N/A until the playhead leaves the "
            "first sample, and N/A for a pelvis-centered origin. Not world "
            "travel and not treadmill belt distance.",
        ),
        (
            "xyz",
            "Position",
            "Current X (lateral), Y (up), Z (forward) of the selected joint, in cm estimates.",
        ),
    )
    values: dict[str, tk.Label] = {}
    for i, (key, title, tip) in enumerate(rows):
        cap = tk.Label(
            card,
            text=title,
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        )
        cap.grid(row=i, column=0, sticky="w", padx=(PAD_SM, PAD_SM), pady=(2, 2))
        val = tk.Label(
            card,
            text="—",
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_MONO_SM,
            anchor="e",
            justify=tk.RIGHT,
        )
        val.grid(row=i, column=1, sticky="ew", padx=(0, PAD_SM), pady=(2, 2))
        create_tooltip(cap, tip, wraplength=320)
        create_tooltip(val, tip, wraplength=320)
        values[key] = val
    gui.overview_path_metric_values = values
    gui.lbl_overview_path_length = values["path_length"]
    gui.lbl_overview_path_displacement = values["displacement"]
    gui.lbl_overview_path_xyz = values["xyz"]
    gui.lbl_overview_path_metrics = values["path_length"]
    return card


def _build_overview_path_stats_panel(gui, parent: tk.Misc) -> None:
    """Compact path metrics plus collapsed advanced extents / origin / legend."""
    from stablewalk.ui.overview_path_display import coordinate_frame_caption
    from stablewalk.ui.theme import GLYPH_EXPAND
    from stablewalk.ui.tk.metric_details_dialog import attach_metric_details_control
    from stablewalk.ui.tk.tab_grid import grid_ew, stretch_columns

    shell = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    shell.grid(row=2, column=0, sticky="ew", pady=(PAD_XS, 0))
    shell.columnconfigure(0, weight=1)
    gui.overview_traj_footer = shell
    gui._overview_path_stats_expanded = False

    compact = _build_overview_path_metrics_strip(gui, shell)
    compact.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
    gui._overview_path_metrics_strip = compact

    toggle = tk.Frame(shell, bg=PANEL, highlightthickness=0)
    toggle.grid(row=1, column=0, sticky="ew")
    toggle.columnconfigure(1, weight=1)
    glyph = tk.Label(
        toggle,
        text=GLYPH_EXPAND,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        width=2,
        anchor="w",
    )
    glyph.grid(row=0, column=0, sticky="w")
    title = tk.Label(
        toggle,
        text="Advanced",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_XS,
        anchor="w",
    )
    title.grid(row=0, column=1, sticky="ew")

    body = tk.Frame(shell, bg=PANEL, highlightthickness=0)
    body.columnconfigure(0, weight=1)
    gui._overview_path_stats_body = body
    gui._overview_path_stats_toggle = toggle
    gui._overview_path_stats_glyph = glyph

    stretch_columns(body)
    gui.overview_path_extent_values = _build_overview_path_extents(body)
    gui.lbl_overview_coord_caption = tk.Label(
        body,
        text=coordinate_frame_caption(),
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
    )
    grid_ew(gui.lbl_overview_coord_caption, row=1, pady=(0, PAD_XS))
    chrome_row = tk.Frame(body, bg=PANEL, highlightthickness=0)
    grid_ew(chrome_row, row=2, pady=(0, 0))
    legend_row = tk.Frame(chrome_row, bg=PANEL, highlightthickness=0)
    legend_row.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor="w")
    gui.overview_traj_legend_labels = _build_compact_graph_legend(legend_row)
    gui._overview_traj_legend_row = legend_row
    attach_metric_details_control(
        chrome_row, "trajectory_path", bg=PANEL, side=tk.RIGHT
    )

    # Legacy single-line labels (kept for callers; not mapped).
    gui.lbl_overview_traj_info = tk.Label(body, text="", bg=PANEL)
    gui.lbl_overview_units_legend = tk.Label(body, text="", bg=PANEL)
    gui.lbl_overview_units_legend.pack_forget()
    gui.lbl_overview_traj_legend = tk.Label(body, text="", bg=PANEL)
    gui.lbl_overview_traj_metrics = tk.Label(body, text="", bg=PANEL)
    gui.lbl_overview_traj_detail = tk.Label(body, text="", bg=PANEL)
    gui.lbl_overview_traj_video = tk.Label(body, text="", bg=PANEL)
    gui.lbl_overview_category_note = tk.Label(body, text="", bg=PANEL)

    def _toggle(_event: object | None = None) -> None:
        set_overview_path_stats_expanded(
            gui, not bool(getattr(gui, "_overview_path_stats_expanded", False))
        )

    for widget in (toggle, glyph, title):
        widget.bind("<Button-1>", _toggle, add="+")
        try:
            widget.configure(cursor="hand2")
        except tk.TclError:
            pass
    create_tooltip(
        toggle,
        "Bounding box, axis ranges, origin, and path legend. "
        "Collapsed by default so Path length, Displacement, and Position stay readable.",
        wraplength=360,
    )


def _build_overview_joint_information_panel(gui, parent: tk.Misc) -> None:
    """Collapsible live Joint Information — Details button only until expanded."""
    from stablewalk.ui.joint_information import JOINT_INFO_FIELD_ORDER, JOINT_INFO_TITLES

    panel = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    panel.grid(row=3, column=0, sticky="ew", pady=(PAD_XS, 0))
    panel.columnconfigure(0, weight=1)
    gui.overview_joint_info_panel = panel

    gui._overview_joint_info_expanded = False
    toggle = ttk.Button(
        panel,
        text="Details",
        style="Compact.TButton",
        width=8,
        takefocus=False,
    )
    toggle.pack(anchor="e")
    create_tooltip(
        toggle,
        "Joint details: angle, ROM, XYZ, ω, α, confidence, contact, phase",
        wraplength=320,
    )

    body = tk.Frame(panel, bg=PANEL, highlightthickness=0)
    gui._overview_joint_info_body = body

    grid = tk.Frame(body, bg=PANEL, highlightthickness=0)
    grid.pack(fill=tk.X, expand=False)
    gui.overview_joint_info_grid = grid

    columns = 3
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
        try:
            value_lbl.configure(wraplength=130)
        except tk.TclError:
            pass
        value_labels[field_id] = value_lbl
    gui.overview_joint_info_values = value_labels

    gui.lbl_overview_joint_info_hint = tk.Label(
        body,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
    )
    gui.lbl_overview_joint_info_hint.pack_forget()

    def _toggle_details() -> None:
        expanded = not bool(getattr(gui, "_overview_joint_info_expanded", False))
        gui._overview_joint_info_expanded = expanded
        if expanded:
            body.pack(fill=tk.X, expand=False, pady=(PAD_XS, 0))
            toggle.configure(text="Hide")
        else:
            body.pack_forget()
            toggle.configure(text="Details")
        # The canvas owns all remaining row height; refit after the row changes.
        try:
            parent.update_idletasks()
            fit = getattr(gui, "_render_overview_traj_canvas", None)
            if callable(fit):
                fit(force=True)
        except (tk.TclError, TypeError):
            pass
        refresh = getattr(gui, "_refresh_overview_joint_information", None)
        if callable(refresh):
            try:
                item_id = None
                active = getattr(gui, "_active_dof_item_id", None)
                if callable(active):
                    item_id = active()
                refresh(item_id)
            except Exception:
                pass

    toggle.configure(command=_toggle_details)
    gui.btn_overview_joint_info_toggle = toggle
    body.pack_forget()


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
    # Keep off the default Overview surface; Graphs opens from the View toggle.
    host.grid(row=5, column=0, sticky="nsew", pady=(PAD_XS, 0))
    host.grid_remove()
    host.columnconfigure(0, weight=1)
    host.rowconfigure(1, weight=1)
    gui.overview_joint_motion_panel = host
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

    # Figure created on first expand — strip starts collapsed.
    gui.fig_joint_motion = None
    gui.canvas_joint_motion = None
    gui._joint_motion_chart_ready = False

    def _ensure_joint_motion_chart() -> bool:
        if getattr(gui, "_joint_motion_chart_ready", False) and gui.fig_joint_motion is not None:
            return True
        gui.fig_joint_motion = Figure(figsize=(10.0, 4.0), dpi=120, facecolor=PANEL)
        gui.fig_joint_motion.subplots(2, 3, sharex=True)
        gui.canvas_joint_motion = FigureCanvasTkAgg(gui.fig_joint_motion, master=body)
        canvas_widget = gui.canvas_joint_motion.get_tk_widget()
        canvas_widget.configure(bg=PANEL, highlightthickness=0)
        canvas_widget.grid(row=1, column=0, sticky="nsew")
        attach_chart_interactions(gui.fig_joint_motion, gui.canvas_joint_motion)
        gui._joint_motion_chart_ready = True
        return True

    gui._ensure_joint_motion_chart = _ensure_joint_motion_chart

    tools = build_chart_tools_bar(
        gui,
        tools_host,
        fig_attr="fig_joint_motion",
        canvas_attr="canvas_joint_motion",
        export_name="joint_motion_analysis",
    )
    from stablewalk.ui.tk.tab_grid import grid_ew, stretch_columns

    stretch_columns(tools_host)
    grid_ew(tools)

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
        if expanded and not bool(getattr(gui, "_overview_analysis_switching", False)):
            apply_spec = getattr(gui, "_apply_overview_signal_spectral_expanded", None)
            if callable(apply_spec) and bool(
                getattr(gui, "_overview_signal_spectral_expanded", False)
            ):
                gui._overview_analysis_switching = True
                try:
                    apply_spec(False)
                finally:
                    gui._overview_analysis_switching = False
        if expanded:
            _ensure_joint_motion_chart()
            host.grid(row=5, column=0, sticky="nsew", pady=(PAD_XS, 0))
            header.grid(row=0, column=0, sticky="ew")
            body.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
        else:
            body.grid_remove()
            header.grid_remove()
            host.grid_remove()
        apply_overview_joint_motion_row_weight(parent, expanded=expanded, row=5)
        btn = getattr(gui, "_btn_overview_joint_motion_toggle", None)
        if btn is not None:
            btn.configure(
                text=disclosure_label("Joint Graphs", expanded=expanded)
            )
        graphs_btn = getattr(gui, "_overview_graphs_btn", None)
        if graphs_btn is not None:
            try:
                graphs_btn.configure(text="Hide graphs" if expanded else "Graphs")
            except tk.TclError:
                pass
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
        text=disclosure_label("Joint Graphs", expanded=False),
        style="Compact.TButton",
        width=14,
        command=_toggle,
        takefocus=False,
    )
    toggle.grid(row=0, column=1, sticky="e", padx=(PAD_XS, 0))
    gui._btn_overview_joint_motion_toggle = toggle
    gui._toggle_overview_joint_motion = _toggle
    gui._apply_overview_joint_motion_expanded = _apply_expanded

    # Start fully collapsed — Graphs opens from View Mode in More details.
    body.grid_remove()
    header.grid_remove()
    host.grid_remove()
    apply_overview_joint_motion_row_weight(parent, expanded=False, row=5)


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
        (DOF_TRAJ_PATH_COLOR, "Path (time)"),
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
        if getattr(sidebar, "_wrap_syncing", False):
            return
        width = sidebar.winfo_width()
        if width < 48:
            return
        line_wrap = max(96, width - 8)
        from stablewalk.ui.theme import set_wraplength_safe

        setattr(sidebar, "_wrap_syncing", True)
        try:
            for lbl in getattr(gui, "dof_analysis_legend_labels", []):
                set_wraplength_safe(lbl, line_wrap)
        finally:
            setattr(sidebar, "_wrap_syncing", False)

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
    cell = create_elevated_card(parent)
    cell.grid(row=row, column=column, sticky="nsew", padx=PAD_XS // 2, pady=PAD_XS // 2)
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
        stretch = col in {
            "joint",
            "contact_state",
            "gait_phase",
            "dof",
            "contact_status",
        }
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
    tree.tag_configure("low_confidence", background=TREE_LOW_CONFIDENCE_BG, foreground=TEXT)
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
    inside the Video pane. Portrait clips may show a display-only Fit/Fill zoom.
    """
    gui._top_video_weight = _TOP_VIDEO_WEIGHT
    if aspect is not None:
        gui._recorded_video_aspect = float(aspect)
    sync_video_zoom_chrome(gui, aspect)


def sync_video_zoom_chrome(gui, aspect: float | None = None) -> None:
    """Show Fit/Fill only for portrait clips. Default remains Fit (no crop)."""
    from stablewalk.ui.video_display_fit import is_portrait_aspect

    combo = getattr(gui, "_video_zoom_combo", None)
    if combo is None:
        return
    if aspect is None:
        aspect = getattr(gui, "_recorded_video_aspect", None)
        if aspect is None:
            aspect = getattr(gui, "_video_aspect", None)
    try:
        packed = bool(combo.winfo_manager())
    except tk.TclError:
        packed = False
    if is_portrait_aspect(aspect):
        if not packed:
            try:
                combo.pack(side=tk.LEFT, padx=(0, PAD_SM))
            except tk.TclError:
                pass
        return
    var = getattr(gui, "_video_display_zoom_var", None)
    if var is not None:
        try:
            if str(var.get() or "Fit") != "Fit":
                var.set("Fit")
        except tk.TclError:
            pass
    if packed:
        try:
            combo.pack_forget()
        except tk.TclError:
            pass


def build_dashboard_layout(gui) -> None:
    """Build the main dashboard grid and attach widgets to ``gui``."""
    from stablewalk.ui.tk.dashboard_responsive import (
        finalize_responsive_dashboard,
        install_responsive_shell,
    )
    from stablewalk.ui.tk.dashboard_sections import (
        SECTION_KINEMATIC_TITLE,
        SECTION_VISUAL_TITLE,
        SEC1_METRICS_ROW_MINSIZE,
        SEC1_PANEL_MINWIDTH,
        SEC1_SKELETON_WEIGHT,
        SEC1_TRAJ_PATH_WEIGHT,
        SEC1_TRAJ_PANEL_MINWIDTH,
        SEC1_TRAJ_VIZ_ROW_MINSIZE,
        SEC1_VIDEO_WEIGHT,
        SEC1_VIZ_ROW_WEIGHT,
        bind_panel_word_wrap,
        build_gait_metrics_section,
        build_real_to_sim_section,
        build_gait_summary_cards,
        build_overview_clinical_summary,
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

    # ── Tab 1: Clinical Gait Overview — main one-screen dashboard ─────────
    # Row 0: slim title · 1: Video 40% | Skeleton 25% | Path panel 35% ·
    # 2: KPI row · Advanced / Joint Graphs stay built but unmapped.
    from stablewalk.ui.tk.dashboard_overview_view_mode import (
        build_overview_view_mode_selector,
    )
    from stablewalk.ui.tk.dashboard_page_frame import install_page_header
    from stablewalk.ui.tk.dashboard_sections import (
        OVERVIEW_HOST_HEADER_ROW,
        OVERVIEW_HOST_VIZ_ROW,
        apply_overview_host_row_weights,
        apply_overview_joint_motion_row_weight,
        apply_overview_viz_grid_weights,
    )
    from stablewalk.ui.tk.overview_responsive import install_overview_scroll_host

    overview_host = install_overview_scroll_host(gui, tab_overview)
    apply_overview_host_row_weights(overview_host, viz_row=OVERVIEW_HOST_VIZ_ROW)
    apply_overview_joint_motion_row_weight(overview_host, expanded=False, row=5)
    gui._overview_layout_host = overview_host

    page_header = install_page_header(
        overview_host, "overview", actions=True, row=0, pady=(0, OVERVIEW_ROW_GAP)
    )
    if page_header is not None:
        from stablewalk.ui.tk.dashboard_sections import build_overview_session_badge

        badge_host = page_header._overview_session_badge_host  # type: ignore[attr-defined]
        build_overview_session_badge(gui, badge_host)
        # Session badge is developer/session chrome — keep built, hide by default.
        try:
            badge_host.grid_remove()
        except tk.TclError:
            pass

    from stablewalk.ui.tk.dashboard_overview_panes import (
        install_overview_workspace_split,
    )

    section1, info_inner = install_overview_workspace_split(gui, overview_host)

    # KPI strip + warnings live in the secondary info pane (scrollable).
    clinical_summary = build_overview_clinical_summary(gui, info_inner)
    clinical_summary.grid(
        row=1,
        column=0,
        sticky="ew",
        pady=(OVERVIEW_ROW_GAP, PAD_XS),
    )
    advanced = getattr(gui, "_overview_advanced_section", None)
    if advanced is not None:
        # Built for extras wiring — not shown on the default surface.
        try:
            advanced.grid(in_=info_inner, row=3, column=0, sticky="ew")
            advanced.grid_remove()
        except tk.TclError:
            try:
                advanced.grid_remove()
            except tk.TclError:
                pass

    # View toggle sits in the page header so it never occupies a spacer row.
    view_mode_parent = getattr(page_header, "actions_host", None) or page_header
    view_mode_bar = build_overview_view_mode_selector(
        gui, view_mode_parent or overview_host
    )
    if view_mode_parent is None:
        # Last resort: share the header row — never steal the viz row.
        view_mode_bar.grid(row=OVERVIEW_HOST_HEADER_ROW, column=0, sticky="ne")
    else:
        view_mode_bar.pack(side=tk.RIGHT, anchor="e")

    # Default: equal Video | Skeleton | 3D Path columns filling the primary pane.
    apply_overview_viz_grid_weights(section1, content_row=0)
    gui._section_visual = section1
    gui._primary_viz_host = section1
    gui._primary_viz_content_row = 0
    gui._overview_summary_row = 1
    gui._viz_tabs_active = False
    gui._overview_summary_expanded = False

    from stablewalk.ui.tk.analysis_split import (
        create_h_split,
        overview_pane_minsizes,
        split_add,
    )
    from stablewalk.ui.tk.dashboard_overview_panes import (
        bind_overview_viewer_split_sashes,
    )
    from stablewalk.ui.tk.layout_scale import dpi_scale_factor

    # Holder frames live in the splitter from the start so Video / Skeleton /
    # Path are never reparented after their clip canvases are created.
    viz_split = create_h_split(section1)
    viz_split.grid(row=0, column=0, columnspan=3, sticky="nsew")
    gui._overview_viz_split = viz_split
    pane_video = tk.Frame(viz_split, bg=PANEL, highlightthickness=0)
    pane_skel = tk.Frame(viz_split, bg=PANEL, highlightthickness=0)
    pane_path = tk.Frame(viz_split, bg=PANEL, highlightthickness=0)
    for pane in (pane_video, pane_skel, pane_path):
        pane.columnconfigure(0, weight=1)
        pane.rowconfigure(0, weight=1)
    gui._overview_pane_video = pane_video
    gui._overview_pane_skeleton = pane_skel
    gui._overview_pane_path = pane_path
    min_v, min_s, min_p = overview_pane_minsizes(dpi_scale_factor(getattr(gui, "root", None)))
    split_add(viz_split, pane_video, minsize=min_v)
    split_add(viz_split, pane_skel, minsize=min_s)
    split_add(viz_split, pane_path, minsize=min_p)
    gui._overview_split_layout = "three_col"
    bind_overview_viewer_split_sashes(gui, viz_split)

    _build_overview_joint_motion_analysis_panel(gui, overview_host)
    from stablewalk.ui.tk.overview_signal_spectral import (
        build_overview_signal_spectral_panel,
    )

    build_overview_signal_spectral_panel(gui, overview_host)

    from stablewalk.ui.tk.dashboard_sections import overview_panel_padx

    overview_path_title = _viz_labelframe_title(pane_path, "3D Path")
    overview_traj_panel = ttk.LabelFrame(
        pane_path,
        labelwidget=overview_path_title,
        style="Card.TLabelframe",
        padding=_SKELETON_PANEL_PAD,
    )
    # Always mapped — selected-joint trajectory must never disappear.
    overview_traj_panel.grid(
        row=0,
        column=0,
        sticky="nsew",
        padx=0,
    )
    overview_traj_panel.columnconfigure(0, weight=1)
    # Path viewport + scrollable inspector own the panel height.
    overview_traj_panel.rowconfigure(
        0, weight=0, minsize=OVERVIEW_VIZ_TOOLBAR_MIN_H
    )  # camera toolbar
    overview_traj_panel.rowconfigure(
        1, weight=1, minsize=_OVERVIEW_TRAJ_CANVAS_MINSIZE
    )  # path viewport + inspector
    overview_traj_panel.rowconfigure(2, weight=0)  # extents + legend
    overview_traj_panel.rowconfigure(3, weight=0)  # joint information (hidden)
    gui.overview_traj_panel = overview_traj_panel
    gui.lbl_overview_traj_title = getattr(
        overview_path_title, "title_label", overview_path_title
    )

    from stablewalk.ui.tk.trajectory_camera_toolbar import build_trajectory_camera_toolbar

    path_toolbar = tk.Frame(overview_traj_panel, bg=ELEVATED, highlightthickness=0)
    path_toolbar.grid(
        row=0,
        column=0,
        sticky="ew",
        padx=OVERVIEW_VIZ_TOOLBAR_PADX,
        pady=OVERVIEW_VIZ_TOOLBAR_PADY,
    )
    path_toolbar.columnconfigure(0, weight=1)
    path_toolbar.columnconfigure(1, weight=0)
    path_toolbar.rowconfigure(0, weight=0)
    path_toolbar.rowconfigure(1, weight=0)
    gui._overview_path_toolbar = path_toolbar

    # Compact camera cluster stays in this panel. Rotate buttons are omitted
    # because drag-orbit is always available and those controls overflow Path.
    gui.overview_traj_camera_bar = build_trajectory_camera_toolbar(
        gui,
        path_toolbar,
        ax_attr="ax_dof_traj_overview",
        canvas_attr="canvas_dof_traj_overview",
        compact=True,
        show_rotate=False,
    )
    gui.overview_traj_camera_bar.grid(row=0, column=0, sticky="w")

    gui._overview_path_toolbar_extras = None

    def _reflow_path_toolbar(event: tk.Event) -> None:
        width = int(getattr(event, "width", 0) or 0)
        if width < 80:
            return
        try:
            caption = getattr(gui, "lbl_overview_coord_caption", None)
            if caption is not None:
                caption.configure(wraplength=max(160, width - 16))
            overview_traj_panel.rowconfigure(
                0,
                weight=0,
                minsize=OVERVIEW_VIZ_TOOLBAR_MIN_H,
            )
        except tk.TclError:
            pass

    path_toolbar.bind("<Configure>", _reflow_path_toolbar, add="+")

    # Cube fills the pane. Inspector starts collapsed under the plot (toggle).
    path_body = tk.Frame(overview_traj_panel, bg=PANEL, highlightthickness=0)
    path_body.grid(row=1, column=0, sticky="nsew")
    path_body.columnconfigure(0, weight=1, minsize=_OVERVIEW_PATH_VIEW_MINWIDTH)
    path_body.columnconfigure(1, weight=0, minsize=0)
    path_body.rowconfigure(0, weight=1)
    gui.overview_traj_path_body = path_body

    gui.overview_traj_canvas_host = tk.Frame(
        path_body, bg=PANEL, highlightthickness=0
    )
    gui.overview_traj_canvas_host.grid(
        row=0, column=0, columnspan=2, sticky="nsew", padx=0
    )
    gui.overview_traj_canvas_host.columnconfigure(0, weight=1)
    gui.overview_traj_canvas_host.rowconfigure(0, weight=1)

    # Inspector is a sibling row (never overlay). Hidden until the user expands.
    readout = tk.Frame(path_body, bg=PANEL, highlightthickness=0)
    readout.columnconfigure(0, weight=1)
    readout.rowconfigure(0, weight=1)
    gui.overview_traj_readout = readout
    inspector_inner, inspector_canvas = _install_coord_inspector_scroll(readout)
    gui._coord_inspector_scroll_canvas = inspector_canvas
    gui.overview_traj_readout_values = _build_overview_traj_readout(
        inspector_inner, gui
    )
    gui.overview_coordinate_inspector_values = gui.overview_traj_readout_values
    gui._coord_inspector_family_colors = getattr(
        inspector_inner, "_coord_inspector_family_colors", None
    )
    gui._overview_inspector_expanded = False
    gui._overview_path_inspector_layout = "collapsed"

    # Path statistics live outside the matplotlib figure in a collapsible
    # strip so axis labels and the cube keep the leftover panel height.
    _build_overview_path_stats_panel(gui, overview_traj_panel)
    gui.lbl_overview_traj_motion = None
    gui.overview_traj_mount = gui.overview_traj_canvas_host
    gui._overview_traj_dock_visible = True

    _build_overview_joint_information_panel(gui, overview_traj_panel)
    joint_info = getattr(gui, "overview_joint_info_panel", None)
    if joint_info is not None:
        try:
            joint_info.grid_remove()
        except tk.TclError:
            pass

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
        pane_video,
        labelwidget=_viz_labelframe_title(pane_video, "Video"),
        style="Card.TLabelframe",
        padding=_SKELETON_PANEL_PAD,
    )
    video_frame.grid(
        row=0,
        column=0,
        sticky="nsew",
        padx=0,
    )
    gui.video_frame = video_frame
    video_frame.columnconfigure(0, weight=1)
    video_frame.rowconfigure(0, weight=0)  # demo title
    video_frame.rowconfigure(1, weight=0)  # demo meta
    video_frame.rowconfigure(2, weight=0, minsize=OVERVIEW_VIZ_TOOLBAR_MIN_H)  # video toolbar
    video_frame.rowconfigure(3, weight=1)  # video viewport
    video_frame.rowconfigure(4, weight=0)  # clinical video info cards
    video_frame.rowconfigure(5, weight=0)  # overlay controls (hidden)

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

    video_toolbar = tk.Frame(video_frame, bg=ELEVATED, highlightthickness=0)
    video_toolbar.grid(
        row=2,
        column=0,
        sticky="ew",
        padx=OVERVIEW_VIZ_TOOLBAR_PADX,
        pady=OVERVIEW_VIZ_TOOLBAR_PADY,
    )
    video_toolbar.columnconfigure(0, weight=1)
    gui._overview_video_toolbar = video_toolbar
    tk.Label(
        video_toolbar,
        text="Video",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(side=tk.LEFT, padx=(PAD_XS, PAD_SM))
    gui._video_display_zoom_var = tk.StringVar(value="Fit")
    zoom_combo = ttk.Combobox(
        video_toolbar,
        textvariable=gui._video_display_zoom_var,
        values=("Fit", "Fill"),
        state="readonly",
        width=5,
    )
    gui._video_zoom_combo = zoom_combo

    def _on_video_display_zoom(_event: object = None) -> None:
        refit = getattr(gui, "_on_video_label_resize", None)
        if callable(refit):
            try:
                refit()
            except Exception:
                pass

    zoom_combo.bind("<<ComboboxSelected>>", _on_video_display_zoom)
    tk.Label(
        video_toolbar,
        text="Click the walker to analyze that person",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(side=tk.LEFT, padx=(PAD_SM, PAD_XS))
    create_tooltip(
        zoom_combo,
        "Display-only zoom for portrait clips. Fit shows the whole person. "
        "Fill uses the panel and may crop the shown image. "
        "Analysis frames are never cropped.",
        wraplength=320,
    )
    from stablewalk.ui.tk.dashboard_panel_maximize import install_maximize_button

    install_maximize_button(
        gui, video_toolbar, "video", pack_side=tk.RIGHT, padx=(PAD_XS, 0)
    )

    build_overlay_mode_controls(
        gui,
        video_frame,
        row=5,
        on_change=_on_overlay_controls_changed,
    )
    # Landmark overlay chrome stays available but off the main dashboard surface.
    from stablewalk.ui.tk.overlay_mode_controls import set_overlay_mode_controls_visible

    set_overlay_mode_controls_visible(gui, False)

    gui._video_clip_canvas, gui.video_display_host, gui._video_clip_window_id = (
        install_clipped_viewport(
            video_frame,
            bg=PANEL,
            row=3,
            column=0,
            sticky="nsew",
            min_height=_OVERVIEW_MEDIA_VIEWPORT_MIN_H,
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
    from stablewalk.ui.tk.clip_viewport import ensure_widget_fills_host

    ensure_widget_fills_host(gui.video_label)
    ensure_widget_fills_host(gui._video_clip_canvas)

    gui._demo_overlay = tk.Label(gui.video_display_host, text="")
    configure_demo_overlay(gui._demo_overlay)
    gui._demo_overlay.grid(row=0, column=0, sticky="n", pady=(4, 0))
    gui._demo_overlay.grid_remove()

    # Lightweight current-frame clearance overlay. It is a Tk layer, so updates
    # reuse the dashboard values without recomposing video or rerunning analysis.
    clearance_overlay = tk.Frame(
        gui.video_display_host,
        bg=ELEVATED,
        highlightthickness=0,
    )
    gui.lbl_overview_clearance_overlay_left = tk.Label(
        clearance_overlay,
        text="L Foot: —",
        bg=ELEVATED,
        fg=SIDE_LEFT,
        font=FONT_UI_XS,
        anchor="w",
        padx=5,
        pady=1,
    )
    gui.lbl_overview_clearance_overlay_left.pack(fill=tk.X)
    gui.lbl_overview_clearance_overlay_right = tk.Label(
        clearance_overlay,
        text="R Foot: —",
        bg=ELEVATED,
        fg=SIDE_RIGHT,
        font=FONT_UI_XS,
        anchor="w",
        padx=5,
        pady=1,
    )
    gui.lbl_overview_clearance_overlay_right.pack(fill=tk.X)
    # Metrics live under the viewer row — never overlay the video frame.
    try:
        clearance_overlay.place_forget()
    except tk.TclError:
        pass
    gui._overview_foot_clearance_overlay = clearance_overlay

    # Compact clinical HUD — built for data binding, hidden on the clean Overview.
    from stablewalk.ui.tk.dashboard_sections import build_overview_playback_hud

    build_overview_playback_hud(gui, video_frame)
    gui._overview_playback_hud.grid(row=4, column=0, sticky="ew", pady=(PAD_XS, 0))
    gui._overview_playback_hud.grid_remove()
    gui._overview_hud_shown = False
    gui._overview_hud_place_kwargs = {}

    skel_frame = ttk.LabelFrame(
        pane_skel,
        labelwidget=_viz_labelframe_title(pane_skel, "Skeleton"),
        style="Card.TLabelframe",
        padding=_SKELETON_PANEL_PAD,
    )
    skel_frame.grid(
        row=0,
        column=0,
        sticky="nsew",
        padx=0,
    )
    skel_frame.columnconfigure(0, weight=1)  # skeleton canvas (owns remaining width)
    skel_frame.columnconfigure(1, weight=0)
    skel_frame.rowconfigure(0, weight=0, minsize=OVERVIEW_VIZ_TOOLBAR_MIN_H)  # control bar
    skel_frame.rowconfigure(1, weight=1)  # visualization (almost full panel)
    skel_frame.rowconfigure(2, weight=0)  # foot clearance detail
    gui.skel_frame = skel_frame

    # ── Minimal chrome: Joint · Advanced ▾ ─────────────────────────────────
    # Play lives on the global transport. Advanced is a floating dropdown.
    from stablewalk.ui.tk.skeleton_overlay_toolbar import build_overlay_control_bar

    overlay_bar, advanced_slot = build_overlay_control_bar(
        gui, skel_frame, primary_only=True
    )
    overlay_bar.grid(
        row=0,
        column=0,
        columnspan=2,
        sticky="ew",
        padx=OVERVIEW_VIZ_TOOLBAR_PADX,
        pady=OVERVIEW_VIZ_TOOLBAR_PADY,
    )
    gui._overview_overlay_bar = overlay_bar

    from stablewalk.models.gait_wireframe import (
        GAIT_WIRE_JOINT_COUNT,
        gait_wire_included_joint_lines,
    )

    gui._gait_wire_joints_expanded = False
    wire_host = tk.Frame(overlay_bar, bg=ELEVATED, highlightthickness=0)
    wire_host.pack(side=tk.TOP, fill=tk.X, padx=PAD_XS, pady=(0, PAD_XS))
    wire_header = tk.Frame(wire_host, bg=ELEVATED, highlightthickness=0)
    wire_header.pack(side=tk.TOP, fill=tk.X)
    gui.lbl_gait_wire_joint_count = tk.Label(
        wire_header,
        text=f"{GAIT_WIRE_JOINT_COUNT} joints",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_gait_wire_joint_count.pack(side=tk.LEFT)
    create_tooltip(
        gui.lbl_gait_wire_joint_count,
        "Canonical 16-joint gait wire model (stable joint IDs and names).",
        wraplength=320,
    )

    def _toggle_gait_wire_joints() -> None:
        expanded = not bool(getattr(gui, "_gait_wire_joints_expanded", False))
        gui._gait_wire_joints_expanded = expanded
        btn = getattr(gui, "btn_gait_wire_joint_list", None)
        lst = getattr(gui, "lbl_gait_wire_joint_list", None)
        if btn is not None:
            btn.configure(
                text=disclosure_label("Included joints", expanded=expanded)
            )
        if lst is None:
            return
        if expanded:
            lst.pack(side=tk.TOP, fill=tk.X, pady=(PAD_XS, 0))
        else:
            lst.pack_forget()

    gui.btn_gait_wire_joint_list = ttk.Button(
        wire_header,
        text=disclosure_label("Included joints", expanded=False),
        style="Compact.TButton",
        command=_toggle_gait_wire_joints,
        takefocus=False,
    )
    gui.btn_gait_wire_joint_list.pack(side=tk.LEFT, padx=(PAD_SM, 0))
    create_tooltip(
        gui.btn_gait_wire_joint_list,
        "Show or hide the joints included in the 16-joint gait model.",
        wraplength=320,
    )
    gui.lbl_gait_wire_joint_list = tk.Label(
        wire_host,
        text="\n".join(gait_wire_included_joint_lines()),
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="nw",
        justify=tk.LEFT,
    )

    gui.skeleton_display_mode = tk.StringVar(
        value=MODE_TO_SKELETON_LABEL[DEFAULT_SKELETON_DISPLAY_MODE]
    )
    from stablewalk.ui.tk.dashboard_panel_maximize import install_maximize_button

    tk.Label(
        advanced_slot,
        text="Type",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
    ).pack(side=tk.LEFT, padx=(0, PAD_XS))
    from stablewalk.ui.tk.layout_scale import combobox_char_width as _combo_w

    gui.cmb_skeleton_mode = ttk.Combobox(
        advanced_slot,
        textvariable=gui.skeleton_display_mode,
        values=SKELETON_MODE_LABELS,
        state="readonly",
        width=_combo_w(SKELETON_MODE_LABELS, minimum=18, maximum=28),
    )
    gui.cmb_skeleton_mode.pack(side=tk.LEFT, padx=(0, PAD_SM))
    create_tooltip(
        gui.cmb_skeleton_mode,
        "Skeleton type: keep Walking / 2D / 3D / Side / Biomechanical, or switch "
        "Full Skeleton vs 16-Joint Gait Model.",
        wraplength=280,
    )
    gui.cmb_skeleton_mode.bind("<<ComboboxSelected>>", gui._on_skeleton_display_mode)
    install_maximize_button(
        gui, advanced_slot, "skeleton", pack_side=tk.LEFT, padx=(0, 0)
    )
    gui.lbl_skeleton_status = None

    gui.skel_canvas_host = tk.Frame(skel_frame, bg=PANEL, highlightthickness=0)
    # Canvas owns nearly all panel height (row weight=1).
    gui.skel_canvas_host.grid(row=1, column=0, sticky="nsew")
    gui.skel_canvas_host.columnconfigure(0, weight=1)
    gui.skel_canvas_host.rowconfigure(0, weight=1)
    skel_frame.columnconfigure(0, weight=1)
    skel_frame.columnconfigure(1, weight=0)

    from stablewalk.ui.tk.dashboard_sections import build_skeleton_live_info_panel

    extras_slot = getattr(gui, "_overview_advanced_extras_slot", overview_host)
    live_info = build_skeleton_live_info_panel(gui, extras_slot)
    live_info.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))

    overview_metrics = build_overview_metrics_row(gui, extras_slot)
    overview_metrics.grid(row=1, column=0, sticky="ew", pady=(0, PAD_XS))

    from stablewalk.ui.scientific_labels import DISCLAIMER_OVERVIEW_VISIBLE

    gui.lbl_overview_visible_disclaimer = tk.Label(
        extras_slot,
        text=DISCLAIMER_OVERVIEW_VISIBLE,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=960,
    )
    gui.lbl_overview_visible_disclaimer.grid(
        row=2, column=0, sticky="ew", pady=(PAD_XS, 0)
    )
    create_tooltip(
        gui.lbl_overview_visible_disclaimer,
        DISCLAIMER_OVERVIEW_VISIBLE,
        wraplength=420,
    )

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
    clearance_badges.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(1, 0))
    clearance_badges.grid_remove()

    # Asymmetric Moments live under the skeleton as a collapsible strip so the
    # canvas keeps leftover panel height. Camera / inspector stay on 3D Path.
    from stablewalk.ui.theme import GLYPH_EXPAND
    from stablewalk.ui.tk.asymmetric_moments_panel import build_asymmetric_moments_section

    skel_frame.rowconfigure(3, weight=0)
    moments_shell = tk.Frame(skel_frame, bg=PANEL, highlightthickness=0)
    moments_shell.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(PAD_XS, 0))
    moments_shell.columnconfigure(0, weight=1)
    moments_toggle = tk.Frame(moments_shell, bg=PANEL, highlightthickness=0)
    moments_toggle.grid(row=0, column=0, sticky="ew")
    moments_toggle.columnconfigure(1, weight=1)
    moments_glyph = tk.Label(
        moments_toggle,
        text=GLYPH_EXPAND,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        width=2,
        anchor="w",
    )
    moments_glyph.grid(row=0, column=0, sticky="w")
    moments_title = tk.Label(
        moments_toggle,
        text="Asymmetric moments",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_XS,
        anchor="w",
    )
    moments_title.grid(row=0, column=1, sticky="ew")
    moments_body = tk.Frame(moments_shell, bg=PANEL, highlightthickness=0)
    moments_body.columnconfigure(0, weight=1)
    gui._asymmetric_moments_section = build_asymmetric_moments_section(
        gui, moments_body
    )
    gui._asymmetric_moments_section.grid(row=0, column=0, sticky="ew")
    gui._overview_moments_body = moments_body
    gui._overview_moments_expanded = False
    gui._overview_moments_glyph = moments_glyph

    def _toggle_moments(_event: object | None = None) -> None:
        from stablewalk.ui.theme import GLYPH_COLLAPSE, GLYPH_EXPAND

        expanded = not bool(getattr(gui, "_overview_moments_expanded", False))
        gui._overview_moments_expanded = expanded
        try:
            if expanded:
                moments_body.grid(row=1, column=0, sticky="ew", pady=(PAD_XS, 0))
                moments_glyph.configure(text=GLYPH_COLLAPSE)
            else:
                moments_body.grid_remove()
                moments_glyph.configure(text=GLYPH_EXPAND)
        except tk.TclError:
            return
        if not bool(getattr(gui, "playing", False)):
            schedule = getattr(gui, "_schedule_overview_media_refit", None)
            if callable(schedule):
                try:
                    schedule()
                except Exception:
                    pass

    for widget in (moments_toggle, moments_glyph, moments_title):
        widget.bind("<Button-1>", _toggle_moments, add="+")
        try:
            widget.configure(cursor="hand2")
        except tk.TclError:
            pass

    # Live Rancho gait-phase strip — synced with Video · Skeleton · Path · Timeline.
    from stablewalk.ui.tk.gait_phase_indicator import build_gait_phase_indicator

    phase_parent = getattr(gui, "_overview_info_inner", None) or section1
    phase_host = ttk.Frame(phase_parent)
    phase_host.grid(row=0, column=0, sticky="ew", pady=(0, OVERVIEW_ROW_GAP))
    phase_host.columnconfigure(0, weight=1)
    gui._overview_phase_strip = phase_host
    build_gait_phase_indicator(gui, phase_host).grid(row=0, column=0, sticky="ew")

    from stablewalk.ui.tk.dashboard_sections import build_foot_clearance_detail_panel

    # Estimated Foot Clearance sits in the secondary info pane so it cannot
    # clip Video / Skeleton / Path.
    gui._foot_clearance_detail_host = build_foot_clearance_detail_panel(
        gui, phase_parent
    )
    gui._foot_clearance_detail_host.grid(
        row=2,
        column=0,
        sticky="ew",
        pady=(OVERVIEW_ROW_GAP, 0),
    )
    skel_frame.rowconfigure(2, weight=0)

    # Lazy skeleton figure: host + placeholder only at dashboard build so
    # Overview opens without allocating a large Matplotlib canvas up front.
    gui._skel_clip_canvas, gui._skel_plot_host, gui._skel_clip_window_id = (
        install_clipped_viewport(
            gui.skel_canvas_host,
            bg=PANEL,
            row=0,
            column=0,
            sticky="nsew",
            padx=(_SKELETON_CANVAS_PAD[0], _SKELETON_CANVAS_PAD[2]),
            pady=(_SKELETON_CANVAS_PAD[1], _SKELETON_CANVAS_PAD[3]),
            min_height=_OVERVIEW_MEDIA_VIEWPORT_MIN_H,
        )
    )
    gui._skel_plot_host.columnconfigure(0, weight=1)
    gui._skel_plot_host.rowconfigure(0, weight=1)
    gui.fig_3d = None
    gui.ax_3d = None
    gui.canvas_3d = None
    gui._skeleton_figure_ready = False
    gui._skel_figure_placeholder = tk.Label(
        gui._skel_plot_host,
        text="3D skeleton",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="center",
    )
    gui._skel_figure_placeholder.grid(row=0, column=0, sticky="nsew")

    def _ensure_skeleton_figure() -> bool:
        if getattr(gui, "_skeleton_figure_ready", False) and getattr(
            gui, "canvas_3d", None
        ) is not None:
            return True
        host = getattr(gui, "_skel_plot_host", None)
        if host is None:
            return False
        try:
            placeholder = getattr(gui, "_skel_figure_placeholder", None)
            if placeholder is not None:
                try:
                    placeholder.destroy()
                except tk.TclError:
                    pass
                gui._skel_figure_placeholder = None
            gui.fig_3d = Figure(figsize=(3.2, 2.8), dpi=100, facecolor=PANEL)
            gui.ax_3d = gui.fig_3d.add_subplot(111)
            setup_skeleton_axes(gui.ax_3d)
            _layout_skeleton_figure(gui.ax_3d)
            gui.canvas_3d = FigureCanvasTkAgg(gui.fig_3d, master=host)
            skel_canvas = gui.canvas_3d.get_tk_widget()
            skel_canvas.configure(bg=PANEL, highlightthickness=0)
            skel_canvas.grid(row=0, column=0, sticky="nsew")
            from stablewalk.ui.tk.clip_viewport import ensure_widget_fills_host

            ensure_widget_fills_host(skel_canvas)
            _bind_skeleton_figure_resize(gui.canvas_3d, gui.fig_3d, gui.ax_3d)
            gui.canvas_3d.mpl_connect("pick_event", gui._on_skeleton_pick)
            gui.canvas_3d.mpl_connect(
                "button_press_event", gui._on_skeleton_button_press
            )
            gui.canvas_3d.mpl_connect(
                "motion_notify_event", gui._on_skeleton_motion
            )
            from stablewalk.ui.viewers.gait_skeleton_renderer import (
                bind_skeleton_view_navigation,
            )

            bind_skeleton_view_navigation(gui.canvas_3d, gui.ax_3d)
            gui._skeleton_figure_ready = True
            gui._skeleton_artists_dirty = True
            return True
        except Exception:
            gui._skeleton_figure_ready = False
            return False

    gui._ensure_skeleton_figure = _ensure_skeleton_figure

    # ── Tab 2: Motion Analysis — synchronized multi-graph workspace ───────
    from stablewalk.ui.tk.dashboard_page_frame import install_page_header
    from stablewalk.ui.tk.dashboard_motion_workspace import (
        build_motion_analysis_workspace,
        mount_motion_path_panel,
    )

    install_page_header(tab_motion, "motion", row=0)
    tab_motion.rowconfigure(0, weight=0)
    tab_motion.rowconfigure(1, weight=1)
    tab_motion.columnconfigure(0, weight=1)

    motion_pages = ttk.Notebook(tab_motion)
    motion_pages.grid(row=1, column=0, sticky="nsew")
    gui._motion_page_notebook = motion_pages
    kin_page = ttk.Frame(motion_pages)
    gait_page = ttk.Frame(motion_pages)
    spec_page = ttk.Frame(motion_pages)
    feat_page = ttk.Frame(motion_pages)
    kin_page.columnconfigure(0, weight=1)
    kin_page.rowconfigure(1, weight=1)
    gait_page.columnconfigure(0, weight=1)
    gait_page.rowconfigure(0, weight=1)
    spec_page.columnconfigure(0, weight=1)
    spec_page.rowconfigure(0, weight=1)
    feat_page.columnconfigure(0, weight=1)
    feat_page.rowconfigure(0, weight=1)
    from stablewalk.ui.tk.gait_correlation_tabs import SECTION_TITLE

    motion_pages.add(kin_page, text="Kinematics")
    motion_pages.add(gait_page, text=SECTION_TITLE)
    motion_pages.add(spec_page, text="Spectral Analysis")
    motion_pages.add(feat_page, text="Spectral Features")

    section2 = build_motion_analysis_workspace(gui, kin_page)
    from stablewalk.ui.tk.dashboard_gait_correlation import (
        build_gait_correlation_section,
    )
    from stablewalk.ui.tk.dashboard_spectral import build_spectral_analysis_workspace
    from stablewalk.ui.tk.dashboard_spectral_features import (
        build_spectral_features_workspace,
    )

    build_gait_correlation_section(gui, gait_page)
    build_spectral_analysis_workspace(gui, spec_page)
    build_spectral_features_workspace(gui, feat_page)
    motion_joint_row = getattr(gui, "_motion_joint_row", section2)

    from stablewalk.ui.tk.dashboard_biomechanics import build_biomechanics_tab

    build_biomechanics_tab(gui, tab_biomechanics)
    gui._section_biomechanics = tab_biomechanics

    from stablewalk.ui.tk.dashboard_summary import build_results_summary_tab

    build_results_summary_tab(gui, tab_results_summary)
    gui._section_results_summary = tab_results_summary

    from stablewalk.ui.tk.dashboard_notebook import ENABLE_COMPARE_TAB

    if ENABLE_COMPARE_TAB and tab_compare is not None:
        from stablewalk.ui.tk.dashboard_compare import build_comparison_tab

        build_comparison_tab(gui, tab_compare)
        gui._section_compare = tab_compare
    else:
        # Compare tab is hidden; keep no-op hooks so session pin/refresh stays safe.
        gui._section_compare = None
        gui._tab_compare = None
        gui._comparison_mode = None
        gui._comparison_refresh_pending = False
        gui._ensure_comparison_mode_loaded = lambda: None
        gui._refresh_comparison_mode = lambda: None

    # Hidden host for legacy Knee Motion panel (feature preserved, not primary).
    # Not gridded — widgets stay built for refresh / Details / export paths.
    legacy_host = ttk.Frame(tab_motion)
    gui._motion_legacy_host = legacy_host
    bottom_row = ttk.Frame(legacy_host)
    bottom_row.grid(row=0, column=0, sticky="nsew")
    gui._motion_content_min_height = _MOTION_TOP_ROW_MIN_H + PAD_XS
    bottom_row.columnconfigure(0, weight=1, minsize=_MOTION_KNEE_PANEL_MIN_W)
    bottom_row.columnconfigure(1, weight=1, minsize=_MOTION_TRAJ_PANEL_MIN_W)
    bottom_row.rowconfigure(0, weight=1)
    gui._dashboard_bottom_row = bottom_row
    gui.dashboard = bottom_row
    # No Motion-page scroll shell — workspace fills the tab without scrolling.
    gui._motion_scroll_outer = None
    gui._motion_scroll_canvas = None
    gui._sync_motion_scroll = lambda *_a, **_k: None

    # Temporal gait metrics answer stability timing — hosted on Biomechanics.
    from stablewalk.ui.tk.dashboard_motion_metrics import build_motion_temporal_metrics_panel

    biomech_metrics_host = getattr(gui, "_biomech_temporal_host", None)
    if biomech_metrics_host is not None:
        motion_metrics_panel = build_motion_temporal_metrics_panel(gui, biomech_metrics_host)
        from stablewalk.ui.tk.tab_grid import grid_fill, stretch_columns, stretch_rows

        stretch_columns(biomech_metrics_host)
        stretch_rows(biomech_metrics_host)
        grid_fill(motion_metrics_panel)
    else:
        motion_metrics_panel = build_motion_temporal_metrics_panel(gui, bottom_row)
        motion_metrics_panel.grid_remove()
    gui._motion_temporal_metrics_panel = motion_metrics_panel

    knee_panel = ttk.LabelFrame(
        bottom_row,
        text=_card_title("Knee Motion"),
        style="Card.TLabelframe",
        padding=(PAD_SM, PAD_SM),
    )
    # Preserved for Details / export paths — not on the default workspace surface.
    knee_panel.grid(row=0, column=0, sticky="nsew", padx=(0, DASHBOARD_GUTTER))
    knee_panel.grid_remove()
    gui.knee_panel = knee_panel
    knee_panel.columnconfigure(0, weight=1)
    knee_panel.rowconfigure(0, weight=0)
    knee_panel.rowconfigure(1, weight=1)
    knee_panel.rowconfigure(2, weight=0)
    knee_panel.rowconfigure(3, weight=0)

    chart_toolbar = tk.Frame(knee_panel, bg=PANEL, highlightthickness=0)
    chart_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
    mode_lbl = tk.Label(
        chart_toolbar, text="Mode", bg=PANEL, fg=MUTED, font=FONT_UI_SM
    )
    mode_lbl.pack(side=tk.LEFT, padx=(0, PAD_XS))
    create_tooltip(
        mode_lbl,
        "Video Time plots flexion against recording time. "
        "Gait Cycle % normalizes each stride from heel strike to 100%.",
        wraplength=360,
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
    angle_src_lbl = tk.Label(
        gui._knee_source_toolbar,
        text="Angle source",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
    )
    angle_src_lbl.pack(side=tk.LEFT, padx=(PAD_MD, PAD_XS))
    create_tooltip(
        angle_src_lbl,
        "Auto selects OpenSim IK when available, otherwise pose-derived knee angles. "
        "Pose-derived and OpenSim IK force a single source.",
        wraplength=360,
    )
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
    from stablewalk.ui.design_system.components import EmptyState

    gui.lbl_knee_chart_empty = EmptyState(
        gui.knee_chart_host,
        "Analyze a session to show the joint-angle graph.",
        bg=PANEL,
    )
    gui.lbl_knee_chart_empty.grid(row=0, column=0, sticky="nsew")

    # Matplotlib knee figure is created on first Motion tab open.
    gui.fig = None
    gui.ax_chart = None
    gui.chart_canvas = None
    gui._chart_widget = None
    gui._motion_knee_chart_ready = False
    from stablewalk.ui.viewers.chart_interactions import (
        attach_chart_interactions,
        build_chart_tools_bar,
    )

    knee_tools = build_chart_tools_bar(
        gui, knee_panel, fig_attr="fig", canvas_attr="chart_canvas", export_name="knee_flexion"
    )
    knee_tools.grid(row=2, column=0, sticky="ew", pady=(PAD_XS, 0))

    knee_legend_row = tk.Frame(knee_panel, bg=PANEL, highlightthickness=0)
    knee_legend_row.grid(row=3, column=0, sticky="ew", pady=(PAD_XS, 0))
    knee_legend_row.grid_remove()

    gui.lbl_knee_no_cycle = tk.Label(
        knee_panel,
        text="",
        bg=PANEL,
        fg=ORANGE,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=520,
    )
    gui.lbl_knee_no_cycle.grid(row=4, column=0, sticky="ew", pady=(PAD_XS, 0))
    gui.lbl_knee_no_cycle.grid_remove()

    knee_summary_row = tk.Frame(knee_panel, bg=PANEL, highlightthickness=0)
    knee_summary_row.grid(row=5, column=0, sticky="ew", pady=(PAD_SM, 0))
    knee_summary_row.columnconfigure(0, weight=1)
    gui.lbl_knee_summary_compact = tk.Label(
        knee_summary_row,
        text="L ROM —  ·  R ROM —  ·  Asymmetry —  ·  Cycles —  ·  Repeatability —",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=520,
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
    gui.btn_knee_details.grid(row=0, column=1, sticky="e", padx=(PAD_SM, 0))
    create_tooltip(
        gui.btn_knee_details,
        "Technical knee-motion report (pose-derived estimates).",
        wraplength=320,
    )

    contact_parent = getattr(gui, "_biomech_contact_host", None) or bottom_row
    contact_on_biomech = contact_parent is not bottom_row
    if contact_on_biomech:
        # Biomechanics workspace owns chrome (tabs / export / interpretation).
        # Chart panel is a plain fill host so the graph stays fully visible.
        contact_gait_panel = tk.Frame(contact_parent, bg=PANEL, highlightthickness=0)
        from stablewalk.ui.tk.tab_grid import grid_fill, stretch_columns, stretch_rows

        stretch_columns(contact_parent)
        stretch_rows(contact_parent)
        grid_fill(contact_gait_panel)
        tools_parent = getattr(gui, "_biomech_contact_tools_host", None) or contact_gait_panel
    else:
        contact_gait_panel = ttk.LabelFrame(
            contact_parent,
            text=_card_title(LABEL_CONTACT_VGRF_SECTION),
            style="Card.TLabelframe",
            padding=_ANALYSIS_PANEL_PAD,
        )
        contact_gait_panel.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="nsew",
            padx=(0, 0),
            pady=(PAD_XS, 0),
        )
        bottom_row.rowconfigure(1, weight=2, minsize=CONTACT_GAIT_CHART_MIN_H)
        tools_parent = contact_gait_panel
    gui.contact_gait_panel = contact_gait_panel
    contact_gait_panel.columnconfigure(0, weight=1)
    if contact_on_biomech:
        contact_gait_panel.rowconfigure(0, weight=1)
    else:
        contact_gait_panel.rowconfigure(0, weight=0)
        contact_gait_panel.rowconfigure(1, weight=1)

    gui.fig_contact_gait = None
    gui.ax_contact_gait = None
    gui.canvas_contact_gait = None
    gui._contact_gait_chart_ready = False
    if contact_on_biomech and not getattr(gui, "_contact_chart_panel", None):
        gui._contact_chart_panel = "contact"

    def _contact_export_name() -> str:
        return str(getattr(gui, "_biomech_export_name", None) or "biomech_contact")

    gui.contact_gait_tools = build_chart_tools_bar(
        gui,
        tools_parent,
        fig_attr="fig_contact_gait",
        canvas_attr="canvas_contact_gait",
        export_name="biomech_contact",
        export_name_getter=_contact_export_name,
        ensure_chart=lambda: getattr(gui, "_ensure_contact_gait_chart", lambda: False)(),
    )
    if contact_on_biomech:
        # Parked until Contact/GRF tab is active (workspace swaps export bars).
        gui.contact_gait_tools.grid(row=0, column=0, sticky="ew")
        try:
            tools_parent.grid_remove()
        except Exception:
            pass
    else:
        gui.contact_gait_tools.grid(row=0, column=0, sticky="ew", pady=(0, 2))

    gui.contact_gait_chart_host = tk.Frame(
        contact_gait_panel,
        bg=PANEL,
        highlightthickness=0,
    )
    gui.contact_gait_chart_host.grid(
        row=0 if contact_on_biomech else 1, column=0, sticky="nsew"
    )
    gui.contact_gait_chart_host.columnconfigure(0, weight=1)
    gui.contact_gait_chart_host.rowconfigure(0, weight=1)

    def _ensure_contact_gait_chart() -> bool:
        if getattr(gui, "_contact_gait_chart_ready", False) and gui.canvas_contact_gait is not None:
            return True
        host = getattr(gui, "contact_gait_chart_host", None)
        if host is None:
            return False
        # Large single-panel figure when hosted in Biomechanics tabs.
        if contact_on_biomech:
            from stablewalk.ui.tk.dashboard_biomechanics import BIOMECH_CHART_FIGSIZE

            figsize = BIOMECH_CHART_FIGSIZE
        else:
            figsize = (10.0, 2.95)
        gui.fig_contact_gait = Figure(figsize=figsize, dpi=120, facecolor=PANEL)
        gui.ax_contact_gait = None
        gui.canvas_contact_gait = FigureCanvasTkAgg(
            gui.fig_contact_gait, master=host
        )
        cg_widget = gui.canvas_contact_gait.get_tk_widget()
        cg_widget.configure(bg=PANEL, highlightthickness=0)
        cg_widget.grid(row=0, column=0, sticky="nsew")
        _bind_figure_resize(gui.canvas_contact_gait, gui.fig_contact_gait, min_px=80)
        attach_chart_interactions(gui.fig_contact_gait, gui.canvas_contact_gait)
        gui._contact_gait_chart_ready = True
        return True

    def _fit_contact_gait_canvas() -> None:
        if not _ensure_contact_gait_chart():
            return
        fitted = _fit_figure_to_host(
            gui.canvas_contact_gait,
            gui.fig_contact_gait,
            host=gui.contact_gait_chart_host,
            min_px=80,
        )
        if fitted and getattr(gui, "_biomech_chart_family", "") == "contact":
            from stablewalk.ui.viewers.contact_gait_viewer import (
                relayout_contact_gait_figure,
            )

            relayout_contact_gait_figure(gui.fig_contact_gait)
            try:
                gui.canvas_contact_gait.draw_idle()
            except Exception:
                pass

    gui._ensure_contact_gait_chart = _ensure_contact_gait_chart
    gui._fit_contact_gait_canvas = _fit_contact_gait_canvas
    _contact_fit_job: dict[str, object] = {"id": None}

    def _on_contact_panel_configure(_event: object | None = None) -> None:
        if should_defer_layout(gui):
            return
        if not getattr(gui, "_contact_gait_chart_ready", False):
            return
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
        text=_card_title("3D Joint Path"),
        style="Card.TLabelframe",
        padding=_ANALYSIS_PANEL_PAD,
    )
    traj_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 0))
    gui.traj_panel = traj_panel
    # Mount into the Motion workspace right column (primary surface).
    try:
        mount_motion_path_panel(gui, traj_panel)
    except Exception:
        pass
    traj_panel.columnconfigure(0, weight=1)
    traj_panel.rowconfigure(0, weight=1)
    traj_panel.rowconfigure(1, weight=0)

    gui.traj_panel_header_host = tk.Frame(traj_panel, bg=PANEL, highlightthickness=0)
    gui.traj_panel_header_host.grid(row=0, column=0, sticky="ew")
    gui.traj_panel_header_host.columnconfigure(0, weight=1)
    gui.traj_panel_header_host.grid_remove()

    # One content frame under traj_panel (no separate stack wrapper).
    gui.dof_analysis_body = tk.Frame(traj_panel, bg=PANEL, highlightthickness=0)
    gui.dof_analysis_body.grid(row=0, column=0, sticky="nsew")
    gui.dof_analysis_body.columnconfigure(0, weight=1)
    gui.dof_analysis_body.rowconfigure(0, weight=1)
    gui.traj_panel_stack = gui.dof_analysis_body

    gui.dof_analysis_empty_host = tk.Frame(gui.dof_analysis_body, bg=PANEL)
    gui.dof_analysis_empty_host.grid(row=0, column=0, sticky="nsew")
    gui.dof_analysis_empty_host.columnconfigure(0, weight=1)
    gui.dof_analysis_empty_host.rowconfigure(0, weight=0)

    from stablewalk.ui.design_system.components import EmptyState

    gui.lbl_dof_analysis_empty = EmptyState(
        gui.dof_analysis_empty_host,
        EMPTY_SELECT_DOF_CHART,
        bg=PANEL,
    )
    gui.lbl_dof_analysis_empty.grid(row=0, column=0, sticky="nsew")

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
    gui.lbl_dof_coord_mode_value.configure(text="Root-relative")
    _status_label(2, "Trajectory Mode:", "lbl_dof_traj_mode_value")
    gui.lbl_dof_traj_mode_value.configure(text="Current progress")
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
    # The LabelFrame is already titled "3D Path"; hide the
    # duplicate inner heading so the 3D graph reclaims that vertical band.
    joint_header.grid_remove()

    joint_controls = tk.Frame(gui.dof_analysis_graph_section, bg=PANEL, highlightthickness=0)
    joint_controls.grid(row=2, column=0, sticky="ew", pady=(0, 0))
    joint_controls.columnconfigure(0, weight=1)

    controls_left = tk.Frame(joint_controls, bg=PANEL, highlightthickness=0)
    controls_left.grid(row=0, column=0, sticky="w")

    gui.var_dof_traj_display = tk.StringVar(value="Current progress")
    traj_disp_lbl = tk.Label(
        controls_left, text="Display", bg=PANEL, fg=MUTED, font=FONT_UI_SM
    )
    traj_disp_lbl.pack(side=tk.LEFT, padx=(0, 4))
    create_tooltip(
        traj_disp_lbl,
        "Current progress grows the path with playback. "
        "Full trajectory shows the complete session path.",
        wraplength=320,
    )
    from stablewalk.ui.tk.layout_scale import combobox_char_width as _combo_w

    _traj_display_values = ("Current progress", "Full trajectory")
    gui.cmb_dof_traj_display = ttk.Combobox(
        controls_left,
        textvariable=gui.var_dof_traj_display,
        values=_traj_display_values,
        state="readonly",
        width=_combo_w(_traj_display_values, minimum=16, maximum=24),
    )
    gui.cmb_dof_traj_display.pack(side=tk.LEFT, padx=(0, 10))
    gui.cmb_dof_traj_display.bind(
        "<<ComboboxSelected>>", lambda _e: gui._on_dof_traj_display_changed()
    )

    gui.var_dof_coord_mode = tk.StringVar(value="Root-relative")
    coord_lbl = tk.Label(controls_left, text="Coord", bg=PANEL, fg=MUTED, font=FONT_UI_SM)
    coord_lbl.pack(side=tk.LEFT, padx=(0, 4))
    create_tooltip(
        coord_lbl,
        "Root-relative centers the path on the pelvis (preferred). "
        "Global uses world coordinates (pose-derived, approximate scale).",
        wraplength=320,
    )
    _coord_values = ("Root-relative", "Global")
    gui.cmb_dof_coord_mode = ttk.Combobox(
        controls_left,
        textvariable=gui.var_dof_coord_mode,
        values=_coord_values,
        state="readonly",
        width=_combo_w(_coord_values, minimum=14, maximum=20),
    )
    gui.cmb_dof_coord_mode.pack(side=tk.LEFT, padx=(0, 10))
    gui.cmb_dof_coord_mode.bind(
        "<<ComboboxSelected>>", lambda _e: gui._on_dof_coord_mode_changed()
    )

    gui.var_dof_projection = tk.StringVar(value="3D")
    view_lbl = tk.Label(controls_left, text="View", bg=PANEL, fg=MUTED, font=FONT_UI_SM)
    view_lbl.pack(side=tk.LEFT, padx=(0, 4))
    create_tooltip(
        view_lbl,
        "3D perspective, or orthographic Frontal / Sagittal plane views.",
        wraplength=280,
    )
    _proj_values = ("3D", "Frontal Plane", "Sagittal Plane")
    gui.cmb_dof_projection = ttk.Combobox(
        controls_left,
        textvariable=gui.var_dof_projection,
        values=_proj_values,
        state="readonly",
        width=_combo_w(_proj_values, minimum=14, maximum=20),
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
        highlightthickness=0,
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

    # Canvas lives directly in the graph frame (no inner wrapper).
    gui.dof_analysis_graph_canvas_host = gui.dof_analysis_graph_frame
    gui.dof_analysis_graph_inner = gui.dof_analysis_graph_frame
    gui._traj_clip_canvas = None
    gui._traj_clip_window_id = None

    from stablewalk.ui.tk.dashboard_path_science import build_path_science_stats_panel

    # Always-visible scientific statistics under the trajectory (main visual).
    path_science = build_path_science_stats_panel(gui, gui.dof_analysis_graph_section)
    path_science.grid(row=4, column=0, sticky="ew", pady=(PAD_SM, 0))

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
    traj_legend_row.grid(row=5, column=0, sticky="ew", pady=(2, 0))
    gui.var_dof_show_body_reference = tk.BooleanVar(value=False)
    gui.chk_dof_show_body_reference = ttk.Checkbutton(
        traj_legend_row,
        text="Show body reference",
        variable=gui.var_dof_show_body_reference,
        command=gui._on_dof_body_reference_changed,
        takefocus=False,
    )
    gui.chk_dof_show_body_reference.pack(side=tk.RIGHT)
    create_tooltip(
        gui.chk_dof_show_body_reference,
        "Optional pose-derived body sketch (off by default for performance).",
        wraplength=300,
    )
    gui.dof_traj_legend_labels = _build_compact_graph_legend(traj_legend_row)

    gui.dof_analysis_interp_frame.grid(row=6, column=0, sticky="ew", pady=(2, 0))
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

    from stablewalk.ui.tk.metric_details_dialog import attach_metric_details_control

    details_host = tk.Frame(traj_legend_row, bg=PANEL, highlightthickness=0)
    details_host.pack(side=tk.LEFT, padx=(0, PAD_XS))
    attach_metric_details_control(
        details_host, "trajectory_path", bg=PANEL, side=tk.LEFT
    )

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
    gui.lbl_motion_traj_metrics.grid(row=2, column=0, sticky="ew", pady=(1, 0))
    # Path length / ROM live in the science stats grid — keep this spare.
    gui.lbl_motion_traj_metrics.grid_remove()
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

    # Motion / Overview trajectory figures are created on first need.
    gui.fig_dof_traj = None
    gui.ax_dof_traj = None
    gui.canvas_dof_traj = None
    gui.fig_dof_traj_overview = None
    gui.ax_dof_traj_overview = None
    gui.canvas_dof_traj_overview = None
    gui._motion_traj_chart_ready = False
    gui.dof_analysis_empty_host.grid_remove()

    def _ensure_motion_knee_chart() -> bool:
        if getattr(gui, "_motion_knee_chart_ready", False) and gui.chart_canvas is not None:
            return True
        host = getattr(gui, "knee_chart_host", None)
        if host is None:
            return False
        from stablewalk.ui.viewers.chart_interactions import attach_chart_interactions

        gui.fig = Figure(figsize=(6.5, 2.5), dpi=120, facecolor=PANEL)
        gui.ax_chart = gui.fig.add_subplot(111)
        gui.ax_chart.set_facecolor(PANEL)
        gui.chart_canvas = FigureCanvasTkAgg(gui.fig, master=host)
        gui._chart_widget = gui.chart_canvas.get_tk_widget()
        gui._chart_widget.configure(bg=PANEL, highlightthickness=0)
        gui._chart_widget.grid(row=0, column=0, sticky="nsew")
        empty = getattr(gui, "lbl_knee_chart_empty", None)
        if empty is not None:
            try:
                empty.grid_remove()
            except tk.TclError:
                pass
        _bind_figure_resize(gui.chart_canvas, gui.fig, min_px=80)
        attach_chart_interactions(gui.fig, gui.chart_canvas)
        gui._init_chart()
        gui._motion_knee_chart_ready = True
        return True

    def _ensure_motion_traj_chart() -> bool:
        if getattr(gui, "_motion_traj_chart_ready", False) and gui.canvas_dof_traj is not None:
            return True
        host = getattr(gui, "dof_analysis_graph_canvas_host", None)
        if host is None:
            return False
        from stablewalk.ui.tk.trajectory_camera_toolbar import bind_trajectory_camera_canvas
        from stablewalk.ui.viewers.dof_trajectory_3d import (
            relayout_single_dof_viewport,
            setup_single_dof_trajectory_axes,
        )

        gui.fig_dof_traj = Figure(figsize=(6.5, 2.5), dpi=120, facecolor=PANEL)
        gui.ax_dof_traj = gui.fig_dof_traj.add_subplot(111, projection="3d")
        gui.ax_dof_traj._stablewalk_motion_dock = True
        gui.ax_dof_traj._stablewalk_overview_cm_ticks = True
        setup_single_dof_trajectory_axes(gui.ax_dof_traj)
        relayout_single_dof_viewport(gui.ax_dof_traj)
        gui.canvas_dof_traj = FigureCanvasTkAgg(gui.fig_dof_traj, master=host)
        traj_canvas = gui.canvas_dof_traj.get_tk_widget()
        traj_canvas.configure(bg=PANEL, highlightthickness=0)
        _ensure_trajectory_canvas_gridded(gui.canvas_dof_traj)
        bind_trajectory_camera_canvas(
            gui, canvas_attr="canvas_dof_traj", ax_attr="ax_dof_traj"
        )
        _bind_trajectory_figure_resize(
            gui.canvas_dof_traj,
            gui.fig_dof_traj,
            gui.ax_dof_traj,
            graph_host=gui.dof_analysis_graph_canvas_host,
            extra_hosts=(
                gui.dof_analysis_graph_section,
                gui.traj_panel,
                getattr(gui, "overview_traj_mount", None),
                getattr(gui, "overview_traj_panel", None),
            ),
        )
        gui._motion_traj_chart_ready = True
        return True

    def _ensure_motion_tab_charts() -> bool:
        # Contact / vGRF figures live on Biomechanics — never build them here.
        ok_knee = _ensure_motion_knee_chart()
        ok_traj = _ensure_motion_traj_chart()
        ok_ws = True
        ensure_ws = getattr(gui, "_ensure_motion_workspace_chart", None)
        if callable(ensure_ws):
            try:
                ok_ws = bool(ensure_ws())
            except Exception:
                ok_ws = False
        return ok_knee and ok_traj and ok_ws

    gui._ensure_motion_knee_chart = _ensure_motion_knee_chart
    gui._ensure_motion_traj_chart = _ensure_motion_traj_chart
    gui._ensure_motion_tab_charts = _ensure_motion_tab_charts

    def _on_traj_panel_resize(_event: object = None) -> None:
        if should_defer_layout(gui):
            return
        after_id = getattr(gui, "_traj_resize_after", None)
        if after_id is not None:
            try:
                gui.root.after_cancel(after_id)
            except tk.TclError:
                pass

        def _reflow() -> None:
            gui._traj_resize_after = None
            if should_defer_layout(gui):
                return
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
            if getattr(gui, "canvas_dof_traj", None) is not None:
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
        # Skeleton host Configure is already handled by
        # ``_bind_skeleton_figure_resize`` on ``_skel_plot_host``. A second
        # ``_fit_skeleton_canvas`` here caused duplicate fits / black flashes.
        return

    def _on_knee_panel_configure(_event: object = None) -> None:
        if should_defer_layout(gui):
            return
        if getattr(gui, "chart_canvas", None) is None:
            return
        job = getattr(gui, "_knee_resize_after", None)
        if job is not None:
            try:
                gui.root.after_cancel(job)
            except (tk.TclError, ValueError):
                pass

        def _refit() -> None:
            gui._knee_resize_after = None
            if should_defer_layout(gui):
                return
            canvas = getattr(gui, "chart_canvas", None)
            fig = getattr(gui, "fig_chart", None) or getattr(gui, "fig", None)
            if canvas is not None and fig is not None:
                _fit_figure_to_host(canvas, fig)

        gui._knee_resize_after = gui.root.after(_FIGURE_RESIZE_DEBOUNCE_MS, _refit)

    traj_panel.bind("<Configure>", _on_traj_panel_resize)
    gui.dof_analysis_graph_frame.bind("<Configure>", _on_traj_panel_resize)
    overview_traj = getattr(gui, "overview_traj_panel", None)
    if overview_traj is not None:
        overview_traj.bind("<Configure>", _on_traj_panel_resize)
        # Host mount already covered by overview_traj panel Configure —
        # avoid a second identical bind on overview_traj_canvas_host.
    # Skeleton fit: ``_bind_skeleton_figure_resize`` on plot host only
    # (do not also bind skel_canvas_host — duplicate fits caused black flashes).
    if hasattr(gui, "knee_panel"):
        gui.knee_panel.bind("<Configure>", _on_knee_panel_configure)

    gui.lbl_dof_traj_path = gui.dof_analysis_graph_section

    gui._motion_tools_row = None

    from stablewalk.ui.tk.dashboard_page_frame import install_page_header

    install_page_header(tab_advanced_content, "advanced", row=0)

    # Technical workspace: Session · OpenSim · Real-to-Sim · Exports · Reports.
    from stablewalk.ui.tk.dashboard_advanced_workspace import (
        advanced_category_slot,
        build_advanced_workspace,
    )

    tab_advanced_content.columnconfigure(0, weight=1)
    # The Advanced tab scroll shell owns viewport expansion. Keep the category
    # workspace at its natural height so collapsed sections return their space.
    tab_advanced_content.rowconfigure(1, weight=0)
    advanced_workspace = build_advanced_workspace(gui, tab_advanced_content)
    advanced_workspace.grid(row=1, column=0, sticky="ew")
    gui._advanced_lazy_slot = advanced_workspace
    gui._advanced_content_built = False
    advanced_empty = tk.Label(
        tab_advanced_content,
        text=(
            "Advanced & Export is the Session / OpenSim / Real-to-Sim / export "
            "workspace. Expand a card below. This tab does not run a second "
            "gait analysis. OpenSim IK Partial means export finished without "
            "a completed Inverse Kinematics solve — not biomechanical validation."
        ),
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI,
        wraplength=820,
        justify="left",
        anchor="w",
    )
    advanced_empty.grid(row=2, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_SM, 0))
    gui._advanced_empty_state = advanced_empty

    # Analysis evidence lives on Biomechanics; keep a hidden alias host for APIs.
    advanced_info = ttk.Frame(tab_advanced_content)
    gui._section_advanced_info_advanced = advanced_info

    def _ensure_advanced_tab_content() -> bool:
        if getattr(gui, "_advanced_content_built", False):
            return True
        empty = getattr(gui, "_advanced_empty_state", None)
        try:
            return _build_advanced_tab_content()
        except Exception as exc:
            import logging

            logging.getLogger(__name__).exception("Advanced tab content failed to load")
            if empty is not None:
                try:
                    empty.configure(
                        text=(
                            "Advanced tab content failed to load. "
                            f"{type(exc).__name__}: {exc}. "
                            "Session / OpenSim / export cards above remain available."
                        )
                    )
                    empty.grid()
                except Exception:
                    pass
            return False

    def _build_advanced_tab_content() -> bool:
        from stablewalk.ui.design_system.components import CollapsibleSection
        from stablewalk.ui.tk.generated_files_panel import build_generated_files_section
        from stablewalk.ui.tk.dashboard_pipeline_status import build_pipeline_status_panel

        session_slot = advanced_category_slot(gui, "session")
        reports_slot = advanced_category_slot(gui, "reports")
        exports_slot = advanced_category_slot(gui, "exports")
        real_to_sim_slot = advanced_category_slot(gui, "real_to_sim")
        if (
            session_slot is None
            or reports_slot is None
            or exports_slot is None
            or real_to_sim_slot is None
        ):
            empty = getattr(gui, "_advanced_empty_state", None)
            if empty is not None:
                try:
                    empty.configure(
                        text=(
                            "Advanced workspace cards are installed, but lazy "
                            "panels are not ready yet. Expand Session or OpenSim. "
                            "This is not a second analysis engine."
                        )
                    )
                    empty.grid()
                except Exception:
                    pass
            return False

        generated_files = build_generated_files_section(gui, session_slot)
        generated_files.grid(row=0, column=0, sticky="ew", pady=(PAD_SM, 0))

        real_to_sim_section = build_real_to_sim_section(gui, real_to_sim_slot)
        real_to_sim_section.grid(row=0, column=0, sticky="ew", pady=(PAD_XS, 0))

        pipe_wrap = CollapsibleSection(
            real_to_sim_slot,
            "Pipeline Status",
            expanded=False,
            bg=PANEL,
        )
        pipe_wrap.frame.grid(row=1, column=0, sticky="ew", pady=(PAD_XS, 0))
        pipe_wrap.body.columnconfigure(0, weight=1)
        gui._advanced_pipeline_status_section = pipe_wrap
        advanced_pipeline = build_pipeline_status_panel(
            gui,
            pipe_wrap.body,
            host_attr="_pipeline_status_host_advanced",
            section_attr="_section_pipeline_status_advanced",
            framed=False,
        )
        advanced_pipeline.grid(row=0, column=0, sticky="ew")
        gui._section_pipeline_status_advanced = advanced_pipeline
        gui._pipeline_panel_host = advanced_pipeline

        section3 = build_gait_metrics_section(gui, exports_slot)
        section3.grid(row=0, column=0, sticky="ew", pady=(PAD_SM, 0))

        motion_tools = ttk.Frame(session_slot)
        motion_tools.columnconfigure(0, weight=1)
        motion_tools.grid(row=1, column=0, sticky="ew", pady=(PAD_XS, 0))
        gui._motion_tools_row = motion_tools

        table_title = tk_labelframe_title(motion_tools, "Detailed Joint Data")
        table_panel = ttk.LabelFrame(
            motion_tools,
            labelwidget=table_title,
            style="Card.TLabelframe",
            padding=DASHBOARD_VIZ_CARD_PAD,
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
            text_cols=frozenset({"joint", "contact_state", "gait_phase"}),
            style="Large.Treeview",
        )
        gui.dof_pos_tree.bind("<<TreeviewSelect>>", gui._on_dof_pos_tree_select)
        # Keep the existing table as the data source for "View Joint Data",
        # without reserving a large empty block in the Advanced workspace.
        motion_tools.grid_remove()

        gui._advanced_content_built = True
        empty = getattr(gui, "_advanced_empty_state", None)
        if empty is not None:
            try:
                empty.grid_remove()
            except Exception:
                pass
        try:
            update_gait = getattr(gui, "_update_gait_cycle_panel", None)
            if callable(update_gait):
                update_gait(getattr(gui, "_gait_cycle", None))
            sync_cycles = getattr(gui, "_sync_gait_cycles_labels", None)
            if callable(sync_cycles):
                sync_cycles()
        except Exception:
            pass
        sync = getattr(gui, "_sync_tab_advanced_scroll", None)
        if callable(sync):
            try:
                sync()
            except Exception:
                pass
        return True

    gui._ensure_advanced_tab_content = _ensure_advanced_tab_content

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

    # ── Section 1 legacy summary dock (hidden; widgets kept for update hooks) ─
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
    sidebar.grid_remove()  # Clinical summary strip above the viz replaces this dock
    sidebar_inner = _install_sidebar_scroll(gui, sidebar)
    gui.sidebar = sidebar
    gui._overview_summary_host = sidebar

    _side_wrap = max(80, _SIDEBAR_MIN_WIDTH - 24)

    def _sync_sidebar_wrap(_event: object | None = None) -> None:
        if getattr(sidebar, "_wrap_syncing", False):
            return
        width = sidebar.winfo_width()
        if width < 40:
            return
        wrap = max(72, width - 20)
        from stablewalk.ui.theme import set_wraplength_safe

        setattr(sidebar, "_wrap_syncing", True)
        try:
            for attr in (
                "lbl_stab_headline",
                "lbl_stab_metrics",
                "lbl_stab_reason",
                "lbl_opensim_compact_ready",
            ):
                lbl = getattr(gui, attr, None)
                if lbl is not None and hasattr(lbl, "configure"):
                    set_wraplength_safe(lbl, wrap)
        finally:
            setattr(sidebar, "_wrap_syncing", False)

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

    dof_panel = ttk.Frame(motion_joint_row)
    dof_panel.grid(row=0, column=0, sticky="ew")
    dof_panel.columnconfigure(0, weight=0)
    dof_panel.columnconfigure(1, weight=1)
    dof_panel.columnconfigure(2, weight=0)
    gui._sidebar_dof_panel = dof_panel
    compare_lbl = tk.Label(
        dof_panel,
        text="Compare Joints",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    compare_lbl.grid(row=0, column=0, sticky="w", padx=(0, PAD_SM))
    gui.lbl_motion_compare_joints_title = compare_lbl

    gui.dof_chips_frame = tk.Frame(dof_panel, bg=PANEL, highlightthickness=0)
    gui.dof_chips_frame.grid(row=0, column=1, sticky="ew")

    btn_row = ttk.Frame(dof_panel)
    btn_row.grid(row=0, column=2, sticky="e", padx=(PAD_XS, 0))
    if not hasattr(gui, "var_motion_compare_add") or gui.var_motion_compare_add is None:
        gui.var_motion_compare_add = tk.StringVar(value=ADD_POINT_PLACEHOLDER)
    gui.cmb_motion_compare_add = ttk.Combobox(
        btn_row,
        textvariable=gui.var_motion_compare_add,
        values=[ADD_POINT_PLACEHOLDER],
        state="readonly",
        width=16,
        takefocus=False,
    )
    gui.cmb_motion_compare_add.pack(side=tk.LEFT, padx=(0, PAD_XS))
    gui.cmb_motion_compare_add.bind(
        "<<ComboboxSelected>>",
        lambda _e: gui._add_motion_compare_joint(),
    )
    create_tooltip(
        gui.cmb_motion_compare_add,
        "Add a joint to overlay on the graph without replacing Selected Joint. "
        "Ctrl+click the skeleton also adds to this compare set.",
        wraplength=340,
    )
    ttk.Button(
        btn_row,
        text="Clear all",
        style="Compact.TButton",
        command=gui._clear_dof_selection,
    ).pack(side=tk.LEFT)
    refresh_add = getattr(gui, "_refresh_add_point_combo", None)
    if callable(refresh_add):
        try:
            refresh_add()
        except Exception:
            pass

    def _reflow_compare_joints(_event: object | None = None) -> None:
        try:
            width = int(dof_panel.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 80:
            return
        stacked = width < 1020
        previous = getattr(gui, "_motion_compare_stacked", None)
        if stacked == previous:
            sync = getattr(gui, "_sync_motion_graph_height_share", None)
            if callable(sync):
                try:
                    sync()
                except Exception:
                    pass
            return
        gui._motion_compare_stacked = stacked
        try:
            if stacked:
                gui.dof_chips_frame.grid(
                    row=1, column=0, columnspan=3, sticky="ew"
                )
                btn_row.grid(
                    row=0,
                    column=2,
                    columnspan=1,
                    sticky="e",
                    padx=(PAD_XS, 0),
                    pady=0,
                )
            else:
                gui.dof_chips_frame.grid(
                    row=0, column=1, columnspan=1, sticky="ew"
                )
                btn_row.grid(
                    row=0, column=2, sticky="e", padx=(PAD_XS, 0), pady=0
                )
        except tk.TclError:
            gui._motion_compare_stacked = None
        sync = getattr(gui, "_sync_motion_graph_height_share", None)
        if callable(sync):
            try:
                gui._motion_graph_row_minsize = None
                sync()
            except Exception:
                pass

    dof_panel.bind("<Configure>", _reflow_compare_joints, add="+")

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

    from stablewalk.ui.tk.dashboard_notebook import ENABLE_COMPARE_TAB

    gui.btn_toggle_comparison = None
    gui.btn_compare_gait = None
    if ENABLE_COMPARE_TAB:
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

    from stablewalk.ui.media.demo_comparison import (
        COMPARISON_TREE_COLUMNS,
        COMPARISON_TREE_HEADINGS,
        COMPARISON_TREE_TEXT_COLS,
        COMPARISON_TREE_WIDTHS,
    )

    gui.demo_comparison_tree = _make_data_tree(
        cmp_body,
        COMPARISON_TREE_COLUMNS,
        COMPARISON_TREE_HEADINGS,
        col_widths=COMPARISON_TREE_WIDTHS,
        height=4,
        text_cols=COMPARISON_TREE_TEXT_COLS,
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

    # Reference-counted bind_all so rapid Enter/Leave across nested widgets
    # cannot stack duplicate global MouseWheel handlers.
    gui._opensim_wheel_depth = 0

    def _opensim_wheel_enter(_event: object | None = None) -> None:
        depth = int(getattr(gui, "_opensim_wheel_depth", 0) or 0)
        gui._opensim_wheel_depth = depth + 1
        if depth == 0:
            gui._opensim_canvas.bind_all("<MouseWheel>", _opensim_mousewheel)

    def _opensim_wheel_leave(_event: object | None = None) -> None:
        depth = int(getattr(gui, "_opensim_wheel_depth", 0) or 0)
        depth = max(0, depth - 1)
        gui._opensim_wheel_depth = depth
        if depth == 0:
            try:
                gui._opensim_canvas.unbind_all("<MouseWheel>")
            except tk.TclError:
                pass

    for widget in (opensim_inner, gui._opensim_canvas, scroll_host):
        widget.bind("<Enter>", _opensim_wheel_enter)
        widget.bind("<Leave>", _opensim_wheel_leave)

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
    gui.lbl_opensim_markers_count.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_ik_residuals = ttk.Label(
        opensim_inner, text="IK residuals: —", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_ik_residuals.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_pose_ik = ttk.Label(
        opensim_inner, text="Pose vs IK knees: —", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_pose_ik.pack(anchor=tk.W, pady=(0, PAD_XS))

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

    tree = getattr(gui, "dof_pos_tree", None)
    if tree is not None:
        host = getattr(tree, "_sw_host", None)
        if host is not None:
            try:
                host.pack_forget()
            except tk.TclError:
                pass

    data_host = getattr(gui, "_dashboard_data_row", None) or getattr(
        gui, "_advanced_workspace", None
    )
    if data_host is not None:
        bind_panel_word_wrap(
            gui,
            data_host,
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

    from stablewalk.ui.tk.timer_registry import schedule_unique

    # Coalesce reflow — avoid idle + delayed duplicate after() jobs.
    schedule_unique(gui, "_tab_reflow_after", 120, _reflow)
