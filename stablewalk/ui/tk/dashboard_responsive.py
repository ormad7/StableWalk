"""
Responsive layout helpers for the StableWalk Tk dashboard.

Layout reacts to the *current* root window size (not a fixed design resolution).
Width modes control the primary visualization grid; height modes control whether
the analysis region scrolls vertically.
"""

from __future__ import annotations

from enum import Enum

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.theme import (
    BG,
    DASHBOARD_GUTTER,
    PAD_SM,
    PAD_XS,
    WIDTH_COMPACT_TRANSPORT,
    WIDTH_METRIC_REFLOW,
    WIDTH_STACK_ANALYSIS,
)

# Supported layout range.
MIN_WINDOW_WIDTH = 960
MIN_WINDOW_HEIGHT = 600

# Width breakpoints (tested on common laptop / desktop sizes).
WIDTH_LARGE = 1550          # Video | 3D | Summary (3 columns)
WIDTH_MEDIUM = 1200         # Video | 3D; summary inline or below
WIDTH_MEDIUM_INLINE = 1350  # Within medium: summary stays in right column

# Height breakpoint — below this the analysis block scrolls.
HEIGHT_TALL = 760

# Sidebar width caps (fraction of window width).
_SIDEBAR_FRAC_LARGE = 0.18
_SIDEBAR_FRAC_MEDIUM = 0.16
_SIDEBAR_MIN = 132
_SIDEBAR_MAX = 280

# Debounce rapid resize events.
_RESIZE_DEBOUNCE_MS = 80

# Legacy aliases (scripts may import these).
WIDTH_COMPACT_SIDEBAR = WIDTH_MEDIUM_INLINE
WIDTH_STACKED_VISUALS = WIDTH_MEDIUM
WIDTH_NARROW_VISUALS = WIDTH_MEDIUM


class WidthMode(str, Enum):
    LARGE = "large"
    MEDIUM = "medium"
    SMALL = "small"


class HeightMode(str, Enum):
    TALL = "tall"
    SHORT = "short"


def classify_layout(width: int, height: int) -> tuple[WidthMode, HeightMode]:
    """Return layout modes for the given root window dimensions."""
    if width >= WIDTH_LARGE:
        wmode = WidthMode.LARGE
    elif width >= WIDTH_MEDIUM:
        wmode = WidthMode.MEDIUM
    else:
        wmode = WidthMode.SMALL

    hmode = HeightMode.TALL if height >= HEIGHT_TALL else HeightMode.SHORT
    return wmode, hmode


def initial_window_geometry(root: tk.Tk) -> str:
    """Size and center the window from the current screen — not a fixed resolution."""
    root.update_idletasks()
    sw = max(MIN_WINDOW_WIDTH, root.winfo_screenwidth())
    sh = max(MIN_WINDOW_HEIGHT, root.winfo_screenheight())
    w = min(max(MIN_WINDOW_WIDTH, int(sw * 0.90)), sw - 24)
    h = min(max(MIN_WINDOW_HEIGHT, int(sh * 0.90)), sh - 56)
    x = max(0, (sw - w) // 2)
    y = max(0, (sh - h) // 2)
    return f"{w}x{h}+{x}+{y}"


def install_responsive_shell(
    gui, parent: tk.Misc
) -> tuple[ttk.Frame, ttk.Frame, ttk.Frame, ttk.Frame, ttk.Frame, ttk.Frame]:
    """Fixed-viewport dashboard notebook (six tabs; no full-page scroll)."""
    from stablewalk.ui.tk.dashboard_notebook import install_dashboard_notebook

    return install_dashboard_notebook(gui, parent)


def _install_sidebar_scroll(gui, sidebar: tk.Misc) -> tk.Misc:
    """Sidebar host — panels stack directly."""
    sidebar.columnconfigure(0, weight=1)
    sidebar.rowconfigure(0, weight=1)
    inner = ttk.Frame(sidebar, style="Card.TFrame", padding=(PAD_XS, PAD_XS))
    inner.grid(row=0, column=0, sticky="nsew")
    gui._sidebar_scroll_canvas = None
    gui._sidebar_scroll_inner = inner
    gui._sidebar_scroll_window_id = None
    return inner


def setup_compact_sidebar(gui) -> None:
    """Summary dock below video|skeleton — compact gait scores, collapsible."""
    import tkinter as tk

    inner: tk.Misc = gui._sidebar_scroll_inner
    for child in inner.winfo_children():
        try:
            child.pack_forget()
        except tk.TclError:
            pass

    header = ttk.Frame(inner, style="Card.TFrame")
    header.pack(fill=tk.X, pady=(0, PAD_SM))
    header.columnconfigure(0, weight=1)

    expanded = bool(getattr(gui, "_overview_summary_expanded", True))
    toggle_lbl = "▼" if expanded else "▶"
    gui._overview_summary_toggle = ttk.Button(
        header,
        text=f"{toggle_lbl} Gait Analysis Summary",
        style="Compact.TButton",
        command=lambda: _toggle_overview_summary(gui),
    )
    gui._overview_summary_toggle.grid(row=0, column=0, sticky="w")

    content = ttk.Frame(inner, style="Card.TFrame")
    gui._overview_summary_content = content
    if expanded:
        content.pack(fill=tk.X)

    gui._sidebar_stability_panel.pack(in_=content, fill=tk.X, pady=(0, 0))

    for attr in (
        "_sidebar_gait_cycle_panel",
        "_sidebar_physics_force_panel",
        "_sidebar_dof_panel",
        "_sidebar_opensim_panel",
    ):
        panel = getattr(gui, attr, None)
        if panel is not None:
            try:
                panel.pack_forget()
            except tk.TclError:
                pass
    if hasattr(gui, "btn_toggle_comparison"):
        try:
            gui.btn_toggle_comparison.pack_forget()
        except tk.TclError:
            pass
    gui._sidebar_sections = ()


def _toggle_overview_summary(gui) -> None:
    """Expand or collapse the Overview Gait Analysis Summary dock."""
    import tkinter as tk

    expanded = not bool(getattr(gui, "_overview_summary_expanded", True))
    gui._overview_summary_expanded = expanded
    content = getattr(gui, "_overview_summary_content", None)
    toggle = getattr(gui, "_overview_summary_toggle", None)
    if content is not None:
        try:
            if expanded:
                content.pack(fill=tk.X)
            else:
                content.pack_forget()
        except tk.TclError:
            pass
    if toggle is not None:
        try:
            toggle.configure(
                text=("▼ Gait Analysis Summary" if expanded else "▶ Gait Analysis Summary")
            )
        except tk.TclError:
            pass
    # Keep video/skeleton free of overlap after collapse — re-fit panels.
    refit = getattr(gui, "_on_video_label_resize", None)
    if callable(refit):
        try:
            refit()
        except Exception:
            pass
    fit = getattr(gui, "_fit_skeleton_canvas", None)
    if callable(fit):
        try:
            fit()
        except Exception:
            pass


def layout_sidebar_panels(gui) -> None:
    setup_compact_sidebar(gui)


def apply_viz_tab_visibility(gui) -> None:
    """Show only the active primary visualization when tab mode is active."""
    if not getattr(gui, "_viz_tabs_active", False):
        return
    tab = getattr(gui, "_viz_tab_var", None)
    choice = tab.get() if tab is not None else "video"
    video = gui.video_frame
    skel = gui.skel_frame
    content_row = getattr(gui, "_primary_viz_content_row", 1)
    if choice == "video":
        video.grid(row=content_row, column=0, sticky="nsew", padx=(0, 0), pady=(0, 0))
        skel.grid_remove()
    else:
        skel.grid(row=content_row, column=0, sticky="nsew", padx=(0, 0), pady=(0, 0))
        video.grid_remove()
    fit_skel = getattr(gui, "_fit_skeleton_canvas", None)
    if fit_skel is not None:
        fit_skel()
    refit = getattr(gui, "_on_video_label_resize", None)
    if refit is not None:
        refit()


def _sidebar_inline(wmode: WidthMode, width: int) -> bool:
    """True when the summary column sits beside the primary visuals."""
    if wmode is WidthMode.LARGE:
        return True
    if wmode is WidthMode.MEDIUM:
        return width >= WIDTH_MEDIUM_INLINE
    return False


def _apply_visual_layout(gui, wmode: WidthMode) -> None:
    """Side-by-side visuals, or tabbed single-pane on small screens."""
    host = getattr(gui, "_primary_viz_host", None)
    video = getattr(gui, "video_frame", None)
    skel = getattr(gui, "skel_frame", None)
    tab_bar = getattr(gui, "_viz_tab_bar", None)
    if host is None or video is None or skel is None:
        return

    # When the professional View Mode workspace owns the Overview layout, defer
    # to it instead of forcing the legacy column grid.
    if getattr(gui, "_overview_view_mode_active", False):
        from stablewalk.ui.tk.dashboard_overview_view_mode import (
            apply_overview_view_mode,
        )

        gui._viz_tabs_active = False
        try:
            apply_overview_view_mode(gui, animate=False, persist=False)
        except Exception:
            pass
        return

    from stablewalk.ui.tk.dashboard_layout import _OVERVIEW_COL_GUTTER

    use_tabs = wmode is WidthMode.SMALL
    prev_tabs = getattr(gui, "_viz_tabs_active", False)
    gui._viz_tabs_active = use_tabs

    if use_tabs:
        content_row = 1
        gui._primary_viz_content_row = content_row
        if tab_bar is not None:
            tab_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, PAD_XS))
        host.rowconfigure(0, weight=0, minsize=0)
        host.rowconfigure(1, weight=0)
        host.rowconfigure(2, weight=0, minsize=0)
        host.columnconfigure(0, weight=1)
        host.columnconfigure(1, weight=0, minsize=0)
        host.columnconfigure(2, weight=0, minsize=0)
        apply_viz_tab_visibility(gui)
    else:
        content_row = 0
        gui._primary_viz_content_row = content_row
        if tab_bar is not None:
            tab_bar.grid_remove()
        host.rowconfigure(0, weight=0)
        host.rowconfigure(1, weight=0, minsize=0)
        host.rowconfigure(2, weight=0, minsize=0)
        from stablewalk.ui.tk.dashboard_sections import (
            SEC1_SKELETON_WEIGHT,
            SEC1_VIDEO_WEIGHT,
        )

        host.columnconfigure(0, weight=SEC1_VIDEO_WEIGHT, uniform="sec1")
        host.columnconfigure(1, weight=SEC1_SKELETON_WEIGHT, uniform="sec1")
        host.columnconfigure(2, weight=0, minsize=0)
        video.grid(
            row=content_row,
            column=0,
            sticky="nsew",
            padx=(0, _OVERVIEW_COL_GUTTER),
            pady=(0, 0),
        )
        skel.grid(
            row=content_row,
            column=1,
            sticky="nsew",
            padx=(_OVERVIEW_COL_GUTTER, _OVERVIEW_COL_GUTTER),
            pady=(0, 0),
        )

    if use_tabs != prev_tabs:
        gui._visuals_stacked = use_tabs


def _apply_sidebar_layout(gui, wmode: WidthMode, width: int) -> None:
    """Overview tab keeps video | 3D | summary at desktop widths."""
    section1 = getattr(gui, "_section_visual", None)
    sidebar = getattr(gui, "sidebar", None)
    if section1 is None or sidebar is None:
        return

    # The View Mode workspace decides where the summary/trajectory column goes.
    if getattr(gui, "_overview_view_mode_active", False):
        return

    from stablewalk.ui.tk.dashboard_sections import (
        SEC1_SKELETON_WEIGHT,
        SEC1_VIDEO_WEIGHT,
    )

    use_tabs = getattr(gui, "_viz_tabs_active", False)
    if use_tabs:
        section1.columnconfigure(0, weight=1, uniform="sec1")
        section1.columnconfigure(1, weight=0, minsize=0)
        section1.columnconfigure(2, weight=0, minsize=0)
    else:
        section1.columnconfigure(0, weight=SEC1_VIDEO_WEIGHT, uniform="sec1")
        section1.columnconfigure(1, weight=SEC1_SKELETON_WEIGHT, uniform="sec1")
        section1.columnconfigure(2, weight=0, minsize=0)
    section1.rowconfigure(0, weight=1)
    section1.rowconfigure(1, weight=0, minsize=0)

    if not use_tabs:
        summary_row = int(getattr(gui, "_overview_summary_row", 1))
        sidebar.grid(
            row=summary_row,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=0,
            pady=(PAD_XS, 0),
        )

    motion = getattr(gui, "_tab_motion", None)
    if motion is not None:
        try:
            motion.rowconfigure(0, weight=1)
            motion.rowconfigure(1, weight=0)
        except tk.TclError:
            pass


def _reflow_analysis_panel_stack(gui, width: int) -> None:
    """Side-by-side knee, joint path, and temporal metrics; contact strip unchanged."""
    bottom = getattr(gui, "_dashboard_bottom_row", None)
    traj = getattr(gui, "traj_panel", None)
    knee = getattr(gui, "knee_panel", None)
    contact = getattr(gui, "contact_gait_panel", None)
    metrics = getattr(gui, "_motion_temporal_metrics_panel", None)
    if bottom is None or traj is None or knee is None:
        return

    from stablewalk.ui.tk.dashboard_layout import (
        CONTACT_GAIT_CHART_MIN_H,
        DASHBOARD_GUTTER,
        MOTION_TOP_COLUMN_WEIGHTS,
        _MOTION_KNEE_PANEL_MIN_W,
        _MOTION_METRICS_PANEL_MIN_W,
        _MOTION_TOP_ROW_MIN_H,
        _MOTION_TRAJ_PANEL_MIN_W,
    )

    stack = width < WIDTH_STACK_ANALYSIS
    try:
        if stack:
            bottom.columnconfigure(0, weight=1, minsize=0, uniform="")
            bottom.columnconfigure(1, weight=1, minsize=0, uniform="")
            bottom.columnconfigure(2, weight=1, minsize=0, uniform="")
            bottom.rowconfigure(0, weight=1, minsize=300)
            bottom.rowconfigure(1, weight=1, minsize=_MOTION_TOP_ROW_MIN_H)
            bottom.rowconfigure(2, weight=0, minsize=180)
            bottom.rowconfigure(3, weight=0, minsize=CONTACT_GAIT_CHART_MIN_H)
            knee.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=(0, 0), pady=(0, PAD_XS))
            traj.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=(0, 0), pady=(0, PAD_XS))
            if metrics is not None:
                metrics.grid(
                    row=2,
                    column=0,
                    columnspan=3,
                    rowspan=1,
                    sticky="nsew",
                    padx=(0, 0),
                    pady=(0, PAD_XS),
                )
            if contact is not None:
                contact.grid(
                    row=3,
                    column=0,
                    columnspan=3,
                    sticky="nsew",
                    padx=(0, 0),
                    pady=(0, 0),
                )
            gui._motion_content_min_height = (
                300
                + _MOTION_TOP_ROW_MIN_H
                + 180
                + CONTACT_GAIT_CHART_MIN_H
                + 3 * PAD_XS
            )
            bottom.configure(height=gui._motion_content_min_height)
        else:
            # Scientific workstation proportions: angle 30%, trajectory 50%,
            # temporal metrics 20%; contact / phase / vGRF spans the full row.
            bottom.columnconfigure(
                0,
                weight=MOTION_TOP_COLUMN_WEIGHTS[0],
                minsize=_MOTION_KNEE_PANEL_MIN_W,
                uniform="motion_top",
            )
            bottom.columnconfigure(
                1,
                weight=MOTION_TOP_COLUMN_WEIGHTS[1],
                minsize=_MOTION_TRAJ_PANEL_MIN_W,
                uniform="motion_top",
            )
            bottom.columnconfigure(
                2,
                weight=MOTION_TOP_COLUMN_WEIGHTS[2],
                minsize=_MOTION_METRICS_PANEL_MIN_W,
                uniform="motion_top",
            )
            bottom.rowconfigure(0, weight=3, minsize=_MOTION_TOP_ROW_MIN_H)
            bottom.rowconfigure(1, weight=2, minsize=CONTACT_GAIT_CHART_MIN_H)
            bottom.rowconfigure(2, weight=0, minsize=0)
            bottom.rowconfigure(3, weight=0, minsize=0)
            knee.grid(
                row=0,
                column=0,
                columnspan=1,
                sticky="nsew",
                padx=(0, DASHBOARD_GUTTER),
                pady=(0, 0),
            )
            traj.grid(row=0, column=1, columnspan=1, sticky="nsew", padx=(0, DASHBOARD_GUTTER), pady=(0, 0))
            if metrics is not None:
                metrics.grid(
                    row=0,
                    column=2,
                    columnspan=1,
                    rowspan=1,
                    sticky="nsew",
                    padx=(0, 0),
                    pady=(0, 0),
                )
            if contact is not None:
                contact.grid(
                    row=1,
                    column=0,
                    columnspan=3,
                    sticky="nsew",
                    padx=(0, 0),
                    pady=(PAD_XS, 0),
                )
            gui._motion_content_min_height = (
                _MOTION_TOP_ROW_MIN_H + CONTACT_GAIT_CHART_MIN_H + PAD_XS
            )
            bottom.configure(height=gui._motion_content_min_height)
    except tk.TclError:
        pass


def _apply_height_layout(gui, hmode: HeightMode, height: int, width: int) -> None:
    """Content-driven section heights — only reflow analysis panel stacking."""
    _reflow_analysis_panel_stack(gui, width)
    scroll_sync = getattr(gui, "_sync_motion_scroll", None) or getattr(
        gui, "_sync_dashboard_scroll", None
    )
    if scroll_sync is not None:
        scroll_sync()


def _reflow_metric_summary(gui, width: int) -> None:
    slots = getattr(gui, "dof_analysis_summary_slots", None)
    hosts = getattr(gui, "dof_analysis_summary_hosts", None)
    if not slots or not hosts:
        return
    cols = 4 if width >= WIDTH_METRIC_REFLOW else 2
    for idx, host in enumerate(hosts):
        row, col = divmod(idx, cols)
        try:
            host.grid(row=row, column=col, sticky="nsew", padx=(0, 4), pady=1)
        except tk.TclError:
            pass
    primary = hosts[0].master if hosts else None
    if primary is not None:
        for c in range(4):
            try:
                primary.columnconfigure(c, weight=1 if c < cols else 0, uniform="metric")
            except tk.TclError:
                pass


def _apply_graph_column_minsizes(gui, width: int) -> None:
    """Graph hosts use content-driven width — no forced cube column minsize."""
    inner = getattr(gui, "dof_analysis_graph_inner", None)
    if inner is None:
        return
    try:
        inner.columnconfigure(0, weight=1, minsize=0)
    except tk.TclError:
        pass


def _apply_transport_layout(gui, width: int) -> None:
    """Compact playback controls on narrow windows; timeline always flexes."""
    compact = width < WIDTH_MEDIUM
    narrow = width < WIDTH_COMPACT_TRANSPORT

    for attr, visible in (
        ("_transport_sep1", not narrow),
        ("_transport_sep2", not compact),
        ("_transport_frame", not narrow),
        ("_lbl_sampling", not narrow),
        ("_lbl_speed", not narrow),
    ):
        widget = getattr(gui, attr, None)
        if widget is None:
            continue
        try:
            if visible:
                widget.grid()
            else:
                widget.grid_remove()
        except tk.TclError:
            pass

    scale = getattr(gui, "_speed_scale", None)
    if scale is not None:
        try:
            scale.configure(length=56 if compact else 72)
        except tk.TclError:
            pass

    sampling = getattr(gui, "cmb_sampling", None)
    if sampling is not None:
        try:
            sampling.configure(width=5 if compact else 6)
        except tk.TclError:
            pass


def _sync_scroll_bottom_clearance(gui) -> None:
    """Advanced tab scroll only — main dashboard uses fixed notebook tabs."""
    sync = getattr(gui, "_sync_tab_advanced_scroll", None)
    if sync is not None:
        sync()


def apply_responsive_layout(gui, *, width: int | None = None, height: int | None = None) -> None:
    """Recompute layout for the current root window size."""
    root = gui.root
    width = width if width is not None else root.winfo_width()
    height = height if height is not None else root.winfo_height()
    if width < 2 or height < 2:
        return

    wmode, hmode = classify_layout(width, height)
    prev_w = getattr(gui, "_layout_width_mode", None)
    prev_h = getattr(gui, "_layout_height_mode", None)

    if prev_w != wmode or prev_h != hmode or getattr(gui, "_force_layout_reflow", False):
        _apply_visual_layout(gui, wmode)
        _apply_sidebar_layout(gui, wmode, width)
        gui._force_layout_reflow = False

    _apply_height_layout(gui, hmode, height, width)
    _reflow_metric_summary(gui, width)
    _apply_graph_column_minsizes(gui, width)
    _apply_transport_layout(gui, width)
    _sync_scroll_bottom_clearance(gui)

    gui._layout_width_mode = wmode
    gui._layout_height_mode = hmode

    sync = getattr(gui, "_sync_dashboard_scroll", None) or getattr(
        gui, "_sync_analysis_scroll", None
    )
    if sync is not None:
        sync()

    fit_skel = getattr(gui, "_fit_skeleton_canvas", None)
    if fit_skel is not None:
        fit_skel()
    fit_traj = getattr(gui, "_fit_dof_traj_canvas", None)
    if fit_traj is not None:
        fit_traj()
    sync_wrap = getattr(gui, "_sync_utility_sidebar_wrap", None)
    if sync_wrap is not None:
        sync_wrap()

    from stablewalk.ui.theme import bind_responsive_wrap

    overview = getattr(gui, "_overview_metrics_row", None)
    if overview is not None:
        bind_responsive_wrap(gui, overview, ("lbl_overview_demo_compare",), margin=12)

    refit_video = getattr(gui, "_on_video_label_resize", None)
    if refit_video is not None:
        refit_video()


def bind_responsive_handlers(gui) -> None:
    """Attach debounced resize handler — root window Configure events only."""
    root = gui.root
    gui._responsive_after_id: str | None = None
    gui._last_layout_size: tuple[int, int] = (0, 0)

    def _schedule(event: tk.Event | None = None) -> None:
        if event is not None and event.widget is not root:
            return
        if event is not None:
            w, h = event.width, event.height
            if w < 2 or h < 2:
                return
            last = gui._last_layout_size
            if abs(w - last[0]) < 2 and abs(h - last[1]) < 2:
                return

        if gui._responsive_after_id is not None:
            try:
                root.after_cancel(gui._responsive_after_id)
            except tk.TclError:
                pass

        def _run() -> None:
            gui._responsive_after_id = None
            w = root.winfo_width()
            h = root.winfo_height()
            if w < 2 or h < 2:
                return
            gui._last_layout_size = (w, h)
            apply_responsive_layout(gui, width=w, height=h)

        gui._responsive_after_id = root.after(_RESIZE_DEBOUNCE_MS, _run)

    root.bind("<Configure>", _schedule, add="+")
    root.after_idle(lambda: _schedule(None))


def finalize_responsive_dashboard(gui) -> None:
    setup_compact_sidebar(gui)
    bind_responsive_handlers(gui)
