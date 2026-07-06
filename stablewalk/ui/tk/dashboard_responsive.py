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

from stablewalk.ui.theme import BG, PAD_XS

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
    w = min(max(MIN_WINDOW_WIDTH, int(sw * 0.86)), sw - 32)
    h = min(max(MIN_WINDOW_HEIGHT, int(sh * 0.86)), sh - 72)
    x = max(0, (sw - w) // 2)
    y = max(0, (sh - h) // 2)
    return f"{w}x{h}+{x}+{y}"


def install_responsive_shell(gui, parent: tk.Misc) -> ttk.Frame:
    """Dashboard body host that fills the viewport."""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)

    inner = ttk.Frame(parent)
    inner.grid(row=0, column=0, sticky="nsew")
    gui._dash_scroll_outer = None
    gui._dash_scroll_canvas = None
    gui._dash_scrollbar = None
    gui._dash_scroll_inner = inner
    gui._dash_scroll_window_id = None
    gui._sync_dashboard_scroll = lambda _event=None: None
    return inner


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
    """Flat sidebar: stability card, joints, action buttons."""
    inner: tk.Misc = gui._sidebar_scroll_inner
    for child in inner.winfo_children():
        try:
            child.pack_forget()
        except tk.TclError:
            pass

    gui._sidebar_stability_panel.pack(in_=inner, fill=tk.X, pady=(0, PAD_XS))
    gui._sidebar_dof_panel.pack(in_=inner, fill=tk.X, pady=(0, PAD_XS))
    if hasattr(gui, "btn_toggle_comparison"):
        gui.btn_toggle_comparison.pack(in_=inner, fill=tk.X, pady=(0, PAD_XS))
    gui._sidebar_opensim_panel.pack(in_=inner, fill=tk.X, pady=(0, 0))
    gui._sidebar_sections = ()


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

    from stablewalk.ui.tk.dashboard_layout import DASHBOARD_GUTTER, _TOP_SKELETON_WEIGHT, _TOP_VIDEO_WEIGHT

    use_tabs = wmode is WidthMode.SMALL
    prev_tabs = getattr(gui, "_viz_tabs_active", False)
    gui._viz_tabs_active = use_tabs

    if use_tabs:
        content_row = 1
        gui._primary_viz_content_row = content_row
        if tab_bar is not None:
            tab_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, PAD_XS))
        host.rowconfigure(0, weight=0, minsize=0)
        host.rowconfigure(1, weight=1)
        host.columnconfigure(0, weight=1)
        host.columnconfigure(1, weight=0, minsize=0)
        apply_viz_tab_visibility(gui)
    else:
        content_row = 0
        gui._primary_viz_content_row = content_row
        if tab_bar is not None:
            tab_bar.grid_remove()
        host.rowconfigure(0, weight=1)
        host.rowconfigure(1, weight=0, minsize=0)
        host.columnconfigure(0, weight=_TOP_VIDEO_WEIGHT, uniform="viz")
        host.columnconfigure(1, weight=_TOP_SKELETON_WEIGHT, uniform="viz")
        video.grid(row=content_row, column=0, sticky="nsew", padx=(0, DASHBOARD_GUTTER), pady=(0, 0))
        skel.grid(row=content_row, column=1, sticky="nsew", padx=(0, 0), pady=(0, 0))

    if use_tabs != prev_tabs:
        gui._visuals_stacked = use_tabs


def _apply_sidebar_layout(gui, wmode: WidthMode, width: int) -> None:
    body = gui._dashboard_body
    sidebar = gui.sidebar
    analysis_outer = getattr(gui, "_analysis_scroll_outer", None)
    primary_viz = getattr(gui, "_primary_viz_host", None)
    inline = _sidebar_inline(wmode, width)

    if inline:
        if primary_viz is not None:
            primary_viz.grid(row=0, column=0, columnspan=2, sticky="nsew")
        sidebar.grid(
            row=0,
            column=2,
            rowspan=1,
            sticky="nsew",
            padx=(8, 0),
            pady=(0, 0),
        )
        if analysis_outer is not None:
            analysis_outer.grid(row=1, column=0, columnspan=3, sticky="nsew")
            gui._analysis_scroll_row = 1
        try:
            body.rowconfigure(0, weight=52)
            body.rowconfigure(1, weight=48)
            body.rowconfigure(2, weight=0, minsize=0)
            cap = max(_SIDEBAR_MIN, min(_SIDEBAR_MAX, int(width * _SIDEBAR_FRAC_LARGE)))
            body.columnconfigure(2, weight=18, minsize=cap)
        except tk.TclError:
            pass
    else:
        if primary_viz is not None:
            primary_viz.grid(row=0, column=0, columnspan=3, sticky="nsew")
        sidebar.grid(
            row=1,
            column=0,
            columnspan=3,
            rowspan=1,
            sticky="ew",
            padx=(0, 0),
            pady=(PAD_XS, 0),
        )
        if analysis_outer is not None:
            analysis_outer.grid(row=2, column=0, columnspan=3, sticky="nsew")
            gui._analysis_scroll_row = 2
        try:
            body.rowconfigure(0, weight=52)
            body.rowconfigure(1, weight=0, minsize=0)
            body.rowconfigure(2, weight=48)
            body.columnconfigure(2, weight=0, minsize=0)
        except tk.TclError:
            pass

    gui._sidebar_was_compact = not inline


def _apply_height_layout(gui, hmode: HeightMode, height: int, width: int) -> None:
    body = getattr(gui, "_dashboard_body", None)
    if body is None:
        return

    wmode, _ = classify_layout(width, height)
    inline = _sidebar_inline(wmode, width)
    chrome = 230
    avail = max(360, height - chrome)

    if hmode is HeightMode.SHORT:
        viz_share = 0.46
        analysis_share = 0.54
        viz_min = max(180, int(avail * viz_share))
        analysis_min = max(160, int(avail * analysis_share))
    else:
        viz_share = 0.52
        analysis_share = 0.48
        viz_min = max(200, int(avail * viz_share))
        analysis_min = max(180, int(avail * analysis_share))

    wmode, _ = classify_layout(width, height)
    if wmode is WidthMode.SMALL:
        viz_min = max(160, int(viz_min * 0.92))

    try:
        body.rowconfigure(0, weight=52, minsize=viz_min)
        if inline:
            body.rowconfigure(1, weight=48, minsize=analysis_min)
        else:
            body.rowconfigure(1, weight=0, minsize=0)
            body.rowconfigure(2, weight=48, minsize=analysis_min)
    except tk.TclError:
        pass

    traj = getattr(gui, "traj_panel", None)
    if traj is not None:
        graph_min = max(140, int(analysis_min * 0.68))
        try:
            traj.rowconfigure(1, weight=1, minsize=graph_min)
        except tk.TclError:
            pass

    scroll_outer = getattr(gui, "_analysis_scroll_outer", None)
    scrollbar = getattr(gui, "_analysis_scrollbar", None)
    canvas = getattr(gui, "_analysis_scroll_canvas", None)
    if scroll_outer is None or canvas is None:
        return

    if hmode is HeightMode.SHORT:
        try:
            scroll_outer.rowconfigure(0, weight=1)
            if scrollbar is not None:
                scrollbar.grid(row=0, column=1, sticky="ns")
            canvas.configure(yscrollcommand=scrollbar.set if scrollbar else None)
        except tk.TclError:
            pass
    else:
        try:
            if scrollbar is not None:
                scrollbar.grid(row=0, column=1, sticky="ns")
        except tk.TclError:
            pass


def _reflow_metric_summary(gui, width: int) -> None:
    slots = getattr(gui, "dof_analysis_summary_slots", None)
    hosts = getattr(gui, "dof_analysis_summary_hosts", None)
    if not slots or not hosts:
        return
    cols = 4 if width >= 560 else 2
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
    inner = getattr(gui, "dof_analysis_graph_inner", None)
    if inner is None:
        return
    wmode, _ = classify_layout(width, gui.root.winfo_height())
    if wmode is WidthMode.SMALL:
        explain_min = max(88, min(120, int(width * 0.14)))
        cube_min = max(160, int(width * 0.55))
    else:
        cube_min = max(180, min(520, int(width * 0.38)))
        explain_min = max(88, min(130, int(width * 0.10)))
    try:
        inner.columnconfigure(0, weight=3, minsize=cube_min)
        inner.columnconfigure(1, weight=1, minsize=explain_min)
    except tk.TclError:
        pass


def _apply_transport_layout(gui, width: int) -> None:
    """Compact playback controls on narrow windows; timeline always flexes."""
    compact = width < WIDTH_MEDIUM
    narrow = width < 1050

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

    gui._layout_width_mode = wmode
    gui._layout_height_mode = hmode

    sync = getattr(gui, "_sync_analysis_scroll", None)
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
