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
    disclosure_label,
)
from stablewalk.ui.tk.layout_scale import (
    MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
    WIDTH_XLARGE,
    layout_scale_from_gui,
    scaled_breakpoints,
)
from stablewalk.ui.tk.resize_coalesce import RESIZE_IGNORE_PX, RESIZE_SETTLE_MS

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

# Trailing quiet period before expensive layout / figure fits.
_RESIZE_DEBOUNCE_MS = RESIZE_SETTLE_MS

# Legacy aliases (scripts may import these).
WIDTH_COMPACT_SIDEBAR = WIDTH_MEDIUM_INLINE
WIDTH_STACKED_VISUALS = WIDTH_MEDIUM
WIDTH_NARROW_VISUALS = WIDTH_MEDIUM


class WidthMode(str, Enum):
    XLARGE = "xlarge"
    LARGE = "large"
    MEDIUM = "medium"
    SMALL = "small"


class HeightMode(str, Enum):
    TALL = "tall"
    SHORT = "short"


def classify_layout(width: int, height: int) -> tuple[WidthMode, HeightMode]:
    """Return layout modes for the given root window dimensions."""
    if width >= WIDTH_XLARGE:
        wmode = WidthMode.XLARGE
    elif width >= WIDTH_LARGE:
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
) -> tuple[ttk.Frame, ttk.Frame, ttk.Frame, ttk.Frame, ttk.Frame | None, ttk.Frame]:
    """Fixed-viewport dashboard notebook (Compare tab is optional)."""
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
    gui._overview_summary_toggle = ttk.Button(
        header,
        text=disclosure_label("Session Metrics", expanded=expanded),
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
    toggle = getattr(gui, "btn_toggle_comparison", None)
    if toggle is not None:
        try:
            toggle.pack_forget()
        except tk.TclError:
            pass
    gui._sidebar_sections = ()


def _toggle_overview_summary(gui) -> None:
    """Expand or collapse the Overview Session Metrics dock."""
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
                text=disclosure_label("Session Metrics", expanded=expanded)
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
    content_row = getattr(gui, "_primary_viz_content_row", 0)
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
    if wmode in (WidthMode.LARGE, WidthMode.XLARGE):
        return True
    if wmode is WidthMode.MEDIUM:
        return width >= WIDTH_MEDIUM_INLINE
    return False


def _apply_visual_layout(gui, wmode: WidthMode) -> None:
    """Side-by-side visuals, or tabbed single-pane on small screens."""
    host = getattr(gui, "_primary_viz_host", None) or getattr(
        gui, "_section_visual", None
    )
    video = getattr(gui, "video_frame", None)
    skel = getattr(gui, "skel_frame", None)
    tab_bar = getattr(gui, "_viz_tab_bar", None)
    if host is None or video is None or skel is None:
        return

    # View Mode owns Video | Skeleton | Path. Those widgets are created once;
    # resize must not remount them or switch to the legacy tabbed viz.
    if getattr(gui, "_overview_view_mode_active", False):
        gui._viz_tabs_active = False
        return

    from stablewalk.ui.tk.analysis_split import split_mapped
    from stablewalk.ui.tk.dashboard_layout import _OVERVIEW_COL_GUTTER

    use_tabs = wmode is WidthMode.SMALL
    prev_tabs = getattr(gui, "_viz_tabs_active", False)
    gui._viz_tabs_active = use_tabs

    paned = getattr(gui, "_overview_viz_split", None)
    if paned is not None and use_tabs:
        try:
            paned.grid_remove()
        except tk.TclError:
            pass
    elif split_mapped(paned) and not use_tabs:
        try:
            n_panes = len(paned.panes())
        except tk.TclError:
            n_panes = 0
        if n_panes >= 3:
            return
        try:
            paned.grid_remove()
        except tk.TclError:
            pass

    if use_tabs:
        content_row = 1
        gui._primary_viz_content_row = content_row
        if tab_bar is not None:
            tab_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, PAD_XS))
        host.rowconfigure(0, weight=0, minsize=0)
        host.rowconfigure(1, weight=1)
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
        host.rowconfigure(0, weight=1, minsize=0)
        host.rowconfigure(1, weight=0, minsize=0)
        host.rowconfigure(2, weight=0, minsize=0)
        from stablewalk.ui.tk.dashboard_sections import apply_overview_viz_grid_weights

        apply_overview_viz_grid_weights(host, content_row=content_row)
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

    from stablewalk.ui.tk.dashboard_sections import apply_overview_viz_grid_weights

    apply_overview_viz_grid_weights(
        section1, content_row=int(getattr(gui, "_primary_viz_content_row", 0))
    )

    if getattr(gui, "overview_traj_panel", None) is not None:
        return

    use_tabs = getattr(gui, "_viz_tabs_active", False)
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
            motion.rowconfigure(0, weight=0)  # page header
            motion.rowconfigure(1, weight=1)  # motion content
        except tk.TclError:
            pass


def _reflow_analysis_panel_stack(gui, width: int) -> None:
    """Side-by-side knee and joint path on Motion; contact/metrics stay on Biomech when hosted there."""
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

    contact_on_motion = contact is not None and getattr(contact, "master", None) is bottom
    metrics_on_motion = metrics is not None and getattr(metrics, "master", None) is bottom

    scale = layout_scale_from_gui(gui)
    stack_floor = int(scaled_breakpoints(scale)["motion_stack"])
    if abs(scale - 1.0) < 0.08:
        stack_floor = WIDTH_STACK_ANALYSIS
    stack = width < stack_floor
    try:
        if stack:
            bottom.columnconfigure(0, weight=1, minsize=0, uniform="")
            bottom.columnconfigure(1, weight=1, minsize=0, uniform="")
            bottom.columnconfigure(2, weight=0, minsize=0, uniform="")
            bottom.rowconfigure(0, weight=1, minsize=300)
            bottom.rowconfigure(1, weight=1, minsize=_MOTION_TOP_ROW_MIN_H)
            bottom.rowconfigure(2, weight=0, minsize=180 if metrics_on_motion else 0)
            bottom.rowconfigure(
                3, weight=0, minsize=CONTACT_GAIT_CHART_MIN_H if contact_on_motion else 0
            )
            knee.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=(0, 0), pady=(0, PAD_XS))
            traj.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=(0, 0), pady=(0, PAD_XS))
            if metrics_on_motion and metrics is not None:
                metrics.grid(
                    row=2,
                    column=0,
                    columnspan=2,
                    rowspan=1,
                    sticky="nsew",
                    padx=(0, 0),
                    pady=(0, PAD_XS),
                )
            if contact_on_motion and contact is not None:
                contact.grid(
                    row=3,
                    column=0,
                    columnspan=2,
                    sticky="nsew",
                    padx=(0, 0),
                    pady=(0, 0),
                )
            gui._motion_content_min_height = (
                300
                + _MOTION_TOP_ROW_MIN_H
                + (180 if metrics_on_motion else 0)
                + (CONTACT_GAIT_CHART_MIN_H if contact_on_motion else 0)
                + 2 * PAD_XS
            )
        else:
            # Joint-focused Motion: knee | path at 50/50. Contact + temporal KPIs
            # answer stability on Biomechanics when hosted there.
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
            bottom.columnconfigure(2, weight=0, minsize=0, uniform="")
            bottom.rowconfigure(0, weight=1, minsize=_MOTION_TOP_ROW_MIN_H)
            bottom.rowconfigure(
                1,
                weight=2 if contact_on_motion else 0,
                minsize=CONTACT_GAIT_CHART_MIN_H if contact_on_motion else 0,
            )
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
            traj.grid(
                row=0,
                column=1,
                columnspan=1,
                sticky="nsew",
                padx=(0, 0),
                pady=(0, 0),
            )
            if metrics_on_motion and metrics is not None:
                # Legacy fallback: keep a third column if metrics were never moved.
                bottom.columnconfigure(
                    2,
                    weight=1,
                    minsize=_MOTION_METRICS_PANEL_MIN_W,
                    uniform="motion_top",
                )
                metrics.grid(
                    row=0,
                    column=2,
                    columnspan=1,
                    rowspan=1,
                    sticky="nsew",
                    padx=(DASHBOARD_GUTTER, 0),
                    pady=(0, 0),
                )
            if contact_on_motion and contact is not None:
                contact.grid(
                    row=1,
                    column=0,
                    columnspan=2 if not metrics_on_motion else 3,
                    sticky="nsew",
                    padx=(0, 0),
                    pady=(PAD_XS, 0),
                )
            gui._motion_content_min_height = _MOTION_TOP_ROW_MIN_H + (
                CONTACT_GAIT_CHART_MIN_H + PAD_XS if contact_on_motion else PAD_XS
            )
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

    # Keep frame-step and sampling/speed visible — restack instead of hiding.
    for attr, visible in (
        ("_transport_sep1", not compact),
        ("_transport_sep2", not compact),
        ("_transport_frame", True),
        ("_transport_analysis", True),
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

    for attr, visible in (
        ("_lbl_sampling", not narrow),
        ("_lbl_speed", not narrow),
    ):
        widget = getattr(gui, attr, None)
        if widget is None:
            continue
        try:
            if visible:
                if str(widget.winfo_manager() or "") != "pack":
                    widget.pack(side=tk.LEFT, padx=(0, PAD_XS))
            else:
                widget.pack_forget()
        except tk.TclError:
            pass

    try:
        from stablewalk.ui.tk.dashboard_status_bar import _reflow_status_bar

        _reflow_status_bar(gui, width=width)
    except Exception:
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

    timeline = getattr(gui, "_transport_timeline", None)
    row = getattr(gui, "_transport_row", None)
    frame_grp = getattr(gui, "_transport_frame", None)
    analysis = getattr(gui, "_transport_analysis", None)
    if timeline is not None and row is not None:
        try:
            if compact:
                timeline.grid(row=0, column=1, columnspan=5, sticky="ew", pady=0)
                if frame_grp is not None:
                    frame_grp.grid(
                        row=1, column=0, sticky="w", pady=(PAD_XS, 0)
                    )
                if analysis is not None:
                    analysis.grid(
                        row=1,
                        column=1,
                        columnspan=5,
                        sticky="w",
                        pady=(PAD_XS, 0),
                    )
                row.rowconfigure(1, weight=0)
            else:
                if frame_grp is not None:
                    frame_grp.grid(row=0, column=2, sticky="w")
                if analysis is not None:
                    analysis.grid(row=0, column=4, sticky="w")
                timeline.grid(row=0, column=5, sticky="ew", pady=0)
                row.rowconfigure(1, weight=0)
        except tk.TclError:
            pass

    actions = getattr(gui, "_input_bar_actions", None)
    source_row = getattr(gui, "_input_bar_row", None)
    if actions is not None and source_row is not None:
        try:
            if narrow:
                actions.pack_forget()
                actions.pack(in_=source_row.master, fill=tk.X, pady=(PAD_XS, 0))
            else:
                actions.pack_forget()
                actions.pack(in_=source_row, side=tk.LEFT)
        except tk.TclError:
            pass
    combo = getattr(gui, "preset_combo", None)
    if combo is not None:
        try:
            from stablewalk.ui.tk.layout_scale import combobox_char_width

            labels = combo.cget("values") or ()
            if narrow:
                combo.configure(
                    width=combobox_char_width(labels, minimum=16, maximum=24)
                )
            else:
                combo.configure(
                    width=combobox_char_width(labels, minimum=24, maximum=48)
                )
        except tk.TclError:
            pass


def _sync_scroll_bottom_clearance(gui) -> None:
    """Advanced tab scroll only — main dashboard uses fixed notebook tabs."""
    sync = getattr(gui, "_sync_tab_advanced_scroll", None)
    if sync is not None:
        sync()


def _stack_two_column_body(
    body: tk.Misc | None,
    left: tk.Misc | None,
    right: tk.Misc | None,
    *,
    stack: bool,
    left_weight: int = 3,
    right_weight: int = 2,
) -> None:
    """Side-by-side on wide Motion inner pages; stacked when the row would crush."""
    if body is None or left is None or right is None:
        return
    key = "_sw_inner_stacked"
    if getattr(body, key, None) is stack:
        return
    try:
        setattr(body, key, stack)
        if stack:
            body.columnconfigure(0, weight=1)
            body.columnconfigure(1, weight=0)
            body.rowconfigure(0, weight=3, minsize=220)
            body.rowconfigure(1, weight=2, minsize=140)
            left.grid_configure(row=0, column=0, columnspan=2, sticky="nsew")
            right.grid_configure(row=1, column=0, columnspan=2, sticky="nsew")
        else:
            body.columnconfigure(0, weight=int(left_weight), minsize=420)
            body.columnconfigure(1, weight=int(right_weight), minsize=240)
            body.rowconfigure(0, weight=1, minsize=0)
            body.rowconfigure(1, weight=0, minsize=0)
            left.grid_configure(row=0, column=0, columnspan=1, sticky="nsew")
            right.grid_configure(row=0, column=1, columnspan=1, sticky="nsew")
    except tk.TclError:
        pass


def _reflow_motion_inner_figures(gui, *, width: int) -> None:
    """Resize Spectral / Features / Asymmetry canvases without recomputing analysis."""
    try:
        from stablewalk.ui.tk.dashboard_layout import _fit_figure_to_host
    except Exception:
        return
    # Spectral Analysis, Spectral Features, and Asymmetry use PanedWindow
    # splits. Re-gridding those hosts would steal them from the sash.
    place_features = getattr(gui, "_place_spectral_features_sash", None)
    if callable(place_features):
        try:
            gui._spectral_features_sash_for_width = None
            place_features()
        except Exception:
            pass
    place_asym = getattr(gui, "_place_asymmetry_sash", None)
    if callable(place_asym):
        try:
            gui._asymmetry_sash_for_width = None
            place_asym()
        except Exception:
            pass
    relayout_spectral = getattr(gui, "_relayout_spectral_plots", None)
    if callable(relayout_spectral):
        try:
            relayout_spectral()
        except Exception:
            pass
    relayout_four = getattr(gui, "_relayout_four_gait_plots", None)
    if callable(relayout_four):
        try:
            relayout_four()
        except Exception:
            pass
    pairs = (
        ("canvas_spectral_features", "fig_spectral_features", "_spectral_features_graph_host"),
        ("canvas_asymmetry", "fig_asymmetry", "_asymmetry_graph_host"),
        ("canvas_four_gait", "fig_four_gait", "_four_gait_graph_host"),
    )
    for canvas_attr, fig_attr, host_attr in pairs:
        canvas = getattr(gui, canvas_attr, None)
        fig = getattr(gui, fig_attr, None)
        if canvas is None or fig is None:
            continue
        host = getattr(gui, host_attr, None) if host_attr else None
        try:
            widget = canvas.get_tk_widget()
            if not widget.winfo_ismapped():
                continue
            _fit_figure_to_host(canvas, fig, host=host or widget.master, min_px=80)
            # Spectral 3×2 and Asymmetry 2×2 set their own subplots_adjust;
            # tight_layout collides titles into neighboring axes.
        except Exception:
            continue
    warn = getattr(gui, "lbl_spectral_warnings", None)
    if warn is not None:
        try:
            warn.configure(wraplength=max(280, int(width) - 48))
        except tk.TclError:
            pass
    feat_def = getattr(gui, "lbl_spectral_feature_def", None)
    if feat_def is not None:
        try:
            feat_def.configure(wraplength=max(180, min(420, int(width) // 3)))
        except tk.TclError:
            pass
    for attr in (
        "lbl_spectral_features_meta",
        "lbl_asymmetry_meta",
        "lbl_gait_corr_note",
    ):
        lbl = getattr(gui, attr, None)
        if lbl is None:
            continue
        try:
            lbl.configure(wraplength=max(280, int(width) - 48))
        except tk.TclError:
            pass


def apply_responsive_layout(gui, *, width: int | None = None, height: int | None = None) -> None:
    """Recompute layout for the current root window size."""
    from stablewalk.ui.tk.layout_playback import is_playing, mark_layout_dirty

    root = gui.root
    width = width if width is not None else root.winfo_width()
    height = height if height is not None else root.winfo_height()
    if width < 2 or height < 2:
        return

    # While playing or mid-resize: transport chrome only — no tab-wide figure fits.
    if is_playing(gui) or bool(getattr(gui, "_ui_resize_active", False)):
        mark_layout_dirty(gui)
        _apply_transport_layout(gui, width)
        gui._last_layout_size = (width, height)
        return

    wmode, hmode = classify_layout(width, height)
    prev_w = getattr(gui, "_layout_width_mode", None)
    prev_h = getattr(gui, "_layout_height_mode", None)

    if prev_w != wmode or prev_h != hmode or getattr(gui, "_force_layout_reflow", False):
        _apply_visual_layout(gui, wmode)
        _apply_sidebar_layout(gui, wmode, width)
        gui._force_layout_reflow = False

    _apply_height_layout(gui, hmode, height, width)
    try:
        from stablewalk.ui.tk.dashboard_notebook import sync_main_tab_geometry

        sync_main_tab_geometry(gui, width=width, height=height)
    except Exception:
        pass
    _reflow_metric_summary(gui, width)
    _apply_graph_column_minsizes(gui, width)
    _apply_transport_layout(gui, width)
    try:
        from stablewalk.ui.tk.dashboard_advanced_workspace import reflow_advanced_workspace

        reflow_advanced_workspace(gui, width=width)
        sync_engineering = getattr(gui, "_sync_engineering_dashboard_wrap", None)
        if callable(sync_engineering):
            sync_engineering()
    except Exception:
        pass

    _sync_scroll_bottom_clearance(gui)

    try:
        reflow_motion = getattr(gui, "_reflow_motion_workspace", None)
        if callable(reflow_motion):
            reflow_motion()
    except Exception:
        pass
    try:
        sync_graph = getattr(gui, "_sync_motion_graph_height_share", None)
        if callable(sync_graph):
            gui._motion_graph_row_minsize = None
            sync_graph()
    except Exception:
        pass
    _reflow_motion_inner_figures(gui, width=width)

    try:
        from stablewalk.ui.tk.dashboard_biomechanics import reflow_biomech_workspace

        reflow_biomech_workspace(gui, width=width, height=height)
    except Exception:
        pass

    try:
        from stablewalk.ui.tk.dashboard_summary import reflow_results_layout

        reflow_results_layout(gui, width=width)
    except Exception:
        pass

    controller = getattr(gui, "_comparison_mode", None)
    reflow_compare = getattr(controller, "_reflow_side_by_side_workspace", None)
    if callable(reflow_compare):
        try:
            reflow_compare()
        except Exception:
            pass
    try:
        from stablewalk.ui.tk.dashboard_research import reflow_research_workspace

        reflow_research_workspace(gui, width=width)
    except Exception:
        pass

    # Overview KPI wrap / summary stack / viz stack — every resize, not only
    # when WidthMode changes (1480 vs 1366 stay in the same mode bucket).
    try:
        from stablewalk.ui.tk.overview_responsive import reflow_overview_layout

        reflow_overview_layout(gui, width=width, height=height)
    except Exception:
        pass

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
            from stablewalk.ui.tk.resize_coalesce import size_change_is_noise

            if size_change_is_noise(last[0], last[1], w, h, threshold=RESIZE_IGNORE_PX):
                return
            from stablewalk.ui.tk.redraw_policy import set_resize_active

            set_resize_active(gui, True)

        if gui._responsive_after_id is not None:
            try:
                root.after_cancel(gui._responsive_after_id)
            except tk.TclError:
                pass

        def _run() -> None:
            gui._responsive_after_id = None
            from stablewalk.ui.tk.redraw_policy import set_resize_active

            set_resize_active(gui, False)
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
