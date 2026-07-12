"""
Fixed-viewport dashboard shell using ttk.Notebook tabs.

Primary analysis content lives in tab pages — not one tall scrolling page.
Only the Advanced & Export tab may scroll vertically.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.theme import BG, PAD_XS


TAB_OVERVIEW = "Overview"
TAB_MOTION = "Motion Analysis"
TAB_BIOMECHANICS = "Biomechanics"
TAB_ADVANCED = "Advanced & Export"


def install_dashboard_notebook(gui, parent: tk.Misc) -> tuple[ttk.Frame, ttk.Frame, ttk.Frame, ttk.Frame]:
    """
    Install a four-tab notebook as the main dashboard viewport.

    Returns ``(tab_overview, tab_motion, tab_biomechanics, tab_advanced_content)``.
    Widgets are created once per tab; switching tabs never recreates children.
    """
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)

    notebook = ttk.Notebook(parent)
    notebook.grid(row=0, column=0, sticky="nsew")
    gui._dashboard_notebook = notebook

    tab_overview = ttk.Frame(notebook, padding=(PAD_XS, PAD_XS))
    tab_motion = ttk.Frame(notebook, padding=(PAD_XS, PAD_XS))
    tab_biomechanics = ttk.Frame(notebook, padding=(PAD_XS, PAD_XS))
    tab_advanced_outer = ttk.Frame(notebook, padding=0)

    notebook.add(tab_overview, text=f"  {TAB_OVERVIEW}  ")
    notebook.add(tab_motion, text=f"  {TAB_MOTION}  ")
    notebook.add(tab_biomechanics, text=f"  {TAB_BIOMECHANICS}  ")
    notebook.add(tab_advanced_outer, text=f"  {TAB_ADVANCED}  ")

    tab_overview.columnconfigure(0, weight=1)
    tab_overview.rowconfigure(0, weight=1)
    tab_overview.rowconfigure(1, weight=0)

    tab_motion.columnconfigure(0, weight=1)
    tab_motion.rowconfigure(0, weight=1)
    tab_motion.rowconfigure(1, weight=0)

    tab_biomechanics.columnconfigure(0, weight=1)
    tab_biomechanics.rowconfigure(0, weight=1)

    tab_advanced_content = _install_tab_scroll(gui, tab_advanced_outer)

    gui._tab_overview = tab_overview
    gui._tab_motion = tab_motion
    gui._tab_biomechanics = tab_biomechanics
    gui._tab_advanced_outer = tab_advanced_outer
    gui._tab_advanced_content = tab_advanced_content

    # Legacy aliases — main dashboard no longer scrolls as one page.
    gui._dash_scroll_outer = None
    gui._dash_scroll_canvas = gui._tab_advanced_scroll_canvas
    gui._dash_scrollbar = gui._tab_advanced_scrollbar
    gui._dash_scroll_inner = gui._tab_advanced_scroll_inner
    gui._dash_scroll_content = tab_overview
    gui._dash_scroll_sections_host = tab_overview
    gui._dash_scroll_window_id = gui._tab_advanced_scroll_window_id
    gui._dash_scroll_bottom_spacer = None
    gui._dash_scroll_bottom_pad = 0

    gui._analysis_scroll_outer = None
    gui._analysis_scroll_canvas = None
    gui._analysis_scrollbar = None
    gui._analysis_scroll_inner = None
    gui._analysis_scroll_window_id = None

    def _sync_main(_event: object | None = None) -> None:
        sync = getattr(gui, "_sync_tab_advanced_scroll", None)
        if sync is not None:
            sync()

    gui._sync_dashboard_scroll = _sync_main
    gui._sync_analysis_scroll = _sync_main

    def _on_tab_changed(_event: object | None = None) -> None:
        sync_mount = getattr(gui, "_sync_trajectory_mount_for_active_tab", None)
        if sync_mount is not None:
            try:
                sync_mount()
            except Exception:
                pass
        _schedule_tab_reflow(gui)
        try:
            selected = notebook.select()
            tab = notebook.nametowidget(selected)
            if tab is getattr(gui, "_tab_motion", None):
                from stablewalk.ui.tk.dashboard_shell import print_motion_widget_hierarchy

                print_motion_widget_hierarchy(gui)
                activate_motion_tab_trajectory(gui)
                gui.root.after_idle(lambda: _log_motion_tab_geometry(gui))
        except tk.TclError:
            pass

    notebook.bind("<<NotebookTabChanged>>", _on_tab_changed, add="+")
    gui._on_dashboard_tab_changed = _on_tab_changed

    return tab_overview, tab_motion, tab_biomechanics, tab_advanced_content


def is_advanced_tab_selected(gui) -> bool:
    """True when the Advanced & Export notebook tab is active."""
    notebook = getattr(gui, "_dashboard_notebook", None)
    if notebook is None:
        return False
    try:
        selected = notebook.select()
        tab = notebook.nametowidget(selected)
        return tab is getattr(gui, "_tab_advanced_outer", None)
    except tk.TclError:
        return False


def is_motion_tab_selected(gui) -> bool:
    """True when the Motion Analysis notebook tab is active."""
    notebook = getattr(gui, "_dashboard_notebook", None)
    if notebook is None:
        return False
    try:
        selected = notebook.select()
        tab = notebook.nametowidget(selected)
        return tab is getattr(gui, "_tab_motion", None)
    except tk.TclError:
        return False


def is_overview_tab_selected(gui) -> bool:
    """True when the Overview notebook tab is active."""
    notebook = getattr(gui, "_dashboard_notebook", None)
    if notebook is None:
        return False
    try:
        selected = notebook.select()
        tab = notebook.nametowidget(selected)
        return tab is getattr(gui, "_tab_overview", None)
    except tk.TclError:
        return False


def is_trajectory_graph_visible(gui) -> bool:
    """True when the joint trajectory canvas should paint immediately."""
    if is_motion_tab_selected(gui):
        return True
    if is_overview_tab_selected(gui) and getattr(gui, "_overview_traj_dock_visible", False):
        return True
    return False


def activate_motion_tab_trajectory(gui) -> None:
    """Fit and force-paint the trajectory canvas when Motion Analysis is shown."""
    from stablewalk.ui.tk.dashboard_layout import (
        _ensure_trajectory_canvas_gridded,
        _hide_trajectory_debug_placeholder,
    )

    if not hasattr(gui, "canvas_dof_traj"):
        return
    sync_panel = getattr(gui, "_sync_dof_analysis_panel_state", None)
    if sync_panel is not None:
        sync_panel()
    fit = getattr(gui, "_fit_dof_traj_canvas", None)
    if fit is not None:
        fit()
    _ensure_trajectory_canvas_gridded(gui.canvas_dof_traj)
    lift = getattr(gui, "_lift_trajectory_canvas", None)
    if lift is not None:
        lift()
    has_session = bool(
        getattr(gui, "skeleton_player", None)
        and getattr(gui.skeleton_player, "frame_count", 0) > 0
    )
    refresh = getattr(gui, "_refresh_selected_dof_trajectory_3d", None)
    if refresh is not None:
        if has_session:
            _hide_trajectory_debug_placeholder(gui)
            refresh(force_draw=True)
        elif not getattr(gui, "_traj_startup_test_drawn", False):
            from stablewalk.ui.tk.dashboard_layout import _draw_trajectory_startup_test

            _draw_trajectory_startup_test(gui)
    render = getattr(gui, "_render_dof_traj_canvas", None)
    if render is not None:
        render(force=True)


def _install_tab_scroll(gui, parent: tk.Misc) -> ttk.Frame:
    """Vertical scroll shell used only inside the Advanced & Export tab."""
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
    canvas.configure(yscrollcommand=vsb.set, yscrollincrement=1)

    scroll_content = ttk.Frame(canvas)
    inner_id = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
    scroll_content.columnconfigure(0, weight=1)

    sections_host = ttk.Frame(scroll_content)
    sections_host.grid(row=0, column=0, sticky="ew")
    sections_host.columnconfigure(0, weight=1)

    def _sync(_event: object | None = None) -> None:
        try:
            canvas.update_idletasks()
            cw = max(canvas.winfo_width(), 1)
            canvas.coords(inner_id, 0, 0)
            canvas.itemconfigure(inner_id, width=cw)
            bbox = canvas.bbox("all")
            if bbox is not None:
                canvas.configure(scrollregion=bbox)
        except tk.TclError:
            pass

    def _on_scroll(*_args: object) -> None:
        try:
            canvas.update_idletasks()
        except tk.TclError:
            pass

    def _scroll_cmd(*args: object) -> None:
        canvas.yview(*args)
        _on_scroll()

    vsb.configure(command=_scroll_cmd)

    scroll_content.bind("<Configure>", _sync, add="+")
    canvas.bind("<Configure>", _sync, add="+")

    def _wheel(event: tk.Event) -> str | None:
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            _on_scroll()
            return "break"
        return None

    for widget in (canvas, scroll_content, sections_host):
        widget.bind("<MouseWheel>", _wheel, add="+")

    gui._tab_advanced_scroll_outer = outer
    gui._tab_advanced_scroll_canvas = canvas
    gui._tab_advanced_scrollbar = vsb
    gui._tab_advanced_scroll_inner = scroll_content
    gui._tab_advanced_scroll_window_id = inner_id
    gui._sync_tab_advanced_scroll = _sync

    return sections_host


def _schedule_tab_reflow(gui) -> None:
    after_id = getattr(gui, "_tab_reflow_after", None)
    if after_id is not None:
        try:
            gui.root.after_cancel(after_id)
        except tk.TclError:
            pass

    def _run() -> None:
        gui._tab_reflow_after = None
        reflow_tab_canvases(gui)
        sync = getattr(gui, "_sync_tab_advanced_scroll", None)
        if sync is not None:
            sync()

    gui._tab_reflow_after = gui.root.after_idle(_run)


def reflow_tab_canvases(gui) -> None:
    """Reflow matplotlib/video layout after a tab switch (no widget recreation)."""
    from stablewalk.ui.tk.clip_viewport import sync_clipped_viewport

    refit_video = getattr(gui, "_on_video_label_resize", None)
    if refit_video is not None:
        refit_video()
    fit_skel = getattr(gui, "_fit_skeleton_canvas", None)
    if fit_skel is not None:
        fit_skel()
    if hasattr(gui, "_update_chart"):
        try:
            gui._update_chart()
        except Exception:
            pass
    sync_clipped_viewport(
        getattr(gui, "_knee_clip_canvas", None),
        getattr(gui, "_knee_clip_window_id", None),
    )
    sync_clipped_viewport(
        getattr(gui, "_traj_clip_canvas", None),
        getattr(gui, "_traj_clip_window_id", None),
    )
    fit_traj = getattr(gui, "_fit_dof_traj_canvas", None)
    if fit_traj is not None:
        fit_traj()
    if not getattr(gui, "_traj_startup_test_drawn", False):
        has_session = bool(
            getattr(gui, "skeleton_player", None)
            and getattr(gui.skeleton_player, "frame_count", 0) > 0
        )
        if not has_session:
            from stablewalk.ui.tk.dashboard_layout import _draw_trajectory_startup_test

            try:
                _draw_trajectory_startup_test(gui)
            except Exception:
                pass
    refresh_traj = getattr(gui, "_refresh_selected_dof_trajectory_3d", None)
    has_session = bool(
        getattr(gui, "skeleton_player", None)
        and getattr(gui.skeleton_player, "frame_count", 0) > 0
    )
    if refresh_traj is not None and (
        has_session or (getattr(gui, "selection", None) and gui.selection.selected)
    ):
        try:
            refresh_traj(force_draw=True)
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Failed to refresh Motion Analysis trajectory graph"
            )
    render = getattr(gui, "_render_dof_traj_canvas", None)
    if render is not None and is_motion_tab_selected(gui):
        try:
            render(force=True)
        except Exception:
            pass
    sync_wrap = getattr(gui, "_sync_utility_sidebar_wrap", None)
    if sync_wrap is not None:
        sync_wrap()


def select_dashboard_tab(gui, tab: str) -> None:
    """Programmatically select a dashboard tab by label."""
    notebook = getattr(gui, "_dashboard_notebook", None)
    if notebook is None:
        return
    labels = {TAB_OVERVIEW: 0, TAB_MOTION: 1, TAB_BIOMECHANICS: 2, TAB_ADVANCED: 3}
    index = labels.get(tab)
    if index is None:
        return
    try:
        notebook.select(index)
        gui.root.update_idletasks()
        reflow_tab_canvases(gui)
        sync_mount = getattr(gui, "_sync_trajectory_mount_for_active_tab", None)
        if sync_mount is not None:
            try:
                sync_mount()
            except Exception:
                pass
        if tab == TAB_MOTION:
            activate_motion_tab_trajectory(gui)
            _log_motion_tab_geometry(gui)
    except tk.TclError:
        pass


def _log_motion_tab_geometry(gui) -> None:
    """Print Motion Analysis layout diagnostics when STABLEWALK_MOTION_LAYOUT_DEBUG is set."""
    import os

    flag = os.environ.get("STABLEWALK_MOTION_LAYOUT_DEBUG", "").strip().lower()
    if flag not in ("1", "true", "yes"):
        return
    from stablewalk.ui.tk.gui_visual_qa import capture_motion_tab_geometry

    snap = capture_motion_tab_geometry(gui)
    print(
        "Motion Analysis layout:",
        f"tab={snap.motion_tab_w}x{snap.motion_tab_h}",
        f"knee={snap.knee_panel_w}x{snap.knee_panel_h}",
        f"traj_panel={snap.traj_panel_w}x{snap.traj_panel_h}",
        f"graph_frame={snap.graph_frame_w}x{snap.graph_frame_h}",
        f"canvas={snap.canvas_w}x{snap.canvas_h}",
        f"traj_width={snap.traj_width_fraction:.1%}",
        sep=" | ",
    )
    if snap.issues:
        print("Motion Analysis layout issues:", "; ".join(snap.issues))


def run_tab_switch_stress_test(gui, *, cycles: int = 50) -> list[tuple[str, bool, str]]:
    """
    Switch tabs repeatedly during a live session and assert singleton widgets.

    Returns list of ``(check_name, passed, detail)`` tuples.
    """
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    from stablewalk.ui.tk.dashboard_shell import assert_dashboard_widget_singletons

    tabs = (TAB_OVERVIEW, TAB_MOTION, TAB_BIOMECHANICS, TAB_ADVANCED, TAB_MOTION, TAB_OVERVIEW)
    for i in range(cycles):
        select_dashboard_tab(gui, tabs[i % len(tabs)])
        try:
            gui.root.update_idletasks()
        except Exception:
            pass

    results: list[tuple[str, bool, str]] = []

    def _count_canvas(attr: str) -> int:
        canvas = getattr(gui, attr, None)
        if canvas is None:
            return 0
        return 1 if isinstance(canvas, FigureCanvasTkAgg) else 0

    video_count = 1 if getattr(gui, "video_label", None) is not None else 0
    results.append(("video_widget_singleton", video_count == 1, f"count={video_count}"))
    for attr, label in (
        ("chart_canvas", "knee"),
        ("canvas_dof_traj", "joint_path"),
        ("canvas_3d", "3d"),
    ):
        count = _count_canvas(attr)
        results.append((f"{label}_canvas_singleton", count == 1, f"count={count}"))

    notebook = getattr(gui, "_dashboard_notebook", None)
    results.append(
        (
            "notebook_present",
            notebook is not None,
            "ok" if notebook is not None else "dashboard notebook missing",
        )
    )
    results.append(
        (
            "main_page_scroll_removed",
            getattr(gui, "_dash_scroll_outer", None) is None,
            "ok" if getattr(gui, "_dash_scroll_outer", None) is None else "legacy scroll active",
        )
    )

    try:
        assert_dashboard_widget_singletons(gui)
        results.append(("dashboard_widget_singletons", True, "all singleton checks passed"))
    except AssertionError as exc:
        results.append(("dashboard_widget_singletons", False, str(exc)))

    from stablewalk.ui.tk.render_diagnostics import run_playback_render_stress_test

    results.extend(run_playback_render_stress_test(gui, frames=50, scroll_during_playback=True))
    results.append(("tab_switch_cycles", cycles >= 50, f"completed {cycles} switches"))
    return results
