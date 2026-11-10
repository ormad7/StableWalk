"""
    Fixed-viewport dashboard shell using ttk.Notebook tabs.

Each tab answers one examiner question (see dashboard_page_frame.PAGE_BRIEFINGS).
Primary analysis content lives in tab pages — not one tall scrolling page.
Only the Advanced tab may scroll vertically.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk

from stablewalk.ui.theme import DASHBOARD_TAB_PAD, PAD_SM, PAD_XS, PANEL

logger = logging.getLogger(__name__)

# Soft floors for expanding content rows (weight=1 still claims leftover height).
TAB_CONTENT_MIN_H = 220
TAB_CHROME_RESERVE_H = 96


TAB_OVERVIEW = "Overview"
TAB_MOTION = "Motion Analysis"
TAB_BIOMECHANICS = "Biomechanics"
TAB_RESULTS_SUMMARY = "Results"
TAB_COMPARE = "Compare"
TAB_ADVANCED = "Advanced"

# Temporarily hide Compare from the notebook. Backend in dashboard_compare.py
# is preserved so the tab can be re-enabled later.
ENABLE_COMPARE_TAB = True

_TAB_WIDGET_ATTRS = {
    TAB_OVERVIEW: "_tab_overview",
    TAB_MOTION: "_tab_motion",
    TAB_BIOMECHANICS: "_tab_biomechanics",
    TAB_RESULTS_SUMMARY: "_tab_results_summary",
    TAB_COMPARE: "_tab_compare",
    TAB_ADVANCED: "_tab_advanced_outer",
}


def dashboard_tab_widget(gui, tab: str) -> tk.Misc | None:
    """Resolve a dashboard tab label to its notebook page widget."""
    attr = _TAB_WIDGET_ATTRS.get(tab)
    if attr is None:
        return None
    return getattr(gui, attr, None)


def visible_dashboard_tab_labels() -> tuple[str, ...]:
    """Labels shown in the main notebook, in order."""
    labels = [
        TAB_OVERVIEW,
        TAB_MOTION,
        TAB_BIOMECHANICS,
        TAB_RESULTS_SUMMARY,
    ]
    if ENABLE_COMPARE_TAB:
        labels.append(TAB_COMPARE)
    labels.append(TAB_ADVANCED)
    return tuple(labels)


def notebook_tab_texts(gui) -> tuple[str, ...]:
    """Current ttk.Notebook tab titles (widget-backed, not hardcoded indices)."""
    notebook = getattr(gui, "_dashboard_notebook", None)
    if notebook is None:
        return ()
    texts: list[str] = []
    try:
        for tab_id in notebook.tabs():
            texts.append(str(notebook.tab(tab_id, "text")))
    except tk.TclError:
        return ()
    return tuple(texts)


def configure_tab_expand_row(tab: tk.Misc, expand_row: int, *, rows: int = 8) -> None:
    """Give *expand_row* all leftover height; keep other rows at their natural size."""
    tab.columnconfigure(0, weight=1)
    for row in range(max(rows, expand_row + 1)):
        try:
            tab.rowconfigure(
                row,
                weight=1 if row == expand_row else 0,
                minsize=0,
            )
        except tk.TclError:
            pass


def apply_tab_stretch_weights(gui) -> None:
    """Central content expands; headers, toolbars, and footers stay compact.

    Re-applied on resize so later reflow cannot leave chrome rows stretching
    or content rows locked at weight 0. Soft minsizes / uniform groups are
    left intact — only weights are set.
    """
    from stablewalk.ui.tk.tab_grid import compact_columns, compact_rows, stretch_columns, stretch_rows

    overview = getattr(gui, "_tab_overview", None)
    if overview is not None:
        configure_tab_expand_row(overview, 0)
        host = getattr(gui, "_overview_layout_host", None) or getattr(
            gui, "_overview_content", None
        )
        if host is not None:
            from stablewalk.ui.tk.dashboard_sections import (
                apply_overview_host_row_weights,
                overview_kpi_row_min_h,
            )

            host_h = 0
            try:
                host_h = int(host.winfo_height() or 0)
            except tk.TclError:
                host_h = 0
            wrap = str(getattr(gui, "_overview_kpi_layout", "") or "") == "2x2"
            apply_overview_host_row_weights(
                host,
                short=bool(getattr(gui, "_overview_video_short", False)),
                kpi_min=overview_kpi_row_min_h(wrap=wrap),
                host_h=host_h if host_h > 80 else None,
            )
        section = getattr(gui, "_section_visual", None) or getattr(
            gui, "_primary_viz_host", None
        )
        if section is not None and not getattr(gui, "_viz_tabs_active", False):
            from stablewalk.ui.tk.dashboard_sections import (
                apply_overview_viz_grid_weights,
                overview_available_viz_h,
                overview_video_row_min_h,
            )

            available_h = overview_available_viz_h(gui, host)
            apply_overview_viz_grid_weights(
                section,
                content_row=int(getattr(gui, "_primary_viz_content_row", 0) or 0),
                layout=str(getattr(gui, "_overview_viz_layout", "three_col") or "three_col"),
                short=bool(getattr(gui, "_overview_short_window", False)),
                video_min_h=overview_video_row_min_h(
                    short=bool(getattr(gui, "_overview_video_short", False))
                ),
                available_h=available_h if available_h > 80 else None,
            )
        for panel, chrome_rows, expand_row in (
            (getattr(gui, "video_frame", None), (0, 1, 2, 4), 3),
            (getattr(gui, "skel_frame", None), (0, 2), 1),
            (getattr(gui, "overview_traj_panel", None), (0, 2, 3), 1),
        ):
            if panel is None:
                continue
            try:
                stretch_columns(panel, 0)
                compact_rows(panel, *chrome_rows)
                stretch_rows(panel, expand_row)
            except tk.TclError:
                pass

    motion = getattr(gui, "_tab_motion", None)
    if motion is not None:
        configure_tab_expand_row(motion, 1)
        workspace = getattr(gui, "_motion_workspace_host", None)
        if workspace is not None:
            stretch_columns(workspace, 0)
            compact_rows(workspace, 0, 2, 3)
            stretch_rows(workspace, 1)
        graphs = getattr(gui, "_motion_graphs_panel", None)
        if graphs is not None:
            from stablewalk.ui.tk.dashboard_motion_workspace import (
                _MOTION_GRAPH_CANVAS_ROW,
            )

            stretch_columns(graphs, 0)
            compact_rows(graphs, 0, 1)
            stretch_rows(graphs, _MOTION_GRAPH_CANVAS_ROW)
        middle = getattr(gui, "_motion_workspace_middle", None)
        if middle is not None:
            stretch_columns(middle, 0)
            stretch_rows(middle, 0)
            compact_rows(middle, 1)
        graph_host = getattr(gui, "motion_graph_host", None)
        if graph_host is not None:
            stretch_columns(graph_host, 0)
            stretch_rows(graph_host, 0)

    biomech = getattr(gui, "_tab_biomechanics", None)
    if biomech is not None:
        configure_tab_expand_row(biomech, 2)
        body = getattr(gui, "_biomech_body", None)
        if body is not None:
            if bool(getattr(gui, "_biomech_stacked", False)):
                stretch_columns(body, 0)
                compact_columns(body, 1, 2)
                compact_rows(body, 0, 2)
                stretch_rows(body, 1)
            else:
                compact_columns(body, 0)
                try:
                    body.columnconfigure(1, weight=4)
                    body.columnconfigure(2, weight=1)
                except tk.TclError:
                    stretch_columns(body, 1)
                    compact_columns(body, 2)
                stretch_rows(body, 0)
                compact_rows(body, 1, 2)
        chart = getattr(gui, "_biomech_chart_panel", None)
        if chart is not None:
            stretch_columns(chart, 0)
            compact_rows(chart, 0, 1, 3)
            stretch_rows(chart, 2)
        clinical = getattr(gui, "_biomech_clinical_host", None)
        if clinical is not None:
            stretch_columns(clinical, 0)
            compact_rows(clinical, 0, 2)
            stretch_rows(clinical, 1)

    results = getattr(gui, "_tab_results_summary", None)
    if results is not None:
        configure_tab_expand_row(results, 1)
        compact_rows(results, 0, 2)
        try:
            # Keep export buttons and status text from being compressed to 0.
            results.rowconfigure(0, minsize=36)
            results.rowconfigure(1, minsize=200)
            results.rowconfigure(2, minsize=28)
        except tk.TclError:
            pass
        content = getattr(gui, "_results_content_row", None)
        if content is not None:
            stretch_columns(content, 0)
            compact_rows(content, 0)
            stretch_rows(content, 1)
            preview_mode = str(getattr(gui, "_results_preview_layout", "") or "")
            if preview_mode == "stacked":
                stretch_rows(content, 2)
            else:
                compact_rows(content, 2)
                if preview_mode == "rail":
                    stretch_columns(content, 1)
                else:
                    compact_columns(content, 1)
        pipeline = getattr(gui, "_results_pipeline_slot", None)
        if pipeline is not None:
            stretch_columns(pipeline, 0)
            stretch_rows(pipeline, 0)
        preview = getattr(gui, "_results_preview_host", None)
        if preview is not None:
            stretch_columns(preview, 0)
            stretch_rows(preview, 0)

    compare = getattr(gui, "_tab_compare", None)
    if compare is not None:
        configure_tab_expand_row(compare, 1)
        host = None
        controller = getattr(gui, "_comparison_mode", None)
        if controller is not None:
            host = getattr(controller, "root", None)
            parent = getattr(host, "master", None) if host is not None else None
            if parent is not None:
                stretch_columns(parent, 0)
                stretch_rows(parent, 0)
            if host is not None:
                stretch_columns(host, 0)
                compact_rows(host, 0, 1, 3, 4, 5)
                stretch_rows(host, 2)
            workspace = getattr(controller, "_compare_workspace", None)
            apply_compare_stretch = getattr(controller, "_apply_compare_stretch_weights", None)
            if callable(apply_compare_stretch):
                apply_compare_stretch()
            elif workspace is not None:
                if bool(getattr(controller, "_compare_stacked", False)):
                    stretch_columns(workspace, 0, 1)
                    compact_columns(workspace, 2)
                    stretch_rows(workspace, 0, 1)
                    compact_rows(workspace, 2, 3)
                else:
                    stretch_columns(workspace, 0, 1, 2)
                    stretch_rows(workspace, 0)
                    compact_rows(workspace, 1, 2, 3)
            transport = getattr(controller, "_transport_host", None)
            if transport is not None:
                stretch_columns(transport, 0)
                compact_rows(transport, 0)
        else:
            try:
                for child in compare.winfo_children():
                    info = child.grid_info() or {}
                    if int(info.get("row", -1) or -1) == 1:
                        stretch_columns(child, 0)
                        stretch_rows(child, 0)
                        break
            except tk.TclError:
                pass

    advanced_outer = getattr(gui, "_tab_advanced_outer", None)
    if advanced_outer is not None:
        configure_tab_expand_row(advanced_outer, 0)
    advanced_content = getattr(gui, "_tab_advanced_content", None)
    if advanced_content is not None:
        stretch_columns(advanced_content, 0)
        compact_rows(advanced_content, 0, 1)
    advanced_workspace = getattr(gui, "_advanced_workspace", None)
    if advanced_workspace is not None:
        compact_rows(advanced_workspace, 0, 1, 2, 3, 4)


def cap_expanding_row_minsize(
    host: tk.Misc,
    row: int,
    *,
    preferred: int,
    available: int,
    chrome: int = TAB_CHROME_RESERVE_H,
    floor: int = TAB_CONTENT_MIN_H,
) -> int:
    """Set a minsize that fits the window instead of pushing sibling rows off-screen."""
    room = max(floor, int(available) - int(chrome))
    minsize = max(floor, min(int(preferred), room))
    try:
        cfg = host.grid_rowconfigure(row)
        weight = int(cfg.get("weight") or 1)
        host.rowconfigure(row, weight=max(1, weight), minsize=minsize)
    except tk.TclError:
        pass
    return minsize


def sync_notebook_tab_fill(gui) -> None:
    """Size the selected notebook page to the notebook client area.

    On Windows, ttk.Notebook leaves the child tab at its requested height.
    Overview then laid out into ~570 px and left a large unused band below.
    """
    notebook = getattr(gui, "_dashboard_notebook", None)
    if notebook is None or getattr(gui, "_notebook_tab_fill_applying", False):
        return
    try:
        selected = notebook.select()
        if not selected:
            return
        tab = notebook.nametowidget(selected)
        notebook.update_idletasks()
        nb_w = int(notebook.winfo_width() or 0)
        nb_h = int(notebook.winfo_height() or 0)
        if nb_w < 80 or nb_h < 80:
            return
        strip = int(tab.winfo_rooty() or 0) - int(notebook.winfo_rooty() or 0)
        inset = int(tab.winfo_rootx() or 0) - int(notebook.winfo_rootx() or 0)
        if strip < 8 or strip > 96:
            strip = 32
        if inset < 0 or inset > 28:
            inset = 4
        client_w = max(80, nb_w - max(2, inset * 2))
        client_h = max(120, nb_h - strip - 4)
        sig = (int(client_w), int(client_h), str(tab))
        if sig == getattr(gui, "_notebook_tab_fill_sig", None):
            return
        gui._notebook_tab_fill_applying = True
        try:
            try:
                tab.grid_propagate(False)
            except tk.TclError:
                pass
            try:
                tab.pack_propagate(False)
            except tk.TclError:
                pass
            tab.configure(width=int(client_w), height=int(client_h))
            gui._notebook_tab_fill_sig = sig
        finally:
            gui._notebook_tab_fill_applying = False
    except (tk.TclError, KeyError):
        return


def sync_main_tab_geometry(gui, *, width: int | None = None, height: int | None = None) -> None:
    """Re-assert tab expand rows and cap content minsizes for the current window."""
    sync_notebook_tab_fill(gui)
    root = getattr(gui, "root", None)
    if height is None and root is not None:
        try:
            height = int(root.winfo_height())
        except tk.TclError:
            height = 0
    height = int(height or 0)

    def _available(widget: tk.Misc | None, fallback: int) -> int:
        if widget is not None:
            try:
                actual = int(widget.winfo_height() or 0)
                if actual > 1:
                    return actual
            except tk.TclError:
                pass
        return max(1, int(fallback))

    tab_h = height
    overview = getattr(gui, "_tab_overview", None)
    if overview is not None:
        try:
            tab_h = max(tab_h, int(overview.winfo_height() or 0))
        except tk.TclError:
            pass
    apply_tab_stretch_weights(gui)
    motion = getattr(gui, "_tab_motion", None)
    if motion is not None:
        host = getattr(gui, "_motion_workspace_host", None)
        if host is not None:
            cap_expanding_row_minsize(
                host,
                1,
                preferred=int(getattr(gui, "_motion_graph_row_minsize", TAB_CONTENT_MIN_H) or TAB_CONTENT_MIN_H),
                available=_available(host, tab_h),
                chrome=0,
                floor=80,
            )
    biomech = getattr(gui, "_tab_biomechanics", None)
    if biomech is not None:
        body = getattr(gui, "_biomech_body", None)
        if body is not None:
            stacked = bool(getattr(gui, "_biomech_stacked", False))
            expand = 1 if stacked else 0
            cap_expanding_row_minsize(
                body,
                expand,
                preferred=280,
                available=_available(body, tab_h),
                chrome=0,
                floor=80,
            )
    compare = getattr(gui, "_tab_compare", None)
    if compare is not None:
        workspace = getattr(gui, "_compare_workspace", None)
        if workspace is None:
            controller = getattr(gui, "_comparison_mode", None)
            workspace = getattr(controller, "_compare_workspace", None) if controller is not None else None
        scroll_host = None
        controller = getattr(gui, "_comparison_mode", None)
        if controller is not None:
            scroll_host = getattr(controller, "_compare_workspace_scroll_host", None)
        cap_target = scroll_host if scroll_host is not None else workspace
        if cap_target is not None:
            cap_expanding_row_minsize(
                cap_target,
                0,
                preferred=220,
                available=_available(cap_target, tab_h),
                chrome=0,
                floor=96,
            )


def install_dashboard_notebook(
    gui, parent: tk.Misc
) -> tuple[ttk.Frame, ttk.Frame, ttk.Frame, ttk.Frame, ttk.Frame | None, ttk.Frame]:
    """
    Install the main dashboard notebook as the viewport.

    Returns
    ``(tab_overview, tab_motion, tab_biomechanics, tab_results_summary, tab_compare, tab_advanced_content)``.
    ``tab_compare`` is ``None`` when ``ENABLE_COMPARE_TAB`` is False.
    Widgets are created once per tab; switching tabs never recreates children.
    """
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)

    notebook = ttk.Notebook(parent)
    notebook.grid(row=0, column=0, sticky="nsew")
    gui._dashboard_notebook = notebook

    # Overview uses a tighter top inset so Video | Skeleton | Path sit higher.
    tab_overview = ttk.Frame(notebook, padding=(PAD_SM, PAD_XS, PAD_SM, PAD_SM))
    tab_motion = ttk.Frame(notebook, padding=DASHBOARD_TAB_PAD)
    tab_biomechanics = ttk.Frame(notebook, padding=DASHBOARD_TAB_PAD)
    tab_results_summary = ttk.Frame(notebook, padding=DASHBOARD_TAB_PAD)
    tab_compare = None
    if ENABLE_COMPARE_TAB:
        tab_compare = ttk.Frame(notebook, padding=DASHBOARD_TAB_PAD)
    # Same outer pad as sibling tabs; Advanced content may still scroll inside.
    tab_advanced_outer = ttk.Frame(notebook, padding=DASHBOARD_TAB_PAD)

    notebook.add(tab_overview, text=TAB_OVERVIEW)
    notebook.add(tab_motion, text=TAB_MOTION)
    notebook.add(tab_biomechanics, text=TAB_BIOMECHANICS)
    notebook.add(tab_results_summary, text=TAB_RESULTS_SUMMARY)
    if tab_compare is not None:
        notebook.add(tab_compare, text=TAB_COMPARE)
    notebook.add(tab_advanced_outer, text=TAB_ADVANCED)

    # One expanding row per tab — extra weighted empty rows create large voids
    # and push chrome (KPIs, interpretation, metrics) below the fold.
    configure_tab_expand_row(tab_overview, 0)  # nested Overview host fills the tab
    configure_tab_expand_row(tab_motion, 1)  # page header | workspace
    configure_tab_expand_row(tab_biomechanics, 2)  # header | banner | body
    configure_tab_expand_row(tab_results_summary, 1)  # header | pipeline | footer
    if tab_compare is not None:
        configure_tab_expand_row(tab_compare, 1)  # header | workspace
    configure_tab_expand_row(tab_advanced_outer, 0)  # scroll shell fills the tab

    tab_advanced_content = _install_tab_scroll(gui, tab_advanced_outer)

    gui._tab_overview = tab_overview
    gui._tab_motion = tab_motion
    gui._tab_biomechanics = tab_biomechanics
    gui._tab_results_summary = tab_results_summary
    gui._tab_compare = tab_compare
    gui._tab_advanced_outer = tab_advanced_outer
    gui._tab_advanced_content = tab_advanced_content

    # Legacy aliases — main dashboard no longer scrolls as one page.
    # Advanced is a direct Frame host (no Canvas.create_window).
    gui._dash_scroll_outer = None
    gui._dash_scroll_canvas = None
    gui._dash_scrollbar = gui._tab_advanced_scrollbar
    gui._dash_scroll_inner = gui._tab_advanced_scroll_inner
    gui._dash_scroll_content = tab_overview
    gui._dash_scroll_sections_host = tab_overview
    gui._dash_scroll_window_id = None
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
        from stablewalk.ui.tk.dashboard_tab_lifecycle import handle_notebook_tab_changed

        applying = bool(getattr(gui, "_applying_programmatic_tab", False))
        # Nested virtual event while the synchronous programmatic call is running:
        # mark it consumed so ``select_dashboard_tab`` does not leave skip=True
        # for the *next* real user click.
        if applying and _event is not None:
            gui._notebook_tab_changed_consumed = True
            return

        # Programmatic ``select_dashboard_tab`` already ran this path once; ignore
        # the queued virtual event only (not the nested synchronous call).
        if getattr(gui, "_skip_next_notebook_tab_changed", False) and not applying:
            gui._skip_next_notebook_tab_changed = False
            # A leftover skip must not drop a real user click (e.g. Advanced).
            try:
                current = notebook.nametowidget(notebook.select())
            except (tk.TclError, KeyError):
                current = None
            from stablewalk.ui.tk.dashboard_tab_lifecycle import resolve_tab_id

            selected_id = resolve_tab_id(gui, current)
            if selected_id != getattr(gui, "_active_dashboard_tab_id", None):
                try:
                    handle_notebook_tab_changed(gui, notebook)
                except Exception:
                    logger.exception("Dashboard tab change handler failed")
                realize_advanced_tab_shell(gui)
            else:
                realize_advanced_tab_shell(gui)
            return

        # Enter/leave first so heavy widgets exist before reflow/paint, and so
        # leaving a tab pauses local timers and releases temporary buffers.
        try:
            handle_notebook_tab_changed(gui, notebook)
        except Exception:
            logger.exception("Dashboard tab change handler failed")
            realize_advanced_tab_shell(gui)
        sync_mount = getattr(gui, "_sync_trajectory_mount_for_active_tab", None)
        if sync_mount is not None:
            try:
                sync_mount()
            except Exception:
                logger.exception("Trajectory mount sync on tab change failed")
        # While playing, skip full matplotlib reflow (expensive set_size_inches).
        # Advanced still needs its Canvas inner-window sized on first map.
        if not getattr(gui, "playing", False):
            _schedule_tab_reflow(gui)
        else:
            realize_advanced_tab_shell(gui)
        # Playback only paints the visible tab, so refresh the destination tab
        # once on switch to guarantee its charts match the current global frame.
        refresh_active = getattr(gui, "_refresh_active_tab_charts", None)
        if refresh_active is not None:
            try:
                refresh_active()
            except Exception:
                logger.exception("Active-tab chart refresh failed")
                try:
                    gui.root.after_idle(refresh_active)
                except tk.TclError:
                    pass
        try:
            selected = notebook.select()
            tab = notebook.nametowidget(selected)
            if tab is getattr(gui, "_tab_motion", None):
                gui.root.after_idle(lambda: _log_motion_tab_geometry(gui))
            # Subtle enter transition — skipped while playing / reduce-motion.
            from stablewalk.ui.tk.ui_motion import mark_viz_reveal, play_tab_transition

            play_tab_transition(gui, tab)
            if tab is getattr(gui, "_tab_motion", None):
                mark_viz_reveal(gui, kind="traj")
                mark_viz_reveal(gui, kind="graph")
            elif tab is getattr(gui, "_tab_overview", None):
                mark_viz_reveal(gui, kind="3d")
                mark_viz_reveal(gui, kind="graph")
            elif tab is getattr(gui, "_tab_biomechanics", None):
                mark_viz_reveal(gui, kind="biomech")
            elif tab is getattr(gui, "_tab_compare", None):
                mark_viz_reveal(gui, kind="compare")
        except tk.TclError:
            pass
        realize_advanced_tab_shell(gui)

    notebook.bind("<<NotebookTabChanged>>", _on_tab_changed, add="+")
    notebook.bind(
        "<Configure>",
        lambda _e: sync_notebook_tab_fill(gui),
        add="+",
    )
    tab_advanced_outer.bind(
        "<Map>",
        lambda _e: realize_advanced_tab_shell(gui),
        add="+",
    )
    gui._on_dashboard_tab_changed = _on_tab_changed
    gui._skip_next_notebook_tab_changed = False
    gui._notebook_tab_changed_consumed = False
    gui._active_dashboard_tab_id = TAB_OVERVIEW

    return (
        tab_overview,
        tab_motion,
        tab_biomechanics,
        tab_results_summary,
        tab_compare,
        tab_advanced_content,
    )


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


def realize_advanced_tab_shell(gui) -> None:
    """Ensure Advanced widgets exist, fill the tab, and are not covered by a veil.

    Advanced is a direct Frame child of the notebook page. A leftover
    PANEL-colored enter-veil still looks like an empty black pane.
    """
    if not is_advanced_tab_selected(gui):
        return
    ensure = getattr(gui, "_ensure_advanced_tab_content", None)
    if callable(ensure):
        try:
            ensure()
        except Exception:
            logger.exception("Advanced tab content failed to load")
    sync = getattr(gui, "_sync_tab_advanced_scroll", None)
    if callable(sync):
        try:
            sync()
        except Exception:
            logger.exception("Advanced scroll geometry sync failed")
    try:
        from stablewalk.ui.tk.ui_motion import dismiss_tab_transition_veil

        dismiss_tab_transition_veil(gui)
    except Exception:
        logger.exception("Advanced tab veil dismiss failed")
    root = getattr(gui, "root", None)
    if root is None:
        return

    def _retry() -> None:
        later = getattr(gui, "_sync_tab_advanced_scroll", None)
        if callable(later):
            try:
                later()
            except Exception:
                logger.exception("Advanced scroll retry sync failed")
        try:
            from stablewalk.ui.tk.ui_motion import dismiss_tab_transition_veil

            dismiss_tab_transition_veil(gui)
        except Exception:
            logger.exception("Advanced tab veil retry dismiss failed")

    try:
        from stablewalk.ui.tk.timer_registry import schedule_unique

        schedule_unique(gui, "_advanced_shell_realize_after", 80, _retry)
    except Exception:
        try:
            root.after(80, _retry)
        except tk.TclError:
            pass


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


def is_biomechanics_tab_selected(gui) -> bool:
    """True when the Biomechanics notebook tab is active."""
    notebook = getattr(gui, "_dashboard_notebook", None)
    if notebook is None:
        return False
    try:
        selected = notebook.select()
        tab = notebook.nametowidget(selected)
        return tab is getattr(gui, "_tab_biomechanics", None)
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
    """Fit and force-paint Motion Analysis when the tab is shown."""
    refresh = getattr(gui, "refresh_motion_analysis_tab", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:
            pass
        root = getattr(gui, "root", None)
        if root is not None:
            def _repaint() -> None:
                try:
                    update = getattr(gui, "_update_joint_motion_analysis_chart", None)
                    if callable(update):
                        update(force_rebuild=True)
                except Exception:
                    pass

            try:
                root.after(120, _repaint)
            except tk.TclError:
                pass
        return
    # Fallback if GUI method is unavailable (tests / partial stubs).
    from stablewalk.ui.tk.dashboard_layout import (
        _ensure_trajectory_canvas_gridded,
        _hide_trajectory_debug_placeholder,
    )

    ensure = getattr(gui, "_ensure_motion_tab_charts", None)
    if callable(ensure):
        try:
            ensure()
        except Exception:
            pass
    if getattr(gui, "canvas_dof_traj", None) is None:
        return
    sync_panel = getattr(gui, "_sync_dof_analysis_panel_state", None)
    if sync_panel is not None:
        sync_panel()
    fit = getattr(gui, "_fit_dof_traj_canvas", None)
    if fit is not None:
        fit()
    sync_motion = getattr(gui, "_sync_motion_scroll", None)
    if sync_motion is not None:
        try:
            sync_motion()
        except Exception:
            pass
    _ensure_trajectory_canvas_gridded(gui.canvas_dof_traj)
    lift = getattr(gui, "_lift_trajectory_canvas", None)
    if lift is not None:
        lift()
    has_session = bool(
        getattr(gui, "skeleton_player", None)
        and getattr(gui.skeleton_player, "frame_count", 0) > 0
    )
    refresh_traj = getattr(gui, "_refresh_selected_dof_trajectory_3d", None)
    if refresh_traj is not None:
        if has_session:
            _hide_trajectory_debug_placeholder(gui)
            refresh_traj(force_draw=True)
        elif not getattr(gui, "_traj_startup_test_drawn", False):
            from stablewalk.ui.tk.dashboard_layout import _draw_trajectory_startup_test

            _draw_trajectory_startup_test(gui)
    render = getattr(gui, "_render_dof_traj_canvas", None)
    if render is not None:
        render(force=True)


def _install_tab_scroll(gui, parent: tk.Misc) -> tk.Frame:
    """Painted Advanced host — direct Frame, no Canvas.create_window.

    Overview and Results already left page-level ``create_window`` because that
    pattern leaves ttk widgets mapped and sized but unpainted on Windows
    (solid PANEL / black empty tab). Overflow uses ``place`` inside the
    notebook page, which still clips and actually draws ttk children.
    """
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)

    outer = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    outer.grid(row=0, column=0, sticky="nsew")
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(0, weight=1)

    viewport = tk.Frame(outer, bg=PANEL, highlightthickness=0)
    viewport.grid(row=0, column=0, sticky="nsew")

    vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL)
    vsb.grid(row=0, column=1, sticky="ns")
    vsb.grid_remove()

    sections_host = tk.Frame(viewport, bg=PANEL, highlightthickness=0)
    sections_host.columnconfigure(0, weight=1)
    sections_host.place(x=0, y=0, relwidth=1)

    state = {"y": 0, "active": False}

    def _content_h() -> int:
        return max(int(sections_host.winfo_reqheight() or 0), 1)

    def _view_h() -> int:
        try:
            return max(int(viewport.winfo_height() or 0), 1)
        except tk.TclError:
            return 1

    def _apply_y(y: float) -> None:
        try:
            view_h = _view_h()
            req_h = _content_h()
            max_y = max(0, req_h - view_h)
            y_i = max(0, min(int(round(y)), max_y))
            state["y"] = y_i
            sections_host.place(x=0, y=-y_i, relwidth=1)
            needs = req_h > view_h + 2
            gui._tab_advanced_needs_scroll = needs
            if needs:
                if not vsb.grid_info():
                    vsb.grid()
                vsb.set(y_i / req_h, (y_i + view_h) / req_h)
            else:
                if vsb.grid_info():
                    vsb.grid_remove()
                vsb.set(0.0, 1.0)
                if y_i:
                    state["y"] = 0
                    sections_host.place(x=0, y=0, relwidth=1)
        except tk.TclError:
            pass

    def _sync(_event: object | None = None) -> None:
        if state["active"]:
            return
        state["active"] = True
        try:
            viewport.update_idletasks()
            _apply_y(state["y"])
        except tk.TclError:
            pass
        finally:
            state["active"] = False

    def _yview(*args: object) -> None:
        view_h = _view_h()
        req_h = _content_h()
        if not args:
            return
        cmd = str(args[0])
        if cmd == "moveto":
            y = float(args[1]) * req_h
        elif cmd == "scroll":
            amount = int(args[1])
            what = str(args[2]) if len(args) > 2 else "units"
            step = view_h if what == "pages" else 40
            y = state["y"] + amount * step
        else:
            return
        _apply_y(y)

    vsb.configure(command=_yview)

    def _wheel(event: tk.Event) -> str | None:
        if not getattr(gui, "_tab_advanced_needs_scroll", False):
            return None
        if event.delta:
            _apply_y(state["y"] - int(event.delta / 120) * 40)
            return "break"
        return None

    for widget in (viewport, sections_host, outer):
        widget.bind("<MouseWheel>", _wheel, add="+")
        widget.bind("<Configure>", _sync, add="+")

    gui._tab_advanced_scroll_outer = outer
    gui._tab_advanced_scroll_canvas = None
    gui._tab_advanced_scroll_viewport = viewport
    gui._tab_advanced_scrollbar = vsb
    gui._tab_advanced_scroll_inner = sections_host
    gui._tab_advanced_scroll_window_id = None
    gui._sync_tab_advanced_scroll = _sync
    gui._advanced_yview = _yview
    gui._tab_advanced_needs_scroll = False

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

    active = None
    notebook = getattr(gui, "_dashboard_notebook", None)
    if notebook is not None:
        try:
            active = notebook.nametowidget(notebook.select())
        except (tk.TclError, KeyError):
            active = None
    overview_active = active is getattr(gui, "_tab_overview", None)
    motion_active = active is getattr(gui, "_tab_motion", None)
    biomechanics_active = active is getattr(gui, "_tab_biomechanics", None)
    trajectory_active = motion_active or (
        overview_active and getattr(gui, "_overview_traj_dock_visible", False)
    )

    if overview_active:
        # Debounced media refit — avoid stacking immediate video + skeleton fits.
        schedule = getattr(gui, "_schedule_overview_media_refit", None)
        if callable(schedule):
            try:
                schedule(delay_ms=80)
            except Exception:
                refit_video = getattr(gui, "_on_video_label_resize", None)
                if refit_video is not None:
                    refit_video()
                fit_skel = getattr(gui, "_fit_skeleton_canvas", None)
                if fit_skel is not None:
                    fit_skel()
        else:
            refit_video = getattr(gui, "_on_video_label_resize", None)
            if refit_video is not None:
                refit_video()
            fit_skel = getattr(gui, "_fit_skeleton_canvas", None)
            if fit_skel is not None:
                fit_skel()
    if motion_active:
        ensure_motion = getattr(gui, "_ensure_motion_tab_charts", None)
        if callable(ensure_motion):
            try:
                ensure_motion()
            except Exception:
                pass
        if hasattr(gui, "_update_chart"):
            try:
                gui._update_chart()
            except Exception:
                pass
        sync_clipped_viewport(
            getattr(gui, "_knee_clip_canvas", None),
            getattr(gui, "_knee_clip_window_id", None),
        )
        # Reuse existing contact/vGRF artists. The chart update already detects
        # an unrealized/zero-size first paint and rebuilds only in that case.
        try:
            update_contact = getattr(gui, "_update_contact_gait_chart", None)
            if callable(update_contact):
                update_contact()
        except Exception:
            pass
    if trajectory_active:
        if motion_active:
            ensure_traj = getattr(gui, "_ensure_motion_traj_chart", None)
            if callable(ensure_traj):
                try:
                    ensure_traj()
                except Exception:
                    pass
        sync_clipped_viewport(
            getattr(gui, "_traj_clip_canvas", None),
            getattr(gui, "_traj_clip_window_id", None),
        )
        fit_traj = getattr(gui, "_fit_dof_traj_canvas", None)
        if fit_traj is not None:
            fit_traj()
    if motion_active:
        fit_contact = getattr(gui, "_fit_contact_gait_canvas", None)
        if fit_contact is not None:
            try:
                fit_contact()
            except Exception:
                pass
        sync_motion = getattr(gui, "_sync_motion_scroll", None)
        if sync_motion is not None:
            try:
                sync_motion()
            except Exception:
                pass
    if biomechanics_active:
        fit_attr = (
            "_fit_contact_gait_canvas"
            if getattr(gui, "_biomech_chart_family", "biomech") == "contact"
            else "_fit_biomech_canvas"
        )
        fit_biomech = getattr(gui, fit_attr, None)
        if fit_biomech is not None:
            try:
                fit_biomech()
            except Exception:
                pass
    if motion_active and not getattr(gui, "_traj_startup_test_drawn", False):
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
    if trajectory_active and refresh_traj is not None and (
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
    if render is not None and motion_active:
        try:
            render(force=True)
        except Exception:
            pass
    sync_wrap = getattr(gui, "_sync_utility_sidebar_wrap", None)
    if sync_wrap is not None:
        sync_wrap()


def select_dashboard_tab(gui, tab: str) -> None:
    """Programmatically select a dashboard tab by widget reference, not index."""
    notebook = getattr(gui, "_dashboard_notebook", None)
    if notebook is None:
        return
    page = dashboard_tab_widget(gui, tab)
    if page is None:
        return
    try:
        try:
            previous = notebook.nametowidget(notebook.select())
        except (tk.TclError, KeyError):
            previous = None
        notebook.select(page)
        # ``notebook.select`` queues ``<<NotebookTabChanged>>``; run lifecycle
        # synchronously so callers do not depend on ``root.update()``, then
        # ignore the queued virtual event to avoid a double activate/refresh.
        if previous is not page:
            gui._applying_programmatic_tab = True
            gui._notebook_tab_changed_consumed = False
            try:
                on_changed = getattr(gui, "_on_dashboard_tab_changed", None)
                if callable(on_changed):
                    try:
                        on_changed()
                    except Exception:
                        logger.exception("Programmatic tab change handler failed")
                else:
                    try:
                        from stablewalk.ui.tk.dashboard_tab_lifecycle import (
                            handle_notebook_tab_changed,
                        )

                        handle_notebook_tab_changed(gui, notebook)
                    except Exception:
                        logger.exception("Programmatic tab lifecycle failed")
            finally:
                gui._applying_programmatic_tab = False
                # Only swallow the queued virtual event if it has not already
                # been delivered nested inside the synchronous call. A leftover
                # skip=True would drop the next real user click (Advanced).
                if not getattr(gui, "_notebook_tab_changed_consumed", False):
                    gui._skip_next_notebook_tab_changed = True
                else:
                    gui._skip_next_notebook_tab_changed = False
                gui._notebook_tab_changed_consumed = False
            realize_advanced_tab_shell(gui)
        # Avoid forcing a full geometry pass on every Compare open — that stacks
        # with activate_tab refresh and can freeze the UI thread.
        if previous is page:
            try:
                gui.root.update_idletasks()
            except tk.TclError:
                pass
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
        # Compare: activate_tab / on_activate_compare_tab already refresh once.
        # Do not call _refresh_comparison_mode here (was a synchronous double-hit).
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

    tabs = (
        TAB_OVERVIEW,
        TAB_MOTION,
        TAB_BIOMECHANICS,
        TAB_RESULTS_SUMMARY,
        *( (TAB_COMPARE,) if ENABLE_COMPARE_TAB else () ),
        TAB_ADVANCED,
        TAB_MOTION,
        TAB_OVERVIEW,
    )
    player = getattr(gui, "skeleton_player", None)
    baseline_frame = (
        float(player.state.frame_float) if player is not None else None
    )
    baseline_playing = bool(getattr(gui, "playing", False))
    selection = getattr(gui, "selection", None)
    baseline_selected = frozenset(getattr(selection, "selected", ()) or ())
    baseline_active = getattr(selection, "active_item_id", None)
    for i in range(cycles):
        select_dashboard_tab(gui, tabs[i % len(tabs)])
        try:
            gui.root.update_idletasks()
        except Exception:
            pass

    results: list[tuple[str, bool, str]] = []
    current_frame = float(player.state.frame_float) if player is not None else None
    results.append(
        (
            "tab_switch_preserves_frame",
            (
                current_frame == baseline_frame
                if not baseline_playing
                else current_frame is not None
            ),
            f"before={baseline_frame}, after={current_frame}",
        )
    )
    results.append(
        (
            "tab_switch_preserves_play_state",
            bool(getattr(gui, "playing", False)) == baseline_playing,
            f"before={baseline_playing}, after={bool(getattr(gui, 'playing', False))}",
        )
    )
    current_selected = frozenset(getattr(selection, "selected", ()) or ())
    current_active = getattr(selection, "active_item_id", None)
    results.append(
        (
            "tab_switch_preserves_selection",
            current_selected == baseline_selected and current_active == baseline_active,
            f"selected={len(current_selected)}, active={current_active}",
        )
    )

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
