"""Tab enter/leave lifecycle — lazy build, pause work, release temp buffers.

Does not change analysis algorithms or user-facing features. Heavy widgets are
created on first visit; leaving a tab pauses local timers/animations and drops
temporary image buffers while keeping session state for a fast restore.
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Any

from stablewalk.ui.tk.dashboard_notebook import (
    TAB_ADVANCED,
    TAB_BIOMECHANICS,
    TAB_COMPARE,
    TAB_MOTION,
    TAB_OVERVIEW,
    TAB_RESULTS_SUMMARY,
)

logger = logging.getLogger(__name__)

_TAB_ATTRS = (
    ("_tab_overview", TAB_OVERVIEW),
    ("_tab_motion", TAB_MOTION),
    ("_tab_biomechanics", TAB_BIOMECHANICS),
    ("_tab_results_summary", TAB_RESULTS_SUMMARY),
    ("_tab_compare", TAB_COMPARE),
    ("_tab_advanced_outer", TAB_ADVANCED),
)


def resolve_tab_id(gui: Any, tab_widget: tk.Misc | None) -> str | None:
    if tab_widget is None:
        return None
    for attr, name in _TAB_ATTRS:
        if tab_widget is getattr(gui, attr, None):
            return name
    # Advanced scroll content is nested inside the outer notebook page.
    if tab_widget is getattr(gui, "_tab_advanced_content", None):
        return TAB_ADVANCED
    return None


def active_tab_id(gui: Any) -> str | None:
    notebook = getattr(gui, "_dashboard_notebook", None)
    if notebook is None:
        return None
    try:
        return resolve_tab_id(gui, notebook.nametowidget(notebook.select()))
    except (tk.TclError, KeyError):
        return None


def cancel_after_attr(gui: Any, attr: str) -> None:
    job = getattr(gui, attr, None)
    if job is None:
        return
    try:
        gui.root.after_cancel(job)
    except (tk.TclError, ValueError, AttributeError):
        pass
    setattr(gui, attr, None)


def cancel_pipeline_animations(gui: Any) -> None:
    """Stop all Real-to-Sim completion / connector pulse timers."""
    from stablewalk.ui.tk.dashboard_pipeline_visual import cancel_all_pipeline_animations

    cancel_all_pipeline_animations(gui)


def cancel_resize_jobs(gui: Any) -> None:
    # Keep in sync with timer_registry.ANCILLARY_AFTER_ATTRS.
    from stablewalk.ui.tk.timer_registry import cancel_ancillary_after_jobs

    cancel_ancillary_after_jobs(gui)


def release_compare_temp_buffers(gui: Any) -> None:
    controller = getattr(gui, "_comparison_mode", None)
    if controller is None:
        return
    release = getattr(controller, "release_temp_buffers", None)
    if callable(release):
        try:
            release()
        except Exception:
            logger.debug("compare buffer release failed", exc_info=True)


def deactivate_tab(gui: Any, tab_id: str | None) -> None:
    """Pause timers / animations and drop temporary buffers for a left tab."""
    if not tab_id:
        return
    try:
        from stablewalk.ui.tk.playback_active_tab import mark_hidden_tab_panels_dirty

        mark_hidden_tab_panels_dirty(gui)
    except Exception:
        logger.debug("hidden-tab dirty mark failed", exc_info=True)
    cancel_resize_jobs(gui)
    try:
        from stablewalk.ui.tk.timer_registry import cancel_after_attr

        cancel_after_attr(gui, "_tab_activate_paint_after")
    except Exception:
        pass

    if tab_id == TAB_COMPARE:
        controller = getattr(gui, "_comparison_mode", None)
        if controller is not None:
            leave = getattr(controller, "on_leave_compare_tab", None)
            if callable(leave):
                try:
                    leave()
                except Exception:
                    logger.debug("compare leave cleanup failed", exc_info=True)
            else:
                pause = getattr(controller, "pause", None)
                if callable(pause) and getattr(controller, "_playing", False):
                    try:
                        pause(paint=False)
                    except TypeError:
                        pause()
                cancel = getattr(controller, "cancel_deferred_work", None)
                if callable(cancel):
                    try:
                        cancel()
                    except Exception:
                        logger.debug("compare deferred cancel failed", exc_info=True)
                release_compare_temp_buffers(gui)

    if tab_id in (TAB_RESULTS_SUMMARY, TAB_ADVANCED):
        cancel_pipeline_animations(gui)

    if tab_id in (TAB_MOTION, TAB_BIOMECHANICS, TAB_COMPARE):
        # Drop Agg blit bitmaps while the tab is hidden (rebuilt on next paint).
        try:
            from stablewalk.ui.viewers.chart_playhead import clear_static_chart_background

            for attr in (
                "fig_knee",
                "fig_biomech",
                "fig_contact_gait",
                "fig_joint_motion",
                "fig_chart",
            ):
                clear_static_chart_background(getattr(gui, attr, None))
            controller = getattr(gui, "_comparison_mode", None)
            if controller is not None:
                for attr in (
                    "fig_chart",
                    "fig_heat",
                    "fig_path",
                    "fig_skel_a",
                    "fig_skel_b",
                    "fig_skel_ov",
                ):
                    clear_static_chart_background(getattr(controller, attr, None))
        except Exception:
            logger.debug("playhead bg clear failed", exc_info=True)

    if tab_id == TAB_OVERVIEW:
        # Suspend Overview-local resize/sash work. Trim (do not clear) the RGB
        # decode cache and keep the last shared frame so return-to-Overview
        # playback can reuse processed frames immediately.
        cancel_after_attr(gui, "_overview_sash_sync_after")
        cancel_after_attr(gui, "_overview_media_refit_after")
        setattr(gui, "_video_photo_key", None)
        setattr(gui, "_overview_hud_frame_i", None)
        rgb_cache = getattr(gui, "_rgb_cache", None)
        trim = getattr(rgb_cache, "trim", None)
        if callable(trim):
            try:
                trim(24)
            except Exception:
                pass
        else:
            clear = getattr(rgb_cache, "clear", None)
            if callable(clear):
                try:
                    clear()
                except Exception:
                    pass
        # Drop Agg blit caches for Overview figures (rebuilt on next paint).
        try:
            from stablewalk.ui.viewers.chart_playhead import clear_static_chart_background

            clear_static_chart_background(getattr(gui, "fig_3d", None))
            clear_static_chart_background(getattr(gui, "fig_dof_traj_overview", None))
        except Exception:
            logger.debug("overview blit clear failed", exc_info=True)


def ensure_tab_content(gui: Any, tab_id: str | None) -> None:
    """Build deferred heavy widgets the first time a tab is opened."""
    if not tab_id:
        return
    if tab_id == TAB_COMPARE:
        # Research dataset is the default Compare page. Building Laboratory
        # ComparisonModeController here freezes the UI on first Compare open.
        if compare_laboratory_page_visible(gui):
            ensure = getattr(gui, "_ensure_comparison_mode_loaded", None)
            if callable(ensure):
                ensure()
    elif tab_id == TAB_BIOMECHANICS:
        family = str(getattr(gui, "_biomech_chart_family", "biomech") or "biomech")
        if family == "contact":
            ensure = getattr(gui, "_ensure_contact_gait_chart", None)
        else:
            ensure = getattr(gui, "_ensure_biomech_chart", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:
                logger.debug("biomech chart ensure failed", exc_info=True)
    elif tab_id == TAB_RESULTS_SUMMARY:
        ensure = getattr(gui, "_ensure_results_tab_content", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:
                logger.debug("results tab ensure failed", exc_info=True)
    elif tab_id == TAB_ADVANCED:
        ensure = getattr(gui, "_ensure_advanced_tab_content", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:
                logger.exception("advanced tab ensure failed")
    elif tab_id == TAB_MOTION:
        ensure = getattr(gui, "_ensure_motion_tab_charts", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:
                logger.debug("motion charts ensure failed", exc_info=True)


def compare_laboratory_page_visible(gui: Any) -> bool:
    """True when Compare's Laboratory inner page (A/B cache) is the selected page."""
    pages = getattr(gui, "_compare_page_notebook", None)
    if pages is None:
        return getattr(gui, "_comparison_mode", None) is not None
    try:
        return str(pages.tab(pages.select(), "text")).strip() == "Laboratory"
    except tk.TclError:
        return False


def _sync_global_transport_for_tab(gui: Any, tab_id: str | None) -> None:
    """Hide the global transport only on Compare Laboratory (it has its own scrubber)."""
    bar = getattr(gui, "_transport_bar", None)
    status = getattr(gui, "_status_bar", None)
    if bar is None:
        return
    hide = tab_id == TAB_COMPARE and compare_laboratory_page_visible(gui)
    try:
        if hide:
            bar.pack_forget()
            return
        if str(bar.winfo_manager() or "") == "pack":
            return
        pack_opts = {"side": tk.BOTTOM, "fill": tk.X, "padx": 8, "pady": (0, 8)}
        if status is not None and str(status.winfo_manager() or "") == "pack":
            bar.pack(after=status, **pack_opts)
        else:
            bar.pack(**pack_opts)
    except tk.TclError:
        pass


def activate_tab(gui: Any, tab_id: str | None) -> None:
    """Ensure content exists, then resume tab-local background work if needed."""
    ensure_tab_content(gui, tab_id)
    _sync_global_transport_for_tab(gui, tab_id)
    if tab_id == TAB_COMPARE:
        controller = getattr(gui, "_comparison_mode", None)
        if controller is not None:
            activate = getattr(controller, "on_activate_compare_tab", None)
            if callable(activate):
                try:
                    activate()
                except Exception as exc:
                    try:
                        from stablewalk.ui.tk.compare_state import log_compare_exception

                        msg = log_compare_exception(gui, "activate_tab", exc)
                        status = getattr(controller, "lbl_status", None)
                        if status is not None:
                            status.configure(text=msg)
                        show_err = getattr(controller, "_show_compare_error", None)
                        if callable(show_err):
                            show_err(msg)
                    except Exception:
                        logger.debug("compare activate failed", exc_info=True)
            else:
                arm = getattr(controller, "_arm_snapshot_poll", None)
                if callable(arm):
                    try:
                        arm()
                    except Exception:
                        logger.debug("compare poll arm failed", exc_info=True)
                if getattr(controller, "_visuals_dirty", False) or not getattr(
                    controller, "_graphs_built", True
                ):
                    request = getattr(controller, "request_refresh", None)
                    if callable(request):
                        try:
                            request()
                        except Exception:
                            logger.debug("compare resume refresh failed", exc_info=True)
    elif tab_id == TAB_RESULTS_SUMMARY:
        # Defer panel refresh so the notebook switch stays responsive.
        def _results_refresh() -> None:
            update = getattr(gui, "_update_analysis_summary_panel", None)
            if callable(update):
                try:
                    update()
                except Exception:
                    logger.debug("results panel refresh failed", exc_info=True)
            refresh_pipe = getattr(gui, "_refresh_pipeline_status_panel", None)
            if callable(refresh_pipe):
                try:
                    refresh_pipe(force=False)
                except Exception:
                    logger.debug("results pipeline refresh failed", exc_info=True)

        from stablewalk.ui.tk.timer_registry import schedule_unique

        schedule_unique(gui, "_tab_activate_paint_after", 16, _results_refresh)
    elif tab_id == TAB_ADVANCED:
        def _advanced_refresh() -> None:
            refresh_pipe = getattr(gui, "_refresh_pipeline_status_panel", None)
            if callable(refresh_pipe):
                try:
                    refresh_pipe(force=False)
                except Exception:
                    logger.debug("advanced pipeline refresh failed", exc_info=True)
            try:
                update_gait = getattr(gui, "_update_gait_cycle_panel", None)
                if callable(update_gait):
                    update_gait(getattr(gui, "_gait_cycle", None))
            except Exception:
                logger.debug("advanced gait cards refresh failed", exc_info=True)
            try:
                sync_cycles = getattr(gui, "_sync_gait_cycles_labels", None)
                if callable(sync_cycles):
                    sync_cycles()
            except Exception:
                logger.debug("advanced gait cycles refresh failed", exc_info=True)
            # Disk scan is cached — safe to call; skips IO when fresh.
            notify = getattr(gui, "_notify_generated_files_changed", None)
            if callable(notify):
                try:
                    notify(force=False)
                except TypeError:
                    try:
                        notify()
                    except Exception:
                        logger.debug("generated files refresh failed", exc_info=True)
                except Exception:
                    logger.debug("generated files refresh failed", exc_info=True)
            # OpenSim auto export/IK is deferred until Advanced is visited.
            if getattr(gui, "_opensim_auto_pending", False):
                maybe = getattr(gui, "_maybe_auto_opensim_pipeline", None)
                if callable(maybe):
                    try:
                        maybe()
                    except Exception:
                        logger.debug("deferred OpenSim auto failed", exc_info=True)
            from stablewalk.ui.tk.dashboard_notebook import realize_advanced_tab_shell

            realize_advanced_tab_shell(gui)

        from stablewalk.ui.tk.timer_registry import schedule_unique

        schedule_unique(gui, "_tab_activate_paint_after", 16, _advanced_refresh)
        from stablewalk.ui.tk.dashboard_notebook import realize_advanced_tab_shell

        realize_advanced_tab_shell(gui)
    elif tab_id == TAB_BIOMECHANICS:
        # Lazy-create only the chart family the workspace currently shows.
        # Contact/vGRF figures are built when those internal tabs are selected.
        family = str(getattr(gui, "_biomech_chart_family", "biomech") or "biomech")
        if family == "contact":
            ensure_contact = getattr(gui, "_ensure_contact_gait_chart", None)
            if callable(ensure_contact):
                try:
                    ensure_contact()
                except Exception:
                    logger.debug("contact chart ensure failed", exc_info=True)
        else:
            ensure = getattr(gui, "_ensure_biomech_chart", None)
            if callable(ensure):
                try:
                    ensure()
                except Exception:
                    logger.debug("biomech chart ensure failed", exc_info=True)

        def _biomech_refresh() -> None:
            refresh = getattr(gui, "_refresh_active_biomech_chart", None)
            if callable(refresh):
                try:
                    refresh()
                    return
                except Exception:
                    logger.debug("biomech chart refresh failed on activate", exc_info=True)
            fam = str(getattr(gui, "_biomech_chart_family", "biomech") or "biomech")
            fit = getattr(
                gui,
                "_fit_contact_gait_canvas"
                if fam == "contact"
                else "_fit_biomech_canvas",
                None,
            )
            if callable(fit):
                try:
                    fit()
                except Exception:
                    logger.debug("biomech chart fit failed on activate", exc_info=True)

        from stablewalk.ui.tk.timer_registry import schedule_unique

        schedule_unique(gui, "_tab_activate_paint_after", 20, _biomech_refresh)
        refresh_panel = getattr(gui, "_update_biomechanics_panel", None)
        if callable(refresh_panel):
            try:
                gui._biomech_static_sig = None
                refresh_panel()
            except Exception:
                logger.debug("biomech findings refresh on activate failed", exc_info=True)
    elif tab_id == TAB_MOTION:
        # Ensure lazy canvases exist once. Full rebuild runs from
        # ``_refresh_active_tab_charts`` (and is skipped while playing).
        ensure = getattr(gui, "_ensure_motion_tab_charts", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:
                logger.debug("motion charts ensure failed", exc_info=True)

        def _motion_geometry() -> None:
            try:
                reflow = getattr(gui, "_reflow_motion_workspace", None)
                if callable(reflow):
                    reflow()
            except Exception:
                logger.debug("motion middle reflow on activate failed", exc_info=True)
            try:
                sync = getattr(gui, "_sync_motion_graph_height_share", None)
                if callable(sync):
                    gui._motion_graph_row_minsize = None
                    sync()
            except Exception:
                logger.debug("motion graph height share on activate failed", exc_info=True)
            try:
                from stablewalk.ui.tk.dashboard_notebook import (
                    activate_motion_tab_trajectory,
                )

                activate_motion_tab_trajectory(gui)
            except Exception:
                logger.debug("motion chart paint on activate failed", exc_info=True)

        from stablewalk.ui.tk.timer_registry import schedule_unique

        schedule_unique(gui, "_tab_activate_paint_after", 20, _motion_geometry)
    elif tab_id == TAB_OVERVIEW:
        import time

        gui._overview_tab_enter_ts = time.perf_counter()
        setattr(gui, "_skel_fit_size_sig", None)
        # Create the heavy skeleton figure on first Overview visit only.
        ensure_skel = getattr(gui, "_ensure_skeleton_figure", None)
        if callable(ensure_skel):
            try:
                ensure_skel()
            except Exception:
                logger.debug("skeleton figure ensure failed", exc_info=True)
        playing = bool(getattr(gui, "playing", False))
        viz = getattr(gui, "_section_visual", None) or getattr(
            gui, "_primary_viz_host", None
        )
        viz_mapped = False
        if viz is not None:
            try:
                viz_mapped = bool(viz.winfo_ismapped()) and str(
                    viz.winfo_manager() or ""
                ) == "grid"
            except tk.TclError:
                viz_mapped = False
        try:
            from stablewalk.ui.tk.dashboard_overview_panes import (
                remount_overview_viz_if_unmapped,
            )

            remount_overview_viz_if_unmapped(gui)
        except Exception:
            logger.debug("overview panel remount on activate failed", exc_info=True)
        # Do not rebuild Overview on tab activate. Full view-mode remount is
        # only for a missing viz host, and never while playing.
        if not viz_mapped and not playing:
            try:
                from stablewalk.ui.tk.dashboard_overview_view_mode import (
                    apply_overview_view_mode,
                )

                apply_overview_view_mode(gui)
            except Exception:
                logger.debug(
                    "overview view-mode remount on activate failed", exc_info=True
                )
        sync = getattr(gui, "_sync_overview_scroll", None)
        if callable(sync):
            try:
                sync()
            except Exception:
                logger.debug("overview scroll sync on activate failed", exc_info=True)
        # Coalesce media refit; skip entirely while playing with a mapped workspace.
        if not (playing and viz_mapped):
            schedule = getattr(gui, "_schedule_overview_media_refit", None)
            if callable(schedule):
                try:
                    schedule(delay_ms=40 if not viz_mapped else 120)
                except Exception:
                    logger.debug(
                        "overview media refit on activate failed", exc_info=True
                    )
        if not playing:
            update = getattr(gui, "_update_overview_clinical_summary", None)
            if callable(update):
                try:
                    update()
                except Exception:
                    logger.debug("overview clinical refresh failed", exc_info=True)
        from stablewalk.ui.tk.perf_diagnostics import (
            perf_diagnostics_enabled,
            record_system_snapshot,
        )

        if perf_diagnostics_enabled(gui):
            record_system_snapshot(gui)
            try:
                from stablewalk.ui.tk.gui_memory import (
                    approximate_rss_mb,
                    collect_active_timer_ids,
                    collect_owned_figures,
                )

                rss = approximate_rss_mb()
                timers = len(collect_active_timer_ids(gui))
                figs = len(collect_owned_figures(gui))
                rss_txt = f"{rss:.1f} MiB" if rss is not None else "—"
                print(
                    f"[Perf] Overview tab enter  "
                    f"timers={timers} figures={figs} rss={rss_txt}",
                    flush=True,
                )
            except Exception:
                pass


def handle_notebook_tab_changed(gui: Any, notebook: tk.Misc) -> None:
    """Central enter/leave hook for ``<<NotebookTabChanged>>``."""
    import time

    from stablewalk.ui.tk.perf_diagnostics import timed_section

    prev = getattr(gui, "_active_dashboard_tab_id", None)
    t0 = time.perf_counter()
    rss_before = None
    try:
        from stablewalk.ui.tk.perf_diagnostics import perf_diagnostics_enabled
        from stablewalk.ui.tk.gui_memory import approximate_rss_mb

        if perf_diagnostics_enabled(gui):
            rss_before = approximate_rss_mb()
    except Exception:
        rss_before = None
    try:
        current_widget = notebook.nametowidget(notebook.select())  # type: ignore[attr-defined]
    except (tk.TclError, KeyError, AttributeError):
        current_widget = None
    current = resolve_tab_id(gui, current_widget)
    with timed_section(gui, "tab_switch", warn_label="tab switch"):
        if prev and prev != current:
            deactivate_tab(gui, prev)
            # Drop queued paints for the tab we left so they cannot stall playback.
            cancel = getattr(gui, "_cancel_queued_redraw_callbacks", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    logger.debug("queued redraw cancel failed", exc_info=True)
        gui._active_dashboard_tab_id = current
        activate_tab(gui, current)
    # Soft RSS budget: trim reusable caches without changing UI layout.
    try:
        from stablewalk.ui.tk.gui_memory import enforce_rss_soft_budget

        enforce_rss_soft_budget(gui)
    except Exception:
        logger.debug("rss soft budget trim failed", exc_info=True)
    # Optional tab-switch instrumentation (STABLEWALK_PERF_LOG / diag).
    try:
        from stablewalk.ui.tk.perf_diagnostics import (
            perf_diagnostics_enabled,
            record_system_snapshot,
        )

        if perf_diagnostics_enabled(gui):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            snap = record_system_snapshot(gui)
            rss_after = getattr(snap, "rss_mb", None) if snap else None
            delta = ""
            if rss_before is not None and rss_after is not None:
                delta = f"  rss_delta={rss_after - rss_before:+.1f} MiB"
            print(
                f"[Perf] tab switch {prev or '—'} → {current or '—'}  "
                f"{elapsed_ms:.1f} ms  "
                f"timers={getattr(snap, 'after_callback_count', '?')}  "
                f"figures={getattr(snap, 'figure_count', '?')}{delta}",
                flush=True,
            )
            enter_ts = getattr(gui, "_overview_tab_enter_ts", None)
            if current == TAB_OVERVIEW and enter_ts is not None:
                load_ms = (time.perf_counter() - float(enter_ts)) * 1000.0
                print(f"[Perf] Overview load time  {load_ms:.1f} ms", flush=True)
    except Exception:
        pass


__all__ = [
    "activate_tab",
    "active_tab_id",
    "compare_laboratory_page_visible",
    "cancel_pipeline_animations",
    "cancel_resize_jobs",
    "deactivate_tab",
    "ensure_tab_content",
    "handle_notebook_tab_changed",
    "release_compare_temp_buffers",
    "resolve_tab_id",
]
