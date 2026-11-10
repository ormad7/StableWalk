"""Playback-aware layout helpers for the Tk dashboard.

While video is playing, Configure/reflow handlers must not thrash Tk geometry
or Matplotlib figure fits. Mark dirty instead and flush once on pause.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any


def is_playing(gui: Any) -> bool:
    return bool(getattr(gui, "playing", False))


def resolve_gui(widget: tk.Misc | None) -> Any | None:
    """Return the StableWalkGUI stashed on the Tk root, if any."""
    if widget is None:
        return None
    try:
        root = widget.winfo_toplevel()
    except tk.TclError:
        return None
    return getattr(root, "_stablewalk_gui", None)


def mark_layout_dirty(gui: Any) -> None:
    gui._layout_dirty_while_playing = True


def should_defer_layout(gui: Any | None = None, *, widget: tk.Misc | None = None) -> bool:
    """True when layout work should be skipped (marks dirty when playing).

    Explicit one-shot refreshes (e.g. opening Motion Analysis while playing) set
    ``gui._force_layout_once`` so the tab can size itself without waiting for pause.
    """
    if gui is None:
        gui = resolve_gui(widget)
    if gui is None:
        return False
    if getattr(gui, "_force_layout_once", False):
        return False
    if is_playing(gui):
        mark_layout_dirty(gui)
        return True
    # Live window drag: skip Matplotlib fits / tab reflows until size settles.
    if bool(getattr(gui, "_ui_resize_active", False)):
        return True
    return False


def force_layout_pass(gui: Any):
    """Context manager: allow layout while playing for a single refresh pass."""

    class _ForceLayout:
        def __enter__(self_inner):
            self_inner._prev = bool(getattr(gui, "_force_layout_once", False))
            gui._force_layout_once = True
            return gui

        def __exit__(self_inner, *_exc):
            gui._force_layout_once = self_inner._prev
            return False

    return _ForceLayout()


def flush_layout_dirty(gui: Any) -> None:
    """Run one deferred layout pass after playback stops."""
    if not getattr(gui, "_layout_dirty_while_playing", False):
        return
    gui._layout_dirty_while_playing = False
    root = getattr(gui, "root", None)
    if root is None:
        _run_deferred_layout(gui)
        return

    prev = getattr(gui, "_layout_flush_after", None)
    if prev is not None:
        try:
            root.after_cancel(prev)
        except (tk.TclError, ValueError):
            pass

    def _run() -> None:
        gui._layout_flush_after = None
        _run_deferred_layout(gui)

    try:
        gui._layout_flush_after = root.after(40, _run)
    except tk.TclError:
        _run_deferred_layout(gui)


def _run_deferred_layout(gui: Any) -> None:
    try:
        from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout

        apply_responsive_layout(gui)
    except Exception:
        pass
    for attr in (
        "_fit_skeleton_canvas",
        "_fit_dof_traj_canvas",
        "_on_video_label_resize",
        "_sync_dashboard_scroll",
        "_sync_motion_scroll",
    ):
        fn = getattr(gui, attr, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
    biomech_fit_attr = (
        "_fit_contact_gait_canvas"
        if getattr(gui, "_biomech_chart_family", "biomech") == "contact"
        else "_fit_biomech_canvas"
    )
    biomech_fit = getattr(gui, biomech_fit_attr, None)
    if callable(biomech_fit):
        try:
            biomech_fit()
        except Exception:
            pass
    # Motion Analysis can stay blank if it was opened while playing and never
    # received a forced restore after pause.
    refresh_motion = getattr(gui, "refresh_motion_analysis_tab", None)
    if callable(refresh_motion):
        try:
            from stablewalk.ui.tk.dashboard_notebook import is_motion_tab_selected

            if is_motion_tab_selected(gui):
                refresh_motion()
        except Exception:
            pass
