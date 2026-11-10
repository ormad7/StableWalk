"""Named Tk ``after`` job registry — prevent duplicate timers and clean shutdown.

Does not own the main playback clock (``gui._after_id``) or Compare's
``_play_job``; those stay on their owners. This module lists ancillary jobs
that must be cancelled on load / reset / session clear / tab leave.
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Deferred layout / resize / idle work (not the main playback timer).
ANCILLARY_AFTER_ATTRS: tuple[str, ...] = (
    "_presentation_after_id",
    "_preset_after_id",
    "_video_resize_after",
    "_scroll_sync_after",
    "_traj_resize_after",
    "_skel_resize_after",
    "_knee_resize_after",
    "_tab_reflow_after",
    "_layout_flush_after",
    "_dof_traj_reflow_after",
    "_contact_resize_job",
    "_biomech_resize_job",
    "_biomech_chart_refresh_after",
    "_overview_sash_sync_after",
    "_overview_media_refit_after",
    "_overview_viz_room_after_id",
    "_overview_clinical_refresh_after",
    "_generated_files_refresh_after",
    "_tab_activate_paint_after",
    "_advanced_shell_realize_after",
    "_post_analysis_defer_after",
    "_progressive_metrics_after",
    "_progressive_graphs_after",
    "_progressive_biomech_after",
    "_progressive_opensim_after",
)


def cancel_after_attr(owner: Any, attr: str, *, root: tk.Misc | None = None) -> None:
    """Cancel a single ``after`` job stored on ``owner.attr`` and clear it."""
    job = getattr(owner, attr, None)
    if job is None:
        return
    host = root
    if host is None:
        host = getattr(owner, "root", None)
    if host is not None:
        try:
            host.after_cancel(job)
        except (tk.TclError, ValueError, TypeError):
            pass
    setattr(owner, attr, None)


def cancel_ancillary_after_jobs(gui: Any) -> None:
    """Cancel all registered ancillary after-jobs on the GUI (keep playback)."""
    root = getattr(gui, "root", None)
    for attr in ANCILLARY_AFTER_ATTRS:
        cancel_after_attr(gui, attr, root=root)


def schedule_unique(
    gui: Any,
    attr: str,
    delay_ms: int,
    callback: Callable[[], None],
    *,
    replace: bool = True,
) -> Any:
    """
    Schedule ``callback`` once under ``gui.attr``.

    If ``replace`` and a job is already pending, it is cancelled first so
    Configure storms cannot stack duplicate callbacks.
    """
    root = getattr(gui, "root", None)
    if root is None:
        callback()
        return None
    if replace:
        cancel_after_attr(gui, attr, root=root)

    def _run() -> None:
        setattr(gui, attr, None)
        try:
            callback()
        except Exception:
            logger.debug("scheduled job %s failed", attr, exc_info=True)

    try:
        job = root.after(max(0, int(delay_ms)), _run)
    except tk.TclError:
        callback()
        return None
    setattr(gui, attr, job)
    return job


def count_known_after_jobs(gui: Any) -> int:
    """Count non-None ancillary + playback + compare play jobs (diagnostics)."""
    n = 0
    if getattr(gui, "_after_id", None) is not None:
        n += 1
    for attr in ANCILLARY_AFTER_ATTRS:
        if getattr(gui, attr, None) is not None:
            n += 1
    compare = getattr(gui, "_comparison_mode", None)
    if compare is not None and getattr(compare, "_play_job", None) is not None:
        n += 1
    return n


__all__ = [
    "ANCILLARY_AFTER_ATTRS",
    "cancel_after_attr",
    "cancel_ancillary_after_jobs",
    "count_known_after_jobs",
    "schedule_unique",
]
