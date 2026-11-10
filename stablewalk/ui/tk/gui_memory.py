"""
GUI memory / timer diagnostics and load-time teardown helpers.

Keeps Compare and Overview from retaining stale timers, captures, frame
buffers, and orphan Matplotlib figures across video loads.
"""

from __future__ import annotations

import logging
import sys
import traceback
from typing import Any

logger = logging.getLogger(__name__)

# Known ``root.after`` / widget.after job attributes on the main GUI.
_GUI_TIMER_ATTRS: tuple[str, ...] = (
    "_after_id",
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
    "_snapshot_poll_job",
    "_overview_sash_sync_after",
    "_overview_media_refit_after",
)

_COMPARE_TIMER_ATTRS: tuple[str, ...] = (
    "_play_job",
    "_video_resize_job",
    "_scroll_watch_job",
    "_visual_refresh_job",
    "_snapshot_poll_job",
    "_compare_timeout_job",
)

_GUI_FIGURE_ATTRS: tuple[str, ...] = (
    "fig",
    "fig_3d",
    "fig_dof_traj",
    "fig_dof_traj_overview",
    "fig_contact_gait",
    "fig_biomech",
    "fig_joint_motion",
    "fig_robot",
)

_COMPARE_FIGURE_ATTRS: tuple[str, ...] = (
    "fig_chart",
    "fig_heat",
    "fig_path",
    "fig_skel_a",
    "fig_skel_b",
    "fig_skel_ov",
)

_SKELETON_CACHE_MAX = 96
# Soft ceiling for a typical analyzed session (progressive cache trim).
TYPICAL_SESSION_RSS_BUDGET_MB = 500.0


def approximate_rss_mb() -> float | None:
    """Best-effort resident set size in MiB (Windows / Unix)."""
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
            GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        else:
            with open("/proc/self/status", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        return float(parts[1]) / 1024.0
    except Exception:
        pass
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(usage) / (1024.0 * 1024.0)
        return float(usage) / 1024.0
    except Exception:
        return None


def count_active_threads() -> int:
    import threading

    return int(threading.active_count())


def collect_perf_system_snapshot(gui: Any) -> dict[str, Any]:
    """Compact system counters for Performance Diagnostics."""
    timers = collect_active_timer_ids(gui)
    figs = collect_owned_figures(gui)
    playback = 0
    if getattr(gui, "_after_id", None) is not None:
        playback += 1
    compare = getattr(gui, "_comparison_mode", None)
    if compare is not None and getattr(compare, "_play_job", None) is not None:
        playback += 1
    workers = 0
    if getattr(gui, "_pipeline_callback", None) is not None:
        workers += 1
    if getattr(gui, "_analysis_compute_active", False):
        workers += 1
    opensim = getattr(gui, "_opensim_worker", None)
    if opensim is not None and getattr(opensim, "is_alive", lambda: False)():
        workers += 1
    if compare is not None and (
        getattr(compare, "_compare_worker", None) is not None
        or getattr(compare, "_loading_keys", None)
    ):
        cw = getattr(compare, "_compare_worker", None)
        if cw is None or getattr(cw, "is_alive", lambda: False)():
            workers += 1
    return {
        "rss_mb": approximate_rss_mb(),
        "thread_count": count_active_threads(),
        "after_callback_count": len(timers),
        "figure_count": len(figs),
        "playback_timer_count": playback,
        "analysis_worker_count": workers,
        "cached_frames": count_cached_frames(gui),
        "active_captures": count_active_captures(),
    }


def collect_active_timer_ids(gui: Any) -> list[str]:
    """Return string ids of known scheduled ``after`` jobs (non-None only)."""
    found: list[str] = []
    for attr in _GUI_TIMER_ATTRS:
        job = getattr(gui, attr, None)
        if job is not None:
            found.append(f"gui.{attr}={job}")
    compare = getattr(gui, "_comparison_mode", None)
    if compare is not None:
        for attr in _COMPARE_TIMER_ATTRS:
            job = getattr(compare, attr, None)
            if job is not None:
                found.append(f"compare.{attr}={job}")
        deferred = getattr(compare, "_deferred_jobs", None) or []
        for i, job in enumerate(list(deferred)):
            if job is not None:
                found.append(f"compare._deferred_jobs[{i}]={job}")
    return found


def collect_owned_figures(gui: Any) -> list[Any]:
    """Figures owned by the live dashboard (do not close these)."""
    figs: list[Any] = []
    seen: set[int] = set()
    for attr in _GUI_FIGURE_ATTRS:
        fig = getattr(gui, attr, None)
        if fig is not None and id(fig) not in seen:
            seen.add(id(fig))
            figs.append(fig)
    compare = getattr(gui, "_comparison_mode", None)
    if compare is not None:
        for attr in _COMPARE_FIGURE_ATTRS:
            fig = getattr(compare, attr, None)
            if fig is not None and id(fig) not in seen:
                seen.add(id(fig))
                figs.append(fig)
    return figs


def count_cached_frames(gui: Any) -> int:
    """Sum of RGB / PIL / PhotoImage frame buffer entries."""
    total = 0
    rgb = getattr(gui, "_rgb_cache", None)
    if rgb is not None:
        try:
            total += int(len(rgb))
        except Exception:
            store = getattr(rgb, "_store", None) or getattr(rgb, "_cache", None)
            if store is not None:
                total += len(store)
    if getattr(gui, "_photo", None) is not None:
        total += 1
    if getattr(gui, "_pb_shared_frame_rgb", None) is not None:
        total += 1
    compare = getattr(gui, "_comparison_mode", None)
    if compare is not None:
        for attr in ("_pil_frame_cache", "_photo_cache"):
            cache = getattr(compare, attr, None)
            if isinstance(cache, dict):
                total += len(cache)
    return total


def count_compare_sessions(gui: Any) -> int:
    cache = getattr(gui, "_compare_session_cache", None)
    if cache is None:
        return 0
    try:
        return len(cache.keys())
    except Exception:
        sessions = getattr(cache, "_sessions", None)
        return len(sessions) if isinstance(sessions, dict) else 0


def count_active_captures() -> int:
    try:
        from stablewalk.core.pipeline_reset import _active_captures

        return len(_active_captures)
    except Exception:
        return 0


def trim_skeleton_cache(gui: Any, *, max_entries: int = _SKELETON_CACHE_MAX) -> int:
    """Drop oldest skeleton entries; return number removed."""
    cache = getattr(gui, "_skeleton_cache", None)
    if not isinstance(cache, dict) or len(cache) <= max_entries:
        return 0
    overflow = len(cache) - max_entries
    for key in list(cache.keys())[:overflow]:
        cache.pop(key, None)
    return overflow


def enforce_rss_soft_budget(
    gui: Any,
    *,
    budget_mb: float = TYPICAL_SESSION_RSS_BUDGET_MB,
) -> dict[str, Any]:
    """Trim reusable caches when RSS approaches the typical-session budget.

    Does not change UI layout — only drops RGB / skeleton / overlay / blit
    buffers and orphan figures. Safe to call on tab switch and after analysis.
    """
    rss = approximate_rss_mb()
    result: dict[str, Any] = {
        "rss_mb": rss,
        "budget_mb": float(budget_mb),
        "trimmed": False,
        "actions": {},
    }
    if rss is None or rss < float(budget_mb) * 0.85:
        return result

    actions = result["actions"]
    # Progressive: soft pressure first, then harder when over budget.
    hard = rss >= float(budget_mb)
    rgb = getattr(gui, "_rgb_cache", None)
    if rgb is not None and hasattr(rgb, "trim"):
        try:
            keep = 16 if hard else 32
            actions["rgb_trimmed"] = int(rgb.trim(keep) or 0)
        except Exception:
            pass
    skel_max = 24 if hard else 48
    actions["skeletons_trimmed"] = trim_skeleton_cache(gui, max_entries=skel_max)
    overlay = getattr(gui, "_landmark_overlay_cache", None)
    if isinstance(overlay, dict) and overlay:
        keep = 8 if hard else 16
        overflow = max(0, len(overlay) - keep)
        for key in list(overlay.keys())[:overflow]:
            overlay.pop(key, None)
        actions["overlay_trimmed"] = overflow
    try:
        from stablewalk.ui.viewers.dof_trajectory_3d import clear_trajectory_path_cache

        # Drop high-water path geometry only under hard pressure; tip rebuilds
        # from progressive cache on the next visible paint.
        if hard:
            clear_trajectory_path_cache()
            actions["trajectory_cache_cleared"] = 1
    except Exception:
        pass
    if hard:
        try:
            from stablewalk.ui.viewers.chart_playhead import clear_static_chart_background

            cleared = 0
            for fig in collect_owned_figures(gui):
                try:
                    clear_static_chart_background(fig)
                    cleared += 1
                except Exception:
                    pass
            if cleared:
                actions["blit_cleared"] = cleared
        except Exception:
            pass
    actions["orphan_figures"] = dispose_orphan_matplotlib_figures(gui)
    compare = getattr(gui, "_comparison_mode", None)
    if compare is not None and hard:
        for attr in ("_pil_frame_cache", "_photo_cache"):
            cache = getattr(compare, attr, None)
            if isinstance(cache, dict) and len(cache) > 8:
                overflow = len(cache) - 8
                for key in list(cache.keys())[:overflow]:
                    cache.pop(key, None)
                actions[f"compare_{attr}_trimmed"] = overflow
    result["trimmed"] = any(int(v or 0) > 0 for v in actions.values())
    result["rss_mb_after"] = approximate_rss_mb()
    return result


def dispose_orphan_matplotlib_figures(gui: Any) -> int:
    """Close pyplot figures that are not owned by the live dashboard canvases.

    Uses quiet close-by-number — never ``plt.figure(num)``, which can pop a
    blank black TkAgg window over the application.
    """
    owned_ids = {id(fig) for fig in collect_owned_figures(gui)}
    try:
        from stablewalk.ui.tk.tk_lifecycle import close_pyplot_figures_quietly

        return close_pyplot_figures_quietly(skip_ids=owned_ids)
    except Exception:
        logger.debug("orphan figure dispose failed", exc_info=True)
        return 0


def clear_owned_figure_axes(gui: Any) -> None:
    """Clear axis content on owned figures without destroying Tk canvases."""
    for fig in collect_owned_figures(gui):
        try:
            for ax in list(getattr(fig, "axes", []) or []):
                try:
                    ax.cla()
                except Exception:
                    pass
            # Drop blit / interaction caches that hold large arrays.
            for attr in (
                "_chart_blit_background",
                "_biomech_blit_background",
                "_stablewalk_blit_bg",
            ):
                if hasattr(fig, attr):
                    try:
                        delattr(fig, attr)
                    except Exception:
                        setattr(fig, attr, None)
        except Exception:
            pass


def retain_compare_sessions_for_load(gui: Any, *, also_keep: set[str] | None = None) -> int:
    """Keep only A/B slots (+ optional keys); drop other cached sessions."""
    try:
        from stablewalk.ui.tk.session_cache import ensure_compare_cache

        cache = ensure_compare_cache(gui)
        retain = getattr(cache, "retain_pinned_slots", None)
        if callable(retain):
            return int(retain(also_keep=also_keep) or 0)
    except Exception:
        logger.debug("compare session retain failed", exc_info=True)
    return 0


def teardown_for_new_video(gui: Any, *, also_keep_session_keys: set[str] | None = None) -> dict[str, int]:
    """Stop playback, cancel jobs, release captures/buffers, prune sessions.

    Safe to call in addition to ``_reset_session`` / ``_stop_all_gui_timers``.
    Returns a small stats dict for logging.
    """
    stats = {
        "timers_cancelled": 0,
        "captures_released": 0,
        "sessions_pruned": 0,
        "orphan_figures_closed": 0,
        "skeletons_trimmed": 0,
    }
    # Playback + scheduled work
    stop = getattr(gui, "_stop_all_gui_timers", None)
    if callable(stop):
        before = len(collect_active_timer_ids(gui))
        try:
            stop()
        except Exception:
            logger.debug("stop_all_gui_timers failed", exc_info=True)
        after = len(collect_active_timer_ids(gui))
        stats["timers_cancelled"] = max(0, before - after)

    try:
        from stablewalk.core.pipeline_reset import release_all_captures

        before_caps = count_active_captures()
        release_all_captures()
        stats["captures_released"] = before_caps
    except Exception:
        pass

    try:
        from stablewalk.ui.tk.dashboard_tab_lifecycle import release_compare_temp_buffers

        release_compare_temp_buffers(gui)
    except Exception:
        pass

    clear_fn = getattr(gui, "_clear_shared_frame_buffer", None)
    if callable(clear_fn):
        try:
            clear_fn()
        except Exception:
            pass
    rgb = getattr(gui, "_rgb_cache", None)
    if rgb is not None and hasattr(rgb, "clear"):
        try:
            rgb.clear()
        except Exception:
            pass
    for attr in ("_photo", "_last_video_rgb", "_video_photo_key"):
        try:
            setattr(gui, attr, None)
        except Exception:
            pass

    stats["sessions_pruned"] = retain_compare_sessions_for_load(
        gui, also_keep=also_keep_session_keys
    )
    stats["orphan_figures_closed"] = dispose_orphan_matplotlib_figures(gui)
    stats["skeletons_trimmed"] = trim_skeleton_cache(gui)
    # Clear skeleton cache entirely on load (caller often resets dict too).
    try:
        gui._skeleton_cache = {}
    except Exception:
        pass
    return stats


def format_gui_memory_dump(gui: Any) -> str:
    """Human-readable snapshot for the Debug → Dump memory / GUI state command."""
    lines: list[str] = ["=== StableWalk GUI memory / lifecycle dump ==="]
    timers = collect_active_timer_ids(gui)
    lines.append(f"active timers ({len(timers)}):")
    if timers:
        for t in timers:
            lines.append(f"  - {t}")
    else:
        lines.append("  (none)")

    figs = collect_owned_figures(gui)
    lines.append(f"active Matplotlib figures (owned): {len(figs)}")
    try:
        import matplotlib.pyplot as plt

        lines.append(f"pyplot figure count: {len(plt.get_fignums())}")
    except Exception:
        lines.append("pyplot figure count: n/a")

    lines.append(f"cached compare sessions: {count_compare_sessions(gui)}")
    cache = getattr(gui, "_compare_session_cache", None)
    if cache is not None:
        try:
            lines.append(
                f"  slots: A={getattr(cache, 'left_key', '?')}  "
                f"B={getattr(cache, 'right_key', '?')}"
            )
            lines.append(f"  keys: {', '.join(cache.keys()) or '(empty)'}")
        except Exception:
            pass

    lines.append(f"cached frame buffers: {count_cached_frames(gui)}")
    sk = getattr(gui, "_skeleton_cache", None)
    lines.append(f"skeleton cache entries: {len(sk) if isinstance(sk, dict) else 0}")
    lines.append(f"active VideoCapture refs: {count_active_captures()}")

    playing = bool(getattr(gui, "playing", False))
    compare = getattr(gui, "_comparison_mode", None)
    cmp_playing = bool(getattr(compare, "_playing", False)) if compare else False
    lines.append(f"playback: main={'on' if playing else 'off'}  compare={'on' if cmp_playing else 'off'}")

    rss = approximate_rss_mb()
    if rss is not None:
        lines.append(f"approximate RSS: {rss:.1f} MiB")
    else:
        lines.append("approximate RSS: n/a")

    # Lightweight widget canvas check
    try:
        from stablewalk.ui.tk.gui_layout_debug import audit_gui_widget_counts

        counts = audit_gui_widget_counts(gui)
        if isinstance(counts, dict):
            lines.append(
                "widgets: "
                + ", ".join(f"{k}={v}" for k, v in list(counts.items())[:8])
            )
    except Exception:
        pass

    return "\n".join(lines)


def dump_gui_memory_state(gui: Any, *, to_console: bool = True, show_dialog: bool = True) -> str:
    """Print (and optionally dialog) the memory / GUI lifecycle dump."""
    text = format_gui_memory_dump(gui)
    if to_console:
        print(text, flush=True)
        logger.info("GUI memory dump\n%s", text)
    if show_dialog:
        try:
            from stablewalk.ui.tk.tk_lifecycle import safe_messagebox

            short = "\n".join(text.splitlines()[:18])
            safe_messagebox(
                "info",
                "GUI memory / timers",
                short + "\n\n(Full dump printed to console.)",
                parent=getattr(gui, "root", None),
            )
        except Exception:
            pass
    return text


__all__ = [
    "TYPICAL_SESSION_RSS_BUDGET_MB",
    "approximate_rss_mb",
    "clear_owned_figure_axes",
    "collect_active_timer_ids",
    "collect_owned_figures",
    "count_active_captures",
    "count_cached_frames",
    "count_compare_sessions",
    "dispose_orphan_matplotlib_figures",
    "dump_gui_memory_state",
    "enforce_rss_soft_budget",
    "format_gui_memory_dump",
    "retain_compare_sessions_for_load",
    "teardown_for_new_video",
    "trim_skeleton_cache",
    "_SKELETON_CACHE_MAX",
]
