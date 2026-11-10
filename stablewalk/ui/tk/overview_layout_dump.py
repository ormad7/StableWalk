"""Dump Overview widget/layout tree for Debug → Dump Overview Layout."""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Any

logger = logging.getLogger(__name__)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _widget_line(name: str, w: Any, *, depth: int = 0) -> str:
    ind = "  " * depth
    if w is None:
        return f"{ind}{name}: <missing>"
    mapped = _safe(lambda: bool(w.winfo_ismapped()), False)
    viewable = _safe(lambda: bool(w.winfo_viewable()), False)
    geo = _safe(
        lambda: f"{w.winfo_x()},{w.winfo_y()} {w.winfo_width()}x{w.winfo_height()}",
        "?",
    )
    req = _safe(lambda: f"{w.winfo_reqwidth()}x{w.winfo_reqheight()}", "?")
    mgr = _safe(lambda: w.winfo_manager() or "", "")
    parent = _safe(lambda: str(w.master), None)
    cls = _safe(lambda: w.winfo_class(), type(w).__name__)
    extra = ""
    if mgr == "grid":
        info = _safe(lambda: dict(w.grid_info()), {}) or {}
        extra = (
            f" grid(row={info.get('row')}, col={info.get('column')}, "
            f"sticky={info.get('sticky')!r})"
        )
    return (
        f"{ind}{name}: class={cls} mapped={mapped} viewable={viewable} "
        f"geo={geo} req={req} mgr={mgr!r} parent={parent}{extra}"
    )


def dump_overview_layout(gui: Any) -> str:
    """Return a multi-line Overview hierarchy + geometry dump."""
    lines: list[str] = ["=== Overview layout dump ==="]

    attrs = (
        "_tab_overview",
        "_overview_scroll_outer",
        "_overview_scroll_canvas",
        "_overview_content",
        "_overview_layout_host",
        "_overview_clinical_summary",
        "_overview_main_split",
        "_overview_info_outer",
        "_overview_info_inner",
        "_section_visual",
        "video_frame",
        "skel_frame",
        "overview_traj_panel",
        "_transport_row",
        "canvas_3d",
        "fig_3d",
        "canvas_dof_traj_overview",
        "fig_dof_traj_overview",
        "video_label",
    )
    for name in attrs:
        lines.append(_widget_line(name, getattr(gui, name, None)))

    content = getattr(gui, "_overview_content", None)
    if content is not None:
        lines.append("--- content row weights ---")
        for r in range(8):
            cfg = _safe(lambda r=r: content.grid_rowconfigure(r), {})
            lines.append(f"  row {r}: {cfg}")
        lines.append("--- content children ---")
        for i, child in enumerate(_safe(content.winfo_children, []) or []):
            lines.append(_widget_line(f"child[{i}]", child, depth=1))

    section = getattr(gui, "_section_visual", None)
    if section is not None:
        lines.append("--- visualization workspace children ---")
        for i, child in enumerate(_safe(section.winfo_children, []) or []):
            lines.append(_widget_line(f"viz_child[{i}]", child, depth=1))

    # Timers / figures
    try:
        from stablewalk.ui.tk.gui_memory import (
            approximate_rss_mb,
            collect_active_timer_ids,
            collect_owned_figures,
        )
        from stablewalk.ui.tk.perf_diagnostics import count_playback_timers

        timers = collect_active_timer_ids(gui)
        figs = collect_owned_figures(gui)
        rss = approximate_rss_mb()
        lines.append(
            f"--- runtime: playback_timers={count_playback_timers(gui)} "
            f"known_after={len(timers)} figures={len(figs)} "
            f"rss={rss if rss is not None else '—'} ---"
        )
    except Exception as exc:
        lines.append(f"--- runtime: unavailable ({exc}) ---")

    lines.append(
        f"clinical_cap_h={getattr(gui, '_overview_clinical_cap_h', None)} "
        f"scroll_needed={getattr(gui, '_overview_scroll_needed', None)} "
        f"viz_stack={getattr(gui, '_overview_viz_stack_requested', None)}"
    )
    text = "\n".join(lines)
    print(text, flush=True)
    logger.info(text)
    return text


__all__ = ["dump_overview_layout"]
