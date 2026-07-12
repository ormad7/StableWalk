"""
GUI layout diagnostics for StableWalk dashboard audits.

Enable with environment variable ``STABLEWALK_GUI_DEBUG=1`` or
``gui._gui_layout_debug = True`` after construction.
"""

from __future__ import annotations

import os
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from stablewalk.ui.tk.clip_viewport import clip_canvas_item_count


def gui_debug_enabled(gui: Any) -> bool:
    if getattr(gui, "_gui_layout_debug", False):
        return True
    return os.environ.get("STABLEWALK_GUI_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _count_figure_canvases(root: Any, attr_names: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in attr_names:
        canvas = getattr(root, name, None)
        if canvas is None:
            counts[name] = 0
            continue
        if isinstance(canvas, FigureCanvasTkAgg):
            counts[name] = 1
        else:
            counts[name] = 0
    return counts


def _count_widget_class(root: Any, cls: type) -> int:
    total = 0

    def _walk(widget: Any) -> None:
        nonlocal total
        try:
            if isinstance(widget, cls):
                total += 1
            for child in widget.winfo_children():
                _walk(child)
        except Exception:
            return

    try:
        _walk(root)
    except Exception:
        pass
    return total


def audit_gui_widget_counts(gui: Any) -> dict[str, int]:
    """Return singleton counts for critical dashboard canvases."""
    attrs = (
        "canvas_3d",
        "chart_canvas",
        "canvas_dof_traj",
        "canvas_robot",
    )
    counts = _count_figure_canvases(gui, attrs)
    root = getattr(gui, "root", gui)
    counts["FigureCanvasTkAgg_total"] = _count_widget_class(root, FigureCanvasTkAgg)
    counts["video_label_instances"] = 1 if getattr(gui, "video_label", None) is not None else 0
    return counts


def format_gui_layout_audit(gui: Any) -> str:
    counts = audit_gui_widget_counts(gui)
    lines = ["StableWalk GUI layout audit", ""]
    labels = {
        "canvas_3d": "3D reconstruction canvas instances",
        "chart_canvas": "Knee graph canvas instances",
        "canvas_dof_traj": "Joint path canvas instances",
        "canvas_robot": "Robot panel canvas instances",
        "video_label_instances": "Video canvas instances",
        "FigureCanvasTkAgg_total": "FigureCanvasTkAgg widgets (tree total)",
    }
    for key, label in labels.items():
        lines.append(f"{label}: {counts.get(key, 0)}")
    lines.append(f"Video clip canvas items: {clip_canvas_item_count(getattr(gui, '_video_clip_canvas', None))}")
    lines.append(f"3D clip canvas items: {clip_canvas_item_count(getattr(gui, '_skel_clip_canvas', None))}")
    lines.append(f"Knee clip canvas items: {clip_canvas_item_count(getattr(gui, '_knee_clip_canvas', None))}")
    lines.append(f"Joint path clip canvas items: {clip_canvas_item_count(getattr(gui, '_traj_clip_canvas', None))}")
    scroll = getattr(gui, "_tab_advanced_scroll_canvas", None) or getattr(
        gui, "_dash_scroll_canvas", None
    )
    analysis = getattr(gui, "_analysis_scroll_canvas", None)
    notebook = getattr(gui, "_dashboard_notebook", None)
    lines.append(f"Dashboard notebook: {'yes' if notebook is not None else 'no'}")
    lines.append(f"Advanced tab scroll: {'yes' if scroll is not None else 'no'}")
    lines.append(f"Nested analysis scroll canvas: {'yes' if analysis is not None else 'no'}")
    transport = getattr(gui, "_transport_row", None)
    lines.append(f"Fixed transport bar: {'yes' if transport is not None else 'no'}")
    return "\n".join(lines)


def print_gui_layout_audit(gui: Any) -> None:
    print(format_gui_layout_audit(gui), flush=True)


def log_foot_clearance_debug_if_enabled(panel: Any) -> None:
    """Print canonical floor/heel/toe clearance debug lines when GUI debug is on."""
    if os.environ.get("STABLEWALK_GUI_DEBUG", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    lines = getattr(panel, "debug_lines", ())
    for line in lines:
        print(f"[Foot clearance debug] {line}", flush=True)


def log_gui_layout_audit_if_enabled(gui: Any, *, context: str = "") -> None:
    if not gui_debug_enabled(gui):
        return
    prefix = f"[GUI debug{': ' + context if context else ''}]"
    for line in format_gui_layout_audit(gui).splitlines():
        print(f"{prefix} {line}", flush=True)
