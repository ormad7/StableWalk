"""Skip full-tab Matplotlib/Tk redraws when the user cannot see the widget.

Playback and window-resize must not rebuild every chart on every tick.
Visible, mapped canvases may blit playheads or ``draw_idle``; hidden tabs
and unmapped hosts stay frozen until they become active.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any


def widget_is_mapped(widget: Any) -> bool:
    """True when a Tk widget currently occupies screen space."""
    if widget is None:
        return False
    try:
        return bool(widget.winfo_ismapped())
    except tk.TclError:
        return False


def mpl_canvas_is_mapped(canvas: Any) -> bool:
    """True when a Matplotlib TkAgg canvas is mapped (or not yet realized)."""
    if canvas is None:
        return False
    getter = getattr(canvas, "get_tk_widget", None)
    if not callable(getter):
        return widget_is_mapped(canvas)
    try:
        widget = getter()
    except Exception:
        return False
    if widget is None:
        return False
    try:
        # ``winfo_exists`` with width 1 is common before first map — allow it
        # so the first paint can create artists. After it exists and is unmapped,
        # skip the rasterizer.
        if not widget.winfo_exists():
            return False
        if int(widget.winfo_width() or 0) <= 1 and int(widget.winfo_height() or 0) <= 1:
            return bool(widget.winfo_ismapped())
        return bool(widget.winfo_ismapped())
    except tk.TclError:
        return False


def skip_unmapped_canvas_paint(canvas: Any) -> bool:
    """True when a paint/blit should be skipped because the canvas is off-screen."""
    if canvas is None:
        return True
    getter = getattr(canvas, "get_tk_widget", None)
    if not callable(getter):
        return not widget_is_mapped(canvas)
    try:
        widget = getter()
    except Exception:
        return True
    if widget is None:
        return True
    try:
        if not widget.winfo_exists():
            return True
        # Never created / zero-size on a visible tab: allow the first layout paint.
        mapped = bool(widget.winfo_ismapped())
        if mapped:
            return False
        # Exists but not mapped → hidden notebook tab or grid_remove.
        return True
    except tk.TclError:
        return True


def full_tab_redraw_blocked(gui: Any) -> bool:
    """True during playback or an in-flight window resize — no tab-wide rebuilds."""
    if bool(getattr(gui, "playing", False)):
        return True
    if bool(getattr(gui, "_ui_resize_active", False)):
        return True
    return False


def mark_resize_active(gui: Any, active: bool) -> None:
    try:
        gui._ui_resize_active = bool(active)
    except Exception:
        pass


set_resize_active = mark_resize_active
