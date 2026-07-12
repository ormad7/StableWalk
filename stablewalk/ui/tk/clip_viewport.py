"""
Clipped Tk viewports for dashboard media and plot hosts.

A Canvas window clips child widgets to the host bounds so video frames and
matplotlib canvases cannot paint outside their panel during scroll or resize.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def install_clipped_viewport(
    parent: tk.Misc,
    *,
    bg: str,
    row: int = 0,
    column: int = 0,
    sticky: str = "nsew",
    padx: int | tuple[int, ...] = 0,
    pady: int | tuple[int, ...] = 0,
) -> tuple[tk.Canvas, tk.Frame, int]:
    """
    Create a clipping canvas with one inner frame for child widgets.

    Returns ``(clip_canvas, inner_frame, window_item_id)``.
    """
    try:
        parent.rowconfigure(row, weight=1)
        parent.columnconfigure(column, weight=1)
    except tk.TclError:
        pass

    canvas = tk.Canvas(
        parent,
        bg=bg,
        highlightthickness=0,
        borderwidth=0,
        bd=0,
        relief="flat",
    )
    canvas.grid(row=row, column=column, sticky=sticky, padx=padx, pady=pady)

    inner = tk.Frame(canvas, bg=bg, highlightthickness=0, bd=0)
    window_id = canvas.create_window(0, 0, window=inner, anchor="nw")

    def _sync(_event: object | None = None) -> None:
        try:
            canvas.update_idletasks()
            cw = max(int(canvas.winfo_width()), 1)
            ch = max(int(canvas.winfo_height()), 1)
            canvas.coords(window_id, 0, 0)
            canvas.itemconfigure(window_id, width=cw, height=ch)
        except tk.TclError:
            pass

    canvas.bind("<Configure>", _sync, add="+")
    inner.bind("<Configure>", _sync, add="+")
    _sync()
    return canvas, inner, window_id


def sync_clipped_viewport(canvas: tk.Canvas | None, window_id: int | None) -> None:
    """Resize the inner frame window to match the clipping canvas bounds."""
    if canvas is None or window_id is None:
        return
    try:
        canvas.update_idletasks()
        cw = max(int(canvas.winfo_width()), 1)
        ch = max(int(canvas.winfo_height()), 1)
        canvas.coords(window_id, 0, 0)
        canvas.itemconfigure(window_id, width=cw, height=ch)
    except tk.TclError:
        pass


def clip_canvas_item_count(canvas: tk.Canvas | None) -> int:
    """Return Tk canvas item count (should stay constant during playback)."""
    if canvas is None:
        return 0
    try:
        return len(canvas.find_all())
    except tk.TclError:
        return 0


def is_widget_inside_clip_host(widget: tk.Misc, inner: tk.Misc | None) -> bool:
    """True when widget is a descendant of the clipped inner frame."""
    if widget is None or inner is None:
        return False
    current: tk.Misc | None = widget
    while current is not None:
        if current is inner:
            return True
        current = getattr(current, "master", None)
    return False
