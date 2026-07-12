"""
Reusable vertical scroll shell for Tk dashboard content.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.theme import BG, PAD_SM


def install_vertical_scroll_shell(
    gui,
    parent: tk.Misc,
    *,
    bottom_pad: int = PAD_SM,
) -> ttk.Frame:
    """
    Create a single vertical scroll container and return the sections host frame.

    Architecture:
    ROOT → main (pack) → canvas (grid) → scroll_content (create_window)
      → sections_host (row 0, returned) + bottom_spacer (row 1)

    The canvas window width tracks the visible canvas width on every Configure.
    """
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)

    outer = ttk.Frame(parent)
    outer.grid(row=0, column=0, sticky="nsew")
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(0, weight=1)

    canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, borderwidth=0)
    vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    canvas.configure(yscrollcommand=vsb.set)

    # Single scroll content frame — all dashboard sections are direct children.
    scroll_content = ttk.Frame(canvas)
    inner_id = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
    scroll_content.columnconfigure(0, weight=1)

    bottom_spacer = ttk.Frame(scroll_content, height=bottom_pad)
    bottom_spacer.grid(row=1, column=0, sticky="ew")
    bottom_spacer.grid_propagate(False)
    gui._dash_scroll_bottom_spacer = bottom_spacer
    gui._dash_scroll_bottom_pad = bottom_pad

    sections_host = ttk.Frame(scroll_content)
    sections_host.grid(row=0, column=0, sticky="ew")
    sections_host.columnconfigure(0, weight=1)
    gui._dash_scroll_sections_host = sections_host

    def _sync(_event: object | None = None) -> None:
        try:
            canvas.update_idletasks()
            bbox = canvas.bbox("all")
            if bbox is not None:
                canvas.configure(scrollregion=bbox)
            cw = max(canvas.winfo_width(), 1)
            canvas.itemconfigure(inner_id, width=cw)
        except tk.TclError:
            pass

    scroll_content.bind("<Configure>", _sync, add="+")
    canvas.bind("<Configure>", _sync, add="+")

    def _wheel(event: tk.Event) -> None:
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

    canvas.bind("<Enter>", lambda _e: canvas.focus_set(), add="+")
    canvas.bind("<MouseWheel>", _wheel, add="+")
    scroll_content.bind("<MouseWheel>", _wheel, add="+")
    sections_host.bind("<MouseWheel>", _wheel, add="+")

    gui._dash_scroll_outer = outer
    gui._dash_scroll_canvas = canvas
    gui._dash_scrollbar = vsb
    gui._dash_scroll_inner = scroll_content
    gui._dash_scroll_content = sections_host
    gui._dash_scroll_window_id = inner_id
    gui._sync_dashboard_scroll = _sync
    gui._sync_analysis_scroll = _sync

    gui._analysis_scroll_outer = None
    gui._analysis_scroll_canvas = None
    gui._analysis_scrollbar = None
    gui._analysis_scroll_inner = None
    gui._analysis_scroll_window_id = None

    return sections_host


def update_scroll_bottom_padding(gui, pad: int, *, extra_margin: int = 20) -> None:
    """
    Reserve space at the bottom of the scroll content so the last row clears
    the fixed playback transport bar and status bar.
    """
    spacer = getattr(gui, "_dash_scroll_bottom_spacer", None)
    if spacer is None:
        return
    total = max(48, int(pad) + int(extra_margin))
    try:
        spacer.configure(height=total)
        spacer.grid_propagate(False)
        gui._dash_scroll_bottom_pad = total
    except tk.TclError:
        return
    sync = getattr(gui, "_sync_dashboard_scroll", None)
    if sync is not None:
        sync()


def measure_fixed_chrome_height(gui) -> int:
    """Combined height of transport bar + status bar (pixels)."""
    total = 0
    for attr in ("_transport_row", "status"):
        widget = getattr(gui, attr, None)
        if widget is None:
            continue
        try:
            widget.update_idletasks()
            h = int(widget.winfo_height())
            if h > 1:
                total += h
        except tk.TclError:
            pass
    transport = getattr(gui, "_transport_row", None)
    if transport is not None:
        try:
            parent = transport.master
            if parent is not None:
                parent.update_idletasks()
                ph = int(parent.winfo_height())
                if ph > total:
                    total = ph
        except tk.TclError:
            pass
    return max(total, 56)
