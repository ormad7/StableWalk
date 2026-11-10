"""
Reusable vertical scroll shell for Tk dashboard content.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.theme import BG, PAD_SM


def apply_overflow_scroll_geometry(
    canvas: tk.Canvas,
    window_id: int,
    scrollbar: tk.Misc,
    *,
    content_height: int,
) -> bool:
    """Match canvas window width to the viewport; keep height at content size.

    Stretching the inner window to the canvas height creates empty scroll
    space. Scroll only when content is taller than the view. Returns True
    when the scrollbar should be active.
    """
    try:
        view_w = max(1, int(canvas.winfo_width() or 1))
        view_h = max(1, int(canvas.winfo_height() or 1))
        req_h = max(1, int(content_height))
        # Unmapped 1×1 canvases must not shrink content to 1px (blank rail)
        # or enable a fake inner scrollbar. Still give the embedded window a
        # natural size so a PANEL canvas is not left 0×0 / empty.
        if view_w < 32 or view_h < 32:
            nat_w, nat_h = 1, req_h
            try:
                win_name = canvas.itemcget(window_id, "window")
                inner = canvas.nametowidget(str(win_name)) if win_name else None
                if inner is not None:
                    nat_w = max(int(inner.winfo_reqwidth() or 0), 1)
                    nat_h = max(req_h, int(inner.winfo_reqheight() or 0), 1)
            except (tk.TclError, KeyError, TypeError, ValueError):
                pass
            if nat_w >= 32 and nat_h >= 1:
                canvas.coords(window_id, 0, 0)
                canvas.itemconfigure(window_id, width=nat_w, height=nat_h)
                canvas.configure(scrollregion=(0, 0, nat_w, nat_h))
            return False
        canvas.coords(window_id, 0, 0)
        canvas.itemconfigure(window_id, width=view_w, height=req_h)
        canvas.configure(scrollregion=(0, 0, view_w, req_h))
        needs_scroll = req_h > view_h + 2
        if needs_scroll:
            if not scrollbar.grid_info():
                scrollbar.grid()
        else:
            if scrollbar.grid_info():
                scrollbar.grid_remove()
            canvas.yview_moveto(0.0)
        return needs_scroll
    except tk.TclError:
        return False


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
    vsb.grid_remove()
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
    sync_state = {"active": False}

    def _sync(_event: object | None = None) -> None:
        if sync_state["active"]:
            return
        sync_state["active"] = True
        try:
            canvas.update_idletasks()
            req_h = max(int(scroll_content.winfo_reqheight() or 0), 1)
            apply_overflow_scroll_geometry(
                canvas, inner_id, vsb, content_height=req_h
            )
        except tk.TclError:
            pass
        finally:
            sync_state["active"] = False

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
    for attr in ("_transport_row", "_status_bar"):
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
