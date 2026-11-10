"""
Clipped Tk viewports for dashboard media and plot hosts.

A Canvas window clips child widgets to the host bounds so video frames and
matplotlib canvases cannot paint outside their panel during scroll or resize.
"""

from __future__ import annotations

import tkinter as tk

from stablewalk.ui.tk.layout_playback import should_defer_layout


def ensure_widget_fills_host(widget: tk.Misc | None) -> None:
    """Keep a canvas filling its parent via grid sticky or pack fill/expand."""
    if widget is None:
        return
    try:
        info = widget.grid_info()
    except tk.TclError:
        info = {}
    if info:
        try:
            widget.grid_configure(sticky="nsew")
        except tk.TclError:
            pass
        parent = info.get("in") or getattr(widget, "master", None)
        row = info.get("row")
        col = info.get("column")
        if parent is not None:
            try:
                if row is not None:
                    parent.rowconfigure(int(row), weight=1)
                if col is not None:
                    parent.columnconfigure(int(col), weight=1)
            except (tk.TclError, TypeError, ValueError):
                pass
        return
    try:
        pack_info = widget.pack_info()
    except tk.TclError:
        pack_info = {}
    if pack_info:
        try:
            widget.pack_configure(fill=tk.BOTH, expand=True)
        except tk.TclError:
            pass
        return
    parent = getattr(widget, "master", None)
    if parent is None:
        return
    try:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        widget.grid(row=0, column=0, sticky="nsew")
    except tk.TclError:
        try:
            widget.pack(fill=tk.BOTH, expand=True)
        except tk.TclError:
            pass


def install_clipped_viewport(
    parent: tk.Misc,
    *,
    bg: str,
    row: int = 0,
    column: int = 0,
    sticky: str = "nsew",
    padx: int | tuple[int, ...] = 0,
    pady: int | tuple[int, ...] = 0,
    min_height: int = 160,
) -> tuple[tk.Canvas, tk.Frame, int]:
    """
    Create a clipping canvas with one inner frame for child widgets.

    Returns ``(clip_canvas, inner_frame, window_item_id)``.
    """
    min_h = max(1, int(min_height))
    try:
        parent.rowconfigure(row, weight=1, minsize=min_h)
        parent.columnconfigure(column, weight=1)
    except tk.TclError:
        pass
    # Request 1×1 so Windows Tk does not treat min_height as the allocation.
    # The grid cell (and row minsize) owns the visible size.
    canvas = tk.Canvas(
        parent,
        bg=bg,
        highlightthickness=0,
        borderwidth=0,
        bd=0,
        relief="flat",
        height=1,
        width=1,
    )
    canvas.grid(row=row, column=column, sticky=sticky, padx=padx, pady=pady)
    ensure_widget_fills_host(canvas)

    inner = tk.Frame(canvas, bg=bg, highlightthickness=0, bd=0)
    window_id = canvas.create_window(0, 0, window=inner, anchor="nw")
    state = {"last": (0, 0)}

    def _sync(_event: object | None = None) -> None:
        if should_defer_layout(widget=canvas):
            return
        try:
            cw = max(int(canvas.winfo_width()), 1)
            ch = max(int(canvas.winfo_height() or 0), 1)
            # Never bump Canvas height above the grid allocation. Forcing
            # min_h overflowed short Overview panes on Windows.
            last = state["last"]
            if abs(cw - last[0]) < 2 and abs(ch - last[1]) < 2:
                return
            state["last"] = (cw, ch)
            canvas.coords(window_id, 0, 0)
            canvas.itemconfigure(window_id, width=cw, height=ch)
        except tk.TclError:
            pass

    # Canvas-only Configure — binding the inner frame caused double sync loops.
    canvas.bind("<Configure>", _sync, add="+")
    _sync()
    return canvas, inner, window_id


def sync_clipped_viewport(canvas: tk.Canvas | None, window_id: int | None) -> None:
    """Resize the inner frame window to match the clipping canvas bounds."""
    if canvas is None or window_id is None:
        return
    if should_defer_layout(widget=canvas):
        return
    try:
        cw = max(int(canvas.winfo_width()), 1)
        ch = max(int(canvas.winfo_height() or 0), 1)
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
