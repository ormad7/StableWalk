"""Overflow scroll matches content height — no fake empty vertical space."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.tk.scroll_shell import apply_overflow_scroll_geometry


def test_overflow_scroll_does_not_inflate_short_content():
    root = tk.Tk()
    root.withdraw()
    outer = ttk.Frame(root)
    outer.pack(fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(outer, width=320, height=400, highlightthickness=0)
    vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    vsb.grid_remove()
    inner = tk.Frame(canvas, width=300, height=80)
    inner.pack_propagate(False)
    win = canvas.create_window((0, 0), window=inner, anchor="nw")
    root.update_idletasks()
    canvas.configure(width=320, height=400)
    root.update_idletasks()

    needs = apply_overflow_scroll_geometry(canvas, win, vsb, content_height=80)
    root.update_idletasks()

    assert needs is False
    assert canvas.itemcget(win, "height") == "80"
    sr = canvas.cget("scrollregion")
    assert int(float(str(sr).split()[-1])) == 80
    assert not vsb.grid_info()
    root.destroy()


def test_overflow_scroll_enables_when_content_taller_than_view():
    root = tk.Tk()
    root.withdraw()
    outer = ttk.Frame(root)
    outer.pack()
    canvas = tk.Canvas(outer, width=200, height=120, highlightthickness=0)
    vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    vsb.grid_remove()
    inner = tk.Frame(canvas)
    win = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(width=200, height=120)
    root.update_idletasks()

    needs = apply_overflow_scroll_geometry(canvas, win, vsb, content_height=500)
    root.update_idletasks()

    assert needs is True
    assert canvas.itemcget(win, "height") == "500"
    assert vsb.grid_info()
    root.destroy()


def test_overflow_scroll_sizes_inner_window_when_canvas_unmapped():
    """A 1×1 unmapped canvas must not leave the embedded window at 0×0."""
    root = tk.Tk()
    root.withdraw()
    canvas = tk.Canvas(root, width=10, height=10, highlightthickness=0)
    vsb = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
    inner = tk.Frame(canvas, width=240, height=180)
    inner.pack_propagate(False)
    tk.Label(inner, text="Advanced content").pack()
    win = canvas.create_window((0, 0), window=inner, anchor="nw")
    root.update_idletasks()

    needs = apply_overflow_scroll_geometry(canvas, win, vsb, content_height=180)
    root.update_idletasks()

    assert needs is False
    width = int(float(canvas.itemcget(win, "width") or 0))
    height = int(float(canvas.itemcget(win, "height") or 0))
    assert width >= 32
    assert height >= 32
    root.destroy()
