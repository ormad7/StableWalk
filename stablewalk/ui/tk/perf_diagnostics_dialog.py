"""Debug dialog: Performance Diagnostics report (avg / max / p95 + system)."""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any


def show_perf_diagnostics_dialog(gui: Any) -> None:
    """Show rolling timing + system counters; refresh on demand."""
    from stablewalk.ui.tk.perf_diagnostics import (
        format_perf_diagnostics_report,
        record_system_snapshot,
        set_perf_diagnostics,
    )

    # Ensure mode is on so the report includes live samples.
    if not getattr(gui, "_perf_diagnostics", False):
        set_perf_diagnostics(gui, True)

    record_system_snapshot(gui)
    text = format_perf_diagnostics_report(gui)
    print(text, flush=True)

    root = getattr(gui, "root", None)
    parent = root if root is not None else None
    win = tk.Toplevel(parent)
    win.title("Performance Diagnostics")
    win.geometry("720x560")
    try:
        if parent is not None:
            win.transient(parent)
    except tk.TclError:
        pass

    frame = ttk.Frame(win, padding=8)
    frame.pack(fill=tk.BOTH, expand=True)
    ttk.Label(
        frame,
        text=(
            "Backend: Tkinter + Matplotlib Agg (not Qt / native OpenGL). "
            "Sections: decode · video · skeleton · trajectory · 3d · mpl_draw · "
            "tk_paint · chart · compare · tab_switch · startup.\n"
            "Budgets: UI callback ≤16 ms during play · chart refresh ≤50 ms · "
            "one playback timer · one analysis worker · paint only when data changes."
        ),
        wraplength=680,
    ).pack(anchor="w")

    box = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Consolas", 9))
    box.pack(fill=tk.BOTH, expand=True, pady=(6, 8))
    box.insert("1.0", text)
    box.configure(state=tk.DISABLED)

    btn_row = ttk.Frame(frame)
    btn_row.pack(fill=tk.X)

    def _refresh() -> None:
        record_system_snapshot(gui)
        report = format_perf_diagnostics_report(gui)
        box.configure(state=tk.NORMAL)
        box.delete("1.0", tk.END)
        box.insert("1.0", report)
        box.configure(state=tk.DISABLED)
        print(report, flush=True)

    def _copy() -> None:
        report = box.get("1.0", tk.END)
        try:
            win.clipboard_clear()
            win.clipboard_append(report)
        except tk.TclError:
            pass

    ttk.Button(btn_row, text="Refresh", command=_refresh).pack(side=tk.LEFT)
    ttk.Button(btn_row, text="Copy", command=_copy).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Button(btn_row, text="Close", command=win.destroy).pack(side=tk.RIGHT)


__all__ = ["show_perf_diagnostics_dialog"]
