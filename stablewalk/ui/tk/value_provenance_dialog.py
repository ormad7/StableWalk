"""Debug dialog: Inspect value provenance for displayed gait metrics."""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any


def show_value_provenance_dialog(gui: Any) -> None:
    """Show live provenance + static display contracts for the current playhead."""
    from stablewalk.ui.value_provenance import (
        format_contract_audit,
        get_provenance_bundle,
    )

    # Refresh joint-info provenance before opening.
    refresh = getattr(gui, "_refresh_overview_joint_information", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:
            pass
    refresh_hud = getattr(gui, "_update_overview_playback_hud", None)
    if callable(refresh_hud):
        try:
            refresh_hud()
        except Exception:
            pass
    refresh_bio = getattr(gui, "_update_biomechanics_panel", None)
    if callable(refresh_bio):
        try:
            refresh_bio()
        except Exception:
            pass

    bundle = get_provenance_bundle(gui)
    live = bundle.format_all()
    contracts = format_contract_audit()
    text = (
        "VALUE PROVENANCE (live playhead)\n"
        "================================\n"
        "Does not invent missing data. N/A rows include a reason.\n\n"
        f"{live}\n\n"
        "DISPLAY CONTRACTS (audit checklist)\n"
        "===================================\n"
        f"{contracts}\n"
    )
    print(text, flush=True)

    root = getattr(gui, "root", None)
    parent = root if root is not None else None
    win = tk.Toplevel(parent)
    win.title("Inspect value provenance")
    win.geometry("780x560")
    try:
        if parent is not None:
            win.transient(parent)
    except tk.TclError:
        pass

    frame = ttk.Frame(win, padding=8)
    frame.pack(fill=tk.BOTH, expand=True)
    ttk.Label(
        frame,
        text="Metric name · raw · display · source frame · function · units · filtering · confidence",
    ).pack(anchor="w")
    box = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Consolas", 9))
    box.pack(fill=tk.BOTH, expand=True, pady=(6, 8))
    box.insert("1.0", text)
    box.configure(state=tk.DISABLED)

    btn_row = ttk.Frame(frame)
    btn_row.pack(fill=tk.X)

    def _copy() -> None:
        try:
            win.clipboard_clear()
            win.clipboard_append(text)
        except tk.TclError:
            pass

    ttk.Button(btn_row, text="Copy", command=_copy).pack(side=tk.LEFT)
    ttk.Button(btn_row, text="Close", command=win.destroy).pack(side=tk.RIGHT)
