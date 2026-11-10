"""Compact matplotlib navigation toolbar for dashboard plot windows."""

from __future__ import annotations

import tkinter as tk
from typing import Any

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from stablewalk.ui.theme import FONT_UI_XS, PANEL, TEXT_SECONDARY


class CompactNavigationToolbar(NavigationToolbar2Tk):
    """Default tools minus the bulky subplot-config button."""

    toolitems = tuple(
        item for item in NavigationToolbar2Tk.toolitems if item[0] != "Subplots"
    )


def install_compact_mpl_toolbar(
    canvas: Any,
    parent: tk.Misc,
    *,
    row: int = 1,
) -> Any:
    """Grid a dark, short toolbar under ``canvas``. Returns the toolbar or None."""
    try:
        toolbar = CompactNavigationToolbar(canvas, parent, pack_toolbar=False)
    except Exception:
        return None
    try:
        toolbar.configure(background=PANEL, highlightthickness=0)
    except tk.TclError:
        pass
    for child in toolbar.winfo_children():
        try:
            child.configure(background=PANEL)
        except tk.TclError:
            pass
        try:
            if str(child.winfo_class()) in {"Label", "Message"}:
                child.configure(
                    foreground=TEXT_SECONDARY,
                    font=FONT_UI_XS,
                    background=PANEL,
                )
        except tk.TclError:
            pass
    try:
        toolbar.update()
    except Exception:
        pass
    try:
        toolbar.grid(row=int(row), column=0, sticky="ew")
        parent.rowconfigure(int(row), weight=0, minsize=28)
    except tk.TclError:
        pass
    return toolbar
