"""Compact help icon with click popup for dashboard metrics."""

from __future__ import annotations

import tkinter as tk

from stablewalk.ui.theme import BORDER, ELEVATED, FONT_UI_SM, FONT_UI_XS, MUTED, PANEL, TEXT


def add_metric_help_icon(parent: tk.Misc, title: str, body: str) -> tk.Label:
    """Small ? icon; click opens a plain-English help popup."""
    icon = tk.Label(
        parent,
        text="?",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        cursor="hand2",
        padx=2,
    )
    icon.pack(side=tk.LEFT)

    popup: tk.Toplevel | None = None

    def _close(_event=None) -> None:
        nonlocal popup
        if popup is not None:
            popup.destroy()
            popup = None

    def _open(_event=None) -> None:
        nonlocal popup
        _close()
        popup = tk.Toplevel(parent)
        popup.title(title)
        popup.configure(bg=ELEVATED)
        popup.resizable(False, False)
        frame = tk.Frame(popup, bg=ELEVATED, highlightthickness=1, highlightbackground=BORDER)
        frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        tk.Label(
            frame,
            text=title,
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_UI_SM,
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(8, 4))
        tk.Label(
            frame,
            text=body,
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_UI_XS,
            anchor="w",
            justify=tk.LEFT,
            wraplength=360,
        ).pack(fill=tk.X, padx=10, pady=(0, 10))
        popup.bind("<Escape>", _close)
        popup.protocol("WM_DELETE_WINDOW", _close)
        x = icon.winfo_rootx()
        y = icon.winfo_rooty() + icon.winfo_height() + 4
        popup.geometry(f"+{x}+{y}")

    icon.bind("<Button-1>", _open)
    return icon


def bind_compact_interpretation_label(lbl: tk.Label, card_text: str) -> None:
    lbl.configure(text=card_text)


def bind_metric_help_tooltip(widget: tk.Misc, title: str, body: str) -> None:
    """Hover tooltip on a metric title — no visible ? icon."""
    from stablewalk.ui.theme import create_tooltip

    create_tooltip(widget, f"{title}\n\n{body}", wraplength=340)
