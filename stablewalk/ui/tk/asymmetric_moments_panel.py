"""Asymmetric Moments cards on the existing Overview 3D skeleton view.

Selectable cards for the top three detected moments. Jump-to-frame reuses
global playback so video and the existing 3D skeleton stay synchronized.
Camera controls and the Coordinate Inspector remain on the 3D Path panel.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from stablewalk.ui.asymmetric_moments_view import moment_card_payload
from stablewalk.ui.theme import (
    ACCENT_ALT,
    BORDER,
    DANGER,
    ELEVATED,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    TEXT_SECONDARY,
    create_tooltip,
)

_MAX_CARDS = 3
_SYMMETRIC_COPY = (
    "Symmetric gait: no left–right difference exceeded the documented score "
    "threshold for the minimum duration."
)
_EMPTY_COPY = _SYMMETRIC_COPY
_PENDING_COPY = "Analyze a walk to detect asymmetric moments."
_INSUFFICIENT_COPY = (
    "Insufficient data: left and right gait cycles could not be compared "
    "with enough confident frames."
)
_FAILED_COPY = "Analysis failed: asymmetric-moment detection did not complete."
_DISCLAIMER = "Measured movement differences. Not a medical diagnosis."
_GENERIC_NOTES = (
    "Left and right gait cycles are resampled to 0–100% before hip, knee, and ankle comparison.",
    "Right-limb samples are compared with left-limb samples half a gait cycle later.",
    "Descriptions are measured movement differences, not medical diagnoses.",
)


def build_asymmetric_moments_section(gui: Any, parent: tk.Misc) -> tk.Frame:
    """Install the Asymmetric Moments strip under the 3D skeleton canvas."""
    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.columnconfigure(0, weight=1)

    header = tk.Frame(host, bg=PANEL, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
    header.columnconfigure(0, weight=1)

    title = tk.Label(
        header,
        text="Asymmetric Moments",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
    )
    title.grid(row=0, column=0, sticky="w")
    create_tooltip(
        title,
        "Top three phase-normalized left–right differences. Select a card or "
        "Jump to frame to seek the video and the full 3D wire-body to that "
        "frame. Affected joints draw in red; the rest of the skeleton stays visible.",
        wraplength=360,
    )
    note = tk.Label(
        header,
        text=_DISCLAIMER,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="e",
    )
    note.grid(row=0, column=1, sticky="e", padx=(PAD_SM, 0))

    cards_row = tk.Frame(host, bg=PANEL, highlightthickness=0)
    cards_row.grid(row=1, column=0, sticky="ew")
    for col in range(_MAX_CARDS):
        cards_row.columnconfigure(col, weight=1, uniform="asym_moment")

    empty = tk.Label(
        host,
        text=_PENDING_COPY,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=520,
    )
    empty.grid(row=2, column=0, sticky="ew")

    gui._asymmetry_moments_host = host
    gui._asymmetry_moments_cards_row = cards_row
    gui._asymmetry_moments_empty = empty
    gui._asymmetry_moment_card_widgets = []
    gui._refresh_asymmetric_moments_panel = lambda: refresh_asymmetric_moments_panel(gui)

    for index in range(_MAX_CARDS):
        gui._asymmetry_moment_card_widgets.append(_build_card(gui, cards_row, index))

    refresh_asymmetric_moments_panel(gui)
    return host


def _build_card(gui: Any, parent: tk.Misc, index: int) -> dict[str, Any]:
    card = tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
        padx=PAD_SM,
        pady=PAD_XS,
    )
    card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else PAD_XS, 0))

    time_lbl = tk.Label(card, text="—", bg=ELEVATED, fg=TEXT, font=FONT_UI_SM, anchor="w")
    time_lbl.pack(fill=tk.X)
    score_lbl = tk.Label(card, text="Score —", bg=ELEVATED, fg=ACCENT_ALT, font=FONT_UI_XS, anchor="w")
    score_lbl.pack(fill=tk.X)
    joints_lbl = tk.Label(
        card,
        text="Joints: —",
        bg=ELEVATED,
        fg=TEXT_SECONDARY,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=180,
    )
    joints_lbl.pack(fill=tk.X, pady=(2, 0))
    values_lbl = tk.Label(
        card,
        text="L / R —",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=180,
    )
    values_lbl.pack(fill=tk.X)
    explain_lbl = tk.Label(
        card,
        text="",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=180,
    )
    explain_lbl.pack(fill=tk.X, pady=(2, 0))
    measures_lbl = tk.Label(
        card,
        text="",
        bg=ELEVATED,
        fg=TEXT_SECONDARY,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=180,
    )
    measures_lbl.pack(fill=tk.X)

    jump = ttk.Button(
        card,
        text="Jump to frame",
        style="Compact.TButton",
        command=lambda i=index: _on_jump(gui, i),
        takefocus=False,
    )
    jump.pack(anchor="w", pady=(PAD_XS, 0))

    def _select(_event: tk.Event | None = None, i: int = index) -> None:
        _on_select(gui, i)

    for widget in (card, time_lbl, score_lbl, joints_lbl, values_lbl, explain_lbl, measures_lbl):
        widget.bind("<Button-1>", _select, add="+")
        widget.configure(cursor="hand2")

    def _reflow(event: tk.Event) -> None:
        width = max(80, int(getattr(event, "width", 0) or 0) - 16)
        for lbl in (joints_lbl, values_lbl, explain_lbl, measures_lbl):
            try:
                lbl.configure(wraplength=width)
            except tk.TclError:
                pass

    card.bind("<Configure>", _reflow, add="+")

    return {
        "frame": card,
        "time": time_lbl,
        "score": score_lbl,
        "joints": joints_lbl,
        "values": values_lbl,
        "explain": explain_lbl,
        "measures": measures_lbl,
        "jump": jump,
    }


def analysis_attempted(gui: Any, report: Any | None = None) -> bool:
    """True after a session analysis pass, even if moments were withheld."""
    if report is not None:
        return True
    cache = getattr(gui, "_session_analysis", None)
    if cache is not None and bool(getattr(cache, "ready", False)):
        return True
    return (
        getattr(gui, "_gait_cycle", None) is not None
        or getattr(gui, "_biomech_analysis", None) is not None
    )


def empty_state_copy(gui: Any, report: Any | None = None) -> str:
    """Distinguish pending, symmetric gait, insufficient data, and analysis failed."""
    report = getattr(gui, "_gait_asymmetry_moments", None) if report is None else report
    moments = list(getattr(report, "moments", None) or []) if report is not None else []
    if moments:
        return ""
    outcome = str(getattr(report, "outcome", "") or "") if report is not None else ""
    notes = [
        str(note).strip()
        for note in (getattr(report, "notes", None) or [])
        if str(note).strip()
        and str(note).strip() not in _GENERIC_NOTES
        and not str(note).startswith("Moments require the per-frame score")
    ]
    if outcome == "moments":
        return ""
    if outcome == "symmetric_gait":
        return notes[-1] if notes and "symmetric gait" in notes[-1].lower() else _SYMMETRIC_COPY
    if outcome == "analysis_failed":
        return notes[-1] if notes else _FAILED_COPY
    if outcome == "insufficient_data":
        return notes[-1] if notes else _INSUFFICIENT_COPY
    available = bool(getattr(report, "available", False)) if report is not None else False
    if available:
        return _SYMMETRIC_COPY
    if not analysis_attempted(gui, report):
        return _PENDING_COPY
    return notes[-1] if notes else _INSUFFICIENT_COPY


def refresh_asymmetric_moments_panel(gui: Any) -> None:
    widgets = getattr(gui, "_asymmetry_moment_card_widgets", None) or []
    empty = getattr(gui, "_asymmetry_moments_empty", None)
    report = getattr(gui, "_gait_asymmetry_moments", None)
    moments = list(getattr(report, "moments", None) or []) if report is not None else []
    selected = getattr(gui, "_selected_asymmetry_moment_index", None)

    if empty is not None:
        if not moments:
            text = empty_state_copy(gui, report)
            try:
                empty.configure(text=text)
                empty.grid()
            except tk.TclError:
                pass
        else:
            try:
                empty.grid_remove()
            except tk.TclError:
                pass

    for index, card in enumerate(widgets):
        if index >= len(moments):
            try:
                card["frame"].grid_remove()
            except tk.TclError:
                pass
            continue
        payload = moment_card_payload(moments[index], index)
        try:
            card["frame"].grid()
            card["time"].configure(text=payload["timestamp_label"])
            card["score"].configure(
                text=(
                    f"Score {payload['score_label']}  ·  {payload['severity']}"
                    f"  ·  conf {payload['confidence_label']}"
                )
            )
            card["joints"].configure(text=f"Joints: {payload['joints_label']}")
            values_w = card.get("values")
            if values_w is not None:
                values_w.configure(text=payload.get("values_label") or "L / R —")
            card["explain"].configure(text=payload["explanation"])
            is_selected = selected == index
            measures = payload["measurements"] if is_selected else []
            card["measures"].configure(
                text="\n".join(measures) if measures else ""
            )
            border = DANGER if is_selected else BORDER
            card["frame"].configure(highlightbackground=border, highlightcolor=border)
        except tk.TclError:
            pass


def _on_select(gui: Any, index: int) -> None:
    handler = getattr(gui, "_select_asymmetric_moment", None)
    if callable(handler):
        handler(index)


def _on_jump(gui: Any, index: int) -> None:
    handler = getattr(gui, "_jump_to_asymmetric_moment", None)
    if callable(handler):
        handler(index)
    else:
        _on_select(gui, index)
