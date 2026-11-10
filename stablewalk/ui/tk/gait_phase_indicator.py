"""Live gait-cycle phase indicator for Overview playback.

Updates from precomputed ``FrameDisplaySnapshot`` only — no analysis work
on the play timer.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from stablewalk.analysis.gait_cycle_phase import (
    GAIT_CYCLE_PHASE_LABELS,
    GAIT_CYCLE_PHASE_ORDER,
    format_gait_cycle_phase,
)
from stablewalk.ui.theme import (
    ACCENT,
    ELEVATED,
    FONT_CAPTION,
    FONT_UI_SEMIBOLD,
    FONT_UI_XS,
    MUTED,
    MUTED_DIM,
    PAD_SM,
    PAD_XS,
    PANEL,
    SUCCESS,
    TEXT,
)


def build_gait_phase_indicator(gui: Any, parent: tk.Misc) -> tk.Frame:
    """Install the phase chip strip + cycle progress bar under Overview viz."""
    root = tk.Frame(
        parent,
        bg=PANEL,
        highlightthickness=0,
    )
    root.columnconfigure(0, weight=1)

    header = tk.Frame(root, bg=PANEL, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(0, 0))
    header.columnconfigure(0, weight=1)

    current = tk.Label(
        header,
        text="—",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SEMIBOLD,
        anchor="w",
    )
    current.grid(row=0, column=0, sticky="w")

    pct_lbl = tk.Label(
        header,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="e",
    )
    pct_lbl.grid(row=0, column=1, sticky="e")

    chips = tk.Frame(root, bg=PANEL, highlightthickness=0)
    chips.grid(row=1, column=0, sticky="ew", padx=PAD_SM, pady=(0, 0))
    for col in range(len(GAIT_CYCLE_PHASE_ORDER)):
        chips.columnconfigure(col, weight=1, uniform="gait_phase_chip")

    chip_labels: dict[str, tk.Label] = {}
    for col, key in enumerate(GAIT_CYCLE_PHASE_ORDER):
        cell = tk.Frame(
            chips,
            bg=PANEL,
            highlightthickness=0,
        )
        cell.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else PAD_XS, 0))
        lbl = tk.Label(
            cell,
            text=GAIT_CYCLE_PHASE_LABELS[key],
            bg=PANEL,
            fg=MUTED_DIM,
            font=FONT_CAPTION,
            anchor="center",
            padx=2,
            pady=1,
        )
        lbl.pack(fill=tk.BOTH, expand=True)
        chip_labels[key] = lbl

    track = tk.Canvas(
        root,
        height=4,
        bg=ELEVATED,
        highlightthickness=0,
        borderwidth=0,
    )
    track.grid(row=2, column=0, sticky="ew", padx=PAD_SM, pady=(0, 2))
    fill_id = track.create_rectangle(0, 0, 0, 4, fill=ACCENT, width=0)

    gui._gait_phase_indicator = root
    gui._gait_phase_current_lbl = current
    gui._gait_phase_pct_lbl = pct_lbl
    gui._gait_phase_chip_labels = chip_labels
    gui._gait_phase_progress_canvas = track
    gui._gait_phase_progress_fill = fill_id
    gui._gait_phase_live_key = None
    gui._gait_phase_live_frame = None
    gui._gait_phase_live_pct = None

    def _on_track_configure(_event: object | None = None) -> None:
        update_gait_phase_progress_bar(gui, getattr(gui, "_gait_phase_live_pct", None))

    track.bind("<Configure>", _on_track_configure, add="+")
    return root


def update_gait_phase_progress_bar(gui: Any, cycle_percent: float | None) -> None:
    """Resize the thin cycle progress fill (cheap canvas coords only)."""
    canvas = getattr(gui, "_gait_phase_progress_canvas", None)
    fill_id = getattr(gui, "_gait_phase_progress_fill", None)
    if canvas is None or fill_id is None:
        return
    try:
        width = max(int(canvas.winfo_width()), 1)
        height = max(int(canvas.winfo_height()), 1)
    except tk.TclError:
        return
    if cycle_percent is None:
        canvas.coords(fill_id, 0, 0, 0, height)
        return
    frac = max(0.0, min(1.0, float(cycle_percent) / 100.0))
    canvas.coords(fill_id, 0, 0, max(1, int(width * frac)), height)


def update_gait_phase_indicator(
    gui: Any,
    frame_index: int | None,
    *,
    force: bool = False,
) -> None:
    """Refresh chips / label / bar from the session frame_display cache."""
    if frame_index is None:
        return
    frame_i = int(frame_index)
    if (
        not force
        and getattr(gui, "_gait_phase_live_frame", None) == frame_i
        and getattr(gui, "playing", False)
    ):
        return

    from stablewalk.ui.tk.playback_render_policy import configure_text_if_changed

    cache = getattr(gui, "_session_analysis", None)
    snap = cache.frame_snapshot(frame_i) if cache is not None else None

    phase_key = getattr(snap, "cycle_phase", None) if snap is not None else None
    phase_label = (
        getattr(snap, "cycle_phase_label", None)
        if snap is not None
        else None
    ) or format_gait_cycle_phase(phase_key)
    cycle_pct = getattr(snap, "cycle_percent", None) if snap is not None else None

    if snap is None or phase_key is None:
        # Fallback: still show macro dashboard phase when cycle % unknown.
        phase_label = (
            getattr(snap, "phase_dashboard", None) if snap is not None else None
        ) or "Estimating…"
        if snap is None:
            phase_label = "—"

    current = getattr(gui, "_gait_phase_current_lbl", None)
    configure_text_if_changed(current, phase_label)

    pct_lbl = getattr(gui, "_gait_phase_pct_lbl", None)
    pct_text = f"{cycle_pct:.0f}% gait cycle" if cycle_pct is not None else ""
    configure_text_if_changed(pct_lbl, pct_text)

    prev_key = getattr(gui, "_gait_phase_live_key", None)
    if phase_key != prev_key or force:
        chips: dict[str, tk.Label] = getattr(gui, "_gait_phase_chip_labels", {}) or {}
        for key, lbl in chips.items():
            active = key == phase_key
            try:
                lbl.configure(
                    bg=SUCCESS if active else PANEL,
                    fg=PANEL if active else MUTED_DIM,
                    font=FONT_UI_SEMIBOLD if active else FONT_CAPTION,
                )
                parent = lbl.master
                if isinstance(parent, tk.Frame):
                    parent.configure(
                        bg=SUCCESS if active else PANEL,
                    )
            except tk.TclError:
                pass
        gui._gait_phase_live_key = phase_key

    if cycle_pct != getattr(gui, "_gait_phase_live_pct", None) or force:
        update_gait_phase_progress_bar(gui, cycle_pct)
        gui._gait_phase_live_pct = cycle_pct

    gui._gait_phase_live_frame = frame_i
    gui._gait_phase_frame_index = frame_i

    # Mirror into Coordinate Inspector when the field exists.
    values = getattr(gui, "overview_traj_readout_values", None) or {}
    phase_field = values.get("gait_phase")
    if phase_field is not None:
        configure_text_if_changed(phase_field, phase_label)


__all__ = [
    "build_gait_phase_indicator",
    "update_gait_phase_indicator",
    "update_gait_phase_progress_bar",
]
