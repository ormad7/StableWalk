"""Compact Research Validation block on the Results page.

Reads already-computed session products. Does not invent values or claim
that a research requirement passed.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from stablewalk.analysis.research_validation import (
    ResearchValidationSnapshot,
    collect_research_validation,
)
from stablewalk.ui.design_system.components import (
    CollapsibleSection,
    StatusBadge,
    update_status_badge,
)
from stablewalk.ui.theme import (
    ELEVATED,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    PAD_SM,
    PAD_XS,
    TEXT,
    TEXT_SECONDARY,
    create_tooltip,
)
from stablewalk.ui.viewers.chart_style import apply_chart_panel_style

_ITEM_KEYS = (
    "wire16",
    "frames_cycles",
    "acf",
    "spectrum",
    "symmetry",
    "moments",
    "reconstruction3d",
    "confidence",
)

_STATUS_TONE = {
    "computed": "ok",
    "insufficient": "warn",
    "unavailable": "neutral",
    "failed": "fail",
}


def build_research_validation_section(gui: Any, parent: tk.Misc) -> tk.Frame:
    """Install the compact Research Validation disclosure on Results."""
    section = CollapsibleSection(
        parent,
        "Research Validation",
        expanded=True,
        bg=ELEVATED,
    )
    section.frame.grid(row=4, column=0, sticky="ew", pady=(PAD_SM, PAD_XS))
    section.frame.columnconfigure(0, weight=1)
    try:
        section.frame.rowconfigure(1, weight=0)
    except tk.TclError:
        pass
    body = section.body
    body.configure(bg=ELEVATED)
    body.columnconfigure(0, weight=1)
    body.columnconfigure(1, weight=1)

    note = tk.Label(
        body,
        text=(
            "Computed session outputs only. Status is not a requirement pass/fail "
            "and not a diagnosis."
        ),
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
    )
    note.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, PAD_XS))
    gui._research_validation_note = note

    cards = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
    cards.grid(row=1, column=0, columnspan=2, sticky="ew")
    for col in range(2):
        cards.columnconfigure(col, weight=1)
    gui._research_validation_cards = {}
    for index, key in enumerate(_ITEM_KEYS):
        row, col = divmod(index, 2)
        card = _build_item_card(cards, row, col)
        gui._research_validation_cards[key] = card

    plots = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
    plots.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(PAD_XS, 0))
    plots.columnconfigure(0, weight=1)
    plots.columnconfigure(1, weight=1)

    fig_spec = Figure(figsize=(3.6, 1.45), dpi=100)
    ax_spec = fig_spec.add_subplot(111)
    apply_chart_panel_style(ax_spec)
    canvas_spec = FigureCanvasTkAgg(fig_spec, master=plots)
    spec_w = canvas_spec.get_tk_widget()
    spec_w.grid(row=0, column=0, sticky="nsew", padx=(0, PAD_XS))
    gui._research_fft_fig = fig_spec
    gui._research_fft_ax = ax_spec
    gui._research_fft_canvas = canvas_spec

    fig_3d = Figure(figsize=(3.6, 1.45), dpi=100)
    ax_3d = fig_3d.add_subplot(111)
    apply_chart_panel_style(ax_3d)
    canvas_3d = FigureCanvasTkAgg(fig_3d, master=plots)
    w3d = canvas_3d.get_tk_widget()
    w3d.grid(row=0, column=1, sticky="nsew")
    gui._research_3d_fig = fig_3d
    gui._research_3d_ax = ax_3d
    gui._research_3d_canvas = canvas_3d

    jump = ttk.Button(
        body,
        text="Open strongest moment on 3D skeleton",
        style="Compact.TButton",
        command=lambda: _jump_strongest(gui),
        state=tk.DISABLED,
        takefocus=False,
    )
    jump.grid(row=3, column=0, sticky="w", pady=(PAD_XS, 0))
    gui._research_validation_jump = jump
    create_tooltip(
        jump,
        "Seek video and the existing Overview 3D wire-body to the strongest detected moment.",
        wraplength=360,
    )

    gui._research_validation_section = section
    gui._refresh_research_validation = lambda: refresh_research_validation(gui)
    gui._download_research_report = lambda: download_research_report(gui)
    refresh_research_validation(gui)
    return section.frame


def _build_item_card(parent: tk.Misc, row: int, col: int) -> dict[str, Any]:
    host = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)
    host.grid(row=row, column=col, sticky="nsew", padx=(0 if col == 0 else PAD_XS, 0), pady=PAD_XS)
    host.columnconfigure(1, weight=1)
    title = tk.Label(host, text="—", bg=ELEVATED, fg=TEXT_SECONDARY, font=FONT_UI_XS, anchor="w")
    title.grid(row=0, column=0, sticky="w", padx=(PAD_SM, PAD_XS), pady=(PAD_XS, 0))
    badge = StatusBadge(host, "not computed", tone="neutral", bg_parent=ELEVATED)
    badge.frame.grid(row=0, column=1, sticky="e", padx=(0, PAD_SM), pady=(PAD_XS, 0))
    summary = tk.Label(
        host,
        text="",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=320,
    )
    summary.grid(row=1, column=0, columnspan=2, sticky="ew", padx=PAD_SM, pady=(0, PAD_XS))
    return {"frame": host, "title": title, "badge": badge, "summary": summary}


def refresh_research_validation(gui: Any) -> None:
    cards = getattr(gui, "_research_validation_cards", None)
    if not cards:
        return
    snapshot = collect_research_validation(gui)
    gui._research_validation_snapshot = snapshot
    for item in snapshot.items:
        card = cards.get(item.key)
        if card is None:
            continue
        try:
            card["title"].configure(text=item.title)
            card["summary"].configure(
                text=item.summary if item.status == "computed" else (item.reason or item.summary)
            )
            update_status_badge(
                card["badge"],
                _status_tone_label(item.status),
                tone=_STATUS_TONE.get(item.status, "neutral"),
            )
        except tk.TclError:
            pass
    _draw_fft(gui, snapshot)
    _draw_moment_wire(gui, snapshot)
    jump = getattr(gui, "_research_validation_jump", None)
    has_moment = snapshot.strongest_moment is not None
    if jump is not None:
        try:
            jump.configure(state=tk.NORMAL if has_moment else tk.DISABLED)
        except tk.TclError:
            pass
    btn = getattr(gui, "btn_download_research_report", None)
    if btn is not None:
        enable = bool(snapshot.source) or any(i.status != "unavailable" for i in snapshot.items)
        try:
            btn.configure(state=tk.NORMAL if enable else tk.DISABLED)
        except tk.TclError:
            pass


def _status_tone_label(status: str) -> str:
    return {
        "computed": "computed",
        "insufficient": "insufficient",
        "unavailable": "not computed",
        "failed": "failed",
    }.get(status, status)


def _draw_fft(gui: Any, snapshot: ResearchValidationSnapshot) -> None:
    ax = getattr(gui, "_research_fft_ax", None)
    canvas = getattr(gui, "_research_fft_canvas", None)
    if ax is None or canvas is None:
        return
    ax.clear()
    apply_chart_panel_style(ax)
    spec = snapshot.plot_series.get("spectrum") or {}
    drew = False
    if spec.get("fft_freq_hz") and spec.get("fft_power"):
        ax.plot(spec["fft_freq_hz"], spec["fft_power"], color="#94a3b8", lw=1.0, label="FFT")
        drew = True
    if spec.get("psd_freq_hz") and spec.get("psd"):
        ax.plot(spec["psd_freq_hz"], spec["psd"], color="#38bdf8", lw=1.4, label="PSD")
        drew = True
    if drew:
        ax.set_xlim(0, 4.0)
        ax.set_xlabel("Hz")
        ax.set_ylabel("Power")
        ax.set_title("FFT / PSD", color=TEXT, fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    else:
        item = snapshot.item("spectrum")
        ax.text(
            0.5,
            0.5,
            (item.reason if item is not None else None) or "FFT/PSD not computed",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
            wrap=True,
        )
        ax.set_axis_off()
    try:
        canvas.draw_idle()
    except Exception:
        pass


def _draw_moment_wire(gui: Any, snapshot: ResearchValidationSnapshot) -> None:
    ax = getattr(gui, "_research_3d_ax", None)
    canvas = getattr(gui, "_research_3d_canvas", None)
    if ax is None or canvas is None:
        return
    ax.clear()
    moment = snapshot.strongest_moment
    rec = getattr(gui, "gait_motion", None)
    item = snapshot.item("reconstruction3d")
    if moment is None or rec is None or item is None or item.status != "computed":
        apply_chart_panel_style(ax)
        ax.text(
            0.5,
            0.5,
            (item.reason if item is not None else None) or "No 3D moment reconstruction",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        try:
            canvas.draw_idle()
        except Exception:
            pass
        return
    try:
        from stablewalk.models.gait_wireframe import (
            SKELETON_DISPLAY_MODE_GAIT16,
            project_snapshot_to_gait_wire,
            session_body_scale,
        )
        from stablewalk.ui.asymmetric_moments_view import moment_highlight_joints
        from stablewalk.ui.viewers.gait_skeleton_renderer import (
            draw_gait_skeleton,
            setup_skeleton_axes,
        )

        snaps = list(rec.snapshots)
        frame_index = int(moment.get("frame_index") or 0)
        snap = next((s for s in snaps if int(s.frame_index) == frame_index), None)
        if snap is None:
            t = float(moment.get("time_s") or 0.0)
            snap = min(snaps, key=lambda s: abs(float(s.time_s) - t))
        display = project_snapshot_to_gait_wire(snap, body_scale=session_body_scale(rec))
        report = getattr(gui, "_gait_asymmetry_moments", None)
        obj = (list(getattr(report, "moments", None) or []) or [None])[0]
        highlight = moment_highlight_joints(obj, SKELETON_DISPLAY_MODE_GAIT16) if obj else set()
        setup_skeleton_axes(ax, display_mode=SKELETON_DISPLAY_MODE_GAIT16)
        draw_gait_skeleton(
            ax,
            display,
            clear=True,
            paused=True,
            lite_draw=True,
            highlight_joints=highlight or None,
            dim_unhighlighted=False,
            display_mode=SKELETON_DISPLAY_MODE_GAIT16,
            show_labels=False,
            show_legend=False,
            title="",
            show_ground_plane=True,
            show_com=False,
            show_support_polygon=False,
            show_contact_points=False,
            show_gait_direction=False,
        )
        ax.set_title(
            f"Wire-body · {float(snap.time_s):.2f}s · f{int(snap.frame_index)}",
            color=TEXT,
            fontsize=9,
        )
    except Exception:
        apply_chart_panel_style(ax)
        ax.text(
            0.5,
            0.5,
            "3D wire could not be drawn from the stored snapshot.",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    try:
        canvas.draw_idle()
    except Exception:
        pass


def _jump_strongest(gui: Any) -> None:
    handler = getattr(gui, "_jump_to_asymmetric_moment", None)
    if callable(handler):
        handler(0)


def download_research_report(gui: Any) -> None:
    snapshot = getattr(gui, "_research_validation_snapshot", None)
    if snapshot is None:
        snapshot = collect_research_validation(gui)
        gui._research_validation_snapshot = snapshot
    if not snapshot.source and all(item.status == "unavailable" for item in snapshot.items):
        messagebox.showinfo(
            "Research report",
            "Load and analyze a walk before downloading a research report.",
        )
        return
    stem = "research_validation"
    if snapshot.source:
        from pathlib import Path as _Path

        stem = _Path(str(snapshot.source)).stem + "_research_validation"
    path = filedialog.asksaveasfilename(
        title="Download Research Report",
        defaultextension=".zip",
        initialfile=f"{stem}.zip",
        filetypes=[("ZIP archive", "*.zip"), ("All files", "*.*")],
    )
    if not path:
        return
    try:
        from stablewalk.io.research_report_export import export_research_report_zip
        from pathlib import Path

        export_research_report_zip(snapshot, Path(path), session=gui)
        messagebox.showinfo("Research report", f"Research report saved to:\n{path}")
    except OSError as exc:
        messagebox.showerror("Export error", str(exc))


__all__ = [
    "build_research_validation_section",
    "download_research_report",
    "refresh_research_validation",
]
