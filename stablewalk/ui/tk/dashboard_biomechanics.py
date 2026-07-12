"""Biomechanics tab layout for the StableWalk dashboard."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from stablewalk.ui.theme import (
    DASHBOARD_GUTTER,
    DASHBOARD_SIDE_PAD,
    FONT_METRIC,
    FONT_PANEL_HEADER,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    PANEL,
    TEXT,
)
from stablewalk.ui.tk.dashboard_layout import _bind_figure_resize, _card_title

_ANALYSIS_PANEL_PAD = (8, 8)


def build_biomechanics_tab(gui, parent: ttk.Frame) -> None:
    """Install Biomechanics tab: summary cards + synchronized plots."""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    header = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
    tk.Label(
        header,
        text="Biomechanical Analysis (estimated from pose)",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(side=tk.LEFT)

    body = ttk.Frame(parent)
    body.grid(row=1, column=0, sticky="nsew")
    body.columnconfigure(0, weight=0, minsize=220)
    body.columnconfigure(1, weight=1)
    body.rowconfigure(0, weight=1)

    sidebar = ttk.Frame(body, style="Card.TFrame", padding=DASHBOARD_SIDE_PAD)
    sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, DASHBOARD_GUTTER))

    chart_panel = ttk.LabelFrame(
        body,
        text=_card_title("Biomechanics Timeline"),
        style="Card.TLabelframe",
        padding=_ANALYSIS_PANEL_PAD,
    )
    chart_panel.grid(row=0, column=1, sticky="nsew")
    chart_panel.columnconfigure(0, weight=1)
    chart_panel.rowconfigure(0, weight=1)

    def _metric_row(parent_frame, title: str, attr: str) -> tk.Label:
        cell = tk.Frame(parent_frame, bg=PANEL, highlightthickness=0)
        cell.pack(fill=tk.X, pady=2)
        tk.Label(cell, text=title, bg=PANEL, fg=MUTED, font=FONT_UI_XS, anchor="w").pack(
            side=tk.LEFT
        )
        lbl = tk.Label(cell, text="—", bg=PANEL, fg=TEXT, font=FONT_METRIC, anchor="w")
        lbl.pack(side=tk.LEFT, padx=(6, 0))
        setattr(gui, attr, lbl)
        return lbl

    ttk.Label(
        sidebar,
        text="Summary",
        style="SideMuted.TLabel",
    ).pack(anchor=tk.W, pady=(0, 4))

    _metric_row(sidebar, "Gait Quality", "lbl_biomech_gait_quality")
    _metric_row(sidebar, "Symmetry", "lbl_biomech_symmetry")
    _metric_row(sidebar, "Stability Margin", "lbl_biomech_stability_margin")
    _metric_row(sidebar, "Cadence", "lbl_biomech_cadence")
    _metric_row(sidebar, "Walking Speed", "lbl_biomech_walking_speed")
    _metric_row(sidebar, "Video Quality", "lbl_biomech_video_quality")

    gui.lbl_biomech_rom = tk.Label(
        sidebar,
        text="ROM: —",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=200,
    )
    gui.lbl_biomech_rom.pack(anchor=tk.W, pady=(6, 0))

    gui.lbl_biomech_interpretation = tk.Label(
        sidebar,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=200,
    )
    gui.lbl_biomech_interpretation.pack(anchor=tk.W, pady=(6, 0))

    gui.fig_biomech = Figure(figsize=(7.0, 5.5), dpi=100, facecolor=PANEL)
    gui.biomech_chart_host = tk.Frame(chart_panel, bg=PANEL, highlightthickness=0)
    gui.biomech_chart_host.grid(row=0, column=0, sticky="nsew")
    gui.biomech_chart_host.columnconfigure(0, weight=1)
    gui.biomech_chart_host.rowconfigure(0, weight=1)
    gui.canvas_biomech = FigureCanvasTkAgg(gui.fig_biomech, master=gui.biomech_chart_host)
    w = gui.canvas_biomech.get_tk_widget()
    w.configure(bg=PANEL, highlightthickness=0)
    w.grid(row=0, column=0, sticky="nsew")
    _bind_figure_resize(gui.canvas_biomech, gui.fig_biomech, min_px=80)
    chart_panel.bind("<Configure>", lambda _e: gui._update_biomechanics_chart())


__all__ = ["build_biomechanics_tab"]
