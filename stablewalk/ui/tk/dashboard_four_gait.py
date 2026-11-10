"""Four Gait Signals workspace — L/R thigh and shank angles plus R(0).

Presentation and async orchestration. Numerics live in
``stablewalk.analysis.four_gait_signals``. Playback only moves the playhead.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from stablewalk.analysis.four_gait_signals import (
    FOCUS_PAIRS,
    FOUR_GAIT_CHANNEL_IDS,
    FOUR_GAIT_SHORT_LABELS,
    INTERPRETATION,
    LAG_INTERPRETATION,
    XCORR_PAIRS,
    MATRIX_INTERPRETATION,
    SPECTRAL_COMPARE_INTERPRETATION,
    SPECTRUM_INTERPRETATION,
    COMPARE_PAIRS,
    FourGaitSignalsReport,
    SignalMode,
    correlation_at_lag,
    lag_seconds,
)
from stablewalk.analysis.spectral.config import DEFAULT_GAIT_BAND_HZ
from stablewalk.ui.theme import (
    ELEVATED,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    TEXT_SECONDARY,
    create_tooltip,
)
from stablewalk.ui.viewers.chart_playhead import PlayheadState, playhead_from_gui

logger = logging.getLogger(__name__)

_FIGSIZE = (7.2, 6.0)
_FIG_MIN_W = 640
_FIG_MIN_H = 520
_PLOT_PANE_MIN = 500
_SIDE_PANE_MIN = 240

_EMPTY_BY_VIEW: dict[str, str] = {
    "time": "Load a walking video to plot left/right thigh and shank angles.",
    "matrix": "Load a walking video to compute the 4×4 Pearson correlation matrix.",
    "corr": "Load a walking video to compute autocorrelation and cross-correlation.",
    "freq": "Load a walking video to compute thigh and shank spectra.",
    "compare": (
        "Load a walking video to compute spectral-shape correlation "
        "and left/right frequency comparisons."
    ),
}


def _page_selected(gui: Any) -> bool:
    from stablewalk.ui.tk.gait_correlation_tabs import (
        FOUR_GAIT_TAB_VIEWS,
        SECTION_TITLE,
        gait_corr_inner_title,
        section_selected,
    )

    inner = gait_corr_inner_title(gui)
    if getattr(gui, "_gait_corr_notebook", None) is not None:
        if inner not in FOUR_GAIT_TAB_VIEWS:
            return False
        if getattr(gui, "_motion_page_notebook", None) is None:
            return True
        return section_selected(gui)
    nb = getattr(gui, "_motion_page_notebook", None)
    if nb is None:
        return True
    try:
        text = str(nb.tab(nb.select(), "text")).strip()
    except tk.TclError:
        return False
    if text == SECTION_TITLE:
        return inner in FOUR_GAIT_TAB_VIEWS
    return text == "Four Gait Signals"


def _mode(gui: Any) -> SignalMode:
    raw = str(getattr(gui, "_four_gait_mode", "filtered") or "filtered")
    return "raw" if raw == "raw" else "filtered"


def _combined(gui: Any) -> bool:
    return bool(getattr(gui, "_four_gait_combined", False))


def _lr_overlay(gui: Any) -> bool:
    return bool(getattr(gui, "_four_gait_lr_overlay", False))


def _view(gui: Any) -> str:
    raw = str(getattr(gui, "_four_gait_view", "time") or "time")
    if raw in {"corr", "freq", "compare", "matrix"}:
        return raw
    return "time"


def _lag_samples(gui: Any) -> int:
    try:
        return int(getattr(gui, "_four_gait_lag_samples", 0) or 0)
    except (TypeError, ValueError):
        return 0


def build_four_gait_workspace(gui: Any, parent: tk.Misc) -> tk.Frame:
    """Install the Four Gait Signals page into ``parent``."""
    from stablewalk.ui.design_system.components import EmptyState
    from stablewalk.ui.tk.analysis_split import create_h_split, split_add
    from stablewalk.ui.tk.layout_scale import dpi_scale_factor

    host = ttk.Frame(parent)
    host.grid(row=0, column=0, sticky="nsew")
    parent.rowconfigure(0, weight=1)
    parent.columnconfigure(0, weight=1)
    host.columnconfigure(0, weight=1)
    host.rowconfigure(2, weight=1)
    gui._four_gait_host = host
    gui._four_gait_report = None
    gui._gait_timing = None
    gui._four_gait_token = 0
    gui._four_gait_worker = None
    gui._four_gait_pending = None
    gui._four_gait_mode = "filtered"
    gui._four_gait_combined = False
    gui._four_gait_lr_overlay = False
    gui._four_gait_view = "time"
    gui._four_gait_lag_samples = 0
    gui._four_gait_cache_key = None

    tools = tk.Frame(host, bg=ELEVATED, highlightthickness=0)
    tools.grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_SM, PAD_XS))
    tools.columnconfigure(10, weight=1)

    tk.Label(
        tools,
        text="Signal",
        bg=ELEVATED,
        fg=TEXT_SECONDARY,
        font=FONT_UI_SM,
    ).grid(row=0, column=0, padx=(PAD_SM, PAD_XS), pady=PAD_XS)

    gui.var_four_gait_mode = tk.StringVar(value="filtered")

    def _on_mode() -> None:
        gui._four_gait_mode = str(gui.var_four_gait_mode.get() or "filtered")
        render_four_gait_workspace(gui)

    ttk.Radiobutton(
        tools,
        text="Filtered",
        value="filtered",
        variable=gui.var_four_gait_mode,
        command=_on_mode,
        takefocus=False,
    ).grid(row=0, column=1, padx=PAD_XS, pady=PAD_XS)
    ttk.Radiobutton(
        tools,
        text="Raw",
        value="raw",
        variable=gui.var_four_gait_mode,
        command=_on_mode,
        takefocus=False,
    ).grid(row=0, column=2, padx=PAD_XS, pady=PAD_XS)
    create_tooltip(
        tools,
        "Filtered = short-gap fill + Savitzky–Golay (degrees, not detrended). "
        "Raw = resampled landmarks only. Correlation uses the selected series.",
        wraplength=380,
    )

    gui.var_four_gait_combined = tk.BooleanVar(value=False)

    def _on_combined() -> None:
        gui._four_gait_combined = bool(gui.var_four_gait_combined.get())
        _rebuild_signal_figure(gui)
        render_four_gait_workspace(gui)

    chk = ttk.Checkbutton(
        tools,
        text="Combined overlay",
        variable=gui.var_four_gait_combined,
        command=_on_combined,
        takefocus=False,
    )
    chk.grid(row=0, column=3, padx=(PAD_SM, PAD_XS), pady=PAD_XS)
    create_tooltip(chk, "Show all four traces on one extra graph under the 2×2 grid.")
    gui._four_gait_combined_chk = chk

    gui.var_four_gait_lr_overlay = tk.BooleanVar(value=False)

    def _on_lr_overlay() -> None:
        gui._four_gait_lr_overlay = bool(gui.var_four_gait_lr_overlay.get())
        _rebuild_signal_figure(gui)
        render_four_gait_workspace(gui)

    lr_chk = ttk.Checkbutton(
        tools,
        text="L/R overlay",
        variable=gui.var_four_gait_lr_overlay,
        command=_on_lr_overlay,
        takefocus=False,
    )
    lr_chk.grid(row=0, column=9, padx=(PAD_SM, PAD_XS), pady=PAD_XS)
    create_tooltip(
        lr_chk,
        "Overlay left and right traces for the same segment "
        "(thigh = hip→knee, shank = knee→ankle). "
        "This is segment tilt, not joint flexion.",
        wraplength=360,
    )
    gui._four_gait_lr_chk = lr_chk

    section_mode = bool(getattr(gui, "_four_gait_use_section_tabs", False))
    view_cmd = None if section_mode else (lambda: _on_four_gait_view(gui))

    view_lbl = tk.Label(
        tools,
        text="View",
        bg=ELEVATED,
        fg=TEXT_SECONDARY,
        font=FONT_UI_SM,
    )
    view_lbl.grid(row=0, column=4, padx=(PAD_SM, PAD_XS), pady=PAD_XS)
    gui.var_four_gait_view = tk.StringVar(value="time")

    time_btn = ttk.Radiobutton(
        tools,
        text="Time",
        value="time",
        variable=gui.var_four_gait_view,
        command=view_cmd,
        takefocus=False,
    )
    time_btn.grid(row=0, column=5, padx=PAD_XS, pady=PAD_XS)
    corr_btn = ttk.Radiobutton(
        tools,
        text="ACF / XCorr",
        value="corr",
        variable=gui.var_four_gait_view,
        command=view_cmd,
        takefocus=False,
    )
    corr_btn.grid(row=0, column=6, padx=PAD_XS, pady=PAD_XS)
    freq_btn = ttk.Radiobutton(
        tools,
        text="Frequency",
        value="freq",
        variable=gui.var_four_gait_view,
        command=view_cmd,
        takefocus=False,
    )
    freq_btn.grid(row=0, column=7, padx=PAD_XS, pady=PAD_XS)
    create_tooltip(
        freq_btn,
        "2×2 Welch PSD of Left/Right Thigh and Shank on the full-frame FPS "
        "grid — not 0.25 s / 0.5 s summaries. Short clips stay limited; "
        "Δf is bin spacing, not confidence.",
        wraplength=380,
    )
    cmp_btn = ttk.Radiobutton(
        tools,
        text="Compare",
        value="compare",
        variable=gui.var_four_gait_view,
        command=view_cmd,
        takefocus=False,
    )
    cmp_btn.grid(row=0, column=8, padx=PAD_XS, pady=PAD_XS)
    create_tooltip(
        cmp_btn,
        "4×4 Pearson r of unit-energy PSD shapes, left–right and "
        "upper–lower overlays, and magnitude-squared coherence when enough "
        "Welch segments exist. Not time-domain cross-correlation. "
        "Similar f0 is not proof of symmetric gait.",
        wraplength=380,
    )
    gui._four_gait_view_chrome = (view_lbl, time_btn, corr_btn, freq_btn, cmp_btn)
    if section_mode:
        for widget in gui._four_gait_view_chrome:
            try:
                widget.grid_remove()
            except tk.TclError:
                pass

    gui.lbl_four_gait_status = tk.Label(
        tools,
        text="Thigh / shank segment angles — not a clinical diagnosis",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="e",
    )
    gui.lbl_four_gait_status.grid(row=0, column=10, sticky="e", padx=PAD_SM, pady=PAD_XS)

    gui.lbl_four_gait_warnings = tk.Label(
        host,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=880,
    )
    gui.lbl_four_gait_warnings.grid(row=1, column=0, sticky="ew", padx=PAD_MD, pady=(0, PAD_XS))

    body = ttk.Frame(host)
    body.grid(row=2, column=0, sticky="nsew")
    body.columnconfigure(0, weight=1)
    body.rowconfigure(0, weight=1)
    gui._four_gait_body = body

    scale = dpi_scale_factor(parent)
    from stablewalk.ui.tk.layout_scale import scaled_px

    plot_min = scaled_px(_PLOT_PANE_MIN, scale, minimum=320)
    side_min = scaled_px(_SIDE_PANE_MIN, scale, minimum=220)
    paned = create_h_split(body)
    paned.grid(row=0, column=0, sticky="nsew")
    gui._four_gait_split = paned

    plot_pane = ttk.Frame(paned)
    plot_pane.columnconfigure(0, weight=1)
    plot_pane.rowconfigure(0, weight=1)
    gui._four_gait_plot_pane = plot_pane

    graph_host = tk.Frame(plot_pane, bg=PANEL, highlightthickness=0)
    graph_host.grid(row=0, column=0, sticky="nsew")
    graph_host.columnconfigure(0, weight=1)
    graph_host.rowconfigure(0, weight=1)
    gui._four_gait_graph_host = graph_host

    gui.fig_four_gait = Figure(figsize=_FIGSIZE, dpi=100, facecolor=PANEL)
    from stablewalk.ui.viewers.four_gait_charts import build_signal_axes

    gui._four_gait_axes = build_signal_axes(gui.fig_four_gait, combined=False)
    canvas = FigureCanvasTkAgg(gui.fig_four_gait, master=graph_host)
    widget = canvas.get_tk_widget()
    try:
        widget.configure(width=1, height=1)
    except tk.TclError:
        pass
    widget.grid(row=0, column=0, sticky="nsew")
    gui.canvas_four_gait = canvas
    gui._four_gait_canvas_widget = widget
    _attach_interactions(gui)

    gui.lbl_four_gait_empty = EmptyState(
        graph_host,
        "Load a walking video to plot left/right thigh and shank angles.",
        bg=PANEL,
    )
    gui.lbl_four_gait_empty.grid(row=0, column=0, sticky="nsew")

    side = ttk.Frame(paned)
    side.columnconfigure(0, weight=1)
    side.rowconfigure(2, weight=1)
    gui._four_gait_side = side

    lag_bar = tk.Frame(side, bg=PANEL, highlightthickness=0)
    lag_bar.grid(row=0, column=0, sticky="ew", padx=PAD_XS, pady=(PAD_XS, 0))
    lag_bar.columnconfigure(1, weight=1)
    tk.Label(
        lag_bar,
        text="C(τ)",
        bg=PANEL,
        fg=TEXT_SECONDARY,
        font=FONT_UI_SM,
    ).grid(row=0, column=0, padx=(PAD_XS, PAD_XS))
    gui.var_four_gait_lag = tk.IntVar(value=0)
    gui.scl_four_gait_lag = ttk.Scale(
        lag_bar,
        from_=-25,
        to=25,
        orient=tk.HORIZONTAL,
        variable=gui.var_four_gait_lag,
        command=lambda _v: _on_lag(gui),
    )
    gui.scl_four_gait_lag.grid(row=0, column=1, sticky="ew", padx=PAD_XS)
    gui.lbl_four_gait_lag = tk.Label(
        lag_bar,
        text="0 samples · 0.000 s",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_four_gait_lag.grid(row=1, column=0, columnspan=2, sticky="ew", padx=PAD_XS)

    heat_host = tk.Frame(side, bg=PANEL, highlightthickness=0)
    heat_host.grid(row=1, column=0, sticky="nsew", padx=PAD_XS, pady=PAD_XS)
    gui._four_gait_heat_host = heat_host
    heat_host.columnconfigure(0, weight=1)
    heat_host.rowconfigure(0, weight=1)
    gui.fig_four_gait_corr = Figure(figsize=(4.4, 3.6), dpi=100, facecolor=PANEL)
    gui.ax_four_gait_corr = gui.fig_four_gait_corr.add_subplot(111)
    heat_canvas = FigureCanvasTkAgg(gui.fig_four_gait_corr, master=heat_host)
    heat_widget = heat_canvas.get_tk_widget()
    heat_widget.grid(row=0, column=0, sticky="nsew")
    gui.canvas_four_gait_corr = heat_canvas

    table_host = ttk.Frame(side)
    table_host.grid(row=2, column=0, sticky="nsew", padx=PAD_XS, pady=PAD_XS)
    table_host.columnconfigure(0, weight=1)
    table_host.rowconfigure(0, weight=1)
    cols = ("pair", "r", "n", "note")
    tree = ttk.Treeview(
        table_host,
        columns=cols,
        show="headings",
        height=8,
        selectmode="browse",
        style="Compact.Treeview",
    )
    tree.heading("pair", text="Comparison")
    tree.heading("r", text="Pearson r")
    tree.heading("n", text="N / lag")
    tree.heading("note", text="")
    tree.column("pair", width=200, minwidth=148, stretch=True)
    tree.column("r", width=72, minwidth=64, stretch=False, anchor="e")
    tree.column("n", width=64, minwidth=52, stretch=False, anchor="e")
    tree.column("note", width=80, minwidth=64, stretch=False)
    tree.grid(row=0, column=0, sticky="nsew")
    scroll = ttk.Scrollbar(table_host, orient="vertical", command=tree.yview)
    scroll.grid(row=0, column=1, sticky="ns")
    tree.configure(yscrollcommand=scroll.set)
    gui._four_gait_tree = tree
    tree.tag_configure("focus", foreground=TEXT)

    gui.lbl_four_gait_interp = tk.Label(
        side,
        text=INTERPRETATION,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="nw",
        justify=tk.LEFT,
        wraplength=260,
    )
    gui.lbl_four_gait_interp.grid(row=3, column=0, sticky="ew", padx=PAD_XS, pady=(0, PAD_SM))

    gui._four_gait_plot_min = plot_min
    gui._four_gait_side_min = side_min
    split_add(paned, plot_pane, minsize=plot_min, stretch="always")
    split_add(paned, side, minsize=side_min, stretch="never")

    def _wrap(_event: object | None = None) -> None:
        try:
            width = int(host.winfo_width() or 0)
            side_w = int(side.winfo_width() or 0)
        except tk.TclError:
            return
        try:
            gui.lbl_four_gait_warnings.configure(wraplength=max(220, width - 48))
            gui.lbl_four_gait_interp.configure(wraplength=max(160, side_w - 16))
        except tk.TclError:
            pass

    host.bind("<Configure>", _wrap, add="+")
    side.bind("<Configure>", _wrap, add="+")
    graph_host.bind("<Configure>", lambda _e: _relayout(gui), add="+")
    heat_host.bind("<Configure>", lambda _e: _relayout(gui), add="+")

    gui._schedule_four_gait_signals = lambda force=False: schedule_four_gait_signals(
        gui, force=force
    )
    gui._update_four_gait_playhead = lambda: update_four_gait_playhead(gui)
    gui._relayout_four_gait_plots = lambda: _relayout(gui)
    gui._rebuild_four_gait_figure = lambda: _rebuild_signal_figure(gui)

    def _on_page(_event: object = None) -> None:
        schedule_four_gait_signals(gui, force=False)
        if _page_selected(gui):
            _relayout(gui)

    nb = getattr(gui, "_motion_page_notebook", None)
    if nb is not None:
        nb.bind("<<NotebookTabChanged>>", lambda e: _on_page(), add="+")

    _set_plot_visible(gui, False)
    _sync_four_gait_chrome(gui)
    return host


def _on_four_gait_view(gui: Any) -> None:
    gui._four_gait_view = str(gui.var_four_gait_view.get() or "time")
    _rebuild_signal_figure(gui)
    render_four_gait_workspace(gui)


def _sync_four_gait_chrome(gui: Any) -> None:
    view = _view(gui)
    time_view = view == "time"
    for attr in ("_four_gait_combined_chk", "_four_gait_lr_chk"):
        chk = getattr(gui, attr, None)
        if chk is None:
            continue
        try:
            if time_view:
                chk.grid()
            else:
                chk.grid_remove()
        except tk.TclError:
            pass
    _set_side_pane_visible(gui, not time_view)
    heat = getattr(gui, "_four_gait_heat_host", None)
    if heat is not None:
        try:
            # C(τ) is the side matrix on Correlation Matrix and ACF.
            # Spectral Correlation draws the 4×4 shape matrix on the main figure.
            if view in {"matrix", "corr"}:
                heat.grid()
            else:
                heat.grid_remove()
        except tk.TclError:
            pass


def _set_side_pane_visible(gui: Any, visible: bool) -> None:
    """Keep C(τ) / table off the Four Motion Signals 2×2."""
    paned = getattr(gui, "_four_gait_split", None)
    side = getattr(gui, "_four_gait_side", None)
    if paned is None or side is None:
        return
    try:
        panes = [str(p) for p in paned.panes()]
    except tk.TclError:
        return
    key = str(side)
    if visible:
        if key not in panes:
            from stablewalk.ui.tk.analysis_split import split_add

            split_add(
                paned,
                side,
                minsize=int(getattr(gui, "_four_gait_side_min", _SIDE_PANE_MIN) or _SIDE_PANE_MIN),
                stretch="never",
            )
        return
    if key in panes:
        try:
            paned.forget(side)
        except tk.TclError:
            pass


def _on_lag(gui: Any) -> None:
    try:
        gui._four_gait_lag_samples = int(round(float(gui.var_four_gait_lag.get())))
    except (TypeError, ValueError, tk.TclError):
        gui._four_gait_lag_samples = 0
    _update_lag_label(gui)
    report = getattr(gui, "_four_gait_report", None)
    if report is None:
        return
    ax = getattr(gui, "ax_four_gait_corr", None)
    heat = getattr(gui, "canvas_four_gait_corr", None)
    if ax is None or heat is None:
        return
    try:
        from stablewalk.ui.viewers.four_gait_charts import (
            draw_correlation_heatmap,
        )

        if _view(gui) != "compare":
            draw_correlation_heatmap(
                ax, report, _mode(gui), lag_samples=_lag_samples(gui)
            )
            heat.draw_idle()
    except Exception:
        logger.debug("four-gait C(lag) redraw failed", exc_info=True)
    _fill_table(gui, report, _mode(gui))


def _update_lag_label(gui: Any) -> None:
    lbl = getattr(gui, "lbl_four_gait_lag", None)
    if lbl is None:
        return
    report = getattr(gui, "_four_gait_report", None)
    samples = _lag_samples(gui)
    if report is None:
        text = f"{samples:+d} samples"
    else:
        text = (
            f"{samples:+d} samples · {lag_seconds(report, samples):+.3f} s · "
            f"{float(report.fps):.2f} Hz"
        )
    try:
        lbl.configure(text=text)
    except tk.TclError:
        pass


def _sync_lag_scale(gui: Any, report: FourGaitSignalsReport | None) -> None:
    scale = getattr(gui, "scl_four_gait_lag", None)
    if scale is None:
        return
    max_lag = 25
    if report is not None:
        max_lag = max(1, int(getattr(report, "max_lag_samples", 0) or 0))
    try:
        scale.configure(from_=-max_lag, to=max_lag)
        cur = _lag_samples(gui)
        cur = max(-max_lag, min(max_lag, cur))
        gui._four_gait_lag_samples = cur
        gui.var_four_gait_lag.set(cur)
    except tk.TclError:
        pass
    _update_lag_label(gui)


def _rebuild_signal_figure(gui: Any) -> None:
    from stablewalk.ui.viewers.four_gait_charts import (
        build_compare_axes,
        build_corr_axes,
        build_matrix_axes,
        build_signal_axes,
    )

    fig = getattr(gui, "fig_four_gait", None)
    if fig is None:
        return
    view = _view(gui)
    if view == "corr":
        gui._four_gait_axes = build_corr_axes(fig)
    elif view == "freq":
        gui._four_gait_axes = build_signal_axes(fig, combined=False)
    elif view == "compare":
        gui._four_gait_axes = build_compare_axes(fig)
    elif view == "matrix":
        gui._four_gait_axes = build_matrix_axes(fig)
    else:
        gui._four_gait_axes = build_signal_axes(
            fig, combined=_combined(gui), lr_overlay=_lr_overlay(gui)
        )
    _attach_interactions(gui)


def _attach_interactions(gui: Any) -> None:
    fig = getattr(gui, "fig_four_gait", None)
    canvas = getattr(gui, "canvas_four_gait", None)
    if fig is None or canvas is None:
        return
    try:
        from stablewalk.ui.viewers.chart_interactions import attach_chart_interactions

        attach_chart_interactions(
            fig,
            canvas,
            on_seek_time=lambda t: _seek_time(gui, t),
            hover_seek=True,
            hover_seek_enabled=lambda: not bool(getattr(gui, "playing", False)),
            sync_crosshair=False,
        )
    except Exception:
        logger.debug("four-gait chart interactions failed", exc_info=True)


def _seek_time(gui: Any, time_s: float) -> None:
    seek = getattr(gui, "_seek_motion_workspace_time", None)
    if callable(seek):
        try:
            seek(float(time_s))
        except Exception:
            logger.debug("four-gait seek failed", exc_info=True)


def _set_plot_visible(gui: Any, visible: bool) -> None:
    empty = getattr(gui, "lbl_four_gait_empty", None)
    widget = getattr(gui, "_four_gait_canvas_widget", None)
    try:
        if visible:
            if empty is not None:
                empty.grid_remove()
            if widget is not None:
                widget.grid(row=0, column=0, sticky="nsew")
        else:
            if widget is not None:
                widget.grid_remove()
            if empty is not None:
                empty.grid(row=0, column=0, sticky="nsew")
    except tk.TclError:
        pass


def _relayout(gui: Any) -> None:
    fig = getattr(gui, "fig_four_gait", None)
    canvas = getattr(gui, "canvas_four_gait", None)
    host = getattr(gui, "_four_gait_graph_host", None)
    if fig is None or canvas is None or host is None:
        return
    try:
        from stablewalk.ui.tk.dashboard_layout import _fit_figure_to_host
        from stablewalk.ui.viewers.four_gait_charts import layout_four_gait_figure

        _fit_figure_to_host(canvas, fig, host=host, min_px=80)
        layout_four_gait_figure(
            fig,
            combined=_combined(gui) or _view(gui) == "corr",
            matrix=_view(gui) == "matrix",
            overlay=_view(gui) == "time" and _lr_overlay(gui),
            spectral=_view(gui) == "compare",
        )
        from stablewalk.ui.viewers.chart_style import unclip_figure_chrome

        unclip_figure_chrome(fig)
        canvas.draw_idle()
    except Exception:
        pass
    heat = getattr(gui, "canvas_four_gait_corr", None)
    heat_fig = getattr(gui, "fig_four_gait_corr", None)
    if heat is not None and heat_fig is not None:
        try:
            from stablewalk.ui.tk.dashboard_layout import _fit_figure_to_host

            _fit_figure_to_host(
                heat, heat_fig, host=heat.get_tk_widget().master, min_px=80
            )
            heat.draw_idle()
        except Exception:
            pass


def _playhead_state(gui: Any) -> PlayheadState | None:
    try:
        return playhead_from_gui(gui)
    except Exception:
        return None


def _attach_event_crosscheck(gui: Any, cycles: Any | None) -> None:
    """Fill ACF vs heel-strike notes on a cached report. Does not rewrite ACF."""
    report = getattr(gui, "_four_gait_report", None)
    if report is None or cycles is None:
        return
    if int(getattr(report, "event_stride_n", 0) or 0) > 0:
        return
    try:
        from stablewalk.analysis.four_gait_signals import (
            compare_acf_and_event_periods,
            event_stride_period_s,
        )

        event_period, event_n, _reason = event_stride_period_s(cycles)
        report.event_stride_period_s = event_period
        report.event_stride_n = event_n
        report.acf_event_note = compare_acf_and_event_periods(
            getattr(report, "gait_period_s", None),
            event_period,
            n_event_intervals=event_n,
        )
        from stablewalk.analysis.gait_timing import refresh_shared_gait_timing

        refresh_shared_gait_timing(gui)
    except Exception:
        logger.debug("ACF / heel-strike cross-check failed", exc_info=True)


def schedule_four_gait_signals(gui: Any, *, force: bool = False) -> None:
    from stablewalk.ui.tk.gait_correlation_tabs import recording_cache_key

    rec = getattr(gui, "gait_motion", None)
    if rec is None:
        gui._four_gait_report = None
        gui._four_gait_cache_key = None
        _paint_empty(gui)
        return
    key = recording_cache_key(rec)
    if not force:
        if (
            getattr(gui, "_four_gait_cache_key", None) == key
            and getattr(gui, "_four_gait_report", None) is not None
        ):
            _attach_event_crosscheck(gui, getattr(gui, "_gait_cycle", None))
            if _should_render_four_gait(gui):
                render_four_gait_workspace(gui)
            return
        worker_obj = getattr(gui, "_four_gait_worker", None)
        if (
            worker_obj is not None
            and getattr(worker_obj, "is_alive", lambda: False)()
            and getattr(gui, "_four_gait_cache_key", None) == key
        ):
            return
    token = int(getattr(gui, "_four_gait_token", 0)) + 1
    gui._four_gait_token = token
    sequence = getattr(gui, "sequence", None)
    cycles = getattr(gui, "_gait_cycle", None)
    status = getattr(gui, "lbl_four_gait_status", None)
    if status is not None:
        try:
            status.configure(text="Computing thigh / shank angles…")
        except tk.TclError:
            pass
    gui._four_gait_pending = None
    gui._four_gait_cache_key = key

    def worker() -> None:
        try:
            from stablewalk.analysis.four_gait_signals import analyze_four_gait_signals

            report = analyze_four_gait_signals(
                rec, sequence=sequence, cycles=cycles
            )
            err = None
        except Exception as exc:  # pragma: no cover
            report = None
            err = exc
            logger.exception("Four gait signals failed")
        gui._four_gait_pending = (token, report, err)

    def drain() -> None:
        if int(getattr(gui, "_four_gait_token", 0)) != token:
            return
        pending = getattr(gui, "_four_gait_pending", None)
        if pending is None:
            worker_obj = getattr(gui, "_four_gait_worker", None)
            if worker_obj is not None and getattr(worker_obj, "is_alive", lambda: False)():
                try:
                    gui.root.after(50, drain)
                    return
                except tk.TclError:
                    return
            return
        _tok, report, err = pending
        if int(_tok) != token:
            return
        gui._four_gait_pending = None
        gui._four_gait_worker = None
        if err is not None:
            gui._four_gait_report = None
            try:
                from stablewalk.analysis.gait_timing import refresh_shared_gait_timing

                refresh_shared_gait_timing(gui)
            except Exception:
                logger.debug("shared gait timing refresh after four-gait error", exc_info=True)
            if status is not None:
                try:
                    status.configure(text=f"Four gait signals failed: {err}")
                except tk.TclError:
                    pass
            _paint_empty(gui, error=str(err))
            return
        gui._four_gait_report = report
        try:
            from stablewalk.analysis.gait_timing import refresh_shared_gait_timing

            refresh_shared_gait_timing(gui)
        except Exception:
            logger.debug("shared gait timing refresh after four-gait failed", exc_info=True)
        if status is not None:
            try:
                status.configure(
                    text="Thigh / shank segment angles — not a clinical diagnosis"
                )
            except tk.TclError:
                pass
        if _should_render_four_gait(gui) or not getattr(gui, "playing", False):
            render_four_gait_workspace(gui)

    thread = threading.Thread(target=worker, name="four-gait-signals", daemon=True)
    gui._four_gait_worker = thread
    thread.start()
    try:
        gui.root.after(50, drain)
    except tk.TclError:
        pass


def _should_render_four_gait(gui: Any) -> bool:
    """True when a four-gait sub-tab is the visible Motion page."""
    return _page_selected(gui)


def _empty_message(gui: Any, *, error: str | None = None) -> str:
    view = _view(gui)
    if error:
        return f"Could not draw this view: {error}"
    report = getattr(gui, "_four_gait_report", None)
    if report is not None and not getattr(report, "available", False):
        notes = [str(w) for w in (getattr(report, "warnings", None) or []) if w]
        if notes:
            return " · ".join(notes[:3])
        return "Four-gait analysis is unavailable for this video."
    return _EMPTY_BY_VIEW.get(view, _EMPTY_BY_VIEW["time"])


def _paint_empty(gui: Any, *, error: str | None = None) -> None:
    empty = getattr(gui, "lbl_four_gait_empty", None)
    if empty is not None:
        try:
            empty.configure(text=_empty_message(gui, error=error))
        except tk.TclError:
            pass
    _set_plot_visible(gui, False)
    warn = getattr(gui, "lbl_four_gait_warnings", None)
    if warn is not None:
        try:
            warn.configure(text="")
        except tk.TclError:
            pass
    tree = getattr(gui, "_four_gait_tree", None)
    if tree is not None:
        try:
            for item in tree.get_children():
                tree.delete(item)
        except tk.TclError:
            pass
    ax = getattr(gui, "ax_four_gait_corr", None)
    canvas = getattr(gui, "canvas_four_gait_corr", None)
    if ax is not None:
        ax.clear()
        ax.text(
            0.5,
            0.5,
            "Correlation unavailable",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        if canvas is not None:
            try:
                canvas.draw_idle()
            except Exception:
                pass


def render_four_gait_workspace(gui: Any) -> None:
    from stablewalk.ui.viewers.four_gait_charts import (
        draw_correlation_heatmap,
        layout_four_gait_figure,
        render_four_gait_compare_figure,
        render_four_gait_corr_figure,
        render_four_gait_figure,
        render_four_gait_matrix_figure,
        render_four_gait_spectrum_figure,
    )

    report: FourGaitSignalsReport | None = getattr(gui, "_four_gait_report", None)
    fig = getattr(gui, "fig_four_gait", None)
    axes = getattr(gui, "_four_gait_axes", None)
    canvas = getattr(gui, "canvas_four_gait", None)
    if fig is None or canvas is None:
        return
    _sync_four_gait_chrome(gui)
    if report is None or not getattr(report, "available", False):
        _paint_empty(gui)
        return
    if not axes:
        _rebuild_signal_figure(gui)
        axes = getattr(gui, "_four_gait_axes", None)
    if not axes:
        _paint_empty(gui, error="Plot axes were not initialized.")
        return
    _set_plot_visible(gui, True)
    _sync_lag_scale(gui, report)
    mode = _mode(gui)
    view = _view(gui)
    playhead = _playhead_state(gui) if view == "time" else None
    try:
        if view == "corr":
            if len(axes) != 6:
                _rebuild_signal_figure(gui)
                axes = getattr(gui, "_four_gait_axes", None) or axes
            render_four_gait_corr_figure(fig, axes, report, mode)
            layout_four_gait_figure(fig, combined=True)
        elif view == "freq":
            if len(axes) != 4:
                _rebuild_signal_figure(gui)
                axes = getattr(gui, "_four_gait_axes", None) or axes
            render_four_gait_spectrum_figure(fig, axes, report, mode)
            layout_four_gait_figure(fig, combined=False)
        elif view == "compare":
            if len(axes) != 5:
                _rebuild_signal_figure(gui)
                axes = getattr(gui, "_four_gait_axes", None) or axes
            render_four_gait_compare_figure(fig, axes, report, mode)
            layout_four_gait_figure(fig, spectral=True)
        elif view == "matrix":
            if len(axes) != 1:
                _rebuild_signal_figure(gui)
                axes = getattr(gui, "_four_gait_axes", None) or axes
            render_four_gait_matrix_figure(fig, axes, report, mode)
            layout_four_gait_figure(fig, matrix=True)
        else:
            overlay = _lr_overlay(gui)
            combined = _combined(gui)
            if overlay:
                want = 3 if combined else 2
            else:
                want = 5 if combined else 4
            if len(axes) != want:
                _rebuild_signal_figure(gui)
                axes = getattr(gui, "_four_gait_axes", None) or axes
            render_four_gait_figure(
                fig,
                axes,
                report,
                mode,
                playhead=playhead,
                combined=combined,
                lr_overlay=overlay,
            )
            layout_four_gait_figure(fig, combined=combined, overlay=overlay)
        canvas.draw_idle()
    except Exception as exc:
        logger.exception("four-gait render failed")
        _paint_empty(gui, error=f"{type(exc).__name__}: {exc}")
        return
    ax = getattr(gui, "ax_four_gait_corr", None)
    heat = getattr(gui, "canvas_four_gait_corr", None)
    if ax is not None and heat is not None:
        try:
            if view != "compare":
                draw_correlation_heatmap(
                    ax, report, mode, lag_samples=_lag_samples(gui)
                )
            heat.draw_idle()
        except Exception:
            logger.exception("four-gait heatmap failed")
    try:
        _fill_table(gui, report, mode)
    except Exception:
        logger.exception("four-gait table fill failed")
    interp = getattr(gui, "lbl_four_gait_interp", None)
    if interp is not None:
        try:
            if view == "freq":
                interp.configure(text=SPECTRUM_INTERPRETATION)
            elif view == "compare":
                interp.configure(text=SPECTRAL_COMPARE_INTERPRETATION)
            elif view == "corr":
                interp.configure(text=LAG_INTERPRETATION)
            elif view == "matrix":
                interp.configure(text=MATRIX_INTERPRETATION)
            else:
                interp.configure(text=INTERPRETATION)
        except tk.TclError:
            pass
    warn = getattr(gui, "lbl_four_gait_warnings", None)
    if warn is not None:
        notes = list(report.warnings)
        period = report.gait_period_s
        if period is not None:
            notes.insert(0, f"Session gait period {period:.2f} s")
        if view == "freq":
            notes.insert(
                0,
                f"PSD @ {float(report.fps):.2f} Hz (Hann, linear detrend, "
                "full-frame FPS grid). Δf is bin spacing, not confidence. "
                "Short clips stay limited — descriptive results are shown, "
                "not a reliable classification.",
            )
        if view == "compare":
            notes.insert(
                0,
                "4×4 shape r is Pearson of unit-energy PSD shapes on shared "
                "bins — not a visual match and not C(lag). Similar f0 is not "
                "proof of symmetric gait. MSC is withheld without enough "
                "Welch segments.",
            )
        if view == "corr":
            notes.insert(
                0,
                "Four ACF panels plus left–right xcorr. Period is the ACF "
                "stride peak (0.55–2.20 s) when cycles allow; heel-strike "
                "intervals are a check only. Negative r(0) is walking "
                "antiphase — not instability.",
            )
            extra = str(getattr(report, "acf_event_note", "") or "").strip()
            if extra:
                notes.insert(1, extra)
            timing = getattr(gui, "_gait_timing", None) or getattr(report, "timing", None)
            if timing is not None:
                est = getattr(timing.cycles, "estimated_from_session_stride", None)
                det = int(getattr(timing.cycles, "detected", 0) or 0)
                if est is not None:
                    notes.insert(
                        1,
                        f"Estimated {est:.1f} session-stride cycles vs {det} "
                        "detected HS–HS cycles — independent counts.",
                    )
                elif det:
                    notes.insert(
                        1,
                        f"{det} detected HS–HS cycles (not an ACF estimate).",
                    )
        if view == "matrix":
            notes.insert(
                0,
                "Large heatmap = zero-lag Pearson C(0). Side heatmap = C(τ) "
                "at the slider. Negative left–right C(0) is walking antiphase "
                "— not a diagnosis. Blank cells are not filled in.",
            )
        if view == "time":
            notes.insert(0, report.angle_definition)
        limit = 6 if view in ("freq", "compare", "matrix") else 4
        text = " · ".join(notes[:limit]) if notes else report.angle_definition
        try:
            warn.configure(text=text)
        except tk.TclError:
            pass


def update_four_gait_playhead(gui: Any) -> None:
    if not _page_selected(gui) or _view(gui) != "time":
        return
    report = getattr(gui, "_four_gait_report", None)
    if report is None:
        return
    axes = getattr(gui, "_four_gait_axes", None)
    canvas = getattr(gui, "canvas_four_gait", None)
    if not axes or canvas is None:
        return
    playhead = _playhead_state(gui)
    if playhead is None:
        return
    from stablewalk.ui.viewers.chart_playhead import update_playhead_on_axes

    try:
        if not update_playhead_on_axes(list(axes), playhead):
            render_four_gait_workspace(gui)
            return
        canvas.draw_idle()
    except Exception:
        pass


def _fmt_corr(val: Any) -> str:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(num) else f"{num:.3f}"


def _focus_pair_note(name: str, value: Any, *, zero_lag: bool) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = float("nan")
    if not np.isfinite(num):
        return "insufficient — not filled in"
    if zero_lag and name in {"thigh_lr", "shank_lr"} and num < 0.0:
        return "walking antiphase is expected"
    if zero_lag:
        return "C(0) same-time r"
    return "C(τ) lagged r"


def _fill_matrix_table(gui: Any, report: FourGaitSignalsReport, mode: SignalMode) -> None:
    tree = getattr(gui, "_four_gait_tree", None)
    if tree is None:
        return
    r0 = np.asarray(report.correlation_for(mode), dtype=float)
    n0 = np.asarray(report.n_paired_for(mode), dtype=int)
    tau = _lag_samples(gui)
    ctau = np.asarray(correlation_at_lag(report, mode, tau), dtype=float)
    lag_s = lag_seconds(report, tau)
    try:
        for item in tree.get_children():
            tree.delete(item)
        tree.insert(
            "",
            "end",
            values=("Zero-lag Pearson C(0)", "", "", "symmetric 4×4"),
            tags=("focus",),
        )
        for name, i, j, lab in FOCUS_PAIRS:
            r = r0[i, j] if i < r0.shape[0] and j < r0.shape[1] else float("nan")
            n = int(n0[i, j]) if i < n0.shape[0] and j < n0.shape[1] else 0
            tree.insert(
                "",
                "end",
                values=(lab, _fmt_corr(r), str(n), _focus_pair_note(name, r, zero_lag=True)),
                tags=("focus",),
            )
        for i, lab in enumerate(FOUR_GAIT_SHORT_LABELS):
            if i >= r0.shape[0]:
                break
            r = r0[i, i]
            n = int(n0[i, i]) if i < n0.shape[0] else 0
            note = (
                "unit diagonal"
                if np.isfinite(r)
                else "insufficient / constant"
            )
            tree.insert(
                "",
                "end",
                values=(f"C(0) {lab} vs self", _fmt_corr(r), str(n), note),
            )
        tree.insert(
            "",
            "end",
            values=(
                f"Selectable-lag C(τ)  ·  τ = {lag_s:+.3f} s",
                "",
                f"{tau:+d}",
                "not C(0)" if tau != 0 else "same as C(0) at τ = 0",
            ),
            tags=("focus",),
        )
        for name, i, j, lab in FOCUS_PAIRS:
            r = (
                ctau[i, j]
                if i < ctau.shape[0] and j < ctau.shape[1]
                else float("nan")
            )
            tree.insert(
                "",
                "end",
                values=(
                    f"C({lag_s:+.2f}s) {lab}",
                    _fmt_corr(r),
                    f"{tau:+d}",
                    _focus_pair_note(name, r, zero_lag=False),
                ),
            )
    except tk.TclError:
        pass


def _fill_table(gui: Any, report: FourGaitSignalsReport, mode: SignalMode) -> None:
    if _view(gui) == "matrix":
        _fill_matrix_table(gui, report, mode)
        return
    tree = getattr(gui, "_four_gait_tree", None)
    if tree is None:
        return
    corr = correlation_at_lag(report, mode, _lag_samples(gui))
    nmat = report.n_paired_for(mode)
    labels = FOUR_GAIT_SHORT_LABELS
    focus = {(_i, _j): lab for _name, _i, _j, lab in FOCUS_PAIRS}
    acfs = report.autocorr_for(mode)
    xcorrs = report.xcorr_for(mode)

    def _fmt_r(val: Any) -> str:
        try:
            num = float(val)
        except (TypeError, ValueError):
            return "—"
        return "—" if not np.isfinite(num) else f"{num:.3f}"

    try:
        for item in tree.get_children():
            tree.delete(item)
        period = report.gait_period_s
        contrib = getattr(report, "gait_period_channel_ids", ()) or ()
        session_note = report.gait_period_reason or (
            "Median of reliable thigh/shank ACF stride lags"
            + (f" ({', '.join(contrib)})" if contrib else "")
        )
        tree.insert(
            "",
            "end",
            values=(
                "Session stride period (Four Gait ACF)",
                "—" if period is None else f"{float(period):.3f} s",
                str(len(contrib)) if contrib else "",
                session_note,
            ),
            tags=("focus",),
        )
        timing = getattr(gui, "_gait_timing", None) or getattr(report, "timing", None)
        if timing is not None:
            ch_stride = timing.channel_stride.value_s
            ch_id = timing.channel_stride.channel_id or "selected channel"
            tree.insert(
                "",
                "end",
                values=(
                    "Selected-channel stride period",
                    "—" if ch_stride is None else f"{float(ch_stride):.3f} s",
                    ch_id,
                    "Spectral ACF on the selected DOF — not the session consensus.",
                ),
            )
            est_ch = timing.cycles.estimated_from_channel_stride
            est_sess = timing.cycles.estimated_from_session_stride
            tree.insert(
                "",
                "end",
                values=(
                    "Estimated stride cycles",
                    "—" if est_sess is None else f"{float(est_sess):.1f} session",
                    "—" if est_ch is None else f"{float(est_ch):.1f} channel",
                    "Duration / ACF stride. Not a heel-strike count.",
                ),
            )
            tree.insert(
                "",
                "end",
                values=(
                    "Detected / valid HS–HS cycles",
                    f"{timing.cycles.detected} detected",
                    f"{timing.cycles.valid} valid",
                    "Contact FSM ipsilateral HS–HS. Independent of ACF estimates.",
                ),
            )
        event_p = getattr(report, "event_stride_period_s", None)
        event_n = int(getattr(report, "event_stride_n", 0) or 0)
        tree.insert(
            "",
            "end",
            values=(
                "Heel-strike interval (independent)",
                "—" if event_p is None else f"{float(event_p):.3f} s",
                str(event_n) if event_n else "",
                str(getattr(report, "acf_event_note", "") or "not an ACF measurement"),
            ),
        )
        tree.insert(
            "",
            "end",
            values=(
                "Sampling rate",
                f"{float(report.fps):.2f} Hz",
                str(int(report.times_s.size)),
                "full-frame FPS grid",
            ),
            tags=("focus",),
        )
        lo, hi = DEFAULT_GAIT_BAND_HZ
        spectra = report.spectrum_for(mode)
        peaks_map = report.spectrum_peaks_for(mode)
        powers = report.band_power_for(mode)
        series = report.series_for(mode)
        for cid, lab in zip(FOUR_GAIT_CHANNEL_IDS, labels):
            spec = spectra.get(cid)
            col = np.asarray(series.get(cid), dtype=float)
            n_fin = int(np.isfinite(col).sum()) if col.size else 0
            if spec is None:
                tree.insert(
                    "",
                    "end",
                    values=(f"f0 · {lab}", "—", str(n_fin), "unavailable"),
                )
                continue
            f0 = spec.dominant_hz
            quality = str(getattr(spec, "quality_status", "") or spec.method or "")
            df = getattr(spec, "resolution_hz", None)
            f0_note = quality
            if quality in {"insufficient", "limited"}:
                f0_note = f"{quality} — descriptive, not a classification"
            if df is not None and np.isfinite(float(df)):
                f0_note = f"{f0_note}; Δf {float(df):.2f} Hz (bin spacing)"
            if quality == "insufficient" or f0 is None:
                f0_txt = "—"
                if quality == "insufficient":
                    f0_note = f"{quality} — not filled in"
            else:
                f0_txt = f"{float(f0):.3f} Hz"
            tree.insert(
                "",
                "end",
                values=(f"f0 · {lab}", f0_txt, str(n_fin), f0_note),
                tags=("focus",),
            )
            ent = spec.spectral_entropy
            tree.insert(
                "",
                "end",
                values=(
                    f"Entropy · {lab}",
                    "—" if ent is None else f"{float(ent):.3f}",
                    "",
                    "normalized [0, 1]",
                ),
            )
            energy = getattr(spec, "band_energy", None)
            if energy is None:
                energy = powers.get(cid)
            tree.insert(
                "",
                "end",
                values=(
                    f"Energy · {lab}",
                    "—" if energy is None else f"{float(energy):.4g}",
                    f"{lo:.2f}–{hi:.2f} Hz",
                    "∫ P(f) df",
                ),
            )
            peaks = peaks_map.get(cid) or []
            peak_txt = ", ".join(f"{hz:.2f}" for hz, _p in peaks) if peaks else "—"
            tree.insert(
                "",
                "end",
                values=(
                    f"Peaks · {lab}",
                    peak_txt,
                    str(len(peaks)),
                    "gait-band local max",
                ),
            )
            for note in spec.notes:
                text = str(note).strip()
                if not text or "searched in" in text.lower():
                    continue
                tree.insert(
                    "",
                    "end",
                    values=(f"Quality · {lab}", "", "", text),
                )
        compares = report.freq_compare_for(mode)
        spec_corr = np.asarray(report.spectral_corr_for(mode), dtype=float)
        n_bins = np.asarray(report.n_spectral_bins_for(mode), dtype=int)
        coherences = report.coherence_for(mode)
        idx = {name: (i, j) for name, i, j, _lab in FOCUS_PAIRS}
        for name, _a, _b, lab in COMPARE_PAIRS:
            cmp = compares.get(name)
            if cmp is None:
                tree.insert("", "end", values=(f"Δf0 · {lab}", "—", "", "unavailable"))
                continue
            fa, fb = cmp.f_a_hz, cmp.f_b_hz
            pair_txt = (
                "—"
                if fa is None or fb is None
                else f"{float(fa):.3f} vs {float(fb):.3f} Hz"
            )
            dhz = "—" if cmp.delta_hz is None else f"{float(cmp.delta_hz):+.3f} Hz"
            dpct = "—" if cmp.delta_pct is None else f"{float(cmp.delta_pct):+.1f}%"
            tree.insert(
                "",
                "end",
                values=(f"Δf0 · {lab}", pair_txt, dhz, dpct),
                tags=("focus",),
            )
            i, j = idx.get(name, (-1, -1))
            sr = (
                spec_corr[i, j]
                if 0 <= i < spec_corr.shape[0] and 0 <= j < spec_corr.shape[1]
                else float("nan")
            )
            nb = (
                int(n_bins[i, j])
                if 0 <= i < n_bins.shape[0] and 0 <= j < n_bins.shape[1]
                else 0
            )
            tree.insert(
                "",
                "end",
                values=(
                    f"Shape r · {lab}",
                    _fmt_r(sr),
                    str(nb),
                    "PSD shape, not C(lag)",
                ),
                tags=("focus",),
            )
            coh = coherences.get(name)
            if coh is None or not coh.sufficient:
                tree.insert(
                    "",
                    "end",
                    values=(f"MSC · {lab}", "—", "", "withheld"),
                )
                continue
            mean = "—" if coh.mean_band is None else f"{float(coh.mean_band):.3f}"
            tree.insert(
                "",
                "end",
                values=(
                    f"MSC · {lab}",
                    mean,
                    f"{int(coh.n_segments)} segs",
                    "gait-band mean γ²",
                ),
            )
        for cid, lab in zip(FOUR_GAIT_CHANNEL_IDS, labels):
            ac = acfs.get(cid)
            if ac is None:
                tree.insert("", "end", values=(f"ACF {lab}", "—", "", "unavailable"))
                continue
            withheld = bool(ac.insufficient_reason) or ac.reliability_label == "insufficient"
            ptxt = "—" if withheld or ac.gait_period_s is None else f"{float(ac.gait_period_s):.3f} s"
            note = ac.insufficient_reason or ac.reliability_label
            tree.insert("", "end", values=(f"ACF period · {lab}", ptxt, str(ac.n_samples), note or ""))
        r0 = np.asarray(report.correlation_for(mode), dtype=float)
        pair_idx = {name: (i, j) for name, i, j, _lab in FOCUS_PAIRS}
        for name, _l, _r, lab in XCORR_PAIRS:
            xc = xcorrs.get(name)
            i, j = pair_idx.get(name, (-1, -1))
            r_zero = (
                r0[i, j]
                if 0 <= i < r0.shape[0] and 0 <= j < r0.shape[1]
                else float("nan")
            )
            if xc is None or xc.peak_value is None:
                tree.insert(
                    "",
                    "end",
                    values=(f"XCorr {lab}", "—", "", f"r(0)={_fmt_r(r_zero)}"),
                    tags=("focus",),
                )
                continue
            lag = "—" if xc.peak_lag_s is None else f"{float(xc.peak_lag_s):+.3f} s"
            tree.insert(
                "",
                "end",
                values=(
                    f"XCorr peak · {lab}",
                    _fmt_r(xc.peak_value),
                    lag,
                    f"r(0)={_fmt_r(r_zero)}; best phase",
                ),
                tags=("focus",),
            )
        lag_s = lag_seconds(report, _lag_samples(gui))
        for _name, i, j, lab in FOCUS_PAIRS:
            r = corr[i, j] if i < corr.shape[0] and j < corr.shape[1] else float("nan")
            n = int(nmat[i, j]) if i < nmat.shape[0] and j < nmat.shape[1] else 0
            tree.insert(
                "",
                "end",
                values=(f"C({lag_s:+.2f}s) {lab}", _fmt_r(r), str(n), "focus"),
                tags=("focus",),
            )
    except tk.TclError:
        pass
