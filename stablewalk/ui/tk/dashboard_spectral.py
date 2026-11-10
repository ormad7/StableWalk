"""Spectral Analysis workspace — autocorrelation, PSD, R(0), eigenvalues.

Presentation + async orchestration. Numerics live in ``stablewalk.analysis.spectral``.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from stablewalk.analysis.spectral.channels import (
    ALL_CHANNEL_SPECS,
    DEFAULT_UNIVARIATE_CHANNEL_ID,
    CHANNEL_BY_ID,
)
from stablewalk.ui.theme import (
    ELEVATED,
    FONT_KPI_TITLE,
    FONT_UI_SM,
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
from stablewalk.ui.viewers.chart_style import apply_chart_panel_style

logger = logging.getLogger(__name__)

_FIGSIZE = (9.6, 8.0)

SPECTRAL_PLOT_SPECS: tuple[tuple[str, str], ...] = (
    ("signal", "Signal"),
    ("acf", "Temporal ACF"),
    ("psd", "Welch PSD"),
    ("lr_psd", "L/R PSD"),
    ("r0", "R(0)"),
    ("eigen", "R(0) λ"),
)

# Readable 3×2 floor at 96 DPI. Smaller hosts scroll instead of clipping labels.
_SPECTRAL_FIG_MIN_W = 720
_SPECTRAL_FIG_MIN_H = 740
_SPECTRAL_PLOT_PANE_MIN = 520
_SPECTRAL_TABLE_PANE_MIN = 260


def spectral_split_minsizes(scale: float = 1.0) -> tuple[int, int]:
    from stablewalk.ui.tk.layout_scale import scaled_px

    return (
        scaled_px(_SPECTRAL_PLOT_PANE_MIN, scale, minimum=400),
        scaled_px(_SPECTRAL_TABLE_PANE_MIN, scale, minimum=200),
    )


def spectral_figure_min_px(scale: float = 1.0) -> tuple[int, int]:
    from stablewalk.ui.tk.layout_scale import scaled_px

    return (
        scaled_px(_SPECTRAL_FIG_MIN_W, scale, minimum=560),
        scaled_px(_SPECTRAL_FIG_MIN_H, scale, minimum=580),
    )

_JOINT_TO_CHANNEL: dict[str, str] = {
    "left_hip": "dof:left_hip_flexion",
    "right_hip": "dof:right_hip_flexion",
    "left_knee": "dof:left_knee_flexion",
    "right_knee": "dof:right_knee_flexion",
    "left_ankle": "dof:left_ankle_flexion",
    "right_ankle": "dof:right_ankle_flexion",
    "left_heel": "pos:left_heel:y",
    "right_heel": "pos:right_heel:y",
    "left_toe": "pos:left_toe:y",
    "right_toe": "pos:right_toe:y",
}

_CHANNEL_LABELS: tuple[tuple[str, str], ...] = tuple(
    (spec.channel_id, spec.label) for spec in ALL_CHANNEL_SPECS
)


def _channel_choices() -> tuple[str, ...]:
    return tuple(label for _cid, label in _CHANNEL_LABELS)


def _id_for_label(label: str) -> str:
    for cid, lab in _CHANNEL_LABELS:
        if lab == label:
            return cid
    return DEFAULT_UNIVARIATE_CHANNEL_ID


def _label_for_id(channel_id: str) -> str:
    spec = CHANNEL_BY_ID.get(channel_id)
    return spec.label if spec else channel_id


def build_spectral_analysis_workspace(gui: Any, parent: tk.Misc) -> tk.Frame:
    """Install the Spectral Analysis page into ``parent``."""
    from stablewalk.ui.design_system.components import EmptyState

    host = ttk.Frame(parent)
    host.grid(row=0, column=0, sticky="nsew")
    parent.rowconfigure(0, weight=1)
    parent.columnconfigure(0, weight=1)
    host.columnconfigure(0, weight=1)
    host.rowconfigure(2, weight=1)
    gui._spectral_host = host
    gui._spectral_report = None
    gui._gait_timing = None
    gui._spectral_token = 0
    gui._spectral_busy = False
    gui._spectral_channel_id = DEFAULT_UNIVARIATE_CHANNEL_ID
    gui._spectral_worker = None
    gui._spectral_pending = None

    tools = tk.Frame(host, bg=ELEVATED, highlightthickness=0)
    tools.grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_SM, PAD_XS))
    tools.columnconfigure(1, weight=1)
    tools.columnconfigure(3, weight=0)

    tk.Label(
        tools,
        text="Channel",
        bg=ELEVATED,
        fg=TEXT_SECONDARY,
        font=FONT_KPI_TITLE,
    ).grid(row=0, column=0, padx=(PAD_SM, PAD_XS), pady=PAD_XS)

    from stablewalk.ui.tk.layout_scale import combobox_char_width

    gui.var_spectral_channel = tk.StringVar(value=_label_for_id(DEFAULT_UNIVARIATE_CHANNEL_ID))
    channel_choices = _channel_choices()
    tools.columnconfigure(1, weight=1)
    combo = ttk.Combobox(
        tools,
        textvariable=gui.var_spectral_channel,
        values=channel_choices,
        state="readonly",
        width=combobox_char_width(channel_choices, minimum=24, maximum=48),
        takefocus=False,
    )
    combo.grid(row=0, column=1, sticky="ew", padx=(0, PAD_SM), pady=PAD_XS)
    gui.cmb_spectral_channel = combo
    create_tooltip(
        combo,
        "Signal used for autocorrelation and PSD. The R(0) matrix always uses "
        "left/right hip, knee, and ankle flexion when those channels exist.",
        wraplength=380,
    )

    def _on_channel(_event: object = None) -> None:
        gui._spectral_channel_id = _id_for_label(gui.var_spectral_channel.get())
        schedule_spectral_analysis(gui, force=True)

    combo.bind("<<ComboboxSelected>>", _on_channel)

    btn_csv = ttk.Button(
        tools, text="CSV", width=6, command=lambda: export_spectral(gui, "csv"), takefocus=False
    )
    btn_csv.grid(row=0, column=2, padx=(0, PAD_XS), pady=PAD_XS)
    btn_json = ttk.Button(
        tools, text="JSON", width=6, command=lambda: export_spectral(gui, "json"), takefocus=False
    )
    btn_json.grid(row=0, column=3, sticky="w", pady=PAD_XS)
    create_tooltip(btn_csv, "Export spectral series, matrices, and parameters as CSV")
    create_tooltip(btn_json, "Export the full spectral report as JSON")
    try:
        from stablewalk.ui.tk.dashboard_panel_maximize import install_maximize_button_grid

        install_maximize_button_grid(
            gui, tools, "spectral", row=0, column=5, sticky="e", padx=(PAD_SM, PAD_SM)
        )
    except Exception:
        pass

    gui.lbl_spectral_status = tk.Label(
        tools,
        text="Spectral analysis (derived) — not a clinical diagnosis",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="e",
        wraplength=280,
        justify=tk.RIGHT,
    )
    tools.columnconfigure(4, weight=1)
    gui.lbl_spectral_status.grid(row=0, column=4, sticky="e", padx=PAD_SM, pady=PAD_XS)
    from stablewalk.ui.tk.layout_scale import bind_status_wrap

    bind_status_wrap(tools, gui.lbl_spectral_status, reserve=520, minimum=100)

    gui.lbl_spectral_warnings = tk.Label(
        host,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=900,
    )
    gui.lbl_spectral_warnings.grid(row=1, column=0, sticky="ew", padx=PAD_MD, pady=(0, PAD_XS))

    def _wrap_spectral_copy(_event: object | None = None) -> None:
        try:
            width = int(host.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 80:
            return
        try:
            gui.lbl_spectral_warnings.configure(wraplength=max(220, width - 48))
        except tk.TclError:
            pass

    host.bind("<Configure>", _wrap_spectral_copy, add="+")

    body = ttk.Frame(host)
    body.grid(row=2, column=0, sticky="nsew")
    body.columnconfigure(0, weight=1)
    body.rowconfigure(0, weight=1)
    gui._spectral_body = body

    from stablewalk.ui.tk.analysis_split import (
        bind_persistent_h_sash,
        create_h_split,
        split_add,
        sync_tree_xscroll,
    )
    from stablewalk.ui.tk.layout_scale import dpi_scale_factor

    scale = dpi_scale_factor(parent)
    plot_min, table_min = spectral_split_minsizes(scale)
    paned = create_h_split(body)
    paned.grid(row=0, column=0, sticky="nsew")
    gui._spectral_split = paned

    plot_pane = ttk.Frame(paned)
    plot_pane.columnconfigure(0, weight=1)
    plot_pane.rowconfigure(1, weight=1)
    gui._spectral_plot_pane = plot_pane

    expand_bar = tk.Frame(plot_pane, bg=ELEVATED, highlightthickness=0)
    expand_bar.grid(row=0, column=0, sticky="ew", padx=PAD_XS, pady=(PAD_XS, 0))
    tk.Label(
        expand_bar,
        text="Expand plot",
        bg=ELEVATED,
        fg=TEXT_SECONDARY,
        font=FONT_UI_SM,
    ).pack(side=tk.LEFT, padx=(PAD_SM, PAD_XS))
    for plot_id, title in SPECTRAL_PLOT_SPECS:
        btn = ttk.Button(
            expand_bar,
            text=title,
            width=max(8, len(title) + 1),
            command=lambda pid=plot_id: open_spectral_plot_window(gui, pid),
            takefocus=False,
        )
        btn.pack(side=tk.LEFT, padx=(0, PAD_XS), pady=PAD_XS)
        create_tooltip(btn, f"Open {title} in a larger window. Right-click a plot to expand it.")
    gui._spectral_expand_bar = expand_bar

    graph_host = tk.Frame(plot_pane, bg=PANEL, highlightthickness=0)
    graph_host.grid(row=1, column=0, sticky="nsew")
    graph_host.columnconfigure(0, weight=1)
    graph_host.rowconfigure(0, weight=1)
    graph_host.grid_propagate(False)
    gui._spectral_graph_host = graph_host

    scroll_canvas = tk.Canvas(graph_host, bg=PANEL, highlightthickness=0, borderwidth=0)
    vsb = ttk.Scrollbar(graph_host, orient="vertical", command=scroll_canvas.yview)
    hsb = ttk.Scrollbar(graph_host, orient="horizontal", command=scroll_canvas.xview)
    scroll_canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    scroll_canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    gui._spectral_scroll_canvas = scroll_canvas
    gui._spectral_vsb = vsb
    gui._spectral_hsb = hsb

    inner = tk.Frame(scroll_canvas, bg=PANEL, highlightthickness=0)
    inner_id = scroll_canvas.create_window((0, 0), window=inner, anchor="nw")
    gui._spectral_scroll_inner = inner
    gui._spectral_scroll_window_id = inner_id

    gui.lbl_spectral_empty = EmptyState(
        graph_host,
        "Load and analyze a walking video.\n"
        "Temporal ACF, Welch PSD, and the R(0) inter-joint matrix are derived "
        "from pose time series.",
        bg=PANEL,
    )
    gui.lbl_spectral_empty.grid(row=0, column=0, sticky="nsew")

    fig = Figure(figsize=_FIGSIZE, dpi=100, facecolor=PANEL)
    axes = fig.subplots(3, 2)
    for ax in axes.ravel():
        apply_chart_panel_style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
    canvas = FigureCanvasTkAgg(fig, master=inner)
    widget = canvas.get_tk_widget()
    widget.configure(bg=PANEL, highlightthickness=0)
    widget.pack(fill=tk.BOTH, expand=True)
    gui.fig_spectral = fig
    gui.canvas_spectral = canvas
    gui._spectral_axes = (
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1],
        axes[2, 0],
        axes[2, 1],
    )
    # Layout: signal | temporal ACF ; PSD | L/R PSD ; R(0) | R(0) eigenvalues
    gui._spectral_canvas_widget = widget
    gui._spectral_expand_windows = []
    gui._spectral_expand_by_id = {}
    _attach_spectral_interactions(gui)

    side = ttk.Frame(paned)
    gui._spectral_side = side
    side.columnconfigure(0, weight=1)
    side.rowconfigure(1, weight=1)

    tk.Label(
        side,
        text="Numerical results",
        bg=PANEL,
        fg=TEXT,
        font=FONT_KPI_TITLE,
        anchor="w",
    ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(PAD_XS, PAD_XS), padx=PAD_XS)

    cols = ("metric", "value", "meaning")
    tree = ttk.Treeview(
        side,
        columns=cols,
        show="headings",
        height=16,
        selectmode="browse",
        style="Compact.Treeview",
    )
    tree.heading("metric", text="Feature")
    tree.heading("value", text="Value")
    tree.heading("meaning", text="Meaning")
    tree.column("metric", width=150, minwidth=110, stretch=True)
    tree.column("value", width=128, minwidth=96, stretch=True)
    tree.column("meaning", width=220, minwidth=96, stretch=True)
    scroll = ttk.Scrollbar(side, orient="vertical", command=tree.yview)
    xscroll = ttk.Scrollbar(side, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=scroll.set, xscrollcommand=xscroll.set)
    side.rowconfigure(1, weight=1)
    side.rowconfigure(2, weight=0)
    tree.grid(row=1, column=0, sticky="nsew")
    scroll.grid(row=1, column=1, sticky="ns")
    xscroll.grid(row=2, column=0, sticky="ew")
    gui._spectral_tree = tree
    gui._spectral_tree_xscroll = xscroll

    gui.lbl_spectral_row_detail = tk.Label(
        side,
        text="Select a row to read the complete value and description.",
        bg=PANEL,
        fg=TEXT_SECONDARY,
        font=FONT_UI_SM,
        wraplength=240,
        justify=tk.LEFT,
        anchor="nw",
    )
    gui.lbl_spectral_row_detail.grid(
        row=3, column=0, columnspan=2, sticky="ew", padx=PAD_XS, pady=(PAD_XS, 0)
    )

    def _on_spectral_row(_event: object | None = None) -> None:
        _update_spectral_row_detail(gui)

    tree.bind("<<TreeviewSelect>>", _on_spectral_row, add="+")

    formula_bar = tk.Frame(side, bg=PANEL, highlightthickness=0)
    formula_bar.grid(row=4, column=0, columnspan=2, sticky="ew", padx=PAD_XS, pady=(PAD_XS, 0))
    gui._spectral_methods_open = False
    gui.btn_spectral_methods = ttk.Button(
        formula_bar,
        text="Show methods",
        width=14,
        takefocus=False,
    )
    gui.btn_spectral_methods.pack(side=tk.LEFT)

    gui.lbl_spectral_methods = tk.Label(
        side,
        text=(
            "R(0) is the same-time inter-joint correlation matrix; its eigenvalues "
            "are spatial modes of that matrix. Temporal ACF ρ(τ) and Welch PSD "
            "are computed on the selected channel after linear detrend and z-score. "
            "They are not interchangeable. Cadence here is a spectral estimate, "
            "not Overview cadence. Not a clinical diagnosis."
        ),
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        wraplength=260,
        justify=tk.LEFT,
        anchor="w",
    )
    gui.lbl_spectral_methods.grid(
        row=5, column=0, columnspan=2, sticky="ew", pady=(PAD_XS, PAD_XS), padx=PAD_XS
    )
    gui.lbl_spectral_methods.grid_remove()

    def _toggle_spectral_methods() -> None:
        open_ = not bool(getattr(gui, "_spectral_methods_open", False))
        gui._spectral_methods_open = open_
        try:
            if open_:
                gui.lbl_spectral_methods.grid()
                gui.btn_spectral_methods.configure(text="Hide methods")
            else:
                gui.lbl_spectral_methods.grid_remove()
                gui.btn_spectral_methods.configure(text="Show methods")
        except tk.TclError:
            pass

    gui.btn_spectral_methods.configure(command=_toggle_spectral_methods)

    split_add(paned, plot_pane, minsize=plot_min)
    split_add(paned, side, minsize=table_min)

    gui._place_spectral_sash = bind_persistent_h_sash(
        paned,
        gui,
        frac_attr="_spectral_sash_frac",
        drag_attr="_spectral_sash_dragging",
        default=0.70,
        min_left=plot_min,
        min_right=table_min,
        min_width=500,
    )

    def _relayout_spectral_plots(_event: object | None = None) -> None:
        relayout_spectral_plots(gui)

    gui._relayout_spectral_plots = _relayout_spectral_plots
    graph_host.bind("<Configure>", _relayout_spectral_plots, add="+")

    def _wrap_spectral_side(_event: object | None = None) -> None:
        try:
            width = int(side.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 40:
            return
        wrap = max(140, width - 16)
        try:
            gui.lbl_spectral_methods.configure(wraplength=wrap)
            detail = getattr(gui, "lbl_spectral_row_detail", None)
            if detail is not None:
                detail.configure(wraplength=wrap)
        except tk.TclError:
            pass
        sync_tree_xscroll(tree, xscroll, side)

    side.bind("<Configure>", _wrap_spectral_side, add="+")
    _bind_spectral_plot_expand(gui)
    _set_spectral_plot_visible(gui, False)

    gui._schedule_spectral_analysis = lambda force=False: schedule_spectral_analysis(gui, force=force)
    gui._update_spectral_playhead = lambda: update_spectral_playhead(gui)
    gui._export_spectral_csv = lambda: export_spectral(gui, "csv")
    gui._export_spectral_json = lambda: export_spectral(gui, "json")
    gui._sync_spectral_channel_from_joint = lambda item_id: sync_channel_from_joint(gui, item_id)

    def _on_page(_event: object = None) -> None:
        schedule_spectral_analysis(gui, force=False)

    nb = getattr(gui, "_motion_page_notebook", None)
    if nb is not None:
        nb.bind("<<NotebookTabChanged>>", lambda e: _on_page(), add="+")

    return host


def _spectral_seek_time(gui: Any, time_s: float) -> None:
    seek = getattr(gui, "_seek_motion_workspace_time", None)
    if callable(seek):
        try:
            seek(float(time_s))
        except Exception:
            logger.debug("spectral seek failed", exc_info=True)


def _attach_spectral_interactions(gui: Any) -> None:
    fig = getattr(gui, "fig_spectral", None)
    canvas = getattr(gui, "canvas_spectral", None)
    if fig is None or canvas is None:
        return
    try:
        from stablewalk.ui.viewers.chart_interactions import (
            attach_chart_interactions,
        )

        attach_chart_interactions(
            fig,
            canvas,
            on_seek_time=lambda t: _spectral_seek_time(gui, t),
            hover_seek=True,
            hover_seek_enabled=lambda: not bool(getattr(gui, "playing", False)),
            sync_crosshair=False,
        )
    except Exception:
        logger.debug("spectral chart interactions failed", exc_info=True)


def _set_spectral_plot_visible(gui: Any, visible: bool) -> None:
    empty = getattr(gui, "lbl_spectral_empty", None)
    scroll = getattr(gui, "_spectral_scroll_canvas", None)
    vsb = getattr(gui, "_spectral_vsb", None)
    hsb = getattr(gui, "_spectral_hsb", None)
    try:
        if visible:
            if empty is not None:
                empty.grid_remove()
            if scroll is not None:
                scroll.grid(row=0, column=0, sticky="nsew")
            if vsb is not None:
                vsb.grid(row=0, column=1, sticky="ns")
            if hsb is not None:
                hsb.grid(row=1, column=0, sticky="ew")
        else:
            if scroll is not None:
                scroll.grid_remove()
            if vsb is not None:
                vsb.grid_remove()
            if hsb is not None:
                hsb.grid_remove()
            if empty is not None:
                empty.grid(row=0, column=0, sticky="nsew")
    except tk.TclError:
        pass


def relayout_spectral_plots(gui: Any) -> None:
    """Size the 3×2 figure to a readable floor; scroll when the host is smaller."""
    if getattr(gui, "_spectral_relayouting", False):
        return
    fig = getattr(gui, "fig_spectral", None)
    canvas = getattr(gui, "canvas_spectral", None)
    widget = getattr(gui, "_spectral_canvas_widget", None)
    host = getattr(gui, "_spectral_graph_host", None)
    scroll = getattr(gui, "_spectral_scroll_canvas", None)
    inner = getattr(gui, "_spectral_scroll_inner", None)
    inner_id = getattr(gui, "_spectral_scroll_window_id", None)
    if fig is None or canvas is None or widget is None or host is None:
        return
    if getattr(gui, "lbl_spectral_empty", None) is not None:
        try:
            if str(gui.lbl_spectral_empty.winfo_manager() or "") == "grid":
                report = getattr(gui, "_spectral_report", None)
                if report is None or not getattr(report, "available", False):
                    return
        except tk.TclError:
            pass
    gui._spectral_relayouting = True
    try:
        from stablewalk.ui.tk.layout_scale import dpi_scale_factor
        from stablewalk.ui.viewers.spectral_charts import layout_spectral_figure

        try:
            host.update_idletasks()
            host_w = int(host.winfo_width() or 0)
            host_h = int(host.winfo_height() or 0)
        except tk.TclError:
            return
        # Leave room for the scrollbars so fill-to-host does not always overflow.
        view_w = max(80, host_w - 18)
        view_h = max(80, host_h - 18)
        if view_w < 80 or view_h < 80:
            return
        scale = dpi_scale_factor(getattr(gui, "root", None) or host)
        min_w, min_h = spectral_figure_min_px(scale)
        fig_w = max(min_w, view_w)
        fig_h = max(min_h, view_h)
        dpi = float(fig.get_dpi() or 100.0)
        try:
            cur_w, cur_h = fig.get_size_inches()
            if abs(cur_w * dpi - fig_w) < 2.0 and abs(cur_h * dpi - fig_h) < 2.0:
                if scroll is not None and inner_id is not None:
                    scroll.configure(scrollregion=(0, 0, fig_w, fig_h))
                return
        except Exception:
            pass
        try:
            fig.set_size_inches(fig_w / dpi, fig_h / dpi, forward=False)
        except Exception:
            return
        try:
            canvas.resize(type("E", (), {"width": int(fig_w), "height": int(fig_h)})())
        except (TypeError, AttributeError, tk.TclError):
            try:
                fig.set_size_inches(fig_w / dpi, fig_h / dpi, forward=True)
            except Exception:
                pass
        try:
            widget.configure(width=int(fig_w), height=int(fig_h))
        except tk.TclError:
            pass
        if inner is not None:
            try:
                inner.configure(width=int(fig_w), height=int(fig_h))
            except tk.TclError:
                pass
        if scroll is not None and inner_id is not None:
            try:
                scroll.itemconfigure(inner_id, width=int(fig_w), height=int(fig_h))
                scroll.configure(scrollregion=(0, 0, int(fig_w), int(fig_h)))
            except tk.TclError:
                pass
        layout_spectral_figure(fig, single=False)
        try:
            canvas.draw_idle()
        except Exception:
            pass
        hsb = getattr(gui, "_spectral_hsb", None)
        vsb = getattr(gui, "_spectral_vsb", None)
        if hsb is not None:
            try:
                if int(fig_w) <= view_w + 2:
                    hsb.grid_remove()
                else:
                    hsb.grid()
            except tk.TclError:
                pass
        if vsb is not None:
            try:
                if int(fig_h) <= view_h + 2:
                    vsb.grid_remove()
                else:
                    vsb.grid()
            except tk.TclError:
                pass
    finally:
        gui._spectral_relayouting = False


def _bind_spectral_plot_expand(gui: Any) -> None:
    canvas = getattr(gui, "canvas_spectral", None)
    if canvas is None:
        return
    if getattr(canvas, "_sw_spectral_expand_cid", None) is not None:
        return

    def _on_press(event: object) -> None:
        button = int(getattr(event, "button", 0) or 0)
        if button != 3:
            return
        inaxes = getattr(event, "inaxes", None)
        axes = getattr(gui, "_spectral_axes", None) or ()
        for ax, (plot_id, _title) in zip(axes, SPECTRAL_PLOT_SPECS, strict=False):
            if inaxes is ax:
                open_spectral_plot_window(gui, plot_id)
                return

    try:
        canvas._sw_spectral_expand_cid = canvas.mpl_connect("button_press_event", _on_press)
    except Exception:
        pass


def open_spectral_plot_window(gui: Any, plot_id: str) -> None:
    """Open one Spectral Analysis plot in a larger, independently zoomable window."""
    plot_id = str(plot_id)
    titles = dict(SPECTRAL_PLOT_SPECS)
    if plot_id not in titles:
        return
    existing = (getattr(gui, "_spectral_expand_by_id", None) or {}).get(plot_id)
    if existing is not None:
        window = existing.get("window")
        try:
            if window is not None and window.winfo_exists():
                window.lift()
                window.focus_force()
                _refresh_spectral_expand_window(gui, existing)
                return
        except tk.TclError:
            pass
    root = getattr(gui, "root", None)
    try:
        dlg = tk.Toplevel(root) if root is not None else tk.Toplevel()
    except tk.TclError:
        return
    dlg.title(f"{titles[plot_id]} — Spectral Analysis")
    dlg.minsize(720, 480)
    dlg.geometry("1040x700")
    dlg.configure(bg=PANEL)
    shell = ttk.Frame(dlg)
    shell.pack(fill=tk.BOTH, expand=True)
    shell.columnconfigure(0, weight=1)
    shell.rowconfigure(1, weight=1)
    header = tk.Frame(shell, bg=ELEVATED, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew")
    tk.Label(
        header,
        text=titles[plot_id],
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_KPI_TITLE,
        anchor="w",
    ).pack(side=tk.LEFT, padx=PAD_SM, pady=PAD_XS)
    ttk.Button(header, text="Close", width=8, command=dlg.destroy, takefocus=False).pack(
        side=tk.RIGHT, padx=PAD_SM, pady=PAD_XS
    )
    host = tk.Frame(shell, bg=PANEL, highlightthickness=0)
    host.grid(row=1, column=0, sticky="nsew")
    host.columnconfigure(0, weight=1)
    host.rowconfigure(0, weight=1)
    plot_host = tk.Frame(host, bg=PANEL, highlightthickness=0)
    plot_host.grid(row=0, column=0, sticky="nsew")
    plot_host.columnconfigure(0, weight=1)
    plot_host.rowconfigure(0, weight=1)
    fig = Figure(figsize=(10.0, 6.4), dpi=100, facecolor=PANEL)
    ax = fig.add_subplot(111)
    apply_chart_panel_style(ax)
    canvas = FigureCanvasTkAgg(fig, master=plot_host)
    widget = canvas.get_tk_widget()
    widget.configure(bg=PANEL, highlightthickness=0)
    widget.grid(row=0, column=0, sticky="nsew")
    try:
        from stablewalk.ui.tk.mpl_toolbar import install_compact_mpl_toolbar

        install_compact_mpl_toolbar(canvas, host, row=1)
    except Exception:
        pass
    try:
        from stablewalk.ui.viewers.chart_interactions import attach_chart_interactions

        attach_chart_interactions(
            fig,
            canvas,
            on_seek_time=(lambda t: _spectral_seek_time(gui, t)) if plot_id == "signal" else None,
            hover_seek=plot_id == "signal",
            hover_seek_enabled=lambda: not bool(getattr(gui, "playing", False)),
            sync_crosshair=False,
        )
    except Exception:
        pass
    try:
        from stablewalk.ui.tk.dashboard_layout import _bind_figure_resize
        from stablewalk.ui.viewers.spectral_charts import SPECTRAL_SINGLE_MARGINS

        _bind_figure_resize(canvas, fig, margins=SPECTRAL_SINGLE_MARGINS, min_px=80)
    except Exception:
        pass
    record = {
        "plot_id": plot_id,
        "window": dlg,
        "fig": fig,
        "ax": ax,
        "canvas": canvas,
    }
    by_id = getattr(gui, "_spectral_expand_by_id", None)
    if by_id is None:
        by_id = {}
        gui._spectral_expand_by_id = by_id
    by_id[plot_id] = record
    windows = getattr(gui, "_spectral_expand_windows", None)
    if windows is None:
        windows = []
        gui._spectral_expand_windows = windows
    windows.append(record)

    def _on_close() -> None:
        try:
            by_id.pop(plot_id, None)
        except Exception:
            pass
        try:
            windows.remove(record)
        except ValueError:
            pass
        try:
            dlg.destroy()
        except tk.TclError:
            pass

    dlg.protocol("WM_DELETE_WINDOW", _on_close)
    _refresh_spectral_expand_window(gui, record)


def _refresh_spectral_expand_window(gui: Any, record: dict[str, Any]) -> None:
    from stablewalk.ui.viewers.spectral_charts import draw_spectral_plot, layout_spectral_figure

    report = getattr(gui, "_spectral_report", None)
    ax = record.get("ax")
    fig = record.get("fig")
    canvas = record.get("canvas")
    plot_id = str(record.get("plot_id") or "")
    if ax is None or fig is None or canvas is None:
        return
    channel_id = str(getattr(gui, "_spectral_channel_id", DEFAULT_UNIVARIATE_CHANNEL_ID))
    if report is None or not getattr(report, "available", False):
        ax.clear()
        apply_chart_panel_style(ax)
        ax.text(
            0.5,
            0.5,
            "Load a walking video to compute spectral analysis.",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
    else:
        draw_spectral_plot(ax, plot_id, report, channel_id, compact=False)
        if plot_id == "signal":
            playhead = _playhead_state(gui)
            if playhead is not None:
                from stablewalk.ui.viewers.chart_playhead import apply_playhead_to_axes

                apply_playhead_to_axes([ax], playhead)
    layout_spectral_figure(fig, single=True)
    try:
        canvas.draw_idle()
    except Exception:
        pass


def _refresh_spectral_expand_windows(gui: Any) -> None:
    by_id = getattr(gui, "_spectral_expand_by_id", None) or {}
    for record in list(by_id.values()):
        window = record.get("window")
        try:
            if window is None or not window.winfo_exists():
                continue
        except tk.TclError:
            continue
        _refresh_spectral_expand_window(gui, record)


def sync_channel_from_joint(gui: Any, item_id: str | None) -> None:
    if not item_id:
        return
    cid = _JOINT_TO_CHANNEL.get(str(item_id))
    if not cid:
        return
    gui._spectral_channel_id = cid
    var = getattr(gui, "var_spectral_channel", None)
    if var is not None:
        try:
            var.set(_label_for_id(cid))
        except tk.TclError:
            pass


def _spectral_page_selected(gui: Any) -> bool:
    nb = getattr(gui, "_motion_page_notebook", None)
    if nb is None:
        return False
    try:
        text = str(nb.tab(nb.select(), "text"))
    except tk.TclError:
        return False
    return text.strip() == "Spectral Analysis"


def _spectral_features_page_selected(gui: Any) -> bool:
    nb = getattr(gui, "_motion_page_notebook", None)
    if nb is None:
        return False
    try:
        text = str(nb.tab(nb.select(), "text"))
    except tk.TclError:
        return False
    return text.strip() == "Spectral Features"


def schedule_spectral_analysis(gui: Any, *, force: bool = False) -> None:
    rec = getattr(gui, "gait_motion", None)
    if rec is None:
        gui._spectral_report = None
        _paint_empty(gui, None)
        render_fn = getattr(gui, "_render_spectral_features", None)
        if callable(render_fn):
            render_fn()
        return
    if gui._spectral_report is not None and not force:
        if _spectral_page_selected(gui):
            render_spectral_workspace(gui)
        if _spectral_features_page_selected(gui):
            render_fn = getattr(gui, "_render_spectral_features", None)
            if callable(render_fn):
                render_fn()
        return
    token = int(getattr(gui, "_spectral_token", 0)) + 1
    gui._spectral_token = token
    channel_id = str(getattr(gui, "_spectral_channel_id", DEFAULT_UNIVARIATE_CHANNEL_ID))
    sequence = getattr(gui, "sequence", None)
    cycles = getattr(gui, "_gait_cycle", None)
    status = getattr(gui, "lbl_spectral_status", None)
    if status is not None:
        try:
            status.configure(text="Computing spectral analysis…")
        except tk.TclError:
            pass

    gui._spectral_pending = None

    def worker() -> None:
        try:
            from stablewalk.analysis.spectral.report import analyze_spectral

            report = analyze_spectral(
                rec,
                sequence=sequence,
                univariate_channel_id=channel_id,
                cycles=cycles,
            )
            err = None
        except Exception as exc:  # pragma: no cover - GUI guard
            report = None
            err = exc
            logger.exception("Spectral analysis failed")
        # Do not call Tk from this thread — Windows Tk is not thread-safe.
        gui._spectral_pending = (token, report, err)

    def drain() -> None:
        if int(getattr(gui, "_spectral_token", 0)) != token:
            return
        pending = getattr(gui, "_spectral_pending", None)
        if pending is None:
            worker_obj = getattr(gui, "_spectral_worker", None)
            if worker_obj is not None and getattr(worker_obj, "is_alive", lambda: False)():
                try:
                    gui.root.after(50, drain)
                except Exception:
                    pass
            return
        p_token, report, err = pending
        if p_token != token:
            return
        gui._spectral_pending = None
        if err is not None:
            if status is not None:
                try:
                    status.configure(text=f"Spectral analysis failed: {err}")
                except tk.TclError:
                    pass
            gui._spectral_error = str(err)
            try:
                from stablewalk.analysis.gait_pipeline_trace import log_stage

                log_stage("fft_psd", "failed", error=type(err).__name__)
            except Exception:
                pass
            refresh_rv = getattr(gui, "_refresh_research_validation", None)
            if callable(refresh_rv):
                try:
                    refresh_rv()
                except Exception:
                    logger.debug("research validation refresh after spectral failure", exc_info=True)
            return
        gui._spectral_error = None
        gui._spectral_report = report
        try:
            from stablewalk.analysis.gait_timing import refresh_shared_gait_timing

            refresh_shared_gait_timing(gui)
        except Exception:
            logger.debug("shared gait timing refresh after spectral failed", exc_info=True)
        try:
            from stablewalk.analysis.gait_pipeline_trace import log_stage

            available = bool(getattr(report, "available", False)) if report is not None else False
            log_stage(
                "fft_psd",
                "complete" if available else "insufficient",
                available=available,
                channels=len(getattr(report, "channels", None) or ()),
            )
        except Exception:
            logger.debug("spectral stage log failed", exc_info=True)
        render_spectral_workspace(gui)
        overview_fn = getattr(gui, "_render_overview_signal_spectral", None)
        if callable(overview_fn):
            try:
                overview_fn()
            except Exception:
                logger.debug("overview signal/spectral refresh failed", exc_info=True)
        render_fn = getattr(gui, "_render_spectral_features", None)
        if callable(render_fn):
            render_fn()
        refresh_rv = getattr(gui, "_refresh_research_validation", None)
        if callable(refresh_rv):
            try:
                refresh_rv()
            except Exception:
                logger.debug("research validation refresh after spectral failed", exc_info=True)

    thread = threading.Thread(target=worker, name="stablewalk-spectral", daemon=True)
    gui._spectral_worker = thread
    thread.start()
    try:
        gui.root.after(50, drain)
    except Exception:
        drain()


def insufficient_spectral_copy(report: Any | None) -> str:
    """Empty-state copy: no fabricated numbers, actual warnings when present."""
    if report is None:
        return (
            "Load and analyze a walking video.\n"
            "Autocorrelation, Welch PSD, and R(0) are derived from pose time series."
        )
    warns = [str(w).strip() for w in (getattr(report, "warnings", None) or []) if str(w).strip()]
    cycle_warns = [w for w in warns if "insufficient cycles" in w.lower()]
    if cycle_warns:
        return "insufficient cycles\n" + "\n".join(cycle_warns[:3])
    if warns:
        return "Insufficient data for spectral analysis.\n" + "\n".join(warns[:4])
    return "Spectral analysis is unavailable for this recording."


def _paint_empty(gui: Any, report: Any | None = None) -> None:
    empty = getattr(gui, "lbl_spectral_empty", None)
    if empty is not None:
        try:
            empty.configure(text=insufficient_spectral_copy(report))
        except tk.TclError:
            pass
    _set_spectral_plot_visible(gui, False)
    _fill_table(gui, [])


def render_spectral_workspace(gui: Any) -> None:
    from stablewalk.ui.viewers.spectral_charts import render_spectral_figure

    report = getattr(gui, "_spectral_report", None)
    fig = getattr(gui, "fig_spectral", None)
    axes = getattr(gui, "_spectral_axes", None)
    canvas = getattr(gui, "canvas_spectral", None)
    if fig is None or axes is None or canvas is None:
        return
    if report is None or not report.available:
        _paint_empty(gui, report)
        warn = getattr(gui, "lbl_spectral_warnings", None)
        if warn is not None:
            try:
                if report is not None:
                    warn.configure(text="  ".join(report.warnings[:4]) or "Insufficient data.")
                else:
                    warn.configure(text="No gait recording is loaded.")
            except tk.TclError:
                pass
        _refresh_spectral_expand_windows(gui)
        return
    _set_spectral_plot_visible(gui, True)
    channel_id = str(getattr(gui, "_spectral_channel_id", DEFAULT_UNIVARIATE_CHANNEL_ID))
    playhead = _playhead_state(gui)
    render_spectral_figure(fig, axes, report, channel_id, playhead=playhead)
    relayout_spectral_plots(gui)
    try:
        canvas.draw_idle()
    except Exception:
        pass
    _fill_results_table(gui, report, channel_id)
    _refresh_spectral_expand_windows(gui)
    status = getattr(gui, "lbl_spectral_status", None)
    ac = report.primary_autocorr()
    rel = ac.reliability_label if ac else "unavailable"
    expl = str(getattr(ac, "reliability_explanation", "") or "") if ac else ""
    qstat = str(getattr(ac, "quality_status", "") or rel) if ac else "unavailable"
    if status is not None:
        try:
            status.configure(
                text=(
                    f"Spectral {qstat} ({rel})"
                    + (f" — {expl}" if expl else "")
                    + " · not pose coverage or video quality · not a diagnosis"
                )
            )
        except tk.TclError:
            pass
    warn = getattr(gui, "lbl_spectral_warnings", None)
    if warn is not None:
        try:
            warn.configure(text="  ".join(report.warnings[:5]) or "No data-quality warnings.")
        except tk.TclError:
            pass


def _playhead_state(gui: Any) -> PlayheadState | None:
    try:
        return playhead_from_gui(gui)
    except Exception:
        return None


def update_spectral_playhead(gui: Any) -> None:
    if not _spectral_page_selected(gui):
        return
    report = getattr(gui, "_spectral_report", None)
    if report is None:
        return
    axes = getattr(gui, "_spectral_axes", None)
    canvas = getattr(gui, "canvas_spectral", None)
    if not axes or canvas is None:
        return
    playhead = _playhead_state(gui)
    if playhead is None:
        return
    from stablewalk.ui.viewers.chart_playhead import update_playhead_on_axes

    try:
        if not update_playhead_on_axes([axes[0]], playhead):
            render_spectral_workspace(gui)
            return
        canvas.draw_idle()
    except Exception:
        pass
    by_id = getattr(gui, "_spectral_expand_by_id", None) or {}
    signal = by_id.get("signal")
    if signal:
        ax = signal.get("ax")
        exp_canvas = signal.get("canvas")
        if ax is not None and exp_canvas is not None:
            try:
                if update_playhead_on_axes([ax], playhead):
                    exp_canvas.draw_idle()
            except Exception:
                pass


def _update_spectral_row_detail(gui: Any) -> None:
    tree = getattr(gui, "_spectral_tree", None)
    detail = getattr(gui, "lbl_spectral_row_detail", None)
    if tree is None or detail is None:
        return
    try:
        sel = tree.selection()
        if not sel:
            detail.configure(text="Select a row to read the complete value and description.")
            return
        values = tree.item(sel[0], "values") or ()
        metric = str(values[0]) if len(values) > 0 else ""
        value = str(values[1]) if len(values) > 1 else ""
        meaning = str(values[2]) if len(values) > 2 else ""
        lines = [part for part in (f"{metric}: {value}".strip(": "), meaning) if part]
        detail.configure(text="\n".join(lines) if lines else "—")
    except tk.TclError:
        pass


def _fill_table(gui: Any, rows: list[tuple[str, str, str]]) -> None:
    tree = getattr(gui, "_spectral_tree", None)
    if tree is None:
        return
    try:
        for item in tree.get_children():
            tree.delete(item)
        for metric, value, meaning in rows:
            tree.insert("", "end", values=(metric, value, meaning))
        children = tree.get_children()
        if children:
            tree.selection_set(children[0])
            tree.focus(children[0])
        _update_spectral_row_detail(gui)
        from stablewalk.ui.tk.analysis_split import sync_tree_xscroll

        sync_tree_xscroll(
            tree,
            getattr(gui, "_spectral_tree_xscroll", None),
            getattr(gui, "_spectral_side", None),
        )
    except tk.TclError:
        pass


def _fmt(value: float | None, unit: str = "", digits: int = 3) -> str:
    if value is None or not (value == value) or value in (float("inf"), float("-inf")):
        return "unavailable"
    if unit:
        return f"{value:.{digits}f} {unit}"
    return f"{value:.{digits}f}"


def _provenance_meaning(base: str, provenance: Any | None) -> str:
    """Append source, window, preprocessing, peak method, and confidence."""
    if provenance is None:
        return base
    bits = [base.rstrip()]
    source = str(getattr(provenance, "source_signal", "") or "").strip()
    if source:
        bits.append(f"Source: {source}.")
    window = getattr(provenance, "time_window_s", None)
    if window and len(window) == 2:
        bits.append(f"Time window: {float(window[0]):.3f}–{float(window[1]):.3f} s.")
    preprocess = str(getattr(provenance, "preprocessing", "") or "").strip()
    if preprocess:
        bits.append(f"Preprocess: {preprocess}.")
    peak = str(getattr(provenance, "peak_selection", "") or "").strip()
    if peak:
        bits.append(f"Peak selection: {peak}.")
    conf = getattr(provenance, "confidence", None)
    label = str(getattr(provenance, "confidence_label", "") or "").strip()
    if conf is not None:
        bits.append(f"Confidence: {float(conf):.2f} ({label or 'n/a'}).")
    elif label:
        bits.append(f"Confidence: {label}.")
    return " ".join(bits)


def _fill_results_table(gui: Any, report: Any, channel_id: str) -> None:
    ac = report.autocorr.get(channel_id)
    sp = report.spectra.get(channel_id)
    timing = getattr(gui, "_gait_timing", None) or getattr(report, "timing", None)
    insuff = getattr(ac, "insufficient_reason", None) if ac is not None else None
    stride_s = getattr(ac, "stride_period_s", None) if ac else None
    if insuff or stride_s is None:
        period_txt = "unavailable"
        cadence_txt = "unavailable"
    else:
        period_txt = _fmt(stride_s, "s")
        cadence_txt = _fmt(
            getattr(ac, "cadence_steps_per_min", None) if ac else None, "steps/min", 1
        )
    cadence_meaning = (
        "60 / selected-channel ACF step period. Independent of stride; not 120/T_stride."
    )
    if ac is not None and getattr(ac, "cadence_assumes_two_steps_per_stride", False):
        cadence_meaning = (
            "120 / selected-channel stride period (assumes two steps per stride; "
            "used only when no independent step peak exists). Not Overview cadence."
        )
    rows: list[tuple[str, str, str]] = [
        ("Channel", _label_for_id(channel_id), "Selected joint / DOF time series"),
        ("Sampling rate", _fmt(report.bundle.fps, "Hz", 2), "Uniform grid after resampling to 1/FPS"),
        ("Samples", str(int(report.bundle.times_s.size)), "Length of the uniform series"),
        (
            "Preprocess",
            "detrend + z-score",
            "Linear detrend then unit-variance normalize; missing landmarks stay NaN",
        ),
    ]
    if timing is not None and timing.session_stride.value_s is not None:
        session_note = (
            "Four Gait median of reliable thigh/shank ACF stride lags. "
            "Not this channel's ACF and not a heel-strike interval."
        )
        if timing.session_stride.notes:
            session_note = f"{session_note} {timing.session_stride.notes[0]}"
        rows.append(
            (
                "Session stride period",
                _fmt(timing.session_stride.value_s, "s"),
                _provenance_meaning(session_note, timing.session_stride.provenance),
            )
        )
    elif timing is not None and timing.session_stride.notes:
        rows.append(
            (
                "Session stride period",
                "unavailable",
                timing.session_stride.notes[0],
            )
        )
    if ac:
        n_peaks = len(getattr(ac, "peaks", None) or [])
        assign_note = str(getattr(ac, "stride_assignment_note", "") or "")
        stride_meaning = (
            "This channel's ACF stride lag after exclusive-window assignment. "
            "Not the session Four Gait consensus and not event-based HS–HS."
        )
        if assign_note:
            stride_meaning = f"{stride_meaning} {assign_note}"
        step_meaning = (
            "ACF peak in the step-lag window on this channel. Independent of "
            "stride; not forced to stride/2."
        )
        if timing is not None:
            stride_meaning = _provenance_meaning(stride_meaning, timing.channel_stride.provenance)
            cadence_meaning = _provenance_meaning(
                cadence_meaning, timing.cadence_spectral.provenance
            )
            step_meaning = _provenance_meaning(step_meaning, timing.channel_step.provenance)
        else:
            stride_meaning = (
                f"{stride_meaning} Source: {ac.channel_id}. "
                f"Preprocess: {getattr(ac, 'preprocessing', '')}. "
                f"Peak selection: {getattr(ac, 'peak_selection', '')}. "
                f"Confidence: {ac.reliability:.2f} ({ac.reliability_label})."
            )
        rows.extend(
            [
                ("Selected-channel stride period", period_txt, stride_meaning),
                ("Cadence (spectral)", cadence_txt, cadence_meaning),
                (
                    "Selected-channel step period",
                    _fmt(ac.step_period_s, "s") if not insuff else "unavailable",
                    step_meaning,
                ),
                ("Valid ACF peaks", str(n_peaks), "Positive local maxima of ρ(τ) in the walking lag window"),
                (
                    "Estimated stride cycles (channel)",
                    _fmt(ac.n_stride_cycles, "", 1) if stride_s is not None and not insuff else "unavailable",
                    "Clip duration / this channel's ACF stride period. Not a heel-strike count.",
                ),
            ]
        )
        if timing is not None:
            est_session = timing.cycles.estimated_from_session_stride
            rows.append(
                (
                    "Estimated stride cycles (session)",
                    _fmt(est_session, "", 1),
                    "Clip duration / session Four Gait stride period. Independent of the channel estimate and of detected HS–HS.",
                )
            )
            rows.append(
                (
                    "Detected / valid HS–HS cycles",
                    f"{timing.cycles.detected} detected / {timing.cycles.valid} valid",
                    "Complete ipsilateral heel-strike intervals from the contact FSM. Independent of ACF estimates. Low-confidence counts are kept.",
                )
            )
            if timing.consistency_notes:
                rows.append(
                    (
                        "Why estimates differ",
                        "kept independent",
                        " ".join(timing.consistency_notes),
                    )
                )
        rows.append(
            (
                "ACF spectral quality",
                f"{ac.reliability:.2f} ({getattr(ac, 'quality_status', ac.reliability_label)}; {ac.reliability_label})",
                getattr(ac, "reliability_explanation", "")
                or "Duration, gaps, cycles, and ACF peak — not pose coverage.",
            )
        )
    if sp:
        rows.extend(
            [
                ("Dominant frequency", _fmt(sp.dominant_hz, "Hz"), "arg max P(f) inside the walking band"),
                ("Peak power", _fmt(sp.dominant_power, "z²/Hz", 4), "P(f₀) of the z-scored series (same units as the Welch PSD axis)"),
                ("Spectral centroid", _fmt(sp.spectral_centroid_hz, "Hz"), "Power-weighted mean frequency in the walking band"),
                (
                    "Spectral bandwidth",
                    _fmt(getattr(sp, "spectral_bandwidth_hz", None), "Hz"),
                    "Power-weighted RMS spread around the centroid",
                ),
                (
                    "Spectral entropy",
                    _fmt(getattr(sp, "spectral_entropy", None), "", 3),
                    "Normalized Shannon entropy of P(f) in the walking band (0–1)",
                ),
                ("Harmonic ratio", _fmt(sp.harmonic_ratio, "", 2), "P(2 f₀) / P(f₀) when 2 f₀ is in the walking band"),
                ("Gait-band power fraction", _fmt(sp.band_power_fraction, "", 2), "Share of total PSD inside the walking band"),
                ("PSD method", str(sp.method), "Welch density (Hann); FFT periodogram stored for export"),
            ]
        )
    ch = report.bundle.get(channel_id)
    if ch is not None:
        rows.extend(
            [
                ("Missing-sample fraction", _fmt(ch.missing_fraction, "", 3), "Share of grid samples with no confident landmark"),
                ("Interpolated fraction", _fmt(ch.interpolated_fraction, "", 3), "Short interior gaps filled; long gaps stay NaN"),
                ("Mean landmark visibility", _fmt(ch.mean_visibility, "", 2), "Mean pose visibility on this joint"),
            ]
        )
    feats = getattr(report, "features", None)
    vec = getattr(feats, "session_vector", None) if feats is not None else None
    if isinstance(vec, dict):
        rows.extend(
            [
                ("ACF stride peak ρ", _fmt(vec.get("acf_stride_peak"), "", 3), "Normalized autocorrelation at the stride lag"),
                ("L/R Δf (session)", _fmt(vec.get("lr_freq_delta_hz"), "Hz"), "Right minus left dominant frequency"),
                ("L/R xcorr peak lag", _fmt(vec.get("lr_xcorr_lag_s"), "s"), "Lag of the left/right cross-correlation peak"),
                ("L/R relative phase", _fmt(vec.get("lr_relative_phase_deg"), "deg", 1), "360° × lag / gait period, wrapped ±180°"),
            ]
        )
    mats = report.matrices
    if mats is not None and mats.lagged:
        lag_txt = ", ".join(f"{m.tau_s:.2f}s" for m in mats.lagged[:6])
        rows.append(("Lagged R(τ) lags", lag_txt or "unavailable", "Delays used for the lagged correlation matrices"))
        if mats.cross_corr:
            xc0 = mats.cross_corr[0]
            rows.append(
                (
                    "L/R xcorr (first pair)",
                    f"τ*={_fmt(xc0.peak_lag_s, 's')} ρ={_fmt(xc0.peak_value, '', 3)}",
                    "Peak lag and ρ of the first left/right pair",
                )
            )
    for cmp_ in report.lr_compare[:3]:
        rows.append(
            (
                f"L/R {cmp_.pair_id} Δf",
                _fmt(cmp_.delta_hz, "Hz"),
                "Right minus left dominant frequency for this pair",
            )
        )
    eig = report.eigen.get("correlation")
    if eig is not None and eig.eigenvalues.size:
        rows.append(
            (
                "Leading eigenvalue (corr.)",
                _fmt(float(eig.eigenvalues[0]), "", 3),
                "Largest eigenvalue of the joint-by-joint R(0) heatmap",
            )
        )
        rows.append(("Modes", str(int(eig.eigenvalues.size)), "Number of R(0) correlation eigenmodes"))
    rows.append(("Disclaimer", "Not a clinical diagnosis", "Derived from monocular pose time series"))
    _fill_table(gui, rows)


def export_spectral(gui: Any, kind: str) -> None:
    report = getattr(gui, "_spectral_report", None)
    if report is None:
        messagebox.showinfo("Spectral analysis", "Compute spectral analysis before exporting.")
        return
    from stablewalk.analysis.spectral.export import export_spectral_csv, export_spectral_json
    from stablewalk.config import ANALYSIS_EXPORT_DIR

    ANALYSIS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if kind == "json":
        path = filedialog.asksaveasfilename(
            title="Export spectral analysis JSON",
            defaultextension=".json",
            initialdir=str(ANALYSIS_EXPORT_DIR),
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        export_spectral_json(report, path)
    else:
        path = filedialog.asksaveasfilename(
            title="Export spectral analysis CSV",
            defaultextension=".csv",
            initialdir=str(ANALYSIS_EXPORT_DIR),
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        export_spectral_csv(report, path)
    status = getattr(gui, "lbl_spectral_status", None)
    if status is not None:
        try:
            status.configure(text=f"Exported {kind.upper()}")
        except tk.TclError:
            pass
