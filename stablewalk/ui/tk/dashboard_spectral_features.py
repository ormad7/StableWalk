"""Spectral Features dashboard — values computed from the loaded video only."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from stablewalk.analysis.spectral.channels import CHANNEL_BY_ID
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
from stablewalk.ui.viewers.chart_style import SERIES_LINE_WIDTH, apply_chart_panel_style

_SCOPES = ("session", "left", "right", "bilateral", "cycle")
_TREE_COLS = ("name", "value", "units", "left", "right", "quality", "scope", "channel")

# Features computed on the z-scored (standardized) series — display-only label.
_ZSCORED_FEATURE_NAMES = frozenset(
    {
        "dominant_frequency_hz",
        "dominant_power",
        "gait_band_power",
        "spectral_entropy",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "harmonic_ratio",
        "stride_harmonic_power_ratio",
        "acf_stride_peak",
        "acf_stride_lag_s",
        "acf_peak_prominence",
        "stride_duration_s",
        "regularity_acf",
        "spectral_cadence_steps_per_min",
        "leading_eigenvalue",
        "explained_variance_mode1",
        "principal_mode_loadings",
        "lr_freq_delta_hz",
        "lr_entropy_delta",
        "lr_xcorr_lag_s",
        "lr_relative_phase_deg",
    }
)

_FEATURES_TABLE_MIN = 360
_FEATURES_VIZ_MIN = 280
_FEATURES_FIG_MIN_W = 360
_FEATURES_FIG_MIN_H = 220
_FEATURES_FIG_MARGINS = dict(left=0.16, right=0.96, top=0.72, bottom=0.20)


def feature_numeric_display(rec: Any) -> str:
    """Table value column: number only. Units stay in the Units column."""
    if rec is None:
        return "—"
    if not bool(getattr(rec, "available", False)):
        text = str(rec.display_value() or "").strip()
        if text.lower() in {"", "unavailable", "—", "-"}:
            return "N/A"
        return text
    value = getattr(rec, "value", None)
    if value is None:
        return rec.display_value()
    try:
        if value != value or value in (float("inf"), float("-inf")):
            return "—"
    except Exception:
        return "—"
    return f"{float(value):.4g}"


def feature_units_display(rec: Any) -> str:
    """Human units. Stored ``1`` means a dimensionless ratio, not the integer 1."""
    units = str(getattr(rec, "units", "") or "").strip()
    if units in {"", "1"}:
        return "dimensionless"
    if units == "units^2/Hz":
        return "z²/Hz"
    if units == "units^2":
        return "z²"
    return units


def feature_uses_zscore(rec: Any) -> bool:
    return str(getattr(rec, "name", "") or "") in _ZSCORED_FEATURE_NAMES


def feature_preprocess_copy(rec: Any) -> str:
    if feature_uses_zscore(rec):
        return (
            "Preprocessing: linear detrend, then z-score (unit variance). "
            "The plotted series is that standardized copy — dimensionless, not deg or m."
        )
    return (
        "Preprocessing: linear detrend in the channel’s native units. "
        "The plotted series is detrended, not z-scored. ACF/PSD features use a separate z-score."
    )


_SPECTRAL_NO_INTERPRETATION = (
    "Do not interpret this as pose accuracy, video quality, or a clinical finding."
)


def feature_quality_reason(rec: Any) -> str:
    quality = str(getattr(rec, "quality", "") or "unavailable")
    notes = [str(n).strip() for n in (getattr(rec, "notes", None) or []) if str(n).strip()]
    missing = None
    if hasattr(rec, "unavailable_reason"):
        try:
            missing = rec.unavailable_reason()
        except Exception:
            missing = None
    if quality == "insufficient":
        reason = (
            (missing or "N/A — source series, peak, or cycle count is insufficient. "
             "This is not a value of 0.")
            + " "
            + _SPECTRAL_NO_INTERPRETATION
        )
    elif quality == "low":
        reason = (
            "Low spectral reliability (duration, gaps, or weak ACF peak). "
            "The numeric result is retained. "
            + _SPECTRAL_NO_INTERPRETATION
        )
    elif quality == "moderate":
        reason = (
            "Usable spectral reliability for this recording — not pose coverage, "
            "landmark visibility, or a diagnosis."
        )
    else:
        reason = "Quality is taken from spectral reliability of the source series."
    if notes:
        return f"{quality}: {reason} {' '.join(notes[:2])}"
    return f"{quality}: {reason}"


def spectral_features_split_minsizes(scale: float = 1.0) -> tuple[int, int]:
    from stablewalk.ui.tk.layout_scale import scaled_px

    return (
        scaled_px(_FEATURES_TABLE_MIN, scale, minimum=280),
        scaled_px(_FEATURES_VIZ_MIN, scale, minimum=220),
    )


def _label(channel_id: str | None) -> str:
    if not channel_id:
        return ""
    spec = CHANNEL_BY_ID.get(channel_id)
    return spec.label if spec else channel_id


def insufficient_features_copy(report: Any | None) -> str:
    """Empty-state copy when the current video has no usable spectral features."""
    if report is None:
        return "No spectral features yet.\nAnalyze a walking video first."
    warns = [str(w).strip() for w in (getattr(report, "warnings", None) or []) if str(w).strip()]
    cycle_warns = [w for w in warns if "insufficient cycles" in w.lower()]
    if cycle_warns:
        return "insufficient cycles\n" + "\n".join(cycle_warns[:3])
    if warns:
        return "Insufficient data for spectral features.\n" + "\n".join(warns[:4])
    return "Spectral features are unavailable for this recording."


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    try:
        if value != value or value in (float("inf"), float("-inf")):
            return "—"
    except Exception:
        return "—"
    return f"{value:.4g}"


def build_spectral_features_workspace(gui: Any, parent: tk.Misc) -> tk.Frame:
    from stablewalk.ui.design_system.components import EmptyState

    host = ttk.Frame(parent)
    host.grid(row=0, column=0, sticky="nsew")
    parent.rowconfigure(0, weight=1)
    parent.columnconfigure(0, weight=1)
    host.columnconfigure(0, weight=1)
    host.rowconfigure(2, weight=1)
    gui._spectral_features_host = host

    tools = tk.Frame(host, bg=ELEVATED, highlightthickness=0)
    tools.grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_SM, PAD_XS))
    tools.columnconfigure(6, weight=1)

    tk.Label(
        tools, text="Scope", bg=ELEVATED, fg=TEXT_SECONDARY, font=FONT_KPI_TITLE
    ).grid(row=0, column=0, padx=(PAD_SM, PAD_XS), pady=PAD_XS)
    gui.var_spectral_feature_scope = tk.StringVar(value="session")
    from stablewalk.ui.tk.layout_scale import combobox_char_width

    scope = ttk.Combobox(
        tools,
        textvariable=gui.var_spectral_feature_scope,
        values=_SCOPES,
        state="readonly",
        width=combobox_char_width(_SCOPES, minimum=10, maximum=16),
        takefocus=False,
    )
    scope.grid(row=0, column=1, padx=(0, PAD_SM), pady=PAD_XS)
    scope.bind("<<ComboboxSelected>>", lambda _e: render_spectral_features_workspace(gui))
    create_tooltip(scope, "Session, left, right, bilateral, or per-cycle features.")

    tk.Label(
        tools, text="Cycle", bg=ELEVATED, fg=TEXT_SECONDARY, font=FONT_KPI_TITLE
    ).grid(row=0, column=2, padx=(PAD_SM, PAD_XS), pady=PAD_XS)
    gui.var_spectral_feature_cycle = tk.StringVar(value="all")
    cycle_box = ttk.Combobox(
        tools,
        textvariable=gui.var_spectral_feature_cycle,
        values=("all",),
        state="readonly",
        width=8,
        takefocus=False,
    )
    cycle_box.grid(row=0, column=3, padx=(0, PAD_SM), pady=PAD_XS)
    gui.cmb_spectral_feature_cycle = cycle_box
    cycle_box.bind("<<ComboboxSelected>>", lambda _e: render_spectral_features_workspace(gui))

    btn_csv = ttk.Button(
        tools, text="CSV", width=6, command=lambda: _export(gui, "csv"), takefocus=False
    )
    btn_json = ttk.Button(
        tools, text="JSON", width=6, command=lambda: _export(gui, "json"), takefocus=False
    )
    btn_csv.grid(row=0, column=4, padx=(0, PAD_XS), pady=PAD_XS)
    btn_json.grid(row=0, column=5, pady=PAD_XS)
    create_tooltip(btn_csv, "Export spectral features (same file as spectral CSV, features section).")
    create_tooltip(btn_json, "Export spectral JSON including the features block.")
    try:
        from stablewalk.ui.tk.dashboard_panel_maximize import install_maximize_button_grid

        install_maximize_button_grid(
            gui, tools, "spectral_features", row=0, column=7, sticky="e", padx=(PAD_SM, PAD_SM)
        )
    except Exception:
        pass

    gui.lbl_spectral_features_status = tk.Label(
        tools,
        text="Derived features — not a clinical diagnosis",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="e",
        wraplength=260,
        justify=tk.RIGHT,
    )
    gui.lbl_spectral_features_status.grid(
        row=0, column=6, sticky="e", padx=PAD_SM, pady=PAD_XS
    )
    from stablewalk.ui.tk.layout_scale import bind_status_wrap

    bind_status_wrap(tools, gui.lbl_spectral_features_status, reserve=480, minimum=100)

    gui.lbl_spectral_features_meta = tk.Label(
        host,
        text="Load a walking video. Every row is computed from that recording.",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=920,
    )
    gui.lbl_spectral_features_meta.grid(
        row=1, column=0, sticky="ew", padx=PAD_MD, pady=(0, PAD_XS)
    )

    def _wrap_features_copy(_event: object | None = None) -> None:
        try:
            width = int(host.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 80:
            return
        try:
            gui.lbl_spectral_features_meta.configure(wraplength=max(220, width - 48))
        except tk.TclError:
            pass

    host.bind("<Configure>", _wrap_features_copy, add="+")

    body = ttk.Frame(host)
    body.grid(row=2, column=0, sticky="nsew")
    body.columnconfigure(0, weight=1)
    body.rowconfigure(0, weight=1)
    gui._spectral_features_body = body

    from stablewalk.ui.tk.analysis_split import (
        bind_persistent_h_sash,
        create_h_split,
        split_add,
        sync_tree_xscroll,
    )
    from stablewalk.ui.tk.layout_scale import dpi_scale_factor

    scale = dpi_scale_factor(parent)
    table_min, viz_min = spectral_features_split_minsizes(scale)
    paned = create_h_split(body)
    paned.grid(row=0, column=0, sticky="nsew")
    gui._spectral_features_split = paned

    table_host = ttk.Frame(paned)
    gui._spectral_features_table_host = table_host
    table_host.columnconfigure(0, weight=1)
    table_host.rowconfigure(0, weight=1)

    gui.lbl_spectral_features_empty = EmptyState(
        table_host,
        "No spectral features yet.\nAnalyze a walking video first.",
        bg=PANEL,
    )
    gui.lbl_spectral_features_empty.grid(row=0, column=0, sticky="nsew")

    tree = ttk.Treeview(
        table_host,
        columns=_TREE_COLS,
        show="headings",
        selectmode="browse",
        height=18,
        style="Compact.Treeview",
    )
    headings = {
        "name": "Feature",
        "value": "Value",
        "units": "Units",
        "left": "Left",
        "right": "Right",
        "quality": "Quality",
        "scope": "Scope",
        "channel": "Channel",
    }
    widths = {
        "name": 180,
        "value": 88,
        "units": 110,
        "left": 72,
        "right": 72,
        "quality": 96,
        "scope": 80,
        "channel": 150,
    }
    for col in _TREE_COLS:
        tree.heading(col, text=headings[col])
        tree.column(
            col,
            width=widths[col],
            minwidth=64 if col != "name" else 110,
            stretch=True,
        )
    scroll = ttk.Scrollbar(table_host, orient="vertical", command=tree.yview)
    xscroll = ttk.Scrollbar(table_host, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=scroll.set, xscrollcommand=xscroll.set)
    table_host.rowconfigure(0, weight=1)
    table_host.rowconfigure(1, weight=0)
    gui._spectral_features_tree = tree
    gui._spectral_features_scroll = scroll
    gui._spectral_features_xscroll = xscroll
    tree.bind("<<TreeviewSelect>>", lambda _e: _on_feature_select(gui))

    viz = ttk.Frame(paned)
    gui._spectral_features_viz = viz
    viz.columnconfigure(0, weight=1)
    viz.rowconfigure(1, weight=1)
    tk.Label(
        viz,
        text="Underlying signal",
        bg=PANEL,
        fg=TEXT,
        font=FONT_KPI_TITLE,
        anchor="w",
    ).grid(row=0, column=0, sticky="ew", padx=PAD_XS, pady=(PAD_XS, 0))
    fig = Figure(figsize=(5.2, 3.6), dpi=100, facecolor=PANEL)
    ax = fig.add_subplot(1, 1, 1)
    apply_chart_panel_style(ax)
    graph_host = tk.Frame(viz, bg=PANEL, highlightthickness=0)
    graph_host.grid(row=1, column=0, sticky="nsew", padx=PAD_XS)
    graph_host.columnconfigure(0, weight=1)
    graph_host.rowconfigure(0, weight=1)
    graph_host.grid_propagate(False)
    canvas = FigureCanvasTkAgg(fig, master=graph_host)
    widget = canvas.get_tk_widget()
    widget.configure(bg=PANEL, highlightthickness=0)
    widget.grid(row=0, column=0, sticky="nsew")
    gui._spectral_features_graph_host = graph_host
    gui.fig_spectral_features = fig
    gui.ax_spectral_features = ax
    gui.canvas_spectral_features = canvas
    try:
        from stablewalk.ui.tk.dashboard_layout import _bind_figure_resize

        _bind_figure_resize(canvas, fig, margins=_FEATURES_FIG_MARGINS, min_px=80)
    except Exception:
        pass
    gui.lbl_spectral_feature_summary = tk.Label(
        viz,
        text="Select a feature to inspect its value, units, and quality.",
        bg=PANEL,
        fg=TEXT_SECONDARY,
        font=FONT_UI_SM,
        wraplength=280,
        justify=tk.LEFT,
        anchor="w",
    )
    gui.lbl_spectral_feature_summary.grid(
        row=2, column=0, sticky="ew", padx=PAD_XS, pady=(PAD_XS, 0)
    )
    formula_bar = tk.Frame(viz, bg=PANEL, highlightthickness=0)
    formula_bar.grid(row=3, column=0, sticky="ew", padx=PAD_XS)
    gui._spectral_feature_formula_open = False
    gui.btn_spectral_feature_formula = ttk.Button(
        formula_bar,
        text="Show formula",
        width=14,
        takefocus=False,
    )
    gui.btn_spectral_feature_formula.pack(side=tk.LEFT, pady=(PAD_XS, 0))
    gui.lbl_spectral_feature_def = tk.Label(
        viz,
        text="Select a feature for formula, source signal, preprocessing, and quality.",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        wraplength=280,
        justify=tk.LEFT,
        anchor="nw",
    )
    gui.lbl_spectral_feature_def.grid(row=4, column=0, sticky="ew", padx=PAD_XS, pady=(PAD_XS, PAD_XS))
    gui.lbl_spectral_feature_def.grid_remove()

    def _toggle_feature_formula() -> None:
        open_ = not bool(getattr(gui, "_spectral_feature_formula_open", False))
        gui._spectral_feature_formula_open = open_
        try:
            if open_:
                gui.lbl_spectral_feature_def.grid()
                gui.btn_spectral_feature_formula.configure(text="Hide formula")
            else:
                gui.lbl_spectral_feature_def.grid_remove()
                gui.btn_spectral_feature_formula.configure(text="Show formula")
        except tk.TclError:
            pass

    gui.btn_spectral_feature_formula.configure(command=_toggle_feature_formula)

    split_add(paned, table_host, minsize=table_min)
    split_add(paned, viz, minsize=viz_min)

    gui._place_spectral_features_sash = bind_persistent_h_sash(
        paned,
        gui,
        frac_attr="_spectral_features_sash_frac",
        drag_attr="_spectral_features_sash_dragging",
        default=0.56,
        min_left=table_min,
        min_right=viz_min,
        min_width=480,
    )

    def _wrap_feature_def(_event: object | None = None) -> None:
        try:
            width = int(viz.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 40:
            return
        wrap = max(160, width - 16)
        try:
            gui.lbl_spectral_feature_def.configure(wraplength=wrap)
            summary = getattr(gui, "lbl_spectral_feature_summary", None)
            if summary is not None:
                summary.configure(wraplength=wrap)
        except tk.TclError:
            pass
        sync_tree_xscroll(tree, xscroll, table_host)

    viz.bind("<Configure>", _wrap_feature_def, add="+")
    table_host.bind(
        "<Configure>",
        lambda _e: sync_tree_xscroll(tree, xscroll, table_host),
        add="+",
    )

    gui._render_spectral_features = lambda: render_spectral_features_workspace(gui)
    return host


def render_spectral_features_workspace(gui: Any) -> None:
    report = getattr(gui, "_spectral_report", None)
    tree = getattr(gui, "_spectral_features_tree", None)
    empty = getattr(gui, "lbl_spectral_features_empty", None)
    scroll = getattr(gui, "_spectral_features_scroll", None)
    features = getattr(report, "features", None) if report is not None else None
    if tree is None:
        return
    if report is None or features is None or not report.available:
        try:
            tree.grid_remove()
            if scroll is not None:
                scroll.grid_remove()
            xscroll = getattr(gui, "_spectral_features_xscroll", None)
            if xscroll is not None:
                xscroll.grid_remove()
            if empty is not None:
                empty.configure(text=insufficient_features_copy(report))
                empty.grid(row=0, column=0, sticky="nsew")
        except tk.TclError:
            pass
        meta = getattr(gui, "lbl_spectral_features_meta", None)
        if meta is not None:
            try:
                meta.configure(text=insufficient_features_copy(report).replace("\n", " "))
            except tk.TclError:
                pass
        _clear_viz(gui)
        return
    try:
        if empty is not None:
            empty.grid_remove()
        tree.grid(row=0, column=0, sticky="nsew")
        if scroll is not None:
            scroll.grid(row=0, column=1, sticky="ns")
        xscroll = getattr(gui, "_spectral_features_xscroll", None)
        if xscroll is not None:
            xscroll.grid(row=1, column=0, sticky="ew")
    except tk.TclError:
        pass

    cycles = sorted(
        {r.cycle_index for r in features.records if r.cycle_index is not None}
    )
    cycle_vals = ["all"] + [str(i) for i in cycles]
    box = getattr(gui, "cmb_spectral_feature_cycle", None)
    if box is not None:
        try:
            box.configure(values=cycle_vals)
        except tk.TclError:
            pass

    scope = str(getattr(gui, "var_spectral_feature_scope", None) and gui.var_spectral_feature_scope.get() or "session")
    cycle_sel = str(gui.var_spectral_feature_cycle.get()) if getattr(gui, "var_spectral_feature_cycle", None) else "all"
    cycle_index = None if cycle_sel in ("", "all") else int(cycle_sel)
    rows = features.records_for(scope=scope, cycle_index=cycle_index if scope == "cycle" else None)
    if scope != "cycle":
        rows = [r for r in features.records if r.scope == scope]

    row_map: dict[str, Any] = {}
    try:
        for item in tree.get_children():
            tree.delete(item)
        for index, rec in enumerate(rows):
            iid = str(index)
            row_map[iid] = rec
            tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    rec.label,
                    feature_numeric_display(rec),
                    feature_units_display(rec),
                    _fmt(rec.left_value),
                    _fmt(rec.right_value),
                    rec.quality,
                    rec.scope,
                    _label(rec.channel_id),
                ),
            )
    except tk.TclError:
        pass
    gui._spectral_feature_row_map = row_map
    if rows:
        try:
            first = tree.get_children()
            if first:
                tree.selection_set(first[0])
                tree.focus(first[0])
                _on_feature_select(gui)
        except tk.TclError:
            pass

    n_avail = sum(1 for r in rows if r.available)
    meta = getattr(gui, "lbl_spectral_features_meta", None)
    qtxt = (
        f"Pose/video heuristic {features.video_quality_score:.0f}/100 "
        "(not spectral reliability)"
        if features.video_quality_score is not None
        else "Pose/video heuristic not scored"
    )
    if meta is not None:
        try:
            meta.configure(
                text=(
                    f"{n_avail}/{len(rows)} features available · "
                    f"{features.n_valid_cycles} valid / {features.n_detected_cycles} detected HS–HS cycles · "
                    f"{qtxt} · derived, not a diagnosis. "
                    + " ".join(features.warnings[:2])
                )
            )
        except tk.TclError:
            pass
    status = getattr(gui, "lbl_spectral_features_status", None)
    if status is not None:
        try:
            status.configure(
                text=f"Derived · {n_avail} available of {len(rows)} · not a clinical diagnosis"
            )
        except tk.TclError:
            pass
    _clear_viz(gui)


def _on_feature_select(gui: Any) -> None:
    tree = getattr(gui, "_spectral_features_tree", None)
    report = getattr(gui, "_spectral_report", None)
    if tree is None or report is None or not report.available:
        return
    sel = tree.selection()
    if not sel:
        return
    rec = (getattr(gui, "_spectral_feature_row_map", None) or {}).get(str(sel[0]))
    channel_id = (rec.channel_id if rec is not None else None) or report.univariate_channel_id
    defn = getattr(gui, "lbl_spectral_feature_def", None)
    summary = getattr(gui, "lbl_spectral_feature_summary", None)
    if rec is not None:
        source = _label(rec.channel_id) or _label(channel_id) or "session / matrix"
        try:
            if summary is not None:
                quality = str(getattr(rec, "quality", "") or "")
                value_txt = rec.display_value()
                if quality == "insufficient":
                    why = rec.unavailable_reason() or "required spectral input is missing"
                    value_txt = f"N/A — {why}"
                extra = ""
                if quality == "low":
                    extra = " Numeric result kept; not a clinical interpretation."
                summary.configure(
                    text=(
                        f"{rec.label}: {value_txt} · {quality}. "
                        f"{rec.definition or ''}{extra}".strip()
                    )
                )
            if defn is not None:
                defn.configure(
                    text="\n".join(
                        [
                            f"Formula: {rec.definition or 'See Spectral Analysis documentation.'}",
                            f"Source signal: {source}",
                            feature_preprocess_copy(rec),
                            f"Quality: {feature_quality_reason(rec)}",
                        ]
                    )
                )
        except tk.TclError:
            pass
    ax = getattr(gui, "ax_spectral_features", None)
    fig = getattr(gui, "fig_spectral_features", None)
    canvas = getattr(gui, "canvas_spectral_features", None)
    if ax is None or canvas is None:
        return
    series = report.bundle.get(channel_id) if channel_id else None
    ax.clear()
    apply_chart_panel_style(ax)
    if fig is not None:
        try:
            fig.set_facecolor(PANEL)
            fig.set_constrained_layout(False)
        except Exception:
            pass
    if series is None:
        ax.text(0.5, 0.5, "No source signal", ha="center", va="center", color=MUTED, transform=ax.transAxes)
    else:
        from stablewalk.pose.knee_convention import (
            KNEE_DETRENDED_AXIS,
            KNEE_ZSCORE_AXIS,
            is_knee_flexion_dof,
        )
        from stablewalk.ui.viewers.chart_style import style_chart_title

        knee = is_knee_flexion_dof(getattr(series.spec, "dof_id", None))
        use_z = feature_uses_zscore(rec) if rec is not None else False
        y = series.normalized if use_z and getattr(series, "normalized", None) is not None and getattr(series.normalized, "size", 0) else series.processed
        if y is None or getattr(y, "size", 0) == 0:
            y = series.processed
        ax.plot(series.times_s, y, color=TEXT, lw=SERIES_LINE_WIDTH)
        ax.set_xlabel("Time (s)")
        if use_z:
            ax.set_ylabel(KNEE_ZSCORE_AXIS if knee else "Z-scored detrended series (dimensionless)")
            style_chart_title(ax, f"{series.spec.label}\nz-scored (detrended)", pad=8)
        else:
            ax.set_ylabel(KNEE_DETRENDED_AXIS if knee else f"Linearly detrended ({series.spec.units})")
            style_chart_title(ax, f"{series.spec.label}\nlinearly detrended", pad=8)
    try:
        if fig is not None:
            fig.subplots_adjust(**_FEATURES_FIG_MARGINS)
        canvas.draw_idle()
    except Exception:
        pass


def _clear_viz(gui: Any) -> None:
    ax = getattr(gui, "ax_spectral_features", None)
    canvas = getattr(gui, "canvas_spectral_features", None)
    if ax is None:
        return
    ax.clear()
    apply_chart_panel_style(ax)
    fig = getattr(gui, "fig_spectral_features", None)
    if fig is not None:
        try:
            fig.set_facecolor(PANEL)
        except Exception:
            pass
    ax.text(
        0.5,
        0.5,
        "Select a feature to view its source signal.",
        ha="center",
        va="center",
        color=MUTED,
        transform=ax.transAxes,
    )
    try:
        if fig is not None:
            fig.subplots_adjust(**_FEATURES_FIG_MARGINS)
        if canvas is not None:
            canvas.draw_idle()
    except Exception:
        pass


def _export(gui: Any, kind: str) -> None:
    from stablewalk.ui.tk.dashboard_spectral import export_spectral

    export_spectral(gui, kind)
