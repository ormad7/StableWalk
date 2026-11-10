"""Left–right gait asymmetry dashboard (Motion inner tab).

Reuses the existing 3D skeleton on Overview / Kinematics. This page does
not create a second 3D viewer.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from stablewalk.ui.colors import ACCENT, BORDER, SIDE_LEFT, SIDE_RIGHT
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
from stablewalk.ui.viewers.chart_style import SERIES_LINE_WIDTH, apply_chart_panel_style, style_chart_title

logger = logging.getLogger(__name__)

_TREE_COLS = ("metric", "units", "left", "right", "delta", "si", "asi", "nsi", "cycles", "quality")
_ASYMMETRY_FIG_MIN_W = 600
_ASYMMETRY_FIG_MIN_H = 580
_PLOT_IDS = ("knee", "foot", "rom", "phase")


def layout_asymmetry_figure(fig: Figure, *, height_px: float | None = None) -> None:
    """Keep titles, axis labels, and legends inside the axes."""
    try:
        fig.set_constrained_layout(False)
    except Exception:
        pass
    if height_px is None:
        try:
            height_px = float(fig.get_size_inches()[1]) * float(fig.get_dpi() or 100.0)
        except Exception:
            height_px = 580.0
    tall = float(height_px) >= _ASYMMETRY_FIG_MIN_H
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.94 if tall else 0.91,
        bottom=0.09 if tall else 0.12,
        hspace=0.34 if tall else 0.46,
        wspace=0.26 if tall else 0.34,
    )


_INDEX_STRIP = (
    "AI / SI = 200(L−R)/(L+R) signed %  ·  "
    "ASI = 200|L−R|/(|L|+|R|) unsigned %  ·  "
    "similarity SI_norm = 2 min(L,R)/(L+R) in [0, 1]  ·  "
    "not interchangeable, not a diagnosis"
)


def build_asymmetry_workspace(gui: Any, parent: tk.Misc) -> tk.Frame:
    from stablewalk.ui.design_system.components import EmptyState

    host = ttk.Frame(parent)
    host.grid(row=0, column=0, sticky="nsew")
    parent.rowconfigure(0, weight=1)
    parent.columnconfigure(0, weight=1)
    host.columnconfigure(0, weight=1)
    host.rowconfigure(3, weight=1)
    gui._asymmetry_host = host
    gui._asymmetry_report = None
    gui._asymmetry_token = 0
    gui._asymmetry_pending = None
    gui._asymmetry_worker = None
    gui._asymmetry_cache_key = None

    tools = tk.Frame(host, bg=ELEVATED, highlightthickness=0)
    tools.grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_SM, PAD_XS))
    tools.columnconfigure(8, weight=1)

    tk.Label(
        tools, text="Subject mass (kg)", bg=ELEVATED, fg=TEXT_SECONDARY, font=FONT_KPI_TITLE
    ).grid(row=0, column=0, padx=(PAD_SM, PAD_XS), pady=PAD_XS)
    gui.var_asymmetry_mass = tk.StringVar(value="")
    mass_entry = ttk.Entry(tools, textvariable=gui.var_asymmetry_mass, width=8)
    mass_entry.grid(row=0, column=1, padx=(0, PAD_SM), pady=PAD_XS)
    create_tooltip(
        mass_entry,
        "Optional. Enables experimental I·α sagittal moment estimates. "
        "Leave blank to keep moments disabled. Not OpenSim Inverse Dynamics.",
    )
    apply = ttk.Button(
        tools,
        text="Apply",
        width=8,
        command=lambda: schedule_asymmetry_analysis(gui, force=True),
        takefocus=False,
    )
    apply.grid(row=0, column=2, padx=(0, PAD_SM), pady=PAD_XS)

    gui.var_asymmetry_highlight = tk.BooleanVar(value=False)
    gui._asymmetry_highlight = False

    def _on_highlight() -> None:
        gui._asymmetry_highlight = bool(gui.var_asymmetry_highlight.get())
        if getattr(gui, "_asymmetry_report", None) is not None:
            render_asymmetry_workspace(gui)

    highlight = ttk.Checkbutton(
        tools,
        text="Highlight unusual L/R",
        variable=gui.var_asymmetry_highlight,
        command=_on_highlight,
        takefocus=False,
    )
    highlight.grid(row=0, column=6, padx=(PAD_SM, PAD_XS), pady=PAD_XS)
    create_tooltip(
        highlight,
        "Shade gait-cycle phases with large phase-matched left–right "
        "differences and list time windows. Selecting a window seeks video "
        "and the existing 3D skeleton. Not a clinical finding.",
    )

    btn_csv = ttk.Button(
        tools, text="CSV", width=6, command=lambda: _export(gui, "csv"), takefocus=False
    )
    btn_json = ttk.Button(
        tools, text="JSON", width=6, command=lambda: _export(gui, "json"), takefocus=False
    )
    btn_csv.grid(row=0, column=3, padx=(0, PAD_XS), pady=PAD_XS)
    btn_json.grid(row=0, column=4, pady=PAD_XS)
    try:
        from stablewalk.ui.tk.dashboard_panel_maximize import install_maximize_button_grid

        install_maximize_button_grid(
            gui, tools, "asymmetry", row=0, column=5, sticky="e", padx=(PAD_SM, 0)
        )
    except Exception:
        pass

    gui.lbl_asymmetry_status = tk.Label(
        tools,
        text="Derived L/R indices — not a clinical diagnosis",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="e",
        wraplength=260,
        justify=tk.RIGHT,
    )
    gui.lbl_asymmetry_status.grid(row=0, column=8, sticky="e", padx=PAD_SM, pady=PAD_XS)
    from stablewalk.ui.tk.layout_scale import bind_status_wrap

    bind_status_wrap(tools, gui.lbl_asymmetry_status, reserve=420, minimum=100)

    gui.lbl_asymmetry_meta = tk.Label(
        host,
        text=(
            "3D skeleton and playback stay on Overview / Kinematics. "
            "Coordinates are inferred monocular 3D unless a calibrated metric session is loaded."
        ),
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=960,
    )
    gui.lbl_asymmetry_meta.grid(row=1, column=0, sticky="ew", padx=PAD_MD, pady=(0, PAD_XS))

    def _wrap_asymmetry_copy(_event: object | None = None) -> None:
        try:
            width = int(host.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 80:
            return
        try:
            gui.lbl_asymmetry_meta.configure(wraplength=max(220, width - 48))
        except tk.TclError:
            pass

    host.bind("<Configure>", _wrap_asymmetry_copy, add="+")

    body = ttk.Frame(host)
    body.grid(row=3, column=0, sticky="nsew", padx=PAD_SM, pady=(0, PAD_SM))
    body.columnconfigure(0, weight=1)
    body.rowconfigure(0, weight=1)
    gui._asymmetry_body = body

    from stablewalk.ui.tk.analysis_split import create_h_split, split_add

    paned = create_h_split(body)
    paned.grid(row=0, column=0, sticky="nsew")
    gui._asymmetry_split = paned

    table_host = ttk.Frame(paned)
    gui._asymmetry_table_host = table_host
    table_host.rowconfigure(0, weight=1)
    table_host.rowconfigure(1, weight=0)
    table_host.rowconfigure(2, weight=0)
    table_host.columnconfigure(0, weight=1)
    table_host.columnconfigure(1, weight=0)

    tree = ttk.Treeview(
        table_host, columns=_TREE_COLS, show="headings", height=12, selectmode="browse"
    )
    headings = {
        "metric": "Metric",
        "units": "Units",
        "left": "Left",
        "right": "Right",
        "delta": "Δ",
        "si": "AI / SI %",
        "asi": "ASI %",
        "nsi": "Similarity",
        "cycles": "N L/R",
        "quality": "Quality",
    }
    widths = {
        "metric": 176,
        "units": 88,
        "left": 64,
        "right": 64,
        "delta": 78,
        "si": 76,
        "asi": 72,
        "nsi": 84,
        "cycles": 58,
        "quality": 100,
    }
    for col in _TREE_COLS:
        tree.heading(col, text=headings[col])
        tree.column(
            col,
            width=widths[col],
            minwidth=120 if col == "metric" else 64,
            stretch=False,
        )
    scroll = ttk.Scrollbar(table_host, orient="vertical", command=tree.yview)
    xscroll = ttk.Scrollbar(table_host, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=scroll.set, xscrollcommand=xscroll.set)
    tree.grid(row=0, column=0, sticky="nsew")
    scroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    gui._asymmetry_tree = tree
    table_host.rowconfigure(0, weight=1)
    table_host.rowconfigure(1, weight=0)
    table_host.rowconfigure(2, weight=0)
    table_host.rowconfigure(3, weight=0)
    table_host.rowconfigure(4, weight=0)
    table_host.rowconfigure(5, weight=0)

    formula_bar = tk.Frame(table_host, bg=PANEL, highlightthickness=0)
    formula_bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=PAD_XS, pady=(PAD_XS, 0))
    gui._asymmetry_formula_open = False
    gui.btn_asymmetry_formula = ttk.Button(
        formula_bar,
        text="Show formulas",
        width=14,
        takefocus=False,
    )
    gui.btn_asymmetry_formula.pack(side=tk.LEFT)
    gui.lbl_asymmetry_index_strip = tk.Label(
        formula_bar,
        text=_INDEX_STRIP,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=360,
    )
    gui.lbl_asymmetry_index_strip.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(PAD_SM, 0))

    gui.lbl_asymmetry_legend = tk.Label(
        table_host,
        text=(
            "Asymmetry index AI / SI = 200(L−R)/(L+R) signed relative % of the mean "
            "(0 = symmetry). Absolute symmetry index ASI = 200|L−R|/(|L|+|R|) "
            "unsigned relative % (0 = symmetry). Normalized similarity SI_norm = "
            "2 min(L,R)/(L+R) in [0, 1] (1 = symmetry). These are not interchangeable. "
            "Peak timing Δ is circular percentage points, not SI. Overall symmetry "
            "on Results is a multi-feature composite — not knee SI_norm."
        ),
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=420,
    )
    gui.lbl_asymmetry_legend.grid(
        row=3, column=0, columnspan=2, sticky="ew", padx=PAD_XS, pady=(PAD_XS, 0)
    )
    gui.lbl_asymmetry_legend.grid_remove()
    gui.lbl_asymmetry_detail = tk.Label(
        table_host,
        text="Select a metric for units, cycle counts, and quality.",
        bg=PANEL,
        fg=TEXT_SECONDARY,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=420,
    )
    gui.lbl_asymmetry_detail.grid(
        row=4, column=0, columnspan=2, sticky="ew", padx=PAD_XS, pady=(PAD_XS, 0)
    )
    gui._asymmetry_row_by_iid = {}
    gui._asymmetry_detail_summary = "Select a metric for units, cycle counts, and quality."
    gui._asymmetry_detail_formula = ""

    interval_bar = tk.Frame(table_host, bg=PANEL, highlightthickness=0)
    interval_bar.grid(row=5, column=0, columnspan=2, sticky="ew", padx=PAD_XS, pady=(PAD_XS, 0))
    interval_bar.columnconfigure(0, weight=1)
    tk.Label(
        interval_bar,
        text="Unusual L/R sections (phase-matched). Click to seek video / 3D.",
        bg=PANEL,
        fg=TEXT_SECONDARY,
        font=FONT_UI_SM,
        anchor="w",
    ).grid(row=0, column=0, sticky="ew")
    gui.lst_asymmetry_intervals = tk.Listbox(
        interval_bar,
        height=4,
        activestyle="dotbox",
        exportselection=False,
    )
    gui.lst_asymmetry_intervals.grid(row=1, column=0, sticky="ew", pady=(2, 0))
    gui._asymmetry_interval_items = []

    def _on_interval(_event: object | None = None) -> None:
        _seek_unusual_interval(gui)

    gui.lst_asymmetry_intervals.bind("<<ListboxSelect>>", _on_interval, add="+")

    def _toggle_asymmetry_formulas() -> None:
        open_ = not bool(getattr(gui, "_asymmetry_formula_open", False))
        gui._asymmetry_formula_open = open_
        try:
            if open_:
                gui.lbl_asymmetry_legend.grid()
                gui.btn_asymmetry_formula.configure(text="Hide formulas")
            else:
                gui.lbl_asymmetry_legend.grid_remove()
                gui.btn_asymmetry_formula.configure(text="Show formulas")
        except tk.TclError:
            pass
        _apply_asymmetry_detail_text(gui)

    gui.btn_asymmetry_formula.configure(command=_toggle_asymmetry_formulas)

    def _on_select(_event: object | None = None) -> None:
        _show_asymmetry_detail(gui)

    tree.bind("<<TreeviewSelect>>", _on_select, add="+")

    def _wrap_table_copy(_event: object | None = None) -> None:
        try:
            width = int(table_host.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 80:
            return
        wrap = max(180, width - 16)
        try:
            gui.lbl_asymmetry_legend.configure(wraplength=wrap)
            gui.lbl_asymmetry_detail.configure(wraplength=wrap)
            strip = getattr(gui, "lbl_asymmetry_index_strip", None)
            if strip is not None:
                strip.configure(wraplength=max(160, width - 140))
        except tk.TclError:
            pass

    table_host.bind("<Configure>", _wrap_table_copy, add="+")

    fig = Figure(figsize=(8.8, 7.4), dpi=100, facecolor=PANEL)
    gs = fig.add_gridspec(2, 2, height_ratios=(1.62, 0.88))
    import numpy as np

    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ],
        dtype=object,
    )
    layout_asymmetry_figure(fig)
    graph_host = tk.Frame(paned, bg=PANEL, highlightthickness=0)
    split_add(paned, table_host, minsize=280)
    split_add(paned, graph_host, minsize=360)

    def _place_asymmetry_sash(_event: object | None = None) -> None:
        try:
            pw = int(paned.winfo_width() or 0)
            if pw < 500:
                return
            bucket = pw // 80
            if getattr(gui, "_asymmetry_sash_for_width", None) == bucket:
                return
            paned.sash_place(0, max(280, min(int(pw * 0.38), pw - 400)), 0)
            gui._asymmetry_sash_for_width = bucket
        except tk.TclError:
            pass

    paned.bind("<Configure>", _place_asymmetry_sash, add="+")
    gui._place_asymmetry_sash = _place_asymmetry_sash
    graph_host.columnconfigure(0, weight=1)
    graph_host.rowconfigure(0, weight=1)
    graph_host.grid_propagate(False)
    canvas = FigureCanvasTkAgg(fig, master=graph_host)
    widget = canvas.get_tk_widget()
    widget.configure(bg=PANEL, highlightthickness=0)
    widget.grid(row=0, column=0, sticky="nsew")
    gui._asymmetry_graph_host = graph_host
    gui.fig_asymmetry = fig
    gui.ax_asymmetry = axes
    gui.canvas_asymmetry = canvas
    gui._asymmetry_canvas_widget = widget
    try:
        from stablewalk.ui.tk.dashboard_layout import _bind_figure_resize

        _bind_figure_resize(canvas, fig, min_px=80)
    except Exception:
        pass
    gui._asymmetry_phase_lines = []

    gui.lbl_asymmetry_empty = EmptyState(
        host,
        "No gait recording.\nLoad a walking video to compute left–right asymmetry.",
        bg=PANEL,
    )
    gui.lbl_asymmetry_empty.grid(row=3, column=0, sticky="nsew")

    gui._schedule_asymmetry_analysis = lambda force=False: schedule_asymmetry_analysis(
        gui, force=force
    )
    gui._update_asymmetry_playhead = lambda: update_asymmetry_playhead(gui)
    gui._render_asymmetry = lambda: render_asymmetry_workspace(gui)

    nb = getattr(gui, "_motion_page_notebook", None)
    if nb is not None:
        nb.bind("<<NotebookTabChanged>>", lambda _e: schedule_asymmetry_analysis(gui), add="+")
    return host


def _asymmetry_page_selected(gui: Any) -> bool:
    from stablewalk.ui.tk.gait_correlation_tabs import (
        SECTION_TITLE,
        asymmetry_tab_selected,
    )

    if getattr(gui, "_gait_corr_notebook", None) is not None:
        return asymmetry_tab_selected(gui)
    nb = getattr(gui, "_motion_page_notebook", None)
    if nb is None:
        return True
    try:
        text = str(nb.tab(nb.select(), "text")).strip()
    except tk.TclError:
        return False
    if text == SECTION_TITLE:
        return asymmetry_tab_selected(gui)
    return text == "Asymmetry"


def _parse_mass(gui: Any) -> float | None:
    var = getattr(gui, "var_asymmetry_mass", None)
    if var is None:
        return None
    raw = str(var.get() or "").strip()
    if not raw:
        return None
    try:
        mass = float(raw)
    except ValueError:
        return None
    return mass if mass > 0 else None


def schedule_asymmetry_analysis(gui: Any, *, force: bool = False) -> None:
    from stablewalk.ui.tk.gait_correlation_tabs import recording_cache_key

    rec = getattr(gui, "gait_motion", None)
    if rec is None:
        gui._asymmetry_report = None
        gui._asymmetry_cache_key = None
        _paint_empty(gui)
        return
    key = recording_cache_key(rec)
    if not force:
        if (
            getattr(gui, "_asymmetry_cache_key", None) == key
            and gui._asymmetry_report is not None
        ):
            if _asymmetry_page_selected(gui):
                render_asymmetry_workspace(gui)
            return
        worker_obj = getattr(gui, "_asymmetry_worker", None)
        if (
            worker_obj is not None
            and getattr(worker_obj, "is_alive", lambda: False)()
            and getattr(gui, "_asymmetry_cache_key", None) == key
        ):
            return
    token = int(getattr(gui, "_asymmetry_token", 0)) + 1
    gui._asymmetry_token = token
    sequence = getattr(gui, "sequence", None)
    cycles = getattr(gui, "_gait_cycle", None)
    mass = _parse_mass(gui)
    run_name = getattr(gui, "_opensim_run_name", None) or getattr(gui, "opensim_run_name", None)
    status = getattr(gui, "lbl_asymmetry_status", None)
    if status is not None:
        try:
            status.configure(text="Computing left–right asymmetry…")
        except tk.TclError:
            pass

    gui._asymmetry_pending = None
    gui._asymmetry_cache_key = key

    def worker() -> None:
        try:
            from stablewalk.analysis.asymmetry3d.report import analyze_asymmetry3d

            report = analyze_asymmetry3d(
                rec,
                cycles=cycles,
                sequence=sequence,
                subject_mass_kg=mass,
                opensim_run_name=str(run_name) if run_name else None,
                detect_if_missing=cycles is None,
            )
            err = None
        except Exception as exc:  # pragma: no cover - GUI guard
            report = None
            err = exc
            logger.exception("Asymmetry analysis failed")
        gui._asymmetry_pending = (token, report, err)

    def drain() -> None:
        if int(getattr(gui, "_asymmetry_token", 0)) != token:
            return
        pending = getattr(gui, "_asymmetry_pending", None)
        if pending is None:
            worker_obj = getattr(gui, "_asymmetry_worker", None)
            if worker_obj is not None and getattr(worker_obj, "is_alive", lambda: False)():
                try:
                    gui.root.after(50, drain)
                except Exception:
                    pass
            return
        p_token, report, err = pending
        if p_token != token:
            return
        gui._asymmetry_pending = None
        if err is not None:
            if status is not None:
                try:
                    status.configure(text=f"Asymmetry analysis failed: {err}")
                except tk.TclError:
                    pass
            failed = type("FailedAsymmetry", (), {})()
            failed.available = False
            failed.failed = True
            failed.warnings = [f"Analysis failed: {err}"]
            gui._asymmetry_report = failed
            _paint_empty(gui, failed)
            refresh_rv = getattr(gui, "_refresh_research_validation", None)
            if callable(refresh_rv):
                try:
                    refresh_rv()
                except Exception:
                    logger.debug("research validation refresh after asymmetry failure", exc_info=True)
            return
        gui._asymmetry_report = report
        render_asymmetry_workspace(gui)
        refresh_rv = getattr(gui, "_refresh_research_validation", None)
        if callable(refresh_rv):
            try:
                refresh_rv()
            except Exception:
                logger.debug("research validation refresh after asymmetry failed", exc_info=True)

    thread = threading.Thread(target=worker, name="stablewalk-asymmetry3d", daemon=True)
    gui._asymmetry_worker = thread
    thread.start()
    try:
        gui.root.after(50, drain)
    except Exception:
        drain()


def insufficient_asymmetry_copy(report: Any | None) -> str:
    """Empty-state copy: symmetric vs insufficient vs failed; never invents L/R indices."""
    if report is None:
        return "No gait recording.\nLoad a walking video to compute left–right asymmetry."
    if bool(getattr(report, "failed", False)):
        warns = [str(w).strip() for w in (getattr(report, "warnings", None) or []) if str(w).strip()]
        detail = warns[0] if warns else "left–right asymmetry could not be computed."
        return f"Analysis failed.\n{detail}"
    warns = [str(w).strip() for w in (getattr(report, "warnings", None) or []) if str(w).strip()]
    if warns:
        return "Insufficient data for asymmetry analysis.\n" + "\n".join(warns[:4])
    return "Insufficient data for asymmetry analysis.\nNeed ≥ 4 kinematic frames and left/right gait cycles."


def _paint_empty(gui: Any, report: Any | None = None) -> None:
    empty = getattr(gui, "lbl_asymmetry_empty", None)
    widget = getattr(gui, "_asymmetry_canvas_widget", None)
    tree = getattr(gui, "_asymmetry_tree", None)
    if empty is not None:
        try:
            empty.configure(text=insufficient_asymmetry_copy(report))
            empty.lift()
        except tk.TclError:
            pass
    if widget is not None:
        try:
            widget.lower()
        except tk.TclError:
            pass
    if tree is not None:
        try:
            for item in tree.get_children():
                tree.delete(item)
        except tk.TclError:
            pass
    _fill_unusual_intervals(gui, None)


def _fill_unusual_intervals(gui: Any, report: Any | None) -> None:
    box = getattr(gui, "lst_asymmetry_intervals", None)
    if box is None:
        return
    try:
        box.delete(0, tk.END)
    except tk.TclError:
        return
    items = []
    quality = getattr(report, "quality", None) if report is not None else None
    intervals = list(getattr(quality, "unusual_intervals", None) or [])
    if not intervals:
        try:
            box.insert(
                tk.END,
                "No phase-matched L/R sections flagged. "
                "Highlight unusual L/R only shades the curves.",
            )
        except tk.TclError:
            pass
        gui._asymmetry_interval_items = []
        return
    for interval in intervals:
        label = (
            f"{interval.start_s:.2f}–{interval.end_s:.2f} s  "
            f"{interval.side} {interval.channel}  "
            f"Δ {interval.peak_delta:.1f}  "
            f"φ {interval.phase_start_pct:.0f}–{interval.phase_end_pct:.0f}%"
        )
        try:
            box.insert(tk.END, label)
        except tk.TclError:
            break
        items.append(interval)
    gui._asymmetry_interval_items = items


def _seek_unusual_interval(gui: Any) -> None:
    box = getattr(gui, "lst_asymmetry_intervals", None)
    items = list(getattr(gui, "_asymmetry_interval_items", None) or [])
    if box is None or not items:
        return
    try:
        sel = box.curselection()
    except tk.TclError:
        return
    if not sel:
        return
    idx = int(sel[0])
    if idx < 0 or idx >= len(items):
        return
    interval = items[idx]
    rec = getattr(gui, "gait_motion", None)
    try:
        from stablewalk.analysis.asymmetry3d.quality import (
            resolve_unusual_interval_target,
        )

        frame, time_s = resolve_unusual_interval_target(interval, rec)
    except Exception:
        frame, time_s = getattr(interval, "start_frame", None), float(interval.start_s)
    gui._asymmetry_last_seek = (frame, float(time_s), str(interval.side))
    gui._asymmetry_seek_side = str(interval.side or "")
    seek = getattr(gui, "_seek_motion_workspace_time", None)
    if callable(seek):
        try:
            seek(float(time_s))
        except Exception:
            logger.debug("asymmetry interval seek failed", exc_info=True)


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        if value != value or value in (float("inf"), float("-inf")):
            return "—"
    except Exception:
        return "—"
    return f"{value:.{digits}g}"


def _fmt_delta(row: Any) -> str:
    kind = str(getattr(row, "comparison_kind", "") or "")
    diff = getattr(row, "difference", None)
    units = str(getattr(row, "difference_units", "") or "")
    if diff is None:
        return "—"
    if kind == "timing" or units == "percentage points":
        return f"{_fmt(diff, 2)} pp"
    if units:
        return f"{_fmt(diff, 3)} {units}"
    return _fmt(diff, 3)


def _quality_label(row: Any) -> str:
    quality = str(getattr(row, "quality", "") or "")
    if quality in {"insufficient", "unavailable"}:
        return quality
    if not bool(getattr(row, "reliable", False)) and "unreliable" not in quality:
        return f"{quality} · unreliable" if quality else "unreliable"
    return quality or "—"


def _metric_units(row: Any) -> str:
    kind = str(getattr(row, "comparison_kind", "") or "")
    units = str(getattr(row, "units", "") or "").strip()
    if kind == "timing":
        return "% gait cycle"
    return units or "—"


def _quality_explanation(row: Any) -> str:
    quality = _quality_label(row)
    n_l = int(getattr(row, "n_left_cycles", 0) or 0)
    n_r = int(getattr(row, "n_right_cycles", 0) or 0)
    bits: list[str] = [f"Quality: {quality}"]
    if quality == "insufficient":
        if n_l < 1 or n_r < 1:
            bits.append("Need ≥1 left and ≥1 right ipsilateral HS–HS cycle.")
        if getattr(row, "left", None) is None or getattr(row, "right", None) is None:
            bits.append("A left or right value is missing, so SI/ASI are not published.")
    elif quality == "unavailable":
        name = str(getattr(row, "name", "") or "").lower()
        if "clearance_ground" in name or "ground" in name:
            bits.append(
                "Ground clearance withheld — the estimated floor plane is "
                "noisy or implausible. No physical calibration was invented."
            )
        elif "step_length" in name:
            bits.append(
                "Spatial step length needs a calibrated metric session and "
                "a reliable camera view."
            )
        elif "moment" in name:
            bits.append(
                "Estimated I·α moments need subject mass (kg). Not OpenSim "
                "Inverse Dynamics."
            )
        else:
            bits.append(
                "This quantity is disabled for the current session "
                "(for example no mass, no metric scale, or withheld ground)."
            )
    elif "unreliable" in quality or not bool(getattr(row, "reliable", False)):
        bits.append("Marked unreliable — not a reassuring symmetry score.")
    note = str(getattr(row, "note", "") or "").strip()
    if note:
        bits.append(note)
    return " ".join(bits)


def asymmetry_plot_id_for_row(row: Any) -> str:
    """Which of the four plots a table row belongs to (display only)."""
    name = str(getattr(row, "name", "") or "").lower()
    if any(key in name for key in ("stance", "swing", "step_duration", "stride_duration", "moment")):
        return "phase"
    if "rom" in name or "thigh" in name or "shank" in name:
        return "rom"
    if any(key in name for key in ("heel", "toe", "vertical", "step_length", "clearance")):
        return "foot"
    if "hip" in name or "ankle" in name:
        return "rom" if "rom" in name else "knee"
    return "knee"


def _highlight_asymmetry_plot(gui: Any, plot_id: str | None) -> None:
    axes = getattr(gui, "ax_asymmetry", None)
    canvas = getattr(gui, "canvas_asymmetry", None)
    if axes is None:
        return
    mapping = {
        "knee": axes[0, 0],
        "foot": axes[0, 1],
        "rom": axes[1, 0],
        "phase": axes[1, 1],
    }
    for pid, ax in mapping.items():
        active = bool(plot_id) and pid == plot_id
        for side, spine in ax.spines.items():
            if side in {"bottom", "left"} or active:
                spine.set_visible(True)
                spine.set_color(ACCENT if active else BORDER)
                spine.set_linewidth(2.15 if active else 1.15)
            else:
                spine.set_visible(False)
        title = ax.get_title() or ""
        marker = "▸ "
        if title.startswith(marker):
            title = title[len(marker) :]
        if active and title:
            ax.set_title(marker + title, color=ACCENT, fontsize=11)
        elif title:
            ax.set_title(title, color=TEXT, fontsize=11)
    try:
        if canvas is not None:
            canvas.draw_idle()
    except Exception:
        pass


def _show_asymmetry_detail(gui: Any) -> None:
    label = getattr(gui, "lbl_asymmetry_detail", None)
    tree = getattr(gui, "_asymmetry_tree", None)
    rows = getattr(gui, "_asymmetry_row_by_iid", None) or {}
    if label is None or tree is None:
        return
    try:
        selected = tree.selection()
    except tk.TclError:
        return
    row = rows.get(selected[0]) if selected else None
    if row is None:
        gui._asymmetry_detail_summary = "Select a metric for units, cycle counts, and quality."
        gui._asymmetry_detail_formula = ""
        _apply_asymmetry_detail_text(gui)
        _highlight_asymmetry_plot(gui, None)
        return
    kind = str(getattr(row, "comparison_kind", "scalar") or "scalar")
    parts = [
        f"{row.display} · units {_metric_units(row)}",
        f"L {_fmt(row.left)}  R {_fmt(row.right)} {_metric_units(row)}",
        f"N {int(row.n_left_cycles)} L / {int(row.n_right_cycles)} R HS–HS cycles",
        _quality_explanation(row),
    ]
    if kind == "timing":
        parts.append(
            f"Δ {_fmt(row.difference, 2)} percentage points (circular; not SI)"
        )
    elif row.difference is not None:
        units = row.difference_units or row.units
        parts.append(f"Δ {_fmt(row.difference, 3)} {units} (not SI/ASI/SI_norm)")
    else:
        parts.append(
            f"SI {_fmt(row.si_percent, 2)} signed% · "
            f"ASI {_fmt(row.asi_percent, 2)} abs% · "
            f"SI_norm {_fmt(row.si_norm, 3)} [0–1]"
            + (
                " (P95−P5 ROM; not Results mean-curve ROM SI_norm)"
                if str(getattr(row, "name", "")).endswith("_flexion_rom")
                else ""
            )
        )
    formula = str(getattr(row, "formula", "") or "").strip()
    note = str(getattr(row, "note", "") or "").strip()
    extra_bits = [bit for bit in (formula, note) if bit]
    gui._asymmetry_detail_summary = " · ".join(parts)
    gui._asymmetry_detail_formula = ("\n" + "\n".join(extra_bits)) if extra_bits else ""
    _apply_asymmetry_detail_text(gui)
    _highlight_asymmetry_plot(gui, asymmetry_plot_id_for_row(row))


def _apply_asymmetry_detail_text(gui: Any) -> None:
    label = getattr(gui, "lbl_asymmetry_detail", None)
    if label is None:
        return
    summary = str(getattr(gui, "_asymmetry_detail_summary", "") or "")
    extra = str(getattr(gui, "_asymmetry_detail_formula", "") or "")
    text = summary + extra if bool(getattr(gui, "_asymmetry_formula_open", False)) else summary
    try:
        label.configure(text=text)
    except tk.TclError:
        pass


def render_asymmetry_workspace(gui: Any) -> None:
    report = getattr(gui, "_asymmetry_report", None)
    empty = getattr(gui, "lbl_asymmetry_empty", None)
    widget = getattr(gui, "_asymmetry_canvas_widget", None)
    tree = getattr(gui, "_asymmetry_tree", None)
    if report is None or not report.available:
        _paint_empty(gui, report)
        meta = getattr(gui, "lbl_asymmetry_meta", None)
        if meta is not None:
            try:
                meta.configure(text=insufficient_asymmetry_copy(report).replace("\n", " "))
            except tk.TclError:
                pass
        return
    if empty is not None:
        try:
            empty.lower()
        except tk.TclError:
            pass
    if widget is not None:
        try:
            widget.lift()
        except tk.TclError:
            pass
    if tree is not None:
        try:
            for item in tree.get_children():
                tree.delete(item)
            gui._asymmetry_row_by_iid = {}
            for row in report.scalars:
                if row.name.startswith("traj_rmse_"):
                    continue
                iid = tree.insert(
                    "",
                    "end",
                    values=(
                        row.display,
                        _metric_units(row),
                        _fmt(row.left),
                        _fmt(row.right),
                        _fmt_delta(row),
                        _fmt(row.si_percent, 2),
                        _fmt(row.asi_percent, 2),
                        _fmt(row.si_norm, 3),
                        f"{int(row.n_left_cycles)}/{int(row.n_right_cycles)}",
                        _quality_label(row),
                    ),
                )
                gui._asymmetry_row_by_iid[iid] = row
            kids = tree.get_children()
            if kids:
                tree.selection_set(kids[0])
                tree.focus(kids[0])
            _show_asymmetry_detail(gui)
        except tk.TclError:
            pass
    _fill_unusual_intervals(gui, report)

    kin = report.kinematics
    events = report.events
    meta = getattr(gui, "lbl_asymmetry_meta", None)
    if meta is not None:
        try:
            meta.configure(
                text=(
                    f"{kin.provenance} · {kin.coordinate_system} · {kin.units} · "
                    f"view {events.view_type} ({events.reliability}) · "
                    f"{len(events.left_cycles)} L / {len(events.right_cycles)} R "
                    f"ipsilateral HS–HS cycles · "
                    f"HS {len(events.heel_strikes)} · TO {len(events.toe_offs)}. "
                    "Alignment: ipsilateral heel-strike"
                    + (
                        " (verified)."
                        if bool(getattr(events, "alignment_verified", False))
                        else " (HS match not verified)."
                    )
                    + " "
                    "Overall Results symmetry is a composite, not knee SI_norm. "
                    "3D skeleton remains on Overview / Kinematics. "
                    + (
                        getattr(report, "quality", None).summary_line()
                        if getattr(report, "quality", None) is not None
                        else ""
                    )
                )
            )
        except tk.TclError:
            pass
    status = getattr(gui, "lbl_asymmetry_status", None)
    if status is not None:
        try:
            n_ok = sum(
                1
                for s in report.scalars
                if s.left is not None
                and s.right is not None
                and int(getattr(s, "n_left_cycles", 0) or 0) >= 1
                and int(getattr(s, "n_right_cycles", 0) or 0) >= 1
            )
            n_ins = sum(
                1
                for s in report.scalars
                if str(getattr(s, "quality", "") or "") in {"insufficient", "unavailable"}
            )
            status.configure(
                text=(
                    f"{n_ok} scalars with L, R, and ≥1 HS–HS cycle · "
                    f"{n_ins} insufficient/unavailable · "
                    "SI / ASI / SI_norm / timing Δ are distinct · not a diagnosis"
                )
            )
        except tk.TclError:
            pass
    _draw_figure(gui, report)


def _pair(report, name: str):
    return report.trajectory_by_name(name)


def _draw_mean_band(
    ax,
    pair,
    *,
    ylabel: str,
    title: str,
    highlight_bands: list[tuple[float, float]] | None = None,
) -> None:
    apply_chart_panel_style(ax)
    if pair is None or not pair.aligned:
        ax.text(
            0.5,
            0.5,
            "Need left and right gait cycles\n(unaligned frames are not compared)",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
        style_chart_title(ax, title, pad=6)
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel(ylabel)
        return
    pct = list(pair.phase_pct)
    ax.plot(pct, pair.left_mean, color=SIDE_LEFT, lw=SERIES_LINE_WIDTH, label="Left")
    ax.plot(pct, pair.right_mean, color=SIDE_RIGHT, lw=SERIES_LINE_WIDTH, label="Right")
    ax.fill_between(
        pct,
        pair.left_mean - pair.left_std,
        pair.left_mean + pair.left_std,
        color=SIDE_LEFT,
        alpha=0.15,
        linewidth=0,
    )
    ax.fill_between(
        pct,
        pair.right_mean - pair.right_std,
        pair.right_mean + pair.right_std,
        color=SIDE_RIGHT,
        alpha=0.15,
        linewidth=0,
    )
    if highlight_bands:
        for lo, hi in highlight_bands:
            if hi < lo:
                lo, hi = hi, lo
            ax.axvspan(float(lo), float(hi), color=ACCENT, alpha=0.14, linewidth=0, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel(ylabel)
    style_chart_title(ax, title, pad=4)
    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=7,
        facecolor=PANEL,
        edgecolor=BORDER,
        labelcolor=TEXT,
        framealpha=0.88,
        borderpad=0.25,
    )


def _draw_figure(gui: Any, report: Any) -> None:
    axes = getattr(gui, "ax_asymmetry", None)
    fig = getattr(gui, "fig_asymmetry", None)
    canvas = getattr(gui, "canvas_asymmetry", None)
    if axes is None or canvas is None:
        return
    ax_knee, ax_foot = axes[0]
    ax_rom, ax_phase = axes[1]
    for ax in (ax_knee, ax_foot, ax_rom, ax_phase):
        ax.clear()
    bands = []
    if bool(getattr(gui, "_asymmetry_highlight", False)):
        quality = getattr(report, "quality", None)
        bands = list(getattr(quality, "unusual_phase_bands", None) or [])
    _draw_mean_band(
        ax_knee,
        _pair(report, "knee_flexion"),
        ylabel="Knee flexion (deg)",
        title="Knee flexion — phase-matched HS–HS",
        highlight_bands=bands,
    )
    heel_units = str(getattr(report.kinematics, "units", "") or "m")
    _draw_mean_band(
        ax_foot,
        _pair(report, "heel_vertical"),
        ylabel=f"Heel height ({heel_units})",
        title="Heel height — phase-matched HS–HS",
        highlight_bands=bands,
    )

    apply_chart_panel_style(ax_rom)
    rom_names = (
        ("knee_flexion_rom", "Knee"),
        ("hip_flexion_rom", "Hip"),
        ("ankle_flexion_rom", "Ankle"),
        ("thigh_rom", "Thigh"),
        ("shank_rom", "Shank"),
    )
    labels = []
    left_vals = []
    right_vals = []
    for name, short in rom_names:
        row = report.scalar_by_name(name)
        if row is None:
            continue
        if row.left is None and row.right is None:
            continue
        labels.append(short)
        left_vals.append(float("nan") if row.left is None else row.left)
        right_vals.append(float("nan") if row.right is None else row.right)
    if labels:
        import numpy as np

        x = np.arange(len(labels), dtype=float)
        w = 0.36
        ax_rom.bar(x - w / 2, left_vals, w, color=SIDE_LEFT, label="Left")
        ax_rom.bar(x + w / 2, right_vals, w, color=SIDE_RIGHT, label="Right")
        ax_rom.set_xticks(x)
        ax_rom.set_xticklabels(labels, fontsize=8)
        ax_rom.set_ylabel("ROM (deg)")
        style_chart_title(ax_rom, "ROM P95−P5 (joint / segment)", pad=4)
        ax_rom.legend(
            loc="upper right",
            frameon=True,
            fontsize=7,
            facecolor=PANEL,
            edgecolor=BORDER,
            labelcolor=TEXT,
            framealpha=0.88,
        )
    else:
        ax_rom.text(
            0.5,
            0.5,
            "ROM unavailable\n(need usable left and right flexion series)",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax_rom.transAxes,
        )

    apply_chart_panel_style(ax_phase)
    events = report.events
    st_l = report.scalar_by_name("stance_duration")
    sw_l = report.scalar_by_name("swing_duration")
    if (
        st_l is not None
        and sw_l is not None
        and st_l.left is not None
        and st_l.right is not None
        and sw_l.left is not None
        and sw_l.right is not None
    ):
        import numpy as np

        cats = np.arange(2)
        stance = [st_l.left, st_l.right]
        swing = [sw_l.left, sw_l.right]
        ax_phase.bar(cats, stance, color="#64748b", label="Stance")
        ax_phase.bar(cats, swing, bottom=stance, color="#f59e0b", label="Swing")
        ax_phase.set_xticks(cats)
        ax_phase.set_xticklabels(["Left", "Right"])
        ax_phase.set_ylabel("Duration (s)")
        style_chart_title(ax_phase, "Stance / swing", pad=4)
        ax_phase.legend(
            loc="upper right",
            frameon=True,
            fontsize=7,
            facecolor=PANEL,
            edgecolor=BORDER,
            labelcolor=TEXT,
            framealpha=0.88,
        )
    elif report.moments is not None and report.moments.estimated_available:
        _plot_moments(ax_phase, report)
    else:
        reason = events.reliability_reason
        if report.moments is not None and report.moments.estimated_disabled_reason:
            reason = report.moments.estimated_disabled_reason
        ax_phase.text(
            0.5,
            0.5,
            reason,
            ha="center",
            va="center",
            color=MUTED,
            transform=ax_phase.transAxes,
            wrap=True,
        )
        style_chart_title(ax_phase, "Stance / swing", pad=6)

    for ax in (ax_knee, ax_foot, ax_rom, ax_phase):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        if ax.xaxis.label is not None:
            ax.xaxis.label.set_color(MUTED)
            ax.xaxis.label.set_fontsize(9)
        if ax.yaxis.label is not None:
            ax.yaxis.label.set_color(MUTED)
            ax.yaxis.label.set_fontsize(9)
    gui._asymmetry_phase_lines = []
    for ax in (ax_knee, ax_foot):
        line = ax.axvline(0.0, color=TEXT, lw=1.0, alpha=0.0)
        gui._asymmetry_phase_lines.append(line)
    try:
        if fig is not None:
            fig.set_facecolor(PANEL)
            h_px = float(fig.get_size_inches()[1]) * float(fig.get_dpi() or 100.0)
            layout_asymmetry_figure(fig, height_px=h_px)
        canvas.draw_idle()
    except Exception:
        pass
    selected = None
    tree = getattr(gui, "_asymmetry_tree", None)
    rows = getattr(gui, "_asymmetry_row_by_iid", None) or {}
    if tree is not None:
        try:
            sel = tree.selection()
            if sel:
                selected = rows.get(sel[0])
        except tk.TclError:
            selected = None
    if selected is not None:
        _highlight_asymmetry_plot(gui, asymmetry_plot_id_for_row(selected))


def _plot_moments(ax, report) -> None:
    apply_chart_panel_style(ax)
    series = report.moments.estimated
    knee = [s for s in series if s.joint == "knee" and s.available]
    if not knee:
        ax.text(0.5, 0.5, "Estimated moments unavailable", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    for s in knee:
        color = SIDE_LEFT if s.side == "left" else SIDE_RIGHT
        ax.plot(s.times_s, s.moment_nm, color=color, lw=SERIES_LINE_WIDTH, label=s.side.title())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Moment (N·m, experimental)")
    style_chart_title(ax, "Estimated knee moment (I·α, not ID)", pad=6)
    ax.legend(loc="upper right", frameon=True, fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)


def _gait_pct(
    report,
    time_s: float,
    *,
    prefer_side: str | None = None,
) -> float | None:
    left = list(report.events.left_cycles)
    right = list(report.events.right_cycles)
    side = str(prefer_side or "").lower()
    if side.startswith("right"):
        cycles = right + left
    elif side.startswith("left"):
        cycles = left + right
    else:
        cycles = left + right
    for cycle in cycles:
        if cycle.start_time_s <= time_s <= cycle.end_time_s and cycle.duration_s > 1e-6:
            return 100.0 * (time_s - cycle.start_time_s) / cycle.duration_s
    return None


def update_asymmetry_playhead(gui: Any) -> None:
    if not _asymmetry_page_selected(gui):
        return
    report = getattr(gui, "_asymmetry_report", None)
    lines = getattr(gui, "_asymmetry_phase_lines", None) or []
    canvas = getattr(gui, "canvas_asymmetry", None)
    if report is None or not lines or canvas is None:
        return
    time_s = 0.0
    try:
        from stablewalk.ui.playback_state import read_playback_state

        state = read_playback_state(gui)
        if state is not None:
            time_s = float(state.current_time_seconds)
    except Exception:
        rec = getattr(gui, "gait_motion", None)
        idx = getattr(gui, "current_frame", 0)
        if rec is not None and rec.snapshots:
            snap = rec.snapshot_at(int(idx or 0))
            if snap is not None:
                time_s = float(snap.time_s)
    pct = _gait_pct(
        report,
        time_s,
        prefer_side=str(getattr(gui, "_asymmetry_seek_side", "") or ""),
    )
    for line in lines:
        if pct is None:
            line.set_alpha(0.0)
        else:
            line.set_xdata([pct, pct])
            line.set_alpha(0.85)
    try:
        canvas.draw_idle()
    except Exception:
        pass


def _export(gui: Any, kind: str) -> None:
    report = getattr(gui, "_asymmetry_report", None)
    if report is None:
        messagebox.showinfo("Asymmetry", "Compute asymmetry before exporting.")
        return
    from stablewalk.analysis.asymmetry3d.export import export_asymmetry_csv, export_asymmetry_json
    from stablewalk.config import ANALYSIS_EXPORT_DIR

    ANALYSIS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if kind == "json":
        path = filedialog.asksaveasfilename(
            title="Export gait asymmetry JSON",
            defaultextension=".json",
            initialdir=str(ANALYSIS_EXPORT_DIR),
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        export_asymmetry_json(report, path)
    else:
        path = filedialog.asksaveasfilename(
            title="Export gait asymmetry CSV",
            defaultextension=".csv",
            initialdir=str(ANALYSIS_EXPORT_DIR),
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        export_asymmetry_csv(report, path)
    status = getattr(gui, "lbl_asymmetry_status", None)
    if status is not None:
        try:
            status.configure(text=f"Exported {kind.upper()}")
        except tk.TclError:
            pass
