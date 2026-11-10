"""Research dataset + independent Video A/B comparison workspace."""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from stablewalk.research.categories import CLINICAL_POLICY
from stablewalk.research.dataset import import_local_video, load_catalog
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

logger = logging.getLogger(__name__)

_SLOT_COLORS = {"A": "#38bdf8", "B": "#f97316"}


def build_research_compare_workspace(gui: Any, parent: tk.Misc) -> tk.Frame:
    host = ttk.Frame(parent)
    host.grid(row=0, column=0, sticky="nsew")
    parent.rowconfigure(0, weight=1)
    parent.columnconfigure(0, weight=1)
    host.columnconfigure(0, weight=1)
    host.rowconfigure(2, weight=1)
    gui._research_host = host
    gui._research_slot_a = None
    gui._research_slot_b = None
    gui._research_comparison = None
    gui._research_token = 0
    gui._research_token_a = 0
    gui._research_token_b = 0
    gui._research_busy_a = False
    gui._research_busy_b = False
    gui._research_prev_a = ""
    gui._research_prev_b = ""
    gui._research_same_source_ok = False

    tools = tk.Frame(host, bg=ELEVATED, highlightthickness=0)
    tools.grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_SM, PAD_XS))
    tools.columnconfigure(0, weight=1)
    btn_row = tk.Frame(tools, bg=ELEVATED, highlightthickness=0)
    btn_row.grid(row=0, column=0, sticky="ew")
    ttk.Button(btn_row, text="Import local video…", command=lambda: _import_clip(gui), takefocus=False).pack(
        side=tk.LEFT, padx=(PAD_SM, PAD_XS), pady=PAD_XS
    )
    ttk.Button(btn_row, text="Refresh catalog", command=lambda: _refresh_catalog(gui), takefocus=False).pack(
        side=tk.LEFT, padx=PAD_XS, pady=PAD_XS
    )
    gui.btn_research_analyze_a = ttk.Button(
        btn_row, text="Analyze A", command=lambda: schedule_slot_analysis(gui, "A"), takefocus=False
    )
    gui.btn_research_analyze_a.pack(side=tk.LEFT, padx=(PAD_MD, PAD_XS), pady=PAD_XS)
    gui.btn_research_analyze_b = ttk.Button(
        btn_row, text="Analyze B", command=lambda: schedule_slot_analysis(gui, "B"), takefocus=False
    )
    gui.btn_research_analyze_b.pack(side=tk.LEFT, padx=PAD_XS, pady=PAD_XS)
    gui.btn_research_compare = ttk.Button(
        btn_row, text="Compare", command=lambda: run_comparison(gui), takefocus=False, state=tk.DISABLED
    )
    gui.btn_research_compare.pack(side=tk.LEFT, padx=PAD_XS, pady=PAD_XS)
    gui.btn_research_export = ttk.Button(
        btn_row, text="Export report", command=lambda: _export(gui), takefocus=False, state=tk.DISABLED
    )
    gui.btn_research_export.pack(side=tk.LEFT, padx=PAD_XS, pady=PAD_XS)
    gui.btn_research_reset = ttk.Button(
        btn_row, text="Reset", command=lambda: reset_research_compare(gui), takefocus=False
    )
    gui.btn_research_reset.pack(side=tk.LEFT, padx=PAD_XS, pady=PAD_XS)
    create_tooltip(gui.btn_research_compare, "Enabled only when Video A and Video B have both finished analysis.")
    create_tooltip(gui.btn_research_export, "Enabled after a successful independent comparison.")
    create_tooltip(gui.btn_research_reset, "Clear both slots, stale results, and charts.")
    gui.lbl_research_status = tk.Label(
        tools,
        text="Import licensed local videos. Pipelines A and B never share pose objects.",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        wraplength=720,
        justify=tk.LEFT,
    )
    gui.lbl_research_status.grid(row=1, column=0, sticky="ew", padx=PAD_SM, pady=(0, PAD_XS))

    def _wrap_research_tools(_event: object | None = None) -> None:
        try:
            width = int(tools.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 80:
            return
        try:
            gui.lbl_research_status.configure(wraplength=max(200, width - 24))
        except tk.TclError:
            pass

    tools.bind("<Configure>", _wrap_research_tools, add="+")

    gui.lbl_research_policy = tk.Label(
        host,
        text=CLINICAL_POLICY + " Remote videos are never downloaded.",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        wraplength=1100,
        justify=tk.LEFT,
        anchor="w",
    )
    gui.lbl_research_policy.grid(row=1, column=0, sticky="ew", padx=PAD_MD)

    def _wrap_research_policy(event: object | None = None) -> None:
        try:
            mapped = int(getattr(event, "width", 0) or gui.lbl_research_policy.winfo_width() or 0)
            gui.lbl_research_policy.configure(wraplength=max(200, mapped - 8))
        except tk.TclError:
            pass

    gui.lbl_research_policy.bind("<Configure>", _wrap_research_policy, add="+")

    from stablewalk.ui.tk.analysis_split import MIN_W_CHART, create_h_split, split_add
    from stablewalk.ui.tk.layout_scale import combobox_char_width

    work = ttk.Frame(host)
    work.grid(row=2, column=0, sticky="nsew")
    work.columnconfigure(0, weight=1)
    work.rowconfigure(0, weight=1)
    gui._research_work = work

    paned = create_h_split(work)
    gui._research_split = paned
    gui._research_stacked = False

    side = ttk.Frame(work)
    side.columnconfigure(0, weight=1)
    side.columnconfigure(1, weight=1)
    side.rowconfigure(4, weight=1)
    gui._research_side = side

    tk.Label(side, text="Catalog", bg=PANEL, fg=TEXT, font=FONT_KPI_TITLE, anchor="w").grid(
        row=0, column=0, columnspan=2, sticky="ew"
    )
    gui.var_research_a = tk.StringVar(value="")
    gui.var_research_b = tk.StringVar(value="")
    ttk.Label(side, text="Video A").grid(row=1, column=0, sticky="w", pady=(PAD_SM, 0), padx=(0, PAD_XS))
    ttk.Label(side, text="Video B").grid(row=1, column=1, sticky="w", pady=(PAD_SM, 0), padx=(PAD_XS, 0))
    gui.cmb_research_a = ttk.Combobox(
        side,
        textvariable=gui.var_research_a,
        state="readonly",
        width=combobox_char_width(("Select a catalog video"), minimum=14, maximum=22),
    )
    gui.cmb_research_a.grid(row=2, column=0, sticky="ew", padx=(0, PAD_XS))
    gui.cmb_research_b = ttk.Combobox(
        side,
        textvariable=gui.var_research_b,
        state="readonly",
        width=combobox_char_width(("Select a catalog video"), minimum=14, maximum=22),
    )
    gui.cmb_research_b.grid(row=2, column=1, sticky="ew", padx=(PAD_XS, 0))
    gui.cmb_research_a.bind(
        "<<ComboboxSelected>>", lambda _e: _on_video_selected(gui, "A"), add="+"
    )
    gui.cmb_research_b.bind(
        "<<ComboboxSelected>>", lambda _e: _on_video_selected(gui, "B"), add="+"
    )
    gui.lbl_research_slot_a = tk.Label(
        side,
        text="A: not loaded",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        wraplength=160,
        justify=tk.LEFT,
    )
    gui.lbl_research_slot_a.grid(row=3, column=0, sticky="ew", padx=(0, PAD_XS), pady=(PAD_XS, 0))
    gui.lbl_research_slot_b = tk.Label(
        side,
        text="B: not loaded",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        wraplength=160,
        justify=tk.LEFT,
    )
    gui.lbl_research_slot_b.grid(row=3, column=1, sticky="ew", padx=(PAD_XS, 0), pady=(PAD_XS, 0))
    cols = ("id", "category", "view")
    tree = ttk.Treeview(side, columns=cols, show="headings", height=8, selectmode="browse")
    tree.heading("id", text="Video ID")
    tree.heading("category", text="Category")
    tree.heading("view", text="View")
    tree.column("id", width=140)
    tree.column("category", width=120)
    tree.column("view", width=80)
    tree.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(PAD_SM, 0))
    gui._research_tree = tree

    body = ttk.Frame(work)
    gui._research_body = body
    paned.grid(row=0, column=0, sticky="nsew")
    split_add(paned, side, minsize=340)
    split_add(paned, body, minsize=MIN_W_CHART)
    body.columnconfigure(0, weight=1)
    body.rowconfigure(0, weight=3, minsize=140)
    body.rowconfigure(1, weight=2, minsize=120)
    body.rowconfigure(2, weight=0)
    body.rowconfigure(3, weight=0, minsize=88)

    graph_host = tk.Frame(body, bg=PANEL, highlightthickness=0)
    graph_host.grid(row=0, column=0, sticky="nsew")
    graph_host.columnconfigure(0, weight=1)
    graph_host.rowconfigure(0, weight=1)
    fig = Figure(figsize=(8.6, 5.6), dpi=100, facecolor=PANEL)
    axes = fig.subplots(2, 2)
    canvas = FigureCanvasTkAgg(fig, master=graph_host)
    widget = canvas.get_tk_widget()
    widget.configure(bg=PANEL, highlightthickness=0)
    widget.grid(row=0, column=0, sticky="nsew")
    gui._research_graph_host = graph_host
    gui.fig_research = fig
    gui.ax_research = axes
    gui.canvas_research = canvas
    try:
        from stablewalk.ui.tk.dashboard_layout import _bind_figure_resize

        _bind_figure_resize(canvas, fig, min_px=80)
    except Exception:
        pass

    from stablewalk.ui.design_system.components import EmptyState

    gui.lbl_research_empty = EmptyState(
        graph_host,
        "Select two different catalog videos, then Analyze A and Analyze B.\n"
        "Pipelines never share pose objects. Charts stay empty until both finish.",
        bg=PANEL,
    )
    gui.lbl_research_empty.grid(row=0, column=0, sticky="nsew")

    skel_host = tk.Frame(body, bg=PANEL, highlightthickness=0)
    skel_host.grid(row=1, column=0, sticky="nsew")
    skel_host.columnconfigure(0, weight=1)
    skel_host.rowconfigure(0, weight=1)
    fig3 = Figure(figsize=(8.6, 3.8), dpi=100, facecolor=PANEL)
    ax_a = fig3.add_subplot(1, 2, 1, projection="3d")
    ax_b = fig3.add_subplot(1, 2, 2, projection="3d")
    canvas3 = FigureCanvasTkAgg(fig3, master=skel_host)
    canvas3.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
    canvas3.get_tk_widget().grid(row=0, column=0, sticky="nsew")
    gui._research_skel_host = skel_host
    gui.fig_research_skel = fig3
    gui.ax_research_skel_a = ax_a
    gui.ax_research_skel_b = ax_b
    gui.canvas_research_skel = canvas3
    try:
        from stablewalk.ui.tk.dashboard_layout import _bind_figure_resize

        _bind_figure_resize(canvas3, fig3, min_px=80)
    except Exception:
        pass

    phase = tk.Frame(body, bg=ELEVATED)
    phase.grid(row=2, column=0, sticky="ew", pady=(PAD_XS, 0))
    tk.Label(phase, text="Gait phase %", bg=ELEVATED, fg=TEXT_SECONDARY, font=FONT_UI_SM).pack(
        side=tk.LEFT, padx=PAD_SM
    )
    gui.var_research_phase = tk.DoubleVar(value=0.0)
    scale = ttk.Scale(
        phase,
        from_=0,
        to=100,
        variable=gui.var_research_phase,
        command=lambda _v: _on_phase(gui),
    )
    scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=PAD_SM)
    create_tooltip(scale, "Synchronized gait-phase playback — each video uses its own cycles.")

    delta_host = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
    delta_host.grid(row=3, column=0, sticky="nsew", pady=(PAD_XS, 0))
    delta_host.columnconfigure(0, weight=1)
    delta_host.rowconfigure(1, weight=1)
    tk.Label(
        delta_host,
        text="Δ = B − A  ·  cadence · symmetry · pose detection · other kinematic/spectral rows",
        bg=ELEVATED,
        fg=TEXT_SECONDARY,
        font=FONT_KPI_TITLE,
        anchor="w",
    ).grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_XS, 0))
    cols = ("metric", "a", "b", "delta", "units")
    tree_delta = ttk.Treeview(
        delta_host, columns=cols, show="headings", height=5, selectmode="browse"
    )
    headings = {
        "metric": "Metric",
        "a": "A",
        "b": "B",
        "delta": "Δ (B−A)",
        "units": "Units",
    }
    widths = {"metric": 180, "a": 90, "b": 90, "delta": 90, "units": 80}
    for col in cols:
        tree_delta.heading(col, text=headings[col])
        tree_delta.column(
            col,
            width=widths[col],
            minwidth=56 if col != "metric" else 80,
            stretch=col in {"metric", "units"},
            anchor="w" if col == "metric" else "e",
        )
    tree_delta.grid(row=1, column=0, sticky="nsew", padx=PAD_SM, pady=(0, PAD_XS))
    gui._research_delta_tree = tree_delta

    def _sync_delta_cols(event: object | None = None) -> None:
        try:
            width = int(getattr(event, "width", 0) or delta_host.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 80:
            return
        usable = max(80, width - 24)
        metric_w = max(96, min(200, int(usable * 0.34)))
        num_w = max(64, min(100, int(usable * 0.16)))
        units_w = max(56, usable - metric_w - 3 * num_w)
        try:
            tree_delta.column("metric", width=metric_w)
            tree_delta.column("a", width=num_w)
            tree_delta.column("b", width=num_w)
            tree_delta.column("delta", width=num_w)
            tree_delta.column("units", width=units_w)
        except tk.TclError:
            pass

    delta_host.bind("<Configure>", _sync_delta_cols, add="+")

    gui._refresh_research_catalog = lambda: _refresh_catalog(gui)
    _refresh_catalog(gui)
    _paint_empty_charts(gui)
    _sync_research_actions(gui)
    return host


def _selected_id(gui: Any, slot: str) -> str:
    var = gui.var_research_a if slot == "A" else gui.var_research_b
    raw = str(var.get() or "")
    return raw.split(" — ", 1)[0].strip()


def _slot_result(gui: Any, slot: str):
    return getattr(gui, f"_research_slot_{slot.lower()}", None)


def _slot_ready(gui: Any, slot: str) -> bool:
    result = _slot_result(gui, slot)
    return result is not None and bool(getattr(result, "ready", False))


def _set_slot_status(gui: Any, slot: str, text: str) -> None:
    lbl = getattr(gui, f"lbl_research_slot_{slot.lower()}", None)
    if lbl is None:
        return
    try:
        lbl.configure(text=f"{slot}: {text}")
    except tk.TclError:
        pass


def _sync_research_actions(gui: Any) -> None:
    """Enable Compare only when both independent analyses are complete."""
    from stablewalk.research.pipeline import slots_are_independent

    a_busy = bool(getattr(gui, "_research_busy_a", False))
    b_busy = bool(getattr(gui, "_research_busy_b", False))
    a_ready = _slot_ready(gui, "A")
    b_ready = _slot_ready(gui, "B")
    independent = False
    if a_ready and b_ready:
        independent = slots_are_independent(
            gui._research_slot_a, gui._research_slot_b
        )
    compare_on = a_ready and b_ready and independent and not a_busy and not b_busy
    export_on = getattr(gui, "_research_comparison", None) is not None
    pairs = (
        ("btn_research_analyze_a", tk.DISABLED if a_busy else tk.NORMAL),
        ("btn_research_analyze_b", tk.DISABLED if b_busy else tk.NORMAL),
        ("btn_research_compare", tk.NORMAL if compare_on else tk.DISABLED),
        ("btn_research_export", tk.NORMAL if export_on else tk.DISABLED),
    )
    for attr, state in pairs:
        btn = getattr(gui, attr, None)
        if btn is None:
            continue
        try:
            btn.configure(state=state)
        except tk.TclError:
            pass
    if a_busy:
        _set_slot_status(gui, "A", "analyzing…")
    elif not a_ready:
        _set_slot_status(gui, "A", "not loaded")
    if b_busy:
        _set_slot_status(gui, "B", "analyzing…")
    elif not b_ready:
        _set_slot_status(gui, "B", "not loaded")


def _clear_slot_result(gui: Any, slot: str) -> None:
    setattr(gui, f"_research_slot_{slot.lower()}", None)
    gui._research_comparison = None
    _fill_delta_table(gui, None)
    empty = getattr(gui, "lbl_research_empty", None)
    if empty is not None:
        try:
            empty.lift()
        except tk.TclError:
            pass
    _set_slot_status(gui, slot, "not loaded")
    _sync_research_actions(gui)


def _on_video_selected(gui: Any, slot: str) -> None:
    """Drop stale analysis when the catalog selection changes."""
    other = "B" if slot == "A" else "A"
    this_id = _selected_id(gui, slot)
    other_id = _selected_id(gui, other)
    prev_attr = f"_research_prev_{slot.lower()}"
    var = gui.var_research_a if slot == "A" else gui.var_research_b
    if this_id and other_id and this_id == other_id:
        allow = False
        try:
            allow = bool(
                messagebox.askyesno(
                    "Same catalog entry",
                    "Video A and Video B currently point at the same catalog clip.\n\n"
                    "Choose Yes only if you want two independent pose runs on that file. "
                    "Otherwise pick a different clip.",
                    parent=getattr(gui, "root", None),
                )
            )
        except Exception:
            allow = False
        gui._research_same_source_ok = allow
        if not allow:
            previous = str(getattr(gui, prev_attr, "") or "")
            try:
                var.set(previous)
            except tk.TclError:
                pass
            _set_status(gui, "Choose a different catalog video for the other slot.")
            return
    setattr(gui, prev_attr, str(var.get() or ""))
    _clear_slot_result(gui, slot)
    _paint_empty_charts(gui)
    _set_status(
        gui,
        f"Video {slot} selection changed — analyze this slot again.",
    )


def reset_research_compare(gui: Any) -> None:
    """Safe reset: drop both pipelines, comparison, and charts."""
    gui._research_token_a = int(getattr(gui, "_research_token_a", 0)) + 1
    gui._research_token_b = int(getattr(gui, "_research_token_b", 0)) + 1
    gui._research_busy_a = False
    gui._research_busy_b = False
    gui._research_slot_a = None
    gui._research_slot_b = None
    gui._research_comparison = None
    gui._research_same_source_ok = False
    try:
        gui.var_research_a.set("")
        gui.var_research_b.set("")
    except tk.TclError:
        pass
    gui._research_prev_a = ""
    gui._research_prev_b = ""
    _fill_delta_table(gui, None)
    _paint_empty_charts(gui)
    _sync_research_actions(gui)
    _set_status(gui, "Comparison reset. Select two different catalog videos.")


def _fill_delta_table(gui: Any, report: Any | None) -> None:
    tree = getattr(gui, "_research_delta_tree", None)
    if tree is None:
        return
    try:
        for item in tree.get_children():
            tree.delete(item)
    except tk.TclError:
        return
    if report is None:
        tree.insert("", "end", values=("No comparison yet", "—", "—", "—", ""))
        return
    preferred = (
        "cadence",
        "knee_asi",
        "pose_quality",
        "knee_rom",
        "stride_time",
        "dominant_frequency",
        "n_frames",
    )
    rows = list(getattr(report, "scalars", None) or [])
    by_name = {str(getattr(row, "name", "")): row for row in rows}
    ordered = [by_name[name] for name in preferred if name in by_name]
    seen = {str(getattr(row, "name", "")) for row in ordered}
    ordered.extend(row for row in rows if str(getattr(row, "name", "")) not in seen)
    if not ordered:
        tree.insert("", "end", values=("No numeric overlap", "—", "—", "—", ""))
        return

    def _fmt(value: float | None) -> str:
        if value is None:
            return "—"
        return f"{value:.3g}"

    for row in ordered:
        tree.insert(
            "",
            "end",
            values=(
                getattr(row, "display", "") or getattr(row, "name", ""),
                _fmt(getattr(row, "value_a", None)),
                _fmt(getattr(row, "value_b", None)),
                _fmt(getattr(row, "delta", None)),
                getattr(row, "units", "") or "",
            ),
        )


def _refresh_catalog(gui: Any) -> None:
    catalog = load_catalog()
    ids = catalog.ids()
    labels = [f"{c.video_id} — {c.category_label}" for c in catalog.clips]
    from stablewalk.ui.tk.layout_scale import combobox_char_width

    combo_w = combobox_char_width(labels or ("Select a catalog video",), minimum=14, maximum=22)
    for box in (getattr(gui, "cmb_research_a", None), getattr(gui, "cmb_research_b", None)):
        if box is not None:
            box.configure(values=labels, width=combo_w)
    tree = getattr(gui, "_research_tree", None)
    if tree is not None:
        for item in tree.get_children():
            tree.delete(item)
        for clip in catalog.clips:
            tree.insert("", "end", values=(clip.video_id, clip.category_label, clip.camera_view))
    gui._research_catalog = catalog
    _ = ids


def _import_clip(gui: Any) -> None:
    path = filedialog.askopenfilename(
        title="Import local walking video",
        filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"), ("All", "*.*")],
    )
    if not path:
        return
    category = simpledialog.askstring(
        "Gait category",
        "Category key (normal_gait, hemiplegic_gait, limping_gait, ataxic_gait, "
        "parkinsonian_gait, shuffling_gait, other_abnormal_gait, unknown_or_unverified, "
        "dementia_research):",
        initialvalue="unknown_or_unverified",
        parent=gui.root,
    )
    if category is None:
        return
    try:
        clip = import_local_video(path, gait_category=category)
    except Exception as exc:
        messagebox.showerror("Import failed", str(exc), parent=gui.root)
        return
    _refresh_catalog(gui)
    label = f"{clip.video_id} — {clip.category_label}"
    if not gui.var_research_a.get():
        gui.var_research_a.set(label)
        gui._research_prev_a = label
    elif not gui.var_research_b.get():
        if _selected_id(gui, "A") == clip.video_id:
            _set_status(
                gui,
                f"Imported {clip.video_id}. Select it as Video B only if you want two pose runs on that file.",
            )
            return
        gui.var_research_b.set(label)
        gui._research_prev_b = label
    _set_status(gui, f"Imported {clip.video_id} (local copy, no download).")


def schedule_slot_analysis(gui: Any, slot: str) -> None:
    if bool(getattr(gui, f"_research_busy_{slot.lower()}", False)):
        _set_status(gui, f"Video {slot} is already analyzing.")
        return
    video_id = _selected_id(gui, slot)
    catalog = getattr(gui, "_research_catalog", None) or load_catalog()
    clip = catalog.get(video_id) if video_id else None
    if clip is None or clip.resolved_path is None or not clip.resolved_path.is_file():
        messagebox.showinfo(
            "Research compare",
            f"Select a catalog video for Video {slot} before analyzing.",
            parent=gui.root,
        )
        return
    other = "B" if slot == "A" else "A"
    other_id = _selected_id(gui, other)
    if other_id and other_id == video_id and not bool(getattr(gui, "_research_same_source_ok", False)):
        try:
            allow = bool(
                messagebox.askyesno(
                    "Same catalog entry",
                    "Video A and Video B are the same catalog clip.\n\n"
                    "Yes = two independent pose pipelines on that file.\n"
                    "No = pick a different clip.",
                    parent=gui.root,
                )
            )
        except Exception:
            allow = False
        gui._research_same_source_ok = allow
        if not allow:
            return
    token_attr = f"_research_token_{slot.lower()}"
    token = int(getattr(gui, token_attr, 0)) + 1
    setattr(gui, token_attr, token)
    setattr(gui, f"_research_busy_{slot.lower()}", True)
    gui._research_comparison = None
    _sync_research_actions(gui)
    _set_status(gui, f"Analyzing Video {slot} independently…")
    max_frames = None
    try:
        from stablewalk import config

        max_frames = config.GUI_MAX_FRAMES_PER_LOAD
    except Exception:
        max_frames = 80
    path = clip.resolved_path

    def worker() -> None:
        try:
            from stablewalk.research.pipeline import analyze_independent_clip

            result = analyze_independent_clip(
                video_path=path,
                metadata=clip,
                video_id=clip.video_id,
                label=clip.category_label,
                run_pose=True,
                max_frames=max_frames,
            )
            err = None
        except Exception as exc:
            result = None
            err = exc
            logger.exception("Research slot %s analysis failed", slot)

        def apply() -> None:
            if int(getattr(gui, token_attr, 0)) != token:
                return
            setattr(gui, f"_research_busy_{slot.lower()}", False)
            if err is not None:
                _set_slot_status(gui, slot, "failed")
                _set_status(gui, f"Video {slot} failed: {err}")
                _sync_research_actions(gui)
                return
            setattr(gui, f"_research_slot_{slot.lower()}", result)
            _set_slot_status(
                gui,
                slot,
                f"ready · {result.n_frames} frames",
            )
            _set_status(
                gui,
                f"Video {slot} ready · {result.n_frames} frames · id(seq)={result.sequence_id}",
            )
            _sync_research_actions(gui)
            if _slot_ready(gui, "A") and _slot_ready(gui, "B"):
                run_comparison(gui)

        root = getattr(gui, "root", None)
        if root is None:
            return
        try:
            root.after(0, apply)
        except (tk.TclError, RuntimeError):
            pass

    threading.Thread(target=worker, name=f"stablewalk-research-{slot}", daemon=True).start()


def run_comparison(gui: Any) -> None:
    from stablewalk.research.compare import compare_independent_clips
    from stablewalk.research.pipeline import slots_are_independent, slots_share_source_file

    a = getattr(gui, "_research_slot_a", None)
    b = getattr(gui, "_research_slot_b", None)
    if a is None or not getattr(a, "ready", False):
        _set_status(gui, "Video A is not analyzed yet.")
        _sync_research_actions(gui)
        return
    if b is None or not getattr(b, "ready", False):
        _set_status(gui, "Video B is not analyzed yet.")
        _sync_research_actions(gui)
        return
    if not slots_are_independent(a, b):
        gui._research_comparison = None
        _fill_delta_table(gui, None)
        _sync_research_actions(gui)
        _set_status(gui, "Blocked: A and B share pose or analysis objects.")
        return
    report = compare_independent_clips(a, b)
    gui._research_comparison = report
    empty = getattr(gui, "lbl_research_empty", None)
    if empty is not None:
        try:
            empty.lower()
        except tk.TclError:
            pass
    _draw_charts(gui, report)
    _fill_delta_table(gui, report)
    _on_phase(gui)
    n = sum(1 for s in report.scalars if s.delta is not None)
    note = ""
    if slots_share_source_file(a, b):
        note = " · same source file (two independent pose runs)"
    _set_status(gui, f"Independent comparison · {n} numeric Δ rows{note} · not a diagnosis")
    _sync_research_actions(gui)


def _set_status(gui: Any, text: str) -> None:
    lbl = getattr(gui, "lbl_research_status", None)
    if lbl is None:
        return
    try:
        lbl.configure(text=text)
    except tk.TclError:
        pass


def _paint_empty_charts(gui: Any) -> None:
    axes = getattr(gui, "ax_research", None)
    if axes is None:
        return
    empty = getattr(gui, "lbl_research_empty", None)
    if empty is not None:
        try:
            empty.lift()
        except tk.TclError:
            pass
    for ax in axes.ravel():
        ax.clear()
        apply_chart_panel_style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            0.5,
            "Waiting for independent Video A and Video B",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
            fontsize=10,
        )
    for ax in (
        getattr(gui, "ax_research_skel_a", None),
        getattr(gui, "ax_research_skel_b", None),
    ):
        if ax is None:
            continue
        ax.clear()
        apply_chart_panel_style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            ax.set_zticks([])
        except Exception:
            pass
        ax.text2D(0.5, 0.5, "No clip", ha="center", va="center", color=MUTED, transform=ax.transAxes)
    canvas = getattr(gui, "canvas_research", None)
    if canvas is not None:
        try:
            canvas.draw_idle()
        except Exception:
            pass
    c3 = getattr(gui, "canvas_research_skel", None)
    if c3 is not None:
        try:
            c3.draw_idle()
        except Exception:
            pass


def _draw_charts(gui: Any, report) -> None:
    axes = getattr(gui, "ax_research", None)
    canvas = getattr(gui, "canvas_research", None)
    fig = getattr(gui, "fig_research", None)
    if axes is None or canvas is None:
        return
    ax_ang, ax_psd = axes[0]
    ax_acf, ax_bar = axes[1]
    for ax in (ax_ang, ax_psd, ax_acf, ax_bar):
        ax.clear()
        apply_chart_panel_style(ax)

    wave = report.waveform_by_name("knee_flexion")
    if wave and wave.aligned:
        ax_ang.plot(wave.phase_pct, wave.series_a, color=_SLOT_COLORS["A"], lw=SERIES_LINE_WIDTH, label="A")
        ax_ang.plot(wave.phase_pct, wave.series_b, color=_SLOT_COLORS["B"], lw=SERIES_LINE_WIDTH, label="B")
        ax_ang.set_xlabel("Gait cycle (%)")
        ax_ang.set_ylabel("deg")
        ax_ang.set_title("Knee flexion (phase-aligned)", color=TEXT, fontsize=11)
        ax_ang.legend(frameon=False, fontsize=9)
        gui._research_phase_line = ax_ang.axvline(0.0, color=TEXT, lw=1.0, alpha=0.8)
    else:
        ax_ang.text(0.5, 0.5, "Need gait cycles on both videos", ha="center", va="center", color=MUTED, transform=ax_ang.transAxes)
        gui._research_phase_line = None

    _plot_psd(ax_psd, report.slot_a, report.slot_b)
    _plot_acf(ax_acf, report.slot_a, report.slot_b)

    names = ["knee_asi", "dominant_frequency", "acf_stride_lag", "cadence"]
    labels = []
    da = []
    db = []
    for name in names:
        row = report.scalar_by_name(name)
        if row is None:
            continue
        if row.value_a is None and row.value_b is None:
            continue
        labels.append(row.display.replace(" ", "\n"))
        da.append(float("nan") if row.value_a is None else row.value_a)
        db.append(float("nan") if row.value_b is None else row.value_b)
    if labels:
        import numpy as np

        x = np.arange(len(labels), dtype=float)
        w = 0.36
        ax_bar.bar(x - w / 2, da, w, color=_SLOT_COLORS["A"], label="A")
        ax_bar.bar(x + w / 2, db, w, color=_SLOT_COLORS["B"], label="B")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, fontsize=8)
        ax_bar.set_title("Selected scalars", color=TEXT, fontsize=11)
        ax_bar.legend(frameon=False, fontsize=9)
    try:
        if fig is not None:
            fig.set_constrained_layout(False)
            fig.subplots_adjust(
                left=0.08, right=0.98, top=0.90, bottom=0.12, hspace=0.48, wspace=0.30
            )
        canvas.draw_idle()
    except Exception:
        pass


def _plot_psd(ax, a, b) -> None:
    drawn = False
    for slot, color in ((a, _SLOT_COLORS["A"]), (b, _SLOT_COLORS["B"])):
        spec = getattr(slot.spectral, "primary_spectrum", lambda: None)()
        if spec is None:
            continue
        freq = getattr(spec, "freq_hz", None)
        pxx = getattr(spec, "psd", None)
        if freq is None or pxx is None:
            continue
        ax.plot(freq, pxx, color=color, lw=SERIES_LINE_WIDTH, label=slot.video_id)
        drawn = True
    ax.set_title("PSD", color=TEXT, fontsize=11)
    ax.set_xlabel("Hz")
    if drawn:
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "PSD unavailable", ha="center", va="center", color=MUTED, transform=ax.transAxes)


def _plot_acf(ax, a, b) -> None:
    drawn = False
    for slot, color in ((a, _SLOT_COLORS["A"]), (b, _SLOT_COLORS["B"])):
        ac = getattr(slot.spectral, "primary_autocorr", lambda: None)()
        if ac is None:
            continue
        ax.plot(ac.lags_s, ac.rho, color=color, lw=SERIES_LINE_WIDTH, label=slot.video_id)
        drawn = True
    ax.set_title("Autocorrelation", color=TEXT, fontsize=11)
    ax.set_xlabel("Lag (s)")
    if drawn:
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "ACF unavailable", ha="center", va="center", color=MUTED, transform=ax.transAxes)


def _on_phase(gui: Any) -> None:
    from stablewalk.research.gait_phase import map_gait_phase

    pct = float(gui.var_research_phase.get())
    line = getattr(gui, "_research_phase_line", None)
    if line is not None:
        line.set_xdata([pct, pct])
        canvas = getattr(gui, "canvas_research", None)
        if canvas is not None:
            try:
                canvas.draw_idle()
            except Exception:
                pass
    a = getattr(gui, "_research_slot_a", None)
    b = getattr(gui, "_research_slot_b", None)
    _draw_skeleton(gui.ax_research_skel_a, a, pct, "A")
    _draw_skeleton(gui.ax_research_skel_b, b, pct, "B")
    c3 = getattr(gui, "canvas_research_skel", None)
    if c3 is not None:
        try:
            c3.draw_idle()
        except Exception:
            pass


def _draw_skeleton(ax, slot, pct: float, tag: str) -> None:
    ax.clear()
    apply_chart_panel_style(ax)
    ax.set_title(f"Skeleton {tag}", color=TEXT, fontsize=10)
    if slot is None or slot.gait_motion is None:
        ax.text2D(0.5, 0.5, f"Load Video {tag}", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    from stablewalk.models.joint_registry import JOINT_PARENTS
    from stablewalk.research.gait_phase import map_gait_phase

    mapping = map_gait_phase(slot.gait_motion, slot.cycles, pct)
    snap = slot.gait_motion.snapshot_at(mapping.frame_index)
    if snap is None:
        return
    joints = snap.joints
    for child, parent in JOINT_PARENTS.items():
        if parent is None:
            continue
        a = joints.get(parent)
        b = joints.get(child)
        if a is None or b is None:
            continue
        ax.plot(
            [a.position.x, b.position.x],
            [a.position.z, b.position.z],
            [a.position.y, b.position.y],
            color=_SLOT_COLORS.get(tag, TEXT),
            lw=1.6,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def _export(gui: Any) -> None:
    report = getattr(gui, "_research_comparison", None)
    if report is None:
        messagebox.showinfo(
            "Research compare",
            "Analyze Video A and Video B, then click Compare before exporting.",
            parent=gui.root,
        )
        return
    from stablewalk.config import ANALYSIS_EXPORT_DIR
    from stablewalk.research.report import (
        export_comparison_csv,
        export_comparison_json,
        export_comparison_markdown,
    )

    ANALYSIS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = filedialog.asksaveasfilename(
        title="Export comparison report",
        defaultextension=".md",
        initialdir=str(ANALYSIS_EXPORT_DIR),
        filetypes=[("Markdown", "*.md"), ("JSON", "*.json"), ("CSV", "*.csv")],
    )
    if not path:
        return
    dest = Path(path)
    if dest.suffix.lower() == ".json":
        export_comparison_json(report, dest)
    elif dest.suffix.lower() == ".csv":
        export_comparison_csv(report, dest)
    else:
        export_comparison_markdown(report, dest)
    _set_status(gui, f"Exported {dest.name}")


def reflow_research_workspace(gui: Any, *, width: int) -> None:
    """Resize Compare research charts and wrap copy without rerunning analysis."""
    policy = getattr(gui, "lbl_research_policy", None)
    if policy is not None:
        try:
            mapped = int(policy.winfo_width() or 0)
        except tk.TclError:
            mapped = 0
        wrap = mapped - 16 if mapped > 80 else max(280, int(width) - 48)
        try:
            policy.configure(wraplength=wrap)
        except tk.TclError:
            pass
    work = getattr(gui, "_research_work", None)
    paned = getattr(gui, "_research_split", None)
    side = getattr(gui, "_research_side", None)
    body = getattr(gui, "_research_body", None)
    if work is not None and paned is not None and side is not None and body is not None:
        from stablewalk.ui.tk.analysis_split import (
            MIN_W_CHART,
            split_add,
            split_clear,
        )
        from stablewalk.ui.tk.layout_scale import layout_scale_from_gui, scaled_breakpoints

        scale = layout_scale_from_gui(gui)
        floor = min(int(scaled_breakpoints(scale)["compare_stack"]), 1200)
        stack = int(width) < floor
        if getattr(gui, "_research_stacked", None) is not stack:
            gui._research_stacked = stack
            gui._research_sash_for_width = None
            try:
                if stack:
                    split_clear(paned)
                    paned.grid_remove()
                    work.rowconfigure(0, weight=0, minsize=160)
                    work.rowconfigure(1, weight=1)
                    side.grid(
                        row=0,
                        column=0,
                        sticky="ew",
                        padx=(PAD_SM, PAD_SM),
                        pady=(0, PAD_XS),
                    )
                    body.grid(
                        row=1,
                        column=0,
                        sticky="nsew",
                        padx=(PAD_SM, PAD_SM),
                        pady=0,
                    )
                else:
                    try:
                        side.grid_forget()
                        body.grid_forget()
                    except tk.TclError:
                        pass
                    work.rowconfigure(0, weight=1)
                    work.rowconfigure(1, weight=0, minsize=0)
                    paned.grid(row=0, column=0, sticky="nsew")
                    split_add(paned, side, minsize=340, stretch="always")
                    split_add(paned, body, minsize=MIN_W_CHART, stretch="always")
            except tk.TclError:
                pass
        if not stack:
            bucket = int(width) // 80
            if getattr(gui, "_research_sash_for_width", None) != bucket:
                try:
                    paned.update_idletasks()
                    pw = int(paned.winfo_width() or width or 0)
                    if pw > 400:
                        paned.sash_place(0, max(340, int(pw * 0.36)), 0)
                        gui._research_sash_for_width = bucket
                except tk.TclError:
                    pass
    for attr in ("lbl_research_slot_a", "lbl_research_slot_b"):
        lbl = getattr(gui, attr, None)
        if lbl is None:
            continue
        try:
            col_w = max(80, int(width) // 2 - 40)
            lbl.configure(wraplength=col_w)
        except tk.TclError:
            pass
    pairs = (
        ("canvas_research", "fig_research"),
        ("canvas_research_skel", "fig_research_skel"),
    )
    try:
        from stablewalk.ui.tk.dashboard_layout import _fit_figure_to_host
    except Exception:
        return
    for canvas_attr, fig_attr in pairs:
        canvas = getattr(gui, canvas_attr, None)
        fig = getattr(gui, fig_attr, None)
        if canvas is None or fig is None:
            continue
        try:
            widget = canvas.get_tk_widget()
            if not widget.winfo_ismapped():
                continue
            _fit_figure_to_host(canvas, fig, host=widget.master, min_px=80)
        except Exception:
            continue
