"""
Biomechanics Laboratory — dual-session gait Comparison Mode.

Layout
------
Chrome:    presets (Normal vs Abnormal / Performance / User vs Reference)
Top:       Session A metrics | Difference Panel | Session B metrics
Middle:    Session A video | Session B video  (+ skeletons)
Bottom:    COM paths | Joint angle overlay | Difference heatmap
Scroll:    vertical scroll when content exceeds the window; transport stays fixed
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from stablewalk.analysis.session_compare import (
    CompareMetrics,
    SessionComparisonResult,
    compare_session_metrics,
)
from stablewalk.ui.colors import ACCENT_ALT, COM, CRITICAL, MUTED, PANEL, TEXT
from stablewalk.ui.theme import (
    BORDER,
    DANGER,
    ELEVATED,
    FONT_HEADING,
    FONT_METRIC,
    FONT_SECTION,
    FONT_SUMMARY_INTERPRETATION,
    FONT_SUMMARY_METRIC_TITLE,
    FONT_SUMMARY_METRIC_VALUE,
    FONT_UI_SM,
    PAD_SM,
    PAD_XS,
    SUCCESS,
    SURFACE,
    WARNING,
    create_tooltip,
)
from stablewalk.ui.tk.session_cache import (
    AnalyzedSessionSnapshot,
    ensure_compare_cache,
    sessions_are_identical,
)
from stablewalk.pose.skeleton_3d import SKELETON_3D_CONNECTIONS
from stablewalk.ui.viewers.difference_heatmap import draw_difference_heatmap
from stablewalk.ui.viewers.plot_3d import (
    apply_display_limits,
    draw_skeleton_3d,
    remember_skeleton_camera,
    setup_3d_axes,
    smooth_reset_skeleton_camera,
)

logger = logging.getLogger(__name__)

# Session A / B — research colors (not body L/R green/red).
COLOR_SESSION_A = COM
COLOR_SESSION_B = ACCENT_ALT
COLOR_LEFT = COLOR_SESSION_A  # Session A (legacy alias)
COLOR_RIGHT = COLOR_SESSION_B  # Session B (legacy alias)

# Layout minsizes — keep panels readable; scroll when the window is short.
COMPARE_METRICS_MIN_H = 280
COMPARE_VIDEO_MIN_H = 360
COMPARE_SKELETON_MIN_H = 240
COMPARE_GRAPH_MIN_H = 320
COMPARE_REPORT_MIN_H = 160

_METRIC_FIELDS = (
    "Gait Quality",
    "Cadence",
    "Speed",
    "Step Length",
    "Knee ROM",
    "Symmetry",
    "COM Excursion",
    "Stability",
)

_DIFF_PRIORITY = (
    "Cadence difference",
    "Step length difference",
    "ROM difference",
    "Hip ROM difference",
    "Ankle ROM difference",
    "Joint angle difference",
    "COM excursion",
    "Stability difference",
    "Stability",
    "Walking speed",
    "Symmetry",
    "Gait quality",
    "Timeline difference",
)

def build_comparison_tab(gui: Any, parent: tk.Misc) -> ttk.Frame:
    """Install a lightweight host; build heavy comparison figures on first use."""
    host = ttk.Frame(parent)
    host.pack(fill=tk.BOTH, expand=True)
    placeholder = ttk.Label(
        host,
        text="Comparison workspace loads when this tab is opened.",
        anchor="center",
    )
    placeholder.pack(fill=tk.BOTH, expand=True, padx=PAD_SM, pady=PAD_SM)
    gui._comparison_mode = None
    gui._comparison_refresh_pending = False

    def _is_compare_active() -> bool:
        notebook = getattr(gui, "_dashboard_notebook", None)
        if notebook is None:
            return False
        try:
            return notebook.nametowidget(notebook.select()) is getattr(
                gui, "_tab_compare", None
            )
        except (tk.TclError, KeyError):
            return False

    def _ensure() -> ComparisonModeController:
        controller = getattr(gui, "_comparison_mode", None)
        if controller is not None:
            return controller
        placeholder.destroy()
        controller = ComparisonModeController(gui, host)
        gui._comparison_mode = controller
        gui._comparison_refresh_pending = False
        return controller

    def _request_refresh() -> None:
        controller = getattr(gui, "_comparison_mode", None)
        if controller is not None:
            controller.refresh()
        elif _is_compare_active():
            _ensure()
        else:
            gui._comparison_refresh_pending = True

    gui._ensure_comparison_mode_loaded = _ensure
    gui._refresh_comparison_mode = _request_refresh
    return host


class ComparisonModeController:
    """Owns Comparison Mode widgets and dual-session playback sync."""

    def __init__(self, gui: Any, parent: tk.Misc) -> None:
        self.gui = gui
        self.root = ttk.Frame(parent)
        self.root.pack(fill=tk.BOTH, expand=True)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)

        self._progress = 0.0
        self._playing = False
        self._play_job: str | None = None
        self._speed = 1.0
        self._sample_stride = 1
        self._skeleton_mode = tk.StringVar(value="side")
        self._show_path_left = tk.BooleanVar(value=True)
        self._show_path_right = tk.BooleanVar(value=True)
        self._photo_a: ImageTk.PhotoImage | None = None
        self._photo_b: ImageTk.PhotoImage | None = None
        self._video_resize_job: str | None = None
        self._last_paint: tuple[int, int] | None = None
        self._skel_paint_tick = 0
        self._skel_paint_stride = 2
        self._loading_keys: set[str] = set()
        self._failed_snapshot_keys: set[str] = set()
        self._snapshot_results: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._snapshot_poll_job: str | None = None
        self._snapshot_build_lock = threading.Lock()
        # Persistent playhead / marker artists so the synchronized cursor MOVES
        # each frame instead of forcing a full graph rebuild.
        self._path_markers: dict[str, tuple[Any, np.ndarray]] = {}
        self._heat_playhead: Any = None
        self._heat_bins = 40

        self._build_chrome()
        self._build_body()
        self.refresh()
        self._snapshot_poll_job = self.gui.root.after(
            100, self._drain_snapshot_results
        )

    # ── chrome ──────────────────────────────────────────────────────────

    def _build_chrome(self) -> None:
        bar = ttk.Frame(self.root)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
        bar.columnconfigure(4, weight=1)

        tk.Label(
            bar,
            text="Biomechanics Lab — Comparison",
            bg=PANEL,
            fg=TEXT,
            font=FONT_SECTION,
        ).grid(row=0, column=0, sticky="w", padx=(0, PAD_SM))

        self.left_key_var = tk.StringVar(value="normal")
        self.right_key_var = tk.StringVar(value="abnormal")

        ttk.Label(bar, text="Session A", style="Card.TLabel").grid(
            row=0, column=1, padx=(PAD_SM, 2)
        )
        self.cmb_left = ttk.Combobox(
            bar, textvariable=self.left_key_var, width=16, state="readonly"
        )
        self.cmb_left.grid(row=0, column=2, padx=(0, PAD_SM))
        self.cmb_left.bind("<<ComboboxSelected>>", lambda _e: self._on_slot_change())

        ttk.Label(bar, text="Session B", style="Card.TLabel").grid(
            row=0, column=3, padx=(0, 2)
        )
        self.cmb_right = ttk.Combobox(
            bar, textvariable=self.right_key_var, width=16, state="readonly"
        )
        self.cmb_right.grid(row=0, column=4, sticky="w")
        self.cmb_right.bind("<<ComboboxSelected>>", lambda _e: self._on_slot_change())

        presets = ttk.Frame(bar)
        presets.grid(row=0, column=5, padx=(PAD_SM, PAD_SM))
        btn_normal_abn = ttk.Button(
            presets,
            text="Normal vs Abnormal",
            style="Compact.TButton",
            command=lambda: self.apply_preset("normal", "abnormal"),
        )
        btn_normal_abn.pack(side=tk.LEFT, padx=(0, 4))
        btn_normal_perf = ttk.Button(
            presets,
            text="Normal vs Performance",
            style="Compact.TButton",
            command=lambda: self.apply_preset("normal", "athletic"),
        )
        btn_normal_perf.pack(side=tk.LEFT, padx=(0, 4))
        btn_abn_perf = ttk.Button(
            presets,
            text="Abnormal vs Performance",
            style="Compact.TButton",
            command=lambda: self.apply_preset("abnormal", "athletic"),
        )
        btn_abn_perf.pack(side=tk.LEFT, padx=(0, 4))
        btn_user_ref = ttk.Button(
            presets,
            text="User vs Reference",
            style="Compact.TButton",
            command=self.apply_user_vs_reference,
        )
        btn_user_ref.pack(side=tk.LEFT, padx=(0, 4))
        btn_pin_a = ttk.Button(
            presets,
            text="Pin → A",
            style="Compact.TButton",
            command=lambda: self.pin_current(slot="left"),
        )
        btn_pin_a.pack(side=tk.LEFT, padx=(0, 4))
        btn_pin_b = ttk.Button(
            presets,
            text="Pin → B",
            style="Compact.TButton",
            command=lambda: self.pin_current(slot="right"),
        )
        btn_pin_b.pack(side=tk.LEFT)

        create_tooltip(btn_normal_abn, "Load Normal into Session A and Abnormal into Session B")
        create_tooltip(btn_normal_perf, "Load Normal into Session A and Performance into Session B")
        create_tooltip(btn_abn_perf, "Load Abnormal into Session A and Performance into Session B")
        create_tooltip(btn_user_ref, "Compare the active recording against a reference session")
        create_tooltip(btn_pin_a, "Pin the current session into Session A")
        create_tooltip(btn_pin_b, "Pin the current session into Session B")
        create_tooltip(self.cmb_left, "Choose Session A source recording")
        create_tooltip(self.cmb_right, "Choose Session B source recording")

        rb_side = ttk.Radiobutton(
            bar,
            text="Two Skeletons",
            variable=self._skeleton_mode,
            value="side",
            command=self._on_skeleton_mode,
        )
        rb_side.grid(row=1, column=0, columnspan=2, sticky="w", pady=(PAD_XS, 0))
        rb_overlay = ttk.Radiobutton(
            bar,
            text="Overlay Skeletons",
            variable=self._skeleton_mode,
            value="overlay",
            command=self._on_skeleton_mode,
        )
        rb_overlay.grid(row=1, column=2, columnspan=2, sticky="w", pady=(PAD_XS, 0))
        create_tooltip(rb_side, "Show Session A and Session B skeletons side by side")
        create_tooltip(rb_overlay, "Overlay both skeletons in a shared coordinate frame")
        tk.Label(
            bar,
            text="Timeline: synchronized normalised progress (0–100%)",
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_SM,
        ).grid(row=1, column=4, columnspan=3, sticky="w", pady=(PAD_XS, 0))

        self.btn_export_report = ttk.Button(
            bar,
            text="Export Lab Report",
            style="Accent.TButton",
            width=16,
            command=self.export_report,
        )
        self.btn_export_report.grid(row=0, column=7, padx=(PAD_SM, PAD_SM))
        create_tooltip(self.btn_export_report, "Export a printable lab comparison report")

        self.lbl_status = tk.Label(
            bar, text="", bg=PANEL, fg=MUTED, font=FONT_UI_SM, anchor="e"
        )
        self.lbl_status.grid(row=0, column=8, sticky="e")

    def _build_body(self) -> None:
        """Scrollable Top / Middle / Bottom layout with readable min heights."""
        scroll_host = ttk.Frame(self.root)
        scroll_host.grid(row=1, column=0, sticky="nsew", pady=(0, PAD_XS))
        scroll_host.columnconfigure(0, weight=1)
        scroll_host.rowconfigure(0, weight=1)

        transport_host = ttk.Frame(self.root)
        transport_host.grid(row=2, column=0, sticky="ew")
        transport_host.columnconfigure(0, weight=1)

        sections = self._install_compare_scroll(scroll_host)
        sections.columnconfigure(0, weight=1)

        # Top — Session A | Difference | Session B
        metrics = ttk.Frame(sections)
        metrics.grid(row=0, column=0, sticky="ew", pady=(0, PAD_SM))
        metrics.configure(height=COMPARE_METRICS_MIN_H)
        self._build_metrics_row(metrics)

        # Middle — Video A | Video B
        videos = ttk.Frame(sections)
        videos.grid(row=1, column=0, sticky="ew", pady=(0, PAD_SM))
        self._build_videos_row(videos)

        # Skeletons (side-by-side or overlay) — keep visible without crushing videos
        skeletons = ttk.Frame(sections)
        skeletons.grid(row=2, column=0, sticky="ew", pady=(0, PAD_SM))
        self._build_skeletons_row(skeletons)

        # Bottom — COM paths | Joint overlay | Difference heatmap
        graphs = ttk.Frame(sections)
        graphs.grid(row=3, column=0, sticky="ew", pady=(0, PAD_SM))
        self._build_graphs_row(graphs)

        report = ttk.Frame(sections)
        report.grid(row=4, column=0, sticky="ew", pady=(0, PAD_SM))
        self._build_report_section(report)

        self._build_transport(transport_host)
        # Bind wheel on nested panels so scroll works over videos/graphs.
        bind_wheel = getattr(self, "_compare_bind_wheel", None)
        if callable(bind_wheel):
            bind_wheel(sections)
        self._sync_compare_scroll()

    def _install_compare_scroll(self, parent: ttk.Frame) -> ttk.Frame:
        """Local vertical scroll shell (does not touch the Advanced-tab scroll)."""
        from stablewalk.ui.theme import BG

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        outer = ttk.Frame(parent)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=vsb.set)

        scroll_content = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        scroll_content.columnconfigure(0, weight=1)

        sections = ttk.Frame(scroll_content)
        sections.grid(row=0, column=0, sticky="ew")
        sections.columnconfigure(0, weight=1)

        spacer = ttk.Frame(scroll_content, height=PAD_SM)
        spacer.grid(row=1, column=0, sticky="ew")
        spacer.grid_propagate(False)

        self._compare_scroll_canvas = canvas
        self._compare_scroll_inner = scroll_content
        self._compare_scroll_window_id = inner_id
        self._compare_scroll_sections = sections

        def _sync(_event: object | None = None) -> None:
            try:
                canvas.update_idletasks()
                bbox = canvas.bbox("all")
                if bbox is not None:
                    canvas.configure(scrollregion=bbox)
                cw = max(int(canvas.winfo_width()), 1)
                canvas.itemconfigure(inner_id, width=cw)
            except tk.TclError:
                pass

        self._sync_compare_scroll = _sync
        scroll_content.bind("<Configure>", _sync, add="+")
        canvas.bind("<Configure>", _sync, add="+")

        def _wheel(event: tk.Event) -> str | None:
            if getattr(event, "delta", None):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                return "break"
            return None

        def _bind_wheel(widget: tk.Misc) -> None:
            widget.bind("<MouseWheel>", _wheel, add="+")
            try:
                for child in widget.winfo_children():
                    _bind_wheel(child)
            except tk.TclError:
                pass

        self._compare_bind_wheel = _bind_wheel
        for widget in (canvas, scroll_content, sections):
            widget.bind("<MouseWheel>", _wheel, add="+")
        canvas.bind("<Enter>", lambda _e: canvas.focus_set(), add="+")
        return sections

    def _section_min_host(self, parent: tk.Misc, *, height: int) -> tk.Frame:
        """Fixed-min-height host so scroll content does not crush panels."""
        host = tk.Frame(parent, bg=PANEL, height=height, highlightthickness=0)
        host.pack(fill=tk.BOTH, expand=True)
        host.pack_propagate(False)
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)
        return host

    def _build_metrics_row(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)
        host = self._section_min_host(parent, height=COMPARE_METRICS_MIN_H)
        host.columnconfigure(0, weight=1)
        host.columnconfigure(1, weight=1)
        host.columnconfigure(2, weight=1)
        host.rowconfigure(0, weight=1)

        self.card_left = self._metric_card(host, "Session A")
        self.card_left["frame"].grid(row=0, column=0, sticky="nsew", padx=(0, PAD_XS))
        self.card_summary = self._summary_card(host)
        self.card_summary.grid(row=0, column=1, sticky="nsew", padx=PAD_XS)
        self.card_right = self._metric_card(host, "Session B")
        self.card_right["frame"].grid(row=0, column=2, sticky="nsew", padx=(PAD_XS, 0))

    def _metric_card(self, parent: tk.Misc, title: str) -> dict[str, Any]:
        from stablewalk.ui.tk.kpi_cards import create_kpi_card

        frame = tk.Frame(parent, bg=ELEVATED, highlightbackground=BORDER, highlightthickness=1)
        hdr = tk.Label(
            frame, text=title, bg=ELEVATED, fg=TEXT, font=FONT_HEADING, anchor="w"
        )
        hdr.pack(fill=tk.X, padx=PAD_SM, pady=(PAD_SM, PAD_XS))
        fields: dict[str, tk.Label] = {}
        kpis: dict[str, Any] = {}
        body = tk.Frame(frame, bg=ELEVATED, highlightthickness=0)
        body.pack(fill=tk.BOTH, expand=True, padx=PAD_SM, pady=(0, PAD_SM))
        for name in _METRIC_FIELDS:
            kpi = create_kpi_card(
                body,
                key=name,
                title=name,
                tooltip=f"{name} — laboratory comparison metric",
                compact=True,
                show_bar=name in ("Gait Quality", "Cadence", "Symmetry", "Speed"),
                fill=False,
            )
            fields[name] = kpi.value_lbl
            kpis[name] = kpi
        return {"frame": frame, "header": hdr, "fields": fields, "kpis": kpis}

    def _summary_card(self, parent: tk.Misc) -> tk.Frame:
        frame = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(
            frame,
            text="Difference Panel",
            bg=SURFACE,
            fg=TEXT,
            font=FONT_HEADING,
            anchor="w",
        ).pack(fill=tk.X, padx=PAD_SM, pady=(PAD_SM, PAD_XS))
        tk.Label(
            frame,
            text="ROM · Cadence · Step length · Joint · COM · Stability",
            bg=SURFACE,
            fg=MUTED,
            font=FONT_UI_SM,
            anchor="w",
        ).pack(fill=tk.X, padx=PAD_SM, pady=(0, PAD_XS))
        self.diff_host = tk.Frame(frame, bg=SURFACE)
        self.diff_host.pack(fill=tk.BOTH, expand=True, padx=PAD_SM, pady=(0, PAD_SM))
        self._diff_labels: list[tk.Label] = []
        return frame

    def _build_videos_row(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        host = self._section_min_host(parent, height=COMPARE_VIDEO_MIN_H)
        host.columnconfigure(0, weight=1)
        host.columnconfigure(1, weight=1)
        host.rowconfigure(0, weight=1)

        self.video_a = self._video_panel(host, "Session A — Video")
        self.video_a["frame"].grid(
            row=0, column=0, sticky="nsew", padx=(0, PAD_XS), pady=0
        )
        self.video_b = self._video_panel(host, "Session B — Video")
        self.video_b["frame"].grid(
            row=0, column=1, sticky="nsew", padx=(PAD_XS, 0), pady=0
        )

    def _build_skeletons_row(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        host = self._section_min_host(parent, height=COMPARE_SKELETON_MIN_H)
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)

        self.skel_side = ttk.Frame(host)
        self.skel_side.grid(row=0, column=0, sticky="nsew")
        self.skel_side.columnconfigure(0, weight=1)
        self.skel_side.columnconfigure(1, weight=1)
        self.skel_side.rowconfigure(0, weight=1)

        self.fig_skel_a = Figure(figsize=(4.2, 3.0), dpi=100, facecolor=PANEL)
        self.ax_skel_a = self.fig_skel_a.add_subplot(111, projection="3d")
        setup_3d_axes(self.ax_skel_a)
        self.canvas_skel_a = FigureCanvasTkAgg(self.fig_skel_a, master=self.skel_side)
        wa = self.canvas_skel_a.get_tk_widget()
        wa.grid(row=0, column=0, sticky="nsew", padx=(0, PAD_XS))
        self._bind_compare_canvas_resize(wa, self.fig_skel_a, self.canvas_skel_a)
        self._bind_skeleton_camera_controls(self.canvas_skel_a, self.ax_skel_a)

        self.fig_skel_b = Figure(figsize=(4.2, 3.0), dpi=100, facecolor=PANEL)
        self.ax_skel_b = self.fig_skel_b.add_subplot(111, projection="3d")
        setup_3d_axes(self.ax_skel_b)
        self.canvas_skel_b = FigureCanvasTkAgg(self.fig_skel_b, master=self.skel_side)
        wb = self.canvas_skel_b.get_tk_widget()
        wb.grid(row=0, column=1, sticky="nsew", padx=(PAD_XS, 0))
        self._bind_compare_canvas_resize(wb, self.fig_skel_b, self.canvas_skel_b)
        self._bind_skeleton_camera_controls(self.canvas_skel_b, self.ax_skel_b)

        self.skel_overlay = ttk.Frame(host)
        self.fig_skel_ov = Figure(figsize=(8.0, 3.0), dpi=100, facecolor=PANEL)
        self.ax_skel_ov = self.fig_skel_ov.add_subplot(111, projection="3d")
        setup_3d_axes(self.ax_skel_ov)
        self.canvas_skel_ov = FigureCanvasTkAgg(self.fig_skel_ov, master=self.skel_overlay)
        wo = self.canvas_skel_ov.get_tk_widget()
        wo.pack(fill=tk.BOTH, expand=True)
        self._bind_compare_canvas_resize(wo, self.fig_skel_ov, self.canvas_skel_ov)
        self._bind_skeleton_camera_controls(self.canvas_skel_ov, self.ax_skel_ov)

    def _build_graphs_row(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        host = self._section_min_host(parent, height=COMPARE_GRAPH_MIN_H)
        host.columnconfigure(0, weight=1)
        host.columnconfigure(1, weight=1)
        host.columnconfigure(2, weight=1)
        host.rowconfigure(0, weight=1)

        path_fr = ttk.LabelFrame(
            host, text="COM Paths (synced)", style="Card.TLabelframe"
        )
        path_fr.grid(row=0, column=0, sticky="nsew", padx=(0, PAD_XS))
        path_fr.columnconfigure(0, weight=1)
        path_fr.rowconfigure(1, weight=1)
        toggles = ttk.Frame(path_fr)
        toggles.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(
            toggles,
            text="Show A",
            variable=self._show_path_left,
            command=self._redraw_paths,
        ).pack(side=tk.LEFT, padx=(PAD_XS, PAD_SM))
        ttk.Checkbutton(
            toggles,
            text="Show B",
            variable=self._show_path_right,
            command=self._redraw_paths,
        ).pack(side=tk.LEFT)
        self.fig_path = Figure(figsize=(4.0, 3.2), dpi=100, facecolor=PANEL)
        self.ax_path = self.fig_path.add_subplot(111, projection="3d")
        setup_3d_axes(self.ax_path)
        self.canvas_path = FigureCanvasTkAgg(self.fig_path, master=path_fr)
        wp = self.canvas_path.get_tk_widget()
        wp.grid(row=1, column=0, sticky="nsew")
        self._bind_compare_canvas_resize(wp, self.fig_path, self.canvas_path)

        chart_fr = ttk.LabelFrame(
            host, text="Joint Angle Overlay", style="Card.TLabelframe"
        )
        chart_fr.grid(row=0, column=1, sticky="nsew", padx=(0, PAD_XS))
        chart_fr.columnconfigure(0, weight=1)
        chart_fr.rowconfigure(0, weight=0)
        chart_fr.rowconfigure(1, weight=1)
        from stablewalk.ui.viewers.chart_interactions import (
            attach_chart_interactions,
            build_chart_tools_bar,
        )

        self.fig_chart = Figure(figsize=(4.0, 3.2), dpi=100, facecolor=PANEL)
        self.ax_chart = self.fig_chart.add_subplot(111)
        self._style_chart_ax()
        tools_host = SimpleNamespace(
            fig_chart=None,
            canvas_chart=None,
            root=getattr(self, "root", None) or parent.winfo_toplevel(),
        )
        self.canvas_chart = FigureCanvasTkAgg(self.fig_chart, master=chart_fr)
        tools_host.fig_chart = self.fig_chart
        tools_host.canvas_chart = self.canvas_chart
        self._compare_chart_tools = build_chart_tools_bar(
            tools_host,
            chart_fr,
            fig_attr="fig_chart",
            canvas_attr="canvas_chart",
            export_name="compare_knee",
        )
        self._compare_chart_tools.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        wc = self.canvas_chart.get_tk_widget()
        wc.grid(row=1, column=0, sticky="nsew")
        self._bind_compare_canvas_resize(wc, self.fig_chart, self.canvas_chart)
        attach_chart_interactions(self.fig_chart, self.canvas_chart)
        self._chart_playhead = None
        self.canvas_chart.mpl_connect("motion_notify_event", self._on_chart_hover)
        self._hover_annot = self.ax_chart.annotate(
            "",
            xy=(0, 0),
            xytext=(8, 8),
            textcoords="offset points",
            color=TEXT,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc=ELEVATED, ec=BORDER, alpha=0.95),
        )
        self._hover_annot.set_visible(False)

        heat_fr = ttk.LabelFrame(
            host, text="Difference Heatmap", style="Card.TLabelframe"
        )
        heat_fr.grid(row=0, column=2, sticky="nsew")
        heat_fr.columnconfigure(0, weight=1)
        heat_fr.rowconfigure(0, weight=1)
        self.fig_heat = Figure(figsize=(4.4, 3.2), dpi=100, facecolor=PANEL)
        self.ax_heat = self.fig_heat.add_subplot(111)
        self.canvas_heat = FigureCanvasTkAgg(self.fig_heat, master=heat_fr)
        wh = self.canvas_heat.get_tk_widget()
        wh.grid(row=0, column=0, sticky="nsew")
        self._bind_compare_canvas_resize(wh, self.fig_heat, self.canvas_heat)
        # Leave room for y-tick labels / colorbar — never crush labels.
        self.fig_heat.subplots_adjust(left=0.24, right=0.86, top=0.88, bottom=0.18)

    def _bind_compare_canvas_resize(
        self,
        widget: tk.Misc,
        fig: Figure,
        canvas: FigureCanvasTkAgg,
    ) -> None:
        """Keep matplotlib figures filling their panel as the window resizes."""
        state: dict[str, object] = {"job": None, "last": (0, 0)}

        def _apply() -> None:
            state["job"] = None
            if getattr(canvas, "_stablewalk_fitting", False):
                return
            setattr(canvas, "_stablewalk_fitting", True)
            try:
                host = widget.master
                if host is not None:
                    try:
                        host.update_idletasks()
                        w = max(int(host.winfo_width()), 80)
                        h = max(int(host.winfo_height()), 80)
                    except tk.TclError:
                        return
                else:
                    w = max(int(widget.winfo_width()), 80)
                    h = max(int(widget.winfo_height()), 80)
                last = state.get("last")
                if last == (w, h):
                    return
                state["last"] = (w, h)
                dpi = float(fig.get_dpi() or 100.0)
                fig.set_size_inches(w / dpi, h / dpi, forward=False)
                try:
                    canvas.resize(type("E", (), {"width": int(w), "height": int(h)})())
                except (TypeError, AttributeError, tk.TclError):
                    fig.set_size_inches(w / dpi, h / dpi, forward=True)
                canvas.draw_idle()
            except Exception:
                pass
            finally:
                setattr(canvas, "_stablewalk_fitting", False)

        def _on_configure(_event: object | None = None) -> None:
            if getattr(canvas, "_stablewalk_fitting", False):
                return
            job = state.get("job")
            if job is not None:
                try:
                    widget.after_cancel(job)  # type: ignore[arg-type]
                except Exception:
                    pass
            try:
                state["job"] = widget.after(16, _apply)
            except Exception:
                _apply()

        # Bind host only — configuring the canvas widget would re-enter here.
        host = widget.master
        if host is not None:
            host.bind("<Configure>", _on_configure, add="+")
        else:
            widget.bind("<Configure>", _on_configure, add="+")

    def _bind_skeleton_camera_controls(self, canvas, ax) -> None:
        """Remember orbit gestures; double-click eases back to the default view."""

        def _remember(_event: object | None = None) -> None:
            remember_skeleton_camera(ax)

        def _reset(event) -> None:
            if not bool(getattr(event, "dblclick", False)):
                return
            smooth_reset_skeleton_camera(
                ax,
                canvas=canvas,
                scheduler=self.gui.root,
            )

        # Persist the orbit while dragging and after release so playback never
        # snaps back to the default Blender/OpenSim framing mid-session.
        canvas.mpl_connect("motion_notify_event", _remember)
        canvas.mpl_connect("button_release_event", _remember)
        canvas.mpl_connect("button_press_event", _reset)

    def _video_panel(self, parent: tk.Misc, title: str) -> dict[str, Any]:
        frame = ttk.LabelFrame(parent, text=title, style="Card.TLabelframe")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        label = tk.Label(frame, text="No session", bg=PANEL, fg=MUTED, font=FONT_UI_SM)
        label.grid(row=0, column=0, sticky="nsew")
        panel: dict[str, Any] = {"frame": frame, "label": label, "_last_size": None}

        def _on_resize(event: tk.Event) -> None:
            size = (int(event.width), int(event.height))
            if size == panel["_last_size"] or min(size) < 20:
                return
            panel["_last_size"] = size
            if self._video_resize_job is not None:
                try:
                    self.gui.root.after_cancel(self._video_resize_job)
                except (tk.TclError, ValueError):
                    pass

            def _refit() -> None:
                self._video_resize_job = None
                self._last_paint = None
                self._paint_frame(force=True)

            self._video_resize_job = self.gui.root.after(90, _refit)

        label.bind("<Configure>", _on_resize, add="+")
        return panel

    def _build_report_section(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        host = self._section_min_host(parent, height=COMPARE_REPORT_MIN_H)
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)
        interp = ttk.LabelFrame(
            host, text="Laboratory Comparison Report", style="Card.TLabelframe"
        )
        interp.grid(row=0, column=0, sticky="nsew")
        self.txt_interp = tk.Text(
            interp,
            height=8,
            wrap=tk.WORD,
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_SUMMARY_INTERPRETATION,
            relief=tk.FLAT,
            padx=PAD_SM,
            pady=PAD_SM,
            highlightthickness=0,
        )
        self.txt_interp.pack(fill=tk.BOTH, expand=True)
        self.txt_interp.configure(state=tk.DISABLED)

    def _build_transport(self, parent: ttk.Frame) -> None:
        """Fixed playback strip — always visible under the scroll area."""
        parent.columnconfigure(0, weight=1)
        transport = ttk.Frame(parent)
        transport.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))

        self.btn_play = ttk.Button(transport, text="▶ Play", width=10, command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=(0, PAD_XS))
        ttk.Button(transport, text="⏮", width=3, command=lambda: self.seek(0.0)).pack(
            side=tk.LEFT, padx=(0, PAD_XS)
        )
        ttk.Button(
            transport, text="◀", width=3, command=lambda: self.step_frames(-1)
        ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(
            transport, text="▶", width=3, command=lambda: self.step_frames(1)
        ).pack(side=tk.LEFT, padx=(0, PAD_XS))

        self.timeline = ttk.Scale(
            transport,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            command=self._on_timeline,
        )
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=PAD_XS)

        self.lbl_progress = tk.Label(
            transport,
            text="Gait cycle 0%",
            bg=PANEL,
            fg=TEXT,
            font=FONT_HEADING,
            width=14,
            anchor="e",
        )
        self.lbl_progress.pack(side=tk.LEFT, padx=(PAD_XS, PAD_SM))

        self.lbl_time = tk.Label(
            transport, text="0.00 s", bg=PANEL, fg=MUTED, font=FONT_METRIC, width=18, anchor="e"
        )
        self.lbl_time.pack(side=tk.LEFT, padx=(PAD_XS, PAD_SM))

        ttk.Label(transport, text="Speed").pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="1.0×")
        sp = ttk.Combobox(
            transport,
            textvariable=self.speed_var,
            values=("0.25×", "0.5×", "1.0×", "1.5×", "2.0×"),
            width=6,
            state="readonly",
        )
        sp.pack(side=tk.LEFT, padx=(2, PAD_SM))
        sp.bind("<<ComboboxSelected>>", self._on_speed)

        ttk.Label(transport, text="Sample").pack(side=tk.LEFT)
        self.sample_var = tk.StringVar(value="1")
        sm = ttk.Combobox(
            transport,
            textvariable=self.sample_var,
            values=("1", "2", "3", "5"),
            width=4,
            state="readonly",
        )
        sm.pack(side=tk.LEFT, padx=(2, 0))
        sm.bind("<<ComboboxSelected>>", self._on_sample)

    # ── public API ──────────────────────────────────────────────────────

    def enter(self) -> None:
        """Switch into Comparison Mode without restarting analysis."""
        # Auto-pin the active analysis so at least one slot is populated.
        try:
            from stablewalk.ui.tk.session_cache import pin_current_session

            pin_current_session(self.gui)
        except Exception:
            pass
        cache = ensure_compare_cache(self.gui)
        # Defaults: Session A = Normal, Session B = Abnormal (independent keys).
        cache.left_key = "normal"
        cache.right_key = "abnormal"
        self.left_key_var.set("normal")
        self.right_key_var.set("abnormal")
        self.refresh()
        from stablewalk.ui.tk.dashboard_notebook import TAB_COMPARE, select_dashboard_tab

        select_dashboard_tab(self.gui, TAB_COMPARE)
        self.gui._comparison_mode_active = True

    def apply_preset(self, left_key: str, right_key: str) -> None:
        """Select a named pair (e.g. Normal vs Abnormal / Performance)."""
        cache = ensure_compare_cache(self.gui)
        # Always set the requested keys — never coerce B onto A.
        cache.set_slot("left", left_key)
        cache.set_slot("right", right_key)
        self.left_key_var.set(left_key)
        self.right_key_var.set(right_key)
        self.refresh()
        missing = [k for k in (left_key, right_key) if cache.get(k) is None]
        if missing:
            self.lbl_status.configure(
                text=f"Load & analyze first: {', '.join(missing)}",
                fg=DANGER,
            )

    def apply_user_vs_reference(self) -> None:
        """Pin the current user session as Video B against a Normal/reference A."""
        from stablewalk.ui.tk.session_cache import pin_current_session

        snap = pin_current_session(self.gui)
        cache = ensure_compare_cache(self.gui)
        keys = cache.keys()
        ref_key = "normal" if "normal" in keys else None
        if ref_key is None:
            for candidate in keys:
                if candidate not in ("abnormal", "athletic") and (
                    snap is None or candidate != snap.key
                ):
                    ref_key = candidate
                    break
        if ref_key is None and keys:
            ref_key = keys[0]
        if ref_key is None:
            self.lbl_status.configure(
                text="Analyze a Reference (Normal) session first",
                fg=DANGER,
            )
            return
        cache.left_key = ref_key
        self.left_key_var.set(ref_key)
        if snap is not None:
            cache.right_key = snap.key
            self.right_key_var.set(snap.key)
            self.refresh()
            self.lbl_status.configure(
                text=f"User vs Reference: {snap.label} vs {ref_key}",
                fg=SUCCESS,
            )
            return
        self.refresh()
        self.lbl_status.configure(
            text="Analyze your video, then click User vs Reference",
            fg=DANGER,
        )

    def pin_current(self, *, slot: str = "left") -> None:
        """Pin the currently analyzed session into Session A or Session B."""
        from stablewalk.ui.tk.session_cache import pin_current_session

        snap = pin_current_session(self.gui)
        if snap is None:
            from tkinter import messagebox

            messagebox.showinfo(
                "Comparison Mode",
                "Analyze a video first, then pin it as Session A or Session B.",
            )
            return
        cache = ensure_compare_cache(self.gui)
        if slot == "right":
            cache.set_slot("right", snap.key)
            self.right_key_var.set(snap.key)
        else:
            cache.set_slot("left", snap.key)
            self.left_key_var.set(snap.key)
        self.refresh()
        side = "B" if slot == "right" else "A"
        self.lbl_status.configure(
            text=f"Pinned {snap.label} → Session {side}",
            fg=SUCCESS,
        )

    def step_frames(self, delta: int) -> None:
        """Step both sessions using shared normalized progress (not equal frame counts)."""
        cache = ensure_compare_cache(self.gui)
        left, right = cache.left(), cache.right()
        n = max(
            (left.n_frames if left else 1),
            (right.n_frames if right else 1),
            1,
        )
        step = 1.0 / max(n - 1, 1)
        self.seek(self._progress + delta * step)

    def export_report(self) -> None:
        """Delegate to the application PDF export handler."""
        export_fn = getattr(self.gui, "_export_comparison_report", None)
        if callable(export_fn):
            export_fn()
        else:
            from tkinter import messagebox

            messagebox.showinfo(
                "Export Comparison Report",
                "Comparison report export is not available in this build.",
            )

    def _ensure_slot_snapshot(self, key: str) -> None:
        """Load a demo snapshot in the background if it is not already pinned."""
        cache = ensure_compare_cache(self.gui)
        if (
            cache.get(key) is not None
            or key in self._loading_keys
            or key in self._failed_snapshot_keys
        ):
            return
        self._loading_keys.add(key)

        def _worker() -> None:
            snap = None
            try:
                from stablewalk.ui.tk.session_cache import try_build_demo_snapshot

                # Serialize CPU-heavy demo analysis so background loading does
                # not starve video playback or Tk's main thread.
                with self._snapshot_build_lock:
                    snap = try_build_demo_snapshot(key)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Failed to prepare comparison snapshot %s", key
                )

            self._snapshot_results.put((key, snap))

        threading.Thread(
            target=_worker,
            name=f"stablewalk-compare-{key}",
            daemon=True,
        ).start()

    def _drain_snapshot_results(self) -> None:
        """Transfer worker results into Tk-owned state on the main thread."""
        changed = False
        try:
            while True:
                key, snap = self._snapshot_results.get_nowait()
                self._loading_keys.discard(key)
                if snap is not None:
                    ensure_compare_cache(self.gui).pin(snap)
                else:
                    self._failed_snapshot_keys.add(key)
                changed = True
        except queue.Empty:
            pass
        if changed:
            self.refresh()
        try:
            self._snapshot_poll_job = self.gui.root.after(
                100, self._drain_snapshot_results
            )
        except tk.TclError:
            self._snapshot_poll_job = None

    def refresh(self) -> None:
        cache = ensure_compare_cache(self.gui)
        # Keep A/B keys independent: try loading missing demos from disk first.
        for key in (cache.left_key, cache.right_key):
            self._ensure_slot_snapshot(key)

        choices = cache.choice_keys()
        self._key_choices = choices
        self.cmb_left.configure(values=choices)
        self.cmb_right.configure(values=choices)

        # Preserve requested keys even when a session is not yet loaded.
        # Never rewrite Session B to Session A's key.
        if cache.left_key not in choices and choices:
            # Only fall back for an empty/invalid left key — not onto right.
            if not cache.left_key:
                cache.left_key = "normal" if "normal" in choices else choices[0]
        if cache.right_key not in choices and choices:
            if not cache.right_key:
                cache.right_key = (
                    "abnormal"
                    if "abnormal" in choices
                    else next((k for k in choices if k != cache.left_key), choices[0])
                )
        self.left_key_var.set(cache.left_key)
        self.right_key_var.set(cache.right_key)

        left = cache.left()
        right = cache.right()
        identical = sessions_are_identical(
            left,
            right,
            left_key=cache.left_key,
            right_key=cache.right_key,
        )

        def _metrics_ready(snap) -> bool:
            if snap is None:
                return False
            m = snap.metrics
            return any(
                v is not None
                for v in (
                    m.gait_quality,
                    m.cadence_spm,
                    m.walking_speed_m_s,
                    m.step_length_m,
                    m.knee_rom_deg,
                    m.symmetry_pct,
                )
            )

        missing = []
        if left is None:
            missing.append(cache.left_key)
        if right is None:
            missing.append(cache.right_key)
        right_incomplete = right is not None and not _metrics_ready(right)
        left_incomplete = left is not None and not _metrics_ready(left)
        if identical:
            self.lbl_status.configure(
                text="Both comparison sessions are identical.",
                fg=WARNING,
            )
        elif missing:
            loading = [key for key in missing if key in self._loading_keys]
            if loading:
                self.lbl_status.configure(
                    text=f"Preparing cached sessions: {', '.join(loading)}…",
                    fg=MUTED,
                )
            else:
                self.lbl_status.configure(
                    text=f"Load sessions to compare: {', '.join(missing)}",
                    fg=DANGER,
                )
        elif right_incomplete or left_incomplete:
            which = []
            if left_incomplete:
                which.append("Session A")
            if right_incomplete:
                which.append("Session B")
            self.lbl_status.configure(
                text=(
                    f"{' / '.join(which)} metrics incomplete — "
                    "wait for demo load or pin a finished analysis."
                ),
                fg=WARNING,
            )
        else:
            self.lbl_status.configure(
                text="Timeline synced by progress % · dual independent sessions",
                fg=SUCCESS,
            )

        self._set_video_titles(left, right)
        self._fill_metric_card(
            self.card_left,
            left.metrics if left else CompareMetrics(label="Session A"),
            header_prefix="Session A",
        )
        self._fill_metric_card(
            self.card_right,
            right.metrics if right else CompareMetrics(label="Session B"),
            header_prefix="Session B",
        )
        if identical:
            self._fill_diffs(None, identical=True)
            self._set_interp(
                "BIOMECHANICS LABORATORY — GAIT COMPARISON\n\n"
                "Both comparison sessions are identical.\n"
                "Select two different sessions (e.g. Normal vs Abnormal) "
                "to compute differences."
            )
        elif left and right and _metrics_ready(left) and _metrics_ready(right):
            # Diffs always come from two independent result objects.
            result = compare_session_metrics(left.metrics, right.metrics)
            self._fill_diffs(result)
            self._set_interp(result.lab_report_summary or result.interpretation)
        else:
            self._fill_diffs(None)
            if missing and any(k in self._loading_keys for k in missing):
                self._set_interp(
                    "BIOMECHANICS LABORATORY — GAIT COMPARISON\n\n"
                    "Loading the second session from demo cache…\n"
                    "Difference metrics appear when Session A and Session B "
                    "both have finished analysis."
                )
            else:
                self._set_interp(
                    "BIOMECHANICS LABORATORY — GAIT COMPARISON\n\n"
                    "Analyze two independent sessions (demos or your videos), pin them "
                    "as Session A / Session B, or use presets:\n"
                    "  • Normal vs Abnormal\n"
                    "  • Normal vs Performance\n"
                    "  • Abnormal vs Performance\n"
                    "  • User vs Reference\n\n"
                    "Already-analyzed sessions are reused — no pipeline restart.\n"
                    "Session B stays empty until its snapshot finishes loading."
                )

        self._redraw_paths()
        self._redraw_chart_series()
        self._redraw_heatmap()
        self._paint_frame(force=True)
        sync = getattr(self, "_sync_compare_scroll", None)
        if callable(sync):
            sync()

    def _set_video_titles(
        self,
        left: AnalyzedSessionSnapshot | None,
        right: AnalyzedSessionSnapshot | None,
    ) -> None:
        a_name = (left.label if left else None) or "Session A"
        b_name = (right.label if right else None) or "Session B"
        try:
            self.video_a["frame"].configure(text=f"Session A — {a_name}")
            self.video_b["frame"].configure(text=f"Session B — {b_name}")
        except Exception:
            pass

    # ── slot / transport ────────────────────────────────────────────────

    def _on_slot_change(self) -> None:
        cache = ensure_compare_cache(self.gui)
        left_key = self.left_key_var.get()
        right_key = self.right_key_var.get()
        cache.set_slot("left", left_key)
        cache.set_slot("right", right_key)
        self._ensure_slot_snapshot(left_key)
        self._ensure_slot_snapshot(right_key)
        self.refresh()

    def _on_skeleton_mode(self) -> None:
        mode = self._skeleton_mode.get()
        if mode == "overlay":
            self.skel_side.grid_remove()
            self.skel_overlay.grid(row=0, column=0, sticky="nsew")
        else:
            self.skel_overlay.grid_remove()
            self.skel_side.grid(row=0, column=0, sticky="nsew")
        self._paint_frame(force=True)
        sync = getattr(self, "_sync_compare_scroll", None)
        if callable(sync):
            sync()

    def _on_timeline(self, value: str) -> None:
        try:
            self._progress = float(value)
        except ValueError:
            return
        self._paint_frame()

    def _on_speed(self, _e: object | None = None) -> None:
        raw = self.speed_var.get().replace("×", "").strip()
        try:
            self._speed = max(0.1, float(raw))
        except ValueError:
            self._speed = 1.0

    def _on_sample(self, _e: object | None = None) -> None:
        try:
            self._sample_stride = max(1, int(self.sample_var.get()))
        except ValueError:
            self._sample_stride = 1

    def seek(self, progress: float) -> None:
        self._progress = float(np.clip(progress, 0.0, 1.0))
        self.timeline.set(self._progress)
        self._paint_frame(force=True)

    def toggle_play(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        self._playing = True
        self._skel_paint_tick = 0
        self.btn_play.configure(text="⏸ Pause")
        self._tick()

    def pause(self) -> None:
        self._playing = False
        self.btn_play.configure(text="▶ Play")
        if self._play_job is not None:
            try:
                self.gui.root.after_cancel(self._play_job)
            except Exception:
                pass
            self._play_job = None
        # Crisp final skeleton paint at the paused frame.
        self._paint_frame(force=True)

    def _tick(self) -> None:
        if not self._playing:
            return
        cache = ensure_compare_cache(self.gui)
        left, right = cache.left(), cache.right()
        # Normalized progress sync: advance a shared 0–1 timeline so unequal
        # frame counts stay locked by percentage, not by absolute frame index.
        n = max(
            (left.n_frames if left else 1),
            (right.n_frames if right else 1),
            1,
        )
        fps = 30.0
        if left is not None and left.fps:
            fps = float(left.fps)
        elif right is not None and right.fps:
            fps = float(right.fps)
        step = (self._sample_stride / max(n - 1, 1))
        self._progress = min(1.0, self._progress + step)
        self.timeline.set(self._progress)
        self._paint_frame()
        if self._progress >= 1.0:
            self.pause()
            return
        delay_ms = max(16, int(1000 / (max(fps, 1.0) * self._speed)))
        self._play_job = self.gui.root.after(delay_ms, self._tick)

    # ── paint ───────────────────────────────────────────────────────────

    def _index_for(self, snap: AnalyzedSessionSnapshot | None) -> int:
        if snap is None or snap.n_frames <= 0:
            return 0
        return int(round(self._progress * (snap.n_frames - 1)))

    def _pose_index(self, snap: AnalyzedSessionSnapshot | None) -> int:
        idx = self._index_for(snap)
        if snap is None:
            return 0
        if snap.pose_indices:
            return snap.pose_indices[min(idx, len(snap.pose_indices) - 1)]
        return idx

    def _paint_frame(self, *, force: bool = False) -> None:
        cache = ensure_compare_cache(self.gui)
        left, right = cache.left(), cache.right()
        ia, ib = self._index_for(left), self._index_for(right)
        if not force and self._last_paint == (ia, ib):
            # Frame indices unchanged but progress moved sub-frame — keep the
            # continuous cursors gliding across the graphs.
            self._update_chart_playhead()
            self._update_heat_playhead()
            self._update_progress_readout()
            return
        self._last_paint = (ia, ib)

        # Everything below is locked to the same normalized gait progress and is
        # refreshed together on every playhead move.
        self._paint_video(self.video_a, left, ia)
        self._paint_video(self.video_b, right, ib)
        # Skeleton cla()+redraw is expensive; stride while playing keeps video
        # lockstep while cutting CPU. Always paint when scrubbing/pausing.
        paint_skel = True
        if self._playing and not force:
            paint_skel = (self._skel_paint_tick % max(self._skel_paint_stride, 1)) == 0
            self._skel_paint_tick += 1
        if paint_skel:
            self._paint_skeletons(left, right)
        self._update_path_markers()
        self._update_chart_playhead()
        self._update_heat_playhead()
        self._update_time_label(left, right, ia, ib)
        self._update_progress_readout()

    def _paint_video(
        self, panel: dict[str, Any], snap: AnalyzedSessionSnapshot | None, list_idx: int
    ) -> None:
        lbl: tk.Label = panel["label"]
        if snap is None:
            lbl.configure(image="", text="No session loaded", fg=MUTED)
            return
        img = self._load_frame_image(snap, list_idx)
        if img is None:
            lbl.configure(
                image="",
                text=f"{snap.label}\nframe {list_idx + 1}/{snap.n_frames}",
                fg=MUTED,
            )
            return
        # Fit the available panel while preserving aspect ratio.
        try:
            max_w = int(lbl.winfo_width())
            max_h = int(lbl.winfo_height())
            if max_w <= 2 or max_h <= 2:
                host: tk.Misc = panel["frame"]
                host.update_idletasks()
                max_w = int(host.winfo_width())
                max_h = int(host.winfo_height())
            if max_w <= 2 or max_h <= 2:
                max_w, max_h = 480, 300
        except tk.TclError:
            max_w, max_h = 480, 300
        fitted = img.copy()
        try:
            fast = Image.Resampling.BILINEAR
            hq = Image.Resampling.LANCZOS
        except AttributeError:  # Pillow < 9.1
            fast = Image.BILINEAR  # type: ignore[attr-defined]
            hq = Image.LANCZOS  # type: ignore[attr-defined]
        # Fast filter while playing keeps both videos smooth; crisp when paused.
        resample = fast if self._playing else hq
        fitted.thumbnail((max_w - 8, max_h - 8), resample)
        photo = ImageTk.PhotoImage(fitted)
        if panel is self.video_a:
            self._photo_a = photo
        else:
            self._photo_b = photo
        lbl.configure(image=photo, text="")

    def _load_frame_image(
        self, snap: AnalyzedSessionSnapshot, list_idx: int
    ) -> Image.Image | None:
        if snap.frame_paths:
            path = snap.frame_paths[min(list_idx, len(snap.frame_paths) - 1)]
            try:
                return Image.open(path).convert("RGB")
            except OSError:
                pass
        if snap.video_path and Path(str(snap.video_path)).is_file():
            try:
                import cv2

                cap = cv2.VideoCapture(str(snap.video_path))
                if not cap.isOpened():
                    return None
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                frame_i = int(round(self._progress * max(total - 1, 0)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                ok, bgr = cap.read()
                cap.release()
                if not ok or bgr is None:
                    return None
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
            except Exception:
                return None
        return None

    def _paint_skeletons(
        self,
        left: AnalyzedSessionSnapshot | None,
        right: AnalyzedSessionSnapshot | None,
    ) -> None:
        mode = self._skeleton_mode.get()
        if mode == "overlay":
            self.ax_skel_ov.cla()
            setup_3d_axes(self.ax_skel_ov)
            drawn = False
            for snap, color, alpha in (
                (left, COLOR_LEFT, 0.55),
                (right, COLOR_RIGHT, 0.75),
            ):
                sk = self._skeleton_at(snap)
                if sk is None or not sk.joints:
                    continue
                self._draw_skel_colored(self.ax_skel_ov, sk, color=color, alpha=alpha, clear=False)
                drawn = True
            if not drawn:
                self.ax_skel_ov.text2D(
                    0.5, 0.5, "No skeletons", transform=self.ax_skel_ov.transAxes, ha="center", color=MUTED
                )
            self.ax_skel_ov.set_title("Overlay Skeletons", color=TEXT, fontsize=9)
            self.canvas_skel_ov.draw_idle()
            return

        for ax, canvas, snap, title, color in (
            (self.ax_skel_a, self.canvas_skel_a, left, "Skeleton A", COLOR_LEFT),
            (self.ax_skel_b, self.canvas_skel_b, right, "Skeleton B", COLOR_RIGHT),
        ):
            ax.cla()
            setup_3d_axes(ax)
            sk = self._skeleton_at(snap)
            if sk is None or not sk.joints:
                ax.text2D(0.5, 0.5, "No skeleton", transform=ax.transAxes, ha="center", color=MUTED)
            else:
                draw_skeleton_3d(ax, sk, clear=False, frame_label=title)
            ax.set_title(title, color=color, fontsize=9)
            canvas.draw_idle()

    def _skeleton_at(self, snap: AnalyzedSessionSnapshot | None) -> Any:
        if snap is None:
            return None
        pose_i = self._pose_index(snap)
        sk = snap.skeletons.get(pose_i)
        if sk is not None:
            return sk
        # Fallback: first available
        if snap.skeletons:
            keys = sorted(snap.skeletons.keys())
            idx = keys[min(self._index_for(snap), len(keys) - 1)]
            return snap.skeletons.get(idx)
        return None

    def _draw_skel_colored(self, ax, skeleton, *, color: str, alpha: float, clear: bool) -> None:
        # Lightweight bone draw for overlay mode (reuse joints).
        if clear:
            ax.cla()
            setup_3d_axes(ax)
        joints = skeleton.joints
        for a, b in SKELETON_3D_CONNECTIONS:
            ja, jb = joints.get(a), joints.get(b)
            if ja is None or jb is None:
                continue
            ax.plot(
                [ja.x, jb.x],
                [ja.y, jb.y],
                [ja.z, jb.z],
                color=color,
                alpha=alpha,
                linewidth=1.45,
            )
        xs = [j.x for j in joints.values()]
        ys = [j.y for j in joints.values()]
        zs = [j.z for j in joints.values()]
        if xs:
            ax.scatter(xs, ys, zs, c=color, s=12, alpha=alpha, depthshade=False)
            apply_display_limits(ax, skeleton, padding=1.18, flat_frontal=False)
            try:
                ax.set_box_aspect((1, 1, 1), zoom=0.90)
            except TypeError:
                ax.set_box_aspect((1, 1, 1))
            except (AttributeError, ValueError):
                pass

    def _redraw_paths(self) -> None:
        cache = ensure_compare_cache(self.gui)
        left, right = cache.left(), cache.right()

        self.ax_path.cla()
        setup_3d_axes(self.ax_path)
        # Rebuild the static path lines once; keep a movable position marker per
        # session so playback only nudges the marker (no full 3D redraw/frame).
        self._path_markers = {}
        drawn_arrays: list[np.ndarray] = []
        for slot, snap, show, color, name in (
            ("A", left, self._show_path_left.get(), COLOR_LEFT, "Session A"),
            ("B", right, self._show_path_right.get(), COLOR_RIGHT, "Session B"),
        ):
            if not show or snap is None or snap.path_xyz is None:
                continue
            p = np.asarray(snap.path_xyz, dtype=float)
            if p.ndim != 2 or p.shape[0] == 0:
                continue
            drawn_arrays.append(p)
            label = snap.label or name
            self.ax_path.plot(p[:, 0], p[:, 1], p[:, 2], color=color, linewidth=2.0, label=label)
            i = min(self._index_for(snap), len(p) - 1)
            marker, = self.ax_path.plot(
                [p[i, 0]],
                [p[i, 1]],
                [p[i, 2]],
                marker="o",
                color=color,
                markersize=9,
                markeredgecolor="white",
                markeredgewidth=1.2,
                linestyle="None",
                zorder=30,
            )
            self._path_markers[slot] = (marker, p)
        self._fit_compare_path_axes(drawn_arrays)
        self.ax_path.set_title("3D Path A | B", color=TEXT, fontsize=9)
        if self.ax_path.get_legend_handles_labels()[0]:
            try:
                self.ax_path.legend(
                    loc="upper right", fontsize=7, facecolor=ELEVATED, edgecolor=BORDER
                )
            except Exception:
                pass
        self.canvas_path.draw_idle()

    def _fit_compare_path_axes(self, arrays: list[np.ndarray]) -> None:
        """Center, equal-scale, and fit the 3D path cube to the drawn paths.

        Equal cubic limits around the data centroid (1:1:1 box aspect) keep the
        two paths comparable without distortion, and the tight span-based extent
        fills the panel instead of floating in an oversized grid.
        """
        pts = [p for p in arrays if p is not None and getattr(p, "size", 0)]
        if not pts:
            return
        allp = np.concatenate(pts, axis=0)
        mins = allp.min(axis=0)
        maxs = allp.max(axis=0)
        center = (mins + maxs) * 0.5
        span = float(np.max(maxs - mins))
        if not np.isfinite(span) or span <= 1e-6:
            span = 0.2
        half = span * 0.58  # equal cube half-extent + ~15% breathing room
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        self.ax_path.set_xlim(cx - half, cx + half)
        self.ax_path.set_ylim(cy - half, cy + half)
        self.ax_path.set_zlim(cz - half, cz + half)
        try:
            self.ax_path.set_box_aspect((1, 1, 1))
        except (AttributeError, ValueError, TypeError):
            pass

    def _update_path_markers(self) -> None:
        """Move the current-position marker on each 3D path (no rebuild)."""
        markers = getattr(self, "_path_markers", None)
        if not markers:
            return
        cache = ensure_compare_cache(self.gui)
        snaps = {"A": cache.left(), "B": cache.right()}
        moved = False
        for slot, (marker, p) in markers.items():
            snap = snaps.get(slot)
            if snap is None or p.shape[0] == 0:
                continue
            i = min(self._index_for(snap), len(p) - 1)
            try:
                marker.set_data_3d([p[i, 0]], [p[i, 1]], [p[i, 2]])
                moved = True
            except Exception:
                pass
        if moved:
            self.canvas_path.draw_idle()

    def _redraw_chart_series(self) -> None:
        cache = ensure_compare_cache(self.gui)
        left, right = cache.left(), cache.right()

        self.ax_chart.cla()
        self._style_chart_ax()
        for snap, color in (
            (left, COLOR_LEFT),
            (right, COLOR_RIGHT),
        ):
            if snap is None or not snap.knee_t:
                continue
            t = np.asarray(snap.knee_t, dtype=float)
            y = np.asarray(
                [np.nan if v is None else float(v) for v in snap.knee_y], dtype=float
            )
            if len(t) < 2:
                continue
            tn = (t - t[0]) / max(t[-1] - t[0], 1e-6)
            self.ax_chart.plot(tn, y, color=color, linewidth=1.6, label=snap.label or "Session")
        if self.ax_chart.get_legend_handles_labels()[0]:
            self.ax_chart.legend(
                loc="upper right", fontsize=8, facecolor=ELEVATED, edgecolor=BORDER
            )
        self.ax_chart.set_xlabel("Normalized time", color=MUTED, fontsize=8)
        self.ax_chart.set_ylabel("Knee flexion (°)", color=MUTED, fontsize=8)
        from stablewalk.ui.viewers.chart_hover import ChartHoverPoint, append_line_hover_points
        from stablewalk.ui.viewers.chart_interactions import finalize_chart_interactions

        hover_points: list[ChartHoverPoint] = []
        for snap, _color in (
            (left, COLOR_LEFT),
            (right, COLOR_RIGHT),
        ):
            if snap is None or not snap.knee_t:
                continue
            t = np.asarray(snap.knee_t, dtype=float)
            y = np.asarray(
                [np.nan if v is None else float(v) for v in snap.knee_y], dtype=float
            )
            if len(t) < 2:
                continue
            tn = (t - t[0]) / max(t[-1] - t[0], 1e-6)
            append_line_hover_points(
                self.ax_chart,
                tn,
                y,
                metric_name="Knee flexion",
                joint_name=snap.label or "Session",
                unit="deg",
                timestamps=tn,
                hover_points=hover_points,
            )
        self._chart_playhead = self.ax_chart.axvline(
            self._progress,
            color=TEXT,
            linewidth=1.6,
            alpha=0.9,
            linestyle="-",
            zorder=20,
        )
        finalize_chart_interactions(self.fig_chart, self.canvas_chart, hover_points=hover_points)
        self.canvas_chart.draw_idle()

    def _redraw_paths_and_chart(self) -> None:
        self._redraw_paths()
        self._redraw_chart_series()
        self._redraw_heatmap()

    def _redraw_heatmap(self) -> None:
        if not hasattr(self, "fig_heat"):
            return
        cache = ensure_compare_cache(self.gui)
        left, right = cache.left(), cache.right()
        try:
            # Clear the whole figure so repeated refreshes never stack colorbars,
            # then draw the static heatmap once and add a movable playhead line.
            self.fig_heat.clf()
            self.ax_heat = self.fig_heat.add_subplot(111)
            draw_difference_heatmap(
                self.ax_heat,
                left.sequence if left else None,
                right.sequence if right else None,
                progress=None,
                n_bins=self._heat_bins,
            )
            x = float(np.clip(self._progress, 0.0, 1.0)) * max(self._heat_bins - 1, 1)
            self._heat_playhead = self.ax_heat.axvline(
                x, color=TEXT, linewidth=1.6, alpha=0.9, zorder=20
            )
            self.canvas_heat.draw_idle()
        except Exception:
            logger.debug("Difference heatmap redraw failed", exc_info=True)
            self._heat_playhead = None

    def _update_heat_playhead(self) -> None:
        """Move the heatmap's vertical cursor to the current normalized progress."""
        ph = getattr(self, "_heat_playhead", None)
        if ph is None:
            return
        try:
            x = float(np.clip(self._progress, 0.0, 1.0)) * max(self._heat_bins - 1, 1)
            ph.set_xdata([x, x])
            self.canvas_heat.draw_idle()
        except Exception:
            pass

    def _update_chart_playhead(self) -> None:
        if self._chart_playhead is not None:
            try:
                self._chart_playhead.set_xdata([self._progress, self._progress])
                self.canvas_chart.draw_idle()
            except Exception:
                pass

    def _update_progress_readout(self) -> None:
        """Show the shared normalized gait progress that drives every panel."""
        lbl = getattr(self, "lbl_progress", None)
        if lbl is None:
            return
        try:
            lbl.configure(text=f"Gait cycle {self._progress * 100:.0f}%")
        except tk.TclError:
            pass

    def _style_chart_ax(self) -> None:
        from stablewalk.ui.viewers.chart_style import apply_chart_grid, apply_chart_panel_style

        apply_chart_panel_style(self.ax_chart)
        self.fig_chart.patch.set_facecolor(PANEL)
        self.ax_chart.tick_params(colors=MUTED, labelsize=7)
        apply_chart_grid(self.ax_chart, y_minor=True)
        for spine in self.ax_chart.spines.values():
            spine.set_color(BORDER)

    def _on_chart_hover(self, event) -> None:
        if event.inaxes != self.ax_chart or event.xdata is None or event.ydata is None:
            if self._hover_annot.get_visible():
                self._hover_annot.set_visible(False)
                self.canvas_chart.draw_idle()
            return
        self._hover_annot.xy = (event.xdata, event.ydata)
        self._hover_annot.set_text(f"t={event.xdata:.2f}\nθ={event.ydata:.1f}°")
        self._hover_annot.set_visible(True)
        self.canvas_chart.draw_idle()

    def _update_time_label(
        self,
        left: AnalyzedSessionSnapshot | None,
        right: AnalyzedSessionSnapshot | None,
        ia: int,
        ib: int,
    ) -> None:
        def _t(snap: AnalyzedSessionSnapshot | None, i: int) -> float:
            if snap is None:
                return 0.0
            if snap.knee_t and i < len(snap.knee_t):
                return float(snap.knee_t[min(i, len(snap.knee_t) - 1)])
            return i / max(snap.fps, 1e-6)

        ta, tb = _t(left, ia), _t(right, ib)
        self.lbl_time.configure(
            text=f"A F{ia} {ta:.2f}s | B F{ib} {tb:.2f}s"
        )

    def _fill_metric_card(
        self,
        card: dict[str, Any],
        metrics: CompareMetrics,
        *,
        header_prefix: str = "Session",
    ) -> None:
        from stablewalk.ui.tk.kpi_cards import parse_numeric, split_value_unit, update_kpi_card

        title = metrics.label or header_prefix
        card["header"].configure(text=f"{header_prefix} — {title}")
        kpis = card.get("kpis") or {}
        for name, val in metrics.display_map().items():
            kpi = kpis.get(name)
            if kpi is not None:
                value, unit = split_value_unit(str(val))
                available = str(val).strip() not in ("", "—", "-", "N/A")
                numeric = parse_numeric(val) if available else None
                fraction = None
                if numeric is not None and name in ("Gait Quality", "Symmetry"):
                    fraction = min(1.0, numeric / 100.0)
                elif numeric is not None and name == "Cadence":
                    fraction = min(1.0, numeric / 180.0)
                update_kpi_card(
                    kpi,
                    value=value if available else "—",
                    unit=unit if available else "",
                    available=available,
                    fraction=fraction,
                    numeric=numeric,
                    quality="neutral" if available else "unavailable",
                    tooltip=f"{name}: {val}",
                )
                continue
            lbl = card["fields"].get(name)
            if lbl is not None:
                lbl.configure(text=val, fg=TEXT)

    def _fill_diffs(
        self,
        result: SessionComparisonResult | None,
        *,
        identical: bool = False,
    ) -> None:
        for child in self.diff_host.winfo_children():
            child.destroy()
        if identical:
            tk.Label(
                self.diff_host,
                text="Both comparison sessions are identical.",
                bg=SURFACE,
                fg=WARNING,
                font=FONT_UI_SM,
                wraplength=280,
                justify=tk.LEFT,
            ).pack(anchor="w")
            return
        if result is None:
            tk.Label(
                self.diff_host,
                text="Waiting for two independent sessions…",
                bg=SURFACE,
                fg=MUTED,
                font=FONT_UI_SM,
            ).pack(anchor="w")
            return
        priority = set(_DIFF_PRIORITY)
        ordered = [d for d in result.diffs if d.name in _DIFF_PRIORITY]
        ordered.extend(d for d in result.diffs if d.name not in priority)
        shown = 0
        for d in ordered:
            if d.name == "Stability" and d.delta is None:
                continue
            if d.delta is None and d.name not in ("Stability",):
                continue
            color = TEXT
            if d.tone_right == "better":
                color = SUCCESS
            elif d.tone_right == "worse":
                color = DANGER
            tk.Label(
                self.diff_host,
                text=d.display,
                bg=SURFACE,
                fg=color,
                font=FONT_UI_SM,
                anchor="w",
            ).pack(fill=tk.X, pady=1)
            shown += 1
            if shown >= 10:
                break

    def _set_interp(self, text: str) -> None:
        self.txt_interp.configure(state=tk.NORMAL)
        self.txt_interp.delete("1.0", tk.END)
        self.txt_interp.insert("1.0", text)
        self.txt_interp.configure(state=tk.DISABLED)


__all__ = ["ComparisonModeController", "build_comparison_tab"]
