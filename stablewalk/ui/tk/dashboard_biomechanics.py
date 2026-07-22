"""Biomechanics tab layout for the StableWalk dashboard."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from stablewalk.ui.theme import (
    DASHBOARD_CARD_PAD,
    DASHBOARD_GUTTER,
    DASHBOARD_SIDE_PAD,
    FONT_UI_XS,
    MUTED,
    PANEL,
    SIDEBAR_MIN_WIDTH,
    TEXT,
    bind_responsive_wrap,
)
from stablewalk.ui.scientific_labels import (
    LABEL_BIOMECH_TAB_HEADER,
    LABEL_BIOMECH_TIMELINE,
    LABEL_CADENCE,
    LABEL_COMPOSITE_GAIT_QUALITY,
    LABEL_GAIT_SYMMETRY,
    LABEL_JOINT_ROM,
    LABEL_STABILITY_MARGIN,
    LABEL_STABILITY_STATE,
    LABEL_VIDEO_QUALITY,
    LABEL_WALKING_SPEED,
)
from stablewalk.ui.tk.dashboard_layout import (
    _bind_figure_resize,
    _card_title,
    _fit_figure_to_host,
)
from stablewalk.ui.metric_tooltips import attach_metric_tooltip, get_metric_tooltip
from stablewalk.ui.summary_metric_style import interpret_summary_metric
from stablewalk.ui.tk.kpi_cards import (
    create_kpi_card,
    create_kpi_grid_card,
    parse_numeric,
    update_kpi_card,
)

_ANALYSIS_PANEL_PAD = DASHBOARD_CARD_PAD

# (key, caption, metric-tooltip key) for the compact summary strip above the graphs.
_BIOMECH_SUMMARY_FIELDS: tuple[tuple[str, str, str | None], ...] = (
    ("heel_strikes", "Heel Strikes", None),
    ("cadence", "Avg Cadence", "cadence"),
    ("stance", "Stance %", None),
    ("swing", "Swing %", None),
    ("gait_quality", "Gait Quality", "gait_quality"),
    ("stability", "Stability", "stability"),
)

_BIOMECH_SIDEBAR_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("gait_quality", LABEL_COMPOSITE_GAIT_QUALITY, "gait_quality", "lbl_biomech_gait_quality"),
    ("symmetry", LABEL_GAIT_SYMMETRY, "symmetry", "lbl_biomech_symmetry"),
    ("stability_margin", LABEL_STABILITY_MARGIN, "stability", "lbl_biomech_stability_margin"),
    ("stability_state", LABEL_STABILITY_STATE, "stability", "lbl_biomech_stability_state"),
    ("cadence", LABEL_CADENCE, "cadence", "lbl_biomech_cadence"),
    ("walking_speed", LABEL_WALKING_SPEED, "walking_speed", "lbl_biomech_walking_speed"),
)

# Short chip labels for stability states (keep the compact card from overflowing).
_STABILITY_SHORT = {
    "Stable": "Stable",
    "Reduced Stability": "Reduced",
    "Unstable": "Unstable",
}


def _quality_for_key(
    key: str,
    value_text: str,
    *,
    available: bool,
    view_type: str | None = None,
    stability_valid_ratio: float | None = None,
) -> str:
    if not available:
        return "unavailable"
    from types import SimpleNamespace

    from stablewalk.ui.summary_metric_style import (
        is_view_limited_for_quality,
        soften_view_limited_level,
    )

    view_limited = is_view_limited_for_quality(view_type)

    if key in ("heel_strikes",):
        return "neutral" if available else "unavailable"
    if key in ("stance", "swing"):
        num = parse_numeric(value_text)
        if num is None:
            return "neutral"
        target = 60.0 if key == "stance" else 40.0
        delta = abs(num - target)
        normal_max = 14.0 if view_limited else 10.0
        borderline_max = 28.0 if view_limited else 20.0
        if delta <= normal_max:
            return "normal"
        if delta <= borderline_max:
            return "borderline"
        return soften_view_limited_level("abnormal", view_limited=view_limited)
    if key == "stability_state":
        lower = value_text.lower()
        if "unavailable" in lower:
            return "unavailable"
        if "unstable" in lower:
            return soften_view_limited_level("abnormal", view_limited=view_limited)
        if "reduced" in lower:
            return "borderline"
        if "stable" in lower:
            return "normal"
        return "neutral"

    field = SimpleNamespace(available=True, value=value_text, tier="calculated")
    mapped = {
        "gait_quality": "gait_quality",
        "cadence": "cadence",
        "walking_speed": "walking_speed",
        "symmetry": "symmetry",
        "stability": "stability_margin",
        "stability_margin": "stability_margin",
        "video_quality": "video_quality",
    }.get(key, key)
    return interpret_summary_metric(
        mapped,
        field,  # type: ignore[arg-type]
        view_type=view_type,
        stability_valid_ratio=stability_valid_ratio,
    )


def _build_biomech_summary_card(gui, parent: tk.Misc) -> tk.Frame:
    """A compact horizontal strip of at-a-glance biomechanics KPI cards."""
    card = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    for col in range(len(_BIOMECH_SUMMARY_FIELDS)):
        card.columnconfigure(col, weight=1, uniform="biomech_summary")

    gui._biomech_summary_kpis = {}
    gui._biomech_summary_value_labels = {}
    for col, (key, caption, tip) in enumerate(_BIOMECH_SUMMARY_FIELDS):
        tip_text = get_metric_tooltip(tip) if tip else f"{caption} — live session metric"
        kpi = create_kpi_grid_card(
            card,
            key=key,
            title=caption,
            tooltip=tip_text,
            compact=True,
            show_bar=key in ("stance", "swing", "gait_quality"),
        )
        kpi.frame.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 4, 0))
        gui._biomech_summary_kpis[key] = kpi
        # Compatibility: playback code historically updated value labels directly.
        gui._biomech_summary_value_labels[key] = kpi.value_lbl

    return card


def update_biomechanics_summary_card(gui, ba, gait_cycle, *, frame_index=None) -> None:
    """Refresh the compact summary strip; safe to call every playback frame."""
    kpis = getattr(gui, "_biomech_summary_kpis", None)
    if not kpis:
        return

    biomech = getattr(gui, "_biomech", None)
    view_type = getattr(biomech, "view_type", None) if biomech is not None else None
    sm = getattr(ba, "stability_margin", None) if ba is not None else None
    valid_ratio = getattr(sm, "valid_frame_ratio", None) if sm is not None else None

    def _set(
        key: str,
        value: str,
        *,
        unit: str = "",
        available: bool = True,
        fraction: float | None = None,
        numeric: float | None = None,
        tip: str | None = None,
    ) -> None:
        card = kpis.get(key)
        if card is None:
            return
        display = f"{value}{(' ' + unit) if unit else ''}".strip()
        update_kpi_card(
            card,
            value=value,
            unit=unit,
            quality=_quality_for_key(
                key,
                display,
                available=available,
                view_type=view_type,
                stability_valid_ratio=valid_ratio,
            ),  # type: ignore[arg-type]
            available=available,
            fraction=fraction,
            numeric=numeric if numeric is not None else parse_numeric(value),
            tooltip=tip,
        )

    metrics = getattr(gait_cycle, "metrics", None) if gait_cycle is not None else None
    gm = getattr(ba, "gait_metrics", None) if ba is not None else None

    if ba is None and metrics is None:
        for key in kpis:
            _set(key, "—", available=False)
        return

    if metrics is not None:
        hs = int((metrics.left_heel_strike_count or 0) + (metrics.right_heel_strike_count or 0))
        _set("heel_strikes", str(hs) if hs > 0 else "—", available=hs > 0, numeric=float(hs))
    else:
        _set("heel_strikes", "—", available=False)

    cadence = None
    if gm is not None and gm.cadence is not None:
        cadence = gm.cadence.value
    if cadence is None and metrics is not None:
        cadence = metrics.cadence_steps_per_min
    if cadence:
        _set("cadence", f"{cadence:.0f}", unit="steps/min", available=True, numeric=float(cadence))
    else:
        _set("cadence", "—", available=False)

    stance = swing = None
    if gm is not None:
        if gm.stance_pct is not None:
            stance = gm.stance_pct.value
        if gm.swing_pct is not None:
            swing = gm.swing_pct.value
    if (stance is None or swing is None) and metrics is not None:
        st = metrics.average_stance_duration_s
        sw = metrics.average_swing_duration_s
        cycle = (st or 0.0) + (sw or 0.0)
        if cycle > 0:
            if stance is None:
                stance = (st or 0.0) / cycle * 100.0
            if swing is None:
                swing = (sw or 0.0) / cycle * 100.0
    if stance is not None:
        _set(
            "stance",
            f"{stance:.0f}",
            unit="%",
            available=True,
            fraction=float(stance) / 100.0,
            numeric=float(stance),
        )
    else:
        _set("stance", "—", available=False)
    if swing is not None:
        _set(
            "swing",
            f"{swing:.0f}",
            unit="%",
            available=True,
            fraction=float(swing) / 100.0,
            numeric=float(swing),
        )
    else:
        _set("swing", "—", available=False)

    gq = getattr(ba, "gait_quality", None) if ba is not None else None
    score = gq.score if gq is not None else None
    if score is not None:
        _set(
            "gait_quality",
            f"{score:.0f}",
            unit="/100",
            available=True,
            fraction=float(score) / 100.0,
            numeric=float(score),
        )
    else:
        _set("gait_quality", "—", available=False)

    text = "—"
    available = False
    numeric = None
    tip = None
    sm = getattr(ba, "stability_margin", None) if ba is not None else None
    if sm is not None:
        if frame_index is not None:
            for f in sm.per_frame:
                if f.frame_index == frame_index:
                    text = _STABILITY_SHORT.get(f.stability_state, f.stability_state)
                    available = True
                    tip = f.stability_state
                    break
        if text == "—":
            pct = getattr(sm, "stable_pct", None)
            if pct is not None:
                if pct >= 80.0:
                    classification = "Stable"
                elif pct >= 50.0:
                    classification = "Reduced Stability"
                else:
                    classification = "Unstable"
                text = _STABILITY_SHORT.get(classification, classification)
                available = True
                numeric = float(pct)
                tip = f"{pct:.0f}% stable frames"
    _set("stability", text, available=available, numeric=numeric, tip=tip)


def build_biomechanics_tab(gui, parent: ttk.Frame) -> None:
    """Install Biomechanics tab: summary cards + synchronized plots."""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)

    header = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
    gui._biomechanics_header = header
    tk.Label(
        header,
        text=LABEL_BIOMECH_TAB_HEADER,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(side=tk.LEFT)

    body = ttk.Frame(parent)
    body.grid(row=1, column=0, sticky="nsew")
    body.columnconfigure(0, weight=0, minsize=max(SIDEBAR_MIN_WIDTH, 200))
    body.columnconfigure(1, weight=3)
    body.rowconfigure(0, weight=1)

    sidebar = ttk.Frame(body, style="Card.TFrame", padding=DASHBOARD_SIDE_PAD)
    sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, DASHBOARD_GUTTER))

    chart_panel = ttk.LabelFrame(
        body,
        text=_card_title(LABEL_BIOMECH_TIMELINE),
        style="Card.TLabelframe",
        padding=_ANALYSIS_PANEL_PAD,
    )
    chart_panel.grid(row=0, column=1, sticky="nsew")
    chart_panel.columnconfigure(0, weight=1)

    gui._biomech_summary_card = _build_biomech_summary_card(gui, chart_panel)
    gui._biomech_summary_card.grid(row=0, column=0, sticky="ew", pady=(0, 6))

    ttk.Label(
        sidebar,
        text="Session Metrics",
        style="SideMuted.TLabel",
    ).pack(anchor=tk.W, pady=(0, 4))

    gui._biomech_kpi_cards = {}
    for key, title, tip_key, attr in _BIOMECH_SIDEBAR_FIELDS:
        tip = get_metric_tooltip(tip_key) or title
        kpi = create_kpi_card(
            sidebar,
            key=key,
            title=title,
            tooltip=tip,
            compact=True,
            show_bar=key in ("gait_quality", "symmetry", "stability_margin", "cadence"),
            fill=False,
        )
        gui._biomech_kpi_cards[key] = kpi
        # Keep legacy label attributes so existing update paths still resolve.
        setattr(gui, attr, kpi.value_lbl)

    rom_title = ttk.Label(
        sidebar,
        text=LABEL_JOINT_ROM,
        style="SideMuted.TLabel",
    )
    rom_title.pack(anchor=tk.W, pady=(8, 2))
    attach_metric_tooltip(rom_title, "rom")
    gui.lbl_biomech_rom = tk.Label(
        sidebar,
        text="—",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
    )
    gui.lbl_biomech_rom.pack(anchor=tk.W, fill=tk.X)
    attach_metric_tooltip(gui.lbl_biomech_rom, "rom")
    bind_responsive_wrap(gui, sidebar, ("lbl_biomech_rom",), margin=12)

    video_kpi = create_kpi_card(
        sidebar,
        key="video_quality",
        title=LABEL_VIDEO_QUALITY,
        tooltip=get_metric_tooltip("video_quality") or LABEL_VIDEO_QUALITY,
        compact=True,
        show_bar=True,
        fill=False,
    )
    gui._biomech_kpi_cards["video_quality"] = video_kpi
    gui.lbl_biomech_video_score = video_kpi.value_lbl
    gui.lbl_biomech_video_checks = tk.Label(
        sidebar,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=220,
    )
    gui.lbl_biomech_video_checks.pack(anchor=tk.W, fill=tk.X, pady=(0, 4))

    gui.fig_biomech = Figure(figsize=(11.0, 9.5), dpi=100, facecolor=PANEL)
    gui.biomech_chart_host = tk.Frame(chart_panel, bg=PANEL, highlightthickness=0)
    gui.biomech_chart_host.columnconfigure(0, weight=1)
    gui.biomech_chart_host.rowconfigure(0, weight=1)
    gui.canvas_biomech = FigureCanvasTkAgg(gui.fig_biomech, master=gui.biomech_chart_host)
    w = gui.canvas_biomech.get_tk_widget()
    w.configure(bg=PANEL, highlightthickness=0)
    w.grid(row=0, column=0, sticky="nsew")
    _bind_figure_resize(gui.canvas_biomech, gui.fig_biomech, min_px=80)
    from stablewalk.ui.viewers.chart_interactions import (
        attach_chart_interactions,
        build_chart_tools_bar,
    )

    attach_chart_interactions(gui.fig_biomech, gui.canvas_biomech)
    # Summary strip (row 0) · tools bar (row 1) · chart host (row 2, expands).
    chart_panel.rowconfigure(0, weight=0)
    chart_panel.rowconfigure(1, weight=0)
    chart_panel.rowconfigure(2, weight=1)
    gui.biomech_chart_tools = build_chart_tools_bar(
        gui,
        chart_panel,
        fig_attr="fig_biomech",
        canvas_attr="canvas_biomech",
        export_name="biomechanics",
    )
    gui.biomech_chart_tools.grid(row=1, column=0, sticky="ew", pady=(0, 2))
    gui.biomech_chart_host.grid(row=2, column=0, sticky="nsew")
    gui._fit_biomech_canvas = lambda: _fit_figure_to_host(
        gui.canvas_biomech,
        gui.fig_biomech,
        host=gui.biomech_chart_host,
        min_px=80,
    )
    _biomech_cfg_job: dict[str, object] = {"id": None}

    def _on_biomech_panel_configure(_event: object | None = None) -> None:
        job = _biomech_cfg_job.get("id")
        if job is not None:
            try:
                gui.root.after_cancel(job)  # type: ignore[arg-type]
            except Exception:
                pass

        def _run() -> None:
            _biomech_cfg_job["id"] = None
            fit = getattr(gui, "_fit_biomech_canvas", None)
            if callable(fit):
                try:
                    fit()
                except Exception:
                    pass

        try:
            _biomech_cfg_job["id"] = gui.root.after(32, _run)
        except Exception:
            _run()

    chart_panel.bind("<Configure>", _on_biomech_panel_configure, add="+")


__all__ = ["build_biomechanics_tab", "update_biomechanics_summary_card"]
