"""Results Summary tab — clinical-style gait analysis report."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.theme import (
    BG,
    BORDER,
    DASHBOARD_CARD_PAD,
    DASHBOARD_SIDE_PAD,
    ELEVATED,
    FONT_PANEL_HEADER,
    FONT_SUMMARY_CATEGORY,
    FONT_SUMMARY_INTERPRETATION,
    FONT_SUMMARY_METRIC_TIER,
    FONT_SUMMARY_METRIC_TITLE,
    FONT_SUMMARY_METRIC_VALUE,
    FONT_SUMMARY_METRIC_VALUE_SM,
    FONT_SUMMARY_REPORT_SUBTITLE,
    FONT_SUMMARY_REPORT_TITLE,
    FONT_SUMMARY_SECTION_RULE,
    FONT_UI_XS,
    MUTED,
    PAD_MD,
    PAD_SM,
    PANEL,
    SUCCESS,
    TEXT,
    WARNING,
    create_tooltip,
    bind_responsive_wrap,
)
from stablewalk.ui.scientific_labels import (
    LABEL_CADENCE,
    LABEL_CADENCE_SHORT,
    LABEL_COM_FULL,
    LABEL_GAIT_QUALITY,
    LABEL_GAIT_SYMMETRY,
    LABEL_JOINT_ROM_SUMMARY,
    LABEL_STABILITY_MARGIN,
    LABEL_TRACKING_CONFIDENCE,
    LABEL_PIPELINE_CONFIDENCE,
    LABEL_VIDEO_QUALITY,
    LABEL_VIRTUAL_GRF_PANEL,
    LABEL_WALKING_SPEED,
    LABEL_ANALYSIS_CONFIDENCE,
    format_tier_badge,
)

_GAIT_PERFORMANCE_METRICS: tuple[tuple[str, str], ...] = (
    ("gait_quality", LABEL_GAIT_QUALITY),
    ("walking_speed", LABEL_WALKING_SPEED),
    ("cadence", LABEL_CADENCE),
    ("symmetry", LABEL_GAIT_SYMMETRY),
    ("stability_margin", LABEL_STABILITY_MARGIN),
)

_BIOMECHANICS_METRICS: tuple[tuple[str, str, bool, int], ...] = (
    ("com", LABEL_COM_FULL, False, 1),
    ("vgrf", LABEL_VIRTUAL_GRF_PANEL, False, 1),
    ("joint_rom", LABEL_JOINT_ROM_SUMMARY, True, 2),
)

_CONFIDENCE_METRICS: tuple[tuple[str, str], ...] = (
    ("video_quality", LABEL_VIDEO_QUALITY),
    ("tracking_confidence", LABEL_TRACKING_CONFIDENCE),
    ("pipeline_confidence", LABEL_PIPELINE_CONFIDENCE),
)

_GAIT_EVENT_KEYS: tuple[tuple[str, str], ...] = (
    ("heel_strike", "Heel Strike"),
    ("toe_off", "Toe Off"),
    ("double_support", "Double Support"),
    ("single_support", "Single Support (ipsilateral)"),
)

_EVENT_NAME_TO_KEY = {
    "Heel Strike": "heel_strike",
    "Toe Off": "toe_off",
    "Double Support": "double_support",
    "Single Support": "single_support",
}


def _install_summary_scroll(parent: ttk.Frame) -> tk.Frame:
    """Vertical scroll shell for the report body."""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)

    outer = ttk.Frame(parent)
    outer.grid(row=0, column=0, sticky="nsew")
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(0, weight=1)

    canvas = tk.Canvas(outer, bg=PANEL, highlightthickness=0, borderwidth=0, bd=0)
    vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    canvas.configure(yscrollcommand=vsb.set, yscrollincrement=1)

    scroll_content = tk.Frame(canvas, bg=PANEL, highlightthickness=0)
    inner_id = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
    scroll_content.columnconfigure(0, weight=1)

    def _sync(_event: object | None = None) -> None:
        try:
            canvas.update_idletasks()
            cw = max(canvas.winfo_width(), 1)
            canvas.coords(inner_id, 0, 0)
            canvas.itemconfigure(inner_id, width=cw)
            bbox = canvas.bbox("all")
            if bbox is not None:
                canvas.configure(scrollregion=bbox)
        except tk.TclError:
            pass

    def _scroll_cmd(*args: object) -> None:
        canvas.yview(*args)

    vsb.configure(command=_scroll_cmd)
    scroll_content.bind("<Configure>", _sync, add="+")
    canvas.bind("<Configure>", _sync, add="+")

    def _wheel(event: tk.Event) -> str | None:
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"
        return None

    for widget in (canvas, scroll_content):
        widget.bind("<MouseWheel>", _wheel, add="+")

    return scroll_content


def _section_header(parent: tk.Misc, *, row: int, title: str) -> None:
    block = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    block.grid(row=row, column=0, sticky="ew", pady=(PAD_MD, PAD_SM))
    block.columnconfigure(0, weight=1)

    tk.Label(
        block,
        text="─" * 32,
        bg=PANEL,
        fg=BORDER,
        font=FONT_SUMMARY_SECTION_RULE,
        anchor="w",
    ).grid(row=0, column=0, sticky="ew")

    tk.Label(
        block,
        text=title,
        bg=PANEL,
        fg=TEXT,
        font=FONT_SUMMARY_CATEGORY,
        anchor="w",
    ).grid(row=1, column=0, sticky="w", pady=(4, 0))


def _build_metric_card(
    parent: tk.Misc,
    *,
    row: int,
    column: int,
    key: str,
    title: str,
    gui,
    multiline: bool = False,
    columnspan: int = 1,
) -> None:
    """One elevated KPI metric card with title, large value, unit, trend, quality."""
    from stablewalk.ui.tk.kpi_cards import create_kpi_grid_card, set_kpi_tooltip

    kpi = create_kpi_grid_card(
        parent,
        key=key,
        title=title,
        compact=multiline,
        show_bar=key
        in (
            "gait_quality",
            "symmetry",
            "cadence",
            "walking_speed",
            "video_quality",
            "tracking_confidence",
        ),
    )
    kpi.frame.grid(
        row=row,
        column=column,
        columnspan=columnspan,
        sticky="nsew",
        padx=(0, PAD_SM),
        pady=(0, PAD_SM),
    )
    if multiline:
        kpi.value_lbl.configure(
            font=FONT_SUMMARY_METRIC_VALUE_SM,
            wraplength=280,
            justify=tk.LEFT,
        )

    gui._summary_field_labels[key] = kpi.value_lbl
    gui._summary_tier_labels[key] = kpi.quality_lbl
    gui._summary_kpi_cards = getattr(gui, "_summary_kpi_cards", {})
    gui._summary_kpi_cards[key] = kpi
    from stablewalk.ui.metric_tooltips import get_metric_tooltip

    tip = get_metric_tooltip(key) or "Updates when analysis completes or playhead moves"
    set_kpi_tooltip(kpi, tip)


def _build_event_card(
    parent: tk.Misc,
    *,
    row: int,
    column: int,
    key: str,
    title: str,
    gui,
) -> None:
    """KPI-styled card for a detected gait event (status + detail preserved)."""
    from stablewalk.ui.tk.kpi_cards import create_kpi_grid_card, set_kpi_tooltip

    kpi = create_kpi_grid_card(
        parent,
        key=key,
        title=title,
        compact=True,
        show_bar=False,
    )
    kpi.frame.grid(
        row=row,
        column=column,
        sticky="nsew",
        padx=(0, PAD_SM),
        pady=(0, PAD_SM),
    )
    kpi.value_lbl.configure(font=FONT_SUMMARY_METRIC_VALUE_SM)
    kpi.quality_lbl.configure(
        wraplength=180,
        justify=tk.LEFT,
    )

    # status → large value; detail → quality line (same information as before)
    gui._summary_event_labels[key] = (kpi.value_lbl, kpi.quality_lbl)
    gui._summary_event_kpis = getattr(gui, "_summary_event_kpis", {})
    gui._summary_event_kpis[key] = kpi
    from stablewalk.ui.metric_tooltips import get_metric_tooltip

    tip = get_metric_tooltip(key)
    if tip:
        set_kpi_tooltip(kpi, tip)


def build_results_summary_tab(gui, parent: ttk.Frame) -> None:
    """Install the Results Summary tab with export actions and report layout."""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)
    parent.rowconfigure(2, weight=0)

    header = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", pady=(0, PAD_SM))
    header.columnconfigure(0, weight=1)

    title_block = tk.Frame(header, bg=PANEL, highlightthickness=0)
    title_block.grid(row=0, column=0, sticky="w")
    tk.Label(
        title_block,
        text="Gait Analysis Report",
        bg=PANEL,
        fg=TEXT,
        font=FONT_SUMMARY_REPORT_TITLE,
        anchor="w",
    ).pack(anchor="w")
    tk.Label(
        title_block,
        text="StableWalk session summary · estimated biomechanical parameters",
        bg=PANEL,
        fg=MUTED,
        font=FONT_SUMMARY_REPORT_SUBTITLE,
        anchor="w",
    ).pack(anchor="w", pady=(2, 0))

    btn_row = tk.Frame(header, bg=PANEL, highlightthickness=0)
    btn_row.grid(row=0, column=1, sticky="e")
    gui._results_summary_btn_row = btn_row
    gui.btn_export_summary_json = ttk.Button(
        btn_row,
        text="Export JSON",
        command=lambda: gui._export_analysis_summary_json(),
        state=tk.DISABLED,
    )
    gui.btn_export_summary_json.pack(side=tk.LEFT, padx=(0, 4))
    gui.btn_export_summary_report = ttk.Button(
        btn_row,
        text="Export Report",
        command=lambda: gui._export_analysis_summary_report(),
        state=tk.DISABLED,
    )
    gui.btn_export_summary_report.pack(side=tk.LEFT)
    create_tooltip(gui.btn_export_summary_json, "Export the current summary as JSON")
    create_tooltip(gui.btn_export_summary_report, "Export a human-readable text report")

    scroll_host = ttk.Frame(parent)
    scroll_host.grid(row=1, column=0, sticky="nsew")
    scroll_host.columnconfigure(0, weight=1)
    scroll_host.rowconfigure(0, weight=1)

    report_body = _install_summary_scroll(scroll_host)
    report_body.columnconfigure(0, weight=1)

    session_card = ttk.LabelFrame(
        report_body,
        text="  Session  ",
        style="Card.TLabelframe",
        padding=DASHBOARD_CARD_PAD,
    )
    session_card.grid(row=0, column=0, sticky="ew", pady=(0, PAD_SM))
    session_card.columnconfigure(0, weight=1)

    gui.lbl_summary_source = tk.Label(
        session_card,
        text="No video analyzed",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_summary_source.grid(row=0, column=0, sticky="ew")

    gui.lbl_summary_playhead = tk.Label(
        session_card,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_summary_playhead.grid(row=1, column=0, sticky="ew", pady=(4, 0))

    metrics_host = tk.Frame(report_body, bg=PANEL, highlightthickness=0)
    metrics_host.grid(row=1, column=0, sticky="ew")
    metrics_host.columnconfigure(0, weight=1)

    gui._summary_field_labels: dict[str, tk.Label] = {}
    gui._summary_tier_labels: dict[str, tk.Label] = {}
    gui._summary_kpi_cards: dict[str, object] = {}
    gui._summary_event_labels: dict[str, tuple[tk.Label, tk.Label]] = {}

    row = 0
    _section_header(metrics_host, row=row, title="Gait Performance")
    row += 1

    gait_grid = tk.Frame(metrics_host, bg=PANEL, highlightthickness=0)
    gait_grid.grid(row=row, column=0, sticky="ew", pady=(0, PAD_SM))
    for col in range(3):
        gait_grid.columnconfigure(col, weight=1)
    for index, (key, title) in enumerate(_GAIT_PERFORMANCE_METRICS):
        _build_metric_card(
            gait_grid,
            row=index // 3,
            column=index % 3,
            key=key,
            title=title,
            gui=gui,
        )
    row += 1

    _section_header(metrics_host, row=row, title="Biomechanics")
    row += 1

    bio_grid = tk.Frame(metrics_host, bg=PANEL, highlightthickness=0)
    bio_grid.grid(row=row, column=0, sticky="ew", pady=(0, PAD_SM))
    for col in range(2):
        bio_grid.columnconfigure(col, weight=1)
    for index, (key, title, multiline, colspan) in enumerate(_BIOMECHANICS_METRICS):
        _build_metric_card(
            bio_grid,
            row=index // 2,
            column=index % 2 if colspan == 1 else 0,
            key=key,
            title=title,
            gui=gui,
            multiline=multiline,
            columnspan=colspan,
        )
    row += 1

    _section_header(metrics_host, row=row, title="Detected Events")
    row += 1

    events_grid = tk.Frame(metrics_host, bg=PANEL, highlightthickness=0)
    events_grid.grid(row=row, column=0, sticky="ew", pady=(0, PAD_SM))
    for col in range(4):
        events_grid.columnconfigure(col, weight=1)
    for index, (key, title) in enumerate(_GAIT_EVENT_KEYS):
        _build_event_card(
            events_grid,
            row=0,
            column=index,
            key=key,
            title=title,
            gui=gui,
        )
    row += 1

    _section_header(metrics_host, row=row, title="Analysis Confidence")
    row += 1

    quality_grid = tk.Frame(metrics_host, bg=PANEL, highlightthickness=0)
    quality_grid.grid(row=row, column=0, sticky="ew", pady=(0, PAD_SM))
    for col in range(3):
        quality_grid.columnconfigure(col, weight=1)
    for index, (key, title) in enumerate(_CONFIDENCE_METRICS):
        _build_metric_card(
            quality_grid,
            row=0,
            column=index,
            key=key,
            title=title,
            gui=gui,
        )
    row += 1

    interp_card = ttk.LabelFrame(
        report_body,
        text="  Interpretation  ",
        style="Card.TLabelframe",
        padding=DASHBOARD_SIDE_PAD,
    )
    interp_card.grid(row=2, column=0, sticky="ew", pady=(PAD_MD, PAD_SM))
    interp_card.columnconfigure(0, weight=1)

    tk.Label(
        interp_card,
        text="Auto-generated clinical-style summary (not a medical diagnosis)",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(0, 4))

    gui.lbl_summary_scientific_interpretation = tk.Label(
        interp_card,
        text=(
            "Run gait analysis to generate an automatic interpretation of the estimated "
            "biomechanical parameters for this recording."
        ),
        bg=PANEL,
        fg=MUTED,
        font=FONT_SUMMARY_INTERPRETATION,
        anchor="w",
        justify=tk.LEFT,
    )
    gui.lbl_summary_scientific_interpretation.grid(
        row=1, column=0, sticky="ew", padx=PAD_SM, pady=(0, PAD_SM)
    )
    bind_responsive_wrap(
        gui,
        report_body,
        ("lbl_summary_scientific_interpretation",),
        margin=24,
    )

    from stablewalk.ui.tk.dashboard_pipeline_status import build_pipeline_status_panel

    pipeline_panel = build_pipeline_status_panel(
        gui,
        parent,
        host_attr="_pipeline_status_host_summary",
        section_attr="_section_pipeline_status_summary",
    )
    pipeline_panel.grid(row=2, column=0, sticky="nsew", pady=(PAD_SM, 0))


def _apply_summary_field(
    key: str,
    field,
    fields: dict[str, tk.Label],
    tiers: dict[str, tk.Label],
    *,
    kpi_cards: dict | None = None,
    gui=None,
) -> None:
    from stablewalk.ui.metric_tooltips import combine_metric_tooltip, get_metric_tooltip
    from stablewalk.ui.summary_metric_style import (
        format_status_line,
        interpret_summary_metric,
        metric_visual_style,
    )
    from stablewalk.ui.tk.kpi_cards import parse_numeric, split_value_unit, update_kpi_card

    lbl = fields.get(key)
    tier_lbl = tiers.get(key)
    card = (kpi_cards or {}).get(key)
    if lbl is None and card is None:
        return

    if field is None:
        if card is not None:
            update_kpi_card(
                card,
                value="—",
                available=False,
                tooltip=combine_metric_tooltip(get_metric_tooltip(key), "No data"),
            )
        elif lbl is not None:
            lbl.configure(text="—", fg=MUTED)
            if tier_lbl is not None:
                tier_lbl.configure(text="")
            create_tooltip(
                lbl,
                combine_metric_tooltip(get_metric_tooltip(key), "No data"),
                wraplength=340,
            )
        return

    tip = field.reason if field.reason else field.value
    if not field.available:
        tip = field.reason or "Not available"
    tip = combine_metric_tooltip(get_metric_tooltip(key), tip)
    view_type = None
    valid_ratio = None
    if gui is not None:
        biomech = getattr(gui, "_biomech", None)
        view_type = getattr(biomech, "view_type", None) if biomech is not None else None
        ba = getattr(gui, "_biomech_analysis", None)
        if ba is not None and ba.stability_margin is not None:
            valid_ratio = ba.stability_margin.valid_frame_ratio
    level = interpret_summary_metric(
        key,
        field,
        view_type=view_type,
        stability_valid_ratio=valid_ratio,
    )
    style = metric_visual_style(level)
    status = format_status_line(level, field.tier)

    if card is not None:
        value, unit = split_value_unit(field.value)
        # Keep rich multi-clause values intact (ROM / COM narratives).
        if key in {"joint_rom", "com", "vgrf", "stability_margin"}:
            value, unit = field.value, ""
        fraction = None
        numeric = parse_numeric(field.value)
        if numeric is not None and key in (
            "gait_quality",
            "symmetry",
            "video_quality",
            "tracking_confidence",
        ):
            fraction = min(1.0, numeric / 100.0)
        elif numeric is not None and key == "cadence":
            fraction = min(1.0, numeric / 180.0)
        elif numeric is not None and key == "walking_speed":
            fraction = min(1.0, numeric / 2.0)
        update_kpi_card(
            card,
            value=value if field.available else field.value,
            unit=unit if field.available else "",
            quality=level,  # type: ignore[arg-type]
            available=bool(field.available),
            fraction=fraction,
            numeric=numeric if field.available else None,
            tooltip=tip,
        )
        # Preserve tier text alongside quality label when both are useful.
        if status:
            card.quality_lbl.configure(text=status, fg=style.value_fg)
        return

    color = style.value_fg if field.available else WARNING
    lbl.configure(text=field.value, fg=color)
    if tier_lbl is not None:
        tier_lbl.configure(text=status or format_tier_badge(field.tier))
    create_tooltip(lbl, tip, wraplength=340)


def update_results_summary_panel(gui, summary) -> None:
    """Refresh summary labels from an AnalysisSummary instance."""
    from stablewalk.analysis.analysis_summary import AnalysisSummary

    if not isinstance(summary, AnalysisSummary):
        return

    fields = gui._summary_field_labels if hasattr(gui, "_summary_field_labels") else {}
    tiers = gui._summary_tier_labels if hasattr(gui, "_summary_tier_labels") else {}
    kpi_cards = getattr(gui, "_summary_kpi_cards", {}) or {}
    if not fields and not kpi_cards:
        return

    src = getattr(gui, "lbl_summary_source", None)
    if src is not None:
        if summary.source:
            src.configure(text=f"Source recording: {summary.source}", fg=TEXT)
        else:
            src.configure(text="No video analyzed", fg=MUTED)

    playhead = getattr(gui, "lbl_summary_playhead", None)
    if playhead is not None:
        if summary.timestamp_s is not None and summary.frame_index is not None:
            playhead.configure(
                text=f"Playhead sync: frame {summary.frame_index} at {summary.timestamp_s:.2f} s",
                fg=MUTED,
            )
        else:
            playhead.configure(text="")

    mapping = {
        "gait_quality": summary.overall_gait_quality,
        "cadence": summary.cadence,
        "walking_speed": summary.walking_speed,
        "symmetry": summary.symmetry,
        "stability_margin": summary.stability_margin,
        "com": summary.center_of_mass,
        "vgrf": summary.estimated_virtual_grf,
        "joint_rom": summary.joint_rom_summary,
        "video_quality": summary.video_quality,
        "tracking_confidence": summary.tracking_confidence,
        "pipeline_confidence": summary.pipeline_confidence,
        "confidence": summary.analysis_confidence,
    }
    for key, field in mapping.items():
        _apply_summary_field(
            key, field, fields, tiers, kpi_cards=kpi_cards, gui=gui
        )

    event_labels = getattr(gui, "_summary_event_labels", {})
    event_kpis = getattr(gui, "_summary_event_kpis", {}) or {}
    if event_labels:
        from stablewalk.ui.metric_tooltips import combine_metric_tooltip, get_metric_tooltip
        from stablewalk.ui.tk.kpi_cards import update_kpi_card

        events_by_key = {
            _EVENT_NAME_TO_KEY.get(ev.name, ""): ev for ev in summary.gait_events
        }
        for key, (status_lbl, detail_lbl) in event_labels.items():
            ev = events_by_key.get(key)
            kpi = event_kpis.get(key)
            if ev is None:
                if kpi is not None:
                    update_kpi_card(
                        kpi,
                        value="—",
                        unit="",
                        available=False,
                        quality="unavailable",
                        tooltip=get_metric_tooltip(key) or "Gait event status",
                    )
                    detail_lbl.configure(text="")
                else:
                    status_lbl.configure(text="—", fg=MUTED)
                    detail_lbl.configure(text="")
                continue
            tip = combine_metric_tooltip(
                get_metric_tooltip(key), f"{ev.name}: {ev.detail}"
            )
            if kpi is not None:
                if ev.detected:
                    update_kpi_card(
                        kpi,
                        value="Detected",
                        unit="",
                        available=True,
                        quality="normal",
                        numeric=1.0,
                        tooltip=tip,
                    )
                else:
                    update_kpi_card(
                        kpi,
                        value="✗ Not detected",
                        unit="",
                        available=True,
                        quality="borderline",
                        numeric=0.0,
                        tooltip=tip,
                    )
                # Preserve event detail on the quality line (same info as before).
                detail_lbl.configure(text=ev.detail, fg=MUTED)
            elif ev.detected:
                status_lbl.configure(text="Detected", fg=SUCCESS)
                detail_lbl.configure(text=ev.detail)
                create_tooltip(status_lbl, tip, wraplength=340)
                create_tooltip(detail_lbl, tip, wraplength=340)
            else:
                status_lbl.configure(text="✗ Not detected", fg=WARNING)
                detail_lbl.configure(text=ev.detail)
                create_tooltip(status_lbl, tip, wraplength=340)
                create_tooltip(detail_lbl, tip, wraplength=340)

    interp_lbl = getattr(gui, "lbl_summary_scientific_interpretation", None)
    if interp_lbl is not None:
        text = (summary.scientific_interpretation or "").strip()
        if text:
            interp_lbl.configure(text=text, fg=TEXT)
            create_tooltip(
                interp_lbl,
                "Auto-generated from estimated and derived parameters; not a clinical diagnosis.",
            )
        elif summary.source:
            interp_lbl.configure(
                text="Interpretation unavailable for this session.",
                fg=WARNING,
            )
        else:
            interp_lbl.configure(
                text=(
                    "Run gait analysis to generate an automatic interpretation of the estimated "
                    "biomechanical parameters for this recording."
                ),
                fg=MUTED,
            )

    export_state = tk.NORMAL if summary.source else tk.DISABLED
    for attr in ("btn_export_summary_json", "btn_export_summary_report"):
        btn = getattr(gui, attr, None)
        if btn is not None:
            btn.configure(state=export_state)

    gui._analysis_summary_cache = summary


__all__ = ["build_results_summary_tab", "update_results_summary_panel"]
