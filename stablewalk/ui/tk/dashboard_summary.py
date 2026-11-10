"""Results Summary tab — clinical-style gait analysis report."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.theme import (
    DASHBOARD_CARD_PAD,
    DANGER,
    ELEVATED,
    FONT_SUMMARY_INTERPRETATION,
    FONT_SUMMARY_METRIC_VALUE_SM,
    FONT_UI,
    FONT_UI_SEMIBOLD,
    FONT_UI_XS,
    MUTED,
    ORANGE,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    SUCCESS,
    TEXT,
    TEXT_SECONDARY,
    WARNING,
    create_tooltip,
    bind_responsive_wrap,
    format_labelframe_title,
)
from stablewalk.ui.scientific_labels import (
    DISCLAIMER_RESEARCH,
    LABEL_ANALYSIS_CONFIDENCE,
    LABEL_BIOMECHANICAL_VALIDITY,
    LABEL_CADENCE,
    LABEL_COM_FULL,
    LABEL_CONTACT_COVERAGE,
    LABEL_CONTACT_COVERAGE_LEFT,
    LABEL_CONTACT_COVERAGE_RIGHT,
    LABEL_CONTACT_NOT_FORCE,
    LABEL_CYCLE_EVIDENCE,
    LABEL_GAIT_QUALITY,
    LABEL_GAIT_SYMMETRY,
    LABEL_JOINT_ROM_SUMMARY,
    LABEL_POSE_COVERAGE,
    LABEL_SPECTRAL_RELIABILITY,
    LABEL_STABILITY_MARGIN,
    LABEL_VIRTUAL_GRF_PANEL,
    LABEL_WALKING_SPEED,
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
    ("pose_coverage", LABEL_POSE_COVERAGE),
    ("contact_coverage", LABEL_CONTACT_COVERAGE),
    ("spectral_reliability", LABEL_SPECTRAL_RELIABILITY),
    ("biomechanical_validity", LABEL_BIOMECHANICAL_VALIDITY),
)
_CONTACT_EVIDENCE_METRICS: tuple[tuple[str, str], ...] = (
    ("contact_coverage_left", LABEL_CONTACT_COVERAGE_LEFT),
    ("contact_coverage_right", LABEL_CONTACT_COVERAGE_RIGHT),
    ("cycle_evidence", LABEL_CYCLE_EVIDENCE),
)

_GAIT_EVENT_KEYS: tuple[tuple[str, str], ...] = (
    ("heel_strike", "Heel Strike (est.)"),
    ("toe_off", "Toe Off (est.)"),
    ("double_support", "Double Support (est.)"),
    ("single_support", "Single Support (est., ipsilateral)"),
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
    vsb.grid_remove()
    canvas.configure(yscrollcommand=vsb.set, yscrollincrement=1)

    scroll_content = tk.Frame(canvas, bg=PANEL, highlightthickness=0)
    inner_id = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
    scroll_content.columnconfigure(0, weight=1)

    needs_scroll = {"on": False}
    sync_state = {"active": False}

    def _sync(_event: object | None = None) -> None:
        if sync_state["active"]:
            return
        sync_state["active"] = True
        try:
            from stablewalk.ui.tk.scroll_shell import apply_overflow_scroll_geometry

            canvas.update_idletasks()
            req_h = max(int(scroll_content.winfo_reqheight() or 0), 1)
            needs_scroll["on"] = apply_overflow_scroll_geometry(
                canvas, inner_id, vsb, content_height=req_h
            )
        except tk.TclError:
            pass
        finally:
            sync_state["active"] = False

    def _scroll_cmd(*args: object) -> None:
        canvas.yview(*args)

    vsb.configure(command=_scroll_cmd)
    scroll_content.bind("<Configure>", _sync, add="+")
    canvas.bind("<Configure>", _sync, add="+")

    def _wheel(event: tk.Event) -> str | None:
        if not needs_scroll["on"]:
            return None
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"
        return None

    for widget in (canvas, scroll_content):
        widget.bind("<MouseWheel>", _wheel, add="+")

    return scroll_content


def _section_header(parent: tk.Misc, *, row: int, title: str) -> None:
    """Shared SectionHeader + SoftSeparator (design-system chrome)."""
    from stablewalk.ui.design_system.components import SectionHeader, SoftSeparator

    block = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    block.grid(row=row, column=0, sticky="ew", pady=(PAD_MD, PAD_SM))
    block.columnconfigure(0, weight=1)
    header = SectionHeader(block, title, bg=PANEL)
    header.frame.grid(row=0, column=0, sticky="ew")
    SoftSeparator(block).grid(row=1, column=0, sticky="ew", pady=(PAD_XS, 0))

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
    tier: str = "secondary",
) -> None:
    """One elevated KPI metric card with title, large value, unit, trend, quality."""
    from stablewalk.ui.tk.kpi_cards import create_kpi_grid_card, set_kpi_tooltip

    kpi = create_kpi_grid_card(
        parent,
        key=key,
        title=title,
        compact=multiline or tier != "primary",
        show_bar=key
        in (
            "gait_quality",
            "symmetry",
            "cadence",
            "walking_speed",
            "video_quality",
            "tracking_confidence",
        )
        and tier != "supporting",
        tier=tier,  # type: ignore[arg-type]
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
        tier="technical",
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


# Report preview rail beside findings + processing details.
# Hidden until a session report exists. Floor keeps the preview readable
# without pinning it to a narrow configured widget width.
RESULTS_REPORT_PREVIEW_WIDTH = 380
RESULTS_PREVIEW_RAIL_MIN = 320
RESULTS_PREVIEW_RAIL_SHARE = 0.32
RESULTS_PREVIEW_STACK_BELOW_W = 840
RESULTS_PROCESSING_DETAILS_TITLE = "Processing Details"


def results_preview_rail_floor(width: int) -> int:
    """Readable Report Preview rail floor for the current content width."""
    share = int(max(0, int(width or 0)) * RESULTS_PREVIEW_RAIL_SHARE)
    return max(
        RESULTS_PREVIEW_RAIL_MIN,
        min(RESULTS_REPORT_PREVIEW_WIDTH, share),
    )


from stablewalk.ui.tk.results_warning_copy import (
    classify_results_warning,
    format_primary_action_warning,
    format_results_warning_copy,
    warning_needs_user_action as _warning_needs_user_action,
)


def results_processing_details_should_expand(
    headline_kind: str,
    *,
    attention_warnings: object = (),
    required_stage_failed: bool = False,
) -> bool:
    """Expand Processing Details only when the examiner needs the pipeline.

    Collapsed after a clean required-stage completion. Expanded when a
    required stage failed, analysis is incomplete / in progress, or a
    blocking warning needs action. Waiting sessions stay collapsed so
    technical processing remains secondary.
    """
    if required_stage_failed:
        return True
    kind = str(headline_kind or "").strip().lower()
    if kind in {"incomplete", "processing", "partial"}:
        return True
    if kind in {"complete", "waiting", ""}:
        return any(
            _warning_needs_user_action(item) for item in (attention_warnings or ())
        )
    return True


def _short_gait_summary_text(interpretation: str, *, max_chars: int = 280) -> str:
    """First 1–2 sentences of the existing interpretation — display only."""
    import re

    raw = " ".join(str(interpretation or "").split()).strip()
    if not raw:
        return ""
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", raw)
        if part.strip()
    ]
    short = " ".join(parts[:2]).strip() if parts else raw
    if len(short) > max_chars:
        return short[: max_chars - 1].rstrip() + "…"
    return short


def _required_stage_failed(views: object) -> bool:
    from stablewalk.ui.tk.dashboard_results_pipeline import (
        OPTIONAL_RESULTS_STAGE_KEYS,
        STATUS_FAILED,
    )

    return any(
        getattr(view, "key", "") not in OPTIONAL_RESULTS_STAGE_KEYS
        and str(getattr(view, "status", "") or "") == STATUS_FAILED
        for view in (views or ())
    )


def _required_attention_warnings(views: object) -> list[str]:
    """Action-needed required-stage warnings — not technical processing notes."""
    from stablewalk.ui.tk.dashboard_results_pipeline import (
        OPTIONAL_RESULTS_STAGE_KEYS,
        STATUS_FAILED,
        _stage_warning_preview,
    )

    warnings: list[str] = []
    seen: set[str] = set()
    for view in views or ():
        if getattr(view, "key", "") in OPTIONAL_RESULTS_STAGE_KEYS:
            continue
        status = str(getattr(view, "status", "") or "")
        preview = _stage_warning_preview(view)
        label = str(getattr(view, "label", "") or "Stage").strip()
        source = preview if preview and preview != "None" else ""
        if classify_results_warning(source, status=status) != "critical":
            continue
        text = format_results_warning_copy(
            source, label=label, status=status, kind="critical"
        )
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        warnings.append(text)
    return warnings


def apply_results_processing_details_expansion(
    gui,
    *,
    headline_kind: str | None = None,
    attention_warnings: object = None,
    required_stage_failed: bool | None = None,
) -> bool:
    """Collapse Processing Details on clean completion; expand when attention is needed."""
    section = getattr(gui, "_results_processing_section", None)
    setter = getattr(section, "set_expanded", None)
    kind = (
        str(headline_kind)
        if headline_kind is not None
        else str(getattr(gui, "_results_pipeline_headline_kind", "") or "")
    )
    warnings = (
        attention_warnings
        if attention_warnings is not None
        else getattr(gui, "_results_attention_warnings", ())
    )
    failed = (
        bool(required_stage_failed)
        if required_stage_failed is not None
        else bool(getattr(gui, "_results_required_stage_failed", False))
    )
    should_expand = results_processing_details_should_expand(
        kind,
        attention_warnings=warnings,
        required_stage_failed=failed,
    )
    if callable(setter):
        try:
            setter(should_expand)
        except tk.TclError:
            pass
    _sync_results_processing_stretch(gui, expanded=should_expand)
    return should_expand


def _sync_results_processing_stretch(gui, *, expanded: bool | None = None) -> None:
    """Collapsed Processing Details keep natural height; leftover goes to preview/findings."""
    details = getattr(gui, "_results_processing_section", None)
    parent_pipe = getattr(gui, "_results_pipeline_slot", None)
    content = getattr(gui, "_results_content_row", None)
    preview_visible = bool(getattr(gui, "_results_preview_visible", False))
    if expanded is None:
        expanded = bool(getattr(details, "expanded", False)) if details is not None else False
    preview_layout = str(getattr(gui, "_results_preview_layout", "") or "")
    preview_stacked = preview_layout == "stacked"
    try:
        if parent_pipe is not None:
            parent_pipe.rowconfigure(0, weight=1 if expanded else 0, minsize=0)
            parent_pipe.grid_configure(sticky="nsew" if expanded else "ew")
        if details is not None:
            details.frame.grid_configure(sticky="nsew" if expanded else "ew")
            details.frame.rowconfigure(1, weight=1 if expanded else 0)
        if content is not None:
            # Findings stay compact when Processing Details or Report Preview
            # own leftover height. Collapsed Processing Details must not keep
            # a weight-1 void beside or above the preview.
            if preview_visible:
                content.rowconfigure(0, weight=0, minsize=0)
                if expanded and preview_stacked:
                    content.rowconfigure(1, weight=2, minsize=0)
                    content.rowconfigure(2, weight=1, minsize=120)
                elif expanded:
                    content.rowconfigure(1, weight=1, minsize=0)
                else:
                    content.rowconfigure(1, weight=0, minsize=0)
                    if preview_stacked:
                        content.rowconfigure(2, weight=1, minsize=120)
            else:
                content.rowconfigure(0, weight=0 if expanded else 1, minsize=0)
                content.rowconfigure(1, weight=1 if expanded else 0, minsize=0)
                content.rowconfigure(2, weight=0, minsize=0)
    except tk.TclError:
        pass


def _primary_status_presentation(headline_kind: str, headline: str) -> tuple[str, str]:
    """Examiner-facing status line. Does not change pipeline headline math."""
    kind = str(headline_kind or "").strip().lower()
    if kind == "complete":
        return "Analysis Complete", "ok"
    if kind == "processing":
        return "Processing", "warn"
    if kind == "waiting":
        return "Waiting for analysis", "neutral"
    if kind in ("incomplete", "partial"):
        return "Analysis incomplete", "fail" if kind == "incomplete" else "warn"
    text = str(headline or "").strip() or "Waiting for analysis"
    return text, "neutral"


def sync_results_primary_findings(gui, views: object = None) -> None:
    """Refresh the findings-first Results header from existing session data."""
    from stablewalk.ui.design_system.components import update_status_badge

    status_lbl = getattr(gui, "lbl_results_primary_status", None)
    if status_lbl is None and getattr(gui, "_results_primary_status_badge", None) is None:
        return

    if views is None:
        views = getattr(gui, "_results_pipeline_views", None)
    warnings = _required_attention_warnings(views)
    gui._results_attention_warnings = tuple(warnings)
    gui._results_required_stage_failed = _required_stage_failed(views)

    headline = str(getattr(gui, "_results_pipeline_headline", "") or "")
    kind = str(getattr(gui, "_results_pipeline_headline_kind", "") or "")
    status_text, tone = _primary_status_presentation(kind, headline)

    badge = getattr(gui, "_results_primary_status_badge", None)
    if badge is not None:
        update_status_badge(badge, status_text, tone=tone)  # type: ignore[arg-type]
    elif status_lbl is not None:
        fg = {
            "ok": SUCCESS,
            "warn": ORANGE,
            "fail": DANGER,
            "neutral": MUTED,
        }.get(tone, TEXT)
        try:
            status_lbl.configure(text=status_text, fg=fg)
        except tk.TclError:
            pass

    summary = getattr(gui, "_analysis_summary_cache", None)
    conf_lbl = getattr(gui, "lbl_results_primary_confidence", None)
    if conf_lbl is not None:
        validity = getattr(summary, "result_validity", None) if summary is not None else None
        if validity is not None:
            conf_text = "\n".join(validity.headline_lines())
            worst = str(validity.worst_status())
            conf_fg = (
                WARNING
                if worst in ("limited", "insufficient")
                else MUTED
                if worst == "unavailable"
                else TEXT
            )
        else:
            field = getattr(summary, "analysis_confidence", None) if summary is not None else None
            value = str(getattr(field, "value", "") or "").strip() if field is not None else ""
            available = bool(getattr(field, "available", False)) if field is not None else False
            reason = str(getattr(field, "reason", "") or "").strip() if field is not None else ""
            if available and value:
                conf_text = f"{LABEL_ANALYSIS_CONFIDENCE}: {value}"
                lowered = value.lower()
                conf_fg = (
                    WARNING
                    if any(token in lowered for token in ("low", "insufficient"))
                    else TEXT
                )
            elif field is not None:
                conf_text = f"{LABEL_ANALYSIS_CONFIDENCE}: Unavailable"
                if reason:
                    conf_text = f"{conf_text} — {reason}"
                conf_fg = MUTED
            else:
                conf_text = "Pose, contact, spectral, and biomechanical validity: waiting for analysis"
                conf_fg = MUTED
        try:
            conf_lbl.configure(text=conf_text, fg=conf_fg)
        except tk.TclError:
            pass

    gait_lbl = getattr(gui, "lbl_results_primary_summary", None)
    if gait_lbl is not None:
        interpretation = str(
            getattr(summary, "scientific_interpretation", "") or ""
        ).strip() if summary is not None else ""
        short = _short_gait_summary_text(interpretation)
        if short:
            gait_text, gait_fg = short, TEXT
        elif getattr(summary, "source", "") if summary is not None else "":
            gait_text, gait_fg = "No gait summary is available for this session.", MUTED
        else:
            gait_text, gait_fg = (
                "Run gait analysis to see a short summary of what StableWalk found.",
                MUTED,
            )
        try:
            gait_lbl.configure(text=gait_text, fg=gait_fg)
        except tk.TclError:
            pass

    warn_lbl = getattr(gui, "lbl_results_primary_warnings", None)
    if warn_lbl is not None:
        if warnings:
            shown = [format_primary_action_warning(item) for item in warnings[:2]]
            extra = len(warnings) - min(2, len(warnings))
            if extra > 0:
                shown.append(f"+{extra} more in Processing Details")
            try:
                warn_lbl.configure(text="\n".join(shown), fg=DANGER)
                warn_lbl.grid()
            except tk.TclError:
                pass
        else:
            try:
                warn_lbl.configure(text="")
                warn_lbl.grid_remove()
            except tk.TclError:
                pass

    apply_results_processing_details_expansion(
        gui,
        headline_kind=kind,
        attention_warnings=warnings,
        required_stage_failed=bool(getattr(gui, "_results_required_stage_failed", False)),
    )


def _build_results_primary_findings(gui, parent: tk.Misc) -> None:
    """Findings-first header: what StableWalk found, not how it was processed."""
    from stablewalk.ui.design_system.components import QuietPanel, StatusBadge

    host = QuietPanel(parent, bg=ELEVATED)
    host.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, PAD_SM))
    host.columnconfigure(0, weight=1)
    gui._results_primary_host = host

    inner = tk.Frame(host, bg=ELEVATED, highlightthickness=0)
    inner.grid(row=0, column=0, sticky="ew", padx=PAD_MD, pady=PAD_SM)
    inner.columnconfigure(0, weight=1)
    gui._results_primary_inner = inner

    title_row = tk.Frame(inner, bg=ELEVATED, highlightthickness=0)
    title_row.grid(row=0, column=0, sticky="ew")
    title_row.columnconfigure(0, weight=1)

    gui._results_primary_status_badge = StatusBadge(
        title_row, "Waiting for analysis", tone="neutral", bg_parent=ELEVATED
    )
    gui._results_primary_status_badge.frame.grid(row=0, column=0, sticky="w")
    gui.lbl_results_primary_status = gui._results_primary_status_badge.label

    gui.lbl_results_primary_summary = tk.Label(
        inner,
        text="Run gait analysis to see a short summary of what StableWalk found.",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI,
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
    )
    gui.lbl_results_primary_summary.grid(row=1, column=0, sticky="ew", pady=(PAD_XS, 0))

    gui.lbl_results_primary_confidence = tk.Label(
        inner,
        text=(
            "Pose coverage, contact coverage, spectral reliability, and "
            "biomechanical validity: waiting for analysis"
        ),
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SEMIBOLD,
        anchor="w",
        justify=tk.LEFT,
    )
    gui.lbl_results_primary_confidence.grid(row=2, column=0, sticky="ew", pady=(PAD_XS, 0))

    gui.lbl_results_primary_warnings = tk.Label(
        inner,
        text="",
        bg=ELEVATED,
        fg=DANGER,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
    )
    gui.lbl_results_primary_warnings.grid(row=3, column=0, sticky="ew", pady=(PAD_XS, 0))
    gui.lbl_results_primary_warnings.grid_remove()


def _install_results_content_host(gui, parent: tk.Misc) -> ttk.Frame:
    """Expanding Results workspace — no page-level canvas scroll.

    Header buttons and footer status stay on the tab. Report Preview scrolls
    internally only when its text overflows, so the page never nests two
    vertical scrollbars around the pipeline.
    """
    content = ttk.Frame(parent)
    content.grid(row=1, column=0, sticky="nsew", pady=(0, PAD_SM))
    content.columnconfigure(0, weight=1)
    content.rowconfigure(0, weight=0)
    content.rowconfigure(1, weight=1)
    gui._results_scroll_canvas = None
    gui._results_scrollbar = None
    gui._results_needs_scroll = False
    gui._sync_results_scroll = lambda *_args: None
    return content


def build_results_summary_tab(gui, parent: ttk.Frame) -> None:
    """Install Results tab shell; heavy report widgets load on first open."""
    from stablewalk.ui.tk.dashboard_notebook import TAB_ADVANCED, select_dashboard_tab
    from stablewalk.ui.tk.dashboard_page_frame import install_page_header

    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=0, minsize=36)
    parent.rowconfigure(1, weight=1, minsize=200)
    parent.rowconfigure(2, weight=0, minsize=28)

    # Header spans the full page width; export actions sit in the title row.
    header = install_page_header(parent, "results", actions=True, row=0)
    btn_row = getattr(header, "actions_host", None)
    if btn_row is None:
        btn_row = tk.Frame(header, bg=PANEL, highlightthickness=0)
        btn_row.grid(row=0, column=1, sticky="e")
    gui._results_summary_btn_row = btn_row
    # One consistent toolbar: primary Export JSON, secondary report + Advanced.
    gui.btn_export_summary_json = ttk.Button(
        btn_row,
        text="Export JSON",
        style="ExportAccent.TButton",
        command=lambda: gui._export_analysis_summary_json(),
        state=tk.DISABLED,
    )
    gui.btn_export_summary_json.grid(row=0, column=0, padx=(0, 4))
    gui.btn_export_summary_report = ttk.Button(
        btn_row,
        text="Export Report",
        style="Export.TButton",
        command=lambda: gui._export_analysis_summary_report(),
        state=tk.DISABLED,
    )
    gui.btn_export_summary_report.grid(row=0, column=1, padx=(0, 4))
    create_tooltip(gui.btn_export_summary_json, "Export the current summary as JSON")
    create_tooltip(gui.btn_export_summary_report, "Export a human-readable text report")
    gui.btn_download_research_report = ttk.Button(
        btn_row,
        text="Download Research Report",
        style="Export.TButton",
        command=lambda: gui._download_research_report()
        if callable(getattr(gui, "_download_research_report", None))
        else None,
        state=tk.DISABLED,
    )
    gui.btn_download_research_report.grid(row=0, column=2, padx=(0, 4))
    create_tooltip(
        gui.btn_download_research_report,
        "Download computed research-validation results, definitions, plots, "
        "video metadata, and failure reasons. Uncomputed items are not marked as passed.",
        wraplength=360,
    )
    gui.btn_results_open_advanced = ttk.Button(
        btn_row,
        text="Open Advanced",
        style="Export.TButton",
        command=lambda: select_dashboard_tab(gui, TAB_ADVANCED),
    )
    gui.btn_results_open_advanced.grid(row=0, column=3)
    create_tooltip(
        gui.btn_results_open_advanced,
        "Open Advanced for session exports, OpenSim, and Real-to-Sim packages",
    )

    # Findings on row 0; pipeline + optional report preview share row 1.
    content = _install_results_content_host(gui, parent)
    content.columnconfigure(0, weight=1)
    content.columnconfigure(1, weight=0, minsize=0)
    content.rowconfigure(0, weight=0)
    content.rowconfigure(1, weight=1)
    gui._results_content_row = content

    gui._results_pipeline_slot = ttk.Frame(content)
    gui._results_pipeline_slot.grid(row=1, column=0, sticky="nsew")
    gui._results_pipeline_slot.columnconfigure(0, weight=1)
    gui._results_pipeline_slot.rowconfigure(0, weight=1)

    # The preview starts collapsed.  Its responsive rail/stack geometry is
    # applied only when a report is available.
    preview_host = ttk.Frame(content)
    preview_host.columnconfigure(0, weight=1)
    preview_host.rowconfigure(0, weight=1)
    gui._results_preview_host = preview_host
    gui._results_preview_visible = False

    gui._results_report_slot = ttk.Frame(parent)
    gui._results_report_slot.grid(row=2, column=0, sticky="ew")
    gui._results_report_slot.columnconfigure(0, weight=1)
    gui._results_report_slot.rowconfigure(0, weight=0)

    gui._summary_field_labels = {}
    gui._summary_tier_labels = {}
    gui._summary_kpi_cards = {}
    gui._summary_event_labels = {}
    gui._results_tab_content_built = False

    def _ensure_results_tab_content() -> bool:
        if getattr(gui, "_results_tab_content_built", False):
            return True
        _build_results_tab_body(gui)
        gui._results_tab_content_built = True
        # Seed the visual pipeline immediately so Results is never a blank panel.
        try:
            from stablewalk.ui.tk.dashboard_results_pipeline import (
                update_results_analysis_pipeline,
            )

            update_results_analysis_pipeline(gui, None)
        except Exception:
            pass
        # Prefer cache after body build — empty sessions stay collapsed.
        sync_results_report_preview(gui)
        return True

    gui._ensure_results_tab_content = _ensure_results_tab_content
    # Build on install — avoid a large empty Results area until first tab click.
    _ensure_results_tab_content()


def _build_results_tab_body(gui) -> None:
    """Create findings header, collapsible pipeline, and technical sections (once)."""
    from stablewalk.ui.design_system.components import CollapsibleSection
    from stablewalk.ui.tk.dashboard_results_pipeline import build_results_analysis_pipeline
    from stablewalk.ui.tk.kpi_cards import create_technical_details_panel

    parent_pipe = getattr(gui, "_results_pipeline_slot", None)
    parent_report = getattr(gui, "_results_report_slot", None)
    preview_host = getattr(gui, "_results_preview_host", None)
    content = getattr(gui, "_results_content_row", None)
    if parent_pipe is None or parent_report is None:
        return

    if content is not None:
        _build_results_primary_findings(gui, content)
        findings = getattr(gui, "_results_primary_inner", None) or getattr(
            gui, "_results_primary_host", None
        )
        if findings is not None:
            try:
                from stablewalk.ui.tk.dashboard_research_validation import (
                    build_research_validation_section,
                )

                build_research_validation_section(gui, findings)
            except Exception:
                pass

    def _on_processing_toggle(expanded: bool) -> None:
        _sync_results_processing_stretch(gui, expanded=expanded)

    # Technical pipeline lives one level down from findings.
    details = CollapsibleSection(
        parent_pipe,
        RESULTS_PROCESSING_DETAILS_TITLE,
        expanded=False,
        bg=PANEL,
        on_toggle=_on_processing_toggle,
    )
    details.frame.grid(row=0, column=0, sticky="ew")
    details.frame.columnconfigure(0, weight=1)
    details.body.columnconfigure(0, weight=1)
    details.body.rowconfigure(0, weight=1)
    gui._results_processing_section = details
    parent_pipe.columnconfigure(0, weight=1)
    parent_pipe.rowconfigure(0, weight=0)
    _sync_results_processing_stretch(gui, expanded=False)

    pipeline_root = build_results_analysis_pipeline(gui, details.body)
    pipeline_root.grid(row=0, column=0, sticky="nsew")
    gui._section_pipeline_status_summary = pipeline_root

    if preview_host is not None:
        _build_results_report_preview(gui, preview_host)

    footer = tk.Frame(parent_report, bg=PANEL, highlightthickness=0)
    footer.grid(row=0, column=0, sticky="ew")
    footer.columnconfigure(0, weight=1)

    session_row = tk.Frame(footer, bg=PANEL, highlightthickness=0)
    session_row.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
    session_row.columnconfigure(0, weight=1)

    gui.lbl_summary_source = tk.Label(
        session_row,
        text="No video analyzed",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
    )
    gui.lbl_summary_source.grid(row=0, column=0, sticky="ew")

    # Playhead sync is shown on the global status bar — keep a hidden label
    # so refresh paths can still update a tooltip on the source line.
    gui.lbl_summary_playhead = tk.Label(
        session_row,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_summary_playhead.grid(row=1, column=0, sticky="ew")
    gui.lbl_summary_playhead.grid_remove()

    gui._summary_field_labels = {}
    gui._summary_tier_labels = {}
    gui._summary_kpi_cards = {}
    gui._summary_event_labels = {}

    # Per-metric validity — secondary expandable detail (collapsed).
    confidence_collapse = CollapsibleSection(
        footer, "Result validity", expanded=False, bg=PANEL
    )
    confidence_collapse.frame.grid(row=1, column=0, sticky="ew", pady=(PAD_XS, 0))
    quality_grid = confidence_collapse.body
    gui._results_quality_grid = quality_grid
    for col in range(4):
        quality_grid.columnconfigure(col, weight=1)
    for index, (key, title) in enumerate(_CONFIDENCE_METRICS):
        _build_metric_card(
            quality_grid,
            row=0,
            column=index,
            key=key,
            title=title,
            gui=gui,
            tier="secondary",
        )
    for index, (key, title) in enumerate(_CONTACT_EVIDENCE_METRICS):
        _build_metric_card(
            quality_grid,
            row=1,
            column=index,
            key=key,
            title=title,
            gui=gui,
            tier="secondary",
        )

    # Long technical content — collapsed by default.
    details = create_technical_details_panel(
        footer,
        title="Technical Details",
        bg=PANEL,
        start_expanded=False,
        pack=False,
    )
    details.frame.grid(row=2, column=0, sticky="ew", pady=(PAD_SM, 0))
    gui._results_technical_details = details
    tech = details.content
    tech.columnconfigure(0, weight=1)

    tech_row = 0
    _section_header(tech, row=tech_row, title="Detected Events (estimated timing)")
    tech_row += 1

    events_grid = tk.Frame(tech, bg=PANEL, highlightthickness=0)
    gui._results_events_grid = events_grid
    events_grid.grid(row=tech_row, column=0, sticky="ew", pady=(0, 2))
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
    tech_row += 1

    disclaimer = tk.Label(
        tech,
        text=f"{LABEL_CONTACT_NOT_FORCE}  {DISCLAIMER_RESEARCH}",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=720,
        justify=tk.LEFT,
    )
    disclaimer.grid(row=tech_row, column=0, sticky="ew", pady=(0, PAD_SM))
    gui._results_disclaimer_label = disclaimer
    tech_row += 1

    _section_header(tech, row=tech_row, title="Derived outputs")
    tech_row += 1

    gait_grid = tk.Frame(tech, bg=PANEL, highlightthickness=0)
    gait_grid.grid(row=tech_row, column=0, sticky="ew", pady=(0, PAD_SM))
    for col in range(3):
        gait_grid.columnconfigure(col, weight=1)
    _gait_layout = (
        ("gait_quality", LABEL_GAIT_QUALITY, 0, 0, 2, "secondary"),
        ("walking_speed", LABEL_WALKING_SPEED, 0, 2, 1, "technical"),
        ("cadence", LABEL_CADENCE, 1, 0, 1, "technical"),
        ("symmetry", LABEL_GAIT_SYMMETRY, 1, 1, 1, "technical"),
        ("stability_margin", LABEL_STABILITY_MARGIN, 1, 2, 1, "technical"),
    )
    for key, title, r, c, span, tier in _gait_layout:
        _build_metric_card(
            gait_grid,
            row=r,
            column=c,
            key=key,
            title=title,
            gui=gui,
            columnspan=span,
            tier=tier,
        )
    tech_row += 1

    _section_header(tech, row=tech_row, title="Biomechanics artifacts")
    tech_row += 1

    bio_grid = tk.Frame(tech, bg=PANEL, highlightthickness=0)
    bio_grid.grid(row=tech_row, column=0, sticky="ew", pady=(0, PAD_SM))
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
            tier="technical" if key != "joint_rom" else "secondary",
        )

    interp_collapse = CollapsibleSection(
        footer, "Processing Interpretation", expanded=False, bg=PANEL
    )
    interp_collapse.frame.grid(row=3, column=0, sticky="ew", pady=(PAD_SM, 0))
    tk.Label(
        interp_collapse.body,
        text="Auto-generated from estimated parameters — not a medical diagnosis.",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(fill=tk.X, pady=(0, PAD_XS))
    gui.lbl_summary_scientific_interpretation = tk.Label(
        interp_collapse.body,
        text=(
            "Run gait analysis to generate an automatic interpretation of the estimated "
            "biomechanical parameters for this recording."
        ),
        bg=PANEL,
        fg=MUTED,
        font=FONT_SUMMARY_INTERPRETATION,
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
    )
    gui.lbl_summary_scientific_interpretation.pack(fill=tk.X)
    bind_responsive_wrap(
        gui,
        footer,
        ("lbl_summary_scientific_interpretation",),
        margin=24,
    )


def _build_results_report_preview(gui, parent: tk.Misc) -> None:
    """Install the side Report Preview card (shown only when content exists)."""
    card = ttk.LabelFrame(
        parent,
        text=format_labelframe_title("Report Preview"),
        style="Card.TLabelframe",
        padding=DASHBOARD_CARD_PAD,
    )
    card.grid(row=0, column=0, sticky="nsew")
    card.columnconfigure(0, weight=1)
    card.rowconfigure(1, weight=1)
    gui._results_preview_card = card

    gui.lbl_results_preview_source = tk.Label(
        card,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_results_preview_source.grid(row=0, column=0, sticky="ew", pady=(0, PAD_SM))

    body = tk.Frame(card, bg=PANEL, highlightthickness=0)
    body.grid(row=1, column=0, sticky="nsew")
    body.columnconfigure(0, weight=1)
    body.rowconfigure(0, weight=1)

    gui.lbl_results_preview_body = tk.Text(
        body,
        bg=PANEL,
        fg=TEXT,
        font=FONT_SUMMARY_INTERPRETATION,
        height=1,
        width=1,
        wrap=tk.WORD,
        relief=tk.FLAT,
        borderwidth=0,
        highlightthickness=0,
        padx=2,
        pady=2,
        cursor="arrow",
    )
    gui.lbl_results_preview_body.grid(row=0, column=0, sticky="nsew")
    preview_scroll = ttk.Scrollbar(
        body,
        orient=tk.VERTICAL,
        command=gui.lbl_results_preview_body.yview,
    )
    gui._results_preview_scrollbar = preview_scroll
    gui._results_preview_needs_scroll = False
    gui.lbl_results_preview_body.tag_configure(
        "heading",
        font=FONT_UI_SEMIBOLD,
        foreground=TEXT,
        spacing1=14,
        spacing3=6,
    )
    gui.lbl_results_preview_body.tag_configure(
        "paragraph",
        font=FONT_SUMMARY_INTERPRETATION,
        foreground=TEXT_SECONDARY,
        spacing1=2,
        spacing3=10,
        lmargin1=0,
        lmargin2=0,
    )
    gui.lbl_results_preview_body.configure(state=tk.DISABLED)
    preview_sync_state = {"active": False}

    def _preview_yscroll(first: str, last: str) -> None:
        try:
            preview_scroll.set(first, last)
        except tk.TclError:
            return
        try:
            start = float(first)
            end = float(last)
            # Unmapped / zero-size Text reports (0, 0) — that is not overflow.
            overflow = 0.0 <= start < end < 0.999
        except (TypeError, ValueError):
            overflow = False
        gui._results_preview_needs_scroll = overflow
        try:
            if overflow:
                preview_scroll.grid(row=0, column=1, sticky="ns")
            else:
                preview_scroll.grid_remove()
                gui.lbl_results_preview_body.yview_moveto(0)
        except tk.TclError:
            pass

    def _fit_preview_text_height() -> None:
        """Match Text ``height`` to the allocated pixels so yview is truthful.

        A grid-stretched Text with height=1 still reports a one-line viewport,
        which falsely enables the scrollbar and clips the report body.
        """
        widget = gui.lbl_results_preview_body
        try:
            px = max(1, int(widget.winfo_height() or 1))
        except tk.TclError:
            return
        if px < 8:
            return
        try:
            line_px = int(
                widget.tk.call("font", "metrics", widget.cget("font"), "-linespace")
            )
        except tk.TclError:
            line_px = 16
        line_px = max(12, line_px)
        lines = max(1, px // line_px)
        try:
            current = int(str(widget.cget("height") or 1).split(".")[0] or 1)
        except (tk.TclError, TypeError, ValueError):
            current = 1
        if current == lines:
            return
        try:
            state = str(widget.cget("state") or tk.NORMAL)
            widget.configure(state=tk.NORMAL)
            widget.configure(height=lines)
            if state == tk.DISABLED:
                widget.configure(state=tk.DISABLED)
        except tk.TclError:
            pass

    gui.lbl_results_preview_body.configure(yscrollcommand=_preview_yscroll)

    def _sync_preview_scroll(_event: object | None = None) -> None:
        if preview_sync_state["active"]:
            return
        preview_sync_state["active"] = True
        try:
            width = max(1, int(body.winfo_width() or 0))
            gui.lbl_results_preview_source.configure(wraplength=max(120, width - 12))
            _fit_preview_text_height()
            gui.lbl_results_preview_body.update_idletasks()
            first, last = gui.lbl_results_preview_body.yview()
            _preview_yscroll(str(first), str(last))
        except (tk.TclError, TypeError, ValueError):
            return
        finally:
            preview_sync_state["active"] = False

    def _scroll_preview(event: tk.Event) -> str | None:
        if not bool(getattr(gui, "_results_preview_needs_scroll", False)):
            return None
        if getattr(event, "delta", 0):
            gui.lbl_results_preview_body.yview_scroll(
                int(-event.delta / 120), "units"
            )
            return "break"
        return None

    gui._sync_results_preview_scroll = _sync_preview_scroll
    body.bind("<Configure>", _sync_preview_scroll, add="+")
    gui.lbl_results_preview_body.bind("<Configure>", _sync_preview_scroll, add="+")
    gui.lbl_results_preview_body.bind("<MouseWheel>", _scroll_preview, add="+")
    try:
        body.after_idle(_sync_preview_scroll)
    except tk.TclError:
        pass


def _results_preview_sections(
    summary,
    *,
    session_state: str = "",
    metrics_status: str = "",
) -> list[tuple[str, list[str]]]:
    """Structured report sections using only values already in the summary."""
    if summary is None:
        return []
    # No analyzed recording → keep the preview rail collapsed so the pipeline
    # can fill the full horizontal content area.
    source = str(getattr(summary, "source", "") or "").strip()
    if not source:
        return []

    def _sentences(text: str) -> list[str]:
        import re

        raw = " ".join(str(text or "").split())
        if not raw:
            return []
        return [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", raw)
            if part.strip()
        ]

    interim = session_state not in ("", "complete")
    metrics_not_run = metrics_status == "unavailable"

    def _field_lines(fields: tuple[tuple[str, str], ...]) -> list[str]:
        lines: list[str] = []
        for attr, fallback_label in fields:
            field = getattr(summary, attr, None)
            if field is None:
                continue
            label = str(getattr(field, "label", "") or fallback_label).strip()
            available = bool(getattr(field, "available", False))
            value = str(getattr(field, "value", "") or "").strip()
            reason = str(getattr(field, "reason", "") or "").strip()
            if available and value:
                prefix = "Interim derived — " if interim else ""
                lines.append(f"{prefix}{label}: {value}")
            else:
                unavailable = f"{label}: Unavailable"
                lines.append(f"{unavailable} — {reason}" if reason else unavailable)
        return lines

    interpretation = str(
        getattr(summary, "scientific_interpretation", "") or ""
    ).strip()
    interpretation_sentences = _sentences(interpretation)

    if interim:
        if metrics_not_run:
            preview_notice = (
                "Interim derived values only. The Metrics pipeline stage has not run; "
                "this preview is not a completed report."
            )
        else:
            preview_notice = (
                "Interim derived values only. Required pipeline stages are not yet "
                "complete; this preview is a partial report."
            )
    else:
        preview_notice = (
            "Values below are derived or estimated from available analysis data."
        )

    sections: list[tuple[str, list[str]]] = [
        (
            "Summary",
            [preview_notice] + interpretation_sentences,
        ),
        (
            "Gait Quality",
            _field_lines(
                (
                    ("overall_gait_quality", "Gait quality"),
                    ("cadence", "Cadence"),
                    ("walking_speed", "Walking speed"),
                )
            ),
        ),
        (
            "Stability",
            _field_lines((("stability_margin", "Stability"),)),
        ),
        (
            "Symmetry",
            _field_lines((("symmetry", "Symmetry"),)),
        ),
        (
            "Biomechanics",
            _field_lines(
                (
                    ("center_of_mass", "Center of mass"),
                    ("estimated_virtual_grf", "Estimated virtual GRF"),
                    ("joint_rom_summary", "Joint ROM"),
                )
            ),
        ),
        (
            "Confidence",
            _field_lines(
                (
                    ("analysis_confidence", "Analysis confidence"),
                    ("tracking_confidence", "Tracking confidence"),
                    ("pipeline_confidence", "Pipeline confidence"),
                    ("video_quality", "Video quality"),
                    ("contact_coverage_left", "Left contact coverage"),
                    ("contact_coverage_right", "Right contact coverage"),
                    ("cycle_evidence", "Cycle evidence"),
                )
            ),
        ),
    ]

    limitation_lines: list[str] = []
    limitation_tokens = (
        "limitation",
        "monocular",
        "not a diagnosis",
        "not clinical",
        "unavailable",
        "insufficient",
    )
    for sentence in interpretation_sentences:
        if any(token in sentence.lower() for token in limitation_tokens):
            limitation_lines.append(sentence)
    for attr, _label in (
        ("overall_gait_quality", "Gait quality"),
        ("stability_margin", "Stability"),
        ("symmetry", "Symmetry"),
        ("center_of_mass", "Center of mass"),
        ("estimated_virtual_grf", "Estimated virtual GRF"),
        ("joint_rom_summary", "Joint ROM"),
        ("analysis_confidence", "Analysis confidence"),
        ("contact_coverage_left", "Left contact coverage"),
        ("contact_coverage_right", "Right contact coverage"),
        ("cycle_evidence", "Cycle evidence"),
    ):
        field = getattr(summary, attr, None)
        if field is None:
            continue
        reason = str(getattr(field, "reason", "") or "").strip()
        if not bool(getattr(field, "available", False)) and reason:
            limitation_lines.append(reason)
    limitation_lines = list(dict.fromkeys(limitation_lines))
    sections.append(
        (
            "Limitations",
            limitation_lines
            or ["No additional limitations were provided in the current summary."],
        )
    )

    return [
        (heading, paragraphs or ["Unavailable"])
        for heading, paragraphs in sections
    ]


def _results_preview_text(
    summary,
    *,
    session_state: str = "",
    metrics_status: str = "",
) -> str:
    """Plain-text form of the structured report preview."""
    return "\n\n".join(
        f"{heading}\n" + "\n\n".join(paragraphs)
        for heading, paragraphs in _results_preview_sections(
            summary,
            session_state=session_state,
            metrics_status=metrics_status,
        )
    )


def _populate_results_preview(
    widget: tk.Misc,
    summary: object,
    *,
    session_state: str = "",
    metrics_status: str = "",
) -> None:
    """Render a readable section hierarchy without changing report content."""
    sections = _results_preview_sections(
        summary,
        session_state=session_state,
        metrics_status=metrics_status,
    )
    if isinstance(widget, tk.Text):
        try:
            widget.configure(state=tk.NORMAL)
            widget.delete("1.0", tk.END)
            for section_index, (heading, paragraphs) in enumerate(sections):
                if section_index:
                    widget.insert(tk.END, "\n")
                widget.insert(tk.END, f"{heading}\n", "heading")
                for paragraph in paragraphs:
                    widget.insert(tk.END, f"{paragraph}\n", "paragraph")
            widget.configure(state=tk.DISABLED)
        except tk.TclError:
            pass
        return
    try:
        widget.configure(
            text=_results_preview_text(
                summary,
                session_state=session_state,
                metrics_status=metrics_status,
            )
        )
    except tk.TclError:
        pass


_UNSET = object()


def sync_results_report_preview(gui, summary: object = _UNSET) -> bool:
    """Show or hide the report preview rail; expand the pipeline when hidden.

    Pass ``summary=None`` to force-hide (pipeline fills the row). Omit ``summary``
    to read ``gui._analysis_summary_cache``. Returns True when the preview is visible.
    """
    content = getattr(gui, "_results_content_row", None)
    host = getattr(gui, "_results_preview_host", None)
    if content is None or host is None:
        return False

    if summary is _UNSET:
        summary = getattr(gui, "_analysis_summary_cache", None)
    session_state = str(
        getattr(gui, "_results_pipeline_session_state", "") or ""
    )
    metrics_status = str(
        getattr(gui, "_results_metrics_stage_status", "") or ""
    )
    body_text = _results_preview_text(
        summary,
        session_state=session_state,
        metrics_status=metrics_status,
    )
    source = str(getattr(summary, "source", "") or "").strip() if summary else ""
    visible = bool(body_text)

    body_lbl = getattr(gui, "lbl_results_preview_body", None)
    source_lbl = getattr(gui, "lbl_results_preview_source", None)
    preview_card = getattr(gui, "_results_preview_card", None)
    if metrics_status == "unavailable" and session_state != "complete":
        preview_title = "Report Preview — Interim (Metrics Not Run)"
    elif metrics_status == "failed" and session_state != "complete":
        preview_title = "Report Preview — Interim (Metrics Failed)"
    else:
        preview_title = {
            "partial": "Report Preview — Interim / Partial Results",
            "processing": "Report Preview — Interim (Processing)",
            "incomplete": "Report Preview — Interim / Incomplete Results",
            "waiting": "Report Preview — Interim / Partial Results",
        }.get(session_state, "Report Preview")
    if preview_card is not None:
        try:
            preview_card.configure(text=format_labelframe_title(preview_title))
        except tk.TclError:
            pass
    if body_lbl is not None:
        _populate_results_preview(
            body_lbl,
            summary,
            session_state=session_state,
            metrics_status=metrics_status,
        )
    if source_lbl is not None:
        try:
            source_text = source or "Session report"
            if session_state not in ("", "complete"):
                source_text = f"{source_text} · Interim derived values"
            source_lbl.configure(text=source_text)
        except tk.TclError:
            pass
    sync = getattr(gui, "_sync_results_preview_scroll", None)
    if callable(sync):
        try:
            sync()
        except tk.TclError:
            pass

    was_visible = bool(getattr(gui, "_results_preview_visible", False))
    gui._results_preview_visible = visible
    try:
        if visible:
            reflow_results_layout(gui)
        else:
            # No preview — pipeline fills the full horizontal content area.
            try:
                host.grid_remove()
            except tk.TclError:
                pass
            content.columnconfigure(0, weight=1)
            content.columnconfigure(1, weight=0, minsize=0)
            content.rowconfigure(0, weight=0, minsize=0)
            content.rowconfigure(1, weight=1, minsize=0)
            content.rowconfigure(2, weight=0, minsize=0)
    except tk.TclError:
        return visible

    if visible != was_visible:
        try:
            from stablewalk.ui.tk.dashboard_results_pipeline import (
                _scale_results_pipeline_cards,
            )

            _scale_results_pipeline_cards(gui)
        except Exception:
            pass
    _sync_results_processing_stretch(gui)
    return visible


def reflow_results_layout(gui, *, width: int | None = None) -> str:
    """Reflow Results actions and preview without changing report state."""
    content = getattr(gui, "_results_content_row", None)
    host = getattr(gui, "_results_preview_host", None)
    if content is None or host is None:
        return "hidden"
    if width is None:
        try:
            width = int(content.winfo_width() or 0)
        except tk.TclError:
            width = 0
    else:
        try:
            actual_width = int(content.winfo_width() or 0)
            if actual_width > 40:
                width = actual_width
        except tk.TclError:
            pass
    width = int(width or 0)
    if width < 80:
        try:
            width = int(content.winfo_reqwidth() or 0)
        except tk.TclError:
            width = RESULTS_PREVIEW_STACK_BELOW_W

    # Footer cards use the same available width as the Results page. Reposition
    # existing widgets only; no report or metric content is rebuilt.
    events_grid = getattr(gui, "_results_events_grid", None)
    event_kpis = getattr(gui, "_summary_event_kpis", None) or {}
    if events_grid is not None and event_kpis:
        event_cols = 4 if width >= 1000 else 2
        try:
            for column in range(4):
                events_grid.columnconfigure(
                    column, weight=1 if column < event_cols else 0
                )
            for index, (key, _title) in enumerate(_GAIT_EVENT_KEYS):
                kpi = event_kpis.get(key)
                if kpi is not None:
                    row, column = divmod(index, event_cols)
                    kpi.frame.grid_configure(row=row, column=column, columnspan=1)
                    kpi.quality_lbl.configure(
                        wraplength=max(140, min(220, width // event_cols - 32))
                    )
        except tk.TclError:
            pass

    quality_grid = getattr(gui, "_results_quality_grid", None)
    metric_kpis = getattr(gui, "_summary_kpi_cards", None) or {}
    if quality_grid is not None:
        quality_cols = 4 if width >= 1100 else 2
        try:
            for column in range(4):
                quality_grid.columnconfigure(
                    column, weight=1 if column < quality_cols else 0
                )
            for index, (key, _title) in enumerate(
                _CONFIDENCE_METRICS + _CONTACT_EVIDENCE_METRICS
            ):
                kpi = metric_kpis.get(key)
                if kpi is not None:
                    row, column = divmod(index, quality_cols)
                    kpi.frame.grid_configure(row=row, column=column, columnspan=1)
        except tk.TclError:
            pass

    disclaimer = getattr(gui, "_results_disclaimer_label", None)
    if disclaimer is not None:
        try:
            disclaimer.configure(wraplength=max(220, width - 32))
        except tk.TclError:
            pass

    source_label = getattr(gui, "lbl_summary_source", None)
    if source_label is not None:
        try:
            source_label.configure(
                wraplength=max(220, width - 32),
                justify=tk.LEFT,
            )
        except tk.TclError:
            pass

    primary_wrap = max(280, width - 48)
    for attr in (
        "lbl_results_primary_summary",
        "lbl_results_primary_warnings",
        "lbl_results_primary_confidence",
    ):
        label = getattr(gui, attr, None)
        if label is None:
            continue
        try:
            label.configure(wraplength=primary_wrap)
        except tk.TclError:
            pass

    actions = getattr(gui, "_results_summary_btn_row", None)
    action_buttons = (
        getattr(gui, "btn_export_summary_json", None),
        getattr(gui, "btn_export_summary_report", None),
        getattr(gui, "btn_download_research_report", None),
        getattr(gui, "btn_results_open_advanced", None),
    )
    present = tuple(button for button in action_buttons if button is not None)
    if actions is not None and present:
        compact = width < 720
        try:
            for column in range(4):
                actions.columnconfigure(column, weight=1 if compact else 0)
            last = len(present) - 1
            for index, button in enumerate(present):
                if compact:
                    row, column = divmod(index, 2)
                    button.grid(
                        row=row,
                        column=column,
                        sticky="ew",
                        padx=(0, PAD_XS if column == 0 else 0),
                        pady=(0, PAD_XS if row == 0 else 0),
                    )
                else:
                    button.grid(
                        row=0,
                        column=index,
                        sticky="",
                        padx=(0, 4) if index < last else 0,
                        pady=0,
                    )
        except tk.TclError:
            pass

    if not bool(getattr(gui, "_results_preview_visible", False)):
        try:
            host.grid_remove()
            content.columnconfigure(0, weight=1, minsize=0)
            content.columnconfigure(1, weight=0, minsize=0)
            details = getattr(gui, "_results_processing_section", None)
            processing_open = bool(getattr(details, "expanded", False))
            content.rowconfigure(0, weight=0 if processing_open else 1, minsize=0)
            content.rowconfigure(1, weight=1 if processing_open else 0, minsize=0)
            content.rowconfigure(2, weight=0, minsize=0)
        except tk.TclError:
            pass
        gui._results_preview_layout = "hidden"
        _sync_results_processing_stretch(gui)
        return "hidden"

    details = getattr(gui, "_results_processing_section", None)
    processing_open = (
        True if details is None else bool(getattr(details, "expanded", False))
    )
    # Collapsed Processing Details must not keep a tall empty left column
    # beside Report Preview. Stack the preview so it owns leftover height.
    stacked = width < RESULTS_PREVIEW_STACK_BELOW_W or (
        details is not None and not processing_open
    )
    try:
        content.rowconfigure(0, weight=0, minsize=0)
        if stacked:
            content.columnconfigure(0, weight=1, minsize=0)
            content.columnconfigure(1, weight=0, minsize=0)
            content.rowconfigure(1, weight=2 if processing_open else 0, minsize=0)
            content.rowconfigure(2, weight=1, minsize=120)
            host.grid(
                row=2,
                column=0,
                sticky="nsew",
                padx=0,
                pady=(PAD_SM, 0),
            )
        else:
            # Findings stay full width. Preview shares the remaining row
            # with Processing Details at a readable rail floor.
            content.columnconfigure(0, weight=2, minsize=0)
            rail_floor = results_preview_rail_floor(width)
            content.columnconfigure(
                1, weight=1, minsize=rail_floor
            )
            content.rowconfigure(1, weight=1, minsize=0)
            content.rowconfigure(2, weight=0, minsize=0)
            host.grid(
                row=1,
                column=1,
                sticky="nsew",
                padx=(PAD_MD, 0),
                pady=0,
            )
    except tk.TclError:
        pass
    mode = "stacked" if stacked else "rail"
    gui._results_preview_layout = mode
    _sync_results_processing_stretch(gui)
    sync = getattr(gui, "_sync_results_preview_scroll", None)
    if callable(sync):
        try:
            sync()
        except tk.TclError:
            pass
    return mode


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
    status = format_status_line(level, field.tier, key=key)
    session_state = (
        str(getattr(gui, "_results_pipeline_session_state", "") or "")
        if gui is not None
        else ""
    )
    interim = bool(field.available and session_state not in ("", "complete"))
    if interim:
        status = "Interim derived"
        tip = (
            "Interim derived value — required Results pipeline stages are not "
            f"complete ({session_state}).\n{tip}"
        )

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


def sync_results_playhead(gui) -> str:
    """Keep the Results playhead aligned with the authoritative transport state."""
    from stablewalk.ui.playback_state import read_playback_state

    playhead = getattr(gui, "lbl_summary_playhead", None)
    state = read_playback_state(gui)
    if playhead is None or state is None or state.total_frames <= 0:
        if playhead is not None:
            playhead.configure(text="")
        return ""
    sync_txt = (
        f"Playhead sync: frame {state.format_frame_label()} · "
        f"{state.format_time_label()}"
    )
    playhead.configure(text=sync_txt, fg=MUTED)
    return sync_txt


def update_results_summary_panel(
    gui,
    summary,
    *,
    refresh_pipeline: bool = True,
) -> None:
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
    src_tip = "Source recording for this Results pipeline."
    sync_txt = sync_results_playhead(gui)
    if sync_txt:
        src_tip = sync_txt
    if src is not None:
        if summary.source:
            src.configure(text=f"Source recording: {summary.source}", fg=TEXT)
        else:
            src.configure(text="No video analyzed", fg=MUTED)
        create_tooltip(src, src_tip, wraplength=340)

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
        "pose_coverage": summary.pose_coverage,
        "contact_coverage": summary.contact_coverage,
        "contact_coverage_left": summary.contact_coverage_left,
        "contact_coverage_right": summary.contact_coverage_right,
        "cycle_evidence": summary.cycle_evidence,
        "spectral_reliability": summary.spectral_reliability,
        "biomechanical_validity": summary.biomechanical_validity,
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
                        value="Not detected",
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
                status_lbl.configure(text="Not detected", fg=WARNING)
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

    try:
        sync_results_primary_findings(gui)
    except Exception:
        pass
    try:
        from stablewalk.ui.tk.dashboard_research_validation import (
            refresh_research_validation,
        )

        refresh_research_validation(gui)
    except Exception:
        pass

    export_state = tk.NORMAL if summary.source else tk.DISABLED
    for attr in (
        "btn_export_summary_json",
        "btn_export_summary_report",
    ):
        btn = getattr(gui, attr, None)
        if btn is not None:
            btn.configure(state=export_state)
    # Download Research Report is enabled by refresh_research_validation from
    # computed session outputs; only force-enable when a session source exists.
    research_btn = getattr(gui, "btn_download_research_report", None)
    if research_btn is not None and summary.source:
        research_btn.configure(state=tk.NORMAL)

    gui._analysis_summary_cache = summary

    # Refresh Results visual pipeline so Metrics/Report stages track the summary.
    if refresh_pipeline and getattr(gui, "_results_pipeline_host", None) is not None:
        try:
            from stablewalk.ui.tk.dashboard_results_pipeline import (
                update_results_analysis_pipeline,
            )

            report = getattr(gui, "_pipeline_status_report_cache", None)
            if report is None:
                assess = getattr(gui, "_assess_pipeline_status_report", None)
                if callable(assess):
                    report = assess(force=False)
            update_results_analysis_pipeline(gui, report)
        except Exception:
            pass

    # Expand pipeline full-width when no report preview; otherwise show the rail.
    try:
        sync_results_report_preview(gui, summary)
    except Exception:
        pass


__all__ = [
    "RESULTS_PROCESSING_DETAILS_TITLE",
    "RESULTS_REPORT_PREVIEW_WIDTH",
    "apply_results_processing_details_expansion",
    "build_results_summary_tab",
    "results_preview_rail_floor",
    "results_processing_details_should_expand",
    "sync_results_primary_findings",
    "sync_results_report_preview",
    "update_results_summary_panel",
]
