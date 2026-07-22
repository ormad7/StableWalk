"""Pipeline Status panel — vertical Real-to-Sim pipeline diagram."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.monitoring.pipeline_status import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    PipelineDiagramStage,
    PipelineStatusReport,
    STATUS_LABEL,
    STATUS_SYMBOL,
    build_pipeline_diagram,
)
from stablewalk.ui.theme import (
    BORDER,
    DASHBOARD_CARD_PAD,
    ELEVATED,
    FONT_HEADING,
    FONT_TITLE,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    MUTED_DIM,
    ORANGE,
    PANEL,
    SUCCESS,
    TEXT,
    create_tooltip,
)
from stablewalk.ui.tk.dashboard_pipeline_visual import (
    PIPELINE_STATUS_FG,
    PIPELINE_STATUS_STRIP,
    animate_connector_activation,
    install_progress_track,
    metrics_from_dialog_content,
    set_connector_state,
    set_progress_fill,
)
from stablewalk.ui.tk.pipeline_stage_dialog import (
    bind_pipeline_stage_click,
    build_pipeline_stage_dialog_content,
    resolve_stage_ui_spec,
)

_STATUS_FG = PIPELINE_STATUS_FG
_STATUS_STRIP = PIPELINE_STATUS_STRIP


def build_pipeline_status_panel(
    gui,
    parent: tk.Misc,
    *,
    host_attr: str = "_pipeline_status_host",
    section_attr: str = "_section_pipeline_status",
) -> ttk.LabelFrame:
    """Install a Pipeline Status monitoring panel."""
    section = ttk.LabelFrame(
        parent,
        text="  Pipeline Status  ",
        style="Card.TLabelframe",
        padding=DASHBOARD_CARD_PAD,
    )
    section.columnconfigure(0, weight=1)

    intro = tk.Label(
        section,
        text=(
            "Artifact-grounded workflow for the current video session — each stage is a "
            "job showing execution time, output files, and confidence. "
            "▶ expands full details · click a stage for the run log."
        ),
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=680,
        justify=tk.LEFT,
    )
    intro.grid(row=0, column=0, sticky="ew", pady=(0, 8))

    host = tk.Frame(section, bg=PANEL, highlightthickness=0)
    host.grid(row=1, column=0, sticky="nsew")
    host.columnconfigure(0, weight=1)

    setattr(gui, host_attr, host)
    hosts = list(getattr(gui, "_pipeline_status_hosts", []))
    if host not in hosts:
        hosts.append(host)
    gui._pipeline_status_hosts = hosts
    gui._pipeline_status_stage_cards: dict[str, tk.Frame] = {}
    gui._pipeline_status_connectors: dict[str, tk.Misc] = {}
    gui._pipeline_status_expanded: dict[str, bool] = getattr(
        gui, "_pipeline_status_expanded", {}
    )

    section.rowconfigure(1, weight=1)
    setattr(gui, section_attr, section)
    if section_attr == "_section_pipeline_status":
        gui._section_pipeline_status = section
    from stablewalk.ui.metric_tooltips import attach_metric_tooltip

    attach_metric_tooltip(section, "pipeline_status")
    attach_metric_tooltip(intro, "pipeline_status")
    return section


def _stage_tooltip(stage: PipelineDiagramStage) -> str:
    from stablewalk.ui.metric_tooltips import combine_metric_tooltip, get_metric_tooltip

    base = get_metric_tooltip("pipeline_status")
    detail = (
        f"{stage.label}\n\n"
        f"{stage.tooltip}\n\n"
        f"Status: {stage.display_line()}\n"
        f"{stage.detail}\n\n"
        "Click for full stage details · ▶ expands summary"
    )
    return combine_metric_tooltip(base, detail)


def _build_connector(
    parent: tk.Misc,
    row: int,
    gui,
    connector_key: str,
) -> tk.Label:
    rail = tk.Frame(parent, bg=PANEL, width=18)
    rail.grid(row=row, column=0, sticky="ns")
    rail.grid_propagate(False)

    arrow = tk.Label(
        rail,
        text="\u2193",
        bg=PANEL,
        fg=MUTED_DIM,
        font=FONT_HEADING,
    )
    arrow.pack(pady=(0, 2))

    tk.Frame(parent, bg=PANEL).grid(row=row, column=1, sticky="ew")

    connectors = getattr(gui, "_pipeline_status_connectors", {})
    connectors[connector_key] = arrow
    gui._pipeline_status_connectors = connectors
    return arrow


def _toggle_status_expanded(gui, stage_key: str, widgets: dict) -> None:
    expanded_map = getattr(gui, "_pipeline_status_expanded", {})
    expanded = not bool(expanded_map.get(stage_key, False))
    expanded_map[stage_key] = expanded
    gui._pipeline_status_expanded = expanded_map

    detail_extra = widgets.get("detail_extra")
    toggle = widgets.get("toggle")
    if detail_extra is not None:
        if expanded:
            detail_extra.grid()
        else:
            detail_extra.grid_remove()
    if toggle is not None:
        toggle.configure(text="\u25bc" if expanded else "\u25b6")


_TIME_ICON = "\u23f1"  # ⏱ stopwatch
_CONF_ICON = "\u25c9"  # ◉ fisheye
_FILE_ICON = "\u25a4"  # ▤ document-like square


def _progress_fraction(status: str, confidence_pct: int | None) -> float:
    """CI-style completion bar fill for a stage."""
    if status == STATUS_COMPLETED:
        return 1.0
    if status == STATUS_PARTIAL:
        if confidence_pct is not None:
            return max(0.15, min(0.9, confidence_pct / 100.0))
        return 0.5
    return 0.0


def _stage_sub_items(report: PipelineStatusReport | None, item_keys: tuple[str, ...]):
    if report is None:
        return []
    by_key = {item.key: item for item in report.all_items()}
    return [by_key[k] for k in item_keys if k in by_key]


def _status_chip(parent: tk.Misc, status: str) -> tk.Frame:
    """A CI-style status pill: bordered chip coloured by the stage result."""
    fg = _STATUS_FG.get(status, MUTED)
    chip = tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=fg,
        highlightcolor=fg,
    )
    tk.Label(
        chip,
        text=f"{STATUS_SYMBOL[status]} {STATUS_LABEL[status]}",
        bg=ELEVATED,
        fg=fg,
        font=(FONT_UI_XS[0], FONT_UI_XS[1], "bold"),
    ).pack(padx=6, pady=1)
    return chip


def _ci_metric(parent: tk.Misc, column: int, icon: str, text: str, fg: str) -> None:
    """One compact metadata pill in the always-visible job summary row."""
    chip = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)
    chip.grid(row=0, column=column, sticky="w", padx=(0, 12))
    tk.Label(chip, text=icon, bg=ELEVATED, fg=fg, font=FONT_UI_XS).pack(side=tk.LEFT, padx=(0, 3))
    tk.Label(chip, text=text, bg=ELEVATED, fg=MUTED, font=FONT_UI_XS).pack(side=tk.LEFT)


def _add_detail_section(
    parent: tk.Misc,
    row: int,
    label: str,
    text: str,
    *,
    fg: str = MUTED,
    label_fg: str = TEXT,
) -> int:
    tk.Label(
        parent,
        text=label,
        bg=ELEVATED,
        fg=label_fg,
        font=(FONT_UI_XS[0], FONT_UI_XS[1], "bold"),
        anchor="w",
    ).grid(row=row, column=0, sticky="w")
    tk.Label(
        parent,
        text=text,
        bg=ELEVATED,
        fg=fg,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=520,
        justify=tk.LEFT,
    ).grid(row=row + 1, column=0, sticky="ew", pady=(0, 6))
    return row + 2


def _build_stage_card(
    parent: tk.Misc,
    row: int,
    stage: PipelineDiagramStage,
    gui,
    report: PipelineStatusReport | None,
    *,
    step: int,
    total: int,
) -> tk.Frame:
    status = stage.status
    dot_color = _STATUS_STRIP.get(status, BORDER)
    badge_fg = _STATUS_FG.get(status, MUTED)

    content = build_pipeline_stage_dialog_content(
        stage.key, report, gui=gui, diagram_stage=stage
    )
    metrics = metrics_from_dialog_content(content)

    outer = tk.Frame(parent, bg=PANEL)
    outer.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 0))
    outer.columnconfigure(1, weight=1)

    # Rail: CI job index over a status dot, keeping the vertical workflow feel.
    rail = tk.Frame(outer, bg=PANEL, width=26)
    rail.grid(row=0, column=0, sticky="ns", padx=(0, 6))
    rail.grid_propagate(False)
    tk.Label(
        rail,
        text=f"{step}/{total}",
        bg=PANEL,
        fg=MUTED_DIM,
        font=FONT_UI_XS,
    ).pack(pady=(8, 0))
    tk.Label(
        rail,
        text=STATUS_SYMBOL.get(status, "\u2022"),
        bg=PANEL,
        fg=dot_color,
        font=FONT_TITLE,
    ).pack(pady=(2, 0))

    card = tk.Frame(
        outer,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    card.grid(row=0, column=1, sticky="ew")
    card.columnconfigure(1, weight=1)

    strip = tk.Frame(card, bg=dot_color, width=4)
    strip.grid(row=0, column=0, sticky="ns")
    strip.grid_propagate(False)

    body = tk.Frame(card, bg=ELEVATED, padx=10, pady=8)
    body.grid(row=0, column=1, sticky="ew")
    body.columnconfigure(0, weight=1)

    # --- Header: expander + stage name + status pill ---
    header = tk.Frame(body, bg=ELEVATED)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(1, weight=1)

    toggle = tk.Label(
        header,
        text="\u25b6",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        cursor="hand2",
    )
    toggle.grid(row=0, column=0, sticky="w", padx=(0, 6))

    tk.Label(
        header,
        text=stage.label,
        bg=ELEVATED,
        fg=TEXT,
        font=(FONT_UI_SM[0], FONT_UI_SM[1], "bold"),
        anchor="w",
    ).grid(row=0, column=1, sticky="w")

    _status_chip(header, status).grid(row=0, column=2, sticky="e", padx=(8, 0))

    # --- Job summary row: execution time · confidence · output files ---
    metrics_row = tk.Frame(body, bg=ELEVATED)
    metrics_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))

    duration_txt = metrics.duration_text if metrics.duration_text != "\u2014" else "not timed"
    _ci_metric(metrics_row, 0, _TIME_ICON, duration_txt, MUTED)

    if metrics.confidence_pct is not None:
        conf_txt = f"{metrics.confidence_pct}% confidence"
        conf_fg = SUCCESS if status == STATUS_COMPLETED else badge_fg
    else:
        conf_txt = "confidence n/a"
        conf_fg = MUTED_DIM
    _ci_metric(metrics_row, 1, _CONF_ICON, conf_txt, conf_fg)

    file_count = metrics.files_count
    files_txt = f"{file_count} output file{'' if file_count == 1 else 's'}"
    _ci_metric(metrics_row, 2, _FILE_ICON, files_txt, SUCCESS if file_count else MUTED_DIM)

    # --- One-line result summary ---
    tk.Label(
        body,
        text=stage.detail,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=520,
        justify=tk.LEFT,
    ).grid(row=2, column=0, sticky="ew", pady=(6, 0))

    # --- Completion bar (green when the job passed) ---
    track, fill = install_progress_track(body, row=3)
    track.grid_configure(pady=(8, 0))
    fill_color = (
        SUCCESS
        if status == STATUS_COMPLETED
        else (ORANGE if status == STATUS_PARTIAL else BORDER)
    )
    set_progress_fill(fill, _progress_fraction(status, metrics.confidence_pct), color=fill_color)

    # --- Expandable details (CI "job log") ---
    spec = resolve_stage_ui_spec(stage.key)
    detail_extra = tk.Frame(body, bg=ELEVATED)
    detail_extra.grid(row=4, column=0, sticky="ew", pady=(8, 0))
    detail_extra.columnconfigure(0, weight=1)

    drow = 0
    drow = _add_detail_section(detail_extra, drow, "Purpose", spec.purpose)
    drow = _add_detail_section(detail_extra, drow, "Input", spec.input_desc)
    drow = _add_detail_section(detail_extra, drow, "Output", spec.output_desc)
    drow = _add_detail_section(detail_extra, drow, "Algorithms", spec.algorithms)
    drow = _add_detail_section(detail_extra, drow, "Execution time", content.duration_text)
    drow = _add_detail_section(detail_extra, drow, "Output files", content.generated_files)
    drow = _add_detail_section(detail_extra, drow, "Confidence", content.confidence)

    # For anything not fully completed, spell out exactly what is missing.
    if status != STATUS_COMPLETED:
        sub_items = _stage_sub_items(report, stage.item_keys)
        incomplete = [it for it in sub_items if it.status != STATUS_COMPLETED]
        tk.Label(
            detail_extra,
            text="What's missing",
            bg=ELEVATED,
            fg=ORANGE,
            font=(FONT_UI_XS[0], FONT_UI_XS[1], "bold"),
            anchor="w",
        ).grid(row=drow, column=0, sticky="w")
        drow += 1
        if incomplete:
            for it in incomplete:
                line = tk.Frame(detail_extra, bg=ELEVATED)
                line.grid(row=drow, column=0, sticky="ew", pady=(0, 3))
                line.columnconfigure(1, weight=1)
                drow += 1
                tk.Label(
                    line,
                    text=STATUS_SYMBOL.get(it.status, "\u2022"),
                    bg=ELEVATED,
                    fg=_STATUS_FG.get(it.status, MUTED),
                    font=FONT_UI_XS,
                ).grid(row=0, column=0, sticky="nw", padx=(0, 5))
                tk.Label(
                    line,
                    text=f"{it.label} — {it.detail}",
                    bg=ELEVATED,
                    fg=MUTED,
                    font=FONT_UI_XS,
                    anchor="w",
                    wraplength=480,
                    justify=tk.LEFT,
                ).grid(row=0, column=1, sticky="ew")
        else:
            tk.Label(
                detail_extra,
                text="Stage has not produced any completed sub-steps yet.",
                bg=ELEVATED,
                fg=MUTED,
                font=FONT_UI_XS,
                anchor="w",
                wraplength=520,
                justify=tk.LEFT,
            ).grid(row=drow, column=0, sticky="ew", pady=(0, 6))
            drow += 1

    expanded_map = getattr(gui, "_pipeline_status_expanded", {})
    is_expanded = bool(expanded_map.get(stage.key, False))
    if is_expanded:
        toggle.configure(text="\u25bc")
    else:
        detail_extra.grid_remove()

    widgets = {
        "toggle": toggle,
        "detail_extra": detail_extra,
    }

    def _on_toggle(_event: object | None = None) -> str:
        _toggle_status_expanded(gui, stage.key, widgets)
        return "break"

    toggle.bind("<Button-1>", _on_toggle)

    tip = _stage_tooltip(stage)
    for widget in (card, body, header, strip):
        create_tooltip(widget, tip)

    bind_pipeline_stage_click(
        card,
        gui,
        stage.key,
        diagram_stage=stage,
        skip_widgets=(toggle,),
    )
    return card


def _render_pipeline_status_host(host: tk.Misc, report: PipelineStatusReport | None, gui) -> None:
    for child in host.winfo_children():
        child.destroy()

    gui._pipeline_status_connectors = {}

    if report is None or not report.groups:
        tk.Label(
            host,
            text="Load and analyze a walking video to view pipeline status.",
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_SM,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        return

    stages = build_pipeline_diagram(report)
    host.columnconfigure(0, weight=1)

    prev_active = getattr(gui, "_pipeline_status_connector_active", {})
    new_active: dict[str, bool] = {}

    total = len(stages)
    row = 0
    for index, stage in enumerate(stages):
        card = _build_stage_card(host, row, stage, gui, report, step=index + 1, total=total)
        gui._pipeline_status_stage_cards[stage.key] = card
        row += 1

        if index < len(stages) - 1:
            next_stage = stages[index + 1]
            connector_key = f"{stage.key}->{next_stage.key}"
            arrow = _build_connector(host, row, gui, connector_key)
            active = stage.status == STATUS_COMPLETED
            new_active[connector_key] = active
            was_active = bool(prev_active.get(connector_key))
            if active and not was_active:
                animate_connector_activation(gui, connector_key, arrow, active=True)
            else:
                set_connector_state(arrow, active=active, dormant_fg=MUTED_DIM)
            row += 1

    gui._pipeline_status_connector_active = new_active


def update_pipeline_status_panel(gui, report: PipelineStatusReport | None) -> None:
    """Refresh all installed pipeline diagram hosts."""
    hosts = list(getattr(gui, "_pipeline_status_hosts", []))
    primary = getattr(gui, "_pipeline_status_host", None)
    if primary is not None and primary not in hosts:
        hosts.append(primary)

    if not hosts:
        return

    gui._pipeline_status_stage_cards = {}
    for host in hosts:
        _render_pipeline_status_host(host, report, gui)

    gui._pipeline_status_report_cache = report


__all__ = ["build_pipeline_status_panel", "update_pipeline_status_panel"]
