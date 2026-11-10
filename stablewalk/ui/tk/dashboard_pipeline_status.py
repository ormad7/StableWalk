"""Pipeline Status panel — compact collapsible stages with top summary."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.monitoring.pipeline_status import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
    PipelineDiagramStage,
    PipelineStatusReport,
    STATUS_LABEL,
    build_pipeline_diagram,
)
from stablewalk.ui.theme import (
    BORDER,
    DASHBOARD_CARD_PAD,
    DANGER,
    ELEVATED,
    FONT_CAPTION,
    FONT_UI_SEMIBOLD,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    MUTED_DIM,
    ORANGE,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    SUCCESS,
    TEXT,
    create_tooltip,
    tk_labelframe_title,
    GLYPH_COLLAPSE,
    GLYPH_EXPAND,
    pack_empty_state,
    create_elevated_card,
)
from stablewalk.ui.tk.dashboard_pipeline_visual import (
    PIPELINE_STATUS_FG,
    PIPELINE_STATUS_STRIP,
    metrics_from_dialog_content,
)
from stablewalk.ui.tk.pipeline_stage_dialog import (
    bind_pipeline_stage_click,
    build_pipeline_stage_dialog_content,
    resolve_stage_ui_spec,
)

_STATUS_FG = PIPELINE_STATUS_FG
_STATUS_STRIP = PIPELINE_STATUS_STRIP

# Recovery actions for incomplete stages (gui method name → button label).
_STAGE_RECOVERY: dict[str, tuple[str, str]] = {
    "opensim": ("_export_opensim_session", "Export OpenSim Files"),
    "retargeting": ("_run_real_to_sim_pipeline", "Run Real-to-Sim"),
    "export": ("_export_motion_reference", "Export Motion Reference"),
    "virtual_grf": ("_ensure_results_tab_content", "Refresh Results"),
}


def build_pipeline_status_panel(
    gui,
    parent: tk.Misc,
    *,
    host_attr: str = "_pipeline_status_host",
    section_attr: str = "_section_pipeline_status",
    framed: bool = True,
) -> ttk.LabelFrame | ttk.Frame:
    """Install a Pipeline Status monitoring panel."""
    if framed:
        title = tk_labelframe_title(parent, "Pipeline Status")
        section = ttk.LabelFrame(
            parent,
            labelwidget=title,
            style="Card.TLabelframe",
            padding=DASHBOARD_CARD_PAD,
        )
    else:
        section = ttk.Frame(parent)
    section.columnconfigure(0, weight=1)

    summary = tk.Frame(section, bg=PANEL, highlightthickness=0)
    summary.grid(row=0, column=0, sticky="ew", pady=(0, PAD_SM))
    summary.columnconfigure(0, weight=1)
    setattr(gui, f"{host_attr}_summary", summary)

    intro = tk.Label(
        section,
        text="Stages stay collapsed — expand one for details. Missing stages mean the full pipeline did not complete.",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=720,
        justify=tk.LEFT,
    )
    intro.grid(row=1, column=0, sticky="ew", pady=(0, PAD_SM))

    host = tk.Frame(section, bg=PANEL, highlightthickness=0)
    host.grid(row=2, column=0, sticky="nsew")
    host.columnconfigure(0, weight=1)

    setattr(gui, host_attr, host)
    hosts = list(getattr(gui, "_pipeline_status_hosts", []))
    if host not in hosts:
        hosts.append(host)
    gui._pipeline_status_hosts = hosts
    summaries = list(getattr(gui, "_pipeline_status_summaries", []))
    if summary not in summaries:
        summaries.append(summary)
    gui._pipeline_status_summaries = summaries
    gui._pipeline_status_stage_cards: dict[str, tk.Frame] = {}
    gui._pipeline_status_connectors: dict[str, tk.Misc] = {}
    gui._pipeline_status_expanded: dict[str, bool] = getattr(
        gui, "_pipeline_status_expanded", {}
    )

    section.rowconfigure(2, weight=1)
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
        f"Status: {STATUS_LABEL.get(stage.status, stage.status)}\n"
        f"{stage.detail}\n\n"
        "Click the chevron for details · card opens the full stage dialog"
    )
    return combine_metric_tooltip(base, detail)


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
        toggle.configure(text=GLYPH_COLLAPSE if expanded else GLYPH_EXPAND)


def _status_chip(parent: tk.Misc, status: str) -> tk.Frame:
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
        text=STATUS_LABEL.get(status, status),
        bg=ELEVATED,
        fg=fg,
        font=FONT_CAPTION,
    ).pack(padx=6, pady=1)
    return chip


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
        font=FONT_CAPTION,
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


def _stage_sub_items(report: PipelineStatusReport | None, item_keys: tuple[str, ...]):
    if report is None:
        return []
    by_key = {item.key: item for item in report.all_items()}
    return [by_key[k] for k in item_keys if k in by_key]


def _pipeline_summary_counts(
    stages: list[PipelineDiagramStage],
) -> tuple[int, int, int, int]:
    completed = sum(1 for s in stages if s.status == STATUS_COMPLETED)
    partial = sum(1 for s in stages if s.status == STATUS_PARTIAL)
    failed = sum(1 for s in stages if s.status == STATUS_UNAVAILABLE)
    return completed, partial, failed, len(stages)


def _overall_confidence_text(report: PipelineStatusReport | None) -> str:
    if report is None:
        return "—"
    # Prefer diagram-level confidence if present on report dict path via groups.
    conf_vals: list[float] = []
    for item in report.all_items():
        if item.status == STATUS_UNAVAILABLE:
            continue
        # confidence may be embedded in detail like "72%" — skip inventing.
        tip = (item.tooltip or "") + " " + (item.detail or "")
        if "confidence" in tip.lower():
            continue
    # Use completed / assessed ratio as display confidence proxy when no field exists.
    stages = build_pipeline_diagram(report)
    assessed = [s for s in stages if s.status != STATUS_UNAVAILABLE]
    if not assessed:
        return "n/a"
    score = (
        sum(1.0 if s.status == STATUS_COMPLETED else 0.5 for s in assessed)
        / len(stages)
        * 100.0
    )
    return f"{score:.0f}%"


def _render_summary_bar(
    host: tk.Misc,
    stages: list[PipelineDiagramStage],
    report: PipelineStatusReport | None,
) -> None:
    for child in host.winfo_children():
        child.destroy()
    completed, partial, failed, total = _pipeline_summary_counts(stages)
    conf = _overall_confidence_text(report)

    chips = (
        (f"{completed} completed", SUCCESS if completed else MUTED),
        (f"{partial} partial", ORANGE if partial else MUTED),
        (f"{failed} missing", DANGER if failed and completed == 0 else (ORANGE if failed else MUTED)),
        (f"confidence {conf}", TEXT),
    )
    row = tk.Frame(host, bg=PANEL, highlightthickness=0)
    row.pack(fill=tk.X)
    for text, fg in chips:
        chip = tk.Frame(
            row,
            bg=ELEVATED,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        chip.pack(side=tk.LEFT, padx=(0, PAD_XS))
        tk.Label(chip, text=text, bg=ELEVATED, fg=fg, font=FONT_UI_XS).pack(
            padx=8, pady=3
        )

    if failed or partial:
        note = (
            "Pipeline incomplete — some stages are partial or missing. "
            "Do not treat this session as a full Real-to-Sim run."
        )
    elif completed == total and total > 0:
        note = "All listed stages completed for this session."
    else:
        note = "Analyze a video to populate pipeline stages."
    tk.Label(
        host,
        text=note,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=720,
        justify=tk.LEFT,
    ).pack(fill=tk.X, pady=(PAD_XS, 0))


def _recovery_command(gui, method_name: str):
    def _run() -> None:
        fn = getattr(gui, method_name, None)
        if callable(fn):
            try:
                fn()
            except TypeError:
                fn()
            except Exception as exc:
                try:
                    from stablewalk.ui.tk.tk_lifecycle import safe_messagebox

                    safe_messagebox(
                        "error",
                        "Recovery action",
                        str(exc),
                        parent=getattr(gui, "root", None),
                    )
                except Exception:
                    pass

    return _run


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
    outer.grid(row=row, column=0, sticky="ew", pady=(0, PAD_XS))
    outer.columnconfigure(0, weight=1)

    card = create_elevated_card(outer)
    card.grid(row=0, column=0, sticky="ew")
    card.columnconfigure(1, weight=1)

    strip = tk.Frame(card, bg=dot_color, width=4)
    strip.grid(row=0, column=0, sticky="ns")
    strip.grid_propagate(False)

    body = tk.Frame(card, bg=ELEVATED, padx=PAD_MD, pady=PAD_SM)
    body.grid(row=0, column=1, sticky="ew")
    body.columnconfigure(1, weight=1)

    # Compact header: expand · name · status · confidence
    header = tk.Frame(body, bg=ELEVATED)
    header.grid(row=0, column=0, columnspan=2, sticky="ew")
    header.columnconfigure(1, weight=1)

    toggle = tk.Label(
        header,
        text=GLYPH_EXPAND,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        cursor="hand2",
    )
    toggle.grid(row=0, column=0, sticky="w", padx=(0, PAD_SM))

    tk.Label(
        header,
        text=f"{step}. {stage.label}",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_SEMIBOLD,
        anchor="w",
    ).grid(row=0, column=1, sticky="w")

    if metrics.confidence_pct is not None:
        conf_txt = f"{metrics.confidence_pct}%"
        conf_fg = SUCCESS if status == STATUS_COMPLETED else badge_fg
    else:
        conf_txt = "—"
        conf_fg = MUTED_DIM
    tk.Label(
        header,
        text=conf_txt,
        bg=ELEVATED,
        fg=conf_fg,
        font=FONT_UI_SM,
        anchor="e",
    ).grid(row=0, column=2, sticky="e", padx=(PAD_SM, PAD_SM))

    _status_chip(header, status).grid(row=0, column=3, sticky="e")

    # One-line short result (always visible)
    tk.Label(
        body,
        text=stage.detail,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=560,
        justify=tk.LEFT,
    ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

    # Expandable details — hidden until requested
    spec = resolve_stage_ui_spec(stage.key)
    detail_extra = tk.Frame(body, bg=ELEVATED)
    detail_extra.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    detail_extra.columnconfigure(0, weight=1)

    drow = 0
    drow = _add_detail_section(detail_extra, drow, "Purpose", spec.purpose)
    drow = _add_detail_section(detail_extra, drow, "Why this status", stage.detail)
    drow = _add_detail_section(detail_extra, drow, "Input", spec.input_desc)
    drow = _add_detail_section(detail_extra, drow, "Output", spec.output_desc)
    drow = _add_detail_section(detail_extra, drow, "Validation quality", content.quality)
    drow = _add_detail_section(detail_extra, drow, "Confidence", content.confidence)
    drow = _add_detail_section(
        detail_extra,
        drow,
        "Warnings",
        content.warnings,
        fg=ORANGE if content.warnings.startswith("•") else MUTED,
    )

    if status != STATUS_COMPLETED:
        sub_items = _stage_sub_items(report, stage.item_keys)
        incomplete = [it for it in sub_items if it.status != STATUS_COMPLETED]
        tk.Label(
            detail_extra,
            text="What's missing",
            bg=ELEVATED,
            fg=ORANGE,
            font=FONT_CAPTION,
            anchor="w",
        ).grid(row=drow, column=0, sticky="w")
        drow += 1
        if incomplete:
            for it in incomplete:
                tk.Label(
                    detail_extra,
                    text=f"· {it.label} — {it.detail}",
                    bg=ELEVATED,
                    fg=MUTED,
                    font=FONT_UI_XS,
                    anchor="w",
                    wraplength=520,
                    justify=tk.LEFT,
                ).grid(row=drow, column=0, sticky="ew", pady=(0, 2))
                drow += 1
        else:
            tk.Label(
                detail_extra,
                text="Stage has not produced completed sub-steps yet.",
                bg=ELEVATED,
                fg=MUTED,
                font=FONT_UI_XS,
                anchor="w",
            ).grid(row=drow, column=0, sticky="ew")
            drow += 1

        recovery = _STAGE_RECOVERY.get(stage.key)
        if recovery is not None:
            method_name, label = recovery
            if getattr(gui, method_name, None) is not None:
                btn = ttk.Button(
                    detail_extra,
                    text=label,
                    style="Compact.TButton",
                    command=_recovery_command(gui, method_name),
                )
                btn.grid(row=drow, column=0, sticky="w", pady=(PAD_SM, 0))
                create_tooltip(btn, f"Attempt recovery: {label}")

    expanded_map = getattr(gui, "_pipeline_status_expanded", {})
    is_expanded = bool(expanded_map.get(stage.key, False))
    if is_expanded:
        toggle.configure(text=GLYPH_COLLAPSE)
    else:
        detail_extra.grid_remove()

    widgets = {"toggle": toggle, "detail_extra": detail_extra}

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


def _render_pipeline_status_host(
    host: tk.Misc, report: PipelineStatusReport | None, gui
) -> None:
    for child in host.winfo_children():
        child.destroy()

    gui._pipeline_status_connectors = {}

    if report is None or not report.groups:
        pack_empty_state(
            host,
            "Load and analyze a walking video to view pipeline status.",
        )
        return

    stages = build_pipeline_diagram(report)
    host.columnconfigure(0, weight=1)

    total = len(stages)
    for index, stage in enumerate(stages):
        card = _build_stage_card(
            host, index, stage, gui, report, step=index + 1, total=total
        )
        gui._pipeline_status_stage_cards[stage.key] = card


def update_pipeline_status_panel(gui, report: PipelineStatusReport | None) -> None:
    """Refresh all installed pipeline diagram hosts."""
    # Never rebuild stage widgets during playback — keep last paint.
    if getattr(gui, "playing", False):
        gui._pipeline_status_report_cache = report
        return

    hosts = list(getattr(gui, "_pipeline_status_hosts", []))
    primary = getattr(gui, "_pipeline_status_host", None)
    if primary is not None and primary not in hosts:
        hosts.append(primary)

    summaries = list(getattr(gui, "_pipeline_status_summaries", []))

    # Results page uses a dedicated visual analysis pipeline.
    try:
        from stablewalk.ui.tk.dashboard_results_pipeline import (
            update_results_analysis_pipeline,
        )

        if getattr(gui, "_results_pipeline_host", None) is not None:
            update_results_analysis_pipeline(gui, report)
    except Exception:
        pass

    if not hosts and not summaries:
        gui._pipeline_status_report_cache = report
        return

    stages = build_pipeline_diagram(report) if report is not None else []
    for summary in summaries:
        try:
            if report is None or not stages:
                for child in summary.winfo_children():
                    child.destroy()
                tk.Label(
                    summary,
                    text="No pipeline assessment yet.",
                    bg=PANEL,
                    fg=MUTED,
                    font=FONT_UI_XS,
                    anchor="w",
                ).pack(fill=tk.X)
            else:
                _render_summary_bar(summary, stages, report)
        except tk.TclError:
            pass

    gui._pipeline_status_stage_cards = {}
    for host in hosts:
        _render_pipeline_status_host(host, report, gui)

    gui._pipeline_status_report_cache = report


__all__ = ["build_pipeline_status_panel", "update_pipeline_status_panel"]
