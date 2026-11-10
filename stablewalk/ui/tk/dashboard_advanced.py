"""Advanced & Export tab — professional engineering pipeline dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import tkinter as tk
from tkinter import ttk

from stablewalk import config
from stablewalk.monitoring.pipeline_status import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
)
from stablewalk.opensim_ik_status import (
    IK_COMPLETE,
    IK_FAILED,
    IK_PARTIAL,
    IK_RUNNING,
    PIPELINE_COMPLETE,
    ik_mot_unavailable_reason,
    resolve_opensim_ik_status_from_gui,
    resolve_opensim_pipeline_status_from_gui,
    resolve_opensim_status_pair_from_gui,
)
from stablewalk.ui.scientific_labels import (
    LABEL_VIRTUAL_GRF_PANEL,
    LABEL_VIRTUAL_GRF_UNAVAILABLE_MSG,
)
from stablewalk.ui.theme import (
    BORDER,
    ELEVATED,
    FONT_SUMMARY_METRIC_TITLE,
    FONT_UI_SM,
    FONT_UI_XS,
    DANGER,
    INFO,
    MUTED,
    SUCCESS,
    PAD_SM,
    PANEL,
    TEXT,
    WARNING,
    create_tooltip,
    GLYPH_COLLAPSE,
    GLYPH_EXPAND,
    create_elevated_card,
)
from stablewalk.ui.tk.dashboard_pipeline_visual import (
    PIPELINE_STATUS_FG,
    PIPELINE_STATUS_STRIP,
)
from stablewalk.ui.tk.pipeline_stage_dialog import (
    bind_pipeline_stage_click,
    resolve_stage_ui_spec,
)

if TYPE_CHECKING:
    from stablewalk.monitoring.pipeline_status import PipelineStageState
else:
    PipelineStageState = str

ENGINEERING_STATUS_SYMBOL = {
    STATUS_COMPLETED: "",
    STATUS_PARTIAL: "",
    STATUS_UNAVAILABLE: "",
}

ENGINEERING_STATUS_LABEL = {
    STATUS_COMPLETED: "Completed",
    STATUS_PARTIAL: "Partial",
    STATUS_UNAVAILABLE: "Missing",
}

ENGINEERING_STATUS_FG = PIPELINE_STATUS_FG
ENGINEERING_STATUS_STRIP = PIPELINE_STATUS_STRIP

_OPENSIM_IK_ROW_FG = {
    "Complete": SUCCESS,
    "Running": INFO,
    "Partial": WARNING,
    "Failed": DANGER,
}


def _opensim_row_fg(row: "OpensimStatusRow") -> str:
    if row.key == "ik":
        return _OPENSIM_IK_ROW_FG.get(row.value, ENGINEERING_STATUS_FG.get(row.status, MUTED))
    return ENGINEERING_STATUS_FG.get(row.status, MUTED)


@dataclass(frozen=True)
class EngineeringStageView:
    key: str
    title: str
    status: PipelineStageState
    detail: str
    tooltip: str = ""


def _status_display(status: PipelineStageState) -> str:
    return ENGINEERING_STATUS_LABEL[status]


def _rts_stage_status(raw: str | None) -> PipelineStageState:
    if raw in ("complete", "completed"):
        return STATUS_COMPLETED
    if raw in ("partial",):
        return STATUS_PARTIAL
    return STATUS_UNAVAILABLE


def _toggle_engineering_expanded(gui, key: str) -> None:
    cards = getattr(gui, "_adv_engineering_cards", {})
    entry = cards.get(key)
    if entry is None:
        return
    expanded = not bool(entry.get("expanded", False))
    entry["expanded"] = expanded
    detail_extra = entry.get("detail_extra")
    toggle = entry.get("toggle")
    if detail_extra is not None:
        if expanded:
            detail_extra.grid()
        else:
            detail_extra.grid_remove()
    if toggle is not None:
        toggle.configure(text=GLYPH_COLLAPSE if expanded else GLYPH_EXPAND)


def _build_engineering_card(
    parent: tk.Misc,
    *,
    row: int,
    column: int,
    key: str,
    title: str,
    gui,
    detail_attr: str,
    status_attr: str,
) -> tk.Frame:
    """One elevated engineering stage card with status badge and detail line."""
    card = create_elevated_card(parent)
    card.grid(
        row=row,
        column=column,
        sticky="nsew",
        padx=(0, PAD_SM),
        pady=(0, PAD_SM),
    )
    card.columnconfigure(0, weight=1)

    strip_color = ENGINEERING_STATUS_STRIP[STATUS_UNAVAILABLE]
    strip = tk.Frame(card, bg=strip_color, width=4)
    strip.grid(row=0, column=0, rowspan=2, sticky="ns")
    strip.grid_propagate(False)

    body = tk.Frame(card, bg=ELEVATED, highlightthickness=0)
    body.grid(row=0, column=1, sticky="nsew", padx=PAD_SM, pady=PAD_SM)
    body.columnconfigure(0, weight=1)

    header = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew")
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
        text=title,
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_SUMMARY_METRIC_TITLE,
        anchor="w",
    ).grid(row=0, column=1, sticky="w")

    status_lbl = tk.Label(
        body,
        text=_status_display(STATUS_UNAVAILABLE),
        bg=ELEVATED,
        fg=ENGINEERING_STATUS_FG[STATUS_UNAVAILABLE],
        font=FONT_UI_XS,
        anchor="w",
    )
    status_lbl.grid(row=1, column=0, sticky="w", pady=(2, 0))
    setattr(gui, status_attr, status_lbl)

    detail_lbl = tk.Label(
        body,
        text="Not computed yet",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        wraplength=220,
    )
    detail_lbl.grid(row=2, column=0, sticky="ew", pady=(4, 0))
    setattr(gui, detail_attr, detail_lbl)

    def _sync_wrap(_event: object | None = None) -> None:
        if getattr(card, "_wrap_syncing", False):
            return
        from stablewalk.ui.theme import set_wraplength_safe

        setattr(card, "_wrap_syncing", True)
        try:
            width = max(120, int(card.winfo_width()) - 48)
            set_wraplength_safe(detail_lbl, width)
        except tk.TclError:
            pass
        finally:
            setattr(card, "_wrap_syncing", False)

    card.bind("<Configure>", _sync_wrap, add="+")

    spec = resolve_stage_ui_spec(key)
    detail_extra = tk.Frame(body, bg=ELEVATED)
    detail_extra.grid(row=3, column=0, sticky="ew", pady=(8, 0))
    detail_extra.columnconfigure(0, weight=1)

    extra_labels: list[tk.Label] = []
    for idx, (label, text) in enumerate(
        (
            ("Purpose", spec.purpose),
            ("Input", spec.input_desc),
            ("Output", spec.output_desc),
            ("Algorithms", spec.algorithms),
        )
    ):
        extra_label = tk.Label(
            detail_extra,
            text=f"{label}: {text}",
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
            justify=tk.LEFT,
            wraplength=320,
        )
        extra_label.grid(row=idx, column=0, sticky="ew", pady=(0, 2))
        extra_labels.append(extra_label)

    detail_extra.grid_remove()

    def _on_toggle(_event: object | None = None) -> str:
        _toggle_engineering_expanded(gui, key)
        return "break"

    toggle.bind("<Button-1>", _on_toggle)

    gui._adv_engineering_cards[key] = {
        "card": card,
        "strip": strip,
        "status": status_lbl,
        "detail": detail_lbl,
        "toggle": toggle,
        "detail_extra": detail_extra,
        "extra_labels": extra_labels,
        "expanded": False,
    }
    create_tooltip(
        card,
        f"{title}\n\nClick for full stage details · {GLYPH_EXPAND} expands summary",
    )
    bind_pipeline_stage_click(card, gui, key, skip_widgets=(toggle,))
    return card


def build_engineering_dashboard(gui, parent: tk.Misc) -> ttk.Frame:
    """Install the engineering pipeline card grid on the Advanced tab."""
    from stablewalk.ui.tk.dashboard_pipeline_flow import build_pipeline_flow_diagram

    # The parent category already supplies the Real-to-Sim title/collapse chrome.
    section = ttk.Frame(parent)
    section.columnconfigure(0, weight=1)

    flow_host = build_pipeline_flow_diagram(gui, section)
    flow_host.grid(row=0, column=0, sticky="ew", pady=(0, PAD_SM))
    grid_host = tk.Frame(section, bg=PANEL, highlightthickness=0)
    grid_host.grid(row=1, column=0, sticky="ew")
    for col in range(2):
        grid_host.columnconfigure(col, weight=1, uniform="eng_card")

    gui._adv_engineering_cards: dict[str, dict[str, tk.Misc]] = {}

    stage_defs = (
        ("perception", "Perception", "lbl_rts_stage1", "lbl_adv_eng_status_perception", 0, 0),
        ("retargeting", "Retargeting", "lbl_rts_stage2", "lbl_adv_eng_status_retargeting", 0, 1),
        ("biomechanics", "Biomechanics", "lbl_adv_eng_detail_biomechanics", "lbl_adv_eng_status_biomechanics", 1, 0),
        ("virtual_grf", LABEL_VIRTUAL_GRF_PANEL, "lbl_rts_stage4", "lbl_adv_eng_status_virtual_grf", 1, 1),
        ("amp_export", "AMP Export", "lbl_rts_stage3", "lbl_adv_eng_status_amp_export", 2, 0),
        ("opensim", "OpenSim", "lbl_adv_eng_detail_opensim", "lbl_adv_eng_status_opensim", 2, 1),
    )
    for key, title, detail_attr, status_attr, row, col in stage_defs:
        _build_engineering_card(
            grid_host,
            row=row,
            column=col,
            key=key,
            title=title,
            gui=gui,
            detail_attr=detail_attr,
            status_attr=status_attr,
        )

    gui.lbl_rts_summary = tk.Label(
        section,
        text="Run Real-to-Sim Pipeline to export motion for Isaac Lab.",
        bg=PANEL,
        fg=INFO,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=760,
        justify=tk.LEFT,
    )
    gui.lbl_rts_summary.grid(row=2, column=0, sticky="ew", pady=(PAD_SM, 0))

    def _reflow_engineering(event: object | None = None) -> None:
        try:
            width = int(
                getattr(event, "width", 0)
                or section.winfo_width()
                or section.winfo_reqwidth()
                or 0
            )
        except (tk.TclError, TypeError, ValueError):
            return
        if width < 80:
            return
        columns = 1 if width < 680 else 2
        try:
            for column in range(2):
                grid_host.columnconfigure(
                    column,
                    weight=1 if column < columns else 0,
                    minsize=0,
                    uniform="eng_card" if column < columns else "",
                )
            for index, entry in enumerate(gui._adv_engineering_cards.values()):
                row, column = divmod(index, columns)
                entry["card"].grid_configure(row=row, column=column)
                wrap = max(160, width // columns - 56)
                entry["detail"].configure(wraplength=wrap)
                for label in entry.get("extra_labels", ()):
                    label.configure(wraplength=wrap)
            gui.lbl_rts_summary.configure(wraplength=max(220, width - 24))
        except tk.TclError:
            pass
        gui._adv_engineering_columns = columns

    section.bind("<Configure>", _reflow_engineering, add="+")
    gui._sync_engineering_dashboard_wrap = _reflow_engineering

    gui._section_engineering_dashboard = section
    gui._section_real_to_sim = section
    return section


def _collect_engineering_stages(gui) -> dict[str, EngineeringStageView]:
    """Derive engineering stage status from existing session state (no new calculations)."""
    stages: dict[str, EngineeringStageView] = {}

    gait_motion = getattr(gui, "gait_motion", None)
    gait_cycle = getattr(gui, "_gait_cycle", None)
    report = gui._load_rts_report_for_session() if hasattr(gui, "_load_rts_report_for_session") else None
    stage_by_id: dict[str, dict] = {}
    if report:
        for stage in report.get("stages", []):
            stage_by_id[stage.get("stage", "")] = stage

    # Perception
    if gait_motion is None or gait_cycle is None:
        stages["perception"] = EngineeringStageView(
            "perception",
            "Perception",
            STATUS_UNAVAILABLE,
            "Gait style: — (analyze a video first)",
            "Load and analyze a walking video to extract gait style.",
        )
    else:
        s1 = stage_by_id.get("1_perception", {})
        status = _rts_stage_status(s1.get("status")) if s1 else STATUS_COMPLETED
        detail = getattr(gui, "lbl_rts_stage1", None)
        detail_text = detail.cget("text") if detail is not None else "Gait style extracted"
        stages["perception"] = EngineeringStageView(
            "perception",
            "Perception",
            status,
            detail_text,
            "Gait style fingerprint from pose and cycle timing.",
        )

    # Retargeting
    s2 = stage_by_id.get("2_retargeting", {})
    if gait_motion is None:
        stages["retargeting"] = EngineeringStageView(
            "retargeting",
            "Retargeting",
            STATUS_UNAVAILABLE,
            "Human → Unitree G1 scale: —",
            "Humanoid retargeting requires analyzed gait motion.",
        )
    elif s2:
        stages["retargeting"] = EngineeringStageView(
            "retargeting",
            "Retargeting",
            _rts_stage_status(s2.get("status")),
            s2.get("detail", "Retargeting complete")[:120],
            "Scale human motion to Unitree G1 proportions.",
        )
    else:
        detail_lbl = getattr(gui, "lbl_rts_stage2", None)
        detail_text = (
            detail_lbl.cget("text")
            if detail_lbl is not None
            else "Human → Unitree G1: ready (uniform scale from leg length)"
        )
        stages["retargeting"] = EngineeringStageView(
            "retargeting",
            "Retargeting",
            STATUS_PARTIAL,
            detail_text,
            "Retargeting parameters ready; full export may be pending.",
        )

    # Biomechanics
    ba = getattr(gui, "_biomech_analysis", None)
    if ba is None:
        stages["biomechanics"] = EngineeringStageView(
            "biomechanics",
            "Biomechanics",
            STATUS_UNAVAILABLE,
            "Biomechanical analysis not run for this session.",
            "COM, stability margin, symmetry, and contact metrics.",
        )
    else:
        has_core = bool(ba.center_of_mass and ba.stability_margin and ba.contact)
        if has_core and not ba.warnings and not ba.abnormalities:
            bio_status: PipelineStageState = STATUS_COMPLETED
        elif has_core:
            bio_status = STATUS_PARTIAL
        else:
            bio_status = STATUS_PARTIAL
        parts: list[str] = []
        quality_ok = False
        try:
            from stablewalk.analysis.result_validity import (
                build_result_validity,
                withhold_gait_quality_heuristic,
            )

            quality_ok = not withhold_gait_quality_heuristic(
                build_result_validity(
                    biomechanical=ba,
                    contact=getattr(ba, "contact", None),
                    cycles=getattr(ba, "cycles", None),
                )
            )
        except Exception:
            quality_ok = False
        if (
            quality_ok
            and ba.gait_quality is not None
            and getattr(ba.gait_quality, "available", True)
            and ba.gait_quality.score is not None
        ):
            parts.append(
                f"unvalidated heuristic {ba.gait_quality.score:.0f}/100"
            )
        if ba.contact and ba.contact.per_frame:
            hs = sum(
                1
                for f in ba.contact.per_frame
                if f.left_heel_strike or f.right_heel_strike
            )
            parts.append(f"{hs} heel-strike pulse(s)")
        if (
            ba.stability_margin is not None
            and ba.stability_margin.stable_pct is not None
        ):
            parts.append(f"{ba.stability_margin.stable_pct:.0f}% stable frames")
        detail = " · ".join(parts) if parts else "Biomechanical bundle available"
        stages["biomechanics"] = EngineeringStageView(
            "biomechanics",
            "Biomechanics",
            bio_status,
            detail,
            "Foot contact, COM, base of support, and derived gait metrics.",
        )

    # Virtual GRF
    vgrf = getattr(gui, "_virtual_grf", None)
    contact_sync = (report or {}).get("contact_sync")
    if contact_sync:
        mean_r = float(contact_sync.get("mean_reward", 0.0))
        interp = str(contact_sync.get("interpretation", ""))[:80]
        vgrf_status: PipelineStageState = (
            STATUS_COMPLETED if mean_r >= 0.65 else (STATUS_PARTIAL if mean_r >= 0.35 else STATUS_UNAVAILABLE)
        )
        stages["virtual_grf"] = EngineeringStageView(
            "virtual_grf",
            LABEL_VIRTUAL_GRF_PANEL,
            vgrf_status,
            f"Contact sync: {mean_r:.0%} mean — {interp}",
            "Estimated virtual ground reaction forces from pelvis kinematics.",
        )
    elif vgrf is not None and vgrf.available:
        stages["virtual_grf"] = EngineeringStageView(
            "virtual_grf",
            LABEL_VIRTUAL_GRF_PANEL,
            STATUS_COMPLETED if vgrf.confidence >= 0.55 else STATUS_PARTIAL,
            (
                f"{vgrf.estimation_method_label} "
                f"({vgrf.confidence:.0%} confidence)"
            ),
            "Pose-derived vertical loading proxy — not force-plate measured.",
        )
    else:
        detail_lbl = getattr(gui, "lbl_rts_stage4", None)
        detail_text = (
            detail_lbl.cget("text")
            if detail_lbl is not None
            else LABEL_VIRTUAL_GRF_UNAVAILABLE_MSG
        )
        stages["virtual_grf"] = EngineeringStageView(
            "virtual_grf",
            LABEL_VIRTUAL_GRF_PANEL,
            STATUS_UNAVAILABLE,
            detail_text,
            "Requires pose sequence and foot contact timing.",
        )

    # AMP Export
    run_name = "session"
    if hasattr(gui, "_resolve_session_video_source"):
        run_name = Path(gui._resolve_session_video_source() or "session").stem or "session"
    amp_path = config.MOTION_REFERENCE_EXPORT_DIR / run_name / "amp_reference_motion.npz"
    s3 = stage_by_id.get("3_simulation_amp", {})
    if amp_path.is_file():
        amp_status: PipelineStageState = (
            STATUS_COMPLETED
            if (not s3 or s3.get("status") in ("complete", "completed", None))
            else STATUS_PARTIAL
        )
        stages["amp_export"] = EngineeringStageView(
            "amp_export",
            "AMP Export",
            amp_status,
            f"AMP reference exported ({amp_path.name})",
            "Isaac Lab / AMP reference motion artifact.",
        )
    elif s3 and s3.get("status") in ("complete", "partial", "completed"):
        stages["amp_export"] = EngineeringStageView(
            "amp_export",
            "AMP Export",
            _rts_stage_status(s3.get("status")),
            s3.get("detail", "AMP export reported in pipeline")[:120],
            "Isaac Lab / AMP reference motion artifact.",
        )
    else:
        stages["amp_export"] = EngineeringStageView(
            "amp_export",
            "AMP Export",
            STATUS_UNAVAILABLE,
            "AMP reference: click Export AMP Reference or Real-to-Sim Pipeline",
            "Export amp_reference_motion.npz for reinforcement-learning pipelines.",
        )

    # OpenSim card: pipeline status is independent of IK.
    sdk = bool(getattr(gui, "_opensim_sdk_available", False))
    presentation = bool(getattr(gui, "_presentation_mode", False))
    has_session = bool(getattr(gui, "sequence", None) or getattr(gui, "gait_motion", None))
    export_complete = bool(getattr(gui, "_opensim_export_completed", False))
    pair = resolve_opensim_status_pair_from_gui(gui)
    pipeline = pair.pipeline
    ik = pair.ik

    if presentation:
        os_status: PipelineStageState = STATUS_PARTIAL
        os_detail = "Presentation demo mode — synthetic workflow"
    elif pipeline.key == PIPELINE_COMPLETE and ik.key == IK_COMPLETE:
        os_status = STATUS_COMPLETED
        os_detail = pair.short_label
    elif pipeline.key == PIPELINE_COMPLETE:
        os_status = STATUS_PARTIAL
        os_detail = pair.short_label
    elif ik.key in (IK_RUNNING, IK_FAILED, IK_PARTIAL) or sdk or export_complete or has_session:
        os_status = STATUS_PARTIAL
        os_detail = pair.short_label
    else:
        os_status = STATUS_UNAVAILABLE
        os_detail = pair.short_label if pair.short_label else "OpenSim not configured for this session"

    stages["opensim"] = EngineeringStageView(
        "opensim",
        "OpenSim",
        os_status,
        os_detail,
        pair.tooltip,
    )

    return stages


@dataclass(frozen=True)
class OpensimStatusRow:
    """One compact OpenSim readout (pipeline snapshot only)."""

    key: str
    title: str
    value: str
    status: PipelineStageState
    tooltip: str = ""


def _ik_mot_row_value(ik, files: dict) -> str:
    reason = ik_mot_unavailable_reason(
        ik, ik_output_on_disk=bool((files or {}).get("IK_MOT"))
    )
    return reason or "Unavailable — Inverse Kinematics has not completed"


def _ik_mot_row_tooltip(ik, files: dict) -> str:
    reason = _ik_mot_row_value(ik, files)
    return (
        f"{reason}\n\nExport completion is not IK success. "
        "Open the diagnostic log from Advanced / OpenSim details."
    )


def collect_opensim_status_rows(gui) -> tuple[OpensimStatusRow, ...]:
    """SDK / model / TRC / MOT / IK from the current pipeline report."""
    report = getattr(gui, "_pipeline_status_report_cache", None)
    assess = getattr(gui, "_assess_pipeline_status_report", None)
    if report is None and callable(assess):
        try:
            report = assess(force=False)
        except Exception:
            report = None
    items = {it.key: it for it in report.all_items()} if report is not None else {}

    sdk_on = bool(getattr(gui, "_opensim_sdk_available", False))
    sdk_ver = getattr(gui, "_opensim_sdk_version", None)
    if sdk_on:
        sdk_value = f"Installed{f' · {sdk_ver}' if sdk_ver else ''}"
        sdk_status: PipelineStageState = STATUS_COMPLETED
        sdk_tip = "OpenSim Python SDK is available."
    else:
        sdk_value = "Not installed"
        sdk_status = STATUS_UNAVAILABLE
        sdk_tip = "OpenSim SDK is not available in this environment."

    model_item = items.get("osim_model")
    model_name = str(getattr(gui, "_opensim_model_name", None) or "").strip()
    if model_item is not None:
        model_status = model_item.status
        model_value = model_name or model_item.detail or "Not assessed"
        model_tip = model_item.tooltip or model_item.detail
    elif model_name:
        model_status = (
            STATUS_COMPLETED
            if getattr(gui, "_opensim_model_valid", False)
            else STATUS_PARTIAL
        )
        model_value = model_name
        model_tip = "Configured OpenSim musculoskeletal model."
    else:
        model_status = STATUS_UNAVAILABLE
        model_value = "No model loaded"
        model_tip = "No OpenSim .osim model is loaded."

    def _artifact(key: str, fallback: str) -> tuple[str, PipelineStageState, str]:
        item = items.get(key)
        if item is None:
            return fallback, STATUS_UNAVAILABLE, fallback
        return (
            ENGINEERING_STATUS_LABEL.get(item.status, item.display_line()),
            item.status,
            item.detail or item.tooltip or item.display_line(),
        )

    trc_value, trc_status, trc_tip = _artifact("osim_trc", "Unavailable")
    mot_value, mot_status, mot_tip = _artifact("osim_mot", "Unavailable")
    pipeline = resolve_opensim_pipeline_status_from_gui(gui)
    pair = resolve_opensim_status_pair_from_gui(gui)
    ik = pair.ik
    files = {}
    getter = getattr(gui, "_opensim_export_files_on_disk", None)
    if callable(getter):
        try:
            files = dict(getter() or {})
        except Exception:
            files = {}
    if not files:
        cached = getattr(gui, "_opensim_export_files", None)
        if isinstance(cached, dict):
            files = cached

    mapped_ready = bool(files.get("MAPPED"))
    ik_complete = ik.key == "complete"

    return (
        OpensimStatusRow(
            "pipeline",
            "Pipeline",
            pipeline.label,
            pipeline.pipeline_status,
            pipeline.tooltip,
        ),
        OpensimStatusRow("sdk", "SDK", sdk_value, sdk_status, sdk_tip),
        OpensimStatusRow("model", "Model", model_value, model_status, model_tip),
        OpensimStatusRow("trc", "TRC", trc_value, trc_status, trc_tip),
        OpensimStatusRow("mot", "MOT", mot_value, mot_status, mot_tip),
        OpensimStatusRow(
            "mapped",
            "Mapped TRC",
            "Complete" if mapped_ready else "Unavailable",
            STATUS_COMPLETED if mapped_ready else STATUS_UNAVAILABLE,
            "Mapped marker TRC ready for experimental IK."
            if mapped_ready
            else "Mapped TRC has not been exported.",
        ),
        OpensimStatusRow(
            "ik_output",
            "IK .mot",
            "Complete" if ik_complete else _ik_mot_row_value(ik, files),
            STATUS_COMPLETED if ik_complete else STATUS_UNAVAILABLE,
            "IK motion file from a successful run in this session."
            if ik_complete
            else _ik_mot_row_tooltip(ik, files),
        ),
        OpensimStatusRow(
            "ik",
            "IK",
            ik.label,
            ik.pipeline_status,
            pair.tooltip,
        ),
    )


def build_opensim_status_panel(gui, parent: tk.Misc) -> tk.Frame:
    """Compact OpenSim readout for the Advanced tab (no extra compute)."""
    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.columnconfigure(1, weight=1)
    tk.Label(
        host,
        text="OpenSim status",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 2))

    widgets: dict[str, tk.Label] = {}
    for index, row in enumerate(collect_opensim_status_rows(gui), start=1):
        tk.Label(
            host,
            text=row.title,
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
            width=9,
        ).grid(row=index, column=0, sticky="w", padx=(0, PAD_SM))
        value = tk.Label(
            host,
            text=row.value,
            bg=PANEL,
            fg=_opensim_row_fg(row),
            font=FONT_UI_XS,
            anchor="w",
            justify=tk.LEFT,
            wraplength=280,
        )
        value.grid(row=index, column=1, sticky="ew")
        widgets[row.key] = value
        if row.tooltip:
            create_tooltip(value, row.tooltip, wraplength=360)
    log_btn = ttk.Button(
        host,
        text="View diagnostic log",
        style="Compact.TButton",
        command=lambda: getattr(gui, "_show_opensim_log", lambda: None)(),
    )
    log_btn.grid(row=len(widgets) + 1, column=0, columnspan=2, sticky="ew", pady=(PAD_SM, 0))
    gui.btn_adv_opensim_view_log = log_btn
    gui._adv_opensim_status_values = widgets
    gui._adv_opensim_status_host = host
    return host


def update_opensim_status_panel(gui) -> None:
    """Refresh the Advanced OpenSim status rows from the pipeline snapshot."""
    widgets = getattr(gui, "_adv_opensim_status_values", None)
    if not widgets:
        return
    for row in collect_opensim_status_rows(gui):
        lbl = widgets.get(row.key)
        if lbl is None:
            continue
        try:
            lbl.configure(
                text=row.value,
                fg=_opensim_row_fg(row),
            )
        except tk.TclError:
            continue
        if row.tooltip:
            create_tooltip(lbl, row.tooltip, wraplength=360)


def _apply_stage_view(gui, view: EngineeringStageView) -> None:
    cards = getattr(gui, "_adv_engineering_cards", {})
    entry = cards.get(view.key)
    if entry is None:
        return

    status_lbl = entry.get("status")
    strip = entry.get("strip")
    detail_lbl = entry.get("detail")

    if status_lbl is not None:
        status_lbl.configure(
            text=_status_display(view.status),
            fg=ENGINEERING_STATUS_FG[view.status],
        )
    if strip is not None:
        strip.configure(bg=ENGINEERING_STATUS_STRIP[view.status])
    if detail_lbl is not None:
        color = TEXT if view.status == STATUS_COMPLETED else (
            WARNING if view.status == STATUS_PARTIAL else MUTED
        )
        detail_lbl.configure(text=view.detail, fg=color)
        tip = view.tooltip or view.detail
        create_tooltip(detail_lbl, tip)
    card = entry.get("card")
    if card is not None:
        create_tooltip(card, f"{view.title}\n\n{view.tooltip}\n\n{view.detail}")


def update_engineering_dashboard(gui) -> None:
    """Refresh engineering stage cards from the current session."""
    from stablewalk.ui.tk.dashboard_pipeline_flow import update_pipeline_flow_diagram

    update_pipeline_flow_diagram(gui)
    stages = _collect_engineering_stages(gui)
    for view in stages.values():
        _apply_stage_view(gui, view)
    update_opensim_status_panel(gui)

    bio_detail = stages.get("biomechanics")
    if bio_detail is not None:
        bio_lbl = getattr(gui, "lbl_adv_eng_detail_biomechanics", None)
        if bio_lbl is not None:
            bio_lbl.configure(text=bio_detail.detail)

    os_detail = stages.get("opensim")
    if os_detail is not None:
        os_lbl = getattr(gui, "lbl_adv_eng_detail_opensim", None)
        if os_lbl is not None:
            os_lbl.configure(text=os_detail.detail)


__all__ = [
    "ENGINEERING_STATUS_LABEL",
    "ENGINEERING_STATUS_SYMBOL",
    "EngineeringStageView",
    "OpensimStatusRow",
    "build_engineering_dashboard",
    "build_opensim_status_panel",
    "collect_opensim_status_rows",
    "update_engineering_dashboard",
    "update_opensim_status_panel",
]
