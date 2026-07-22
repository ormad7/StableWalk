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
from stablewalk.ui.scientific_labels import (
    LABEL_VIRTUAL_GRF_PANEL,
    LABEL_VIRTUAL_GRF_UNAVAILABLE_MSG,
)
from stablewalk.ui.theme import (
    BORDER,
    DASHBOARD_CARD_PAD,
    ELEVATED,
    FONT_SUMMARY_METRIC_TITLE,
    FONT_UI_SM,
    FONT_UI_XS,
    INFO,
    MUTED,
    PAD_MD,
    PAD_SM,
    PANEL,
    TEXT,
    WARNING,
    create_tooltip,
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
    STATUS_COMPLETED: "\u2713",  # ✓
    STATUS_PARTIAL: "\u26a0",  # ⚠
    STATUS_UNAVAILABLE: "\u2717",  # ✗
}

ENGINEERING_STATUS_LABEL = {
    STATUS_COMPLETED: "Completed",
    STATUS_PARTIAL: "Partial",
    STATUS_UNAVAILABLE: "Missing",
}

ENGINEERING_STATUS_FG = PIPELINE_STATUS_FG
ENGINEERING_STATUS_STRIP = PIPELINE_STATUS_STRIP


@dataclass(frozen=True)
class EngineeringStageView:
    key: str
    title: str
    status: PipelineStageState
    detail: str
    tooltip: str = ""


def _status_display(status: PipelineStageState) -> str:
    return (
        f"{ENGINEERING_STATUS_SYMBOL[status]} "
        f"{ENGINEERING_STATUS_LABEL[status]}"
    )


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
        toggle.configure(text="\u25bc" if expanded else "\u25b6")


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
    card = tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
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
    body.grid(row=0, column=1, sticky="nsew", padx=PAD_MD, pady=PAD_MD)
    body.columnconfigure(0, weight=1)

    header = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
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
        text="—",
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
        try:
            width = max(120, int(card.winfo_width()) - 48)
            detail_lbl.configure(wraplength=width)
        except tk.TclError:
            pass

    card.bind("<Configure>", _sync_wrap, add="+")
    body.bind("<Configure>", _sync_wrap, add="+")

    spec = resolve_stage_ui_spec(key)
    detail_extra = tk.Frame(body, bg=ELEVATED)
    detail_extra.grid(row=3, column=0, sticky="ew", pady=(8, 0))
    detail_extra.columnconfigure(0, weight=1)

    for idx, (label, text) in enumerate(
        (
            ("Purpose", spec.purpose),
            ("Input", spec.input_desc),
            ("Output", spec.output_desc),
            ("Algorithms", spec.algorithms),
        )
    ):
        tk.Label(
            detail_extra,
            text=f"{label}: {text}",
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
            justify=tk.LEFT,
            wraplength=320,
        ).grid(row=idx, column=0, sticky="ew", pady=(0, 2))

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
        "expanded": False,
    }
    create_tooltip(
        card,
        f"{title}\n\nClick for full stage details · ▶ expands summary",
    )
    bind_pipeline_stage_click(card, gui, key, skip_widgets=(toggle,))
    return card


def build_engineering_dashboard(gui, parent: tk.Misc) -> ttk.LabelFrame:
    """Install the engineering pipeline card grid on the Advanced tab."""
    from stablewalk.ui.tk.dashboard_pipeline_flow import build_pipeline_flow_diagram

    section = ttk.LabelFrame(
        parent,
        text="  Real-to-Sim Pipeline  ",
        style="Card.TLabelframe",
        padding=DASHBOARD_CARD_PAD,
    )
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
        text="Run Real-to-Sim Pipeline below to export motion for Isaac Lab.",
        bg=PANEL,
        fg=INFO,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=760,
        justify=tk.LEFT,
    )
    gui.lbl_rts_summary.grid(row=2, column=0, sticky="ew", pady=(PAD_SM, 0))

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
        if ba.gait_quality is not None:
            parts.append(f"gait quality {ba.gait_quality.score:.0f}/100")
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

    # OpenSim
    sdk = bool(getattr(gui, "_opensim_sdk_available", False))
    presentation = bool(getattr(gui, "_presentation_mode", False))
    has_session = bool(getattr(gui, "sequence", None) or getattr(gui, "gait_motion", None))
    export_complete = bool(getattr(gui, "_opensim_export_completed", False))
    model_valid = bool(getattr(gui, "_opensim_model_valid", False))

    if presentation:
        os_status: PipelineStageState = STATUS_PARTIAL
        os_detail = "Presentation demo mode — synthetic workflow"
    elif sdk and model_valid and export_complete:
        os_status = STATUS_COMPLETED
        os_detail = "SDK ready · model loaded · session exported"
    elif sdk and (model_valid or export_complete):
        os_status = STATUS_PARTIAL
        os_detail = "SDK ready · " + ("model loaded" if model_valid else "model not loaded")
        if export_complete:
            os_detail += " · export complete"
        else:
            os_detail += " · export pending"
    elif sdk:
        os_status = STATUS_PARTIAL
        os_detail = "SDK installed — load model and export session files"
    elif export_complete or has_session:
        os_status = STATUS_PARTIAL
        os_detail = "Export-only mode — OpenSim SDK not installed"
    else:
        os_status = STATUS_UNAVAILABLE
        os_detail = "OpenSim not configured for this session"

    stages["opensim"] = EngineeringStageView(
        "opensim",
        "OpenSim",
        os_status,
        os_detail,
        "OpenSim SDK, model loading, and motion export status.",
    )

    return stages


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
    "build_engineering_dashboard",
    "update_engineering_dashboard",
]
