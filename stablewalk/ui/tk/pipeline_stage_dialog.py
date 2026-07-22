"""Clickable Real-to-Sim pipeline stage detail dialogs (UI only — no pipeline logic)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tkinter as tk
from tkinter import ttk

from stablewalk import config
from stablewalk.monitoring.pipeline_status import (
    STATUS_COMPLETED,
    STATUS_LABEL,
    STATUS_PARTIAL,
    STATUS_SYMBOL,
    STATUS_UNAVAILABLE,
    PIPELINE_DIAGRAM_STAGE_SPECS,
    PipelineStageItem,
    PipelineStageState,
    PipelineStatusReport,
)
from stablewalk.ui.theme import (
    ACCENT,
    BORDER,
    ELEVATED,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    PANEL,
    TEXT,
)
from stablewalk.ui.tk.dashboard_pipeline_visual import PIPELINE_STATUS_FG

if TYPE_CHECKING:
    from stablewalk.monitoring.pipeline_status import PipelineDiagramStage


@dataclass(frozen=True)
class PipelineStageUiSpec:
    """Static copy for a pipeline stage dialog."""

    key: str
    title: str
    purpose: str
    input_desc: str
    output_desc: str
    algorithms: str = "—"
    item_keys: tuple[str, ...] = ()
    rts_stage_ids: tuple[str, ...] = ()
    artifact_names: tuple[str, ...] = ()


# Canonical stage metadata — presentation only.
_STAGE_UI_SPECS: dict[str, PipelineStageUiSpec] = {
    "video": PipelineStageUiSpec(
        key="video",
        title="Video",
        purpose="Load a walking video and extract 2D/3D pose landmarks for downstream analysis.",
        input_desc="Walking video file, stream URL, or verified pose JSON session.",
        output_desc="Detected pose frames, gait motion recording, and session metadata.",
        algorithms="MediaPipe Pose / BlazePose · frame decode · landmark filtering",
        item_keys=("pose_mediapipe",),
        rts_stage_ids=("1_perception",),
        artifact_names=("*_poses.json",),
    ),
    "pose_estimation": PipelineStageUiSpec(
        key="pose_estimation",
        title="Pose Estimation",
        purpose="Extract 2D/3D body pose from video using MediaPipe, SMPL, or ROMP providers.",
        input_desc="Walking video or pre-recorded pose sequence.",
        output_desc="Landmark poses, optional SMPL mesh motion, and pose backend report.",
        algorithms="MediaPipe Pose · SMPL/SMPL-X · ROMP (optional backends)",
        item_keys=("pose_mediapipe", "pose_smpl", "pose_romp"),
        rts_stage_ids=("1_perception",),
        artifact_names=("*_poses.json", "smpl_motion.npz"),
    ),
    "perception": PipelineStageUiSpec(
        key="perception",
        title="Perception",
        purpose="Capture the visual gait fingerprint: stride, cadence, hip sway, and arm swing.",
        input_desc="Analyzed pose sequence and gait cycle features.",
        output_desc="Gait style fingerprint embedded in stablewalk_motion.npz.",
        algorithms="Gait cycle segmentation · stride/cadence estimators · style fingerprinting",
        item_keys=("pose_mediapipe", "motion_skeleton", "bio_cadence"),
        rts_stage_ids=("1_perception",),
        artifact_names=("stablewalk_motion.npz",),
    ),
    "smpl": PipelineStageUiSpec(
        key="smpl",
        title="SMPL",
        purpose="Optional monocular SMPL body mesh inference for richer 3D body shape.",
        input_desc="Video frames and licensed SMPL model assets.",
        output_desc="SMPL pose parameters and smpl_motion.npz when exported.",
        algorithms="SMPL / SMPL-X regression · monocular mesh fitting",
        item_keys=("pose_smpl", "pose_romp"),
        artifact_names=("smpl_motion.npz",),
    ),
    "reconstruction_3d": PipelineStageUiSpec(
        key="reconstruction_3d",
        title="3D Reconstruction",
        purpose="Build a hip-centered 3D skeleton, joint angles, and kinematic velocities.",
        input_desc="Pose landmarks across the analyzed clip.",
        output_desc="GaitMotionRecording snapshots with 3D joints, angles, and velocities.",
        algorithms="Hip-centered kinematics · joint angle IK · finite-difference velocities",
        item_keys=("motion_skeleton", "motion_angles", "motion_velocities"),
        artifact_names=("stablewalk_motion.npz",),
    ),
    "retargeting": PipelineStageUiSpec(
        key="retargeting",
        title="Retargeting",
        purpose="Scale human motion to humanoid robot proportions and map joint angles to robot DOFs.",
        input_desc="Human motion reference (stablewalk_motion.npz) and retarget config.",
        output_desc="retargeted_motion.npz ready for simulation pipelines.",
        algorithms="Anthropometric scaling · DOF remapping · joint limit clamping",
        item_keys=("retarget_scale", "retarget_joints", "retarget_export"),
        rts_stage_ids=("2_retargeting",),
        artifact_names=("retargeted_motion.npz", "stablewalk_motion.npz"),
    ),
    "opensim": PipelineStageUiSpec(
        key="opensim",
        title="OpenSim",
        purpose="Map landmarks to OpenSim markers, run inverse kinematics, and export musculoskeletal motion.",
        input_desc="Pose TRC, OpenSim .osim model, and marker mapping configuration.",
        output_desc="Mapped .trc, inverse kinematics .mot, and coverage report.",
        algorithms="Marker mapping · OpenSim Inverse Kinematics · coverage scoring",
        item_keys=(
            "osim_model",
            "osim_trc",
            "osim_mot",
            "osim_ik",
            "osim_mapping",
            "osim_coverage",
        ),
        artifact_names=("*.trc", "*_ik.mot", "*_mapped.trc"),
    ),
    "biomechanics": PipelineStageUiSpec(
        key="biomechanics",
        title="Biomechanics",
        purpose="Estimate gait events, foot contact, cadence, center of mass, and base of support from pose.",
        input_desc="Gait motion recording and foot-contact heuristics.",
        output_desc="Contact masks, gait events, COM/BoS trajectories, and temporal metrics.",
        algorithms="Foot-contact heuristics · heel-strike/toe-off detection · COM/BoS geometry",
        item_keys=(
            "bio_contact_mask",
            "bio_heel_strike",
            "bio_toe_off",
            "bio_cadence",
            "bio_com",
            "bio_bos",
        ),
        rts_stage_ids=("5_biomechanical",),
        artifact_names=("biomechanical_report.json",),
    ),
    "virtual_grf": PipelineStageUiSpec(
        key="virtual_grf",
        title="Estimated Virtual GRF",
        purpose="Provide a pose-based vertical ground reaction force proxy synchronized with contact timing.",
        input_desc="Contact masks, COM kinematics, and body mass estimate.",
        output_desc="Estimated virtual GRF time series (not force-plate measured).",
        algorithms="Contact-synchronized COM acceleration proxy · vertical GRF estimator",
        item_keys=("bio_vgrf",),
        rts_stage_ids=("4_physics_vgrf",),
        artifact_names=("contact_sync.npz", "virtual_grf.npz"),
    ),
    "amp_dataset": PipelineStageUiSpec(
        key="amp_dataset",
        title="AMP Dataset",
        purpose="Prepare AMP reference motion for adversarial motion prior training in Isaac Lab.",
        input_desc="Retargeted or motion-reference NPZ and Isaac Lab export settings.",
        output_desc="amp_reference_motion.npz and amp_reference_manifest.json.",
        algorithms="AMP observation packing · reference motion serialization",
        item_keys=("isaac_export_only",),
        rts_stage_ids=("3_simulation_amp",),
        artifact_names=("amp_reference_motion.npz", "amp_reference_manifest.json"),
    ),
    "amp_export": PipelineStageUiSpec(
        key="amp_export",
        title="AMP Export",
        purpose="Export AMP reference motion for external Isaac Lab / AMP training workflows.",
        input_desc="Motion reference bundle and gait style fingerprint.",
        output_desc="amp_reference_motion.npz for offline Isaac Lab training.",
        algorithms="AMP reference export · manifest writer",
        item_keys=("isaac_export_only",),
        rts_stage_ids=("3_simulation_amp",),
        artifact_names=("amp_reference_motion.npz", "amp_reference_manifest.json"),
    ),
    "export": PipelineStageUiSpec(
        key="export",
        title="Export",
        purpose="Write session artifacts for downstream simulation, analysis, and sharing.",
        input_desc="Completed analysis stages and export configuration.",
        output_desc="stablewalk_motion.npz, retargeted_motion.npz, AMP reference, and OpenSim exports.",
        algorithms="Motion NPZ writer · Real-to-Sim report aggregation",
        item_keys=("retarget_export", "isaac_export_only", "osim_ik"),
        artifact_names=(
            "stablewalk_motion.npz",
            "retargeted_motion.npz",
            "amp_reference_motion.npz",
            "real_to_sim_pipeline_report.json",
        ),
    ),
    "isaac_lab": PipelineStageUiSpec(
        key="isaac_lab",
        title="Isaac Lab",
        purpose="NVIDIA Isaac Lab simulation environment and optional RL/AMP training (external to StableWalk).",
        input_desc="AMP reference motion and Isaac Lab installation.",
        output_desc="Environment readiness check and export-only motion artifacts.",
        algorithms="Isaac Lab env probe · AMP training bridge (external)",
        item_keys=("isaac_env_ready", "isaac_export_only", "isaac_training_running", "isaac_training_complete"),
        artifact_names=("amp_reference_motion.npz",),
    ),
}

# Aliases from flow / engineering cards to canonical specs.
_STAGE_ALIASES: dict[str, str] = {
    "video": "video",
    "smpl": "smpl",
    "retargeting": "retargeting",
    "opensim": "opensim",
    "biomechanics": "biomechanics",
    "virtual_grf": "virtual_grf",
    "amp_dataset": "amp_dataset",
    "export": "export",
    "pose_estimation": "pose_estimation",
    "reconstruction_3d": "reconstruction_3d",
    "isaac_lab": "isaac_lab",
    "perception": "perception",
    "amp_export": "amp_export",
}


@dataclass(frozen=True)
class PipelineStageDialogContent:
    """Resolved dialog payload for one pipeline stage."""

    stage_key: str
    title: str
    purpose: str
    input_desc: str
    output_desc: str
    algorithms: str
    status: PipelineStageState
    status_line: str
    evidence: str
    generated_files: str
    confidence: str
    execution_summary: str
    duration_text: str = "—"


def resolve_stage_ui_spec(stage_key: str) -> PipelineStageUiSpec:
    canonical = _STAGE_ALIASES.get(stage_key, stage_key)
    if canonical in _STAGE_UI_SPECS:
        return _STAGE_UI_SPECS[canonical]

    for key, label, item_keys, tooltip in PIPELINE_DIAGRAM_STAGE_SPECS:
        if key == stage_key:
            return PipelineStageUiSpec(
                key=key,
                title=label,
                purpose=tooltip,
                input_desc="Prior pipeline stage outputs for the current session.",
                output_desc="Stage artifacts and in-memory analysis results.",
                algorithms="Stage-specific analysis algorithms",
                item_keys=item_keys,
            )
    return PipelineStageUiSpec(
        key=stage_key,
        title=stage_key.replace("_", " ").title(),
        purpose="Real-to-Sim pipeline stage.",
        input_desc="Session inputs from prior stages.",
        output_desc="Stage outputs when executed.",
        algorithms="—",
        item_keys=(),
    )


def _aggregate_status(items: list[PipelineStageItem]) -> tuple[PipelineStageState, str]:
    if not items:
        return STATUS_UNAVAILABLE, "Not assessed"
    active = [item for item in items if item.status != STATUS_UNAVAILABLE]
    pool = active or items
    statuses = [item.status for item in pool]
    if statuses and all(s == STATUS_COMPLETED for s in statuses):
        state = STATUS_COMPLETED
    elif any(s == STATUS_COMPLETED for s in statuses) or any(s == STATUS_PARTIAL for s in statuses):
        state = STATUS_PARTIAL
    else:
        state = STATUS_UNAVAILABLE
    parts = [f"{item.label}: {item.detail}" for item in pool[:4] if item.detail]
    if len(pool) > 4:
        parts.append(f"+{len(pool) - 4} more sub-steps")
    return state, "; ".join(parts) if parts else "No execution evidence"


def _format_items_evidence(items: list[PipelineStageItem]) -> str:
    if not items:
        return "No monitored sub-steps matched this stage."
    lines = [
        f"{STATUS_SYMBOL[item.status]} {item.label} — {item.detail}"
        for item in items
    ]
    return "\n".join(lines)


def _session_run_name(gui: Any, report: PipelineStatusReport | None) -> str:
    if report and report.run_name:
        return report.run_name
    if gui is not None:
        run = getattr(gui, "_active_run_name", None)
        if run:
            return str(run)
        if hasattr(gui, "_resolve_session_video_source"):
            src = gui._resolve_session_video_source()
            if src:
                return Path(str(src)).stem or "session"
    return "session"


def _find_artifacts(run_name: str, patterns: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    search_roots = [
        config.MOTION_REFERENCE_EXPORT_DIR / run_name,
        config.OPENSIM_DIR / run_name,
    ]
    pose_json = config.POSES_DIR / f"{run_name}_poses.json"
    if pose_json.is_file():
        found.append(str(pose_json.relative_to(config.PROJECT_ROOT)))

    for pattern in patterns:
        if pattern == "*_poses.json":
            continue
        for root in search_roots:
            if not root.is_dir():
                continue
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    rel = str(path.relative_to(config.PROJECT_ROOT))
                    if rel not in found:
                        found.append(rel)
    return found


def _confidence_for_stage(stage_key: str, gui: Any, items: list[PipelineStageItem]) -> str:
    if gui is None:
        return "—"

    contact = getattr(gui, "_foot_contact", None)
    vgrf = getattr(gui, "_estimated_vgrf", None)
    sequence = getattr(gui, "sequence", None)
    pose_indices = getattr(gui, "pose_indices", None) or []

    if stage_key in ("video", "pose_estimation", "perception", "smpl"):
        if sequence and pose_indices:
            detected = sum(1 for i in pose_indices if sequence.frames[i].detected)
            total = len(pose_indices)
            if total:
                pct = 100.0 * detected / total
                return f"Pose detection coverage: {detected}/{total} frames ({pct:.0f}%)"
        return "No pose session loaded"

    if stage_key in ("biomechanics", "virtual_grf") and contact is not None:
        conf = contact.metrics.contact_confidence
        return f"Foot contact confidence: {conf:.0%}"

    if stage_key == "virtual_grf" and vgrf is not None and len(vgrf.confidence) > 0:
        import numpy as np

        mean_conf = float(np.nanmean(vgrf.confidence))
        return f"Virtual GRF estimator confidence: {mean_conf:.0%}"

    if stage_key == "opensim":
        ctx = None
        if hasattr(gui, "_build_pipeline_status_context"):
            try:
                ctx = gui._build_pipeline_status_context()
            except Exception:
                ctx = None
        if ctx is not None and ctx.opensim_coverage_percent is not None:
            return f"OpenSim marker coverage: {ctx.opensim_coverage_percent:.0f}%"

    if items:
        completed = sum(1 for item in items if item.status == STATUS_COMPLETED)
        return f"Sub-step completion: {completed}/{len(items)} monitored items complete"
    return "—"


def _rts_execution_summary(gui: Any, rts_stage_ids: tuple[str, ...]) -> str:
    if not rts_stage_ids:
        return "Real-to-Sim pipeline has not been run for this stage in the current session."
    report = None
    if gui is not None and hasattr(gui, "_load_rts_report_for_session"):
        report = gui._load_rts_report_for_session()
    if not report:
        return "Real-to-Sim pipeline report not found on disk for this session."

    lines: list[str] = []
    stages = report.get("stages") or []
    for stage_id in rts_stage_ids:
        match = next((s for s in stages if s.get("stage") == stage_id), None)
        if match is None:
            continue
        status = str(match.get("status") or "unknown")
        detail = str(match.get("detail") or "")
        out = match.get("output_path")
        line = f"{stage_id}: {status}"
        if detail:
            line += f" — {detail}"
        if out:
            line += f" → {out}"
        lines.append(line)

    if lines:
        return "\n".join(lines)

    summary_bits = []
    if report.get("run_name"):
        summary_bits.append(f"Run: {report['run_name']}")
    if report.get("isaac_lab_note"):
        summary_bits.append(str(report["isaac_lab_note"]))
    return "\n".join(summary_bits) if summary_bits else "Pipeline report present; no stage-specific execution lines."


def _rts_stage_duration(gui: Any, rts_stage_ids: tuple[str, ...]) -> float | None:
    if not rts_stage_ids or gui is None or not hasattr(gui, "_load_rts_report_for_session"):
        return None
    report = gui._load_rts_report_for_session()
    if not report:
        return None
    total = 0.0
    found = False
    for stage_id in rts_stage_ids:
        match = next((s for s in (report.get("stages") or []) if s.get("stage") == stage_id), None)
        if match is None:
            continue
        raw = match.get("duration_s")
        if raw is None:
            continue
        try:
            total += float(raw)
            found = True
        except (TypeError, ValueError):
            continue
    return total if found else None


def build_pipeline_stage_dialog_content(
    stage_key: str,
    report: PipelineStatusReport | None,
    *,
    gui: Any = None,
    diagram_stage: PipelineDiagramStage | None = None,
) -> PipelineStageDialogContent:
    """Assemble dialog fields from cached status — read-only, no pipeline mutations."""
    spec = resolve_stage_ui_spec(stage_key)
    item_keys = spec.item_keys
    if diagram_stage is not None and diagram_stage.item_keys:
        item_keys = diagram_stage.item_keys

    items_by_key: dict[str, PipelineStageItem] = {}
    if report is not None:
        items_by_key = {item.key: item for item in report.all_items()}

    matched = [items_by_key[k] for k in item_keys if k in items_by_key]
    status, summary = _aggregate_status(matched)
    if diagram_stage is not None:
        status = diagram_stage.status
        summary = diagram_stage.detail

    run_name = _session_run_name(gui, report)
    artifacts = _find_artifacts(run_name, spec.artifact_names)
    files_text = "\n".join(f"• {path}" for path in artifacts) if artifacts else "No matching files on disk yet."

    from stablewalk.ui.tk.dashboard_pipeline_visual import format_duration_seconds

    duration_s = _rts_stage_duration(gui, spec.rts_stage_ids)
    duration_text = format_duration_seconds(duration_s)

    return PipelineStageDialogContent(
        stage_key=stage_key,
        title=diagram_stage.label if diagram_stage is not None else spec.title,
        purpose=diagram_stage.tooltip if diagram_stage is not None else spec.purpose,
        input_desc=spec.input_desc,
        output_desc=spec.output_desc,
        algorithms=spec.algorithms,
        status=status,
        status_line=f"{STATUS_SYMBOL[status]} {STATUS_LABEL[status]} — {summary}",
        evidence=_format_items_evidence(matched),
        generated_files=files_text,
        confidence=_confidence_for_stage(stage_key, gui, matched),
        execution_summary=_rts_execution_summary(gui, spec.rts_stage_ids),
        duration_text=duration_text,
    )


def _add_dialog_section(parent: tk.Misc, row: int, label: str, body: str, *, fg: str = TEXT) -> int:
    tk.Label(
        parent,
        text=label,
        bg=ELEVATED,
        fg=ACCENT,
        font=(FONT_UI_SM[0], FONT_UI_SM[1], "bold"),
        anchor="w",
    ).grid(row=row, column=0, sticky="w", pady=(0, 2))
    tk.Label(
        parent,
        text=body,
        bg=ELEVATED,
        fg=fg,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=520,
    ).grid(row=row + 1, column=0, sticky="ew", pady=(0, 10))
    return row + 2


def show_pipeline_stage_dialog(parent: tk.Misc, content: PipelineStageDialogContent) -> None:
    """Open a modal stage detail dialog."""
    dlg = tk.Toplevel(parent)
    dlg.title(f"Pipeline — {content.title}")
    dlg.configure(bg=PANEL)
    dlg.transient(parent.winfo_toplevel())
    dlg.resizable(True, True)
    dlg.minsize(440, 460)
    dlg.geometry("560x560")

    outer = tk.Frame(dlg, bg=PANEL, padx=12, pady=12)
    outer.pack(fill=tk.BOTH, expand=True)
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(0, weight=1)

    canvas = tk.Canvas(outer, bg=PANEL, highlightthickness=0, borderwidth=0)
    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    canvas.configure(yscrollcommand=scrollbar.set)

    body = tk.Frame(canvas, bg=ELEVATED, highlightthickness=1, highlightbackground=BORDER)
    window_id = canvas.create_window((0, 0), window=body, anchor="nw")
    body.columnconfigure(0, weight=1)

    def _sync_scroll(_event: object | None = None) -> None:
        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfigure(window_id, width=max(canvas.winfo_width(), 420))

    body.bind("<Configure>", _sync_scroll, add="+")
    canvas.bind("<Configure>", _sync_scroll, add="+")

    tk.Label(
        body,
        text=content.title,
        bg=ELEVATED,
        fg=TEXT,
        font=(FONT_UI_SM[0], FONT_UI_SM[1] + 2, "bold"),
        anchor="w",
    ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))

    status_fg = PIPELINE_STATUS_FG.get(content.status, MUTED)
    tk.Label(
        body,
        text=content.status_line,
        bg=ELEVATED,
        fg=status_fg,
        font=FONT_UI_SM,
        anchor="w",
        wraplength=520,
        justify=tk.LEFT,
    ).grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

    sections = tk.Frame(body, bg=ELEVATED)
    sections.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
    sections.columnconfigure(0, weight=1)

    row = 0
    for label, text, fg in (
        ("Status", content.status_line, status_fg),
        ("Purpose", content.purpose, TEXT),
        ("Input", content.input_desc, TEXT),
        ("Output", content.output_desc, TEXT),
        ("Algorithms", content.algorithms, TEXT),
        ("Generated files", content.generated_files, TEXT),
        ("Execution time", content.duration_text, TEXT),
        ("Evidence", content.evidence, MUTED),
        ("Confidence", content.confidence, MUTED),
        ("Execution summary", content.execution_summary, MUTED),
    ):
        row = _add_dialog_section(sections, row, label, text, fg=fg)

    btn_row = tk.Frame(outer, bg=PANEL)
    btn_row.grid(row=1, column=0, columnspan=2, sticky="e", pady=(8, 0))
    ttk.Button(btn_row, text="Close", command=dlg.destroy).pack(side=tk.RIGHT)

    dlg.bind("<Escape>", lambda _e: dlg.destroy())
    dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
    dlg.grab_set()
    dlg.focus_force()


def bind_pipeline_stage_click(
    widget: tk.Misc,
    gui: Any,
    stage_key: str,
    *,
    diagram_stage: PipelineDiagramStage | None = None,
    skip_widgets: tuple[tk.Misc, ...] = (),
) -> None:
    """Make a pipeline stage widget open the detail dialog on click."""

    skip = set(skip_widgets)

    def _open(_event: object | None = None) -> None:
        report = getattr(gui, "_pipeline_status_report_cache", None)
        if report is None and hasattr(gui, "_assess_pipeline_status_report"):
            try:
                report = gui._assess_pipeline_status_report()
            except Exception:
                report = None
        content = build_pipeline_stage_dialog_content(
            stage_key,
            report,
            gui=gui,
            diagram_stage=diagram_stage,
        )
        show_pipeline_stage_dialog(widget, content)

    def _bind_tree(node: tk.Misc) -> None:
        if node in skip:
            return
        try:
            node.configure(cursor="hand2")
        except tk.TclError:
            pass
        node.bind("<Button-1>", _open, add="+")
        for child in node.winfo_children():
            _bind_tree(child)

    _bind_tree(widget)


__all__ = [
    "PipelineStageDialogContent",
    "PipelineStageUiSpec",
    "bind_pipeline_stage_click",
    "build_pipeline_stage_dialog_content",
    "resolve_stage_ui_spec",
    "show_pipeline_stage_dialog",
]
