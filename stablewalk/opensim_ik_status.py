"""Authoritative OpenSim IK *process* status for every UI surface.

This module is display-only. It does not run IK, write files, or infer success
from leftover artifacts. ``Complete`` is allowed only when the session recorded
a successful IK run (``ik_state == "Completed"``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

# Match pipeline_status.PipelineStageState without importing that module
# (pipeline_status calls this resolver — avoid a circular import).
PipelineStageState = Literal["completed", "partial", "unavailable"]

IK_NOT_STARTED = "not_started"
IK_RUNNING = "running"
IK_PARTIAL = "partial"
IK_COMPLETE = "complete"
IK_FAILED = "failed"

IK_STATUS_LABEL = {
    IK_NOT_STARTED: "Not Started",
    IK_RUNNING: "Running",
    IK_PARTIAL: "Partial",
    IK_COMPLETE: "Complete",
    IK_FAILED: "Failed",
}

_IK_TO_PIPELINE: dict[str, PipelineStageState] = {
    IK_COMPLETE: "completed",
    IK_PARTIAL: "partial",
    IK_RUNNING: "partial",
    IK_FAILED: "partial",
    IK_NOT_STARTED: "unavailable",
}

PIPELINE_NOT_STARTED = "not_started"
PIPELINE_PARTIAL = "partial"
PIPELINE_COMPLETE = "complete"

PIPELINE_STATUS_LABEL = {
    PIPELINE_NOT_STARTED: "Not Started",
    PIPELINE_PARTIAL: "Partial",
    PIPELINE_COMPLETE: "Complete",
}

_PIPELINE_TO_STAGE: dict[str, PipelineStageState] = {
    PIPELINE_COMPLETE: "completed",
    PIPELINE_PARTIAL: "partial",
    PIPELINE_NOT_STARTED: "unavailable",
}

_PROCESS_COMPLETED = "Completed"
_PROCESS_RUNNING = "Running"
_PROCESS_FAILED = "Failed"


@dataclass(frozen=True)
class OpensimIkStatus:
    """One IK process readout shared by Advanced, the footer, Results, and reports."""

    key: str
    label: str
    detail: str
    tooltip: str

    @property
    def pipeline_status(self) -> PipelineStageState:
        return _IK_TO_PIPELINE[self.key]

    @property
    def process_completed(self) -> bool:
        return self.key == IK_COMPLETE


@dataclass(frozen=True)
class OpensimPipelineStatus:
    """Export-pipeline readout (TRC/MOT). Independent of IK process success."""

    key: str
    label: str
    detail: str
    tooltip: str

    @property
    def pipeline_status(self) -> PipelineStageState:
        return _PIPELINE_TO_STAGE[self.key]

    @property
    def process_completed(self) -> bool:
        return self.key == PIPELINE_COMPLETE


@dataclass(frozen=True)
class OpensimStatusPair:
    """Pipeline and IK together — the only consistent OpenSim summary."""

    pipeline: OpensimPipelineStatus
    ik: OpensimIkStatus
    short_label: str
    tooltip: str


def _normalize_process_state(ik_state: str | None) -> str | None:
    raw = str(ik_state or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered == "running":
        return _PROCESS_RUNNING
    if lowered == "completed":
        return _PROCESS_COMPLETED
    if lowered == "failed":
        return _PROCESS_FAILED
    return raw


def _export_line(*, exported: bool, name: str) -> str:
    return f"{name} export: {'Complete' if exported else 'Not complete'}"


def _build_tooltip(
    *,
    key: str,
    detail: str,
    sdk_available: bool,
    model_valid: bool,
    trc_exported: bool,
    mot_exported: bool,
    mapped_trc_ready: bool,
    ik_output_on_disk: bool,
    process_state: str | None,
) -> str:
    lines = [
        f"IK: {IK_STATUS_LABEL[key]}",
        detail,
        "",
        f"SDK: {'Available' if sdk_available else 'Unavailable'}",
        f"Model: {'Ready' if model_valid else 'Not ready'}",
        _export_line(exported=trc_exported, name="TRC"),
        _export_line(exported=mot_exported, name="MOT"),
        f"Mapped TRC: {'Ready' if mapped_trc_ready else 'Not ready'}",
        f"IK process: {process_state or 'Not Started'}",
    ]
    if key == IK_PARTIAL:
        lines.extend(
            [
                "",
                "Partial means the IK process has not completed successfully.",
                "TRC/MOT export status is tracked separately and is not IK success.",
            ]
        )
    if key == IK_COMPLETE:
        lines.extend(
            [
                "",
                "Complete is shown only after a successful IK run in this session.",
            ]
        )
    if ik_output_on_disk and key != IK_COMPLETE:
        lines.extend(
            [
                "",
                "An IK .mot file is on disk, but file presence is not treated as "
                "a successful IK run.",
            ]
        )
    lines.extend(
        [
            "",
            "Experimental IK from mapped markers — not clinical mocap grade.",
        ]
    )
    return "\n".join(lines)


def resolve_opensim_ik_status(
    *,
    ik_state: str | None = None,
    ik_message: str = "",
    block_reason: str | None = None,
    export_completed: bool = False,
    mapped_trc_ready: bool = False,
    trc_exported: bool = False,
    mot_exported: bool = False,
    ik_output_on_disk: bool = False,
    sdk_available: bool = False,
    model_valid: bool = False,
    has_session: bool = False,
    presentation: bool = False,
) -> OpensimIkStatus:
    """Resolve the canonical IK process status.

    ``ik_output_on_disk`` may appear in the tooltip only. It never upgrades
    the status to Complete.
    """
    process = _normalize_process_state(ik_state)
    message = " ".join(str(ik_message or "").split())
    block = " ".join(str(block_reason or "").split()) or None
    export_done = bool(export_completed or (trc_exported and mot_exported))

    if process == _PROCESS_RUNNING:
        key = IK_RUNNING
        detail = "IK running…"
    elif process == _PROCESS_FAILED:
        key = IK_FAILED
        detail = message or block or "IK failed"
    elif process == _PROCESS_COMPLETED:
        key = IK_COMPLETE
        detail = message or "IK completed"
    elif presentation:
        key = IK_PARTIAL
        detail = "Presentation demo — real IK is not run in this mode"
    elif block:
        toward_ik = bool(
            sdk_available or export_done or mapped_trc_ready or has_session
        )
        key = IK_PARTIAL if toward_ik else IK_NOT_STARTED
        detail = block
    elif export_done or mapped_trc_ready:
        key = IK_PARTIAL
        if mapped_trc_ready and not process:
            detail = "Mapped TRC ready; IK not completed"
        else:
            detail = "TRC/MOT export finished; IK has not completed"
    elif has_session and (sdk_available or model_valid):
        key = IK_PARTIAL
        detail = "Session loaded; IK has not started"
    else:
        key = IK_NOT_STARTED
        detail = "IK not executed"

    tooltip = _build_tooltip(
        key=key,
        detail=detail,
        sdk_available=sdk_available,
        model_valid=model_valid,
        trc_exported=bool(trc_exported or export_completed),
        mot_exported=bool(mot_exported or export_completed),
        mapped_trc_ready=mapped_trc_ready,
        ik_output_on_disk=ik_output_on_disk,
        process_state=process,
    )
    return OpensimIkStatus(
        key=key,
        label=IK_STATUS_LABEL[key],
        detail=detail,
        tooltip=tooltip,
    )


def resolve_opensim_ik_status_from_gui(gui: Any) -> OpensimIkStatus:
    """Read live GUI fields — the same inputs every OpenSim status widget uses."""
    files: dict[str, bool] = {}
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

    status = resolve_opensim_ik_status(
        ik_state=getattr(gui, "_opensim_ik_state", None),
        ik_message=str(getattr(gui, "_opensim_ik_message", "") or ""),
        block_reason=getattr(gui, "_opensim_ik_block_reason", None),
        export_completed=bool(getattr(gui, "_opensim_export_completed", False)),
        mapped_trc_ready=bool(files.get("MAPPED")),
        trc_exported=bool(files.get("TRC")),
        mot_exported=bool(files.get("MOT")),
        ik_output_on_disk=bool(files.get("IK_MOT")),
        sdk_available=bool(getattr(gui, "_opensim_sdk_available", False)),
        model_valid=bool(getattr(gui, "_opensim_model_valid", False)),
        has_session=bool(
            getattr(gui, "sequence", None) or getattr(gui, "gait_motion", None)
        ),
        presentation=bool(getattr(gui, "_presentation_mode", False)),
    )
    notes = _reconstruction_notes_from_holder(gui, status.key)
    if notes:
        status = OpensimIkStatus(
            key=status.key,
            label=status.label,
            detail=status.detail,
            tooltip=status.tooltip + "\n\n" + notes,
        )
    return status


def resolve_opensim_ik_status_from_context(ctx: Any) -> OpensimIkStatus:
    """Resolve IK status from a pipeline snapshot context."""
    files = dict(getattr(ctx, "opensim_export_files", None) or {})
    state = getattr(ctx, "opensim_ik_state", None)
    if not state:
        if getattr(ctx, "opensim_ik_completed", False):
            state = _PROCESS_COMPLETED
        elif getattr(ctx, "opensim_ik_running", False):
            state = _PROCESS_RUNNING
        elif str(getattr(ctx, "opensim_ik_block_reason", "") or "").lower().startswith(
            "ik failed"
        ):
            state = _PROCESS_FAILED
    return resolve_opensim_ik_status(
        ik_state=state,
        ik_message="",
        block_reason=getattr(ctx, "opensim_ik_block_reason", None),
        export_completed=bool(files.get("TRC") and files.get("MOT")),
        mapped_trc_ready=bool(files.get("MAPPED")),
        trc_exported=bool(files.get("TRC")),
        mot_exported=bool(files.get("MOT")),
        ik_output_on_disk=bool(files.get("IK_MOT")),
        sdk_available=bool(getattr(ctx, "opensim_sdk_available", False)),
        model_valid=bool(getattr(ctx, "opensim_model_loaded", False)),
        has_session=bool(
            getattr(ctx, "gait_motion", None)
            or getattr(ctx, "run_name", None)
            or getattr(ctx, "source", None)
        ),
        presentation=False,
    )


def _gui_export_files(gui: Any) -> dict[str, bool]:
    files: dict[str, bool] = {}
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
    return files


def resolve_opensim_pipeline_status(
    *,
    export_completed: bool = False,
    trc_exported: bool = False,
    mot_exported: bool = False,
    mapped_trc_ready: bool = False,
    sdk_available: bool = False,
    has_session: bool = False,
    presentation: bool = False,
) -> OpensimPipelineStatus:
    """Resolve TRC/MOT export pipeline status. Never uses IK success."""
    trc = bool(trc_exported or export_completed)
    mot = bool(mot_exported or export_completed)
    if presentation:
        key = PIPELINE_PARTIAL
        detail = "Presentation demo — export pipeline is not a real session"
    elif trc and mot:
        key = PIPELINE_COMPLETE
        detail = "TRC and MOT export completed"
    elif trc or mot or mapped_trc_ready:
        key = PIPELINE_PARTIAL
        missing = []
        if not trc:
            missing.append("TRC")
        if not mot:
            missing.append("MOT")
        if missing:
            detail = "OpenSim export pipeline incomplete (" + " / ".join(missing) + " not exported)"
        else:
            detail = "OpenSim export pipeline is incomplete"
    elif has_session:
        key = PIPELINE_PARTIAL
        detail = "Session loaded; OpenSim files have not been exported"
    else:
        key = PIPELINE_NOT_STARTED
        detail = "OpenSim export has not been run"
    tooltip = "\n".join(
        [
            f"OpenSim Pipeline: {PIPELINE_STATUS_LABEL[key]}",
            detail,
            "",
            f"SDK: {'Available' if sdk_available else 'Unavailable'}",
            _export_line(exported=trc, name="TRC"),
            _export_line(exported=mot, name="MOT"),
            "Pipeline Complete means TRC and MOT were exported.",
            "It does not mean Inverse Kinematics completed.",
        ]
    )
    return OpensimPipelineStatus(
        key=key,
        label=PIPELINE_STATUS_LABEL[key],
        detail=detail,
        tooltip=tooltip,
    )


def resolve_opensim_pipeline_status_from_gui(gui: Any) -> OpensimPipelineStatus:
    files = _gui_export_files(gui)
    return resolve_opensim_pipeline_status(
        export_completed=bool(getattr(gui, "_opensim_export_completed", False)),
        trc_exported=bool(files.get("TRC")),
        mot_exported=bool(files.get("MOT")),
        mapped_trc_ready=bool(files.get("MAPPED")),
        sdk_available=bool(getattr(gui, "_opensim_sdk_available", False)),
        has_session=bool(
            getattr(gui, "sequence", None) or getattr(gui, "gait_motion", None)
        ),
        presentation=bool(getattr(gui, "_presentation_mode", False)),
    )


def resolve_opensim_pipeline_status_from_context(ctx: Any) -> OpensimPipelineStatus:
    files = dict(getattr(ctx, "opensim_export_files", None) or {})
    return resolve_opensim_pipeline_status(
        export_completed=bool(files.get("TRC") and files.get("MOT")),
        trc_exported=bool(files.get("TRC")),
        mot_exported=bool(files.get("MOT")),
        mapped_trc_ready=bool(files.get("MAPPED")),
        sdk_available=bool(getattr(ctx, "opensim_sdk_available", False)),
        has_session=bool(
            getattr(ctx, "gait_motion", None)
            or getattr(ctx, "run_name", None)
            or getattr(ctx, "source", None)
        ),
        presentation=False,
    )


def format_opensim_stage_outcome(
    pipeline: OpensimPipelineStatus,
    ik: OpensimIkStatus,
) -> str:
    """Name which OpenSim stage succeeded and what remains unavailable."""
    succeeded: list[str] = []
    unavailable: list[str] = []
    if pipeline.key == PIPELINE_COMPLETE:
        succeeded.append("OpenSim export (TRC and MOT)")
    elif pipeline.key == PIPELINE_PARTIAL:
        succeeded.append("OpenSim export (partial)")
        unavailable.append("complete TRC+MOT export")
    else:
        unavailable.append("OpenSim export (TRC and MOT)")
    if ik.key == IK_COMPLETE:
        succeeded.append("Inverse Kinematics")
    elif ik.key == IK_RUNNING:
        unavailable.append("Inverse Kinematics (still running)")
    elif ik.key == IK_FAILED:
        unavailable.append("Inverse Kinematics (failed)")
    else:
        unavailable.append("Inverse Kinematics (not completed)")
    lines: list[str] = []
    if succeeded:
        lines.append("Succeeded: " + "; ".join(succeeded))
    if unavailable:
        lines.append("Unavailable: " + "; ".join(unavailable))
    return "\n".join(lines)


def _pair_short_label(pipeline: OpensimPipelineStatus, ik: OpensimIkStatus) -> str:
    if pipeline.key == PIPELINE_COMPLETE and ik.key == IK_COMPLETE:
        return "Complete"
    if pipeline.key == PIPELINE_COMPLETE and ik.key == IK_RUNNING:
        return "Export Complete · IK Running"
    if pipeline.key == PIPELINE_COMPLETE and ik.key == IK_FAILED:
        return "Export Complete · IK Failed"
    if pipeline.key == PIPELINE_COMPLETE:
        return "Export Complete · IK unavailable"
    if ik.key == IK_RUNNING:
        return "IK Running"
    if ik.key == IK_FAILED:
        return "IK Failed"
    if ik.key == IK_COMPLETE:
        return "IK Complete"
    if pipeline.key == PIPELINE_PARTIAL or ik.key == IK_PARTIAL:
        return "Partial"
    return "Not Started"


def format_ik_frame_fit_notes(summary: Any | None) -> str:
    """Successful/failed frames and fitting errors, or an unavailable reason."""
    if summary is None:
        return "IK fitting errors: unavailable (no marker-error .sto)."
    n_frames = getattr(summary, "n_frames", None)
    n_failed = getattr(summary, "n_failed_frames", None)
    if n_frames is None:
        return "IK fitting errors: unavailable."
    failed = int(n_failed or 0)
    solved = int(n_frames) - failed
    lines = [
        str(getattr(summary, "summary_line", lambda: "")() or "").strip()
        or "IK fitting errors available",
        (
            f"IK frames: {solved} within residual band / {failed} failed / "
            f"{int(n_frames)} total"
        ),
    ]
    failed_rows = tuple(getattr(summary, "failed_frames", ()) or ())
    if failed_rows:
        preview: list[str] = []
        for item in failed_rows[:5]:
            idx = item.get("frame_index", "—")
            t = item.get("time_s")
            rms = item.get("marker_error_rms_m")
            time_txt = f"{float(t):.2f}s" if isinstance(t, (int, float)) else "—"
            rms_txt = (
                f"{float(rms) * 100:.1f} cm"
                if isinstance(rms, (int, float))
                else "unavailable"
            )
            preview.append(f"f{idx}@{time_txt} RMS {rms_txt}")
        extra = f" (+{len(failed_rows) - 5} more)" if len(failed_rows) > 5 else ""
        lines.append("Failed frames: " + "; ".join(preview) + extra)
    return "\n".join(line for line in lines if line)


def format_ik_reconstruction_notes(
    *,
    mapping: Any = None,
    residual_rms_m: float | None = None,
    ik_key: str = "",
    frame_count: int | None = None,
    ik_frames_solved: int | None = None,
    ik_error_summary: Any | None = None,
) -> str:
    """Coverage / residual lines for Partial IK — never implies validation."""
    lines: list[str] = []
    if mapping is not None:
        cov = getattr(mapping, "coverage_percent", None)
        raw = getattr(mapping, "raw_coverage_percent", None)
        hc = getattr(mapping, "high_confidence_coverage_percent", None)
        mapped = list(getattr(mapping, "mapped_matching_markers", None) or ())
        refs = list(getattr(mapping, "opensim_reference_markers", None) or ())
        missing = list(getattr(mapping, "missing_for_ik", None) or ())
        unavailable = list(getattr(mapping, "unavailable_markers", None) or ())
        if cov is not None:
            lines.append(f"Reconstruction coverage: {float(cov):.1f}% name overlap")
        if raw is not None or hc is not None:
            lines.append(
                "Marker coverage: "
                f"{float(raw or 0.0):.1f}% raw / {float(hc or 0.0):.1f}% high-confidence"
            )
        if refs:
            lines.append(f"Mapped markers: {len(mapped)}/{len(refs)}")
        if frame_count is not None:
            if ik_frames_solved is None:
                lines.append("IK frames solved: unavailable")
            else:
                lines.append(f"IK frames solved: {int(ik_frames_solved)}/{int(frame_count)}")
        preview = missing or unavailable
        if preview:
            shown = ", ".join(str(m) for m in preview[:8])
            extra = f" (+{len(preview) - 8} more)" if len(preview) > 8 else ""
            lines.append(f"Missing / unmapped: {shown}{extra}")
        msg = str(getattr(mapping, "message", "") or "").strip()
        if msg:
            lines.append(msg)
    if residual_rms_m is not None and math.isfinite(float(residual_rms_m)):
        lines.append(f"IK residual mean RMS: {float(residual_rms_m) * 100.0:.1f} cm")
    else:
        lines.append("IK residuals: unavailable (no completed IK .sto).")
    fit_notes = format_ik_frame_fit_notes(ik_error_summary)
    if fit_notes:
        lines.append(fit_notes)
    if ik_key == IK_PARTIAL:
        lines.append(
            "IK Partial is not complete biomechanical validation. "
            "TRC/MOT export is not a successful Inverse Kinematics solve."
        )
    return "\n".join(lines)


def ik_mot_unavailable_reason(
    ik: OpensimIkStatus,
    *,
    ik_output_on_disk: bool = False,
) -> str:
    """Why IK.mot is unavailable. Export completion is not IK success."""
    if ik.key == IK_COMPLETE:
        return ""
    if ik.key == IK_RUNNING:
        return "IK .mot is unavailable because Inverse Kinematics is still running."
    if ik.key == IK_FAILED:
        detail = str(ik.detail or "").strip()
        extra = f" {detail}" if detail else ""
        return f"IK .mot is unavailable because Inverse Kinematics failed.{extra}"
    if ik_output_on_disk:
        return (
            "An IK .mot file is on disk, but file presence is not Inverse Kinematics "
            "success. Open the diagnostic log for the session reason."
        )
    if ik.key == IK_PARTIAL:
        return (
            f"IK .mot is unavailable — {ik.detail}. "
            "TRC/MOT export completion is not IK success. Open the diagnostic log."
        )
    return (
        "IK .mot is unavailable because Inverse Kinematics has not been run. "
        "TRC/MOT export is separate from IK. Open the diagnostic log for details."
    )


def format_opensim_output_inventory(
    *,
    trc_exported: bool = False,
    mot_exported: bool = False,
    mapped_trc_ready: bool = False,
    ik_output_on_disk: bool = False,
    ik_complete: bool = False,
) -> str:
    """List which OpenSim artifacts completed vs remain unavailable."""

    def _state(present: bool) -> str:
        return "Complete" if present else "Unavailable"

    lines = [
        "Outputs:",
        f"  TRC: {_state(trc_exported)}",
        f"  MOT: {_state(mot_exported)}",
        f"  Mapped TRC: {_state(mapped_trc_ready)}",
        f"  IK .mot: {_state(ik_complete)}",
    ]
    if ik_output_on_disk and not ik_complete:
        lines.append(
            "  An IK .mot file is on disk but is not treated as a completed IK run."
        )
    return "\n".join(lines)


def format_opensim_status_pair(
    pipeline: OpensimPipelineStatus,
    ik: OpensimIkStatus,
    *,
    reconstruction_notes: str = "",
    output_inventory: str = "",
) -> OpensimStatusPair:
    """Canonical combined readout: pipeline and IK are named separately."""
    outcome = format_opensim_stage_outcome(pipeline, ik)
    lines = [
        f"OpenSim Pipeline: {pipeline.label}",
        f"IK: {ik.label}",
        "",
        outcome,
        "",
        pipeline.detail,
        ik.detail,
        "",
        "Pipeline Complete means TRC and MOT were exported.",
        "It does not mean Inverse Kinematics completed.",
        "IK Complete is shown only after a successful IK run in this session.",
        "'IK Partial' is not a completed Inverse Kinematics solve.",
    ]
    if ik.key == IK_PARTIAL:
        lines.append("IK Partial means the IK process has not completed successfully.")
        lines.append("Partial IK is not biomechanical validation.")
    inventory = str(output_inventory or "").strip()
    if inventory:
        lines.extend(["", inventory])
    notes = str(reconstruction_notes or "").strip()
    if notes:
        lines.extend(["", notes])
    return OpensimStatusPair(
        pipeline=pipeline,
        ik=ik,
        short_label=_pair_short_label(pipeline, ik),
        tooltip="\n".join(lines),
    )


def _reconstruction_notes_from_holder(holder: Any, ik_key: str) -> str:
    mapping = getattr(holder, "_marker_mapping_comparison", None)
    if mapping is None:
        mapping = getattr(holder, "opensim_mapping", None)
    residual = getattr(holder, "_opensim_ik_mean_rms_m", None)
    if residual is None:
        residual = getattr(holder, "opensim_ik_mean_rms_m", None)
    frames = getattr(holder, "pose_indices", None)
    frame_count = len(frames) if frames else None
    if frame_count is None:
        seq = getattr(holder, "sequence", None)
        if seq is not None:
            frame_count = len(getattr(seq, "frames", ()) or ())
    solved = getattr(holder, "_opensim_ik_frames_solved", None)
    summary = getattr(holder, "_opensim_ik_error_summary", None)
    return format_ik_reconstruction_notes(
        mapping=mapping,
        residual_rms_m=residual,
        ik_key=ik_key,
        frame_count=frame_count,
        ik_frames_solved=solved,
        ik_error_summary=summary,
    )


def resolve_opensim_status_pair_from_gui(gui: Any) -> OpensimStatusPair:
    ik = resolve_opensim_ik_status_from_gui(gui)
    files = _gui_export_files(gui)
    return format_opensim_status_pair(
        resolve_opensim_pipeline_status_from_gui(gui),
        ik,
        reconstruction_notes=_reconstruction_notes_from_holder(gui, ik.key),
        output_inventory=format_opensim_output_inventory(
            trc_exported=bool(files.get("TRC")),
            mot_exported=bool(files.get("MOT")),
            mapped_trc_ready=bool(files.get("MAPPED")),
            ik_output_on_disk=bool(files.get("IK_MOT")),
            ik_complete=ik.key == IK_COMPLETE,
        ),
    )


def resolve_opensim_status_pair_from_context(ctx: Any) -> OpensimStatusPair:
    ik = resolve_opensim_ik_status_from_context(ctx)
    return format_opensim_status_pair(
        resolve_opensim_pipeline_status_from_context(ctx),
        ik,
        reconstruction_notes=_reconstruction_notes_from_holder(ctx, ik.key),
    )


__all__ = [
    "IK_COMPLETE",
    "IK_FAILED",
    "IK_NOT_STARTED",
    "IK_PARTIAL",
    "IK_RUNNING",
    "IK_STATUS_LABEL",
    "PIPELINE_COMPLETE",
    "PIPELINE_NOT_STARTED",
    "PIPELINE_PARTIAL",
    "PIPELINE_STATUS_LABEL",
    "OpensimIkStatus",
    "OpensimPipelineStatus",
    "OpensimStatusPair",
    "format_ik_frame_fit_notes",
    "format_ik_reconstruction_notes",
    "format_opensim_output_inventory",
    "ik_mot_unavailable_reason",
    "format_opensim_stage_outcome",
    "format_opensim_status_pair",
    "resolve_opensim_ik_status",
    "resolve_opensim_ik_status_from_context",
    "resolve_opensim_ik_status_from_gui",
    "resolve_opensim_pipeline_status",
    "resolve_opensim_pipeline_status_from_context",
    "resolve_opensim_pipeline_status_from_gui",
    "resolve_opensim_status_pair_from_context",
    "resolve_opensim_status_pair_from_gui",
]
