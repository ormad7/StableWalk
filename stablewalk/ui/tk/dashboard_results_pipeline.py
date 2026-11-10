"""Results page — interactive processing pipeline (presentation only).

Technical visualization (shown inside Processing Details):
Video → Pose Detection → Filtering → 3D Reconstruction → OpenSim →
Biomechanics → Metrics → Export

Stage cards scale to the available width; the overall progress strip stays
visible above the flow inside Processing Details.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from typing import Any, Literal

from stablewalk.monitoring.pipeline_status import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
    STATUS_LABEL,
    PipelineDiagramStage,
    PipelineStatusReport,
    build_pipeline_diagram,
)
from stablewalk.ui.theme import (
    BORDER,
    DANGER,
    ELEVATED,
    FONT_CAPTION,
    FONT_FLOW_ARROW,
    FONT_FLOW_ICON,
    FONT_KPI_TITLE,
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
    TEXT_SECONDARY,
    create_elevated_card,
    create_tooltip,
)
from stablewalk.ui.tk.dashboard_pipeline_visual import (
    animate_connector_activation,
    animate_stage_completion,
    format_duration_seconds,
    install_progress_track,
    metrics_from_dialog_content,
    parse_confidence_percent,
    set_connector_state,
    set_progress_fill,
)
from stablewalk.ui.tk.pipeline_stage_dialog import (
    bind_pipeline_stage_click,
    build_pipeline_stage_dialog_content,
    resolve_stage_ui_spec,
)

# UI-only display states (not assessed by pipeline_status).
STATUS_RUNNING = "running"
STATUS_FAILED = "failed"

DisplayStatus = Literal[
    "completed", "partial", "unavailable", "running", "failed"
]

# User-facing Results pipeline — eight horizontal stages.
RESULTS_PIPELINE_STAGES: tuple[tuple[str, str, str], ...] = (
    # key, label, source: "diagram" | "synthetic"
    ("video", "Video", "synthetic"),
    ("pose_estimation", "Pose Detection", "diagram"),
    ("filtering", "Filtering", "synthetic"),
    ("reconstruction_3d", "3D Reconstruction", "diagram"),
    ("opensim", "OpenSim", "diagram"),
    ("biomechanics", "Biomechanics", "diagram"),
    ("metrics", "Metrics", "synthetic"),
    ("export", "Export", "diagram"),
)

# OpenSim and file export extend a completed gait analysis; they are not
# prerequisites for the Results analysis itself.
OPTIONAL_RESULTS_STAGE_KEYS = frozenset({"opensim", "export"})
REQUIRED_RESULTS_STAGE_KEYS = frozenset(
    key for key, _label, _kind in RESULTS_PIPELINE_STAGES
    if key not in OPTIONAL_RESULTS_STAGE_KEYS
)

# Compact stage icons (Segoe UI Symbol / Unicode — no emoji).
_STAGE_ICONS: dict[str, str] = {
    "video": "▶",
    "pose_estimation": "◉",
    "filtering": "≋",
    "reconstruction_3d": "◇",
    "opensim": "⬡",
    "biomechanics": "⌖",
    "metrics": "▦",
    "export": "⇩",
}

# Preferred / floor sizes — cards scale up to fill width; never force h-scroll.
# Row minsize stays at the floor so 720p Processing Details cannot push
# playback off-screen; scale up toward MIN_HEIGHT when the host has room.
_STAGE_CARD_WIDTH = 148
_STAGE_CARD_MIN_WIDTH = 96
_STAGE_CARD_MIN_HEIGHT = 226
_STAGE_CARD_ROW_FLOOR = 96
_STAGE_CONNECTOR_WIDTH = 16

# Host-width breakpoints for wrapping the 8-stage flow (2 / 4 / 8 per row).
RESULTS_PIPELINE_WRAP_NARROW = 720
RESULTS_PIPELINE_WRAP_MEDIUM = 1100
_RESULTS_PIPELINE_WRAP_HYSTERESIS = 40


def results_pipeline_wrap_columns(
    width: int,
    n: int = 8,
    *,
    previous: int | None = None,
) -> int:
    """How many stage cards share a row at ``width`` pixels.

    Tightening wrap is immediate; expanding back to more columns waits
    ``_RESULTS_PIPELINE_WRAP_HYSTERESIS`` past the breakpoint so a wrap-induced
    height change cannot oscillate the layout.
    """
    n = max(1, int(n))
    if int(width) < 80:
        return int(previous) if previous and int(previous) > 0 else n
    if width < RESULTS_PIPELINE_WRAP_NARROW:
        target = 2
    elif width < RESULTS_PIPELINE_WRAP_MEDIUM:
        target = 4
    else:
        target = n
    target = min(target, n)
    prev = int(previous) if previous and int(previous) > 0 else target
    if prev == target:
        return target
    if target < prev:
        return target
    if target == 4 and width < RESULTS_PIPELINE_WRAP_NARROW + _RESULTS_PIPELINE_WRAP_HYSTERESIS:
        return prev
    if target == n and width < RESULTS_PIPELINE_WRAP_MEDIUM + _RESULTS_PIPELINE_WRAP_HYSTERESIS:
        return prev
    return target


def _reset_pipeline_host_grid(host: tk.Misc) -> None:
    try:
        cols, rows = host.grid_size()
    except tk.TclError:
        cols, rows = (0, 0)
    for col in range(max(int(cols), 20)):
        host.columnconfigure(col, weight=0, minsize=0, uniform="")
    for row in range(max(int(rows), 6)):
        host.rowconfigure(row, weight=0, minsize=0)

_DISPLAY_LABEL = {
    STATUS_COMPLETED: "Completed",
    STATUS_PARTIAL: "Partial",
    STATUS_UNAVAILABLE: "Not run",
    STATUS_RUNNING: "Running",
    STATUS_FAILED: "Failed",
}

_DISPLAY_STRIP = {
    STATUS_COMPLETED: SUCCESS,
    STATUS_PARTIAL: ORANGE,
    STATUS_UNAVAILABLE: BORDER,
    STATUS_RUNNING: ORANGE,
    STATUS_FAILED: DANGER,
}

_DISPLAY_FG = {
    STATUS_COMPLETED: SUCCESS,
    STATUS_PARTIAL: ORANGE,
    STATUS_UNAVAILABLE: MUTED,
    STATUS_RUNNING: ORANGE,
    STATUS_FAILED: DANGER,
}


@dataclass(frozen=True)
class _ResultsStageView:
    key: str
    label: str
    status: str
    detail: str
    tooltip: str
    confidence_text: str
    duration_text: str
    output_text: str
    artifacts_text: str
    item_keys: tuple[str, ...] = ()
    logs: str = ""
    purpose: str = ""
    quality: str = ""
    warnings: str = ""


@dataclass(frozen=True)
class ResultsPipelineProgress:
    """Overall Results pipeline progress (summary strip — not stage cards)."""

    completion_pct: float
    total_time_s: float | None
    total_time_text: str
    avg_fps: float | None
    avg_fps_text: str
    remaining_count: int
    remaining_text: str
    per_stage_text: str
    completed: int
    total_stages: int


def results_pipeline_session_state(views: list[_ResultsStageView]) -> str:
    """Return the presentation state for required Results stages."""
    required = [
        view for view in views if view.key not in OPTIONAL_RESULTS_STAGE_KEYS
    ]
    if any(view.status == STATUS_RUNNING for view in required):
        return "processing"
    if required and all(view.status == STATUS_COMPLETED for view in required):
        return "complete"
    if any(view.status == STATUS_FAILED for view in required):
        return "incomplete"
    if any(
        view.status in (STATUS_COMPLETED, STATUS_PARTIAL)
        for view in required
    ):
        return "partial"
    return "waiting"


def stage_incomplete_reason(view: _ResultsStageView) -> str:
    """Exact user-facing reason a required stage is not complete."""
    detail = " ".join(str(getattr(view, "detail", "") or "").split()).strip()
    label = str(getattr(view, "label", None) or getattr(view, "key", None) or "Stage").strip()
    prefixes = (
        f"{label} stage not run — ",
        f"{label} stage not run · ",
        f"{label} stage not run - ",
        "Metrics stage not run — ",
        "Metrics stage not run · ",
        "Metrics stage not run - ",
        f"Partial {label.lower()} stage · ",
        "Partial metrics stage · ",
    )
    lowered = detail.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix.lower()):
            detail = detail[len(prefix) :].strip()
            lowered = detail.lower()
            break
    cut = " · derived analysis summary available"
    if cut in lowered:
        detail = detail[: lowered.index(cut)].strip(" ·-")
    if detail and detail not in ("—", "-", "Waiting for biomechanics"):
        return detail
    status = getattr(view, "status", "")
    if status == STATUS_FAILED:
        return "stage failed"
    if status == STATUS_PARTIAL:
        output = " ".join(
            str(getattr(view, "output_text", "") or "").split()
        ).strip()
        if output and output not in ("—", "-", "Partial KPI set"):
            return output
        return "partial results"
    if status == STATUS_RUNNING:
        return "still running"
    if status == STATUS_UNAVAILABLE:
        return "not run"
    return "not completed"


def format_required_analysis_headline(
    views: list[_ResultsStageView],
    *,
    analysis_active: bool = False,
) -> tuple[str, str]:
    """After analysis: ``100% Complete`` or ``Incomplete: <stage> — <reason>``."""
    required = [
        view for view in views if view.key not in OPTIONAL_RESULTS_STAGE_KEYS
    ]
    if analysis_active or any(view.status == STATUS_RUNNING for view in required):
        return "Processing", "processing"
    if required and all(view.status == STATUS_COMPLETED for view in required):
        return "100% Complete", "complete"
    if not any(
        view.status in (STATUS_COMPLETED, STATUS_PARTIAL, STATUS_FAILED)
        for view in required
    ):
        return "Waiting for analysis", "waiting"

    incomplete = [view for view in required if view.status != STATUS_COMPLETED]
    parts: list[str] = []
    for view in incomplete[:2]:
        parts.append(f"{view.label} — {stage_incomplete_reason(view)}")
    extra = len(incomplete) - len(parts)
    if extra > 0:
        parts.append(f"+{extra} more")
    return "Incomplete: " + "; ".join(parts), "incomplete"


def _session_frame_count(gui: Any) -> int | None:
    sequence = getattr(gui, "sequence", None)
    if sequence is not None:
        frames = getattr(sequence, "frames", None)
        if frames is not None:
            try:
                n = len(frames)
            except TypeError:
                n = 0
            if n > 0:
                return int(n)
    recording = getattr(gui, "gait_motion", None) or getattr(
        gui, "_analysis_motion_recording", None
    )
    if callable(recording):
        try:
            recording = recording()
        except Exception:
            recording = None
    if recording is not None:
        n = int(getattr(recording, "frame_count", 0) or 0)
        if n > 0:
            return n
    return None


def _total_rts_duration_s(gui: Any) -> float | None:
    """Sum unique Real-to-Sim stage durations (avoid double-counting shared steps)."""
    loader = getattr(gui, "_load_rts_report_for_session", None)
    if not callable(loader):
        return None
    try:
        report = loader()
    except Exception:
        return None
    if not report:
        return None
    total = 0.0
    found = False
    for stage in report.get("stages") or ():
        raw = stage.get("duration_s")
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value < 0:
            continue
        total += value
        found = True
    return total if found else None


def compute_results_pipeline_progress(
    gui: Any, views: list[_ResultsStageView]
) -> ResultsPipelineProgress:
    """Derive main-analysis progress from required gait-analysis stages only."""
    required_views = [
        view for view in views if view.key not in OPTIONAL_RESULTS_STAGE_KEYS
    ]
    total = max(1, len(required_views))
    completed = sum(1 for v in required_views if v.status == STATUS_COMPLETED)
    partial = sum(1 for v in required_views if v.status == STATUS_PARTIAL)
    remaining_views = [
        v
        for v in required_views
        if v.status in (
            STATUS_UNAVAILABLE,
            STATUS_RUNNING,
            STATUS_FAILED,
            STATUS_PARTIAL,
        )
    ]
    # Partial counts half; running contributes a small in-progress fraction.
    running = sum(1 for v in required_views if v.status == STATUS_RUNNING)
    weighted = completed + 0.5 * partial + 0.25 * running
    completion_pct = max(0.0, min(100.0, 100.0 * weighted / float(total)))

    total_time_s = _total_rts_duration_s(gui)
    # Fallback: sum unique non-empty card duration texts that parse as seconds.
    if total_time_s is None:
        from stablewalk.ui.tk.pipeline_stage_dialog import (
            _rts_stage_duration,
            resolve_stage_ui_spec,
        )

        seen_ids: set[str] = set()
        summed = 0.0
        found = False
        for view in views:
            spec = resolve_stage_ui_spec(view.key)
            key = ",".join(spec.rts_stage_ids) or view.key
            if key in seen_ids:
                continue
            seen_ids.add(key)
            dur = _rts_stage_duration(gui, spec.rts_stage_ids)
            if dur is None:
                continue
            summed += float(dur)
            found = True
        total_time_s = summed if found else None

    total_time_text = format_duration_seconds(total_time_s)
    frames = _session_frame_count(gui)
    avg_fps: float | None = None
    if (
        total_time_s is not None
        and total_time_s > 1e-6
        and frames is not None
        and frames > 0
    ):
        avg_fps = float(frames) / float(total_time_s)
    if avg_fps is None:
        avg_fps_text = "—"
    else:
        avg_fps_text = f"{avg_fps:.1f} FPS"

    remaining_count = len(remaining_views)
    def _remaining_label(view: _ResultsStageView) -> str:
        suffix = {
            STATUS_PARTIAL: "partial",
            STATUS_RUNNING: "running",
            STATUS_FAILED: "failed",
            STATUS_UNAVAILABLE: "not run",
        }.get(view.status)
        return f"{view.label} ({suffix})" if suffix else view.label

    if remaining_count <= 0:
        remaining_text = "None — required analysis complete"
    elif remaining_count == 1:
        remaining_text = f"1 · {_remaining_label(remaining_views[0])}"
    else:
        shown = remaining_views[:3]
        names = ", ".join(_remaining_label(v) for v in shown)
        extra = remaining_count - len(shown)
        if extra > 0:
            names = f"{names} +{extra}"
        remaining_text = f"{remaining_count} · {names}"

    per_parts = [
        f"{v.label} {v.duration_text or '—'}"
        for v in views
        if (v.duration_text or "—") != "—"
    ]
    per_stage_text = "  ·  ".join(per_parts) if per_parts else "No stage timings yet"

    return ResultsPipelineProgress(
        completion_pct=completion_pct,
        total_time_s=total_time_s,
        total_time_text=total_time_text,
        avg_fps=avg_fps,
        avg_fps_text=avg_fps_text,
        remaining_count=remaining_count,
        remaining_text=remaining_text,
        per_stage_text=per_stage_text,
        completed=completed,
        total_stages=len(required_views),
    )


def _field_available(field: Any) -> bool:
    if field is None:
        return False
    return bool(getattr(field, "available", False))


def _field_value(field: Any, default: str = "—") -> str:
    if field is None:
        return default
    value = getattr(field, "value", None)
    if value is None or str(value).strip() in ("", "—", "-"):
        return default
    return str(value).strip()


# Monitored inputs the Metrics stage packages into session KPIs.
_METRICS_REQUIRED_ITEMS: tuple[tuple[str, str], ...] = (
    ("bio_cadence", "Cadence"),
    ("bio_com", "COM"),
    ("bio_bos", "BoS"),
    ("bio_heel_strike", "Heel-strike events"),
    ("bio_toe_off", "Toe-off events"),
)


def metrics_stage_missing_dependencies(
    gui: Any, report: PipelineStatusReport | None
) -> list[str]:
    """Exact Metrics-stage inputs that are absent — never inferred as complete."""
    missing: list[str] = []
    biomech = getattr(gui, "_biomech_analysis", None)
    cycles = getattr(gui, "_gait_cycle", None)
    contact = getattr(gui, "_foot_contact", None)
    if biomech is None:
        missing.append("Biomechanical analysis result")
        if cycles is None and contact is None:
            missing.append("Gait-cycle or foot-contact analysis")
        return missing

    items: dict[str, Any] = {}
    if report is not None:
        items = {item.key: item for item in report.all_items()}
    for key, label in _METRICS_REQUIRED_ITEMS:
        item = items.get(key)
        if item is None:
            missing.append(f"{label} (not assessed)")
            continue
        if item.status == STATUS_UNAVAILABLE:
            detail = str(getattr(item, "detail", "") or "").strip()
            missing.append(f"{label}: {detail}" if detail else f"{label} (not computed)")
    return missing


def _diagram_by_key(report: PipelineStatusReport | None) -> dict[str, PipelineDiagramStage]:
    if report is None:
        return {}
    return {s.key: s for s in build_pipeline_diagram(report)}


def _looks_failed(status: str, detail: str, warnings: str, quality: str) -> bool:
    if status == STATUS_FAILED:
        return True
    blob = f"{detail}\n{warnings}\n{quality}".lower()
    fail_tokens = ("fail", "error", "exception", "crash", "not installed")
    if status == STATUS_UNAVAILABLE and any(t in blob for t in fail_tokens):
        # Only treat as failed when the session actually tried something.
        return "not executed" not in blob and "not assessed" not in blob
    if "failed" in blob or "error:" in blob:
        return True
    return False


def _looks_running(gui: Any, key: str, detail: str) -> bool:
    detail_l = (detail or "").lower()
    if "running" in detail_l or "…" in detail or "..." in detail_l:
        return True
    if bool(getattr(gui, "_analysis_compute_active", False)):
        if key in (
            "pose_estimation",
            "filtering",
            "reconstruction_3d",
            "biomechanics",
            "metrics",
            "video",
        ):
            return True
    ctx = getattr(gui, "_pipeline_status_context_cache", None)
    if ctx is not None and key == "opensim" and bool(
        getattr(ctx, "opensim_ik_running", False)
    ):
        return True
    if bool(getattr(gui, "_opensim_auto_pipeline_busy", False)) and key == "opensim":
        return True
    return False


def _normalize_display_status(
    gui: Any,
    key: str,
    status: str,
    *,
    detail: str,
    warnings: str,
    quality: str,
) -> str:
    if _looks_running(gui, key, detail):
        return STATUS_RUNNING
    if _looks_failed(status, detail, warnings, quality):
        return STATUS_FAILED
    return status


def _session_has_analysis(gui: Any, report: PipelineStatusReport | None) -> bool:
    if report is not None and (report.source or report.run_name):
        return True
    summary = getattr(gui, "_analysis_summary_cache", None)
    if summary is not None and getattr(summary, "source", None):
        return True
    if getattr(gui, "_biomech_analysis", None) is not None:
        return True
    if getattr(gui, "gait_motion", None) is not None:
        return True
    seq = getattr(gui, "sequence", None)
    return bool(seq is not None and getattr(seq, "frames", None))


def resolve_results_pipeline_report(
    gui: Any, report: PipelineStatusReport | None = None
) -> PipelineStatusReport | None:
    """Single snapshot used by Results cards: assessed report, else cache."""
    if report is not None:
        return report
    assess = getattr(gui, "_assess_pipeline_status_report", None)
    if callable(assess):
        try:
            return assess(force=False)
        except Exception:
            pass
    cached = getattr(gui, "_pipeline_status_report_cache", None)
    if cached is not None:
        return cached
    return getattr(gui, "_pipeline_status_report", None)


def _synthetic_video_stage(gui: Any, report: PipelineStatusReport | None) -> _ResultsStageView:
    summary = getattr(gui, "_analysis_summary_cache", None)
    source = ""
    if summary is not None and getattr(summary, "source", None):
        source = str(summary.source)
    elif report is not None and report.source:
        source = report.source
    seq = getattr(gui, "sequence", None)
    frame_count = len(getattr(seq, "frames", ()) or ()) if seq is not None else 0

    if source or frame_count > 0:
        status = STATUS_COMPLETED
        detail = source or f"{frame_count} frames loaded"
        if frame_count > 0 and source:
            detail = f"{source} · {frame_count} frames"
        output = f"Video loaded · {frame_count} frames" if frame_count else "Video loaded"
    else:
        status = STATUS_UNAVAILABLE
        detail = "Waiting for a walking video"
        output = "—"

    content = build_pipeline_stage_dialog_content("video", report, gui=gui)
    metrics = metrics_from_dialog_content(content)
    conf = content.confidence
    vq = getattr(summary, "video_quality", None) if summary is not None else None
    if _field_available(vq):
        conf = _field_value(vq)

    if status == STATUS_UNAVAILABLE and content.status != STATUS_UNAVAILABLE:
        status = content.status
        detail = content.status_line
        output = content.output_desc or output

    status = _normalize_display_status(
        gui, "video", status, detail=detail, warnings=content.warnings, quality=content.quality
    )
    return _ResultsStageView(
        key="video",
        label="Video",
        status=status,
        detail=detail,
        tooltip="Input recording used for pose detection.",
        confidence_text=conf,
        duration_text=metrics.duration_text,
        output_text=output,
        artifacts_text=metrics.files_text,
        item_keys=("pose_mediapipe",),
        logs=content.execution_summary or content.evidence,
        purpose=content.purpose,
        quality=content.quality,
        warnings=content.warnings,
    )


def _synthetic_filtering_stage(
    gui: Any, report: PipelineStatusReport | None, diagram: dict[str, PipelineDiagramStage]
) -> _ResultsStageView:
    content = build_pipeline_stage_dialog_content("filtering", report, gui=gui)
    metrics = metrics_from_dialog_content(content)
    pose = diagram.get("pose_estimation")
    seq = getattr(gui, "sequence", None)
    frame_count = len(getattr(seq, "frames", ()) or ()) if seq is not None else 0
    smooth_on = bool(getattr(getattr(gui, "smooth_motion", None), "get", lambda: True)())

    if pose is not None and pose.status == STATUS_COMPLETED and frame_count > 0:
        status = STATUS_COMPLETED
        detail = (
            "Temporal landmark smoothing applied"
            if smooth_on
            else "Pose cleaned for reconstruction (smoothing off)"
        )
        output = f"Filtered trajectories · {frame_count} frames"
    elif pose is not None and pose.status == STATUS_PARTIAL:
        status = STATUS_PARTIAL
        detail = "Partial pose — filtering limited"
        output = content.output_desc or "Partial filter pass"
    elif frame_count > 0:
        status = STATUS_UNAVAILABLE
        detail = "Loaded trajectory data available; filtering execution not verified"
        output = f"Session data · {frame_count} frames (filtering not run)"
    else:
        status = STATUS_UNAVAILABLE
        detail = "Waiting for pose detection"
        output = "—"

    if status == STATUS_UNAVAILABLE and content.status != STATUS_UNAVAILABLE:
        status = content.status
        detail = content.status_line or detail

    status = _normalize_display_status(
        gui,
        "filtering",
        status,
        detail=detail,
        warnings=content.warnings,
        quality=content.quality,
    )
    return _ResultsStageView(
        key="filtering",
        label="Filtering",
        status=status,
        detail=detail,
        tooltip=content.purpose,
        confidence_text=content.confidence,
        duration_text=metrics.duration_text,
        output_text=output,
        artifacts_text=metrics.files_text,
        item_keys=resolve_stage_ui_spec("filtering").item_keys,
        logs=content.execution_summary or content.evidence,
        purpose=content.purpose,
        quality=content.quality,
        warnings=content.warnings,
    )


def _synthetic_metrics_stage(gui: Any, report: PipelineStatusReport | None) -> _ResultsStageView:
    summary = getattr(gui, "_analysis_summary_cache", None)
    content = build_pipeline_stage_dialog_content("metrics", report, gui=gui)
    metrics = metrics_from_dialog_content(content)
    missing = metrics_stage_missing_dependencies(gui, report)

    kpi_bits: list[str] = []
    if summary is not None:
        for attr, label in (
            ("overall_gait_quality", "Gait quality"),
            ("cadence", "Cadence"),
            ("walking_speed", "Speed"),
            ("symmetry", "Symmetry"),
            ("stability_margin", "Stability"),
        ):
            field = getattr(summary, attr, None)
            if _field_available(field):
                kpi_bits.append(f"{label} {_field_value(field)}")

    has_source = bool(summary and getattr(summary, "source", None))
    monitored = content.status
    derived_detail = " · ".join(kpi_bits[:3]) if kpi_bits else ""
    missing_text = "; ".join(missing)

    if monitored in (STATUS_COMPLETED, STATUS_PARTIAL, STATUS_RUNNING, STATUS_FAILED):
        # Cadence / COM / BoS / events were computed — do not wait for a
        # Results-tab paint to acknowledge the Metrics stage.
        status = monitored
        if status == STATUS_COMPLETED:
            detail = (
                f"Completed derived metrics · {derived_detail}"
                if derived_detail
                else (content.status_line or "Completed gait metrics")
            )
            output = (
                f"{len(kpi_bits)} completed derived metric outputs"
                if kpi_bits
                else "Session metrics computed from biomechanical analysis"
            )
        elif status == STATUS_RUNNING:
            detail = (
                f"Metrics stage running · derived summary available · {derived_detail}"
                if derived_detail
                else "Metrics stage running"
            )
            output = (
                f"{len(kpi_bits)} derived estimates available while stage runs"
                if kpi_bits
                else "—"
            )
        elif status == STATUS_PARTIAL:
            gap = missing_text or "one or more metric inputs incomplete"
            detail = (
                f"Partial metrics stage · {gap}"
                + (f" · {derived_detail}" if derived_detail else "")
            )
            output = (
                f"{len(kpi_bits)} derived metric outputs from partial analysis"
                if kpi_bits
                else "Partial KPI set"
            )
        else:
            detail = (
                f"Metrics stage failed · prior derived summary available · {derived_detail}"
                if derived_detail
                else "Metrics stage failed"
            )
            output = (
                f"{len(kpi_bits)} derived estimates available; stage failed"
                if kpi_bits
                else "—"
            )
    elif kpi_bits:
        # Leftover Overview KPIs must not mark an unrun Metrics stage complete.
        status = STATUS_UNAVAILABLE
        gap = missing_text or "required metric inputs were not computed"
        detail = (
            f"Metrics stage not run — missing: {gap} · "
            f"derived analysis summary available · {derived_detail}"
        )
        output = f"{len(kpi_bits)} derived estimates from analysis summary"
    elif missing:
        status = STATUS_UNAVAILABLE
        detail = f"Metrics stage not run — missing: {missing_text}"
        output = "—"
    elif has_source:
        status = STATUS_PARTIAL
        detail = "Session analyzed — metrics incomplete"
        output = "Partial KPI set"
    else:
        status = STATUS_UNAVAILABLE
        detail = "Waiting for biomechanics"
        output = "—"

    validity = getattr(summary, "result_validity", None) if summary else None
    if validity is not None:
        conf = f"{validity.worst_status()} · see pose/contact/spectral/biomech layers"
        conf_field = None
    else:
        conf_field = getattr(summary, "analysis_confidence", None) if summary else None
        if not _field_available(conf_field):
            conf_field = getattr(summary, "pipeline_confidence", None) if summary else None
        conf = (
            _field_value(conf_field)
            if _field_available(conf_field)
            else (content.confidence or "—")
        )

    status = _normalize_display_status(
        gui, "metrics", status, detail=detail, warnings=content.warnings, quality=content.quality
    )
    return _ResultsStageView(
        key="metrics",
        label="Metrics",
        status=status,
        detail=detail,
        tooltip=content.purpose,
        confidence_text=conf,
        duration_text=metrics.duration_text,
        output_text=output,
        artifacts_text=metrics.files_text,
        item_keys=resolve_stage_ui_spec("metrics").item_keys,
        logs=content.execution_summary or content.evidence,
        purpose=content.purpose,
        quality=content.quality,
        warnings=content.warnings,
    )


def _opensim_results_status(gui: Any) -> tuple[str, str, str] | None:
    """Footer/Advanced OpenSim pair — export success is independent of IK."""
    try:
        from stablewalk.opensim_ik_status import (
            IK_COMPLETE,
            IK_FAILED,
            IK_RUNNING,
            PIPELINE_COMPLETE,
            resolve_opensim_status_pair_from_gui,
        )

        pair = resolve_opensim_status_pair_from_gui(gui)
    except Exception:
        return None
    pipeline_key = pair.pipeline.key
    ik_key = pair.ik.key
    if pipeline_key == PIPELINE_COMPLETE and ik_key == IK_COMPLETE:
        status = STATUS_COMPLETED
    elif ik_key == IK_FAILED:
        status = STATUS_FAILED
    elif ik_key == IK_RUNNING:
        status = STATUS_RUNNING
    elif pipeline_key == PIPELINE_COMPLETE or ik_key not in ("not_started", None, ""):
        status = STATUS_PARTIAL
    else:
        status = STATUS_UNAVAILABLE
    return status, pair.short_label, pair.tooltip


def _diagram_stage_view(
    key: str,
    label: str,
    gui: Any,
    report: PipelineStatusReport | None,
    diagram: dict[str, PipelineDiagramStage],
) -> _ResultsStageView:
    stage = diagram.get(key)
    if stage is None:
        stage = PipelineDiagramStage(
            key=key,
            label=label,
            status=STATUS_UNAVAILABLE,
            detail="Waiting for upstream stages" if not _session_has_analysis(gui, report)
            else "Not executed in this session",
            tooltip=resolve_stage_ui_spec(key).purpose,
            item_keys=resolve_stage_ui_spec(key).item_keys,
        )
    content = build_pipeline_stage_dialog_content(
        key, report, gui=gui, diagram_stage=stage
    )
    metrics = metrics_from_dialog_content(content)
    logs_parts = [
        content.execution_summary,
        content.evidence,
        content.generated_files,
    ]
    logs = "\n\n".join(p for p in logs_parts if p and p.strip() and not p.startswith("No "))
    output = (content.output_desc or stage.detail or "—").strip()
    status = _normalize_display_status(
        gui,
        key,
        stage.status,
        detail=stage.detail,
        warnings=content.warnings,
        quality=content.quality,
    )
    detail = stage.detail
    tooltip = stage.tooltip
    if key == "opensim":
        pair_view = _opensim_results_status(gui)
        if pair_view is not None:
            status, detail, tooltip = pair_view
            status = _normalize_display_status(
                gui,
                key,
                status,
                detail=detail,
                warnings=content.warnings,
                quality=content.quality,
            )
            output = detail
            from stablewalk.opensim_ik_status import (
                IK_COMPLETE,
                ik_mot_unavailable_reason,
                resolve_opensim_status_pair_from_gui,
            )

            pair = resolve_opensim_status_pair_from_gui(gui)
            if pair.ik.key != IK_COMPLETE:
                files = {}
                getter = getattr(gui, "_opensim_export_files_on_disk", None)
                if callable(getter):
                    try:
                        files = dict(getter() or {})
                    except Exception:
                        files = {}
                reason = ik_mot_unavailable_reason(
                    pair.ik, ik_output_on_disk=bool(files.get("IK_MOT"))
                )
                extra = (
                    f"{reason}\nOpen the diagnostic log from Advanced / OpenSim details."
                )
                logs = f"{logs}\n\n{extra}".strip() if logs else extra
    return _ResultsStageView(
        key=key,
        label=label,
        status=status,
        detail=detail,
        tooltip=tooltip,
        confidence_text=content.confidence,
        duration_text=metrics.duration_text,
        output_text=output,
        artifacts_text=metrics.files_text,
        item_keys=stage.item_keys,
        logs=logs or "No technical log for this stage.",
        purpose=content.purpose,
        quality=content.quality,
        warnings=content.warnings,
    )


def build_results_stage_views(
    gui: Any, report: PipelineStatusReport | None
) -> list[_ResultsStageView]:
    """Assemble the Results pipeline stage views."""
    report = resolve_results_pipeline_report(gui, report)
    diagram = _diagram_by_key(report)
    views: list[_ResultsStageView] = []
    for key, label, kind in RESULTS_PIPELINE_STAGES:
        if kind == "synthetic":
            if key == "video":
                views.append(_synthetic_video_stage(gui, report))
            elif key == "filtering":
                views.append(_synthetic_filtering_stage(gui, report, diagram))
            elif key == "metrics":
                views.append(_synthetic_metrics_stage(gui, report))
            else:
                views.append(_diagram_stage_view(key, label, gui, report, diagram))
        else:
            views.append(_diagram_stage_view(key, label, gui, report, diagram))
    return views


def _status_tone_name(status: str) -> str:
    if status == STATUS_COMPLETED:
        return "ok"
    if status in (STATUS_PARTIAL, STATUS_RUNNING):
        return "warn"
    if status in (STATUS_FAILED, STATUS_UNAVAILABLE):
        return "fail" if status == STATUS_FAILED else "neutral"
    return "neutral"


def _stage_status_label(view: _ResultsStageView) -> str:
    if view.key in OPTIONAL_RESULTS_STAGE_KEYS:
        optional_labels = {
            STATUS_COMPLETED: "Optional – Completed",
            STATUS_PARTIAL: "Optional – Incomplete",
            STATUS_UNAVAILABLE: "Optional – Not run",
            STATUS_RUNNING: "Optional – Running",
            STATUS_FAILED: "Optional – Failed",
        }
        return optional_labels.get(view.status, "Optional")
    return _DISPLAY_LABEL.get(
        view.status,
        STATUS_LABEL.get(view.status, view.status),
    )


def _status_chip(parent: tk.Misc, view: _ResultsStageView) -> tk.Frame:
    from stablewalk.ui.design_system.components import StatusBadge

    badge = StatusBadge(
        parent,
        _stage_status_label(view),
        tone=_status_tone_name(view.status),  # type: ignore[arg-type]
        bg_parent=ELEVATED,
    )
    return badge.frame


def _metric_cell(parent: tk.Misc, label: str, value: str, *, fg: str = TEXT) -> tk.Frame:
    cell = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)
    tk.Label(cell, text=label, bg=ELEVATED, fg=MUTED_DIM, font=FONT_CAPTION, anchor="w").pack(
        anchor="w"
    )
    tk.Label(
        cell,
        text=value or "—",
        bg=ELEVATED,
        fg=fg,
        font=FONT_UI_SM,
        anchor="w",
        wraplength=180,
        justify=tk.LEFT,
    ).pack(anchor="w", pady=(1, 0))
    return cell


def _warnings_preview(warnings: str) -> str:
    warn_preview = " ".join((warnings or "").split()).strip()
    if warn_preview in ("", "None recorded.", "—"):
        return "None"
    # The full warning remains available in Details.  Keep the collapsed card
    # short enough to align with neighboring stages at the narrow breakpoint.
    if len(warn_preview) > 44:
        return warn_preview[:41].rstrip() + "…"
    return warn_preview


def _stage_warning_preview(view: _ResultsStageView) -> str:
    """One concise warning for the collapsed presentation card."""
    warning = _warnings_preview(view.warnings)
    if warning != "None":
        return warning
    if view.status in (STATUS_PARTIAL, STATUS_FAILED):
        return _short_partial_reason(view)
    return "None"


def _stage_details_text(view: _ResultsStageView) -> str:
    """Full secondary stage information shown only after expansion."""
    sections = (
        ("Purpose", view.purpose or view.tooltip),
        ("Stage detail", view.detail),
        ("Duration", view.duration_text),
        ("Output", view.output_text),
        ("Quality", view.quality),
        ("Warnings", view.warnings),
        ("Artifacts", view.artifacts_text),
        ("Technical log", view.logs),
    )
    return "\n\n".join(
        f"{heading}\n{str(value).strip() if str(value or '').strip() else '—'}"
        for heading, value in sections
    )


def _calculated_confidence_percent(text: str) -> int | None:
    """Return a percentage only when *text* describes actual confidence."""
    raw = (text or "").strip()
    if not raw or raw in ("—", "-"):
        return None
    lower = raw.lower()
    # Coverage and sub-step completion are useful evidence, but they are not
    # confidence measurements and must not be presented as such.
    if "coverage" in lower or "sub-step completion" in lower:
        return None
    pct = parse_confidence_percent(raw)
    if pct is None:
        return None
    if "confidence" in lower:
        return pct
    # Summary confidence fields are already stored as a bare percentage.
    if raw.endswith("%") and raw[:-1].strip().replace(".", "", 1).isdigit():
        return pct
    return None


def _short_partial_reason(view: _ResultsStageView) -> str:
    reason = (view.detail or view.warnings or "Some stage outputs are unavailable").strip()
    if len(reason) > 46:
        return reason[:43].rstrip() + "…"
    return reason


def _unavailable_stage_text(stage_key: str) -> str:
    if stage_key in OPTIONAL_RESULTS_STAGE_KEYS:
        return "Optional – Not run"
    return "Not exported" if stage_key == "export" else "Not run"


def _build_results_stage_card(
    parent: tk.Misc,
    column: int,
    view: _ResultsStageView,
    gui: Any,
    report: PipelineStatusReport | None,
    *,
    step: int,
    is_last: bool,
    prev_completed: bool,
    grid_row: int = 0,
) -> dict[str, Any]:
    """One pipeline node that expands to its grid cell (no fixed page width)."""
    strip = _DISPLAY_STRIP.get(view.status, BORDER)
    pct = _calculated_confidence_percent(view.confidence_text)
    icon = _STAGE_ICONS.get(view.key, "●")

    outer = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    outer.grid(row=grid_row, column=column, sticky="nsew", padx=(0, 2), pady=(0, PAD_XS))
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(0, weight=1)

    card = create_elevated_card(outer)
    card.grid(row=0, column=0, sticky="nsew")
    card.columnconfigure(0, weight=1)
    card.rowconfigure(1, weight=1)

    accent = tk.Frame(card, bg=strip, height=4)
    accent.grid(row=0, column=0, sticky="ew")
    accent.grid_propagate(False)

    body = tk.Frame(card, bg=ELEVATED, padx=PAD_XS, pady=PAD_SM)
    body.grid(row=1, column=0, sticky="nsew")
    body.columnconfigure(0, weight=1)

    icon_wrap = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
    icon_wrap.grid(row=0, column=0, pady=(0, PAD_XS))
    icon_badge = tk.Label(
        icon_wrap,
        text=icon,
        bg=PANEL,
        fg=_DISPLAY_FG.get(view.status, TEXT),
        font=FONT_FLOW_ICON,
        width=3,
        height=1,
        highlightthickness=0,
    )
    icon_badge.pack()

    title_lbl = tk.Label(
        body,
        text=view.label,
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_SEMIBOLD,
        anchor="center",
        wraplength=_STAGE_CARD_MIN_WIDTH - 12,
        justify=tk.CENTER,
    )
    title_lbl.grid(row=1, column=0, sticky="ew")

    chip_host = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
    chip_host.grid(row=2, column=0, pady=(PAD_XS, PAD_SM))
    _status_chip(chip_host, view).pack()

    conf_fg = (
        SUCCESS
        if view.status == STATUS_COMPLETED and pct is not None
        else _DISPLAY_FG.get(view.status, MUTED)
    )
    conf_txt = f"{pct}%" if pct is not None else ""
    warn_preview = _stage_warning_preview(view)
    from stablewalk.ui.tk.results_warning_copy import (
        classify_results_warning,
        format_results_warning_copy,
    )

    warn_kind = classify_results_warning(
        "" if warn_preview in ("None", "—") else warn_preview,
        status=view.status,
    )
    if warn_kind == "critical":
        warn_label = "Action needed"
        warn_display = format_results_warning_copy(
            warn_preview,
            label=view.label,
            status=view.status,
            kind="critical",
        )
        warn_fg = DANGER
    elif warn_kind == "info":
        warn_label = "Note"
        warn_display = format_results_warning_copy(
            warn_preview,
            label=view.label,
            status=view.status,
            kind="info",
        )
        warn_fg = MUTED
    else:
        warn_label = "Note"
        warn_display = "None"
        warn_fg = MUTED

    metrics = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
    metrics.grid(row=3, column=0, sticky="ew")
    metrics.columnconfigure(0, weight=1)

    confidence_display = (
        conf_txt
        if pct is not None
        else (
            _unavailable_stage_text(view.key)
            if view.status == STATUS_UNAVAILABLE
            else "Not calculated"
        )
    )
    rows: list[tuple[str, str, str]] = [
        ("Status", _stage_status_label(view), _DISPLAY_FG.get(view.status, MUTED)),
        ("Confidence", confidence_display, conf_fg if pct is not None else MUTED),
        (warn_label, warn_display, warn_fg),
    ]
    value_labels: list[tk.Label] = []
    for r, (lab, val, fg) in enumerate(rows):
        row_f = tk.Frame(metrics, bg=ELEVATED, highlightthickness=0)
        row_f.grid(row=r, column=0, sticky="nsew", pady=(2, 0))
        row_f.columnconfigure(0, weight=1)
        metrics.rowconfigure(r, weight=1, minsize=36 if r < 2 else 50)
        tk.Label(
            row_f, text=lab, bg=ELEVATED, fg=MUTED_DIM, font=FONT_CAPTION, anchor="w"
        ).grid(row=0, column=0, sticky="ew")
        val_lbl = tk.Label(
            row_f,
            text=val,
            bg=ELEVATED,
            fg=fg,
            font=FONT_UI_XS,
            anchor="nw",
            wraplength=max(72, _STAGE_CARD_WIDTH - PAD_SM * 2),
            justify=tk.LEFT,
        )
        val_lbl.grid(row=1, column=0, sticky="ew", pady=(1, 0))
        value_labels.append(val_lbl)

    track, fill = install_progress_track(body, row=len(rows), column=0)
    track.grid_configure(pady=(PAD_SM, 0))

    details_button = ttk.Button(
        body,
        text="Details ▸",
        style="Compact.TButton",
        takefocus=False,
    )
    details_button.grid(row=len(rows) + 1, column=0, sticky="ew", pady=(PAD_XS, 0))

    details_host = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
    details_host.grid(row=len(rows) + 2, column=0, sticky="ew", pady=(PAD_SM, 0))
    details_host.columnconfigure(0, weight=1)
    details_text = tk.Text(
        details_host,
        bg=ELEVATED,
        fg=TEXT_SECONDARY,
        font=FONT_UI_XS,
        height=10,
        width=1,
        wrap=tk.WORD,
        relief=tk.FLAT,
        borderwidth=0,
        highlightthickness=0,
        padx=0,
        pady=0,
        cursor="arrow",
    )
    details_scroll = ttk.Scrollbar(
        details_host,
        orient=tk.VERTICAL,
        command=details_text.yview,
    )
    details_text.configure(yscrollcommand=details_scroll.set)
    details_text.grid(row=0, column=0, sticky="nsew")
    details_scroll.grid(row=0, column=1, sticky="ns")
    details_text.insert("1.0", _stage_details_text(view))
    details_text.configure(state=tk.DISABLED)
    details_host.grid_remove()

    def _toggle_details() -> None:
        expanded = bool(getattr(details_button, "_pipeline_details_expanded", False))
        if expanded:
            details_host.grid_remove()
            details_button.configure(text="Details ▸")
        else:
            previous = getattr(gui, "_results_pipeline_open_details", None)
            if previous is not None and previous is not details_button:
                close_previous = getattr(previous, "_pipeline_close_details", None)
                if callable(close_previous):
                    close_previous()
            details_host.grid()
            details_button.configure(text="Details ▾")
            gui._results_pipeline_open_details = details_button
        details_button._pipeline_details_expanded = not expanded  # type: ignore[attr-defined]

    def _close_details() -> None:
        details_host.grid_remove()
        details_button.configure(text="Details ▸")
        details_button._pipeline_details_expanded = False  # type: ignore[attr-defined]
        if getattr(gui, "_results_pipeline_open_details", None) is details_button:
            gui._results_pipeline_open_details = None

    details_button._pipeline_close_details = _close_details  # type: ignore[attr-defined]
    details_button.configure(command=_toggle_details)

    tip_lines = [
        view.label,
        f"Status: {_stage_status_label(view)}",
        f"Confidence: {confidence_display}",
        f"{warn_label}: {warn_display}",
        "Expand Details for timing, outputs, warnings, artifacts, and logs.",
    ]
    tip = "\n".join(tip_lines)
    create_tooltip(card, tip, wraplength=320)

    assess_status = view.status
    if assess_status in (STATUS_RUNNING, STATUS_FAILED):
        assess_status = STATUS_PARTIAL
    elif assess_status not in (STATUS_COMPLETED, STATUS_PARTIAL, STATUS_UNAVAILABLE):
        assess_status = STATUS_UNAVAILABLE
    diagram_stage = PipelineDiagramStage(
        key=view.key,
        label=view.label,
        status=assess_status,  # type: ignore[arg-type]
        detail=view.detail,
        tooltip=view.tooltip,
        item_keys=view.item_keys,
    )
    bind_pipeline_stage_click(
        card,
        gui,
        view.key,
        diagram_stage=diagram_stage,
        skip_widgets=(details_button, details_host, details_text, details_scroll),
    )

    widgets = {
        "progress_track": track,
        "progress_fill": fill,
        "ring": accent,
        "details_button": details_button,
        "details_host": details_host,
    }
    prev_statuses: dict[str, str] = getattr(gui, "_results_pipeline_prev_status", {})
    prev = prev_statuses.get(view.key)
    if view.status == STATUS_COMPLETED:
        if prev != STATUS_COMPLETED:
            animate_stage_completion(
                gui, f"results:{view.key}", widgets, status=STATUS_COMPLETED
            )
        else:
            set_progress_fill(fill, 1.0, color=SUCCESS)
    elif view.status == STATUS_RUNNING:
        set_progress_fill(fill, 0.45, color=ORANGE)
        _start_running_pulse(gui, f"results-run:{view.key}", accent, fill)
    elif view.status == STATUS_FAILED:
        set_progress_fill(fill, 1.0, color=DANGER)
    elif view.status == STATUS_PARTIAL:
        set_progress_fill(fill, 0.55, color=ORANGE)
    else:
        set_progress_fill(fill, 0.0, color=BORDER)

    prev_statuses[view.key] = view.status
    gui._results_pipeline_prev_status = prev_statuses

    if not is_last:
        connector_col = column + 1
        parent.columnconfigure(connector_col, weight=0, minsize=_STAGE_CONNECTOR_WIDTH)
        connector = tk.Frame(
            parent, bg=PANEL, highlightthickness=0, width=_STAGE_CONNECTOR_WIDTH
        )
        connector.grid(row=grid_row, column=connector_col, sticky="ns")
        connector.grid_propagate(False)
        arrow = tk.Label(
            connector,
            text="→",
            bg=PANEL,
            fg=MUTED_DIM,
            font=FONT_FLOW_ARROW,
        )
        arrow.place(relx=0.5, rely=0.45, anchor="center")
        active = view.status == STATUS_COMPLETED or prev_completed
        if prev != STATUS_COMPLETED and view.status == STATUS_COMPLETED:
            animate_connector_activation(
                gui, f"results-conn:{view.key}", arrow, active=True
            )
        else:
            set_connector_state(arrow, active=active, active_fg=SUCCESS, dormant_fg=MUTED_DIM)

    del step
    return {
        "outer": outer,
        "card": card,
        "title": title_lbl,
        "value_labels": value_labels,
    }



def _start_running_pulse(
    gui: Any, anim_key: str, accent: tk.Misc, fill: tk.Misc
) -> None:
    """Pulse orange accent while a stage is running."""
    root = getattr(gui, "root", None)
    if root is None:
        return
    after_ids: dict[str, str] = getattr(gui, "_pipeline_anim_after_ids", {})
    prev = after_ids.pop(anim_key, None)
    if prev is not None:
        try:
            root.after_cancel(prev)
        except tk.TclError:
            pass

    palette = (ORANGE, "#e0a050", ORANGE, "#c47a28")
    state = {"i": 0}

    def _tick() -> None:
        # Stop if card was destroyed or stage no longer running.
        try:
            if not accent.winfo_exists():
                after_ids.pop(anim_key, None)
                return
        except tk.TclError:
            after_ids.pop(anim_key, None)
            return
        color = palette[state["i"] % len(palette)]
        state["i"] += 1
        try:
            accent.configure(bg=color)
            set_progress_fill(fill, 0.35 + 0.15 * (state["i"] % 3), color=color)
        except tk.TclError:
            after_ids.pop(anim_key, None)
            return
        after_ids[anim_key] = root.after(120, _tick)
        gui._pipeline_anim_after_ids = after_ids

    _tick()


def _render_summary_strip(
    host: tk.Misc,
    views: list[_ResultsStageView],
    *,
    gui: Any | None = None,
    progress: ResultsPipelineProgress | None = None,
) -> None:
    gui._results_pipeline_open_details = None
    for child in host.winfo_children():
        child.destroy()
    required_views = [
        view for view in views if view.key not in OPTIONAL_RESULTS_STAGE_KEYS
    ]
    optional_views = [
        view for view in views if view.key in OPTIONAL_RESULTS_STAGE_KEYS
    ]
    completed = sum(1 for v in required_views if v.status == STATUS_COMPLETED)
    running = sum(1 for v in required_views if v.status == STATUS_RUNNING)
    failed = sum(1 for v in required_views if v.status == STATUS_FAILED)
    partial = sum(1 for v in required_views if v.status == STATUS_PARTIAL)
    pending = sum(1 for v in required_views if v.status == STATUS_UNAVAILABLE)
    session_state = results_pipeline_session_state(views)
    analysis_active = bool(
        gui is not None
        and (
            getattr(gui, "_analysis_compute_active", False)
            or session_state == "processing"
        )
    )
    headline, headline_kind = format_required_analysis_headline(
        views, analysis_active=analysis_active
    )
    if gui is not None:
        gui._results_pipeline_headline = headline
        gui._results_pipeline_headline_kind = headline_kind
        gui._results_pipeline_views = views
    if progress is None and gui is not None:
        progress = compute_results_pipeline_progress(gui, views)
    if progress is None:
        progress = ResultsPipelineProgress(
            completion_pct=100.0 * completed / max(1, len(required_views)),
            total_time_s=None,
            total_time_text="—",
            avg_fps=None,
            avg_fps_text="—",
            remaining_count=pending + running + failed,
            remaining_text=f"{pending + running + failed}",
            per_stage_text="No stage timings yet",
            completed=completed,
            total_stages=len(required_views),
        )

    tk.Label(
        host,
        text="How this session was processed · click a stage for details",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(fill=tk.X)

    # Overall pipeline progress indicator (stage cards below stay unchanged).
    progress_card = tk.Frame(
        host,
        bg=ELEVATED,
        highlightthickness=0,
    )
    progress_card.pack(fill=tk.X, pady=(PAD_SM, 0))
    progress_card.columnconfigure(0, weight=1)

    title_row = tk.Frame(progress_card, bg=ELEVATED, highlightthickness=0)
    title_row.grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_SM, PAD_XS))
    tk.Label(
        title_row,
        text="Required analysis progress",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_SEMIBOLD,
        anchor="w",
    ).pack(side=tk.LEFT)
    headline_fg = {
        "complete": SUCCESS,
        "processing": ORANGE,
        "incomplete": DANGER,
        "waiting": MUTED,
        "partial": ORANGE,
    }.get(headline_kind, TEXT)

    # Completion is already stated in the findings header. Keep the
    # stage-specific headline only when processing needs attention.
    headline_lbl = None
    if headline_kind != "complete":
        headline_row = tk.Frame(progress_card, bg=ELEVATED, highlightthickness=0)
        headline_row.grid(row=1, column=0, sticky="ew", padx=PAD_SM, pady=(0, PAD_XS))
        headline_lbl = tk.Label(
            headline_row,
            text=headline,
            bg=ELEVATED,
            fg=headline_fg,
            font=FONT_UI_SEMIBOLD,
            anchor="w",
            justify=tk.LEFT,
            wraplength=520,
        )
        headline_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

    track_host = tk.Frame(progress_card, bg=ELEVATED, highlightthickness=0)
    track_host.grid(row=2, column=0, sticky="ew", padx=PAD_SM)
    track_host.columnconfigure(0, weight=1)
    track, fill = install_progress_track(track_host, row=0, column=0)
    track.configure(height=8)
    try:
        track.grid_configure(sticky="ew", pady=(0, PAD_SM))
    except tk.TclError:
        pass
    bar_color = (
        SUCCESS
        if headline_kind == "complete"
        else ORANGE
        if headline_kind == "processing"
        else DANGER
        if headline_kind == "incomplete"
        else BORDER
    )
    fill_frac = 1.0 if headline_kind == "complete" else (
        0.35 if headline_kind == "processing" else 0.0 if headline_kind == "waiting" else 0.55
    )
    set_progress_fill(fill, fill_frac, color=bar_color)

    metrics_row = tk.Frame(progress_card, bg=ELEVATED, highlightthickness=0)
    metrics_row.grid(row=3, column=0, sticky="ew", padx=PAD_SM, pady=(0, PAD_SM))
    for col in range(4):
        metrics_row.columnconfigure(col, weight=1, uniform="pipe_progress")
    completion_value = (
        "100%"
        if headline_kind == "complete"
        else "Processing"
        if headline_kind == "processing"
        else "Waiting"
        if headline_kind == "waiting"
        else "Incomplete"
    )
    metric_cells = (
        ("Required completion", completion_value),
        ("Total execution time", progress.total_time_text),
        ("Average processing FPS", progress.avg_fps_text),
        ("Required stages incomplete", progress.remaining_text),
    )
    metric_value_labels: list[tk.Label] = []
    for col, (title, value) in enumerate(metric_cells):
        cell = tk.Frame(metrics_row, bg=ELEVATED, highlightthickness=0)
        cell.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else PAD_SM, 0))
        tk.Label(
            cell,
            text=title,
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_KPI_TITLE,
            anchor="w",
        ).pack(anchor="w")
        value_lbl = tk.Label(
            cell,
            text=value,
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_UI_SM,
            anchor="w",
            wraplength=140,
            justify=tk.LEFT,
        )
        value_lbl.pack(anchor="w", pady=(2, 0))
        metric_value_labels.append(value_lbl)

    stage_times = tk.Frame(progress_card, bg=ELEVATED, highlightthickness=0)
    stage_times.grid(row=4, column=0, sticky="ew", padx=PAD_SM, pady=(0, PAD_SM))
    tk.Label(
        stage_times,
        text="Per-stage execution time",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_KPI_TITLE,
        anchor="w",
    ).pack(anchor="w")
    stage_times_lbl = tk.Label(
        stage_times,
        text=progress.per_stage_text,
        bg=ELEVATED,
        fg=TEXT_SECONDARY,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=480,
    )
    stage_times_lbl.pack(anchor="w", fill=tk.X, pady=(2, 0))

    row = tk.Frame(host, bg=PANEL, highlightthickness=0)
    row.pack(fill=tk.X, pady=(PAD_XS, 0))
    for col in range(5):
        row.columnconfigure(col, weight=1, uniform="pipe_status_chips")
    chips = (
        (
            f"{completed}/{len(required_views)} required",
            SUCCESS if completed == len(required_views) and completed else MUTED,
        ),
        (f"{running} running", ORANGE if running else MUTED),
        (f"{failed} failed", DANGER if failed else MUTED),
        (f"{partial} partial", ORANGE if partial else MUTED),
        (f"{pending} pending", MUTED if pending == 0 else MUTED_DIM),
    )
    for col, (text, fg) in enumerate(chips):
        chip = tk.Frame(row, bg=ELEVATED, highlightthickness=0)
        chip.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else PAD_XS, 0))
        tk.Label(chip, text=text, bg=ELEVATED, fg=fg, font=FONT_UI_XS).pack(
            padx=PAD_SM, pady=PAD_XS
        )

    optional_line = "  ·  ".join(
        f"{view.label}: {_stage_status_label(view)}" for view in optional_views
    )
    if optional_line:
        tk.Label(
            host,
            text=optional_line,
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        ).pack(fill=tk.X, pady=(PAD_XS, 0))

    def _sync_progress_wrap(_event: object | None = None) -> None:
        try:
            width = int(progress_card.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 80:
            return
        try:
            stage_times_lbl.configure(wraplength=max(200, width - 36))
        except tk.TclError:
            pass
        if headline_lbl is not None:
            try:
                headline_lbl.configure(wraplength=max(200, width - 36))
            except tk.TclError:
                pass
        cell_wrap = max(72, (width // 4) - 24)
        for lbl in metric_value_labels:
            try:
                lbl.configure(wraplength=cell_wrap)
            except tk.TclError:
                pass

    progress_card.bind("<Configure>", _sync_progress_wrap, add="+")
    _sync_progress_wrap()

    if gui is not None:
        try:
            from stablewalk.ui.tk.dashboard_summary import sync_results_primary_findings

            sync_results_primary_findings(gui, views)
        except Exception:
            pass


def _scale_results_pipeline_cards(gui: Any, _event: object | None = None) -> None:
    """Fit stage-card text wraplengths to the current equal-width cells."""
    host = getattr(gui, "_results_pipeline_host", None)
    cards = getattr(gui, "_results_pipeline_card_widgets", None)
    if host is None:
        return
    try:
        width = int(host.winfo_width() or 0)
        height = int(host.winfo_height() or 0)
    except tk.TclError:
        return
    parent = getattr(host, "master", None)
    if parent is not None:
        try:
            parent_h = int(parent.winfo_height() or 0)
            if parent_h > 1:
                height = parent_h if height <= 1 else min(height, parent_h)
        except tk.TclError:
            pass
    slot = getattr(gui, "_results_pipeline_slot", None)
    if slot is not None:
        try:
            slot_h = int(slot.winfo_height() or 0)
            if slot_h > 1:
                # Processing Details header + progress strip share this slot.
                budget = max(96, slot_h - 48)
                height = budget if height <= 1 else min(height, budget)
        except tk.TclError:
            pass
    n = len(cards or ())
    if width < 80 or n <= 0:
        return
    wrap = int(getattr(gui, "_results_pipeline_wrap_cols", 0) or n)
    wrap = max(1, min(wrap, n))
    row_count = max(1, (n + wrap - 1) // wrap)
    if height > 1:
        available_each = max(
            48,
            (height - max(0, row_count - 1) * PAD_XS) // row_count,
        )
        row_floor = min(_STAGE_CARD_MIN_HEIGHT, available_each)
        for row in range(row_count):
            try:
                host.rowconfigure(row, weight=1, minsize=row_floor)
            except tk.TclError:
                pass
    connectors = max(0, wrap - 1) * _STAGE_CONNECTOR_WIDTH
    card_w = max(_STAGE_CARD_MIN_WIDTH, (width - connectors - 8) // wrap)
    title_wrap = max(56, card_w - 16)
    value_wrap = max(48, card_w - 40)
    for entry in cards or ():
        title = entry.get("title")
        if title is not None:
            try:
                title.configure(wraplength=title_wrap)
            except tk.TclError:
                pass
        for lbl in entry.get("value_labels") or ():
            try:
                lbl.configure(wraplength=value_wrap)
            except tk.TclError:
                pass


def build_results_analysis_pipeline(gui: Any, parent: tk.Misc) -> tk.Frame:
    """Install the responsive Results processing pipeline into ``parent``."""
    root = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    summary = tk.Frame(root, bg=PANEL, highlightthickness=0)
    summary.grid(row=0, column=0, sticky="ew", pady=(0, PAD_SM))
    gui._results_pipeline_summary = summary

    # Equal-width stage row — scales to available width (no horizontal scrollbar).
    host = tk.Frame(root, bg=PANEL, highlightthickness=0)
    host.grid(row=1, column=0, sticky="nsew")
    host.rowconfigure(0, weight=1, minsize=_STAGE_CARD_ROW_FLOOR)

    def _on_host_configure(event: object | None = None) -> None:
        _scale_results_pipeline_cards(gui, event)
        if getattr(gui, "_results_pipeline_rebuilding", False):
            return
        try:
            width = int(host.winfo_width() or 0)
        except tk.TclError:
            return
        n = len(RESULTS_PIPELINE_STAGES)
        prev = getattr(gui, "_results_pipeline_wrap_cols", None)
        wrap = results_pipeline_wrap_columns(width, n, previous=prev)
        if wrap == prev:
            return
        gui._results_pipeline_wrap_cols = wrap
        pending = getattr(gui, "_results_pipeline_wrap_after", None)
        if pending is not None:
            try:
                host.after_cancel(pending)
            except Exception:
                pass

        def _rebuild_wrapped_pipeline() -> None:
            gui._results_pipeline_wrap_after = None
            if getattr(gui, "playing", False):
                return
            report = resolve_results_pipeline_report(gui, None)
            try:
                from stablewalk.ui.tk.playback_render_policy import clear_paint_signatures

                clear_paint_signatures(gui, "results_pipeline")
            except Exception:
                pass
            gui._results_pipeline_rebuilding = True
            try:
                update_results_analysis_pipeline(gui, report)
            except Exception:
                pass
            finally:
                gui._results_pipeline_rebuilding = False

        try:
            gui._results_pipeline_wrap_after = host.after(80, _rebuild_wrapped_pipeline)
        except tk.TclError:
            _rebuild_wrapped_pipeline()

    host.bind("<Configure>", _on_host_configure, add="+")

    gui._results_pipeline_host = host
    gui._results_pipeline_canvas = None  # legacy attr; horizontal canvas removed
    gui._results_pipeline_root = root
    gui._results_pipeline_card_widgets = []
    gui._results_pipeline_expanded = getattr(gui, "_results_pipeline_expanded", {})
    gui._results_pipeline_prev_status = getattr(gui, "_results_pipeline_prev_status", {})
    return root


def _results_pipeline_view_signature(views: list[_ResultsStageView]) -> tuple:
    """Compact signature of visible pipeline data (skip rebuild when unchanged)."""
    return tuple(
        (
            v.key,
            v.status,
            v.duration_text,
            v.confidence_text,
            v.warnings,
            v.artifacts_text,
            v.detail,
        )
        for v in views
    )


def update_results_analysis_pipeline(
    gui: Any,
    report: PipelineStatusReport | None,
    *,
    force: bool = False,
) -> None:
    """Refresh Results pipeline cards from the latest status + summary cache."""
    if getattr(gui, "playing", False) and not force:
        return
    host = getattr(gui, "_results_pipeline_host", None)
    summary = getattr(gui, "_results_pipeline_summary", None)
    if host is None:
        return

    views = build_results_stage_views(gui, resolve_results_pipeline_report(gui, report))
    progress = compute_results_pipeline_progress(gui, views)
    gui._results_pipeline_session_state = results_pipeline_session_state(views)
    gui._results_metrics_stage_status = next(
        (view.status for view in views if view.key == "metrics"),
        STATUS_UNAVAILABLE,
    )
    session_state = gui._results_pipeline_session_state
    analysis_active = bool(
        getattr(gui, "_analysis_compute_active", False)
        or getattr(gui, "_analysis_running", False)
        or session_state == "processing"
    )
    try:
        from stablewalk.ui.tk.analyze_progress import analyze_progress_active

        analysis_active = analysis_active or analyze_progress_active(gui)
    except Exception:
        pass
    headline, headline_kind = format_required_analysis_headline(
        views, analysis_active=analysis_active
    )
    gui._results_pipeline_headline = headline
    gui._results_pipeline_headline_kind = headline_kind
    gui._results_pipeline_views = views
    try:
        from stablewalk.ui.tk.dashboard_summary import (
            sync_results_report_preview,
            update_results_summary_panel,
        )

        if getattr(gui, "_analysis_summary_cache", None) is not None:
            update_results_summary_panel(
                gui,
                gui._analysis_summary_cache,
                refresh_pipeline=False,
            )
        else:
            sync_results_report_preview(gui)
    except Exception:
        pass
    # Always refresh the overall progress strip (independent of stage cards).
    if summary is not None:
        _render_summary_strip(summary, views, gui=gui, progress=progress)

    n = len(views)
    wrap = int(getattr(gui, "_results_pipeline_wrap_cols", 0) or 0)
    if wrap <= 0:
        try:
            width = int(host.winfo_width() or 0)
        except (tk.TclError, AttributeError, TypeError, ValueError):
            width = 0
        wrap = results_pipeline_wrap_columns(width, n or 8)
        gui._results_pipeline_wrap_cols = wrap
    wrap = max(1, min(wrap, n or 1))

    sig = (_results_pipeline_view_signature(views), wrap)
    # Skip full card destroy/rebuild when every visible field is unchanged.
    try:
        has_children = bool(host.winfo_children())
    except tk.TclError:
        has_children = False
    from stablewalk.ui.tk.playback_render_policy import paint_signature_changed

    if has_children and not paint_signature_changed(gui, "results_pipeline", sig):
        return
    if not has_children:
        # First paint (or host was cleared) — record signature for later skips.
        paint_signature_changed(gui, "results_pipeline", sig)

    for child in host.winfo_children():
        child.destroy()
    _reset_pipeline_host_grid(host)

    prev_completed = False
    card_widgets: list[dict[str, Any]] = []
    for index, view in enumerate(views):
        row = index // wrap
        pos = index % wrap
        col = pos * 2
        host.columnconfigure(
            col, weight=1, uniform="results_stage", minsize=_STAGE_CARD_MIN_WIDTH
        )
        host.rowconfigure(row, weight=1, minsize=_STAGE_CARD_ROW_FLOOR)
        last_in_row = pos == wrap - 1 or index == n - 1
        entry = _build_results_stage_card(
            host,
            col,
            view,
            gui,
            report,
            step=index + 1,
            is_last=last_in_row,
            prev_completed=prev_completed,
            grid_row=row,
        )
        card_widgets.append(entry)
        if view.status == STATUS_COMPLETED:
            prev_completed = True

    gui._results_pipeline_card_widgets = card_widgets
    try:
        host.update_idletasks()
    except tk.TclError:
        pass
    _scale_results_pipeline_cards(gui)


__all__ = [
    "OPTIONAL_RESULTS_STAGE_KEYS",
    "REQUIRED_RESULTS_STAGE_KEYS",
    "RESULTS_PIPELINE_STAGES",
    "STATUS_FAILED",
    "STATUS_RUNNING",
    "ResultsPipelineProgress",
    "_STAGE_ICONS",
    "build_results_analysis_pipeline",
    "build_results_stage_views",
    "compute_results_pipeline_progress",
    "format_required_analysis_headline",
    "metrics_stage_missing_dependencies",
    "results_pipeline_wrap_columns",
    "stage_incomplete_reason",
    "resolve_results_pipeline_report",
    "results_pipeline_session_state",
    "update_results_analysis_pipeline",
]
