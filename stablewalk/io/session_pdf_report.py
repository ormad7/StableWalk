"""
Professional StableWalk Research PDF report (ReportLab).

Multi-page research / laboratory report including:
patient/session info, screenshots, pipeline summary, gait quality,
stability, cadence, ROM, COM, symmetry, heel strikes, contact timing,
graph snapshots, comparison summary, clinical-style interpretation,
confidence, and processing information.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stablewalk import config
from stablewalk.analysis.analysis_summary import AnalysisSummary
from stablewalk.analysis.biomechanical.orchestrator import BiomechanicalAnalysisResult
from stablewalk.analysis.estimated_vgrf_analysis import EstimatedVGRFResult
from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult
from stablewalk.analysis.session_compare import com_path_xyz, knee_angle_series
from stablewalk.models.pose_data import PoseSequence
from stablewalk.ui.scientific_labels import (
    DISCLAIMER_RESEARCH,
    METHODS_OVERVIEW,
    UNITS_COORD_LEGEND,
)

logger = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = "2.0"
DISCLAIMER = DISCLAIMER_RESEARCH
REPORT_TITLE = "StableWalk Research Report"


@dataclass
class SessionPdfReportContext:
    """All inputs required to render a professional research PDF."""

    source: str = ""
    run_name: str = ""
    session_label: str = ""
    patient_id: str = ""
    patient_name: str = ""
    notes: str = ""
    generated_at: str = ""
    fps: float | None = None
    n_frames: int = 0
    detected_frames: int = 0
    duration_s: float | None = None
    selected_joints: list[str] = field(default_factory=list)
    summary: AnalysisSummary | None = None
    biomechanical: BiomechanicalAnalysisResult | None = None
    foot_contact: FootContactAnalysisResult | None = None
    cycles: GaitCycleAnalysisResult | None = None
    estimated_vgrf: EstimatedVGRFResult | None = None
    sequence: PoseSequence | None = None
    video_path: str | None = None
    frame_paths: list[Path] = field(default_factory=list)
    pipeline_status: dict[str, Any] | None = None
    skeleton_xyz: np.ndarray | None = None
    skeleton_connections: list[tuple[int, int]] = field(default_factory=list)
    conclusion: str = ""
    recommendations: str = ""
    app_version: str = ""
    pose_backend: str = ""
    comparison_summary: str = ""
    clinical_interpretation: str = ""
    processing_lines: list[str] = field(default_factory=list)
    screenshot_paths: list[Path] = field(default_factory=list)


def _require_reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image as RLImage,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ReportLab is required for PDF reports. Install with:\n"
            "  pip install reportlab>=4.0.0"
        ) from exc
    return {
        "colors": colors,
        "TA_CENTER": TA_CENTER,
        "TA_JUSTIFY": TA_JUSTIFY,
        "TA_LEFT": TA_LEFT,
        "A4": A4,
        "ParagraphStyle": ParagraphStyle,
        "getSampleStyleSheet": getSampleStyleSheet,
        "mm": mm,
        "RLImage": RLImage,
        "PageBreak": PageBreak,
        "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
        "Table": Table,
        "TableStyle": TableStyle,
    }


def generate_project_logo(path: Path | None = None, *, size: tuple[int, int] = (520, 140)) -> Path:
    """Create a simple StableWalk wordmark logo (PNG) if none is bundled."""
    out = path or (config.REPORTS_DIR / "_stablewalk_logo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size > 0:
        return out

    w, h = size
    img = Image.new("RGB", (w, h), "#0f172a")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 18, h], fill="#38bdf8")
    draw.rectangle([0, h - 10, w, h], fill="#0369a1")
    try:
        font_lg = ImageFont.truetype("arial.ttf", 42)
        font_sm = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font_lg = ImageFont.load_default()
        font_sm = ImageFont.load_default()
    draw.text((40, 36), "StableWalk", fill="#f8fafc", font=font_lg)
    draw.text(
        (42, 90),
        "Gait Biomechanics Laboratory Report",
        fill="#94a3b8",
        font=font_sm,
    )
    img.save(out, format="PNG")
    return out


def _load_video_thumbnail(
    *,
    frame_paths: Sequence[Path],
    video_path: str | None,
    progress: float = 0.35,
) -> Image.Image | None:
    if frame_paths:
        idx = int(round(progress * (len(frame_paths) - 1)))
        idx = max(0, min(idx, len(frame_paths) - 1))
        try:
            return Image.open(frame_paths[idx]).convert("RGB")
        except OSError:
            pass
    if video_path and Path(str(video_path)).is_file():
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            try:
                if not cap.isOpened():
                    return None
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                frame_i = int(round(progress * max(total - 1, 0)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    return None
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
        except Exception:
            return None
    return None


def _save_pil(img: Image.Image, path: Path, *, max_side: int = 900) -> Path:
    clone = img.copy()
    clone.thumbnail((max_side, max_side))
    path.parent.mkdir(parents=True, exist_ok=True)
    clone.save(path, format="PNG")
    return path


def _fig_to_png(fig: plt.Figure, path: Path, *, dpi: int = 140) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _render_knee_graph(ctx: SessionPdfReportContext, path: Path) -> Path | None:
    if ctx.sequence is None:
        return None
    t, y = knee_angle_series(ctx.sequence)
    if len(t) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=120)
    ax.plot(t, [np.nan if v is None else v for v in y], color="#0369a1", linewidth=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Knee flexion (°)")
    ax.set_title("Selected joint kinematics — knee flexion (L/R mean)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig, path)


def _render_com_graph(ctx: SessionPdfReportContext, path: Path) -> Path | None:
    ba = ctx.biomechanical
    if ba is None or ba.center_of_mass is None or ba.center_of_mass.positions is None:
        return None
    pos = np.asarray(ba.center_of_mass.positions, dtype=float)
    if pos.ndim != 2 or pos.shape[0] < 3:
        return None
    t = np.arange(pos.shape[0], dtype=float)
    if ctx.fps and ctx.fps > 1e-6:
        t = t / float(ctx.fps)
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=120)
    ax.plot(t, pos[:, 1], color="#0f766e", linewidth=1.6, label="Vertical COM")
    if pos.shape[1] >= 3:
        ax.plot(t, pos[:, 0], color="#7c3aed", linewidth=1.0, alpha=0.7, label="Lateral")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m, body-normalized)")
    ax.set_title("Centre of mass trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig, path)


def _render_trajectory_3d(ctx: SessionPdfReportContext, path: Path) -> Path | None:
    arr = com_path_xyz(ctx.biomechanical)
    if arr is None:
        return None
    fig = plt.figure(figsize=(5.2, 4.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color="#0369a1", linewidth=1.8)
    ax.scatter([arr[0, 0]], [arr[0, 1]], [arr[0, 2]], c="#16a34a", s=36, label="Start")
    ax.scatter([arr[-1, 0]], [arr[-1, 1]], [arr[-1, 2]], c="#dc2626", s=36, label="End")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D COM trajectory")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _fig_to_png(fig, path)


def _render_skeleton_image(ctx: SessionPdfReportContext, path: Path) -> Path | None:
    """Render a simple 3D skeleton stick figure when joint cache is available."""
    xyz = ctx.skeleton_xyz
    if xyz is None or len(xyz) < 3:
        if ctx.sequence is None or not ctx.sequence.frames:
            return None
        frames = [f for f in ctx.sequence.frames if f.detected and f.keypoints]
        if not frames:
            return None
        fr = frames[len(frames) // 2]
        pts = []
        for kp in fr.keypoints:
            if kp is None:
                continue
            x = getattr(kp, "x", None)
            y = getattr(kp, "y", None)
            z = getattr(kp, "z", None)
            if x is None or y is None:
                continue
            pts.append((float(x), float(y), float(z or 0.0)))
        if len(pts) < 5:
            return None
        xyz = np.asarray(pts, dtype=float)

    fig = plt.figure(figsize=(4.6, 4.6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="#0369a1", s=18)
    for a, b in ctx.skeleton_connections:
        if 0 <= a < len(xyz) and 0 <= b < len(xyz):
            ax.plot(
                [xyz[a, 0], xyz[b, 0]],
                [xyz[a, 1], xyz[b, 1]],
                [xyz[a, 2], xyz[b, 2]],
                color="#0f172a",
                linewidth=1.6,
            )
    ax.set_title("Skeleton snapshot")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    return _fig_to_png(fig, path)


def _render_contact_graph(ctx: SessionPdfReportContext, path: Path) -> Path | None:
    contact = ctx.foot_contact
    frames = getattr(contact, "per_frame", None) if contact is not None else None
    if not frames:
        return None
    t = [float(f.time_s) for f in frames]
    left = [float(f.left_contact_binary) for f in frames]
    right = [float(f.right_contact_binary) + 1.15 for f in frames]
    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=120)
    ax.fill_between(t, 0, left, step="mid", alpha=0.55, color="#2563eb", label="Left contact")
    ax.fill_between(t, 1.15, right, step="mid", alpha=0.55, color="#dc2626", label="Right contact")
    ax.set_yticks([0.5, 1.65])
    ax.set_yticklabels(["Left", "Right"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Foot contact timing (estimated)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig, path)


def _render_heel_strike_graph(ctx: SessionPdfReportContext, path: Path) -> Path | None:
    """Mark estimated heel-strike events on the contact timeline."""
    cycles = ctx.cycles
    contact = ctx.foot_contact
    events = getattr(cycles, "events", None) if cycles is not None else None
    frames = getattr(contact, "per_frame", None) if contact is not None else None
    if not events and not frames:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 2.2), dpi=120)
    if frames:
        t = [float(f.time_s) for f in frames]
        left = [float(getattr(f, "left_contact_binary", 0)) for f in frames]
        right = [float(getattr(f, "right_contact_binary", 0)) + 1.15 for f in frames]
        ax.fill_between(t, 0, left, step="mid", alpha=0.35, color="#2563eb")
        ax.fill_between(t, 1.15, right, step="mid", alpha=0.35, color="#dc2626")

    lhs = [
        float(e.time_s)
        for e in (events or [])
        if getattr(e, "event_type", "") == "left_heel_strike"
    ]
    rhs = [
        float(e.time_s)
        for e in (events or [])
        if getattr(e, "event_type", "") == "right_heel_strike"
    ]
    for ti in lhs:
        ax.axvline(ti, color="#1d4ed8", linewidth=1.2, alpha=0.9)
    for ti in rhs:
        ax.axvline(ti, color="#b91c1c", linewidth=1.2, alpha=0.9, linestyle="--")
    if lhs:
        ax.plot([], [], color="#1d4ed8", label=f"Left HS (n={len(lhs)})")
    if rhs:
        ax.plot([], [], color="#b91c1c", linestyle="--", label=f"Right HS (n={len(rhs)})")
    ax.set_yticks([0.5, 1.65])
    ax.set_yticklabels(["Left", "Right"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Heel strikes (estimated timing)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig, path)


def _render_stability_margin_graph(ctx: SessionPdfReportContext, path: Path) -> Path | None:
    ba = ctx.biomechanical
    sm = getattr(ba, "stability_margin", None) if ba is not None else None
    frames = getattr(sm, "per_frame", None) if sm is not None else None
    if not frames:
        return None
    t = [float(f.time_s) for f in frames]
    y = [
        float(f.stability_margin_m)
        if getattr(f, "stability_margin_m", None) is not None
        else np.nan
        for f in frames
    ]
    if not any(np.isfinite(v) for v in y):
        return None
    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=120)
    ax.plot(t, y, color="#0f766e", linewidth=1.5, label="COM–BoS signed distance")
    ax.axhline(
        0.0,
        color="#64748b",
        linestyle="--",
        linewidth=1.0,
        label="BoS boundary (signed distance = 0)",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signed distance (m)")
    ax.set_title("COM–BoS signed distance (unvalidated preview)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig, path)


def _render_vgrf_graph(ctx: SessionPdfReportContext, path: Path) -> Path | None:
    vgrf = ctx.estimated_vgrf
    if vgrf is None or not getattr(vgrf, "available", False):
        return None
    t = np.asarray(getattr(vgrf, "timestamps", []), dtype=float)
    total = np.asarray(getattr(vgrf, "total_vgrf_bw", None) or [], dtype=float)
    if total.size == 0:
        left = np.asarray(getattr(vgrf, "left_vgrf_bw", []), dtype=float)
        right = np.asarray(getattr(vgrf, "right_vgrf_bw", []), dtype=float)
        if left.size and right.size and left.size == right.size:
            total = left + right
    if t.size < 2 or total.size != t.size:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=120)
    ax.plot(t, total, color="#7c3aed", linewidth=1.5, label="Total vGRF")
    left = np.asarray(getattr(vgrf, "left_vgrf_bw", []), dtype=float)
    right = np.asarray(getattr(vgrf, "right_vgrf_bw", []), dtype=float)
    if left.size == t.size:
        ax.plot(t, left, color="#2563eb", linewidth=1.0, alpha=0.75, label="Left")
    if right.size == t.size:
        ax.plot(t, right, color="#dc2626", linewidth=1.0, alpha=0.75, label="Right")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (BW)")
    ax.set_title("Estimated Virtual GRF (pose-based)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig, path)


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _fmt(v: Any, *, digits: int = 2, suffix: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}{suffix}"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(fv):
        return "—"
    return f"{fv:.{digits}f}{suffix}"


def _field_value(summary: AnalysisSummary | None, attr: str) -> str:
    if summary is None:
        return "—"
    fld = getattr(summary, attr, None)
    if fld is None:
        return "—"
    if not getattr(fld, "available", True):
        return getattr(fld, "value", None) or "Not available"
    return str(getattr(fld, "value", "—"))


def build_automatic_conclusion(ctx: SessionPdfReportContext) -> str:
    """Compose a non-diagnostic laboratory conclusion from session metrics."""
    parts: list[str] = []
    s = ctx.summary
    ba = ctx.biomechanical
    if s is not None and s.scientific_interpretation:
        parts.append(s.scientific_interpretation.strip())
    else:
        parts.append(
            "Pose-based gait analysis was completed for this session. "
            "Reported metrics are estimated from monocular video."
        )

    quality = _field_value(s, "overall_gait_quality")
    cadence = _field_value(s, "cadence")
    speed = _field_value(s, "walking_speed")
    stab = _field_value(s, "stability_margin")
    parts.append(
        f"Key session descriptors — gait quality: {quality}; cadence: {cadence}; "
        f"walking speed: {speed}; stability: {stab}."
    )

    if ba is not None and ba.gait_quality is not None:
        score = ba.gait_quality.score
        if score is not None and score < 55:
            parts.append(
                "Overall gait quality score is reduced relative to typical lab demos, "
                "suggesting greater kinematic variability or asymmetry in this clip."
            )
        elif score is not None and score >= 75:
            parts.append(
                "Overall gait quality score is relatively favourable for this recording "
                "under StableWalk's monocular scoring conventions."
            )

    if ctx.cycles is not None:
        n = getattr(ctx.cycles.metrics, "gait_cycle_count", None)
        if n is not None:
            parts.append(f"Usable gait cycles detected: {n}.")

    parts.append(DISCLAIMER)
    return " ".join(parts)


def build_recommendations(ctx: SessionPdfReportContext) -> str:
    """Actionable research/lab recommendations (non-clinical)."""
    tips: list[str] = [
        "Review overlaid joint kinematics and the ROM table before drawing relative conclusions.",
        "Prefer multiple walking trials and consistent camera viewpoint for within-subject comparisons.",
    ]
    s = ctx.summary
    if s is not None:
        track = _field_value(s, "tracking_confidence")
        if "low" in track.lower() or "limited" in track.lower():
            tips.append(
                "Tracking confidence appears limited — re-record with fuller body visibility "
                "and improved lighting before quantitative reporting."
            )
        video_q = _field_value(s, "video_quality")
        if "poor" in video_q.lower() or "low" in video_q.lower():
            tips.append(
                "Video quality flags suggest resolution or motion blur may affect landmark stability."
            )
    if ctx.foot_contact is None:
        tips.append("Foot-contact analysis was unavailable; confirm pose enrichment completed.")
    if ctx.biomechanical is None or ctx.biomechanical.center_of_mass is None:
        tips.append("COM analysis was incomplete; ensure biomechanical analysis ran successfully.")
    tips.append(
        "Export OpenSim mapped TRC/MOT when SDK integration is required for musculoskeletal IK."
    )
    tips.append(
        "Do not use this PDF alone for clinical certification, diagnosis, or disability decisions."
    )
    return "\n".join(f"• {t}" for t in tips)


def build_validation_warnings(ctx: SessionPdfReportContext) -> list[str]:
    """Collect scientific-validity warnings included with exported metrics."""
    warnings: list[str] = []
    if ctx.cycles is not None:
        warnings.extend(ctx.cycles.warnings)
    if ctx.biomechanical is not None:
        warnings.extend(ctx.biomechanical.warnings)

    # OpenSim residual / pose↔IK agreement when session artifacts exist on disk.
    run_name = getattr(ctx, "run_name", None) or ""
    if run_name:
        try:
            from stablewalk.analysis.pipeline_validation import (
                compare_pose_mot_to_ik_mot,
                find_ik_marker_errors_sto,
                find_ik_mot,
                find_pose_mot,
                opensim_session_dir,
                parse_ik_marker_errors,
            )

            session_dir = opensim_session_dir(str(run_name))
            sto = find_ik_marker_errors_sto(session_dir)
            ik_err = parse_ik_marker_errors(sto) if sto else None
            if ik_err is not None:
                warnings.append(ik_err.summary_line())
                warnings.extend(ik_err.warnings)
            pose_mot = find_pose_mot(session_dir, str(run_name))
            ik_mot = find_ik_mot(session_dir, str(run_name))
            if pose_mot is not None and ik_mot is not None:
                agree = compare_pose_mot_to_ik_mot(pose_mot, ik_mot)
                if agree is not None:
                    warnings.append(agree.summary_line())
                    warnings.extend(agree.warnings[:2])
        except Exception:
            logger.debug("OpenSim validation warnings unavailable", exc_info=True)

    return list(dict.fromkeys(w.strip() for w in warnings if w and w.strip()))


def build_session_pdf_context_from_gui(gui: Any) -> SessionPdfReportContext:
    """Collect report inputs from the live StableWalk GUI session."""
    from stablewalk.analysis.analysis_summary import build_analysis_summary

    sequence = getattr(gui, "sequence", None)
    source = (
        getattr(gui, "_session_display_src", None)
        or getattr(gui, "_current_source", "")
        or ""
    )
    run_name = getattr(gui, "_active_run_name", None) or ""
    meta = getattr(gui, "_run_metadata", None)
    video_path = getattr(meta, "source", None) if meta is not None else None
    if video_path is None:
        video_path = getattr(gui, "_current_source", None)
    frames_dir = getattr(meta, "frames_dir", None) if meta is not None else None
    frame_paths: list[Path] = []
    if frames_dir:
        try:
            frame_paths = sorted(Path(frames_dir).glob("*.jpg")) + sorted(
                Path(frames_dir).glob("*.png")
            )
        except OSError:
            frame_paths = []

    summary = getattr(gui, "_analysis_summary_cache", None)
    ba = getattr(gui, "_biomech_analysis", None)
    contact = getattr(gui, "_foot_contact", None)
    cycles = getattr(gui, "_gait_cycle", None)
    vgrf = getattr(gui, "_estimated_vgrf", None)
    if summary is None:
        try:
            report = None
            assess = getattr(gui, "_assess_pipeline_status_report", None)
            if callable(assess):
                report = assess(force=True)
            summary = build_analysis_summary(
                source=str(source),
                biomechanical=ba,
                estimated_vgrf=vgrf,
                contact=contact,
                cycles=cycles,
                pipeline_status=report.to_dict() if report is not None else None,
            )
        except Exception:
            logger.debug("Could not build analysis summary for PDF", exc_info=True)
            summary = None

    selected: list[str] = []
    selection = getattr(gui, "selection", None)
    if selection is not None and getattr(selection, "selected", None):
        try:
            from stablewalk.ui.dof_selection import GUI_DOF_LABELS

            for item_id in selection.selected:
                selected.append(GUI_DOF_LABELS.get(item_id, str(item_id)))
        except Exception:
            selected = [str(x) for x in selection.selected]

    n_frames = len(sequence.frames) if sequence is not None else 0
    detected = (
        sum(1 for f in sequence.frames if getattr(f, "detected", False))
        if sequence is not None
        else 0
    )
    fps = float(getattr(sequence, "fps", 0.0) or getattr(meta, "fps", None) or 0.0) or None
    duration = None
    if n_frames > 1 and fps:
        duration = (n_frames - 1) / fps

    skeleton_xyz = None
    connections: list[tuple[int, int]] = []
    sk_cache = getattr(gui, "_skeleton_cache", None) or {}
    if sk_cache:
        try:
            from stablewalk.pose.skeleton_3d import SKELETON_3D_CONNECTIONS

            pose_indices = list(getattr(gui, "pose_indices", None) or [])
            mid = pose_indices[len(pose_indices) // 2] if pose_indices else next(iter(sk_cache))
            sk = sk_cache.get(mid) or next(iter(sk_cache.values()))
            joint_ids = list(sk.joints.keys())
            pts = []
            for jid in joint_ids:
                j = sk.joints[jid]
                pts.append((float(j.x), float(j.y), float(j.z)))
            skeleton_xyz = np.asarray(pts, dtype=float)
            id_to_i = {jid: i for i, jid in enumerate(joint_ids)}
            for a, b in SKELETON_3D_CONNECTIONS:
                if a in id_to_i and b in id_to_i:
                    connections.append((id_to_i[a], id_to_i[b]))
        except Exception:
            skeleton_xyz = None

    patient_name = str(getattr(gui, "_report_patient_name", "") or "")
    patient_id = str(getattr(gui, "_report_patient_id", "") or run_name or "")
    notes = str(getattr(gui, "_report_session_notes", "") or "")

    try:
        from stablewalk import __version__ as _sw_ver
    except Exception:
        _sw_ver = ""
    pose_backend = ""
    if meta is not None:
        pose_backend = str(getattr(meta, "pose_backend_used", "") or "")
    if not pose_backend:
        try:
            pose_backend = str(gui._selected_pose_backend())
        except Exception:
            pose_backend = "mediapipe"

    pipe = None
    if summary is not None:
        pipe = summary.pipeline_status
    if pipe is None:
        try:
            report = gui._assess_pipeline_status_report(force=True)
            pipe = report.to_dict()
        except Exception:
            pipe = None

    ctx = SessionPdfReportContext(
        source=str(source),
        run_name=str(run_name),
        session_label=str(source or run_name or "Session"),
        patient_id=patient_id,
        patient_name=patient_name or "—",
        notes=notes,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        fps=fps,
        n_frames=n_frames,
        detected_frames=detected,
        duration_s=duration,
        selected_joints=selected,
        summary=summary,
        biomechanical=ba,
        foot_contact=contact,
        cycles=cycles,
        estimated_vgrf=vgrf,
        sequence=sequence,
        video_path=str(video_path) if video_path else None,
        frame_paths=frame_paths,
        pipeline_status=pipe,
        skeleton_xyz=skeleton_xyz,
        skeleton_connections=connections,
        app_version=str(_sw_ver),
        pose_backend=pose_backend,
    )

    # Clinical-style interpretation (non-diagnostic) from AnalysisSummary.
    if summary is not None and getattr(summary, "scientific_interpretation", ""):
        ctx.clinical_interpretation = str(summary.scientific_interpretation).strip()

    # Comparison summary from Compare cache when Session A and B are ready.
    try:
        from stablewalk.analysis.session_compare import compare_session_metrics
        from stablewalk.ui.tk.session_cache import (
            ensure_compare_cache,
            sessions_are_identical,
        )

        cache = ensure_compare_cache(gui)
        left = cache.left()
        right = cache.right()
        if left is not None and right is not None:
            if sessions_are_identical(left, right):
                ctx.comparison_summary = (
                    "Compare Mode: Session A and Session B are identical — "
                    "no metric deltas."
                )
            else:
                result = compare_session_metrics(left.metrics, right.metrics)
                parts = [
                    result.lab_report_summary or result.interpretation or "",
                    result.research_summary or "",
                ]
                ctx.comparison_summary = "\n\n".join(p.strip() for p in parts if p.strip())
    except Exception:
        logger.debug("Could not attach comparison summary for PDF", exc_info=True)

    # Processing information (display / provenance only).
    processing: list[str] = [
        f"StableWalk version: {ctx.app_version or '—'}",
        f"Pose backend: {ctx.pose_backend or '—'}",
        f"Report schema: {REPORT_SCHEMA_VERSION}",
        f"Generated (UTC): {ctx.generated_at}",
        f"Source: {ctx.source or '—'}",
        f"Frames detected/total: {ctx.detected_frames}/{ctx.n_frames}",
        f"FPS: {_fmt(ctx.fps, digits=1)}",
        f"Duration: {_fmt(ctx.duration_s, digits=2, suffix=' s')}",
    ]
    if meta is not None:
        for attr, label in (
            ("pose_backend_requested", "Backend requested"),
            ("pose_backend_used", "Backend used"),
            ("pose_backend_fallback", "Backend fallback"),
        ):
            val = getattr(meta, attr, None)
            if val is not None and str(val) != "":
                processing.append(f"{label}: {val}")
    if cycles is not None:
        m = cycles.metrics
        processing.append(
            f"Usable gait cycles: {getattr(m, 'gait_cycle_count', None) or '—'}"
        )
        processing.append(
            f"Contact confidence: {_fmt(getattr(m, 'contact_confidence', None), digits=2)}"
        )
    processing.append(UNITS_COORD_LEGEND)
    processing.append(DISCLAIMER_RESEARCH)
    ctx.processing_lines = processing

    # Optional live-figure snapshots from the dashboard (display only).
    screenshot_paths: list[Path] = []
    try:
        tmp_shots = Path(tempfile.mkdtemp(prefix="stablewalk_pdf_shots_"))
        for attr, name in (
            ("fig_biomech", "biomech_timeline"),
            ("fig", "knee_flexion"),
            ("fig_contact_gait", "contact_vgrf"),
        ):
            fig = getattr(gui, attr, None)
            if fig is None:
                continue
            out_p = tmp_shots / f"{name}.png"
            try:
                fig.savefig(
                    out_p,
                    dpi=140,
                    bbox_inches="tight",
                    facecolor=fig.get_facecolor(),
                )
                if out_p.is_file():
                    screenshot_paths.append(out_p)
            except Exception:
                continue
    except Exception:
        screenshot_paths = []
    ctx.screenshot_paths = screenshot_paths

    ctx.conclusion = build_automatic_conclusion(ctx)
    ctx.recommendations = build_recommendations(ctx)
    return ctx


def export_session_pdf_report(
    ctx: SessionPdfReportContext,
    output_path: str | Path,
    *,
    logo_path: Path | None = None,
) -> Path:
    """Write a professional multi-page PDF report using ReportLab."""
    rl = _require_reportlab()
    colors = rl["colors"]
    Paragraph = rl["Paragraph"]
    Spacer = rl["Spacer"]
    Table = rl["Table"]
    TableStyle = rl["TableStyle"]
    PageBreak = rl["PageBreak"]
    RLImage = rl["RLImage"]
    SimpleDocTemplate = rl["SimpleDocTemplate"]
    ParagraphStyle = rl["ParagraphStyle"]
    getSampleStyleSheet = rl["getSampleStyleSheet"]
    A4 = rl["A4"]
    mm = rl["mm"]
    TA_CENTER = rl["TA_CENTER"]
    TA_JUSTIFY = rl["TA_JUSTIFY"]
    TA_LEFT = rl["TA_LEFT"]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    config.ensure_output_dirs()

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SWTitle",
            parent=styles["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=6,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SWHeading",
            parent=styles["Heading2"],
            fontSize=12,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SWBody",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#1f2937"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SWSmall",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#64748b"),
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SWMeta",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#334155"),
        )
    )

    tmp = Path(tempfile.mkdtemp(prefix="stablewalk_pdf_"))
    assets: list[Path] = []

    logo = generate_project_logo(logo_path or (config.REPORTS_DIR / "_stablewalk_logo.png"))
    assets.append(logo)

    thumb = _load_video_thumbnail(
        frame_paths=ctx.frame_paths, video_path=ctx.video_path, progress=0.35
    )
    thumb_path = None
    if thumb is not None:
        thumb_path = _save_pil(thumb, tmp / "thumbnail.png", max_side=720)
        assets.append(thumb_path)

    skel_path = _render_skeleton_image(ctx, tmp / "skeleton.png")
    knee_path = _render_knee_graph(ctx, tmp / "knee.png")
    com_fig = _render_com_graph(ctx, tmp / "com.png")
    traj_path = _render_trajectory_3d(ctx, tmp / "traj3d.png")
    contact_path = _render_contact_graph(ctx, tmp / "contact.png")
    hs_path = _render_heel_strike_graph(ctx, tmp / "heel_strikes.png")
    margin_path = _render_stability_margin_graph(ctx, tmp / "stability_margin.png")
    vgrf_path = _render_vgrf_graph(ctx, tmp / "vgrf.png")
    for p in (
        skel_path,
        knee_path,
        com_fig,
        traj_path,
        contact_path,
        hs_path,
        margin_path,
        vgrf_path,
    ):
        if p is not None:
            assets.append(p)
    for p in ctx.screenshot_paths:
        if p is not None and Path(p).is_file():
            assets.append(Path(p))

    def _table(data: list[list[Any]], *, col_widths: list[float] | None = None) -> Any:
        t = Table(data, colWidths=col_widths, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.HexColor("#f8fafc"), colors.white],
                    ),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return t

    def _img(path: Path | None, *, width: float, height: float | None = None) -> Any:
        if path is None or not path.is_file():
            return Paragraph("<i>Figure unavailable</i>", styles["SWSmall"])
        if height is None:
            return RLImage(str(path), width=width, height=width * 0.62, kind="proportional")
        return RLImage(str(path), width=width, height=height, kind="proportional")

    story: list[Any] = []

    # ── Cover ────────────────────────────────────────────────────────────
    story.append(RLImage(str(logo), width=150 * mm, height=36 * mm, kind="proportional"))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(REPORT_TITLE, styles["SWTitle"]))
    story.append(
        Paragraph(
            "Pose-estimated gait biomechanics — research &amp; laboratory review",
            styles["SWSmall"],
        )
    )
    story.append(Paragraph(DISCLAIMER, styles["SWSmall"]))
    story.append(Spacer(1, 4 * mm))

    # ── 1. Patient / session information ────────────────────────────────
    story.append(Paragraph("1. Patient / Session Information", styles["SWHeading"]))
    meta_rows = [
        ["Field", "Value"],
        ["Generated", ctx.generated_at],
        ["Patient / subject", ctx.patient_name or "—"],
        ["Patient / session ID", ctx.patient_id or "—"],
        ["Session label", ctx.session_label or "—"],
        ["Source video", ctx.source or "—"],
        ["Run name", ctx.run_name or "—"],
        ["FPS", _fmt(ctx.fps, digits=1)],
        ["Frames (detected / total)", f"{ctx.detected_frames} / {ctx.n_frames}"],
        ["Duration", _fmt(ctx.duration_s, digits=2, suffix=" s")],
        ["Selected joints", ", ".join(ctx.selected_joints) if ctx.selected_joints else "—"],
        ["Notes", ctx.notes or "—"],
        ["StableWalk version", ctx.app_version or "—"],
        ["Pose backend", ctx.pose_backend or "—"],
        ["Report schema", REPORT_SCHEMA_VERSION],
    ]
    story.append(_table(meta_rows, col_widths=[55 * mm, 120 * mm]))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Methods", styles["SWHeading"]))
    story.append(Paragraph(METHODS_OVERVIEW, styles["SWBody"]))
    story.append(Spacer(1, 1 * mm))
    story.append(Paragraph(UNITS_COORD_LEGEND, styles["SWMeta"]))
    story.append(Spacer(1, 3 * mm))

    # ── 2. Screenshot ───────────────────────────────────────────────────
    story.append(Paragraph("2. Session Screenshot", styles["SWHeading"]))
    media_cells = [
        [
            Paragraph("<b>Video frame</b>", styles["SWMeta"]),
            Paragraph("<b>Skeleton snapshot</b>", styles["SWMeta"]),
        ],
        [
            _img(thumb_path, width=80 * mm, height=55 * mm),
            _img(skel_path, width=80 * mm, height=55 * mm),
        ],
    ]
    media = Table(media_cells, colWidths=[90 * mm, 90 * mm])
    media.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(media)
    if ctx.screenshot_paths:
        story.append(Spacer(1, 2 * mm))
        story.append(
            Paragraph(
                "<i>Dashboard graph snapshots captured at export time:</i>",
                styles["SWMeta"],
            )
        )
        for shot in ctx.screenshot_paths[:3]:
            story.append(Spacer(1, 2 * mm))
            story.append(_img(Path(shot), width=170 * mm))
    story.append(PageBreak())

    # ── 3. Pipeline summary ─────────────────────────────────────────────
    story.append(Paragraph("3. Pipeline Summary", styles["SWHeading"]))
    pipe_rows = [["Stage", "Status", "Detail"]]
    pipe = ctx.pipeline_status or {}
    diagram = pipe.get("diagram") if isinstance(pipe, dict) else None
    if isinstance(diagram, list) and diagram:
        for stage in diagram:
            if not isinstance(stage, dict):
                continue
            pipe_rows.append(
                [
                    str(stage.get("label") or stage.get("key") or "—"),
                    str(stage.get("status") or "—").title(),
                    str(stage.get("detail") or "")[:90],
                ]
            )
    else:
        groups = pipe.get("groups") if isinstance(pipe, dict) else None
        if isinstance(groups, list):
            for group in groups:
                for item in group.get("stages", []) if isinstance(group, dict) else []:
                    pipe_rows.append(
                        [
                            str(item.get("label") or "—"),
                            str(item.get("status") or "—").title(),
                            str(item.get("detail") or "")[:90],
                        ]
                    )
        else:
            pipe_rows.append(["Pipeline", "Unavailable", "No pipeline status for this session"])
    story.append(_table(pipe_rows, col_widths=[45 * mm, 30 * mm, 100 * mm]))
    story.append(PageBreak())

    # ── 4–9. Core metrics ───────────────────────────────────────────────
    ba = ctx.biomechanical
    story.append(Paragraph("4. Gait Quality", styles["SWHeading"]))
    gq_rows = [
        ["Metric", "Value"],
        ["Overall gait quality", _field_value(ctx.summary, "overall_gait_quality")],
    ]
    if ba is not None and ba.gait_quality is not None:
        gq = ba.gait_quality
        gq_rows.append(["Composite score", _fmt(gq.score, digits=1, suffix=" / 100")])
        gq_rows.append(["Composite confidence", _fmt(gq.confidence, digits=2)])
        if gq.explanation:
            gq_rows.append(["Explanation", str(gq.explanation)[:140]])
        if gq.dominant_factors:
            gq_rows.append(["Dominant factors", ", ".join(gq.dominant_factors[:4])])
    story.append(_table(gq_rows, col_widths=[55 * mm, 120 * mm]))

    story.append(Paragraph("5. Stability", styles["SWHeading"]))
    stab_rows = [
        ["Metric", "Value"],
        ["Stability (summary)", _field_value(ctx.summary, "stability_margin")],
    ]
    if ba is not None and ba.stability_margin is not None:
        sm = ba.stability_margin
        if getattr(sm, "reportable", True) is False:
            note = str(
                getattr(sm, "experimental_note", "")
                or "Monocular COM–BoS is experimental and is not a physical stability score."
            )
            stab_rows.extend(
                [
                    ["Reportable", "No"],
                    ["Note", note[:140]],
                ]
            )
        else:
            stab_rows.extend(
                [
                    ["Stable-frame %", _fmt(sm.stable_pct, digits=1, suffix=" %")],
                    ["Mean stability margin", _fmt(sm.mean_margin_m, digits=3, suffix=" m")],
                    ["Valid frame ratio", _fmt(sm.valid_frame_ratio, digits=2)],
                    ["Margin confidence", _fmt(sm.confidence, digits=2)],
                ]
            )
    story.append(_table(stab_rows, col_widths=[55 * mm, 120 * mm]))
    if margin_path is not None:
        story.append(Spacer(1, 2 * mm))
        story.append(_img(margin_path, width=170 * mm))

    story.append(Paragraph("6. Cadence", styles["SWHeading"]))
    cad_rows = [
        ["Metric", "Value"],
        ["Cadence", _field_value(ctx.summary, "cadence")],
        ["Walking speed", _field_value(ctx.summary, "walking_speed")],
    ]
    if ctx.cycles is not None:
        m = ctx.cycles.metrics
        cad_rows.extend(
            [
                ["Gait cycles (usable)", _fmt(getattr(m, "gait_cycle_count", None), digits=0)],
                ["Stride time", _fmt(getattr(m, "stride_time_s", None), digits=3, suffix=" s")],
                [
                    "Step time",
                    _fmt(getattr(m, "step_time_s", None), digits=3, suffix=" s"),
                ],
                [
                    "Stance duration",
                    _fmt(getattr(m, "average_stance_duration_s", None), digits=3, suffix=" s"),
                ],
            ]
        )
    if ba is not None and ba.gait_metrics is not None:
        gm = ba.gait_metrics
        if gm.step_length and gm.step_length.value is not None:
            cad_rows.append(["Step length", _fmt(gm.step_length.value, digits=3, suffix=" m")])
        if gm.stride_length and gm.stride_length.value is not None:
            cad_rows.append(["Stride length", _fmt(gm.stride_length.value, digits=3, suffix=" m")])
    story.append(_table(cad_rows, col_widths=[55 * mm, 120 * mm]))
    story.append(PageBreak())

    story.append(Paragraph("7. Range of Motion (ROM)", styles["SWHeading"]))
    rom_rows = [["Joint", "Side", "Min (°)", "Max (°)", "ROM (°)", "Confidence"]]
    if ba is not None and ba.joint_rom is not None and ba.joint_rom.joints:
        for j in ba.joint_rom.joints:
            rom_rows.append(
                [
                    str(j.joint).title(),
                    str(j.side).title(),
                    _fmt(j.flexion_min_deg, digits=1),
                    _fmt(j.flexion_max_deg, digits=1),
                    _fmt(j.rom_deg, digits=1),
                    _fmt(j.confidence, digits=2),
                ]
            )
    else:
        rom_rows.append(["—", "—", "—", "—", "—", "Unavailable"])
        story.append(
            Paragraph(_field_value(ctx.summary, "joint_rom_summary"), styles["SWBody"])
        )
    story.append(
        _table(rom_rows, col_widths=[28 * mm, 25 * mm, 28 * mm, 28 * mm, 28 * mm, 33 * mm])
    )
    if knee_path is not None:
        story.append(Spacer(1, 2 * mm))
        story.append(_img(knee_path, width=170 * mm))

    story.append(Paragraph("8. Center of Mass (COM)", styles["SWHeading"]))
    story.append(Paragraph(_field_value(ctx.summary, "center_of_mass"), styles["SWBody"]))
    if ba is not None and ba.center_of_mass is not None:
        pos = ba.center_of_mass.positions
        if pos is not None:
            arr = np.asarray(pos, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                y = arr[:, 1]
                story.append(
                    Paragraph(
                        f"Vertical COM range (stature-scaled estimate): "
                        f"{_fmt(float(np.nanmax(y) - np.nanmin(y)) * config.DEFAULT_SUBJECT_HEIGHT_M * 100, digits=1, suffix=' cm')}. "
                        f"Samples: {arr.shape[0]}.",
                        styles["SWBody"],
                    )
                )
    if com_fig is not None:
        story.append(Spacer(1, 2 * mm))
        story.append(_img(com_fig, width=170 * mm))
    if traj_path is not None:
        story.append(Spacer(1, 2 * mm))
        story.append(_img(traj_path, width=130 * mm, height=100 * mm))

    story.append(Paragraph("9. Symmetry", styles["SWHeading"]))
    sym_rows = [
        ["Metric", "Value"],
        ["Overall symmetry", _field_value(ctx.summary, "symmetry")],
    ]
    if ba is not None and ba.symmetry is not None:
        sym = ba.symmetry
        overall = getattr(sym, "overall_symmetry_pct", None)
        if overall is not None and getattr(overall, "value", None) is not None:
            sym_rows.append(
                ["Overall symmetry %", _fmt(overall.value, digits=1, suffix=" %")]
            )
        for attr, label in (
            ("stance_symmetry", "Stance symmetry"),
            ("swing_symmetry", "Swing symmetry"),
            ("step_length_symmetry", "Step-length symmetry"),
            ("stride_length_symmetry", "Stride-length symmetry"),
            ("cadence_consistency", "Cadence consistency"),
        ):
            fld = getattr(sym, attr, None)
            if fld is not None and getattr(fld, "value", None) is not None:
                unit = getattr(fld, "unit", "") or ""
                sym_rows.append([label, _fmt(fld.value, digits=3, suffix=f" {unit}".rstrip())])
    story.append(_table(sym_rows, col_widths=[55 * mm, 120 * mm]))
    story.append(PageBreak())

    # ── 10. Heel strikes ────────────────────────────────────────────────
    story.append(Paragraph("10. Heel Strikes", styles["SWHeading"]))
    hs_rows = [["Event", "Status / detail"]]
    if ctx.summary is not None and ctx.summary.gait_events:
        for ev in ctx.summary.gait_events:
            mark = "Detected" if ev.detected else "Not detected"
            hs_rows.append([ev.name, f"{mark} — {ev.detail}"])
    elif ctx.cycles is not None and getattr(ctx.cycles, "events", None):
        counts: dict[str, int] = {}
        for e in ctx.cycles.events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
        for name, n in sorted(counts.items()):
            hs_rows.append([name.replace("_", " ").title(), f"Count: {n}"])
    else:
        hs_rows.append(["Heel strike", "Unavailable for this session"])
    story.append(_table(hs_rows, col_widths=[55 * mm, 120 * mm]))
    if hs_path is not None:
        story.append(Spacer(1, 2 * mm))
        story.append(_img(hs_path, width=170 * mm))

    # ── 11. Contact timing ──────────────────────────────────────────────
    story.append(Paragraph("11. Contact Timing", styles["SWHeading"]))
    contact = ctx.foot_contact
    if contact is not None:
        cm = contact.metrics
        n_frames_contact = len(getattr(contact, "per_frame", []) or [])
        contact_rows = [
            ["Metric", "Value"],
            ["Double support %", _fmt(getattr(cm, "double_support_pct", None), digits=1, suffix=" %")],
            [
                "Double support duration",
                _fmt(getattr(cm, "double_support_duration_s", None), digits=3, suffix=" s"),
            ],
            [
                "Average stance duration",
                _fmt(getattr(cm, "average_stance_duration_s", None), digits=3, suffix=" s"),
            ],
            [
                "Contact confidence",
                _fmt(getattr(cm, "contact_confidence", None), digits=2),
            ],
            [
                "Valid gait cycles (contact)",
                _fmt(getattr(cm, "valid_gait_cycle_count", None), digits=0),
            ],
            ["Frames analyzed", _fmt(n_frames_contact, digits=0)],
            ["Estimated virtual GRF", _field_value(ctx.summary, "estimated_virtual_grf")],
        ]
        if ctx.estimated_vgrf is not None and ctx.estimated_vgrf.available:
            contact_rows.append(
                [
                    "Peak virtual GRF",
                    _fmt(ctx.estimated_vgrf.metrics.peak_force_bw, digits=2, suffix=" BW"),
                ]
            )
        if ctx.cycles is not None:
            m = ctx.cycles.metrics
            contact_rows.extend(
                [
                    [
                        "Double support % (cycles)",
                        _fmt(getattr(m, "double_support_pct", None), digits=1, suffix=" %"),
                    ],
                    [
                        "Single support % (ipsilateral)",
                        _fmt(getattr(m, "single_support_pct", None), digits=1, suffix=" %"),
                    ],
                ]
            )
        story.append(_table(contact_rows, col_widths=[70 * mm, 105 * mm]))
        if contact_path is not None:
            story.append(Spacer(1, 2 * mm))
            story.append(_img(contact_path, width=170 * mm))
        if vgrf_path is not None:
            story.append(Spacer(1, 2 * mm))
            story.append(_img(vgrf_path, width=170 * mm))
    else:
        story.append(
            Paragraph(
                "Foot contact analysis was not available for this session.",
                styles["SWBody"],
            )
        )
    story.append(PageBreak())

    # ── 12. Graph snapshots ─────────────────────────────────────────────
    story.append(Paragraph("12. Graph Snapshots", styles["SWHeading"]))
    story.append(
        Paragraph(
            "Publication figures generated from the analyzed session "
            "(pose-estimated series; monocular scale approximate).",
            styles["SWBody"],
        )
    )
    fig_blocks = [
        ("Knee flexion", knee_path),
        ("COM trajectory", com_fig),
        ("3D COM path", traj_path),
        ("Stability margin", margin_path),
        ("Contact timing", contact_path),
        ("Heel strikes", hs_path),
        ("Estimated vGRF", vgrf_path),
    ]
    any_fig = False
    for title, path in fig_blocks:
        if path is None:
            continue
        any_fig = True
        story.append(Paragraph(f"<b>{title}</b>", styles["SWMeta"]))
        story.append(_img(path, width=160 * mm))
        story.append(Spacer(1, 2 * mm))
    if not any_fig:
        story.append(Paragraph("No graph snapshots were available.", styles["SWBody"]))
    story.append(PageBreak())

    # ── 13. Comparison summary ──────────────────────────────────────────
    story.append(Paragraph("13. Comparison Summary", styles["SWHeading"]))
    if ctx.comparison_summary.strip():
        story.append(
            Paragraph(
                _escape(ctx.comparison_summary).replace("\n", "<br/>"),
                styles["SWBody"],
            )
        )
        story.append(
            Paragraph(
                "Δ convention: Session B − Session A (pose-estimated). "
                "Color/tone in the GUI is directional only — not statistical significance.",
                styles["SWMeta"],
            )
        )
    else:
        story.append(
            Paragraph(
                "No dual-session comparison was available at export time. "
                "Open the Compare tab, assign Session A and Session B, then re-export "
                "to include a laboratory comparison summary.",
                styles["SWBody"],
            )
        )

    # ── 14. Clinical interpretation ─────────────────────────────────────
    story.append(Paragraph("14. Clinical-Style Interpretation", styles["SWHeading"]))
    story.append(
        Paragraph(
            "<i>Auto-generated research language — not a medical diagnosis or "
            "clinical certification.</i>",
            styles["SWMeta"],
        )
    )
    interp = (
        ctx.clinical_interpretation.strip()
        or (ctx.summary.scientific_interpretation.strip() if ctx.summary and ctx.summary.scientific_interpretation else "")
        or ctx.conclusion
        or "Interpretation unavailable for this session."
    )
    story.append(Paragraph(_escape(interp).replace("\n", "<br/>"), styles["SWBody"]))
    story.append(Paragraph("Laboratory conclusion", styles["SWHeading"]))
    story.append(
        Paragraph(_escape(ctx.conclusion or "—").replace("\n", "<br/>"), styles["SWBody"])
    )
    story.append(Paragraph("Recommendations", styles["SWHeading"]))
    story.append(
        Paragraph(
            _escape(ctx.recommendations or "—").replace("\n", "<br/>"),
            styles["SWBody"],
        )
    )

    # ── 15. Confidence ──────────────────────────────────────────────────
    story.append(Paragraph("15. Confidence", styles["SWHeading"]))
    conf_rows = [
        ["Confidence metric", "Value"],
        ["Video quality", _field_value(ctx.summary, "video_quality")],
        ["Tracking confidence", _field_value(ctx.summary, "tracking_confidence")],
        ["Pipeline confidence", _field_value(ctx.summary, "pipeline_confidence")],
        ["Analysis confidence", _field_value(ctx.summary, "analysis_confidence")],
    ]
    if ctx.cycles is not None:
        conf_rows.append(
            [
                "Contact confidence (cycles)",
                _fmt(getattr(ctx.cycles.metrics, "contact_confidence", None), digits=2),
            ]
        )
    if ba is not None and ba.stability_margin is not None:
        conf_rows.append(
            ["Stability margin confidence", _fmt(ba.stability_margin.confidence, digits=2)]
        )
    if ba is not None and ba.gait_quality is not None:
        conf_rows.append(
            ["Gait quality confidence", _fmt(ba.gait_quality.confidence, digits=2)]
        )
    story.append(_table(conf_rows, col_widths=[70 * mm, 105 * mm]))
    story.append(PageBreak())

    # ── 16. Processing information ──────────────────────────────────────
    story.append(Paragraph("16. Processing Information", styles["SWHeading"]))
    proc_lines = ctx.processing_lines or [
        f"StableWalk version: {ctx.app_version or '—'}",
        f"Pose backend: {ctx.pose_backend or '—'}",
        f"Report schema: {REPORT_SCHEMA_VERSION}",
        f"Generated (UTC): {ctx.generated_at}",
    ]
    story.append(
        Paragraph(
            "<br/>".join(f"• {_escape(line)}" for line in proc_lines),
            styles["SWBody"],
        )
    )

    story.append(Paragraph("Metric Validation Warnings", styles["SWHeading"]))
    validation_warnings = build_validation_warnings(ctx)
    if validation_warnings:
        story.append(
            Paragraph(
                "<br/>".join(f"• {_escape(warning)}" for warning in validation_warnings),
                styles["SWBody"],
            )
        )
    else:
        story.append(
            Paragraph(
                "No mathematical validity warnings were generated for reported metrics.",
                styles["SWBody"],
            )
        )

    story.append(Spacer(1, 8 * mm))
    story.append(
        Paragraph(
            f"Timestamp: {ctx.generated_at} · StableWalk {ctx.app_version or ''} "
            f"· Research Report v{REPORT_SCHEMA_VERSION}"
            f"{(' · backend ' + ctx.pose_backend) if ctx.pose_backend else ''}",
            styles["SWSmall"],
        )
    )
    story.append(Paragraph(DISCLAIMER, styles["SWSmall"]))

    def _footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawCentredString(
            A4[0] / 2.0,
            10 * mm,
            f"StableWalk Research Report  ·  page {doc.page}  ·  {DISCLAIMER[:70]}…",
        )
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title=REPORT_TITLE,
        author="StableWalk",
        subject="Pose-estimated gait biomechanics research report",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)

    for p in assets:
        if p == logo:
            continue
        try:
            if p.is_file() and (tmp in p.parents or "stablewalk_pdf_shots_" in str(p)):
                p.unlink(missing_ok=True)
                parent = p.parent
                if parent != tmp and parent.name.startswith("stablewalk_pdf_shots_"):
                    try:
                        parent.rmdir()
                    except OSError:
                        pass
        except OSError:
            pass
    try:
        tmp.rmdir()
    except OSError:
        pass

    logger.info("Research PDF report written: %s", out)
    return out


def export_session_pdf_report_from_gui(gui: Any, output_path: str | Path) -> Path:
    """Convenience: gather GUI context and write the PDF."""
    ctx = build_session_pdf_context_from_gui(gui)
    if ctx.sequence is None and ctx.biomechanical is None and ctx.summary is None:
        raise ValueError("No analyzed session data available for PDF export.")
    return export_session_pdf_report(ctx, output_path)


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "REPORT_TITLE",
    "SessionPdfReportContext",
    "build_automatic_conclusion",
    "build_recommendations",
    "build_validation_warnings",
    "build_session_pdf_context_from_gui",
    "export_session_pdf_report",
    "export_session_pdf_report_from_gui",
    "generate_project_logo",
]
