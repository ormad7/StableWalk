"""
Professional StableWalk session PDF report (ReportLab).

Generates a multi-page biomechanics laboratory report including session
metadata, pipeline status, gait/biomech metrics, joint ROM, graphs,
skeleton / trajectory figures, foot-contact and COM analysis, plus an
automatic conclusion and recommendations.
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

logger = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = "1.0"
DISCLAIMER = (
    "Monocular pose-estimated biomechanics — for research and education only. "
    "Not a clinical diagnosis or medical device output."
)


@dataclass
class SessionPdfReportContext:
    """All inputs required to render a professional session PDF."""

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
            if not cap.isOpened():
                return None
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_i = int(round(progress * max(total - 1, 0)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
            ok, bgr = cap.read()
            cap.release()
            if not ok or bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
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
    ax.set_title("Foot contact analysis")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig, path)


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
    )
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
    for p in (skel_path, knee_path, com_fig, traj_path, contact_path):
        if p is not None:
            assets.append(p)

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

    story.append(RLImage(str(logo), width=150 * mm, height=40 * mm, kind="proportional"))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Professional Gait Analysis Report", styles["SWTitle"]))
    story.append(Paragraph(DISCLAIMER, styles["SWSmall"]))
    story.append(Spacer(1, 4 * mm))

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
        ["Report schema", REPORT_SCHEMA_VERSION],
    ]
    story.append(Paragraph("Patient / Session Information", styles["SWHeading"]))
    story.append(_table(meta_rows, col_widths=[55 * mm, 120 * mm]))
    story.append(Spacer(1, 4 * mm))

    media_cells = [
        [
            Paragraph("<b>Video thumbnail</b>", styles["SWMeta"]),
            Paragraph("<b>Skeleton image</b>", styles["SWMeta"]),
        ],
        [
            _img(thumb_path, width=80 * mm, height=55 * mm),
            _img(skel_path, width=80 * mm, height=55 * mm),
        ],
    ]
    media = Table(media_cells, colWidths=[90 * mm, 90 * mm])
    media.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(media)
    story.append(PageBreak())

    story.append(Paragraph("Pipeline Summary", styles["SWHeading"]))
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
    story.append(Spacer(1, 3 * mm))
    conf_rows = [
        ["Confidence", "Value"],
        ["Video quality", _field_value(ctx.summary, "video_quality")],
        ["Tracking confidence", _field_value(ctx.summary, "tracking_confidence")],
        ["Pipeline confidence", _field_value(ctx.summary, "pipeline_confidence")],
        ["Analysis confidence", _field_value(ctx.summary, "analysis_confidence")],
    ]
    story.append(_table(conf_rows, col_widths=[60 * mm, 115 * mm]))
    story.append(PageBreak())

    story.append(Paragraph("Gait Metrics", styles["SWHeading"]))
    gait_rows = [
        ["Metric", "Value"],
        ["Overall gait quality", _field_value(ctx.summary, "overall_gait_quality")],
        ["Cadence", _field_value(ctx.summary, "cadence")],
        ["Walking speed", _field_value(ctx.summary, "walking_speed")],
        ["Symmetry", _field_value(ctx.summary, "symmetry")],
        ["Estimated virtual GRF", _field_value(ctx.summary, "estimated_virtual_grf")],
    ]
    if ctx.cycles is not None:
        m = ctx.cycles.metrics
        gait_rows.extend(
            [
                ["Gait cycles (usable)", _fmt(getattr(m, "gait_cycle_count", None), digits=0)],
                [
                    "Stride time",
                    _fmt(getattr(m, "stride_time_s", None), digits=3, suffix=" s"),
                ],
                [
                    "Stance duration",
                    _fmt(getattr(m, "average_stance_duration_s", None), digits=3, suffix=" s"),
                ],
                [
                    "Double support %",
                    _fmt(getattr(m, "double_support_pct", None), digits=1, suffix=" %"),
                ],
            ]
        )
    ba = ctx.biomechanical
    if ba is not None and ba.gait_metrics is not None:
        gm = ba.gait_metrics
        if gm.step_length and gm.step_length.value is not None:
            gait_rows.append(["Step length", _fmt(gm.step_length.value, digits=3, suffix=" m")])
        if gm.stride_length and gm.stride_length.value is not None:
            gait_rows.append(["Stride length", _fmt(gm.stride_length.value, digits=3, suffix=" m")])
    story.append(_table(gait_rows, col_widths=[70 * mm, 105 * mm]))

    story.append(Paragraph("Biomechanics Metrics", styles["SWHeading"]))
    bio_rows = [
        ["Metric", "Value"],
        ["Stability", _field_value(ctx.summary, "stability_margin")],
        ["Centre of mass", _field_value(ctx.summary, "center_of_mass")],
        ["Joint ROM summary", _field_value(ctx.summary, "joint_rom_summary")],
    ]
    if ba is not None and ba.stability_margin is not None:
        sm = ba.stability_margin
        bio_rows.extend(
            [
                ["Stability score (stable %)", _fmt(sm.stable_pct, digits=1, suffix=" %")],
                ["Mean stability margin", _fmt(sm.mean_margin_m, digits=3, suffix=" m")],
            ]
        )
    if ctx.estimated_vgrf is not None and ctx.estimated_vgrf.available:
        bio_rows.append(
            [
                "Peak virtual GRF",
                _fmt(ctx.estimated_vgrf.metrics.peak_force_bw, digits=2, suffix=" BW"),
            ]
        )
    story.append(_table(bio_rows, col_widths=[70 * mm, 105 * mm]))

    stab_score = "—"
    if ba is not None and ba.stability_margin is not None and ba.stability_margin.stable_pct is not None:
        stab_score = f"{ba.stability_margin.stable_pct:.0f} / 100 (stable-frame %)"
    story.append(Paragraph("Stability Score", styles["SWHeading"]))
    story.append(
        Paragraph(
            f"<b>{stab_score}</b> — pose-derived COM–base-of-support proxy "
            "(High / Moderate / Low category also reported in gait summary).",
            styles["SWBody"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Joint ROM Table", styles["SWHeading"]))
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
    story.append(_table(rom_rows, col_widths=[28 * mm, 25 * mm, 28 * mm, 28 * mm, 28 * mm, 33 * mm]))

    story.append(Paragraph("Selected Joint(s)", styles["SWHeading"]))
    story.append(
        Paragraph(
            ", ".join(ctx.selected_joints)
            if ctx.selected_joints
            else "No joint selection recorded — knee flexion shown as default kinematics.",
            styles["SWBody"],
        )
    )
    if knee_path is not None:
        story.append(Spacer(1, 2 * mm))
        story.append(_img(knee_path, width=170 * mm))
    story.append(PageBreak())

    story.append(Paragraph("Graphs &amp; 3D Trajectory", styles["SWHeading"]))
    if com_fig is not None:
        story.append(_img(com_fig, width=170 * mm))
        story.append(Spacer(1, 3 * mm))
    story.append(_img(traj_path, width=140 * mm, height=110 * mm))
    story.append(PageBreak())

    story.append(Paragraph("Foot Contact Analysis", styles["SWHeading"]))
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
                "Valid gait cycles (contact)",
                _fmt(getattr(cm, "valid_gait_cycle_count", None), digits=0),
            ],
            ["Frames analyzed", _fmt(n_frames_contact, digits=0)],
        ]
        story.append(_table(contact_rows, col_widths=[70 * mm, 105 * mm]))
        if contact_path is not None:
            story.append(Spacer(1, 3 * mm))
            story.append(_img(contact_path, width=170 * mm))
    else:
        story.append(
            Paragraph("Foot contact analysis was not available for this session.", styles["SWBody"])
        )
    story.append(PageBreak())

    story.append(Paragraph("COM Analysis", styles["SWHeading"]))
    story.append(Paragraph(_field_value(ctx.summary, "center_of_mass"), styles["SWBody"]))
    if ba is not None and ba.center_of_mass is not None:
        pos = ba.center_of_mass.positions
        if pos is not None:
            arr = np.asarray(pos, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                y = arr[:, 1]
                story.append(
                    Paragraph(
                        f"Vertical COM range: "
                        f"{_fmt(float(np.nanmax(y) - np.nanmin(y)) * 100, digits=1, suffix=' cm')}. "
                        f"Samples: {arr.shape[0]}.",
                        styles["SWBody"],
                    )
                )
    if com_fig is not None:
        story.append(Spacer(1, 2 * mm))
        story.append(_img(com_fig, width=170 * mm))

    story.append(Paragraph("Automatic Conclusion", styles["SWHeading"]))
    story.append(Paragraph((ctx.conclusion or "—").replace("\n", "<br/>"), styles["SWBody"]))
    story.append(Paragraph("Metric Validation Warnings", styles["SWHeading"]))
    validation_warnings = build_validation_warnings(ctx)
    if validation_warnings:
        story.append(
            Paragraph(
                "<br/>".join(f"• {warning}" for warning in validation_warnings),
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
    story.append(Paragraph("Recommendations", styles["SWHeading"]))
    story.append(
        Paragraph((ctx.recommendations or "—").replace("\n", "<br/>"), styles["SWBody"])
    )
    story.append(Spacer(1, 8 * mm))
    story.append(
        Paragraph(
            f"Timestamp: {ctx.generated_at} · StableWalk PDF Report v{REPORT_SCHEMA_VERSION}",
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
            f"StableWalk Biomechanics Report  ·  page {doc.page}  ·  {DISCLAIMER[:70]}…",
        )
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title="StableWalk Gait Analysis Report",
        author="StableWalk",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)

    for p in assets:
        if p == logo:
            continue
        try:
            if p.is_file() and tmp in p.parents:
                p.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        tmp.rmdir()
    except OSError:
        pass

    logger.info("Session PDF report written: %s", out)
    return out


def export_session_pdf_report_from_gui(gui: Any, output_path: str | Path) -> Path:
    """Convenience: gather GUI context and write the PDF."""
    ctx = build_session_pdf_context_from_gui(gui)
    if ctx.sequence is None and ctx.biomechanical is None and ctx.summary is None:
        raise ValueError("No analyzed session data available for PDF export.")
    return export_session_pdf_report(ctx, output_path)


__all__ = [
    "SessionPdfReportContext",
    "build_automatic_conclusion",
    "build_recommendations",
    "build_validation_warnings",
    "build_session_pdf_context_from_gui",
    "export_session_pdf_report",
    "export_session_pdf_report_from_gui",
    "generate_project_logo",
]
