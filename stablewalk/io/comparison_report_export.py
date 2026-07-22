"""
Export a professional multi-page Comparison Mode PDF report.

Designed for MSc thesis documentation: cover page, session media, metrics,
difference table, overlay kinematics, 3D trajectory, and narrative sections.

Implemented with matplotlib PdfPages (no additional PDF dependencies).
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from PIL import Image

from stablewalk.analysis.session_compare import (
    SessionComparisonResult,
    compare_session_metrics,
)
from stablewalk.ui.colors import SIDE_LEFT, SIDE_RIGHT
from stablewalk.ui.tk.session_cache import AnalyzedSessionSnapshot

REPORT_SCHEMA_VERSION = "1.0"
COLOR_A = SIDE_LEFT
COLOR_B = SIDE_RIGHT

# Print-friendly A4
_PAGE_W = 8.27
_PAGE_H = 11.69
_MARGIN = 0.65


def _new_page() -> Figure:
    fig = Figure(figsize=(_PAGE_W, _PAGE_H), dpi=150, facecolor="white")
    fig.subplots_adjust(
        left=_MARGIN / _PAGE_W,
        right=1 - _MARGIN / _PAGE_W,
        top=1 - 0.55 / _PAGE_H,
        bottom=_MARGIN / _PAGE_H,
    )
    return fig


def _footer(fig: Figure, page: int, total: int) -> None:
    fig.text(
        0.5,
        0.028,
        f"StableWalk Gait Comparison Report  ·  p. {page}/{total}  ·  "
        "Monocular pose estimation — not a clinical diagnosis",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )


def _title_block(fig: Figure, title: str, subtitle: str = "") -> None:
    fig.text(0.5, 0.96, title, ha="center", va="top", fontsize=16, fontweight="bold", color="#111827")
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=10, color="#374151")


def _wrapped_text(
    ax,
    text: str,
    *,
    x: float = 0.0,
    y: float = 1.0,
    width: int = 96,
    fontsize: float = 9.5,
    color: str = "#1f2937",
    fontweight: str = "normal",
    va: str = "top",
) -> float:
    ax.set_axis_off()
    lines = []
    for para in (text or "").split("\n"):
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=width) or [""])
    block = "\n".join(lines)
    ax.text(
        x,
        y,
        block,
        transform=ax.transAxes,
        ha="left",
        va=va,
        fontsize=fontsize,
        color=color,
        fontweight=fontweight,
        family="DejaVu Sans",
        linespacing=1.35,
    )
    # Approximate remaining height used
    return max(0.0, y - 0.018 * max(len(lines), 1))


def _load_thumbnail(snap: AnalyzedSessionSnapshot | None, progress: float = 0.35) -> Image.Image | None:
    if snap is None:
        return None
    if snap.frame_paths:
        idx = int(round(progress * (len(snap.frame_paths) - 1)))
        idx = max(0, min(idx, len(snap.frame_paths) - 1))
        try:
            return Image.open(snap.frame_paths[idx]).convert("RGB")
        except OSError:
            pass
    if snap.video_path and Path(str(snap.video_path)).is_file():
        try:
            import cv2

            cap = cv2.VideoCapture(str(snap.video_path))
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


def _fmt(v: float | None, fmt: str) -> str:
    if v is None:
        return "—"
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return "—"


def build_comparison_report_result(
    left: AnalyzedSessionSnapshot,
    right: AnalyzedSessionSnapshot,
) -> SessionComparisonResult:
    return compare_session_metrics(left.metrics, right.metrics)


def export_comparison_report_pdf(
    left: AnalyzedSessionSnapshot,
    right: AnalyzedSessionSnapshot,
    path: Path | str,
    *,
    comparison: SessionComparisonResult | None = None,
    generated_at: datetime | None = None,
    progress: float = 0.35,
) -> Path:
    """
    Write a multi-page PDF comparison report.

    Returns the output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison = comparison or build_comparison_report_result(left, right)
    stamp = generated_at or datetime.now(timezone.utc)
    date_str = stamp.strftime("%Y-%m-%d %H:%M UTC")

    thumb_a = _load_thumbnail(left, progress)
    thumb_b = _load_thumbnail(right, progress)

    pages: list[Figure] = []
    pages.append(_page_cover(left, right, date_str))
    pages.append(_page_thumbnails(left, right, thumb_a, thumb_b))
    pages.append(_page_metrics(left, right, comparison))
    pages.append(_page_graphs(left, right))
    pages.append(_page_trajectory(left, right))
    pages.append(_page_narratives(comparison))

    total = len(pages)
    with PdfPages(path) as pdf:
        for i, fig in enumerate(pages, start=1):
            _footer(fig, i, total)
            pdf.savefig(fig, facecolor="white")
            import matplotlib.pyplot as plt

            plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "StableWalk Gait Comparison Report"
        d["Author"] = "StableWalk"
        d["Subject"] = f"{left.label} vs {right.label}"
        d["Keywords"] = "gait analysis, comparison, StableWalk, MSc"
        d["CreationDate"] = stamp

    return path


def _page_cover(
    left: AnalyzedSessionSnapshot,
    right: AnalyzedSessionSnapshot,
    date_str: str,
) -> Figure:
    fig = _new_page()
    _title_block(fig, "StableWalk", "Gait Comparison Report")
    ax = fig.add_axes([0.1, 0.12, 0.8, 0.72])
    ax.set_axis_off()

    body = (
        f"Title\n"
        f"    Dual-Session Gait Comparison for Thesis Documentation\n\n"
        f"Selected sessions\n"
        f"    Left (reference):  {left.label}  [{left.key}]\n"
        f"    Right (comparator): {right.label}  [{right.key}]\n\n"
        f"Date\n"
        f"    {date_str}\n\n"
        f"Sources\n"
        f"    Left:  {left.source or left.video_path or '—'}\n"
        f"    Right: {right.source or right.video_path or '—'}\n\n"
        f"Frame counts\n"
        f"    Left: {left.n_frames} frames @ {left.fps:.1f} fps\n"
        f"    Right: {right.n_frames} frames @ {right.fps:.1f} fps\n\n"
        f"Methodology note\n"
        f"    Metrics are computed from monocular video pose estimation\n"
        f"    (MediaPipe / StableWalk pipeline). Walking speed, virtual GRF,\n"
        f"    COM, and stability margins are estimated or derived — not\n"
        f"    force-plate or marker-based laboratory measurements.\n\n"
        f"Schema version: {REPORT_SCHEMA_VERSION}\n\n"
        f"Disclaimer\n"
        f"    This document does not constitute medical advice or a clinical\n"
        f"    diagnosis. It is intended for academic documentation, laboratory\n"
        f"    notes, and relative session comparison within a research study."
    )
    ax.text(
        0.0,
        1.0,
        body,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="DejaVu Sans",
        color="#111827",
        linespacing=1.4,
    )
    return fig


def _page_thumbnails(
    left: AnalyzedSessionSnapshot,
    right: AnalyzedSessionSnapshot,
    thumb_a: Image.Image | None,
    thumb_b: Image.Image | None,
) -> Figure:
    fig = _new_page()
    _title_block(fig, "Video Thumbnails", "Representative mid-clip frames from each session")

    axes = fig.subplots(1, 2)
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.18)
    for ax, snap, img, color in (
        (axes[0], left, thumb_a, COLOR_A),
        (axes[1], right, thumb_b, COLOR_B),
    ):
        ax.set_title(snap.label, fontsize=11, color=color, fontweight="bold", pad=8)
        if img is not None:
            ax.imshow(np.asarray(img))
        else:
            ax.text(
                0.5,
                0.5,
                "Thumbnail unavailable\n(no cached frames / video)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="#6b7280",
            )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)
        ax.set_xlabel(
            f"{snap.n_frames} frames · {snap.fps:.1f} fps",
            fontsize=8,
            color="#4b5563",
        )
    return fig


def _page_metrics(
    left: AnalyzedSessionSnapshot,
    right: AnalyzedSessionSnapshot,
    comparison: SessionComparisonResult,
) -> Figure:
    fig = _new_page()
    _title_block(fig, "Side-by-Side Metrics & Difference Table")

    # Metrics table
    ax1 = fig.add_axes([0.1, 0.58, 0.8, 0.28])
    ax1.set_axis_off()
    ax1.set_title("Side-by-side metrics", loc="left", fontsize=11, fontweight="bold", pad=10)

    rows = [
        ["Metric", left.label, right.label],
        ["Gait Quality", _fmt(left.metrics.gait_quality, "{:.0f}"), _fmt(right.metrics.gait_quality, "{:.0f}")],
        ["Cadence (steps/min)", _fmt(left.metrics.cadence_spm, "{:.0f}"), _fmt(right.metrics.cadence_spm, "{:.0f}")],
        [
            "Walking speed (m/s)",
            _fmt(left.metrics.walking_speed_m_s, "{:.2f}"),
            _fmt(right.metrics.walking_speed_m_s, "{:.2f}"),
        ],
        ["Symmetry (%)", _fmt(left.metrics.symmetry_pct, "{:.0f}"), _fmt(right.metrics.symmetry_pct, "{:.0f}")],
        ["Stability", left.metrics.stability_label, right.metrics.stability_label],
        [
            "Stability margin (m)",
            _fmt(left.metrics.stability_margin_m, "{:.3f}"),
            _fmt(right.metrics.stability_margin_m, "{:.3f}"),
        ],
        [
            "COM excursion (m)",
            _fmt(left.metrics.com_excursion_m, "{:.3f}"),
            _fmt(right.metrics.com_excursion_m, "{:.3f}"),
        ],
        [
            "Virtual GRF peak (BW)",
            _fmt(left.metrics.virtual_grf_peak_bw, "{:.2f}"),
            _fmt(right.metrics.virtual_grf_peak_bw, "{:.2f}"),
        ],
        [
            "Usable gait cycles",
            _fmt(
                None if left.metrics.usable_gait_cycles is None else float(left.metrics.usable_gait_cycles),
                "{:.0f}",
            ),
            _fmt(
                None if right.metrics.usable_gait_cycles is None else float(right.metrics.usable_gait_cycles),
                "{:.0f}",
            ),
        ],
    ]
    table = ax1.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="upper center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if r == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f4f6")
        else:
            cell.set_facecolor("white")
        if c == 1:
            cell.set_text_props(color=COLOR_A if r > 0 else "white")
        if c == 2:
            cell.set_text_props(color=COLOR_B if r > 0 else "white")

    # Difference table
    ax2 = fig.add_axes([0.1, 0.12, 0.8, 0.40])
    ax2.set_axis_off()
    ax2.set_title(
        "Difference table (right − left)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    diff_rows = []
    for d in comparison.diffs:
        if d.name == "Stability" and d.left_value is not None:
            delta_txt = d.display.split(":", 1)[-1].strip() if "→" in d.display else _fmt(d.delta, "{:+.0f}")
        else:
            delta_txt = _fmt(d.delta, "{:+.3f}") if d.unit in (" m", " BW", " m/s") else _fmt(d.delta, "{:+.1f}")
            if d.delta is not None and d.unit:
                delta_txt = f"{delta_txt}{d.unit}".replace("++", "+")
        tone = d.tone_right
        tone_label = {"better": "↑ favourable (right)", "worse": "↓ less favourable (right)", "neutral": "similar"}.get(
            tone, "—"
        )
        diff_rows.append([d.name, delta_txt, tone_label])

    if not diff_rows:
        ax2.text(0.0, 0.9, "No numeric differences available.", transform=ax2.transAxes, fontsize=9)
        return fig

    t2 = ax2.table(
        cellText=diff_rows,
        colLabels=["Metric", "Δ (right − left)", "Interpretation"],
        loc="upper center",
        cellLoc="center",
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(8)
    t2.scale(1.0, 1.3)
    for (r, c), cell in t2.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if r == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", fontweight="bold")
            continue
        tone = comparison.diffs[r - 1].tone_right
        if tone == "better":
            cell.set_facecolor("#dcfce7")
        elif tone == "worse":
            cell.set_facecolor("#fee2e2")
        elif r % 2 == 0:
            cell.set_facecolor("#f9fafb")
        else:
            cell.set_facecolor("white")
    return fig


def _page_graphs(left: AnalyzedSessionSnapshot, right: AnalyzedSessionSnapshot) -> Figure:
    fig = _new_page()
    _title_block(fig, "Overlay Graphs", "Knee flexion vs normalised time (both sessions)")
    ax = fig.add_axes([0.12, 0.18, 0.76, 0.68])
    ax.set_facecolor("#fafafa")
    plotted = False
    for snap, color in ((left, COLOR_A), (right, COLOR_B)):
        if not snap.knee_t:
            continue
        t = np.asarray(snap.knee_t, dtype=float)
        y = np.asarray(
            [np.nan if v is None else float(v) for v in snap.knee_y],
            dtype=float,
        )
        if len(t) < 2:
            continue
        tn = (t - t[0]) / max(float(t[-1] - t[0]), 1e-6)
        ax.plot(tn, y, color=color, linewidth=1.8, label=snap.label)
        plotted = True
    ax.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax.set_xlabel("Normalised time (0–1)", fontsize=9)
    ax.set_ylabel("Mean knee flexion (°)", fontsize=9)
    ax.tick_params(labelsize=8)
    if plotted:
        ax.legend(loc="best", fontsize=9, frameon=True)
    else:
        ax.text(0.5, 0.5, "No knee angle series available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Overlay — green: left/reference, red: right/comparator", fontsize=9, color="#4b5563")
    return fig


def _page_trajectory(left: AnalyzedSessionSnapshot, right: AnalyzedSessionSnapshot) -> Figure:
    fig = _new_page()
    _title_block(fig, "3D Trajectory Snapshot", "Overlaid centre-of-mass paths in a shared coordinate frame")
    ax = fig.add_axes([0.1, 0.16, 0.8, 0.70], projection="3d")
    any_path = False
    for snap, color in ((left, COLOR_A), (right, COLOR_B)):
        if snap.path_xyz is None:
            continue
        p = np.asarray(snap.path_xyz, dtype=float)
        if p.ndim != 2 or len(p) < 2:
            continue
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, linewidth=1.8, label=snap.label)
        ax.scatter([p[0, 0]], [p[0, 1]], [p[0, 2]], c=color, s=28, marker="o")
        ax.scatter([p[-1, 0]], [p[-1, 1]], [p[-1, 2]], c=color, s=36, marker="^")
        any_path = True
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)
    ax.tick_params(labelsize=7)
    if any_path:
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.text2D(0.5, 0.5, "No COM trajectories available", transform=ax.transAxes, ha="center")
    ax.set_title("○ start   △ end", fontsize=8, color="#6b7280")
    return fig


def _page_narratives(comparison: SessionComparisonResult) -> Figure:
    fig = _new_page()
    _title_block(fig, "Interpretation & Recommendations")

    sections = [
        ("Laboratory comparison report", comparison.lab_report_summary or comparison.interpretation),
        ("Automatic interpretation", comparison.interpretation),
        ("Clinical summary", comparison.clinical_summary),
        ("Research summary", comparison.research_summary),
        ("Overall recommendation", comparison.overall_recommendation),
    ]

    y = 0.88
    for title, body in sections:
        fig.text(0.1, y, title, fontsize=11, fontweight="bold", color="#111827", ha="left", va="top")
        y -= 0.028
        wrapped = textwrap.fill(body.replace("\n", " "), width=98)
        lines = wrapped.split("\n")
        chunk = "\n".join(lines)
        fig.text(
            0.1,
            y,
            chunk,
            fontsize=8.8,
            color="#1f2937",
            ha="left",
            va="top",
            linespacing=1.35,
            family="DejaVu Sans",
        )
        y -= 0.018 * max(len(lines), 1) + 0.035
        if y < 0.12:
            break

    fig.text(
        0.1,
        0.08,
        "End of StableWalk Comparison Report",
        fontsize=8,
        color="#6b7280",
        style="italic",
    )
    return fig


def export_comparison_report_from_cache(
    cache: Any,
    path: Path | str,
    *,
    progress: float = 0.35,
) -> Path:
    """Export using a SessionCompareCache; raises ValueError if slots incomplete."""
    left = cache.left()
    right = cache.right()
    if left is None or right is None:
        missing = []
        if left is None:
            missing.append(getattr(cache, "left_key", "left"))
        if right is None:
            missing.append(getattr(cache, "right_key", "right"))
        raise ValueError(
            "Both comparison sessions must be analysed before exporting. "
            f"Missing: {', '.join(missing)}"
        )
    return export_comparison_report_pdf(left, right, path, progress=progress)


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "build_comparison_report_result",
    "export_comparison_report_from_cache",
    "export_comparison_report_pdf",
]
