"""
Compact plain-language interpretations for StableWalk dashboard metrics.

Each card follows:
  Metric name / Current value / One-sentence interpretation / Confidence level
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from stablewalk.ui.scientific_labels import LABEL_GAIT_QUALITY, UNIT_CADENCE

if TYPE_CHECKING:
    from stablewalk.analysis.gait_analysis_summary import GaitAnalysisSummary
    from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult, FrameContactState
    from stablewalk.analysis.gait_feature_analysis import GaitFeatureAnalysisResult
    from stablewalk.analysis.stability_scoring import MetricResult, StabilityResult
    from stablewalk.ui.selected_point_analysis import BilateralGroundClearancePanel
    from stablewalk.ui.viewers.knee_angle_chart import KneeAngleSeries
    from stablewalk.ui.viewers.knee_chart_interpretation import KneeMotionSummary

ConfidenceLevel = Literal["High", "Moderate", "Low", "Insufficient"]

PROJECTION_3D = "3D"
PROJECTION_FRONTAL = "Frontal Plane"
PROJECTION_SAGITTAL = "Sagittal Plane"

SUMMARY_MS_SUBTITLE = "Body motion control"
SUMMARY_MS_DEFINITION = (
    "Measures control and steadiness of pelvis and trunk motion."
)
SUMMARY_GQ_SUBTITLE = "Symmetry and gait coordination"
SUMMARY_GQ_DEFINITION = (
    "Measures gait timing, left-right symmetry, contact sequencing, and cycle consistency."
)
SUMMARY_AC_SUBTITLE = "Reliability of this analysis"
SUMMARY_IMPORTANT_NOTE = (
    "Movement Stability and Gait Quality measure different aspects of walking. "
    "A person may show steady torso movement while still having asymmetric or impaired gait."
)

GaitEvidenceBadge = Literal["PROVISIONAL", "INSUFFICIENT GAIT EVIDENCE"]


@dataclass(frozen=True)
class CompactInterpretation:
    name: str
    value: str
    sentence: str
    confidence: ConfidenceLevel


@dataclass(frozen=True)
class TrajectoryPathMetrics:
    total_travel_m: float
    smoothness: ConfidenceLevel
    max_deviation_m: float
    out_of_plane_axis: str


def format_score_over_100(score: float | None) -> str:
    """Display a domain score as ``91 / 100`` without ambiguous asterisks."""
    if score is None:
        return "—"
    return f"{score:.0f} / 100"


def truncate_dashboard_explanation(text: str, max_len: int = 120) -> str:
    """Keep on-dashboard interpretation to one short readable line."""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def gait_quality_evidence_badge(
    summary: "GaitAnalysisSummary",
    *,
    usable_gait_cycles: int,
) -> GaitEvidenceBadge | None:
    """Explicit badge for limited gait evidence (replaces ``*`` suffixes)."""
    gq = summary.gait_quality
    if gq.score is None or gq.active_domains == 0:
        return "INSUFFICIENT GAIT EVIDENCE"
    if summary.validity_status == "INSUFFICIENT_DATA" and gq.completeness_pct < 20.0:
        return "INSUFFICIENT GAIT EVIDENCE"
    if (
        summary.validity_status == "PROVISIONAL"
        or gq.completeness_pct < 35.0
        or usable_gait_cycles < 2
    ):
        return "PROVISIONAL"
    return None


def format_analysis_confidence_level(level: str | None) -> str:
    if not level:
        return "—"
    return level if level == "INSUFFICIENT" else level.title()


def format_comparable_score_label(comparable: str | None) -> str:
    if not comparable:
        return "Comparable score: —"
    return f"Comparable score: {comparable}"


def format_compact_interpretation(card: CompactInterpretation) -> str:
    return (
        f"{card.name}\n{card.value}\n\n"
        f'"{card.sentence}"\n\n'
        f"Confidence: {card.confidence}"
    )


def confidence_from_tier(tier: str | None, *, score: float | None = None) -> ConfidenceLevel:
    if tier in ("LOW_CONFIDENCE", "LOW", "INSUFFICIENT"):
        return "Insufficient" if tier == "INSUFFICIENT" else "Low"
    if score is not None:
        if score >= 0.72:
            return "High"
        if score >= 0.48:
            return "Moderate"
        return "Low"
    if tier in ("HIGH",):
        return "High"
    if tier in ("MEDIUM", "MODERATE"):
        return "Moderate"
    return "Moderate"


def confidence_from_metric(metric: MetricResult | None) -> ConfidenceLevel:
    if metric is None or metric.availability == "UNAVAILABLE":
        return "Insufficient"
    if metric.availability == "LOW_CONFIDENCE":
        return "Low"
    if metric.confidence is not None:
        return confidence_from_tier(None, score=metric.confidence)
    return "Moderate"


# ── Help texts (scientific tooltips; shared with hover tips) ───────────────

def _metric_help_map() -> dict[str, str]:
    from stablewalk.ui.metric_tooltips import metric_help_body

    keys = (
        "knee_motion",
        "foot_clearance",
        "gait_cycle",
        "movement_stability",
        "gait_quality",
        "analysis_confidence",
        "joint_movement_3d",
        "com",
        "bos",
        "grf",
        "rom",
        "cadence",
        "symmetry",
        "stability",
        "heel_strike",
        "toe_off",
        "joint_angle",
        "pipeline_status",
    )
    return {k: metric_help_body(k) for k in keys if metric_help_body(k)}


METRIC_HELP: dict[str, str] = _metric_help_map()


def default_projection_for_view(view_type: str | None) -> str:
    vt = (view_type or "").upper()
    if vt in ("SAGITTAL_LEFT", "SAGITTAL_RIGHT"):
        return PROJECTION_SAGITTAL
    if vt == "FRONTAL":
        return PROJECTION_FRONTAL
    return PROJECTION_3D


def coordinate_mode_label(coord_mode: str = "ROOT-RELATIVE") -> str:
    if coord_mode == "GLOBAL":
        return "GLOBAL"
    return "ROOT-RELATIVE"


def coordinate_mode_display(coord_mode: str = "ROOT-RELATIVE") -> str:
    if coord_mode == "GLOBAL":
        return "GLOBAL"
    return "ROOT-RELATIVE"


def movement_path_title(joint_label: str) -> str:
    return f"{joint_label} 3D Movement Path"


def joint_graph_title(joint_label: str, *, coord_mode: str = "ROOT-RELATIVE") -> str:
    return movement_path_title(joint_label)


def compute_trajectory_path_metrics(
    path: list[Vec3],
    *,
    projection: str = PROJECTION_3D,
) -> TrajectoryPathMetrics | None:
    if len(path) < 2:
        return None
    xs = np.array([p.x for p in path], dtype=float)
    ys = np.array([p.y for p in path], dtype=float)
    zs = np.array([p.z for p in path], dtype=float)
    segments = np.sqrt(
        np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2
    )
    total_travel = float(np.sum(segments))
    direct = float(
        np.sqrt(
            (xs[-1] - xs[0]) ** 2
            + (ys[-1] - ys[0]) ** 2
            + (zs[-1] - zs[0]) ** 2
        )
    )
    ratio = total_travel / max(direct, 1e-6)
    if ratio <= 1.2:
        smooth = "High"
    elif ratio <= 1.55:
        smooth = "Moderate"
    else:
        smooth = "Low"

    if projection == PROJECTION_FRONTAL:
        oop = zs
        oop_name = "Z (forward)"
    elif projection == PROJECTION_SAGITTAL:
        oop = xs
        oop_name = "X (mediolateral)"
    else:
        oop = xs
        oop_name = "X (mediolateral)"

    max_dev = float(np.max(np.abs(oop - np.mean(oop)))) if oop.size else 0.0
    return TrajectoryPathMetrics(
        total_travel_m=total_travel,
        smoothness=smooth,
        max_deviation_m=max_dev,
        out_of_plane_axis=oop_name,
    )


@dataclass(frozen=True)
class TrajectoryReadiness:
    sufficient: bool
    reason: str
    confidence: ConfidenceLevel
    metrics: TrajectoryPathMetrics | None


def evaluate_trajectory_readiness(
    path: list[Vec3],
    *,
    projection: str = PROJECTION_3D,
    min_samples: int = 3,
) -> TrajectoryReadiness:
    if len(path) < 2:
        return TrajectoryReadiness(
            sufficient=False,
            reason="Fewer than 2 tracked frames for this joint.",
            confidence="Insufficient",
            metrics=None,
        )
    metrics = compute_trajectory_path_metrics(path, projection=projection)
    if metrics is None:
        return TrajectoryReadiness(
            sufficient=False,
            reason="Could not compute a movement path from the available frames.",
            confidence="Insufficient",
            metrics=None,
        )
    if len(path) < min_samples:
        return TrajectoryReadiness(
            sufficient=False,
            reason=(
                f"Only {len(path)} valid samples; at least {min_samples} are needed "
                "for a reliable trajectory."
            ),
            confidence="Insufficient",
            metrics=metrics,
        )
    confidence: ConfidenceLevel = metrics.smoothness
    return TrajectoryReadiness(
        sufficient=True,
        reason="",
        confidence=confidence,
        metrics=metrics,
    )


def format_trajectory_confidence(level: ConfidenceLevel | str | None) -> str:
    if not level or level == "Insufficient":
        return "INSUFFICIENT"
    return str(level).upper()


def interpret_joint_trajectory(
    joint_label: str,
    metrics: TrajectoryPathMetrics | None,
    *,
    projection: str,
    view_type: str | None = None,
) -> CompactInterpretation:
    if metrics is None:
        return CompactInterpretation(
            name=movement_path_title(joint_label),
            value="",
            sentence="Select a joint and play the video to build a movement path.",
            confidence="Insufficient",
        )
    plane_word = {
        PROJECTION_SAGITTAL: "sagittal",
        PROJECTION_FRONTAL: "frontal",
        PROJECTION_3D: "3D",
    }.get(projection, "3D")
    dev_cm = metrics.max_deviation_m * 100.0
    if metrics.smoothness == "High" and dev_cm < 3.0:
        sentence = (
            f"The {joint_label.lower()} follows a relatively consistent {plane_word} path "
            f"with limited out-of-plane movement."
        )
    elif metrics.smoothness == "Low":
        sentence = (
            f"The {joint_label.lower()} path varies substantially between frames, "
            f"so this trajectory should be interpreted cautiously."
        )
    else:
        sentence = (
            f"The {joint_label.lower()} shows moderate {plane_word} travel "
            f"with some out-of-plane spread ({metrics.out_of_plane_axis})."
        )
    conf: ConfidenceLevel = metrics.smoothness
    if view_type and view_type.upper() == "UNKNOWN":
        conf = "Low"
    return CompactInterpretation(
        name=movement_path_title(joint_label),
        value="",
        sentence=truncate_dashboard_explanation(sentence, max_len=160),
        confidence=conf,
    )


def interpret_knee_motion(summary: KneeMotionSummary) -> CompactInterpretation:
    parts: list[str] = []
    if summary.left_rom_deg is not None and summary.right_rom_deg is not None:
        parts.append(
            f"L ROM {summary.left_rom_deg:.0f}°  |  R ROM {summary.right_rom_deg:.0f}°"
        )
    if summary.rom_asymmetry_pct is not None:
        parts.append(f"Asymmetry {summary.rom_asymmetry_pct:.1f}%")
    value = "  |  ".join(parts) if parts else "—"
    conf: ConfidenceLevel = "Moderate"
    if summary.cycle_repeatability == "High":
        conf = "High"
    elif summary.cycle_repeatability == "Low":
        conf = "Low"
    if "Insufficient" in summary.plain_language:
        conf = "Insufficient"
    return CompactInterpretation(
        name="Knee Motion",
        value=value,
        sentence=summary.plain_language,
        confidence=conf,
    )


def interpret_foot_clearance(panel: BilateralGroundClearancePanel | None) -> CompactInterpretation:
    if panel is None:
        return CompactInterpretation(
            name="Foot Clearance",
            value="L —  |  R —",
            sentence="Foot clearance appears when both feet are visible in the pose.",
            confidence="Insufficient",
        )
    value = f"L {panel.left_cm}  |  R {panel.right_cm}"
    try:
        l_val = float(panel.left_cm.replace("cm", "").strip())
        r_val = float(panel.right_cm.replace("cm", "").strip())
        if abs(l_val - r_val) < 0.8:
            sentence = "Left and right swing clearance are similar in the current frame."
        elif r_val > l_val:
            sentence = "Right swing clearance is higher than left clearance in the current frame."
        else:
            sentence = "Left swing clearance is higher than right clearance in the current frame."
    except ValueError:
        sentence = panel.phase_summary or "Clearance estimated from foot height above the floor."
    conf = "Low" if "check" in panel.left_cm.lower() or "check" in panel.right_cm.lower() else "Moderate"
    return CompactInterpretation(
        name="Foot Clearance",
        value=value,
        sentence=sentence,
        confidence=conf,
    )


def interpret_gait_phase(
    state: FrameContactState | None,
    result: GaitCycleAnalysisResult | None,
) -> CompactInterpretation:
    if state is None or result is None:
        return CompactInterpretation(
            name="Gait Phase",
            value="Phase —",
            sentence="Gait phase is detected from foot height and contact timing.",
            confidence="Insufficient",
        )
    phase = state.phase.replace("_", " ").title()
    value = f"Phase: {phase}"
    m = result.metrics
    extras: list[str] = []
    if m.cadence_steps_per_min is not None:
        extras.append(f"Cadence {m.cadence_steps_per_min:.0f} {UNIT_CADENCE}")
    if m.double_support_pct is not None:
        extras.append(f"DS {m.double_support_pct:.0f}%")
    if extras:
        value += "  |  " + "  |  ".join(extras)
    if state.phase == "LEFT_STANCE":
        sentence = "The left foot is bearing weight while the right leg is likely swinging."
    elif state.phase == "RIGHT_STANCE":
        sentence = "The right foot is bearing weight while the left leg is likely swinging."
    elif state.phase == "DOUBLE_SUPPORT":
        sentence = "Both feet are near the ground — a brief double-support part of the step."
    elif state.phase == "FLIGHT":
        sentence = "Both feet are clearly above the floor in this frame."
    elif state.phase in ("UNCERTAIN", "FLIGHT_OR_UNCERTAIN"):
        sentence = "Foot contact timing is uncertain in this frame; phase may be mid-transition."
    else:
        sentence = "Gait phase could not be classified for this frame."
    conf = confidence_from_tier(m.confidence_tier, score=m.contact_confidence)
    return CompactInterpretation(
        name="Gait Phase",
        value=value,
        sentence=sentence,
        confidence=conf,
    )


def interpret_domain_metric(
    metric: MetricResult | None,
    *,
    display_name: str,
) -> CompactInterpretation:
    if metric is None or metric.score is None:
        return CompactInterpretation(
            name=display_name,
            value="—",
            sentence=f"{display_name} needs more detected steps or clearer pose data.",
            confidence="Insufficient",
        )
    suffix = "*" if metric.availability == "LOW_CONFIDENCE" else ""
    value = f"{metric.score:.0f}{suffix}"
    sentence = metric.summary or f"{display_name} summarizes one aspect of this walking clip."
    return CompactInterpretation(
        name=display_name,
        value=value,
        sentence=sentence,
        confidence=confidence_from_metric(metric),
    )


def interpret_movement_stability(result: StabilityResult | None) -> CompactInterpretation:
    if result is None or result.gait_summary is None:
        return CompactInterpretation(
            name="Movement Stability",
            value="—",
            sentence="Measures pelvis and trunk steadiness plus smooth joint motion.",
            confidence="Insufficient",
        )
    gs = result.gait_summary
    score = gs.movement_stability.score
    value = f"{score:.0f}" if score is not None else "—"
    if score is not None and score >= 70:
        sentence = "Torso and pelvis motion look relatively steady in this recording."
    elif score is not None and score < 45:
        sentence = "Body sway or joint motion varies noticeably — stability looks limited."
    else:
        sentence = "Movement stability is moderate; some sway or irregular motion is present."
    level = gs.analysis_confidence.level
    conf: ConfidenceLevel = {
        "HIGH": "High",
        "MODERATE": "Moderate",
        "LOW": "Low",
        "INSUFFICIENT": "Insufficient",
    }.get(level, "Moderate")
    return CompactInterpretation(
        name="Movement Stability",
        value=value,
        sentence=sentence,
        confidence=conf,
    )


def interpret_gait_quality(result: StabilityResult | None) -> CompactInterpretation:
    if result is None or result.gait_summary is None:
        return CompactInterpretation(
            name=LABEL_GAIT_QUALITY,
            value="—",
            sentence="Combines timing, clearance, symmetry, and detected gait cycles.",
            confidence="Insufficient",
        )
    gs = result.gait_summary
    score = gs.gait_quality.score
    value = f"{score:.0f}" if score is not None else "—"
    cycles = result.usable_gait_cycles
    if score is not None and cycles < 2:
        sentence = (
            f"Gait quality score is provisional — only {cycles} usable cycle(s) were detected."
        )
    elif score is not None and score >= 68:
        sentence = "Step timing, clearance, and cycle patterns look fairly regular."
    elif score is not None and score < 45:
        sentence = "Walking regularity looks limited — few cycles or uneven timing/clearance."
    else:
        sentence = "Gait quality is mixed; some timing or clearance aspects are inconsistent."
    return CompactInterpretation(
        name=LABEL_GAIT_QUALITY,
        value=value,
        sentence=sentence,
        confidence=interpret_movement_stability(result).confidence,
    )


def interpret_analysis_confidence(result: StabilityResult | None) -> CompactInterpretation:
    if result is None or result.gait_summary is None:
        return CompactInterpretation(
            name="Analysis Confidence",
            value="—",
            sentence="How much trustworthy evidence supported the scores above.",
            confidence="Insufficient",
        )
    ac = result.gait_summary.analysis_confidence
    value = ac.level.title() if ac.level != "INSUFFICIENT" else "Insufficient"
    factors = ac.factors[:2] if ac.factors else []
    if factors:
        sentence = "; ".join(factors) + "."
    elif ac.level == "HIGH":
        sentence = "Enough pose frames and gait cycles support these measurements."
    elif ac.level == "INSUFFICIENT":
        sentence = "Too little reliable gait evidence — treat all scores as preliminary."
    else:
        sentence = f"Analysis completeness is about {ac.completeness_pct:.0f}% for this clip."
    return CompactInterpretation(
        name="Analysis Confidence",
        value=value,
        sentence=sentence,
        confidence=value if value in ("High", "Moderate", "Low", "Insufficient") else "Moderate",
    )
