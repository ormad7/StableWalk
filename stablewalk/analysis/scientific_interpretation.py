"""
Generate a clinical-style interpretation from analysis summary fields.

The text describes measured and estimated biomechanical parameters only.
It does not diagnose disease or infer medical conclusions.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stablewalk.analysis.analysis_summary import AnalysisSummary
    from stablewalk.analysis.biomechanical.orchestrator import BiomechanicalAnalysisResult

_MIN_SENTENCES = 4
_MAX_SENTENCES = 8

_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)")
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_SPEED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*m/s", re.IGNORECASE)

_TYPICAL_COMFORTABLE_SPEED_MS = 1.2
_TYPICAL_ELEVATED_CADENCE = 115.0
_TYPICAL_REDUCED_CADENCE = 90.0


def _first_number(text: str) -> float | None:
    match = _SCORE_RE.search(text)
    return float(match.group(1)) if match else None


def _parse_speed_ms(text: str) -> float | None:
    match = _SPEED_RE.search(text)
    return float(match.group(1)) if match else None


def _stable_frame_pct(
    summary: AnalysisSummary,
    biomechanical: BiomechanicalAnalysisResult | None,
) -> float | None:
    if biomechanical is not None and biomechanical.stability_margin is not None:
        value = biomechanical.stability_margin.stable_pct
        return float(value) if value is not None else None
    field = summary.stability_margin
    if field is None or not field.available:
        return None
    match = _PCT_RE.search(field.value)
    return float(match.group(1)) if match else None


def _dominant_stability_label(
    summary: AnalysisSummary,
    biomechanical: BiomechanicalAnalysisResult | None,
) -> str | None:
    if biomechanical is not None and biomechanical.stability_margin is not None:
        counts: dict[str, int] = {}
        for frame in biomechanical.stability_margin.per_frame:
            counts[frame.stability_state] = counts.get(frame.stability_state, 0) + 1
        if counts:
            return max(counts, key=counts.get)
    field = summary.stability_margin
    if field is None or not field.available:
        return None
    for label in ("Unstable", "Reduced Stability", "Stable"):
        if label in field.value:
            return label
    return None


def _opening_clinical_assessment(
    summary: AnalysisSummary,
    biomechanical: BiomechanicalAnalysisResult | None,
) -> str:
    """Synthesized lead paragraph in clinical report style."""
    stable_pct = _stable_frame_pct(summary, biomechanical)
    dominant = _dominant_stability_label(summary, biomechanical)
    quality = summary.overall_gait_quality
    gq = _first_number(quality.value) if quality and quality.available else None
    sym_field = summary.symmetry
    sym = _first_number(sym_field.value) if sym_field and sym_field.available else None

    if dominant == "Unstable" or (stable_pct is not None and stable_pct < 40):
        stability_phrase = "markedly reduced stability"
    elif dominant == "Reduced Stability" or (stable_pct is not None and stable_pct < 55):
        stability_phrase = "reduced stability"
    elif stable_pct is not None and stable_pct >= 70:
        stability_phrase = "predominantly stable weight-bearing control"
    else:
        stability_phrase = "mixed stability characteristics"

    causes: list[str] = []
    if sym is not None and sym < 70:
        causes.append("increased asymmetry")
    if stable_pct is not None and stable_pct < 55:
        causes.append("a reduced stability margin")
    if gq is not None and gq < 55 and not causes:
        causes.append("limited overall gait coordination")

    if causes:
        cause_text = " and ".join(causes)
        opening = (
            f"The analyzed gait demonstrates {stability_phrase} caused primarily by {cause_text}."
        )
    else:
        opening = (
            f"The analyzed gait demonstrates {stability_phrase} across the estimated "
            "and derived temporal and spatial parameters."
        )

    cadence_field = summary.cadence
    speed_field = summary.walking_speed
    cadence = _first_number(cadence_field.value) if cadence_field and cadence_field.available else None
    speed = _parse_speed_ms(speed_field.value) if speed_field and speed_field.available else None

    speed_clause = ""
    cadence_clause = ""
    if speed is not None and speed < _TYPICAL_COMFORTABLE_SPEED_MS:
        speed_clause = "Walking speed is lower than expected for comfortable adult overground walking"
    elif speed is not None and speed >= 1.35:
        speed_clause = "Walking speed is elevated relative to typical comfortable overground walking"

    if cadence is not None and cadence >= _TYPICAL_ELEVATED_CADENCE:
        cadence_clause = "cadence remains elevated"
    elif cadence is not None and cadence <= _TYPICAL_REDUCED_CADENCE:
        cadence_clause = "cadence is reduced"

    if speed_clause and cadence_clause:
        if "lower" in speed_clause and "elevated" in cadence_clause:
            opening += (
                f" {speed_clause} while {cadence_clause}, suggesting a compensatory gait strategy."
            )
        else:
            opening += f" {speed_clause}, and {cadence_clause}."
    elif speed_clause:
        opening += f" {speed_clause}."
    elif cadence_clause:
        opening += f" {cadence_clause.capitalize()}."

    return opening


def _sentence_gait_quality(summary: AnalysisSummary) -> str:
    field = summary.overall_gait_quality
    if field is None or not field.available:
        return (
            "A derived gait quality score could not be computed from the available "
            "pose and gait-cycle evidence in this recording."
        )
    score = _first_number(field.value)
    if score is None:
        return (
            f"The derived gait quality indicator is reported as {field.value}, "
            "summarizing coordination across detected gait cycles in this clip."
        )
    if score >= 68:
        tone = "relatively consistent"
    elif score < 45:
        tone = "limited"
    else:
        tone = "mixed"
    detail = field.reason.strip() if field.reason else ""
    if detail:
        return (
            f"The composite gait quality score is {field.value}, indicating {tone} walking "
            f"coordination in this recording ({detail})."
        )
    return (
        f"The composite gait quality score is {field.value}, indicating {tone} walking "
        "coordination across the detected cycles in this clip."
    )


def _sentence_stability_margin(
    summary: AnalysisSummary,
    biomechanical: BiomechanicalAnalysisResult | None,
) -> str:
    stable_pct = _stable_frame_pct(summary, biomechanical)
    dominant = _dominant_stability_label(summary, biomechanical)
    if stable_pct is None and dominant is None:
        return (
            "Stability margin estimates were not available because center-of-mass and "
            "base-of-support polygon data were incomplete for this session."
        )
    pct_text = f"{stable_pct:.0f}%" if stable_pct is not None else "an unquantified share"
    if dominant == "Unstable":
        return (
            f"The estimated center of mass is classified outside the base-of-support polygon "
            f"in multiple frames ({pct_text} stable frames), indicating an unstable derived "
            "stability margin profile for this walking sequence."
        )
    if dominant == "Reduced Stability" or (stable_pct is not None and stable_pct < 50):
        return (
            "The estimated center of mass frequently approaches the support boundary, with many "
            f"frames classified in the reduced-stability margin band ({pct_text} stable frames)."
        )
    return (
        f"The estimated center of mass remains predominantly inside the derived base-of-support "
        f"polygon, with {pct_text} of frames classified as stable in the stability margin analysis."
    )


def _sentence_symmetry(summary: AnalysisSummary) -> str:
    field = summary.symmetry
    if field is None or not field.available:
        return (
            "Left-right symmetry indices could not be derived reliably from the bilateral "
            "timing and spatial metrics available in this clip."
        )
    pct = _first_number(field.value)
    if pct is None:
        return (
            f"Bilateral symmetry is summarized as {field.value}, based on derived left-right "
            "comparisons of gait timing and spatial parameters."
        )
    if pct >= 82:
        return (
            f"Derived left-right symmetry is {field.value}, suggesting relatively balanced "
            "step timing and spatial progression between sides in this recording."
        )
    if pct < 65:
        return (
            f"Increased left-right asymmetry is estimated ({field.value} symmetry index), "
            "indicating measurable differences between left and right gait parameters in this clip."
        )
    return (
        f"Derived symmetry is {field.value}, indicating moderate left-right differences in "
        "gait timing or spatial parameters across the analyzed sequence."
    )


def _sentence_com(summary: AnalysisSummary) -> str:
    field = summary.center_of_mass
    if field is None or not field.available:
        return (
            "Center-of-mass trajectory estimates were too limited to describe vertical "
            "oscillation or path variability in this recording."
        )
    reason = field.reason.strip() if field.reason else ""
    if reason:
        return f"Center-of-mass behavior is summarized as {field.value.lower()}: {reason}"
    return (
        f"Center-of-mass behavior is summarized as {field.value.lower()} based on the "
        "estimated three-dimensional pelvis-root trajectory across frames."
    )


def _sentence_vgrf(summary: AnalysisSummary) -> str:
    field = summary.estimated_virtual_grf
    if field is None or not field.available:
        return (
            "Estimated virtual ground reaction force profiles were not available, so loading "
            "patterns cannot be described beyond contact-state timing in this session."
        )
    reason = field.reason.strip() if field.reason else ""
    if reason:
        return (
            f"Estimated virtual GRF (not force-plate measured) is characterized as "
            f"{field.value.lower()}: {reason}"
        )
    return (
        f"Estimated virtual ground reaction force (not force-plate measured) is characterized "
        f"as {field.value.lower()} from pelvis kinematics and foot contact timing."
    )


def _sentence_confidence(summary: AnalysisSummary) -> str:
    conf = summary.analysis_confidence
    vq = summary.video_quality
    tracking = summary.tracking_confidence
    pipeline = summary.pipeline_confidence
    conf_text = conf.value if conf and conf.available else "limited"
    details: list[str] = []
    if vq and vq.available:
        details.append(f"video quality {vq.value}")
    if tracking and tracking.available:
        details.append(f"tracking confidence {tracking.value}")
    if pipeline and pipeline.available:
        details.append(f"pipeline confidence {pipeline.value}")
    if details:
        joined = ", ".join(details)
        return (
            f"Overall analysis confidence is rated {conf_text} for this session ({joined}); "
            "reported values describe estimated biomechanical parameters from monocular video "
            "pose data, not clinical diagnosis."
        )
    return (
        f"Overall analysis confidence is rated {conf_text} for this session; reported values "
        "describe estimated biomechanical parameters from monocular video pose data, not "
        "clinical diagnosis."
    )


def build_scientific_interpretation_sentences(
    summary: AnalysisSummary,
    *,
    biomechanical: BiomechanicalAnalysisResult | None = None,
) -> list[str]:
    """Build 4–8 interpretation sentences grounded in summary fields."""
    opening = _opening_clinical_assessment(summary, biomechanical)
    candidates = [
        opening,
        _sentence_stability_margin(summary, biomechanical),
        _sentence_symmetry(summary),
        _sentence_gait_quality(summary),
        _sentence_com(summary),
        _sentence_vgrf(summary),
        _sentence_confidence(summary),
    ]

    selected: list[str] = []
    for sentence in candidates:
        clean = sentence.strip()
        if not clean:
            continue
        if not clean.endswith("."):
            clean += "."
        if clean not in selected:
            selected.append(clean)

    if len(selected) < _MIN_SENTENCES:
        selected.append(
            "Interpretations are limited to parameters computed for this specific video session "
            "and should not be extrapolated to medical conclusions."
        )
    if len(selected) > _MAX_SENTENCES:
        selected = selected[:_MAX_SENTENCES]

    return selected


def build_scientific_interpretation(
    summary: AnalysisSummary,
    *,
    biomechanical: BiomechanicalAnalysisResult | None = None,
) -> str:
    """Paragraph form of :func:`build_scientific_interpretation_sentences`."""
    return " ".join(
        build_scientific_interpretation_sentences(
            summary,
            biomechanical=biomechanical,
        )
    )


__all__ = [
    "build_scientific_interpretation",
    "build_scientific_interpretation_sentences",
]
