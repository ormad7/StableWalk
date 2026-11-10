"""
Visual interpretation levels for Results Summary metric cards.

Maps existing summary *display strings* to normal / borderline / abnormal
color coding using the shared metrics registry. Does not recompute metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from stablewalk.analysis.analysis_summary import SummaryField
from stablewalk.metrics.definitions import (
    CADENCE_BORDER_HI,
    CADENCE_BORDER_LO,
    CADENCE_NORMAL_HI,
    CADENCE_NORMAL_LO,
    CONFIDENCE_BORDERLINE_MIN,
    CONFIDENCE_NORMAL_MIN,
    GQ_BORDERLINE_MIN,
    GQ_NORMAL_MIN,
    STABLE_PCT_BORDERLINE_MIN,
    STABLE_PCT_NORMAL_MIN,
    SYMMETRY_BORDERLINE_MIN,
    SYMMETRY_NORMAL_MIN,
)
from stablewalk.metrics.status import DisplayStatus
from stablewalk.ui.theme import BORDER, DANGER, MUTED, ORANGE, SUCCESS, TEXT

InterpretationLevel = Literal[
    "normal", "borderline", "abnormal", "neutral", "unavailable"
]

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
_SPEED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*m/s", re.IGNORECASE)


@dataclass(frozen=True)
class MetricVisualStyle:
    """Foreground and card accent for one interpretation level."""

    level: InterpretationLevel
    value_fg: str
    border: str
    status_label: str


def _first_number(text: str) -> float | None:
    match = _NUMBER_RE.search(text)
    return float(match.group(1)) if match else None


def _score_over_100(text: str) -> float | None:
    if "/" in text:
        head = text.split("/", 1)[0].strip()
        try:
            return float(head)
        except ValueError:
            pass
    return _first_number(text)


def _classify_score_100(score: float) -> InterpretationLevel:
    if score >= GQ_NORMAL_MIN:
        return "normal"
    if score >= GQ_BORDERLINE_MIN:
        return "borderline"
    return "abnormal"


def _classify_confidence_score(score: float) -> InterpretationLevel:
    if score >= CONFIDENCE_NORMAL_MIN:
        return "normal"
    if score >= CONFIDENCE_BORDERLINE_MIN:
        return "borderline"
    return "abnormal"


def is_view_limited_for_quality(view_type: str | None) -> bool:
    """True when contact / COM–BoS quality badges should be softened."""
    vt = (view_type or "").upper()
    return vt in ("FRONTAL", "OBLIQUE")


def soften_view_limited_level(
    level: InterpretationLevel,
    *,
    view_limited: bool,
) -> InterpretationLevel:
    """Cap harsh Abnormal badges when the camera view makes KPIs unreliable."""
    if not view_limited or level != "abnormal":
        return level
    return "borderline"


def interpret_summary_metric(
    key: str,
    field: SummaryField | None,
    *,
    view_type: str | None = None,
    stability_valid_ratio: float | None = None,
) -> InterpretationLevel:
    """Classify a summary field for dashboard color coding."""
    if field is None or not field.available:
        return "unavailable"

    value = field.value.strip()
    lower = value.lower()
    view_limited = is_view_limited_for_quality(view_type)

    if lower in ("not available", "—", "-", ""):
        return "unavailable"

    if key == "gait_quality":
        score = _score_over_100(value)
        return _classify_score_100(score) if score is not None else "neutral"

    if key == "symmetry":
        pct = _first_number(value)
        if pct is None:
            return "neutral"
        if pct >= SYMMETRY_NORMAL_MIN:
            return "normal"
        if pct >= SYMMETRY_BORDERLINE_MIN:
            return "borderline"
        return "abnormal"

    if key == "stability_margin":
        # Prefer numeric stable-frame share when present (shared registry bands).
        stable_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*stable", lower)
        if stable_match:
            stable_pct = float(stable_match.group(1))
            if (
                view_limited
                and (
                    stability_valid_ratio is not None and stability_valid_ratio < 0.45
                )
            ) or (
                view_limited
                and stable_pct <= 0.0
            ):
                return "borderline"
            if stable_pct >= STABLE_PCT_NORMAL_MIN:
                return "normal"
            if stable_pct >= STABLE_PCT_BORDERLINE_MIN:
                return "borderline"
            return soften_view_limited_level("abnormal", view_limited=view_limited)
        # Frontal / low-validity COM–BoS is not a clinical Unstable finding.
        if (
            view_limited
            and (
                stability_valid_ratio is not None and stability_valid_ratio < 0.45
            )
        ) or (
            view_limited
            and ("unstable" in lower or "0% stable" in lower or "0 % stable" in lower)
        ):
            return "borderline"
        if "unstable" in lower:
            return soften_view_limited_level("abnormal", view_limited=view_limited)
        if "reduced stability" in lower or "reduced" in lower:
            return "borderline"
        if "stable" in lower and "reduced" not in lower and "unstable" not in lower:
            return "normal"
        if "acceptable" in lower:
            return "normal"
        if "borderline" in lower:
            return "borderline"
        return "neutral"

    if key == "cadence":
        spm = _first_number(value)
        if spm is None:
            return "neutral"
        if CADENCE_NORMAL_LO <= spm <= CADENCE_NORMAL_HI:
            return "normal"
        if CADENCE_BORDER_LO <= spm < CADENCE_NORMAL_LO or CADENCE_NORMAL_HI < spm <= CADENCE_BORDER_HI:
            return "borderline"
        return "abnormal"

    if key == "walking_speed":
        if "unavailable" in lower or "not available" in lower:
            return "unavailable"
        match = _SPEED_RE.search(value)
        if not match:
            return "neutral"
        speed = float(match.group(1))
        if 1.0 <= speed <= 1.35:
            return "normal"
        if 0.75 <= speed < 1.0 or 1.35 < speed <= 1.55:
            return "borderline"
        return "abnormal"

    if key == "center_of_mass":
        if "normal oscillation" in lower:
            return "normal"
        if "minimal" in lower or "reduced" in lower:
            return "borderline"
        if "elevated" in lower:
            return "borderline"
        return "neutral"

    if key == "vgrf":
        if "normal estimated" in lower:
            return "normal"
        if "high estimated" in lower:
            return "borderline"
        if "flat" in lower or "low estimated" in lower:
            return "borderline"
        return "neutral"

    if key == "joint_rom":
        return "normal"

    if key in ("video_quality", "tracking_confidence"):
        score = _score_over_100(value)
        if score is None:
            score = _first_number(value)
        if score is None:
            return "neutral"
        return _classify_confidence_score(score)

    if key == "pipeline_confidence":
        if lower.startswith("high"):
            return "normal"
        if lower.startswith("moderate") or "partial" in lower:
            return "borderline"
        if lower.startswith("low"):
            return "abnormal"
        pct = _first_number(value)
        if pct is not None:
            return _classify_confidence_score(pct)
        return "neutral"

    if key == "confidence":
        if lower.startswith("high"):
            return "normal"
        if lower.startswith("moderate"):
            return "borderline"
        if lower.startswith("low"):
            return "abnormal"
        return "neutral"

    return "neutral"


def interpret_gait_event(*, detected: bool) -> InterpretationLevel:
    return "normal" if detected else "abnormal"


# Heuristic / coverage metrics must not wear a clinical "Acceptable" badge.
_UNVALIDATED_HEURISTIC_KEYS = frozenset(
    {
        "gait_quality",
        "gait_quality_composite",
        "gait_coordination",
        "gait_quality_coordination",
        "symmetry",
    }
)
_HEURISTIC_STATUS_KEYS = frozenset(
    {
        "gait_quality",
        "gait_quality_composite",
        "gait_coordination",
        "gait_quality_coordination",
        "symmetry",
        "movement_stability",
        "pelvis_stability",
        "stability_margin",
        "stable_pct",
        "stability_state",
        "stability",
        "video_quality",
        "tracking_confidence",
        "analysis_confidence",
        "pose_coverage",
        "contact_coverage",
        "spectral_reliability",
        "biomechanical_validity",
    }
)
_COVERAGE_STATUS_KEYS = frozenset(
    {
        "video_quality",
        "tracking_confidence",
        "pose_coverage",
        "contact_coverage",
        "spectral_reliability",
        "biomechanical_validity",
    }
)


def kpi_status_label(level: InterpretationLevel, *, key: str | None = None) -> str:
    """Badge text for a KPI. Heuristic bands are labeled Heuristic, not Acceptable."""
    style = metric_visual_style(level)
    metric = (key or "").strip()
    if level == "unavailable":
        return "N/A"
    if metric in _COVERAGE_STATUS_KEYS and level not in ("unavailable", "neutral"):
        return "Not accuracy"
    if metric in _UNVALIDATED_HEURISTIC_KEYS and level not in ("unavailable", "neutral"):
        return DisplayStatus.UNVALIDATED_HEURISTIC.value
    if metric in _HEURISTIC_STATUS_KEYS and level == "normal":
        return DisplayStatus.HEURISTIC.value
    return style.status_label


def metric_visual_style(level: InterpretationLevel) -> MetricVisualStyle:
    # Shared user-facing vocabulary (Acceptable / Borderline / Reduced).
    if level == "normal":
        return MetricVisualStyle(
            "normal", SUCCESS, SUCCESS, DisplayStatus.ACCEPTABLE.value
        )
    if level == "borderline":
        return MetricVisualStyle(
            "borderline", ORANGE, ORANGE, DisplayStatus.BORDERLINE.value
        )
    if level == "abnormal":
        return MetricVisualStyle(
            "abnormal", DANGER, DANGER, DisplayStatus.REDUCED.value
        )
    if level == "unavailable":
        return MetricVisualStyle("unavailable", MUTED, BORDER, "")
    return MetricVisualStyle("neutral", TEXT, BORDER, "")


def format_status_line(
    level: InterpretationLevel,
    tier: str,
    *,
    key: str | None = None,
) -> str:
    """Status badge text: interpretation label plus data tier."""
    from stablewalk.ui.scientific_labels import format_tier_badge

    label = kpi_status_label(level, key=key)
    tier_text = format_tier_badge(tier)
    if label and tier_text:
        return f"{label} · {tier_text}"
    if label:
        return label
    return tier_text
