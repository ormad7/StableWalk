"""
Visual interpretation levels for Results Summary metric cards.

Maps existing summary *display strings* to normal / borderline / abnormal
color coding. Does not recompute metrics — only classifies rendered values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from stablewalk.analysis.analysis_summary import SummaryField
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
    if score >= 60:
        return "normal"
    if score >= 45:
        return "borderline"
    return "abnormal"


def _classify_confidence_score(score: float) -> InterpretationLevel:
    if score >= 72:
        return "normal"
    if score >= 50:
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
        if pct >= 70:
            return "normal"
        if pct >= 55:
            return "borderline"
        return "abnormal"

    if key == "stability_margin":
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
        if "reduced stability" in lower:
            return "borderline"
        stable_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*stable", lower)
        if stable_match:
            stable_pct = float(stable_match.group(1))
            if stable_pct >= 70:
                return "normal"
            if stable_pct >= 55:
                return "borderline"
            return soften_view_limited_level("abnormal", view_limited=view_limited)
        if "stable" in lower and "reduced" not in lower and "unstable" not in lower:
            return "normal"
        return "neutral"

    if key == "cadence":
        spm = _first_number(value)
        if spm is None:
            return "neutral"
        # Monocular demos often under-count steps; keep a wide adult walking band.
        if 60 <= spm <= 130:
            return "normal"
        if 48 <= spm < 60 or 130 < spm <= 145:
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


def metric_visual_style(level: InterpretationLevel) -> MetricVisualStyle:
    if level == "normal":
        return MetricVisualStyle("normal", SUCCESS, SUCCESS, "Normal")
    if level == "borderline":
        return MetricVisualStyle("borderline", ORANGE, ORANGE, "Borderline")
    if level == "abnormal":
        return MetricVisualStyle("abnormal", DANGER, DANGER, "Abnormal")
    if level == "unavailable":
        return MetricVisualStyle("unavailable", MUTED, BORDER, "")
    return MetricVisualStyle("neutral", TEXT, BORDER, "")


def format_status_line(level: InterpretationLevel, tier: str) -> str:
    """Status badge text: interpretation label plus data tier."""
    from stablewalk.ui.scientific_labels import format_tier_badge

    style = metric_visual_style(level)
    tier_text = format_tier_badge(tier)
    if style.status_label and tier_text:
        return f"{style.status_label} · {tier_text}"
    if style.status_label:
        return style.status_label
    return tier_text
