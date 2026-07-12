"""
Gait analysis semantics: separate movement control from gait quality.

StableWalk distinguishes three high-level concepts:

1. MOVEMENT STABILITY — pelvis/trunk control, smoothness, root-relative consistency
2. GAIT QUALITY — symmetry, sequencing, coordination, clearance (not clinical diagnosis)
3. ANALYSIS CONFIDENCE — pose/view/temporal evidence for interpreting the above

The legacy composite stability score remains for backward compatibility but must not
be the sole interpretation of walking performance.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from stablewalk.analysis.stability_scoring import MetricResult, StabilityResult
    from stablewalk.analysis.stability_validity import StabilityResultValidity

AnalysisConfidenceLevel = Literal["HIGH", "MODERATE", "LOW", "INSUFFICIENT"]

# Domain partition (weights from stability_config are reused per domain).
MOVEMENT_STABILITY_DOMAIN_KEYS: frozenset[str] = frozenset(
    {
        "pelvis_stability",
        "trunk_stability",
        "joint_smoothness",
    }
)

GAIT_QUALITY_DOMAIN_KEYS: frozenset[str] = frozenset(
    {
        "temporal_symmetry",
        "spatial_symmetry",
        "foot_clearance",
        "cycle_consistency",
        "contact_pattern",
    }
)

DOMAIN_SEMANTICS_DOC = """
Movement Stability domains (visible body-motion control):
  pelvis_stability   — mediolateral/vertical pelvis control in gait frame
  trunk_stability    — root-relative trunk sway and lean consistency
  joint_smoothness   — trajectory smoothness, jerk, controlled motion

Gait Quality domains (pattern/coordination, not pathology):
  temporal_symmetry  — stance/swing/step timing symmetry
  spatial_symmetry   — step/stride/progression asymmetry
  foot_clearance     — swing clearance consistency (not height magnitude)
  cycle_consistency  — gait-cycle repeatability and phase coordination
  contact_pattern    — contact sequencing and double-support timing

Analysis Confidence (measurement evidence, not gait performance):
  pose-valid frame %, foot visibility, camera view reliability,
  usable gait cycles, analysis completeness, domain evidence coverage
""".strip()


@dataclass
class DomainGroupScore:
    """Confidence-weighted score for one semantic domain group."""

    label: str
    score: float | None
    completeness_pct: float
    active_domains: int
    total_domains: int
    domain_keys: list[str] = field(default_factory=list)
    unavailable_domains: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": None if self.score is None else round(self.score, 1),
            "completeness_pct": round(self.completeness_pct, 1),
            "active_domains": self.active_domains,
            "total_domains": self.total_domains,
            "domain_keys": list(self.domain_keys),
            "unavailable_domains": list(self.unavailable_domains),
        }


@dataclass
class AnalysisConfidenceSummary:
    """Evidence supporting interpretability of movement and gait-quality scores."""

    level: AnalysisConfidenceLevel
    badge: str
    completeness_pct: float
    usable_gait_cycles: int
    pose_valid_frame_pct: float
    active_domain_count: int
    evidence_index: float
    view_display_name: str | None = None
    view_confidence: float = 0.0
    repeatability_tier: str = "UNAVAILABLE"
    factors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "badge": self.badge,
            "completeness_pct": round(self.completeness_pct, 1),
            "usable_gait_cycles": self.usable_gait_cycles,
            "pose_valid_frame_pct": round(self.pose_valid_frame_pct, 2),
            "active_domain_count": self.active_domain_count,
            "evidence_index": round(self.evidence_index, 3),
            "view_display_name": self.view_display_name,
            "view_confidence": round(self.view_confidence, 3),
            "repeatability_tier": self.repeatability_tier,
            "factors": list(self.factors),
        }


@dataclass
class GaitAnalysisSummary:
    """
    High-level gait analysis semantics for GUI and reports.

    Does not encode clinical diagnosis or demo-category labels.
    """

    movement_stability: DomainGroupScore
    gait_quality: DomainGroupScore
    analysis_confidence: AnalysisConfidenceSummary
    explanation: str
    legacy_composite_score: float
    legacy_classification: str
    validity_status: str | None = None
    comparable_score: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "movement_stability": self.movement_stability.to_dict(),
            "gait_quality": self.gait_quality.to_dict(),
            "analysis_confidence": self.analysis_confidence.to_dict(),
            "explanation": self.explanation,
            "legacy_composite_score": round(self.legacy_composite_score, 1),
            "legacy_classification": self.legacy_classification,
            "validity_status": self.validity_status,
            "comparable_score": self.comparable_score,
        }


def _group_score(
    metrics: list[MetricResult],
    *,
    label: str,
    domain_keys: frozenset[str],
) -> DomainGroupScore:
    group = [m for m in metrics if m.key in domain_keys]
    total_weight = sum(m.weight for m in group)
    active = [
        m for m in group
        if m.availability != "UNAVAILABLE" and m.score is not None
    ]
    unavailable = [
        m.name for m in group if m.availability == "UNAVAILABLE" or m.score is None
    ]

    if not active or total_weight <= 0:
        return DomainGroupScore(
            label=label,
            score=None,
            completeness_pct=0.0,
            active_domains=0,
            total_domains=len(group),
            domain_keys=sorted(domain_keys),
            unavailable_domains=unavailable,
        )

    denom = sum(m.weight * m.confidence for m in active)
    completeness = (denom / total_weight) * 100.0 if total_weight > 0 else 0.0
    if denom <= 0:
        score = None
    else:
        score = sum(m.score * m.weight * m.confidence for m in active) / denom  # type: ignore[operator]

    return DomainGroupScore(
        label=label,
        score=score,
        completeness_pct=completeness,
        active_domains=len(active),
        total_domains=len(group),
        domain_keys=sorted(domain_keys),
        unavailable_domains=unavailable,
    )


def _pose_valid_frame_pct(result: StabilityResult) -> float:
    video = result.analysis_evidence.get("video", {})
    pct = video.get("valid_pose_frame_pct")
    if pct is not None:
        return float(pct)
    total = result.frame_count
    if total <= 0:
        return 0.0
    valid = video.get("valid_pose_frames")
    if valid is not None:
        return 100.0 * float(valid) / float(total)
    return 100.0


def _foot_visibility_factor(result: StabilityResult) -> float | None:
    cycles = result.analysis_evidence.get("cycles", {})
    if not cycles:
        return None
    contact = result.metric("contact_pattern")
    if contact is None:
        return None
    return float(contact.confidence) if contact.availability != "UNAVAILABLE" else 0.0


def _analysis_confidence_level(
    result: StabilityResult,
    *,
    evidence_index: float,
    validity_status: str | None,
) -> AnalysisConfidenceLevel:
    if validity_status == "INSUFFICIENT_DATA":
        return "INSUFFICIENT"
    badge = result.confidence_badge
    if badge == "HIGH CONFIDENCE" and evidence_index >= 0.68:
        return "HIGH"
    if badge == "MODERATE CONFIDENCE" and evidence_index >= 0.45:
        return "MODERATE"
    if badge == "LOW CONFIDENCE" or evidence_index < 0.32:
        return "LOW"
    return "MODERATE"


def _build_analysis_confidence(
    result: StabilityResult,
    *,
    evidence_index: float,
    validity_status: str | None,
    active_domain_count: int,
) -> AnalysisConfidenceSummary:
    pose_pct = _pose_valid_frame_pct(result)
    level = _analysis_confidence_level(
        result, evidence_index=evidence_index, validity_status=validity_status
    )
    factors: list[str] = []

    if pose_pct < 70:
        factors.append(f"pose-valid frames {pose_pct:.0f}%")
    foot_factor = _foot_visibility_factor(result)
    if foot_factor is not None and foot_factor < 0.45:
        factors.append("limited foot landmark visibility")
    if result.view_display_name and result.view_confidence < 0.55:
        factors.append(
            f"camera view ({result.view_display_name}) has moderate reliability"
        )
    if result.usable_gait_cycles == 0:
        factors.append("no usable complete gait cycles")
    elif result.usable_gait_cycles < 3:
        factors.append(
            f"only {result.usable_gait_cycles} usable gait cycle(s)"
        )
    if result.completeness_pct < 45:
        factors.append(f"analysis completeness {result.completeness_pct:.0f}%")
    if active_domain_count < 5:
        factors.append(f"{active_domain_count}/8 stability domains contributed")

    return AnalysisConfidenceSummary(
        level=level,
        badge=result.confidence_badge,
        completeness_pct=result.completeness_pct,
        usable_gait_cycles=result.usable_gait_cycles,
        pose_valid_frame_pct=pose_pct,
        active_domain_count=active_domain_count,
        evidence_index=evidence_index,
        view_display_name=result.view_display_name,
        view_confidence=result.view_confidence,
        repeatability_tier=result.repeatability_tier,
        factors=factors,
    )


def _metric(result: StabilityResult, key: str) -> MetricResult | None:
    return result.metric(key)


def _movement_control_phrase(
    movement: DomainGroupScore,
    result: StabilityResult,
) -> str:
    pelvis = _metric(result, "pelvis_stability")
    trunk = _metric(result, "trunk_stability")
    scores = [
        s for s in (
            movement.score,
            pelvis.score if pelvis else None,
            trunk.score if trunk else None,
        )
        if s is not None
    ]
    avg = statistics.mean(scores) if scores else None

    if avg is not None and avg >= 72:
        return "Visible pelvis and trunk motion is relatively controlled"
    if avg is not None and avg < 48:
        return "Pelvis and trunk motion show irregular control patterns"
    if movement.score is None or movement.completeness_pct < 20:
        return "Visible body-motion control cannot be assessed from available landmarks"
    return "Body motion control is moderately consistent"


def _knee_rom_phrase(result: StabilityResult) -> str | None:
    joint = _metric(result, "joint_smoothness")
    if joint is None:
        return None
    rom = joint.values.get("rom_deg") or {}
    knee_roms = [
        float(v)
        for key, v in rom.items()
        if "knee" in key and v is not None
    ]
    if not knee_roms:
        return None
    max_rom = max(knee_roms)
    if max_rom < 50:
        return None
    movement = _metric(result, "joint_smoothness")
    smooth = (
        movement.score is not None
        and movement.score >= 58
        and movement.availability != "UNAVAILABLE"
    )
    quality = "smooth" if smooth else "variable"
    return (
        f"Large knee range of motion is present ({max_rom:.0f}°), "
        f"but movement control remains {quality}"
    )


def _gait_quality_phrase(
    gait_quality: DomainGroupScore,
    result: StabilityResult,
) -> str:
    unavailable = len(gait_quality.unavailable_domains)
    total = gait_quality.total_domains

    if gait_quality.score is None or gait_quality.completeness_pct < 22:
        return "gait quality cannot be reliably assessed"
    if gait_quality.score < 45:
        return "gait symmetry, sequencing, or clearance show limitations"
    if gait_quality.score >= 68 and unavailable <= 1:
        return "gait timing and coordination measures are comparatively consistent"
    if unavailable >= total // 2:
        return "gait quality is only partially measured"
    return "gait quality measures are mixed across available domains"


def _evidence_limitation_phrases(result: StabilityResult) -> list[str]:
    limits: list[str] = []

    if result.usable_gait_cycles == 0:
        limits.append("no complete gait cycles were detected")
    elif result.usable_gait_cycles == 1:
        limits.append("temporal assessment is limited by only one usable gait cycle")
    elif result.usable_gait_cycles < 3:
        limits.append(
            f"repeatability assessment is limited to "
            f"{result.usable_gait_cycles} usable gait cycle(s)"
        )

    foot = _metric(result, "foot_clearance")
    if foot is not None and foot.availability == "UNAVAILABLE":
        limits.append("foot motion is partially occluded or swing phases were not captured")

    for key, label in (
        ("temporal_symmetry", "left-right timing symmetry could not be measured"),
        ("spatial_symmetry", "spatial step symmetry could not be measured"),
        ("cycle_consistency", "gait-cycle repeatability could not be measured"),
    ):
        m = _metric(result, key)
        if m is not None and m.availability == "UNAVAILABLE":
            limits.append(label)

    if result.completeness_pct < 40:
        limits.append(
            f"overall analysis completeness is {result.completeness_pct:.0f}%"
        )

    return limits


def generate_gait_analysis_explanation(
    summary: GaitAnalysisSummary,
    result: StabilityResult,
) -> str:
    """
    Build a short evidence-based explanation from measured domains.

    Does not reference demo labels (abnormal / normal / athletic) or claim pathology.
    """
    parts: list[str] = []

    movement_clause = _movement_control_phrase(summary.movement_stability, result)
    rom_clause = _knee_rom_phrase(result)
    if rom_clause:
        parts.append(rom_clause)
    else:
        parts.append(movement_clause)

    gait_clause = _gait_quality_phrase(summary.gait_quality, result)
    limits = _evidence_limitation_phrases(result)

    if limits:
        limit_text = ", and ".join(limits[:3])
        if "cannot be reliably assessed" in gait_clause:
            parts.append(f"{gait_clause} because {limit_text}")
        else:
            parts.append(f"{gait_clause}, although {limit_text}")
    else:
        parts.append(gait_clause)

    if summary.analysis_confidence.level in ("LOW", "INSUFFICIENT"):
        parts.append(
            "Interpretation confidence is limited by measurement evidence"
        )

    text = ". ".join(p.strip().rstrip(".") for p in parts if p.strip())
    return f"{text}." if text and not text.endswith(".") else text


def build_gait_analysis_summary(result: StabilityResult) -> GaitAnalysisSummary:
    """Construct the three-pillar gait analysis summary from a stability result."""
    from stablewalk.analysis.stability_validity import assess_stability_result_validity

    validity: StabilityResultValidity = (
        result.validity or assess_stability_result_validity(result)
    )

    movement = _group_score(
        result.metrics,
        label="Movement Stability",
        domain_keys=MOVEMENT_STABILITY_DOMAIN_KEYS,
    )
    gait_quality = _group_score(
        result.metrics,
        label="Gait Quality",
        domain_keys=GAIT_QUALITY_DOMAIN_KEYS,
    )
    analysis_confidence = _build_analysis_confidence(
        result,
        evidence_index=validity.evidence_index,
        validity_status=validity.status,
        active_domain_count=validity.active_domain_count,
    )

    summary = GaitAnalysisSummary(
        movement_stability=movement,
        gait_quality=gait_quality,
        analysis_confidence=analysis_confidence,
        explanation="",
        legacy_composite_score=result.score,
        legacy_classification=result.classification,
        validity_status=validity.status,
        comparable_score=validity.comparable_score,
    )
    summary.explanation = generate_gait_analysis_explanation(summary, result)
    return summary


def format_summary_display(summary: GaitAnalysisSummary) -> dict[str, str]:
    """Compact display strings for the Gait Analysis Summary panel."""

    def _score_text(group: DomainGroupScore) -> str:
        if group.score is None:
            return "—"
        return f"{group.score:.0f} / 100"

    conf = summary.analysis_confidence.level
    legacy_note = ""
    if summary.validity_status == "INSUFFICIENT_DATA":
        legacy_note = (
            f"Legacy composite (partial): {summary.legacy_composite_score:.0f} / 100 "
            "— not reliable for comparison"
        )
    elif summary.validity_status == "PROVISIONAL":
        legacy_note = (
            f"Legacy composite: {summary.legacy_composite_score:.0f} / 100 (provisional)"
        )

    return {
        "movement_stability": _score_text(summary.movement_stability),
        "gait_quality": _score_text(summary.gait_quality),
        "analysis_confidence": conf,
        "explanation": summary.explanation,
        "legacy_note": legacy_note,
        "partial_scores_note": (
            "* partial estimate — limited domain evidence"
            if (
                summary.gait_quality.completeness_pct < 35.0
                or summary.movement_stability.completeness_pct < 35.0
            )
            else ""
        ),
        "comparable": (
            f"Comparable score: {summary.comparable_score}"
            if summary.comparable_score
            else ""
        ),
    }


__all__ = [
    "AnalysisConfidenceLevel",
    "AnalysisConfidenceSummary",
    "DOMAIN_SEMANTICS_DOC",
    "GAIT_QUALITY_DOMAIN_KEYS",
    "GaitAnalysisSummary",
    "DomainGroupScore",
    "MOVEMENT_STABILITY_DOMAIN_KEYS",
    "build_gait_analysis_summary",
    "format_summary_display",
    "generate_gait_analysis_explanation",
]
