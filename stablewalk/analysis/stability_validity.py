"""
Scientific validity assessment for stability scores.

Prevents low-confidence partial estimates from being presented or compared as
authoritative biomechanical conclusions. Uses multi-factor evidence — no single
gate determines the outcome.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from stablewalk.analysis.stability_scoring import StabilityResult

ResultValidityStatus = Literal["VALID", "PROVISIONAL", "INSUFFICIENT_DATA"]
ComparableScoreLevel = Literal["YES", "LIMITED", "NO"]
ComparisonVerdict = Literal[
    "CLEAR_DIFFERENCE",
    "POSSIBLE_DIFFERENCE",
    "NO_RELIABLE_DIFFERENCE",
    "NOT_COMPARABLE",
]

VALIDITY_CONFIGURATION_DOC = """
Stability result validity thresholds (v1)

Derived from StableWalk demo-video distributions (abnormal / normal / athletic):
  abnormal: completeness 29%, usable cycles 0, 4/8 domains, evidence weight 0.29
  normal:   completeness 24%, usable cycles 1, 4/8 domains, evidence weight 0.24
  athletic: completeness 55%, usable cycles 2, 8/8 domains, evidence weight 0.55

Aligned with existing stability_scoring confidence badges:
  HIGH:     completeness >= 85% AND avg domain confidence >= 0.72
  MODERATE: completeness >= 55% AND avg domain confidence >= 0.45
  LOW:      otherwise

Decision uses a weighted evidence index (0–1) over six normalized factors:
  completeness, confidence badge, pose-valid frame %, usable gait cycles,
  active domain count, and confidence-weighted evidence mass.

Hard insufficient gates (any triggers INSUFFICIENT_DATA):
  • analysis completeness < 35%  (matches classification cutoff)
  • confidence-weighted evidence < 0.25
  • fewer than 2 active stability domains
  • zero usable gait cycles AND evidence < 0.40
  • pose-valid frame percentage < 50%

VALID requires evidence index >= 0.68 AND all of:
  • completeness >= 55%
  • confidence badge MODERATE or HIGH (or avg active confidence >= 0.45)
  • usable gait cycles >= 3  (gait_evidence NORMAL tier)
  • active domains >= 5
  • pose-valid frames >= 70%
  • evidence weight >= 0.45

PROVISIONAL: evidence index >= 0.32 and not insufficient.

Comparable score:
  YES      — VALID result with HIGH or MODERATE confidence
  LIMITED  — PROVISIONAL, or VALID with LOW confidence
  NO       — INSUFFICIENT_DATA

Comparison tolerance (score points):
  per_result = base + completeness_penalty + confidence_penalty + temporal_penalty
    base = 5.0
    completeness_penalty = (100 - completeness_pct) * 0.06
    confidence_penalty = (1 - confidence_factor) * 12
    temporal_penalty = max(0, min_valid_cycles - usable_cycles) * 4
  combined_tolerance = sqrt(tolerance_a² + tolerance_b²)

Comparison verdict:
  NOT_COMPARABLE           — either result INSUFFICIENT or comparable NO
  NO_RELIABLE_DIFFERENCE   — |delta| < combined_tolerance OR both low-confidence
  POSSIBLE_DIFFERENCE      — |delta| >= tolerance but at least one PROVISIONAL
  CLEAR_DIFFERENCE         — both VALID and |delta| >= tolerance
""".strip()


@dataclass(frozen=True)
class StabilityValidityThresholds:
    """
    Configurable multi-factor validity thresholds.

    Defaults calibrated against demo-video distributions and existing
    ``stability_scoring`` confidence cutoffs (55% / 85% completeness).
    """

    # Hard insufficient gates
    insufficient_completeness_pct: float = 35.0
    insufficient_evidence_weight: float = 0.25
    insufficient_min_active_domains: int = 2
    insufficient_pose_frame_pct: float = 50.0
    insufficient_zero_cycles_evidence: float = 0.40

    # VALID composite requirements
    valid_evidence_index: float = 0.68
    valid_completeness_pct: float = 55.0
    valid_usable_gait_cycles: int = 3
    valid_active_domains: int = 5
    valid_pose_frame_pct: float = 70.0
    valid_evidence_weight: float = 0.45
    valid_min_avg_confidence: float = 0.45

    # PROVISIONAL floor
    provisional_evidence_index: float = 0.32

    # Evidence-index factor weights (sum = 1.0)
    weight_completeness: float = 0.25
    weight_confidence: float = 0.20
    weight_pose: float = 0.10
    weight_temporal: float = 0.20
    weight_domains: float = 0.15
    weight_evidence_mass: float = 0.10

    # Reference scales for normalization
    completeness_reference_pct: float = 85.0
    temporal_reference_cycles: int = 3
    total_domain_count: int = 8

    # Comparison tolerance
    comparison_base_tolerance: float = 5.0
    comparison_completeness_coeff: float = 0.06
    comparison_confidence_coeff: float = 12.0
    comparison_temporal_coeff: float = 4.0


DEFAULT_STABILITY_VALIDITY_THRESHOLDS = StabilityValidityThresholds()


@dataclass
class StabilityResultValidity:
    """Whether an overall stability score is scientifically interpretable."""

    status: ResultValidityStatus
    comparable_score: ComparableScoreLevel
    evidence_index: float
    evidence_weight: float
    active_domain_count: int
    pose_valid_frame_pct: float
    avg_active_confidence: float
    partial_estimate: float | None
    reasons: list[str] = field(default_factory=list)
    factor_scores: dict[str, float] = field(default_factory=dict)

    @property
    def is_displayable_score(self) -> bool:
        """Primary UI may show score / 100 only for VALID or PROVISIONAL."""
        return self.status in ("VALID", "PROVISIONAL")

    @property
    def show_partial_estimate(self) -> bool:
        return (
            self.status == "INSUFFICIENT_DATA"
            and self.partial_estimate is not None
            and self.partial_estimate > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "comparable_score": self.comparable_score,
            "evidence_index": round(self.evidence_index, 3),
            "evidence_weight": round(self.evidence_weight, 3),
            "active_domain_count": self.active_domain_count,
            "pose_valid_frame_pct": round(self.pose_valid_frame_pct, 2),
            "avg_active_confidence": round(self.avg_active_confidence, 3),
            "partial_estimate": (
                None if self.partial_estimate is None
                else round(self.partial_estimate, 1)
            ),
            "reasons": list(self.reasons),
            "factor_scores": {k: round(v, 3) for k, v in self.factor_scores.items()},
        }


@dataclass
class StabilityComparisonResult:
    """Outcome of comparing two stability assessments."""

    verdict: ComparisonVerdict
    score_delta: float
    combined_tolerance: float
    tolerance_a: float
    tolerance_b: float
    comparable_a: ComparableScoreLevel
    comparable_b: ComparableScoreLevel
    validity_a: ResultValidityStatus
    validity_b: ResultValidityStatus
    explanation: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score_delta": round(self.score_delta, 2),
            "combined_tolerance": round(self.combined_tolerance, 2),
            "tolerance_a": round(self.tolerance_a, 2),
            "tolerance_b": round(self.tolerance_b, 2),
            "comparable_a": self.comparable_a,
            "comparable_b": self.comparable_b,
            "validity_a": self.validity_a,
            "validity_b": self.validity_b,
            "explanation": self.explanation,
            "warnings": list(self.warnings),
        }


def _confidence_factor(badge: str) -> float:
    return {
        "HIGH CONFIDENCE": 1.0,
        "MODERATE CONFIDENCE": 0.65,
        "LOW CONFIDENCE": 0.25,
    }.get(badge, 0.25)


def _badge_meets_moderate(badge: str) -> bool:
    return badge in ("HIGH CONFIDENCE", "MODERATE CONFIDENCE")


def _active_domains(result: StabilityResult) -> list[Any]:
    return [
        m for m in result.metrics
        if m.availability != "UNAVAILABLE" and m.score is not None
    ]


def _evidence_weight(result: StabilityResult) -> float:
    total_weight = sum(m.weight for m in result.metrics)
    if total_weight <= 0:
        return 0.0
    active_mass = sum(
        c.weight * c.confidence
        for c in result.contributions
        if c.availability != "UNAVAILABLE" and c.score is not None
    )
    return active_mass / total_weight


def _avg_active_confidence(result: StabilityResult) -> float:
    confs = [
        c.confidence
        for c in result.contributions
        if c.availability != "UNAVAILABLE" and c.score is not None
    ]
    return statistics.mean(confs) if confs else 0.0


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


def _factor_scores(
    result: StabilityResult,
    *,
    pose_valid_pct: float,
    active_count: int,
    evidence_weight: float,
    thresholds: StabilityValidityThresholds,
) -> dict[str, float]:
    conf_factor = _confidence_factor(result.confidence_badge)
    return {
        "completeness": min(
            result.completeness_pct / thresholds.completeness_reference_pct,
            1.0,
        ),
        "confidence": conf_factor,
        "pose": min(pose_valid_pct / 100.0, 1.0),
        "temporal": min(
            result.usable_gait_cycles / thresholds.temporal_reference_cycles,
            1.0,
        ),
        "domains": active_count / thresholds.total_domain_count,
        "evidence_mass": min(evidence_weight, 1.0),
    }


def _evidence_index(
    factors: dict[str, float],
    thresholds: StabilityValidityThresholds,
) -> float:
    return (
        thresholds.weight_completeness * factors["completeness"]
        + thresholds.weight_confidence * factors["confidence"]
        + thresholds.weight_pose * factors["pose"]
        + thresholds.weight_temporal * factors["temporal"]
        + thresholds.weight_domains * factors["domains"]
        + thresholds.weight_evidence_mass * factors["evidence_mass"]
    )


def _hard_insufficient(
    result: StabilityResult,
    *,
    pose_valid_pct: float,
    active_count: int,
    evidence_weight: float,
    thresholds: StabilityValidityThresholds,
) -> list[str]:
    reasons: list[str] = []
    if result.completeness_pct < thresholds.insufficient_completeness_pct:
        reasons.append(
            f"Analysis completeness {result.completeness_pct:.0f}% is below "
            f"{thresholds.insufficient_completeness_pct:.0f}%."
        )
    if evidence_weight < thresholds.insufficient_evidence_weight:
        reasons.append(
            f"Confidence-weighted evidence ({evidence_weight:.0%}) is below "
            f"{thresholds.insufficient_evidence_weight:.0%}."
        )
    if active_count < thresholds.insufficient_min_active_domains:
        reasons.append(
            f"Only {active_count} stability domain(s) contributed — need at least "
            f"{thresholds.insufficient_min_active_domains}."
        )
    if (
        result.usable_gait_cycles == 0
        and evidence_weight < thresholds.insufficient_zero_cycles_evidence
    ):
        reasons.append(
            "No usable gait cycles and limited cross-domain evidence for a "
            "temporally grounded score."
        )
    if pose_valid_pct < thresholds.insufficient_pose_frame_pct:
        reasons.append(
            f"Pose-valid frames ({pose_valid_pct:.0f}%) are below "
            f"{thresholds.insufficient_pose_frame_pct:.0f}%."
        )
    return reasons


def _meets_valid_requirements(
    result: StabilityResult,
    *,
    pose_valid_pct: float,
    active_count: int,
    evidence_weight: float,
    avg_conf: float,
    evidence_index: float,
    thresholds: StabilityValidityThresholds,
) -> tuple[bool, list[str]]:
    checks: list[tuple[bool, str]] = [
        (
            evidence_index >= thresholds.valid_evidence_index,
            f"Evidence index {evidence_index:.2f} < {thresholds.valid_evidence_index:.2f}",
        ),
        (
            result.completeness_pct >= thresholds.valid_completeness_pct,
            f"Completeness {result.completeness_pct:.0f}% < "
            f"{thresholds.valid_completeness_pct:.0f}%",
        ),
        (
            _badge_meets_moderate(result.confidence_badge)
            or avg_conf >= thresholds.valid_min_avg_confidence,
            "Confidence below MODERATE and average domain confidence insufficient",
        ),
        (
            result.usable_gait_cycles >= thresholds.valid_usable_gait_cycles,
            f"Usable gait cycles ({result.usable_gait_cycles}) < "
            f"{thresholds.valid_usable_gait_cycles}",
        ),
        (
            active_count >= thresholds.valid_active_domains,
            f"Active domains ({active_count}) < {thresholds.valid_active_domains}",
        ),
        (
            pose_valid_pct >= thresholds.valid_pose_frame_pct,
            f"Pose-valid frames ({pose_valid_pct:.0f}%) < "
            f"{thresholds.valid_pose_frame_pct:.0f}%",
        ),
        (
            evidence_weight >= thresholds.valid_evidence_weight,
            f"Evidence weight ({evidence_weight:.0%}) < "
            f"{thresholds.valid_evidence_weight:.0%}",
        ),
    ]
    failures = [msg for ok, msg in checks if not ok]
    return len(failures) == 0, failures


def assess_stability_result_validity(
    result: StabilityResult,
    *,
    thresholds: StabilityValidityThresholds | None = None,
    pose_valid_frame_pct: float | None = None,
) -> StabilityResultValidity:
    """
    Determine whether the overall numerical stability score is scientifically
    interpretable for display and comparison.
    """
    cfg = thresholds or DEFAULT_STABILITY_VALIDITY_THRESHOLDS
    pose_pct = (
        pose_valid_frame_pct
        if pose_valid_frame_pct is not None
        else _pose_valid_frame_pct(result)
    )
    active = _active_domains(result)
    active_count = len(active)
    evidence_weight = _evidence_weight(result)
    avg_conf = _avg_active_confidence(result)
    factors = _factor_scores(
        result,
        pose_valid_pct=pose_pct,
        active_count=active_count,
        evidence_weight=evidence_weight,
        thresholds=cfg,
    )
    index = _evidence_index(factors, cfg)

    hard_reasons = _hard_insufficient(
        result,
        pose_valid_pct=pose_pct,
        active_count=active_count,
        evidence_weight=evidence_weight,
        thresholds=cfg,
    )

    if hard_reasons:
        status: ResultValidityStatus = "INSUFFICIENT_DATA"
        reasons = hard_reasons
    else:
        valid_ok, valid_failures = _meets_valid_requirements(
            result,
            pose_valid_pct=pose_pct,
            active_count=active_count,
            evidence_weight=evidence_weight,
            avg_conf=avg_conf,
            evidence_index=index,
            thresholds=cfg,
        )
        if valid_ok:
            status = "VALID"
            reasons = [
                "Sufficient analysis completeness, confidence, domain coverage, "
                "and temporal evidence for an interpretable overall score."
            ]
        elif index >= cfg.provisional_evidence_index:
            status = "PROVISIONAL"
            reasons = [
                "Meaningful biomechanical evidence exists, but temporal or "
                "reconstruction confidence is limited."
            ]
            if valid_failures:
                reasons.extend(valid_failures)
        else:
            status = "INSUFFICIENT_DATA"
            reasons = [
                "Composite evidence index is too low for a scientifically "
                "interpretable overall score."
            ]
            if valid_failures:
                reasons.extend(valid_failures)

    comparable = _comparable_level(status, result.confidence_badge)
    partial = result.score if result.score > 0 else None

    return StabilityResultValidity(
        status=status,
        comparable_score=comparable,
        evidence_index=index,
        evidence_weight=evidence_weight,
        active_domain_count=active_count,
        pose_valid_frame_pct=pose_pct,
        avg_active_confidence=avg_conf,
        partial_estimate=partial,
        reasons=reasons,
        factor_scores=factors,
    )


def _comparable_level(
    status: ResultValidityStatus,
    confidence_badge: str,
) -> ComparableScoreLevel:
    if status == "INSUFFICIENT_DATA":
        return "NO"
    if status == "PROVISIONAL":
        return "LIMITED"
    if confidence_badge == "LOW CONFIDENCE":
        return "LIMITED"
    return "YES"


def score_comparison_tolerance(
    result: StabilityResult,
    validity: StabilityResultValidity | None = None,
    *,
    thresholds: StabilityValidityThresholds | None = None,
) -> float:
    """
    Estimate score uncertainty in points for pairwise comparison.

    Uses metric confidence (badge + avg domain confidence), domain availability
    (completeness), and temporal evidence (usable gait cycles).
    """
    cfg = thresholds or DEFAULT_STABILITY_VALIDITY_THRESHOLDS
    v = validity or assess_stability_result_validity(result, thresholds=cfg)
    conf = max(
        _confidence_factor(result.confidence_badge),
        min(v.avg_active_confidence, 1.0),
    )
    completeness_penalty = (100.0 - result.completeness_pct) * cfg.comparison_completeness_coeff
    confidence_penalty = (1.0 - conf) * cfg.comparison_confidence_coeff
    temporal_penalty = (
        max(0, cfg.valid_usable_gait_cycles - result.usable_gait_cycles)
        * cfg.comparison_temporal_coeff
    )
    return (
        cfg.comparison_base_tolerance
        + completeness_penalty
        + confidence_penalty
        + temporal_penalty
    )


def compare_stability_results(
    result_a: StabilityResult,
    result_b: StabilityResult,
    *,
    label_a: str = "A",
    label_b: str = "B",
    thresholds: StabilityValidityThresholds | None = None,
) -> StabilityComparisonResult:
    """
    Compare two stability results with explicit reliability gating.

    Returns CLEAR_DIFFERENCE, POSSIBLE_DIFFERENCE, NO_RELIABLE_DIFFERENCE, or
    NOT_COMPARABLE based on validity status and combined score tolerance.
    """
    cfg = thresholds or DEFAULT_STABILITY_VALIDITY_THRESHOLDS
    validity_a = assess_stability_result_validity(result_a, thresholds=cfg)
    validity_b = assess_stability_result_validity(result_b, thresholds=cfg)

    delta = result_a.score - result_b.score
    tol_a = score_comparison_tolerance(result_a, validity_a, thresholds=cfg)
    tol_b = score_comparison_tolerance(result_b, validity_b, thresholds=cfg)
    combined_tol = math.sqrt(tol_a * tol_a + tol_b * tol_b)

    warnings: list[str] = []
    abs_delta = abs(delta)

    if (
        validity_a.status == "INSUFFICIENT_DATA"
        or validity_b.status == "INSUFFICIENT_DATA"
        or validity_a.comparable_score == "NO"
        or validity_b.comparable_score == "NO"
    ):
        verdict: ComparisonVerdict = "NOT_COMPARABLE"
        explanation = (
            f"{label_a} ({result_a.score:.0f}) versus {label_b} ({result_b.score:.0f}) "
            f"is not a meaningful ranking because at least one analysis lacks "
            f"sufficient evidence "
            f"({label_a}: {validity_a.status}, {label_a} comparable: "
            f"{validity_a.comparable_score}; "
            f"{label_b}: {validity_b.status}, {label_b} comparable: "
            f"{validity_b.comparable_score})."
        )
    elif abs_delta < combined_tol:
        verdict = "NO_RELIABLE_DIFFERENCE"
        explanation = (
            f"{label_a} {result_a.score:.0f} versus {label_b} {result_b.score:.0f} "
            f"is not a meaningful ranking because both analyses are "
            f"{'low-confidence' if validity_a.status != 'VALID' or validity_b.status != 'VALID' else 'limited-confidence'} "
            f"and the score difference ({abs_delta:.1f} points) is smaller than "
            f"the estimated evidence uncertainty (±{combined_tol:.1f} points)."
        )
    elif validity_a.status == "VALID" and validity_b.status == "VALID":
        verdict = "CLEAR_DIFFERENCE"
        direction = label_a if delta > 0 else label_b
        explanation = (
            f"{direction} scores higher ({abs_delta:.1f} points above tolerance "
            f"±{combined_tol:.1f}) with VALID evidence on both sides."
        )
    else:
        verdict = "POSSIBLE_DIFFERENCE"
        direction = label_a if delta > 0 else label_b
        explanation = (
            f"{direction} may score higher ({abs_delta:.1f} points), but at least "
            f"one result is PROVISIONAL — treat as indicative, not conclusive "
            f"(tolerance ±{combined_tol:.1f})."
        )
        warnings.append(
            "One or both analyses are PROVISIONAL; domain-level review recommended."
        )

    return StabilityComparisonResult(
        verdict=verdict,
        score_delta=delta,
        combined_tolerance=combined_tol,
        tolerance_a=tol_a,
        tolerance_b=tol_b,
        comparable_a=validity_a.comparable_score,
        comparable_b=validity_b.comparable_score,
        validity_a=validity_a.status,
        validity_b=validity_b.status,
        explanation=explanation,
        warnings=warnings,
    )


def format_validity_display(validity: StabilityResultValidity) -> dict[str, str]:
    """UI presentation hints for the Analysis Summary stability panel."""
    if validity.status == "INSUFFICIENT_DATA":
        return {
            "title": "Stability Assessment",
            "primary": "INSUFFICIENT DATA",
            "secondary": (
                f"Partial estimate: {validity.partial_estimate:.0f} / 100"
                if validity.show_partial_estimate
                else ""
            ),
            "secondary_label": "NOT RELIABLE FOR COMPARISON",
            "score_text": "",
            "badge": validity.status.replace("_", " "),
            "comparable": f"Comparable score: {validity.comparable_score}",
        }
    if validity.status == "PROVISIONAL":
        score = validity.partial_estimate or 0.0
        return {
            "title": "Stability",
            "primary": f"{score:.0f} / 100",
            "secondary": "PROVISIONAL RESULT",
            "secondary_label": "",
            "score_text": f"{score:.0f}",
            "badge": "PROVISIONAL RESULT",
            "comparable": f"Comparable score: {validity.comparable_score}",
        }
    score = validity.partial_estimate or 0.0
    return {
        "title": "Stability",
        "primary": f"{score:.0f} / 100",
        "secondary": "",
        "secondary_label": "",
        "score_text": f"{score:.0f}",
        "badge": "",
        "comparable": f"Comparable score: {validity.comparable_score}",
    }


__all__ = [
    "ComparableScoreLevel",
    "ComparisonVerdict",
    "DEFAULT_STABILITY_VALIDITY_THRESHOLDS",
    "ResultValidityStatus",
    "StabilityComparisonResult",
    "StabilityResultValidity",
    "StabilityValidityThresholds",
    "VALIDITY_CONFIGURATION_DOC",
    "assess_stability_result_validity",
    "compare_stability_results",
    "format_validity_display",
    "score_comparison_tolerance",
]
