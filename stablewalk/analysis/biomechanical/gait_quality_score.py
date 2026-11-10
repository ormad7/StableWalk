"""Interpretable Gait Quality Score (0–100)."""

from __future__ import annotations

from stablewalk.analysis.biomechanical.advanced_gait_metrics import AdvancedGaitMetrics
from stablewalk.analysis.biomechanical.stability_margin import StabilityMarginAnalysis
from stablewalk.analysis.biomechanical.symmetry_metrics import SymmetryAnalysis
from stablewalk.analysis.biomechanical.types import GaitQualityScore
from stablewalk.analysis.biomech_stability import StabilityResult
from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult
from stablewalk.analysis.quality_layers import (
    GAIT_QUALITY_FORMULA,
    GAIT_QUALITY_KIND,
    GAIT_QUALITY_LIMITATIONS,
    GAIT_QUALITY_WEIGHTS,
    gait_quality_confidence,
    gait_quality_status,
)


def _clamp_score(x: float) -> float:
    return max(0.0, min(100.0, x))


def _norm_symmetry(sym: SymmetryAnalysis) -> float | None:
    if sym.overall_symmetry_pct and sym.overall_symmetry_pct.value is not None:
        return min(100.0, sym.overall_symmetry_pct.value)
    return None


def _movement_stability_score(stability: StabilityResult) -> float | None:
    """Prefer Movement Stability domains — avoid the legacy 8-domain composite."""
    summary = getattr(stability, "gait_summary", None)
    if summary is not None and summary.movement_stability.score is not None:
        return float(summary.movement_stability.score)
    return None


def _joint_smoothness_score(stability: StabilityResult) -> float | None:
    metric = stability.metric("joint_smoothness")
    if metric is None or metric.score is None:
        return None
    return float(metric.score)


def compute_gait_quality_score(
    *,
    stability: StabilityResult | None = None,
    stability_margin: StabilityMarginAnalysis | None = None,
    symmetry: SymmetryAnalysis | None = None,
    gait_metrics: AdvancedGaitMetrics | None = None,
    cycles: GaitCycleAnalysisResult | None = None,
    contact: FootContactAnalysisResult | None = None,
) -> GaitQualityScore:
    """Heuristic 0–100 session descriptor — not a validated gait-quality instrument.

    Weighted mean of available pose-derived components (stability proxy,
    symmetry, cadence/cycle consistency, contact score, joint smoothness).
    Unreportable physical COM–BoS / force-plate terms stay omitted.
    Distinct from Overview "Gait Coordination", which averages a different set.
    """
    del gait_metrics  # Reserved for future cadence-derived components.
    components: dict[str, float] = {}
    weights = dict(GAIT_QUALITY_WEIGHTS)

    if stability:
        move = _movement_stability_score(stability)
        if move is not None:
            components["stability"] = move
        else:
            components["stability"] = float(stability.score)
        smooth = _joint_smoothness_score(stability)
        if smooth is not None:
            components["motion_quality"] = smooth

    if (
        stability_margin
        and stability_margin.stable_pct is not None
        and getattr(stability_margin, "reportable", True)
    ):
        components["stability_margin"] = float(stability_margin.stable_pct)

    sym_score = _norm_symmetry(symmetry) if symmetry else None
    if sym_score is not None:
        components["symmetry"] = sym_score

    if symmetry and symmetry.cadence_consistency and symmetry.cadence_consistency.value:
        components["cadence_consistency"] = symmetry.cadence_consistency.value * 100.0

    if cycles and cycles.metrics.gait_cycle_consistency is not None:
        components["cycle_consistency"] = cycles.metrics.gait_cycle_consistency * 100.0

    if contact:
        components["contact_reliability"] = contact.metrics.contact_confidence * 100.0

    active = {k: v for k, v in components.items() if v is not None}
    if not active:
        # Do not invent a numeric 0/100 — unavailable must stay unavailable.
        return GaitQualityScore(
            score=None,
            confidence=0.0,
            dominant_factors=["Insufficient data"],
            explanation=(
                "Heuristic gait-quality descriptor could not be scored — "
                "insufficient pose or cycle data."
            ),
            components={},
            available=False,
            kind=GAIT_QUALITY_KIND,
            status="unavailable",
            formula=GAIT_QUALITY_FORMULA,
            limitations=GAIT_QUALITY_LIMITATIONS,
        )

    total_w = sum(weights[k] for k in active)
    score = sum(active[k] * weights[k] for k in active) / total_w

    conf_parts: list[float] = []
    if contact is not None:
        conf_parts.append(float(contact.metrics.contact_confidence))
    if cycles is not None:
        conf_parts.append(float(cycles.metrics.contact_confidence))
    if stability is not None:
        conf_parts.append(float(stability.completeness_pct) / 100.0)
    confidence = gait_quality_confidence(n_active=len(active), input_confidences=conf_parts)
    status = gait_quality_status(len(active))

    ranked = sorted(active.items(), key=lambda kv: kv[1])
    weak = [k for k, v in ranked if v < 55.0]
    strong = [k for k, v in sorted(active.items(), key=lambda kv: -kv[1])[:2]]

    factors: list[str] = []
    if weak:
        factors.extend(f"low {k.replace('_', ' ')}" for k in weak[:3])
    if strong:
        factors.extend(f"strong {k.replace('_', ' ')}" for k in strong[:2])

    explanation_parts = [
        f"Heuristic gait-quality descriptor {_clamp_score(score):.0f}/100 "
        f"({len(active)}/{len(weights)} components; {status}). "
        "Not a validated gait-quality instrument."
    ]
    if weak:
        explanation_parts.append(
            "Primary limitations: " + ", ".join(weak[:3]).replace("_", " ") + "."
        )
    if sym_score is not None and sym_score < 65:
        explanation_parts.append(
            "Overall symmetry is a weighted composite of available L/R ratios — "
            "not a clinically validated symmetry diagnosis."
        )
    elif sym_score is not None:
        explanation_parts.append(
            f"Overall symmetry {sym_score:.0f}% is a weighted composite of "
            "spatiotemporal and ROM ratios, not a clinical symmetry grade."
        )

    return GaitQualityScore(
        score=_clamp_score(score),
        confidence=confidence,
        dominant_factors=factors[:5] or ["balanced profile"],
        explanation=" ".join(explanation_parts),
        components=active,
        available=True,
        kind=GAIT_QUALITY_KIND,
        status=status,
        formula=GAIT_QUALITY_FORMULA,
        limitations=GAIT_QUALITY_LIMITATIONS,
    )


__all__ = ["compute_gait_quality_score"]
