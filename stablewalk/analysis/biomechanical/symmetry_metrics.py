"""Left–right symmetry metrics for biomechanical gait analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from stablewalk.analysis.biomechanical.types import MetricWithConfidence
from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult, symmetry_ratio
from stablewalk.analysis.gait_feature_analysis import GaitFeatureAnalysisResult, symmetry_index


@dataclass
class SymmetryAnalysis:
    step_length_symmetry: MetricWithConfidence | None = None
    stride_length_symmetry: MetricWithConfidence | None = None
    stance_symmetry: MetricWithConfidence | None = None
    swing_symmetry: MetricWithConfidence | None = None
    cadence_consistency: MetricWithConfidence | None = None
    knee_rom_symmetry: MetricWithConfidence | None = None
    hip_rom_symmetry: MetricWithConfidence | None = None
    ankle_rom_symmetry: MetricWithConfidence | None = None
    overall_symmetry_pct: MetricWithConfidence | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in (
            "step_length_symmetry",
            "stride_length_symmetry",
            "stance_symmetry",
            "swing_symmetry",
            "cadence_consistency",
            "knee_rom_symmetry",
            "hip_rom_symmetry",
            "ankle_rom_symmetry",
            "overall_symmetry_pct",
        ):
            m = getattr(self, key)
            if m is not None:
                out[key] = m.to_dict()
        out["warnings"] = list(self.warnings)
        return out


def _metric(
    value: float | None,
    *,
    confidence: float,
    unit: str = "ratio",
    note: str = "",
) -> MetricWithConfidence | None:
    if value is None or not math.isfinite(value):
        return None
    if unit == "ratio" and not 0.0 <= value <= 1.0:
        return None
    if unit == "%" and not 0.0 <= value <= 100.0:
        return None
    return MetricWithConfidence(
        value=round(value, 4),
        unit=unit,
        kind="derived",
        confidence=confidence,
        note=note,
    )


def _cadence_consistency(cycles: GaitCycleAnalysisResult | None) -> float | None:
    if cycles is None or not cycles.metrics.metrics_reliable:
        return None
    cv = cycles.metrics.gait_cycle_consistency
    if cv is not None:
        return cv
    durations = [c.duration_s for c in cycles.cycles if c.duration_s > 1e-6]
    if len(durations) < 2:
        return None
    mean_d = sum(durations) / len(durations)
    if mean_d <= 1e-6:
        return None
    variance = sum((d - mean_d) ** 2 for d in durations) / len(durations)
    cv_raw = (variance ** 0.5) / mean_d
    return max(0.0, min(1.0, 1.0 - cv_raw))


def analyze_symmetry(
    cycles: GaitCycleAnalysisResult | None,
    features: GaitFeatureAnalysisResult | None,
    contact: FootContactAnalysisResult | None,
    *,
    knee_rom_left: float | None = None,
    knee_rom_right: float | None = None,
    hip_rom_left: float | None = None,
    hip_rom_right: float | None = None,
    ankle_rom_left: float | None = None,
    ankle_rom_right: float | None = None,
) -> SymmetryAnalysis:
    """Compute bilateral symmetry metrics with confidence."""
    result = SymmetryAnalysis()
    reliable_cycles = bool(cycles and cycles.metrics.metrics_reliable)
    conf_base = cycles.metrics.contact_confidence if reliable_cycles and cycles else 0.0
    if not reliable_cycles:
        reason = (
            cycles.metrics.reliability_reason
            if cycles is not None
            else "No gait-cycle analysis was available."
        )
        result.warnings.append(f"Gait symmetry unavailable: {reason}")
    m = cycles.metrics if cycles else None
    nf = features.features if features else None

    if m and reliable_cycles:
        result.stance_symmetry = _metric(
            symmetry_ratio(m.left_stance_time_s, m.right_stance_time_s),
            confidence=conf_base,
            note="min(L,R) / max(L,R) stance duration",
        )
        result.swing_symmetry = _metric(
            symmetry_ratio(m.left_swing_time_s, m.right_swing_time_s),
            confidence=conf_base,
            note="min(L,R) / max(L,R) swing duration",
        )

    result.cadence_consistency = _metric(
        _cadence_consistency(cycles),
        confidence=conf_base * 0.9,
        note="1 − coefficient of variation of gait cycle duration",
    )

    if nf and reliable_cycles:
        result.step_length_symmetry = _metric(
            nf.step_length_symmetry,
            confidence=conf_base * 0.85,
            note="2×min(L,R) / (L+R) step length from foot landmarks",
        )
        result.stride_length_symmetry = _metric(
            nf.stride_length_symmetry,
            confidence=conf_base * 0.85,
            note="2×min(L,R) / (L+R) stride length from foot landmarks",
        )

    result.knee_rom_symmetry = _metric(
        symmetry_index(knee_rom_left, knee_rom_right) if reliable_cycles else None,
        confidence=conf_base * 0.8,
        note="Knee ROM L/R symmetry index",
    )
    result.hip_rom_symmetry = _metric(
        symmetry_index(hip_rom_left, hip_rom_right) if reliable_cycles else None,
        confidence=conf_base * 0.75,
        note="Hip ROM L/R symmetry index",
    )
    result.ankle_rom_symmetry = _metric(
        symmetry_index(ankle_rom_left, ankle_rom_right) if reliable_cycles else None,
        confidence=conf_base * 0.75,
        note="Ankle ROM L/R symmetry index",
    )

    parts: list[float] = []
    weights: list[float] = []
    for sym, w in (
        (result.stride_length_symmetry, 1.4),
        (result.step_length_symmetry, 1.3),
        (result.stance_symmetry, 1.2),
        (result.swing_symmetry, 1.2),
        (result.cadence_consistency, 1.1),
        (result.knee_rom_symmetry, 1.0),
        (result.hip_rom_symmetry, 0.9),
        (result.ankle_rom_symmetry, 0.9),
    ):
        if sym is not None and sym.value is not None:
            parts.append(sym.value * w)
            weights.append(w)

    overall = sum(parts) / sum(weights) if len(parts) >= 2 else None
    if reliable_cycles and overall is None:
        result.warnings.append(
            "Overall symmetry unavailable: fewer than two valid bilateral components."
        )
    result.overall_symmetry_pct = _metric(
        overall * 100.0 if overall is not None else None,
        confidence=conf_base,
        unit="%",
        note="Weighted mean of spatiotemporal and ROM symmetry indices × 100",
    )
    return result


__all__ = ["SymmetryAnalysis", "analyze_symmetry"]
