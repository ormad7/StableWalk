"""Left–right symmetry metrics for biomechanical gait analysis."""

from __future__ import annotations

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
        return out


def _metric(
    value: float | None,
    *,
    confidence: float,
    unit: str = "ratio",
    note: str = "",
) -> MetricWithConfidence | None:
    if value is None:
        return None
    return MetricWithConfidence(
        value=round(value, 4),
        unit=unit,
        kind="derived",
        confidence=confidence,
        note=note,
    )


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
    conf_base = 0.5
    if cycles is not None:
        conf_base = max(conf_base, cycles.metrics.contact_confidence)
    if contact is not None:
        conf_base = max(conf_base, contact.metrics.contact_confidence)

    result = SymmetryAnalysis()
    m = cycles.metrics if cycles else None
    nf = features.features if features else None

    if m:
        result.stance_symmetry = _metric(
            symmetry_ratio(m.left_stance_time_s, m.right_stance_time_s),
            confidence=conf_base,
            note="Temporal stance duration L/R ratio",
        )
        result.swing_symmetry = _metric(
            symmetry_ratio(m.left_swing_time_s, m.right_swing_time_s),
            confidence=conf_base,
            note="Temporal swing duration L/R ratio",
        )
        cv = m.gait_cycle_consistency
        result.cadence_consistency = _metric(
            cv,
            confidence=conf_base * 0.9,
            note="1 − gait cycle duration CV",
        )

    if nf:
        result.step_length_symmetry = _metric(
            nf.step_length_symmetry,
            confidence=conf_base * 0.85,
            note="Step length symmetry index",
        )
        result.stride_length_symmetry = _metric(
            nf.stride_length_symmetry,
            confidence=conf_base * 0.85,
            note="Stride length symmetry index",
        )

    result.knee_rom_symmetry = _metric(
        symmetry_index(knee_rom_left, knee_rom_right),
        confidence=conf_base * 0.8,
        note="Knee ROM L/R symmetry",
    )
    result.hip_rom_symmetry = _metric(
        symmetry_index(hip_rom_left, hip_rom_right),
        confidence=conf_base * 0.75,
        note="Hip ROM L/R symmetry",
    )
    result.ankle_rom_symmetry = _metric(
        symmetry_index(ankle_rom_left, ankle_rom_right),
        confidence=conf_base * 0.75,
        note="Ankle ROM L/R symmetry",
    )

    parts: list[float] = []
    weights: list[float] = []
    for sym, w in (
        (result.step_length_symmetry, 1.2),
        (result.stride_length_symmetry, 1.0),
        (result.stance_symmetry, 1.3),
        (result.swing_symmetry, 1.1),
        (result.cadence_consistency, 1.0),
        (result.knee_rom_symmetry, 0.9),
        (result.hip_rom_symmetry, 0.8),
        (result.ankle_rom_symmetry, 0.8),
    ):
        if sym is not None and sym.value is not None:
            parts.append(sym.value * w)
            weights.append(w)

    overall = sum(parts) / sum(weights) if weights else None
    result.overall_symmetry_pct = _metric(
        overall * 100.0 if overall is not None else None,
        confidence=conf_base,
        unit="%",
        note="Weighted mean symmetry index × 100",
    )
    return result


__all__ = ["SymmetryAnalysis", "analyze_symmetry"]
