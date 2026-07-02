"""
Numeric gait feature vectors for anomaly detection, comparison, and ML.

Maps pose-derived metrics to a fixed-length vector analogous to
spatiotemporal parameters exported by clinical gait labs (cadence, stride,
symmetry indices).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from stablewalk.analysis.report import GaitAnalysisReport, analyze_gait
from stablewalk.models.pose_data import PoseSequence

# Order fixed for ML models and CSV export
FEATURE_NAMES: tuple[str, ...] = (
    "symmetry_score",
    "cadence_hz",
    "cadence_steps_per_min",
    "stride_length_normalized",
    "stride_length_m",
    "step_time_variability",
    "stride_length_variability",
    "stance_left_fraction",
    "stance_right_fraction",
    "stance_symmetry_ratio",
    "com_lateral_std",
    "com_vertical_range",
    "com_path_length",
    "grf_symmetry",
    "knee_symmetry_deg",
    "hip_symmetry_deg",
    "stability_score",
    "double_peak_fraction",
)


@dataclass
class GaitFeatureVector:
    """Single-walk feature snapshot (NaN where unavailable)."""

    source_id: str = ""
    values: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    def to_array(self) -> list[float]:
        import math

        out: list[float] = []
        for name in FEATURE_NAMES:
            v = self.values.get(name, float("nan"))
            out.append(float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else float("nan"))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "values": dict(self.values), "flags": list(self.flags)}


def extract_gait_features(
    sequence: PoseSequence,
    report: GaitAnalysisReport | None = None,
    *,
    source_id: str = "",
    body_mass_kg: float = 70.0,
) -> GaitFeatureVector:
    """
    Build a feature vector from an enriched pose sequence.

    If ``report`` is omitted, runs ``analyze_gait`` internally.
    """
    if report is None:
        report, _ = analyze_gait(sequence, body_mass_kg=body_mass_kg)

    m = report.metrics
    s = report.stability
    sm = s.metrics

    cadence_hz = None
    if m.cadence_steps_per_min is not None:
        cadence_hz = m.cadence_steps_per_min / 60.0

    grf_sym = sm.grf_symmetry if sm else None
    dp_frac = report.grf.double_peak_fraction if report.grf else None

    values: dict[str, float] = {}
    _set(values, "symmetry_score", sm.symmetry_score if sm else s.symmetry_score)
    _set(values, "cadence_hz", cadence_hz)
    _set(values, "cadence_steps_per_min", m.cadence_steps_per_min)
    _set(values, "stride_length_normalized", m.stride_length_normalized)
    _set(values, "stride_length_m", m.stride_length_m)
    _set(values, "step_time_variability", sm.step_time_variability if sm else None)
    _set(values, "stride_length_variability", sm.stride_length_variability if sm else None)
    _set(values, "stance_left_fraction", sm.stance_left_fraction if sm else m.contact.left_stance_fraction)
    _set(values, "stance_right_fraction", sm.stance_right_fraction if sm else m.contact.right_stance_fraction)
    _set(values, "stance_symmetry_ratio", m.contact.stance_symmetry_ratio)
    _set(values, "com_lateral_std", sm.com_lateral_std if sm else s.com_lateral_std)
    _set(values, "com_vertical_range", sm.com_vertical_range if sm else s.com_vertical_range)
    _set(values, "com_path_length", sm.com_path_length if sm else 0.0)
    _set(values, "grf_symmetry", grf_sym)
    _set(values, "knee_symmetry_deg", sm.knee_symmetry_deg if sm else s.knee_symmetry_deg)
    _set(values, "hip_symmetry_deg", sm.hip_symmetry_deg if sm else s.hip_symmetry_deg)
    _set(values, "stability_score", s.score)
    _set(values, "double_peak_fraction", dp_frac)

    return GaitFeatureVector(
        source_id=source_id or sequence.source_video or "walk",
        values=values,
        flags=list(s.abnormal_patterns),
    )


def _set(d: dict[str, float], key: str, val: float | None) -> None:
    if val is not None:
        d[key] = float(val)
