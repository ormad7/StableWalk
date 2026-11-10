"""Canonical metric definitions shared by Overview, Biomechanics, Results, Compare, exports.

Thresholds here are **pose-derived heuristics** unless marked otherwise.
They must not be presented as clinical cut-offs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SourceKind = Literal["measured", "estimated", "derived", "opensim"]
ThresholdKind = Literal["engine", "heuristic", "literature", "none"]


@dataclass(frozen=True, slots=True)
class BandThresholds:
    """Inclusive lower bounds for normal / borderline bands (abnormal below borderline)."""

    normal_min: float | None = None
    borderline_min: float | None = None
    # Optional upper bounds (e.g. cadence)
    normal_max: float | None = None
    borderline_max: float | None = None
    higher_is_better: bool = True


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """One displayed metric contract."""

    key: str
    display_name: str
    unit: str
    source: SourceKind
    threshold_source: ThresholdKind
    threshold_note: str
    user_explanation: str
    technical_explanation: str
    bands: BandThresholds | None = None
    # Analysis writer (module.function) — single source of numeric truth.
    analysis_ref: str = ""
    # Distinct from another metric that shares a similar UI name.
    aliases: tuple[str, ...] = ()


# Engine margin thresholds (analysis/biomechanical/stability_margin.py).
MARGIN_STABLE_M = 0.04
MARGIN_REDUCED_M = 0.0
STABLE_PCT_VALID_FRAME_RATIO = 0.50
MIN_FRAME_CONFIDENCE = 0.35

# Heuristic session bands (shared Overview + Results/Biomech KPI colors).
GQ_NORMAL_MIN = 60.0
GQ_BORDERLINE_MIN = 45.0
STABLE_PCT_NORMAL_MIN = 70.0
STABLE_PCT_BORDERLINE_MIN = 55.0
SYMMETRY_NORMAL_MIN = 70.0
SYMMETRY_BORDERLINE_MIN = 55.0
CADENCE_NORMAL_LO = 60.0
CADENCE_NORMAL_HI = 130.0
CADENCE_BORDER_LO = 48.0
CADENCE_BORDER_HI = 145.0
CONFIDENCE_NORMAL_MIN = 72.0
CONFIDENCE_BORDERLINE_MIN = 50.0


METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    "gait_quality_composite": MetricDefinition(
        key="gait_quality_composite",
        display_name="Overall Gait Quality",
        unit="/100",
        source="derived",
        threshold_source="heuristic",
        threshold_note=(
            f"Pose-derived heuristic bands: ≥{GQ_NORMAL_MIN:.0f} higher band, "
            f"≥{GQ_BORDERLINE_MIN:.0f} mid band (not a clinical cut-off; "
            "independent of source video labels)."
        ),
        user_explanation=(
            "Composite session score from several pose-derived signals. "
            "A high score does not mean every submetric is present, and it is "
            "not a clinical normality grade."
        ),
        technical_explanation=(
            "Weighted mean of available components (stability, stable-frame %, "
            "symmetry, cadence/cycle consistency, contact reliability, motion quality). "
            "Missing components are omitted and weights renormalized. "
            "Distinct from Gait Coordination (domain-group average)."
        ),
        bands=BandThresholds(
            normal_min=GQ_NORMAL_MIN, borderline_min=GQ_BORDERLINE_MIN
        ),
        analysis_ref="stablewalk.analysis.biomechanical.gait_quality_score.compute_gait_quality_score",
        aliases=("gait_quality", "overall_gait_quality"),
    ),
    "gait_quality_coordination": MetricDefinition(
        key="gait_quality_coordination",
        display_name="Gait Coordination",
        unit="/100",
        source="derived",
        threshold_source="heuristic",
        threshold_note="Domain-group average; separate from Composite Gait Quality.",
        user_explanation="Average of temporal/spatial symmetry, clearance, cycle, and contact domains.",
        technical_explanation=(
            "gait_analysis_summary._group_score over GAIT_QUALITY_DOMAIN_KEYS — "
            "not the Composite Gait Quality weighted formula."
        ),
        bands=BandThresholds(normal_min=68.0, borderline_min=45.0),
        analysis_ref="stablewalk.analysis.gait_analysis_summary.build_gait_analysis_summary",
    ),
    "stable_pct": MetricDefinition(
        key="stable_pct",
        display_name="Stability (stable-frame share)",
        unit="%",
        source="derived",
        threshold_source="heuristic",
        threshold_note=(
            f"Session bands are pose-derived heuristics (≥{STABLE_PCT_NORMAL_MIN:.0f}% / "
            f"≥{STABLE_PCT_BORDERLINE_MIN:.0f}%). Per-frame engine uses margin ≥ {MARGIN_STABLE_M} m."
        ),
        user_explanation=(
            "Share of valid frames whose geometric COM–BoS proxy met an internal "
            "4 cm bin. Monocular sessions withhold this as a KPI because that bin "
            "has no documented Hof or clinical basis for this method."
        ),
        technical_explanation=(
            f"stable_pct = 100 × (# frames with state Stable) / (# valid margins). "
            f"Stable when margin_m ≥ {MARGIN_STABLE_M}; cleared to unavailable when "
            f"valid_frame_ratio < {STABLE_PCT_VALID_FRAME_RATIO}."
        ),
        bands=BandThresholds(
            normal_min=STABLE_PCT_NORMAL_MIN, borderline_min=STABLE_PCT_BORDERLINE_MIN
        ),
        analysis_ref="stablewalk.analysis.biomechanical.stability_margin.analyze_stability_margin",
        aliases=("stability_state", "stability"),
    ),
    "stability_margin_m": MetricDefinition(
        key="stability_margin_m",
        display_name="Stability Margin",
        unit="m",
        source="derived",
        threshold_source="engine",
        threshold_note=(
            f"Engine: ≥{MARGIN_STABLE_M} m Stable; ≥{MARGIN_REDUCED_M} m Reduced; "
            f"<0 Unstable (pose-derived COM–BoS)."
        ),
        user_explanation="Signed distance from estimated COM projection to the support boundary.",
        technical_explanation=(
            "Per-frame margin from analyze_stability_margin._classify_margin. "
            "Session mean_margin_m requires sufficient valid frames."
        ),
        bands=BandThresholds(
            normal_min=MARGIN_STABLE_M, borderline_min=0.02, higher_is_better=True
        ),
        analysis_ref="stablewalk.analysis.biomechanical.stability_margin.analyze_stability_margin",
    ),
    "stability_frame_state": MetricDefinition(
        key="stability_frame_state",
        display_name="Stability State",
        unit="",
        source="derived",
        threshold_source="engine",
        threshold_note=(
            f"Per-frame engine labels from COM–BoS margin "
            f"(Stable ≥{MARGIN_STABLE_M} m). Pose-derived, not clinical."
        ),
        user_explanation="Instantaneous pose-derived stability class for the current frame.",
        technical_explanation="BiomechanicalFrameStability.stability_state from margin thresholds.",
        analysis_ref="stablewalk.analysis.biomechanical.stability_margin._classify_margin",
    ),
    "symmetry": MetricDefinition(
        key="symmetry",
        display_name="Symmetry",
        unit="%",
        source="derived",
        threshold_source="heuristic",
        threshold_note=(
            f"Pose-derived heuristic bands: ≥{SYMMETRY_NORMAL_MIN:.0f}% higher band, "
            f"≥{SYMMETRY_BORDERLINE_MIN:.0f}% mid band (not a clinical laterality "
            "cut-off; independent of source video labels)."
        ),
        user_explanation=(
            "Left–right similarity from pose-derived gait features. "
            "Not a clinical symmetry grade."
        ),
        technical_explanation="SymmetryAnalysis.overall_symmetry_pct (derived).",
        bands=BandThresholds(
            normal_min=SYMMETRY_NORMAL_MIN, borderline_min=SYMMETRY_BORDERLINE_MIN
        ),
        analysis_ref="stablewalk.analysis.biomechanical.symmetry_metrics.analyze_symmetry",
    ),
    "cadence": MetricDefinition(
        key="cadence",
        display_name="Cadence",
        unit="steps/min",
        source="estimated",
        threshold_source="literature",
        threshold_note=(
            f"Adult walking heuristic band {CADENCE_NORMAL_LO:.0f}–{CADENCE_NORMAL_HI:.0f} "
            "steps/min (literature-informed, not a diagnosis gate)."
        ),
        user_explanation="Estimated walking step rate from detected events.",
        technical_explanation="AdvancedGaitMetrics.cadence or gait cycle metrics.cadence_steps_per_min.",
        bands=BandThresholds(
            normal_min=CADENCE_NORMAL_LO,
            normal_max=CADENCE_NORMAL_HI,
            borderline_min=CADENCE_BORDER_LO,
            borderline_max=CADENCE_BORDER_HI,
            higher_is_better=True,
        ),
        analysis_ref="stablewalk.analysis.biomechanical.advanced_gait_metrics",
    ),
    "analysis_confidence": MetricDefinition(
        key="analysis_confidence",
        display_name="Analysis Confidence",
        unit="%",
        source="derived",
        threshold_source="heuristic",
        threshold_note=(
            f"Data-quality heuristic: ≥{CONFIDENCE_NORMAL_MIN:.0f}% / "
            f"≥{CONFIDENCE_BORDERLINE_MIN:.0f}%."
        ),
        user_explanation="Aggregate landmark and cycle data-quality confidence for this session.",
        technical_explanation="Mean of available tracking / cycle quality signals (not a clinical certainty).",
        bands=BandThresholds(
            normal_min=CONFIDENCE_NORMAL_MIN, borderline_min=CONFIDENCE_BORDERLINE_MIN
        ),
        analysis_ref="stablewalk.ui.overview_clinical_summary.extract_confidence_visualization",
    ),
    "gait_cycles": MetricDefinition(
        key="gait_cycles",
        display_name="Detected Gait Cycles",
        unit="",
        source="measured",
        threshold_source="heuristic",
        threshold_note="≥2 cycles preferred for cycle-normalized views.",
        user_explanation="Number of detected gait cycles in the analyzed session.",
        technical_explanation="Detected cycle count from gait cycle analysis (measured events).",
        bands=BandThresholds(normal_min=2.0, borderline_min=1.0),
        analysis_ref="stablewalk.analysis.gait_cycle_analysis",
    ),
}

_ALIAS_TO_KEY: dict[str, str] = {}
for _def in METRIC_DEFINITIONS.values():
    _ALIAS_TO_KEY[_def.key] = _def.key
    for alias in _def.aliases:
        _ALIAS_TO_KEY[alias] = _def.key


def resolve_metric_key(key: str) -> str:
    """Map UI aliases to the canonical registry key."""
    return _ALIAS_TO_KEY.get(key, key)


def get_metric_definition(key: str) -> MetricDefinition | None:
    return METRIC_DEFINITIONS.get(resolve_metric_key(key))


__all__ = [
    "BandThresholds",
    "CADENCE_BORDER_HI",
    "CADENCE_BORDER_LO",
    "CADENCE_NORMAL_HI",
    "CADENCE_NORMAL_LO",
    "CONFIDENCE_BORDERLINE_MIN",
    "CONFIDENCE_NORMAL_MIN",
    "GQ_BORDERLINE_MIN",
    "GQ_NORMAL_MIN",
    "MARGIN_STABLE_M",
    "METRIC_DEFINITIONS",
    "MetricDefinition",
    "STABLE_PCT_BORDERLINE_MIN",
    "STABLE_PCT_NORMAL_MIN",
    "SYMMETRY_BORDERLINE_MIN",
    "SYMMETRY_NORMAL_MIN",
    "get_metric_definition",
    "resolve_metric_key",
]
