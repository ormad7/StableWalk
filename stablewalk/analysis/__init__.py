"""
Analysis domain: gait metrics, GRF estimation, stability scoring, and ML screening.

Primary stability entry (GUI + demo validation): ``analyze_biomech_stability``.
Legacy penalty-based stability lives in ``analyze_gait`` / ``StabilityAnalyzer``.

Imports are lazy so ``import stablewalk.analysis.<submodule>`` does not pay for
the full analysis graph at GUI startup.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "GRFAnalyzer",
    "GRFTimeSeries",
    "ForceAnalyzer",
    "GaitMetrics",
    "GaitMetricsResult",
    "GaitAnalysisReport",
    "analyze_gait",
    "analyze_biomech_stability",
    "StabilityResult",
    "METRIC_WEIGHTS",
    "SCORING_NOTES",
    "analyze_gait_cycles",
    "analyze_gait_cycles_from_pose_sequence",
    "GaitCycleAnalysisResult",
    "analyze_gait_features",
    "GaitFeatureAnalysisResult",
    "CycleConsistencyResult",
    "FeatureNormalization",
    "estimate_body_segment_dimensions",
    "symmetry_index",
    "assess_opensim_id_readiness",
    "OpenSimIDReadinessReport",
    "estimate_virtual_grf",
    "VirtualForceResult",
    "StabilityAnalyzer",
    "StabilityReport",
    "StabilityMetrics",
    "analyze_stability",
    "analyze_spectral",
    "analyze_gait_autocorrelation",
    "GaitAutocorrReport",
    "analyze_gait_spectrum",
    "GaitSpectrumReport",
    "detect_gait_asymmetry_moments",
    "GaitAsymmetryMomentReport",
    "SpectralAnalysisReport",
    "SpectralAnalysisConfig",
    "SpectralFeatureSet",
    "extract_spectral_features",
    "analyze_asymmetry3d",
    "Asymmetry3DReport",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "GRFAnalyzer": ("stablewalk.analysis.forces", "GRFAnalyzer"),
    "GRFTimeSeries": ("stablewalk.analysis.forces", "GRFTimeSeries"),
    "ForceAnalyzer": ("stablewalk.analysis.forces", "ForceAnalyzer"),
    "GaitMetrics": ("stablewalk.analysis.metrics", "GaitMetrics"),
    "GaitMetricsResult": ("stablewalk.analysis.metrics", "GaitMetricsResult"),
    "GaitAnalysisReport": ("stablewalk.analysis.report", "GaitAnalysisReport"),
    "analyze_gait": ("stablewalk.analysis.report", "analyze_gait"),
    "analyze_biomech_stability": (
        "stablewalk.analysis.biomech_stability",
        "analyze_biomech_stability",
    ),
    "StabilityResult": ("stablewalk.analysis.biomech_stability", "StabilityResult"),
    "METRIC_WEIGHTS": ("stablewalk.analysis.biomech_stability", "METRIC_WEIGHTS"),
    "SCORING_NOTES": ("stablewalk.analysis.biomech_stability", "SCORING_NOTES"),
    "analyze_gait_cycles": (
        "stablewalk.analysis.gait_cycle_analysis",
        "analyze_gait_cycles",
    ),
    "analyze_gait_cycles_from_pose_sequence": (
        "stablewalk.analysis.gait_cycle_analysis",
        "analyze_gait_cycles_from_pose_sequence",
    ),
    "GaitCycleAnalysisResult": (
        "stablewalk.analysis.gait_cycle_analysis",
        "GaitCycleAnalysisResult",
    ),
    "analyze_gait_features": (
        "stablewalk.analysis.gait_feature_analysis",
        "analyze_gait_features",
    ),
    "GaitFeatureAnalysisResult": (
        "stablewalk.analysis.gait_feature_analysis",
        "GaitFeatureAnalysisResult",
    ),
    "CycleConsistencyResult": (
        "stablewalk.analysis.gait_feature_analysis",
        "CycleConsistencyResult",
    ),
    "FeatureNormalization": (
        "stablewalk.analysis.gait_feature_analysis",
        "FeatureNormalization",
    ),
    "estimate_body_segment_dimensions": (
        "stablewalk.analysis.gait_feature_analysis",
        "estimate_body_segment_dimensions",
    ),
    "symmetry_index": ("stablewalk.analysis.gait_feature_analysis", "symmetry_index"),
    "assess_opensim_id_readiness": (
        "stablewalk.analysis.opensim_id_readiness",
        "assess_opensim_id_readiness",
    ),
    "OpenSimIDReadinessReport": (
        "stablewalk.analysis.opensim_id_readiness",
        "OpenSimIDReadinessReport",
    ),
    "estimate_virtual_grf": ("stablewalk.analysis.virtual_grf", "estimate_virtual_grf"),
    "VirtualForceResult": ("stablewalk.analysis.virtual_grf", "VirtualForceResult"),
    "StabilityAnalyzer": ("stablewalk.analysis.stability", "StabilityAnalyzer"),
    "StabilityReport": ("stablewalk.analysis.stability", "StabilityReport"),
    "StabilityMetrics": ("stablewalk.analysis.stability", "StabilityMetrics"),
    "analyze_stability": ("stablewalk.analysis.stability", "analyze_stability"),
    "analyze_spectral": ("stablewalk.analysis.spectral.report", "analyze_spectral"),
    "analyze_gait_autocorrelation": (
        "stablewalk.analysis.gait_acf",
        "analyze_gait_autocorrelation",
    ),
    "GaitAutocorrReport": (
        "stablewalk.analysis.gait_acf",
        "GaitAutocorrReport",
    ),
    "analyze_gait_spectrum": (
        "stablewalk.analysis.gait_spectrum",
        "analyze_gait_spectrum",
    ),
    "GaitSpectrumReport": (
        "stablewalk.analysis.gait_spectrum",
        "GaitSpectrumReport",
    ),
    "detect_gait_asymmetry_moments": (
        "stablewalk.analysis.gait_asymmetry_moments",
        "detect_gait_asymmetry_moments",
    ),
    "GaitAsymmetryMomentReport": (
        "stablewalk.analysis.gait_asymmetry_moments",
        "GaitAsymmetryMomentReport",
    ),
    "SpectralAnalysisReport": (
        "stablewalk.analysis.spectral.report",
        "SpectralAnalysisReport",
    ),
    "SpectralAnalysisConfig": (
        "stablewalk.analysis.spectral.config",
        "SpectralAnalysisConfig",
    ),
    "SpectralFeatureSet": (
        "stablewalk.analysis.spectral.features",
        "SpectralFeatureSet",
    ),
    "extract_spectral_features": (
        "stablewalk.analysis.spectral.features",
        "extract_spectral_features",
    ),
    "analyze_asymmetry3d": (
        "stablewalk.analysis.asymmetry3d.report",
        "analyze_asymmetry3d",
    ),
    "Asymmetry3DReport": (
        "stablewalk.analysis.asymmetry3d.report",
        "Asymmetry3DReport",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = target
    import importlib

    mod = importlib.import_module(mod_name)
    value = getattr(mod, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
