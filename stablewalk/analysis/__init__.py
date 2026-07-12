"""
Analysis domain: gait metrics, GRF estimation, stability scoring, and ML screening.

Primary stability entry (GUI + demo validation): ``analyze_biomech_stability``.
Legacy penalty-based stability lives in ``analyze_gait`` / ``StabilityAnalyzer``.
"""

from stablewalk.analysis.biomech_stability import (
    METRIC_WEIGHTS,
    SCORING_NOTES,
    StabilityResult,
    analyze_biomech_stability,
)
from stablewalk.analysis.gait_cycle_analysis import (
    GaitCycleAnalysisResult,
    analyze_gait_cycles,
    analyze_gait_cycles_from_pose_sequence,
)
from stablewalk.analysis.gait_feature_analysis import (
    CycleConsistencyResult,
    FeatureNormalization,
    GaitFeatureAnalysisResult,
    analyze_gait_features,
    estimate_body_segment_dimensions,
    symmetry_index,
)
from stablewalk.analysis.opensim_id_readiness import (
    OpenSimIDReadinessReport,
    assess_opensim_id_readiness,
)
from stablewalk.analysis.virtual_grf import (
    VirtualForceResult,
    estimate_virtual_grf,
)
from stablewalk.analysis.forces import GRFAnalyzer, GRFTimeSeries, ForceAnalyzer
from stablewalk.analysis.metrics import GaitMetrics, GaitMetricsResult
from stablewalk.analysis.report import GaitAnalysisReport, analyze_gait
from stablewalk.analysis.stability import (
    StabilityAnalyzer,
    StabilityReport,
    StabilityMetrics,
    analyze_stability,
)

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
]
