"""
Backward-compatible stability module.

The v1 implementation is preserved below for reference and script imports.
Production scoring delegates to ``stability_scoring`` (v2).
"""

from __future__ import annotations

# Re-export v2 as the primary API.
from stablewalk.analysis.stability_config import (  # noqa: F401
    DEFAULT_STABILITY_CONFIG,
    LEGACY_V1_DOCUMENTATION,
    StabilityScoreConfig,
)
from stablewalk.analysis.stability_scoring import (  # noqa: F401
    SCORING_NOTES_V2,
    MetricResult,
    StabilityResult,
    STABLE_MIN,
    MODERATE_MIN,
    analyze_biomech_stability,
    generate_why_explanation,
)
from stablewalk.analysis.stability_validity import (  # noqa: F401
    StabilityComparisonResult,
    StabilityResultValidity,
    StabilityValidityThresholds,
    VALIDITY_CONFIGURATION_DOC,
    assess_stability_result_validity,
    compare_stability_results,
    format_validity_display,
)
from stablewalk.analysis.gait_analysis_summary import (  # noqa: F401
    GaitAnalysisSummary,
    build_gait_analysis_summary,
    format_summary_display,
    DOMAIN_SEMANTICS_DOC,
)

# Legacy v1 weight names (for old scripts comparing to v1).
METRIC_WEIGHTS = {
    "symmetry": 0.22,
    "step_consistency": 0.24,
    "body_stability": 0.14,
    "range_of_motion": 0.22,
    "trajectory_smoothness": 0.11,
    "pose_quality": 0.07,
}
SCORING_NOTES = SCORING_NOTES_V2

from stablewalk.analysis._biomech_stability_v1 import (  # noqa: E402
    _extract_series,
    _mean_abs_diff,
    _mean_jerk,
    _robust_range,
)

__all__ = [
    "StabilityResult",
    "MetricResult",
    "analyze_biomech_stability",
    "generate_why_explanation",
    "StabilityScoreConfig",
    "DEFAULT_STABILITY_CONFIG",
    "LEGACY_V1_DOCUMENTATION",
    "STABLE_MIN",
    "MODERATE_MIN",
    "METRIC_WEIGHTS",
    "SCORING_NOTES",
    "SCORING_NOTES_V2",
    "StabilityResultValidity",
    "StabilityComparisonResult",
    "StabilityValidityThresholds",
    "VALIDITY_CONFIGURATION_DOC",
    "assess_stability_result_validity",
    "compare_stability_results",
    "format_validity_display",
    "GaitAnalysisSummary",
    "build_gait_analysis_summary",
    "format_summary_display",
    "DOMAIN_SEMANTICS_DOC",
    "_extract_series",
    "_mean_abs_diff",
    "_mean_jerk",
    "_robust_range",
]
