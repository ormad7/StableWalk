"""
Advanced gait analytics: anomaly detection, classification, comparison, optional ML.

See ``RESEARCH.md`` for biomechanics context and research relevance.
"""

from stablewalk.analysis.advanced.anomaly import AnomalyReport, GaitAnomalyDetector
from stablewalk.analysis.advanced.classification import ClassificationResult, GaitClassifier
from stablewalk.analysis.advanced.comparison import GaitSessionComparison, compare_gait_sessions
from stablewalk.analysis.advanced.features import GaitFeatureVector, extract_gait_features
from stablewalk.analysis.advanced.pipeline import analyze_gait_advanced

__all__ = [
    "GaitFeatureVector",
    "extract_gait_features",
    "GaitAnomalyDetector",
    "AnomalyReport",
    "GaitClassifier",
    "ClassificationResult",
    "GaitSessionComparison",
    "compare_gait_sessions",
    "analyze_gait_advanced",
]
