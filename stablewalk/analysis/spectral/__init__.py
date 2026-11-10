"""Spectral characteristics and autocorrelation-matrix analysis.

All public results are **derived** from pose time series. They are not
clinical instability diagnoses.
"""

from __future__ import annotations

from stablewalk.analysis.spectral.config import SpectralAnalysisConfig, default_config
from stablewalk.analysis.spectral.features import SpectralFeatureSet, extract_spectral_features
from stablewalk.analysis.spectral.report import SpectralAnalysisReport, analyze_spectral

__all__ = [
    "SpectralAnalysisConfig",
    "SpectralAnalysisReport",
    "SpectralFeatureSet",
    "analyze_spectral",
    "extract_spectral_features",
    "default_config",
]
