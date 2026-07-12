"""
Advanced biomechanical gait analysis for StableWalk.

Operates after pose estimation, gait-event detection, and contact-mask generation.
All outputs distinguish measured vs estimated vs derived values.
"""

from stablewalk.analysis.biomechanical.orchestrator import (
    BiomechanicalAnalysisResult,
    run_biomechanical_analysis,
)

__all__ = [
    "BiomechanicalAnalysisResult",
    "run_biomechanical_analysis",
]
