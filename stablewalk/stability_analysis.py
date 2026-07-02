"""
Legacy import path for stability analysis.

Prefer ``stablewalk.analysis.stability`` in new code.
"""

from __future__ import annotations

import warnings

from stablewalk.analysis.stability import (  # noqa: F401
    StabilityReport,
    analyze_stability,
)

warnings.warn(
    "stablewalk.stability_analysis is deprecated; use stablewalk.analysis.stability",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["StabilityReport", "analyze_stability"]
