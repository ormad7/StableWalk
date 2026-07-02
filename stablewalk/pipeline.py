"""
Legacy import path for the gait pipeline.

Prefer ``stablewalk.core.pipeline`` in new code.
"""

from __future__ import annotations

import warnings

from stablewalk.core.pipeline import (  # noqa: F401
    PipelineResult,
    run_gait_pipeline,
)

warnings.warn(
    "stablewalk.pipeline is deprecated; use stablewalk.core.pipeline",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PipelineResult", "run_gait_pipeline"]
