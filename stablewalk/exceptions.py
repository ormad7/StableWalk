"""
StableWalk exception hierarchy.

Use typed errors at service boundaries so CLI and GUI can show consistent messages
without catching bare ``Exception``.
"""

from __future__ import annotations


class StableWalkError(Exception):
    """Base error for recoverable StableWalk failures."""


class VideoProcessingError(StableWalkError):
    """Video ingest, validation, or frame extraction failed."""


class PoseEstimationError(StableWalkError):
    """Pose detection or sequence export failed."""


class PoseLoadError(StableWalkError):
    """Pose JSON could not be read or parsed."""


class AnalysisError(StableWalkError):
    """Gait / stability / force analysis failed."""


class ConfigurationError(StableWalkError):
    """Invalid paths, missing assets, or bad configuration."""
