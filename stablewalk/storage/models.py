"""Data models for persisted analysis sessions and kinematic samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class AnalysisSession:
    """Metadata for one saved analysis session."""

    session_id: str
    video_source: str
    created_at: str = field(default_factory=utc_now_iso)
    fps: float | None = None
    sample_count: int = 0
    selected_dofs: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class KinematicSample:
    """One DOF/joint measurement at a single playback frame."""

    frame_number: int
    time_s: float
    dof_name: str
    joint_name: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    angle_deg: float | None = None
    velocity: float | None = None
    velocity_deg_s: float | None = None
    next_angle_deg: float | None = None
    delta_angle_deg: float | None = None
