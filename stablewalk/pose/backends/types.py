"""
Standardized per-frame outputs for human motion / pose reconstruction backends.

All backends map into this schema so gait analysis, OpenSim export, and research
comparisons can share a common interchange format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CoordinateSystemMetadata:
    """Describes the 3D coordinate frame produced by a backend."""

    name: str
    units: str
    origin_description: str
    x_axis: str
    y_axis: str
    z_axis: str
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "units": self.units,
            "origin": self.origin_description,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "z_axis": self.z_axis,
            "notes": self.notes,
        }


@dataclass
class HumanMotionFrame:
    """
    One frame of standardized human motion output.

    ``joint_positions_3d`` uses **canonical StableWalk joint ids**
    (see ``models.joint_registry``): pelvis, left_hip, left_knee, …
    """

    frame_index: int
    timestamp_s: float
    joint_positions_3d: dict[str, tuple[float, float, float]]
    landmark_confidence: dict[str, float]
    backend_name: str
    coordinate_system: CoordinateSystemMetadata
    detected: bool = True
    joint_rotations: dict[str, tuple[float, float, float, float]] | None = None
    root_position: tuple[float, float, float] | None = None
    root_orientation: tuple[float, float, float, float] | None = None
    body_shape: dict[str, float] | None = None
    raw_landmarks: dict[str, tuple[float, float, float]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanMotionSequence:
    """Full video/session output from one pose or HMR backend."""

    frames: list[HumanMotionFrame]
    fps: float
    source_video: str
    backend_name: str
    coordinate_system: CoordinateSystemMetadata

    @property
    def valid_frame_count(self) -> int:
        return sum(1 for f in self.frames if f.detected)

    @property
    def valid_frame_ratio(self) -> float:
        if not self.frames:
            return 0.0
        return self.valid_frame_count / len(self.frames)
