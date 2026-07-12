"""
Unified internal motion representation for MediaPipe and SMPL backends.

Downstream gait analysis, Real-to-Sim export, and OpenSim-compatible bridges
consume this schema so perception backends are interchangeable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stablewalk.models.joint_registry import JOINT_IDS, ROOT_JOINT_ID
from stablewalk.pose.backends.types import CoordinateSystemMetadata, HumanMotionFrame, HumanMotionSequence


UNIFIED_MOTION_SCHEMA_VERSION = "1.0"


@dataclass
class UnifiedMotionFrame:
    """One time-aligned frame of human motion (backend-agnostic)."""

    frame_index: int
    timestamp_s: float
    root_position: tuple[float, float, float] | None = None
    root_orientation: tuple[float, float, float, float] | None = None  # w, x, y, z
    joint_positions_3d: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    joint_rotations: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    body_shape_parameters: dict[str, float] = field(default_factory=dict)
    pose_confidence: float = 0.0
    detected: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScaleInformation:
    """Body scale metadata attached to a motion sequence."""

    subject_height_m: float | None = None
    leg_length_m: float | None = None
    scale_to_meters: float = 1.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_height_m": self.subject_height_m,
            "leg_length_m": self.leg_length_m,
            "scale_to_meters": self.scale_to_meters,
            "notes": self.notes,
        }


@dataclass
class UnifiedHumanMotion:
    """
    Normalized motion sequence produced by any pose / HMR backend.

    MediaPipe fills joint positions and heuristic depth; SMPL adds mesh-based
    root motion, joint rotations (axis-angle or quat), and shape β parameters.
    """

    fps: float
    timestamps: np.ndarray
    frames: list[UnifiedMotionFrame]
    source_backend: str
    coordinate_system: CoordinateSystemMetadata
    scale_information: ScaleInformation
    source_video: str = ""
    provider_name: str = ""
    schema_version: str = UNIFIED_MOTION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def valid_frame_count(self) -> int:
        return sum(1 for f in self.frames if f.detected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "valid_frame_count": self.valid_frame_count,
            "source_backend": self.source_backend,
            "provider_name": self.provider_name,
            "source_video": self.source_video,
            "coordinate_system": self.coordinate_system.to_dict(),
            "scale_information": self.scale_information.to_dict(),
            "metadata": self.metadata,
        }


def _mean_confidence(frame: HumanMotionFrame) -> float:
    if not frame.landmark_confidence:
        return 1.0 if frame.detected else 0.0
    vals = [float(v) for v in frame.landmark_confidence.values()]
    return float(sum(vals) / len(vals)) if vals else 0.0


def human_motion_sequence_to_unified(
    sequence: HumanMotionSequence,
    *,
    scale: ScaleInformation | None = None,
    provider_name: str = "",
) -> UnifiedHumanMotion:
    """Convert a backend ``HumanMotionSequence`` into ``UnifiedHumanMotion``."""
    frames: list[UnifiedMotionFrame] = []
    timestamps = np.array([f.timestamp_s for f in sequence.frames], dtype=np.float64)

    for hf in sequence.frames:
        joint_rots: dict[str, tuple[float, float, float, float]] = {}
        if hf.joint_rotations:
            joint_rots = dict(hf.joint_rotations)

        shape: dict[str, float] = {}
        if hf.body_shape:
            shape = dict(hf.body_shape)

        frames.append(
            UnifiedMotionFrame(
                frame_index=hf.frame_index,
                timestamp_s=hf.timestamp_s,
                root_position=hf.root_position,
                root_orientation=hf.root_orientation,
                joint_positions_3d=dict(hf.joint_positions_3d),
                joint_rotations=joint_rots,
                body_shape_parameters=shape,
                pose_confidence=_mean_confidence(hf),
                detected=hf.detected,
                metadata=dict(hf.metadata),
            )
        )

    default_scale = scale or ScaleInformation(
        notes=f"Backend {sequence.backend_name} — see coordinate_system metadata."
    )

    return UnifiedHumanMotion(
        fps=float(sequence.fps),
        timestamps=timestamps,
        frames=frames,
        source_backend=sequence.backend_name,
        coordinate_system=sequence.coordinate_system,
        scale_information=default_scale,
        source_video=sequence.source_video,
        provider_name=provider_name or sequence.backend_name,
        metadata={"backend_name": sequence.backend_name},
    )


def unified_joint_positions_array(
    motion: UnifiedHumanMotion,
    *,
    joint_ids: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Stack joint positions as ``(N, J, 3)`` float64 array."""
    jids = joint_ids or [ROOT_JOINT_ID, *JOINT_IDS]
    n = motion.frame_count
    j = len(jids)
    arr = np.full((n, j, 3), np.nan, dtype=np.float64)
    for i, frame in enumerate(motion.frames):
        for j_idx, jid in enumerate(jids):
            pos = frame.joint_positions_3d.get(jid)
            if pos is not None:
                arr[i, j_idx] = pos
    return jids, arr


def unified_metadata_json(motion: UnifiedHumanMotion) -> str:
    return json.dumps(motion.to_dict())


__all__ = [
    "UNIFIED_MOTION_SCHEMA_VERSION",
    "UnifiedMotionFrame",
    "UnifiedHumanMotion",
    "ScaleInformation",
    "human_motion_sequence_to_unified",
    "unified_joint_positions_array",
    "unified_metadata_json",
]
