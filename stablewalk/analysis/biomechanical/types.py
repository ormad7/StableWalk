"""Shared types for biomechanical analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

ValueKind = Literal["measured", "estimated", "derived"]

StabilityStateName = Literal["Stable", "Reduced Stability", "Unstable"]


class StabilityState(str, Enum):
    STABLE = "Stable"
    REDUCED = "Reduced Stability"
    UNSTABLE = "Unstable"


@dataclass
class MetricWithConfidence:
    """A scalar metric with provenance and confidence."""

    value: float | None
    unit: str = ""
    kind: ValueKind = "estimated"
    confidence: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BiomechanicalFrameCOM:
    frame_index: int
    time_s: float
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    acceleration: tuple[float, float, float]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "confidence": self.confidence,
        }


@dataclass
class BiomechanicalFrameBoS:
    frame_index: int
    time_s: float
    support_type: str  # left_stance | right_stance | double_support | swing
    polygon_xy: list[tuple[float, float]]  # horizontal-plane vertices
    centroid: tuple[float, float]
    area_m2: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "support_type": self.support_type,
            "polygon_xy": self.polygon_xy,
            "centroid": self.centroid,
            "area_m2": self.area_m2,
            "confidence": self.confidence,
        }


@dataclass
class BiomechanicalFrameStability:
    frame_index: int
    time_s: float
    stability_margin_m: float | None
    stability_state: StabilityStateName
    com_projection: tuple[float, float]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JointROMStats:
    joint: str
    side: str
    flexion_min_deg: float | None = None
    flexion_max_deg: float | None = None
    extension_min_deg: float | None = None
    extension_max_deg: float | None = None
    rom_deg: float | None = None
    mean_deg: float | None = None
    std_deg: float | None = None
    confidence: float = 0.0
    kind: ValueKind = "estimated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GaitQualityScore:
    score: float  # 0–100
    confidence: float
    dominant_factors: list[str]
    explanation: str
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "confidence": round(self.confidence, 3),
            "dominant_factors": list(self.dominant_factors),
            "explanation": self.explanation,
            "components": {k: round(v, 2) for k, v in self.components.items()},
        }
