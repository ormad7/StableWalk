"""
Stability margin — projected COM relative to Base of Support.

Margin and state are **derived estimates** from pose-based COM and foot polygons.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from stablewalk.analysis.biomechanical.base_of_support import BaseOfSupportAnalysis
from stablewalk.analysis.biomechanical.com_estimation import CenterOfMassAnalysis
from stablewalk.analysis.biomechanical.types import BiomechanicalFrameStability, StabilityState

logger = logging.getLogger(__name__)

# Margin thresholds as fraction of typical foot length (~0.25 m at body scale).
_STABLE_MARGIN_M = 0.04
_REDUCED_MARGIN_M = 0.0
_MIN_VALID_FRAME_RATIO = 0.50
_MIN_FRAME_CONFIDENCE = 0.35


def _point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > py) != (yj > py):
            denominator = yj - yi
            if abs(denominator) > 1e-12:
                crossing_x = (xj - xi) * (py - yi) / denominator + xi
                if px < crossing_x:
                    inside = not inside
        j = i
    return inside


def _distance_to_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5


def _signed_margin_to_polygon(
    px: float,
    py: float,
    polygon: list[tuple[float, float]],
) -> float | None:
    """
    Signed distance to polygon edge.

    Positive when COM projection is inside; negative when outside.
    Two-point support (single-limb line) uses a thickened foot model so the
    margin is not forced negative solely because a line has zero area.
    """
    if len(polygon) < 2:
        return None
    if len(polygon) == 2:
        d = _distance_to_segment(
            px, py, polygon[0][0], polygon[0][1], polygon[1][0], polygon[1][1]
        )
        # ~3 cm half-width at body scale 1.0 (~5 cm after ×1.7 m stature).
        foot_half_width = 0.03
        return foot_half_width - d

    min_dist = float("inf")
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        d = _distance_to_segment(px, py, ax, ay, bx, by)
        min_dist = min(min_dist, d)

    inside = _point_in_polygon(px, py, polygon)
    return min_dist if inside else -min_dist


def _classify_margin(margin_m: float | None) -> str:
    if margin_m is None:
        return "Unavailable"
    if margin_m >= _STABLE_MARGIN_M:
        return StabilityState.STABLE.value
    if margin_m >= _REDUCED_MARGIN_M:
        return StabilityState.REDUCED.value
    return StabilityState.UNSTABLE.value


@dataclass
class StabilityMarginAnalysis:
    per_frame: list[BiomechanicalFrameStability] = field(default_factory=list)
    mean_margin_m: float | None = None
    stable_pct: float | None = None
    valid_frame_ratio: float = 0.0
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    kind: str = "derived"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "frame_count": len(self.per_frame),
            "mean_margin_m": self.mean_margin_m,
            "stable_pct": None if self.stable_pct is None else round(self.stable_pct, 2),
            "valid_frame_ratio": round(self.valid_frame_ratio, 4),
            "confidence": round(self.confidence, 4),
            "warnings": list(self.warnings),
            "note": "COM–BoS margin from estimated pose kinematics.",
        }


def analyze_stability_margin(
    com: CenterOfMassAnalysis,
    bos: BaseOfSupportAnalysis,
    *,
    meters_per_unit: float = 1.0,
) -> StabilityMarginAnalysis:
    """Compute per-frame stability margin and state in absolute metres.

    ``meters_per_unit`` converts body-normalized pose coordinates to metres.
    """
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise ValueError("meters_per_unit must be finite and positive")
    bos_by_frame = {f.frame_index: f for f in bos.per_frame}
    per_frame: list[BiomechanicalFrameStability] = []
    margins: list[float] = []
    stable_count = 0

    for com_f in com.per_frame:
        bos_f = bos_by_frame.get(com_f.frame_index)
        if bos_f is None or not bos_f.polygon_xy:
            per_frame.append(
                BiomechanicalFrameStability(
                    frame_index=com_f.frame_index,
                    time_s=com_f.time_s,
                    stability_margin_m=None,
                    stability_state="Unavailable",
                    com_projection=(com_f.position[0], com_f.position[2]),
                    confidence=com_f.confidence * 0.4,
                )
            )
            continue

        px, py = com_f.position[0], com_f.position[2]
        margin_normalized = _signed_margin_to_polygon(px, py, bos_f.polygon_xy)
        margin = (
            margin_normalized * meters_per_unit
            if margin_normalized is not None
            else None
        )
        conf = min(com_f.confidence, bos_f.confidence)
        state = (
            _classify_margin(margin)
            if margin is not None
            and math.isfinite(margin)
            and conf >= _MIN_FRAME_CONFIDENCE
            else "Unavailable"
        )

        if margin is not None and math.isfinite(margin) and conf >= _MIN_FRAME_CONFIDENCE:
            margins.append(margin)
            if state == StabilityState.STABLE.value:
                stable_count += 1

        per_frame.append(
            BiomechanicalFrameStability(
                frame_index=com_f.frame_index,
                time_s=com_f.time_s,
                stability_margin_m=margin,
                stability_state=state,
                com_projection=(px, py),
                confidence=conf,
            )
        )

    n = len(per_frame)
    valid_ratio = len(margins) / n if n else 0.0
    valid_confidences = [
        frame.confidence
        for frame in per_frame
        if frame.stability_margin_m is not None
        and math.isfinite(frame.stability_margin_m)
        and frame.confidence >= _MIN_FRAME_CONFIDENCE
    ]
    confidence = (
        float(sum(valid_confidences) / len(valid_confidences))
        if valid_confidences
        else 0.0
    )
    warnings: list[str] = []
    if valid_ratio < _MIN_VALID_FRAME_RATIO:
        mean_margin = None
        stable_pct = None
        warnings.append(
            "Stability margin unavailable: only "
            f"{valid_ratio:.0%} of frames had a valid COM/support polygon with "
            f"confidence ≥ {_MIN_FRAME_CONFIDENCE:.2f}."
        )
    else:
        mean_margin = float(sum(margins) / len(margins))
        stable_pct = stable_count / len(margins) * 100.0

    return StabilityMarginAnalysis(
        per_frame=per_frame,
        mean_margin_m=mean_margin,
        stable_pct=stable_pct,
        valid_frame_ratio=valid_ratio,
        confidence=confidence,
        warnings=warnings,
    )


__all__ = ["StabilityMarginAnalysis", "analyze_stability_margin"]
