"""
Stability margin — projected COM relative to Base of Support.

Margin and state are **derived estimates** from pose-based COM and foot polygons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from stablewalk.analysis.biomechanical.base_of_support import BaseOfSupportAnalysis
from stablewalk.analysis.biomechanical.com_estimation import CenterOfMassAnalysis
from stablewalk.analysis.biomechanical.types import BiomechanicalFrameStability, StabilityState

logger = logging.getLogger(__name__)

# Margin thresholds as fraction of typical foot length (~0.25 m at body scale).
_STABLE_MARGIN_M = 0.04
_REDUCED_MARGIN_M = 0.0


def _point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / max(yj - yi, 1e-9) + xi
        ):
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
    """
    if len(polygon) < 2:
        return None
    if len(polygon) == 2:
        d = _distance_to_segment(px, py, polygon[0][0], polygon[0][1], polygon[1][0], polygon[1][1])
        return d if _point_in_polygon(px, py, polygon) else -d

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
        return StabilityState.REDUCED.value
    if margin_m >= _STABLE_MARGIN_M:
        return StabilityState.STABLE.value
    if margin_m >= _REDUCED_MARGIN_M:
        return StabilityState.REDUCED.value
    return StabilityState.UNSTABLE.value


@dataclass
class StabilityMarginAnalysis:
    per_frame: list[BiomechanicalFrameStability] = field(default_factory=list)
    mean_margin_m: float | None = None
    stable_pct: float = 0.0
    kind: str = "derived"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "frame_count": len(self.per_frame),
            "mean_margin_m": self.mean_margin_m,
            "stable_pct": round(self.stable_pct, 2),
            "note": "COM–BoS margin from estimated pose kinematics.",
        }


def analyze_stability_margin(
    com: CenterOfMassAnalysis,
    bos: BaseOfSupportAnalysis,
) -> StabilityMarginAnalysis:
    """Compute per-frame stability margin and state."""
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
                    stability_state=StabilityState.REDUCED.value,
                    com_projection=(com_f.position[0], com_f.position[2]),
                    confidence=com_f.confidence * 0.4,
                )
            )
            continue

        px, py = com_f.position[0], com_f.position[2]
        margin = _signed_margin_to_polygon(px, py, bos_f.polygon_xy)
        state = _classify_margin(margin)
        conf = min(com_f.confidence, bos_f.confidence)

        if margin is not None:
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
    mean_margin = float(sum(margins) / len(margins)) if margins else None
    stable_pct = (stable_count / n * 100.0) if n else 0.0

    return StabilityMarginAnalysis(
        per_frame=per_frame,
        mean_margin_m=mean_margin,
        stable_pct=stable_pct,
    )


__all__ = ["StabilityMarginAnalysis", "analyze_stability_margin"]
