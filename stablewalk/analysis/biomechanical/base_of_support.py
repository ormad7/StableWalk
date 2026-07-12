"""
Base of Support (BoS) estimation from foot contact states.

Support polygons are **estimated** from pose-derived foot landmarks in the
horizontal (X–Z) plane (+Y vertical).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from stablewalk.analysis.biomechanical.types import BiomechanicalFrameBoS
from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot

logger = logging.getLogger(__name__)


def _foot_points(snap: SkeletonSnapshot, side: str) -> list[tuple[float, float]]:
    """Horizontal-plane (x, z) foot contact points."""
    pts: list[tuple[float, float]] = []
    for jid in (f"{side}_heel", f"{side}_ankle", f"{side}_toe", f"{side}_foot"):
        j = snap.joints.get(jid)
        if j is not None:
            pts.append((j.position.x, j.position.z))
    # Deduplicate near-identical points
    unique: list[tuple[float, float]] = []
    for p in pts:
        if not any(abs(p[0] - u[0]) < 1e-4 and abs(p[1] - u[1]) < 1e-4 for u in unique):
            unique.append(p)
    return unique


def _convex_hull_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Monotonic chain convex hull."""
    if len(points) <= 2:
        return list(points)
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _polygon_area(vertices: list[tuple[float, float]]) -> float:
    if len(vertices) < 3:
        return 0.0
    area = 0.0
    n = len(vertices)
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) * 0.5


def _polygon_centroid(vertices: list[tuple[float, float]]) -> tuple[float, float]:
    if not vertices:
        return (0.0, 0.0)
    if len(vertices) == 1:
        return vertices[0]
    cx = sum(v[0] for v in vertices) / len(vertices)
    cy = sum(v[1] for v in vertices) / len(vertices)
    return cx, cy


@dataclass
class BaseOfSupportAnalysis:
    per_frame: list[BiomechanicalFrameBoS] = field(default_factory=list)
    fps: float = 30.0
    kind: str = "estimated"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "fps": self.fps,
            "frame_count": len(self.per_frame),
            "note": "Support polygon from pose foot landmarks — estimated.",
        }


def analyze_base_of_support(
    recording: GaitMotionRecording,
    contact: FootContactAnalysisResult,
) -> BaseOfSupportAnalysis:
    """Build per-frame support polygons from contact masks."""
    fps = max(contact.fps, recording.fps, 1e-6)
    per_frame: list[BiomechanicalFrameBoS] = []

    for cf in contact.per_frame:
        snap = recording.snapshot_at(cf.frame_index)
        if snap is None:
            continue

        left_on = bool(cf.left_contact_binary)
        right_on = bool(cf.right_contact_binary)

        if left_on and right_on:
            support_type = "double_support"
        elif left_on:
            support_type = "left_stance"
        elif right_on:
            support_type = "right_stance"
        else:
            support_type = "swing"

        polygon: list[tuple[float, float]] = []
        conf = 0.0

        if left_on:
            polygon.extend(_foot_points(snap, "left"))
            conf += cf.left_confidence
        if right_on:
            polygon.extend(_foot_points(snap, "right"))
            conf += cf.right_confidence

        if support_type == "swing":
            # Use trailing stance foot if available from previous frame
            polygon = []
            conf = max(cf.left_confidence, cf.right_confidence) * 0.3
        elif len(polygon) >= 3:
            polygon = _convex_hull_2d(polygon)
            conf = conf / (2.0 if support_type == "double_support" else 1.0)
        elif len(polygon) == 2:
            conf *= 0.7
        else:
            conf *= 0.4

        conf = min(1.0, max(0.0, conf))
        centroid = _polygon_centroid(polygon)
        area = _polygon_area(polygon)

        per_frame.append(
            BiomechanicalFrameBoS(
                frame_index=cf.frame_index,
                time_s=cf.time_s,
                support_type=support_type,
                polygon_xy=polygon,
                centroid=centroid,
                area_m2=area,
                confidence=conf,
            )
        )

    return BaseOfSupportAnalysis(per_frame=per_frame, fps=fps)


__all__ = ["BaseOfSupportAnalysis", "analyze_base_of_support"]
