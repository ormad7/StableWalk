"""
Anatomical OpenSim marker reconstruction from MediaPipe-exported TRC landmarks.

Inspired by marker-enhancement concepts (segment-local frames, subject-scaled
offsets, temporal filtering) but implemented entirely within StableWalk — no
OpenCap code or proprietary models are used.

Pipeline stages per session:
  1. Read raw StableWalk TRC landmarks (OpenSim lab frame, mm)
  2. Build pelvis / thigh / shank / foot / torso anatomical frames per frame
  3. Place Gait2392 markers via DIRECT mapping or segment-relative offsets
  4. Temporal smoothing + jump detection / robust interpolation
  5. Export mapped TRC + confidence-aware validation report
"""

from __future__ import annotations

import csv
import io
import logging
import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

MappingType = Literal[
    "DIRECT",
    "DERIVED_ANATOMICAL",
    "TEMPORAL_ESTIMATED",
    "UNAVAILABLE",
]

BiomechanicalRisk = Literal["LOW", "MODERATE", "HIGH", "N/A"]


class IkReadinessLevel(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    NOT_READY = "NOT READY"


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class AnatomicalFrame:
    """Right-handed segment frame: x ≈ lateral, y ≈ proximal-distal/up, z ≈ anterior."""

    origin: Vec3
    x: Vec3
    y: Vec3
    z: Vec3


@dataclass
class SegmentDimensions:
    hip_width: float
    shoulder_width: float
    thigh_length_left: float
    thigh_length_right: float
    shank_length_left: float
    shank_length_right: float
    foot_length_left: float
    foot_length_right: float
    torso_length: float


@dataclass(frozen=True)
class MarkerSpec:
    opensim_name: str
    mapping_type: MappingType
    source_landmarks: tuple[str, ...]
    calculation: str
    confidence_base: float
    biomechanical_risk: BiomechanicalRisk
    compute: Callable[
        [dict[str, Vec3], dict[str, AnatomicalFrame], SegmentDimensions],
        tuple[Vec3 | None, float],
    ]


@dataclass
class MarkerFrameResult:
    position: Vec3 | None
    confidence: float
    mapping_type: MappingType
    source_landmarks: list[str]
    warnings: list[str] = field(default_factory=list)
    was_interpolated: bool = False
    was_filtered: bool = False


@dataclass
class MarkerReconstructionResult:
    """Full-session marker reconstruction output."""

    marker_names: list[str]
    frames: list[tuple[int, float, dict[str, MarkerFrameResult]]]
    fps: float
    units: str
    catalog_rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    direct_count: int = 0
    derived_count: int = 0
    temporal_count: int = 0
    unavailable_count: int = 0
    raw_coverage_percent: float = 0.0
    high_confidence_coverage_percent: float = 0.0
    ik_readiness: IkReadinessLevel = IkReadinessLevel.NOT_READY
    ik_readiness_score: float = 0.0
    low_confidence_markers: list[str] = field(default_factory=list)
    trajectory_comparison: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    def frame_positions(self, frame_idx: int) -> dict[str, Vec3]:
        for fi, _t, markers in self.frames:
            if fi == frame_idx:
                return {
                    name: m.position
                    for name, m in markers.items()
                    if m.position is not None
                }
        return {}


@dataclass(frozen=True)
class MarkerReconstructionConfig:
    """Tunable reconstruction / filtering parameters."""

    filter_window: int = 5
    max_velocity_mm_s: float = 3500.0
    max_acceleration_mm_s2: float = 80000.0
    max_jump_mm: float = 120.0
    high_confidence_threshold: float = 0.65
    ik_readiness_high: float = 72.0
    ik_readiness_moderate: float = 48.0


DEFAULT_RECONSTRUCTION_CONFIG = MarkerReconstructionConfig()

# All Gait2392 IK task markers (subject01_Setup_IK.xml order subset).
GAIT2392_MARKER_NAMES: tuple[str, ...] = (
    "Sternum",
    "R.Acromium",
    "L.Acromium",
    "Top.Head",
    "R.ASIS",
    "L.ASIS",
    "V.Sacral",
    "R.Thigh.Upper",
    "R.Thigh.Front",
    "R.Thigh.Rear",
    "R.Shank.Upper",
    "R.Shank.Front",
    "R.Shank.Rear",
    "R.Heel",
    "R.Midfoot.Sup",
    "R.Midfoot.Lat",
    "R.Toe.Lat",
    "R.Toe.Med",
    "R.Toe.Tip",
    "L.Thigh.Upper",
    "L.Thigh.Front",
    "L.Thigh.Rear",
    "L.Shank.Upper",
    "L.Shank.Front",
    "L.Shank.Rear",
    "L.Heel",
    "L.Midfoot.Sup",
    "L.Midfoot.Lat",
    "L.Toe.Lat",
    "L.Toe.Med",
    "L.Toe.Tip",
)


# ---------------------------------------------------------------------------
# Vector / frame math
# ---------------------------------------------------------------------------
def _v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _v_len(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _v_norm(a: Vec3, fallback: Vec3 = (1.0, 0.0, 0.0)) -> Vec3:
    ln = _v_len(a)
    if ln < 1e-9:
        return fallback
    return (a[0] / ln, a[1] / ln, a[2] / ln)


def _v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    )


def _point_in_frame(frame: AnatomicalFrame, a: float, b: float, c: float) -> Vec3:
    """origin + a*local_x + b*local_y + c*local_z"""
    ox, oy, oz = frame.origin
    return (
        ox + a * frame.x[0] + b * frame.y[0] + c * frame.z[0],
        oy + a * frame.x[1] + b * frame.y[1] + c * frame.y[1],
        oz + a * frame.x[2] + b * frame.y[2] + c * frame.z[2],
    )


def _dist(a: Vec3, b: Vec3) -> float:
    return _v_len(_v_sub(a, b))


def _get(lm: dict[str, Vec3], name: str) -> Vec3 | None:
    return lm.get(name)


def _direct(lm: dict[str, Vec3], sw_name: str, conf: float = 0.9) -> tuple[Vec3 | None, float]:
    p = lm.get(sw_name)
    return (p, conf if p is not None else 0.0)


# ---------------------------------------------------------------------------
# Anatomical frames
# ---------------------------------------------------------------------------
def build_anatomical_frames(
    lm: dict[str, Vec3],
    dims: SegmentDimensions,
) -> dict[str, AnatomicalFrame]:
    frames: dict[str, AnatomicalFrame] = {}
    lh, rh = lm.get("L_HIP"), lm.get("R_HIP")
    ls, rs = lm.get("L_SHOULDER"), lm.get("R_SHOULDER")
    head = lm.get("HEAD")

    if lh is None or rh is None:
        return frames

    pelvis_origin = _lerp(lh, rh, 0.5)
    hip_x = _v_norm(_v_sub(rh, lh))
    shoulder_mid = _lerp(ls, rs, 0.5) if ls and rs else pelvis_origin
    if head is not None:
        up_ref = _v_sub(shoulder_mid, pelvis_origin)
        if _v_len(up_ref) < 1e-6:
            up_ref = (0.0, 1.0, 0.0)
    else:
        up_ref = _v_sub(shoulder_mid, pelvis_origin)
    pelvis_y = _v_norm(up_ref, (0.0, 1.0, 0.0))
    pelvis_z = _v_norm(_v_cross(hip_x, pelvis_y), (0.0, 0.0, 1.0))
    pelvis_y = _v_norm(_v_cross(pelvis_z, hip_x), pelvis_y)
    frames["pelvis"] = AnatomicalFrame(pelvis_origin, hip_x, pelvis_y, pelvis_z)

    if ls and rs:
        torso_origin = _lerp(ls, rs, 0.5)
        torso_x = _v_norm(_v_sub(rs, ls))
        torso_y = _v_norm(_v_sub(torso_origin, pelvis_origin), (0.0, 1.0, 0.0))
        torso_z = _v_norm(_v_cross(torso_x, torso_y), (0.0, 0.0, 1.0))
        torso_y = _v_norm(_v_cross(torso_z, torso_x), torso_y)
        frames["torso"] = AnatomicalFrame(torso_origin, torso_x, torso_y, torso_z)

    for side, hip_key, knee_key, ankle_key, heel_key, toe_key in (
        ("left", "L_HIP", "L_KNEE", "L_ANKLE", "L_HEEL", "L_TOE"),
        ("right", "R_HIP", "R_KNEE", "R_ANKLE", "R_HEEL", "R_TOE"),
    ):
        hip, knee, ankle = lm.get(hip_key), lm.get(knee_key), lm.get(ankle_key)
        if hip and knee:
            thigh_y = _v_norm(_v_sub(knee, hip))
            lat_sign = -1.0 if side == "left" else 1.0
            thigh_x = _v_norm(_v_scale(hip_x, lat_sign))
            thigh_z = _v_norm(_v_cross(thigh_x, thigh_y))
            thigh_x = _v_norm(_v_cross(thigh_y, thigh_z), thigh_x)
            frames[f"{side}_thigh"] = AnatomicalFrame(hip, thigh_x, thigh_y, thigh_z)

        if knee and ankle:
            shank_y = _v_norm(_v_sub(ankle, knee))
            lat_sign = -1.0 if side == "left" else 1.0
            shank_x = _v_norm(_v_scale(hip_x, lat_sign))
            shank_z = _v_norm(_v_cross(shank_x, shank_y))
            shank_x = _v_norm(_v_cross(shank_y, shank_z), shank_x)
            frames[f"{side}_shank"] = AnatomicalFrame(knee, shank_x, shank_y, shank_z)

        heel, toe = lm.get(heel_key), lm.get(toe_key)
        foot_pts = [p for p in (ankle, heel, toe) if p is not None]
        if len(foot_pts) >= 2 and ankle:
            foot_origin = ankle
            if heel and toe:
                foot_y = _v_norm(_v_sub(toe, heel))
            elif toe:
                foot_y = _v_norm(_v_sub(toe, ankle))
            else:
                foot_y = _v_norm(_v_sub(heel, ankle))  # type: ignore[arg-type]
            lat_sign = -1.0 if side == "left" else 1.0
            foot_x = _v_norm(_v_scale(hip_x, lat_sign))
            foot_z = _v_norm(_v_cross(foot_x, foot_y))
            foot_x = _v_norm(_v_cross(foot_y, foot_z), foot_x)
            frames[f"{side}_foot"] = AnatomicalFrame(foot_origin, foot_x, foot_y, foot_z)

    return frames


def measure_segment_dimensions(lm: dict[str, Vec3]) -> SegmentDimensions:
    lh, rh = lm.get("L_HIP"), lm.get("R_HIP")
    ls, rs = lm.get("L_SHOULDER"), lm.get("R_SHOULDER")
    hip_w = _dist(lh, rh) if lh and rh else 200.0
    shoulder_w = _dist(ls, rs) if ls and rs else hip_w * 1.2

    def seg(a: str, b: str, default: float) -> float:
        pa, pb = lm.get(a), lm.get(b)
        return _dist(pa, pb) if pa and pb else default

    thigh_l = seg("L_HIP", "L_KNEE", hip_w * 1.1)
    thigh_r = seg("R_HIP", "R_KNEE", hip_w * 1.1)
    shank_l = seg("L_KNEE", "L_ANKLE", thigh_l * 0.85)
    shank_r = seg("R_KNEE", "R_ANKLE", thigh_r * 0.85)
    foot_l = seg("L_HEEL", "L_TOE", shank_l * 0.55)
    foot_r = seg("R_HEEL", "R_TOE", shank_r * 0.55)
    torso = seg("L_HIP", "HEAD", hip_w * 2.5) if lm.get("HEAD") else hip_w * 2.0

    return SegmentDimensions(
        hip_width=max(hip_w, 50.0),
        shoulder_width=max(shoulder_w, 50.0),
        thigh_length_left=max(thigh_l, 80.0),
        thigh_length_right=max(thigh_r, 80.0),
        shank_length_left=max(shank_l, 70.0),
        shank_length_right=max(shank_r, 70.0),
        foot_length_left=max(foot_l, 40.0),
        foot_length_right=max(foot_r, 40.0),
        torso_length=max(torso, 100.0),
    )


# ---------------------------------------------------------------------------
# Marker placement rules
# ---------------------------------------------------------------------------
def _derive_thigh_upper(side: str):
    key = f"{side}_thigh"

    def _fn(
        lm: dict[str, Vec3],
        frames: dict[str, AnatomicalFrame],
        dims: SegmentDimensions,
    ) -> tuple[Vec3 | None, float]:
        fr = frames.get(key)
        if fr is None:
            return None, 0.0
        length = dims.thigh_length_left if side == "left" else dims.thigh_length_right
        return _point_in_frame(fr, 0.0, 0.35 * length, 0.0), 0.68

    return _fn


def _derive_thigh_surface(side: str, anterior: bool):
    key = f"{side}_thigh"

    def _fn(
        lm: dict[str, Vec3],
        frames: dict[str, AnatomicalFrame],
        dims: SegmentDimensions,
    ) -> tuple[Vec3 | None, float]:
        fr = frames.get(key)
        if fr is None:
            return None, 0.0
        length = dims.thigh_length_left if side == "left" else dims.thigh_length_right
        t = 0.28 * length
        z_off = 0.14 * length if anterior else -0.14 * length
        return _point_in_frame(fr, 0.0, t, z_off), 0.58 if anterior else 0.52

    return _fn


def _derive_shank_upper(side: str):
    key = f"{side}_shank"

    def _fn(
        lm: dict[str, Vec3],
        frames: dict[str, AnatomicalFrame],
        dims: SegmentDimensions,
    ) -> tuple[Vec3 | None, float]:
        fr = frames.get(key)
        if fr is None:
            return None, 0.0
        length = dims.shank_length_left if side == "left" else dims.shank_length_right
        return _point_in_frame(fr, 0.0, 0.35 * length, 0.0), 0.66

    return _fn


def _derive_shank_rear(side: str):
    key = f"{side}_shank"

    def _fn(
        lm: dict[str, Vec3],
        frames: dict[str, AnatomicalFrame],
        dims: SegmentDimensions,
    ) -> tuple[Vec3 | None, float]:
        fr = frames.get(key)
        if fr is None:
            return None, 0.0
        length = dims.shank_length_left if side == "left" else dims.shank_length_right
        return _point_in_frame(fr, 0.0, 0.30 * length, -0.12 * length), 0.54

    return _fn


def _derive_midfoot(side: str, lateral: bool):
    key = f"{side}_foot"

    def _fn(
        lm: dict[str, Vec3],
        frames: dict[str, AnatomicalFrame],
        dims: SegmentDimensions,
    ) -> tuple[Vec3 | None, float]:
        fr = frames.get(key)
        if fr is None:
            return None, 0.0
        fl = dims.foot_length_left if side == "left" else dims.foot_length_right
        heel, toe = lm.get(f"{side[0].upper()}_HEEL"), lm.get(f"{side[0].upper()}_TOE")
        if lateral:
            if heel and lm.get(f"{side[0].upper()}_ANKLE"):
                mid = _lerp(lm[f"{side[0].upper()}_ANKLE"], heel, 0.5)  # type: ignore[index]
                return mid, 0.62
            return _point_in_frame(fr, 0.12 * fl, 0.45 * fl, 0.0), 0.55
        pts = [
            lm.get(f"{side[0].upper()}_ANKLE"),
            heel,
            toe,
        ]
        valid = [p for p in pts if p is not None]
        if len(valid) < 2:
            return None, 0.0
        cx = sum(p[0] for p in valid) / len(valid)
        cy = sum(p[1] for p in valid) / len(valid)
        cz = sum(p[2] for p in valid) / len(valid)
        return (cx, cy + 0.04 * fl, cz), 0.60

    return _fn


def _derive_toe_offset(side: str, medial: bool):
    prefix = "L" if side == "left" else "R"

    def _fn(
        lm: dict[str, Vec3],
        frames: dict[str, AnatomicalFrame],
        dims: SegmentDimensions,
    ) -> tuple[Vec3 | None, float]:
        toe = lm.get(f"{prefix}_TOE")
        fr = frames.get(f"{side}_foot")
        if toe is None or fr is None:
            return None, 0.0
        fl = dims.foot_length_left if side == "left" else dims.foot_length_right
        sign = -1.0 if medial else 1.0
        lat = sign * 0.10 * fl
        return _point_in_frame(fr, lat, 0.92 * fl, 0.0), 0.48

    return _fn


def _derive_v_sacral(
    lm: dict[str, Vec3],
    frames: dict[str, AnatomicalFrame],
    dims: SegmentDimensions,
) -> tuple[Vec3 | None, float]:
    fr = frames.get("pelvis")
    if fr is None:
        return None, 0.0
    return (
        _point_in_frame(fr, 0.0, -0.06 * dims.hip_width, -0.10 * dims.hip_width),
        0.64,
    )


def _derive_sternum(
    lm: dict[str, Vec3],
    frames: dict[str, AnatomicalFrame],
    dims: SegmentDimensions,
) -> tuple[Vec3 | None, float]:
    fr = frames.get("torso")
    if fr is None:
        ls, rs = lm.get("L_SHOULDER"), lm.get("R_SHOULDER")
        if ls and rs:
            mid = _lerp(ls, rs, 0.5)
            head = lm.get("HEAD")
            if head:
                return _lerp(mid, head, 0.25), 0.55
            return mid, 0.50
        return None, 0.0
    return _point_in_frame(fr, 0.0, -0.08 * dims.torso_length, 0.18 * dims.shoulder_width), 0.62


def build_marker_catalog() -> list[MarkerSpec]:
    """Static catalog for all Gait2392 markers."""
    specs: list[MarkerSpec] = []

    direct_map = {
        "R.Acromium": ("R_SHOULDER", 0.88, "LOW"),
        "L.Acromium": ("L_SHOULDER", 0.88, "LOW"),
        "Top.Head": ("HEAD", 0.82, "LOW"),
        "R.ASIS": ("R_HIP", 0.85, "LOW"),
        "L.ASIS": ("L_HIP", 0.85, "LOW"),
        "R.Heel": ("R_HEEL", 0.86, "LOW"),
        "L.Heel": ("L_HEEL", 0.86, "LOW"),
        "R.Toe.Tip": ("R_TOE", 0.84, "LOW"),
        "L.Toe.Tip": ("L_TOE", 0.84, "LOW"),
        "R.Shank.Front": ("R_ANKLE", 0.80, "MODERATE"),
        "L.Shank.Front": ("L_ANKLE", 0.80, "MODERATE"),
    }
    for name, (src, conf, risk) in direct_map.items():
        specs.append(
            MarkerSpec(
                name,
                "DIRECT",
                (src,),
                f"Direct MediaPipe landmark {src}",
                conf,
                risk,  # type: ignore[arg-type]
                lambda lm, fr, ds, s=src, c=conf: _direct(lm, s, c),
            )
        )

    derived: list[tuple[str, tuple[str, ...], str, float, BiomechanicalRisk, Callable]] = [
        ("V.Sacral", ("L_HIP", "R_HIP"), "Pelvis frame posterior-inferior offset scaled by hip width", 0.64, "MODERATE", _derive_v_sacral),
        ("Sternum", ("L_SHOULDER", "R_SHOULDER", "HEAD"), "Torso frame anterior offset from shoulder line", 0.62, "MODERATE", _derive_sternum),
        ("R.Thigh.Upper", ("R_HIP", "R_KNEE"), "35% along thigh segment (hip→knee)", 0.68, "MODERATE", _derive_thigh_upper("right")),
        ("L.Thigh.Upper", ("L_HIP", "L_KNEE"), "35% along thigh segment (hip→knee)", 0.68, "MODERATE", _derive_thigh_upper("left")),
        ("R.Thigh.Front", ("R_HIP", "R_KNEE"), "Thigh frame anterior surface (~28% segment)", 0.58, "MODERATE", _derive_thigh_surface("right", True)),
        ("L.Thigh.Front", ("L_HIP", "L_KNEE"), "Thigh frame anterior surface (~28% segment)", 0.58, "MODERATE", _derive_thigh_surface("left", True)),
        ("R.Thigh.Rear", ("R_HIP", "R_KNEE"), "Thigh frame posterior surface (~28% segment)", 0.52, "HIGH", _derive_thigh_surface("right", False)),
        ("L.Thigh.Rear", ("L_HIP", "L_KNEE"), "Thigh frame posterior surface (~28% segment)", 0.52, "HIGH", _derive_thigh_surface("left", False)),
        ("R.Shank.Upper", ("R_KNEE", "R_ANKLE"), "35% along shank segment (knee→ankle)", 0.66, "MODERATE", _derive_shank_upper("right")),
        ("L.Shank.Upper", ("L_KNEE", "L_ANKLE"), "35% along shank segment (knee→ankle)", 0.66, "MODERATE", _derive_shank_upper("left")),
        ("R.Shank.Rear", ("R_KNEE", "R_ANKLE"), "Shank frame posterior offset (~30% segment)", 0.54, "HIGH", _derive_shank_rear("right")),
        ("L.Shank.Rear", ("L_KNEE", "L_ANKLE"), "Shank frame posterior offset (~30% segment)", 0.54, "HIGH", _derive_shank_rear("left")),
        ("R.Midfoot.Sup", ("R_ANKLE", "R_HEEL", "R_TOE"), "Foot frame dorsal centroid", 0.60, "MODERATE", _derive_midfoot("right", False)),
        ("L.Midfoot.Sup", ("L_ANKLE", "L_HEEL", "L_TOE"), "Foot frame dorsal centroid", 0.60, "MODERATE", _derive_midfoot("left", False)),
        ("R.Midfoot.Lat", ("R_ANKLE", "R_HEEL"), "Foot frame lateral midfoot", 0.55, "MODERATE", _derive_midfoot("right", True)),
        ("L.Midfoot.Lat", ("L_ANKLE", "L_HEEL"), "Foot frame lateral midfoot", 0.55, "MODERATE", _derive_midfoot("left", True)),
        ("R.Toe.Lat", ("R_TOE", "R_HEEL"), "Foot frame lateral toe offset", 0.48, "HIGH", _derive_toe_offset("right", False)),
        ("L.Toe.Lat", ("L_TOE", "L_HEEL"), "Foot frame lateral toe offset", 0.48, "HIGH", _derive_toe_offset("left", False)),
        ("R.Toe.Med", ("R_TOE", "R_HEEL"), "Foot frame medial toe offset", 0.48, "HIGH", _derive_toe_offset("right", True)),
        ("L.Toe.Med", ("L_TOE", "L_HEEL"), "Foot frame medial toe offset", 0.48, "HIGH", _derive_toe_offset("left", True)),
    ]
    for name, sources, calc, conf, risk, fn in derived:
        specs.append(
            MarkerSpec(name, "DERIVED_ANATOMICAL", sources, calc, conf, risk, fn)
        )

    return specs


MARKER_CATALOG: list[MarkerSpec] = build_marker_catalog()
CATALOG_BY_NAME: dict[str, MarkerSpec] = {s.opensim_name: s for s in MARKER_CATALOG}


# ---------------------------------------------------------------------------
# Temporal processing
# ---------------------------------------------------------------------------
def _moving_average(values: list[float | None], window: int) -> list[float | None]:
    w = max(1, window | 1)
    half = w // 2
    out: list[float | None] = []
    for i in range(len(values)):
        chunk = [
            values[j]
            for j in range(max(0, i - half), min(len(values), i + half + 1))
            if values[j] is not None
        ]
        out.append(float(statistics.mean(chunk)) if chunk else None)
    return out


def _smooth_marker_trajectories(
    raw_series: dict[str, list[Vec3 | None]],
    *,
    window: int,
) -> dict[str, list[Vec3 | None]]:
    smoothed: dict[str, list[Vec3 | None]] = {}
    for name, pts in raw_series.items():
        xs = _moving_average([p[0] if p else None for p in pts], window)
        ys = _moving_average([p[1] if p else None for p in pts], window)
        zs = _moving_average([p[2] if p else None for p in pts], window)
        smoothed[name] = [
            (x, y, z) if x is not None and y is not None and z is not None else None
            for x, y, z in zip(xs, ys, zs)
        ]
    return smoothed


def _repair_jumps(
    values: list[Vec3 | None],
    *,
    max_jump_mm: float,
) -> tuple[list[Vec3 | None], list[bool]]:
    """Replace implausible single-frame spikes with neighbor interpolation."""
    out = list(values)
    repaired = [False] * len(values)
    for i in range(1, len(out) - 1):
        cur = out[i]
        prev = out[i - 1]
        nxt = out[i + 1]
        if cur is None or prev is None or nxt is None:
            continue
        if _dist(cur, prev) > max_jump_mm or _dist(cur, nxt) > max_jump_mm:
            out[i] = _lerp(prev, nxt, 0.5)
            repaired[i] = True
    return out, repaired


def _interpolate_gaps(values: list[Vec3 | None]) -> tuple[list[Vec3 | None], list[bool]]:
    out = list(values)
    interpolated = [False] * len(values)
    i = 0
    while i < len(out):
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] is None:
            j += 1
        left = out[i - 1] if i > 0 else None
        right = out[j] if j < len(out) else None
        if left and right:
            gap = j - i
            for k in range(gap):
                t = (k + 1) / (gap + 1)
                out[i + k] = _lerp(left, right, t)
                interpolated[i + k] = True
        i = max(j, i + 1)
    return out, interpolated


def _sanity_check_trajectories(
    series: list[Vec3 | None],
    times: list[float],
    *,
    cfg: MarkerReconstructionConfig,
) -> list[str]:
    warnings: list[str] = []
    for i in range(1, len(series)):
        if series[i] is None or series[i - 1] is None:
            continue
        dt = max(times[i] - times[i - 1], 1e-6)
        disp = _dist(series[i], series[i - 1])  # type: ignore[arg-type]
        vel = disp / dt
        if disp > cfg.max_jump_mm:
            warnings.append(f"frame {i}: jump {disp:.0f} mm")
        if vel > cfg.max_velocity_mm_s:
            warnings.append(f"frame {i}: velocity {vel:.0f} mm/s")
        if i >= 2 and series[i - 2] is not None:
            v1 = _v_scale(_v_sub(series[i - 1], series[i - 2]), 1.0 / dt)  # type: ignore[arg-type]
            v2 = _v_scale(_v_sub(series[i], series[i - 1]), 1.0 / dt)  # type: ignore[arg-type]
            acc = _v_len(_v_sub(v2, v1)) / dt
            if acc > cfg.max_acceleration_mm_s2:
                warnings.append(f"frame {i}: acceleration {acc:.0f} mm/s²")
    return warnings


# ---------------------------------------------------------------------------
# Session reconstruction
# ---------------------------------------------------------------------------
def reconstruct_markers_from_trc_frames(
    frames: list[tuple[int, float, dict[str, Vec3]]],
    *,
    fps: float,
    config: MarkerReconstructionConfig | None = None,
) -> MarkerReconstructionResult:
    """
    Reconstruct all catalog markers from raw StableWalk TRC frame data.

    ``frames``: list of (frame_num, time_s, {L_HIP: (x,y,z), ...}).
    """
    cfg = config or DEFAULT_RECONSTRUCTION_CONFIG
    catalog = MARKER_CATALOG
    names = [s.opensim_name for s in catalog if s.opensim_name in GAIT2392_MARKER_NAMES]

    raw_per_marker: dict[str, list[Vec3 | None]] = {n: [] for n in names}
    derived_per_marker: dict[str, list[MarkerFrameResult]] = {n: [] for n in names}
    times: list[float] = []

    for _fn, time_s, landmarks in frames:
        times.append(time_s)
        dims = measure_segment_dimensions(landmarks)
        seg_frames = build_anatomical_frames(landmarks, dims)
        for spec in catalog:
            if spec.opensim_name not in names:
                continue
            pos, conf = spec.compute(landmarks, seg_frames, dims)
            raw_per_marker[spec.opensim_name].append(pos)
            derived_per_marker[spec.opensim_name].append(
                MarkerFrameResult(
                    position=pos,
                    confidence=conf if pos is not None else 0.0,
                    mapping_type=spec.mapping_type,
                    source_landmarks=list(spec.source_landmarks),
                    warnings=[],
                )
            )

    filtered = _smooth_marker_trajectories(raw_per_marker, window=cfg.filter_window)

    final_series: dict[str, list[Vec3 | None]] = {}
    interp_flags: dict[str, list[bool]] = {}
    session_warnings: list[str] = []

    for name in names:
        jump_repaired, jump_flags = _repair_jumps(
            filtered[name],
            max_jump_mm=cfg.max_jump_mm,
        )
        smoothed, gap_flags = _interpolate_gaps(jump_repaired)
        final_series[name] = smoothed
        interp_flags[name] = [g or j for g, j in zip(gap_flags, jump_flags)]
        w = _sanity_check_trajectories(smoothed, times, cfg=cfg)
        if w:
            session_warnings.extend([f"{name}: {x}" for x in w[:3]])

    out_frames: list[tuple[int, float, dict[str, MarkerFrameResult]]] = []
    trajectory_comparison: dict[str, dict[str, list[float]]] = {}

    for i, (frame_num, time_s, _lm) in enumerate(frames):
        frame_markers: dict[str, MarkerFrameResult] = {}
        for name in names:
            spec = CATALOG_BY_NAME[name]
            raw = raw_per_marker[name][i]
            derived = derived_per_marker[name][i]
            filt = final_series[name][i]
            mtype: MappingType = spec.mapping_type
            conf = derived.confidence
            warnings: list[str] = []
            interpolated = interp_flags[name][i]
            was_filtered = raw != filt and raw is not None and filt is not None

            if interpolated:
                mtype = "TEMPORAL_ESTIMATED"
                conf = min(conf, 0.50)
                warnings.append("Filled by temporal interpolation")

            if filt is None:
                frame_markers[name] = MarkerFrameResult(
                    None, 0.0, "UNAVAILABLE", list(spec.source_landmarks), warnings
                )
                continue

            if was_filtered:
                warnings.append("Temporal smoothing applied")

            frame_markers[name] = MarkerFrameResult(
                filt,
                conf,
                mtype,
                list(spec.source_landmarks),
                warnings,
                was_interpolated=interpolated,
                was_filtered=was_filtered,
            )

            if i == 0:
                trajectory_comparison[name] = {
                    "raw_x": [], "filtered_x": [],
                    "raw_y": [], "filtered_y": [],
                }
            tc = trajectory_comparison[name]
            tc["raw_x"].append(raw[0] if raw else float("nan"))
            tc["filtered_x"].append(filt[0])
            tc["raw_y"].append(raw[1] if raw else float("nan"))
            tc["filtered_y"].append(filt[1])

        out_frames.append((frame_num, time_s, frame_markers))

    ref_count = len(GAIT2392_MARKER_NAMES)
    present_names = [
        n for n in names
        if any(
            out_frames[i][2][n].position is not None
            for i in range(len(out_frames))
        )
    ]
    raw_coverage = 100.0 * len(present_names) / ref_count

    direct_n = sum(
        1 for n in present_names
        if CATALOG_BY_NAME[n].mapping_type == "DIRECT"
    )
    derived_n = sum(
        1 for n in present_names
        if CATALOG_BY_NAME[n].mapping_type == "DERIVED_ANATOMICAL"
    )
    temporal_n = sum(
        1 for n in present_names
        if any(
            out_frames[i][2][n].mapping_type == "TEMPORAL_ESTIMATED"
            for i in range(len(out_frames))
        )
    )
    unavailable_n = ref_count - len(present_names)

    hc_names = [
        n for n in present_names
        if statistics.mean(
            [
                out_frames[i][2][n].confidence
                for i in range(len(out_frames))
                if out_frames[i][2][n].position is not None
            ]
        ) >= cfg.high_confidence_threshold
    ]
    hc_coverage = 100.0 * len(hc_names) / ref_count

    ik_score = _compute_ik_readiness_score(names, out_frames, ref_count, cfg)
    readiness = _ik_readiness_level(ik_score, cfg)

    low_conf = sorted(
        n for n in present_names
        if statistics.mean(
            [out_frames[i][2][n].confidence for i in range(len(out_frames)) if out_frames[i][2][n].position is not None]
        ) < cfg.high_confidence_threshold
    )

    catalog_rows = format_mapping_catalog_table()

    return MarkerReconstructionResult(
        marker_names=names,
        frames=out_frames,
        fps=fps,
        units="mm",
        catalog_rows=catalog_rows,
        warnings=session_warnings[:20],
        direct_count=direct_n,
        derived_count=derived_n,
        temporal_count=temporal_n,
        unavailable_count=unavailable_n,
        raw_coverage_percent=round(raw_coverage, 1),
        high_confidence_coverage_percent=round(hc_coverage, 1),
        ik_readiness=readiness,
        ik_readiness_score=round(ik_score, 1),
        low_confidence_markers=low_conf,
        trajectory_comparison=trajectory_comparison,
    )


def _compute_ik_readiness_score(
    marker_names: list[str],
    frames: list[tuple[int, float, dict[str, MarkerFrameResult]]],
    ref_count: int,
    cfg: MarkerReconstructionConfig,
) -> float:
    if not frames:
        return 0.0
    type_weight = {
        "DIRECT": 1.0,
        "DERIVED_ANATOMICAL": 0.72,
        "TEMPORAL_ESTIMATED": 0.45,
        "UNAVAILABLE": 0.0,
    }
    total = 0.0
    for name in marker_names:
        confs = [
            frames[i][2][name].confidence
            for i in range(len(frames))
            if frames[i][2][name].position is not None
        ]
        if not confs:
            continue
        mean_conf = statistics.mean(confs)
        mtype = CATALOG_BY_NAME[name].mapping_type
        total += mean_conf * type_weight.get(mtype, 0.5)
    return 100.0 * total / max(ref_count, 1)


def _ik_readiness_level(score: float, cfg: MarkerReconstructionConfig) -> IkReadinessLevel:
    if score >= cfg.ik_readiness_high:
        return IkReadinessLevel.HIGH
    if score >= cfg.ik_readiness_moderate:
        return IkReadinessLevel.MODERATE
    if score >= 25.0:
        return IkReadinessLevel.LOW
    return IkReadinessLevel.NOT_READY


def format_mapping_catalog_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in MARKER_CATALOG:
        rows.append({
            "OpenSim Marker": spec.opensim_name,
            "Mapping Type": spec.mapping_type,
            "Source MediaPipe Landmarks": ", ".join(spec.source_landmarks),
            "Calculation": spec.calculation,
            "Confidence": f"{spec.confidence_base:.2f}",
            "Biomechanical Risk": spec.biomechanical_risk,
        })
    for name in GAIT2392_MARKER_NAMES:
        if name not in CATALOG_BY_NAME:
            rows.append({
                "OpenSim Marker": name,
                "Mapping Type": "UNAVAILABLE",
                "Source MediaPipe Landmarks": "",
                "Calculation": "Not reconstructable from monocular MediaPipe",
                "Confidence": "0.00",
                "Biomechanical Risk": "N/A",
            })
    return rows


def mapping_catalog_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def write_mapped_trc_from_reconstruction(
    result: MarkerReconstructionResult,
    output_path: Path,
    *,
    data_rate: str = "30.0",
) -> Path:
    """Write OpenSim TRC from reconstructed marker trajectories."""
    output_path = Path(output_path)
    names = result.marker_names
    n_frames = len(result.frames)
    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{output_path.name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{data_rate}\t{data_rate}\t{n_frames}\t{len(names)}\t{result.units}\t{data_rate}\t1\t{n_frames}",
        "Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t",
        "\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, len(names) + 1)),
        "",
    ]
    for frame_num, time_s, markers in result.frames:
        cells = [str(frame_num), f"{time_s:.6f}"]
        for name in names:
            m = markers.get(name)
            if m and m.position:
                cells.extend(f"{c:.6f}" for c in m.position)
            else:
                cells.extend(["", "", ""])
        lines.append("\t".join(cells))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


# Skeleton joint id → StableWalk TRC landmark name (for debug visualization).
SKELETON_JOINT_TO_LANDMARK: dict[str, str] = {
    "left_shoulder": "L_SHOULDER",
    "right_shoulder": "R_SHOULDER",
    "head": "HEAD",
    "left_hip": "L_HIP",
    "right_hip": "R_HIP",
    "left_knee": "L_KNEE",
    "right_knee": "R_KNEE",
    "left_ankle": "L_ANKLE",
    "right_ankle": "R_ANKLE",
    "left_heel": "L_HEEL",
    "right_heel": "R_HEEL",
    "left_toe": "L_TOE",
    "right_toe": "R_TOE",
}


def landmarks_from_skeleton_joints(
    joints: dict[str, tuple[float, float, float]],
) -> dict[str, Vec3]:
    """Convert canonical skeleton joint positions to TRC-style landmark dict."""
    out: dict[str, Vec3] = {}
    for jid, lm_name in SKELETON_JOINT_TO_LANDMARK.items():
        pos = joints.get(jid)
        if pos is not None:
            out[lm_name] = pos
    return out


def reconstruct_markers_single_frame(
    landmarks: dict[str, Vec3],
) -> dict[str, MarkerFrameResult]:
    """Anatomical marker placement for one frame (no temporal filtering)."""
    dims = measure_segment_dimensions(landmarks)
    seg_frames = build_anatomical_frames(landmarks, dims)
    results: dict[str, MarkerFrameResult] = {}
    for spec in MARKER_CATALOG:
        if spec.opensim_name not in GAIT2392_MARKER_NAMES:
            continue
        pos, conf = spec.compute(landmarks, seg_frames, dims)
        results[spec.opensim_name] = MarkerFrameResult(
            position=pos,
            confidence=conf if pos is not None else 0.0,
            mapping_type=spec.mapping_type if pos is not None else "UNAVAILABLE",
            source_landmarks=list(spec.source_landmarks),
        )
    return results


def project_marker_to_display_xy(
    position: Vec3,
    *,
    display_mode: str = "biomechanical",
) -> tuple[float, float]:
    """Project a 3D marker into skeleton display coordinates."""
    x, y, z = position
    if display_mode in ("3d_normalized", "biomechanical"):
        return (x + 0.22 * z, y)
    return (x, y)


__all__ = [
    "MarkerReconstructionConfig",
    "DEFAULT_RECONSTRUCTION_CONFIG",
    "MarkerReconstructionResult",
    "MarkerFrameResult",
    "MarkerSpec",
    "MappingType",
    "IkReadinessLevel",
    "GAIT2392_MARKER_NAMES",
    "MARKER_CATALOG",
    "CATALOG_BY_NAME",
    "reconstruct_markers_from_trc_frames",
    "write_mapped_trc_from_reconstruction",
    "format_mapping_catalog_table",
    "mapping_catalog_csv",
    "build_anatomical_frames",
    "measure_segment_dimensions",
    "SKELETON_JOINT_TO_LANDMARK",
    "landmarks_from_skeleton_joints",
    "reconstruct_markers_single_frame",
    "project_marker_to_display_xy",
]
