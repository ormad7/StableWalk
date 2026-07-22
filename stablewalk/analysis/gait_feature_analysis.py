"""
Anthropometrically normalized gait features and gait-cycle trajectory analysis.

Feature normalization tiers (documented on each output field):

* **RAW** — absolute values in pose/recording units (meters, seconds, degrees).
* **BODY_NORMALIZED** — divided by robust median segment dimensions (leg length,
  hip width, shoulder width) estimated from many high-confidence frames.
* **GAIT_CYCLE_NORMALIZED** — resampled to 0–100% gait cycle (101 samples),
  comparable across walking speeds and frame rates.

Does not compare subjects using raw spatial measurements alone.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np

from stablewalk.analysis.gait_cycle_analysis import (
    DetectedGaitCycle,
    GaitCycleAnalysisResult,
    GaitEvent,
    symmetry_ratio,
)
from stablewalk.analysis.ground_reference import (
    GroundReferencePlane,
    estimate_ground_plane,
    foot_clearance_m,
    vertical_coordinate,
)
from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot, Vec3
from stablewalk.models.joint_registry import ROOT_JOINT_ID
from stablewalk.models.pose_data import PoseFrame, PoseSequence
from stablewalk.pose.kinematics import compute_joint_angles

logger = logging.getLogger(__name__)

GAIT_CYCLE_SAMPLE_COUNT = 101
MIN_ANTHRO_FRAMES = 8
MIN_JOINT_VISIBILITY = 0.45
SYMMETRY_EPS = 1e-6


class FeatureNormalization(str, Enum):
    RAW = "RAW"
    BODY_NORMALIZED = "BODY_NORMALIZED"
    GAIT_CYCLE_NORMALIZED = "GAIT_CYCLE_NORMALIZED"


AngleSource = Literal["mediapipe_angles", "opensim_ik", "unavailable"]


@dataclass(frozen=True)
class BodySegmentDimensions:
    """Robust median body dimensions from high-confidence 3D pose frames."""

    hip_width: float
    shoulder_width: float
    leg_length_left: float
    leg_length_right: float
    leg_length_average: float
    thigh_length_left: float
    thigh_length_right: float
    shank_length_left: float
    shank_length_right: float
    frame_count_used: int = 0
    confidence_tier: str = "LOW"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hip_width_m": self.hip_width,
            "shoulder_width_m": self.shoulder_width,
            "leg_length_left_m": self.leg_length_left,
            "leg_length_right_m": self.leg_length_right,
            "leg_length_average_m": self.leg_length_average,
            "thigh_length_left_m": self.thigh_length_left,
            "thigh_length_right_m": self.thigh_length_right,
            "shank_length_left_m": self.shank_length_left,
            "shank_length_right_m": self.shank_length_right,
            "frame_count_used": self.frame_count_used,
            "confidence_tier": self.confidence_tier,
        }


@dataclass
class NormalizedGaitFeatures:
    """Spatial/temporal gait descriptors with explicit normalization metadata."""

    step_length_m: float | None = None
    stride_length_m: float | None = None
    foot_clearance_left_m: float | None = None
    foot_clearance_right_m: float | None = None
    pelvis_mediolateral_range_m: float | None = None
    pelvis_vertical_range_m: float | None = None
    trunk_sway_m: float | None = None

    normalized_step_length: float | None = None
    normalized_stride_length: float | None = None
    normalized_foot_clearance_left: float | None = None
    normalized_foot_clearance_right: float | None = None
    normalized_pelvis_sway: float | None = None
    normalized_vertical_pelvis_motion: float | None = None
    normalized_trunk_sway: float | None = None

    step_length_symmetry: float | None = None
    stride_length_symmetry: float | None = None
    foot_clearance_symmetry: float | None = None
    leg_length_symmetry: float | None = None

    normalization: dict[str, FeatureNormalization] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in (
            "step_length_m",
            "stride_length_m",
            "foot_clearance_left_m",
            "foot_clearance_right_m",
            "pelvis_mediolateral_range_m",
            "pelvis_vertical_range_m",
            "trunk_sway_m",
            "normalized_step_length",
            "normalized_stride_length",
            "normalized_foot_clearance_left",
            "normalized_foot_clearance_right",
            "normalized_pelvis_sway",
            "normalized_vertical_pelvis_motion",
            "normalized_trunk_sway",
            "step_length_symmetry",
            "stride_length_symmetry",
            "foot_clearance_symmetry",
            "leg_length_symmetry",
        ):
            out[key] = getattr(self, key)
        out["normalization"] = {k: v.value for k, v in self.normalization.items()}
        return out


@dataclass
class CycleTrajectory:
    """One signal resampled to 0–100% gait cycle."""

    name: str
    percent: tuple[float, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    per_cycle: list[tuple[float, ...]]
    normalization: FeatureNormalization = FeatureNormalization.GAIT_CYCLE_NORMALIZED
    source: AngleSource = "mediapipe_angles"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "normalization": self.normalization.value,
            "source": self.source,
            "sample_count": len(self.percent),
            "mean": list(self.mean),
            "std": list(self.std),
            "cycle_count": len(self.per_cycle),
        }


@dataclass
class CycleConsistencyResult:
    """Cross-cycle trajectory comparison at 0–100% gait phase."""

    trajectories: dict[str, CycleTrajectory] = field(default_factory=dict)
    cycle_to_cycle_rmse: dict[str, float] = field(default_factory=dict)
    left_right_phase_consistency: float | None = None
    cycle_repeatability_score: float | None = None
    cycle_count: int = 0
    angle_source: AngleSource = "mediapipe_angles"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectories": {k: v.to_dict() for k, v in self.trajectories.items()},
            "cycle_to_cycle_rmse": self.cycle_to_cycle_rmse,
            "left_right_phase_consistency": self.left_right_phase_consistency,
            "cycle_repeatability_score": self.cycle_repeatability_score,
            "cycle_count": self.cycle_count,
            "angle_source": self.angle_source,
            "warnings": list(self.warnings),
        }


@dataclass
class GaitFeatureAnalysisResult:
    """Combined normalized gait feature output."""

    dimensions: BodySegmentDimensions
    features: NormalizedGaitFeatures
    cycle_consistency: CycleConsistencyResult
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": self.dimensions.to_dict(),
            "features": self.features.to_dict(),
            "cycle_consistency": self.cycle_consistency.to_dict(),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def safe_divide(numerator: float, denominator: float, *, default: float | None = None) -> float | None:
    if abs(denominator) < SYMMETRY_EPS:
        return default
    return numerator / denominator


def symmetry_index(
    left: float | None,
    right: float | None,
    *,
    eps: float = SYMMETRY_EPS,
) -> float | None:
    """
    General left-right symmetry index in [0, 1].

    Uses ``2 * min(L, R) / (L + R + eps)`` — stable when values are near zero.
    """
    if left is None or right is None:
        return None
    if left < 0 or right < 0:
        return None
    denom = left + right + eps
    if denom <= eps:
        return None
    return float(2.0 * min(left, right) / denom)


def _joint_pos(snapshot: SkeletonSnapshot, joint_id: str) -> Vec3 | None:
    sample = snapshot.joints.get(joint_id)
    return sample.position if sample else None


def _distance(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _frame_confidence(snapshot: SkeletonSnapshot) -> float:
    """Proxy confidence from available lower-body joints."""
    keys = (
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    present = sum(1 for k in keys if k in snapshot.joints)
    return present / len(keys)


def _collect_segment_samples(
    recording: GaitMotionRecording,
    *,
    min_confidence: float = MIN_JOINT_VISIBILITY,
) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {
        "hip_width": [],
        "shoulder_width": [],
        "leg_length_left": [],
        "leg_length_right": [],
        "thigh_length_left": [],
        "thigh_length_right": [],
        "shank_length_left": [],
        "shank_length_right": [],
    }
    for snap in recording.snapshots:
        if _frame_confidence(snap) < min_confidence:
            continue
        lh = _joint_pos(snap, "left_hip")
        rh = _joint_pos(snap, "right_hip")
        ls = _joint_pos(snap, "left_shoulder")
        rs = _joint_pos(snap, "right_shoulder")
        lk = _joint_pos(snap, "left_knee")
        rk = _joint_pos(snap, "right_knee")
        la = _joint_pos(snap, "left_ankle")
        ra = _joint_pos(snap, "right_ankle")

        if lh and rh:
            samples["hip_width"].append(_distance(lh, rh))
        if ls and rs:
            samples["shoulder_width"].append(_distance(ls, rs))
        if lh and la:
            samples["leg_length_left"].append(_distance(lh, la))
        if rh and ra:
            samples["leg_length_right"].append(_distance(rh, ra))
        if lh and lk:
            samples["thigh_length_left"].append(_distance(lh, lk))
        if rh and rk:
            samples["thigh_length_right"].append(_distance(rh, rk))
        if lk and la:
            samples["shank_length_left"].append(_distance(lk, la))
        if rk and ra:
            samples["shank_length_right"].append(_distance(rk, ra))

    return samples


def estimate_body_segment_dimensions(
    recording: GaitMotionRecording,
    *,
    min_confidence: float = MIN_JOINT_VISIBILITY,
) -> BodySegmentDimensions:
    """
    Robust median segment dimensions from multiple high-confidence frames.

    Never uses a single frame estimate.
    """
    samples = _collect_segment_samples(recording, min_confidence=min_confidence)
    used = max(len(samples["hip_width"]), len(samples["leg_length_left"]))

    def med(key: str, fallback: float) -> float:
        vals = samples[key]
        return float(np.median(vals)) if vals else fallback

    hip = med("hip_width", 0.25)
    shoulder = med("shoulder_width", hip * 1.2)
    leg_l = med("leg_length_left", hip * 1.1)
    leg_r = med("leg_length_right", hip * 1.1)
    leg_avg = (leg_l + leg_r) / 2.0

    tier = "HIGH" if used >= MIN_ANTHRO_FRAMES else ("MEDIUM" if used >= 4 else "LOW")

    return BodySegmentDimensions(
        hip_width=max(hip, SYMMETRY_EPS),
        shoulder_width=max(shoulder, SYMMETRY_EPS),
        leg_length_left=max(leg_l, SYMMETRY_EPS),
        leg_length_right=max(leg_r, SYMMETRY_EPS),
        leg_length_average=max(leg_avg, SYMMETRY_EPS),
        thigh_length_left=max(med("thigh_length_left", leg_l * 0.45), SYMMETRY_EPS),
        thigh_length_right=max(med("thigh_length_right", leg_r * 0.45), SYMMETRY_EPS),
        shank_length_left=max(med("shank_length_left", leg_l * 0.45), SYMMETRY_EPS),
        shank_length_right=max(med("shank_length_right", leg_r * 0.45), SYMMETRY_EPS),
        frame_count_used=used,
        confidence_tier=tier,
    )


def _pelvis_position(snapshot: SkeletonSnapshot) -> Vec3 | None:
    pelvis = _joint_pos(snapshot, ROOT_JOINT_ID)
    if pelvis:
        return pelvis
    lh = _joint_pos(snapshot, "left_hip")
    rh = _joint_pos(snapshot, "right_hip")
    if lh and rh:
        return Vec3(
            x=(lh.x + rh.x) / 2.0,
            y=(lh.y + rh.y) / 2.0,
            z=(lh.z + rh.z) / 2.0,
        )
    return None


def _shoulder_mid(snapshot: SkeletonSnapshot) -> Vec3 | None:
    ls = _joint_pos(snapshot, "left_shoulder")
    rs = _joint_pos(snapshot, "right_shoulder")
    if ls and rs:
        return Vec3(
            x=(ls.x + rs.x) / 2.0,
            y=(ls.y + rs.y) / 2.0,
            z=(ls.z + rs.z) / 2.0,
        )
    return None


def resample_cycle_trajectory(
    times: list[float],
    values: list[float],
    *,
    t_start: float,
    t_end: float,
    n_samples: int = GAIT_CYCLE_SAMPLE_COUNT,
) -> np.ndarray | None:
    """Resample ``(time, value)`` pairs to ``n_samples`` evenly spaced 0–100%."""
    if t_end <= t_start or len(times) < 3 or len(values) != len(times):
        return None
    t_arr = np.asarray(times, dtype=float)
    v_arr = np.asarray(values, dtype=float)
    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    v_arr = v_arr[order]
    grid_t = np.linspace(t_start, t_end, n_samples)
    return np.interp(grid_t, t_arr, v_arr)


def gait_cycle_percent_grid(n_samples: int = GAIT_CYCLE_SAMPLE_COUNT) -> tuple[float, ...]:
    return tuple(float(i) * 100.0 / (n_samples - 1) for i in range(n_samples))


def _step_lengths_from_events(
    recording: GaitMotionRecording,
    events: list[GaitEvent],
) -> tuple[list[float], list[float]]:
    """Progression-axis foot separation at heel strike."""
    from stablewalk.analysis.biomechanical.walking_speed import _step_lengths_from_foot_events

    return _step_lengths_from_foot_events(recording, events)


def compute_normalized_gait_features(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult,
    dimensions: BodySegmentDimensions,
) -> NormalizedGaitFeatures:
    """Compute RAW and BODY_NORMALIZED spatial gait features."""
    features = NormalizedGaitFeatures()
    norm = features.normalization

    complete_events = [
        event
        for event in cycles.events
        if any(
            cycle.start_time_s <= event.time_s <= cycle.end_time_s
            for cycle in cycles.cycles
        )
    ]
    if cycles.metrics.metrics_reliable:
        left_steps, right_steps = _step_lengths_from_events(recording, complete_events)
    else:
        left_steps, right_steps = [], []
    step_l = statistics.mean(left_steps) if left_steps else None
    step_r = statistics.mean(right_steps) if right_steps else None
    if step_l is not None and step_r is not None:
        features.step_length_m = (step_l + step_r) / 2.0
        norm["step_length_m"] = FeatureNormalization.RAW
        features.stride_length_m = step_l + step_r
        norm["stride_length_m"] = FeatureNormalization.RAW
    elif step_l is not None:
        features.step_length_m = step_l
        features.stride_length_m = step_l * 2.0
        norm["step_length_m"] = FeatureNormalization.RAW
        norm["stride_length_m"] = FeatureNormalization.RAW
    elif step_r is not None:
        features.step_length_m = step_r
        features.stride_length_m = step_r * 2.0
        norm["step_length_m"] = FeatureNormalization.RAW
        norm["stride_length_m"] = FeatureNormalization.RAW

    leg = dimensions.leg_length_average
    hip_w = dimensions.hip_width
    shoulder_w = dimensions.shoulder_width

    if features.step_length_m is not None:
        features.normalized_step_length = safe_divide(features.step_length_m, leg)
        norm["normalized_step_length"] = FeatureNormalization.BODY_NORMALIZED
    if features.stride_length_m is not None:
        features.normalized_stride_length = safe_divide(features.stride_length_m, leg)
        norm["normalized_stride_length"] = FeatureNormalization.BODY_NORMALIZED

    features.step_length_symmetry = symmetry_index(step_l, step_r)
    features.stride_length_symmetry = symmetry_index(
        step_l * 2 if step_l else None,
        step_r * 2 if step_r else None,
    )
    features.leg_length_symmetry = symmetry_index(
        dimensions.leg_length_left,
        dimensions.leg_length_right,
    )
    for key in ("step_length_symmetry", "stride_length_symmetry", "leg_length_symmetry"):
        norm[key] = FeatureNormalization.RAW

    plane = cycles.ground_plane or estimate_ground_plane(
        recording,
        float(max(recording.frame_count - 1, 0)),
    )
    clearances_l: list[float] = []
    clearances_r: list[float] = []
    pelvis_x: list[float] = []
    pelvis_y: list[float] = []
    trunk_offsets: list[float] = []

    for state in cycles.per_frame:
        snap = recording.snapshot_at(state.frame_index)
        if snap is None:
            continue
        pelvis = _pelvis_position(snap)
        if pelvis:
            pelvis_x.append(pelvis.x)
            pelvis_y.append(vertical_coordinate(pelvis))
        shoulder_mid = _shoulder_mid(snap)
        if pelvis and shoulder_mid:
            trunk_offsets.append(abs(shoulder_mid.x - pelvis.x))

        if state.left.foot_clearance_m is not None and state.left_contact == 0:
            clearances_l.append(state.left.foot_clearance_m)
        if state.right.foot_clearance_m is not None and state.right_contact == 0:
            clearances_r.append(state.right.foot_clearance_m)

    if clearances_l:
        features.foot_clearance_left_m = float(np.median(clearances_l))
        norm["foot_clearance_left_m"] = FeatureNormalization.RAW
        features.normalized_foot_clearance_left = safe_divide(
            features.foot_clearance_left_m, leg
        )
        norm["normalized_foot_clearance_left"] = FeatureNormalization.BODY_NORMALIZED
    if clearances_r:
        features.foot_clearance_right_m = float(np.median(clearances_r))
        norm["foot_clearance_right_m"] = FeatureNormalization.RAW
        features.normalized_foot_clearance_right = safe_divide(
            features.foot_clearance_right_m, leg
        )
        norm["normalized_foot_clearance_right"] = FeatureNormalization.BODY_NORMALIZED

    features.foot_clearance_symmetry = symmetry_index(
        features.foot_clearance_left_m,
        features.foot_clearance_right_m,
    )
    norm["foot_clearance_symmetry"] = FeatureNormalization.RAW

    if len(pelvis_x) >= 3:
        features.pelvis_mediolateral_range_m = float(np.percentile(pelvis_x, 95) - np.percentile(pelvis_x, 5))
        norm["pelvis_mediolateral_range_m"] = FeatureNormalization.RAW
        features.normalized_pelvis_sway = safe_divide(
            features.pelvis_mediolateral_range_m, hip_w
        )
        norm["normalized_pelvis_sway"] = FeatureNormalization.BODY_NORMALIZED
    if len(pelvis_y) >= 3:
        features.pelvis_vertical_range_m = float(np.percentile(pelvis_y, 95) - np.percentile(pelvis_y, 5))
        norm["pelvis_vertical_range_m"] = FeatureNormalization.RAW
        features.normalized_vertical_pelvis_motion = safe_divide(
            features.pelvis_vertical_range_m, leg
        )
        norm["normalized_vertical_pelvis_motion"] = FeatureNormalization.BODY_NORMALIZED
    if len(trunk_offsets) >= 3:
        features.trunk_sway_m = float(np.percentile(trunk_offsets, 95) - np.percentile(trunk_offsets, 5))
        norm["trunk_sway_m"] = FeatureNormalization.RAW
        features.normalized_trunk_sway = safe_divide(features.trunk_sway_m, shoulder_w)
        norm["normalized_trunk_sway"] = FeatureNormalization.BODY_NORMALIZED

    return features


def read_opensim_mot_timeseries(mot_path: Path) -> dict[str, np.ndarray] | None:
    """Parse OpenSim ``.mot`` angle columns (time + coordinates)."""
    try:
        text = mot_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_end = 0
    for i, ln in enumerate(lines):
        if ln.lower() == "endheader":
            header_end = i + 1
            break
    if header_end >= len(lines):
        return None
    header = lines[header_end].split()
    if not header or header[0].lower() != "time":
        return None
    cols: dict[str, list[float]] = {name: [] for name in header}
    for ln in lines[header_end + 1 :]:
        parts = ln.split()
        if len(parts) != len(header):
            continue
        try:
            for name, val in zip(header, parts):
                cols[name].append(float(val))
        except ValueError:
            continue
    if len(cols.get("time", [])) < 2:
        return None
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def _angle_at_time_from_mot(
    mot: dict[str, np.ndarray],
    time_s: float,
    column_candidates: tuple[str, ...],
) -> float | None:
    times = mot.get("time")
    if times is None or len(times) < 2:
        return None
    col = next((c for c in column_candidates if c in mot), None)
    if col is None:
        return None
    return float(np.interp(time_s, times, mot[col]))


def analyze_cycle_consistency(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult,
    *,
    sequence: PoseSequence | None = None,
    ik_mot_path: Path | str | None = None,
) -> CycleConsistencyResult:
    """
    Resample gait-cycle trajectories to 101 samples (0–100%) and compare cycles.
    """
    result = CycleConsistencyResult(cycle_count=len(cycles.cycles))
    mot = read_opensim_mot_timeseries(Path(ik_mot_path)) if ik_mot_path else None
    if ik_mot_path and mot is None:
        result.warnings.append(f"OpenSim IK file unreadable: {ik_mot_path}")
    elif mot is not None:
        result.angle_source = "opensim_ik"

    plane = cycles.ground_plane or estimate_ground_plane(
        recording,
        float(max(recording.frame_count - 1, 0)),
    )
    percent = gait_cycle_percent_grid()

    def knee_left(_snap, frame):
        if frame and frame.joint_angles:
            return frame.joint_angles.left_knee
        return None

    def knee_right(_snap, frame):
        if frame and frame.joint_angles:
            return frame.joint_angles.right_knee
        return None

    def hip_left(_snap, frame):
        if frame and frame.joint_angles:
            return frame.joint_angles.left_hip
        return None

    def hip_right(_snap, frame):
        if frame and frame.joint_angles:
            return frame.joint_angles.right_hip
        return None

    def ankle_left(_snap, frame):
        if frame and frame.joint_angles:
            return frame.joint_angles.left_ankle_flexion or frame.joint_angles.left_ankle
        return None

    def ankle_right(_snap, frame):
        if frame and frame.joint_angles:
            return frame.joint_angles.right_ankle_flexion or frame.joint_angles.right_ankle
        return None

    def pelvis_vert(snap, _frame):
        p = _pelvis_position(snap) if snap else None
        return vertical_coordinate(p) if p else None

    def pelvis_ml(snap, _frame):
        p = _pelvis_position(snap) if snap else None
        return p.x if p else None

    def clearance_left(snap, _frame):
        if snap is None:
            return None
        toe = _joint_pos(snap, "left_toe") or _joint_pos(snap, "left_heel")
        return foot_clearance_m(toe, plane) if toe else None

    def clearance_right(snap, _frame):
        if snap is None:
            return None
        toe = _joint_pos(snap, "right_toe") or _joint_pos(snap, "right_heel")
        return foot_clearance_m(toe, plane) if toe else None

    ik_used = False

    signal_defs = [
        ("left_knee_angle", knee_left, ("knee_angle_l", "knee_l")),
        ("right_knee_angle", knee_right, ("knee_angle_r", "knee_r")),
        ("left_hip_angle", hip_left, ("hip_flexion_l", "hip_l")),
        ("right_hip_angle", hip_right, ("hip_flexion_r", "hip_r")),
        ("left_ankle_angle", ankle_left, ("ankle_angle_l", "ankle_l")),
        ("right_ankle_angle", ankle_right, ("ankle_angle_r", "ankle_r")),
        ("pelvis_vertical", pelvis_vert, None),
        ("pelvis_mediolateral", pelvis_ml, None),
        ("left_foot_clearance", clearance_left, None),
        ("right_foot_clearance", clearance_right, None),
    ]

    for name, pose_fn, ik_cols in signal_defs:
        per_cycle: list[np.ndarray] = []
        src: AngleSource = "mediapipe_angles"

        for cycle in cycles.cycles:
            times: list[float] = []
            values: list[float] = []
            for snap in recording.snapshots:
                if snap.frame_index < cycle.start_frame or snap.frame_index > cycle.end_frame:
                    continue
                val = None
                if mot is not None and ik_cols:
                    for col in ik_cols:
                        if col in mot:
                            val = _angle_at_time_from_mot(mot, snap.time_s, (col,))
                            if val is not None:
                                src = "opensim_ik"
                                ik_used = True
                            break
                if val is None and sequence is not None:
                    fi = snap.frame_index
                    if 0 <= fi < len(sequence.frames):
                        val = pose_fn(snap, sequence.frames[fi])
                if val is None:
                    val = pose_fn(snap, None)
                if val is None:
                    continue
                times.append(snap.time_s)
                values.append(float(val))

            arr = resample_cycle_trajectory(
                times,
                values,
                t_start=cycle.start_time_s,
                t_end=cycle.end_time_s,
            )
            if arr is not None:
                per_cycle.append(arr)

        if len(per_cycle) < 1:
            continue

        stack = np.vstack(per_cycle)
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)
        result.trajectories[name] = CycleTrajectory(
            name=name,
            percent=percent,
            mean=tuple(float(x) for x in mean),
            std=tuple(float(x) for x in std),
            per_cycle=[tuple(float(x) for x in row) for row in per_cycle],
            source=src,
        )

        if len(per_cycle) >= 2:
            rmses = []
            for row in per_cycle:
                rmses.append(float(np.sqrt(np.mean((row - mean) ** 2))))
            result.cycle_to_cycle_rmse[name] = float(np.mean(rmses))

    if ik_used:
        result.angle_source = "opensim_ik"
    elif mot is not None:
        result.warnings.append("OpenSim IK file present but joint columns not matched — using pose angles.")

    # Left-right phase consistency: correlation of mean knee angles across cycle %
    lk = result.trajectories.get("left_knee_angle")
    rk = result.trajectories.get("right_knee_angle")
    if lk and rk and len(lk.mean) == len(rk.mean):
        a = np.asarray(lk.mean)
        b = np.asarray(rk.mean)
        if np.std(a) > 1e-9 and np.std(b) > 1e-9:
            corr = float(np.corrcoef(a, b)[0, 1])
            if not math.isnan(corr):
                result.left_right_phase_consistency = max(0.0, corr)

    rmse_vals = list(result.cycle_to_cycle_rmse.values())
    if rmse_vals:
        mean_rmse = statistics.mean(rmse_vals)
        # Lower RMSE → higher repeatability (heuristic scale)
        result.cycle_repeatability_score = max(0.0, min(100.0, 100.0 - mean_rmse * 8.0))

    if len(cycles.cycles) < 2:
        result.warnings.append("Fewer than 2 gait cycles — cycle envelope statistics are limited.")

    return result


def analyze_gait_features(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult,
    *,
    sequence: PoseSequence | None = None,
    ik_mot_path: Path | str | None = None,
) -> GaitFeatureAnalysisResult:
    """Full normalized gait feature + cycle consistency analysis."""
    warnings: list[str] = []
    dimensions = estimate_body_segment_dimensions(recording)
    if dimensions.confidence_tier == "LOW":
        warnings.append(
            "Body segment dimensions estimated from few high-confidence frames."
        )
    if not cycles.metrics.metrics_reliable:
        warnings.append(
            "Step and stride metrics unavailable: "
            f"{cycles.metrics.reliability_reason}"
        )

    features = compute_normalized_gait_features(recording, cycles, dimensions)
    cycle_consistency = analyze_cycle_consistency(
        recording,
        cycles,
        sequence=sequence,
        ik_mot_path=ik_mot_path,
    )
    warnings.extend(cycle_consistency.warnings)

    return GaitFeatureAnalysisResult(
        dimensions=dimensions,
        features=features,
        cycle_consistency=cycle_consistency,
        warnings=warnings,
    )


__all__ = [
    "FeatureNormalization",
    "BodySegmentDimensions",
    "NormalizedGaitFeatures",
    "CycleTrajectory",
    "CycleConsistencyResult",
    "GaitFeatureAnalysisResult",
    "GAIT_CYCLE_SAMPLE_COUNT",
    "estimate_body_segment_dimensions",
    "compute_normalized_gait_features",
    "analyze_cycle_consistency",
    "analyze_gait_features",
    "symmetry_index",
    "safe_divide",
    "resample_cycle_trajectory",
    "gait_cycle_percent_grid",
    "read_opensim_mot_timeseries",
]
