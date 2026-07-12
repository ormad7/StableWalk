"""
Automatic camera-view estimation and view-based measurement reliability.

Classifies monocular gait recordings by dominant camera geometry (frontal,
sagittal, oblique) without using filenames or ground-truth labels.

View reliability scales domain *confidence* — raw metric values are unchanged.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from stablewalk.analysis.motion_frames import MotionFrameSeries, build_motion_frame_series
from stablewalk.models.pose_data import PoseFrame

MIN_VISIBILITY = 0.35


class GaitViewType(str, Enum):
    FRONTAL = "FRONTAL"
    SAGITTAL_LEFT = "SAGITTAL_LEFT"
    SAGITTAL_RIGHT = "SAGITTAL_RIGHT"
    OBLIQUE = "OBLIQUE"
    UNKNOWN = "UNKNOWN"


ReliabilityTier = Literal["HIGH", "MODERATE", "LOW"]


VIEW_DISPLAY_NAMES: dict[GaitViewType, str] = {
    GaitViewType.FRONTAL: "Frontal",
    GaitViewType.SAGITTAL_LEFT: "Left Side",
    GaitViewType.SAGITTAL_RIGHT: "Right Side",
    GaitViewType.OBLIQUE: "Oblique",
    GaitViewType.UNKNOWN: "Unknown",
}


# Domain-level reliability coefficients (0–1) by estimated view.
_DOMAIN_RELIABILITY: dict[GaitViewType, dict[str, float]] = {
    GaitViewType.FRONTAL: {
        "temporal_symmetry": 0.95,
        "spatial_symmetry": 0.55,
        "pelvis_stability": 0.90,
        "trunk_stability": 0.88,
        "foot_clearance": 0.62,
        "joint_smoothness": 0.72,
        "cycle_consistency": 0.78,
        "contact_pattern": 0.88,
    },
    GaitViewType.SAGITTAL_LEFT: {
        "temporal_symmetry": 0.92,
        "spatial_symmetry": 0.50,
        "pelvis_stability": 0.35,
        "trunk_stability": 0.40,
        "foot_clearance": 0.88,
        "joint_smoothness": 0.90,
        "cycle_consistency": 0.85,
        "contact_pattern": 0.90,
    },
    GaitViewType.SAGITTAL_RIGHT: {
        "temporal_symmetry": 0.92,
        "spatial_symmetry": 0.50,
        "pelvis_stability": 0.35,
        "trunk_stability": 0.40,
        "foot_clearance": 0.88,
        "joint_smoothness": 0.90,
        "cycle_consistency": 0.85,
        "contact_pattern": 0.90,
    },
    GaitViewType.OBLIQUE: {
        "temporal_symmetry": 0.85,
        "spatial_symmetry": 0.65,
        "pelvis_stability": 0.60,
        "trunk_stability": 0.62,
        "foot_clearance": 0.72,
        "joint_smoothness": 0.78,
        "cycle_consistency": 0.72,
        "contact_pattern": 0.80,
    },
    GaitViewType.UNKNOWN: {
        "temporal_symmetry": 0.70,
        "spatial_symmetry": 0.55,
        "pelvis_stability": 0.50,
        "trunk_stability": 0.50,
        "foot_clearance": 0.55,
        "joint_smoothness": 0.60,
        "cycle_consistency": 0.55,
        "contact_pattern": 0.65,
    },
}

# Human-readable metric labels for GUI / reports.
_METRIC_RELIABILITY_BY_VIEW: dict[GaitViewType, dict[str, ReliabilityTier]] = {
    GaitViewType.FRONTAL: {
        "knee_flexion": "MODERATE",
        "hip_flexion": "MODERATE",
        "pelvis_lateral_sway": "HIGH",
        "trunk_lateral_sway": "HIGH",
        "foot_clearance": "MODERATE",
        "stride_length": "LOW",
        "step_timing": "HIGH",
        "frontal_symmetry": "HIGH",
    },
    GaitViewType.SAGITTAL_LEFT: {
        "knee_flexion": "HIGH",
        "hip_flexion": "HIGH",
        "pelvis_lateral_sway": "LOW",
        "trunk_lateral_sway": "LOW",
        "foot_clearance": "HIGH",
        "stride_length": "MODERATE",
        "step_timing": "HIGH",
        "frontal_symmetry": "LOW",
    },
    GaitViewType.SAGITTAL_RIGHT: {
        "knee_flexion": "HIGH",
        "hip_flexion": "HIGH",
        "pelvis_lateral_sway": "LOW",
        "trunk_lateral_sway": "LOW",
        "foot_clearance": "HIGH",
        "stride_length": "MODERATE",
        "step_timing": "HIGH",
        "frontal_symmetry": "LOW",
    },
    GaitViewType.OBLIQUE: {
        "knee_flexion": "MODERATE",
        "hip_flexion": "MODERATE",
        "pelvis_lateral_sway": "MODERATE",
        "trunk_lateral_sway": "MODERATE",
        "foot_clearance": "MODERATE",
        "stride_length": "MODERATE",
        "step_timing": "HIGH",
        "frontal_symmetry": "MODERATE",
    },
    GaitViewType.UNKNOWN: {
        "knee_flexion": "MODERATE",
        "hip_flexion": "MODERATE",
        "pelvis_lateral_sway": "LOW",
        "trunk_lateral_sway": "LOW",
        "foot_clearance": "MODERATE",
        "stride_length": "LOW",
        "step_timing": "MODERATE",
        "frontal_symmetry": "LOW",
    },
}

_METRIC_GUI_LABELS: dict[str, str] = {
    "knee_flexion": "Knee flexion",
    "hip_flexion": "Hip flexion",
    "pelvis_lateral_sway": "Pelvis lateral sway",
    "trunk_lateral_sway": "Trunk lateral sway",
    "foot_clearance": "Foot clearance",
    "stride_length": "Stride length",
    "step_timing": "Step timing",
    "frontal_symmetry": "Frontal symmetry",
}


@dataclass(frozen=True)
class GaitViewEstimate:
    """Automatic camera-view classification from pose geometry."""

    view_type: GaitViewType
    view_confidence: float
    signals: dict[str, float] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return VIEW_DISPLAY_NAMES.get(self.view_type, "Unknown")

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_type": self.view_type.value,
            "view_confidence": round(self.view_confidence, 3),
            "display_name": self.display_name,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
        }


@dataclass(frozen=True)
class ViewReliabilityProfile:
    """View-based reliability coefficients for stability domains and metrics."""

    view_type: GaitViewType
    domain_coefficients: dict[str, float]
    metric_tiers: dict[str, ReliabilityTier]

    def domain_coefficient(self, domain_key: str) -> float:
        return float(self.domain_coefficients.get(domain_key, 0.65))

    def metric_tier(self, metric_key: str) -> ReliabilityTier:
        return self.metric_tiers.get(metric_key, "MODERATE")

    def metric_table(self) -> list[tuple[str, ReliabilityTier]]:
        """Rows for GUI: (display label, tier)."""
        rows: list[tuple[str, ReliabilityTier]] = []
        for key, tier in self.metric_tiers.items():
            rows.append((_METRIC_GUI_LABELS.get(key, key.replace("_", " ").title()), tier))
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_type": self.view_type.value,
            "domain_coefficients": dict(self.domain_coefficients),
            "metric_tiers": dict(self.metric_tiers),
        }


@dataclass(frozen=True)
class CrossVideoComparability:
    """Whether demo videos can be compared on absolute stability scores."""

    level: Literal["HIGH", "MODERATE", "LOW"]
    score: float
    factors: dict[str, Any]
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "score": round(self.score, 3),
            "factors": self.factors,
            "warning": self.warning,
        }


def build_view_reliability_profile(view_type: GaitViewType) -> ViewReliabilityProfile:
    return ViewReliabilityProfile(
        view_type=view_type,
        domain_coefficients=dict(_DOMAIN_RELIABILITY.get(view_type, _DOMAIN_RELIABILITY[GaitViewType.UNKNOWN])),
        metric_tiers=dict(_METRIC_RELIABILITY_BY_VIEW.get(view_type, _METRIC_RELIABILITY_BY_VIEW[GaitViewType.UNKNOWN])),
    )


def _keypoint_map(frame: PoseFrame) -> dict[str, Any]:
    return {kp.name: kp for kp in frame.keypoints}


def _frame_geometry_signals(frame: PoseFrame) -> dict[str, float] | None:
    kp = _keypoint_map(frame)
    lh, rh = kp.get("left_hip"), kp.get("right_hip")
    ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
    if not all(
        p and p.visibility >= MIN_VISIBILITY
        for p in (lh, rh, ls, rs)
    ):
        return None

    ys = [p.y for p in kp.values() if p.visibility >= MIN_VISIBILITY]
    body_span = max(max(ys) - min(ys), 1e-3)

    hip_sep = abs(float(lh.x) - float(rh.x)) / body_span
    shoulder_sep = abs(float(ls.x) - float(rs.x)) / body_span
    overlap = 1.0 if hip_sep < 0.06 else 0.0

    depth_asym = (
        (float(lh.z) + float(ls.z)) / 2.0 - (float(rh.z) + float(rs.z)) / 2.0
    )
    torso_yaw = abs((ls.x + rs.x) / 2.0 - (lh.x + rh.x) / 2.0) / body_span

    return {
        "hip_separation": hip_sep,
        "shoulder_separation": shoulder_sep,
        "lr_overlap": overlap,
        "depth_asymmetry": depth_asym,
        "torso_yaw": torso_yaw,
    }


def _progression_from_frames(frames: Sequence[PoseFrame]) -> dict[str, float]:
    """Pelvis progression axis ratios without a full GaitMotionRecording."""
    from stablewalk.pose.reconstruction import compute_root_trajectory_point
    from stablewalk.pose.skeleton_3d import reconstruct_skeleton_3d, sequence_skeleton_scale

    kps = [f.keypoints for f in frames if f.keypoints]
    scale = sequence_skeleton_scale(kps) if kps else 1.0
    valid: list[np.ndarray] = []
    for frame in frames:
        if not frame.keypoints:
            continue
        root = compute_root_trajectory_point(frame.keypoints)
        if root is None:
            continue
        skel = reconstruct_skeleton_3d(frame.keypoints, scale=scale)
        valid.append(np.asarray(root, dtype=float) * max(float(skel.scale), 1e-6))

    if len(valid) < 3:
        return {"progression_x_ratio": 0.0, "progression_z_ratio": 0.0, "path_length": 0.0}

    arr = np.asarray(valid, dtype=float)
    disp = arr[-1] - arr[0]
    path = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
    total = float(np.linalg.norm(disp))
    if total < 1e-6:
        return {"progression_x_ratio": 0.0, "progression_z_ratio": 0.0, "path_length": path}

    ax = np.abs(disp) / total
    return {
        "progression_x_ratio": float(ax[0]),
        "progression_y_ratio": float(ax[1]),
        "progression_z_ratio": float(ax[2]),
        "path_length": path,
    }


def _progression_axis_signals(series: MotionFrameSeries | None, frames: Sequence[PoseFrame]) -> dict[str, float]:
    if series is not None:
        valid = [p for p in series.global_pelvis if p is not None]
        if len(valid) >= 3:
            arr = np.asarray(valid, dtype=float)
            disp = arr[-1] - arr[0]
            path = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
            total = float(np.linalg.norm(disp))
            if total >= 1e-6:
                ax = np.abs(disp) / total
                return {
                    "progression_x_ratio": float(ax[0]),
                    "progression_y_ratio": float(ax[1]),
                    "progression_z_ratio": float(ax[2]),
                    "path_length": path,
                }
    return _progression_from_frames(frames)


def estimate_gait_view(
    frames: Sequence[PoseFrame],
    *,
    motion_frames: MotionFrameSeries | None = None,
    body_height: float | None = None,
) -> GaitViewEstimate:
    """
    Classify dominant camera view from body geometry and pelvis trajectory.

    Does not use video filenames or gait labels.
    """
    per_frame: list[dict[str, float]] = []
    for frame in frames:
        sig = _frame_geometry_signals(frame)
        if sig:
            per_frame.append(sig)

    if not per_frame:
        return GaitViewEstimate(GaitViewType.UNKNOWN, 0.0, {"tracked_geometry_frames": 0.0})

    hip_sep = statistics.mean(s["hip_separation"] for s in per_frame)
    shoulder_sep = statistics.mean(s["shoulder_separation"] for s in per_frame)
    overlap_frac = statistics.mean(s["lr_overlap"] for s in per_frame)
    depth_asym = statistics.mean(s["depth_asymmetry"] for s in per_frame)
    torso_yaw = statistics.mean(s["torso_yaw"] for s in per_frame)

    series = motion_frames
    prog = _progression_axis_signals(series, frames)
    prog_x = prog.get("progression_x_ratio", 0.0)
    prog_z = prog.get("progression_z_ratio", 0.0)

    signals = {
        "hip_separation_mean": hip_sep,
        "shoulder_separation_mean": shoulder_sep,
        "lr_overlap_fraction": overlap_frac,
        "depth_asymmetry_mean": depth_asym,
        "torso_yaw_mean": torso_yaw,
        "progression_x_ratio": prog_x,
        "progression_z_ratio": prog_z,
        "tracked_geometry_frames": float(len(per_frame)),
    }
    if body_height is not None:
        signals["body_height_m"] = float(body_height)

    # Sagittal: narrow L/R width, high overlap, progression mostly image-lateral (X).
    sagittal_score = (
        0.30 * overlap_frac
        + 0.20 * _clamp01(0.14 - hip_sep, scale=0.14)
        + 0.15 * _clamp01(0.16 - shoulder_sep, scale=0.16)
        + 0.35 * _clamp01(prog_x - 0.40, scale=0.55)
    )
    # Frontal: wide hips/shoulders, low overlap, limited lateral progression.
    frontal_score = (
        0.28 * _clamp01(hip_sep - 0.085, scale=0.10)
        + 0.27 * _clamp01(shoulder_sep - 0.11, scale=0.12)
        + 0.25 * (1.0 - overlap_frac)
        + 0.20 * _clamp01(0.35 - prog_x, scale=0.35)
    )
    oblique_score = (
        0.45 * (1.0 - abs(sagittal_score - frontal_score))
        + 0.30 * _clamp01(torso_yaw, scale=0.08)
        + 0.25 * _clamp01(min(hip_sep, shoulder_sep), scale=0.10)
    )

    signals.update(
        {
            "sagittal_score": sagittal_score,
            "frontal_score": frontal_score,
            "oblique_score": oblique_score,
        }
    )

    # Strong lateral progression → sagittal side view (athletic demo pattern).
    if prog_x >= 0.50 and sagittal_score >= 0.30:
        best_type = (
            GaitViewType.SAGITTAL_LEFT if depth_asym >= 0.0 else GaitViewType.SAGITTAL_RIGHT
        )
        confidence = _clamp(0.55 + 0.35 * prog_x + 0.10 * sagittal_score)
        return GaitViewEstimate(best_type, confidence, signals)

    # Wide body profile with limited lateral translation → frontal.
    if frontal_score >= 0.34 and frontal_score >= sagittal_score + 0.06:
        confidence = _clamp(0.45 + 0.45 * frontal_score)
        return GaitViewEstimate(GaitViewType.FRONTAL, confidence, signals)

    scores = {
        GaitViewType.SAGITTAL_LEFT: sagittal_score,
        GaitViewType.SAGITTAL_RIGHT: sagittal_score,
        GaitViewType.FRONTAL: frontal_score,
        GaitViewType.OBLIQUE: oblique_score,
    }
    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    if best_score < 0.28:
        return GaitViewEstimate(GaitViewType.UNKNOWN, best_score, signals)

    if best_type in (GaitViewType.SAGITTAL_LEFT, GaitViewType.SAGITTAL_RIGHT):
        best_type = (
            GaitViewType.SAGITTAL_LEFT if depth_asym >= 0.0 else GaitViewType.SAGITTAL_RIGHT
        )

    confidence = _clamp(best_score, 0.0, 1.0)
    if best_type == GaitViewType.OBLIQUE and confidence > 0.85:
        confidence = 0.75

    return GaitViewEstimate(best_type, confidence, signals)


def analyze_gait_view_geometry(
    frames: Sequence[PoseFrame],
    *,
    motion_frames: MotionFrameSeries | None = None,
    body_height: float | None = None,
) -> tuple[GaitViewEstimate, ViewReliabilityProfile]:
    """Run view estimation and build the reliability profile."""
    estimate = estimate_gait_view(
        frames,
        motion_frames=motion_frames,
        body_height=body_height,
    )
    profile = build_view_reliability_profile(estimate.view_type)
    return estimate, profile


def apply_view_reliability(confidence: float, domain_key: str, profile: ViewReliabilityProfile) -> float:
    """Scale domain confidence by view reliability (raw score unchanged)."""
    return max(0.0, min(1.0, confidence * profile.domain_coefficient(domain_key)))


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _clamp01(value: float, *, scale: float) -> float:
    if scale <= 1e-9:
        return 0.0
    return _clamp(value / scale)


def assess_cross_video_comparability(
    video_summaries: Sequence[Mapping[str, Any]],
) -> CrossVideoComparability:
    """
    Assess whether absolute stability scores are comparable across videos.

    ``video_summaries`` entries should include keys:
    view_type, view_confidence, valid_pose_frame_pct, body_height_m (optional),
    gait_cycles, duration_s, fps.
    """
    if not video_summaries:
        return CrossVideoComparability("LOW", 0.0, {}, "No videos to compare.")

    views = {str(v.get("view_type", "UNKNOWN")) for v in video_summaries}
    sagittal = any(v.startswith("SAGITTAL") for v in views)
    frontal = "FRONTAL" in views
    view_conf = [float(v.get("view_confidence", 0.0) or 0.0) for v in video_summaries]
    pose_pcts = [float(v.get("valid_pose_frame_pct", 0.0) or 0.0) for v in video_summaries]
    cycles = [float(v.get("gait_cycles", 0.0) or 0.0) for v in video_summaries]
    durations = [float(v.get("duration_s", 0.0) or 0.0) for v in video_summaries]
    fps_vals = [float(v.get("fps", 0.0) or 0.0) for v in video_summaries]
    heights = [
        float(v["body_height_m"])
        for v in video_summaries
        if v.get("body_height_m") is not None
    ]

    view_homogeneity = 1.0 if len(views) == 1 else (0.55 if len(views) == 2 else 0.25)
    view_conf_mean = statistics.mean(view_conf) if view_conf else 0.0
    pose_mean = statistics.mean(pose_pcts) if pose_pcts else 0.0
    pose_spread = (max(pose_pcts) - min(pose_pcts)) if pose_pcts else 100.0
    cycle_mean = statistics.mean(cycles) if cycles else 0.0
    cycle_spread = (max(cycles) - min(cycles)) if cycles else 0.0
    duration_spread = (max(durations) - min(durations)) if durations else 0.0
    fps_spread = (max(fps_vals) - min(fps_vals)) if fps_vals else 0.0
    height_spread = (max(heights) - min(heights)) if len(heights) >= 2 else 0.0

    score = (
        0.35 * view_homogeneity
        + 0.20 * _clamp01(view_conf_mean, scale=0.85)
        + 0.20 * _clamp01(pose_mean, scale=95.0)
        + 0.10 * _clamp01(1.0 - pose_spread / 40.0, scale=1.0)
        + 0.08 * _clamp01(cycle_mean, scale=4.0)
        + 0.07 * _clamp01(1.0 - cycle_spread / 4.0, scale=1.0)
    )
    if duration_spread > 4.0:
        score -= 0.05
    if fps_spread > 5.0:
        score -= 0.05
    if height_spread > 0.15:
        score -= 0.08
    score = _clamp(score)

    if len(views) > 1:
        if sagittal and frontal:
            score = min(score, 0.42)
        else:
            score = min(score, 0.58)

    if score >= 0.72:
        level: Literal["HIGH", "MODERATE", "LOW"] = "HIGH"
        warning = None
    elif score >= 0.45:
        level = "MODERATE"
        warning = None
    else:
        level = "LOW"
        warning = (
            "Direct absolute stability-score comparison is limited by heterogeneous "
            "camera viewpoints."
        )

    if len(views) > 1:
        warning = (
            "Direct absolute stability-score comparison is limited by heterogeneous "
            "camera viewpoints."
        )
        if level == "HIGH":
            level = "MODERATE"
        if sagittal and frontal:
            level = "LOW"

    factors = {
        "unique_view_types": sorted(views),
        "view_homogeneity": round(view_homogeneity, 3),
        "mean_view_confidence": round(view_conf_mean, 3),
        "valid_pose_frame_pct": pose_pcts,
        "pose_pct_spread": round(pose_spread, 2),
        "gait_cycles": cycles,
        "cycle_spread": round(cycle_spread, 2),
        "duration_spread_s": round(duration_spread, 2),
        "fps_spread": round(fps_spread, 2),
        "body_height_spread_m": round(height_spread, 3) if heights else None,
    }
    return CrossVideoComparability(level, score, factors, warning)


__all__ = [
    "CrossVideoComparability",
    "GaitViewEstimate",
    "GaitViewType",
    "ReliabilityTier",
    "ViewReliabilityProfile",
    "analyze_gait_view_geometry",
    "apply_view_reliability",
    "assess_cross_video_comparability",
    "build_view_reliability_profile",
    "estimate_gait_view",
]
