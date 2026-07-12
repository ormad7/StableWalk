"""
Global motion vs root-relative biomechanical coordinate representations.

GLOBAL MOTION COORDINATES
    Pelvis trajectory in camera-aligned meters before per-frame hip centering.
    Used for walking progression, speed, path length — not stability penalties.

ROOT-RELATIVE BIOMECHANICAL COORDINATES
    Joint positions from ``GaitMotionRecording`` (pelvis-centered canonical frame).
    Used for trunk sway, joint smoothness, and oscillation orthogonal to progression.

Local gait frame (for pelvis stability)
    FORWARD: robust estimate of pelvis progression direction (not single-frame).
    VERTICAL: canonical +Y.
    MEDIOLATERAL: cross(vertical, forward).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from stablewalk.models.gait_motion import GaitMotionRecording, Vec3
from stablewalk.models.joint_registry import ROOT_JOINT_ID
from stablewalk.models.pose_data import PoseFrame
from stablewalk.pose.coordinates import hip_center
from stablewalk.pose.reconstruction import compute_root_trajectory_point
from stablewalk.pose.skeleton_3d import reconstruct_skeleton_3d, sequence_skeleton_scale

CANONICAL_VERTICAL = np.array([0.0, 1.0, 0.0], dtype=float)
MIN_VISIBILITY = 0.3
MIN_PROGRESSION_SPEED = 1e-5


@dataclass(frozen=True)
class GaitProgressionFrame:
    """Orthonormal gait coordinate basis in global motion space."""

    forward: np.ndarray
    vertical: np.ndarray
    mediolateral: np.ndarray
    confidence: float = 0.0

    def project(self, vector: np.ndarray) -> tuple[float, float, float]:
        """Return (forward, vertical, mediolateral) scalar components."""
        return (
            float(np.dot(vector, self.forward)),
            float(np.dot(vector, self.vertical)),
            float(np.dot(vector, self.mediolateral)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward_axis": self.forward.tolist(),
            "vertical_axis": self.vertical.tolist(),
            "mediolateral_axis": self.mediolateral.tolist(),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class MotionFrameSeries:
    """Per-frame global pelvis path and root-relative joint positions."""

    frame_indices: list[int] = field(default_factory=list)
    timestamps_s: list[float] = field(default_factory=list)
    global_pelvis: list[np.ndarray | None] = field(default_factory=list)
    pelvis_confidence: list[float] = field(default_factory=list)
    root_relative: dict[int, dict[str, Vec3]] = field(default_factory=dict)
    progression: GaitProgressionFrame | None = None
    body_scale: float = 1.0

    @property
    def tracked_ratio(self) -> float:
        if not self.frame_indices:
            return 0.0
        valid = sum(1 for p in self.global_pelvis if p is not None)
        return valid / len(self.frame_indices)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v.copy()
    return v / n


def _gram_schmidt_horizontal(forward_hint: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    """Remove vertical component and re-normalize."""
    f = forward_hint - np.dot(forward_hint, vertical) * vertical
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return f / n


def estimate_gait_progression_frame(
    positions: np.ndarray,
    *,
    confidences: Sequence[float] | None = None,
) -> GaitProgressionFrame:
    """
    Robust forward-axis estimate from a pelvis trajectory.

    Uses high-confidence velocity directions and falls back to robust displacement.
    """
    vertical = CANONICAL_VERTICAL.copy()
    if positions.shape[0] < 3:
        return GaitProgressionFrame(
            forward=np.array([0.0, 0.0, 1.0]),
            vertical=vertical,
            mediolateral=np.array([1.0, 0.0, 0.0]),
            confidence=0.0,
        )

    if confidences is not None and len(confidences) == positions.shape[0]:
        conf = np.asarray(confidences, dtype=float)
        threshold = max(float(np.percentile(conf, 40)), MIN_VISIBILITY)
        mask = conf >= threshold
    else:
        mask = np.ones(positions.shape[0], dtype=bool)

    vel = np.diff(positions, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    vel_mask = mask[1:] & mask[:-1] & (speed > MIN_PROGRESSION_SPEED)

    forward: np.ndarray | None = None
    confidence = 0.0

    if vel_mask.sum() >= 3:
        dirs = vel[vel_mask] / speed[vel_mask][:, None]
        forward = _normalize(np.median(dirs, axis=0))
        confidence = min(1.0, vel_mask.sum() / max(positions.shape[0] - 1, 1))

    if forward is None or not math.isfinite(float(forward[0])):
        idx = np.where(mask)[0]
        if idx.size >= 2:
            start = positions[idx[0]]
            end = positions[idx[-1]]
            disp = end - start
            if float(np.linalg.norm(disp)) > MIN_PROGRESSION_SPEED:
                forward = _normalize(disp)
                confidence = 0.35
            else:
                q10 = int(max(0, positions.shape[0] * 0.1))
                q90 = int(min(positions.shape[0] - 1, positions.shape[0] * 0.9))
                disp = positions[q90] - positions[q10]
                forward = _normalize(disp) if float(np.linalg.norm(disp)) > MIN_PROGRESSION_SPEED else np.array(
                    [0.0, 0.0, 1.0]
                )
                confidence = 0.25
        else:
            forward = np.array([0.0, 0.0, 1.0])
            confidence = 0.0

    forward = _gram_schmidt_horizontal(forward, vertical)
    mediolateral = _normalize(np.cross(vertical, forward))
    if float(np.linalg.norm(mediolateral)) < 1e-9:
        mediolateral = np.array([1.0, 0.0, 0.0])

    return GaitProgressionFrame(
        forward=forward,
        vertical=vertical,
        mediolateral=mediolateral,
        confidence=confidence,
    )


def _global_pelvis_meters(frame: PoseFrame, *, uniform_scale: float) -> tuple[np.ndarray, float] | None:
    if not frame.keypoints:
        return None
    root = compute_root_trajectory_point(frame.keypoints)
    if root is None:
        return None
    skel = reconstruct_skeleton_3d(frame.keypoints, scale=uniform_scale)
    scale = max(float(skel.scale), 1e-6)
    pos = np.array(root, dtype=float) * scale
    center = hip_center(frame.keypoints)
    conf = 0.0
    if center:
        kp = {k.name: k for k in frame.keypoints}
        lh, rh = kp.get("left_hip"), kp.get("right_hip")
        if lh and rh:
            conf = min(lh.visibility, rh.visibility)
        else:
            mid = kp.get("mid_hip")
            conf = mid.visibility if mid else 0.0
    return pos, conf


def _snapshot_joint_positions(recording: GaitMotionRecording, frame_index: int) -> dict[str, Vec3]:
    for snap in recording.snapshots:
        if snap.frame_index == frame_index:
            return {jid: js.position for jid, js in snap.joints.items()}
    if 0 <= frame_index < len(recording.snapshots):
        snap = recording.snapshots[frame_index]
        return {jid: js.position for jid, js in snap.joints.items()}
    return {}


def build_motion_frame_series(
    frames: Sequence[PoseFrame],
    recording: GaitMotionRecording,
) -> MotionFrameSeries:
    """Build global pelvis path and root-relative joint map for stability analysis."""
    keypoint_frames = [f.keypoints for f in frames if f.keypoints]
    uniform_scale = sequence_skeleton_scale(keypoint_frames) if keypoint_frames else 1.0

    series = MotionFrameSeries(body_scale=uniform_scale)
    global_positions: list[np.ndarray] = []

    for frame in frames:
        series.frame_indices.append(frame.frame_index)
        series.timestamps_s.append(float(frame.timestamp_s))
        gp = _global_pelvis_meters(frame, uniform_scale=uniform_scale)
        if gp is None:
            series.global_pelvis.append(None)
            series.pelvis_confidence.append(0.0)
        else:
            pos, conf = gp
            series.global_pelvis.append(pos)
            series.pelvis_confidence.append(conf)
            if conf >= MIN_VISIBILITY:
                global_positions.append(pos)

        rr = _snapshot_joint_positions(recording, frame.frame_index)
        if rr:
            series.root_relative[frame.frame_index] = rr

    if len(global_positions) >= 3:
        pos_arr = np.asarray(global_positions, dtype=float)
        conf_arr = [
            c for p, c in zip(series.global_pelvis, series.pelvis_confidence) if p is not None
        ]
        series.progression = estimate_gait_progression_frame(pos_arr, confidences=conf_arr)

    return series


def _shoulder_center_relative(joints: dict[str, Vec3]) -> np.ndarray | None:
    ls = joints.get("left_shoulder")
    rs = joints.get("right_shoulder")
    if ls is None or rs is None:
        return None
    return np.array(
        [
            (ls.x + rs.x) / 2.0,
            (ls.y + rs.y) / 2.0,
            (ls.z + rs.z) / 2.0,
        ],
        dtype=float,
    )


def project_global_trajectory(
    series: MotionFrameSeries,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project global pelvis positions into gait-frame components.

    Returns (times, forward, vertical, mediolateral) arrays with NaN for gaps.
    """
    prog = series.progression
    if prog is None:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty

    times: list[float] = []
    fwd: list[float] = []
    vert: list[float] = []
    ml: list[float] = []

    origin: np.ndarray | None = None
    for t, pos in zip(series.timestamps_s, series.global_pelvis):
        if pos is None:
            continue
        if origin is None:
            origin = pos.copy()
        rel = pos - origin
        f, v, m = prog.project(rel)
        times.append(t)
        fwd.append(f)
        vert.append(v)
        ml.append(m)

    return (
        np.asarray(times, dtype=float),
        np.asarray(fwd, dtype=float),
        np.asarray(vert, dtype=float),
        np.asarray(ml, dtype=float),
    )


def compute_pelvis_motion_comparison(series: MotionFrameSeries) -> dict[str, float | None]:
    """Diagnostic metrics: global vs gait-frame pelvis decomposition."""
    valid = [p for p in series.global_pelvis if p is not None]
    if len(valid) < 2:
        return {
            "global_pelvis_displacement_m": None,
            "forward_pelvis_displacement_m": None,
            "mediolateral_pelvis_sway_m": None,
            "vertical_pelvis_oscillation_m": None,
            "root_relative_trunk_sway_m": None,
            "tracked_frame_ratio": series.tracked_ratio,
        }

    global_disp = float(np.linalg.norm(valid[-1] - valid[0]))
    _, fwd, vert, ml = project_global_trajectory(series)
    ml_det = _detrend_array(ml) if ml.size else ml
    vert_det = _detrend_array(vert) if vert.size else vert

    trunk_ml: list[float] = []
    prog = series.progression
    if prog is not None:
        for frame_idx, joints in series.root_relative.items():
            shoulder = _shoulder_center_relative(joints)
            if shoulder is None:
                continue
            _, _, m = prog.project(shoulder)
            trunk_ml.append(m)

    return {
        "global_pelvis_displacement_m": global_disp,
        "forward_pelvis_displacement_m": float(np.ptp(fwd)) if fwd.size else None,
        "mediolateral_pelvis_sway_m": float(np.std(ml_det)) if ml_det.size else None,
        "vertical_pelvis_oscillation_m": float(np.std(vert_det)) if vert_det.size else None,
        "root_relative_trunk_sway_m": float(np.std(trunk_ml)) if len(trunk_ml) >= 3 else None,
        "tracked_frame_ratio": series.tracked_ratio,
        "progression_confidence": prog.confidence if prog else 0.0,
    }


def compute_gait_frame_pelvis_metrics(
    series: MotionFrameSeries,
    *,
    hip_width: float,
    fps: float,
) -> dict[str, float | None]:
    """
    Pelvis stability features in the local gait frame.

    Penalizes mediolateral sway and vertical oscillation — not forward progression.
    """
    width = max(hip_width, 1e-6)
    _, fwd, vert, ml = project_global_trajectory(series)
    times = np.asarray(
        [t for t, p in zip(series.timestamps_s, series.global_pelvis) if p is not None],
        dtype=float,
    )

    if ml.size < 3 or times.size < 3:
        return {
            "pelvis_mediolateral_range": None,
            "pelvis_vertical_range": None,
            "pelvis_mediolateral_rms": None,
            "pelvis_forward_displacement": None,
            "pelvis_velocity_variability": None,
            "pelvis_acceleration_variability": None,
            "pelvis_jerk_metric": None,
            "normalized_pelvis_sway": None,
            "tracked_frame_ratio": series.tracked_ratio,
            "coordinate_representation": "gait_frame_global_pelvis",
        }

    ml_det = _detrend_array(ml)
    vert_det = _detrend_array(vert)
    ml_rms = float(np.sqrt(np.mean(ml_det**2)))
    ml_range = float(np.ptp(ml_det))
    vert_range = float(np.std(vert_det))
    vert_ptp = float(np.ptp(vert_det))

    ml_vel = _gradient(ml_det, times, fps=fps)
    ml_acc = _gradient(ml_vel, times, fps=fps)
    vert_vel = _gradient(vert_det, times, fps=fps)

    jerk = _mean_abs_second_diff(ml_det)

    def _cv(vals: np.ndarray) -> float | None:
        if vals.size < 2:
            return None
        mean = float(np.mean(np.abs(vals)))
        if mean < 1e-9:
            return None
        return float(np.std(vals) / mean)

    return {
        "pelvis_mediolateral_range": ml_range,
        "pelvis_vertical_range": vert_range,
        "pelvis_vertical_ptp": vert_ptp,
        "pelvis_mediolateral_rms": ml_rms,
        "pelvis_forward_displacement": float(np.ptp(fwd)),
        "pelvis_velocity_variability": _cv(ml_vel),
        "pelvis_acceleration_variability": _cv(ml_acc),
        "pelvis_jerk_metric": jerk,
        "normalized_pelvis_sway": ml_rms / width,
        "tracked_frame_ratio": series.tracked_ratio,
        "coordinate_representation": "gait_frame_global_pelvis",
    }


def compute_trunk_gait_frame_metrics(
    series: MotionFrameSeries,
    *,
    shoulder_width: float,
) -> dict[str, float | None]:
    """Trunk stability from shoulder center in root-relative + gait-frame coordinates."""
    prog = series.progression
    if prog is None:
        return {}

    width = max(shoulder_width, 1e-6)
    ml_offsets: list[float] = []
    lean_angles: list[float] = []
    com_dist: list[float] = []

    for joints in series.root_relative.values():
        shoulder = _shoulder_center_relative(joints)
        if shoulder is None:
            continue
        f, v, m = prog.project(shoulder)
        ml_offsets.append(m)
        lean_angles.append(math.degrees(math.atan2(v, f + 1e-9)))
        com_dist.append(float(np.linalg.norm(shoulder)) / width)

    if len(ml_offsets) < 3:
        return {}

    ml_arr = _detrend_array(np.asarray(ml_offsets, dtype=float))
    return {
        "trunk_lateral_sway_ratio": float(np.std(ml_arr)) / width,
        "trunk_lean_variation_deg": float(np.std(lean_angles)),
        "trunk_upper_oscillation": float(np.std(_detrend_array(np.asarray(com_dist)))),
        "trunk_frame_count": float(len(ml_offsets)),
    }


def root_relative_joint_series(
    series: MotionFrameSeries,
    joint_id: str,
) -> list[tuple[float, np.ndarray | None]]:
    """(timestamp, root-relative position) for one joint."""
    out: list[tuple[float, np.ndarray | None]] = []
    for t, frame_idx in zip(series.timestamps_s, series.frame_indices):
        joints = series.root_relative.get(frame_idx, {})
        js = joints.get(joint_id)
        if js is None:
            out.append((t, None))
        else:
            out.append((t, np.array([js.x, js.y, js.z], dtype=float)))
    return out


def root_relative_position_jerk(
    series: MotionFrameSeries,
    joint_id: str,
    *,
    fps: float,
    progression: GaitProgressionFrame | None = None,
) -> float | None:
    """
    Irregularity of root-relative joint motion projected orthogonal to progression.

    Whole-body translation is already removed by the pelvis-centered recording frame.
    """
    prog = progression or series.progression
    if prog is None:
        return None

    times: list[float] = []
    orth: list[float] = []
    for t, pos in root_relative_joint_series(series, joint_id):
        if pos is None:
            continue
        _, v, m = prog.project(pos)
        times.append(t)
        orth.append(math.hypot(v, m))

    if len(orth) < 5:
        return None
    arr = np.asarray(orth, dtype=float)
    t_arr = np.asarray(times, dtype=float)
    vel = _gradient(arr, t_arr, fps=fps)
    return _mean_abs_second_diff(vel)


def forward_step_displacement(
    series: MotionFrameSeries,
    event_frame_indices: Sequence[int],
    *,
    leg_length: float,
) -> list[float]:
    """Forward gait-frame pelvis displacement between events, normalized by leg length."""
    prog = series.progression
    if prog is None:
        return []

    leg = max(leg_length, 1e-6)
    frame_to_fwd: dict[int, float] = {}
    origin: np.ndarray | None = None

    for frame_idx, pos in zip(series.frame_indices, series.global_pelvis):
        if pos is None:
            continue
        if origin is None:
            origin = pos.copy()
        f, _, _ = prog.project(pos - origin)
        frame_to_fwd[frame_idx] = f

    out: list[float] = []
    for i in range(len(event_frame_indices) - 1):
        a, b = event_frame_indices[i], event_frame_indices[i + 1]
        if a in frame_to_fwd and b in frame_to_fwd:
            out.append(abs(frame_to_fwd[b] - frame_to_fwd[a]) / leg)
    return out


def swing_foot_progression_amplitude(
    series: MotionFrameSeries,
    frame_index: int,
    side: str,
) -> float | None:
    """Root-relative foot forward excursion during swing in the gait frame."""
    prog = series.progression
    joints = series.root_relative.get(frame_index)
    if prog is None or not joints:
        return None

    names = (f"{side}_heel", f"{side}_ankle", f"{side}_toe")
    fwd_vals: list[float] = []
    for name in names:
        js = joints.get(name)
        if js is None:
            continue
        f, _, _ = prog.project(np.array([js.x, js.y, js.z]))
        fwd_vals.append(f)
    if not fwd_vals:
        return None
    return max(fwd_vals) - min(fwd_vals)


def _detrend_array(arr: np.ndarray) -> np.ndarray:
    idx = np.arange(len(arr), dtype=float)
    valid = np.isfinite(arr)
    if valid.sum() < 3:
        return arr.copy()
    coef = np.polyfit(idx[valid], arr[valid], 1)
    return arr - np.polyval(coef, idx)


def _gradient(y: np.ndarray, times: np.ndarray, *, fps: float) -> np.ndarray:
    if len(y) < 2:
        return np.zeros_like(y)
    if len(times) >= 2 and (float(np.max(times)) - float(np.min(times))) > 1e-9:
        return np.gradient(y, times)
    return np.gradient(y, 1.0 / max(fps, 1e-6))


def _mean_abs_second_diff(arr: np.ndarray) -> float | None:
    if arr.size < 5:
        return None
    return float(np.mean(np.abs(np.diff(arr, n=2))))
