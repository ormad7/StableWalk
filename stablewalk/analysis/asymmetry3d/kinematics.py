"""Frame-by-frame 3D kinematics from the existing GaitMotionRecording.

This module does **not** run a second reconstruction. Positions come from
the MediaPipe → canonical adapter already stored on each snapshot.
Coordinates remain ``sw_canonical_y_up`` (pelvis origin, +X right, +Y up,
+Z forward) in **body-normalized metres** (~1.0 body height) unless the
recording metadata marks a calibrated metric session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stablewalk.analysis.spectral.series import interpolate_short_gaps, smooth_preserve_gait
from stablewalk.coordinates.coordinate_map import (
    CANONICAL_FRAME_ID,
    CANONICAL_UNITS,
    TARGET_SKELETON_HEIGHT,
)
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.joint_registry import DOF_IDS, JOINT_IDS, ROOT_JOINT_ID
from stablewalk.pose.reconstruction import ASSUMPTIONS as RECONSTRUCTION_ASSUMPTIONS
from stablewalk.pose.skeleton_3d import LIMB_RATIOS

PROVENANCE_INFERRED = "inferred_monocular_3d"
PROVENANCE_CALIBRATED = "calibrated_metric"

VISIBILITY_MIN = 0.50
MAX_GAP_S = 0.10
SMOOTH_WINDOW_S = 0.12
MAX_WALKING_OMEGA_DEG_S = 1200.0
MAX_WALKING_ALPHA_DEG_S2 = 25000.0
BONE_LENGTH_REL_TOL = 0.30

# Canonical lower-limb + trunk bones used for length constraints.
BONE_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("pelvis", "spine"),
)

PAIRED_DOFS: tuple[tuple[str, str, str], ...] = (
    ("left_hip_flexion", "right_hip_flexion", "hip"),
    ("left_knee_flexion", "right_knee_flexion", "knee"),
    ("left_ankle_flexion", "right_ankle_flexion", "ankle"),
)

PAIRED_JOINTS: tuple[tuple[str, str], ...] = (
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_heel", "right_heel"),
    ("left_toe", "right_toe"),
)

# Sagittal segment tilt: proximal → distal (same convention as four-gait signals).
SEGMENT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("left_thigh", "left_hip", "left_knee"),
    ("right_thigh", "right_hip", "right_knee"),
    ("left_shank", "left_knee", "left_ankle"),
    ("right_shank", "right_knee", "right_ankle"),
)

_LIMB_KEY_MAP = {
    "mid_hip": "pelvis",
    "mid_shoulder": "spine",
    "nose": "head",
}

TRACKED_JOINTS: tuple[str, ...] = tuple(
    dict.fromkeys((*JOINT_IDS, ROOT_JOINT_ID))
)


def infer_coordinate_provenance(
    recording: GaitMotionRecording,
    *,
    calibrated_metric: bool = False,
) -> tuple[str, str, str]:
    """Return (provenance, coordinate_system, units)."""
    meta = recording.metadata or {}
    if calibrated_metric or bool(meta.get("calibrated_metric")):
        units = str(meta.get("metric_units") or "m")
        return PROVENANCE_CALIBRATED, str(recording.coordinate_system), units
    frame = str(recording.coordinate_system or CANONICAL_FRAME_ID)
    units = "body_normalized_m"
    if frame in (CANONICAL_FRAME_ID, "pelvis_centered_y_up_z_forward", "sw_canonical_y_up"):
        return PROVENANCE_INFERRED, CANONICAL_FRAME_ID, units
    return PROVENANCE_INFERRED, frame, units


def _expected_bone_length(parent: str, child: str, stature: float) -> float | None:
    for (a, b), ratio in LIMB_RATIOS.items():
        pa = _LIMB_KEY_MAP.get(a, a)
        pb = _LIMB_KEY_MAP.get(b, b)
        if {pa, pb} == {parent, child}:
            return float(ratio) * float(stature)
    return None


def observed_stature(
    positions: np.ndarray,
    joint_ids: tuple[str, ...],
) -> float | None:
    """Median reconstructed stature from head–foot and hip–foot spans.

    Demo recordings are often labelled body-normalized but keep a raw
    vertical span of ~2–3 instead of 1.0. Using the observed span keeps
    Winter ratios on the same scale as the stored joints.
    """
    if positions.ndim != 3 or positions.shape[0] < 2:
        return None
    index = {jid: i for i, jid in enumerate(joint_ids)}
    head_i = next((index[j] for j in ("head", "nose") if j in index), None)
    ankle_ids = [
        index[j]
        for j in ("left_ankle", "right_ankle", "left_heel", "right_heel")
        if j in index
    ]
    hip_ids = [index[j] for j in ("left_hip", "right_hip") if j in index]
    estimates: list[float] = []
    if ankle_ids:
        feet = positions[:, ankle_ids, 1]
        if not np.isfinite(feet).any():
            return None
        foot_y = np.nanmin(np.where(np.isfinite(feet), feet, np.inf), axis=1)
        foot_y = np.where(np.isfinite(foot_y), foot_y, np.nan)
        if head_i is not None:
            head_span = positions[:, head_i, 1] - foot_y
            finite = head_span[np.isfinite(head_span) & (head_span > 0.12)]
            if finite.size >= 4:
                estimates.append(float(np.median(finite)))
        if hip_ids:
            hips = positions[:, hip_ids, 1]
            if np.isfinite(hips).any():
                with np.errstate(all="ignore"):
                    hip_y = np.nanmean(hips, axis=1)
                hip_span = (hip_y - foot_y) / 0.53
                finite = hip_span[np.isfinite(hip_span) & (hip_span > 0.20)]
                if finite.size >= 4:
                    estimates.append(float(np.median(finite)))
    if not estimates:
        return None
    med = float(np.median(np.asarray(estimates, dtype=float)))
    if not np.isfinite(med) or med < 0.20:
        return None
    return med


def _visibility(snap, joint_id: str) -> float:
    vis = (snap.metadata or {}).get("landmark_visibility") or {}
    if joint_id in vis:
        try:
            return float(vis[joint_id])
        except (TypeError, ValueError):
            return 0.0
    return 1.0 if joint_id in snap.joints else 0.0


def _central_diff(y: np.ndarray, dt: float) -> np.ndarray:
    out = np.full(y.shape, np.nan, dtype=float)
    if y.size < 2 or dt <= 0:
        return out
    interior = (y[2:] - y[:-2]) / (2.0 * dt)
    ok = np.isfinite(y[2:]) & np.isfinite(y[:-2])
    out[1:-1] = np.where(ok, interior, np.nan)
    if np.isfinite(y[0]) and np.isfinite(y[1]):
        out[0] = (y[1] - y[0]) / dt
    if np.isfinite(y[-1]) and np.isfinite(y[-2]):
        out[-1] = (y[-1] - y[-2]) / dt
    return out


def _percentile_span(values: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return None
    p_lo, p_hi = np.percentile(finite, [lo, hi])
    return float(p_hi - p_lo)


@dataclass
class KinematicBundle:
    """Synchronized 3D kinematics on the recording timeline."""

    times_s: np.ndarray
    fps: float
    coordinate_system: str
    provenance: str
    units: str
    joint_ids: tuple[str, ...]
    positions: np.ndarray  # (T, J, 3)
    visibility: np.ndarray  # (T, J)
    dof_ids: tuple[str, ...]
    angle_deg: np.ndarray  # (T, D)
    omega_deg_s: np.ndarray
    alpha_deg_s2: np.ndarray
    bone_pairs: tuple[tuple[str, str], ...]
    orientations: np.ndarray  # (T, B, 3) unit bone vectors
    bone_length: np.ndarray  # (T, B)
    bone_length_expected: np.ndarray  # (B,)
    low_confidence: np.ndarray  # (T,)
    unrealistic: np.ndarray  # (T,)
    missing_fraction: float
    warnings: list[str] = field(default_factory=list)
    reconstruction_note: str = RECONSTRUCTION_ASSUMPTIONS.strip().splitlines()[0]
    segment_ids: tuple[str, ...] = field(
        default_factory=lambda: tuple(spec[0] for spec in SEGMENT_SPECS)
    )
    segment_deg: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=float))
    stature: float | None = None

    @property
    def n_frames(self) -> int:
        return int(self.times_s.size)

    def joint_index(self, joint_id: str) -> int | None:
        try:
            return self.joint_ids.index(joint_id)
        except ValueError:
            return None

    def dof_index(self, dof_id: str) -> int | None:
        try:
            return self.dof_ids.index(dof_id)
        except ValueError:
            return None

    def series_for_joint(self, joint_id: str, axis: int) -> np.ndarray:
        idx = self.joint_index(joint_id)
        if idx is None:
            return np.full(self.n_frames, np.nan)
        return self.positions[:, idx, axis]

    def series_for_dof(self, dof_id: str) -> np.ndarray:
        idx = self.dof_index(dof_id)
        if idx is None:
            return np.full(self.n_frames, np.nan)
        return self.angle_deg[:, idx]

    def omega_for_dof(self, dof_id: str) -> np.ndarray:
        idx = self.dof_index(dof_id)
        if idx is None:
            return np.full(self.n_frames, np.nan)
        return self.omega_deg_s[:, idx]

    def series_for_segment(self, segment_id: str) -> np.ndarray:
        try:
            idx = self.segment_ids.index(segment_id)
        except ValueError:
            return np.full(self.n_frames, np.nan)
        if self.segment_deg.ndim != 2 or idx >= self.segment_deg.shape[1]:
            return np.full(self.n_frames, np.nan)
        return self.segment_deg[:, idx]

    def quality_summary(self) -> dict[str, Any]:
        n = max(self.n_frames, 1)
        return {
            "n_frames": self.n_frames,
            "fps": self.fps,
            "missing_fraction": self.missing_fraction,
            "low_confidence_fraction": float(np.mean(self.low_confidence)) if n else 0.0,
            "unrealistic_fraction": float(np.mean(self.unrealistic)) if n else 0.0,
            "provenance": self.provenance,
            "coordinate_system": self.coordinate_system,
            "units": self.units,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.quality_summary(),
            "joint_ids": list(self.joint_ids),
            "dof_ids": list(self.dof_ids),
            "warnings": list(self.warnings),
            "reconstruction_note": self.reconstruction_note,
            "kind": "derived",
        }


def build_kinematics(
    recording: GaitMotionRecording,
    *,
    calibrated_metric: bool = False,
) -> KinematicBundle:
    """Extract positions, orientations, angles, ω, α from existing snapshots."""
    provenance, frame, units = infer_coordinate_provenance(
        recording, calibrated_metric=calibrated_metric
    )
    snaps = list(recording.snapshots)
    fps = float(recording.fps or 30.0)
    fps = fps if fps > 1e-6 else 30.0
    dt = 1.0 / fps
    n = len(snaps)
    joint_ids = TRACKED_JOINTS
    dof_ids = tuple(d for d in DOF_IDS if d.endswith("_flexion") or d in DOF_IDS)
    j_count = len(joint_ids)
    d_count = len(dof_ids)

    times = np.array([float(s.time_s) for s in snaps], dtype=float) if snaps else np.zeros(0)
    if times.size >= 2:
        deltas = np.diff(times)
        positive = deltas[deltas > 1e-9]
        if positive.size:
            dt = float(np.median(positive))
            fps = 1.0 / dt

    positions = np.full((n, j_count, 3), np.nan, dtype=float)
    visibility = np.zeros((n, j_count), dtype=float)
    angles = np.full((n, d_count), np.nan, dtype=float)
    omega_snap = np.full((n, d_count), np.nan, dtype=float)

    from stablewalk.pose.knee_convention import (
        analysis_knee_flexion_deg,
        is_knee_flexion_dof,
        recording_knee_storage,
    )

    knee_storage = recording_knee_storage(recording)
    for t, snap in enumerate(snaps):
        detected = bool((snap.metadata or {}).get("detected", True))
        for j, jid in enumerate(joint_ids):
            vis = _visibility(snap, jid)
            visibility[t, j] = vis
            sample = snap.joints.get(jid)
            if sample is None or not detected or vis < VISIBILITY_MIN:
                continue
            positions[t, j, 0] = float(sample.position.x)
            positions[t, j, 1] = float(sample.position.y)
            positions[t, j, 2] = float(sample.position.z)
        for d, did in enumerate(dof_ids):
            if is_knee_flexion_dof(did):
                val = analysis_knee_flexion_deg(snap, did, storage=knee_storage)
                if val is None or not detected:
                    continue
                angles[t, d] = float(val)
                continue
            dof = snap.get_dof(did)
            if dof is None or dof.angle_deg is None or not detected:
                continue
            val = float(dof.angle_deg)
            if not math.isfinite(val):
                continue
            angles[t, d] = val
            if dof.velocity_deg_s is not None and math.isfinite(float(dof.velocity_deg_s)):
                omega_snap[t, d] = float(dof.velocity_deg_s)

    warnings: list[str] = []
    max_gap = max(1, int(round(MAX_GAP_S * fps)))
    win = max(5, int(round(SMOOTH_WINDOW_S * fps)) | 1)

    pos_smooth = positions.copy()
    for j in range(j_count):
        for ax in range(3):
            col = positions[:, j, ax]
            filled, _interp = interpolate_short_gaps(col, max_gap_samples=max_gap)
            pos_smooth[:, j, ax] = smooth_preserve_gait(
                filled, window_samples=win, polyorder=2
            )

    ang_smooth = angles.copy()
    for d in range(d_count):
        filled, _interp = interpolate_short_gaps(angles[:, d], max_gap_samples=max_gap)
        ang_smooth[:, d] = smooth_preserve_gait(filled, window_samples=win, polyorder=2)

    omega = np.full_like(ang_smooth, np.nan)
    alpha = np.full_like(ang_smooth, np.nan)
    for d in range(d_count):
        derived = _central_diff(ang_smooth[:, d], dt)
        # Prefer snapshot ω when present and finite; otherwise filtered FD.
        use = omega_snap[:, d].copy()
        missing = ~np.isfinite(use)
        use[missing] = derived[missing]
        omega[:, d] = use
        alpha[:, d] = _central_diff(use, dt)

    bone_len = np.full((n, len(BONE_PAIRS)), np.nan, dtype=float)
    orientations = np.full((n, len(BONE_PAIRS), 3), np.nan, dtype=float)
    expected = np.full(len(BONE_PAIRS), np.nan, dtype=float)
    j_index = {jid: i for i, jid in enumerate(joint_ids)}
    stature = observed_stature(pos_smooth, joint_ids)
    scale = float(stature) if stature is not None else float(TARGET_SKELETON_HEIGHT)
    for b, (parent, child) in enumerate(BONE_PAIRS):
        exp = _expected_bone_length(parent, child, scale)
        expected[b] = float(exp) if exp is not None else float("nan")
        ip = j_index.get(parent)
        ic = j_index.get(child)
        if ip is None or ic is None:
            continue
        vec = pos_smooth[:, ic, :] - pos_smooth[:, ip, :]
        lengths = np.linalg.norm(vec, axis=1)
        bone_len[:, b] = lengths
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = vec / lengths[:, None]
        orientations[:, b, :] = unit

    # Session-median bone length as the constraint target when ratios missing.
    median_len = np.array(
        [
            float(np.nanmedian(bone_len[:, b])) if np.isfinite(bone_len[:, b]).any() else np.nan
            for b in range(len(BONE_PAIRS))
        ],
        dtype=float,
    )
    # Inferred monocular scale is often not 1.0 body height. Frame-to-frame
    # stretch is judged against the session median, not Winter table lengths.
    if provenance == PROVENANCE_INFERRED:
        stretch_target = median_len
        target = np.where(np.isfinite(expected), expected, median_len)
    else:
        stretch_target = np.where(np.isfinite(expected), expected, median_len)
        target = stretch_target

    low_conf = np.zeros(n, dtype=bool)
    unrealistic = np.zeros(n, dtype=bool)
    vis_ok = visibility >= VISIBILITY_MIN
    lower = [
        j_index[j]
        for j in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
        if j in j_index
    ]
    for t in range(n):
        if lower:
            frac = float(np.mean(vis_ok[t, lower]))
            if frac < 0.5:
                low_conf[t] = True
        stretch = False
        for b in range(len(BONE_PAIRS)):
            L = bone_len[t, b]
            tgt = stretch_target[b]
            if not np.isfinite(L) or not np.isfinite(tgt) or tgt < 1e-6:
                continue
            if abs(L - tgt) / tgt > BONE_LENGTH_REL_TOL:
                stretch = True
                break
        if stretch:
            unrealistic[t] = True
        om = omega[t][np.isfinite(omega[t])]
        al = alpha[t][np.isfinite(alpha[t])]
        if om.size and float(np.max(np.abs(om))) > MAX_WALKING_OMEGA_DEG_S:
            unrealistic[t] = True
        if al.size and float(np.max(np.abs(al))) > MAX_WALKING_ALPHA_DEG_S2:
            unrealistic[t] = True

    segment_ids = tuple(spec[0] for spec in SEGMENT_SPECS)
    segment_deg = np.full((n, len(SEGMENT_SPECS)), np.nan, dtype=float)
    for s, (_name, prox_id, dist_id) in enumerate(SEGMENT_SPECS):
        ip = j_index.get(prox_id)
        ic = j_index.get(dist_id)
        if ip is None or ic is None:
            continue
        dy = pos_smooth[:, ic, 1] - pos_smooth[:, ip, 1]
        dz = pos_smooth[:, ic, 2] - pos_smooth[:, ip, 2]
        ok = np.isfinite(dy) & np.isfinite(dz)
        tilt = np.full(n, np.nan, dtype=float)
        tilt[ok] = np.degrees(np.arctan2(dz[ok], -dy[ok]))
        filled, _interp = interpolate_short_gaps(tilt, max_gap_samples=max_gap)
        segment_deg[:, s] = smooth_preserve_gait(
            filled, window_samples=win, polyorder=2
        )

    missing = float(np.mean(~np.isfinite(pos_smooth[:, :, 0]))) if n else 1.0
    if provenance == PROVENANCE_INFERRED:
        warnings.append(
            "Joint XYZ are inferred monocular 3D (body-normalized), not calibrated mocap metres."
        )
    if missing > 0.25:
        warnings.append(f"High missing joint fraction ({missing:.0%}).")
    if n and float(np.mean(unrealistic)) > 0.15:
        warnings.append("Frequent unrealistic bone lengths or angular rates.")

    return KinematicBundle(
        times_s=times,
        fps=fps,
        coordinate_system=frame,
        provenance=provenance,
        units=units,
        joint_ids=joint_ids,
        positions=pos_smooth,
        visibility=visibility,
        dof_ids=dof_ids,
        angle_deg=ang_smooth,
        omega_deg_s=omega,
        alpha_deg_s2=alpha,
        bone_pairs=BONE_PAIRS,
        orientations=orientations,
        bone_length=bone_len,
        bone_length_expected=target,
        low_confidence=low_conf,
        unrealistic=unrealistic,
        missing_fraction=missing,
        warnings=warnings,
        reconstruction_note=RECONSTRUCTION_ASSUMPTIONS.strip(),
        segment_ids=segment_ids,
        segment_deg=segment_deg,
        stature=stature,
    )


def rom_percentile(values: np.ndarray, *, lo: float = 5.0, hi: float = 95.0) -> float | None:
    return _percentile_span(values, lo, hi)


def peak_abs(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    return float(np.max(np.abs(finite)))


def rms(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    return float(np.sqrt(np.mean(finite * finite)))
