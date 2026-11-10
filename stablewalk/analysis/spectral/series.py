"""Extract and preprocess synchronized gait channels.

Missing landmarks become NaN (never the ``Vec3(0,0,0)`` filler used by
``JointTimeSeries.positions``). Short gaps may be interpolated; long gaps
and series edges are never extrapolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from stablewalk.analysis.spectral.channels import (
    ALL_CHANNEL_SPECS,
    CHANNEL_BY_ID,
    ChannelSpec,
)
from stablewalk.analysis.spectral.config import SpectralAnalysisConfig
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.joint_registry import ROOT_JOINT_ID
from stablewalk.models.pose_data import PoseSequence

KIND_DERIVED = "derived"


@dataclass
class ChannelSeries:
    """One scalar channel: raw (resampled) and processed copies."""

    spec: ChannelSpec
    times_s: np.ndarray
    raw: np.ndarray
    processed: np.ndarray
    valid_mask: np.ndarray
    visibility: np.ndarray
    missing_fraction: float
    interpolated_fraction: float
    fps: float
    coordinate_system: str
    mean_visibility: float = 0.0
    warnings: list[str] = field(default_factory=list)
    # Z-scored processed samples (NaNs preserved). Used for ACF / FFT / PSD.
    normalized: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))

    @property
    def n_valid(self) -> int:
        return int(np.isfinite(self.processed).sum())

    @property
    def duration_s(self) -> float:
        if self.times_s.size < 2:
            return 0.0
        return float(self.times_s[-1] - self.times_s[0])

    def finite_processed(self) -> np.ndarray:
        return self.processed[np.isfinite(self.processed)]


@dataclass
class SpectralSeriesBundle:
    """Uniform-grid bundle of gait channels."""

    times_s: np.ndarray
    fps: float
    coordinate_system: str
    channels: dict[str, ChannelSeries]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, channel_id: str) -> ChannelSeries | None:
        return self.channels.get(channel_id)

    def matrix_block(
        self,
        channel_ids: Sequence[str],
        *,
        processed: bool = True,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Return (T, C) array, kept ids, and row-valid mask (all finite)."""
        cols: list[np.ndarray] = []
        kept: list[str] = []
        for cid in channel_ids:
            series = self.channels.get(cid)
            if series is None:
                continue
            col = series.processed if processed else series.raw
            if np.isfinite(col).sum() < 8:
                continue
            cols.append(col)
            kept.append(cid)
        if not cols:
            t = self.times_s.size
            return np.zeros((t, 0), dtype=float), [], np.zeros(t, dtype=bool)
        data = np.column_stack(cols)
        row_ok = np.all(np.isfinite(data), axis=1)
        return data, kept, row_ok


def _uniform_grid(times: np.ndarray, fps: float) -> np.ndarray:
    if times.size == 0:
        return np.asarray([], dtype=float)
    t0 = float(times[0])
    t1 = float(times[-1])
    if t1 <= t0:
        return np.asarray([t0], dtype=float)
    dt = 1.0 / max(fps, 1e-6)
    n = int(round((t1 - t0) / dt)) + 1
    n = max(n, 1)
    return t0 + dt * np.arange(n, dtype=float)


def _resample_to_grid(
    src_t: np.ndarray,
    src_y: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Resample finite source samples onto a uniform FPS grid.

    Linear interpolation is used between adjacent finite samples whose gap is
    at most ~1.5 frames. Missing landmarks stay NaN — coordinates are never
    invented across dropped detections or at the series edges.
    """
    out = np.full(grid.shape, np.nan, dtype=float)
    if src_t.size == 0 or grid.size == 0:
        return out
    dt = abs(float(grid[1] - grid[0])) if grid.size > 1 else 1.0
    half = 0.51 * dt
    max_gap = 1.51 * dt
    finite = np.isfinite(src_t) & np.isfinite(src_y)
    if not np.any(finite):
        return out
    t = np.asarray(src_t[finite], dtype=float)
    y = np.asarray(src_y[finite], dtype=float)
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    for i, tg in enumerate(grid):
        j = int(np.searchsorted(t, tg))
        if j < t.size and abs(float(t[j]) - float(tg)) <= half:
            out[i] = float(y[j])
            continue
        if j > 0 and abs(float(t[j - 1]) - float(tg)) <= half:
            out[i] = float(y[j - 1])
            continue
        if 0 < j < t.size:
            t0, t1 = float(t[j - 1]), float(t[j])
            span = t1 - t0
            if span <= 0.0 or span > max_gap:
                continue
            w = (float(tg) - t0) / span
            if w < 0.0 or w > 1.0:
                continue
            out[i] = (1.0 - w) * float(y[j - 1]) + w * float(y[j])
    return out


def normalize_unit_variance(values: np.ndarray) -> np.ndarray:
    """Z-score finite samples (mean 0, std 1). Constants become 0. NaNs stay NaN."""
    y = np.asarray(values, dtype=float).copy()
    mask = np.isfinite(y)
    if int(mask.sum()) < 2:
        return y
    mu = float(np.mean(y[mask]))
    sd = float(np.std(y[mask]))
    if sd < 1e-18:
        y[mask] = 0.0
        return y
    y[mask] = (y[mask] - mu) / sd
    return y


def interpolate_short_gaps(
    values: np.ndarray,
    *,
    max_gap_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-fill interior NaN runs no longer than ``max_gap_samples``.

    Leading/trailing NaNs are never filled (no extrapolation).
    Returns (filled, interpolated_mask).
    """
    y = np.asarray(values, dtype=float).copy()
    interpolated = np.zeros(y.shape, dtype=bool)
    if y.size == 0 or max_gap_samples <= 0:
        return y, interpolated
    finite = np.isfinite(y)
    n = y.size
    i = 0
    while i < n:
        if finite[i]:
            i += 1
            continue
        j = i
        while j < n and not finite[j]:
            j += 1
        gap = j - i
        left = i - 1
        right = j
        interior = left >= 0 and right < n
        short = gap <= max_gap_samples
        if interior and short:
            y0 = y[left]
            y1 = y[right]
            span = right - left
            for k in range(i, j):
                w = (k - left) / span
                y[k] = (1.0 - w) * y0 + w * y1
                interpolated[k] = True
        i = j
    return y, interpolated


def _savgol_coeffs(window_length: int, polyorder: int) -> np.ndarray:
    half = window_length // 2
    x = np.arange(-half, half + 1, dtype=float)
    order = min(polyorder, window_length - 1)
    a = np.vander(x, N=order + 1, increasing=True)
    ata = a.T @ a
    coeffs = np.linalg.solve(ata, a.T)[0]
    return coeffs.astype(float)


def smooth_preserve_gait(
    values: np.ndarray,
    *,
    window_samples: int,
    polyorder: int,
) -> np.ndarray:
    """Centered Savitzky–Golay on finite samples; NaNs stay NaN.

    Window is odd and short relative to a gait cycle so ~1–3 Hz content remains.
    """
    y = np.asarray(values, dtype=float).copy()
    n = y.size
    w = int(window_samples)
    if w < 3 or n < 5:
        return y
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 3:
        return y
    polyorder = max(1, min(int(polyorder), w - 1))
    coeffs = _savgol_coeffs(w, polyorder)
    half = w // 2
    out = y.copy()
    for i in range(half, n - half):
        sl = y[i - half : i + half + 1]
        if not np.all(np.isfinite(sl)):
            continue
        out[i] = float(np.dot(coeffs, sl))
    return out


def detrend_signal(values: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(values, dtype=float).copy()
    mask = np.isfinite(y)
    if mask.sum() < 2 or mode in ("none", "", None):
        return y
    x = np.arange(y.size, dtype=float)
    if mode == "mean":
        y[mask] = y[mask] - float(np.mean(y[mask]))
        return y
    if mode == "linear":
        coef = np.polyfit(x[mask], y[mask], 1)
        y[mask] = y[mask] - np.polyval(coef, x[mask])
        return y
    raise ValueError(f"Unknown detrend mode: {mode}")


def _visibility_for_joint(snap, joint_id: str | None) -> float:
    if not joint_id:
        return 1.0
    vis = (snap.metadata or {}).get("landmark_visibility") or {}
    if joint_id in vis:
        try:
            return float(vis[joint_id])
        except (TypeError, ValueError):
            return 0.0
    return 1.0 if joint_id in snap.joints else 0.0


def _dof_value(snap, dof_id: str | None, *, storage: str | None = None) -> float:
    if not dof_id:
        return float("nan")
    from stablewalk.pose.knee_convention import (
        analysis_knee_flexion_deg,
        is_knee_flexion_dof,
    )

    if is_knee_flexion_dof(dof_id) and storage is not None:
        value = analysis_knee_flexion_deg(snap, dof_id, storage=storage)  # type: ignore[arg-type]
        return float(value) if value is not None else float("nan")
    dof = snap.get_dof(dof_id)
    if dof is None or dof.angle_deg is None:
        return float("nan")
    val = float(dof.angle_deg)
    return val if math.isfinite(val) else float("nan")


# Body-normalized monocular frames use ~0.4 for a thigh. Below this the
# proximal/distal pair is degenerate and the signed tilt is not defined.
_MIN_SEGMENT_LENGTH = 1e-4


def segment_sagittal_tilt_deg(proximal: Any, distal: Any) -> float:
    """Signed sagittal tilt from downward vertical, degrees.

    Uses the real 3D joint coordinates. Canonical frame: +Y up, +Z forward.
    The angle is the YZ (sagittal) projection: 0° ≈ hanging down; positive
    when the distal end is more anterior. Returns NaN when the 3D segment is
    too short or has no reliable sagittal component.
    """
    try:
        dy = float(distal.y) - float(proximal.y)
        dz = float(distal.z) - float(proximal.z)
        dx = float(distal.x) - float(proximal.x)
    except (TypeError, AttributeError, ValueError):
        return float("nan")
    if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(dz)):
        return float("nan")
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < _MIN_SEGMENT_LENGTH:
        return float("nan")
    sagittal = math.hypot(dz, dy)
    if sagittal < _MIN_SEGMENT_LENGTH:
        return float("nan")
    return float(math.degrees(math.atan2(dz, -dy)))


def _segment_value(snap, spec: ChannelSpec) -> float:
    proximal_id = spec.proximal_joint_id
    distal_id = spec.distal_joint_id
    if not proximal_id or not distal_id:
        return float("nan")
    proximal = snap.joints.get(proximal_id)
    distal = snap.joints.get(distal_id)
    if proximal is None or distal is None:
        return float("nan")
    prox_pos = getattr(proximal, "position", None)
    dist_pos = getattr(distal, "position", None)
    if prox_pos is None or dist_pos is None:
        return float("nan")
    return segment_sagittal_tilt_deg(prox_pos, dist_pos)


def _pos_value(snap, joint_id: str | None, axis: str | None) -> float:
    if not joint_id or not axis:
        return float("nan")
    joint = snap.joints.get(joint_id)
    if joint is None:
        return float("nan")
    val = float(getattr(joint.position, axis))
    return val if math.isfinite(val) else float("nan")


def _pelvis_yaw_deg(snap) -> float:
    lh = snap.joints.get("left_hip")
    rh = snap.joints.get("right_hip")
    if lh is None or rh is None:
        return float("nan")
    dx = float(rh.position.x) - float(lh.position.x)
    dz = float(rh.position.z) - float(lh.position.z)
    if abs(dx) < 1e-12 and abs(dz) < 1e-12:
        return float("nan")
    return float(np.degrees(np.arctan2(dz, dx)))


def _root_trajectory_lookup(sequence: PoseSequence | None) -> dict[int, tuple[float, float, float]]:
    if sequence is None:
        return {}
    from stablewalk.pose.reconstruction import compute_root_trajectory_point

    out: dict[int, tuple[float, float, float]] = {}
    for frame in sequence.frames:
        if not frame.detected or not frame.keypoints:
            continue
        pt = compute_root_trajectory_point(frame.keypoints)
        if pt is None:
            continue
        out[int(frame.frame_index)] = (float(pt[0]), float(pt[1]), float(pt[2]))
    return out


def build_spectral_series(
    recording: GaitMotionRecording,
    *,
    sequence: PoseSequence | None = None,
    config: SpectralAnalysisConfig | None = None,
    channel_ids: Iterable[str] | None = None,
) -> SpectralSeriesBundle:
    """Build uniform-grid raw + processed channels from a gait recording."""
    cfg = config or SpectralAnalysisConfig()
    fps = max(float(recording.fps), 1e-6)
    snaps = list(recording.snapshots)
    warnings: list[str] = []
    if not snaps:
        return SpectralSeriesBundle(
            times_s=np.asarray([], dtype=float),
            fps=fps,
            coordinate_system=str(recording.coordinate_system or cfg.coordinate_system),
            channels={},
            warnings=["No skeleton snapshots in the gait recording."],
        )

    src_t = np.asarray([float(s.time_s) for s in snaps], dtype=float)
    if src_t.size >= 2:
        dts = np.diff(src_t)
        if np.any(dts <= 0):
            warnings.append("Non-monotonic timestamps; using frame index / FPS.")
            src_t = np.arange(len(snaps), dtype=float) / fps
        median_dt = float(np.median(dts[dts > 0])) if np.any(dts > 0) else 1.0 / fps
        if abs(median_dt - 1.0 / fps) > 0.25 / fps:
            warnings.append(
                f"Snapshot spacing ({median_dt:.4f} s) differs from 1/FPS "
                f"({1.0 / fps:.4f} s); resampling onto an FPS grid."
            )
    grid = _uniform_grid(src_t, fps)
    max_gap = max(1, int(round(cfg.max_gap_s * fps)))
    win = max(3, int(round(cfg.smooth_window_s * fps)))
    if win % 2 == 0:
        win += 1

    roots = _root_trajectory_lookup(sequence)
    from stablewalk.pose.knee_convention import recording_knee_storage

    knee_storage = recording_knee_storage(recording)
    specs = list(ALL_CHANNEL_SPECS)
    if channel_ids is not None:
        wanted = {str(c) for c in channel_ids}
        specs = [s for s in specs if s.channel_id in wanted]
        for cid in wanted:
            if cid not in CHANNEL_BY_ID:
                warnings.append(f"Unknown channel id ignored: {cid}")

    channels: dict[str, ChannelSeries] = {}
    for spec in specs:
        raw_src = np.full(len(snaps), np.nan, dtype=float)
        vis_src = np.full(len(snaps), np.nan, dtype=float)
        for i, snap in enumerate(snaps):
            if spec.source == "segment_sagittal":
                vis = min(
                    _visibility_for_joint(snap, spec.proximal_joint_id),
                    _visibility_for_joint(snap, spec.distal_joint_id),
                )
            else:
                vis = _visibility_for_joint(snap, spec.joint_id)
            vis_src[i] = vis
            if spec.source == "segment_sagittal":
                if vis < cfg.visibility_min:
                    continue
                raw_src[i] = _segment_value(snap, spec)
                continue
            if spec.joint_id and vis < cfg.visibility_min:
                continue
            if spec.source == "pose_root_trajectory":
                axis = spec.axis or "x"
                pt = roots.get(int(snap.frame_index))
                if pt is None:
                    continue
                raw_src[i] = {"x": pt[0], "y": pt[1], "z": pt[2]}[axis]
            elif spec.source == "derived_hip_line":
                raw_src[i] = _pelvis_yaw_deg(snap)
            elif spec.dof_id:
                raw_src[i] = _dof_value(snap, spec.dof_id, storage=knee_storage)
            else:
                raw_src[i] = _pos_value(snap, spec.joint_id, spec.axis)
        raw = _resample_to_grid(src_t, raw_src, grid)
        vis = _resample_to_grid(src_t, vis_src, grid)
        filled, interp_mask = interpolate_short_gaps(raw, max_gap_samples=max_gap)
        remaining_nan = ~np.isfinite(filled)
        if remaining_nan.any():
            n_gap = int(remaining_nan.sum())
            if n_gap:
                pass  # left as NaN on purpose
        processed = smooth_preserve_gait(
            filled, window_samples=win, polyorder=cfg.smooth_polyorder
        )
        processed = detrend_signal(processed, cfg.detrend)
        normalized = normalize_unit_variance(processed)
        miss = float(np.mean(~np.isfinite(raw))) if raw.size else 1.0
        interp_frac = float(np.mean(interp_mask)) if interp_mask.size else 0.0
        vis_finite = vis[np.isfinite(vis)]
        mean_vis = float(np.mean(vis_finite)) if vis_finite.size else 0.0
        ch_warn: list[str] = []
        if spec.source == "pose_root_trajectory" and not roots:
            ch_warn.append(
                "Pelvis translation needs the pose sequence (root trajectory); "
                "hip-centered pelvis XYZ is near zero and is not used here."
            )
        if miss > cfg.max_missing_fraction:
            ch_warn.append(
                f"High missing fraction ({miss:.0%}) after confidence masking."
            )
        if spec.kind == "position_m" and spec.joint_id == ROOT_JOINT_ID and spec.source == "gait_motion.joints":
            ch_warn.append("Hip-centered pelvis position is not a walking trajectory.")
        channels[spec.channel_id] = ChannelSeries(
            spec=spec,
            times_s=grid,
            raw=raw,
            processed=processed,
            valid_mask=np.isfinite(processed),
            visibility=vis,
            missing_fraction=miss,
            interpolated_fraction=interp_frac,
            fps=fps,
            coordinate_system=str(recording.coordinate_system or cfg.coordinate_system),
            mean_visibility=mean_vis,
            warnings=ch_warn,
            normalized=normalized,
        )

    duration = float(grid[-1] - grid[0]) if grid.size >= 2 else 0.0
    if duration < cfg.min_duration_s:
        warnings.append(
            f"Clip duration {duration:.2f} s is shorter than {cfg.min_duration_s:.1f} s; "
            "periodic estimates are unreliable."
        )
    if grid.size < cfg.min_samples:
        warnings.append(
            f"Only {grid.size} uniform samples (need ≥ {cfg.min_samples})."
        )
    if fps < 20:
        warnings.append(f"Low FPS ({fps:.1f}); gait harmonics may alias.")

    diffs = np.diff(src_t) if src_t.size >= 2 else np.asarray([])
    drop = int(np.sum(diffs > 1.5 / fps)) if diffs.size else 0
    if drop:
        warnings.append(
            f"{drop} snapshot gap(s) larger than one frame — detections were dropped."
        )

    return SpectralSeriesBundle(
        times_s=grid,
        fps=fps,
        coordinate_system=str(recording.coordinate_system or cfg.coordinate_system),
        channels=channels,
        warnings=warnings,
        metadata={
            "n_snapshots": len(snaps),
            "n_grid": int(grid.size),
            "duration_s": duration,
            "fps": fps,
            "smooth_window_samples": win,
            "max_gap_samples": max_gap,
            "detrend": cfg.detrend,
            "normalized": "zscore",
            "knee_angle_storage": knee_storage,
            "knee_raw_signal": "anatomical_knee_flexion_deg",
            "visibility_min": cfg.visibility_min,
            "has_pose_sequence": sequence is not None,
            "kind": KIND_DERIVED,
        },
    )
