"""
Traceable 3D joint path from ``GaitMotionRecording`` snapshots.

Source array
------------
``recording.snapshot_at(i).joints[joint_id].position`` — hip-centered
canonical XYZ written by the pose adapter after MediaPipe → 3D reconstruction.
This is **not** MediaPipe image landmarks, not the 15-column reconstruction
export, and not an interpolated display spline.

Units
-----
Recording coordinates are monocular, stature-normalized or heuristic meters
(``sw_canonical_y_up``). They are **not** optically calibrated lab meters.
Displayed centimetres after ``stature_display_scale`` are **estimated /
relative** body-scale units.

Missing data
------------
Frames with a missing or non-finite joint are omitted. Consecutive valid
samples separated by more than ``1.5 / fps`` seconds are a tracking gap:
that interval is **not** added to path length and should not be drawn as
a connecting segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from stablewalk.models.gait_motion import Vec3

SOURCE_ARRAY = "GaitMotionRecording.snapshots[i].joints[joint_id].position"
UNITS_RECORDING = "m (monocular canonical, heuristic Z)"
UNITS_DISPLAY_CM = "cm (stature-scaled estimate, not mocap)"
COORDINATE_FRAME = "sw_canonical_y_up  +X right  +Y up  +Z forward"


@dataclass(frozen=True)
class JointPathSample:
    """One reconstructed sample that may be plotted."""

    frame_index: int
    time_s: float
    position: Vec3
    visibility: float | None = None


@dataclass(frozen=True)
class JointPathTrace:
    """Independent geometry of a reconstructed joint path."""

    joint_id: str
    source_array: str
    coordinate_system: str
    units: str
    units_note: str
    samples: tuple[JointPathSample, ...]
    path_length_m: float
    prefix_length_m: tuple[float, ...]
    valid_segment_count: int
    skipped_gap_count: int
    mean_visibility: float | None
    fps: float


def is_finite_vec(point: Vec3 | None) -> bool:
    return bool(
        point is not None
        and math.isfinite(float(point.x))
        and math.isfinite(float(point.y))
        and math.isfinite(float(point.z))
    )


def euclidean_m(a: Vec3, b: Vec3) -> float:
    dx = float(b.x) - float(a.x)
    dy = float(b.y) - float(a.y)
    dz = float(b.z) - float(a.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def max_connected_dt(fps: float) -> float:
    """Largest timestamp step treated as a continuous sample interval."""
    rate = float(fps) if fps and fps > 1e-6 else 30.0
    return 1.5 / rate


def segment_is_connected(t0: float, t1: float, *, fps: float) -> bool:
    dt = float(t1) - float(t0)
    if dt <= 1e-12:
        return False
    return dt <= max_connected_dt(fps) + 1e-12


def series_with_gap_breaks(
    times: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    fps: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert a NaN sample between finite values whose Δt exceeds the connected-run gate.

    Finite zeros are preserved. Samples that are already NaN stay NaN. Plotting
    the returned arrays lets matplotlib break the stroke at tracking gaps
    instead of drawing a chord across missing time.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    n = int(t.size)
    if n == 0:
        return t.copy(), y.copy()
    if fps is None:
        dts = np.diff(t)
        dts = dts[np.isfinite(dts) & (dts > 1e-9)]
        fps = float(1.0 / np.median(dts)) if dts.size else 30.0
    max_dt = max_connected_dt(fps) + 1e-12
    t_out: list[float] = []
    y_out: list[float] = []
    prev_finite: int | None = None
    for i in range(n):
        ti = float(t[i]) if np.isfinite(t[i]) else float("nan")
        yi = float(y[i]) if np.isfinite(y[i]) else float("nan")
        if (
            prev_finite is not None
            and np.isfinite(ti)
            and np.isfinite(yi)
            and np.isfinite(t[prev_finite])
            and np.isfinite(y[prev_finite])
        ):
            dt = ti - float(t[prev_finite])
            if dt > max_dt:
                t_out.append(0.5 * (float(t[prev_finite]) + ti))
                y_out.append(float("nan"))
        t_out.append(ti)
        y_out.append(yi)
        if np.isfinite(ti) and np.isfinite(yi):
            prev_finite = i
    return np.asarray(t_out, dtype=float), np.asarray(y_out, dtype=float)


def polyline_length_m(points: Sequence[Vec3]) -> float:
    """Sum of Euclidean segments — no gap rule (all consecutive pairs)."""
    if len(points) < 2:
        return 0.0
    return sum(euclidean_m(a, b) for a, b in zip(points, points[1:], strict=False))


def gap_aware_prefix_lengths_m(
    samples: Sequence[JointPathSample] | Sequence[tuple[Vec3, float]],
    *,
    fps: float,
) -> tuple[float, ...]:
    """Prefix traveled distance; gaps contribute 0."""
    if not samples:
        return ()
    prefixes = [0.0]
    prev_p, prev_t = _unpack(samples[0])
    for item in samples[1:]:
        point, time_s = _unpack(item)
        if (
            is_finite_vec(prev_p)
            and is_finite_vec(point)
            and segment_is_connected(prev_t, time_s, fps=fps)
        ):
            prefixes.append(prefixes[-1] + euclidean_m(prev_p, point))
        else:
            prefixes.append(prefixes[-1])
        prev_p, prev_t = point, time_s
    return tuple(prefixes)


def _unpack(item: JointPathSample | tuple[Any, ...]) -> tuple[Vec3, float]:
    if isinstance(item, JointPathSample):
        return item.position, float(item.time_s)
    return item[0], float(item[1])


def index_nearest_time(
    samples: Sequence[JointPathSample] | Sequence[tuple[Vec3, float]],
    time_s: float,
    *,
    frame_index: int | None = None,
) -> int:
    """Playhead sample index — timestamp first, then frame proximity."""
    if not samples:
        return 0
    target = float(time_s)

    def _time(item: JointPathSample | tuple[Any, ...]) -> float:
        if isinstance(item, JointPathSample):
            return float(item.time_s)
        return float(item[1])

    def _frame(i: int) -> int:
        item = samples[i]
        if isinstance(item, JointPathSample):
            return int(item.frame_index)
        return i

    return min(
        range(len(samples)),
        key=lambda i: (
            abs(_time(samples[i]) - target),
            abs(_frame(i) - int(frame_index if frame_index is not None else i)),
        ),
    )


def instantaneous_speed_m_s(
    samples: Sequence[JointPathSample] | Sequence[tuple[Vec3, float]],
    index: int,
    *,
    fps: float,
) -> float | None:
    """Central or backward difference over a connected pair (m/s)."""
    if index < 1 or index >= len(samples):
        return None
    p0, t0 = _unpack(samples[index - 1])
    p1, t1 = _unpack(samples[index])
    if not segment_is_connected(t0, t1, fps=fps):
        return None
    dt = float(t1) - float(t0)
    if dt <= 1e-12:
        return None
    return euclidean_m(p0, p1) / dt


def mean_visibility(samples: Sequence[JointPathSample]) -> float | None:
    vals = [s.visibility for s in samples if s.visibility is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def trace_joint_path(
    recording: Any,
    joint_id: str,
    end_frame: float | int,
    *,
    position_scale: float = 1.0,
    transform: Callable[[Vec3, int], Vec3] | None = None,
) -> JointPathTrace:
    """Walk snapshots 0..end and collect finite reconstructed positions."""
    frame_count = int(getattr(recording, "frame_count", 0) or 0)
    fps = float(getattr(recording, "fps", 0.0) or 0.0) or 30.0
    last = int(end_frame)
    last = max(0, min(last, max(frame_count - 1, 0)))
    samples: list[JointPathSample] = []
    for index in range(0, last + 1):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        sample = snap.joints.get(joint_id) if snap.joints else None
        if sample is None:
            continue
        position = sample.position
        if transform is not None:
            position = transform(position, int(getattr(snap, "frame_index", index)))
        if abs(float(position_scale) - 1.0) > 1e-12:
            position = Vec3(
                position.x * position_scale,
                position.y * position_scale,
                position.z * position_scale,
            )
        if not is_finite_vec(position):
            continue
        vis = None
        meta = getattr(snap, "metadata", None) or {}
        vis_map = meta.get("landmark_visibility") if isinstance(meta, dict) else None
        if isinstance(vis_map, dict) and joint_id in vis_map:
            try:
                vis = float(vis_map[joint_id])
            except (TypeError, ValueError):
                vis = None
        samples.append(
            JointPathSample(
                frame_index=int(getattr(snap, "frame_index", index)),
                time_s=float(getattr(snap, "time_s", index / fps)),
                position=position,
                visibility=vis,
            )
        )
    prefixes = gap_aware_prefix_lengths_m(samples, fps=fps)
    length = float(prefixes[-1]) if prefixes else 0.0
    connected = 0
    gaps = 0
    for i in range(1, len(samples)):
        if segment_is_connected(samples[i - 1].time_s, samples[i].time_s, fps=fps):
            connected += 1
        else:
            gaps += 1
    return JointPathTrace(
        joint_id=joint_id,
        source_array=SOURCE_ARRAY,
        coordinate_system=COORDINATE_FRAME,
        units=UNITS_RECORDING if abs(position_scale - 1.0) < 1e-12 else UNITS_DISPLAY_CM,
        units_note=(
            "Monocular reconstructed XYZ. Distances are estimated / relative; "
            "not optically calibrated laboratory metres."
        ),
        samples=tuple(samples),
        path_length_m=length,
        prefix_length_m=prefixes,
        valid_segment_count=connected,
        skipped_gap_count=gaps,
        mean_visibility=mean_visibility(samples),
        fps=fps,
    )
