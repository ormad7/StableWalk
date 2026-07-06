"""
Robust heel-strike / foot-contact detection from 2D foot landmarks.

Detects physiologically plausible gait events from smoothed, body-scaled vertical
foot trajectories with peak prominence, swing-amplitude hysteresis, and FPS-aware
minimum step intervals. Designed to reject MediaPipe landmark jitter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from stablewalk.models.pose_data import PoseFrame, PoseSequence

# ---------------------------------------------------------------------------
# Detection parameters (documented, data-driven — not demo-specific)
# ---------------------------------------------------------------------------
MIN_STEP_INTERVAL_S = 0.38
"""Minimum time between same-foot contacts (~1.3 Hz max per foot)."""

SMOOTH_WINDOW_S = 0.18
"""Moving-average window for low-pass smoothing before event detection."""

MIN_PROMINENCE_RATIO = 0.14
"""Local contact peak must rise this fraction of the signal range above surroundings."""

MIN_SWING_RATIO = 0.16
"""Foot must lift this fraction of range (swing) before the next contact counts."""

MIN_RANGE_BODY_RATIO = 0.005
"""Minimum foot oscillation amplitude, scaled by body height."""

INTERP_MAX_GAP = 4
MIN_VALID_FRAMES = 8
VISIBILITY_MIN = 0.30

CADENCE_MIN_HZ = 0.35
"""Minimum plausible total footfall rate (both legs) for high confidence."""

CADENCE_MAX_HZ = 2.8
"""Maximum plausible total footfall rate (both legs) for high confidence."""

MERGED_EVENT_MIN_S = 0.18
"""Minimum time between any two footfalls (either leg) after bilateral merge."""

FOOT_LANDMARK_SUFFIXES = ("heel", "ankle", "foot_index")
ConfidenceLevel = Literal["high", "low"]


@dataclass
class LandmarkSignal:
    """Vertical foot trajectory for one landmark on one leg."""

    name: str
    frame_indices: list[int]
    raw_y: np.ndarray
    body_height: float


@dataclass
class SideStepDetection:
    """Detected contacts for one leg."""

    side: str
    landmark_used: str
    event_frame_indices: list[int]
    interval_cv: float | None
    smoothed_y: np.ndarray
    detrended_y: np.ndarray
    raw_y: np.ndarray
    aligned_frame_indices: list[int]
    rejected_low_prominence: int = 0
    rejected_min_interval: int = 0
    rejected_low_swing: int = 0
    rejected_hysteresis: int = 0
    rejected_low_range: int = 0

    @property
    def step_count(self) -> int:
        return len(self.event_frame_indices)


@dataclass
class GaitStepDetectionResult:
    """Bilateral gait step detection with cadence sanity checks."""

    left: SideStepDetection
    right: SideStepDetection
    fps: float
    duration_s: float
    valid_frames: int
    total_frames: int
    body_height: float
    confidence: ConfidenceLevel
    cadence_hz: float | None
    cadence_steps_per_min: float | None
    mean_step_interval_s: float | None
    min_step_interval_s: float | None
    max_step_interval_s: float | None
    notes: list[str] = field(default_factory=list)
    landmark_signals: dict[str, LandmarkSignal] = field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        return self.left.step_count + self.right.step_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_steps": self.left.step_count,
            "right_steps": self.right.step_count,
            "total_steps": self.total_steps,
            "left_events": list(self.left.event_frame_indices),
            "right_events": list(self.right.event_frame_indices),
            "left_landmark": self.left.landmark_used,
            "right_landmark": self.right.landmark_used,
            "fps": self.fps,
            "duration_s": round(self.duration_s, 3),
            "confidence": self.confidence,
            "cadence_hz": None if self.cadence_hz is None else round(self.cadence_hz, 3),
            "cadence_steps_per_min": self.cadence_steps_per_min,
            "mean_step_interval_s": self.mean_step_interval_s,
            "min_step_interval_s": self.min_step_interval_s,
            "max_step_interval_s": self.max_step_interval_s,
            "rejected": {
                "left_low_prominence": self.left.rejected_low_prominence,
                "left_min_interval": self.left.rejected_min_interval,
                "left_low_swing": self.left.rejected_low_swing,
                "left_hysteresis": self.left.rejected_hysteresis,
                "right_low_prominence": self.right.rejected_low_prominence,
                "right_min_interval": self.right.rejected_min_interval,
                "right_low_swing": self.right.rejected_low_swing,
                "right_hysteresis": self.right.rejected_hysteresis,
            },
            "notes": list(self.notes),
        }


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _cv(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return None
    mean = float(np.mean(arr))
    if abs(mean) < 1e-9:
        return None
    return float(np.std(arr) / mean)


def _interpolate_short_gaps(values: list[float | None], max_gap: int = INTERP_MAX_GAP) -> np.ndarray:
    arr = np.full(len(values), np.nan, dtype=float)
    for i, v in enumerate(values):
        if v is not None:
            arr[i] = float(v)
    n = len(arr)
    i = 0
    while i < n:
        if not np.isnan(arr[i]):
            i += 1
            continue
        j = i
        while j < n and np.isnan(arr[j]):
            j += 1
        gap = j - i
        if gap <= max_gap and i > 0 and j < n:
            left, right = arr[i - 1], arr[j]
            for k in range(gap):
                t = (k + 1) / (gap + 1)
                arr[i + k] = left + t * (right - left)
        i = max(j, i + 1)
    return arr


def _detrend_linear(arr: np.ndarray) -> np.ndarray:
    idx = np.arange(len(arr), dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return arr.copy()
    coef = np.polyfit(idx[valid], arr[valid], 1)
    return arr - np.polyval(coef, idx)


def _smooth_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    window = max(1, window)
    if window == 1:
        return arr.copy()
    out = np.full_like(arr, np.nan, dtype=float)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        chunk = arr[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        if valid.size:
            out[i] = float(np.mean(valid))
    return out


def _body_height(frames: list[PoseFrame]) -> float:
    heights: list[float] = []
    for f in frames:
        ys = [kp.y for kp in f.keypoints if kp.visibility >= VISIBILITY_MIN]
        if len(ys) >= 4:
            heights.append(max(ys) - min(ys))
    return float(np.median(heights)) if heights else 0.5


def _landmark_y_series(
    frames: list[PoseFrame],
    landmark_name: str,
) -> tuple[list[int], np.ndarray]:
    indices: list[int] = []
    ys: list[float | None] = []
    for frame in frames:
        indices.append(frame.frame_index)
        y_val: float | None = None
        for kp in frame.keypoints:
            if kp.name == landmark_name and kp.visibility >= VISIBILITY_MIN:
                y_val = float(kp.y)
                break
        ys.append(y_val)
    return indices, _interpolate_short_gaps(ys)


def export_foot_landmark_signals(sequence: PoseSequence) -> dict[str, LandmarkSignal]:
    """Export raw vertical signals for all six foot landmarks."""
    frames = [f for f in sequence.frames if f.detected]
    body_h = _body_height(frames)
    out: dict[str, LandmarkSignal] = {}
    for side in ("left", "right"):
        for suffix in FOOT_LANDMARK_SUFFIXES:
            name = f"{side}_{suffix}"
            frame_indices, raw = _landmark_y_series(frames, name)
            out[name] = LandmarkSignal(
                name=name,
                frame_indices=frame_indices,
                raw_y=raw.copy(),
                body_height=body_h,
            )
    return out


def _find_local_maxima(values: np.ndarray) -> list[int]:
    peaks: list[int] = []
    for i in range(1, len(values) - 1):
        if np.isnan(values[i - 1]) or np.isnan(values[i]) or np.isnan(values[i + 1]):
            continue
        if values[i] >= values[i - 1] and values[i] > values[i + 1]:
            peaks.append(i)
    return peaks


def detect_steps_in_signal(
    signal: list[float | None],
    fps: float,
    *,
    side: str = "left",
    body_scale: float = 1.0,
    landmark_used: str = "ankle",
    aligned_frame_indices: list[int] | None = None,
) -> SideStepDetection:
    """
    Detect foot-contact events from one vertical foot trajectory.

    Input landmark: caller supplies y(t) in image coordinates (down = positive).
    Heel strike ≈ local maximum of detrended, smoothed y.
    """
    scale = max(body_scale, 1e-6)
    arr = _interpolate_short_gaps(signal)
    n = len(arr)
    if aligned_frame_indices is None:
        aligned_frame_indices = list(range(n))

    empty = SideStepDetection(
        side=side,
        landmark_used=landmark_used,
        event_frame_indices=[],
        interval_cv=None,
        smoothed_y=arr.copy(),
        detrended_y=arr.copy(),
        raw_y=arr.copy(),
        aligned_frame_indices=aligned_frame_indices,
        rejected_low_range=1,
    )

    valid_count = int(np.sum(~np.isnan(arr)))
    if valid_count < MIN_VALID_FRAMES:
        return empty

    detrended = _detrend_linear(arr)
    window = max(3, int(round(fps * SMOOTH_WINDOW_S)))
    if window % 2 == 0:
        window += 1
    smoothed = _smooth_moving_average(detrended, window)
    valid = ~np.isnan(smoothed)
    if valid.sum() < MIN_VALID_FRAMES:
        return SideStepDetection(
            side=side,
            landmark_used=landmark_used,
            event_frame_indices=[],
            interval_cv=None,
            smoothed_y=smoothed,
            detrended_y=detrended,
            raw_y=arr,
            aligned_frame_indices=aligned_frame_indices,
            rejected_low_range=1,
        )

    rng = float(np.nanmax(smoothed) - np.nanmin(smoothed))
    min_range = MIN_RANGE_BODY_RATIO * scale
    if rng < min_range:
        return SideStepDetection(
            side=side,
            landmark_used=landmark_used,
            event_frame_indices=[],
            interval_cv=None,
            smoothed_y=smoothed,
            detrended_y=detrended,
            raw_y=arr,
            aligned_frame_indices=aligned_frame_indices,
            rejected_low_range=1,
        )

    min_distance = max(3, int(round(fps * MIN_STEP_INTERVAL_S)))
    min_prominence = max(0.004 * scale, MIN_PROMINENCE_RATIO * rng)
    min_swing = MIN_SWING_RATIO * rng
    fill = float(np.nanmean(smoothed[valid]))

    rejected_low_prominence = 0
    rejected_min_interval = 0
    rejected_low_swing = 0
    rejected_hysteresis = 0

    events: list[int] = []
    last_event = -min_distance
    armed_for_contact = True
    swing_depth = float(np.nanmax(smoothed[valid]))

    for i in _find_local_maxima(np.nan_to_num(smoothed, nan=fill)):
        if np.isnan(smoothed[i]):
            continue

        lo = max(0, i - min_distance)
        hi = min(n, i + min_distance + 1)
        local_min = float(np.min(smoothed[lo:hi]))
        prominence = float(smoothed[i] - local_min)
        if prominence < min_prominence:
            rejected_low_prominence += 1
            continue

        if events and (i - last_event) < min_distance:
            rejected_min_interval += 1
            continue

        if events:
            prev = events[-1]
            segment = smoothed[prev:i + 1]
            swing_amp = float(np.max(segment) - np.min(segment))
            if swing_amp < min_swing:
                rejected_low_swing += 1
                continue

        if not armed_for_contact:
            rejected_hysteresis += 1
            continue

        events.append(i)
        last_event = i
        armed_for_contact = False
        swing_depth = float(smoothed[i])

        # Hysteresis: foot must rise (lower y in detrended signal = smaller value)
        # before next contact. Track until signal drops min_swing below last peak.
        for j in range(i + 1, n):
            if np.isnan(smoothed[j]):
                continue
            if float(swing_depth - smoothed[j]) >= min_swing:
                armed_for_contact = True
                break

    event_frames = [aligned_frame_indices[i] for i in events]
    interval_cv = None
    if len(events) >= 2:
        intervals = [(events[i + 1] - events[i]) / fps for i in range(len(events) - 1)]
        interval_cv = _cv(intervals)

    return SideStepDetection(
        side=side,
        landmark_used=landmark_used,
        event_frame_indices=event_frames,
        interval_cv=interval_cv,
        smoothed_y=smoothed,
        detrended_y=detrended,
        raw_y=arr,
        aligned_frame_indices=aligned_frame_indices,
        rejected_low_prominence=rejected_low_prominence,
        rejected_min_interval=rejected_min_interval,
        rejected_low_swing=rejected_low_swing,
        rejected_hysteresis=rejected_hysteresis,
    )


def _best_landmark_for_side(
    frames: list[PoseFrame],
    side: str,
    fps: float,
    body_height: float,
) -> SideStepDetection:
    """Pick the foot landmark with the largest usable oscillation range."""
    best: SideStepDetection | None = None
    best_range = -1.0

    for suffix in FOOT_LANDMARK_SUFFIXES:
        name = f"{side}_{suffix}"
        frame_indices, raw = _landmark_y_series(frames, name)
        if len(raw) < MIN_VALID_FRAMES:
            continue
        det = detect_steps_in_signal(
            [None if np.isnan(v) else float(v) for v in raw],
            fps,
            side=side,
            body_scale=body_height,
            landmark_used=name,
            aligned_frame_indices=frame_indices,
        )
        valid = det.smoothed_y[~np.isnan(det.smoothed_y)]
        if valid.size < MIN_VALID_FRAMES:
            continue
        rng = float(np.max(valid) - np.min(valid))
        if rng > best_range:
            best_range = rng
            best = det

    if best is not None:
        return best

    return detect_steps_in_signal(
        [],
        fps,
        side=side,
        body_scale=body_height,
        landmark_used=f"{side}_ankle",
    )


def _filter_close_merged_events(
    left: SideStepDetection,
    right: SideStepDetection,
    fps: float,
    *,
    min_gap_s: float = MERGED_EVENT_MIN_S,
) -> tuple[list[int], list[int], int]:
    """
    Drop spurious footfalls when left/right events occur within ``min_gap_s``.

    Keeps the earlier event and rejects the later one (common jitter pattern).
    Returns (filtered_left_frames, filtered_right_frames, rejected_count).
    """
    min_gap_frames = max(3, int(math.ceil(fps * min_gap_s)))
    tagged = [(f, "left") for f in left.event_frame_indices] + [
        (f, "right") for f in right.event_frame_indices
    ]
    tagged.sort(key=lambda t: t[0])
    if not tagged:
        return [], [], 0

    kept: list[tuple[int, str]] = [tagged[0]]
    rejected = 0
    for frame_idx, side in tagged[1:]:
        if frame_idx - kept[-1][0] < min_gap_frames:
            rejected += 1
            continue
        kept.append((frame_idx, side))

    left_out = [f for f, s in kept if s == "left"]
    right_out = [f for f, s in kept if s == "right"]
    return left_out, right_out, rejected


def _with_filtered_events(
    side: SideStepDetection,
    events: list[int],
    fps: float,
) -> SideStepDetection:
    interval_cv = None
    if len(events) >= 2:
        intervals = [(events[i + 1] - events[i]) / fps for i in range(len(events) - 1)]
        interval_cv = _cv(intervals)
    return replace(side, event_frame_indices=events, interval_cv=interval_cv)


def detect_gait_steps(sequence: PoseSequence) -> GaitStepDetectionResult:
    """
    Detect bilateral foot-contact events with cadence sanity checks.

    Returns frame indices of heel-strike / foot-down events for each leg.
    """
    frames = [f for f in sequence.frames if f.detected]
    fps = max(sequence.fps, 1e-6)
    body_height = _body_height(frames)
    duration_s = len(sequence.frames) / fps

    left = _best_landmark_for_side(frames, "left", fps, body_height)
    right = _best_landmark_for_side(frames, "right", fps, body_height)

    filtered_l, filtered_r, merged_rejected = _filter_close_merged_events(
        left, right, fps,
    )
    if merged_rejected:
        left = _with_filtered_events(left, filtered_l, fps)
        right = _with_filtered_events(right, filtered_r, fps)

    notes: list[str] = []
    if merged_rejected:
        notes.append(
            f"Removed {merged_rejected} footfall(s) closer than "
            f"{MERGED_EVENT_MIN_S:.2f}s apart (likely landmark jitter)."
        )

    merged = sorted(left.event_frame_indices + right.event_frame_indices)
    cadence_hz = None
    cadence_spm = None
    mean_interval = None
    min_interval = None
    max_interval = None

    if len(merged) >= 2:
        intervals = [
            (merged[i + 1] - merged[i]) / fps for i in range(len(merged) - 1)
        ]
        span_s = (merged[-1] - merged[0]) / fps
        if span_s > 0:
            cadence_hz = len(merged) / span_s
            cadence_spm = cadence_hz * 60.0
        mean_interval = float(np.mean(intervals))
        min_interval = float(np.min(intervals))
        max_interval = float(np.max(intervals))

    confidence: ConfidenceLevel = "high"
    total = left.step_count + right.step_count

    if total < 2:
        confidence = "low"
        notes.append("Fewer than two foot contacts detected.")
    elif cadence_hz is not None and (
        cadence_hz < CADENCE_MIN_HZ or cadence_hz > CADENCE_MAX_HZ
    ):
        confidence = "low"
        notes.append(
            f"Cadence {cadence_hz:.2f} Hz outside plausible range "
            f"({CADENCE_MIN_HZ:.2f}–{CADENCE_MAX_HZ:.2f} Hz)."
        )
    if min_interval is not None and min_interval < MIN_STEP_INTERVAL_S * 0.85:
        confidence = "low"
        notes.append(
            f"Minimum step interval {min_interval:.2f}s is below "
            f"{MIN_STEP_INTERVAL_S:.2f}s — possible landmark noise."
        )

    return GaitStepDetectionResult(
        left=left,
        right=right,
        fps=fps,
        duration_s=duration_s,
        valid_frames=len(frames),
        total_frames=len(sequence.frames),
        body_height=body_height,
        confidence=confidence,
        cadence_hz=cadence_hz,
        cadence_steps_per_min=cadence_spm,
        mean_step_interval_s=mean_interval,
        min_step_interval_s=min_interval,
        max_step_interval_s=max_interval,
        notes=notes,
        landmark_signals=export_foot_landmark_signals(sequence),
    )


def format_step_detection_report(
    video: str,
    result: GaitStepDetectionResult,
) -> str:
    """Plain-text debug report for step detection."""
    rej = result.to_dict()["rejected"]
    noise_rejected = (
        rej["left_low_prominence"] + rej["right_low_prominence"]
        + rej["left_low_swing"] + rej["right_low_swing"]
        + rej["left_hysteresis"] + rej["right_hysteresis"]
    )
    lines = [
        "STEP DETECTION REPORT",
        "",
        f"Video: {video}",
        f"FPS: {result.fps:.1f}",
        f"Duration: {result.duration_s:.2f}s",
        f"Valid frames: {result.valid_frames}/{result.total_frames}",
        f"Body height scale: {result.body_height:.3f}",
        f"Confidence: {result.confidence.upper()}",
        "",
        f"Left landmark used: {result.left.landmark_used}",
        f"Left gait events ({result.left.step_count}): {result.left.event_frame_indices}",
        f"Right landmark used: {result.right.landmark_used}",
        f"Right gait events ({result.right.step_count}): {result.right.event_frame_indices}",
        "",
        f"Total detected steps: {result.total_steps}",
        f"Average step interval: "
        f"{result.mean_step_interval_s:.3f}s"
        if result.mean_step_interval_s is not None
        else "Average step interval: N/A",
        f"Minimum step interval: "
        f"{result.min_step_interval_s:.3f}s"
        if result.min_step_interval_s is not None
        else "Minimum step interval: N/A",
        f"Maximum step interval: "
        f"{result.max_step_interval_s:.3f}s"
        if result.max_step_interval_s is not None
        else "Maximum step interval: N/A",
        f"Cadence estimate: "
        f"{result.cadence_hz:.2f} Hz ({result.cadence_steps_per_min:.0f} spm)"
        if result.cadence_hz is not None
        else "Cadence estimate: N/A",
        "",
        f"Rejected noise / low-prominence events: {noise_rejected}",
        f"Rejected events due to minimum interval: "
        f"{rej['left_min_interval'] + rej['right_min_interval']}",
        f"Rejected low-prominence events: "
        f"{rej['left_low_prominence'] + rej['right_low_prominence']}",
        f"Rejected low-swing events: "
        f"{rej['left_low_swing'] + rej['right_low_swing']}",
        f"Rejected hysteresis (contact not re-armed): "
        f"{rej['left_hysteresis'] + rej['right_hysteresis']}",
    ]
    merged_rej = result.notes and any("Removed" in n for n in result.notes)
    if merged_rej:
        for n in result.notes:
            if n.startswith("Removed"):
                lines.append(f"Rejected merged close events: {n}")
    if result.notes:
        lines.extend(["", "Notes:"])
        lines.extend(f"  • {n}" for n in result.notes)
    return "\n".join(lines)


__all__ = [
    "GaitStepDetectionResult",
    "SideStepDetection",
    "LandmarkSignal",
    "detect_gait_steps",
    "detect_steps_in_signal",
    "export_foot_landmark_signals",
    "format_step_detection_report",
    "MIN_STEP_INTERVAL_S",
    "SMOOTH_WINDOW_S",
]
