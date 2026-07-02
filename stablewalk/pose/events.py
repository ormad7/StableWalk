"""
Gait cycle phase and event detection (heel strike, toe-off) from 2D ankle kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stablewalk.models.pose_data import PoseFrame

SIDES = ("left", "right")
ANKLE_BY_SIDE = {"left": "left_ankle", "right": "right_ankle"}
FOOT_BY_SIDE = {"left": "left_foot_index", "right": "right_foot_index"}


@dataclass
class GaitEvent:
    """Single gait event on one leg."""

    frame_index: int
    side: str
    event_type: str  # heel_strike | toe_off
    confidence: float = 1.0


@dataclass
class GaitCycleAnnotation:
    """Per-frame gait labels attached to a pose frame."""

    phase_left: str = "unknown"  # stance | swing | unknown
    phase_right: str = "unknown"
    events: list[str] = field(default_factory=list)  # e.g. left_heel_strike


def _ankle_y_series(
    frames: list[PoseFrame],
    side: str,
) -> tuple[list[int], np.ndarray]:
    """Frame indices and ankle y (image coords, down = positive)."""
    name = ANKLE_BY_SIDE[side]
    indices: list[int] = []
    ys: list[float] = []
    for frame in frames:
        if not frame.detected:
            continue
        by_name = {kp.name: kp for kp in frame.keypoints}
        ankle = by_name.get(name)
        if ankle is None or ankle.visibility < 0.4:
            continue
        indices.append(frame.frame_index)
        ys.append(float(ankle.y))
    return indices, np.array(ys, dtype=float)


def _find_extrema(y: np.ndarray, mode: str) -> list[int]:
    """Local maxima (mode='max') or minima (mode='min') indices."""
    if len(y) < 3:
        return []
    extrema: list[int] = []
    for i in range(1, len(y) - 1):
        if mode == "max" and y[i] > y[i - 1] and y[i] > y[i + 1]:
            extrema.append(i)
        elif mode == "min" and y[i] < y[i - 1] and y[i] < y[i + 1]:
            extrema.append(i)
    return extrema


def detect_side_events(
    frame_indices: list[int],
    y: np.ndarray,
    side: str,
) -> list[GaitEvent]:
    """
    Heel strike ≈ ankle lowest in image (local max y).
    Toe-off ≈ ankle highest in image during swing (local min y).
    """
    events: list[GaitEvent] = []
    if len(y) < 5:
        return events

    y_smooth = np.convolve(y, np.ones(3) / 3, mode="same")
    y_range = float(np.max(y_smooth) - np.min(y_smooth))
    if y_range < 0.02:
        return events

    for idx in _find_extrema(y_smooth, "max"):
        events.append(
            GaitEvent(
                frame_index=frame_indices[idx],
                side=side,
                event_type="heel_strike",
                confidence=min(1.0, y_range / 0.08),
            )
        )

    for idx in _find_extrema(y_smooth, "min"):
        events.append(
            GaitEvent(
                frame_index=frame_indices[idx],
                side=side,
                event_type="toe_off",
                confidence=min(1.0, y_range / 0.08),
            )
        )

    events.sort(key=lambda e: e.frame_index)
    return events


def assign_phases(
    frame_indices: list[int],
    y: np.ndarray,
    events: list[GaitEvent],
    side: str,
) -> dict[int, str]:
    """
    Stance between heel strike and toe-off; swing otherwise.
    Simplified: below median y → swing (foot up), above → stance (foot down).
    """
    phases: dict[int, str] = {}
    if len(y) == 0:
        return phases

    median_y = float(np.median(y))
    for i, fidx in enumerate(frame_indices):
        phases[fidx] = "stance" if y[i] >= median_y else "swing"

    # Refine using events when available
    hs_frames = [e.frame_index for e in events if e.side == side and e.event_type == "heel_strike"]
    to_frames = [e.frame_index for e in events if e.side == side and e.event_type == "toe_off"]

    for fidx in frame_indices:
        last_hs = max((h for h in hs_frames if h <= fidx), default=None)
        last_to = max((t for t in to_frames if t <= fidx), default=None)
        if last_hs is not None and (last_to is None or last_hs > last_to):
            phases[fidx] = "stance"
        elif last_to is not None and (last_hs is None or last_to > last_hs):
            phases[fidx] = "swing"

    return phases


def analyze_gait_sequence(frames: list[PoseFrame]) -> tuple[list[GaitEvent], dict[int, GaitCycleAnnotation]]:
    """
    Detect gait events and per-frame phase labels for a pose sequence.

    Returns:
        timeline of GaitEvent objects
        map frame_index → GaitCycleAnnotation
    """
    detected = [f for f in frames if f.detected]
    all_events: list[GaitEvent] = []
    annotations: dict[int, GaitCycleAnnotation] = {}

    phase_maps: dict[str, dict[int, str]] = {"left": {}, "right": {}}

    for side in SIDES:
        indices, y = _ankle_y_series(detected, side)
        side_events = detect_side_events(indices, y, side) if len(indices) else []
        all_events.extend(side_events)
        phase_maps[side] = assign_phases(indices, y, side_events, side)

    for frame in detected:
        ann = GaitCycleAnnotation()
        ann.phase_left = phase_maps["left"].get(frame.frame_index, "unknown")
        ann.phase_right = phase_maps["right"].get(frame.frame_index, "unknown")
        ann.events = [
            f"{e.side}_{e.event_type}"
            for e in all_events
            if e.frame_index == frame.frame_index
        ]
        annotations[frame.frame_index] = ann

    all_events.sort(key=lambda e: (e.frame_index, e.side, e.event_type))
    return all_events, annotations
