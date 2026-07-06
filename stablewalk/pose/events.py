"""
Gait cycle phase and event detection (heel strike, toe-off) from 2D ankle kinematics.

Heel strikes use the shared robust detector in ``gait_step_detection`` so event
counts match the stability pipeline and reject landmark jitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stablewalk.models.pose_data import PoseFrame
from stablewalk.pose.gait_step_detection import (
    detect_gait_steps,
    detect_steps_in_signal,
)

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


def detect_side_events(
    frame_indices: list[int],
    y: np.ndarray,
    side: str,
    *,
    fps: float = 30.0,
    body_height: float = 0.5,
) -> list[GaitEvent]:
    """
    Heel strike ≈ foot contact (local max y after robust filtering).

    Toe-off events are estimated as mid-swing minima between consecutive heel
    strikes only (not every local minimum) to avoid jitter-driven over-counting.
    """
    events: list[GaitEvent] = []
    if len(y) < 5:
        return events

    signal = [float(v) if not np.isnan(v) else None for v in y]
    det = detect_steps_in_signal(
        signal,
        fps,
        side=side,
        body_scale=body_height,
        landmark_used=ANKLE_BY_SIDE[side],
        aligned_frame_indices=frame_indices,
    )

    hs_frames = det.event_frame_indices
    valid = det.smoothed_y[~np.isnan(det.smoothed_y)]
    y_range = float(np.max(valid) - np.min(valid)) if valid.size else 0.0
    conf = min(1.0, y_range / 0.08) if y_range > 0 else 0.5

    for fi in hs_frames:
        events.append(
            GaitEvent(
                frame_index=fi,
                side=side,
                event_type="heel_strike",
                confidence=conf,
            )
        )

    # One toe-off between each pair of consecutive heel strikes (lowest y in between).
    for i in range(len(hs_frames) - 1):
        start_f, end_f = hs_frames[i], hs_frames[i + 1]
        seg_idx = [
            j for j, fidx in enumerate(frame_indices)
            if start_f <= fidx <= end_f
        ]
        if len(seg_idx) < 3:
            continue
        seg_y = det.smoothed_y[seg_idx]
        if np.all(np.isnan(seg_y)):
            continue
        toe_local = int(np.nanargmin(seg_y))
        toe_frame = frame_indices[seg_idx[toe_local]]
        events.append(
            GaitEvent(
                frame_index=toe_frame,
                side=side,
                event_type="toe_off",
                confidence=conf * 0.9,
            )
        )

    events.sort(key=lambda e: (e.frame_index, e.side, e.event_type))
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


def analyze_gait_sequence(
    frames: list[PoseFrame],
    *,
    fps: float = 30.0,
) -> tuple[list[GaitEvent], dict[int, GaitCycleAnnotation]]:
    """
    Detect gait events and per-frame phase labels for a pose sequence.

    Returns:
        timeline of GaitEvent objects
        map frame_index → GaitCycleAnnotation
    """
    from stablewalk.models.pose_data import PoseSequence

    detected = [f for f in frames if f.detected]
    all_events: list[GaitEvent] = []
    annotations: dict[int, GaitCycleAnnotation] = {}

    # Use shared robust bilateral detector for consistent heel-strike timing.
    seq = PoseSequence(source_video="", frames=frames, fps=fps)
    gait = detect_gait_steps(seq)
    body_h = gait.body_height

    phase_maps: dict[str, dict[int, str]] = {"left": {}, "right": {}}

    for side in SIDES:
        side_det = gait.left if side == "left" else gait.right
        indices, y = _ankle_y_series(detected, side)
        side_events: list[GaitEvent] = []
        for fi in side_det.event_frame_indices:
            side_events.append(
                GaitEvent(
                    frame_index=fi,
                    side=side,
                    event_type="heel_strike",
                    confidence=0.9 if gait.confidence == "high" else 0.6,
                )
            )
        # Toe-off between consecutive heel strikes only.
        hs_sorted = sorted(side_det.event_frame_indices)
        for i in range(len(hs_sorted) - 1):
            start_f, end_f = hs_sorted[i], hs_sorted[i + 1]
            seg = [
                (j, float(y[j]))
                for j in range(len(indices))
                if start_f <= indices[j] <= end_f and j < len(y)
            ]
            if len(seg) < 3:
                continue
            toe_j, _ = min(seg, key=lambda t: t[1])
            side_events.append(
                GaitEvent(
                    frame_index=indices[toe_j],
                    side=side,
                    event_type="toe_off",
                    confidence=0.8 if gait.confidence == "high" else 0.5,
                )
            )
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
