"""
Gait contact detection and gait-cycle temporal analysis.

Uses hip-centered 3D pose data and the shared ground-reference plane from
``ground_reference`` (vertical axis +Y by default — never assumes Z is up).

Contact combines heel/toe/ankle clearance, vertical foot velocities, floor
height, FPS-scaled temporal persistence, and entry/exit hysteresis.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from stablewalk.analysis.ground_reference import (
    GroundReferencePlane,
    displayed_foot_clearance_m,
    estimate_ground_plane,
    foot_clearance_m,
    vertical_coordinate,
)
from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot, Vec3
from stablewalk.models.pose_data import PoseSequence

logger = logging.getLogger(__name__)

from stablewalk.analysis.gait_phase_classification import (
    GaitPhaseName,
    classify_gait_phase_from_contacts,
    contact_to_display_state,
    format_gait_phase_display,
    log_phase_consistency_warning,
)
GaitEventType = Literal[
    "left_heel_strike",
    "right_heel_strike",
    "left_toe_off",
    "right_toe_off",
]
ConfidenceTier = Literal["HIGH", "MEDIUM", "LOW_CONFIDENCE"]

# Fallback velocity gates (m/s at body scale) when calibration has too few samples.
_DEFAULT_ENTRY_MAX_VEL_M_S = 0.28
_DEFAULT_EXIT_MIN_VEL_M_S = 0.42

MIN_FOOT_VISIBILITY = 0.35
MIN_FRAMES_FOR_ANALYSIS = 6
MIN_CYCLES_FOR_CADENCE = 1
MIN_STATE_HOLD_FRAMES = 2
MIN_CONTACT_RUN_FRAMES = 3

# Maximum displayed min(heel,toe) clearance while retaining CONTACT (fraction of leg length).
# Prevents ankle-driven contact hysteresis when heel/toe are clearly off the floor.
MAX_DISPLAY_EXIT_LEG_RATIO = 0.10

SIDE_JOINTS: dict[str, dict[str, tuple[str, ...]]] = {
    "left": {
        "heel": ("left_heel",),
        "toe": ("left_toe",),
        "ankle": ("left_ankle",),
        "all": ("left_heel", "left_toe", "left_ankle"),
    },
    "right": {
        "heel": ("right_heel",),
        "toe": ("right_toe",),
        "ankle": ("right_ankle",),
        "all": ("right_heel", "right_toe", "right_ankle"),
    },
}


@dataclass(frozen=True)
class FootLandmarkSample:
    """Per-side foot kinematics at one frame (body-scale meters, +Y vertical)."""

    heel_clearance_m: float | None
    toe_clearance_m: float | None
    ankle_clearance_m: float | None
    foot_clearance_m: float | None
    heel_velocity_m_s: float | None
    toe_velocity_m_s: float | None
    ankle_velocity_m_s: float | None
    visibility: float


@dataclass(frozen=True)
class FrameContactState:
    """Per-frame bilateral contact and gait phase."""

    frame_index: int
    time_s: float
    left_contact: int
    right_contact: int
    phase: GaitPhaseName
    left: FootLandmarkSample
    right: FootLandmarkSample


@dataclass(frozen=True)
class GaitEvent:
    event_type: GaitEventType
    frame_index: int
    time_s: float
    side: str


@dataclass(frozen=True)
class DetectedGaitCycle:
    cycle_index: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float


@dataclass
class GaitTemporalMetrics:
    left_stance_time_s: float | None = None
    right_stance_time_s: float | None = None
    left_swing_time_s: float | None = None
    right_swing_time_s: float | None = None
    average_stance_duration_s: float | None = None
    average_swing_duration_s: float | None = None
    left_right_stance_symmetry: float | None = None
    left_right_swing_symmetry: float | None = None
    double_support_time_s: float | None = None
    double_support_pct: float | None = None
    step_time_s: float | None = None
    stride_time_s: float | None = None
    cadence_steps_per_min: float | None = None
    gait_cycle_count: int = 0
    gait_cycle_consistency: float | None = None
    contact_confidence: float = 0.0
    confidence_tier: ConfidenceTier = "LOW_CONFIDENCE"
    left_heel_strike_count: int = 0
    right_heel_strike_count: int = 0
    left_toe_off_count: int = 0
    right_toe_off_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContactThresholds:
    """Data-driven contact gates (body-scale meters + leg-normalized ratios)."""

    entry_clearance_m: float
    exit_clearance_m: float
    entry_max_vel_m_s: float
    exit_min_vel_m_s: float
    leg_length_m: float
    entry_normalized: float
    exit_normalized: float
    clearance_p10_m: float
    clearance_p50_m: float
    clearance_p90_m: float
    max_display_exit_clearance_m: float
    max_display_exit_normalized: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class GaitCycleAnalysisResult:
    per_frame: list[FrameContactState] = field(default_factory=list)
    events: list[GaitEvent] = field(default_factory=list)
    cycles: list[DetectedGaitCycle] = field(default_factory=list)
    metrics: GaitTemporalMetrics = field(default_factory=GaitTemporalMetrics)
    ground_plane: GroundReferencePlane | None = None
    vertical_axis: str = "y"
    fps: float = 30.0
    contact_thresholds: ContactThresholds | None = None
    warnings: list[str] = field(default_factory=list)

    def frame_at(self, frame_index: int) -> FrameContactState | None:
        for state in self.per_frame:
            if state.frame_index == frame_index:
                return state
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "vertical_axis": self.vertical_axis,
            "ground_plane": None
            if self.ground_plane is None
            else {
                "floor_y": self.ground_plane.floor_y,
                "vertical_axis": self.ground_plane.vertical_axis,
                "scale_mode": self.ground_plane.scale_mode,
            },
            "metrics": self.metrics.to_dict(),
            "events": [asdict(e) for e in self.events],
            "cycles": [asdict(c) for c in self.cycles],
            "per_frame_count": len(self.per_frame),
            "contact_thresholds": (
                None if self.contact_thresholds is None else self.contact_thresholds.to_dict()
            ),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable)
# ---------------------------------------------------------------------------
def resolve_vertical_axis(plane: GroundReferencePlane | None) -> str:
    """Return the analysis vertical axis (+Y default; never assume Z)."""
    if plane is not None and plane.vertical_axis:
        return plane.vertical_axis
    return "y"


def landmark_clearance_m(
    position: Vec3 | None,
    plane: GroundReferencePlane,
    *,
    axis: str,
) -> float | None:
    if position is None:
        return None
    return foot_clearance_m(position, plane, axis=axis)


def landmark_height_m(
    position: Vec3 | None,
    *,
    axis: str,
) -> float | None:
    if position is None:
        return None
    return vertical_coordinate(position, axis=axis)


def smooth_series(values: list[float | None], window: int) -> list[float | None]:
    """NaN-aware moving average."""
    if window <= 1 or not values:
        return list(values)
    w = max(3, window | 1)
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


def vertical_velocities_m_s(
    heights: list[float | None],
    times_s: list[float],
) -> list[float | None]:
    """First derivative of vertical height (m/s) using actual timestamps."""
    n = len(heights)
    if n < 2:
        return [None] * n
    out: list[float | None] = [None] * n
    for i in range(1, n):
        if heights[i] is None or heights[i - 1] is None:
            continue
        dt = times_s[i] - times_s[i - 1]
        if dt <= 1e-9:
            continue
        out[i] = (heights[i] - heights[i - 1]) / dt
    if n >= 3 and out[1] is None and heights[2] is not None and heights[0] is not None:
        dt = times_s[2] - times_s[0]
        if dt > 1e-9:
            out[1] = (heights[2] - heights[0]) / dt
    return out


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def estimate_leg_length_m(
    recording: GaitMotionRecording,
    *,
    axis: str = "y",
) -> float:
    """Robust median hip-to-ankle distance as body scale reference."""
    lengths: list[float] = []
    for index in range(recording.frame_count):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        for hip_id, ankle_id in (
            ("left_hip", "left_ankle"),
            ("right_hip", "right_ankle"),
        ):
            hip = snap.joints.get(hip_id)
            ankle = snap.joints.get(ankle_id)
            if hip is None or ankle is None:
                continue
            hy = vertical_coordinate(hip.position, axis=axis)
            ay = vertical_coordinate(ankle.position, axis=axis)
            drop = abs(hy - ay)
            if 0.12 < drop < 0.62:
                lengths.append(drop)
    if lengths:
        return float(statistics.median(lengths))
    plane = estimate_ground_plane(recording, float(max(recording.frame_count - 1, 0)))
    span = plane.body_vertical_span if plane else 0.45
    return max(span, 0.28) * 0.48


def calibrate_contact_thresholds(
    left_samples: list[FootLandmarkSample],
    right_samples: list[FootLandmarkSample],
    *,
    leg_length_m: float,
) -> ContactThresholds:
    """
    Derive hysteresis thresholds from observed clearance distribution.

    Uses body-scale meters internally; normalized ratios = clearance / leg_length.
    """
    leg = max(leg_length_m, 0.20)
    clearances: list[float] = []
    for ls, rs in zip(left_samples, right_samples):
        for c in (ls.foot_clearance_m, rs.foot_clearance_m):
            if c is not None:
                clearances.append(c)

    vels: list[float] = []
    for sample in left_samples + right_samples:
        if sample.foot_clearance_m is None or sample.foot_clearance_m > leg * 0.12:
            continue
        for v in (
            sample.heel_velocity_m_s,
            sample.toe_velocity_m_s,
            sample.ankle_velocity_m_s,
        ):
            if v is not None and v > 0.0:
                vels.append(v)

    if len(clearances) >= 8:
        p10 = _percentile(clearances, 10.0)
        p50 = _percentile(clearances, 50.0)
        p90 = _percentile(clearances, 90.0)
        spread = max(p90 - p10, leg * 0.04, 0.012)
        entry = p10 + max(0.012, 0.15 * spread)
        exit_c = max(entry + leg * 0.05, p10 + 0.50 * spread)
        entry = min(entry, max(p50 * 0.55, p10 + 0.008))
        exit_c = max(exit_c, entry + leg * 0.04)
    else:
        p10, p50, p90 = leg * 0.02, leg * 0.08, leg * 0.18
        entry = leg * 0.055
        exit_c = leg * 0.14

    if vels:
        p60_v = _percentile(vels, 60.0)
        p85_v = _percentile(vels, 85.0)
        entry_vel = max(0.12, min(p60_v * 0.75, 0.45))
        exit_vel = max(entry_vel + 0.08, min(p85_v * 0.95, 0.65))
    else:
        entry_vel = _DEFAULT_ENTRY_MAX_VEL_M_S
        exit_vel = _DEFAULT_EXIT_MIN_VEL_M_S

    max_display_exit = max(exit_c, leg * MAX_DISPLAY_EXIT_LEG_RATIO)

    return ContactThresholds(
        entry_clearance_m=entry,
        exit_clearance_m=exit_c,
        entry_max_vel_m_s=entry_vel,
        exit_min_vel_m_s=exit_vel,
        leg_length_m=leg,
        entry_normalized=entry / leg,
        exit_normalized=exit_c / leg,
        clearance_p10_m=p10,
        clearance_p50_m=p50,
        clearance_p90_m=p90,
        max_display_exit_clearance_m=max_display_exit,
        max_display_exit_normalized=max_display_exit / leg,
    )


def raw_contact_desire(
    sample: FootLandmarkSample,
    *,
    currently_in_contact: bool,
    thresholds: ContactThresholds | None = None,
    entry_clearance_m: float | None = None,
    exit_clearance_m: float | None = None,
    entry_max_vel_m_s: float | None = None,
    exit_min_vel_m_s: float | None = None,
) -> bool | None:
    """
    Schmitt-trigger contact desire from clearance + vertical velocity.

    Returns None when foot data is insufficient.
    """
    if thresholds is not None:
        entry_clearance_m = thresholds.entry_clearance_m
        exit_clearance_m = thresholds.exit_clearance_m
        entry_max_vel_m_s = thresholds.entry_max_vel_m_s
        exit_min_vel_m_s = thresholds.exit_min_vel_m_s
    else:
        leg = 0.45
        entry_clearance_m = entry_clearance_m if entry_clearance_m is not None else leg * 0.055
        exit_clearance_m = exit_clearance_m if exit_clearance_m is not None else leg * 0.14
        entry_max_vel_m_s = entry_max_vel_m_s if entry_max_vel_m_s is not None else _DEFAULT_ENTRY_MAX_VEL_M_S
        exit_min_vel_m_s = exit_min_vel_m_s if exit_min_vel_m_s is not None else _DEFAULT_EXIT_MIN_VEL_M_S

    clearance = sample.foot_clearance_m
    if clearance is None:
        return None

    leg = thresholds.leg_length_m if thresholds is not None else 0.45
    max_display_exit = (
        thresholds.max_display_exit_clearance_m
        if thresholds is not None
        else max(exit_clearance_m or leg * 0.14, leg * MAX_DISPLAY_EXIT_LEG_RATIO)
    )
    displayed = displayed_foot_clearance_m(
        sample.heel_clearance_m,
        sample.toe_clearance_m,
    )

    vels = [
        v
        for v in (
            sample.heel_velocity_m_s,
            sample.toe_velocity_m_s,
            sample.ankle_velocity_m_s,
        )
        if v is not None
    ]
    max_up_vel = max((v for v in vels if v > 0.0), default=0.0)

    if currently_in_contact:
        if displayed is not None and displayed >= max_display_exit:
            return False
        if (
            displayed is not None
            and displayed >= exit_clearance_m
            and max_up_vel >= exit_min_vel_m_s
        ):
            return False
        if clearance >= exit_clearance_m:
            return False
        if max_up_vel >= exit_min_vel_m_s and clearance > entry_clearance_m:
            return False
        return True

    if clearance <= entry_clearance_m and max_up_vel <= entry_max_vel_m_s:
        return True
    if clearance <= entry_clearance_m * 0.65:
        return True
    return False


def apply_contact_state_machine(
    desires: list[bool | None],
    *,
    min_hold_frames: int = MIN_STATE_HOLD_FRAMES,
    min_run_frames: int = MIN_CONTACT_RUN_FRAMES,
) -> list[int]:
    """
    Temporal persistence + hysteresis debounce on contact desires.

    Prevents single-frame jitter from flipping contact state.
    """
    n = len(desires)
    if n == 0:
        return []

    state = 0
    hold = 0
    pending: int | None = None
    out: list[int] = []

    for desire in desires:
        target = state if desire is None else (1 if desire else 0)
        if desire is None:
            out.append(state)
            continue
        if target == state:
            pending = None
            hold = 0
            out.append(state)
            continue
        if pending != target:
            pending = target
            hold = 1
        else:
            hold += 1
        if hold >= min_hold_frames:
            state = target
            pending = None
            hold = 0
        out.append(state)

    return enforce_min_run_length(out, min_run_frames)


def enforce_displayed_clearance_contact_exit(
    contacts: list[int],
    samples: list[FootLandmarkSample],
    thresholds: ContactThresholds,
) -> list[int]:
    """
    Force SWING when displayed min(heel,toe) clearance exceeds the body-normalized exit cap.

    Overrides temporal hold so CONTACT cannot persist after clear lift-off.
    """
    if not contacts or thresholds is None:
        return list(contacts)
    max_exit = thresholds.max_display_exit_clearance_m
    out = list(contacts)
    for i, sample in enumerate(samples):
        if i >= len(out) or not out[i]:
            continue
        displayed = displayed_foot_clearance_m(
            sample.heel_clearance_m,
            sample.toe_clearance_m,
        )
        if displayed is not None and displayed >= max_exit:
            out[i] = 0
    return out


def enforce_min_run_length(states: list[int], min_len: int) -> list[int]:
    """Remove short contact blips and fill brief gaps (morphological closing)."""
    if min_len <= 1 or not states:
        return list(states)
    out = list(states)

    i = 0
    while i < len(out):
        val = out[i]
        j = i
        while j < len(out) and out[j] == val:
            j += 1
        run = j - i
        if run < min_len and i > 0 and j < len(out):
            for k in range(i, j):
                out[k] = out[i - 1]
        i = j
    return out


def classify_gait_phase(
    left_contact: int,
    right_contact: int,
    *,
    contact_confidence: float | None = None,
    confidence_tier: ConfidenceTier | None = None,
    left_foot_clearance_m: float | None = None,
    right_foot_clearance_m: float | None = None,
) -> GaitPhaseName:
    """Delegate to the central gait-phase classifier."""
    return classify_gait_phase_from_contacts(
        left_contact,
        right_contact,
        contact_confidence=contact_confidence,
        confidence_tier=confidence_tier,
        left_foot_clearance_m=left_foot_clearance_m,
        right_foot_clearance_m=right_foot_clearance_m,
    )


def finalize_per_frame_phases(
    per_frame: list[FrameContactState],
    *,
    contact_confidence: float,
    confidence_tier: ConfidenceTier,
) -> list[FrameContactState]:
    """Re-derive phase from final contact mask and session confidence."""
    updated: list[FrameContactState] = []
    for state in per_frame:
        phase = classify_gait_phase_from_contacts(
            state.left_contact,
            state.right_contact,
            contact_confidence=contact_confidence,
            confidence_tier=confidence_tier,
            left_foot_clearance_m=state.left.foot_clearance_m,
            right_foot_clearance_m=state.right.foot_clearance_m,
        )
        updated.append(
            FrameContactState(
                frame_index=state.frame_index,
                time_s=state.time_s,
                left_contact=state.left_contact,
                right_contact=state.right_contact,
                phase=phase,
                left=state.left,
                right=state.right,
            )
        )
    return updated


def detect_gait_events(
    per_frame: list[FrameContactState],
) -> list[GaitEvent]:
    """Heel strike = 0→1 contact; toe off = 1→0 contact."""
    events: list[GaitEvent] = []
    prev_l = 0
    prev_r = 0
    for state in per_frame:
        if state.left_contact and not prev_l:
            events.append(
                GaitEvent(
                    "left_heel_strike",
                    state.frame_index,
                    state.time_s,
                    "left",
                )
            )
        elif not state.left_contact and prev_l:
            events.append(
                GaitEvent("left_toe_off", state.frame_index, state.time_s, "left")
            )
        if state.right_contact and not prev_r:
            events.append(
                GaitEvent(
                    "right_heel_strike",
                    state.frame_index,
                    state.time_s,
                    "right",
                )
            )
        elif not state.right_contact and prev_r:
            events.append(
                GaitEvent("right_toe_off", state.frame_index, state.time_s, "right")
            )
        prev_l = state.left_contact
        prev_r = state.right_contact
    return events


def segment_durations_s(
    states: list[int],
    times_s: list[float],
    *,
    value: int,
) -> list[float]:
    """Contiguous run durations where state equals ``value``."""
    if not states:
        return []
    durations: list[float] = []
    run_start_idx = 0
    run_val = states[0]
    for i in range(1, len(states)):
        if states[i] != run_val:
            if run_val == value:
                durations.append(times_s[i] - times_s[run_start_idx])
            run_start_idx = i
            run_val = states[i]
    if run_val == value and len(times_s) > run_start_idx:
        durations.append(times_s[-1] - times_s[run_start_idx])
    return durations


def symmetry_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or max(a, b) <= 1e-9:
        return None
    return min(a, b) / max(a, b)


def interval_cv(intervals: list[float]) -> float | None:
    if len(intervals) < 2:
        return None
    mean = statistics.mean(intervals)
    if mean <= 1e-9:
        return None
    return statistics.pstdev(intervals) / mean


def detect_gait_cycles(
    events: list[GaitEvent],
    *,
    duration_s: float,
) -> list[DetectedGaitCycle]:
    """Cycles bounded by consecutive left heel strikes (fallback: any HS)."""
    left_hs = sorted(
        (e for e in events if e.event_type == "left_heel_strike"),
        key=lambda e: e.time_s,
    )
    if len(left_hs) >= 2:
        bounds = left_hs
    else:
        bounds = sorted(events, key=lambda e: e.time_s)
        bounds = [e for e in bounds if e.event_type.endswith("_heel_strike")]

    cycles: list[DetectedGaitCycle] = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        dur = end.time_s - start.time_s
        if dur <= 1e-6:
            continue
        cycles.append(
            DetectedGaitCycle(
                cycle_index=len(cycles),
                start_frame=start.frame_index,
                end_frame=end.frame_index,
                start_time_s=start.time_s,
                end_time_s=end.time_s,
                duration_s=dur,
            )
        )
    if not cycles and duration_s > 0 and bounds:
        cycles.append(
            DetectedGaitCycle(
                cycle_index=0,
                start_frame=bounds[0].frame_index,
                end_frame=bounds[-1].frame_index,
                start_time_s=bounds[0].time_s,
                end_time_s=bounds[-1].time_s,
                duration_s=max(duration_s, 1e-6),
            )
        )
    return cycles


def compute_temporal_metrics(
    per_frame: list[FrameContactState],
    events: list[GaitEvent],
    cycles: list[DetectedGaitCycle],
    *,
    fps: float,
    contact_confidence: float,
    confidence_tier: ConfidenceTier,
) -> GaitTemporalMetrics:
    if not per_frame:
        return GaitTemporalMetrics(
            contact_confidence=contact_confidence,
            confidence_tier=confidence_tier,
        )

    times = [s.time_s for s in per_frame]
    left = [s.left_contact for s in per_frame]
    right = [s.right_contact for s in per_frame]
    duration_s = max(times[-1] - times[0], len(per_frame) / max(fps, 1e-6))

    left_stance_runs = segment_durations_s(left, times, value=1)
    right_stance_runs = segment_durations_s(right, times, value=1)
    left_swing_runs = segment_durations_s(left, times, value=0)
    right_swing_runs = segment_durations_s(right, times, value=0)

    left_stance = statistics.mean(left_stance_runs) if left_stance_runs else None
    right_stance = statistics.mean(right_stance_runs) if right_stance_runs else None
    left_swing = statistics.mean(left_swing_runs) if left_swing_runs else None
    right_swing = statistics.mean(right_swing_runs) if right_swing_runs else None

    stance_vals = [v for v in (left_stance, right_stance) if v is not None]
    swing_vals = [v for v in (left_swing, right_swing) if v is not None]

    ds_frames = sum(1 for s in per_frame if s.left_contact and s.right_contact)
    ds_time = ds_frames / max(fps, 1e-6)
    cycle_durations = [c.duration_s for c in cycles if c.duration_s > 0]
    mean_cycle = statistics.mean(cycle_durations) if cycle_durations else duration_s
    ds_pct = (ds_time / mean_cycle * 100.0) if mean_cycle > 1e-6 else None

    hs_times = sorted(e.time_s for e in events if e.event_type.endswith("_heel_strike"))
    step_intervals = [hs_times[i + 1] - hs_times[i] for i in range(len(hs_times) - 1)]
    step_time = statistics.mean(step_intervals) if step_intervals else None

    left_hs_times = sorted(
        e.time_s for e in events if e.event_type == "left_heel_strike"
    )
    right_hs_times = sorted(
        e.time_s for e in events if e.event_type == "right_heel_strike"
    )
    left_stride = [left_hs_times[i + 1] - left_hs_times[i] for i in range(len(left_hs_times) - 1)]
    right_stride = [right_hs_times[i + 1] - right_hs_times[i] for i in range(len(right_hs_times) - 1)]
    stride_pool = left_stride + right_stride
    stride_time = statistics.mean(stride_pool) if stride_pool else None

    cadence = None
    if step_time and step_time > 1e-6:
        cadence = 60.0 / step_time
    elif hs_times and duration_s > 0:
        cadence = len(hs_times) / duration_s * 60.0

    cycle_cv = interval_cv(cycle_durations)

    return GaitTemporalMetrics(
        left_stance_time_s=left_stance,
        right_stance_time_s=right_stance,
        left_swing_time_s=left_swing,
        right_swing_time_s=right_swing,
        average_stance_duration_s=statistics.mean(stance_vals) if stance_vals else None,
        average_swing_duration_s=statistics.mean(swing_vals) if swing_vals else None,
        left_right_stance_symmetry=symmetry_ratio(left_stance, right_stance),
        left_right_swing_symmetry=symmetry_ratio(left_swing, right_swing),
        double_support_time_s=ds_time,
        double_support_pct=ds_pct,
        step_time_s=step_time,
        stride_time_s=stride_time,
        cadence_steps_per_min=cadence,
        gait_cycle_count=len(cycles),
        gait_cycle_consistency=(1.0 - min(cycle_cv, 1.0)) if cycle_cv is not None else None,
        contact_confidence=contact_confidence,
        confidence_tier=confidence_tier,
        left_heel_strike_count=len(left_hs_times),
        right_heel_strike_count=len(right_hs_times),
        left_toe_off_count=sum(1 for e in events if e.event_type == "left_toe_off"),
        right_toe_off_count=sum(1 for e in events if e.event_type == "right_toe_off"),
    )


def assess_contact_confidence(
    *,
    plane: GroundReferencePlane | None,
    foot_visibility_mean: float,
    valid_frame_ratio: float,
    heel_strike_count: int,
    duration_s: float,
    scale_mode: str,
) -> tuple[float, ConfidenceTier]:
    score = 0.0
    if plane is not None:
        score += 0.35
    score += 0.25 * min(1.0, foot_visibility_mean / 0.85)
    score += 0.20 * min(1.0, valid_frame_ratio)
    if duration_s >= 2.0:
        score += 0.10
    if heel_strike_count >= 2:
        score += 0.10
    elif heel_strike_count == 0:
        score *= 0.45
    if scale_mode == "unknown":
        score *= 0.55

    if score >= 0.72:
        tier: ConfidenceTier = "HIGH"
    elif score >= 0.48:
        tier = "MEDIUM"
    else:
        tier = "LOW_CONFIDENCE"
    return min(1.0, score), tier


# ---------------------------------------------------------------------------
# Recording-level analysis
# ---------------------------------------------------------------------------
def _joint_position(
    snapshot: SkeletonSnapshot,
    joint_id: str,
) -> tuple[Vec3 | None, float]:
    sample = snapshot.joints.get(joint_id)
    if sample is None:
        return None, 0.0
    vis_map = snapshot.metadata.get("landmark_visibility")
    if isinstance(vis_map, dict) and joint_id in vis_map:
        return sample.position, float(vis_map[joint_id])
    return sample.position, 1.0


def _side_visibility(snapshot: SkeletonSnapshot, side: str) -> float:
    joints = SIDE_JOINTS[side]["all"]
    vis = [_joint_position(snapshot, j)[1] for j in joints]
    vis = [v for v in vis if v > 0]
    return float(statistics.mean(vis)) if vis else 0.0


def _valid_foot_position(
    position: Vec3 | None,
    ankle_pos: Vec3 | None,
    *,
    axis: str,
) -> Vec3 | None:
    """Drop heel/toe landmarks that float above the ipsilateral ankle."""
    if position is None:
        return None
    if ankle_pos is None:
        return position
    if vertical_coordinate(position, axis=axis) > vertical_coordinate(ankle_pos, axis=axis) + 0.015:
        return None
    return position


def _lowest_clearance(
    snapshot: SkeletonSnapshot,
    plane: GroundReferencePlane,
    side: str,
    *,
    axis: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    ankle_pos, _ = _joint_position(snapshot, SIDE_JOINTS[side]["ankle"][0])
    heel_pos, _ = _joint_position(snapshot, SIDE_JOINTS[side]["heel"][0])
    toe_pos, _ = _joint_position(snapshot, SIDE_JOINTS[side]["toe"][0])
    heel_pos = _valid_foot_position(heel_pos, ankle_pos, axis=axis)
    toe_pos = _valid_foot_position(toe_pos, ankle_pos, axis=axis)

    heel_c = landmark_clearance_m(heel_pos, plane, axis=axis)
    toe_c = landmark_clearance_m(toe_pos, plane, axis=axis)
    ankle_c = landmark_clearance_m(ankle_pos, plane, axis=axis)
    parts = [c for c in (heel_c, toe_c, ankle_c) if c is not None]
    foot_c = min(parts) if parts else None
    return heel_c, toe_c, ankle_c, foot_c


def analyze_gait_cycles(
    recording: GaitMotionRecording,
) -> GaitCycleAnalysisResult:
    """
    Full gait contact + cycle analysis from a ``GaitMotionRecording``.

    Reuses ``estimate_ground_plane`` and foot clearance from ``ground_reference``.
    """
    fps = max(recording.fps, 1e-6)
    if recording.frame_count < MIN_FRAMES_FOR_ANALYSIS:
        logger.warning(
            "Gait cycle analysis skipped: only %d frames (need %d)",
            recording.frame_count,
            MIN_FRAMES_FOR_ANALYSIS,
        )
        return GaitCycleAnalysisResult(fps=fps)

    end_index = recording.frame_count - 1
    plane = estimate_ground_plane(recording, float(end_index))
    axis = resolve_vertical_axis(plane)

    if plane is None:
        logger.warning(
            "Gait cycle analysis: floor could not be estimated — LOW_CONFIDENCE"
        )
        conf, tier = assess_contact_confidence(
            plane=None,
            foot_visibility_mean=0.0,
            valid_frame_ratio=0.0,
            heel_strike_count=0,
            duration_s=recording.duration_s,
            scale_mode="unknown",
        )
        return GaitCycleAnalysisResult(
            fps=fps,
            vertical_axis=axis,
            metrics=GaitTemporalMetrics(
                contact_confidence=conf,
                confidence_tier=tier,
            ),
        )

    smooth_window = max(3, int(round(fps * 0.06)) | 1)

    frame_indices: list[int] = []
    times: list[float] = []
    left_samples: list[FootLandmarkSample] = []
    right_samples: list[FootLandmarkSample] = []
    vis_accum: list[float] = []

    left_heel_h: list[float | None] = []
    left_toe_h: list[float | None] = []
    left_ankle_h: list[float | None] = []
    right_heel_h: list[float | None] = []
    right_toe_h: list[float | None] = []
    right_ankle_h: list[float | None] = []

    for index in range(recording.frame_count):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        frame_indices.append(index)
        times.append(float(snap.time_s))
        vis_accum.append(
            (_side_visibility(snap, "left") + _side_visibility(snap, "right")) / 2.0
        )

        lh, lt, la, lf = _lowest_clearance(snap, plane, "left", axis=axis)
        rh, rt, ra, rf = _lowest_clearance(snap, plane, "right", axis=axis)
        left_samples.append(
            FootLandmarkSample(lh, lt, la, lf, None, None, None, _side_visibility(snap, "left"))
        )
        right_samples.append(
            FootLandmarkSample(rh, rt, ra, rf, None, None, None, _side_visibility(snap, "right"))
        )

        def _height(jid: str) -> float | None:
            pos, _ = _joint_position(snap, jid)
            return landmark_height_m(pos, axis=axis)

        left_heel_h.append(_height(SIDE_JOINTS["left"]["heel"][0]))
        left_toe_h.append(_height(SIDE_JOINTS["left"]["toe"][0]))
        left_ankle_h.append(_height(SIDE_JOINTS["left"]["ankle"][0]))
        right_heel_h.append(_height(SIDE_JOINTS["right"]["heel"][0]))
        right_toe_h.append(_height(SIDE_JOINTS["right"]["toe"][0]))
        right_ankle_h.append(_height(SIDE_JOINTS["right"]["ankle"][0]))

    def attach_velocities(
        samples: list[FootLandmarkSample],
        heel_h: list[float | None],
        toe_h: list[float | None],
        ankle_h: list[float | None],
    ) -> list[FootLandmarkSample]:
        heel_s = smooth_series(heel_h, smooth_window)
        toe_s = smooth_series(toe_h, smooth_window)
        ankle_s = smooth_series(ankle_h, smooth_window)
        v_heel = vertical_velocities_m_s(heel_s, times)
        v_toe = vertical_velocities_m_s(toe_s, times)
        v_ankle = vertical_velocities_m_s(ankle_s, times)
        out: list[FootLandmarkSample] = []
        for i, s in enumerate(samples):
            out.append(
                FootLandmarkSample(
                    heel_clearance_m=s.heel_clearance_m,
                    toe_clearance_m=s.toe_clearance_m,
                    ankle_clearance_m=s.ankle_clearance_m,
                    foot_clearance_m=s.foot_clearance_m,
                    heel_velocity_m_s=v_heel[i],
                    toe_velocity_m_s=v_toe[i],
                    ankle_velocity_m_s=v_ankle[i],
                    visibility=s.visibility,
                )
            )
        return out

    left_samples = attach_velocities(left_samples, left_heel_h, left_toe_h, left_ankle_h)
    right_samples = attach_velocities(right_samples, right_heel_h, right_toe_h, right_ankle_h)

    leg_length = estimate_leg_length_m(recording, axis=axis)
    thresholds = calibrate_contact_thresholds(
        left_samples, right_samples, leg_length_m=leg_length
    )
    logger.info(
        "Contact calibration — leg=%.3fm entry=%.3fm (%.2f×leg) exit=%.3fm (%.2f×leg) "
        "clearance p10/p50/p90=%.3f/%.3f/%.3fm",
        thresholds.leg_length_m,
        thresholds.entry_clearance_m,
        thresholds.entry_normalized,
        thresholds.exit_clearance_m,
        thresholds.exit_normalized,
        thresholds.clearance_p10_m,
        thresholds.clearance_p50_m,
        thresholds.clearance_p90_m,
    )

    min_hold = max(MIN_STATE_HOLD_FRAMES, int(round(fps * 0.05)))
    min_run = max(MIN_CONTACT_RUN_FRAMES, int(round(fps * 0.08)))

    left_desires: list[bool | None] = []
    left_state = 0
    for sample in left_samples:
        if sample.foot_clearance_m is None or sample.visibility < MIN_FOOT_VISIBILITY:
            left_desires.append(None)
            continue
        desire = raw_contact_desire(
            sample, currently_in_contact=bool(left_state), thresholds=thresholds
        )
        left_desires.append(desire)
        if desire is not None:
            left_state = 1 if desire else 0

    right_desires: list[bool | None] = []
    right_state = 0
    for sample in right_samples:
        if sample.foot_clearance_m is None or sample.visibility < MIN_FOOT_VISIBILITY:
            right_desires.append(None)
            continue
        desire = raw_contact_desire(
            sample, currently_in_contact=bool(right_state), thresholds=thresholds
        )
        right_desires.append(desire)
        if desire is not None:
            right_state = 1 if desire else 0

    left_contacts = apply_contact_state_machine(
        left_desires, min_hold_frames=min_hold, min_run_frames=min_run
    )
    right_contacts = apply_contact_state_machine(
        right_desires, min_hold_frames=min_hold, min_run_frames=min_run
    )
    left_contacts = enforce_displayed_clearance_contact_exit(
        left_contacts, left_samples, thresholds
    )
    right_contacts = enforce_displayed_clearance_contact_exit(
        right_contacts, right_samples, thresholds
    )

    per_frame: list[FrameContactState] = []
    for i, frame_index in enumerate(frame_indices):
        phase = classify_gait_phase(
            left_contacts[i],
            right_contacts[i],
            left_foot_clearance_m=left_samples[i].foot_clearance_m,
            right_foot_clearance_m=right_samples[i].foot_clearance_m,
        )
        per_frame.append(
            FrameContactState(
                frame_index=frame_index,
                time_s=times[i],
                left_contact=left_contacts[i],
                right_contact=right_contacts[i],
                phase=phase,
                left=left_samples[i],
                right=right_samples[i],
            )
        )

    events = detect_gait_events(per_frame)
    duration_s = recording.duration_s or (len(per_frame) / fps)
    cycles = detect_gait_cycles(events, duration_s=duration_s)

    foot_vis = float(statistics.mean(vis_accum)) if vis_accum else 0.0
    valid_ratio = len(per_frame) / max(recording.frame_count, 1)
    hs_count = sum(1 for e in events if e.event_type.endswith("_heel_strike"))
    conf, tier = assess_contact_confidence(
        plane=plane,
        foot_visibility_mean=foot_vis,
        valid_frame_ratio=valid_ratio,
        heel_strike_count=hs_count,
        duration_s=duration_s,
        scale_mode=plane.scale_mode,
    )

    metrics = compute_temporal_metrics(
        per_frame,
        events,
        cycles,
        fps=fps,
        contact_confidence=conf,
        confidence_tier=tier,
    )

    per_frame = finalize_per_frame_phases(
        per_frame,
        contact_confidence=conf,
        confidence_tier=tier,
    )

    warnings = validate_gait_cycle_analysis(per_frame, events, cycles, metrics, thresholds)
    for w in warnings:
        logger.warning("Gait cycle validation: %s", w)

    _log_analysis_summary(metrics)

    return GaitCycleAnalysisResult(
        per_frame=per_frame,
        events=events,
        cycles=cycles,
        metrics=metrics,
        ground_plane=plane,
        vertical_axis=axis,
        fps=fps,
        contact_thresholds=thresholds,
        warnings=warnings,
    )


def validate_gait_cycle_analysis(
    per_frame: list[FrameContactState],
    events: list[GaitEvent],
    cycles: list[DetectedGaitCycle],
    metrics: GaitTemporalMetrics,
    thresholds: ContactThresholds | None,
) -> list[str]:
    """Return human-readable warnings when contact/cycle detection likely failed."""
    warnings: list[str] = []
    n = len(per_frame)
    if n == 0:
        warnings.append("No frames analyzed — contact detection unavailable.")
        return warnings

    uncertain = sum(
        1 for s in per_frame if s.phase in ("FLIGHT_OR_UNCERTAIN", "UNCERTAIN", "FLIGHT")
    )
    uncertain_only = sum(1 for s in per_frame if s.phase in ("UNCERTAIN", "FLIGHT_OR_UNCERTAIN"))
    uncertain_frac = uncertain / n
    left_contact_frames = sum(s.left_contact for s in per_frame)
    right_contact_frames = sum(s.right_contact for s in per_frame)
    double_support = sum(1 for s in per_frame if s.left_contact and s.right_contact)
    hs_total = metrics.left_heel_strike_count + metrics.right_heel_strike_count

    uncertain_only_frac = uncertain_only / n
    if uncertain_frac > 0.80:
        warnings.append(
            f"{uncertain_frac:.0%} of frames are FLIGHT/UNCERTAIN (>80%) — "
            "floor reference or contact thresholds may be miscalibrated."
        )
    contact_frac = (left_contact_frames + right_contact_frames) / max(2 * n, 1)
    if contact_frac > 0.85 and uncertain_frac < 0.20:
        warnings.append(
            f"{contact_frac:.0%} of foot-frames are in contact (>85%) — "
            "floor may be set too high or swing clearance is underestimated."
        )
    if left_contact_frames == 0 and right_contact_frames == 0:
        warnings.append(
            "No foot contact detected on either side — check floor estimation and clearance units."
        )
    if hs_total == 0:
        warnings.append(
            "No heel strikes detected — cadence, stance symmetry, and cycle metrics unavailable."
        )
    if metrics.gait_cycle_count == 0 and hs_total > 0:
        warnings.append(
            "Heel strikes present but no complete gait cycles segmented."
        )
    if metrics.gait_cycle_count == 0 and hs_total == 0:
        warnings.append("No complete gait cycles detected.")

    if thresholds is not None and thresholds.clearance_p10_m > thresholds.entry_clearance_m * 1.5:
        warnings.append(
            f"Lowest clearance percentile ({thresholds.clearance_p10_m*100:.1f} cm) remains "
            f"above entry threshold ({thresholds.entry_clearance_m*100:.1f} cm) — "
            "possible floor/unit mismatch."
        )

    return warnings


def _log_analysis_summary(metrics: GaitTemporalMetrics) -> None:
    cadence = metrics.cadence_steps_per_min
    cadence_str = f"{cadence:.1f}" if cadence is not None else "N/A"
    logger.info(
        "Gait cycle analysis — Detected gait cycles: %d | "
        "Left heel strikes: %d | Right heel strikes: %d | "
        "Left stance mean: %s sec | Right stance mean: %s sec | "
        "Double support: %s %% | Cadence: %s steps/min | "
        "Contact confidence: %.2f (%s)",
        metrics.gait_cycle_count,
        metrics.left_heel_strike_count,
        metrics.right_heel_strike_count,
        _fmt_sec(metrics.left_stance_time_s),
        _fmt_sec(metrics.right_stance_time_s),
        _fmt_pct(metrics.double_support_pct),
        cadence_str,
        metrics.contact_confidence,
        metrics.confidence_tier,
    )


def _fmt_sec(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


def _fmt_pct(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "N/A"


def analyze_gait_cycles_from_pose_sequence(
    sequence: PoseSequence,
) -> GaitCycleAnalysisResult:
    """Convert enriched pose sequence to recording and run cycle analysis."""
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion

    recording = pose_sequence_to_gait_motion(sequence)
    result = analyze_gait_cycles(recording)
    recording.metadata["gait_cycle_analysis"] = result.to_dict()
    return result


def attach_gait_cycle_to_sequence(
    sequence: PoseSequence,
    result: GaitCycleAnalysisResult,
) -> None:
    """Write contact flags and per-side stance/swing onto pose frames."""
    by_index = {s.frame_index: s for s in result.per_frame}
    for frame in sequence.frames:
        state = by_index.get(frame.frame_index)
        if state is None:
            continue
        frame.foot_contact = {
            "left": bool(state.left_contact),
            "right": bool(state.right_contact),
        }
        frame.gait_phase = {
            "left": "stance" if state.left_contact else "swing",
            "right": "stance" if state.right_contact else "swing",
        }


__all__ = [
    "GaitCycleAnalysisResult",
    "GaitTemporalMetrics",
    "GaitEvent",
    "FrameContactState",
    "DetectedGaitCycle",
    "analyze_gait_cycles",
    "analyze_gait_cycles_from_pose_sequence",
    "attach_gait_cycle_to_sequence",
    "classify_gait_phase",
    "detect_gait_events",
    "apply_contact_state_machine",
    "raw_contact_desire",
    "compute_temporal_metrics",
    "assess_contact_confidence",
]
