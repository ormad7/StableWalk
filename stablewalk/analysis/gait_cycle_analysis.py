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
from stablewalk.analysis.timeline_duration import cycle_interval_pose_evidence
from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot, Vec3
from stablewalk.models.pose_data import PoseSequence

logger = logging.getLogger(__name__)

from stablewalk.analysis.gait_phase_classification import (
    FLIGHT_CLEARANCE_THRESHOLD_M,
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
EventSufficiency = Literal["none", "input_insufficient", "processing_failure"]

# Fallback velocity gates (m/s at body scale) when calibration has too few samples.
_DEFAULT_ENTRY_MAX_VEL_M_S = 0.28
_DEFAULT_EXIT_MIN_VEL_M_S = 0.42

MIN_FOOT_VISIBILITY = 0.35
MIN_FRAMES_FOR_ANALYSIS = 6
MIN_CYCLES_FOR_CADENCE = 2
MIN_STATE_HOLD_FRAMES = 2
MIN_CONTACT_RUN_FRAMES = 3
MIN_COMPLETE_CYCLE_DURATION_S = 0.35
MAX_COMPLETE_CYCLE_DURATION_S = 3.0
MIN_RELIABLE_CONTACT_CONFIDENCE = 0.48
# Per-cycle support integration may miss a few samples at 25 fps; do not
# require 98% coverage or cadence disappears on otherwise valid HS intervals.
CYCLE_SUPPORT_COVERAGE_MIN = 0.80

# Maximum displayed min(heel,toe) clearance while retaining CONTACT (fraction of leg length).
# Prevents ankle-driven contact hysteresis when heel/toe are clearly off the floor.
MAX_DISPLAY_EXIT_LEG_RATIO = 0.10
# Contact FSM saturated (walking aid / shuffle): do not invent HS/TO from
# clearance extrema — those peaks are not gait events.
SATURATED_CONTACT_FRAC = 0.85
# Kinematic toe-off: first leave-floor after heel strike, not mid-swing peak.
_KIN_TO_MIN_STANCE_S = 0.18
_KIN_TO_MAX_STANCE_S = 0.85
_KIN_TO_CLEARANCE_FRAC = 0.25

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
    confidence: float | None = None


@dataclass(frozen=True)
class DetectedGaitCycle:
    cycle_index: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    boundary_side: str = ""


@dataclass(frozen=True)
class CycleCandidate:
    """Binary-contact HS–HS interval. Not a valid cycle unless ``accepted``."""

    candidate_index: int
    side: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    n_frames: int
    n_known_contact: int
    n_known_stance: int
    n_known_swing: int
    known_contact_coverage: float
    accepted: bool
    acceptance_reason: str
    source: str = "binary_contact_hs_hs"
    n_valid_pose_frames: int = 0
    valid_pose_coverage: float | None = None
    longest_internal_gap_s: float | None = None

    def to_detected(self, cycle_index: int) -> DetectedGaitCycle:
        return DetectedGaitCycle(
            cycle_index=cycle_index,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            start_time_s=self.start_time_s,
            end_time_s=self.end_time_s,
            duration_s=self.duration_s,
            boundary_side=self.side,
        )


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
    single_support_time_s: float | None = None
    single_support_pct: float | None = None
    flight_time_s: float | None = None
    flight_pct: float | None = None
    left_stance_pct: float | None = None
    right_stance_pct: float | None = None
    left_swing_pct: float | None = None
    right_swing_pct: float | None = None
    step_time_s: float | None = None
    stride_time_s: float | None = None
    cadence_steps_per_min: float | None = None
    gait_cycle_count: int = 0
    gait_cycle_consistency: float | None = None
    contact_confidence: float = 0.0
    confidence_tier: ConfidenceTier = "LOW_CONFIDENCE"
    metrics_reliable: bool = False
    session_metrics_withheld: bool = False
    reliability_reason: str = "No complete gait cycles were analyzed."
    double_support_definition: str = (
        "Total time with both feet in contact, combining both double-support "
        "periods within one complete same-side heel-strike gait cycle."
    )
    single_support_definition: str = (
        "Ipsilateral single-limb support as a percentage of the gait cycle "
        "(mean of left-only and right-only support; typically ~40%)."
    )
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
    cycle_candidates: list[CycleCandidate] = field(default_factory=list)
    metrics: GaitTemporalMetrics = field(default_factory=GaitTemporalMetrics)
    ground_plane: GroundReferencePlane | None = None
    vertical_axis: str = "y"
    fps: float = 30.0
    contact_thresholds: ContactThresholds | None = None
    warnings: list[str] = field(default_factory=list)
    event_sufficiency: EventSufficiency = "none"
    event_sufficiency_reason: str = ""

    def frame_at(self, frame_index: int) -> FrameContactState | None:
        idx = getattr(self, "_by_frame", None)
        if idx is None:
            idx = {int(f.frame_index): f for f in self.per_frame}
            self._by_frame = idx
        return idx.get(int(frame_index))

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
            "cycle_candidates": [asdict(c) for c in self.cycle_candidates],
            "per_frame_count": len(self.per_frame),
            "contact_thresholds": (
                None if self.contact_thresholds is None else self.contact_thresholds.to_dict()
            ),
            "warnings": list(self.warnings),
            "event_sufficiency": self.event_sufficiency,
            "event_sufficiency_reason": self.event_sufficiency_reason,
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


def _clearance_and_velocity(
    sample: FootLandmarkSample,
) -> tuple[float | None, float | None]:
    """Prefer heel, then ankle, then generic foot clearance (confidence-aware)."""
    if sample.visibility < MIN_FOOT_VISIBILITY:
        return None, None
    clearance = sample.heel_clearance_m
    velocity = sample.heel_velocity_m_s
    if clearance is None:
        clearance = sample.ankle_clearance_m
        velocity = sample.ankle_velocity_m_s
    if clearance is None:
        clearance = sample.foot_clearance_m
        velocity = sample.ankle_velocity_m_s
    return clearance, velocity


def kinematic_supplement_allowed(contact_frac: float) -> bool:
    """False when feet never leave contact — kinematic extrema would be fabricated."""
    return math.isfinite(contact_frac) and contact_frac <= SATURATED_CONTACT_FRAC


def _kinematic_prominence_m(leg_length_m: float | None, view_type: str | None) -> float:
    """View-aware clearance prominence. Frontal image-Y is not true sagittal clearance."""
    leg = float(leg_length_m) if leg_length_m and leg_length_m > 0 else 0.45
    base = max(0.012, leg * 0.03)
    key = str(view_type or "").lower()
    if "frontal" in key or key == "unknown":
        return max(base, leg * 0.040)
    if "sagittal" in key:
        return max(0.010, leg * 0.022)
    return base


def detect_kinematic_foot_events(
    samples: list[FootLandmarkSample],
    times_s: list[float],
    frame_indices: list[int],
    *,
    side: str,
    min_interval_s: float = MIN_COMPLETE_CYCLE_DURATION_S,
    min_prominence_m: float = 0.012,
) -> list[GaitEvent]:
    """
    Heel-strike / toe-off from kinematics when the contact FSM has no transitions.

    Heel strike = prominent local heel-clearance minimum, confirmed by a
    descending→ascending vertical-velocity change when velocities are available.
    Toe-off = first leave-floor after that heel strike (clearance rising through
    a fraction of the swing range), **not** the mid-swing clearance peak.

    Returns no events when prominence or visibility is insufficient (walking-aid
    shuffle, occlusion). Does not invent a cycle from a single noisy extremum.
    """
    n = min(len(samples), len(times_s), len(frame_indices))
    if n < 7:
        return []
    clearance: list[float | None] = []
    velocity: list[float | None] = []
    for i in range(n):
        c, v = _clearance_and_velocity(samples[i])
        clearance.append(c)
        velocity.append(v)

    finite = [c for c in clearance if c is not None]
    if len(finite) < 7:
        return []
    spread = max(finite) - min(finite)
    if spread < min_prominence_m:
        return []

    half_win = 2
    if n >= 2 and times_s[-1] > times_s[0]:
        dt = (times_s[-1] - times_s[0]) / max(n - 1, 1)
        if dt > 1e-6:
            half_win = max(2, int(round(0.20 / dt)))

    hs: list[GaitEvent] = []
    last_hs_t = -1e9
    for i in range(half_win, n - half_win):
        c = clearance[i]
        if c is None:
            continue
        prev_c = clearance[i - 1]
        next_c = clearance[i + 1]
        if prev_c is None or next_c is None:
            continue
        if not (c <= prev_c and c <= next_c):
            continue
        neighborhood = [
            clearance[j]
            for j in range(i - half_win, i + half_win + 1)
            if clearance[j] is not None
        ]
        if len(neighborhood) < 5:
            continue
        prominence = max(neighborhood) - c
        t = times_s[i]
        if prominence < min_prominence_m or (t - last_hs_t) < min_interval_s:
            continue
        pv = velocity[i - 1] if i > 0 else None
        nv = velocity[i + 1] if i + 1 < n else None
        if (
            pv is not None
            and nv is not None
            and (abs(pv) + abs(nv)) > 1e-6
            and not (pv <= 1e-4 and nv >= -1e-4)
        ):
            continue
        hs.append(GaitEvent(f"{side}_heel_strike", frame_indices[i], t, side))
        last_hs_t = t

    events: list[GaitEvent] = list(hs)
    leave_thr = min(finite) + _KIN_TO_CLEARANCE_FRAC * spread
    last_to_t = -1e9
    for strike in hs:
        t0 = strike.time_s
        for i in range(n):
            t = times_s[i]
            if t < t0 + _KIN_TO_MIN_STANCE_S:
                continue
            if t > t0 + _KIN_TO_MAX_STANCE_S:
                break
            c = clearance[i]
            prev_c = clearance[i - 1] if i > 0 else None
            if c is None or prev_c is None:
                continue
            if c < leave_thr:
                continue
            if c < prev_c:
                continue
            if (t - last_to_t) < 0.20:
                continue
            events.append(GaitEvent(f"{side}_toe_off", frame_indices[i], t, side))
            last_to_t = t
            break
    events.sort(key=lambda e: e.time_s)
    return events


def annotate_event_confidence(
    events: list[GaitEvent],
    per_frame: list[FrameContactState],
) -> list[GaitEvent]:
    """Attach ipsilateral foot-landmark visibility at the event frame.

    Events have no independent detector posterior. Visibility is the
    evidence that the landmark used for HS/TO was seen, not a clinical
    certainty that the event is a true heel strike.
    """
    by_frame = {state.frame_index: state for state in per_frame}
    out: list[GaitEvent] = []
    for event in events:
        vis = event.confidence
        state = by_frame.get(event.frame_index)
        if vis is None and state is not None:
            sample = state.left if event.side == "left" else state.right
            raw = getattr(sample, "visibility", None)
            vis = float(raw) if raw is not None else None
        if vis == event.confidence:
            out.append(event)
        else:
            out.append(
                GaitEvent(
                    event.event_type,
                    event.frame_index,
                    event.time_s,
                    event.side,
                    vis,
                )
            )
    return out


def merge_gait_events(
    primary: list[GaitEvent],
    supplemental: list[GaitEvent],
    *,
    tol_s: float = 0.12,
) -> list[GaitEvent]:
    """Keep contact-FSM events; add kinematic events only when that type/side is absent nearby."""
    merged = list(primary)
    for extra in supplemental:
        if any(
            e.event_type == extra.event_type
            and e.side == extra.side
            and abs(e.time_s - extra.time_s) <= tol_s
            for e in merged
        ):
            continue
        same_side_type = [
            e for e in primary if e.event_type == extra.event_type and e.side == extra.side
        ]
        # If the contact FSM already found this event class, do not add extras
        # except when the FSM found none for that side/type.
        if same_side_type:
            continue
        merged.append(extra)
    merged.sort(key=lambda e: (e.time_s, e.event_type))
    return merged


def classify_event_sufficiency(
    *,
    frame_count: int,
    duration_s: float,
    ground_plane: GroundReferencePlane | None,
    heel_strike_count: int,
    toe_off_count: int,
    cycle_count: int,
    foot_visibility_mean: float,
    warnings: list[str],
) -> tuple[EventSufficiency, str]:
    """Separate a technical failure from scientifically insufficient input."""
    if frame_count < MIN_FRAMES_FOR_ANALYSIS:
        return (
            "processing_failure",
            f"Only {frame_count} pose frames (need {MIN_FRAMES_FOR_ANALYSIS}).",
        )
    if ground_plane is None:
        return ("processing_failure", "Ground plane could not be estimated.")
    if cycle_count >= MIN_CYCLES_FOR_CADENCE:
        return (
            "none",
            f"{cycle_count} complete same-side cycle(s) — no event-sufficiency "
            "failure. Measurement reliability is a separate flag.",
        )
    if duration_s < 2.0 * MIN_COMPLETE_CYCLE_DURATION_S:
        return (
            "input_insufficient",
            f"Clip is {duration_s:.1f}s — shorter than two minimum gait cycles "
            f"({2.0 * MIN_COMPLETE_CYCLE_DURATION_S:.1f}s). Not a processing crash.",
        )
    if foot_visibility_mean < MIN_FOOT_VISIBILITY:
        return (
            "input_insufficient",
            "Foot landmarks are too occluded for reliable heel-strike/toe-off detection.",
        )
    if heel_strike_count < 2:
        return (
            "input_insufficient",
            "Fewer than two heel-strike events with sufficient prominence. "
            "Short clips, walking aids, or shuffle gait often lack usable intervals. "
            "Cycles were not fabricated.",
        )
    if cycle_count == 0 and heel_strike_count >= 2:
        return (
            "input_insufficient",
            "Heel strikes were detected but no same-side pair formed a valid "
            f"{MIN_COMPLETE_CYCLE_DURATION_S:.2f}–{MAX_COMPLETE_CYCLE_DURATION_S:.1f}s cycle.",
        )
    if toe_off_count == 0 and cycle_count == 0:
        return (
            "input_insufficient",
            "No toe-off transitions and no complete gait cycles.",
        )
    del warnings
    return ("input_insufficient", "Usable gait cycles are below the session minimum.")


def _contact_partition_untrustworthy(warnings: list[str]) -> bool:
    """Withhold session cadence only when feet never leave contact (aid / shuffle).

    A stance+swing+flight sum ≠ 100% is an accounting warning, not missing
    heel-strikes. Do not suppress cadence for that alone.
    """
    text = " ".join(warnings).lower()
    return "foot-frames are in contact" in text


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
    if (
        a is None
        or b is None
        or not math.isfinite(a)
        or not math.isfinite(b)
        or a < 0.0
        or b < 0.0
        or max(a, b) <= 1e-9
    ):
        return None
    return min(a, b) / max(a, b)


def interval_cv(intervals: list[float]) -> float | None:
    if len(intervals) < 2:
        return None
    mean = statistics.mean(intervals)
    if mean <= 1e-9:
        return None
    return statistics.pstdev(intervals) / mean


def contact_bit_is_known(sample: FootLandmarkSample, binary: int) -> bool:
    """Known observation: contact-on, or confirmed flight. Near-ground off is not known."""
    if sample.visibility < MIN_FOOT_VISIBILITY:
        return False
    if binary:
        return True
    if sample.foot_clearance_m is None:
        return False
    return float(sample.foot_clearance_m) >= FLIGHT_CLEARANCE_THRESHOLD_M


def _side_contact_bits(state: FrameContactState, side: str) -> tuple[int, FootLandmarkSample]:
    if str(side).lower().startswith("right"):
        return int(state.right_contact), state.right
    return int(state.left_contact), state.left


def recording_pose_validity(
    recording: GaitMotionRecording,
) -> tuple[list[float], list[bool]]:
    """Detected, not skipped, with joints — the pose evidence for cycle gaps."""
    times: list[float] = []
    valid: list[bool] = []
    for snap in recording.snapshots:
        times.append(float(snap.time_s))
        meta = getattr(snap, "metadata", None) or {}
        detected = bool(meta.get("detected", True))
        skipped = bool(meta.get("skipped", False))
        valid.append(detected and not skipped and bool(getattr(snap, "joints", None)))
    return times, valid


def evaluate_cycle_candidate(
    start: GaitEvent,
    end: GaitEvent,
    *,
    candidate_index: int,
    per_frame: list[FrameContactState] | None,
    require_known_contact: bool,
    pose_times: list[float] | None = None,
    pose_valid: list[bool] | None = None,
    fps: float = 0.0,
) -> CycleCandidate:
    """Score one binary HS–HS interval. Does not retune contact thresholds."""
    side = str(start.side or "").lower() or "left"
    duration = float(end.time_s) - float(start.time_s)
    states = [
        state
        for state in (per_frame or [])
        if start.time_s <= state.time_s < end.time_s
    ]
    n_frames = len(states)
    n_known = n_stance = n_swing = 0
    for state in states:
        binary, sample = _side_contact_bits(state, side)
        known = contact_bit_is_known(sample, binary)
        if known:
            n_known += 1
            if binary:
                n_stance += 1
            else:
                n_swing += 1
    coverage = (n_known / n_frames) if n_frames else 0.0
    n_valid_pose = 0
    pose_coverage: float | None = None
    longest_gap: float | None = None
    pose_gap_reason = ""
    if pose_times is not None and pose_valid is not None:
        evidence = cycle_interval_pose_evidence(
            float(start.time_s),
            float(end.time_s),
            pose_times,
            pose_valid,
            fps=float(fps or 0.0),
        )
        n_valid_pose = evidence.n_valid_pose_frames
        pose_coverage = evidence.valid_pose_coverage
        longest_gap = evidence.longest_internal_gap_s
        if evidence.n_unsupported_gaps > 0:
            pose_gap_reason = (
                "unsupported_pose_gap "
                f"(longest internal gap {evidence.longest_internal_gap_s:.3f} s; "
                f"valid-pose coverage {evidence.valid_pose_coverage:.1%}; "
                "gaps are not interpolated)"
            )
    if duration < MIN_COMPLETE_CYCLE_DURATION_S:
        reason = (
            f"duration_below_min ({duration:.3f} s < {MIN_COMPLETE_CYCLE_DURATION_S:.2f} s)"
        )
        accepted = False
    elif duration > MAX_COMPLETE_CYCLE_DURATION_S:
        reason = (
            f"duration_above_max ({duration:.3f} s > {MAX_COMPLETE_CYCLE_DURATION_S:.2f} s)"
        )
        accepted = False
    elif pose_gap_reason:
        reason = pose_gap_reason
        accepted = False
    elif require_known_contact and n_frames == 0:
        reason = "no_frames_in_interval"
        accepted = False
    elif require_known_contact and n_stance < 1:
        reason = (
            "no_known_stance — binary HS–HS used contact bits that the validity "
            "layer marks Unknown"
        )
        accepted = False
    elif require_known_contact and n_swing < 1:
        reason = (
            "no_known_swing — off bits were near-ground Unknown, not confirmed flight"
        )
        accepted = False
    else:
        accepted = True
        pose_note = ""
        if pose_coverage is not None and longest_gap is not None:
            pose_note = (
                f"; valid-pose coverage {pose_coverage:.1%}; "
                f"longest internal gap {longest_gap:.3f} s"
            )
        if require_known_contact:
            reason = (
                f"accepted: duration {duration:.3f} s in "
                f"[{MIN_COMPLETE_CYCLE_DURATION_S:.2f}, {MAX_COMPLETE_CYCLE_DURATION_S:.2f}] s; "
                f"known stance {n_stance} and confirmed swing {n_swing} "
                f"(known-contact coverage {coverage:.1%})"
                f"{pose_note}"
            )
        else:
            reason = (
                f"duration_in_range ({duration:.3f} s); known-contact coverage not evaluated"
                f"{pose_note}"
            )
    return CycleCandidate(
        candidate_index=candidate_index,
        side=side,
        start_frame=int(start.frame_index),
        end_frame=int(end.frame_index),
        start_time_s=float(start.time_s),
        end_time_s=float(end.time_s),
        duration_s=duration,
        n_frames=n_frames,
        n_known_contact=n_known,
        n_known_stance=n_stance,
        n_known_swing=n_swing,
        known_contact_coverage=coverage,
        accepted=accepted,
        acceptance_reason=reason,
        n_valid_pose_frames=n_valid_pose,
        valid_pose_coverage=pose_coverage,
        longest_internal_gap_s=longest_gap,
    )


def enumerate_cycle_candidates(
    events: list[GaitEvent],
    *,
    per_frame: list[FrameContactState] | None = None,
    require_known_contact: bool | None = None,
    pose_times: list[float] | None = None,
    pose_valid: list[bool] | None = None,
    fps: float = 0.0,
) -> list[CycleCandidate]:
    """Every consecutive same-side HS–HS interval, accepted or rejected."""
    require = bool(per_frame) if require_known_contact is None else bool(require_known_contact)
    out: list[CycleCandidate] = []
    for side in ("left", "right"):
        bounds = sorted(
            (e for e in events if e.event_type == f"{side}_heel_strike"),
            key=lambda e: e.time_s,
        )
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            if end.time_s <= start.time_s:
                continue
            out.append(
                evaluate_cycle_candidate(
                    start,
                    end,
                    candidate_index=len(out),
                    per_frame=per_frame,
                    require_known_contact=require,
                    pose_times=pose_times,
                    pose_valid=pose_valid,
                    fps=fps,
                )
            )
    return out


def detect_gait_cycles(
    events: list[GaitEvent],
    *,
    duration_s: float,
    per_frame: list[FrameContactState] | None = None,
    cycle_candidates: list[CycleCandidate] | None = None,
    pose_times: list[float] | None = None,
    pose_valid: list[bool] | None = None,
    fps: float = 0.0,
) -> list[DetectedGaitCycle]:
    """Return every accepted ipsilateral HS–HS cycle from both sides.

    Alternating heel strikes delimit steps, not gait cycles. A leading or
    trailing partial interval is therefore never promoted to a complete cycle.
    When ``per_frame`` is supplied, binary-contact intervals that lack known
    stance and confirmed swing are candidates only — not valid cycles.
    Intervals that span an unsupported pose gap are rejected explicitly.
    Accepted left and right cycles are both published; the non-winning side
    is not dropped.
    """
    del duration_s  # Retained for API compatibility; partial clip duration is not a cycle.
    candidates = cycle_candidates
    if candidates is None:
        candidates = enumerate_cycle_candidates(
            events,
            per_frame=per_frame,
            pose_times=pose_times,
            pose_valid=pose_valid,
            fps=fps,
        )
    accepted = [item for item in candidates if item.accepted]
    accepted.sort(
        key=lambda item: (
            item.start_time_s,
            0 if str(item.side).lower().startswith("left") else 1,
            item.candidate_index,
        )
    )
    return [item.to_detected(index) for index, item in enumerate(accepted)]


def _cycle_support_durations(
    per_frame: list[FrameContactState],
    cycle: DetectedGaitCycle,
    *,
    fps: float,
) -> dict[str, float] | None:
    """Integrate binary support states over one complete gait cycle."""
    states = [
        state
        for state in per_frame
        if cycle.start_time_s <= state.time_s < cycle.end_time_s
    ]
    if len(states) < 3 or cycle.duration_s <= 1e-9:
        return None

    totals = {
        "left_stance": 0.0,
        "right_stance": 0.0,
        "double_support": 0.0,
        "left_single_support": 0.0,
        "right_single_support": 0.0,
        "single_support": 0.0,
        "flight": 0.0,
    }
    fallback_dt = 1.0 / max(fps, 1e-6)
    for i, state in enumerate(states):
        next_t = states[i + 1].time_s if i + 1 < len(states) else cycle.end_time_s
        dt = min(max(next_t - state.time_s, 0.0), fallback_dt * 2.5)
        if state.left_contact:
            totals["left_stance"] += dt
        if state.right_contact:
            totals["right_stance"] += dt
        if state.left_contact and state.right_contact:
            totals["double_support"] += dt
        elif state.left_contact:
            totals["left_single_support"] += dt
            totals["single_support"] += dt
        elif state.right_contact:
            totals["right_single_support"] += dt
            totals["single_support"] += dt
        else:
            totals["flight"] += dt

    covered = (
        totals["double_support"] + totals["single_support"] + totals["flight"]
    )
    if covered < cycle.duration_s * CYCLE_SUPPORT_COVERAGE_MIN:
        return None
    return totals


def compute_temporal_metrics(
    per_frame: list[FrameContactState],
    events: list[GaitEvent],
    cycles: list[DetectedGaitCycle],
    *,
    fps: float,
    contact_confidence: float,
    confidence_tier: ConfidenceTier,
) -> GaitTemporalMetrics:
    left_hs_times = sorted(
        e.time_s for e in events if e.event_type == "left_heel_strike"
    )
    right_hs_times = sorted(
        e.time_s for e in events if e.event_type == "right_heel_strike"
    )
    event_counts = {
        "left_heel_strike_count": len(left_hs_times),
        "right_heel_strike_count": len(right_hs_times),
        "left_toe_off_count": sum(
            1 for e in events if e.event_type == "left_toe_off"
        ),
        "right_toe_off_count": sum(
            1 for e in events if e.event_type == "right_toe_off"
        ),
    }
    if not per_frame:
        return GaitTemporalMetrics(
            contact_confidence=contact_confidence,
            confidence_tier=confidence_tier,
            reliability_reason="No contact-state frames were available.",
            **event_counts,
        )

    cycle_samples = [
        (cycle, sample)
        for cycle in cycles
        if (sample := _cycle_support_durations(per_frame, cycle, fps=fps)) is not None
    ]
    usable_cycles = [cycle for cycle, _ in cycle_samples]
    support_samples = [sample for _, sample in cycle_samples]
    # Cadence uses detected same-side HS→HS cycles. Support-time integration is
    # only required for stance/swing symmetry — do not hide cadence when the
    # partition is slightly incomplete.
    cadence_cycles = list(cycles) if cycles else []
    n_left_cycles = sum(
        1
        for cycle in cadence_cycles
        if str(getattr(cycle, "boundary_side", "")).lower().startswith("left")
    )
    n_right_cycles = sum(
        1
        for cycle in cadence_cycles
        if str(getattr(cycle, "boundary_side", "")).lower().startswith("right")
    )
    # Cadence still needs ≥2 ipsilateral HS→HS intervals on at least one side.
    # Mixed 1 L + 1 R is two steps, not two strides, so it does not unlock cadence.
    can_compute_cadence = max(n_left_cycles, n_right_cycles) >= MIN_CYCLES_FOR_CADENCE
    can_compute_symmetry = len(support_samples) >= MIN_CYCLES_FOR_CADENCE
    contact_ok = (
        contact_confidence >= MIN_RELIABLE_CONTACT_CONFIDENCE
        and confidence_tier != "LOW_CONFIDENCE"
    )
    if not can_compute_cadence:
        reason = (
            f"Only {n_left_cycles} left / {n_right_cycles} right complete "
            f"same-side heel-strike cycle(s); at least {MIN_CYCLES_FOR_CADENCE} "
            "ipsilateral cycles on one side are required for cadence."
        )
    elif not contact_ok:
        reason = (
            f"Cadence computed from {n_left_cycles} left / {n_right_cycles} right "
            f"HS→HS cycle(s) at reduced reliability (contact "
            f"{contact_confidence:.2f}, {confidence_tier}). "
            f"MEDIUM and >={MIN_RELIABLE_CONTACT_CONFIDENCE:.2f} are required "
            "before the value is marked scientifically reliable."
        )
    else:
        reason = (
            f"Computed from {n_left_cycles} left / {n_right_cycles} right "
            "complete same-side heel-strike cycles."
        )
        if not can_compute_symmetry:
            reason += (
                " Stance/swing symmetry withheld — cycle support coverage "
                f"below {CYCLE_SUPPORT_COVERAGE_MIN:.0%}."
            )

    if not can_compute_cadence:
        return GaitTemporalMetrics(
            gait_cycle_count=len(cadence_cycles),
            contact_confidence=contact_confidence,
            confidence_tier=confidence_tier,
            metrics_reliable=False,
            reliability_reason=reason,
            **event_counts,
        )

    cycle_durations = [cycle.duration_s for cycle in cadence_cycles]
    if not can_compute_symmetry or not contact_ok:
        complete_hs = sorted(
            {
                (event.time_s, event.side)
                for event in events
                if event.event_type.endswith("_heel_strike")
                and any(
                    cycle.start_time_s <= event.time_s <= cycle.end_time_s
                    for cycle in cadence_cycles
                )
            }
        )
        step_intervals = [
            complete_hs[i + 1][0] - complete_hs[i][0]
            for i in range(len(complete_hs) - 1)
            if complete_hs[i][1] != complete_hs[i + 1][1]
            and complete_hs[i + 1][0] > complete_hs[i][0]
        ]
        step_time = statistics.mean(step_intervals) if step_intervals else None
        stride_time = statistics.mean(cycle_durations)
        cadence = (
            60.0 / step_time if step_time and step_time > 1e-6 else 120.0 / stride_time
        )
        cycle_cv = interval_cv(cycle_durations)
        return GaitTemporalMetrics(
            step_time_s=step_time,
            stride_time_s=stride_time,
            cadence_steps_per_min=cadence,
            gait_cycle_count=len(cadence_cycles),
            gait_cycle_consistency=(1.0 - min(cycle_cv, 1.0)) if cycle_cv is not None else None,
            contact_confidence=contact_confidence,
            confidence_tier=confidence_tier,
            metrics_reliable=contact_ok,
            reliability_reason=reason,
            **event_counts,
        )

    left_stance_values = [s["left_stance"] for s in support_samples]
    right_stance_values = [s["right_stance"] for s in support_samples]
    left_swing_values = [
        max(cycle.duration_s - sample["left_stance"], 0.0)
        for cycle, sample in cycle_samples
    ]
    right_swing_values = [
        max(cycle.duration_s - sample["right_stance"], 0.0)
        for cycle, sample in cycle_samples
    ]
    ds_values = [s["double_support"] for s in support_samples]
    left_ss_values = [s["left_single_support"] for s in support_samples]
    right_ss_values = [s["right_single_support"] for s in support_samples]
    # Keep total exclusive SS for flight accounting; report ipsilateral mean clinically.
    ss_values = [
        0.5 * (ls + rs) for ls, rs in zip(left_ss_values, right_ss_values)
    ]
    flight_values = [s["flight"] for s in support_samples]

    def _mean_pct(values: list[float]) -> float:
        return statistics.mean(
            value / cycle.duration_s * 100.0
            for cycle, value in zip(usable_cycles, values)
        )

    complete_hs = sorted(
        {
            (event.time_s, event.side)
            for event in events
            if event.event_type.endswith("_heel_strike")
            and any(
                cycle.start_time_s <= event.time_s <= cycle.end_time_s
                for cycle in usable_cycles
            )
        }
    )
    step_intervals = [
        complete_hs[i + 1][0] - complete_hs[i][0]
        for i in range(len(complete_hs) - 1)
        if complete_hs[i][1] != complete_hs[i + 1][1]
        and complete_hs[i + 1][0] > complete_hs[i][0]
    ]
    step_time = statistics.mean(step_intervals) if step_intervals else None
    stride_time = statistics.mean(cycle_durations)
    cadence = 60.0 / step_time if step_time and step_time > 1e-6 else 120.0 / stride_time
    cycle_cv = interval_cv(cycle_durations)

    return GaitTemporalMetrics(
        left_stance_time_s=statistics.mean(left_stance_values),
        right_stance_time_s=statistics.mean(right_stance_values),
        left_swing_time_s=statistics.mean(left_swing_values),
        right_swing_time_s=statistics.mean(right_swing_values),
        average_stance_duration_s=statistics.mean(left_stance_values + right_stance_values),
        average_swing_duration_s=statistics.mean(left_swing_values + right_swing_values),
        left_right_stance_symmetry=symmetry_ratio(
            statistics.mean(left_stance_values), statistics.mean(right_stance_values)
        ),
        left_right_swing_symmetry=symmetry_ratio(
            statistics.mean(left_swing_values), statistics.mean(right_swing_values)
        ),
        double_support_time_s=statistics.mean(ds_values),
        double_support_pct=_mean_pct(ds_values),
        single_support_time_s=statistics.mean(ss_values),
        single_support_pct=_mean_pct(ss_values),
        flight_time_s=statistics.mean(flight_values),
        flight_pct=_mean_pct(flight_values),
        left_stance_pct=_mean_pct(left_stance_values),
        right_stance_pct=_mean_pct(right_stance_values),
        left_swing_pct=_mean_pct(left_swing_values),
        right_swing_pct=_mean_pct(right_swing_values),
        step_time_s=step_time,
        stride_time_s=stride_time,
        cadence_steps_per_min=cadence,
        gait_cycle_count=len(cadence_cycles),
        gait_cycle_consistency=(1.0 - min(cycle_cv, 1.0)) if cycle_cv is not None else None,
        contact_confidence=contact_confidence,
        confidence_tier=confidence_tier,
        metrics_reliable=contact_ok,
        reliability_reason=reason,
        **event_counts,
    )


def cadence_is_exploratory(metrics: Any | None, cycles: Any | None = None) -> bool:
    """True when cadence must not be shown as a session summary number.

    Low-confidence contact and unilateral (single-channel) HS–HS sets can still
    produce a numeric cadence. That value is exploratory, not a bilateral
    session result. Thresholds are not relaxed here.
    """
    if metrics is None:
        return False
    value = getattr(metrics, "cadence_steps_per_min", None)
    if value is None:
        return False
    if getattr(metrics, "metrics_reliable", None) is False:
        return True
    tier = str(getattr(metrics, "confidence_tier", "") or "").upper()
    if tier == "LOW_CONFIDENCE":
        return True
    clist = list(getattr(cycles, "cycles", None) or [])
    if not clist:
        return False
    n_left = sum(
        1
        for cycle in clist
        if str(getattr(cycle, "boundary_side", "")).lower().startswith("left")
    )
    n_right = sum(
        1
        for cycle in clist
        if str(getattr(cycle, "boundary_side", "")).lower().startswith("right")
    )
    return (n_left == 0) ^ (n_right == 0)


def summary_cadence_spm(metrics: Any | None, cycles: Any | None = None) -> float | None:
    """Cadence for Overview / Results / exports, or None when exploratory."""
    if cadence_is_exploratory(metrics, cycles):
        return None
    if metrics is None:
        return None
    value = getattr(metrics, "cadence_steps_per_min", None)
    if value is None:
        value = getattr(metrics, "cadence_spm", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    *,
    sequence: PoseSequence | None = None,
    view_estimate: Any | None = None,
) -> GaitCycleAnalysisResult:
    """
    Full gait contact + cycle analysis from a ``GaitMotionRecording``.

    Reuses ``estimate_ground_plane`` and foot clearance from ``ground_reference``.
    Optional ``sequence`` / ``view_estimate`` only scale kinematic-supplement
    prominence (frontal views are more conservative). They do not invent events.
    """
    if view_estimate is None and sequence is not None:
        try:
            from stablewalk.analysis.gait_view_analysis import estimate_gait_view

            frames = getattr(sequence, "frames", None) or []
            if frames:
                view_estimate = estimate_gait_view(frames)
        except Exception:
            view_estimate = None
    fps = max(recording.fps, 1e-6)
    if recording.frame_count < MIN_FRAMES_FOR_ANALYSIS:
        logger.warning(
            "Gait cycle analysis skipped: only %d frames (need %d)",
            recording.frame_count,
            MIN_FRAMES_FOR_ANALYSIS,
        )
        return GaitCycleAnalysisResult(
            fps=fps,
            warnings=[
                f"Final gait metrics unavailable: only {recording.frame_count} frames "
                f"were available; at least {MIN_FRAMES_FOR_ANALYSIS} are required."
            ],
            event_sufficiency="processing_failure",
            event_sufficiency_reason=(
                f"Only {recording.frame_count} pose frames (need {MIN_FRAMES_FOR_ANALYSIS})."
            ),
        )

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
                reliability_reason="Ground plane could not be estimated.",
            ),
            warnings=[
                "Final gait metrics unavailable: ground plane could not be estimated."
            ],
            event_sufficiency="processing_failure",
            event_sufficiency_reason="Ground plane could not be estimated.",
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
    hs_contact = sum(1 for e in events if e.event_type.endswith("_heel_strike"))
    to_contact = sum(1 for e in events if e.event_type.endswith("_toe_off"))
    n_pf = max(len(per_frame), 1)
    contact_frac = (
        sum(s.left_contact for s in per_frame) + sum(s.right_contact for s in per_frame)
    ) / float(2 * n_pf)
    if (hs_contact < 2 or to_contact == 0) and kinematic_supplement_allowed(contact_frac):
        kin: list[GaitEvent] = []
        view_key = None
        if view_estimate is not None:
            view_key = str(
                getattr(getattr(view_estimate, "view_type", None), "value", None)
                or getattr(view_estimate, "view_type", None)
                or view_estimate
            )
        prominence = _kinematic_prominence_m(leg_length, view_key)
        kin.extend(
            detect_kinematic_foot_events(
                left_samples, times, frame_indices, side="left",
                min_prominence_m=prominence,
            )
        )
        kin.extend(
            detect_kinematic_foot_events(
                right_samples, times, frame_indices, side="right",
                min_prominence_m=prominence,
            )
        )
        events = merge_gait_events(events, kin)
    events = annotate_event_confidence(events, per_frame)
    duration_s = recording.duration_s or (len(per_frame) / fps)
    pose_times, pose_valid = recording_pose_validity(recording)
    cycle_candidates = enumerate_cycle_candidates(
        events,
        per_frame=per_frame,
        pose_times=pose_times,
        pose_valid=pose_valid,
        fps=fps,
    )
    cycles = detect_gait_cycles(
        events,
        duration_s=duration_s,
        per_frame=per_frame,
        cycle_candidates=cycle_candidates,
        pose_times=pose_times,
        pose_valid=pose_valid,
        fps=fps,
    )

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
    withhold = _contact_partition_untrustworthy(warnings)
    if withhold:
        metrics.metrics_reliable = False
        metrics.session_metrics_withheld = True
        metrics.cadence_steps_per_min = None
        metrics.left_right_stance_symmetry = None
        metrics.left_right_swing_symmetry = None
        metrics.step_time_s = None
        metrics.stride_time_s = None
        metrics.reliability_reason = (
            "Near-constant foot contact (>85% of foot-frames). "
            "Cadence and symmetry are withheld because heel-strike intervals "
            "are not reliable under saturated contact. Cycles were not fabricated. "
            "Walking aids and shuffle gait often cause this."
        )
    # Log each distinct warning once per process — analysis may re-run on load
    # and the repeated WARNING spam slows the UI thread / floods the console.
    seen = getattr(validate_gait_cycle_analysis, "_logged_warnings", None)
    if not isinstance(seen, set):
        seen = set()
        setattr(validate_gait_cycle_analysis, "_logged_warnings", seen)
    for w in warnings:
        key = str(w)
        if key in seen:
            continue
        seen.add(key)
        logger.warning("Gait cycle validation: %s", w)

    _log_analysis_summary(metrics)

    sufficiency, sufficiency_reason = classify_event_sufficiency(
        frame_count=len(per_frame),
        duration_s=duration_s,
        ground_plane=plane,
        heel_strike_count=hs_count,
        toe_off_count=metrics.left_toe_off_count + metrics.right_toe_off_count,
        cycle_count=len(cycles),
        foot_visibility_mean=foot_vis,
        warnings=warnings,
    )
    if withhold:
        sufficiency = "input_insufficient"
        sufficiency_reason = metrics.reliability_reason
    if sufficiency != "none" and sufficiency_reason:
        note = (
            "Scientifically insufficient input: "
            if sufficiency == "input_insufficient"
            else "Technical processing limitation: "
        ) + sufficiency_reason
        if note not in warnings:
            warnings.append(note)

    return GaitCycleAnalysisResult(
        per_frame=per_frame,
        events=events,
        cycles=cycles,
        cycle_candidates=cycle_candidates,
        metrics=metrics,
        ground_plane=plane,
        vertical_axis=axis,
        fps=fps,
        contact_thresholds=thresholds,
        warnings=warnings,
        event_sufficiency=sufficiency,
        event_sufficiency_reason=sufficiency_reason,
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
    if not metrics.metrics_reliable:
        if metrics.cadence_steps_per_min is not None:
            warnings.append(
                "Temporal gait metrics computed at reduced reliability: "
                f"{metrics.reliability_reason}"
            )
        else:
            warnings.append(
                f"Final temporal gait metrics unavailable: {metrics.reliability_reason}"
            )
    if metrics.metrics_reliable:
        for side in ("left", "right"):
            stance = getattr(metrics, f"{side}_stance_pct")
            swing = getattr(metrics, f"{side}_swing_pct")
            if stance is None or swing is None:
                warnings.append(f"{side.title()} stance/swing percentages are unavailable.")
            elif not (0.0 <= stance <= 100.0 and 0.0 <= swing <= 100.0):
                warnings.append(
                    f"{side.title()} stance/swing percentages are outside 0–100% "
                    f"({stance:.2f}%/{swing:.2f}%)."
                )
            elif abs((stance + swing) - 100.0) > 0.5:
                warnings.append(
                    f"{side.title()} stance + swing is {stance + swing:.2f}%, not approximately 100%."
                )
        for label, value in (
            ("Double support", metrics.double_support_pct),
            ("Single support", metrics.single_support_pct),
        ):
            if value is not None and not (0.0 <= value <= 100.0):
                warnings.append(f"{label} is outside 0–100% ({value:.2f}%).")
        support_parts = (
            metrics.double_support_pct,
            metrics.single_support_pct,
            metrics.flight_pct,
        )
        if all(value is not None for value in support_parts):
            support_total = sum(value for value in support_parts if value is not None)
            if abs(support_total - 100.0) > 0.5:
                warnings.append(
                    "Double support + single support + flight is "
                    f"{support_total:.2f}%, not approximately 100%."
                )

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
    result = analyze_gait_cycles(recording, sequence=sequence)
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
    "detect_kinematic_foot_events",
    "annotate_event_confidence",
    "merge_gait_events",
    "kinematic_supplement_allowed",
    "apply_contact_state_machine",
    "raw_contact_desire",
    "compute_temporal_metrics",
    "cadence_is_exploratory",
    "summary_cadence_spm",
    "assess_contact_confidence",
]
