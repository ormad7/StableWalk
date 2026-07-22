"""
Robust foot-contact and gait-event analysis for StableWalk.

Combines heel/toe clearance relative to the estimated ground plane, vertical and
horizontal foot velocities, pose confidence, temporal smoothing, and hysteresis.
Each foot follows a sub-state machine:

  Swing → Heel Strike → Foot Flat → Mid Stance → Toe Off → Swing
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from stablewalk.analysis.gait_cycle_analysis import (
    ContactThresholds,
    DetectedGaitCycle,
    FootLandmarkSample,
    FrameContactState,
    GaitCycleAnalysisResult,
    GaitEvent,
    GaitTemporalMetrics,
    analyze_gait_cycles,
    compute_temporal_metrics,
    detect_gait_cycles,
    smooth_series,
    symmetry_ratio,
)
from stablewalk.analysis.gait_phase_classification import classify_gait_phase_from_contacts
from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot

logger = logging.getLogger(__name__)

FootSubStateName = Literal[
    "swing",
    "heel_strike",
    "foot_flat",
    "mid_stance",
    "toe_off",
]

GaitMacroPhase = Literal["stance", "swing", "double_support", "uncertain"]


class FootSubState(str, Enum):
    SWING = "swing"
    HEEL_STRIKE = "heel_strike"
    FOOT_FLAT = "foot_flat"
    MID_STANCE = "mid_stance"
    TOE_OFF = "toe_off"


# Probability / hysteresis gates (multi-signal — not a single height threshold).
PROB_SMOOTH_WINDOW = 5
PROB_CONTACT_ENTER = 0.58
PROB_CONTACT_EXIT = 0.42
HS_FRAMES = 3
FOOT_FLAT_FRAMES = 4
TOE_OFF_STANCE_FRACTION = 0.22
MID_STANCE_FRACTION = 0.35


@dataclass(frozen=True)
class FootContactFrame:
    """Per-frame bilateral contact, events, and foot sub-states."""

    frame_index: int
    time_s: float
    left_contact_probability: float
    right_contact_probability: float
    left_contact_binary: int
    right_contact_binary: int
    left_heel_strike: int
    right_heel_strike: int
    left_toe_off: int
    right_toe_off: int
    left_foot_substate: FootSubStateName
    right_foot_substate: FootSubStateName
    macro_phase: GaitMacroPhase
    left_confidence: float
    right_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FootContactMetrics:
    """Temporal gait metrics derived from robust contact analysis."""

    cadence_steps_per_min: float | None = None
    step_time_s: float | None = None
    stride_time_s: float | None = None
    left_stance_duration_s: float | None = None
    right_stance_duration_s: float | None = None
    left_swing_duration_s: float | None = None
    right_swing_duration_s: float | None = None
    average_stance_duration_s: float | None = None
    average_swing_duration_s: float | None = None
    double_support_duration_s: float | None = None
    double_support_pct: float | None = None
    left_right_temporal_asymmetry: float | None = None
    valid_gait_cycle_count: int = 0
    gait_event_confidence: float = 0.0
    contact_confidence: float = 0.0
    metrics_reliable: bool = False
    reliability_reason: str = "No complete gait cycles were analyzed."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FootContactAnalysisResult:
    per_frame: list[FootContactFrame] = field(default_factory=list)
    events: list[GaitEvent] = field(default_factory=list)
    cycles: list[DetectedGaitCycle] = field(default_factory=list)
    metrics: FootContactMetrics = field(default_factory=FootContactMetrics)
    gait_cycles: GaitCycleAnalysisResult | None = None
    fps: float = 30.0
    warnings: list[str] = field(default_factory=list)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([f.time_s for f in self.per_frame], dtype=np.float64)

    @property
    def left_contact_probability(self) -> np.ndarray:
        return np.array([f.left_contact_probability for f in self.per_frame], dtype=np.float64)

    @property
    def right_contact_probability(self) -> np.ndarray:
        return np.array([f.right_contact_probability for f in self.per_frame], dtype=np.float64)

    @property
    def left_contact_binary(self) -> np.ndarray:
        return np.array([f.left_contact_binary for f in self.per_frame], dtype=np.int8)

    @property
    def right_contact_binary(self) -> np.ndarray:
        return np.array([f.right_contact_binary for f in self.per_frame], dtype=np.int8)

    def frame_at(self, frame_index: int) -> FootContactFrame | None:
        for state in self.per_frame:
            if state.frame_index == frame_index:
                return state
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "metrics": self.metrics.to_dict(),
            "event_count": len(self.events),
            "cycle_count": len(self.cycles),
            "per_frame_count": len(self.per_frame),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sigmoid(x: float) -> float:
    x = max(-12.0, min(12.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def clearance_contact_score(
    clearance_m: float | None,
    *,
    entry_m: float,
    exit_m: float,
    in_contact: bool,
) -> float | None:
    """Higher score when foot is closer to the ground (multi-signal component)."""
    if clearance_m is None:
        return None
    span = max(exit_m - entry_m, 1e-6)
    if in_contact:
        # Hysteresis: tolerate higher clearance while already in contact.
        z = (exit_m - clearance_m) / span * 5.0
    else:
        z = (entry_m - clearance_m) / span * 5.0 + 0.5
    return _clamp01(_sigmoid(z))


def velocity_contact_score(
    vertical_m_s: float | None,
    horizontal_m_s: float | None,
    *,
    entry_max_m_s: float,
    exit_min_m_s: float,
    in_contact: bool,
) -> float | None:
    """Low foot speed suggests ground contact."""
    speeds: list[float] = []
    if vertical_m_s is not None:
        speeds.append(abs(vertical_m_s))
    if horizontal_m_s is not None:
        speeds.append(abs(horizontal_m_s))
    if not speeds:
        return None
    speed = max(speeds)
    gate = exit_min_m_s if in_contact else entry_max_m_s
    span = max(gate, 0.08)
    z = (gate - speed) / span * 4.0
    return _clamp01(_sigmoid(z))


def combine_contact_probability(
    *,
    heel_score: float | None,
    toe_score: float | None,
    vel_score: float | None,
    visibility: float,
    min_visibility: float = 0.35,
) -> float:
    """Weighted fusion of clearance, velocity, and confidence."""
    parts: list[tuple[float, float]] = []
    if heel_score is not None:
        parts.append((heel_score, 0.30))
    if toe_score is not None:
        parts.append((toe_score, 0.30))
    if vel_score is not None:
        parts.append((vel_score, 0.25))
    if not parts:
        return 0.0
    weight_sum = sum(w for _, w in parts)
    raw = sum(s * w for s, w in parts) / weight_sum
    vis_factor = _clamp01(visibility / max(min_visibility, 1e-6))
    return _clamp01(raw * (0.55 + 0.45 * vis_factor))


def apply_probability_hysteresis(
    probabilities: list[float],
    *,
    enter: float = PROB_CONTACT_ENTER,
    exit_threshold: float = PROB_CONTACT_EXIT,
) -> list[int]:
    """Schmitt trigger on smoothed contact probabilities."""
    states: list[int] = []
    in_contact = False
    for p in probabilities:
        if in_contact:
            if p < exit_threshold:
                in_contact = False
        elif p >= enter:
            in_contact = True
        states.append(1 if in_contact else 0)
    return states


def horizontal_speeds_m_s(
    xs: list[float | None],
    zs: list[float | None],
    times_s: list[float],
) -> list[float | None]:
    """Horizontal foot speed (m/s) from ankle/heel trajectory."""
    n = len(times_s)
    if n < 2:
        return [None] * n
    out: list[float | None] = [None] * n
    for i in range(1, n):
        if xs[i] is None or xs[i - 1] is None or zs[i] is None or zs[i - 1] is None:
            continue
        dt = times_s[i] - times_s[i - 1]
        if dt <= 1e-9:
            continue
        dx = xs[i] - xs[i - 1]
        dz = zs[i] - zs[i - 1]
        out[i] = math.hypot(dx, dz) / dt
    return out


def macro_phase_from_contacts(left: int, right: int) -> GaitMacroPhase:
    if left and right:
        return "double_support"
    if left or right:
        return "stance"
    return "swing"


def run_foot_substate_machine(
    contact_binary: list[int],
    samples: list[FootLandmarkSample],
    *,
    heel_strike_frames: list[int],
    toe_off_frames: list[int],
    fps: float,
) -> list[FootSubStateName]:
    """
    Per-foot state machine: Swing → HS → Foot Flat → Mid Stance → TO → Swing.
    """
    n = len(contact_binary)
    states: list[FootSubStateName] = ["swing"] * n
    hs_set = set(heel_strike_frames)
    to_set = set(toe_off_frames)

    stance_start: int | None = None
    for i in range(n):
        if contact_binary[i]:
            if stance_start is None:
                stance_start = i
            if i in hs_set or (stance_start is not None and i - stance_start < HS_FRAMES):
                states[i] = "heel_strike"
            else:
                stance_len = i - stance_start + 1
                frac = stance_len / max(int(round(fps * 0.55)), 1)
                heel_c = samples[i].heel_clearance_m
                toe_c = samples[i].toe_clearance_m
                flat = (
                    heel_c is not None
                    and toe_c is not None
                    and abs(heel_c - toe_c) < 0.025
                    and heel_c < (samples[i].foot_clearance_m or 0.05) + 0.02
                )
                if i in to_set or frac >= (1.0 - TOE_OFF_STANCE_FRACTION):
                    states[i] = "toe_off"
                elif flat and i - stance_start < HS_FRAMES + FOOT_FLAT_FRAMES:
                    states[i] = "foot_flat"
                elif frac >= MID_STANCE_FRACTION:
                    states[i] = "mid_stance"
                elif flat:
                    states[i] = "foot_flat"
                else:
                    states[i] = "mid_stance"
        else:
            stance_start = None
            if i in to_set:
                states[i] = "toe_off"
            else:
                states[i] = "swing"
    return states


def detect_event_pulses(
    contact_binary: list[int],
    side: str,
) -> tuple[list[int], list[int], list[GaitEvent], list[int], list[int]]:
    """Return HS/TO frame indices, events, and frame-indexed pulse arrays."""
    hs_frames: list[int] = []
    to_frames: list[int] = []
    events: list[GaitEvent] = []
    hs_pulse = [0] * len(contact_binary)
    to_pulse = [0] * len(contact_binary)

    prev = 0
    for i, c in enumerate(contact_binary):
        if c and not prev:
            hs_frames.append(i)
            hs_pulse[i] = 1
            events.append(
                GaitEvent(
                    event_type=f"{side}_heel_strike",  # type: ignore[arg-type]
                    frame_index=i,
                    time_s=0.0,
                    side=side,
                )
            )
        elif not c and prev:
            to_frames.append(i)
            to_pulse[i] = 1
            events.append(
                GaitEvent(
                    event_type=f"{side}_toe_off",  # type: ignore[arg-type]
                    frame_index=i,
                    time_s=0.0,
                    side=side,
                )
            )
        prev = c
    return hs_frames, to_frames, events, hs_pulse, to_pulse


def compute_foot_contact_metrics(
    per_frame: list[FootContactFrame],
    events: list[GaitEvent],
    cycles: list[DetectedGaitCycle],
    *,
    fps: float,
    base_metrics: GaitTemporalMetrics,
) -> FootContactMetrics:
    """Map gait-cycle metrics and add asymmetry / event confidence."""
    left_step = base_metrics.left_stance_time_s
    right_step = base_metrics.right_stance_time_s
    asym = None
    if left_step and right_step:
        asym = symmetry_ratio(left_step, right_step)

    hs_count = sum(1 for e in events if e.event_type.endswith("_heel_strike"))
    to_count = sum(1 for e in events if e.event_type.endswith("_toe_off"))
    event_conf = _clamp01(
        base_metrics.contact_confidence * 0.6
        + min(1.0, hs_count / 4.0) * 0.25
        + min(1.0, to_count / 4.0) * 0.15
    )

    return FootContactMetrics(
        cadence_steps_per_min=base_metrics.cadence_steps_per_min,
        step_time_s=base_metrics.step_time_s,
        stride_time_s=base_metrics.stride_time_s,
        left_stance_duration_s=base_metrics.left_stance_time_s,
        right_stance_duration_s=base_metrics.right_stance_time_s,
        left_swing_duration_s=base_metrics.left_swing_time_s,
        right_swing_duration_s=base_metrics.right_swing_time_s,
        average_stance_duration_s=base_metrics.average_stance_duration_s,
        average_swing_duration_s=base_metrics.average_swing_duration_s,
        double_support_duration_s=base_metrics.double_support_time_s,
        double_support_pct=base_metrics.double_support_pct,
        left_right_temporal_asymmetry=asym,
        valid_gait_cycle_count=base_metrics.gait_cycle_count,
        gait_event_confidence=event_conf,
        contact_confidence=base_metrics.contact_confidence,
        metrics_reliable=base_metrics.metrics_reliable,
        reliability_reason=base_metrics.reliability_reason,
    )


def _ankle_xz(snapshot: SkeletonSnapshot, side: str) -> tuple[float | None, float | None]:
    joint = snapshot.joints.get(f"{side}_ankle")
    if joint is None:
        joint = snapshot.joints.get(f"{side}_heel")
    if joint is None:
        return None, None
    return joint.position.x, joint.position.z


def _build_side_probability_series(
    samples: list[FootLandmarkSample],
    horizontal_speed: list[float | None],
    *,
    thresholds: ContactThresholds | None,
    binary_hint: list[int],
    smooth_window: int,
) -> tuple[list[float], list[int], list[float]]:
    entry = thresholds.entry_clearance_m if thresholds else 0.03
    exit_c = thresholds.exit_clearance_m if thresholds else 0.08
    entry_v = thresholds.entry_max_vel_m_s if thresholds else 0.28
    exit_v = thresholds.exit_min_vel_m_s if thresholds else 0.42

    raw_probs: list[float] = []
    confidences: list[float] = []
    in_contact = False
    for i, sample in enumerate(samples):
        in_contact = bool(binary_hint[i]) if i < len(binary_hint) else in_contact
        heel_s = clearance_contact_score(
            sample.heel_clearance_m, entry_m=entry, exit_m=exit_c, in_contact=in_contact
        )
        toe_s = clearance_contact_score(
            sample.toe_clearance_m, entry_m=entry, exit_m=exit_c, in_contact=in_contact
        )
        vel_s = velocity_contact_score(
            sample.heel_velocity_m_s,
            horizontal_speed[i] if i < len(horizontal_speed) else None,
            entry_max_m_s=entry_v,
            exit_min_m_s=exit_v,
            in_contact=in_contact,
        )
        prob = combine_contact_probability(
            heel_score=heel_s,
            toe_score=toe_s,
            vel_score=vel_s,
            visibility=sample.visibility,
        )
        raw_probs.append(prob)
        confidences.append(_clamp01(sample.visibility))

    smoothed = smooth_series(raw_probs, smooth_window)
    smooth_f = [float(v) if v is not None else 0.0 for v in smoothed]
    binary = apply_probability_hysteresis(smooth_f)
    return smooth_f, binary, confidences


# ---------------------------------------------------------------------------
# Main analysis entry
# ---------------------------------------------------------------------------
def analyze_foot_contact(
    recording: GaitMotionRecording,
    *,
    cycles: GaitCycleAnalysisResult | None = None,
    smooth_window: int = PROB_SMOOTH_WINDOW,
) -> FootContactAnalysisResult:
    """
    Run robust foot-contact + gait-event analysis on a gait motion recording.

    Builds on ``analyze_gait_cycles`` for ground plane and baseline contacts,
    then adds multi-signal probabilities, hysteresis, and per-foot sub-states.
    """
    gait = cycles if cycles is not None else analyze_gait_cycles(recording)
    fps = max(gait.fps, recording.fps, 1e-6)
    thresholds = gait.contact_thresholds
    warnings = list(gait.warnings)

    if not gait.per_frame:
        return FootContactAnalysisResult(
            fps=fps,
            gait_cycles=gait,
            warnings=warnings or ["No per-frame gait data — contact analysis skipped."],
        )

    frame_indices = [s.frame_index for s in gait.per_frame]
    times = [s.time_s for s in gait.per_frame]
    left_samples = [s.left for s in gait.per_frame]
    right_samples = [s.right for s in gait.per_frame]
    left_hint = [s.left_contact for s in gait.per_frame]
    right_hint = [s.right_contact for s in gait.per_frame]

    left_x, left_z, right_x, right_z = [], [], [], []
    for fi in frame_indices:
        snap = recording.snapshot_at(fi)
        if snap is None:
            left_x.append(None)
            left_z.append(None)
            right_x.append(None)
            right_z.append(None)
            continue
        lx, lz = _ankle_xz(snap, "left")
        rx, rz = _ankle_xz(snap, "right")
        left_x.append(lx)
        left_z.append(lz)
        right_x.append(rx)
        right_z.append(rz)

    left_horiz = horizontal_speeds_m_s(left_x, left_z, times)
    right_horiz = horizontal_speeds_m_s(right_x, right_z, times)

    left_prob, left_bin, left_conf = _build_side_probability_series(
        left_samples,
        left_horiz,
        thresholds=thresholds,
        binary_hint=left_hint,
        smooth_window=smooth_window,
    )
    right_prob, right_bin, right_conf = _build_side_probability_series(
        right_samples,
        right_horiz,
        thresholds=thresholds,
        binary_hint=right_hint,
        smooth_window=smooth_window,
    )

    l_hs_f, l_to_f, _, _, _ = detect_event_pulses(left_bin, "left")
    r_hs_f, r_to_f, _, _, _ = detect_event_pulses(right_bin, "right")

    events: list[GaitEvent] = []
    for i, (lb, rb) in enumerate(zip(left_bin, right_bin)):
        fi = gait.per_frame[i].frame_index
        ts = gait.per_frame[i].time_s
        if lb and (i == 0 or not left_bin[i - 1]):
            events.append(GaitEvent("left_heel_strike", fi, ts, "left"))
        if not lb and i > 0 and left_bin[i - 1]:
            events.append(GaitEvent("left_toe_off", fi, ts, "left"))
        if rb and (i == 0 or not right_bin[i - 1]):
            events.append(GaitEvent("right_heel_strike", fi, ts, "right"))
        if not rb and i > 0 and right_bin[i - 1]:
            events.append(GaitEvent("right_toe_off", fi, ts, "right"))

    l_hs_p = [0] * len(left_bin)
    l_to_p = [0] * len(left_bin)
    r_hs_p = [0] * len(right_bin)
    r_to_p = [0] * len(right_bin)
    for e in events:
        idx = next(
            (j for j, s in enumerate(gait.per_frame) if s.frame_index == e.frame_index),
            None,
        )
        if idx is None:
            continue
        if e.event_type == "left_heel_strike":
            l_hs_p[idx] = 1
        elif e.event_type == "left_toe_off":
            l_to_p[idx] = 1
        elif e.event_type == "right_heel_strike":
            r_hs_p[idx] = 1
        elif e.event_type == "right_toe_off":
            r_to_p[idx] = 1

    left_sub = run_foot_substate_machine(
        left_bin, left_samples, heel_strike_frames=l_hs_f, toe_off_frames=l_to_f, fps=fps
    )
    right_sub = run_foot_substate_machine(
        right_bin, right_samples, heel_strike_frames=r_hs_f, toe_off_frames=r_to_f, fps=fps
    )

    # Build FrameContactState list for temporal metrics
    frame_states: list[FrameContactState] = []
    per_frame: list[FootContactFrame] = []
    for i, base in enumerate(gait.per_frame):
        macro = macro_phase_from_contacts(left_bin[i], right_bin[i])
        per_frame.append(
            FootContactFrame(
                frame_index=base.frame_index,
                time_s=base.time_s,
                left_contact_probability=left_prob[i],
                right_contact_probability=right_prob[i],
                left_contact_binary=left_bin[i],
                right_contact_binary=right_bin[i],
                left_heel_strike=l_hs_p[i],
                right_heel_strike=r_hs_p[i],
                left_toe_off=l_to_p[i],
                right_toe_off=r_to_p[i],
                left_foot_substate=left_sub[i],
                right_foot_substate=right_sub[i],
                macro_phase=macro,
                left_confidence=left_conf[i],
                right_confidence=right_conf[i],
            )
        )
        phase = classify_gait_phase_from_contacts(left_bin[i], right_bin[i])
        frame_states.append(
            FrameContactState(
                frame_index=base.frame_index,
                time_s=base.time_s,
                left_contact=left_bin[i],
                right_contact=right_bin[i],
                phase=phase,
                left=base.left,
                right=base.right,
            )
        )

    duration_s = recording.duration_s or (len(per_frame) / fps)
    cycle_list = detect_gait_cycles(events, duration_s=duration_s)
    base_metrics = compute_temporal_metrics(
        frame_states,
        events,
        cycle_list,
        fps=fps,
        contact_confidence=gait.metrics.contact_confidence,
        confidence_tier=gait.metrics.confidence_tier,
    )
    metrics = compute_foot_contact_metrics(
        per_frame, events, cycle_list, fps=fps, base_metrics=base_metrics
    )

    return FootContactAnalysisResult(
        per_frame=per_frame,
        events=events,
        cycles=cycle_list,
        metrics=metrics,
        gait_cycles=gait,
        fps=fps,
        warnings=warnings,
    )


__all__ = [
    "FootSubState",
    "FootSubStateName",
    "FootContactFrame",
    "FootContactMetrics",
    "FootContactAnalysisResult",
    "GaitMacroPhase",
    "analyze_foot_contact",
    "apply_probability_hysteresis",
    "clearance_contact_score",
    "combine_contact_probability",
    "compute_foot_contact_metrics",
    "detect_event_pulses",
    "horizontal_speeds_m_s",
    "macro_phase_from_contacts",
    "run_foot_substate_machine",
    "velocity_contact_score",
]
