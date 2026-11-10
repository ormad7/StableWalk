"""Gait-event wrapper around the existing contact / cycle detector.

Does not reimplement heel-strike or toe-off detection. It reads
``GaitCycleAnalysisResult`` (from ``analyze_gait_cycles``) and attaches
view-based reliability so frontal / unknown cameras do not silently
produce spatial step lengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from stablewalk.analysis.asymmetry3d.indices import (
    ALIGNMENT_EVENT,
    ALIGNMENT_EVENT_LABEL,
)
from stablewalk.analysis.gait_cycle_analysis import (
    MAX_COMPLETE_CYCLE_DURATION_S,
    MIN_COMPLETE_CYCLE_DURATION_S,
    DetectedGaitCycle,
    GaitCycleAnalysisResult,
    GaitEvent,
    analyze_gait_cycles,
)
from stablewalk.analysis.gait_view_analysis import (
    GaitViewEstimate,
    GaitViewType,
    ViewReliabilityProfile,
    analyze_gait_view_geometry,
    build_view_reliability_profile,
)
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.pose_data import PoseSequence

Reliability = Literal["HIGH", "MODERATE", "LOW", "UNAVAILABLE"]

_SPATIAL_UNRELIABLE_VIEWS = frozenset(
    {GaitViewType.FRONTAL, GaitViewType.UNKNOWN}
)


@dataclass
class Interval:
    side: str
    start_s: float
    end_s: float
    duration_s: float
    kind: str


@dataclass
class GaitEventSummary:
    heel_strikes: list[GaitEvent] = field(default_factory=list)
    toe_offs: list[GaitEvent] = field(default_factory=list)
    left_cycles: list[DetectedGaitCycle] = field(default_factory=list)
    right_cycles: list[DetectedGaitCycle] = field(default_factory=list)
    stance_intervals: list[Interval] = field(default_factory=list)
    swing_intervals: list[Interval] = field(default_factory=list)
    left_step_durations_s: list[float] = field(default_factory=list)
    right_step_durations_s: list[float] = field(default_factory=list)
    left_stride_durations_s: list[float] = field(default_factory=list)
    right_stride_durations_s: list[float] = field(default_factory=list)
    view_type: str = "UNKNOWN"
    view_confidence: float = 0.0
    events_reliable: bool = False
    spatial_reliable: bool = False
    camera_unsuitable: bool = False
    reliability: Reliability = "UNAVAILABLE"
    reliability_reason: str = "No gait cycles."
    warnings: list[str] = field(default_factory=list)
    source: str = "gait_cycle_analysis.analyze_gait_cycles"
    alignment_event: str = ALIGNMENT_EVENT
    alignment_label: str = ALIGNMENT_EVENT_LABEL
    alignment_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_heel_strikes": len(self.heel_strikes),
            "n_toe_offs": len(self.toe_offs),
            "n_left_cycles": len(self.left_cycles),
            "n_right_cycles": len(self.right_cycles),
            "n_stance": len(self.stance_intervals),
            "n_swing": len(self.swing_intervals),
            "left_step_duration_s_median": _median(self.left_step_durations_s),
            "right_step_duration_s_median": _median(self.right_step_durations_s),
            "left_stride_duration_s_median": _median(self.left_stride_durations_s),
            "right_stride_duration_s_median": _median(self.right_stride_durations_s),
            "view_type": self.view_type,
            "view_confidence": self.view_confidence,
            "events_reliable": self.events_reliable,
            "spatial_reliable": self.spatial_reliable,
            "camera_unsuitable": self.camera_unsuitable,
            "reliability": self.reliability,
            "reliability_reason": self.reliability_reason,
            "warnings": list(self.warnings),
            "source": self.source,
            "alignment_event": self.alignment_event,
            "alignment_label": self.alignment_label,
            "alignment_verified": self.alignment_verified,
        }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def split_cycles_by_side(
    cycles: list[DetectedGaitCycle],
) -> tuple[list[DetectedGaitCycle], list[DetectedGaitCycle]]:
    left: list[DetectedGaitCycle] = []
    right: list[DetectedGaitCycle] = []
    for cycle in cycles:
        side = str(cycle.boundary_side or "").lower()
        if side.startswith("left"):
            left.append(cycle)
        elif side.startswith("right"):
            right.append(cycle)
    return left, right


def cycles_from_heel_strikes(
    events: list[GaitEvent],
    side: str,
) -> list[DetectedGaitCycle]:
    """Build ipsilateral HS-to-HS cycles for one side.

    Does not replace ``GaitCycleAnalysisResult.cycles`` (Overview cadence
    still uses the primary detector output).
    """
    key = f"{side}_heel_strike"
    bounds = sorted(
        (
            event
            for event in events
            if event.event_type == key
            or (event.side == side and "heel_strike" in event.event_type)
        ),
        key=lambda event: event.time_s,
    )
    cycles: list[DetectedGaitCycle] = []
    seen: set[tuple[int, int]] = set()
    for start, end in zip(bounds, bounds[1:]):
        dur = float(end.time_s - start.time_s)
        if not (MIN_COMPLETE_CYCLE_DURATION_S <= dur <= MAX_COMPLETE_CYCLE_DURATION_S):
            continue
        stamp = (int(start.frame_index), int(end.frame_index))
        if stamp in seen:
            continue
        seen.add(stamp)
        cycles.append(
            DetectedGaitCycle(
                cycle_index=len(cycles),
                start_frame=int(start.frame_index),
                end_frame=int(end.frame_index),
                start_time_s=float(start.time_s),
                end_time_s=float(end.time_s),
                duration_s=dur,
                boundary_side=side,
            )
        )
    return cycles


def left_right_gait_cycles(
    cycles: GaitCycleAnalysisResult,
) -> tuple[list[DetectedGaitCycle], list[DetectedGaitCycle]]:
    """Left and right gait cycles for phase-normalized comparison.

    Uses ``cycles.cycles`` when both sides are present. If the primary
    detector kept only one HS series, the missing side is filled from
    ipsilateral heel-strike events. Cadence is not rewritten.
    """
    left, right = split_cycles_by_side(list(cycles.cycles))
    events = list(cycles.events)
    if not left:
        left = cycles_from_heel_strikes(events, "left")
    if not right:
        right = cycles_from_heel_strikes(events, "right")
    return left, right


def heel_strike_events(events: list[GaitEvent], side: str) -> list[GaitEvent]:
    key = f"{side}_heel_strike"
    return [
        event
        for event in events
        if event.event_type == key
        or (event.side == side and "heel_strike" in event.event_type)
    ]


def heel_strike_times(events: list[GaitEvent], side: str) -> list[float]:
    return [float(event.time_s) for event in heel_strike_events(events, side)]


def _cycle_matches_heel_strike(
    cycle: DetectedGaitCycle,
    strikes: list[GaitEvent],
    *,
    fps: float,
) -> bool:
    """True when the cycle start matches a heel-strike by time or frame."""
    if not strikes:
        return True
    tol_s = max(1.5 / max(float(fps), 1.0), 0.04)
    tol_f = max(2, int(round(1.5)))
    start_t = float(cycle.start_time_s)
    start_f = int(cycle.start_frame)
    for event in strikes:
        if abs(start_t - float(event.time_s)) <= tol_s:
            return True
        if abs(start_f - int(event.frame_index)) <= tol_f:
            return True
    return False


def cycles_aligned_to_heel_strike(
    cycles: list[DetectedGaitCycle],
    events: list[GaitEvent],
    side: str,
    *,
    fps: float = 30.0,
) -> list[DetectedGaitCycle]:
    """Keep ipsilateral HS→HS cycles whose start matches a heel-strike event.

    Detector cycles are already HS-bounded. Matching uses time **or** frame
    index so a small clock offset does not drop a valid cycle. Labelled
    detector cycles are retained when no event list is attached.
    """
    labelled = [
        c for c in cycles if str(c.boundary_side or "").lower().startswith(side)
    ]
    if not labelled:
        return []
    strikes = heel_strike_events(events, side)
    if not strikes:
        return labelled
    kept = [
        cycle
        for cycle in labelled
        if _cycle_matches_heel_strike(cycle, strikes, fps=fps)
    ]
    return kept if kept else labelled


def _event_times(events: list[GaitEvent], event_type: str) -> list[float]:
    return [float(e.time_s) for e in events if e.event_type == event_type]


def _pairwise_durations(times: list[float]) -> list[float]:
    ordered = sorted(times)
    out: list[float] = []
    for a, b in zip(ordered, ordered[1:]):
        dt = b - a
        if 0.2 <= dt <= 3.0:
            out.append(dt)
    return out


def _contralateral_steps(left_hs: list[float], right_hs: list[float]) -> tuple[list[float], list[float]]:
    """Step duration = HS of one side to the next HS of the other side."""
    merged = sorted([(t, "L") for t in left_hs] + [(t, "R") for t in right_hs])
    left_steps: list[float] = []
    right_steps: list[float] = []
    for (t0, s0), (t1, s1) in zip(merged, merged[1:]):
        dt = t1 - t0
        if not (0.15 <= dt <= 2.0) or s0 == s1:
            continue
        if s0 == "L":
            left_steps.append(dt)
        else:
            right_steps.append(dt)
    return left_steps, right_steps


def _phase_intervals(
    hs: list[float],
    to: list[float],
    *,
    side: str,
) -> tuple[list[Interval], list[Interval]]:
    stance: list[Interval] = []
    swing: list[Interval] = []
    hs_s = sorted(hs)
    to_s = sorted(to)
    for strike in hs_s:
        later_to = [t for t in to_s if t > strike + 1e-4]
        if not later_to:
            continue
        toe = later_to[0]
        if toe - strike > 2.5:
            continue
        stance.append(Interval(side, strike, toe, toe - strike, "stance"))
        later_hs = [t for t in hs_s if t > toe + 1e-4]
        if later_hs and later_hs[0] - toe <= 2.5:
            swing.append(Interval(side, toe, later_hs[0], later_hs[0] - toe, "swing"))
    return stance, swing


def summarize_gait_events(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult | None = None,
    *,
    sequence: PoseSequence | None = None,
    view_estimate: GaitViewEstimate | None = None,
    detect_if_missing: bool = False,
) -> GaitEventSummary:
    warnings: list[str] = []
    if cycles is None and detect_if_missing:
        try:
            cycles = analyze_gait_cycles(recording)
        except Exception as exc:  # pragma: no cover - detector guard
            warnings.append(f"Gait-cycle analysis failed: {exc}")
            cycles = GaitCycleAnalysisResult(fps=float(recording.fps or 30.0))
    elif cycles is None:
        warnings.append(
            "No gait-cycle result supplied; heel-strike / toe-off timing is unavailable. "
            "Pass the existing analyze_gait_cycles output rather than a second detector."
        )
        cycles = GaitCycleAnalysisResult(fps=float(recording.fps or 30.0))

    events = list(cycles.events)
    heel = [e for e in events if "heel_strike" in e.event_type]
    toe = [e for e in events if "toe_off" in e.event_type]
    left_raw, right_raw = left_right_gait_cycles(cycles)
    fps = float(getattr(cycles, "fps", None) or getattr(recording, "fps", None) or 30.0)
    left_matched = [
        c
        for c in left_raw
        if _cycle_matches_heel_strike(c, heel_strike_events(events, "left"), fps=fps)
    ]
    right_matched = [
        c
        for c in right_raw
        if _cycle_matches_heel_strike(c, heel_strike_events(events, "right"), fps=fps)
    ]
    left_cycles = cycles_aligned_to_heel_strike(left_raw, events, "left", fps=fps)
    right_cycles = cycles_aligned_to_heel_strike(right_raw, events, "right", fps=fps)
    if left_raw and not left_matched:
        warnings.append(
            "Left cycle start stamps were loosely matched to ipsilateral "
            "heel-strike; detector HS–HS cycles are retained for comparison."
        )
    if right_raw and not right_matched:
        warnings.append(
            "Right cycle start stamps were loosely matched to ipsilateral "
            "heel-strike; detector HS–HS cycles are retained for comparison."
        )
    alignment_verified = bool(left_matched) and bool(right_matched)

    left_hs = _event_times(events, "left_heel_strike")
    right_hs = _event_times(events, "right_heel_strike")
    left_to = _event_times(events, "left_toe_off")
    right_to = _event_times(events, "right_toe_off")
    left_steps, right_steps = _contralateral_steps(left_hs, right_hs)
    left_stride = _pairwise_durations(left_hs)
    right_stride = _pairwise_durations(right_hs)
    st_l, sw_l = _phase_intervals(left_hs, left_to, side="left")
    st_r, sw_r = _phase_intervals(right_hs, right_to, side="right")

    estimate = view_estimate
    profile: ViewReliabilityProfile | None = None
    if estimate is None and sequence is not None and sequence.frames:
        try:
            estimate, profile = analyze_gait_view_geometry(sequence.frames)
        except Exception:
            estimate = None
    if estimate is None:
        view_type = GaitViewType.UNKNOWN
        view_conf = 0.0
        profile = build_view_reliability_profile(view_type)
        warnings.append("Camera view unknown — spatial gait metrics are down-weighted.")
    else:
        view_type = estimate.view_type
        view_conf = float(estimate.view_confidence)
        profile = profile or build_view_reliability_profile(view_type)

    spatial_ok = view_type not in _SPATIAL_UNRELIABLE_VIEWS and view_conf >= 0.35
    n_cycles = len(left_cycles) + len(right_cycles)
    contact_conf = float(getattr(cycles.metrics, "contact_confidence", 0.0) or 0.0)
    events_ok = n_cycles >= 1 and (len(heel) >= 2) and contact_conf >= 0.35
    camera_bad = view_type == GaitViewType.UNKNOWN and view_conf < 0.25
    if view_type == GaitViewType.FRONTAL:
        warnings.append(
            "Frontal view: step/stride *length* is unreliable; timing and "
            "sagittal angles are still estimated with reduced confidence."
        )
        spatial_ok = False
    if not events_ok:
        warnings.append("Insufficient complete gait cycles for event timing.")

    if camera_bad:
        reliability: Reliability = "UNAVAILABLE"
        reason = "Camera angle unsuitable for gait-event detection."
    elif not events_ok:
        reliability = "LOW"
        reason = cycles.metrics.reliability_reason or "Too few gait cycles."
    elif spatial_ok and contact_conf >= 0.6:
        reliability = "HIGH"
        reason = f"{n_cycles} cycles · view {view_type.value}."
    else:
        reliability = "MODERATE"
        reason = (
            f"{n_cycles} cycles · view {view_type.value} · "
            f"spatial={'ok' if spatial_ok else 'limited'}."
        )

    timing_tier = profile.metric_tier("step_timing") if profile else "MODERATE"
    if timing_tier == "LOW" and reliability == "HIGH":
        reliability = "MODERATE"

    return GaitEventSummary(
        heel_strikes=heel,
        toe_offs=toe,
        left_cycles=left_cycles,
        right_cycles=right_cycles,
        stance_intervals=st_l + st_r,
        swing_intervals=sw_l + sw_r,
        left_step_durations_s=left_steps,
        right_step_durations_s=right_steps,
        left_stride_durations_s=left_stride,
        right_stride_durations_s=right_stride,
        view_type=view_type.value,
        view_confidence=view_conf,
        events_reliable=events_ok and not camera_bad,
        spatial_reliable=spatial_ok and events_ok and not camera_bad,
        camera_unsuitable=camera_bad,
        reliability=reliability,
        reliability_reason=reason,
        warnings=warnings,
        alignment_event=ALIGNMENT_EVENT,
        alignment_label=ALIGNMENT_EVENT_LABEL,
        alignment_verified=alignment_verified,
    )
