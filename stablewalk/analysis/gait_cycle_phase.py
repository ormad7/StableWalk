"""Rancho / Perry gait-cycle phases from stride progress (0–100%).

These eight phases are the live Overview indicator labels. Macro contact
phases (LEFT_STANCE / …) remain in ``gait_phase_classification`` for
contact-mask dashboards.
"""

from __future__ import annotations

from typing import Literal

GaitCyclePhaseName = Literal[
    "heel_strike",
    "loading_response",
    "mid_stance",
    "terminal_stance",
    "pre_swing",
    "initial_swing",
    "mid_swing",
    "terminal_swing",
]

# Display order for the live indicator chips.
GAIT_CYCLE_PHASE_ORDER: tuple[GaitCyclePhaseName, ...] = (
    "heel_strike",
    "loading_response",
    "mid_stance",
    "terminal_stance",
    "pre_swing",
    "initial_swing",
    "mid_swing",
    "terminal_swing",
)

GAIT_CYCLE_PHASE_LABELS: dict[GaitCyclePhaseName, str] = {
    "heel_strike": "Heel Strike",
    "loading_response": "Loading Response",
    "mid_stance": "Mid Stance",
    "terminal_stance": "Terminal Stance",
    "pre_swing": "Pre Swing",
    "initial_swing": "Initial Swing",
    "mid_swing": "Mid Swing",
    "terminal_swing": "Terminal Swing",
}

# Upper bounds (% of gait cycle) for Perry/Rancho bins after Heel Strike (0–2%).
# Heel Strike is treated as the initial-contact window at the start of the stride.
_PHASE_UPPER_PCT: tuple[tuple[GaitCyclePhaseName, float], ...] = (
    ("heel_strike", 2.0),
    ("loading_response", 12.0),
    ("mid_stance", 31.0),
    ("terminal_stance", 50.0),
    ("pre_swing", 62.0),
    ("initial_swing", 75.0),
    ("mid_swing", 87.0),
    ("terminal_swing", 100.0),
)


def format_gait_cycle_phase(phase: str | None) -> str:
    """Human-readable Rancho phase label."""
    if not phase:
        return "—"
    key = str(phase).strip().lower()
    if key in GAIT_CYCLE_PHASE_LABELS:
        return GAIT_CYCLE_PHASE_LABELS[key]  # type: ignore[index]
    return key.replace("_", " ").title()


def classify_gait_cycle_phase(
    cycle_percent: float | None,
    *,
    heel_strike_pulse: bool = False,
) -> GaitCyclePhaseName | None:
    """Map 0–100% gait-cycle progress to a Rancho phase.

    ``heel_strike_pulse`` forces Heel Strike when a HS event lands on this frame.
    """
    if heel_strike_pulse:
        return "heel_strike"
    if cycle_percent is None:
        return None
    try:
        pct = float(cycle_percent)
    except (TypeError, ValueError):
        return None
    if not (pct == pct):  # NaN
        return None
    pct = max(0.0, min(100.0, pct))
    for name, upper in _PHASE_UPPER_PCT:
        if pct <= upper + 1e-9:
            return name
    return "terminal_swing"


def estimate_cycle_percent_from_hs_times(
    time_s: float,
    heel_strike_times_s: list[float] | tuple[float, ...],
) -> float | None:
    """0–100% within the enclosing heel-strike → heel-strike interval."""
    hs = sorted(float(t) for t in heel_strike_times_s if t == t)
    if len(hs) < 2:
        return None
    t = float(time_s)
    for i in range(len(hs) - 1):
        t0, t1 = hs[i], hs[i + 1]
        if t0 <= t <= t1 and t1 > t0:
            return max(0.0, min(100.0, 100.0 * (t - t0) / (t1 - t0)))
    if t < hs[0]:
        return 0.0
    return 100.0


def heel_strike_times_from_gait_cycle(gait_cycle: object | None) -> list[float]:
    """Prefer left HS times; fall back to any heel-strike events."""
    if gait_cycle is None:
        return []
    events = list(getattr(gait_cycle, "events", None) or [])
    left = sorted(
        float(e.time_s)
        for e in events
        if getattr(e, "event_type", "") == "left_heel_strike"
    )
    if len(left) >= 2:
        return left
    any_hs = sorted(
        float(e.time_s)
        for e in events
        if str(getattr(e, "event_type", "")).endswith("_heel_strike")
    )
    return any_hs


def heel_strike_frame_set(gait_cycle: object | None) -> set[int]:
    """Frame indices that carry a heel-strike event pulse."""
    if gait_cycle is None:
        return set()
    out: set[int] = set()
    for e in list(getattr(gait_cycle, "events", None) or []):
        if not str(getattr(e, "event_type", "")).endswith("_heel_strike"):
            continue
        try:
            out.add(int(e.frame_index))
        except (TypeError, ValueError, AttributeError):
            continue
    return out


__all__ = [
    "GAIT_CYCLE_PHASE_LABELS",
    "GAIT_CYCLE_PHASE_ORDER",
    "GaitCyclePhaseName",
    "classify_gait_cycle_phase",
    "estimate_cycle_percent_from_hs_times",
    "format_gait_cycle_phase",
    "heel_strike_frame_set",
    "heel_strike_times_from_gait_cycle",
]
