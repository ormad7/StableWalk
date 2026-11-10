"""Session analysis cache — compute once, display many times.

The gait pipeline and per-session ``analyze_*`` suite run when a video/poses
session is loaded. Playback, graphs, skeleton, and metrics read from this
cache and must never re-trigger analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrameDisplaySnapshot:
    """Precomputed per-frame labels for zero-cost playback display."""

    phase_raw: str | None = None
    phase_dashboard: str = "—"
    left_contact: int | None = None
    right_contact: int | None = None
    # Rancho / Perry gait-cycle phase (live Overview indicator).
    cycle_phase: str | None = None
    cycle_phase_label: str = "—"
    cycle_percent: float | None = None


@dataclass
class SessionAnalysisCache:
    """Immutable-ish bag of session results after one analysis pass."""

    generation: int = 0
    stability: Any = None
    biomech_stability: Any = None
    gait_cycle: Any = None
    gait_asymmetry_moments: Any = None
    foot_contact: Any = None
    estimated_vgrf: Any = None
    gait_features: Any = None
    virtual_grf: Any = None
    biomech_analysis: Any = None
    biomech_frame_cache: dict[int, dict] | None = None
    biomech_com_sequence: list[dict] | None = None
    biomech_com_index: dict[int, int] | None = None
    # frame_index → display snapshot (phase / contact) — filled once
    frame_display: dict[int, FrameDisplaySnapshot] = field(default_factory=dict)
    ready: bool = False

    def clear(self) -> None:
        self.generation += 1
        self.stability = None
        self.biomech_stability = None
        self.gait_cycle = None
        self.gait_asymmetry_moments = None
        self.foot_contact = None
        self.estimated_vgrf = None
        self.gait_features = None
        self.virtual_grf = None
        self.biomech_analysis = None
        self.biomech_frame_cache = None
        self.biomech_com_sequence = None
        self.biomech_com_index = None
        self.frame_display.clear()
        self.ready = False

    def frame_snapshot(self, frame_index: int | None) -> FrameDisplaySnapshot | None:
        if frame_index is None or not self.frame_display:
            return None
        return self.frame_display.get(int(frame_index))


def build_gait_frame_display(
    gait_cycle: Any,
    *,
    format_dashboard_phase,
) -> dict[int, FrameDisplaySnapshot]:
    """Precompute phase/contact labels for every gait-cycle frame (once)."""
    out: dict[int, FrameDisplaySnapshot] = {}
    if gait_cycle is None or not getattr(gait_cycle, "per_frame", None):
        return out

    from stablewalk.analysis.gait_cycle_phase import (
        classify_gait_cycle_phase,
        estimate_cycle_percent_from_hs_times,
        format_gait_cycle_phase,
        heel_strike_frame_set,
        heel_strike_times_from_gait_cycle,
    )
    from stablewalk.analysis.gait_phase_classification import (
        classify_gait_phase_from_contacts,
    )

    metrics = getattr(gait_cycle, "metrics", None)
    contact_confidence = getattr(metrics, "contact_confidence", None) if metrics else None
    confidence_tier = getattr(metrics, "confidence_tier", None) if metrics else None
    hs_times = heel_strike_times_from_gait_cycle(gait_cycle)
    hs_frames = heel_strike_frame_set(gait_cycle)

    for state in gait_cycle.per_frame:
        derived = classify_gait_phase_from_contacts(
            state.left_contact,
            state.right_contact,
            contact_confidence=contact_confidence,
            confidence_tier=confidence_tier,
            left_foot_clearance_m=state.left.foot_clearance_m,
            right_foot_clearance_m=state.right.foot_clearance_m,
        )
        frame_i = int(state.frame_index)
        time_s = float(getattr(state, "time_s", 0.0) or 0.0)
        cycle_pct = estimate_cycle_percent_from_hs_times(time_s, hs_times)
        cycle_phase = classify_gait_cycle_phase(
            cycle_pct,
            heel_strike_pulse=frame_i in hs_frames,
        )
        out[frame_i] = FrameDisplaySnapshot(
            phase_raw=derived,
            phase_dashboard=format_dashboard_phase(derived) if derived else "—",
            left_contact=int(state.left_contact),
            right_contact=int(state.right_contact),
            cycle_phase=cycle_phase,
            cycle_phase_label=(
                format_gait_cycle_phase(cycle_phase) if cycle_phase else "—"
            ),
            cycle_percent=cycle_pct,
        )
    return out
