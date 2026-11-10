"""Map a shared 0–100% gait phase onto each clip's own timeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stablewalk.analysis.gait_cycle_analysis import DetectedGaitCycle, GaitCycleAnalysisResult


@dataclass(frozen=True)
class PhaseMapping:
    phase_pct: float
    time_s: float
    frame_index: int
    used_gait_cycle: bool
    note: str


def _times_and_frames(recording) -> tuple[np.ndarray, np.ndarray]:
    snaps = list(getattr(recording, "snapshots", None) or [])
    if not snaps:
        return np.zeros(0), np.zeros(0, dtype=int)
    times = np.array([float(s.time_s) for s in snaps], dtype=float)
    frames = np.array([int(s.frame_index) for s in snaps], dtype=int)
    return times, frames


def map_gait_phase(
    recording,
    cycles: GaitCycleAnalysisResult | None,
    phase_pct: float,
) -> PhaseMapping:
    """Independent mapping: this clip's cycles / duration only."""
    pct = float(np.clip(phase_pct, 0.0, 100.0))
    times, frames = _times_and_frames(recording)
    if times.size == 0:
        return PhaseMapping(pct, 0.0, 0, False, "No frames in this clip.")
    cycle_list: list[DetectedGaitCycle] = list(getattr(cycles, "cycles", None) or [])
    usable = [
        c
        for c in cycle_list
        if getattr(c, "duration_s", 0.0) >= 0.35
        and getattr(c, "end_time_s", 0.0) > getattr(c, "start_time_s", 0.0)
    ]
    if usable:
        cycle = usable[len(usable) // 2]
        t = cycle.start_time_s + (pct / 100.0) * cycle.duration_s
        idx = int(np.argmin(np.abs(times - t)))
        return PhaseMapping(
            pct,
            float(times[idx]),
            int(frames[idx]),
            True,
            "Mapped through this clip's own gait cycle (not the other video).",
        )
    t0 = float(times[0])
    t1 = float(times[-1])
    t = t0 + (pct / 100.0) * max(t1 - t0, 0.0)
    idx = int(np.argmin(np.abs(times - t)))
    return PhaseMapping(
        pct,
        float(times[idx]),
        int(frames[idx]),
        False,
        "No gait cycle on this clip — using video progress 0–100%.",
    )
