"""Phase-normalized left/right gait asymmetry moments.

Detects ipsilateral left and right gait cycles, resamples hip / knee /
ankle flexion onto a shared 0–100% grid, and compares corresponding
phase samples. Per-frame scores are withheld when landmark confidence
is below ``MIN_CONFIDENCE``. Asymmetric moments are emitted only when
the score stays above ``SCORE_THRESHOLD`` for at least ``MIN_DURATION_S``.

This module does not write cadence, gait-quality, or other dashboard
metrics, and it does not name a medical condition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from stablewalk.analysis.asymmetry3d.events import (
    cycles_aligned_to_heel_strike,
    left_right_gait_cycles,
)
from stablewalk.analysis.asymmetry3d.indices import (
    ALIGNMENT_EVENT,
    ALIGNMENT_EVENT_LABEL,
    ASI_DEFINITION,
    ROBINSON_SI_DEFINITION,
    SI_NORM_DEFINITION,
    TIMING_DEFINITION,
    absolute_symmetry_index,
    circular_phase_difference_pp,
    normalized_symmetry_index,
    robinson_si,
)
from stablewalk.analysis.gait_cycle_analysis import (
    DetectedGaitCycle,
    GaitCycleAnalysisResult,
    GaitEvent,
)
from stablewalk.analysis.gait_feature_analysis import (
    GAIT_CYCLE_SAMPLE_COUNT,
    gait_cycle_percent_grid,
    resample_cycle_trajectory,
)
from stablewalk.models.gait_motion import GaitMotionRecording, Vec3
from stablewalk.models.joint_registry import ROOT_JOINT_ID

VIS_MIN = 0.45
SCORE_THRESHOLD = 28.0
MERGE_GAP_S = 0.22
TOP_K = 3
MIN_CONFIDENCE = 0.40
MIN_CYCLES = 1
# Consecutive above-threshold *used* samples must last at least 80 ms.
# Unused / low-confidence gaps do not count toward this duration.
MIN_DURATION_S = 0.08

AsymmetryOutcome = Literal[
    "moments",
    "symmetric_gait",
    "insufficient_data",
    "analysis_failed",
]

# Floor scales so tiny ROM does not inflate the 0–100 score.
_SCALE_FLOOR: dict[str, float] = {
    "hip_angle": 12.0,
    "knee_angle": 15.0,
    "ankle_angle": 8.0,
    "ankle_height": 0.04,
    "heel_height": 0.04,
    "foot_ground": 0.03,
    "step_timing": 0.08,
    "step_length": 0.05,
    "pelvis_tilt": 4.0,
    "pelvis_lateral": 0.03,
}

_WEIGHTS: dict[str, float] = {
    "hip_angle": 1.0,
    "knee_angle": 1.15,
    "ankle_angle": 1.0,
    "ankle_height": 0.95,
    "heel_height": 0.95,
    "foot_ground": 1.0,
    "step_timing": 0.80,
    "step_length": 0.80,
    "pelvis_tilt": 0.70,
    "pelvis_lateral": 0.70,
}

_JOINTS: dict[str, tuple[str, ...]] = {
    "hip_angle": ("right_hip", "left_hip"),
    "knee_angle": ("right_knee", "left_knee"),
    "ankle_angle": ("right_ankle", "left_ankle"),
    "ankle_height": ("right_ankle", "left_ankle"),
    "heel_height": ("right_heel", "left_heel"),
    "foot_ground": ("right_heel", "left_heel", "right_ankle", "left_ankle"),
    "step_timing": ("right_heel", "left_heel"),
    "step_length": ("right_heel", "left_heel"),
    "pelvis_tilt": ("pelvis", "left_hip", "right_hip"),
    "pelvis_lateral": ("pelvis",),
}

_UNITS: dict[str, str] = {
    "hip_angle": "deg",
    "knee_angle": "deg",
    "ankle_angle": "deg",
    "ankle_height": "body_normalized_m",
    "heel_height": "body_normalized_m",
    "foot_ground": "m",
    "step_timing": "s",
    "step_length": "body_normalized_m",
    "pelvis_tilt": "deg",
    "pelvis_lateral": "body_normalized_m",
}

_LABELS: dict[str, str] = {
    "hip_angle": "hip flexion",
    "knee_angle": "knee flexion",
    "ankle_angle": "ankle flexion",
    "ankle_height": "ankle height",
    "heel_height": "heel height",
    "foot_ground": "foot-to-ground distance",
    "step_timing": "step timing",
    "step_length": "step length",
    "pelvis_tilt": "pelvis tilt",
    "pelvis_lateral": "pelvis lateral movement",
}

# Odd-symmetry channels: left counterpart is compared to the negated right value.
_ODD_SYMMETRY = frozenset({"pelvis_tilt", "pelvis_lateral"})


@dataclass
class AsymmetryContribution:
    measurement: str
    joints: tuple[str, ...]
    difference: float
    units: str
    score: float
    left_value: float | None = None
    right_value: float | None = None
    phase_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement": self.measurement,
            "joints": list(self.joints),
            "difference": self.difference,
            "units": self.units,
            "score": self.score,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "phase_pct": self.phase_pct,
        }


@dataclass
class JointCycleComparison:
    """Session-level L/R comparison after 0–100% phase normalization."""

    name: str
    units: str = "deg"
    rom_left: float | None = None
    rom_right: float | None = None
    peak_left: float | None = None
    peak_right: float | None = None
    peak_difference: float | None = None
    peak_phase_left_pct: float | None = None
    peak_phase_right_pct: float | None = None
    timing_difference_pct: float | None = None
    timing_difference_units: str = "percentage points"
    si_percent: float | None = None
    asi_percent: float | None = None
    si_norm: float | None = None
    n_left_cycles: int = 0
    n_right_cycles: int = 0
    quality: str = "insufficient"
    alignment_event: str = ALIGNMENT_EVENT
    formula: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "units": self.units,
            "rom_left": self.rom_left,
            "rom_right": self.rom_right,
            "peak_left": self.peak_left,
            "peak_right": self.peak_right,
            "peak_difference": self.peak_difference,
            "peak_phase_left_pct": self.peak_phase_left_pct,
            "peak_phase_right_pct": self.peak_phase_right_pct,
            "timing_difference_pct": self.timing_difference_pct,
            "timing_difference_units": self.timing_difference_units,
            "si_percent": self.si_percent,
            "asi_percent": self.asi_percent,
            "si_norm": self.si_norm,
            "n_left_cycles": self.n_left_cycles,
            "n_right_cycles": self.n_right_cycles,
            "quality": self.quality,
            "alignment_event": self.alignment_event,
            "formula": self.formula,
        }


@dataclass
class AsymmetryMoment:
    frame_index: int
    time_s: float
    match_time_s: float
    involved_joints: list[str]
    contributing_measurements: list[AsymmetryContribution]
    severity: str
    score: float
    confidence: float
    explanation: str
    duration_s: float = 0.0
    phase_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "match_time_s": self.match_time_s,
            "involved_joints": list(self.involved_joints),
            "contributing_measurements": [c.to_dict() for c in self.contributing_measurements],
            "severity": self.severity,
            "score": self.score,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "duration_s": self.duration_s,
            "phase_pct": self.phase_pct,
            "kind": "measured_movement_difference",
            "clinical": False,
        }


@dataclass
class FrameAsymmetrySample:
    frame_index: int
    time_s: float
    match_time_s: float | None
    score: float | None
    confidence: float
    used: bool
    contributions: list[AsymmetryContribution] = field(default_factory=list)
    phase_pct: float | None = None
    match_phase_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "match_time_s": self.match_time_s,
            "score": self.score,
            "confidence": self.confidence,
            "used": self.used,
            "phase_pct": self.phase_pct,
            "match_phase_pct": self.match_phase_pct,
        }


@dataclass
class GaitAsymmetryMomentReport:
    available: bool
    cycle_duration_s: float | None
    half_cycle_s: float | None
    per_frame: list[FrameAsymmetrySample] = field(default_factory=list)
    moments: list[AsymmetryMoment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = "detected_gait_cycle"
    outcome: AsymmetryOutcome = "insufficient_data"
    joint_comparisons: list[JointCycleComparison] = field(default_factory=list)
    n_left_cycles: int = 0
    n_right_cycles: int = 0
    score_threshold: float = SCORE_THRESHOLD
    min_duration_s: float = MIN_DURATION_S
    min_confidence: float = MIN_CONFIDENCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "stablewalk-gait-asymmetry-moments",
            "schema_version": "1.1",
            "kind": "derived",
            "clinical": False,
            "available": self.available,
            "outcome": self.outcome,
            "cycle_duration_s": self.cycle_duration_s,
            "half_cycle_s": self.half_cycle_s,
            "n_left_cycles": self.n_left_cycles,
            "n_right_cycles": self.n_right_cycles,
            "score_threshold": self.score_threshold,
            "min_duration_s": self.min_duration_s,
            "min_confidence": self.min_confidence,
            "joint_comparisons": [j.to_dict() for j in self.joint_comparisons],
            "per_frame": [s.to_dict() for s in self.per_frame],
            "moments": [m.to_dict() for m in self.moments],
            "notes": list(self.notes),
            "source": self.source,
            "disclaimer": (
                "Scores describe measured left–right movement differences after "
                "phase-normalizing gait cycles to 0–100%. They are not a medical diagnosis."
            ),
        }


def _median(values: Sequence[float]) -> float | None:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _severity(score: float) -> str:
    if score >= 75.0:
        return "marked"
    if score >= 50.0:
        return "moderate"
    if score >= SCORE_THRESHOLD:
        return "mild"
    return "low"


def _in_any_cycle(time_s: float, cycles: Sequence[DetectedGaitCycle]) -> bool:
    return _enclosing_cycle(time_s, cycles) is not None


def _enclosing_cycle(
    time_s: float,
    cycles: Sequence[DetectedGaitCycle],
) -> DetectedGaitCycle | None:
    for cycle in cycles:
        if cycle.start_time_s <= time_s <= cycle.end_time_s:
            return cycle
    return None


def _phase_pct(time_s: float, cycle: DetectedGaitCycle) -> float | None:
    if cycle.duration_s <= 1e-9:
        return None
    return float(100.0 * (time_s - cycle.start_time_s) / cycle.duration_s)


def _time_at_phase(cycle: DetectedGaitCycle, phase_pct: float) -> float:
    pct = min(100.0, max(0.0, float(phase_pct)))
    return float(cycle.start_time_s + (pct / 100.0) * cycle.duration_s)


def _paired_cycle(
    reference: DetectedGaitCycle,
    others: Sequence[DetectedGaitCycle],
    half_t: float,
) -> DetectedGaitCycle | None:
    if not others:
        return None
    target = float(reference.start_time_s) - float(half_t)
    return min(others, key=lambda cycle: abs(cycle.start_time_s - target))


def _resample_side_mean(
    times: np.ndarray,
    values: np.ndarray,
    cycles: Sequence[DetectedGaitCycle],
) -> tuple[np.ndarray, int]:
    curves: list[np.ndarray] = []
    for cycle in cycles:
        if cycle.duration_s < 0.25 or cycle.end_time_s <= cycle.start_time_s:
            continue
        mask = (
            (times >= cycle.start_time_s)
            & (times <= cycle.end_time_s)
            & np.isfinite(values)
        )
        if int(mask.sum()) < 4:
            continue
        resampled = resample_cycle_trajectory(
            [float(t) for t in times[mask]],
            [float(v) for v in values[mask]],
            t_start=cycle.start_time_s,
            t_end=cycle.end_time_s,
            n_samples=GAIT_CYCLE_SAMPLE_COUNT,
        )
        if resampled is None or resampled.size != GAIT_CYCLE_SAMPLE_COUNT:
            continue
        curves.append(np.asarray(resampled, dtype=float))
    n = GAIT_CYCLE_SAMPLE_COUNT
    if not curves:
        return np.full(n, np.nan), 0
    return np.nanmean(np.vstack(curves), axis=0), len(curves)


def _curve_rom_peak(
    curve: np.ndarray,
    phase_pct: tuple[float, ...],
) -> tuple[float | None, float | None, float | None]:
    finite = curve[np.isfinite(curve)]
    if finite.size < 8:
        return None, None, None
    rom = float(np.nanmax(finite) - np.nanmin(finite))
    idx = int(np.nanargmax(curve))
    peak = float(curve[idx])
    phase = float(phase_pct[idx]) if idx < len(phase_pct) else None
    return rom, peak, phase


def _joint_comparisons(
    times: np.ndarray,
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    left_cycles: Sequence[DetectedGaitCycle],
    right_cycles: Sequence[DetectedGaitCycle],
) -> list[JointCycleComparison]:
    phase = gait_cycle_percent_grid()
    rows: list[JointCycleComparison] = []
    for name, (left_s, right_s) in series.items():
        l_mean, n_l = _resample_side_mean(times, left_s, left_cycles)
        r_mean, n_r = _resample_side_mean(times, right_s, right_cycles)
        rom_l, peak_l, ph_l = _curve_rom_peak(l_mean, phase)
        rom_r, peak_r, ph_r = _curve_rom_peak(r_mean, phase)
        peak_diff = (
            None if peak_l is None or peak_r is None else float(peak_l - peak_r)
        )
        enough = n_l >= 1 and n_r >= 1 and rom_l is not None and rom_r is not None
        timing_diff = circular_phase_difference_pp(ph_l, ph_r) if enough else None
        rows.append(
            JointCycleComparison(
                name=name,
                units=_UNITS.get(name, "deg"),
                rom_left=rom_l,
                rom_right=rom_r,
                peak_left=peak_l,
                peak_right=peak_r,
                peak_difference=peak_diff if enough else None,
                peak_phase_left_pct=ph_l,
                peak_phase_right_pct=ph_r,
                timing_difference_pct=timing_diff,
                timing_difference_units="percentage points",
                si_percent=robinson_si(rom_l, rom_r) if enough else None,
                asi_percent=absolute_symmetry_index(rom_l, rom_r) if enough else None,
                si_norm=normalized_symmetry_index(rom_l, rom_r) if enough else None,
                n_left_cycles=n_l,
                n_right_cycles=n_r,
                quality="derived" if enough else "insufficient",
                alignment_event=ALIGNMENT_EVENT,
                formula=(
                    f"ROM of the HS–HS mean curve (deg). {ROBINSON_SI_DEFINITION}. "
                    f"{ASI_DEFINITION}. {SI_NORM_DEFINITION}. {TIMING_DEFINITION}."
                ),
            )
        )
    return rows


def failed_gait_asymmetry_report(reason: str) -> GaitAsymmetryMomentReport:
    notes = [
        "Left and right gait cycles are resampled to 0–100% before hip, knee, and ankle comparison.",
        "Descriptions are measured movement differences, not medical diagnoses.",
        str(reason).strip() or "Asymmetry-moment detection failed.",
    ]
    return GaitAsymmetryMomentReport(
        available=False,
        cycle_duration_s=None,
        half_cycle_s=None,
        notes=notes,
        outcome="analysis_failed",
    )


def _scale(values: np.ndarray, floor: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 8:
        return float(floor)
    p10, p90 = np.percentile(finite, [10.0, 90.0])
    span = abs(float(p90) - float(p10))
    return float(max(floor, span))


def _interp(times: np.ndarray, values: np.ndarray, t: float) -> float:
    if times.size == 0:
        return float("nan")
    mask = np.isfinite(values) & np.isfinite(times)
    if int(mask.sum()) < 2:
        return float("nan")
    tt = times[mask]
    vv = values[mask]
    if t < float(tt[0]) or t > float(tt[-1]):
        return float("nan")
    return float(np.interp(t, tt, vv))


def _hold_interp(times: np.ndarray, values: np.ndarray, t: float) -> float:
    """Last-sample hold for piecewise-constant step series."""
    mask = np.isfinite(values) & np.isfinite(times)
    if not np.any(mask):
        return float("nan")
    tt, vv = times[mask], values[mask]
    idx = int(np.searchsorted(tt, t, side="right") - 1)
    if idx < 0:
        return float("nan")
    return float(vv[idx])


def _visibility(snap, *joint_ids: str) -> float:
    vis = (snap.metadata or {}).get("landmark_visibility") or {}
    vals: list[float] = []
    for jid in joint_ids:
        if jid in vis:
            try:
                vals.append(float(vis[jid]))
            except (TypeError, ValueError):
                vals.append(0.0)
        elif jid in snap.joints or (jid == ROOT_JOINT_ID and ROOT_JOINT_ID in snap.joints):
            vals.append(1.0)
    if not vals:
        return 0.0
    return float(min(vals))


def _joint_axis(snap, joint_id: str, axis: str) -> float:
    sample = snap.joints.get(joint_id)
    if sample is None:
        return float("nan")
    val = float(getattr(sample.position, axis))
    return val if math.isfinite(val) else float("nan")


def _dof_deg(snap, dof_id: str, *, storage: str | None = None) -> float:
    from stablewalk.pose.knee_convention import (
        analysis_knee_flexion_deg,
        is_knee_flexion_dof,
    )

    if is_knee_flexion_dof(dof_id) and storage is not None:
        value = analysis_knee_flexion_deg(snap, dof_id, storage=storage)  # type: ignore[arg-type]
        return float(value) if value is not None else float("nan")
    dof = snap.get_dof(dof_id)
    if dof is None or dof.angle_deg is None:
        return float("nan")
    val = float(dof.angle_deg)
    return val if math.isfinite(val) else float("nan")


def _pelvis_tilt_deg(snap) -> float:
    lh = snap.joints.get("left_hip")
    rh = snap.joints.get("right_hip")
    if lh is None or rh is None:
        return float("nan")
    dy = float(lh.position.y) - float(rh.position.y)
    width = abs(float(rh.position.x) - float(lh.position.x))
    return float(math.degrees(math.atan2(dy, max(width, 1e-4))))


def _pelvis_lateral(snap) -> float:
    pelvis = snap.joints.get(ROOT_JOINT_ID)
    if pelvis is not None:
        return float(pelvis.position.x)
    lh = snap.joints.get("left_hip")
    rh = snap.joints.get("right_hip")
    if lh is None or rh is None:
        return float("nan")
    return 0.5 * (float(lh.position.x) + float(rh.position.x))


def _foot_pos(snap, side: str) -> Vec3 | None:
    for jid in (f"{side}_heel", f"{side}_ankle", f"{side}_toe"):
        sample = snap.joints.get(jid)
        if sample is not None:
            return sample.position
    return None


def _step_length_at_hs(recording: GaitMotionRecording, event: GaitEvent) -> float:
    snaps = {int(s.frame_index): s for s in recording.snapshots}
    snap = snaps.get(int(event.frame_index))
    if snap is None and recording.snapshots:
        snap = min(recording.snapshots, key=lambda s: abs(s.time_s - event.time_s))
    if snap is None:
        return float("nan")
    lf = _foot_pos(snap, "left")
    rf = _foot_pos(snap, "right")
    if lf is None or rf is None:
        return float("nan")
    return abs(float(rf.z) - float(lf.z))


def _assign_step_series(
    times: np.ndarray,
    events: Sequence[GaitEvent],
    recording: GaitMotionRecording,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Piecewise-constant left/right step duration and length aligned to ``times``."""
    left_t = np.full(times.shape, np.nan)
    right_t = np.full(times.shape, np.nan)
    left_l = np.full(times.shape, np.nan)
    right_l = np.full(times.shape, np.nan)
    hs = [e for e in events if e.event_type.endswith("_heel_strike")]
    hs.sort(key=lambda e: e.time_s)
    for i, event in enumerate(hs):
        nxt = next((e for e in hs[i + 1 :] if e.side != event.side), None)
        if nxt is None:
            continue
        dur = float(nxt.time_s - event.time_s)
        length = _step_length_at_hs(recording, event)
        mask = (times >= event.time_s) & (times < nxt.time_s)
        if event.side == "left":
            left_t[mask] = dur
            left_l[mask] = length
        else:
            right_t[mask] = dur
            right_l[mask] = length
    return left_t, right_t, left_l, right_l


def _clearance_series(cycles: GaitCycleAnalysisResult, times: np.ndarray, side: str) -> np.ndarray:
    out = np.full(times.shape, np.nan)
    frames = cycles.per_frame or []
    if not frames:
        return out
    src_t = np.asarray([float(fr.time_s) for fr in frames], dtype=float)
    vals = np.full(src_t.shape, np.nan)
    for i, fr in enumerate(frames):
        sample = getattr(fr, side, None)
        dist = getattr(sample, "foot_clearance_m", None) if sample is not None else None
        if dist is None:
            continue
        try:
            vals[i] = float(dist)
        except (TypeError, ValueError):
            continue
    mask = np.isfinite(vals) & np.isfinite(src_t)
    if int(mask.sum()) < 2:
        return out
    out[:] = np.interp(times, src_t[mask], vals[mask], left=np.nan, right=np.nan)
    return out


def _channel_score(diff: float, scale: float) -> float:
    if not math.isfinite(diff) or scale <= 1e-12:
        return float("nan")
    return float(min(100.0, 100.0 * abs(diff) / scale))


def _combine(contribs: list[AsymmetryContribution]) -> float | None:
    num = 0.0
    den = 0.0
    for item in contribs:
        w = _WEIGHTS.get(item.measurement, 1.0)
        if not math.isfinite(item.score):
            continue
        num += w * (item.score / 100.0) ** 2
        den += w
    if den <= 1e-12:
        return None
    return float(min(100.0, 100.0 * math.sqrt(num / den)))


def _explain(
    moment_score: float,
    time_s: float,
    frame: int,
    contribs: list[AsymmetryContribution],
    *,
    phase_pct: float | None,
    duration_s: float,
) -> str:
    ranked = sorted(contribs, key=lambda c: c.score, reverse=True)
    parts: list[str] = []
    for item in ranked[:3]:
        if item.score < 12.0:
            continue
        label = _LABELS.get(item.measurement, item.measurement)
        mag = abs(item.difference)
        if item.left_value is not None and item.right_value is not None and item.units == "deg":
            parts.append(
                f"{label} L {item.left_value:.1f}° / R {item.right_value:.1f}° (Δ {mag:.1f}°)"
            )
        elif item.units == "deg":
            parts.append(f"{label} differed by {mag:.1f}°")
        elif item.units == "s":
            parts.append(f"{label} differed by {mag:.3f} s")
        else:
            parts.append(f"{label} differed by {mag:.3f} {item.units}")
    phase_txt = f" at {phase_pct:.0f}% gait cycle" if phase_pct is not None else ""
    head = (
        f"{_severity(moment_score).capitalize()} measured movement difference "
        f"at {time_s:.2f} s (frame {frame}){phase_txt}. "
        f"Left and right hip, knee, and ankle were compared at the same "
        f"gait-cycle percent after 0–100% normalization "
        f"(held above {SCORE_THRESHOLD:.0f} for {duration_s:.2f} s)."
    )
    if not parts:
        return head
    return head + " Main contributors: " + "; ".join(parts) + "."


def _island_duration_s(island: Sequence[FrameAsymmetrySample], frame_dt: float) -> float:
    """Used-sample duration; low-confidence holes do not count."""
    if not island:
        return 0.0
    dt = max(float(frame_dt), 1e-4)
    if len(island) == 1:
        return dt
    total = 0.0
    cap = dt * 1.5
    for prev, nxt in zip(island, island[1:]):
        gap = float(nxt.time_s - prev.time_s)
        total += min(gap, cap) if gap > 0 else dt
    return float(total)


def _merge_islands(
    samples: list[FrameAsymmetrySample],
    *,
    threshold: float,
    gap_s: float,
    min_duration_s: float,
    frame_dt: float,
) -> list[tuple[FrameAsymmetrySample, float]]:
    used = [s for s in samples if s.used and s.score is not None and s.score >= threshold]
    if not used:
        return []
    islands: list[list[FrameAsymmetrySample]] = []
    current = [used[0]]
    for sample in used[1:]:
        if sample.time_s - current[-1].time_s <= gap_s:
            current.append(sample)
        else:
            islands.append(current)
            current = [sample]
    islands.append(current)
    peaks: list[tuple[FrameAsymmetrySample, float]] = []
    for island in islands:
        duration = _island_duration_s(island, frame_dt)
        if duration < min_duration_s:
            continue
        peak = max(island, key=lambda s: float(s.score or 0.0))
        peaks.append((peak, duration))
    peaks.sort(key=lambda item: float(item[0].score or 0.0), reverse=True)
    return peaks[:TOP_K]


def contralateral_joint_id(joint_id: str) -> str:
    """Left/right counterpart; axial joints (pelvis) map to themselves."""
    if joint_id.startswith("right_"):
        return "left_" + joint_id[6:]
    if joint_id.startswith("left_"):
        return "right_" + joint_id[5:]
    return joint_id


def difference_joint_pairs(moment: AsymmetryMoment) -> list[tuple[str, str]]:
    """Pairs for current vs contralateral joints at matching gait-cycle percent."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for jid in moment.involved_joints:
        if jid.startswith("left_"):
            continue
        key = (jid, contralateral_joint_id(jid))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def format_contribution_line(item: AsymmetryContribution) -> str:
    label = _LABELS.get(item.measurement, item.measurement.replace("_", " "))
    mag = abs(item.difference)
    left = item.left_value
    right = item.right_value
    if item.units == "deg":
        delta = f"Δ {mag:.1f}°"
        if left is not None and right is not None:
            return f"{label}  L {left:.1f}°  R {right:.1f}°  {delta}"
        return f"{label}  {delta}"
    if item.units == "s":
        delta = f"Δ {mag:.3f} s"
        if left is not None and right is not None:
            return f"{label}  L {left:.3f} s  R {right:.3f} s  {delta}"
        return f"{label}  {delta}"
    unit = item.units
    delta = f"Δ {mag:.3f} {unit}"
    if left is not None and right is not None:
        return f"{label}  L {left:.3f}  R {right:.3f}  {delta}"
    return f"{label}  {delta}"


def detect_gait_asymmetry_moments(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult | None,
) -> GaitAsymmetryMomentReport:
    """Compare corresponding L/R hip, knee, and ankle after 0–100% normalization."""
    notes: list[str] = [
        f"{ALIGNMENT_EVENT_LABEL} before hip, knee, and ankle comparison.",
        (
            "Signed SI, absolute ASI, and SI_norm are separate indices; "
            "the 0–100 moment score is a composite and is not SI_norm. "
            "Peak timing Δ is percentage points (circular), not Robinson SI."
        ),
        (
            f"Moments require the per-frame score to stay at or above {SCORE_THRESHOLD:.0f} "
            f"for at least {MIN_DURATION_S:.2f} s. Low-confidence frames are not classified."
        ),
        "Descriptions are measured movement differences, not medical diagnoses.",
    ]
    empty = GaitAsymmetryMomentReport(
        available=False,
        cycle_duration_s=None,
        half_cycle_s=None,
        notes=notes,
        outcome="insufficient_data",
    )
    if recording is None or not recording.snapshots:
        notes.append("No skeleton snapshots.")
        return empty
    if cycles is None or not (cycles.cycles or cycles.events):
        notes.append("Detected gait cycles are required; none were provided.")
        return empty

    left_raw, right_raw = left_right_gait_cycles(cycles)
    fps = float(getattr(cycles, "fps", None) or getattr(recording, "fps", None) or 30.0)
    events = list(cycles.events)
    left_cycles = cycles_aligned_to_heel_strike(left_raw, events, "left", fps=fps)
    right_cycles = cycles_aligned_to_heel_strike(right_raw, events, "right", fps=fps)
    n_left = len(left_cycles)
    n_right = len(right_cycles)
    if n_left < MIN_CYCLES or n_right < MIN_CYCLES:
        notes.append(
            "Insufficient data: both left and right gait cycles are required "
            f"(found {n_left} left, {n_right} right)."
        )
        return GaitAsymmetryMomentReport(
            available=False,
            cycle_duration_s=None,
            half_cycle_s=None,
            notes=notes,
            outcome="insufficient_data",
            n_left_cycles=n_left,
            n_right_cycles=n_right,
        )

    both = list(left_cycles) + list(right_cycles)
    cycle_t = _median([c.duration_s for c in both if c.duration_s > 0.25])
    if cycle_t is None and cycles.metrics.stride_time_s:
        cycle_t = float(cycles.metrics.stride_time_s)
    if cycle_t is None or cycle_t < 0.35:
        notes.append("Gait-cycle duration is too short or missing.")
        return GaitAsymmetryMomentReport(
            available=False,
            cycle_duration_s=cycle_t,
            half_cycle_s=None,
            notes=notes,
            outcome="insufficient_data",
            n_left_cycles=n_left,
            n_right_cycles=n_right,
        )
    half_t = 0.5 * cycle_t
    fps = max(float(recording.fps or cycles.fps or 30.0), 1e-6)
    frame_dt = 1.0 / fps
    snaps = list(recording.snapshots)
    times = np.asarray([float(s.time_s) for s in snaps], dtype=float)
    n = times.size

    from stablewalk.pose.knee_convention import recording_knee_storage

    knee_storage = recording_knee_storage(recording)
    left_hip = np.array([_dof_deg(s, "left_hip_flexion") for s in snaps])
    right_hip = np.array([_dof_deg(s, "right_hip_flexion") for s in snaps])
    left_knee = np.array([_dof_deg(s, "left_knee_flexion", storage=knee_storage) for s in snaps])
    right_knee = np.array([_dof_deg(s, "right_knee_flexion", storage=knee_storage) for s in snaps])
    left_ankle = np.array([_dof_deg(s, "left_ankle_flexion") for s in snaps])
    right_ankle = np.array([_dof_deg(s, "right_ankle_flexion") for s in snaps])
    left_ankle_y = np.array([_joint_axis(s, "left_ankle", "y") for s in snaps])
    right_ankle_y = np.array([_joint_axis(s, "right_ankle", "y") for s in snaps])
    left_heel_y = np.array([_joint_axis(s, "left_heel", "y") for s in snaps])
    right_heel_y = np.array([_joint_axis(s, "right_heel", "y") for s in snaps])
    pelvis_tilt = np.array([_pelvis_tilt_deg(s) for s in snaps])
    pelvis_lat = np.array([_pelvis_lateral(s) for s in snaps])
    left_clear = _clearance_series(cycles, times, "left")
    right_clear = _clearance_series(cycles, times, "right")
    left_step_t, right_step_t, left_step_l, right_step_l = _assign_step_series(
        times, cycles.events, recording
    )

    angle_series = {
        "hip_angle": (left_hip, right_hip),
        "knee_angle": (left_knee, right_knee),
        "ankle_angle": (left_ankle, right_ankle),
    }
    joint_comparisons = _joint_comparisons(times, angle_series, left_cycles, right_cycles)

    channels = {
        "hip_angle": (right_hip, left_hip, False),
        "knee_angle": (right_knee, left_knee, False),
        "ankle_angle": (right_ankle, left_ankle, False),
        "ankle_height": (right_ankle_y, left_ankle_y, False),
        "heel_height": (right_heel_y, left_heel_y, False),
        "foot_ground": (right_clear, left_clear, False),
        "step_timing": (right_step_t, left_step_t, True),
        "step_length": (right_step_l, left_step_l, True),
        "pelvis_tilt": (pelvis_tilt, pelvis_tilt, False),
        "pelvis_lateral": (pelvis_lat, pelvis_lat, False),
    }
    scales = {
        name: _scale(np.concatenate([a, b]), _SCALE_FLOOR[name])
        for name, (a, b, _hold) in channels.items()
    }

    per_frame: list[FrameAsymmetrySample] = []
    for i, snap in enumerate(snaps):
        t = float(times[i])
        ref = _enclosing_cycle(t, right_cycles) or _enclosing_cycle(t, left_cycles)
        using_right = _enclosing_cycle(t, right_cycles) is not None
        contra_pool = left_cycles if using_right else right_cycles
        match_cycle = _paired_cycle(ref, contra_pool, half_t) if ref is not None else None
        phase = _phase_pct(t, ref) if ref is not None else None
        t_match = _time_at_phase(match_cycle, phase) if match_cycle is not None and phase is not None else None
        vis_now_ids = (
            ("right_hip", "right_knee", "right_ankle", "right_heel", ROOT_JOINT_ID)
            if using_right
            else ("left_hip", "left_knee", "left_ankle", "left_heel", ROOT_JOINT_ID)
        )
        vis_match_ids = (
            ("left_hip", "left_knee", "left_ankle", "left_heel", ROOT_JOINT_ID)
            if using_right
            else ("right_hip", "right_knee", "right_ankle", "right_heel", ROOT_JOINT_ID)
        )
        vis_now = _visibility(snap, *vis_now_ids)
        j = (
            int(np.clip(np.argmin(np.abs(times - t_match)), 0, n - 1))
            if t_match is not None
            else i
        )
        vis_match = _visibility(snaps[j], *vis_match_ids)
        conf = float(min(vis_now, vis_match))
        used = (
            ref is not None
            and match_cycle is not None
            and phase is not None
            and t_match is not None
            and conf >= MIN_CONFIDENCE
            and vis_now >= VIS_MIN
            and vis_match >= VIS_MIN
            and float(times[0]) <= t_match <= float(times[-1])
        )
        contribs: list[AsymmetryContribution] = []
        if used:
            for name, (right_s, left_s, hold) in channels.items():
                getter = _hold_interp if hold else _interp
                r = getter(times, right_s, t if using_right else t_match)
                l = getter(times, left_s, t_match if using_right else t)
                if name in _ODD_SYMMETRY:
                    l = -l
                diff = r - l
                score = _channel_score(diff, scales[name])
                if not math.isfinite(score):
                    continue
                contribs.append(
                    AsymmetryContribution(
                        measurement=name,
                        joints=_JOINTS[name],
                        difference=float(diff),
                        units=_UNITS[name],
                        score=score,
                        left_value=float(l) if math.isfinite(l) else None,
                        right_value=float(r) if math.isfinite(r) else None,
                        phase_pct=phase,
                    )
                )
        combined = _combine(contribs) if contribs else None
        per_frame.append(
            FrameAsymmetrySample(
                frame_index=int(snap.frame_index),
                time_s=t,
                match_time_s=t_match if used else None,
                score=combined,
                confidence=conf,
                used=used and combined is not None,
                contributions=contribs,
                phase_pct=phase if used else None,
                match_phase_pct=phase if used else None,
            )
        )

    peaks = _merge_islands(
        per_frame,
        threshold=SCORE_THRESHOLD,
        gap_s=MERGE_GAP_S,
        min_duration_s=MIN_DURATION_S,
        frame_dt=frame_dt,
    )
    moments: list[AsymmetryMoment] = []
    for peak, duration in peaks:
        ranked = sorted(peak.contributions, key=lambda c: c.score, reverse=True)
        keep = [c for c in ranked if c.score >= max(12.0, 0.35 * float(peak.score or 0.0))]
        if not keep:
            keep = ranked[:2]
        joints: list[str] = []
        for item in keep:
            for jid in item.joints:
                if jid not in joints:
                    joints.append(jid)
        score = float(peak.score or 0.0)
        moments.append(
            AsymmetryMoment(
                frame_index=peak.frame_index,
                time_s=peak.time_s,
                match_time_s=float(peak.match_time_s or peak.time_s),
                involved_joints=joints,
                contributing_measurements=keep,
                severity=_severity(score),
                score=score,
                confidence=peak.confidence,
                explanation=_explain(
                    score,
                    peak.time_s,
                    peak.frame_index,
                    keep,
                    phase_pct=peak.phase_pct,
                    duration_s=duration,
                ),
                duration_s=duration,
                phase_pct=peak.phase_pct,
            )
        )

    n_used = sum(1 for s in per_frame if s.used)
    available = n_used >= max(8, int(0.5 * fps))
    if not available:
        notes.append(
            "Too few confident in-cycle frames for phase-normalized left–right comparison."
        )
        moments = []
        outcome: AsymmetryOutcome = "insufficient_data"
    elif moments:
        outcome = "moments"
    else:
        outcome = "symmetric_gait"
        notes.append(
            "Symmetric gait: no left–right difference exceeded the documented "
            f"score threshold ({SCORE_THRESHOLD:.0f}) for {MIN_DURATION_S:.2f} s."
        )
    return GaitAsymmetryMomentReport(
        available=available,
        cycle_duration_s=cycle_t,
        half_cycle_s=half_t,
        per_frame=per_frame,
        moments=moments,
        notes=notes,
        outcome=outcome,
        joint_comparisons=joint_comparisons,
        n_left_cycles=n_left,
        n_right_cycles=n_right,
    )


__all__ = [
    "AsymmetryContribution",
    "AsymmetryMoment",
    "AsymmetryOutcome",
    "FrameAsymmetrySample",
    "GaitAsymmetryMomentReport",
    "JointCycleComparison",
    "MIN_DURATION_S",
    "SCORE_THRESHOLD",
    "contralateral_joint_id",
    "detect_gait_asymmetry_moments",
    "difference_joint_pairs",
    "failed_gait_asymmetry_report",
    "format_contribution_line",
]
