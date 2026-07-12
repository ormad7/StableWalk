"""
Advanced gait stability metrics from pose, foot contact, and estimated GRF.

Primary API: ``GaitMetrics`` — symmetry, cadence, stride, variability, CoM stability.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from stablewalk.pose.events import GaitEvent, analyze_gait_sequence
from stablewalk.io.pose_loader import detected_frame_indices
from stablewalk.models.pose_data import PoseFrame, PoseSequence
from stablewalk.pose.skeleton_3d import sequence_skeleton_scale

# Why each metric matters for stability (for reports and UI)
METRIC_WHY: dict[str, str] = {
    "symmetry_score": (
        "Overall left/right balance (0–1). Asymmetric step times or peak forces often "
        "indicate limping, injury compensation, or fall risk. Stable walkers show "
        "similar loading and timing on both legs."
    ),
    "symmetry_step_duration": (
        "Ratio of mean step intervals left vs right (1 = equal). Unequal timing "
        "suggests irregular rhythm or favoring one leg."
    ),
    "symmetry_force_magnitude": (
        "Ratio of peak vertical GRF left vs right (1 = equal). Large imbalance "
        "means one leg carries much more body weight per step."
    ),
    "cadence_hz": (
        "Steps per second. Too slow or too fast vs typical adult walking (~1.0–2.0 Hz) "
        "can reflect cautious gait, hurrying, or poor pose detection."
    ),
    "cadence_steps_per_min": "Steps per minute (clinical unit). Often 100–120 in healthy adults.",
    "stride_length_normalized": (
        "Forward hip travel per step (image-normalized). Very short strides can mean "
        "shuffling; inconsistent stride length reduces stability."
    ),
    "stride_length_m": "Approximate stride in meters from body scale × horizontal hip displacement.",
    "step_timing_variability": (
        "Variance of inter-step intervals (seconds²). Higher values = less regular "
        "footfall rhythm, linked to trips and falls in older adults."
    ),
    "step_timing_cv": (
        "Coefficient of variation of step intervals (std/mean). Lower is more "
        "consistent — a common stability indicator in gait research."
    ),
    "com_lateral_std": (
        "Side-to-side hip (CoM proxy) spread. Excessive lateral sway moves the mass "
        "outside the base of support and increases fall risk."
    ),
    "com_lateral_range": "Peak-to-peak horizontal hip displacement during the walk.",
    "com_path_length": "Total hip path length in the image plane — erratic paths suggest poor control.",
}


@dataclass
class ContactTimingStats:
    """Stance/swing timing per leg (fractions of detected timeline)."""

    left_stance_fraction: float
    right_stance_fraction: float
    double_support_fraction: float
    left_mean_stance_s: float | None
    right_mean_stance_s: float | None
    stance_symmetry_ratio: float


@dataclass
class GaitMetricsResult:
    """Advanced stability-oriented gait metrics (export via ``to_dict()``)."""

    symmetry_score: float
    symmetry_step_duration: float | None
    symmetry_force_magnitude: float | None
    cadence_hz: float | None
    cadence_steps_per_min: float | None
    stride_length_normalized: float | None
    stride_length_m: float | None
    step_timing_variability: float | None
    step_timing_cv: float | None
    com_lateral_std: float
    com_lateral_range: float
    com_path_length: float
    step_count: int
    contact: ContactTimingStats
    notes: list[str] = field(default_factory=list)
    metric_why: dict[str, str] = field(default_factory=lambda: dict(METRIC_WHY))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["contact"] = asdict(self.contact)
        out["metric_why"] = dict(self.metric_why)
        return out


# Backward-compatible report (subset + alias fields)
@dataclass
class GaitMetricsReport:
    cadence_steps_per_min: float | None
    stride_period_s: float | None
    stride_length_normalized: float | None
    stride_length_m: float | None
    step_count: int
    heel_strike_count: int
    contact: ContactTimingStats
    step_time_symmetry: float | None
    notes: list[str] = field(default_factory=list)


class GaitMetrics:
    """
    Compute stability-related gait metrics from pose, contact, and GRF.

    Inputs
    ------
    - **Pose** — hip positions for stride (horizontal) and CoM lateral stability.
    - **Contact** — foot contact times for step duration and cadence (optional;
      computed via ``ContactDetector`` if missing).
    - **GRF** — estimated left/right vertical forces for magnitude symmetry (optional;
      computed via ``GRFAnalyzer`` if missing).

    Metrics
    -------
    1. **Symmetry** — step-duration balance and peak force balance (left vs right).
    2. **Cadence** — steps per second (and steps/min).
    3. **Stride length** — mean horizontal hip displacement between same-side steps.
    4. **Variability** — variance and CV of step timing.
    5. **CoM stability** — hip lateral std, range, and path length.

    Example::

        metrics = GaitMetrics().compute(sequence, contact=contact_result, grf=grf_series)
        print(metrics.to_dict())
    """

    def compute(
        self,
        sequence: PoseSequence,
        *,
        contact=None,
        grf=None,
    ) -> GaitMetricsResult:
        """Run full metric pipeline; returns ``GaitMetricsResult`` with ``to_dict()``."""
        notes: list[str] = []
        fps = max(sequence.fps, 1e-6)

        if contact is None:
            from stablewalk.pose.contact import ContactDetector

            contact = ContactDetector().detect(sequence)

        if grf is None:
            from stablewalk.analysis.forces import GRFAnalyzer

            grf = GRFAnalyzer().analyze(sequence)

        events, _ = analyze_gait_sequence(sequence.frames, fps=fps)
        hs = [e for e in events if e.event_type == "heel_strike"]
        frames_map = _frame_by_index(sequence)
        indices = detected_frame_indices(sequence)

        detected_kps = [f.keypoints for f in sequence.frames if f.detected and f.keypoints]
        scale = sequence_skeleton_scale(detected_kps) if detected_kps else None

        # --- Cadence ---
        duration_s = 0.0
        if len(indices) >= 2:
            t0 = sequence.frames[indices[0]].timestamp_s
            t1 = sequence.frames[indices[-1]].timestamp_s
            duration_s = max(t1 - t0, (indices[-1] - indices[0]) / fps)

        step_times = _all_step_times(contact, hs, frames_map, fps)
        step_count = len(step_times)
        cadence_hz = None
        cadence_spm = None
        if duration_s > 0 and step_count >= 1:
            cadence_hz = step_count / duration_s
            cadence_spm = cadence_hz * 60.0
        elif step_count < 2:
            notes.append("Fewer than two steps detected; cadence may be unreliable.")

        # --- Stride (horizontal hip movement) ---
        norm_lens, meter_lens = _stride_lengths_horizontal(hs, contact, frames_map, scale_m=scale)
        stride_norm = sum(norm_lens) / len(norm_lens) if norm_lens else None
        stride_m = sum(meter_lens) / len(meter_lens) if meter_lens else None
        if stride_norm is None:
            notes.append("Stride: need visible pelvis across consecutive same-side steps.")

        # --- Step timing variability ---
        step_var, step_cv = _step_timing_stats(step_times)

        # --- Symmetry: step duration ---
        sym_step = _symmetry_step_duration(contact, hs, fps)

        # --- Symmetry: force magnitude ---
        sym_force = _symmetry_force_magnitude(grf)

        symmetry_score = _combined_symmetry(sym_step, sym_force, _contact_timing(sequence))

        # --- CoM / hip stability ---
        com_lat_std, com_lat_range, com_path = _hip_stability(sequence, indices)

        contact_stats = _contact_timing(sequence, contact)

        return GaitMetricsResult(
            symmetry_score=symmetry_score,
            symmetry_step_duration=sym_step,
            symmetry_force_magnitude=sym_force,
            cadence_hz=cadence_hz,
            cadence_steps_per_min=cadence_spm,
            stride_length_normalized=stride_norm,
            stride_length_m=stride_m,
            step_timing_variability=step_var,
            step_timing_cv=step_cv,
            com_lateral_std=com_lat_std,
            com_lateral_range=com_lat_range,
            com_path_length=com_path,
            step_count=step_count,
            contact=contact_stats,
            notes=notes,
        )

    def to_legacy_report(self, result: GaitMetricsResult) -> GaitMetricsReport:
        """Map to ``GaitMetricsReport`` for older callers."""
        period = None
        if result.cadence_hz and result.cadence_hz > 0:
            period = 1.0 / result.cadence_hz
        return GaitMetricsReport(
            cadence_steps_per_min=result.cadence_steps_per_min,
            stride_period_s=period,
            stride_length_normalized=result.stride_length_normalized,
            stride_length_m=result.stride_length_m,
            step_count=result.step_count,
            heel_strike_count=result.step_count,
            contact=result.contact,
            step_time_symmetry=result.symmetry_step_duration,
            notes=result.notes,
        )


def _pelvis_xy(frame: PoseFrame) -> tuple[float, float] | None:
    kp = {k.name: k for k in frame.keypoints}
    if "mid_hip" in kp and kp["mid_hip"].visibility >= 0.3:
        p = kp["mid_hip"]
        return (p.x, p.y)
    lh, rh = kp.get("left_hip"), kp.get("right_hip")
    if lh and rh and lh.visibility >= 0.3 and rh.visibility >= 0.3:
        return ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)
    return None


def _frame_by_index(sequence: PoseSequence) -> dict[int, PoseFrame]:
    return {f.frame_index: f for f in sequence.frames}


def _intervals_from_frame_list(frame_ids: list[int], time_by_frame: dict[int, float], fps: float) -> list[float]:
    if len(frame_ids) < 2:
        return []
    sorted_ids = sorted(frame_ids)
    out: list[float] = []
    for a, b in zip(sorted_ids, sorted_ids[1:]):
        ta = time_by_frame.get(a, a / fps)
        tb = time_by_frame.get(b, b / fps)
        out.append(tb - ta)
    return out


def _side_step_intervals(contact, side: str, hs: list[GaitEvent], fps: float) -> list[float]:
    time_by = contact.frame_times_s
    ids = contact.left_foot_contacts if side == "left" else contact.right_foot_contacts
    intervals = _intervals_from_frame_list(ids, time_by, fps)
    if intervals:
        return intervals
    side_hs = sorted([e.frame_index for e in hs if e.side == side])
    return _intervals_from_frame_list(side_hs, time_by, fps)


def _all_step_times(contact, hs: list[GaitEvent], frames_map: dict[int, PoseFrame], fps: float) -> list[float]:
    """Merged sorted step event times (seconds) for variability / cadence."""
    times: list[float] = []
    for fi in contact.left_foot_contacts + contact.right_foot_contacts:
        t = contact.frame_times_s.get(fi)
        if t is not None:
            times.append(t)
    if not times and hs:
        for e in hs:
            f = frames_map.get(e.frame_index)
            if f:
                times.append(f.timestamp_s)
    return sorted(times)


def _symmetry_step_duration(contact, hs: list[GaitEvent], fps: float) -> float | None:
    left = _side_step_intervals(contact, "left", hs, fps)
    right = _side_step_intervals(contact, "right", hs, fps)
    if not left or not right:
        return None
    ml = sum(left) / len(left)
    mr = sum(right) / len(right)
    if max(ml, mr) < 1e-8:
        return None
    return min(ml, mr) / max(ml, mr)


def _symmetry_force_magnitude(grf) -> float | None:
    if grf is None or len(grf.left_force_n) == 0:
        return None
    peak_l = float(np.max(grf.left_force_bw[grf.left_contact])) if grf.left_contact.any() else 0.0
    peak_r = float(np.max(grf.right_force_bw[grf.right_contact])) if grf.right_contact.any() else 0.0
    if max(peak_l, peak_r) < 1e-6:
        return None
    return min(peak_l, peak_r) / max(peak_l, peak_r)


def _combined_symmetry(
    sym_step: float | None,
    sym_force: float | None,
    contact: ContactTimingStats,
) -> float:
    parts: list[float] = [contact.stance_symmetry_ratio]
    if sym_step is not None:
        parts.append(sym_step)
    if sym_force is not None:
        parts.append(sym_force)
    return float(sum(parts) / len(parts)) if parts else 0.0


def _step_timing_stats(step_times: list[float]) -> tuple[float | None, float | None]:
    if len(step_times) < 3:
        return None, None
    intervals = [step_times[i + 1] - step_times[i] for i in range(len(step_times) - 1)]
    if not intervals:
        return None, None
    mean = sum(intervals) / len(intervals)
    var = sum((x - mean) ** 2 for x in intervals) / len(intervals)
    cv = math.sqrt(var) / mean if mean > 1e-8 else None
    return var, cv


def _hip_stability(sequence: PoseSequence, indices: list[int]) -> tuple[float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for i in indices:
        p = _pelvis_xy(sequence.frames[i])
        if p:
            xs.append(p[0])
            ys.append(p[1])
    if len(xs) < 2:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / len(xs)
    lat_std = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / len(xs))
    lat_range = max(xs) - min(xs)
    path = sum(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(len(xs) - 1))
    return lat_std, lat_range, path


def _stride_lengths_horizontal(
    hs: list[GaitEvent],
    contact,
    frames: dict[int, PoseFrame],
    *,
    scale_m: float | None,
) -> tuple[list[float], list[float]]:
    """Horizontal (x) pelvis displacement between same-side steps."""
    norm_lengths: list[float] = []
    meter_lengths: list[float] = []

    def add_stride(fa: PoseFrame, fb: PoseFrame) -> None:
        pa, pb = _pelvis_xy(fa), _pelvis_xy(fb)
        if pa is None or pb is None:
            return
        dx = abs(pb[0] - pa[0])
        norm_lengths.append(dx)
        if scale_m:
            meter_lengths.append(dx * scale_m)

    by_side: dict[str, list[int]] = {"left": [], "right": []}
    for side in ("left", "right"):
        ids = contact.left_foot_contacts if side == "left" else contact.right_foot_contacts
        by_side[side] = sorted(ids)
    for side, ids in by_side.items():
        for a, b in zip(ids, ids[1:]):
            fa, fb = frames.get(a), frames.get(b)
            if fa and fb:
                add_stride(fa, fb)

    if not norm_lengths:
        by_hs: dict[str, list[GaitEvent]] = {"left": [], "right": []}
        for e in hs:
            by_hs[e.side].append(e)
        for evs in by_hs.values():
            evs.sort(key=lambda e: e.frame_index)
            for a, b in zip(evs, evs[1:]):
                fa, fb = frames.get(a.frame_index), frames.get(b.frame_index)
                if fa and fb:
                    add_stride(fa, fb)

    return norm_lengths, meter_lengths


def _contact_timing(sequence: PoseSequence, contact=None) -> ContactTimingStats:
    indices = detected_frame_indices(sequence)
    if not indices:
        return ContactTimingStats(0, 0, 0, None, None, 0)

    left_stance = right_stance = double_support = 0
    for idx in indices:
        f = sequence.frames[idx]
        foot_contact = getattr(f, "foot_contact", None)
        if foot_contact:
            pl = foot_contact.get("left", False)
            pr = foot_contact.get("right", False)
        else:
            pl = f.gait_phase.get("left") == "stance"
            pr = f.gait_phase.get("right") == "stance"
        if pl:
            left_stance += 1
        if pr:
            right_stance += 1
        if pl and pr:
            double_support += 1

    n = len(indices)
    fps = max(sequence.fps, 1e-6)
    ls, rs = left_stance / n, right_stance / n
    ds = double_support / n
    sym = min(ls, rs) / max(ls, rs) if max(ls, rs) > 1e-6 else 0.0

    left_stance_s = contact.left_stance.stance_s if contact else left_stance / fps
    right_stance_s = contact.right_stance.stance_s if contact else right_stance / fps

    return ContactTimingStats(
        left_stance_fraction=ls,
        right_stance_fraction=rs,
        double_support_fraction=ds,
        left_mean_stance_s=left_stance_s,
        right_mean_stance_s=right_stance_s,
        stance_symmetry_ratio=sym,
    )


def compute_gait_metrics(sequence: PoseSequence) -> GaitMetricsReport:
    """Backward-compatible: ``GaitMetrics().compute()`` → legacy report."""
    result = GaitMetrics().compute(sequence)
    return GaitMetrics().to_legacy_report(result)
