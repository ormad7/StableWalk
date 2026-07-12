"""
Transparent, explainable walking-stability analysis.
====================================================

Goal
----
Decide whether a walking pattern is **Stable**, **Moderate**, or **Unstable**,
using *simple and transparent* biomechanical metrics derived from the MediaPipe
pose data (joint angles + landmark positions). This is intentionally **not** a
black box: every sub-score is a plain formula with a documented threshold, and
every deduction is reported in words (e.g. "Right knee range of motion is
significantly different from the left knee").

Metric groups
-------------
1. **Gait symmetry** — left vs right knee/hip/ankle angle agreement over time.
2. **Center of mass (CoM)** — simplified body center from hips + shoulders;
   measures excessive side-to-side sway.
3. **Step consistency** — step timing regularity from ankle/heel/toe motion,
   and left/right step balance.
4. **Joint range of motion (ROM)** — how much each knee/hip/ankle moves, and
   whether the two sides are consistent.
5. **Stability score** — a weighted average of six group scores, mapped to
   a 0–100 scale and a Stable / Moderate / Unstable label.

All angles are in degrees; positions are MediaPipe image-normalized coordinates
(x right, y down). Thresholds are tuned for noisy monocular video and are
explained in :data:`SCORING_NOTES`; they are educational, not clinical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from stablewalk.models.pose_data import PoseFrame, PoseSequence
from stablewalk.pose.gait_step_detection import (
    CADENCE_MAX_HZ,
    CADENCE_MIN_HZ,
    GaitStepDetectionResult,
    detect_gait_steps,
)

# ---------------------------------------------------------------------------
# Classification thresholds (final 0–100 score)
# ---------------------------------------------------------------------------
STABLE_MIN = 70.0
MODERATE_MIN = 45.0  # below this => Unstable

# Weights for combining metric groups (must describe intent clearly).
METRIC_WEIGHTS: dict[str, float] = {
    "symmetry": 0.22,           # left/right joint-angle + step-timing agreement
    "step_consistency": 0.24,   # regular, balanced steps (same-side + stride timing)
    "body_stability": 0.14,     # pelvis + torso control (not low-motion bias)
    "range_of_motion": 0.22,    # consistent, adequate joint ROM between sides
    "trajectory_smoothness": 0.11,  # ROM-normalized jerk (controlled vs jerky)
    "pose_quality": 0.07,       # landmark visibility + usable frame coverage
}

# Per-metric thresholds (documented; used in transparent formulas).
SYMMETRY_DIFF_FULL_DEG = 0.0    # 0° L/R difference  -> sub-score 100
SYMMETRY_DIFF_ZERO_DEG = 25.0   # >=25° L/R difference -> sub-score 0 (knee/hip)
SYMMETRY_ANKLE_DIFF_ZERO_DEG = 14.0  # tighter for ankles (foot-drop compensation)
SYMMETRY_FLAG_DEG = 12.0        # report a finding above this average difference

ROM_DIFF_ZERO_RATIO = 0.60      # 60% L/R ROM difference -> sub-score 0
ROM_FLAG_RATIO = 0.30           # report a finding above this ROM difference
ROM_STIFF_DEG = 5.0             # a moving joint with <5° ROM looks "stiff"

BODY_SWAY_ZERO_RATIO = 0.18     # lateral sway = 18% of shoulder width -> 0
BODY_SWAY_FLAG_RATIO = 0.10     # report excessive sway above this

TORSO_SWAY_ZERO_DEG = 12.0      # shoulder-line tilt std >= 12° -> sub-score 0
TORSO_SWAY_FLAG_DEG = 5.0

STEP_CV_ZERO = 0.35             # same-side step-interval CV of 0.35 -> sub-score 0
STEP_CV_FLAG = 0.18             # report irregular timing above this CV
STEP_BALANCE_FLAG = 0.70        # left/right step-count ratio below this -> flag
STEP_MIN_INTERVAL_S = 0.38      # refractory period between same-foot contacts (~1.3 Hz max)
STEP_EXPECTED_RATE_HZ = 1.75    # legacy reference cadence (both legs); coverage uses cadence band
STEP_MIN_PER_SIDE = 3           # need this many contacts per foot for reliable timing CV (full ~3 s clip)
STEP_REFERENCE_DURATION_S = 3.0  # scale STEP_MIN_PER_SIDE down on shorter clips
STEP_ANKLE_ASYM_PENALTY_DEG = 8.0  # ankle L/R angle diff above this reduces step score

TRAJECTORY_JERK_ZERO = 1.25     # ROM-normalized jerk (|Δ²θ|/ROM) at which sub-score -> 0
TRAJECTORY_JERK_FLAG = 0.65
JERK_ROM_FLOOR_DEG = 6.0         # minimum ROM divisor when normalizing jerk

WALK_MIN_MEAN_ROM_DEG = 8.0     # mean joint ROM below this suggests restricted gait
POSE_VISIBILITY_FULL = 0.92       # mean foot visibility -> pose sub-score 100

MIN_FRAMES = 8                  # need at least this many detected frames
INTERP_MAX_GAP = 4              # interpolate foot-y gaps up to this many frames

SCORING_NOTES = (
    "Score starts from six transparent group scores (each 0-100): gait symmetry, "
    "step consistency, body stability (pelvis + torso control), joint range of motion, "
    "trajectory smoothness, and pose quality. The final score is their weighted average "
    "(symmetry 22%, steps 24%, body stability 14%, ROM 22%, smoothness 11%, pose 7%). "
    "Foot contacts use body-height-scaled heel/ankle signals with temporal smoothing, "
    "minimum step interval, peak prominence, swing-amplitude hysteresis, and cadence-band "
    "coverage (over- and under-detection outside plausible cadence both reduce the step score). "
    "Step timing uses same-side and merged stride CV. Trajectory smoothness uses jerk divided by joint ROM "
    "so slow restricted motion is not mistaken for controlled stability. Lateral sway is "
    "detrended and normalized by shoulder width. Stable >= 70, Moderate 45-69, Unstable < 45."
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class MetricResult:
    """One transparent metric group: a 0-100 score plus a plain explanation."""

    key: str
    name: str
    score: float | None          # None when not computable (too little data)
    weight: float
    summary: str                 # one-line, human readable
    findings: list[str] = field(default_factory=list)  # specific problems
    values: dict[str, Any] = field(default_factory=dict)  # raw numbers

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "score": None if self.score is None else round(self.score, 1),
            "weight": self.weight,
            "summary": self.summary,
            "findings": list(self.findings),
            "values": self.values,
        }


@dataclass
class StabilityResult:
    """Full explainable stability assessment."""

    score: float
    classification: str          # Stable | Moderate | Unstable | Insufficient data
    metrics: list[MetricResult]
    primary_issue: str | None
    explanation: str
    frame_count: int
    scoring_notes: str = SCORING_NOTES

    @property
    def is_stable(self) -> bool:
        return self.classification == "Stable"

    def metric(self, key: str) -> MetricResult | None:
        return next((m for m in self.metrics if m.key == key), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "classification": self.classification,
            "primary_issue": self.primary_issue,
            "frame_count": self.frame_count,
            "metrics": [m.to_dict() for m in self.metrics],
            "explanation": self.explanation,
            "scoring_notes": self.scoring_notes,
        }


# ---------------------------------------------------------------------------
# Small numeric helpers (kept simple and transparent)
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _linear_score(value: float, full_at: float, zero_at: float) -> float:
    """Map ``value`` to 0-100: ``full_at`` -> 100, ``zero_at`` -> 0 (linear)."""
    if zero_at == full_at:
        return 100.0
    frac = (value - full_at) / (zero_at - full_at)
    return _clamp(100.0 * (1.0 - frac))


def _robust_range(values: Sequence[float]) -> float:
    """Range of motion using 5th–95th percentile (ignores single-frame outliers)."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size < 3:
        return 0.0
    return float(np.percentile(arr, 95) - np.percentile(arr, 5))


def _mean_abs_diff(left: Sequence[float | None], right: Sequence[float | None]) -> float | None:
    """Average |left - right| over frames where both sides are present."""
    diffs = [
        abs(a - b)
        for a, b in zip(left, right)
        if a is not None and b is not None
    ]
    return float(np.mean(diffs)) if diffs else None


def _find_peaks(values: np.ndarray, min_distance: int, min_prominence: float) -> list[int]:
    """
    Minimal, transparent local-maxima detector (no SciPy dependency).

    A point is a step "contact" peak if it is a local maximum, rises at least
    ``min_prominence`` above the surrounding window, and is at least
    ``min_distance`` frames from the previous accepted peak.
    """
    peaks: list[int] = []
    n = len(values)
    last = -min_distance
    for i in range(1, n - 1):
        if values[i] >= values[i - 1] and values[i] > values[i + 1]:
            lo = max(0, i - min_distance)
            hi = min(n, i + min_distance + 1)
            local_min = float(np.min(values[lo:hi]))
            if values[i] - local_min >= min_prominence and (i - last) >= min_distance:
                peaks.append(i)
                last = i
    return peaks


def _cv(values: Sequence[float]) -> float | None:
    """Coefficient of variation (std / mean). Lower = more regular."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return None
    mean = float(np.mean(arr))
    if abs(mean) < 1e-9:
        return None
    return float(np.std(arr) / mean)


def _interpolate_short_gaps(values: list[float | None], max_gap: int = INTERP_MAX_GAP) -> np.ndarray:
    """Fill short missing runs by linear interpolation; longer gaps stay NaN."""
    arr = np.full(len(values), np.nan, dtype=float)
    for i, v in enumerate(values):
        if v is not None:
            arr[i] = float(v)
    n = len(arr)
    i = 0
    while i < n:
        if not np.isnan(arr[i]):
            i += 1
            continue
        j = i
        while j < n and np.isnan(arr[j]):
            j += 1
        gap = j - i
        if gap <= max_gap and i > 0 and j < n:
            left, right = arr[i - 1], arr[j]
            for k in range(gap):
                t = (k + 1) / (gap + 1)
                arr[i + k] = left + t * (right - left)
        i = max(j, i + 1)
    return arr


def _smooth_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Moving average that ignores NaN (does not compress the timeline)."""
    window = max(1, window)
    if window == 1:
        return arr.copy()
    out = np.full_like(arr, np.nan, dtype=float)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        chunk = arr[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        if valid.size:
            out[i] = float(np.mean(valid))
    return out


def _detrend_linear(arr: np.ndarray) -> np.ndarray:
    """Remove linear trend so forward walking / camera pan is not scored as sway."""
    idx = np.arange(len(arr), dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return arr.copy()
    coef = np.polyfit(idx[valid], arr[valid], 1)
    return arr - np.polyval(coef, idx)


def _mean_jerk(values: Sequence[float | None]) -> float | None:
    """Mean absolute second difference of a joint-angle series (lower = smoother)."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size < 5:
        return None
    d2 = np.diff(arr, n=2)
    return float(np.mean(np.abs(d2)))


def _pelvis_mid_x(frame: PoseFrame) -> float | None:
    kp = _keypoint_map(frame)
    lh, rh = kp.get("left_hip"), kp.get("right_hip")
    if lh and rh and lh.visibility >= 0.3 and rh.visibility >= 0.3:
        return (lh.x + rh.x) / 2.0
    mid = kp.get("mid_hip")
    if mid is not None and mid.visibility >= 0.3:
        return float(mid.x)
    return None


def _shoulder_mid_x(frame: PoseFrame) -> float | None:
    kp = _keypoint_map(frame)
    ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
    if ls and rs and ls.visibility >= 0.3 and rs.visibility >= 0.3:
        return (ls.x + rs.x) / 2.0
    return None


def _shoulder_pelvis_offset(frame: PoseFrame) -> float | None:
    """Lateral offset of shoulder midpoint relative to pelvis (body-relative trunk lean)."""
    px = _pelvis_mid_x(frame)
    sx = _shoulder_mid_x(frame)
    if px is None or sx is None:
        return None
    return sx - px


# ---------------------------------------------------------------------------
# Frame-level data extraction
# ---------------------------------------------------------------------------
@dataclass
class _Series:
    """Per-frame arrays pulled from the pose sequence (already filtered)."""

    knee_l: list[float | None]
    knee_r: list[float | None]
    hip_l: list[float | None]
    hip_r: list[float | None]
    ankle_l: list[float | None]
    ankle_r: list[float | None]
    com_x: list[float]
    com_y: list[float]
    pelvis_x: list[float | None]
    shoulder_offset: list[float | None]
    ankle_l_y: list[float | None]
    ankle_r_y: list[float | None]
    body_width: float
    body_height: float
    mean_foot_visibility: float
    fps: float
    n: int
    total_frames: int
    valid_frame_pct: float


def _keypoint_map(frame: PoseFrame) -> dict[str, Any]:
    return {kp.name: kp for kp in frame.keypoints}


def _com_xy(frame: PoseFrame) -> tuple[float, float] | None:
    """
    Simplified body center: average of available hip + shoulder landmarks.

    Using hips and shoulders (the heaviest segments: pelvis + trunk) gives a
    stable, transparent CoM proxy without a full mass model.
    """
    kp = _keypoint_map(frame)
    pts: list[tuple[float, float]] = []
    for name in ("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
        k = kp.get(name)
        if k is not None and k.visibility >= 0.3:
            pts.append((k.x, k.y))
    if not pts:
        mid = kp.get("mid_hip")
        if mid is not None and mid.visibility >= 0.3:
            pts.append((mid.x, mid.y))
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _body_width(frames: list[PoseFrame]) -> float:
    """Median shoulder width (fallback hip width) — lateral scale for CoM sway."""
    widths: list[float] = []
    for f in frames:
        kp = _keypoint_map(f)
        ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
        if ls and rs and ls.visibility >= 0.3 and rs.visibility >= 0.3:
            widths.append(abs(ls.x - rs.x))
            continue
        lh, rh = kp.get("left_hip"), kp.get("right_hip")
        if lh and rh and lh.visibility >= 0.3 and rh.visibility >= 0.3:
            widths.append(abs(lh.x - rh.x))
    return float(np.median(widths)) if widths else 0.2


def _body_height(frames: list[PoseFrame]) -> float:
    """Median vertical span of visible keypoints — scale for foot clearance."""
    heights: list[float] = []
    for f in frames:
        ys = [kp.y for kp in f.keypoints if kp.visibility >= 0.3]
        if len(ys) >= 4:
            heights.append(max(ys) - min(ys))
    return float(np.median(heights)) if heights else 0.5


def _mean_foot_visibility(frames: list[PoseFrame]) -> float:
    """Average visibility of ankle/heel/foot-index landmarks."""
    names = (
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    )
    vis: list[float] = []
    for f in frames:
        kp = _keypoint_map(f)
        for name in names:
            k = kp.get(name)
            if k is not None:
                vis.append(float(k.visibility))
    return float(np.mean(vis)) if vis else 0.5


def _scale_foot_signal(values: list[float | None], scale: float) -> list[float | None]:
    """Express foot vertical positions in body-scale units (width or height)."""
    denom = max(scale, 1e-6)
    return [None if v is None else float(v) / denom for v in values]


def _ankle_y(frame: PoseFrame, side: str) -> float | None:
    """Vertical position of the foot (prefer heel, fall back to ankle/toe)."""
    kp = _keypoint_map(frame)
    for name in (f"{side}_heel", f"{side}_ankle", f"{side}_foot_index"):
        k = kp.get(name)
        if k is not None and k.visibility >= 0.3:
            return float(k.y)
    return None


def _extract_series(sequence: PoseSequence) -> _Series:
    frames = [f for f in sequence.frames if f.detected and f.joint_angles]
    knee_l, knee_r, hip_l, hip_r, ankle_l, ankle_r = ([] for _ in range(6))
    com_x, com_y, pelvis_x, shoulder_offset, ankle_l_y, ankle_r_y = [], [], [], [], [], []

    for f in frames:
        a = f.joint_angles
        knee_l.append(a.left_knee)
        knee_r.append(a.right_knee)
        hip_l.append(a.left_hip)
        hip_r.append(a.right_hip)
        ankle_l.append(a.left_ankle if a.left_ankle is not None else a.left_ankle_flexion)
        ankle_r.append(a.right_ankle if a.right_ankle is not None else a.right_ankle_flexion)
        com = _com_xy(f)
        if com:
            com_x.append(com[0])
            com_y.append(com[1])
        pelvis_x.append(_pelvis_mid_x(f))
        shoulder_offset.append(_shoulder_pelvis_offset(f))
        ankle_l_y.append(_ankle_y(f, "left"))
        ankle_r_y.append(_ankle_y(f, "right"))

    return _Series(
        knee_l=knee_l, knee_r=knee_r, hip_l=hip_l, hip_r=hip_r,
        ankle_l=ankle_l, ankle_r=ankle_r,
        com_x=com_x, com_y=com_y,
        pelvis_x=pelvis_x, shoulder_offset=shoulder_offset,
        ankle_l_y=ankle_l_y, ankle_r_y=ankle_r_y,
        body_width=_body_width(frames),
        body_height=_body_height(frames),
        mean_foot_visibility=_mean_foot_visibility(frames),
        fps=max(sequence.fps, 1e-6),
        n=len(frames),
        total_frames=len(sequence.frames),
        valid_frame_pct=len(frames) / max(len(sequence.frames), 1),
    )


# ---------------------------------------------------------------------------
# Metric 1 — Gait symmetry (left vs right joint angles)
# ---------------------------------------------------------------------------
def _metric_symmetry(s: _Series, gait: GaitStepDetectionResult) -> MetricResult:
    joints = {
        "knee": (s.knee_l, s.knee_r),
        "hip": (s.hip_l, s.hip_r),
        "ankle": (s.ankle_l, s.ankle_r),
    }
    sub_scores: list[float] = []
    findings: list[str] = []
    values: dict[str, Any] = {}

    for name, (left, right) in joints.items():
        diff = _mean_abs_diff(left, right)
        if diff is None:
            continue
        zero_at = (
            SYMMETRY_ANKLE_DIFF_ZERO_DEG if name == "ankle" else SYMMETRY_DIFF_ZERO_DEG
        )
        score = _linear_score(diff, SYMMETRY_DIFF_FULL_DEG, zero_at)
        weight = 2.0 if name == "ankle" else 1.0
        sub_scores.extend([score] * int(weight))
        values[f"{name}_mean_abs_diff_deg"] = round(diff, 1)
        if diff >= SYMMETRY_FLAG_DEG:
            findings.append(
                f"Left {name} angle differs from the right {name} by "
                f"{diff:.0f}° on average (asymmetric {name} motion)."
            )

    steps_l = gait.left.step_count
    steps_r = gait.right.step_count
    cv_l = gait.left.interval_cv
    cv_r = gait.right.interval_cv
    if cv_l is not None and cv_r is not None and cv_l > 0 and cv_r > 0:
        tempo_ratio = min(cv_l, cv_r) / max(cv_l, cv_r)
        tempo_score = _clamp(100.0 * tempo_ratio)
        sub_scores.append(tempo_score)
        values["left_interval_cv"] = round(cv_l, 3)
        values["right_interval_cv"] = round(cv_r, 3)
        values["step_timing_balance"] = round(tempo_ratio, 2)
        if tempo_ratio < 0.65:
            findings.append(
                "Left and right legs show different step-timing variability "
                f"(CV balance {tempo_ratio:.0%})."
            )

    if max(steps_l, steps_r) > 0:
        step_balance = min(steps_l, steps_r) / max(steps_l, steps_r)
        sub_scores.append(_clamp(100.0 * step_balance))
        values["step_count_balance"] = round(step_balance, 2)

    if not sub_scores:
        return MetricResult(
            key="symmetry", name="Gait symmetry",
            score=None, weight=METRIC_WEIGHTS["symmetry"],
            summary="Not enough left/right joint angles to compare.",
        )

    score = float(np.mean(sub_scores))
    summary = (
        f"Left/right joint angles agree well ({score:.0f}/100)."
        if not findings
        else f"Left/right asymmetry detected ({score:.0f}/100)."
    )
    return MetricResult(
        key="symmetry", name="Gait symmetry", score=score,
        weight=METRIC_WEIGHTS["symmetry"], summary=summary,
        findings=findings, values=values,
    )


# ---------------------------------------------------------------------------
# Metric 2 — Body stability (pelvis + torso sway, body-relative)
# ---------------------------------------------------------------------------
def _metric_body_stability(s: _Series) -> MetricResult:
    sub_scores: list[float] = []
    findings: list[str] = []
    values: dict[str, Any] = {}
    width = max(s.body_width, 1e-6)

    pelvis_vals = [v for v in s.pelvis_x if v is not None]
    if len(pelvis_vals) >= MIN_FRAMES:
        px = _detrend_linear(np.asarray(pelvis_vals, dtype=float))
        pelvis_std = float(np.std(px))
        pelvis_ratio = pelvis_std / width
        pelvis_score = _linear_score(pelvis_ratio, 0.0, BODY_SWAY_ZERO_RATIO)
        sub_scores.append(pelvis_score)
        values["pelvis_lateral_sway_ratio"] = round(pelvis_ratio, 3)
        if pelvis_ratio >= BODY_SWAY_FLAG_RATIO:
            findings.append(
                f"Pelvis sways side-to-side at {pelvis_ratio:.0%} of body width "
                f"(steady walking stays below {BODY_SWAY_FLAG_RATIO:.0%})."
            )

    tilt_vals = [v for v in s.shoulder_offset if v is not None]
    if len(tilt_vals) >= MIN_FRAMES:
        offset = _detrend_linear(np.asarray(tilt_vals, dtype=float))
        offset_std = float(np.std(offset))
        torso_ratio = offset_std / width
        torso_score = _linear_score(torso_ratio, 0.0, BODY_SWAY_ZERO_RATIO)
        sub_scores.append(torso_score)
        values["torso_lateral_offset_ratio"] = round(torso_ratio, 3)
        if torso_ratio >= BODY_SWAY_FLAG_RATIO:
            findings.append(
                f"Shoulders shift {torso_ratio:.0%} of body width relative to the pelvis "
                f"(trunk sway above {BODY_SWAY_FLAG_RATIO:.0%})."
            )

    if len(s.com_x) >= MIN_FRAMES:
        x = np.asarray(s.com_x, dtype=float)
        lateral = _detrend_linear(x)
        com_ratio = float(np.std(lateral)) / width
        com_score = _linear_score(com_ratio, 0.0, BODY_SWAY_ZERO_RATIO)
        sub_scores.append(com_score)
        values["com_lateral_sway_ratio"] = round(com_ratio, 3)

    # Restricted / shuffling gait can show low lateral sway — do not treat as high stability.
    joint_series = (
        s.knee_l, s.knee_r, s.hip_l, s.hip_r, s.ankle_l, s.ankle_r,
    )
    mean_rom = float(np.mean([_robust_range(seq) for seq in joint_series]))
    dynamic_score = _linear_score(mean_rom, WALK_MIN_MEAN_ROM_DEG, 0.0)
    sub_scores.append(dynamic_score)
    values["mean_joint_rom_deg"] = round(mean_rom, 1)
    if mean_rom < WALK_MIN_MEAN_ROM_DEG:
        findings.append(
            f"Overall joint movement is limited (mean ROM {mean_rom:.0f}°), "
            "which may reflect restricted rather than controlled gait."
        )

    if not sub_scores:
        return MetricResult(
            key="body_stability", name="Body stability",
            score=None, weight=METRIC_WEIGHTS["body_stability"],
            summary="Not enough frames to estimate pelvis/torso stability.",
        )

    score = float(np.mean(sub_scores))
    summary = (
        f"Pelvis and trunk stay steady ({score:.0f}/100)."
        if not findings
        else f"Pelvis or trunk sway detected ({score:.0f}/100)."
    )
    return MetricResult(
        key="body_stability", name="Body stability", score=score,
        weight=METRIC_WEIGHTS["body_stability"], summary=summary,
        findings=findings, values=values,
    )


# ---------------------------------------------------------------------------
# Metric 3 — Step consistency (ankle/heel/toe timing)
# ---------------------------------------------------------------------------
def _detect_steps(
    signal: list[float | None],
    fps: float,
    *,
    body_scale: float = 1.0,
) -> tuple[int, float | None, list[int]]:
    """Backward-compatible wrapper around :func:`detect_steps_in_signal`."""
    from stablewalk.pose.gait_step_detection import detect_steps_in_signal

    result = detect_steps_in_signal(
        signal, fps, body_scale=body_scale, landmark_used="ankle",
    )
    # Return list indices (0..n-1) for callers that index into per-frame series.
    index_events = [
        i for i, frame_idx in enumerate(result.aligned_frame_indices)
        if frame_idx in result.event_frame_indices
    ]
    return result.step_count, result.interval_cv, index_events


def _merged_step_cv(peaks_l: list[int], peaks_r: list[int], fps: float) -> float | None:
    """CV of inter-step intervals when left and right contacts are merged in time."""
    merged = sorted(set(peaks_l + peaks_r))
    if len(merged) < 3:
        return None
    intervals = [(merged[i + 1] - merged[i]) / fps for i in range(len(merged) - 1)]
    return _cv(intervals)


def _required_steps_per_side(duration_s: float) -> int:
    """Minimum same-foot contacts needed for timing CV on a clip of this length."""
    scaled = STEP_MIN_PER_SIDE * duration_s / STEP_REFERENCE_DURATION_S
    return max(1, min(STEP_MIN_PER_SIDE, int(math.ceil(scaled))))


def _step_count_coverage(detected_steps: int, duration_s: float) -> tuple[float, float, float]:
    """
    Penalize footfall counts outside a physiologically plausible cadence band.

    Uses the same cadence limits as step detection (both legs combined) so brisk
    walking is not mistaken for over-detection.
    """
    expected_lo = max(2.0, duration_s * CADENCE_MIN_HZ)
    expected_hi = max(2.0, duration_s * CADENCE_MAX_HZ)
    if detected_steps <= 0:
        return 0.0, expected_lo, expected_hi
    if expected_lo <= detected_steps <= expected_hi:
        return 1.0, expected_lo, expected_hi
    if detected_steps < expected_lo:
        return _clamp(detected_steps / expected_lo, 0.0, 1.0), expected_lo, expected_hi
    return _clamp(expected_hi / detected_steps, 0.0, 1.0), expected_lo, expected_hi


def _metric_step_consistency(s: _Series, gait: GaitStepDetectionResult) -> MetricResult:
    scaled_l = _scale_foot_signal(s.ankle_l_y, s.body_height)
    scaled_r = _scale_foot_signal(s.ankle_r_y, s.body_height)
    steps_l = gait.left.step_count
    steps_r = gait.right.step_count
    cv_l = gait.left.interval_cv
    cv_r = gait.right.interval_cv
    peaks_l = gait.left.event_frame_indices
    peaks_r = gait.right.event_frame_indices

    side_cvs = [c for c in (cv_l, cv_r) if c is not None]
    merged_cv = _merged_step_cv(peaks_l, peaks_r, s.fps)

    if not side_cvs and merged_cv is None and steps_l + steps_r == 0:
        return MetricResult(
            key="step_consistency", name="Step consistency",
            score=None, weight=METRIC_WEIGHTS["step_consistency"],
            summary="Not enough clear steps to assess timing.",
        )

    values: dict[str, Any] = {}

    # Same-side timing regularity (lower CV is better).
    if side_cvs:
        mean_cv = float(np.mean(side_cvs))
        timing_score = _linear_score(mean_cv, 0.0, STEP_CV_ZERO)
        if len(side_cvs) == 2:
            cv_balance = min(side_cvs) / max(side_cvs)
            timing_score *= cv_balance
            values["step_timing_cv_balance"] = round(cv_balance, 2)
        else:
            values["step_timing_cv_balance"] = None
    else:
        mean_cv = STEP_CV_ZERO
        timing_score = 0.0
        values["step_timing_cv_balance"] = None

    # Merged stride regularity penalizes irregular alternation between legs.
    if merged_cv is not None:
        stride_score = _linear_score(merged_cv, 0.0, STEP_CV_ZERO)
        timing_score = 0.65 * timing_score + 0.35 * stride_score
        values["merged_stride_cv"] = round(merged_cv, 3)
    else:
        values["merged_stride_cv"] = None

    # Unreliable timing when too few contacts detected per foot (scaled to clip length).
    duration_s = s.n / s.fps
    min_side_steps = min(steps_l, steps_r)
    required_per_side = _required_steps_per_side(duration_s)
    if min_side_steps < required_per_side:
        reliability = min_side_steps / required_per_side
        timing_score *= reliability
        values["step_timing_reliability"] = round(reliability, 2)
        values["required_steps_per_side"] = required_per_side
    else:
        values["step_timing_reliability"] = 1.0
        values["required_steps_per_side"] = required_per_side

    # Penalize footfall counts outside plausible cadence band (not a fixed single rate).
    detected_steps = steps_l + steps_r
    coverage, expected_lo, expected_hi = _step_count_coverage(detected_steps, duration_s)
    timing_score *= coverage
    values["expected_steps_lo"] = round(expected_lo, 1)
    values["expected_steps_hi"] = round(expected_hi, 1)

    # Low-confidence step detection reduces timing score reliability.
    if gait.confidence == "low":
        timing_score *= 0.75
        values["step_detection_confidence"] = "low"
    else:
        values["step_detection_confidence"] = "high"

    balance = 1.0
    if max(steps_l, steps_r) > 0:
        balance = min(steps_l, steps_r) / max(steps_l, steps_r)
    balance_score = _clamp(100.0 * balance)
    if min_side_steps < required_per_side:
        balance_score *= min_side_steps / required_per_side

    # Foot clearance consistency: compare vertical oscillation amplitude L vs R.
    clearance_l = _robust_range(scaled_l)
    clearance_r = _robust_range(scaled_r)
    clearance_score = 100.0
    if clearance_l > 0 and clearance_r > 0:
        mean_clr = (clearance_l + clearance_r) / 2.0
        clr_ratio = abs(clearance_l - clearance_r) / mean_clr
        clearance_score = _linear_score(clr_ratio, 0.0, ROM_DIFF_ZERO_RATIO)
        clearance_balance = round(1.0 - clr_ratio, 2) if clr_ratio <= 1 else 0.0
    else:
        clearance_balance = None

    score = 0.72 * timing_score + 0.13 * balance_score + 0.15 * clearance_score

    # Compensatory / neuropathic patterns: similar foot clearance but asymmetric ankle
    # kinematics should not receive a high step-consistency score.
    ankle_asym = _mean_abs_diff(s.ankle_l, s.ankle_r)
    if ankle_asym is not None and ankle_asym >= STEP_ANKLE_ASYM_PENALTY_DEG:
        asym_factor = _linear_score(
            ankle_asym, STEP_ANKLE_ASYM_PENALTY_DEG, SYMMETRY_ANKLE_DIFF_ZERO_DEG,
        ) / 100.0
        score *= 0.50 + 0.50 * asym_factor
        values["ankle_asymmetry_step_factor"] = round(asym_factor, 2)

    # False regularity: very steady same-side timing but poor merged stride cadence.
    if (
        merged_cv is not None
        and merged_cv >= STEP_CV_FLAG
        and side_cvs
        and float(np.mean(side_cvs)) < STEP_CV_FLAG
    ):
        score *= 0.82
        values["false_regularity_penalty"] = True

    findings: list[str] = []
    if side_cvs and mean_cv >= STEP_CV_FLAG:
        findings.append(
            f"Irregular step timing (same-side CV={mean_cv:.2f}; steady walking "
            f"is usually below {STEP_CV_FLAG:.2f})."
        )
    if merged_cv is not None and merged_cv >= STEP_CV_FLAG:
        findings.append(
            f"Irregular stride timing between legs (merged CV={merged_cv:.2f})."
        )
    if coverage < 0.65:
        findings.append(
            f"Step count ({detected_steps}) is outside a plausible cadence band "
            f"(expected {expected_lo:.0f}–{expected_hi:.0f} foot contacts; coverage {coverage:.0%})."
        )
    if balance < STEP_BALANCE_FLAG:
        findings.append(
            f"Unbalanced steps: left foot took {steps_l} steps vs right foot "
            f"{steps_r} — the two sides are not stepping evenly."
        )
    if clearance_balance is not None and clearance_balance < 0.70:
        findings.append(
            f"Foot clearance differs between legs "
            f"(left {clearance_l:.3f}, right {clearance_r:.3f} body-height units)."
        )

    summary = (
        f"Steps are regular and balanced ({score:.0f}/100)."
        if not findings
        else f"Step timing/balance problems detected ({score:.0f}/100)."
    )
    return MetricResult(
        key="step_consistency", name="Step consistency", score=score,
        weight=METRIC_WEIGHTS["step_consistency"], summary=summary,
        findings=findings,
        values={
            "left_steps": steps_l, "right_steps": steps_r,
            "left_interval_cv": None if cv_l is None else round(cv_l, 3),
            "right_interval_cv": None if cv_r is None else round(cv_r, 3),
            "step_balance": round(balance, 2),
            "foot_clearance_left": round(clearance_l, 4),
            "foot_clearance_right": round(clearance_r, 4),
            "foot_clearance_balance": clearance_balance,
            "step_coverage": round(coverage, 2),
            "expected_steps_lo": values.get("expected_steps_lo"),
            "expected_steps_hi": values.get("expected_steps_hi"),
            "step_detection_confidence": values.get("step_detection_confidence"),
            "cadence_hz": gait.cadence_hz,
            "peak_frames_left": peaks_l,
            "peak_frames_right": peaks_r,
            **{k: v for k, v in values.items() if k not in ("step_detection_confidence",)},
        },
    )


# ---------------------------------------------------------------------------
# Metric 4 — Joint range of motion (consistency between sides)
# ---------------------------------------------------------------------------
def _metric_range_of_motion(s: _Series) -> MetricResult:
    joints = {
        "knee": (s.knee_l, s.knee_r),
        "hip": (s.hip_l, s.hip_r),
        "ankle": (s.ankle_l, s.ankle_r),
    }
    sub_scores: list[float] = []
    findings: list[str] = []
    values: dict[str, Any] = {}

    for name, (left, right) in joints.items():
        rom_l = _robust_range(left)
        rom_r = _robust_range(right)
        values[f"{name}_rom_left_deg"] = round(rom_l, 1)
        values[f"{name}_rom_right_deg"] = round(rom_r, 1)
        if rom_l <= 0 and rom_r <= 0:
            continue

        mean_rom = max((rom_l + rom_r) / 2.0, 1e-6)
        diff_ratio = abs(rom_l - rom_r) / mean_rom
        consistency_score = _linear_score(diff_ratio, 0.0, ROM_DIFF_ZERO_RATIO)
        adequacy_score = _linear_score(mean_rom, WALK_MIN_MEAN_ROM_DEG, 0.0)
        score = 0.65 * consistency_score + 0.35 * adequacy_score
        sub_scores.append(score)

        if diff_ratio >= ROM_FLAG_RATIO:
            bigger, smaller = ("left", "right") if rom_l > rom_r else ("right", "left")
            findings.append(
                f"{bigger.capitalize()} {name} range of motion is significantly "
                f"different from the {smaller} {name} "
                f"(left {rom_l:.0f}°, right {rom_r:.0f}°, {diff_ratio:.0%} difference)."
            )
        if 0 < min(rom_l, rom_r) < ROM_STIFF_DEG:
            stiff = "left" if rom_l < rom_r else "right"
            stiff_score = _linear_score(min(rom_l, rom_r), 0.0, ROM_STIFF_DEG)
            sub_scores.append(stiff_score)
            if diff_ratio >= ROM_FLAG_RATIO:
                sub_scores.append(min(stiff_score, consistency_score))
            findings.append(
                f"The {stiff} {name} barely moves (ROM "
                f"{min(rom_l, rom_r):.0f}°), which can indicate a stiff joint."
            )

    if not sub_scores:
        return MetricResult(
            key="range_of_motion", name="Joint range of motion",
            score=None, weight=METRIC_WEIGHTS["range_of_motion"],
            summary="Not enough joint-angle data to measure range of motion.",
        )

    # Multi-joint asymmetry pattern (common in compensatory / pathological gait).
    severe_asymmetry = sum(
        1 for name in joints
        if values.get(f"{name}_rom_left_deg") is not None
        and values.get(f"{name}_rom_right_deg") is not None
        and max(values[f"{name}_rom_left_deg"], values[f"{name}_rom_right_deg"]) > 0
        and abs(values[f"{name}_rom_left_deg"] - values[f"{name}_rom_right_deg"])
        / max((values[f"{name}_rom_left_deg"] + values[f"{name}_rom_right_deg"]) / 2.0, 1e-6)
        >= ROM_FLAG_RATIO
    )
    if severe_asymmetry >= 2:
        sub_scores.append(max(3.0, 32.0 - severe_asymmetry * 16.0))
        values["multi_joint_asymmetry_count"] = severe_asymmetry

    score = float(np.mean(sub_scores))
    summary = (
        f"Knees, hips and ankles move consistently on both sides ({score:.0f}/100)."
        if not findings
        else f"Range of motion is uneven between sides ({score:.0f}/100)."
    )
    return MetricResult(
        key="range_of_motion", name="Joint range of motion", score=score,
        weight=METRIC_WEIGHTS["range_of_motion"], summary=summary,
        findings=findings, values=values,
    )


# ---------------------------------------------------------------------------
# Metric 5 — Trajectory smoothness (joint jerk)
# ---------------------------------------------------------------------------
def _metric_trajectory_smoothness(s: _Series) -> MetricResult:
    series_map = {
        "knee": (s.knee_l, s.knee_r),
        "hip": (s.hip_l, s.hip_r),
        "ankle": (s.ankle_l, s.ankle_r),
    }
    sub_scores: list[float] = []
    findings: list[str] = []
    values: dict[str, Any] = {}

    # Lower landmark visibility increases coordinate noise; widen jerk tolerance accordingly.
    noise_scale = max(0.85, min(1.6, 1.05 / max(s.mean_foot_visibility, 0.55)))
    jerk_zero = TRAJECTORY_JERK_ZERO * noise_scale

    for name, (left, right) in series_map.items():
        for side_label, seq in (("left", left), ("right", right)):
            jerk = _mean_jerk(seq)
            if jerk is None:
                continue
            rom = _robust_range(seq)
            norm_jerk = jerk / max(rom, JERK_ROM_FLOOR_DEG)
            score = _linear_score(norm_jerk, 0.0, jerk_zero)
            sub_scores.append(score)
            values[f"{name}_{side_label}_mean_jerk"] = round(jerk, 3)
            values[f"{name}_{side_label}_normalized_jerk"] = round(norm_jerk, 3)
            if norm_jerk >= TRAJECTORY_JERK_FLAG * noise_scale:
                findings.append(
                    f"{side_label.capitalize()} {name} motion is jerky "
                    f"(normalized jerk {norm_jerk:.2f})."
                )

    if not sub_scores:
        return MetricResult(
            key="trajectory_smoothness", name="Trajectory smoothness",
            score=None, weight=METRIC_WEIGHTS["trajectory_smoothness"],
            summary="Not enough joint-angle data to assess smoothness.",
        )

    score = float(np.mean(sub_scores))
    summary = (
        f"Joint trajectories are smooth ({score:.0f}/100)."
        if not findings
        else f"Jerky joint motion detected ({score:.0f}/100)."
    )
    values["mean_foot_visibility"] = round(s.mean_foot_visibility, 3)
    values["jerk_tolerance_scale"] = round(noise_scale, 3)
    return MetricResult(
        key="trajectory_smoothness", name="Trajectory smoothness", score=score,
        weight=METRIC_WEIGHTS["trajectory_smoothness"], summary=summary,
        findings=findings, values=values,
    )


# ---------------------------------------------------------------------------
# Metric 6 — Pose quality (visibility + frame coverage)
# ---------------------------------------------------------------------------
def _metric_pose_quality(s: _Series) -> MetricResult:
    visibility_score = _linear_score(s.mean_foot_visibility, POSE_VISIBILITY_FULL, 0.45)
    coverage_score = _clamp(100.0 * s.valid_frame_pct)
    score = 0.55 * visibility_score + 0.45 * coverage_score
    findings: list[str] = []
    if s.mean_foot_visibility < 0.75:
        findings.append(
            f"Foot/ankle landmarks are often low-confidence "
            f"(mean visibility {s.mean_foot_visibility:.0%})."
        )
    if s.valid_frame_pct < 0.85:
        findings.append(
            f"Only {s.valid_frame_pct:.0%} of frames had usable pose data."
        )
    summary = (
        f"Pose tracking quality is good ({score:.0f}/100)."
        if not findings
        else f"Pose tracking quality limits confidence ({score:.0f}/100)."
    )
    return MetricResult(
        key="pose_quality", name="Pose quality", score=score,
        weight=METRIC_WEIGHTS["pose_quality"], summary=summary,
        findings=findings,
        values={
            "mean_foot_visibility": round(s.mean_foot_visibility, 3),
            "valid_frame_pct": round(s.valid_frame_pct, 3),
            "detected_frames": s.n,
            "total_frames": s.total_frames,
        },
    )


# ---------------------------------------------------------------------------
# Combine into the final score + explanation
# ---------------------------------------------------------------------------
def _classify(score: float) -> str:
    if score >= STABLE_MIN:
        return "Stable"
    if score >= MODERATE_MIN:
        return "Moderate"
    return "Unstable"


def _build_explanation(
    score: float,
    classification: str,
    metrics: list[MetricResult],
    primary_issue: str | None,
) -> str:
    lines = [
        f"Walking pattern classified as {classification.upper()} "
        f"(stability score {score:.0f}/100).",
        "",
        "How the score is built (transparent, weighted average):",
    ]
    for m in metrics:
        if m.score is None:
            lines.append(f"  - {m.name}: not available — {m.summary}")
        else:
            lines.append(
                f"  - {m.name}: {m.score:.0f}/100 (weight {m.weight:.0%}) — {m.summary}"
            )

    issues = [f for m in metrics for f in m.findings]
    lines.append("")
    if issues:
        lines.append("Why it is not perfectly stable — specific findings:")
        for f in issues:
            lines.append(f"  • {f}")
    else:
        lines.append("No specific instability findings — gait looks symmetric, "
                     "balanced, and steady.")

    if primary_issue:
        lines.append("")
        lines.append(f"Main concern: {primary_issue}")

    return "\n".join(lines)


def analyze_biomech_stability(sequence: PoseSequence) -> StabilityResult:
    """
    Compute the transparent, explainable stability assessment.

    Combines gait symmetry, step consistency, body stability, joint range of motion,
    and trajectory smoothness into a 0–100 score and a Stable / Moderate / Unstable
    label, each with plain-language reasons.
    """
    s = _extract_series(sequence)
    gait = detect_gait_steps(sequence)

    if s.n < MIN_FRAMES:
        msg = (
            f"Only {s.n} usable frames were detected (need at least {MIN_FRAMES}). "
            "Analyze a longer/clearer walking video for a reliable score."
        )
        return StabilityResult(
            score=0.0,
            classification="Insufficient data",
            metrics=[],
            primary_issue=msg,
            explanation=msg,
            frame_count=s.n,
        )

    metrics = [
        _metric_symmetry(s, gait),
        _metric_step_consistency(s, gait),
        _metric_body_stability(s),
        _metric_range_of_motion(s),
        _metric_trajectory_smoothness(s),
        _metric_pose_quality(s),
    ]

    # Weighted average over the metrics that could be computed (renormalize).
    available = [m for m in metrics if m.score is not None]
    total_weight = sum(m.weight for m in available)
    if total_weight <= 0:
        score = 0.0
    else:
        score = sum(m.score * m.weight for m in available) / total_weight  # type: ignore[operator]

    classification = _classify(score)

    # Primary issue = the metric that hurt the score most (lowest, and below good)
    # and contributed a concrete finding.
    primary_issue: str | None = None
    worst = min(
        (m for m in available if m.score is not None and m.score < STABLE_MIN),
        key=lambda m: m.score,  # type: ignore[arg-type, return-value]
        default=None,
    )
    if worst is not None:
        if worst.findings:
            primary_issue = f"{worst.name} ({worst.score:.0f}/100) — {worst.findings[0]}"
        else:
            primary_issue = f"{worst.name} scored lowest ({worst.score:.0f}/100)."

    explanation = _build_explanation(score, classification, metrics, primary_issue)

    return StabilityResult(
        score=score,
        classification=classification,
        metrics=metrics,
        primary_issue=primary_issue,
        explanation=explanation,
        frame_count=s.n,
    )


__all__ = [
    "StabilityResult",
    "MetricResult",
    "analyze_biomech_stability",
    "STABLE_MIN",
    "MODERATE_MIN",
    "METRIC_WEIGHTS",
    "SCORING_NOTES",
]
