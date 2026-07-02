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
5. **Stability score** — a weighted average of the four group scores, mapped to
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

# ---------------------------------------------------------------------------
# Classification thresholds (final 0–100 score)
# ---------------------------------------------------------------------------
STABLE_MIN = 70.0
MODERATE_MIN = 45.0  # below this => Unstable

# Weights for combining the four metric groups (must describe intent clearly).
METRIC_WEIGHTS: dict[str, float] = {
    "symmetry": 0.30,      # left/right joint-angle agreement
    "range_of_motion": 0.25,  # consistent ROM between sides
    "step_consistency": 0.25,  # regular, balanced steps
    "center_of_mass": 0.20,   # little side-to-side sway
}

# Per-metric thresholds (documented; used in transparent formulas).
SYMMETRY_DIFF_FULL_DEG = 0.0    # 0° L/R difference  -> sub-score 100
SYMMETRY_DIFF_ZERO_DEG = 25.0   # >=25° L/R difference -> sub-score 0
SYMMETRY_FLAG_DEG = 12.0        # report a finding above this average difference

ROM_DIFF_ZERO_RATIO = 0.60      # 60% L/R ROM difference -> sub-score 0
ROM_FLAG_RATIO = 0.30           # report a finding above this ROM difference
ROM_STIFF_DEG = 5.0             # a moving joint with <5° ROM looks "stiff"

COM_SWAY_ZERO_RATIO = 0.50      # lateral sway = 50% of shoulder width -> 0
COM_SWAY_FLAG_RATIO = 0.18      # report excessive sway above this

STEP_CV_ZERO = 0.40             # step-interval CV of 0.40 -> sub-score 0
STEP_CV_FLAG = 0.20             # report irregular timing above this CV
STEP_BALANCE_FLAG = 0.70        # left/right step-count ratio below this -> flag

MIN_FRAMES = 8                  # need at least this many detected frames

SCORING_NOTES = (
    "Score starts from four transparent group scores (each 0-100): gait symmetry, "
    "joint range of motion, step consistency, and center-of-mass sway. The final "
    "score is their weighted average (symmetry 30%, ROM 25%, steps 25%, CoM 20%). "
    "Stable >= 70, Moderate 45-69, Unstable < 45."
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
    ankle_l_y: list[float | None]
    ankle_r_y: list[float | None]
    body_width: float
    fps: float
    n: int


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
    """Mean shoulder width (fallback hip width) — lateral scale for CoM sway."""
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
    com_x, com_y, ankle_l_y, ankle_r_y = [], [], [], []

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
        ankle_l_y.append(_ankle_y(f, "left"))
        ankle_r_y.append(_ankle_y(f, "right"))

    return _Series(
        knee_l=knee_l, knee_r=knee_r, hip_l=hip_l, hip_r=hip_r,
        ankle_l=ankle_l, ankle_r=ankle_r,
        com_x=com_x, com_y=com_y,
        ankle_l_y=ankle_l_y, ankle_r_y=ankle_r_y,
        body_width=_body_width(frames),
        fps=max(sequence.fps, 1e-6),
        n=len(frames),
    )


# ---------------------------------------------------------------------------
# Metric 1 — Gait symmetry (left vs right joint angles)
# ---------------------------------------------------------------------------
def _metric_symmetry(s: _Series) -> MetricResult:
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
        score = _linear_score(diff, SYMMETRY_DIFF_FULL_DEG, SYMMETRY_DIFF_ZERO_DEG)
        sub_scores.append(score)
        values[f"{name}_mean_abs_diff_deg"] = round(diff, 1)
        if diff >= SYMMETRY_FLAG_DEG:
            findings.append(
                f"Left {name} angle differs from the right {name} by "
                f"{diff:.0f}° on average (asymmetric {name} motion)."
            )

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
# Metric 2 — Center of mass (side-to-side sway)
# ---------------------------------------------------------------------------
def _metric_com(s: _Series) -> MetricResult:
    if len(s.com_x) < MIN_FRAMES:
        return MetricResult(
            key="center_of_mass", name="Center of mass",
            score=None, weight=METRIC_WEIGHTS["center_of_mass"],
            summary="Not enough frames to estimate body center.",
        )

    x = np.asarray(s.com_x, dtype=float)
    # Remove the overall walking direction (linear trend) so we measure only the
    # side-to-side oscillation, not the person crossing the frame.
    t = np.arange(x.size, dtype=float)
    trend = np.polyfit(t, x, 1)
    lateral = x - np.polyval(trend, t)
    lateral_std = float(np.std(lateral))

    width = max(s.body_width, 1e-6)
    sway_ratio = lateral_std / width
    score = _linear_score(sway_ratio, 0.0, COM_SWAY_ZERO_RATIO)

    y = np.asarray(s.com_y, dtype=float)
    vertical_range = float(np.max(y) - np.min(y)) if y.size else 0.0

    findings: list[str] = []
    if sway_ratio >= COM_SWAY_FLAG_RATIO:
        findings.append(
            f"Excessive side-to-side movement: the body center sways "
            f"{sway_ratio:.0%} of shoulder width (a steady walk stays well below "
            f"{COM_SWAY_FLAG_RATIO:.0%})."
        )

    summary = (
        f"Body center is steady ({score:.0f}/100)."
        if not findings
        else f"Body center sways too much side-to-side ({score:.0f}/100)."
    )
    return MetricResult(
        key="center_of_mass", name="Center of mass", score=score,
        weight=METRIC_WEIGHTS["center_of_mass"], summary=summary,
        findings=findings,
        values={
            "lateral_sway_ratio": round(sway_ratio, 3),
            "lateral_std_normalized": round(lateral_std, 4),
            "vertical_range_normalized": round(vertical_range, 4),
        },
    )


# ---------------------------------------------------------------------------
# Metric 3 — Step consistency (ankle/heel/toe timing)
# ---------------------------------------------------------------------------
def _detect_steps(signal: list[float | None], fps: float) -> tuple[int, float | None]:
    """
    Detect foot-contact steps from the foot vertical signal.

    The foot reaches its lowest screen point (largest y) at each contact, so we
    detect local maxima. Returns (step_count, interval_CV).
    """
    vals = [v for v in signal if v is not None]
    if len(vals) < MIN_FRAMES:
        return 0, None
    arr = np.asarray(vals, dtype=float)
    rng = float(np.max(arr) - np.min(arr))
    if rng < 1e-4:
        return 0, None
    min_distance = max(3, int(round(fps * 0.3)))  # steps no closer than ~0.3 s
    min_prominence = 0.15 * rng
    peaks = _find_peaks(arr, min_distance, min_prominence)
    if len(peaks) < 2:
        return len(peaks), None
    intervals = [(peaks[i + 1] - peaks[i]) / fps for i in range(len(peaks) - 1)]
    return len(peaks), _cv(intervals)


def _metric_step_consistency(s: _Series) -> MetricResult:
    steps_l, cv_l = _detect_steps(s.ankle_l_y, s.fps)
    steps_r, cv_r = _detect_steps(s.ankle_r_y, s.fps)

    cvs = [c for c in (cv_l, cv_r) if c is not None]
    if not cvs:
        return MetricResult(
            key="step_consistency", name="Step consistency",
            score=None, weight=METRIC_WEIGHTS["step_consistency"],
            summary="Not enough clear steps to assess timing.",
        )

    mean_cv = float(np.mean(cvs))
    timing_score = _linear_score(mean_cv, 0.0, STEP_CV_ZERO)

    # Left/right step-count balance (both feet should take a similar number).
    balance = 1.0
    if max(steps_l, steps_r) > 0:
        balance = min(steps_l, steps_r) / max(steps_l, steps_r)
    balance_score = _clamp(100.0 * balance)

    score = 0.6 * timing_score + 0.4 * balance_score

    findings: list[str] = []
    if mean_cv >= STEP_CV_FLAG:
        findings.append(
            f"Irregular step timing (variation CV={mean_cv:.2f}; steady walking "
            f"is usually below {STEP_CV_FLAG:.2f})."
        )
    if balance < STEP_BALANCE_FLAG:
        findings.append(
            f"Unbalanced steps: left foot took {steps_l} steps vs right foot "
            f"{steps_r} — the two sides are not stepping evenly."
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
        score = _linear_score(diff_ratio, 0.0, ROM_DIFF_ZERO_RATIO)
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

    Combines gait symmetry, joint range of motion, step consistency, and CoM
    sway into a 0–100 score and a Stable / Moderate / Unstable label, each with
    plain-language reasons.
    """
    s = _extract_series(sequence)

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
        _metric_symmetry(s),
        _metric_range_of_motion(s),
        _metric_step_consistency(s),
        _metric_com(s),
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
