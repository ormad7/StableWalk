"""
Gait stability analysis from pose, contact timing, and estimated GRF.

Uses ``StabilityAnalyzer`` to combine spatiotemporal metrics, bilateral symmetry,
CoM movement, and force proxies into a 0–100 stability score with explanations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from stablewalk.analysis.forces import ForceAnalyzer, GRFTimeSeries
from stablewalk.analysis.metrics import GaitMetrics, GaitMetricsReport, GaitMetricsResult
from stablewalk.io.pose_loader import detected_frame_indices
from stablewalk.models.pose_data import PoseFrame, PoseSequence

# Classification thresholds (score is 0–100 after penalties from base 100)
STABLE_MIN_SCORE = 60.0
BORDERLINE_MIN_SCORE = 45.0
UNSTABLE_MAX_SCORE = BORDERLINE_MIN_SCORE - 1  # below borderline → Unstable status

BASE_SCORE = 100.0

# Penalty points (subtracted from 100). Tuned for video-based proxies on ~30 fps clips.
PENALTY_SYMMETRY_LOW = 10.0       # symmetry_score < 0.70
PENALTY_SYMMETRY_SEVERE = 10.0    # symmetry_score < 0.55 (stacks)
PENALTY_STEP_VARIABILITY = 15.0   # step CV > 0.18
PENALTY_STEP_VARIABILITY_HIGH = 5.0  # step CV > 0.25 (stacks)
PENALTY_STRIDE_VARIABILITY = 8.0  # stride CV > 0.20
PENALTY_COM_SWAY = 12.0           # lateral std > 0.045
PENALTY_COM_SWAY_SEVERE = 8.0     # lateral std > 0.060
PENALTY_COM_VERTICAL = 6.0        # vertical range out of band
PENALTY_GRF_ASYMMETRY = 20.0      # peak GRF symmetry < 0.65
PENALTY_GRF_PATTERN = 10.0        # low double-peak fraction
PENALTY_STANCE_ASYMMETRY = 8.0   # stance time ratio < 0.75
PENALTY_JOINT_ASYMMETRY = 8.0     # mean L/R knee or hip diff > 15°
PENALTY_CADENCE = 5.0             # cadence outside 0.8–2.2 Hz
PENALTY_FEW_FRAMES = 10.0         # < 10 analyzed frames

SCORING_HOW_IT_WORKS = """
Stability scoring (0–100)
-------------------------
1. Start at **100** (assumes ideal symmetric, regular, quiet CoM, typical forces).
2. **Subtract** fixed penalty points when a rule fires (asymmetry, irregular steps,
   unstable CoM, abnormal GRF). Multiple rules can apply; penalties stack.
3. Clamp final score to [0, 100].
4. **status**: "Stable" if score ≥ 60, else "Unstable" (label may also be Borderline 45–59).

Thresholds follow published gait-variability bands (CV ~0.15–0.20 for older adults)
and conservative symmetry/GRF cutoffs suitable for noisy monocular pose — not
clinical certification.
""".strip()

THRESHOLD_RATIONALE = """
Why these thresholds
--------------------
- **Symmetry < 0.70 (−10)**: Below this, step timing or stance or force balance is
  clearly unequal — common in limping or compensation.
- **Symmetry < 0.55 (−10 more)**: Severe bilateral mismatch.
- **Step CV > 0.18 (−15)**: Irregular footfall timing; CV above ~0.15–0.20 is often
  flagged as increased fall risk in literature.
- **Step CV > 0.25 (−5 more)**: Very irregular rhythm.
- **CoM lateral std > 0.045 (−12)**: Excess side sway in normalized image coordinates.
- **GRF peak symmetry < 0.65 (−20)**: One leg bears much more estimated load.
- **Double-peak < 20% of stances (−10)**: Atypical vertical force shape vs healthy walk.
- **Stable ≥ 60**: Lenient for student/demo video; stricter studies may use 70+.
""".strip()


METRIC_DESCRIPTIONS = {
    "symmetry_score": (
        "Left/right symmetry (0–1). Combines stance-time balance, knee/hip angle "
        "differences, step-interval balance, and GRF peak balance. 1 = equal legs."
    ),
    "stance_left_fraction": "Fraction of frames with left foot in stance phase.",
    "swing_left_fraction": "Fraction of frames with left foot in swing (1 - stance).",
    "stance_right_fraction": "Fraction of frames with right foot in stance.",
    "swing_right_fraction": "Fraction of frames with right foot in swing.",
    "cadence_hz": "Steps per second (heel strikes / duration).",
    "cadence_steps_per_min": "Steps per minute (common clinical unit).",
    "stride_length_normalized": "Mean pelvis displacement between same-side heel strikes (image-normalized).",
    "stride_length_m": "Stride length in estimated meters (if body scale available).",
    "step_time_variability": (
        "Coefficient of variation (std/mean) of inter-heel-strike intervals. "
        "Lower = more regular rhythm."
    ),
    "stride_length_variability": "CV of stride lengths across steps. Lower = more consistent steps.",
    "com_lateral_std": "Standard deviation of CoM horizontal position (image X). Lower = less side sway.",
    "com_vertical_range": "Peak-to-peak vertical CoM motion (image Y). Very large or small may be abnormal.",
    "com_path_length": "Total distance traveled by CoM proxy in the image plane.",
    "grf_symmetry": "Balance of peak vertical GRF (body weights) between left and right feet.",
    "stability_score": (
        "Overall 0–100: start at 100, subtract penalties for asymmetry, irregular steps, "
        "unstable CoM, and abnormal forces. Higher is more stable."
    ),
    "status": 'Binary "Stable" (≥60) or "Unstable" (<60).',
}


@dataclass
class StabilityMetrics:
    """Computed metric values (all explained in ``METRIC_DESCRIPTIONS``)."""

    symmetry_score: float
    stance_left_fraction: float
    swing_left_fraction: float
    stance_right_fraction: float
    swing_right_fraction: float
    cadence_hz: float | None
    cadence_steps_per_min: float | None
    stride_length_normalized: float | None
    stride_length_m: float | None
    step_time_variability: float | None
    stride_length_variability: float | None
    com_lateral_std: float
    com_vertical_range: float
    com_path_length: float
    grf_symmetry: float | None
    knee_symmetry_deg: float
    hip_symmetry_deg: float
    frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScorePenalty:
    """One deduction from the base score of 100."""

    rule_id: str
    points: float
    reason: str


@dataclass
class StabilityReport:
    """Full stability assessment with structured and human-readable output."""

    label: str  # Stable | Borderline | Unstable
    score: float  # 0–100 (same as stability_score)
    is_stable: bool
    status: str = ""  # Stable | Unstable (binary); set in analyze()
    base_score: float = BASE_SCORE
    penalties: list[ScorePenalty] = field(default_factory=list)
    abnormal_patterns: list[str] = field(default_factory=list)
    explanation: str = ""
    metrics: StabilityMetrics | None = None
    metric_descriptions: dict[str, str] = field(default_factory=lambda: dict(METRIC_DESCRIPTIONS))
    scoring_notes: str = field(default_factory=lambda: SCORING_HOW_IT_WORKS)

    # Legacy fields used by GUI (populated from metrics)
    com_lateral_std: float = 0.0
    com_vertical_range: float = 0.0
    symmetry_score: float = 0.0
    knee_symmetry_deg: float = 0.0
    hip_symmetry_deg: float = 0.0
    frame_count: int = 0

    @property
    def summary(self) -> str:
        lines = [
            f"{self.label} (score {self.score:.0f}/100)",
            self.explanation.split("\n")[0] if self.explanation else "",
        ]
        if self.abnormal_patterns:
            lines.append("Flags: " + ", ".join(self.abnormal_patterns))
        if self.metrics:
            m = self.metrics
            cad = f"{m.cadence_hz:.2f} Hz" if m.cadence_hz is not None else "n/a"
            lines.append(
                f"CoM lateral std={m.com_lateral_std:.4f}  "
                f"symmetry={m.symmetry_score:.0%}  cadence={cad}"
            )
        return "\n".join(lines)

    @property
    def stability_score(self) -> float:
        """Alias for ``score`` (API-friendly name)."""
        return self.score

    def to_dict(self) -> dict[str, Any]:
        """Structured report for JSON export."""
        status = self.status or ("Stable" if self.score >= STABLE_MIN_SCORE else "Unstable")
        out: dict[str, Any] = {
            "stability_score": round(self.score, 2),
            "score": round(self.score, 2),
            "status": status,
            "label": self.label,
            "is_stable": self.is_stable,
            "base_score": self.base_score,
            "penalties": [
                {"rule": p.rule_id, "points": p.points, "reason": p.reason}
                for p in self.penalties
            ],
            "total_penalty": round(sum(p.points for p in self.penalties), 2),
            "abnormal_patterns": list(self.abnormal_patterns),
            "explanation": self.explanation,
            "scoring_notes": self.scoring_notes,
            "threshold_rationale": THRESHOLD_RATIONALE,
            "metric_descriptions": self.metric_descriptions,
        }
        if self.metrics:
            out["metrics"] = self.metrics.to_dict()
        return out


class StabilityAnalyzer:
    """
    Assess gait stability from pose, contact, and estimated GRF.

    **Scoring:** starts at **100**, subtracts penalties for asymmetry, irregular
    steps, unstable CoM, and abnormal force patterns. See ``SCORING_HOW_IT_WORKS``
    and ``THRESHOLD_RATIONALE``.

    Example::

        analyzer = StabilityAnalyzer()
        report = analyzer.analyze(pose_sequence)
        print(report.to_dict())  # stability_score, status, penalties
    """

    def __init__(
        self,
        *,
        body_mass_kg: float = 70.0,
        stable_threshold: float = STABLE_MIN_SCORE,
        borderline_threshold: float = BORDERLINE_MIN_SCORE,
    ) -> None:
        self.body_mass_kg = body_mass_kg
        self.stable_threshold = stable_threshold
        self.borderline_threshold = borderline_threshold

    def analyze(
        self,
        sequence: PoseSequence,
        *,
        grf: GRFTimeSeries | None = None,
        gait_metrics: GaitMetricsReport | None = None,
        physics_result=None,
    ) -> StabilityReport:
        """
        Run full stability analysis.

        Args:
            sequence: Enriched pose sequence (gait phases recommended).
            grf: Precomputed GRF series, or computed via ``ForceAnalyzer``.
            gait_metrics: Precomputed spatiotemporal metrics, or computed here.
            physics_result: Optional ``PhysicsWalkResult`` for CoM in meters.
        """
        advanced: GaitMetricsResult | None = None
        if gait_metrics is None:
            if grf is None:
                grf = ForceAnalyzer(body_mass_kg=self.body_mass_kg).analyze(
                    sequence, physics_result=physics_result
                )
            advanced = GaitMetrics().compute(sequence, grf=grf)
            gait_metrics = GaitMetrics().to_legacy_report(advanced)
        elif grf is None:
            grf = ForceAnalyzer(body_mass_kg=self.body_mass_kg).analyze(
                sequence, physics_result=physics_result
            )

        sm = self._compute_metrics(sequence, gait_metrics, grf, physics_result, advanced)
        penalties, flags = self._compute_penalties(sm, gait_metrics, grf)
        score = self._score_from_penalties(penalties)
        label = self._classify_label(score)
        status = "Stable" if score >= self.stable_threshold else "Unstable"
        explanation = self._build_explanation(sm, penalties, flags, label, score)

        return StabilityReport(
            label=label,
            score=score,
            status=status,
            is_stable=score >= self.stable_threshold,
            base_score=BASE_SCORE,
            penalties=penalties,
            abnormal_patterns=flags,
            explanation=explanation,
            metrics=sm,
            com_lateral_std=sm.com_lateral_std,
            com_vertical_range=sm.com_vertical_range,
            symmetry_score=sm.symmetry_score,
            knee_symmetry_deg=sm.knee_symmetry_deg,
            hip_symmetry_deg=sm.hip_symmetry_deg,
            frame_count=sm.frame_count,
        )

    def _compute_metrics(
        self,
        sequence: PoseSequence,
        gait_metrics: GaitMetricsReport,
        grf: GRFTimeSeries,
        physics_result,
        advanced: GaitMetricsResult | None = None,
    ) -> StabilityMetrics:
        indices = detected_frame_indices(sequence)
        frames = [sequence.frames[i] for i in indices if sequence.frames[i].joint_angles]
        n = len(frames)

        ct = gait_metrics.contact
        swing_l = max(0.0, 1.0 - ct.left_stance_fraction)
        swing_r = max(0.0, 1.0 - ct.right_stance_fraction)

        cadence_hz = None
        if gait_metrics.cadence_steps_per_min is not None:
            cadence_hz = gait_metrics.cadence_steps_per_min / 60.0

        step_cv = (
            advanced.step_timing_cv
            if advanced and advanced.step_timing_cv is not None
            else self._step_interval_cv(sequence, gait_metrics)
        )
        stride_cv = self._stride_length_cv(sequence, gait_metrics)

        com_lat, com_vert, com_path, knee_d, hip_d = self._com_and_angles(
            sequence, frames, physics_result
        )

        sym_angle = self._angle_symmetry_score(knee_d, hip_d)
        sym_stance = ct.stance_symmetry_ratio
        sym_step = gait_metrics.step_time_symmetry or 0.5
        grf_sym = self._grf_symmetry(grf)

        if advanced is not None:
            symmetry = advanced.symmetry_score
            if advanced.com_lateral_std > 0:
                com_lat = advanced.com_lateral_std
        else:
            symmetry = float(
                np.mean(
                    [
                        sym_angle,
                        sym_stance,
                        sym_step,
                        grf_sym if grf_sym is not None else sym_stance,
                    ]
                )
            )

        return StabilityMetrics(
            symmetry_score=symmetry,
            stance_left_fraction=ct.left_stance_fraction,
            swing_left_fraction=swing_l,
            stance_right_fraction=ct.right_stance_fraction,
            swing_right_fraction=swing_r,
            cadence_hz=cadence_hz,
            cadence_steps_per_min=gait_metrics.cadence_steps_per_min,
            stride_length_normalized=gait_metrics.stride_length_normalized,
            stride_length_m=gait_metrics.stride_length_m,
            step_time_variability=step_cv,
            stride_length_variability=stride_cv,
            com_lateral_std=com_lat,
            com_vertical_range=com_vert,
            com_path_length=com_path,
            grf_symmetry=grf_sym,
            knee_symmetry_deg=knee_d,
            hip_symmetry_deg=hip_d,
            frame_count=n,
        )

    def _com_and_angles(
        self,
        sequence: PoseSequence,
        frames: list[PoseFrame],
        physics_result,
    ) -> tuple[float, float, float, float, float]:
        """CoM proxy + mean knee/hip angle asymmetry."""
        if physics_result is not None and physics_result.com_trajectory.size:
            com = physics_result.com_trajectory
            xs, ys = com[:, 0], com[:, 1]
            com_lat = float(np.std(xs))
            com_vert = float(np.max(ys) - np.min(ys)) if len(ys) else 0.0
            diffs = np.sqrt(np.sum(np.diff(com, axis=0) ** 2, axis=1))
            com_path = float(np.sum(diffs))
        else:
            com_points: list[tuple[float, float]] = []
            for f in frames:
                c = _com_xy(f)
                if c:
                    com_points.append(c)
            if len(com_points) >= 2:
                xs = [p[0] for p in com_points]
                ys = [p[1] for p in com_points]
                mean_x = sum(xs) / len(xs)
                com_lat = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / len(xs))
                com_vert = max(ys) - min(ys)
                com_path = sum(
                    math.hypot(com_points[i + 1][0] - com_points[i][0],
                               com_points[i + 1][1] - com_points[i][1])
                    for i in range(len(com_points) - 1)
                )
            else:
                com_lat, com_vert, com_path = 1.0, 1.0, 0.0

        knee_diffs: list[float] = []
        hip_diffs: list[float] = []
        for f in frames:
            a = f.joint_angles
            if not a:
                continue
            if a.left_knee is not None and a.right_knee is not None:
                knee_diffs.append(abs(a.left_knee - a.right_knee))
            if a.left_hip is not None and a.right_hip is not None:
                hip_diffs.append(abs(a.left_hip - a.right_hip))

        knee_d = sum(knee_diffs) / len(knee_diffs) if knee_diffs else 30.0
        hip_d = sum(hip_diffs) / len(hip_diffs) if hip_diffs else 30.0
        return com_lat, com_vert, com_path, knee_d, hip_d

    @staticmethod
    def _angle_symmetry_score(knee_deg: float, hip_deg: float) -> float:
        sk = max(0.0, 1.0 - knee_deg / 25.0)
        sh = max(0.0, 1.0 - hip_deg / 25.0)
        return (sk + sh) / 2.0

    @staticmethod
    def _grf_symmetry(grf: GRFTimeSeries) -> float | None:
        if not grf.left_contact.any() and not grf.right_contact.any():
            return None
        peak_l = float(np.max(grf.left_force_bw[grf.left_contact])) if grf.left_contact.any() else 0.0
        peak_r = float(np.max(grf.right_force_bw[grf.right_contact])) if grf.right_contact.any() else 0.0
        if max(peak_l, peak_r) < 1e-6:
            return None
        return min(peak_l, peak_r) / max(peak_l, peak_r)

    @staticmethod
    def _step_interval_cv(sequence: PoseSequence, gait_metrics: GaitMetricsReport) -> float | None:
        from stablewalk.pose.events import analyze_gait_sequence

        events, _ = analyze_gait_sequence(sequence.frames)
        hs = [e for e in events if e.event_type == "heel_strike"]
        if len(hs) < 3:
            return None
        fps = max(sequence.fps, 1e-6)
        intervals = []
        frames_sorted = sorted(hs, key=lambda e: e.frame_index)
        for a, b in zip(frames_sorted, frames_sorted[1:]):
            intervals.append((b.frame_index - a.frame_index) / fps)
        if not intervals:
            return None
        mean = sum(intervals) / len(intervals)
        if mean < 1e-6:
            return None
        std = math.sqrt(sum((x - mean) ** 2 for x in intervals) / len(intervals))
        return std / mean

    @staticmethod
    def _stride_length_cv(sequence: PoseSequence, gait_metrics: GaitMetricsReport) -> float | None:
        from stablewalk.pose.events import analyze_gait_sequence

        events, _ = analyze_gait_sequence(sequence.frames)
        hs = [e for e in events if e.event_type == "heel_strike"]
        if len(hs) < 3:
            return None
        frames_map = {f.frame_index: f for f in sequence.frames}
        detected_kps = [f.keypoints for f in sequence.frames if f.detected and f.keypoints]
        from stablewalk.pose.skeleton_3d import sequence_skeleton_scale

        scale = sequence_skeleton_scale(detected_kps) if detected_kps else None
        by_side: dict[str, list] = {"left": [], "right": []}
        for e in hs:
            by_side[e.side].append(e)
        norm_lens: list[float] = []
        for evs in by_side.values():
            evs.sort(key=lambda e: e.frame_index)
            for a, b in zip(evs, evs[1:]):
                fa, fb = frames_map.get(a.frame_index), frames_map.get(b.frame_index)
                if not fa or not fb:
                    continue
                pa, pb = _com_xy(fa), _com_xy(fb)
                if pa and pb:
                    d = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
                    norm_lens.append(d)
        if len(norm_lens) < 2:
            return None
        mean = sum(norm_lens) / len(norm_lens)
        if mean < 1e-8:
            return None
        std = math.sqrt(sum((x - mean) ** 2 for x in norm_lens) / len(norm_lens))
        return std / mean

    def _compute_penalties(
        self,
        sm: StabilityMetrics,
        gait_metrics: GaitMetricsReport,
        grf: GRFTimeSeries,
    ) -> tuple[list[ScorePenalty], list[str]]:
        """
        Start from 100; return list of penalties and abnormal-pattern flags.
        """
        penalties: list[ScorePenalty] = []
        flags: list[str] = []

        def add(rule_id: str, points: float, reason: str, flag: str | None = None) -> None:
            penalties.append(ScorePenalty(rule_id=rule_id, points=points, reason=reason))
            if flag:
                flags.append(flag)

        # --- Asymmetry ---
        if sm.symmetry_score < 0.70:
            add(
                "symmetry_low",
                PENALTY_SYMMETRY_LOW,
                f"Symmetry {sm.symmetry_score:.0%} below 70%",
                "asymmetric_legs",
            )
        if sm.symmetry_score < 0.55:
            add(
                "symmetry_severe",
                PENALTY_SYMMETRY_SEVERE,
                f"Symmetry {sm.symmetry_score:.0%} below 55%",
            )

        stance_balance = min(sm.stance_left_fraction, sm.stance_right_fraction) / max(
            sm.stance_left_fraction, sm.stance_right_fraction, 1e-6
        )
        if stance_balance < 0.75:
            add(
                "stance_asymmetry",
                PENALTY_STANCE_ASYMMETRY,
                f"Stance balance ratio {stance_balance:.0%} below 75%",
                "asymmetric_stance_duration",
            )

        if sm.knee_symmetry_deg > 15 or sm.hip_symmetry_deg > 15:
            add(
                "joint_asymmetry",
                PENALTY_JOINT_ASYMMETRY,
                f"Knee/hip angle asymmetry {sm.knee_symmetry_deg:.0f}° / {sm.hip_symmetry_deg:.0f}°",
                "large_joint_angle_asymmetry",
            )

        # --- Inconsistent steps ---
        if sm.step_time_variability is not None and sm.step_time_variability > 0.18:
            add(
                "step_variability",
                PENALTY_STEP_VARIABILITY,
                f"Step timing CV {sm.step_time_variability:.2f} > 0.18",
                "irregular_step_timing",
            )
        if sm.step_time_variability is not None and sm.step_time_variability > 0.25:
            add(
                "step_variability_high",
                PENALTY_STEP_VARIABILITY_HIGH,
                f"Step timing CV {sm.step_time_variability:.2f} > 0.25",
            )

        if sm.stride_length_variability is not None and sm.stride_length_variability > 0.20:
            add(
                "stride_variability",
                PENALTY_STRIDE_VARIABILITY,
                f"Stride length CV {sm.stride_length_variability:.2f} > 0.20",
                "irregular_stride_length",
            )

        # --- Unstable CoM ---
        if sm.com_lateral_std > 0.045:
            add(
                "com_sway",
                PENALTY_COM_SWAY,
                f"CoM lateral std {sm.com_lateral_std:.4f} > 0.045",
                "excessive_com_lateral_sway",
            )
        if sm.com_lateral_std > 0.060:
            add(
                "com_sway_severe",
                PENALTY_COM_SWAY_SEVERE,
                f"CoM lateral std {sm.com_lateral_std:.4f} > 0.060",
            )
        if sm.com_vertical_range > 0.14 or sm.com_vertical_range < 0.02:
            add(
                "com_vertical",
                PENALTY_COM_VERTICAL,
                f"CoM vertical range {sm.com_vertical_range:.4f} outside 0.02–0.14",
                "abnormal_com_vertical_motion",
            )

        # --- Abnormal force patterns ---
        if sm.grf_symmetry is not None and sm.grf_symmetry < 0.65:
            add(
                "grf_asymmetry",
                PENALTY_GRF_ASYMMETRY,
                f"Peak GRF symmetry {sm.grf_symmetry:.0%} below 65%",
                "asymmetric_ground_forces",
            )

        has_stances = bool(grf.double_peak_left or grf.double_peak_right)
        if has_stances and grf.double_peak_fraction < 0.2:
            add(
                "grf_pattern",
                PENALTY_GRF_PATTERN,
                f"Double-peak stances {grf.double_peak_fraction:.0%} below 20%",
                "atypical_force_pattern",
            )

        # --- Cadence / data quality ---
        if sm.cadence_hz is not None and (sm.cadence_hz < 0.8 or sm.cadence_hz > 2.2):
            add(
                "cadence",
                PENALTY_CADENCE,
                f"Cadence {sm.cadence_hz:.2f} Hz outside 0.8–2.2",
                "slow_cadence" if sm.cadence_hz < 0.8 else "fast_cadence",
            )

        if sm.frame_count < 10:
            add(
                "few_frames",
                PENALTY_FEW_FRAMES,
                f"Only {sm.frame_count} analyzed frames",
                "insufficient_frames",
            )

        return penalties, flags

    @staticmethod
    def _score_from_penalties(penalties: list[ScorePenalty]) -> float:
        total_penalty = sum(p.points for p in penalties)
        return max(0.0, min(100.0, BASE_SCORE - total_penalty))

    def _classify_label(self, score: float) -> str:
        if score >= self.stable_threshold:
            return "Stable"
        if score >= self.borderline_threshold:
            return "Borderline"
        return "Unstable"

    def _build_explanation(
        self,
        sm: StabilityMetrics,
        penalties: list[ScorePenalty],
        flags: list[str],
        label: str,
        score: float,
    ) -> str:
        total_pen = sum(p.points for p in penalties)
        parts = [
            f"Gait classified as {label} (stability score {score:.0f}/100).",
            f"Scoring: start {BASE_SCORE:.0f}, subtract penalties ({total_pen:.0f} total) → {score:.0f}.",
            "",
            "Penalties applied:" if penalties else "No penalties applied.",
        ]
        for p in penalties:
            parts.append(f"  - {p.reason}: −{p.points:.0f}")
        parts.extend([
            "",
            "Metrics:",
            f"  1. Symmetry ({sm.symmetry_score:.0%}): balance between left and right legs "
            f"(stance, angles, steps, forces).",
            f"  2. Stance vs swing: left stance {sm.stance_left_fraction:.0%} / swing {sm.swing_left_fraction:.0%}; "
            f"right stance {sm.stance_right_fraction:.0%} / swing {sm.swing_right_fraction:.0%}.",
        ])
        if sm.cadence_hz is not None:
            parts.append(
                f"  3. Cadence: {sm.cadence_hz:.2f} steps/s ({sm.cadence_steps_per_min:.0f} steps/min)."
            )
        else:
            parts.append("  3. Cadence: not enough heel strikes to estimate.")
        if sm.stride_length_normalized is not None:
            sl = f"{sm.stride_length_normalized:.3f} (normalized)"
            if sm.stride_length_m:
                sl += f", ~{sm.stride_length_m:.2f} m"
            parts.append(f"  4. Stride length: mean {sl}.")
        else:
            parts.append("  4. Stride length: unavailable.")
        if sm.step_time_variability is not None:
            parts.append(
                f"  5. Step variability: CV={sm.step_time_variability:.2f} "
                f"(lower is more regular)."
            )
        else:
            parts.append("  5. Step variability: unavailable.")
        parts.append(
            f"  6. CoM movement: lateral spread={sm.com_lateral_std:.4f}, "
            f"vertical range={sm.com_vertical_range:.4f}, path length={sm.com_path_length:.4f}."
        )
        if flags:
            parts.append("")
            parts.append("Abnormal patterns detected:")
            for f in flags:
                parts.append(f"  - {f.replace('_', ' ')}")
        else:
            parts.append("")
            parts.append("No strong abnormal pattern flags.")
        return "\n".join(parts)


def _com_xy(frame: PoseFrame) -> tuple[float, float] | None:
    kp = {k.name: k for k in frame.keypoints}
    points: list[tuple[float, float]] = []
    for name in ("mid_hip", "left_hip", "right_hip", "mid_shoulder"):
        if name in kp and kp[name].visibility >= 0.3:
            points.append((kp[name].x, kp[name].y))
    if not points and "left_hip" in kp and "right_hip" in kp:
        lh, rh = kp["left_hip"], kp["right_hip"]
        if lh.visibility >= 0.3 and rh.visibility >= 0.3:
            points.append(((lh.x + rh.x) / 2, (lh.y + rh.y) / 2))
    if not points:
        return None
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def analyze_stability(sequence: PoseSequence) -> StabilityReport:
    """Backward-compatible entry point using ``StabilityAnalyzer``."""
    return StabilityAnalyzer().analyze(sequence)
