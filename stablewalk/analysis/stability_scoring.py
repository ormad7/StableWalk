"""
Redesigned gait stability scoring (v2).

Eight interpretable domain sub-scores (0–100) combined via ``StabilityScoreConfig``.
Uses gait-cycle analysis, body-normalized spatial metrics, and irregularity-based
smoothness (high athletic amplitude is not penalized when temporally consistent).
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.gait_cycle_analysis import (
    GaitCycleAnalysisResult,
    GaitEvent,
    analyze_gait_cycles,
    symmetry_ratio,
)
from stablewalk.analysis.controlled_motion import (
    _controlled_motion_without_repeatability,
    analyze_controlled_motion,
)
from stablewalk.analysis.gait_feature_analysis import (
    FeatureNormalization,
    GaitFeatureAnalysisResult,
    analyze_gait_features,
    estimate_body_segment_dimensions,
    resample_cycle_trajectory,
    symmetry_index,
)
from stablewalk.analysis.stability_metrics import continuous_low_better_score
from stablewalk.analysis.ground_reference import (
    estimate_ground_plane,
    vertical_coordinate,
)
from stablewalk.analysis.gait_evidence import (
    GaitEvidenceAssessment,
    assess_gait_evidence,
)
from stablewalk.analysis.gait_view_analysis import (
    GaitViewEstimate,
    ViewReliabilityProfile,
    analyze_gait_view_geometry,
    apply_view_reliability,
)
from stablewalk.analysis.motion_frames import (
    MotionFrameSeries,
    build_motion_frame_series,
    compute_gait_frame_pelvis_metrics,
    compute_pelvis_motion_comparison,
    compute_trunk_gait_frame_metrics,
    forward_step_displacement,
    root_relative_position_jerk,
    swing_foot_progression_amplitude,
)
from stablewalk.analysis.stability_config import (
    DEFAULT_STABILITY_CONFIG,
    StabilityScoreConfig,
)
from stablewalk.models.gait_motion import GaitMotionRecording, Vec3
from stablewalk.models.pose_data import PoseFrame, PoseSequence
from stablewalk.pose.enrichment import enrich_pose_sequence

if TYPE_CHECKING:
    from stablewalk.analysis.gait_analysis_summary import GaitAnalysisSummary
    from stablewalk.analysis.stability_validity import StabilityResultValidity

logger = logging.getLogger(__name__)

SCORING_NOTES_V2 = (
    "Stability v2 combines eight gait domains (each 0–100): temporal symmetry, "
    "spatial symmetry, pelvis stability, trunk stability, foot clearance consistency, "
    "joint motion smoothness, gait cycle consistency, and contact pattern quality. "
    "Scores reflect control quality — asymmetry, variability, smoothness, repeatability, "
    "and contact timing — not movement magnitude. Large knee/hip ROM or forward velocity "
    "does not directly reduce stability. Joint smoothness uses gait-cycle-normalized "
    "Controlled Motion Scores (repeatability, jerk, L/R coordination, spike detection). "
    "Temporal symmetry uses continuous asymmetry penalties (no threshold saturation at 100). "
    "Pelvis/trunk use progression-relative gait-frame coordinates. "
    "Gait-cycle evidence gates domain availability: 0 usable cycles → repeatability "
    "UNAVAILABLE; 1 → single-cycle metrics only; 2 → LOW_CONFIDENCE repeatability; "
    "3+ → normal (see StabilityScoreConfig.gait_evidence). Short clips reduce "
    "confidence and exclude unsupported metrics — no fallback perfect scores. "
    "Classification: Stable >= 70, Moderate 45–69, Unstable < 45."
)

# Gait category labels must never be scoring inputs (validation / tests enforce this).
FORBIDDEN_STABILITY_PARAMETERS = frozenset(
    {
        "gait_category",
        "gait_label",
        "ground_truth_label",
        "demo_key",
        "category",
        "gait_type",
        "pathology_label",
        "gui_category",
        "comparison_label",
    }
)

LABEL_FREE_PIPELINE_ASSERTION = (
    "Stability scoring accepts PoseSequence motion data only; "
    "gait category labels must not be passed as parameters."
)

STABLE_MIN = DEFAULT_STABILITY_CONFIG.stable_min
MODERATE_MIN = DEFAULT_STABILITY_CONFIG.moderate_min

DomainAvailability = Literal["AVAILABLE", "LOW_CONFIDENCE", "UNAVAILABLE"]


@dataclass
class DomainContribution:
    """One domain's role in the weighted overall stability score."""

    key: str
    name: str
    score: float | None
    weight: float
    confidence: float
    availability: DomainAvailability
    contribution: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "score": None if self.score is None else round(self.score, 1),
            "weight": self.weight,
            "confidence": round(self.confidence, 2),
            "availability": self.availability,
            "contribution": round(self.contribution, 2),
        }


@dataclass
class MetricResult:
    """One transparent metric group: a 0-100 score plus a plain explanation."""

    key: str
    name: str
    score: float | None
    weight: float
    summary: str
    findings: list[str] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)
    availability: DomainAvailability = "UNAVAILABLE"
    confidence: float = 0.0
    contribution: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "score": None if self.score is None else round(self.score, 1),
            "weight": self.weight,
            "summary": self.summary,
            "findings": list(self.findings),
            "values": self.values,
            "availability": self.availability,
            "confidence": round(self.confidence, 2),
            "contribution": round(self.contribution, 2),
        }


@dataclass
class StabilityResult:
    """Full explainable stability assessment."""

    score: float
    classification: str
    metrics: list[MetricResult]
    primary_issue: str | None
    explanation: str
    frame_count: int
    scoring_notes: str = SCORING_NOTES_V2
    completeness_pct: float = 0.0
    confidence_badge: str = "LOW CONFIDENCE"
    contributions: list[DomainContribution] = field(default_factory=list)
    data_limitations: list[str] = field(default_factory=list)
    view_type: str | None = None
    view_confidence: float = 0.0
    view_display_name: str | None = None
    view_reliability_table: list[tuple[str, str]] = field(default_factory=list)
    analysis_evidence: dict[str, Any] = field(default_factory=dict)
    domain_evidence: dict[str, str] = field(default_factory=dict)
    usable_gait_cycles: int = 0
    video_duration_s: float = 0.0
    repeatability_tier: str = "UNAVAILABLE"
    validity: StabilityResultValidity | None = None
    gait_summary: GaitAnalysisSummary | None = None

    @property
    def is_stable(self) -> bool:
        return self.classification == "Stable"

    def metric(self, key: str) -> MetricResult | None:
        return next((m for m in self.metrics if m.key == key), None)

    def contribution_table_text(self) -> str:
        lines = [
            f"{'Domain':<24}{'Score':>7}{'Weight':>8}{'Conf':>7}{'Status':>16}{'Contrib':>10}",
            "-" * 72,
        ]
        for row in self.contributions:
            score_s = "N/A" if row.score is None else f"{row.score:.0f}"
            lines.append(
                f"{row.name:<24}{score_s:>7}{row.weight:>8.2f}{row.confidence:>7.2f}"
                f"{row.availability:>16}{row.contribution:>10.2f}"
            )
        lines.append("")
        lines.append(f"Analysis completeness: {self.completeness_pct:.0f}%")
        lines.append(f"Confidence badge: {self.confidence_badge}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "classification": self.classification,
            "primary_issue": self.primary_issue,
            "frame_count": self.frame_count,
            "metrics": [m.to_dict() for m in self.metrics],
            "explanation": self.explanation,
            "scoring_notes": self.scoring_notes,
            "completeness_pct": round(self.completeness_pct, 1),
            "confidence_badge": self.confidence_badge,
            "contributions": [c.to_dict() for c in self.contributions],
            "data_limitations": list(self.data_limitations),
            "view_type": self.view_type,
            "view_confidence": round(self.view_confidence, 3),
            "view_display_name": self.view_display_name,
            "view_reliability_table": [
                {"metric": name, "reliability": tier}
                for name, tier in self.view_reliability_table
            ],
            "analysis_evidence": self.analysis_evidence,
            "domain_evidence": dict(self.domain_evidence),
            "usable_gait_cycles": self.usable_gait_cycles,
            "video_duration_s": round(self.video_duration_s, 3),
            "repeatability_tier": self.repeatability_tier,
            "validity": None if self.validity is None else self.validity.to_dict(),
            "gait_summary": None if self.gait_summary is None else self.gait_summary.to_dict(),
        }


@dataclass
class Anthropometry:
    """Body-scale references derived from pose (not clinical tape measure)."""

    body_height: float
    hip_width: float
    shoulder_width: float
    leg_length_left: float
    leg_length_right: float

    @property
    def leg_length_mean(self) -> float:
        return (self.leg_length_left + self.leg_length_right) / 2.0


@dataclass
class StabilityDebugRecord:
    """One raw or normalized feature with scoring contribution."""

    domain: str
    feature: str
    raw_value: float | None
    normalized_value: float | None
    direction: str
    sub_score_component: float | None
    weight_share: float | None = None
    note: str = ""


@dataclass
class StabilityAnalysisContext:
    sequence: PoseSequence
    recording: GaitMotionRecording
    cycles: GaitCycleAnalysisResult
    anthro: Anthropometry
    gait_features: GaitFeatureAnalysisResult | None
    frames: list[PoseFrame]
    fps: float
    ik_available: bool = False
    motion_frames: MotionFrameSeries | None = None
    view_estimate: GaitViewEstimate | None = None
    view_reliability: ViewReliabilityProfile | None = None
    evidence: GaitEvidenceAssessment | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _robust_range(values: Sequence[float | None]) -> float:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size < 3:
        return float(np.ptp(arr)) if arr.size else 0.0
    return float(np.percentile(arr, 95) - np.percentile(arr, 5))


def _cv(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(clean) < 2:
        return None
    mean = statistics.mean(clean)
    if abs(mean) < 1e-9:
        return None
    val = statistics.pstdev(clean) / abs(mean)
    return val if math.isfinite(val) else None


def _safe_gradient(y: np.ndarray, times: Sequence[float], *, fps: float) -> np.ndarray:
    if len(y) < 2:
        return np.zeros_like(y)
    if len(times) >= 2 and (max(times) - min(times)) > 1e-9:
        return np.gradient(y, np.asarray(times, dtype=float))
    return np.gradient(y, 1.0 / max(fps, 1e-6))


def _detrend(arr: np.ndarray) -> np.ndarray:
    idx = np.arange(len(arr), dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return arr.copy()
    coef = np.polyfit(idx[valid], arr[valid], 1)
    return arr - np.polyval(coef, idx)


def _mean_jerk(series: Sequence[float | None]) -> float | None:
    arr = np.asarray([v for v in series if v is not None], dtype=float)
    if arr.size < 5:
        return None
    return float(np.mean(np.abs(np.diff(arr, n=2))))


def _get_motion_frames(ctx: StabilityAnalysisContext) -> MotionFrameSeries:
    if ctx.motion_frames is None:
        ctx.motion_frames = build_motion_frame_series(ctx.frames, ctx.recording)
    return ctx.motion_frames


def compute_pelvis_motion_metrics(
    ctx: StabilityAnalysisContext,
) -> dict[str, float | None]:
    """
    Pelvis kinematics in the local gait frame (global trajectory projected).

    Forward walking progression is reported but not used as an instability metric.
    """
    series = _get_motion_frames(ctx)
    metrics = compute_gait_frame_pelvis_metrics(
        series,
        hip_width=ctx.anthro.hip_width,
        fps=ctx.fps,
    )
    comparison = compute_pelvis_motion_comparison(series)
    metrics.update(comparison)
    return metrics


def _keypoint_map(frame: PoseFrame) -> dict[str, Any]:
    return {kp.name: kp for kp in frame.keypoints}


def _pelvis_xy(frame: PoseFrame) -> tuple[float, float] | None:
    kp = _keypoint_map(frame)
    lh, rh = kp.get("left_hip"), kp.get("right_hip")
    if lh and rh and lh.visibility >= 0.3 and rh.visibility >= 0.3:
        return (lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0
    mid = kp.get("mid_hip")
    if mid and mid.visibility >= 0.3:
        return float(mid.x), float(mid.y)
    return None


def _shoulder_mid(frame: PoseFrame) -> tuple[float, float] | None:
    kp = _keypoint_map(frame)
    ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
    if ls and rs and ls.visibility >= 0.3 and rs.visibility >= 0.3:
        return (ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0
    return None


def _anthropometry_from_dimensions(dim) -> Anthropometry:
    return Anthropometry(
        body_height=max(dim.leg_length_average * 2.2, dim.hip_width * 2.0),
        hip_width=dim.hip_width,
        shoulder_width=dim.shoulder_width,
        leg_length_left=dim.leg_length_left,
        leg_length_right=dim.leg_length_right,
    )


def _estimate_anthropometry(frames: list[PoseFrame]) -> Anthropometry:
    heights, hip_w, shoulder_w = [], [], []
    leg_l, leg_r = [], []
    for f in frames:
        kp = _keypoint_map(f)
        ys = [p.y for p in kp.values() if p.visibility >= 0.3]
        if len(ys) >= 4:
            heights.append(max(ys) - min(ys))
        lh, rh = kp.get("left_hip"), kp.get("right_hip")
        if lh and rh and lh.visibility >= 0.3 and rh.visibility >= 0.3:
            hip_w.append(abs(lh.x - rh.x))
        ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
        if ls and rs and ls.visibility >= 0.3 and rs.visibility >= 0.3:
            shoulder_w.append(abs(ls.x - rs.x))

        def leg_dist(side: str) -> None:
            hip = kp.get(f"{side}_hip")
            ankle = kp.get(f"{side}_ankle")
            if hip and ankle and hip.visibility >= 0.3 and ankle.visibility >= 0.3:
                d = math.hypot(hip.x - ankle.x, hip.y - ankle.y)
                (leg_l if side == "left" else leg_r).append(d)

        leg_dist("left")
        leg_dist("right")

    return Anthropometry(
        body_height=float(np.median(heights)) if heights else 0.5,
        hip_width=float(np.median(hip_w)) if hip_w else 0.2,
        shoulder_width=float(np.median(shoulder_w)) if shoulder_w else 0.25,
        leg_length_left=float(np.median(leg_l)) if leg_l else 0.22,
        leg_length_right=float(np.median(leg_r)) if leg_r else 0.22,
    )


def _build_context(
    sequence: PoseSequence,
    *,
    cycles: GaitCycleAnalysisResult | None = None,
    recording: GaitMotionRecording | None = None,
) -> StabilityAnalysisContext | None:
    frames = [f for f in sequence.frames if f.detected and f.joint_angles]
    if len(frames) < DEFAULT_STABILITY_CONFIG.min_frames:
        return None
    rec = recording or pose_sequence_to_gait_motion(sequence)
    cyc = cycles or analyze_gait_cycles(rec)
    dims = estimate_body_segment_dimensions(rec)
    anthro = _anthropometry_from_dimensions(dims)
    ik_mot = None
    src_video = sequence.source_video or ""
    if src_video:
        from pathlib import Path

        from stablewalk import config

        stem = Path(src_video).stem
        candidate = config.OPENSIM_DIR / stem / f"{stem}_ik.mot"
        if candidate.is_file():
            ik_mot = candidate
    gait_features = analyze_gait_features(
        rec,
        cyc,
        sequence=sequence,
        ik_mot_path=ik_mot,
    )
    ctx = StabilityAnalysisContext(
        sequence=sequence,
        recording=rec,
        cycles=cyc,
        anthro=anthro,
        gait_features=gait_features,
        frames=frames,
        fps=max(sequence.fps, 1e-6),
        ik_available=ik_mot is not None and gait_features.cycle_consistency.angle_source == "opensim_ik",
    )
    ctx.motion_frames = build_motion_frame_series(frames, rec)
    view_est, view_profile = analyze_gait_view_geometry(
        frames,
        motion_frames=ctx.motion_frames,
        body_height=anthro.body_height,
    )
    ctx.view_estimate = view_est
    ctx.view_reliability = view_profile
    ctx.evidence = assess_gait_evidence(
        sequence,
        cyc,
        stability_frame_count=len(frames),
        thresholds=DEFAULT_STABILITY_CONFIG.gait_evidence,
    )
    return ctx


def _metric(
    key: str,
    name: str,
    score: float | None,
    weight: float,
    summary: str,
    findings: list[str],
    values: dict[str, Any],
    debug: list[StabilityDebugRecord],
) -> MetricResult:
    values = dict(values)
    values["debug_features"] = [
        {
            "domain": d.domain,
            "feature": d.feature,
            "raw": d.raw_value,
            "normalized": d.normalized_value,
            "direction": d.direction,
            "component_score": d.sub_score_component,
            "note": d.note,
        }
        for d in debug
    ]
    return MetricResult(
        key=key,
        name=name,
        score=score,
        weight=weight,
        summary=summary,
        findings=findings,
        values=values,
    )


# ---------------------------------------------------------------------------
# Domain scorers
# ---------------------------------------------------------------------------
def score_temporal_symmetry(
    ctx: StabilityAnalysisContext,
    cfg: StabilityScoreConfig,
) -> tuple[float | None, list[str], dict[str, Any], list[StabilityDebugRecord]]:
    m = ctx.cycles.metrics
    debug: list[StabilityDebugRecord] = []
    findings: list[str] = []
    cycle_count = len(ctx.cycles.cycles)

    hs_count = m.left_heel_strike_count + m.right_heel_strike_count
    if hs_count < 2:
        return None, ["Too few heel-strike events for temporal symmetry."], {}, debug

    asym_parts: list[float] = []
    component_scores: list[float] = []
    penalties: list[float] = []

    stance_sym = symmetry_ratio(m.left_stance_time_s, m.right_stance_time_s)
    if stance_sym is not None:
        asym = 1.0 - stance_sym
        penalty, s = continuous_low_better_score(asym, steepness=7.0, reference=0.22)
        asym_parts.append(asym)
        penalties.append(penalty)
        component_scores.append(s)
        debug.append(
            StabilityDebugRecord(
                "temporal_symmetry", "stance_duration_symmetry", stance_sym, asym,
                "continuous asymmetry penalty", s, note="L/R mean stance time ratio",
            )
        )

    swing_sym = symmetry_ratio(m.left_swing_time_s, m.right_swing_time_s)
    if swing_sym is not None:
        asym = 1.0 - swing_sym
        penalty, s = continuous_low_better_score(asym, steepness=7.0, reference=0.22)
        asym_parts.append(asym)
        penalties.append(penalty)
        component_scores.append(s)
        debug.append(
            StabilityDebugRecord(
                "temporal_symmetry", "swing_duration_symmetry", swing_sym, asym,
                "continuous asymmetry penalty", s,
            )
        )

    left_hs = sorted(e.time_s for e in ctx.cycles.events if e.event_type == "left_heel_strike")
    right_hs = sorted(e.time_s for e in ctx.cycles.events if e.event_type == "right_heel_strike")
    left_steps = [left_hs[i + 1] - left_hs[i] for i in range(len(left_hs) - 1)]
    right_steps = [right_hs[i + 1] - right_hs[i] for i in range(len(right_hs) - 1)]
    step_asym = None
    if left_steps and right_steps:
        mean_l, mean_r = statistics.mean(left_steps), statistics.mean(right_steps)
        step_asym = abs(mean_l - mean_r) / max((mean_l + mean_r) / 2.0, 1e-6)
        penalty, s = continuous_low_better_score(step_asym, steepness=6.5, reference=0.20)
        asym_parts.append(step_asym)
        penalties.append(penalty)
        component_scores.append(s)
        debug.append(
            StabilityDebugRecord(
                "temporal_symmetry", "step_time_asymmetry", step_asym, step_asym,
                "continuous asymmetry penalty", s,
            )
        )

    if not component_scores:
        return None, ["Insufficient heel-strike events for temporal symmetry."], {}, debug

    raw_temporal_asymmetry = float(statistics.mean(asym_parts))
    normalized_temporal_penalty = float(statistics.mean(penalties))
    _, final_from_combined = continuous_low_better_score(
        raw_temporal_asymmetry, steepness=7.5, reference=0.20,
    )
    score = float(statistics.mean(component_scores))

    insufficient_cycles = (
        ctx.evidence is not None
        and ctx.evidence.usable_gait_cycles < cfg.gait_evidence.min_usable_cycles_normal
    ) or cycle_count < cfg.gait_evidence.min_usable_cycles_low_confidence
    if insufficient_cycles and ctx.evidence:
        findings.append(
            f"{ctx.evidence.usable_gait_cycles} usable gait cycle(s) — temporal timing "
            f"has reduced reliability (need {cfg.gait_evidence.min_usable_cycles_normal} for normal tier)."
        )
    if hs_count < 4:
        findings.append("Few heel strikes — temporal timing estimates are limited.")

    if stance_sym is not None and stance_sym < 0.75:
        findings.append("Asymmetric left/right stance duration.")
    if swing_sym is not None and swing_sym < 0.75:
        findings.append("Asymmetric left/right swing duration.")

    values = {
        "left_stance_s": m.left_stance_time_s,
        "right_stance_s": m.right_stance_time_s,
        "left_swing_s": m.left_swing_time_s,
        "right_swing_s": m.right_swing_time_s,
        "stance_symmetry": stance_sym,
        "swing_symmetry": swing_sym,
        "step_time_asymmetry": step_asym,
        "raw_temporal_asymmetry": raw_temporal_asymmetry,
        "normalized_temporal_penalty": normalized_temporal_penalty,
        "final_temporal_score": score,
        "combined_continuous_score": final_from_combined,
        "gait_cycle_count": cycle_count,
        "heel_strike_count": hs_count,
        "insufficient_cycles": insufficient_cycles,
    }
    debug.append(
        StabilityDebugRecord(
            "temporal_symmetry", "raw_temporal_asymmetry", raw_temporal_asymmetry,
            raw_temporal_asymmetry, "lower better", score,
        )
    )
    debug.append(
        StabilityDebugRecord(
            "temporal_symmetry", "normalized_temporal_penalty", normalized_temporal_penalty,
            normalized_temporal_penalty, "lower better", score,
        )
    )
    debug.append(
        StabilityDebugRecord(
            "temporal_symmetry", "final_temporal_score", score, score,
            "continuous control score", score,
        )
    )
    return score, findings, values, debug


def _step_displacements(
    ctx: StabilityAnalysisContext,
    side: str,
) -> list[float]:
    """Pelvis forward (gait-frame) displacement between same-side heel strikes / leg length."""
    events = [
        e for e in ctx.cycles.events
        if e.event_type == f"{side}_heel_strike"
    ]
    if len(events) < 2:
        return []
    leg = ctx.anthro.leg_length_left if side == "left" else ctx.anthro.leg_length_right
    series = _get_motion_frames(ctx)
    indices = [e.frame_index for e in events]
    return forward_step_displacement(series, indices, leg_length=leg)


def score_spatial_symmetry(
    ctx: StabilityAnalysisContext,
    cfg: StabilityScoreConfig,
) -> tuple[float | None, list[str], dict[str, Any], list[StabilityDebugRecord]]:
    debug: list[StabilityDebugRecord] = []
    parts: list[float] = []
    findings: list[str] = []

    step_l = _step_displacements(ctx, "left")
    step_r = _step_displacements(ctx, "right")
    if step_l and step_r:
        mean_l, mean_r = statistics.mean(step_l), statistics.mean(step_r)
        asym = abs(mean_l - mean_r) / max((mean_l + mean_r) / 2.0, 1e-6)
        s = cfg.spatial_step_length_asymmetry.score(asym)
        parts.append(s)
        debug.append(
            StabilityDebugRecord(
                "spatial_symmetry", "norm_step_length_asymmetry", asym, asym,
                "lower better", s, note="pelvis Δx / leg length per step",
            )
        )

    stride_l = [
        step_l[i + 1] for i in range(len(step_l) - 1)
    ] if len(step_l) >= 2 else []
    stride_r = [
        step_r[i + 1] for i in range(len(step_r) - 1)
    ] if len(step_r) >= 2 else []
    if stride_l and stride_r:
        asym = abs(statistics.mean(stride_l) - statistics.mean(stride_r)) / max(
            (statistics.mean(stride_l) + statistics.mean(stride_r)) / 2.0, 1e-6
        )
        s = cfg.spatial_stride_length_asymmetry.score(asym)
        parts.append(s)
        debug.append(
            StabilityDebugRecord(
                "spatial_symmetry", "norm_stride_length_asymmetry", asym, asym,
                "lower better", s,
            )
        )

    # Foot progression amplitude during swing (root-relative, gait-frame forward / leg length).
    series = _get_motion_frames(ctx)
    amp_left: list[float] = []
    amp_right: list[float] = []
    for state in ctx.cycles.per_frame:
        for side, store, contact in (
            ("left", amp_left, state.left_contact),
            ("right", amp_right, state.right_contact),
        ):
            if contact:
                continue
            amp = swing_foot_progression_amplitude(series, state.frame_index, side)
            if amp is not None:
                leg = ctx.anthro.leg_length_left if side == "left" else ctx.anthro.leg_length_right
                store.append(amp / max(leg, 1e-6))

    if amp_left and amp_right:
        mean_l, mean_r = statistics.mean(amp_left), statistics.mean(amp_right)
        asym = abs(mean_l - mean_r) / max((mean_l + mean_r) / 2.0, 1e-6)
        s = cfg.spatial_foot_progression_asymmetry.score(asym)
        parts.append(s)
        debug.append(
            StabilityDebugRecord(
                "spatial_symmetry", "foot_progression_asymmetry", asym, asym,
                "lower better", s,
            )
        )
        if asym > 0.25:
            findings.append("Left/right foot progression amplitude differs.")

    if not parts:
        return None, ["Not enough spatial stride data."], {}, debug

    return float(np.mean(parts)), findings, {
        "norm_step_length_left_mean": statistics.mean(step_l) if step_l else None,
        "norm_step_length_right_mean": statistics.mean(step_r) if step_r else None,
    }, debug


def score_pelvis_stability(
    ctx: StabilityAnalysisContext,
    cfg: StabilityScoreConfig,
) -> tuple[float | None, list[str], dict[str, Any], list[StabilityDebugRecord]]:
    debug: list[StabilityDebugRecord] = []
    parts: list[float] = []
    findings: list[str] = []

    pelvis_metrics = compute_pelvis_motion_metrics(ctx)
    if pelvis_metrics.get("tracked_frame_ratio", 0.0) < cfg.min_frames / max(len(ctx.frames), 1):
        return None, ["Insufficient pelvis tracking."], pelvis_metrics, debug

    lat_ratio = pelvis_metrics.get("normalized_pelvis_sway")
    if lat_ratio is None:
        return None, ["Insufficient pelvis tracking."], pelvis_metrics, debug

    s_lat = cfg.pelvis_lateral_sway.score(float(lat_ratio))
    parts.append(s_lat)
    debug.append(
        StabilityDebugRecord(
            "pelvis_stability", "mediolateral_sway_ratio", lat_ratio, lat_ratio,
            "lower better", s_lat,
        )
    )
    if lat_ratio > cfg.pelvis_lateral_sway.poor * 0.6:
        findings.append("Increased mediolateral pelvis sway.")

    vert_range = pelvis_metrics.get("pelvis_vertical_range")
    if vert_range is not None:
        width = max(ctx.anthro.hip_width, 1e-6)
        vert_norm = float(vert_range) / width
        s_vert = cfg.pelvis_vertical_oscillation_cv.score(vert_norm)
        parts.append(s_vert)
        debug.append(
            StabilityDebugRecord(
                "pelvis_stability", "vertical_oscillation_normalized", vert_norm, vert_norm,
                "lower better", s_vert, note="gait-frame vertical std / hip width",
            )
        )

    vel_cv = pelvis_metrics.get("pelvis_acceleration_variability")
    if vel_cv is not None:
        s_vel = cfg.pelvis_velocity_inconsistency.score(float(vel_cv))
        parts.append(s_vel)
        debug.append(
            StabilityDebugRecord(
                "pelvis_stability", "mediolateral_acceleration_cv", vel_cv, vel_cv,
                "lower better", s_vel, note="ML acceleration irregularity, not forward",
            )
        )
    elif pelvis_metrics.get("pelvis_velocity_variability") is not None:
        vel_cv = pelvis_metrics["pelvis_velocity_variability"]
        s_vel = cfg.pelvis_velocity_inconsistency.score(float(vel_cv))
        parts.append(s_vel)
        debug.append(
            StabilityDebugRecord(
                "pelvis_stability", "mediolateral_velocity_cv", vel_cv, vel_cv,
                "lower better", s_vel, note="ML velocity irregularity, not forward",
            )
        )

    jerk = pelvis_metrics.get("pelvis_jerk_metric")
    width = max(ctx.anthro.hip_width, 1e-6)
    if jerk is not None:
        norm_j = jerk / width
        s_j = cfg.pelvis_jerk_normalized.score(norm_j)
        parts.append(s_j)
        debug.append(
            StabilityDebugRecord(
                "pelvis_stability", "lateral_jerk_normalized", norm_j, norm_j,
                "lower better", s_j,
            )
        )

    score = float(np.mean(parts))
    values = dict(pelvis_metrics)
    values["lateral_sway_ratio"] = lat_ratio
    values["pelvis_score"] = score
    return score, findings, values, debug


def score_trunk_stability(
    ctx: StabilityAnalysisContext,
    cfg: StabilityScoreConfig,
) -> tuple[float | None, list[str], dict[str, Any], list[StabilityDebugRecord]]:
    debug: list[StabilityDebugRecord] = []
    parts: list[float] = []
    findings: list[str] = []

    series = _get_motion_frames(ctx)
    trunk = compute_trunk_gait_frame_metrics(
        series,
        shoulder_width=ctx.anthro.shoulder_width,
    )
    if not trunk:
        return None, ["Insufficient trunk landmarks."], {}, debug

    ratio = trunk.get("trunk_lateral_sway_ratio")
    if ratio is not None and trunk.get("trunk_frame_count", 0) >= cfg.min_frames:
        s = cfg.trunk_lateral_sway.score(float(ratio))
        parts.append(s)
        debug.append(
            StabilityDebugRecord(
                "trunk_stability", "root_relative_ml_sway_ratio", ratio, ratio,
                "lower better", s, note="shoulder center in gait frame / shoulder width",
            )
        )
        if ratio > cfg.trunk_lateral_sway.poor * 0.55:
            findings.append("Increased trunk lateral sway relative to pelvis.")

    tilt_std = trunk.get("trunk_lean_variation_deg")
    if tilt_std is not None and trunk.get("trunk_frame_count", 0) >= cfg.min_frames:
        s = cfg.trunk_lean_variation_deg.score(float(tilt_std))
        parts.append(s)
        debug.append(
            StabilityDebugRecord(
                "trunk_stability", "trunk_lean_variation_deg", tilt_std, tilt_std,
                "lower better", s, note="sagittal lean in gait frame",
            )
        )

    osc = trunk.get("trunk_upper_oscillation")
    if osc is not None and trunk.get("trunk_frame_count", 0) >= cfg.min_frames:
        s = cfg.trunk_upper_oscillation.score(float(osc))
        parts.append(s)
        debug.append(
            StabilityDebugRecord(
                "trunk_stability", "upper_body_oscillation", osc, osc,
                "lower better", s, note="root-relative shoulder excursion",
            )
        )

    if not parts:
        return None, ["Insufficient trunk landmarks."], {}, debug
    return float(np.mean(parts)), findings, trunk, debug


def score_foot_clearance(
    ctx: StabilityAnalysisContext,
    cfg: StabilityScoreConfig,
) -> tuple[float | None, list[str], dict[str, Any], list[StabilityDebugRecord]]:
    debug: list[StabilityDebugRecord] = []
    parts: list[float] = []
    findings: list[str] = []

    swing_max: dict[str, float | None] = {"left": None, "right": None}
    swing_all: dict[str, list[float]] = {"left": [], "right": []}
    toe_drags = 0

    for state in ctx.cycles.per_frame:
        for side, sample in (("left", state.left), ("right", state.right)):
            if state.left_contact if side == "left" else state.right_contact:
                continue
            c = sample.foot_clearance_m
            if c is None:
                continue
            swing_all[side].append(c)
            if c < cfg.toe_drag_clearance_m:
                toe_drags += 1

    for side in ("left", "right"):
        vals = swing_all[side]
        if vals:
            swing_max[side] = max(vals)

    # L/R asymmetry of peak clearance (not absolute clearance level).
    if swing_max["left"] is not None and swing_max["right"] is not None:
        asym = abs(swing_max["left"] - swing_max["right"]) / max(
            (swing_max["left"] + swing_max["right"]) / 2.0, 1e-6
        )
        _, s = continuous_low_better_score(asym, steepness=5.5, reference=0.30)
        parts.append(s)
        debug.append(
            StabilityDebugRecord(
                "foot_clearance", "max_swing_clearance_asymmetry", asym, asym,
                "asymmetry only", s,
            )
        )
        if asym > 0.35:
            findings.append("Inconsistent right/left foot swing clearance.")

    # Cycle-to-cycle consistency of swing clearance trajectories.
    clearance_cycles: list[np.ndarray] = []
    for cycle in ctx.cycles.cycles:
        times: list[float] = []
        vals: list[float] = []
        for state in ctx.cycles.per_frame:
            if state.frame_index < cycle.start_frame or state.frame_index > cycle.end_frame:
                continue
            if state.left_contact and state.right_contact:
                continue
            for sample in (state.left, state.right):
                if sample.foot_clearance_m is not None:
                    times.append(float(state.time_s))
                    vals.append(float(sample.foot_clearance_m))
        arr = resample_cycle_trajectory(
            times, vals,
            t_start=float(cycle.start_time_s),
            t_end=float(cycle.end_time_s),
        )
        if arr is not None:
            clearance_cycles.append(arr)

    if len(clearance_cycles) >= 2:
        stack = np.vstack(clearance_cycles)
        mean_c = np.mean(stack, axis=0)
        rom = max(float(np.ptp(mean_c)), cfg.toe_drag_clearance_m * 2)
        rmses = [float(np.sqrt(np.mean((row - mean_c) ** 2))) for row in clearance_cycles]
        norm_rmse = statistics.mean(rmses) / rom
        _, s = continuous_low_better_score(norm_rmse, steepness=5.0, reference=0.40)
        parts.append(s)
        debug.append(
            StabilityDebugRecord(
                "foot_clearance", "clearance_cycle_consistency", norm_rmse, norm_rmse,
                "repeatability", s,
            )
        )

    # Outlier fraction during swing (pose spikes / discontinuities — not high mean).
    all_swing = swing_all["left"] + swing_all["right"]
    outlier_frac = 0.0
    if len(all_swing) >= 8:
        arr = np.asarray(all_swing, dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = max(float(q3 - q1), 1e-6)
        upper = float(q3) + 3.0 * iqr
        lower = max(0.0, float(q1) - 3.0 * iqr)
        outliers = np.sum((arr > upper) | (arr < lower))
        outlier_frac = float(outliers) / len(arr)
        _, s = continuous_low_better_score(outlier_frac, steepness=7.0, reference=0.06)
        parts.append(s)
        debug.append(
            StabilityDebugRecord(
                "foot_clearance", "clearance_outlier_fraction", outlier_frac, outlier_frac,
                "isolated spikes", s,
            )
        )

    if not parts:
        return None, ["Insufficient swing-phase clearance data."], {}, debug

    score = float(np.mean(parts))
    if toe_drags > 0:
        drag_penalty = min(40.0, toe_drags * cfg.toe_drag_penalty_per_event * 0.35)
        score = max(0.0, score - drag_penalty)
        findings.append(f"Possible toe-drag events during swing ({toe_drags} frames).")
        debug.append(
            StabilityDebugRecord(
                "foot_clearance", "toe_drag_frames", float(toe_drags), float(toe_drags),
                "contact risk", score, note="toe-drag penalty applied",
            )
        )

    return score, findings, {
        "max_clearance_left": swing_max.get("left"),
        "max_clearance_right": swing_max.get("right"),
        "toe_drag_frames": toe_drags,
        "clearance_outlier_fraction": outlier_frac,
        "swing_frame_count": len(swing_all["left"]) + len(swing_all["right"]),
        "clearance_cycle_count": len(clearance_cycles),
    }, debug


def _joint_angle_series(ctx: StabilityAnalysisContext) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {
        "left_hip": [], "right_hip": [],
        "left_knee": [], "right_knee": [],
        "left_ankle": [], "right_ankle": [],
    }
    for f in ctx.frames:
        a = f.joint_angles
        out["left_hip"].append(a.left_hip)
        out["right_hip"].append(a.right_hip)
        out["left_knee"].append(a.left_knee)
        out["right_knee"].append(a.right_knee)
        out["left_ankle"].append(a.left_ankle if a.left_ankle is not None else a.left_ankle_flexion)
        out["right_ankle"].append(a.right_ankle if a.right_ankle is not None else a.right_ankle_flexion)
    return out


def score_joint_smoothness(
    ctx: StabilityAnalysisContext,
    cfg: StabilityScoreConfig,
) -> tuple[float | None, list[str], dict[str, Any], list[StabilityDebugRecord]]:
    debug: list[StabilityDebugRecord] = []
    findings: list[str] = []

    if ctx.evidence and ctx.evidence.stability_frames < cfg.min_frames:
        return None, ["Insufficient pose frames for joint motion analysis."], {
            "joints_scored": 0,
        }, debug

    usable_indices: set[int] | None = None
    if ctx.evidence is not None:
        usable_indices = set(ctx.evidence.cycles.usable_cycle_indices)

    filtered_cycles = [
        c for c in ctx.cycles.cycles
        if usable_indices is None or c.cycle_index in usable_indices
    ]
    rep_tier = ctx.evidence.repeatability_tier if ctx.evidence else "NORMAL"

    if not filtered_cycles:
        if rep_tier == "UNAVAILABLE":
            return None, [
                "No usable complete gait cycles — joint repeatability unavailable; "
                "single-cycle metrics not computable."
            ], {"joints_scored": 0, "repeatability_tier": rep_tier}, debug
        filtered_cycles = list(ctx.cycles.cycles)

    cm = analyze_controlled_motion(ctx.frames, filtered_cycles)

    parts: list[float] = []
    rom_report: dict[str, float | None] = {}

    for joint in cm.joints:
        key = joint.key
        rom_report[f"{key}_rom_deg"] = joint.rom_deg
        motion_score = joint.controlled_motion_score
        if rep_tier == "UNAVAILABLE":
            motion_score = _controlled_motion_without_repeatability(joint)
            if joint.repeatability_score is not None:
                findings.append(
                    f"{key}: cross-cycle repeatability excluded (need "
                    f"{cfg.gait_evidence.min_cycles_joint_repeatability}+ usable cycles)."
                )
        elif rep_tier == "LOW_CONFIDENCE":
            findings.append(
                f"{key}: repeatability at LOW_CONFIDENCE ({ctx.evidence.usable_gait_cycles if ctx.evidence else 0} usable cycles)."
            )

        if motion_score is not None:
            parts.append(motion_score)
        debug.append(
            StabilityDebugRecord(
                "joint_smoothness", f"controlled_motion_{key}", motion_score,
                motion_score, "control composite", motion_score,
                note=f"ROM={joint.rom_deg:.1f}° (informational)" if joint.rom_deg else "",
            )
        )
        if joint.repeatability_score is not None:
            debug.append(
                StabilityDebugRecord(
                    "joint_smoothness", f"{key}_cycle_repeatability", joint.repeatability_score,
                    joint.repeatability_score, "repeatability", joint.repeatability_score,
                    note=f"tier={rep_tier}",
                )
            )
        if joint.smoothness_score is not None:
            debug.append(
                StabilityDebugRecord(
                    "joint_smoothness", f"{key}_cycle_smoothness", joint.smoothness_score,
                    joint.smoothness_score, "smoothness", joint.smoothness_score,
                )
            )
        if joint.lr_symmetry_score is not None:
            debug.append(
                StabilityDebugRecord(
                    "joint_smoothness", f"{key}_lr_phase_symmetry", joint.lr_symmetry_score,
                    joint.lr_symmetry_score, "asymmetry", joint.lr_symmetry_score,
                )
            )
        if joint.spike_resilience_score is not None:
            debug.append(
                StabilityDebugRecord(
                    "joint_smoothness", f"{key}_spike_resilience", joint.spike_resilience_score,
                    joint.spike_resilience_score, "smoothness", joint.spike_resilience_score,
                )
            )
        if (
            motion_score is not None
            and motion_score < cfg.moderate_min
        ):
            findings.append(f"Reduced controlled motion quality at {key}.")

    if not parts:
        return None, ["Insufficient gait-cycle joint trajectories for controlled motion."], {
            "joints_scored": 0,
            "repeatability_tier": rep_tier,
        }, debug

    score = float(np.mean(parts))
    return score, findings, {
        "source": "controlled_motion_gait_cycle_normalized",
        "joints_scored": len(parts),
        "controlled_motion_overall": cm.overall_score,
        "repeatability_tier": rep_tier,
        "usable_cycles": len(filtered_cycles),
        "rom_deg": rom_report,
        "controlled_motion_detail": cm.to_dict(),
    }, debug


def _resample_cycle(
    values: list[tuple[float, float]],
    t0: float,
    t1: float,
    n: int = 50,
) -> np.ndarray | None:
    if t1 <= t0 or len(values) < 3:
        return None
    times = np.array([t for t, _ in values])
    ys = np.array([v for _, v in values])
    grid = np.linspace(t0, t1, n)
    return np.interp(grid, times, ys)


def score_cycle_consistency(
    ctx: StabilityAnalysisContext,
    cfg: StabilityScoreConfig,
) -> tuple[float | None, list[str], dict[str, Any], list[StabilityDebugRecord]]:
    debug: list[StabilityDebugRecord] = []
    parts: list[float] = []
    findings: list[str] = []
    gf = ctx.gait_features
    cc = gf.cycle_consistency if gf else None
    usable = (
        ctx.evidence.usable_gait_cycles
        if ctx.evidence
        else len(ctx.cycles.cycles)
    )

    if usable < cfg.gait_evidence.min_cycles_cycle_consistency:
        return None, [
            f"Cycle consistency requires at least "
            f"{cfg.gait_evidence.min_cycles_cycle_consistency} usable complete gait cycles "
            f"(have {usable}). No fallback score applied."
        ], {
            "cycle_count": usable,
            "usable_cycles": usable,
            "angle_source": cc.angle_source if cc else "unavailable",
        }, debug

    usable_cycles = [
        c for c in ctx.cycles.cycles
        if ctx.evidence is None
        or c.cycle_index in ctx.evidence.cycles.usable_cycle_indices
    ]

    if cc and cc.cycle_repeatability_score is not None and usable >= 2:
        s = cc.cycle_repeatability_score
        parts.append(s)
        debug.append(
            StabilityDebugRecord(
                "cycle_consistency",
                "cycle_repeatability_score",
                cc.cycle_repeatability_score,
                cc.cycle_repeatability_score,
                "higher better",
                s,
                note=FeatureNormalization.GAIT_CYCLE_NORMALIZED.value,
            )
        )

    if cc and cc.left_right_phase_consistency is not None:
        s = cfg.cycle_shape_similarity.score(cc.left_right_phase_consistency)
        parts.append(s)
        debug.append(
            StabilityDebugRecord(
                "cycle_consistency",
                "left_right_knee_phase_consistency",
                cc.left_right_phase_consistency,
                cc.left_right_phase_consistency,
                "higher better",
                s,
                note=FeatureNormalization.GAIT_CYCLE_NORMALIZED.value,
            )
        )

    if len(usable_cycles) >= 2:
        durs = [c.duration_s for c in usable_cycles]
        cv = _cv(durs)
        if cv is not None:
            s = cfg.cycle_duration_cv.score(cv)
            parts.append(s)
            debug.append(
                StabilityDebugRecord(
                    "cycle_consistency", "cycle_duration_cv", cv, cv,
                    "lower better", s, note=FeatureNormalization.RAW.value,
                )
            )

    if cc and cc.cycle_to_cycle_rmse:
        mean_rmse = statistics.mean(cc.cycle_to_cycle_rmse.values())
        s = max(0.0, min(100.0, 100.0 - mean_rmse * 10.0))
        parts.append(s)
        debug.append(
            StabilityDebugRecord(
                "cycle_consistency",
                "mean_cycle_rmse",
                mean_rmse,
                mean_rmse,
                "lower better",
                s,
                note=FeatureNormalization.GAIT_CYCLE_NORMALIZED.value,
            )
        )

    if cc and cc.angle_source == "opensim_ik":
        findings.append("Joint-angle cycle trajectories prefer OpenSim IK when available.")
    elif cc:
        findings.append("Joint-angle cycle trajectories use pose-derived angles.")

    if usable < cfg.gait_evidence.min_usable_cycles_normal:
        findings.append(
            f"Only {usable} usable cycle(s) — cycle consistency at reduced confidence "
            f"(need {cfg.gait_evidence.min_usable_cycles_normal} for normal tier)."
        )

    if not parts:
        return None, findings or ["Gait cycle consistency not computable."], {}, debug
    return float(np.mean(parts)), findings, {
        "cycle_count": usable,
        "usable_cycles": usable,
        "angle_source": cc.angle_source if cc else "unavailable",
    }, debug


def score_contact_pattern(
    ctx: StabilityAnalysisContext,
    cfg: StabilityScoreConfig,
) -> tuple[float | None, list[str], dict[str, Any], list[StabilityDebugRecord]]:
    debug: list[StabilityDebugRecord] = []
    parts: list[float] = []
    findings: list[str] = []
    pf = ctx.cycles.per_frame
    if len(pf) < cfg.min_frames:
        return None, ["Insufficient contact timeline."], {}, debug

    fps = ctx.fps
    duration = pf[-1].time_s - pf[0].time_s if len(pf) > 1 else len(pf) / fps

    for side in ("left", "right"):
        contacts = [s.left_contact if side == "left" else s.right_contact for s in pf]
        stance_durs = []
        run, val = 0, contacts[0]
        for c in contacts[1:] + [-1]:
            if c == val:
                run += 1
            else:
                if val == 1:
                    stance_durs.append(run / fps)
                run = 1
                val = c
        cv = _cv(stance_durs)
        if cv is not None:
            s = cfg.contact_stance_cv.score(cv)
            parts.append(s)
            debug.append(
                StabilityDebugRecord(
                    "contact_pattern", f"{side}_stance_duration_cv", cv, cv,
                    "lower better", s,
                )
            )

    toggles = sum(
        1 for i in range(1, len(pf))
        if pf[i].left_contact != pf[i - 1].left_contact
        or pf[i].right_contact != pf[i - 1].right_contact
    )
    toggle_rate = toggles / max(duration, 1e-6)
    s_t = cfg.contact_toggle_rate_hz.score(toggle_rate)
    parts.append(s_t)
    debug.append(
        StabilityDebugRecord(
            "contact_pattern", "contact_toggle_rate_hz", toggle_rate, toggle_rate,
            "lower better", s_t,
        )
    )
    if toggle_rate > cfg.contact_toggle_rate_hz.poor * 0.5:
        findings.append("Rapid foot contact-state changes (possible tracking noise).")

    ds_frac = sum(1 for s in pf if s.left_contact and s.right_contact) / len(pf)
    ds_dev = abs(ds_frac - cfg.expected_double_support_fraction)
    s_ds = cfg.double_support_deviation.score(ds_dev)
    parts.append(s_ds)
    debug.append(
        StabilityDebugRecord(
            "contact_pattern", "double_support_fraction", ds_frac, ds_dev,
            "lower deviation better", s_ds,
        )
    )

    # Event alternation quality.
    hs = [e for e in ctx.cycles.events if e.event_type.endswith("_heel_strike")]
    hs.sort(key=lambda e: e.time_s)
    if len(hs) >= 3:
        same_side_runs = 0
        for i in range(1, len(hs)):
            if hs[i].side == hs[i - 1].side:
                same_side_runs += 1
        alt_score = 100.0 * (1.0 - same_side_runs / max(len(hs) - 1, 1))
        parts.append(alt_score)
        debug.append(
            StabilityDebugRecord(
                "contact_pattern", "heel_strike_alternation", float(same_side_runs),
                same_side_runs / max(len(hs) - 1, 1), "lower same-side repeats better",
                alt_score,
            )
        )
    else:
        findings.append(
            "Fewer than 3 heel strikes — alternation quality not scored."
        )

    if not parts:
        return None, findings or ["Contact pattern not computable."], {
            "double_support_fraction": ds_frac,
            "contact_toggle_rate_hz": toggle_rate,
            "heel_strike_count": len(hs),
        }, debug

    return float(np.mean(parts)), findings, {
        "double_support_fraction": ds_frac,
        "contact_toggle_rate_hz": toggle_rate,
        "heel_strike_count": len(hs),
    }, debug


# ---------------------------------------------------------------------------
# Explanation + main entry
# ---------------------------------------------------------------------------
def generate_why_explanation(metrics: list[MetricResult]) -> str:
    """Plain-language summary of the lowest contributing domains."""
    ranked = sorted(
        (m for m in metrics if m.score is not None and m.findings),
        key=lambda m: m.score or 100.0,
    )
    if not ranked:
        low = sorted(
            (m for m in metrics if m.score is not None),
            key=lambda m: m.score or 100.0,
        )
        if not low or low[0].score >= DEFAULT_STABILITY_CONFIG.stable_min:
            return "No major instability patterns detected in the measured gait domains."
        return (
            f"Reduced stability is mainly associated with lower "
            f"{low[0].name.lower()} ({low[0].score:.0f}/100)."
        )

    issues: list[str] = []
    for m in ranked[:3]:
        issues.extend(m.findings[:1])
    if not issues:
        return "Stability is limited by uneven sub-scores across gait domains."
    joined = ", ".join(issues[:3]).rstrip(".")
    return f"Reduced stability is mainly associated with {joined.lower()}."


def _merge_domain_availability(
    evidence_hint: DomainAvailability,
    computed: DomainAvailability,
) -> DomainAvailability:
    """Take the more conservative availability tier."""
    rank = {"UNAVAILABLE": 0, "LOW_CONFIDENCE": 1, "AVAILABLE": 2}
    return evidence_hint if rank[evidence_hint] < rank[computed] else computed


def _domain_confidence(
    key: str,
    ctx: StabilityAnalysisContext,
    cfg: StabilityScoreConfig,
    *,
    score: float | None,
    values: dict[str, Any],
) -> tuple[DomainAvailability, float]:
    """Map raw domain output to availability + view-adjusted confidence in [0, 1]."""
    evidence_hint: DomainAvailability = (
        ctx.evidence.availability_for_domain(key) if ctx.evidence else "AVAILABLE"
    )

    if score is None:
        return "UNAVAILABLE", 0.0

    if evidence_hint == "UNAVAILABLE" and key in (
        "cycle_consistency",
        "spatial_symmetry",
    ):
        return "UNAVAILABLE", 0.0

    contact_conf = ctx.cycles.metrics.contact_confidence
    hs_total = (
        ctx.cycles.metrics.left_heel_strike_count
        + ctx.cycles.metrics.right_heel_strike_count
    )
    conf: float | None = None
    computed: DomainAvailability = "AVAILABLE"

    if key == "temporal_symmetry":
        if hs_total < cfg.gait_evidence.min_heel_strikes_temporal:
            return "UNAVAILABLE", 0.0
        cycle_count = int(values.get("gait_cycle_count", len(ctx.cycles.cycles)))
        usable = ctx.evidence.usable_gait_cycles if ctx.evidence else cycle_count
        conf = min(1.0, hs_total / 4.0) * contact_conf
        if usable < cfg.gait_evidence.min_usable_cycles_normal:
            conf *= 0.75
        if cycle_count < cfg.min_cycles_for_cycle_score or values.get("insufficient_cycles"):
            if conf < 0.35:
                computed = "UNAVAILABLE"
            else:
                computed = "LOW_CONFIDENCE"
                conf *= 0.55

    elif key == "spatial_symmetry":
        if hs_total < cfg.gait_evidence.min_heel_strikes_temporal:
            return "UNAVAILABLE", 0.0
        conf = min(1.0, hs_total / 3.0) * min(1.0, contact_conf + 0.15)
        if ctx.evidence and ctx.evidence.usable_gait_cycles < cfg.gait_evidence.min_usable_cycles_low_confidence:
            conf *= 0.70

    elif key == "pelvis_stability":
        ratio = float(values.get("tracked_frame_ratio", 0.0))
        if ratio < cfg.min_frames / max(len(ctx.frames), 1):
            return "UNAVAILABLE", 0.0
        conf = min(1.0, ratio / 0.85)

    elif key == "trunk_stability":
        conf = min(1.0, len(ctx.frames) / max(cfg.min_frames, 1))
        if conf < 0.5:
            return "UNAVAILABLE", 0.0

    elif key == "foot_clearance":
        swing_frames = int(values.get("swing_frame_count", 0))
        if swing_frames < 4:
            return "UNAVAILABLE", 0.0
        conf = min(1.0, swing_frames / max(len(ctx.cycles.per_frame) * 0.25, 1.0)) * contact_conf

    elif key == "joint_smoothness":
        joint_count = int(values.get("joints_scored", 0))
        if joint_count < 2:
            return "UNAVAILABLE", 0.0
        conf = min(1.0, joint_count / 4.0)
        rep_tier = values.get("repeatability_tier", "NORMAL")
        if rep_tier == "UNAVAILABLE":
            conf *= 0.72
        elif rep_tier == "LOW_CONFIDENCE":
            conf *= 0.85

    elif key == "cycle_consistency":
        cycles = int(values.get("usable_cycles", values.get("cycle_count", 0)))
        if cycles < cfg.gait_evidence.min_cycles_cycle_consistency:
            return "UNAVAILABLE", 0.0
        if not values.get("angle_source"):
            return "UNAVAILABLE", 0.0
        conf = min(1.0, cycles / max(cfg.gait_evidence.min_usable_cycles_normal, 1)) * contact_conf
        if cycles < cfg.gait_evidence.min_usable_cycles_normal:
            computed = "LOW_CONFIDENCE"
            conf *= 0.80

    elif key == "contact_pattern":
        hs = int(values.get("heel_strike_count", hs_total))
        if len(ctx.cycles.per_frame) < cfg.min_frames:
            return "UNAVAILABLE", 0.0
        conf = min(1.0, hs / 3.0) * contact_conf
        if hs < cfg.gait_evidence.min_heel_strikes_temporal:
            return "UNAVAILABLE", 0.0
        if ctx.evidence and ctx.evidence.usable_gait_cycles < cfg.gait_evidence.min_cycles_contact_pattern:
            conf *= 0.78
            computed = "LOW_CONFIDENCE"

    else:
        conf = 1.0

    if conf is None:
        return "UNAVAILABLE", 0.0

    base_conf = conf
    if ctx.view_reliability is not None:
        conf = apply_view_reliability(conf, key, ctx.view_reliability)
    values["base_confidence"] = round(base_conf, 3)
    values["view_reliability"] = round(
        ctx.view_reliability.domain_coefficient(key) if ctx.view_reliability else 1.0,
        3,
    )
    values["effective_confidence"] = round(conf, 3)
    computed = _merge_domain_availability(
        evidence_hint,
        computed if computed != "AVAILABLE" else _availability_from_confidence(conf)[0],
    )
    if computed == "UNAVAILABLE":
        return "UNAVAILABLE", 0.0
    if computed == "LOW_CONFIDENCE":
        return "LOW_CONFIDENCE", conf
    return _availability_from_confidence(conf)


def _availability_from_confidence(conf: float) -> tuple[DomainAvailability, float]:
    if conf < 0.35:
        return "UNAVAILABLE", 0.0
    if conf < 0.72:
        return "LOW_CONFIDENCE", conf
    return "AVAILABLE", conf


def _finalize_stability_score(
    metrics: list[MetricResult],
    cfg: StabilityScoreConfig,
) -> tuple[float, list[DomainContribution], float, str]:
    """
    Confidence-weighted renormalization over AVAILABLE and LOW_CONFIDENCE domains.

    UNAVAILABLE domains are excluded — never assigned neutral placeholder scores.
    """
    contributions: list[DomainContribution] = []
    total_configured_weight = sum(m.weight for m in metrics)

    for m in metrics:
        effective_conf = m.confidence if m.availability != "UNAVAILABLE" else 0.0
        contrib = (
            float(m.score) * m.weight * effective_conf
            if m.score is not None and m.availability != "UNAVAILABLE"
            else 0.0
        )
        contributions.append(
            DomainContribution(
                key=m.key,
                name=m.name,
                score=m.score,
                weight=m.weight,
                confidence=effective_conf,
                availability=m.availability,
                contribution=contrib,
            )
        )
        m.contribution = contrib

    active = [c for c in contributions if c.availability != "UNAVAILABLE" and c.score is not None]
    denom = sum(c.weight * c.confidence for c in active)
    if denom <= 0:
        return 0.0, contributions, 0.0, "LOW CONFIDENCE"

    final = sum(c.score * c.weight * c.confidence for c in active) / denom  # type: ignore[operator]
    completeness = (denom / max(total_configured_weight, 1e-9)) * 100.0

    avg_conf = statistics.mean([c.confidence for c in active]) if active else 0.0
    if completeness >= 85.0 and avg_conf >= 0.72:
        badge = "HIGH CONFIDENCE"
    elif completeness >= 55.0 and avg_conf >= 0.45:
        badge = "MODERATE CONFIDENCE"
    else:
        badge = "LOW CONFIDENCE"

    return _clamp(final), contributions, completeness, badge


def _log_debug_report(
    score: float,
    metrics: list[MetricResult],
    ctx: StabilityAnalysisContext,
) -> None:
    logger.info("=== Stability v2 score: %.1f ===", score)
    for m in metrics:
        if m.score is None:
            logger.info("  %s: N/A — %s", m.name, m.summary)
            continue
        contrib = m.score * m.weight
        logger.info(
            "  %s: %.1f/100 (weight %.0f%%, contribution %.1f) — %s",
            m.name, m.score, m.weight * 100, contrib, m.summary,
        )
        for feat in m.values.get("debug_features", []):
            logger.debug(
                "    feature %s raw=%s norm=%s component=%s %s",
                feat.get("feature"),
                feat.get("raw"),
                feat.get("normalized"),
                feat.get("component_score"),
                feat.get("note", ""),
            )
    logger.info(
        "  Contact confidence: %.2f (%s) | cycles: %d | frames: %d | fps: %.1f",
        ctx.cycles.metrics.contact_confidence,
        ctx.cycles.metrics.confidence_tier,
        ctx.cycles.metrics.gait_cycle_count,
        len(ctx.frames),
        ctx.fps,
    )


def verify_stability_api_label_free() -> None:
    """Assert the public stability API cannot accept gait-category parameters."""
    import inspect

    sig = inspect.signature(analyze_biomech_stability)
    forbidden = FORBIDDEN_STABILITY_PARAMETERS.intersection(sig.parameters)
    if forbidden:
        raise AssertionError(
            f"Stability API exposes forbidden label parameters: {sorted(forbidden)}"
        )


def analyze_biomech_stability(
    sequence: PoseSequence,
    *,
    config: StabilityScoreConfig | None = None,
    cycles: GaitCycleAnalysisResult | None = None,
) -> StabilityResult:
    """
    Compute redesigned stability assessment (v2).

    Backward-compatible return type for GUI and scripts.
    """
    verify_stability_api_label_free()
    cfg = config or DEFAULT_STABILITY_CONFIG
    cfg.validate()

    ctx = _build_context(sequence, cycles=cycles)
    if ctx is None:
        n = sum(1 for f in sequence.frames if f.detected)
        msg = f"Only {n} usable frames (need {cfg.min_frames})."
        return StabilityResult(
            score=0.0,
            classification="Insufficient data",
            metrics=[],
            primary_issue=msg,
            explanation=msg,
            frame_count=n,
            scoring_notes=SCORING_NOTES_V2,
        )

    weights = cfg.sub_score_weights()
    scorers = [
        ("temporal_symmetry", "Temporal Symmetry", score_temporal_symmetry),
        ("spatial_symmetry", "Spatial Symmetry", score_spatial_symmetry),
        ("pelvis_stability", "Pelvis Stability", score_pelvis_stability),
        ("trunk_stability", "Trunk Stability", score_trunk_stability),
        ("foot_clearance", "Foot Clearance Consistency", score_foot_clearance),
        ("joint_smoothness", "Joint Motion Smoothness", score_joint_smoothness),
        ("cycle_consistency", "Gait Cycle Consistency", score_cycle_consistency),
        ("contact_pattern", "Contact Pattern", score_contact_pattern),
    ]

    metrics: list[MetricResult] = []
    for key, name, fn in scorers:
        score, findings, values, debug = fn(ctx, cfg)
        w = weights[key]
        for d in debug:
            d.weight_share = w
        summary = (
            f"{name} is strong ({score:.0f}/100)."
            if score is not None and score >= cfg.stable_min and not findings
            else f"{name} shows limitations ({score:.0f}/100)."
            if score is not None
            else f"{name} not available."
        )
        metrics.append(_metric(key, name, score, w, summary, findings, values, debug))

    for m in metrics:
        availability, confidence = _domain_confidence(
            m.key, ctx, cfg, score=m.score, values=m.values
        )
        m.availability = availability
        m.confidence = confidence
        if availability == "UNAVAILABLE":
            m.score = None
            m.summary = f"{m.name} unavailable — excluded from overall score."

    final, contributions, completeness, badge = _finalize_stability_score(metrics, cfg)

    classification = (
        "Stable" if final >= cfg.stable_min
        else "Moderate" if final >= cfg.moderate_min
        else "Unstable"
    )
    if completeness < 35.0:
        classification = "Insufficient data"

    data_limitations: list[str] = []
    unavailable = [m for m in metrics if m.availability == "UNAVAILABLE"]
    low_conf = [m for m in metrics if m.availability == "LOW_CONFIDENCE"]
    if unavailable:
        data_limitations.append(
            "Excluded from overall score: "
            + ", ".join(m.name for m in unavailable)
        )
    if low_conf:
        data_limitations.append(
            "Low-confidence domains (included with reduced weight): "
            + ", ".join(m.name for m in low_conf)
        )
    hs_total = (
        ctx.cycles.metrics.left_heel_strike_count
        + ctx.cycles.metrics.right_heel_strike_count
    )
    if hs_total < 3:
        data_limitations.append(
            f"Only {hs_total} heel strike(s) detected — temporal, cycle, and contact "
            "domains may be incomplete."
        )
    if ctx.view_estimate is not None:
        data_limitations.append(
            f"Estimated camera view: {ctx.view_estimate.display_name} "
            f"(confidence {ctx.view_estimate.view_confidence:.0%}). "
            "Domain confidences are scaled by view-based measurement reliability."
        )

    if ctx.evidence:
        data_limitations.extend(ctx.evidence.warnings)
        data_limitations.append(
            f"Repeatability tier: {ctx.evidence.repeatability_tier} "
            f"({ctx.evidence.usable_gait_cycles} usable gait cycle(s))."
        )
        data_limitations.append(cfg.gait_evidence.documentation().split("\n")[0])

    view_table: list[tuple[str, str]] = []
    view_type: str | None = None
    view_confidence = 0.0
    view_display: str | None = None
    if ctx.view_reliability is not None and ctx.view_estimate is not None:
        view_table = ctx.view_reliability.metric_table()
        view_type = ctx.view_estimate.view_type.value
        view_confidence = ctx.view_estimate.view_confidence
        view_display = ctx.view_estimate.display_name

    why = generate_why_explanation(metrics)
    scored = [m for m in metrics if m.availability != "UNAVAILABLE" and m.score is not None]
    lines = [
        f"Overall Stability: {final:.0f}/100 ({classification})",
        f"Analysis completeness: {completeness:.0f}%",
        f"Confidence: {badge}",
    ]
    if ctx.evidence:
        lines.append(f"Usable gait cycles: {ctx.evidence.usable_gait_cycles}")
        lines.append(
            f"Video: {ctx.evidence.video.duration_s:.2f}s, "
            f"{ctx.evidence.video.fps:.1f} FPS, "
            f"{ctx.evidence.video.valid_pose_frames}/{ctx.evidence.video.total_frames} valid pose frames"
        )
        lines.append(f"Repeatability tier: {ctx.evidence.repeatability_tier}")
    if view_display:
        lines.append(f"View: {view_display} (confidence {view_confidence:.0%})")
    lines.extend(["", why, "", "Domain breakdown:"])
    for m in metrics:
        if m.availability == "UNAVAILABLE" or m.score is None:
            lines.append(f"  {m.name}: unavailable ({m.availability})")
        else:
            lines.append(
                f"  {m.name}: {m.score:.0f}/100 "
                f"(weight {m.weight:.0%}, conf {m.confidence:.0%}, {m.availability})"
            )
    lines.extend(["", "Contribution table:"])
    temp_result = StabilityResult(
        score=final,
        classification=classification,
        metrics=metrics,
        primary_issue=None,
        explanation="",
        frame_count=len(ctx.frames),
        completeness_pct=completeness,
        confidence_badge=badge,
        contributions=contributions,
    )
    lines.append(temp_result.contribution_table_text())
    if data_limitations:
        lines.extend(["", "Data limitations:"])
        lines.extend(f"  • {note}" for note in data_limitations)

    primary = None
    worst = min(scored, key=lambda m: m.score or 100.0, default=None)  # type: ignore[arg-type]
    if worst and worst.score is not None and worst.score < cfg.stable_min:
        primary = f"{worst.name} ({worst.score:.0f}/100)"

    _log_debug_report(final, metrics, ctx)

    result = StabilityResult(
        score=final,
        classification=classification,
        metrics=metrics,
        primary_issue=primary,
        explanation="\n".join(lines),
        frame_count=len(ctx.frames),
        scoring_notes=SCORING_NOTES_V2,
        completeness_pct=completeness,
        confidence_badge=badge,
        contributions=contributions,
        data_limitations=data_limitations,
        view_type=view_type,
        view_confidence=view_confidence,
        view_display_name=view_display,
        view_reliability_table=view_table,
        analysis_evidence=ctx.evidence.to_dict() if ctx.evidence else {},
        domain_evidence={
            k: v.evidence_summary for k, v in ctx.evidence.domain_evidence.items()
        } if ctx.evidence else {},
        usable_gait_cycles=ctx.evidence.usable_gait_cycles if ctx.evidence else 0,
        video_duration_s=ctx.evidence.video.duration_s if ctx.evidence else 0.0,
        repeatability_tier=ctx.evidence.repeatability_tier if ctx.evidence else "UNAVAILABLE",
    )
    from stablewalk.analysis.gait_analysis_summary import build_gait_analysis_summary
    from stablewalk.analysis.stability_validity import assess_stability_result_validity

    result.validity = assess_stability_result_validity(result)
    result.gait_summary = build_gait_analysis_summary(result)
    return result


__all__ = [
    "analyze_biomech_stability",
    "StabilityResult",
    "MetricResult",
    "DomainContribution",
    "DomainAvailability",
    "StabilityScoreConfig",
    "DEFAULT_STABILITY_CONFIG",
    "StabilityDebugRecord",
    "compute_pelvis_motion_metrics",
    "generate_why_explanation",
    "SCORING_NOTES_V2",
    "STABLE_MIN",
    "MODERATE_MIN",
    "FORBIDDEN_STABILITY_PARAMETERS",
    "LABEL_FREE_PIPELINE_ASSERTION",
    "verify_stability_api_label_free",
]
