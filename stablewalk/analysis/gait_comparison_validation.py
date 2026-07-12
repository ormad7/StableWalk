"""
Label-blind gait comparison validation for StableWalk demo videos.

Runs the same motion-analysis pipeline on each video independently.
Ground-truth gait labels (abnormal / normal / athletic) are attached only
when assembling comparison reports — never passed into scoring or metrics.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.biomech_stability import analyze_biomech_stability
from stablewalk.analysis.gait_cycle_analysis import (
    GaitCycleAnalysisResult,
    analyze_gait_cycles,
    segment_durations_s,
    symmetry_ratio,
)
from stablewalk.analysis.gait_feature_analysis import analyze_gait_features
from stablewalk.analysis.gait_view_analysis import assess_cross_video_comparability
from stablewalk.analysis.stability_scoring import (
    StabilityAnalysisContext,
    StabilityResult,
    _build_context,
    _joint_angle_series,
    _mean_jerk,
    _robust_range,
    compute_pelvis_motion_metrics,
)
from stablewalk.analysis.stability_validity import (
    compare_stability_results,
    assess_stability_result_validity,
)
from stablewalk.analysis.gait_analysis_summary import build_gait_analysis_summary
from stablewalk.models.pose_data import PoseFrame, PoseSequence
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.ui.viewers.knee_angle_chart import build_knee_angle_series

# Ground-truth labels for report grouping only — not pipeline inputs.
# Internal key ``athletic`` = UI label "Performance" (Health&Gait FGS).
COMPARISON_VIDEOS: tuple[tuple[str, str], ...] = (
    ("abnormal", "abnormal_gait"),
    ("normal", "normal_gait"),
    ("athletic", "athletic_walking"),  # Performance demo
)

FOOT_LANDMARKS = (
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

REPORT_VERSION = "1.0"


@dataclass
class VideoComparisonRecord:
    """One video's label-blind analysis plus ground-truth label for comparison."""

    ground_truth_label: str
    video_stem: str
    sections: dict[str, Any] = field(default_factory=dict)
    stability_result: StabilityResult | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ground_truth_label": self.ground_truth_label,
            "video_stem": self.video_stem,
            **self.sections,
        }


def analyze_motion_label_blind(
    sequence: PoseSequence,
) -> tuple[dict[str, Any], StabilityResult]:
    """
    Run the full StableWalk analysis pipeline on a pose sequence.

    Accepts only ``PoseSequence`` motion data. Gait category labels must not
    be supplied to this function or to ``analyze_biomech_stability``.
    """
    enrich_pose_sequence(sequence)
    recording = pose_sequence_to_gait_motion(sequence)
    cycles = analyze_gait_cycles(recording)
    stability = analyze_biomech_stability(sequence, cycles=cycles)
    ctx = _build_context(sequence, cycles=cycles, recording=recording)
    features = ctx.gait_features if ctx else None

    sections = {
        "video_quality": _video_quality_section(sequence, stability, ctx),
        "gait_events": _gait_events_section(cycles),
        "temporal_features": _temporal_features_section(cycles),
        "spatial_features": _spatial_features_section(features, ctx),
        "kinematic_features": _kinematic_features_section(sequence, ctx, stability),
        "gait_consistency": _gait_consistency_section(features, cycles, stability),
        "stability": _stability_section(stability),
    }
    sections["video_quality"]["contact_confidence_tier"] = sections["gait_events"].get(
        "contact_confidence_tier"
    )
    return sections, stability


def _video_quality_section(
    sequence: PoseSequence,
    stability: StabilityResult,
    ctx: StabilityAnalysisContext | None = None,
) -> dict[str, Any]:
    total = len(sequence.frames)
    detected = sum(1 for f in sequence.frames if f.detected)
    confidences: list[float] = []
    foot_confidences: list[float] = []
    for frame in sequence.frames:
        if not frame.detected or not frame.keypoints:
            continue
        for kp in frame.keypoints:
            confidences.append(float(kp.visibility))
            if kp.name in FOOT_LANDMARKS:
                foot_confidences.append(float(kp.visibility))

    fps = max(sequence.fps, 1e-6)
    duration_s = total / fps if total else 0.0
    if detected and sequence.frames:
        last_t = sequence.frames[min(detected - 1, total - 1)].time_seconds(fps)
        duration_s = max(duration_s, last_t)

    out = {
        "fps": round(fps, 3),
        "duration_s": round(duration_s, 3),
        "frame_count": total,
        "detected_frame_count": detected,
        "valid_pose_frame_pct": round(100.0 * detected / max(total, 1), 2),
        "average_pose_confidence": _mean_or_none(confidences),
        "foot_landmark_confidence": _mean_or_none(foot_confidences),
        "analysis_completeness_pct": round(stability.completeness_pct, 2),
        "contact_confidence_tier": None,
        "view_type": stability.view_type,
        "view_confidence": round(stability.view_confidence, 3),
        "view_display_name": stability.view_display_name,
        "body_height_m": round(ctx.anthro.body_height, 4) if ctx else None,
        "gait_cycles": (
            ctx.cycles.metrics.gait_cycle_count if ctx and ctx.cycles.metrics else None
        ),
    }
    return out


def _gait_events_section(cycles: GaitCycleAnalysisResult) -> dict[str, Any]:
    m = cycles.metrics
    return {
        "left_heel_strikes": m.left_heel_strike_count,
        "right_heel_strikes": m.right_heel_strike_count,
        "left_toe_offs": m.left_toe_off_count,
        "right_toe_offs": m.right_toe_off_count,
        "complete_gait_cycles": m.gait_cycle_count,
        "contact_confidence": round(m.contact_confidence, 3),
        "contact_confidence_tier": m.confidence_tier,
    }


def _temporal_features_section(cycles: GaitCycleAnalysisResult) -> dict[str, Any]:
    m = cycles.metrics
    pf = cycles.per_frame
    stance_swing = _stance_swing_stats(pf, cycles.fps)

    step_sym = None
    hs = sorted(
        (e.time_s, e.side) for e in cycles.events if e.event_type.endswith("_heel_strike")
    )
    if len(hs) >= 4:
        left_intervals = [
            hs[i + 1][0] - hs[i][0]
            for i in range(len(hs) - 1)
            if hs[i][1] == "left" and hs[i + 1][1] == "left"
        ]
        right_intervals = [
            hs[i + 1][0] - hs[i][0]
            for i in range(len(hs) - 1)
            if hs[i][1] == "right" and hs[i + 1][1] == "right"
        ]
        left_step = statistics.mean(left_intervals) if left_intervals else None
        right_step = statistics.mean(right_intervals) if right_intervals else None
        step_sym = symmetry_ratio(left_step, right_step)

    out = {
        "cadence_steps_per_min": m.cadence_steps_per_min,
        "left_stance_mean_s": m.left_stance_time_s,
        "left_stance_std_s": stance_swing.get("left_stance_std_s"),
        "right_stance_mean_s": m.right_stance_time_s,
        "right_stance_std_s": stance_swing.get("right_stance_std_s"),
        "left_swing_mean_s": m.left_swing_time_s,
        "left_swing_std_s": stance_swing.get("left_swing_std_s"),
        "right_swing_mean_s": m.right_swing_time_s,
        "right_swing_std_s": stance_swing.get("right_swing_std_s"),
        "stance_symmetry_index": m.left_right_stance_symmetry,
        "swing_symmetry_index": m.left_right_swing_symmetry,
        "step_time_symmetry": step_sym,
        "double_support_pct": m.double_support_pct,
        "step_time_s": m.step_time_s,
        "stride_time_s": m.stride_time_s,
    }
    return {k: _round_val(v) for k, v in out.items()}


def _stance_swing_stats(
    per_frame: list,
    fps: float,
) -> dict[str, float | None]:
    if not per_frame:
        return {}
    times = [s.time_s for s in per_frame]
    left = [s.left_contact for s in per_frame]
    right = [s.right_contact for s in per_frame]

    def _std(runs: list[float]) -> float | None:
        if len(runs) < 2:
            return None
        return float(statistics.stdev(runs))

    return {
        "left_stance_std_s": _std(segment_durations_s(left, times, value=1)),
        "right_stance_std_s": _std(segment_durations_s(right, times, value=1)),
        "left_swing_std_s": _std(segment_durations_s(left, times, value=0)),
        "right_swing_std_s": _std(segment_durations_s(right, times, value=0)),
    }


def _spatial_features_section(
    features: Any,
    ctx: StabilityAnalysisContext | None,
) -> dict[str, Any]:
    if features is None:
        return {}
    f = features.features
    pelvis = compute_pelvis_motion_metrics(ctx) if ctx else {}
    clearance_asym = None
    if (
        f.normalized_foot_clearance_left is not None
        and f.normalized_foot_clearance_right is not None
    ):
        clearance_asym = symmetry_ratio(
            f.normalized_foot_clearance_left,
            f.normalized_foot_clearance_right,
        )

    out = {
        "normalized_step_length": f.normalized_step_length,
        "normalized_stride_length": f.normalized_stride_length,
        "left_foot_clearance_m": f.foot_clearance_left_m,
        "right_foot_clearance_m": f.foot_clearance_right_m,
        "normalized_foot_clearance_left": f.normalized_foot_clearance_left,
        "normalized_foot_clearance_right": f.normalized_foot_clearance_right,
        "foot_clearance_asymmetry": clearance_asym or f.foot_clearance_symmetry,
        "pelvis_mediolateral_sway_m": f.pelvis_mediolateral_range_m,
        "pelvis_vertical_oscillation_m": f.pelvis_vertical_range_m,
        "normalized_pelvis_sway": f.normalized_pelvis_sway,
        "normalized_vertical_pelvis_motion": f.normalized_vertical_pelvis_motion,
        "pelvis_mediolateral_range_image": pelvis.get("pelvis_mediolateral_range"),
        "pelvis_vertical_range_image": pelvis.get("pelvis_vertical_range"),
    }
    return {k: _round_val(v) for k, v in out.items()}


def _kinematic_features_section(
    sequence: PoseSequence,
    ctx: StabilityAnalysisContext | None,
    stability: StabilityResult,
) -> dict[str, Any]:
    pose_indices = [i for i, f in enumerate(sequence.frames) if f.detected]
    ik_mot_path = None
    if sequence.source_video:
        from stablewalk import config as sw_config

        stem = Path(sequence.source_video).stem
        candidate = sw_config.OPENSIM_DIR / stem / f"{stem}_ik.mot"
        if candidate.is_file():
            ik_mot_path = candidate

    knee = build_knee_angle_series(sequence, pose_indices, ik_mot_path=ik_mot_path)
    left_valid = knee.left_deg[~np.isnan(knee.left_deg)]
    right_valid = knee.right_deg[~np.isnan(knee.right_deg)]
    left_rom = float(np.ptp(left_valid)) if left_valid.size else None
    right_rom = float(np.ptp(right_valid)) if right_valid.size else None
    knee_asym = symmetry_ratio(left_rom, right_rom)

    hip_rom: dict[str, float | None] = {}
    ankle_rom: dict[str, float | None] = {}
    ang_vel_cv: dict[str, float | None] = {}
    norm_jerks: list[float] = []

    if ctx:
        series = _joint_angle_series(ctx)
        for name, seq in series.items():
            valid = [v for v in seq if v is not None]
            if len(valid) < 4:
                continue
            rom = _robust_range(seq)
            if "hip" in name:
                hip_rom[name] = rom
            if "ankle" in name:
                ankle_rom[name] = rom
            diffs = np.diff(np.asarray(valid, dtype=float))
            if len(diffs) >= 2 and ctx.fps > 0:
                vel = diffs * ctx.fps
                mean_v = float(np.mean(np.abs(vel)))
                if mean_v > 1e-6:
                    ang_vel_cv[name] = float(np.std(vel) / mean_v)
            jerk = _mean_jerk(seq)
            if jerk is not None and rom > 1e-6:
                norm_jerks.append(jerk / rom)

    smooth_metric = stability.metric("joint_smoothness")
    motion_smoothness = smooth_metric.score if smooth_metric else None

    out = {
        "left_knee_rom_deg": left_rom,
        "right_knee_rom_deg": right_rom,
        "knee_rom_asymmetry": knee_asym,
        "knee_angle_source": knee.source,
        "hip_rom_deg": hip_rom or None,
        "ankle_rom_deg": ankle_rom or None,
        "angular_velocity_variability": ang_vel_cv or None,
        "mean_normalized_joint_jerk": _mean_or_none(norm_jerks),
        "motion_smoothness_score": motion_smoothness,
    }
    return {k: _round_val(v) if not isinstance(v, dict) else v for k, v in out.items()}


def _gait_consistency_section(
    features: Any,
    cycles: GaitCycleAnalysisResult,
    stability: StabilityResult,
) -> dict[str, Any]:
    cc = features.cycle_consistency if features else None
    contact = stability.metric("contact_pattern")
    contact_vals = contact.values if contact else {}

    rmse_vals = dict(cc.cycle_to_cycle_rmse) if cc else {}
    mean_rmse = _mean_or_none(list(rmse_vals.values())) if rmse_vals else None

    out = {
        "cycle_to_cycle_rmse": rmse_vals or None,
        "mean_cycle_to_cycle_rmse": mean_rmse,
        "gait_cycle_repeatability": cc.cycle_repeatability_score if cc else None,
        "left_right_phase_consistency": cc.left_right_phase_consistency if cc else None,
        "gait_cycle_consistency_temporal": cycles.metrics.gait_cycle_consistency,
        "contact_pattern_score": contact.score if contact else None,
        "contact_toggle_rate_hz": contact_vals.get("contact_toggle_rate_hz"),
        "double_support_fraction": contact_vals.get("double_support_fraction"),
        "heel_strike_count": contact_vals.get("heel_strike_count"),
        "angle_source": cc.angle_source if cc else None,
        "cycle_count": cc.cycle_count if cc else cycles.metrics.gait_cycle_count,
    }
    return {k: _round_val(v) if not isinstance(v, dict) else v for k, v in out.items()}


def _stability_section(stability: StabilityResult) -> dict[str, Any]:
    domains: dict[str, Any] = {}
    for m in stability.metrics:
        domains[m.key] = {
            "name": m.name,
            "score": m.score,
            "confidence": m.confidence,
            "availability": m.availability,
            "contribution": m.contribution,
            "weight": m.weight,
        }
    validity = stability.validity or assess_stability_result_validity(stability)
    gait_summary = stability.gait_summary or build_gait_analysis_summary(stability)
    return {
        "overall_score": round(stability.score, 2),
        "classification": stability.classification,
        "confidence_badge": stability.confidence_badge,
        "analysis_completeness_pct": round(stability.completeness_pct, 2),
        "usable_gait_cycles": stability.usable_gait_cycles,
        "result_validity": validity.to_dict(),
        "comparable_score": validity.comparable_score,
        "gait_analysis_summary": gait_summary.to_dict(),
        "view_type": stability.view_type,
        "view_confidence": round(stability.view_confidence, 3),
        "view_display_name": stability.view_display_name,
        "view_reliability_table": [
            {"metric": name, "reliability": tier}
            for name, tier in stability.view_reliability_table
        ],
        "explanation": stability.explanation,
        "primary_issue": stability.primary_issue,
        "data_limitations": list(stability.data_limitations),
        "domains": domains,
        "contributions": [c.to_dict() for c in stability.contributions],
    }


def build_comparison_report(
    records: Sequence[VideoComparisonRecord],
) -> dict[str, Any]:
    """Assemble full comparison report with diagnostic interpretation."""
    comparability = assess_cross_video_comparability(
        [r.sections.get("video_quality", {}) for r in records]
    )
    return {
        "report_version": REPORT_VERSION,
        "pipeline": "stablewalk_label_blind_v2",
        "label_policy": (
            "Ground-truth gait labels are attached for comparison only. "
            "They are never passed to pose reconstruction, gait metrics, or stability scoring."
        ),
        "cross_video_comparability": comparability.to_dict(),
        "videos": [r.to_dict() for r in records],
        "interpretation": generate_diagnostic_interpretation(records, comparability),
    }


def generate_diagnostic_interpretation(
    records: Sequence[VideoComparisonRecord],
    comparability: Any | None = None,
) -> dict[str, Any]:
    """Answer structured diagnostic questions from measured features."""
    by_label = {r.ground_truth_label: r for r in records}
    abnormal = by_label.get("abnormal")
    normal = by_label.get("normal")
    athletic = by_label.get("athletic")

    numeric_map = _numeric_feature_map(records)

    ab_vs_norm = _rank_discriminators(numeric_map, "abnormal", "normal")
    ath_vs_norm = _rank_discriminators(numeric_map, "athletic", "normal")

    stability_scores = {
        r.ground_truth_label: r.sections.get("stability", {}).get("overall_score")
        for r in records
    }

    ab_score = stability_scores.get("abnormal")
    norm_score = stability_scores.get("normal")
    ath_score = stability_scores.get("athletic")
    ab_norm_gap = (
        abs(ab_score - norm_score)
        if ab_score is not None and norm_score is not None
        else None
    )

    validity = _measurement_validity_assessment(records)
    monocular_limits = _monocular_limitations(records)
    if comparability is not None and comparability.warning:
        monocular_limits.append(comparability.warning)

    stability_comparisons = _stability_pairwise_comparisons(records)

    return {
        "questions": {
            "1_abnormal_vs_normal_top_features": ab_vs_norm[:8],
            "2_athletic_vs_normal_top_features": ath_vs_norm[:8],
            "3_measurement_validity": validity,
            "4_monocular_mediapipe_limitations": monocular_limits,
            "5_stability_reflects_measurements": _stability_reflection(
                records, ab_vs_norm, ath_vs_norm, stability_scores, stability_comparisons
            ),
            "6_abnormal_normal_within_five_points": _within_five_points_explanation(
                records, ab_score, norm_score, ab_norm_gap
            ),
            "7_stability_score_comparisons": stability_comparisons,
        },
        "cross_video_comparability": comparability.to_dict() if comparability else None,
        "stability_scores": stability_scores,
        "score_spread": {
            "abnormal_vs_normal": ab_norm_gap,
            "athletic_vs_normal": (
                abs(ath_score - norm_score)
                if ath_score is not None and norm_score is not None
                else None
            ),
            "max_minus_min": (
                max(stability_scores.values()) - min(stability_scores.values())
                if all(v is not None for v in stability_scores.values())
                else None
            ),
        },
    }


def _numeric_feature_map(
    records: Sequence[VideoComparisonRecord],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for rec in records:
        flat = flatten_sections(rec.sections)
        out[rec.ground_truth_label] = {
            k: v for k, v in flat.items() if isinstance(v, (int, float)) and v is not None
        }
    return out


def _rank_discriminators(
    numeric_map: dict[str, dict[str, float]],
    label_a: str,
    label_b: str,
    *,
    biomechanical_only: bool = True,
) -> list[dict[str, Any]]:
    a = numeric_map.get(label_a, {})
    b = numeric_map.get(label_b, {})
    keys = set(a) & set(b)
    ranked: list[dict[str, Any]] = []
    for key in keys:
        if biomechanical_only and _is_scoring_metadata_feature(key):
            continue
        va, vb = a[key], b[key]
        delta = va - vb
        denom = max(abs(va), abs(vb), 1e-6)
        ranked.append(
            {
                "feature": key,
                f"{label_a}": round(va, 4),
                f"{label_b}": round(vb, 4),
                "delta": round(delta, 4),
                "relative_delta_pct": round(100.0 * abs(delta) / denom, 2),
            }
        )
    ranked.sort(key=lambda x: x["relative_delta_pct"], reverse=True)
    return ranked


def _is_scoring_metadata_feature(key: str) -> bool:
    """Exclude confidence/contribution/weight fields from biomechanical ranking."""
    if key.startswith("stability.contributions"):
        return True
    if ".confidence" in key or ".contribution" in key or ".weight" in key:
        return True
    if key.endswith(".availability"):
        return True
    if key.startswith("stability.domains.") and key.endswith(".score"):
        return False
    if key.startswith("stability."):
        return key not in ("stability.overall_score",)
    return False


def _measurement_validity_assessment(
    records: Sequence[VideoComparisonRecord],
) -> dict[str, Any]:
    per_video: dict[str, Any] = {}
    for rec in records:
        vq = rec.sections.get("video_quality", {})
        ge = rec.sections.get("gait_events", {})
        st = rec.sections.get("stability", {})
        per_video[rec.ground_truth_label] = {
            "valid_pose_frame_pct": vq.get("valid_pose_frame_pct"),
            "foot_landmark_confidence": vq.get("foot_landmark_confidence"),
            "analysis_completeness_pct": st.get("analysis_completeness_pct"),
            "contact_confidence_tier": ge.get("contact_confidence_tier"),
            "heel_strikes_total": (
                (ge.get("left_heel_strikes") or 0) + (ge.get("right_heel_strikes") or 0)
            ),
            "confidence_badge": st.get("confidence_badge"),
            "supported": (
                (vq.get("valid_pose_frame_pct") or 0) >= 70
                and (ge.get("contact_confidence_tier") != "LOW_CONFIDENCE")
                and (st.get("analysis_completeness_pct") or 0) >= 45
            ),
        }
    return {
        "per_video": per_video,
        "summary": (
            "Differences are best supported when valid_pose_frame_pct is high, "
            "contact_confidence_tier is not LOW_CONFIDENCE, and analysis_completeness_pct ≥ 45%."
        ),
    }


def _monocular_limitations(records: Sequence[VideoComparisonRecord]) -> list[str]:
    limits = [
        "Depth and forward motion are heuristic from monocular MediaPipe image landmarks.",
        "Foot contact and clearance depend on estimated floor Y in hip-centered 3D.",
        "Step/stride length uses body-normalized scaling, not calibrated floor distance.",
        "2D knee flexion from interior angles differs from clinical joint coordinate systems.",
        "Short demo clips limit heel-strike counts and cycle-consistency reliability.",
    ]
    low_contact = [
        r.ground_truth_label
        for r in records
        if r.sections.get("gait_events", {}).get("contact_confidence_tier") == "LOW_CONFIDENCE"
    ]
    if low_contact:
        limits.append(
            f"LOW_CONFIDENCE contact tier for: {', '.join(low_contact)} — temporal/spatial "
            "symmetry and contact domains are less reliable."
        )
    return limits


def _stability_pairwise_comparisons(
    records: Sequence[VideoComparisonRecord],
) -> dict[str, Any]:
    """Compare stability results with explicit reliability verdicts."""
    by_label = {r.ground_truth_label: r for r in records}
    pairs = (
        ("abnormal", "normal"),
        ("normal", "athletic"),
        ("abnormal", "athletic"),
    )
    comparisons: dict[str, Any] = {}
    for label_a, label_b in pairs:
        rec_a = by_label.get(label_a)
        rec_b = by_label.get(label_b)
        if rec_a is None or rec_b is None:
            continue
        if rec_a.stability_result is None or rec_b.stability_result is None:
            continue
        result = compare_stability_results(
            rec_a.stability_result,
            rec_b.stability_result,
            label_a=label_a,
            label_b=label_b,
        )
        comparisons[f"{label_a}_vs_{label_b}"] = result.to_dict()
    return comparisons


def _stability_reflection(
    records: Sequence[VideoComparisonRecord],
    ab_vs_norm: list[dict[str, Any]],
    ath_vs_norm: list[dict[str, Any]],
    stability_scores: dict[str, float | None],
    stability_comparisons: dict[str, Any],
) -> str:
    if not stability_scores or any(v is None for v in stability_scores.values()):
        return "Stability scores incomplete — cannot assess reflection."

    not_comparable = [
        key.replace("_vs_", " vs ")
        for key, comp in stability_comparisons.items()
        if comp.get("verdict") in ("NOT_COMPARABLE", "NO_RELIABLE_DIFFERENCE")
    ]
    if not_comparable:
        parts = [
            "Overall stability scores must not be ranked without validity gating.",
            f"Non-comparable or unreliable pairs: {', '.join(not_comparable)}.",
        ]
        for key, comp in stability_comparisons.items():
            if comp.get("verdict") in ("NOT_COMPARABLE", "NO_RELIABLE_DIFFERENCE"):
                parts.append(comp.get("explanation", ""))
        top_ab = ab_vs_norm[0]["feature"] if ab_vs_norm else "n/a"
        top_ath = ath_vs_norm[0]["feature"] if ath_vs_norm else "n/a"
        parts.append(
            f"Largest measured abnormal↔normal separator: {top_ab}. "
            f"Largest measured athletic↔normal separator: {top_ath}. "
            "Review domain contributions — not raw overall scores — when evidence is limited."
        )
        return " ".join(parts)

    ordered = sorted(
        ((k, v) for k, v in stability_scores.items() if v is not None),
        key=lambda x: x[1],
    )
    top_ab = ab_vs_norm[0]["feature"] if ab_vs_norm else "n/a"
    top_ath = ath_vs_norm[0]["feature"] if ath_vs_norm else "n/a"
    return (
        f"Comparable stability ordering (low→high): "
        f"{', '.join(f'{k}={v:.1f}' for k, v in ordered)}. "
        f"Largest measured abnormal↔normal separator: {top_ab}. "
        f"Largest measured athletic↔normal separator: {top_ath}. "
        "Compare domain contributions in each video's stability section to see "
        "whether score gaps match raw feature gaps."
    )


def _within_five_points_explanation(
    records: Sequence[VideoComparisonRecord],
    ab_score: float | None,
    norm_score: float | None,
    gap: float | None,
) -> str:
    if ab_score is None or norm_score is None or gap is None:
        return "Insufficient stability scores for abnormal vs normal comparison."
    if gap > 5.0:
        return (
            f"Abnormal ({ab_score:.1f}) and normal ({norm_score:.1f}) differ by "
            f"{gap:.1f} points — not within five points."
        )

    ab_rec = next((r for r in records if r.ground_truth_label == "abnormal"), None)
    norm_rec = next((r for r in records if r.ground_truth_label == "normal"), None)
    if ab_rec is None or norm_rec is None:
        return "Missing records for explanation."

    lines = [
        f"Abnormal ({ab_score:.1f}) and normal ({norm_score:.1f}) are within "
        f"{gap:.1f} stability points. Likely reasons:",
    ]
    for label, rec in (("abnormal", ab_rec), ("normal", norm_rec)):
        st = rec.sections.get("stability", {})
        lines.append(
            f"  {label}: completeness={st.get('analysis_completeness_pct')}%, "
            f"badge={st.get('confidence_badge')}"
        )
        contribs = st.get("contributions") or []
        active = [c for c in contribs if c.get("contribution", 0) > 0]
        active.sort(key=lambda c: c.get("contribution", 0), reverse=True)
        if active:
            top = active[0]
            lines.append(
                f"    top contributor: {top.get('name')} "
                f"(score={top.get('score')}, contrib={top.get('contribution')})"
            )
        unavailable = [
            c.get("name")
            for c in contribs
            if c.get("availability") == "UNAVAILABLE"
        ]
        if unavailable:
            lines.append(f"    unavailable domains: {', '.join(unavailable[:3])}")

    lines.append(
        "Confidence-weighted renormalization pulls scores toward the mid-range when "
        "domains are LOW_CONFIDENCE or excluded, compressing separation."
    )
    return "\n".join(lines)


def flatten_sections(sections: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in sections.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_sections(value, path))
        else:
            flat[path] = value
    return flat


def write_comparison_csv(report: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for video in report.get("videos", []):
        label = video.get("ground_truth_label", "")
        stem = video.get("video_stem", "")
        flat = flatten_sections(
            {k: v for k, v in video.items() if k not in ("ground_truth_label", "video_stem")}
        )
        row = {"ground_truth_label": label, "video_stem": stem, **flat}
        rows.append(row)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def format_terminal_table(records: Sequence[VideoComparisonRecord]) -> str:
    cols = [
        ("label", 10),
        ("score", 7),
        ("cadence", 8),
        ("L HS", 5),
        ("R HS", 5),
        ("cycles", 7),
        ("compl%", 7),
        ("contact", 10),
        ("validity", 12),
    ]
    header = "".join(f"{name:>{w}}" for name, w in cols)
    lines = ["GAIT COMPARISON (label-blind pipeline)", "=" * len(header), header, "-" * len(header)]
    for rec in records:
        tf = rec.sections.get("temporal_features", {})
        ge = rec.sections.get("gait_events", {})
        st = rec.sections.get("stability", {})
        vq = rec.sections.get("video_quality", {})
        row = [
            rec.ground_truth_label[:10],
            f"{st.get('overall_score', 0):.1f}" if st.get("overall_score") is not None else "N/A",
            f"{tf.get('cadence_steps_per_min', 0):.0f}" if tf.get("cadence_steps_per_min") else "N/A",
            str(ge.get("left_heel_strikes", "—")),
            str(ge.get("right_heel_strikes", "—")),
            str(ge.get("complete_gait_cycles", "—")),
            f"{vq.get('analysis_completeness_pct', st.get('analysis_completeness_pct', 0)):.0f}",
            str(ge.get("contact_confidence_tier", "—"))[:10],
        ]
        validity = (st.get("result_validity") or {}).get("status", "—")
        row_display = row + [str(validity)[:12]]
        lines.append("".join(f"{val:>{w}}" for val, (_, w) in zip(row_display, cols)))
    return "\n".join(lines)


def format_interpretation_text(report: dict[str, Any]) -> str:
    interp = report.get("interpretation", {})
    questions = interp.get("questions", {})
    lines = ["", "DIAGNOSTIC INTERPRETATION", "=" * 72, ""]

    lines.append("1. Features most distinguishing abnormal vs normal:")
    for item in questions.get("1_abnormal_vs_normal_top_features", [])[:5]:
        lines.append(
            f"   - {item['feature']}: abnormal={item.get('abnormal')} "
            f"normal={item.get('normal')} (Δ {item.get('relative_delta_pct')}%)"
        )

    lines.append("")
    lines.append("2. Features most distinguishing athletic vs normal:")
    for item in questions.get("2_athletic_vs_normal_top_features", [])[:5]:
        lines.append(
            f"   - {item['feature']}: athletic={item.get('athletic')} "
            f"normal={item.get('normal')} (Δ {item.get('relative_delta_pct')}%)"
        )

    lines.append("")
    lines.append("3. Measurement validity:")
    lines.append(f"   {questions.get('3_measurement_validity', {}).get('summary', '')}")
    for label, info in questions.get("3_measurement_validity", {}).get("per_video", {}).items():
        lines.append(
            f"   {label}: supported={info.get('supported')} "
            f"pose%={info.get('valid_pose_frame_pct')} "
            f"contact={info.get('contact_confidence_tier')}"
        )

    lines.append("")
    lines.append("4. Monocular / MediaPipe limitations:")
    for item in questions.get("4_monocular_mediapipe_limitations", []):
        lines.append(f"   - {item}")

    lines.append("")
    lines.append("5. Stability score vs measurements:")
    lines.append(f"   {questions.get('5_stability_reflects_measurements', '')}")

    lines.append("")
    lines.append("6. Abnormal vs normal within five points:")
    lines.append(questions.get("6_abnormal_normal_within_five_points", ""))

    lines.append("")
    lines.append("7. Stability score comparison reliability:")
    comparisons = questions.get("7_stability_score_comparisons", {})
    for pair, comp in comparisons.items():
        lines.append(
            f"   {pair.replace('_vs_', ' vs ')}: {comp.get('verdict')} — "
            f"{comp.get('explanation', '')}"
        )

    spread = interp.get("score_spread", {})
    lines.append("")
    lines.append(
        f"Score spread — abnormal vs normal: {spread.get('abnormal_vs_normal')} | "
        f"athletic vs normal: {spread.get('athletic_vs_normal')} | "
        f"max-min: {spread.get('max_minus_min')}"
    )
    return "\n".join(lines)


def _mean_or_none(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return round(float(statistics.mean(vals)), 4)


def _round_val(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 4)
    return value


__all__ = [
    "COMPARISON_VIDEOS",
    "VideoComparisonRecord",
    "analyze_motion_label_blind",
    "build_comparison_report",
    "generate_diagnostic_interpretation",
    "write_comparison_csv",
    "write_comparison_json",
    "format_terminal_table",
    "format_interpretation_text",
    "flatten_sections",
]
