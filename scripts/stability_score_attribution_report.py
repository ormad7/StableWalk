"""
Deep stability score attribution diagnostic for the three demo gait videos.

Label-blind: ground-truth labels are for report grouping only.

Usage:
  python scripts/stability_score_attribution_report.py
  python scripts/stability_score_attribution_report.py --use-cached-poses

Writes:
  data/output/reports/stability_score_attribution.txt
  data/output/reports/stability_score_attribution.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.biomech_stability import analyze_biomech_stability
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.analysis.stability_config import DEFAULT_STABILITY_CONFIG
from stablewalk.analysis.stability_scoring import (
    StabilityAnalysisContext,
    _build_context,
    compute_pelvis_motion_metrics,
)
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.enrichment import enrich_pose_sequence

VIDEOS = (
    ("abnormal", "abnormal_gait"),
    ("normal", "normal_gait"),
    ("athletic", "athletic_walking"),
)

DOMAIN_ORDER = (
    "temporal_symmetry",
    "spatial_symmetry",
    "pelvis_stability",
    "trunk_stability",
    "foot_clearance",
    "joint_smoothness",
    "cycle_consistency",
    "contact_pattern",
)

DOMAIN_TITLES = {
    "temporal_symmetry": "Temporal Symmetry",
    "spatial_symmetry": "Spatial Symmetry",
    "pelvis_stability": "Pelvis Stability",
    "trunk_stability": "Trunk Stability",
    "foot_clearance": "Foot Clearance Consistency",
    "joint_smoothness": "Joint Motion Smoothness",
    "cycle_consistency": "Gait Cycle Consistency",
    "contact_pattern": "Contact Pattern",
}


def _resolve_poses(stem: str, *, use_cached: bool) -> Path:
    path = config.POSES_DIR / f"{stem}_poses.json"
    if path.is_file():
        return path
    if use_cached:
        raise FileNotFoundError(f"Missing cached poses: {path}")
    video = config.DEMO_VIDEOS_DIR / f"{stem}.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"Missing video: {video}")
    from stablewalk.pose.estimation import PoseEstimator
    from stablewalk.video_processing import VideoProcessor

    frames_dir = config.FRAMES_DIR / stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    processor = VideoProcessor()
    result = processor.extract_frames(str(video), frames_dir)
    with PoseEstimator() as est:
        seq = est.process_directory(
            frames_dir,
            source_video=str(video),
            fps=result.fps,
        )
        est.save_sequence(seq, path)
    return path


def _analyze(stem: str, *, use_cached: bool) -> tuple[object, StabilityAnalysisContext | None]:
    seq = load_pose_sequence(_resolve_poses(stem, use_cached=use_cached))
    enrich_pose_sequence(seq)
    recording = pose_sequence_to_gait_motion(seq)
    cycles = analyze_gait_cycles(recording)
    result = analyze_biomech_stability(seq, cycles=cycles)
    ctx = _build_context(seq, cycles=cycles, recording=recording)
    return result, ctx


def _debug_features_table(metric) -> list[dict]:
    rows = []
    for feat in metric.values.get("debug_features", []):
        rows.append(
            {
                "feature": feat.get("feature"),
                "raw": feat.get("raw"),
                "normalized": feat.get("normalized"),
                "component_score": feat.get("component_score"),
                "direction": feat.get("direction"),
                "note": feat.get("note"),
            }
        )
    return rows


def _domain_row(result, key: str) -> dict:
    m = result.metric(key)
    if m is None:
        return {"key": key, "name": DOMAIN_TITLES.get(key, key)}
    raw_contrib = (m.score or 0) * m.weight if m.score is not None else None
    return {
        "key": key,
        "name": m.name,
        "score_before_confidence": m.score,
        "confidence": m.confidence,
        "availability": m.availability,
        "weight": m.weight,
        "raw_weighted_contribution": raw_contrib,
        "final_contribution": m.contribution,
        "summary": m.summary,
        "findings": list(m.findings),
        "values": {k: v for k, v in m.values.items() if k != "debug_features"},
        "debug_features": _debug_features_table(m),
    }


def _finalize_denominator(result) -> float:
    return sum(
        c.weight * c.confidence
        for c in result.contributions
        if c.availability != "UNAVAILABLE" and c.score is not None
    )


def _domain_final_share(result, key: str) -> float | None:
    m = result.metric(key)
    if m is None or m.score is None or m.availability == "UNAVAILABLE":
        return None
    denom = _finalize_denominator(result)
    if denom <= 0:
        return None
    return (m.score * m.weight * m.confidence) / denom


def _pelvis_translation_audit(ctx: StabilityAnalysisContext) -> dict:
    """Quantify global image-X translation vs detrended residual."""
    px_raw, px_det = [], []
    import numpy as np

    from stablewalk.analysis.stability_scoring import _detrend, _pelvis_xy

    for f in ctx.frames:
        p = _pelvis_xy(f)
        if p:
            px_raw.append(p[0])
    if len(px_raw) < 3:
        return {}
    arr = np.asarray(px_raw, dtype=float)
    det = _detrend(arr)
    return {
        "coordinate_source": "MediaPipe normalized image (x=horizontal, y=down)",
        "pelvis_image_x_span_raw": float(np.ptp(arr)),
        "pelvis_image_x_span_detrended": float(np.ptp(det)),
        "pelvis_image_x_std_raw": float(np.std(arr)),
        "pelvis_image_x_std_detrended": float(np.std(det)),
        "first_third_mean_x": float(np.mean(arr[: len(arr) // 3])),
        "last_third_mean_x": float(np.mean(arr[-len(arr) // 3 :])),
        "global_translation_x": float(np.mean(arr[-len(arr) // 3 :]) - np.mean(arr[: len(arr) // 3])),
        "compute_pelvis_motion_metrics": compute_pelvis_motion_metrics(ctx),
    }


def _temporal_raw_audit(ctx: StabilityAnalysisContext) -> dict:
    m = ctx.cycles.metrics
    return {
        "left_stance_s": m.left_stance_time_s,
        "right_stance_s": m.right_stance_time_s,
        "left_swing_s": m.left_swing_time_s,
        "right_swing_s": m.right_swing_time_s,
        "stance_symmetry_ratio": (
            min(m.left_stance_time_s, m.right_stance_time_s)
            / max(m.left_stance_time_s, m.right_stance_time_s)
            if m.left_stance_time_s and m.right_stance_time_s
            else None
        ),
        "swing_symmetry_ratio": (
            min(m.left_swing_time_s, m.right_swing_time_s)
            / max(m.left_swing_time_s, m.right_swing_time_s)
            if m.left_swing_time_s and m.right_swing_time_s
            else None
        ),
        "heel_strikes": m.left_heel_strike_count + m.right_heel_strike_count,
        "gait_cycles": m.gait_cycle_count,
        "double_support_pct": m.double_support_pct,
        "contact_confidence": m.contact_confidence,
        "confidence_tier": m.confidence_tier,
    }


def build_report(*, use_cached: bool) -> dict:
    results: dict[str, tuple] = {}
    reports: dict[str, dict] = {}
    for label, stem in VIDEOS:
        result, ctx = _analyze(stem, use_cached=use_cached)
        results[label] = (result, ctx)
        domains = [_domain_row(result, key) for key in DOMAIN_ORDER]
        reports[label] = {
            "video_stem": stem,
            "final_score": result.score,
            "classification": result.classification,
            "completeness_pct": result.completeness_pct,
            "confidence_badge": result.confidence_badge,
            "explanation": result.explanation,
            "data_limitations": list(result.data_limitations),
            "finalize_denominator": _finalize_denominator(result),
            "domains": domains,
            "temporal_raw_audit": _temporal_raw_audit(ctx) if ctx else {},
            "pelvis_translation_audit": _pelvis_translation_audit(ctx) if ctx else {},
            "contribution_table": [c.to_dict() for c in result.contributions],
        }

    athletic = reports["athletic"]
    normal = reports["normal"]
    ath_result, _ = results["athletic"]
    norm_result, _ = results["normal"]
    score_delta = athletic["final_score"] - normal["final_score"]

    domain_breakdown = []
    for key in DOMAIN_ORDER:
        share_a = _domain_final_share(ath_result, key)
        share_n = _domain_final_share(norm_result, key)
        ma = ath_result.metric(key)
        mn = norm_result.metric(key)
        delta_points = None
        if share_a is not None and share_n is not None:
            delta_points = share_a - share_n
        domain_breakdown.append(
            {
                "domain": DOMAIN_TITLES[key],
                "key": key,
                "athletic_score": ma.score if ma else None,
                "normal_score": mn.score if mn else None,
                "athletic_availability": ma.availability if ma else None,
                "normal_availability": mn.availability if mn else None,
                "athletic_confidence": ma.confidence if ma else None,
                "normal_confidence": mn.confidence if mn else None,
                "athletic_final_share_pts": share_a,
                "normal_final_share_pts": share_n,
                "delta_final_points": delta_points,
                "score_delta": (
                    (ma.score - mn.score)
                    if ma and mn and ma.score is not None and mn.score is not None
                    else None
                ),
            }
        )

    domain_breakdown.sort(
        key=lambda x: x["delta_final_points"] if x["delta_final_points"] is not None else 0.0
    )

    return {
        "videos": reports,
        "athletic_minus_normal": {
            "final_score_delta": score_delta,
            "athletic_final": athletic["final_score"],
            "normal_final": normal["final_score"],
            "domain_breakdown": domain_breakdown,
            "sum_domain_delta_points": sum(
                d["delta_final_points"]
                for d in domain_breakdown
                if d["delta_final_points"] is not None
            ),
        },
        "code_audit": {
            "temporal_hardcoded_100": (
                "No hardcoded temporal_score=100. Score comes from ThresholdBand.score() "
                "which returns 100.0 when normalized asymmetry <= good threshold "
                "(stability_config.ThresholdBand lines 36-37)."
            ),
            "gui_asterisk_meaning": (
                "Sidebar shows f'{score:.0f}*' when availability==LOW_CONFIDENCE "
                "(stablewalk/ui/tk/app.py _update_stability_panel lines 2550-2551)."
            ),
            "pelvis_coordinate_source": (
                "Pelvis stability uses _pelvis_xy() = MediaPipe normalized IMAGE coords, "
                "NOT hip-centered canonical 3D. Image X is treated as mediolateral."
            ),
            "pelvis_75_bin_hypothesis": (
                "Pelvis domain score = mean of 4 ThresholdBand sub-scores; 75 often means "
                "three sub-metrics at 100 and one at 0, or continuous interpolation."
            ),
        },
    }


def _format_domain_table(video: dict) -> list[str]:
    lines = [
        f"\n{'=' * 100}",
        f"VIDEO: {video.get('video_stem')} | Final={video['final_score']:.1f} | "
        f"Completeness={video['completeness_pct']:.0f}% | {video['confidence_badge']}",
        f"{'=' * 100}",
        f"{'Domain':<28} {'Score':>6} {'Conf':>6} {'Wt':>5} {'Contrib':>8} {'Avail':<16} Raw/normalized highlights",
        "-" * 100,
    ]
    for d in video["domains"]:
        highlights = []
        for feat in d.get("debug_features", [])[:3]:
            highlights.append(
                f"{feat['feature']}={feat.get('raw')}→{feat.get('normalized')}→{feat.get('component_score')}"
            )
        if not highlights:
            vals = d.get("values") or {}
            for k in list(vals)[:2]:
                highlights.append(f"{k}={vals[k]}")
        hl = "; ".join(highlights)[:48]
        score_s = "N/A" if d["score_before_confidence"] is None else f"{d['score_before_confidence']:.1f}"
        lines.append(
            f"{d['name']:<28} {score_s:>6} {d['confidence']:>6.2f} {d['weight']:>5.2f} "
            f"{d['final_contribution']:>8.2f} {d['availability']:<16} {hl}"
        )
        for feat in d.get("debug_features", []):
            lines.append(
                f"    · {feat['feature']}: raw={feat.get('raw')} norm={feat.get('normalized')} "
                f"component={feat.get('component_score')} ({feat.get('note') or feat.get('direction')})"
            )
    return lines


def format_text_report(report: dict) -> str:
    lines = [
        "STABILITY SCORE ATTRIBUTION DIAGNOSTIC",
        "Label-blind pipeline — categories for grouping only",
        "",
        "SUMMARY SCORES",
        f"  Abnormal:  {report['videos']['abnormal']['final_score']:.1f}",
        f"  Normal:    {report['videos']['normal']['final_score']:.1f}",
        f"  Athletic:  {report['videos']['athletic']['final_score']:.1f}",
        "",
        "ATHLETIC − NORMAL FINAL SCORE BREAKDOWN (points on 0–100 scale)",
        f"  Total delta: {report['athletic_minus_normal']['final_score_delta']:+.2f} "
        f"(sum of domain shares: {report['athletic_minus_normal']['sum_domain_delta_points']:+.2f})",
        "",
    ]
    for row in sorted(
        report["athletic_minus_normal"]["domain_breakdown"],
        key=lambda r: r["delta_final_points"] or 0.0,
    ):
        dp = row["delta_final_points"]
        dp_s = f"{dp:+.2f}" if dp is not None else "N/A"
        lines.append(
            f"  {row['domain']:<32} {dp_s} pts  "
            f"(ath={row['athletic_score']} norm={row['normal_score']} "
            f"Δscore={row['score_delta']})"
        )

    for label in ("abnormal", "normal", "athletic"):
        lines.extend(_format_domain_table(report["videos"][label]))

    lines.extend(
        [
            "",
            "TEMPORAL SYMMETRY AUDIT (raw inputs, all videos)",
        ]
    )
    for label in ("abnormal", "normal", "athletic"):
        t = report["videos"][label]["temporal_raw_audit"]
        lines.append(
            f"  {label}: HS={t.get('heel_strikes')} cycles={t.get('gait_cycles')} "
            f"stance_sym={t.get('stance_symmetry_ratio')} swing_sym={t.get('swing_symmetry_ratio')} "
            f"DS%={t.get('double_support_pct')} contact_conf={t.get('contact_confidence')}"
        )

    lines.extend(["", "PELVIS TRANSLATION AUDIT (image coordinates)"])
    for label in ("abnormal", "normal", "athletic"):
        p = report["videos"][label]["pelvis_translation_audit"]
        if not p:
            continue
        lines.append(
            f"  {label}: raw_x_span={p.get('pelvis_image_x_span_raw'):.4f} "
            f"detrended_x_span={p.get('pelvis_image_x_span_detrended'):.4f} "
            f"global_translation_x={p.get('global_translation_x'):.4f} "
            f"norm_sway={p.get('compute_pelvis_motion_metrics', {}).get('normalized_pelvis_sway')}"
        )

    lines.extend(["", "CODE AUDIT NOTES"])
    for k, v in report["code_audit"].items():
        lines.append(f"  {k}: {v}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stability score attribution diagnostic")
    parser.add_argument("--use-cached-poses", action="store_true")
    args = parser.parse_args()

    report = build_report(use_cached=args.use_cached_poses)
    out_txt = config.REPORTS_DIR / "stability_score_attribution.txt"
    out_json = config.REPORTS_DIR / "stability_score_attribution.json"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(format_text_report(report), encoding="utf-8")
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    print(out_txt.read_text(encoding="utf-8"))
    print(f"\nWrote {out_txt}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
