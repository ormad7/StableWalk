"""
Final demo validation report for Abnormal / Normal / Performance categories.

Runs the same label-blind pipeline on each video. Category names appear only in
the report — never in scoring.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from stablewalk.analysis.gait_analysis_summary import (
    GAIT_QUALITY_DOMAIN_KEYS,
    MOVEMENT_STABILITY_DOMAIN_KEYS,
    build_gait_analysis_summary,
)
from stablewalk.analysis.gait_comparison_validation import (
    COMPARISON_VIDEOS,
    VideoComparisonRecord,
    _rank_discriminators,
    analyze_motion_label_blind,
    assess_cross_video_comparability,
)
from stablewalk.analysis.stability_validity import (
    compare_stability_results,
    assess_stability_result_validity,
)
from stablewalk.demo.candidate_validation import validate_demo_candidate
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path, example_by_key

UI_LABELS: dict[str, str] = {
    "abnormal": "Abnormal",
    "normal": "Normal",
    "athletic": "Performance",
}

DATASET_INFO: dict[str, dict[str, str]] = {
    "abnormal": {
        "dataset": "University of Utah NeuroLogic Examination (legacy installed)",
        "target_dataset": "GAVD — clinically annotated abnormal gait",
        "sequence_id": "gait_ab_10 (Utah neuropathic gait clip, frames 510–659)",
        "replacement_status": "Pending GAVD candidate — current file REJECTED by validator",
    },
    "normal": {
        "dataset": "Pexels stock footage (legacy installed)",
        "target_dataset": "Health&Gait UGS usual gait speed trial",
        "sequence_id": "Not assigned — legacy distant 4K clip",
        "replacement_status": "Pending Health&Gait-sourced MP4 — current file REJECTED by validator",
    },
    "athletic": {
        "dataset": "Pexels stock footage (legacy installed)",
        "target_dataset": "Health&Gait FGS fast gait speed trial",
        "sequence_id": "Not assigned — legacy vertical athletic clip",
        "replacement_status": "Pending Health&Gait-sourced MP4 — validator REJECT (subject scale gate)",
    },
}

PAIR_LABELS: tuple[tuple[str, str, str], ...] = (
    ("abnormal", "normal", "Abnormal vs Normal"),
    ("normal", "athletic", "Normal vs Performance"),
    ("abnormal", "athletic", "Abnormal vs Performance"),
)


@dataclass
class DemoValidationBundle:
    records: list[VideoComparisonRecord]
    comparability: Any


def _subject_bbox_fraction(sequence) -> float | None:
    """Body height as fraction of frame (MediaPipe y is normalized 0–1)."""
    spans: list[float] = []
    for frame in sequence.frames:
        if not frame.detected or not frame.keypoints:
            continue
        ys = [kp.y for kp in frame.keypoints if kp.visibility >= 0.5]
        if len(ys) >= 4:
            spans.append(max(ys) - min(ys))
    return statistics.median(spans) if spans else None


def _foot_visibility_pct(sequence) -> float | None:
    from stablewalk.analysis.gait_comparison_validation import FOOT_LANDMARKS

    total = 0
    visible = 0
    for frame in sequence.frames:
        if not frame.detected or not frame.keypoints:
            continue
        total += len(FOOT_LANDMARKS)
        kp_map = {kp.name: kp for kp in frame.keypoints}
        for name in FOOT_LANDMARKS:
            kp = kp_map.get(name)
            if kp and kp.visibility >= 0.5:
                visible += 1
    if total == 0:
        return None
    return 100.0 * visible / total


def load_demo_records(*, use_cached_poses: bool = True) -> DemoValidationBundle:
    from stablewalk import config
    from stablewalk.pose.estimation import PoseEstimator
    from stablewalk.video_processing import VideoProcessor

    def _resolve_video(stem: str) -> Path:
        for folder in (config.DEMO_VIDEOS_DIR, config.VIDEOS_DIR, config.LEGACY_VIDEOS_DIR):
            candidate = folder / f"{stem}.mp4"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Demo video not found: {stem}.mp4")

    def _load_or_estimate_poses(video_path: Path) -> Path:
        stem = video_path.stem
        poses_path = config.POSES_DIR / f"{stem}_poses.json"
        if use_cached_poses and poses_path.is_file():
            return poses_path

        frames_dir = config.FRAMES_DIR / stem
        frames_dir.mkdir(parents=True, exist_ok=True)
        processor = VideoProcessor()
        result = processor.extract_frames(str(video_path), frames_dir)
        with PoseEstimator() as est:
            seq = est.process_directory(
                frames_dir,
                source_video=str(video_path),
                fps=result.fps,
            )
            est.save_sequence(seq, poses_path)
        return poses_path

    records: list[VideoComparisonRecord] = []
    for label, stem in COMPARISON_VIDEOS:
        video_path = _resolve_video(stem)
        poses_path = _load_or_estimate_poses(video_path)
        sequence = load_pose_sequence(poses_path)
        enrich_pose_sequence(sequence)
        sections, stability = analyze_motion_label_blind(sequence)
        validator = validate_demo_candidate(video_path, category=label)
        info = dict(DATASET_INFO.get(label, {}))
        if validator.verdict in ("ACCEPT", "ACCEPT_WITH_LIMITATIONS"):
            info["replacement_status"] = f"Installed — validator {validator.verdict}"
        sections["source_quality_extra"] = {
            "dataset_info": info,
            "subject_bbox_fraction": _round(_subject_bbox_fraction(sequence)),
            "foot_visibility_pct": _round(_foot_visibility_pct(sequence)),
            "validator": validator.to_dict(),
        }
        records.append(
            VideoComparisonRecord(
                ground_truth_label=label,
                video_stem=stem,
                sections=sections,
                stability_result=stability,
            )
        )
    comparability = assess_cross_video_comparability(
        [r.sections.get("video_quality", {}) for r in records]
    )
    return DemoValidationBundle(records=records, comparability=comparability)


def _round(val: float | None, nd: int = 2) -> float | None:
    if val is None:
        return None
    return round(float(val), nd)


def _domain_score(stability, key: str) -> float | None:
    m = stability.metric(key)
    if m is None or m.score is None:
        return None
    return round(float(m.score), 1)


def _top_contributors(stability, n: int = 3) -> list[dict[str, Any]]:
    contribs = sorted(
        stability.contributions,
        key=lambda c: abs(c.contribution),
        reverse=True,
    )
    out = []
    for c in contribs[:n]:
        out.append(
            {
                "name": c.name,
                "key": c.key,
                "score": round(c.score, 1) if c.score is not None else None,
                "contribution": round(c.contribution, 2),
                "confidence": round(c.confidence, 2),
            }
        )
    return out


def render_video_section(rec: VideoComparisonRecord) -> str:
    label = rec.ground_truth_label
    ui = UI_LABELS.get(label, label)
    vq = rec.sections.get("video_quality", {})
    ge = rec.sections.get("gait_events", {})
    tf = rec.sections.get("temporal_features", {})
    gc = rec.sections.get("gait_consistency", {})
    st = rec.sections.get("stability", {})
    extra = rec.sections.get("source_quality_extra", {})
    info = extra.get("dataset_info", {})
    stability = rec.stability_result
    gs = stability.gait_summary if stability else None
    validity = (st.get("result_validity") or {})

    lines = [f"## {ui}", ""]

    lines += [
        "### SOURCE QUALITY",
        f"- **Dataset name:** {info.get('dataset', '—')}",
        f"- **Target dataset:** {info.get('target_dataset', '—')}",
        f"- **Sequence ID:** {info.get('sequence_id', '—')}",
        f"- **Replacement status:** {info.get('replacement_status', '—')}",
        f"- **Detected camera view:** {vq.get('view_display_name') or vq.get('view_type', '—')} "
        f"(confidence {vq.get('view_confidence', '—')})",
        f"- **Subject size in frame:** "
        f"{(extra.get('subject_bbox_fraction') or 0) * 100:.1f}% of frame height (median bbox)",
        f"- **Pose valid frame %:** {vq.get('valid_pose_frame_pct', '—')}%",
        f"- **Foot visibility %:** {extra.get('foot_visibility_pct', '—')}%",
        f"- **Duration:** {vq.get('duration_s', '—')} s",
        f"- **FPS:** {vq.get('fps', '—')}",
        "",
    ]

    hs_total = (ge.get("left_heel_strikes") or 0) + (ge.get("right_heel_strikes") or 0)
    to_total = (ge.get("left_toe_offs") or 0) + (ge.get("right_toe_offs") or 0)
    lines += [
        "### GAIT EVIDENCE",
        f"- **Detected steps (heel strikes):** {hs_total}",
        f"- **Complete gait cycles:** {ge.get('complete_gait_cycles', '—')}",
        f"- **Usable gait cycles:** {st.get('usable_gait_cycles', '—')}",
        f"- **Heel-strike count:** L {ge.get('left_heel_strikes', '—')} / "
        f"R {ge.get('right_heel_strikes', '—')}",
        f"- **Toe-off count:** L {ge.get('left_toe_offs', '—')} / "
        f"R {ge.get('right_toe_offs', '—')}",
        f"- **Gait-cycle confidence:** {ge.get('contact_confidence', '—')} "
        f"({ge.get('contact_confidence_tier', '—')})",
        "",
    ]

    if stability:
        lines += [
            "### MOVEMENT STABILITY",
            f"- **Group score:** "
            f"{gs.movement_stability.score if gs else '—'}",
            f"- **Pelvis control:** {_domain_score(stability, 'pelvis_stability')}",
            f"- **Trunk control:** {_domain_score(stability, 'trunk_stability')}",
            f"- **Root-relative smoothness:** {_domain_score(stability, 'joint_smoothness')}",
            f"- **Trajectory consistency:** {gc.get('gait_cycle_repeatability') or gc.get('mean_cycle_to_cycle_rmse', '—')}",
            "",
            "### GAIT QUALITY",
            f"- **Group score:** {gs.gait_quality.score if gs else '—'}",
            f"- **Stance symmetry:** {_domain_score(stability, 'temporal_symmetry')} "
            f"(timing index {tf.get('stance_symmetry_index', '—')})",
            f"- **Swing symmetry:** {tf.get('swing_symmetry_index', '—')}",
            f"- **Temporal symmetry:** {_domain_score(stability, 'temporal_symmetry')}",
            f"- **Foot-clearance symmetry:** {_domain_score(stability, 'foot_clearance')}",
            f"- **Contact sequencing:** {_domain_score(stability, 'contact_pattern')}",
            f"- **Joint coordination / cycle repeatability:** "
            f"{_domain_score(stability, 'cycle_consistency')} "
            f"(repeatability {gc.get('gait_cycle_repeatability', '—')})",
            "",
            "### ANALYSIS VALIDITY",
            f"- **Completeness:** {st.get('analysis_completeness_pct', '—')}%",
            f"- **Confidence:** {st.get('confidence_badge', '—')}",
            f"- **Result validity:** {validity.get('status', '—')}",
            f"- **Cross-video comparability:** {validity.get('comparable_score', '—')}",
            "",
        ]
    return "\n".join(lines)


def render_pair_comparison(
    rec_a: VideoComparisonRecord,
    rec_b: VideoComparisonRecord,
    *,
    title: str,
) -> str:
    sa = rec_a.stability_result
    sb = rec_b.stability_result
    if sa is None or sb is None:
        return f"### {title}\n\nInsufficient data.\n"

    gs_a = sa.gait_summary or build_gait_analysis_summary(sa)
    gs_b = sb.gait_summary or build_gait_analysis_summary(sb)
    cmp_legacy = compare_stability_results(sa, sb, label_a=rec_a.ground_truth_label, label_b=rec_b.ground_truth_label)

    ms_delta = None
    gq_delta = None
    if gs_a.movement_stability.score is not None and gs_b.movement_stability.score is not None:
        ms_delta = gs_a.movement_stability.score - gs_b.movement_stability.score
    if gs_a.gait_quality.score is not None and gs_b.gait_quality.score is not None:
        gq_delta = gs_a.gait_quality.score - gs_b.gait_quality.score

    contrib_a = _top_contributors(sa)
    contrib_b = _top_contributors(sb)

    lines = [
        f"### {title}",
        "",
        f"- **Movement Stability Δ ({UI_LABELS[rec_a.ground_truth_label]} − "
        f"{UI_LABELS[rec_b.ground_truth_label]}):** "
        f"{ms_delta if ms_delta is not None else '—'}",
        f"- **Gait Quality Δ:** {gq_delta if gq_delta is not None else '—'}",
        f"- **Legacy composite Δ:** {cmp_legacy.score_delta:+.1f} "
        f"(tolerance ±{cmp_legacy.combined_tolerance:.1f})",
        f"- **Comparison verdict:** {cmp_legacy.verdict}",
        f"- **Reliable difference:** "
        f"{'Yes' if cmp_legacy.verdict == 'CLEAR_DIFFERENCE' else 'No' if cmp_legacy.verdict in ('NOT_COMPARABLE', 'NO_RELIABLE_DIFFERENCE') else 'Tentative'}",
        "",
        f"**Largest contributing domains ({UI_LABELS[rec_a.ground_truth_label]}):** "
        + ", ".join(f"{c['name']} ({c['contribution']:+.1f})" for c in contrib_a),
        "",
        f"**Largest contributing domains ({UI_LABELS[rec_b.ground_truth_label]}):** "
        + ", ".join(f"{c['name']} ({c['contribution']:+.1f})" for c in contrib_b),
        "",
        f"**Measurement confidence:** {UI_LABELS[rec_a.ground_truth_label]} "
        f"{cmp_legacy.validity_a}/{cmp_legacy.comparable_a}; "
        f"{UI_LABELS[rec_b.ground_truth_label]} {cmp_legacy.validity_b}/{cmp_legacy.comparable_b}",
        "",
        f"_{cmp_legacy.explanation}_",
        "",
    ]
    return "\n".join(lines)


def _generate_conclusions(bundle: DemoValidationBundle) -> dict[str, list[str]]:
    records = bundle.records
    by_label = {r.ground_truth_label: r for r in records}
    numeric = {
        r.ground_truth_label: {
            k: v
            for k, v in _flatten_numeric(r.sections).items()
            if isinstance(v, (int, float))
        }
        for r in records
    }

    supported: list[str] = []
    tentative: list[str] = []
    unsupported: list[str] = []

    ab = by_label.get("abnormal")
    norm = by_label.get("normal")
    perf = by_label.get("athletic")

    if ab and norm:
        ab_st = ab.sections.get("stability", {})
        norm_st = norm.sections.get("stability", {})
        if (
            ab_st.get("result_validity", {}).get("status") == "INSUFFICIENT_DATA"
            or norm_st.get("result_validity", {}).get("status") == "INSUFFICIENT_DATA"
        ):
            unsupported.append(
                "Abnormal vs Normal overall stability or gait-quality ranking — "
                "at least one clip has INSUFFICIENT_DATA validity."
            )
        else:
            tf_ab = ab.sections.get("temporal_features", {})
            tf_norm = norm.sections.get("temporal_features", {})
            st_ab = tf_ab.get("stance_symmetry_index")
            st_norm = tf_norm.get("stance_symmetry_index")
            if st_ab is not None and st_norm is not None and abs(st_ab - st_norm) > 0.08:
                better = "Normal" if st_norm > st_ab else "Abnormal"
                supported.append(
                    f"{better} gait shows higher stance-timing symmetry index "
                    f"({st_norm:.2f} vs {st_ab:.2f}) in this short clip."
                )

    if perf:
        perf_st = perf.sections.get("stability", {})
        perf_vq = perf.sections.get("video_quality", {})
        if perf_st.get("usable_gait_cycles", 0) >= 3:
            supported.append(
                f"Performance clip provides the strongest gait evidence among installed demos "
                f"({perf_st.get('usable_gait_cycles')} usable cycles, "
                f"{perf_st.get('analysis_completeness_pct', 0):.0f}% completeness)."
            )
        if perf_vq.get("view_display_name", "").lower().find("side") >= 0:
            supported.append(
                "Performance clip is classified as a side view — sagittal knee and pelvis "
                "graphs are the most reliable projections for this recording."
            )

    for label, rec in by_label.items():
        ui = UI_LABELS[label]
        st = rec.sections.get("stability", {})
        extra = rec.sections.get("source_quality_extra", {})
        info = extra.get("dataset_info", {})
        if "legacy" in info.get("dataset", "").lower() or "REJECT" in info.get("replacement_status", ""):
            unsupported.append(
                f"{ui} demo video is a legacy placeholder — scores describe the installed clip, "
                f"not the target {info.get('target_dataset', 'dataset')} protocol."
            )
        if (st.get("analysis_completeness_pct") or 0) < 45:
            tentative.append(
                f"{ui} analysis completeness is below 45% — domain scores are partial estimates."
            )
        if st.get("usable_gait_cycles", 0) < 3:
            tentative.append(
                f"{ui} has fewer than 3 usable gait cycles — cycle-normalized knee graphs "
                "and repeatability metrics are limited."
            )

    if bundle.comparability and getattr(bundle.comparability, "warning", None):
        tentative.append(bundle.comparability.warning)

    ab_vs_norm = _rank_discriminators(numeric, "abnormal", "normal")[:3]
    if ab_vs_norm:
        top = ab_vs_norm[0]
        tentative.append(
            f"Largest measured Abnormal↔Normal feature gap: {top['feature']} "
            f"(Δ {top['delta']}, {top['relative_delta_pct']}% relative)."
        )

    perf_vs_norm = _rank_discriminators(numeric, "athletic", "normal")[:1]
    if perf_vs_norm:
        top = perf_vs_norm[0]
        tentative.append(
            f"Performance vs Normal largest raw feature gap: {top['feature']} — "
            "interpret cautiously across different camera geometries."
        )

    unsupported.append(
        "No clinical validation is claimed. Demo categories do not drive scoring weights."
    )
    if not any("Performance" in s and "more stable" in s.lower() for s in supported + tentative):
        unsupported.append(
            "Performance gait is globally more stable than Normal gait — NOT supported "
            "with current legacy Normal clip (low completeness, oblique view, 2 usable cycles)."
        )

    return {
        "scientifically_supported": supported,
        "tentative": tentative,
        "unsupported": unsupported,
    }


def _flatten_numeric(sections: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in sections.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_numeric(value, path))
        else:
            flat[path] = value
    return flat


def render_markdown_report(bundle: DemoValidationBundle) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# StableWalk Final Demo Validation",
        "",
        f"Generated: {now}",
        "",
        "## Protocol",
        "",
        "- Same label-blind analysis pipeline for every video.",
        "- UI categories (Abnormal / Normal / Performance) are **not** passed to scoring.",
        "- Target is **video-selection quality** (visible body, feet, cycles, confidence) — "
        "not forced score ordering Performance > Normal > Abnormal.",
        "- **Note:** Installed MP4s are legacy placeholders pending GAVD / Health&Gait replacement.",
        "",
        "---",
        "",
    ]

    by_label = {r.ground_truth_label: r for r in bundle.records}
    for rec in bundle.records:
        lines.append(render_video_section(rec))
        lines.append("---")
        lines.append("")

    lines += ["## Pairwise Comparisons", ""]
    for key_a, key_b, title in PAIR_LABELS:
        lines.append(render_pair_comparison(by_label[key_a], by_label[key_b], title=title))

    conclusions = _generate_conclusions(bundle)
    lines += [
        "## Final Summary",
        "",
        "### Scientifically supported conclusions",
        "",
    ]
    for item in conclusions["scientifically_supported"] or ["None at current evidence level."]:
        lines.append(f"- {item}")
    lines += ["", "### Tentative conclusions", ""]
    for item in conclusions["tentative"] or ["None."]:
        lines.append(f"- {item}")
    lines += ["", "### Unsupported conclusions", ""]
    for item in conclusions["unsupported"]:
        lines.append(f"- {item}")

    lines += [
        "",
        "## Demo Selection Quality vs Targets",
        "",
        "| Criterion | Abnormal | Normal | Performance |",
        "|-----------|----------|--------|-------------|",
    ]
    for criterion, keys in (
        ("Full body visible", ("valid_pose_frame_pct",)),
        ("Feet visible", ("source_quality_extra.foot_visibility_pct",)),
        ("Usable cycles ≥ 3", ("stability.usable_gait_cycles",)),
        ("Completeness ≥ 45%", ("stability.analysis_completeness_pct",)),
        ("Confidence MODERATE+", ("stability.confidence_badge",)),
    ):
        row = [criterion]
        for rec in bundle.records:
            st = rec.sections.get("stability", {})
            vq = rec.sections.get("video_quality", {})
            extra = rec.sections.get("source_quality_extra", {})
            if criterion.startswith("Full"):
                val = vq.get("valid_pose_frame_pct")
                row.append("✓" if (val or 0) >= 70 else "✗")
            elif criterion.startswith("Feet"):
                val = extra.get("foot_visibility_pct")
                row.append("✓" if (val or 0) >= 80 else "✗")
            elif criterion.startswith("Usable"):
                val = st.get("usable_gait_cycles")
                row.append("✓" if (val or 0) >= 3 else "✗")
            elif criterion.startswith("Complete"):
                val = st.get("analysis_completeness_pct")
                row.append("✓" if (val or 0) >= 45 else "✗")
            else:
                badge = st.get("confidence_badge", "")
                row.append("✓" if badge in ("MODERATE CONFIDENCE", "HIGH CONFIDENCE") else "✗")
        lines.append("| " + " | ".join(str(x) for x in row) + " |")

    lines += [
        "",
        "## Score Attribution (surprising orderings)",
        "",
    ]
    scores = {
        UI_LABELS[r.ground_truth_label]: r.sections.get("stability", {}).get("overall_score")
        for r in bundle.records
    }
    ms_scores = {
        UI_LABELS[r.ground_truth_label]: (
            r.stability_result.gait_summary.movement_stability.score
            if r.stability_result and r.stability_result.gait_summary
            else None
        )
        for r in bundle.records
    }
    gq_scores = {
        UI_LABELS[r.ground_truth_label]: (
            r.stability_result.gait_summary.gait_quality.score
            if r.stability_result and r.stability_result.gait_summary
            else None
        )
        for r in bundle.records
    }
    lines.append(f"- Legacy composite scores: {scores}")
    lines.append(f"- Movement Stability scores: {ms_scores}")
    lines.append(f"- Gait Quality scores: {gq_scores}")
    lines.append(
        "- If Performance scores highest, attribute primarily to higher completeness, "
        "usable cycles, and side-view reliability — not category labels."
    )
    for rec in bundle.records:
        ui = UI_LABELS[rec.ground_truth_label]
        if rec.stability_result:
            lines.append(f"- **{ui} top contributors:** " + ", ".join(
                f"{c['name']} ({c['contribution']:+.1f})" for c in _top_contributors(rec.stability_result)
            ))

    return "\n".join(lines) + "\n"


def write_final_demo_validation_report(path: Path, *, use_cached_poses: bool = True) -> str:
    bundle = load_demo_records(use_cached_poses=use_cached_poses)
    text = render_markdown_report(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
