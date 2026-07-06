"""Debug stability scoring for Abnormal / Normal / Athletic demo videos."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from stablewalk import config
from stablewalk.analysis.biomech_stability import (
    METRIC_WEIGHTS,
    _extract_series,
    analyze_biomech_stability,
)
from stablewalk.pose.gait_step_detection import detect_gait_steps
from stablewalk.analysis.metrics import GaitMetrics
from stablewalk.analysis.stability import analyze_stability
from stablewalk.analysis.forces import ForceAnalyzer
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.pose.events import analyze_gait_sequence
from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path

LANDMARKS = (
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)

METRIC_KEYS = (
    "symmetry",
    "step_consistency",
    "body_stability",
    "range_of_motion",
    "trajectory_smoothness",
    "pose_quality",
)


def _bbox_stats(sequence) -> dict:
    heights, confs = [], {n: [] for n in LANDMARKS}
    for f in sequence.frames:
        if not f.detected or not f.keypoints:
            continue
        ys = [kp.y for kp in f.keypoints if kp.visibility >= 0.3]
        xs = [kp.x for kp in f.keypoints if kp.visibility >= 0.3]
        if ys and xs:
            heights.append(max(ys) - min(ys))
        for kp in f.keypoints:
            if kp.name in confs:
                confs[kp.name].append(kp.visibility)
    avg_h = float(np.mean(heights)) if heights else 0.0
    vis = {k: float(np.mean(v)) if v else 0.0 for k, v in confs.items()}
    return {
        "bbox_height_norm_mean": avg_h,
        "bbox_height_pct_frame": avg_h * 100,
        "landmark_visibility": vis,
        "mean_visibility": float(np.mean(list(vis.values()))) if vis else 0.0,
    }


def _heel_strikes(sequence) -> int:
    events, _ = analyze_gait_sequence(sequence.frames, fps=sequence.fps)
    return sum(1 for e in events if e.event_type == "heel_strike")


def analyze_demo(key: str) -> dict:
    ex = next(e for e in DEMO_GAIT_EXAMPLES if e.key == key)
    path = demo_path(ex)
    with PoseEstimator(video_mode=True) as est:
        seq = est.process_video(path, enrich_gait=False)
    enrich_pose_sequence(seq)

    biomech = analyze_biomech_stability(seq)
    penalty = analyze_stability(seq)
    gait = GaitMetrics().compute(seq, grf=ForceAnalyzer().analyze(seq))
    series = _extract_series(seq)
    bbox = _bbox_stats(seq)

    gait = detect_gait_steps(seq)
    steps_l = gait.left.step_count
    steps_r = gait.right.step_count
    cv_l = gait.left.interval_cv
    cv_r = gait.right.interval_cv
    peaks_l = gait.left.event_frame_indices
    peaks_r = gait.right.event_frame_indices

    metric_map = {m.key: m for m in biomech.metrics}

    return {
        "key": key,
        "path": str(path),
        "fps": seq.fps,
        "total_frames": len(seq.frames),
        "detected_frames": sum(1 for f in seq.frames if f.detected),
        "biomech_score": biomech.score,
        "biomech_class": biomech.classification,
        "penalty_score": penalty.score,
        "penalty_label": penalty.label,
        "penalties": penalty.penalties,
        "metrics": metric_map,
        "gait": gait,
        "series": series,
        "bbox": bbox,
        "peak_steps_l": steps_l,
        "peak_steps_r": steps_r,
        "peak_cv_l": cv_l,
        "peak_cv_r": cv_r,
        "peak_frames_l": peaks_l,
        "peak_frames_r": peaks_r,
        "heel_strikes": _heel_strikes(seq),
        "duration_s": len(seq.frames) / max(seq.fps, 1e-6),
    }


def _print_report(d: dict) -> None:
    print("=" * 72)
    print(f"STABILITY SCORE DEBUG REPORT — {d['key'].upper()}")
    print("=" * 72)
    print(f"Video: {d['path']}")
    print(f"Duration: {d['duration_s']:.2f}s  FPS: {d['fps']:.1f}")
    print(f"Total frames: {d['total_frames']}  Detected: {d['detected_frames']}")
    print()
    print(f"Final Stability Score (GUI / biomech): {d['biomech_score']:.1f}/100 — {d['biomech_class']}")
    print(f"Penalty-based score (legacy):          {d['penalty_score']:.1f}/100 — {d['penalty_label']}")
    print()

    mm = d["metrics"]
    for key in METRIC_KEYS:
        m = mm.get(key)
        label = key.replace("_", " ").title()
        print(f"{label + ' Score:':<28} {m.score if m and m.score is not None else 'N/A'}")
    print()

    body = mm.get("body_stability")
    if body and body.values:
        print(f"Pelvis sway ratio:          {body.values.get('pelvis_lateral_sway_ratio')}")
        print(f"Torso offset ratio:         {body.values.get('torso_lateral_offset_ratio')}")
        print(f"CoM lateral sway ratio:     {body.values.get('com_lateral_sway_ratio')}")

    steps = mm.get("step_consistency")
    if steps and steps.values:
        print(f"Peak-detect steps (L/R):    {steps.values.get('left_steps')} / {steps.values.get('right_steps')}")
        print(f"Step interval CV (L/R):     {steps.values.get('left_interval_cv')} / {steps.values.get('right_interval_cv')}")
        print(f"Foot clearance L/R:         {steps.values.get('foot_clearance_left')} / {steps.values.get('foot_clearance_right')}")

    smooth = mm.get("trajectory_smoothness")
    if smooth and smooth.values:
        jerks = [v for k, v in smooth.values.items() if k.endswith("_mean_jerk")]
        if jerks:
            print(f"Mean joint jerk (avg):        {float(np.mean(jerks)):.3f}")

    g = d["gait"]
    print(f"GaitMetrics cadence:        {g.cadence_hz} Hz ({g.cadence_steps_per_min} spm)")
    print(f"GaitMetrics step_count:     {g.step_count}")
    print(f"GaitMetrics step CV:        {g.step_timing_cv}")
    print(f"Event heel strikes:         {d['heel_strikes']}")
    print()

    print(f"Body bbox height (norm):    {d['bbox']['bbox_height_norm_mean']:.3f} ({d['bbox']['bbox_height_pct_frame']:.1f}% of frame)")
    print(f"Mean landmark visibility:   {d['bbox']['mean_visibility']:.3f}")
    for name in LANDMARKS:
        print(f"  {name:18s} {d['bbox']['landmark_visibility'][name]:.3f}")

    for key in METRIC_KEYS:
        m = mm.get(key)
        if m and m.values:
            print()
            print(f"{key} raw values:")
            for k, v in m.values.items():
                if k.startswith("peak_frames"):
                    continue
                print(f"  {k}: {v}")

    if d["penalties"]:
        print()
        print("Penalty-based deductions:")
        for p in d["penalties"]:
            print(f"  -{p.points:.0f}  {p.rule_id}: {p.reason}")

    findings = []
    for key in METRIC_KEYS:
        m = mm.get(key)
        if m:
            findings.extend(m.findings)
    if findings:
        print()
        print("Biomech findings:")
        for f in findings:
            print(f"  • {f}")
    print()


def main() -> int:
    config.ensure_output_dirs()
    results = {}
    for key in ("abnormal", "normal", "athletic"):
        if not demo_path(next(e for e in DEMO_GAIT_EXAMPLES if e.key == key)).is_file():
            print(f"MISSING: {key}")
            continue
        results[key] = analyze_demo(key)
        _print_report(results[key])

    if len(results) < 3:
        return 1

    print("=" * 72)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 72)
    headers = ["Metric", "Abnormal", "Normal", "Athletic"]
    rows = []

    def score(key: str, metric_key: str) -> str:
        m = results[key]["metrics"].get(metric_key)
        return f"{m.score:.0f}" if m and m.score is not None else "N/A"

    rows.append(("Final score (GUI)", *[
        f"{results[k]['biomech_score']:.0f} ({results[k]['biomech_class']})" for k in results
    ]))
    for mk in METRIC_KEYS:
        rows.append((mk.replace("_", " ").title(), *[score(k, mk) for k in results]))
    rows.append(("Peak steps L+R", *[
        f"{results[k]['peak_steps_l']+results[k]['peak_steps_r']}" for k in results
    ]))
    rows.append(("Heel strikes (events)", *[str(results[k]["heel_strikes"]) for k in results]))
    rows.append(("BBox height % frame", *[
        f"{results[k]['bbox']['bbox_height_pct_frame']:.1f}%" for k in results
    ]))
    rows.append(("Mean visibility", *[
        f"{results[k]['bbox']['mean_visibility']:.2f}" for k in results
    ]))

    col_w = [28, 14, 14, 14]
    print(f"{headers[0]:<{col_w[0]}} {headers[1]:>{col_w[1]}} {headers[2]:>{col_w[2]}} {headers[3]:>{col_w[3]}}")
    print("-" * sum(col_w))
    for row in rows:
        print(f"{row[0]:<{col_w[0]}} {row[1]:>{col_w[1]}} {row[2]:>{col_w[2]}} {row[3]:>{col_w[3]}}")

    print()
    print("BIOMECH WEIGHTS:", METRIC_WEIGHTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
