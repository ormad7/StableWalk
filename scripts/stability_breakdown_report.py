"""Full stability metric breakdown for Abnormal / Normal / Athletic demo videos."""

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
    _mean_abs_diff,
    _mean_jerk,
    _robust_range,
    analyze_biomech_stability,
)
from stablewalk.pose.gait_step_detection import detect_gait_steps
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.pose.events import analyze_gait_sequence
from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path


def _raw_breakdown(key: str) -> dict:
    ex = next(e for e in DEMO_GAIT_EXAMPLES if e.key == key)
    path = demo_path(ex)
    with PoseEstimator(video_mode=True) as est:
        seq = est.process_video(path, enrich_gait=False)
    enrich_pose_sequence(seq)

    result = analyze_biomech_stability(seq)
    s = _extract_series(seq)
    gait = detect_gait_steps(seq)
    steps_l = gait.left.step_count
    steps_r = gait.right.step_count
    cv_l = gait.left.interval_cv
    cv_r = gait.right.interval_cv
    events, _ = analyze_gait_sequence(seq.frames, fps=seq.fps)
    heel_strikes = sum(1 for e in events if e.event_type == "heel_strike")

    pelvis_y = []
    for f in seq.frames:
        if not f.detected:
            continue
        kp = {k.name: k for k in f.keypoints}
        lh, rh = kp.get("left_hip"), kp.get("right_hip")
        if lh and rh and lh.visibility >= 0.3 and rh.visibility >= 0.3:
            pelvis_y.append((lh.y + rh.y) / 2.0)

    shoulder_x = []
    for f in seq.frames:
        if not f.detected:
            continue
        kp = {k.name: k for k in f.keypoints}
        ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
        if ls and rs and ls.visibility >= 0.3 and rs.visibility >= 0.3:
            shoulder_x.append((ls.x + rs.x) / 2.0)

    scaled_l = [v / max(s.body_height, 1e-6) if v is not None else None for v in s.ankle_l_y]
    scaled_r = [v / max(s.body_height, 1e-6) if v is not None else None for v in s.ankle_r_y]
    clr_l, clr_r = _robust_range(scaled_l), _robust_range(scaled_r)
    clr_asym = abs(clr_l - clr_r) / max((clr_l + clr_r) / 2.0, 1e-6)

    jerks = []
    for seq_angles in (s.knee_l, s.knee_r, s.ankle_l, s.ankle_r):
        j = _mean_jerk(seq_angles)
        if j is not None:
            jerks.append(j)

    mm = {m.key: m for m in result.metrics}

    return {
        "key": key,
        "display": ex.display_name,
        "result": result,
        "raw": {
            "temporal_symmetry": mm["symmetry"].values.get("step_timing_balance"),
            "step_regularity_cv": np.mean([c for c in (cv_l, cv_r) if c is not None])
            if any(c is not None for c in (cv_l, cv_r))
            else None,
            "step_interval_cv_l": cv_l,
            "step_interval_cv_r": cv_r,
            "pelvis_lateral_sway": mm["body_stability"].values.get("pelvis_lateral_sway_ratio"),
            "pelvis_vertical_var": float(np.std(pelvis_y)) if len(pelvis_y) >= 3 else None,
            "torso_sway": mm["body_stability"].values.get("torso_lateral_offset_ratio"),
            "shoulder_sway": float(np.std(shoulder_x) / max(s.body_width, 1e-6))
            if len(shoulder_x) >= 3
            else None,
            "com_sway": mm["body_stability"].values.get("com_lateral_sway_ratio"),
            "foot_clearance_consistency": mm["step_consistency"].values.get("foot_clearance_balance"),
            "foot_clearance_asymmetry": round(clr_asym, 3),
            "ankle_smoothness": float(np.mean(jerks)) if jerks else None,
            "knee_smoothness": float(np.mean([
                j for j in (_mean_jerk(s.knee_l), _mean_jerk(s.knee_r)) if j is not None
            ])) if any(_mean_jerk(x) is not None for x in (s.knee_l, s.knee_r)) else None,
            "mean_jerk": float(np.mean(jerks)) if jerks else None,
            "pose_confidence": s.mean_foot_visibility,
            "valid_frame_pct": s.valid_frame_pct,
            "gait_cycles": heel_strikes // 2 if heel_strikes else 0,
            "step_count": steps_l + steps_r,
            "left_steps": steps_l,
            "right_steps": steps_r,
            "knee_symmetry_deg": _mean_abs_diff(s.knee_l, s.knee_r),
            "ankle_symmetry_deg": _mean_abs_diff(s.ankle_l, s.ankle_r),
        },
        "components": {
            "symmetry": mm["symmetry"].score,
            "regularity": mm["step_consistency"].score,
            "pelvis_stability": mm["body_stability"].score,
            "torso_stability": mm["body_stability"].score,
            "foot_clearance": mm["step_consistency"].values.get("foot_clearance_balance"),
            "trajectory_smoothness": mm["trajectory_smoothness"].score,
            "pose_quality": mm.get("pose_quality", mm.get("pose_quality")),
        },
    }


def _print_demo(d: dict) -> None:
    r = d["result"]
    raw = d["raw"]
    print("STABILITY ANALYSIS BREAKDOWN")
    print()
    print(f"Demo: {d['display']}")
    print(f"Final Stability Score: {r.score:.1f}/100 ({r.classification})")
    print()
    print("Raw metrics:")
    labels = [
        ("left/right temporal symmetry", "temporal_symmetry", "{:.2f}"),
        ("step timing regularity (mean CV)", "step_regularity_cv", "{:.3f}"),
        ("step interval coefficient of variation (L/R)", None, None),
        ("pelvis lateral sway", "pelvis_lateral_sway", "{:.3f}"),
        ("pelvis vertical variability", "pelvis_vertical_var", "{:.4f}"),
        ("torso sway", "torso_sway", "{:.3f}"),
        ("shoulder sway", "shoulder_sway", "{:.3f}"),
        ("center-of-mass variability", "com_sway", "{:.3f}"),
        ("foot clearance consistency", "foot_clearance_consistency", "{}"),
        ("left/right foot clearance asymmetry", "foot_clearance_asymmetry", "{:.3f}"),
        ("ankle trajectory smoothness (mean jerk)", "ankle_smoothness", "{:.3f}"),
        ("knee trajectory smoothness (mean jerk)", "knee_smoothness", "{:.3f}"),
        ("jerk / movement smoothness", "mean_jerk", "{:.3f}"),
        ("pose confidence", "pose_confidence", "{:.3f}"),
        ("valid frame percentage", "valid_frame_pct", "{:.1%}"),
        ("detected gait cycles", "gait_cycles", "{}"),
        ("detected step count", "step_count", "{}"),
    ]
    for label, key, fmt in labels:
        if key is None:
            print(
                f"- step interval CV L/R: "
                f"{raw['step_interval_cv_l']} / {raw['step_interval_cv_r']}"
            )
            continue
        val = raw.get(key)
        if val is None:
            print(f"- {label}: N/A")
        elif fmt == "{:.1%}":
            print(f"- {label}: {val:.1%}")
        else:
            print(f"- {label}: {fmt.format(val)}")

    print()
    print("Normalized component scores:")
    for m in r.metrics:
        w = METRIC_WEIGHTS.get(m.key, m.weight)
        print(f"- {m.name}: {m.score:.1f}/100 (weight {w:.0%})")
    print()
    print("Final weighted score:", f"{r.score:.1f}/100")
    print()
    print("-" * 72)


def main() -> int:
    config.ensure_output_dirs()
    keys = ("abnormal", "normal", "athletic")
    data = {}
    for key in keys:
        if not demo_path(next(e for e in DEMO_GAIT_EXAMPLES if e.key == key)).is_file():
            print(f"MISSING demo: {key}")
            return 1
        data[key] = _raw_breakdown(key)
        _print_demo(data[key])

    print("COMPARISON TABLE")
    print()
    header = f"{'Metric':<32} {'Abnormal':>12} {'Normal':>12} {'Athletic':>12}"
    print(header)
    print("-" * len(header))

    def cell(key: str, field: str, fmt: str = "{:.0f}") -> str:
        val = data[key]["raw"].get(field)
        if val is None:
            comp = data[key]["components"].get(field)
            if comp is None:
                return "N/A"
            return fmt.format(comp)
        if isinstance(val, float) and fmt == "{:.0f}":
            return fmt.format(val)
        if isinstance(val, float):
            return fmt.format(val)
        return str(val)

    rows = [
        ("Temporal symmetry", "temporal_symmetry", "{:.2f}"),
        ("Step regularity (CV)", "step_regularity_cv", "{:.3f}"),
        ("Pelvis sway", "pelvis_lateral_sway", "{:.3f}"),
        ("Torso sway", "torso_sway", "{:.3f}"),
        ("Foot clearance consistency", "foot_clearance_consistency", "{}"),
        ("Trajectory smoothness", None, "{:.0f}"),
        ("Pose quality", None, "{:.0f}"),
        ("Final Stability Score", None, "{:.0f}"),
    ]
    for label, field, fmt in rows:
        if field:
            vals = [cell(k, field, fmt) for k in keys]
        elif label == "Trajectory smoothness":
            vals = [f"{data[k]['result'].metric('trajectory_smoothness').score:.0f}" for k in keys]
        elif label == "Pose quality":
            m = data["abnormal"]["result"].metric("pose_quality")
            vals = [
                f"{data[k]['result'].metric('pose_quality').score:.0f}"
                if data[k]["result"].metric("pose_quality")
                else "N/A"
                for k in keys
            ]
        else:
            vals = [f"{data[k]['result'].score:.0f}" for k in keys]
        print(f"{label:<32} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    print()
    print("FINAL SCORE VALIDATION")
    print()
    print(f"{'Demo':<12} {'Score':>8} {'Category':<12}")
    print("-" * 34)
    for key in keys:
        r = data[key]["result"]
        print(f"{key.capitalize():<12} {r.score:>8.1f} {r.classification:<12}")

    print()
    print("METRIC WEIGHTS:", METRIC_WEIGHTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
