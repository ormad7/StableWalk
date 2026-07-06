"""Direct technical comparison: Normal Gait vs Athletic Walking demos."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from stablewalk import config
from stablewalk.analysis.biomech_stability import (
    BODY_SWAY_ZERO_RATIO,
    METRIC_WEIGHTS,
    ROM_DIFF_ZERO_RATIO,
    STEP_CV_ZERO,
    TRAJECTORY_JERK_ZERO,
    _extract_series,
    _linear_score,
    _mean_jerk,
    _robust_range,
    analyze_biomech_stability,
)
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.pose.gait_step_detection import detect_gait_steps
from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path

LANDMARKS = (
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)


def _process(key: str, *, max_frames: int | None = None) -> dict:
    ex = next(e for e in DEMO_GAIT_EXAMPLES if e.key == key)
    path = demo_path(ex)
    with PoseEstimator(video_mode=True) as est:
        seq = est.process_video(path, enrich_gait=False, max_frames=max_frames)
    enrich_pose_sequence(seq)
    result = analyze_biomech_stability(seq)
    gait = detect_gait_steps(seq)
    s = _extract_series(seq)
    return {"key": key, "ex": ex, "path": path, "seq": seq, "result": result, "gait": gait, "series": s}


def _video_meta(path: Path, seq) -> dict:
    cap = cv2.VideoCapture(str(path))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or seq.fps or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = total / fps if fps > 0 else 0.0
    return {
        "resolution": f"{fw}x{fh}",
        "fps": fps,
        "duration_s": duration,
        "total_frames": total,
        "analyzed_frames": len(seq.frames),
    }


def _quality_report(data: dict) -> dict:
    seq = data["seq"]
    s = data["series"]
    gait = data["gait"]
    meta = _video_meta(data["path"], seq)

    vis: dict[str, list[float]] = {n: [] for n in LANDMARKS}
    conf_all: list[float] = []
    body_heights: list[float] = []
    hip_centers: list[tuple[float, float]] = []
    shoulder_centers: list[tuple[float, float]] = []

    for f in seq.frames:
        if not f.detected:
            continue
        kp = {k.name: k for k in f.keypoints}
        for name in LANDMARKS:
            k = kp.get(name)
            if k:
                vis[name].append(k.visibility)
        for k in f.keypoints:
            conf_all.append(k.visibility)
        lh, rh = kp.get("left_hip"), kp.get("right_hip")
        ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
        if lh and rh and ls and rs:
            top = min(ls.y, rs.y, lh.y, rh.y)
            bot = max(ls.y, rs.y, lh.y, rh.y)
            body_heights.append(bot - top)
            hip_centers.append(((lh.x + rh.x) / 2, (lh.y + rh.y) / 2))
            shoulder_centers.append(((ls.x + rs.x) / 2, (ls.y + rs.y) / 2))

    def _mean(name: str) -> float | None:
        arr = vis[name]
        return float(np.mean(arr)) if arr else None

    # Subject translation (hip center drift)
    trans = 0.0
    if len(hip_centers) >= 2:
        deltas = [
            np.hypot(hip_centers[i + 1][0] - hip_centers[i][0], hip_centers[i + 1][1] - hip_centers[i][1])
            for i in range(len(hip_centers) - 1)
        ]
        trans = float(np.sum(deltas))

    cam_motion = "low"
    if len(shoulder_centers) >= 10:
        d = [
            np.hypot(shoulder_centers[i + 1][0] - shoulder_centers[i][0],
                     shoulder_centers[i + 1][1] - shoulder_centers[i][1])
            for i in range(len(shoulder_centers) - 1)
        ]
        md = float(np.mean(d))
        if md > 0.025:
            cam_motion = "high"
        elif md > 0.012:
            cam_motion = "moderate"

    body_frac = float(np.median(body_heights)) if body_heights else 0.0
    steps = gait.total_steps
    cycles = steps // 2

    return {
        **meta,
        "avg_pose_confidence": float(np.mean(conf_all)) if conf_all else 0.0,
        "valid_pose_pct": s.valid_frame_pct,
        "body_height_fraction": body_frac,
        "landmark_visibility": {n: _mean(n) for n in LANDMARKS},
        "camera_motion": cam_motion,
        "subject_translation": trans,
        "subject_size_note": "small" if body_frac < 0.18 else ("medium" if body_frac < 0.28 else "large"),
        "viewpoint": "front-facing" if body_frac > 0.15 else "distant/other",
        "estimated_cycles": cycles,
        "step_detection_confidence": gait.confidence,
        "steps_detected": steps,
    }


def _component_detail(data: dict) -> dict:
    r = data["result"]
    s = data["series"]
    gait = data["gait"]
    mm = {m.key: m for m in r.metrics}

    scaled_l = [v / max(s.body_height, 1e-6) if v is not None else None for v in s.ankle_l_y]
    scaled_r = [v / max(s.body_height, 1e-6) if v is not None else None for v in s.ankle_r_y]
    clr_l, clr_r = _robust_range(scaled_l), _robust_range(scaled_r)
    clr_asym = abs(clr_l - clr_r) / max((clr_l + clr_r) / 2.0, 1e-6) if clr_l and clr_r else None

    body = mm["body_stability"]
    step = mm["step_consistency"]
    traj = mm["trajectory_smoothness"]

    pelvis_r = body.values.get("pelvis_lateral_sway_ratio")
    torso_r = body.values.get("torso_lateral_offset_ratio")
    pelvis_score = _linear_score(float(pelvis_r), 0.0, BODY_SWAY_ZERO_RATIO) if pelvis_r is not None else None
    torso_score = _linear_score(float(torso_r), 0.0, BODY_SWAY_ZERO_RATIO) if torso_r is not None else None

    knee_jerks = [_mean_jerk(s.knee_l), _mean_jerk(s.knee_r)]
    ankle_jerks = [_mean_jerk(s.ankle_l), _mean_jerk(s.ankle_r)]
    knee_jerk = float(np.mean([j for j in knee_jerks if j is not None])) if any(knee_jerks) else None
    ankle_jerk = float(np.mean([j for j in ankle_jerks if j is not None])) if any(ankle_jerks) else None

    duration_s = s.n / s.fps
    expected_steps = max(2.0, duration_s * 1.75)
    ratio = gait.total_steps / expected_steps if expected_steps > 0 else 0.0
    coverage = min(ratio, 1.0 / ratio) if ratio > 0 else 0.0

    return {
        "final_score": r.score,
        "classification": r.classification,
        "symmetry_score": mm["symmetry"].score,
        "step_regularity_score": step.score,
        "rom_score": mm["range_of_motion"].score,
        "body_score": body.score,
        "trajectory_score": traj.score,
        "pose_quality_score": mm["pose_quality"].score if mm.get("pose_quality") else None,
        "pelvis_stability_score": pelvis_score,
        "torso_stability_score": torso_score,
        "foot_clearance_balance": step.values.get("foot_clearance_balance"),
        "foot_clearance_asymmetry": round(clr_asym, 3) if clr_asym is not None else None,
        "knee_jerk": knee_jerk,
        "ankle_jerk": ankle_jerk,
        "mean_jerk": float(np.mean([j for j in ankle_jerks + knee_jerks if j is not None])) if any(ankle_jerks + knee_jerks) else None,
        "step_coverage_penalty": round(coverage, 3),
        "merged_stride_cv": step.values.get("merged_stride_cv"),
        "left_interval_cv": step.values.get("left_interval_cv"),
        "right_interval_cv": step.values.get("right_interval_cv"),
        "step_detection_confidence": step.values.get("step_detection_confidence"),
        "left_steps": gait.left.step_count,
        "right_steps": gait.right.step_count,
        "cadence_hz": gait.cadence_hz,
        "mean_joint_rom": body.values.get("mean_joint_rom_deg"),
        "jerk_tolerance_scale": traj.values.get("jerk_tolerance_scale"),
        "metric_values": {m.key: m.values for m in r.metrics},
    }


def _weighted_contribution(comp: dict) -> dict[str, float]:
    """Approximate contribution of each metric group to final score."""
    out = {}
    for key, weight in METRIC_WEIGHTS.items():
        score_key = {
            "symmetry": "symmetry_score",
            "step_consistency": "step_regularity_score",
            "body_stability": "body_score",
            "range_of_motion": "rom_score",
            "trajectory_smoothness": "trajectory_score",
            "pose_quality": "pose_quality_score",
        }[key]
        val = comp.get(score_key)
        if val is not None:
            out[key] = val * weight
    return out


def _print_report(normal: dict, athletic: dict, *, label: str) -> None:
    nq, aq = normal["quality"], athletic["quality"]
    nc, ac = normal["components"], athletic["components"]

    print(f"\n{'=' * 72}")
    print(f"ANALYSIS MODE: {label}")
    print(f"{'=' * 72}")

    for title, q in [("Normal Gait", nq), ("Athletic Walking", aq)]:
        print(f"\nVIDEO QUALITY - {title}")
        print(f"  Video: {normal['path'].name if title.startswith('Normal') else athletic['path'].name}")
        print(f"  Duration: {q['duration_s']:.2f}s (analyzed {q['analyzed_frames']} frames)")
        print(f"  FPS: {q['fps']:.1f}")
        print(f"  Resolution: {q['resolution']}")
        print(f"  Average pose detection confidence: {q['avg_pose_confidence']:.3f}")
        print(f"  Valid pose frame percentage: {q['valid_pose_pct']:.1%}")
        print(f"  Body bounding-box height / frame: {q['body_height_fraction']:.1%} ({q['subject_size_note']})")
        print("  Landmark visibility:")
        for lm in LANDMARKS:
            v = q["landmark_visibility"].get(lm)
            print(f"    {lm}: {v:.3f}" if v is not None else f"    {lm}: N/A")
        print(f"  Camera motion: {q['camera_motion']}")
        print(f"  Subject translation through frame: {q['subject_translation']:.4f}")
        print(f"  Primary gait viewpoint: {q['viewpoint']}")
        print(f"  Estimated usable gait cycles: {q['estimated_cycles']}")
        print(f"  Step detection confidence: {q['step_detection_confidence']}")
        print(f"  Steps detected: {q['steps_detected']}")

    print("\nSTABILITY COMPONENT COMPARISON")
    print(f"{'Metric':<32} {'Normal':>12} {'Athletic':>12}")
    print("-" * 58)
    rows = [
        ("Final Stability Score", f"{nc['final_score']:.1f}", f"{ac['final_score']:.1f}"),
        ("Symmetry (norm)", f"{nc['symmetry_score']:.1f}", f"{ac['symmetry_score']:.1f}"),
        ("Step Regularity (norm)", f"{nc['step_regularity_score']:.1f}", f"{ac['step_regularity_score']:.1f}"),
        ("ROM (norm)", f"{nc['rom_score']:.1f}", f"{ac['rom_score']:.1f}"),
        ("Body metric (norm)", f"{nc['body_score']:.1f}", f"{ac['body_score']:.1f}"),
        ("Pelvis stability (norm)", f"{nc['pelvis_stability_score']:.1f}" if nc['pelvis_stability_score'] else "N/A", f"{ac['pelvis_stability_score']:.1f}" if ac['pelvis_stability_score'] else "N/A"),
        ("Torso stability (norm)", f"{nc['torso_stability_score']:.1f}" if nc['torso_stability_score'] else "N/A", f"{ac['torso_stability_score']:.1f}" if ac['torso_stability_score'] else "N/A"),
        ("Foot-clearance balance", str(nc['foot_clearance_balance']), str(ac['foot_clearance_balance'])),
        ("Foot-clearance asymmetry", str(nc['foot_clearance_asymmetry']), str(ac['foot_clearance_asymmetry'])),
        ("Knee jerk (raw)", f"{nc['knee_jerk']:.3f}" if nc['knee_jerk'] else "N/A", f"{ac['knee_jerk']:.3f}" if ac['knee_jerk'] else "N/A"),
        ("Ankle jerk (raw)", f"{nc['ankle_jerk']:.3f}" if nc['ankle_jerk'] else "N/A", f"{ac['ankle_jerk']:.3f}" if ac['ankle_jerk'] else "N/A"),
        ("Step coverage factor", str(nc['step_coverage_penalty']), str(ac['step_coverage_penalty'])),
        ("Merged stride CV", str(nc['merged_stride_cv']), str(ac['merged_stride_cv'])),
        ("Left interval CV", str(nc['left_interval_cv']), str(ac['right_interval_cv'])),
        ("Pose quality (norm)", f"{nc['pose_quality_score']:.1f}" if nc['pose_quality_score'] else "N/A", f"{ac['pose_quality_score']:.1f}" if ac['pose_quality_score'] else "N/A"),
    ]
    for row in rows:
        print(f"{row[0]:<32} {row[1]:>12} {row[2]:>12}")

    n_contrib = _weighted_contribution(nc)
    a_contrib = _weighted_contribution(ac)
    print("\nWeighted score contributions (score × weight):")
    for key in METRIC_WEIGHTS:
        nk = n_contrib.get(key, 0)
        ak = a_contrib.get(key, 0)
        print(f"  {key:<22} Normal {nk:6.2f}  Athletic {ak:6.2f}  diff {ak - nk:+.2f}")

    # Largest athletic penalty vs normal
    deltas = {k: ac.get(f"{k}_score" if k != 'step_consistency' else 'step_regularity_score', 0) or 0
              for k in ['symmetry', 'step', 'rom', 'body', 'trajectory', 'pose']}
    # simpler: compare normalized component scores
    comp_keys = [
        ("symmetry_score", "Symmetry"),
        ("step_regularity_score", "Step Regularity"),
        ("rom_score", "ROM"),
        ("body_score", "Body"),
        ("trajectory_score", "Trajectory"),
        ("pose_quality_score", "Pose Quality"),
    ]
    worst = min(comp_keys, key=lambda kv: (ac[kv[0]] or 0) - (nc[kv[0]] or 0))
    print(f"\nLargest Athletic penalty vs Normal: {worst[1]} (diff {(ac[worst[0]] or 0) - (nc[worst[0]] or 0):+.1f})")


def main() -> int:
    config.ensure_output_dirs()
    gui_max = getattr(config, "GUI_MAX_FRAMES_PER_LOAD", None)

    print("ATHLETIC VS NORMAL INVESTIGATION")
    print("=" * 72)

    # Full video analysis
    normal_full = _process("normal")
    athletic_full = _process("athletic")
    normal_full["quality"] = _quality_report(normal_full)
    athletic_full["quality"] = _quality_report(athletic_full)
    normal_full["components"] = _component_detail(normal_full)
    athletic_full["components"] = _component_detail(athletic_full)
    _print_report(normal_full, athletic_full, label="FULL VIDEO")

    # GUI-limited analysis (matches dashboard if capped)
    if gui_max:
        normal_gui = _process("normal", max_frames=gui_max)
        athletic_gui = _process("athletic", max_frames=gui_max)
        normal_gui["quality"] = _quality_report(normal_gui)
        athletic_gui["quality"] = _quality_report(athletic_gui)
        normal_gui["components"] = _component_detail(normal_gui)
        athletic_gui["components"] = _component_detail(athletic_gui)
        _print_report(normal_gui, athletic_gui, label=f"GUI CAP ({gui_max} frames)")

    nf, af = normal_full["components"], athletic_full["components"]
    nq, aq = normal_full["quality"], athletic_full["quality"]

    print("\n" + "=" * 72)
    print("DECISION ANALYSIS")
    print("=" * 72)
    print(f"Full-video scores: Normal {nf['final_score']:.1f}, Athletic {af['final_score']:.1f}")
    if gui_max:
        ng = normal_gui["components"]
        ag = athletic_gui["components"]
        print(f"GUI-cap scores:    Normal {ng['final_score']:.1f}, Athletic {ag['final_score']:.1f}")

    athletic_suitable = (
        aq["body_height_fraction"] >= 0.18
        and aq["landmark_visibility"].get("left_heel", 0) >= 0.7
        and aq["step_detection_confidence"] == "high"
        and aq["valid_pose_pct"] >= 0.9
    )
    print(f"\nAthletic video technically suitable (strict): {athletic_suitable}")
    print(f"  body height fraction: {aq['body_height_fraction']:.1%}")
    print(f"  heel visibility L/R: {aq['landmark_visibility'].get('left_heel')} / {aq['landmark_visibility'].get('right_heel')}")
    print(f"  step detection: {aq['step_detection_confidence']}, steps={aq['steps_detected']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
