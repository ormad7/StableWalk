"""Validate and compare athletic demo video candidates for gait analysis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from stablewalk.analysis.biomech_stability import analyze_biomech_stability
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.pose.gait_step_detection import detect_gait_steps
from stablewalk.ui.media.demo_gait import demo_path, example_by_key

LANDMARKS = (
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)


def _video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {
        "duration_s": total / fps if fps > 0 else 0.0,
        "fps": fps,
        "resolution": f"{fw}x{fh}",
        "total_frames": total,
    }


def validate_video(path: Path, *, max_frames: int | None = None, label: str = "") -> dict:
    meta = _video_meta(path)
    with PoseEstimator(video_mode=True) as est:
        seq = est.process_video(path, enrich_gait=False, max_frames=max_frames)
    enrich_pose_sequence(seq)
    stab = analyze_biomech_stability(seq)
    gait = detect_gait_steps(seq)

    vis: dict[str, list[float]] = {n: [] for n in LANDMARKS}
    body_heights: list[float] = []
    shoulder_centers: list[tuple[float, float]] = []
    detected = 0

    for f in seq.frames:
        if not f.keypoints:
            continue
        if f.detected:
            detected += 1
        kp = {k.name: k for k in f.keypoints}
        for name in LANDMARKS:
            k = kp.get(name)
            if k:
                vis[name].append(k.visibility)
        ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
        lh, rh = kp.get("left_hip"), kp.get("right_hip")
        if ls and rs and lh and rh:
            top = min(ls.y, rs.y, lh.y, rh.y)
            bot = max(ls.y, rs.y, lh.y, rh.y)
            body_heights.append(bot - top)
            shoulder_centers.append(((ls.x + rs.x) / 2, (ls.y + rs.y) / 2))

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

    def _avg(name: str) -> float:
        arr = vis[name]
        return float(np.mean(arr)) if arr else 0.0

    ankle_vis = (_avg("left_ankle") + _avg("right_ankle")) / 2
    heel_vis = (_avg("left_heel") + _avg("right_heel")) / 2
    foot_vis = (_avg("left_foot_index") + _avg("right_foot_index")) / 2
    body_frac = float(np.median(body_heights)) if body_heights else 0.0
    sampled = len(seq.frames)

    mm = {m.key: m for m in stab.metrics}
    return {
        "label": label or path.name,
        "path": str(path),
        **meta,
        "sampled_frames": sampled,
        "pose_detected_frames": sum(1 for f in seq.frames if f.keypoints),
        "gait_detected_frames": detected,
        "detection_pct": detected / max(sampled, 1),
        "landmark_visibility": {n: _avg(n) for n in LANDMARKS},
        "body_height_fraction": body_frac,
        "ankle_visibility": ankle_vis,
        "heel_visibility": heel_vis,
        "foot_index_visibility": foot_vis,
        "usable_gait_cycles": gait.total_steps // 2,
        "steps_detected": gait.total_steps,
        "step_confidence": gait.confidence,
        "camera_motion": cam_motion,
        "stability_score": stab.score,
        "symmetry": mm["symmetry"].score if mm.get("symmetry") else None,
        "step_regularity": mm["step_consistency"].score if mm.get("step_consistency") else None,
        "body": mm["body_stability"].score if mm.get("body_stability") else None,
        "pose_quality": mm["pose_quality"].score if mm.get("pose_quality") else None,
        "classification": stab.classification,
    }


def _print_metrics(r: dict) -> None:
    print(f"\n=== {r['label']} ===")
    print(f"Path: {r['path']}")
    print(f"Duration: {r['duration_s']:.2f}s  FPS: {r['fps']:.1f}  Resolution: {r['resolution']}")
    print(f"Pose detected frames: {r['gait_detected_frames']} / {r['sampled_frames']} ({r['detection_pct']:.1%})")
    print(f"Body height / frame: {r['body_height_fraction']:.1%}")
    for n in LANDMARKS:
        print(f"  {n}: {r['landmark_visibility'][n]:.3f}")
    print(f"Ankle avg: {r['ankle_visibility']:.3f}  Heel avg: {r['heel_visibility']:.3f}  Foot index avg: {r['foot_index_visibility']:.3f}")
    print(f"Usable gait cycles: {r['usable_gait_cycles']}  Steps: {r['steps_detected']}  Confidence: {r['step_confidence']}")
    print(f"Camera motion: {r['camera_motion']}")
    print(f"Stability: {r['stability_score']:.1f}  Symmetry: {r['symmetry']:.1f}  Step: {r['step_regularity']:.1f}  Body: {r['body']:.1f}")


def compare(current: dict, candidate: dict) -> None:
    print("\nATHLETIC VIDEO COMPARISON")
    print(f"{'Metric':<28} {'Current':>12} {'Candidate':>12}")
    print("-" * 54)
    rows = [
        ("Pose detection %", f"{current['detection_pct']:.1%}", f"{candidate['detection_pct']:.1%}"),
        ("Body size in frame", f"{current['body_height_fraction']:.1%}", f"{candidate['body_height_fraction']:.1%}"),
        ("Ankle visibility", f"{current['ankle_visibility']:.3f}", f"{candidate['ankle_visibility']:.3f}"),
        ("Heel visibility", f"{current['heel_visibility']:.3f}", f"{candidate['heel_visibility']:.3f}"),
        ("Foot-index visibility", f"{current['foot_index_visibility']:.3f}", f"{candidate['foot_index_visibility']:.3f}"),
        ("Usable gait cycles", str(current['usable_gait_cycles']), str(candidate['usable_gait_cycles'])),
        ("Step detection confidence", current['step_confidence'], candidate['step_confidence']),
        ("Camera-motion concern", current['camera_motion'], candidate['camera_motion']),
        ("Stability score", f"{current['stability_score']:.1f}", f"{candidate['stability_score']:.1f}"),
    ]
    for row in rows:
        print(f"{row[0]:<28} {row[1]:>12} {row[2]:>12}")


def candidate_better(current: dict, candidate: dict) -> tuple[bool, list[str]]:
    """Score candidate vs current on gait-analysis metrics (not stability score)."""
    reasons: list[str] = []
    score = 0
    checks = [
        ("detection_pct", True, "pose detection rate"),
        ("heel_visibility", True, "heel visibility"),
        ("ankle_visibility", True, "ankle visibility"),
        ("foot_index_visibility", True, "foot-index visibility"),
        ("body_height_fraction", True, "body size in frame"),
        ("usable_gait_cycles", True, "usable gait cycles"),
    ]
    for key, higher_better, name in checks:
        c, n = current[key], candidate[key]
        if higher_better:
            if n > c + 0.02 if isinstance(n, float) and n <= 1 else n > c:
                score += 1
                reasons.append(f"Candidate better: {name} ({n} vs {c})")
            elif c > n + (0.02 if isinstance(n, float) and n <= 1 else 0):
                score -= 1
                reasons.append(f"Current better: {name} ({c} vs {n})")
    if candidate["step_confidence"] == "high" and current["step_confidence"] == "low":
        score += 2
        reasons.append("Candidate has HIGH step detection confidence vs LOW current")
    elif current["step_confidence"] == "high" and candidate["step_confidence"] == "low":
        score -= 2
        reasons.append("Current has HIGH step confidence vs LOW candidate")
    cam_rank = {"low": 2, "moderate": 1, "high": 0}
    if cam_rank.get(candidate["camera_motion"], 0) > cam_rank.get(current["camera_motion"], 0):
        score += 1
        reasons.append(f"Candidate lower camera motion ({candidate['camera_motion']} vs {current['camera_motion']})")
    return score > 0, reasons


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=Path, help="Candidate video path")
    p.add_argument("--max-frames", type=int, default=120)
    args = p.parse_args()

    ex = example_by_key("athletic")
    current_path = demo_path(ex)
    current = validate_video(current_path, max_frames=args.max_frames, label="Current Athletic")
    candidate = validate_video(args.candidate, max_frames=args.max_frames, label="Candidate")
    _print_metrics(current)
    _print_metrics(candidate)
    compare(current, candidate)
    better, reasons = candidate_better(current, candidate)
    print(f"\nCandidate technically better: {better}")
    for r in reasons:
        print(f"  - {r}")
    return 0 if better else 1


if __name__ == "__main__":
    raise SystemExit(main())
