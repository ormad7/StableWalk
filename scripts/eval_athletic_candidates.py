"""Evaluate Pexels athletic walking candidates for demo suitability."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk.analysis.biomech_stability import analyze_biomech_stability
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.ui.media.demo_download import _download_pexels_video
from stablewalk.ui.media.demo_prepare import find_best_walking_segment, trim_to_gait_segment

CANDIDATES = [
    (27861377, "Treadmill gym walk — Nothing Ahead"),
    (5823532, "Woman walking in coat"),
    (27727783, "Tennis court walk"),
    (8533913, "Running (old athletic demo)"),
    (13740494, "Sportswear backup"),
    (36581832, "Profile view leather jacket"),
    (5320110, "Normal gait source (reference)"),
]


def eval_video(path: Path) -> dict:
    with PoseEstimator(video_mode=True) as est:
        seq = est.process_video(path, enrich_gait=False)
    enrich_pose_sequence(seq)
    biomech = analyze_biomech_stability(seq)
    detected = sum(1 for f in seq.frames if f.detected)
    ys = []
    for f in seq.frames:
        if f.detected and f.keypoints:
            kpys = [kp.y for kp in f.keypoints if kp.visibility >= 0.3]
            if kpys:
                ys.append(max(kpys) - min(kpys))
    import numpy as np
    bbox = float(np.mean(ys)) if ys else 0.0
    return {
        "score": biomech.score,
        "class": biomech.classification,
        "detected": detected,
        "total": len(seq.frames),
        "bbox_pct": bbox * 100,
        "metrics": {m.key: m.score for m in biomech.metrics if m.score is not None},
    }


def main() -> int:
    print(f"{'ID':<10} {'Score':>6} {'Class':<10} {'Det%':>6} {'BBox%':>6}  Title")
    print("-" * 80)
    for vid, title in CANDIDATES:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.mp4"
            if not _download_pexels_video(vid, raw):
                print(f"{vid:<10} {'FAIL':>6} download failed  {title}")
                continue
            trimmed = Path(tmp) / "trim.mp4"
            start = find_best_walking_segment(raw, output_frames=120)
            ok, used = trim_to_gait_segment(raw, trimmed, start_frame=start, output_frames=120)
            path = trimmed if ok else raw
            try:
                r = eval_video(path)
            except Exception as exc:
                print(f"{vid:<10} {'ERR':>6} {exc}  {title}")
                continue
            det_pct = 100 * r["detected"] / max(r["total"], 1)
            print(
                f"{vid:<10} {r['score']:6.1f} {r['class']:<10} {det_pct:5.0f}% "
                f"{r['bbox_pct']:5.1f}%  {title}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
