"""Quick eval of Pexels walking candidates."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from stablewalk.analysis.biomech_stability import analyze_biomech_stability
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.ui.media.demo_download import _download_pexels_video
from stablewalk.ui.media.demo_prepare import find_best_walking_segment, trim_to_gait_segment

CANDIDATES = [
    (5319095, "steady pace best"),
    (6830920, "frozen lake"),
    (4540121, "catalog 4540121"),
    (7026843, "winter sportswear"),
    (36581832, "profile leather jacket"),
    (6195655, "catalog 6195655"),
    (4942831, "catalog 4942831"),
    (2803277, "catalog 2803277"),
    (5823532, "front view path"),
]


def eval_vid(vid: int, title: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        if not _download_pexels_video(vid, raw):
            print(f"{vid:8} FAIL download | {title}")
            return
        trim = Path(tmp) / "trim.mp4"
        start = find_best_walking_segment(raw, output_frames=120)
        ok, _ = trim_to_gait_segment(raw, trim, start_frame=start, output_frames=120)
        path = trim if ok else raw
        with PoseEstimator(video_mode=True) as est:
            seq = est.process_video(path, enrich_gait=False)
        enrich_pose_sequence(seq)
        b = analyze_biomech_stability(seq)
        det = sum(1 for f in seq.frames if f.detected)
        ys, vis = [], []
        for f in seq.frames:
            if f.detected and f.keypoints:
                kpys = [kp.y for kp in f.keypoints if kp.visibility >= 0.3]
                if kpys:
                    ys.append(max(kpys) - min(kpys))
                for kp in f.keypoints:
                    if kp.name in ("left_ankle", "right_ankle", "left_heel", "right_heel"):
                        vis.append(kp.visibility)
        bbox = float(np.mean(ys) * 100) if ys else 0.0
        mvis = float(np.mean(vis)) if vis else 0.0
        steps = b.metric("step_consistency")
        sv = steps.values if steps else {}
        ls, rs = sv.get("left_steps"), sv.get("right_steps")
        print(
            f"{vid:8} score={b.score:5.1f} {b.classification:10} "
            f"det={det:3}/{len(seq.frames):3} bbox={bbox:5.1f}% "
            f"foot_vis={mvis:.2f} steps={ls}+{rs} | {title}"
        )


if __name__ == "__main__":
    for vid, title in CANDIDATES:
        eval_vid(vid, title)
