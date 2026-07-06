"""Dump raw stability metric values for Normal vs Athletic (full vs GUI cap)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.analysis.biomech_stability import analyze_biomech_stability
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path

SKIP = {"peak_frames_left", "peak_frames_right"}


def dump(key: str, max_frames: int | None) -> None:
    ex = next(e for e in DEMO_GAIT_EXAMPLES if e.key == key)
    path = demo_path(ex)
    mf_label = str(max_frames) if max_frames else "full"
    with PoseEstimator(video_mode=True) as est:
        seq = est.process_video(path, enrich_gait=False, max_frames=max_frames)
    enrich_pose_sequence(seq)
    r = analyze_biomech_stability(seq)
    print("=" * 60)
    print(f"{key} frames={mf_label} score={r.score:.1f} n={r.frame_count}")
    for m in r.metrics:
        print(f"  [{m.key}] score={m.score:.1f} weight={m.weight:.0%}")
        for k, v in sorted(m.values.items()):
            if k not in SKIP:
                print(f"    {k}: {v}")


def main() -> None:
    caps = [None, config.GUI_MAX_FRAMES_PER_LOAD, config.DEMO_MAX_FRAMES]
    for mf in caps:
        for key in ("normal", "athletic"):
            dump(key, mf)


if __name__ == "__main__":
    main()
