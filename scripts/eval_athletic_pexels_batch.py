"""Batch-evaluate Pexels athletic walking candidates vs current demo."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk.ui.media.demo_download import _download_pexels_video
from stablewalk.ui.media.demo_prepare import find_best_walking_segment, trim_to_gait_segment
from scripts.validate_athletic_video import validate_video, candidate_better
from stablewalk.ui.media.demo_gait import demo_path, example_by_key

# Verified Pexels pages — athletic/sportswear WALKING (not running)
CANDIDATES = [
    (27861377, "Treadmill gym walk — Nothing Ahead"),
    (27727783, "Tennis court walk"),
    (13740494, "Sportswear backup"),
    (36581832, "Profile view leather jacket"),
    (5823532, "Current athletic (woman path)"),
    (8533913, "Running — reject check"),
    (4568380, "Man walking on track"),
    (3195244, "Woman walking on treadmill"),
    (7681049, "Athletic man walking outdoors"),
    (5319095, "Man walking full body"),
    (6774633, "Fitness walking side view"),
    (3254018, "Sportswear woman walking"),
]


def main() -> int:
    current_path = demo_path(example_by_key("athletic"))
    current = validate_video(current_path, max_frames=120, label="Current")

    print(f"{'ID':<10} {'Det%':>6} {'Body%':>6} {'Heel':>6} {'Ankle':>6} {'Steps':>5} {'Conf':>5} {'Score':>6}  Title")
    print("-" * 95)

    best: tuple[float, int, str, Path] | None = None

    for vid, title in CANDIDATES:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.mp4"
            if not _download_pexels_video(vid, raw):
                print(f"{vid:<10} {'FAIL':>6} download failed  {title}")
                continue
            trimmed = Path(tmp) / "trim.mp4"
            start = find_best_walking_segment(raw, output_frames=120)
            ok, _ = trim_to_gait_segment(raw, trimmed, start_frame=start, output_frames=120)
            path = trimmed if ok and trimmed.is_file() else raw
            try:
                r = validate_video(path, max_frames=120, label=title)
            except Exception as exc:
                print(f"{vid:<10} {'ERR':>6} {exc}  {title}")
                continue

            det = r["detection_pct"] * 100
            print(
                f"{vid:<10} {det:5.0f}% {r['body_height_fraction']*100:5.1f}% "
                f"{r['heel_visibility']:5.3f} {r['ankle_visibility']:5.3f} "
                f"{r['steps_detected']:5d} {r['step_confidence'][:4]:>5} "
                f"{r['stability_score']:6.1f}  {title}"
            )

            # Gait-analysis quality score (not stability)
            gait_q = (
                r["detection_pct"] * 30
                + r["heel_visibility"] * 25
                + r["ankle_visibility"] * 20
                + min(r["body_height_fraction"], 0.35) / 0.35 * 15
                + min(r["usable_gait_cycles"], 4) / 4 * 10
                + (5 if r["step_confidence"] == "high" else 0)
            )
            if best is None or gait_q > best[0]:
                best = (gait_q, vid, title, path)

    if best:
        print(f"\nBest candidate by gait quality: {best[1]} — {best[2]} (quality={best[0]:.1f})")
        r = validate_video(best[3], max_frames=120, label=best[2])
        better, reasons = candidate_better(current, r)
        print(f"Technically better than current: {better}")
        for reason in reasons:
            print(f"  - {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
