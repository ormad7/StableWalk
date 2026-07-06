"""
Select the best ProGait side-view walking clip for the abnormal gait demo.

Requires Hugging Face access to ericyxy98/ProGait:
  1. Accept dataset terms: https://huggingface.co/datasets/ericyxy98/ProGait
  2. huggingface-cli login
  3. python scripts/select_progait_demo.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.ui.media.demo_gait import PROGAIT_METADATA_PATH, demo_path, example_by_key
from stablewalk.ui.media.demo_validation import report_to_dict, validate_demo_video

SIDE_VIEW_CANDIDATES: tuple[str, ...] = (
    "videos/inside/1_3_1_s.mp4",
    "videos/inside/1_4_1_s.mp4",
    "videos/inside/1_5_1_s.mp4",
    "videos/inside/2_5_1_s.mp4",
    "videos/inside/4_1_1_s.mp4",
    "videos/inside/4_1_2_s.mp4",
    "videos/outside/1_1_1_s.mp4",
    "videos/outside/1_2_1_s.mp4",
    "videos/outside/1_3_1_s.mp4",
    "videos/outside/2_1_1_s.mp4",
    "videos/outside/2_2_1_s.mp4",
    "videos/outside/3_1_1_s.mp4",
)


def _score_report(report) -> float:
    return (
        report.gait_detected_rate * 3.0
        + report.hip_visibility_rate
        + report.knee_visibility_rate
        + report.ankle_visibility_rate
        + report.foot_visibility_rate
    )


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        return 1

    config.ensure_output_dirs()
    dest = demo_path(example_by_key("abnormal"))
    cache_dir = config.DEMO_VIDEOS_DIR / "progait_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        available = set(list_repo_files("ericyxy98/ProGait", repo_type="dataset"))
    except Exception as exc:
        print(f"Cannot list ProGait dataset: {exc}")
        print("Accept terms and run: huggingface-cli login")
        return 1

    candidates = [c for c in SIDE_VIEW_CANDIDATES if c in available]
    if not candidates:
        side = sorted(p for p in available if p.endswith("_s.mp4") and p.startswith("videos/"))
        candidates = side[:12]

    if not candidates:
        print("No side-view ProGait clips found.")
        return 1

    best_path = None
    best_rel = None
    best_report = None
    best_score = -1.0

    print(f"Testing {len(candidates)} ProGait side-view candidates...\n")
    for rel in candidates:
        try:
            local = hf_hub_download(
                "ericyxy98/ProGait",
                rel,
                repo_type="dataset",
                local_dir=str(cache_dir),
            )
        except Exception as exc:
            print(f"[skip] {rel}: {exc}")
            continue
        report = validate_demo_video(local, max_frames=80)
        score = _score_report(report)
        print(
            f"[{report.compact_status}] {rel} "
            f"gait={report.gait_detected_rate:.0%} foot={report.foot_visibility_rate:.0%}"
        )
        if score > best_score:
            best_score = score
            best_path = Path(local)
            best_rel = rel
            best_report = report

    if best_path is None or best_report is None or best_rel is None:
        print("\nNo suitable ProGait clip could be downloaded.")
        print("Manual steps:")
        print("  1. Accept terms at https://huggingface.co/datasets/ericyxy98/ProGait")
        print("  2. huggingface-cli login")
        print("  3. Re-run this script")
        print(f"  4. Or copy a validated clip to: {dest}")
        return 1

    shutil.copy2(best_path, dest)
    meta = {
        "dataset": "ericyxy98/ProGait",
        "selected_clip": best_rel,
        "selection_score": best_score,
        "local_source": str(best_path),
        "validation": report_to_dict(best_report),
        "project_url": "https://pittisl.github.io/publication/2025-progait/",
    }
    PROGAIT_METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nSelected ProGait clip:")
    print(f"  {best_rel}")
    print(f"Saved demo -> {dest}")
    print(f"Metadata -> {PROGAIT_METADATA_PATH}")
    print("\n" + best_report.format_report())
    return 0 if best_report.gait_detected_rate >= 0.25 else 1


if __name__ == "__main__":
    raise SystemExit(main())
