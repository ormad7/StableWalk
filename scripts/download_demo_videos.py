"""Download StableWalk demo gait videos (Pexels) and prepare ProGait abnormal clip."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.ui.media.demo_download import download_demo_video
from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path
from stablewalk.ui.media.demo_validation import validate_demo_video


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download/prepare StableWalk demo gait videos.")
    parser.add_argument("--force", action="store_true", help="Re-download even when cached.")
    args = parser.parse_args()

    config.ensure_output_dirs()
    print(f"Demo folder: {config.DEMO_VIDEOS_DIR}\n")
    ok = 0
    for ex in DEMO_GAIT_EXAMPLES:
        dest = demo_path(ex)
        print(f"[prepare] {ex.display_name}")
        if ex.key == "abnormal":
            success = download_demo_video(ex.key, force=args.force)
            if not success:
                print("  ProGait requires Hugging Face access.")
                print("  Run: python scripts/select_progait_demo.py")
                print()
                continue
        elif download_demo_video(ex.key, force=args.force):
            success = True
        else:
            print("  download failed — see DEMO_VIDEOS.md for manual steps")
            print()
            continue

        if dest.is_file():
            report = validate_demo_video(dest, max_frames=80)
            print(f"  saved -> {dest.name} ({dest.stat().st_size // 1024} KB)")
            print(f"  {report.compact_status}")
            if report.gait_detected_rate >= 0.25:
                ok += 1
        print()

    print(f"Ready: {ok}/{len(DEMO_GAIT_EXAMPLES)} demo videos validated.")
    if ok < len(DEMO_GAIT_EXAMPLES):
        print("See DEMO_VIDEOS.md for manual download instructions.")
    return 0 if ok == len(DEMO_GAIT_EXAMPLES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
