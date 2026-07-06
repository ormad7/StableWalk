"""Verify demo gait configuration and optional local video files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.ui.media.demo_gait import (
    DEMO_GAIT_EXAMPLES,
    demo_exists,
    demo_path,
    demo_validation_status,
    demo_videos_dir,
    missing_file_message,
)
from stablewalk.ui.media.utah_abnormal import UTAH_METADATA_PATH
from stablewalk.ui.media.demo_validation import validate_demo_video


def main() -> int:
    config.ensure_output_dirs()
    folder = demo_videos_dir()
    print(f"Demo folder: {folder}\n")

    failures = 0
    for ex in DEMO_GAIT_EXAMPLES:
        path = demo_path(ex)
        present = demo_exists(ex)
        status = demo_validation_status(ex) if present else "Demo Video: not downloaded"
        print(f"[{status}] {ex.display_name}")
        print(f"        file: {path.name}")
        print(f"        source: {ex.source_url}")
        if ex.key == "abnormal" and UTAH_METADATA_PATH.is_file():
            meta = json.loads(UTAH_METADATA_PATH.read_text(encoding="utf-8"))
            print(f"        utah clip: {meta.get('video_title', '?')}")
        if present:
            report = validate_demo_video(path, max_frames=80)
            print(
                f"        gait={report.gait_detected_rate:.0%} "
                f"foot={report.foot_visibility_rate:.0%}"
            )
        if "not downloaded" in status or "required" in status.lower():
            failures += 1
            print(f"        help: {missing_file_message(ex).splitlines()[0]}")
        elif present and report.gait_detected_rate < 0.25:
            failures += 1
        print()

    if failures:
        print(f"{failures} demo file(s) need attention — see DEMO_VIDEOS.md")
        return 1
    print("All demo files present and validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
