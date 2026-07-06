"""Validate all StableWalk demo videos and print a comparison report."""

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
)
from stablewalk.ui.media.utah_abnormal import UTAH_METADATA_PATH
from stablewalk.ui.media.demo_validation import validate_demo_video


def _opensim_note(path: Path) -> str:
    if not path.is_file():
        return "not run — video missing"
    try:
        from stablewalk.core.pipeline import run_gait_pipeline

        result = run_gait_pipeline(str(path), validate="none", max_frames=40)
        ik = result.ik_path if hasattr(result, "ik_path") else None
        if ik and Path(ik).is_file():
            return "IK files exported"
        return "pipeline completed (check OpenSim export in GUI)"
    except Exception as exc:
        return f"pipeline error: {exc}"


def main() -> int:
    config.ensure_output_dirs()
    failures = 0
    print("StableWalk Demo Video Validation Report\n" + "=" * 48 + "\n")

    for ex in DEMO_GAIT_EXAMPLES:
        path = demo_path(ex)
        print(ex.display_name)
        print("-" * len(ex.display_name))
        print(f"Source: {ex.source_name}")
        print(f"URL: {ex.source_url}")
        if ex.key == "abnormal" and UTAH_METADATA_PATH.is_file():
            meta = json.loads(UTAH_METADATA_PATH.read_text(encoding="utf-8"))
            print(f"Exact selected video: {meta.get('video_title', '?')} ({meta.get('kaltura_entry_id', '?')})")

        if not demo_exists(ex):
            print("Status: MISSING — manual download required")
            print(f"Place file at: {path}\n")
            failures += 1
            continue

        report = validate_demo_video(path, max_frames=80)
        print(report.format_report())
        print(f"OpenSim IK result: {_opensim_note(path)}\n")

        if report.gait_detected_rate < 0.25:
            failures += 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
