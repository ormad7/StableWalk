#!/usr/bin/env python3
"""
Research-oriented StableWalk demo video selection workflow.

Runs:
  1. GAVD abnormal candidate filtering (metadata)
  2. Health&Gait sample inspection (metadata)
  3. Optional validation of installed demo MP4 files

Usage:
  python scripts/demo_video_selection_workflow.py
  python scripts/demo_video_selection_workflow.py --validate-installed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.demo.candidate_validation import validate_demo_candidate
from stablewalk.demo.gavd_selection import select_gavd_abnormal_candidates
from stablewalk.demo.healthgait_selection import inspect_healthgait_samples
from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path


def main() -> int:
    parser = argparse.ArgumentParser(description="StableWalk demo selection workflow")
    parser.add_argument(
        "--validate-installed",
        action="store_true",
        help="Also validate currently installed demo MP4 files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DEMO_VIDEOS_DIR / "demo_selection_workflow_report.json",
    )
    args = parser.parse_args()

    print("=== StableWalk demo video selection workflow ===\n")

    print("1) GAVD abnormal metadata filter...")
    gavd = select_gavd_abnormal_candidates(top_n=15)

    print("2) Health&Gait samples inspection...")
    health = inspect_healthgait_samples()

    installed: dict[str, object] = {}
    if args.validate_installed:
        print("3) Validating installed demo files...")
        for ex in DEMO_GAIT_EXAMPLES:
            path = demo_path(ex)
            category = "performance" if ex.key == "athletic" else ex.key
            if path.is_file():
                report = validate_demo_candidate(path, category=category)
                installed[ex.key] = report.to_dict()
            else:
                installed[ex.key] = {"verdict": "REJECT", "issues": ["file missing"]}

    report = {
        "protocol_version": "1.0",
        "categories": {
            "abnormal": {"source": "GAVD", "ui_label": "Abnormal"},
            "normal": {"source": "Health&Gait UGS", "ui_label": "Normal"},
            "performance": {
                "source": "Health&Gait FGS",
                "ui_label": "Performance",
                "legacy_key": "athletic",
            },
        },
        "gavd_abnormal_selection": gavd,
        "healthgait_inspection": health,
        "installed_demo_validation": installed,
        "documentation": str(
            (config.DEMO_VIDEOS_DIR / "DEMO_VIDEO_SOURCES.md").resolve()
        ),
        "validator": "python scripts/validate_demo_candidate.py --video <path>",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output.resolve()}")
    print(f"Document sources in: {config.DEMO_VIDEOS_DIR / 'DEMO_VIDEO_SOURCES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
