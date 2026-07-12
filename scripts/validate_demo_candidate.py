#!/usr/bin/env python3
"""
Validate a StableWalk demo video candidate.

Usage:
  python scripts/validate_demo_candidate.py --video path/to/candidate.mp4
  python scripts/validate_demo_candidate.py --video candidate.mp4 --category abnormal --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk.demo.candidate_validation import validate_demo_candidate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="StableWalk demo candidate prevalidation"
    )
    parser.add_argument("--video", required=True, help="Path to candidate MP4/video")
    parser.add_argument(
        "--category",
        choices=("abnormal", "normal", "performance", "athletic"),
        default=None,
        help="Optional demo category hint for the report",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=180,
        help="Maximum frames to sample for pose analysis (default: 180)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON report instead of plain text",
    )
    args = parser.parse_args()

    category = args.category
    if category == "athletic":
        category = "performance"

    report = validate_demo_candidate(
        args.video,
        max_frames=args.max_frames,
        category=category,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format_report())

    if report.verdict == "ACCEPT":
        return 0
    if report.verdict == "ACCEPT_WITH_LIMITATIONS":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
