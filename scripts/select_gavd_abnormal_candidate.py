#!/usr/bin/env python3
"""
Filter GAVD clinical annotations for abnormal gait demo candidates.

Downloads annotation CSV parts from https://github.com/Rahmyyy/GAVD (metadata only).
Videos must be retrieved separately from public YouTube URLs.

Usage:
  python scripts/select_gavd_abnormal_candidate.py
  python scripts/select_gavd_abnormal_candidate.py --top 30 --output report.json
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
from stablewalk.demo.gavd_selection import (
    select_gavd_abnormal_candidates,
    write_gavd_selection_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="GAVD abnormal demo candidate filter")
    parser.add_argument("--top", type=int, default=20, help="Top candidates to list")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DEMO_VIDEOS_DIR / "gavd_abnormal_candidates.json",
        help="JSON report output path",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use cached annotation CSVs only",
    )
    args = parser.parse_args()

    report = select_gavd_abnormal_candidates(
        top_n=args.top,
        download=not args.no_download,
    )
    write_gavd_selection_report(args.output, report)

    print(f"GAVD abnormal candidates: {report['eligible_count']} eligible")
    print(f"Wrote {args.output.resolve()}")
    for item in report.get("top_candidates", [])[:5]:
        print(
            f"  {item['gait_pat']:20} view={item['cam_view']:12} "
            f"cycles~{item['estimated_cycles']} score={item['metadata_score']:.2f} "
            f"{item['url']}"
        )
    print("\nNext: download top URL, trim to seq frames, then:")
    print("  python scripts/validate_demo_candidate.py --video <path> --category abnormal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
