#!/usr/bin/env python3
"""
Inspect Health&Gait dataset_samples.zip before downloading the full archive.

Usage:
  python scripts/inspect_healthgait_samples.py
  python scripts/inspect_healthgait_samples.py --output data/demo_videos/healthgait_inspection.json
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
from stablewalk.demo.healthgait_selection import (
    inspect_healthgait_samples,
    write_healthgait_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Health&Gait sample archive")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DEMO_VIDEOS_DIR / "healthgait_samples_inspection.json",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use cached dataset_samples.zip only",
    )
    args = parser.parse_args()

    report = inspect_healthgait_samples(download=not args.no_download)
    write_healthgait_report(args.output, report)

    print("Health&Gait dataset_samples inspection")
    print(f"  Samples zip: {report.get('samples_zip')}")
    print(f"  Raw MP4 in samples: {report.get('raw_video_in_samples')}")
    print(f"  Limitation: {report.get('public_release_limitation')}")
    print(f"Wrote {args.output.resolve()}")

    if report.get("normal_candidates_ugs"):
        top = report["normal_candidates_ugs"][0]
        print(f"  Top Normal (UGS) sample: {top['participant_id']} v={top['velocity_m_s']} m/s")
    if report.get("performance_candidates_fgs"):
        top = report["performance_candidates_fgs"][0]
        print(
            f"  Top Performance (FGS) sample: {top['participant_id']} "
            f"v={top['velocity_m_s']} m/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
