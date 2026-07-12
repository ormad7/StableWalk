#!/usr/bin/env python3
"""Generate final StableWalk demo validation report (Abnormal / Normal / Performance)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.demo.final_demo_validation import write_final_demo_validation_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="StableWalk final demo validation (Abnormal / Normal / Performance)"
    )
    parser.add_argument(
        "--use-cached-poses",
        action="store_true",
        help="Reuse cached pose JSON instead of re-estimating from MP4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path (default: data/output/reports/final_demo_validation.md)",
    )
    args = parser.parse_args()

    config.ensure_output_dirs()
    out = args.output or (config.REPORTS_DIR / "final_demo_validation.md")
    text = write_final_demo_validation_report(out, use_cached_poses=args.use_cached_poses)
    print(text)
    print(f"\nWrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
