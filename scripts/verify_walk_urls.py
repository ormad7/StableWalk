"""Scan Pexels IDs and refresh data/verified_men_walk.json (pose-checked clips)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.video_catalog import (
    CANDIDATE_MAN_WALK_IDS,
    label_for,
    pexels_url,
    save_verified_catalog,
)
from stablewalk.video_source import quick_validate_source
from stablewalk.video_validation import validate_video_source


def main() -> int:
    verified: list[dict] = []
    for vid in CANDIDATE_MAN_WALK_IDS:
        url = pexels_url(vid)
        ok, _ = quick_validate_source(url)
        if not ok:
            print(f"skip {vid}: cannot open")
            continue
        passed, ratio, msg = validate_video_source(
            url, sample_count=12, min_valid_ratio=0.25
        )
        print(f"{vid}: {'OK' if passed else 'fail'} ({ratio:.0%}) — {msg[:60]}")
        if passed:
            verified.append({"id": vid, "label": label_for(vid)})

    if not verified:
        print("No verified videos found.")
        return 1

    path = save_verified_catalog(verified)
    print(f"Wrote {len(verified)} videos to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
