#!/usr/bin/env python3
"""
Download StableWalk demo video candidates from documented public sources.

These are *candidates* — always run validate_demo_candidate.py before installing
to data/demo_videos/{abnormal_gait,normal_gait,athletic_walking}.mp4.

Usage:
  python scripts/fetch_recommended_demo_videos.py --all
  python scripts/fetch_recommended_demo_videos.py --category normal
  python scripts/fetch_recommended_demo_videos.py --category abnormal --gavd-rank 1
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config

# Direct MP4 links (academic use — cite source datasets in publications).
WEIGHTGAIT_SAMPLES = {
    "normal": {
        "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/8956/1_Normal_Gait.mp4?sequence=15&isAllowed=y",
        "filename": "weightgait_normal_gait.mp4",
        "source": "WeightGait (Edinburgh DataShare)",
        "cite": "Lochhead & Fisher, Computers in Biology and Medicine, 2025",
    },
    "abnormal_limp": {
        "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/8956/4_Limping_Gait.mp4?sequence=18&isAllowed=y",
        "filename": "weightgait_limping_gait.mp4",
        "source": "WeightGait (Edinburgh DataShare)",
        "cite": "Lochhead & Fisher, Computers in Biology and Medicine, 2025",
    },
    "abnormal_shuffle": {
        "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/8956/7_Shuffle_Gait.mp4?sequence=21&isAllowed=y",
        "filename": "weightgait_shuffle_gait.mp4",
        "source": "WeightGait (Edinburgh DataShare)",
        "cite": "Lochhead & Fisher, Computers in Biology and Medicine, 2025",
    },
    "performance_obstacle": {
        "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/8956/3_Normal_Obstacle.mp4?sequence=17&isAllowed=y",
        "filename": "weightgait_normal_obstacle.mp4",
        "source": "WeightGait (Edinburgh DataShare)",
        "cite": "Lochhead & Fisher, Computers in Biology and Medicine, 2025",
        "note": "Interim performance placeholder — prefer Health&Gait FGS when RGB available",
    },
}

GAVD_REPORT = config.DEMO_VIDEOS_DIR / "gavd_abnormal_candidates.json"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "StableWalk/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"  -> {dest.resolve()} ({len(data) / 1024:.1f} KB)")


def _gavd_youtube_url(rank: int) -> str | None:
    if not GAVD_REPORT.is_file():
        print(f"Missing {GAVD_REPORT} — run: python scripts/select_gavd_abnormal_candidate.py")
        return None
    report = json.loads(GAVD_REPORT.read_text(encoding="utf-8"))
    candidates = report.get("top_candidates") or []
    if rank < 1 or rank > len(candidates):
        print(f"GAVD rank {rank} out of range (1..{len(candidates)})")
        return None
    item = candidates[rank - 1]
    print(
        f"GAVD #{rank}: {item.get('gait_pat')} view={item.get('cam_view')} "
        f"cycles~{item.get('estimated_cycles')} url={item.get('url')}"
    )
    return item.get("url")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch StableWalk demo video candidates")
    parser.add_argument(
        "--category",
        choices=("abnormal", "normal", "performance", "all"),
        default="all",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.DEMO_VIDEOS_DIR / "candidates",
        help="Directory for downloaded candidates",
    )
    parser.add_argument(
        "--gavd-rank",
        type=int,
        default=1,
        help="1-based rank from gavd_abnormal_candidates.json (for abnormal)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = (
        ("abnormal", "normal", "performance")
        if args.category == "all"
        else (args.category,)
    )

    for cat in categories:
        if cat == "normal":
            spec = WEIGHTGAIT_SAMPLES["normal"]
            dest = out_dir / spec["filename"]
            _download(spec["url"], dest)
            print(f"Validate: python scripts/validate_demo_candidate.py --video {dest} --category normal")
        elif cat == "performance":
            spec = WEIGHTGAIT_SAMPLES["performance_obstacle"]
            dest = out_dir / spec["filename"]
            _download(spec["url"], dest)
            print(
                f"Validate: python scripts/validate_demo_candidate.py --video {dest} --category performance"
            )
            print(f"Note: {spec.get('note', '')}")
        elif cat == "abnormal":
            for key in ("abnormal_limp", "abnormal_shuffle"):
                spec = WEIGHTGAIT_SAMPLES[key]
                dest = out_dir / spec["filename"]
                _download(spec["url"], dest)
                print(
                    f"Validate: python scripts/validate_demo_candidate.py --video {dest} --category abnormal"
                )
            url = _gavd_youtube_url(args.gavd_rank)
            if url:
                print("\nGAVD (preferred abnormal source) — download with yt-dlp, then validate:")
                print(f"  yt-dlp -f 'bv*[height<=720]' -o '{out_dir / 'gavd_abnormal_%(id)s.%(ext)s'}' {url}")
                print(
                    f"  python scripts/validate_demo_candidate.py --video {out_dir / 'gavd_abnormal_<id>.mp4'} --category abnormal"
                )

    print(f"\nCandidates saved under {out_dir.resolve()}")
    print("See data/demo_videos/RECOMMENDED_REPLACEMENTS.md for full protocol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
