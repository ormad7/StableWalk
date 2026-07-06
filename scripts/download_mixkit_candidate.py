"""Download Mixkit athletic walking candidate for validation."""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = "https://mixkit.co/free-stock-video/man-walking-wearing-activewear-596/"
OUT = ROOT / "data" / "demo_videos" / "_candidate_mixkit_596_raw.mp4"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": PAGE,
}


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    page = _fetch(PAGE).decode("utf-8", "replace")
    urls = sorted(set(re.findall(r"https://assets\.mixkit\.co[^\"'\s>]+\.mp4", page)))
    print(f"Found {len(urls)} mp4 URLs on page")
    for url in urls:
        print(f"  {url}")

    candidates = urls or [
        "https://assets.mixkit.co/videos/596/596-720.mp4",
        "https://assets.mixkit.co/videos/596/596-1080.mp4",
        "https://assets.mixkit.co/videos/preview/mixkit-man-walking-wearing-activewear-596-large.mp4",
    ]
    for url in candidates:
        try:
            print(f"Downloading {url} ...")
            data = _fetch(url)
            if len(data) < 100_000:
                print(f"  skip: too small ({len(data)} bytes)")
                continue
            OUT.write_bytes(data)
            print(f"Saved {OUT} ({len(data) / 1e6:.2f} MB)")
            return 0
        except Exception as exc:
            print(f"  failed: {exc}")

    # Try download modal endpoint
    modal = "https://mixkit.co/free-stock-video/download/596/?context=sidebar"
    try:
        print(f"Trying modal {modal} ...")
        html = _fetch(modal).decode("utf-8", "replace")
        dl = re.findall(r"https://assets\.mixkit\.co[^\"'\s>]+\.mp4", html)
        for url in dl:
            print(f"  modal url: {url}")
            data = _fetch(url)
            OUT.write_bytes(data)
            print(f"Saved {OUT} ({len(data) / 1e6:.2f} MB)")
            return 0
    except Exception as exc:
        print(f"modal failed: {exc}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
