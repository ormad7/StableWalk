"""
Men's walking video catalog — Pexels direct-download URLs.

Only pose-verified clips are used for Preset / Next video cycling.
Add your own URLs in data/men_walk_urls.txt or via the GUI "Enter URL…" button.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from stablewalk import config

logger = logging.getLogger(__name__)

MIN_DETECTED_POSE_FRAMES = 5

VERIFIED_CATALOG_PATH = config.DATA_DIR / "verified_men_walk.json"

# Candidates to scan with scripts/verify_walk_urls.py (not all are walkers)
CANDIDATE_MAN_WALK_IDS: tuple[int, ...] = (
    5319095,
    6830920,
    5823532,
    4540121,
    2803277,
    5738706,
    3756871,
    5582584,
    13853768,
    5544312,
    6195655,
    4942831,
    4493661,
    6779611,
    6614354,
    6568974,
    6205120,
    5744681,
    5634291,
    5527594,
    5038381,
    4967201,
    4896021,
    4824841,
    4682481,
    6560500,
    6754532,
    7187651,
    6192899,
    5088762,
    7678717,
    3191864,
    2491284,
    4763824,
    4057250,
    5273041,
    6344420,
    3255275,
)

KNOWN_LABELS: dict[int, str] = {
    5319095: "Man walking — steady pace (best)",
    6830920: "Men walking — frozen lake",
    5738706: "Man walking — road & pine trees",
    5823532: "Walker — front view path",
    4540121: "Man walking — outdoor clip A",
    2803277: "Man walking — sidewalk (full body)",
    3756871: "Man walking — park trail",
    5582584: "Man walking — low light",
    13853768: "Man in coat — walking in fog",
    5544312: "Man walking — in park",
    6195655: "Man walking — street clip",
    4942831: "Man walking — casual pace",
    4493661: "Man walking — sidewalk",
    6779611: "Man walking — urban",
    6614354: "Man walking — daytime",
    6568974: "Man walking — open area",
    6205120: "Man walking — path B",
    5744681: "Man walking — clip C",
    5634291: "Man walking — clip D",
    5527594: "Man walking — clip E",
}

# Best default for GUI / CLI (100% pose detection in validation)
BEST_MAN_WALK_VIDEO_ID = 5319095

# Built-in fallback if JSON is missing
_BUILTIN_VERIFIED_IDS: tuple[int, ...] = (
    BEST_MAN_WALK_VIDEO_ID,
    6830920,
    5823532,
    2803277,
    3756871,
    4540121,
)

USER_URL_LIST = config.DATA_DIR / "men_walk_urls.txt"

_PEXELS_ID_RE = re.compile(r"/video/(?:[^/]+-)?(\d+)/?")


def pexels_url(video_id: int) -> str:
    return f"https://www.pexels.com/download/video/{video_id}/"


def label_for(video_id: int) -> str:
    return KNOWN_LABELS.get(video_id, f"Man walk — Pexels #{video_id}")


def extract_pexels_id(url: str) -> int | None:
    m = _PEXELS_ID_RE.search(url)
    if m:
        return int(m.group(1))
    m2 = re.search(r"/download/video/(\d+)/?", url)
    return int(m2.group(1)) if m2 else None


def load_verified_ids() -> tuple[int, ...]:
    """Pose-checked video IDs from data/verified_men_walk.json."""
    if VERIFIED_CATALOG_PATH.is_file():
        try:
            data = json.loads(VERIFIED_CATALOG_PATH.read_text(encoding="utf-8"))
            ids: list[int] = []
            for entry in data.get("videos", []):
                vid = entry.get("id")
                if isinstance(vid, int):
                    ids.append(vid)
                elif isinstance(vid, str) and vid.isdigit():
                    ids.append(int(vid))
            if ids:
                return tuple(ids)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Could not read %s: %s", VERIFIED_CATALOG_PATH, exc)
    return _BUILTIN_VERIFIED_IDS


def save_verified_catalog(entries: list[dict]) -> Path:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "comment": "Pose-checked men's walk clips. Regenerate: python scripts/verify_walk_urls.py",
        "videos": entries,
    }
    VERIFIED_CATALOG_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return VERIFIED_CATALOG_PATH


VERIFIED_MAN_WALK_IDS: tuple[int, ...] = load_verified_ids()


def build_verified_men_walk_urls() -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for vid in VERIFIED_MAN_WALK_IDS:
        label = label_for(vid)
        if VERIFIED_CATALOG_PATH.is_file():
            try:
                data = json.loads(VERIFIED_CATALOG_PATH.read_text(encoding="utf-8"))
                for entry in data.get("videos", []):
                    if entry.get("id") == vid and entry.get("label"):
                        label = str(entry["label"])
                        break
            except (OSError, json.JSONDecodeError):
                pass
        out.append((label, pexels_url(vid)))
    return tuple(out)


MEN_WALK_URLS: tuple[tuple[str, str], ...] = build_verified_men_walk_urls()


def load_user_urls() -> list[tuple[str, str]]:
    if not USER_URL_LIST.is_file():
        return []
    entries: list[tuple[str, str]] = []
    for line in USER_URL_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            label, url = line.split("|", 1)
            entries.append((label.strip(), url.strip()))
        else:
            entries.append(("Custom URL", line))
    return entries


def all_men_walk_sources(*, verified_only: bool = False) -> list[tuple[str, str]]:
    """Preset sources: verified Pexels clips (+ optional user URLs)."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    sources = list(MEN_WALK_URLS)
    if not verified_only:
        sources.extend(load_user_urls())
    for label, url in sources:
        if url in seen:
            continue
        seen.add(url)
        out.append((label, url))
    return out


def count_detected_poses(poses_json: Path) -> int:
    try:
        data = json.loads(poses_json.read_text(encoding="utf-8"))
        return int(data.get("detected_count", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def ensure_user_url_template() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if USER_URL_LIST.is_file():
        return
    lines = [
        "# Add men's walking URLs (one per line)",
        "# Format:  Label | https://www.pexels.com/download/video/VIDEO_ID/",
        "# Example page: https://www.pexels.com/search/videos/man%20walking/",
        "#",
    ]
    for label, url in MEN_WALK_URLS[:5]:
        lines.append(f"{label} | {url}")
    lines.append("# Add more lines below…")
    USER_URL_LIST.write_text("\n".join(lines) + "\n", encoding="utf-8")
