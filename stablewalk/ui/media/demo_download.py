"""Download and prepare StableWalk demo gait videos."""

from __future__ import annotations

import logging
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from stablewalk.ui.media.catalog import pexels_url
from stablewalk.ui.media.demo_gait import DemoGaitExample, demo_exists, demo_path

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def _download_pexels_video(video_id: int, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = pexels_url(video_id)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as out:
            while True:
                block = resp.read(1024 * 256)
                if not block:
                    break
                out.write(block)
        tmp.replace(dest)
        return dest.is_file() and dest.stat().st_size > 10_000
    except urllib.error.URLError as exc:
        logger.warning("Pexels download failed for %s: %s", video_id, exc)
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        return False


def _prepare_downloaded_pexels(raw: Path, dest: Path, example: DemoGaitExample) -> bool:
    from stablewalk.ui.media.demo_prepare import prepare_pexels_demo

    ok, detail = prepare_pexels_demo(
        raw,
        dest,
        start_frame=example.trim_start_frame,
        output_frames=120,
    )
    logger.info("Prepared %s: %s", example.filename, detail)
    return dest.is_file()


def download_pexels_demo(example: DemoGaitExample, *, force: bool = False) -> bool:
    from stablewalk.ui.media.demo_gait import demo_cached_file_ready

    dest = demo_path(example)
    if dest.is_file() and not force and demo_cached_file_ready(example):
        return True
    video_id = example.pexels_video_id
    if video_id is None:
        return False
    with tempfile.TemporaryDirectory() as tmpdir:
        raw = Path(tmpdir) / "raw.mp4"
        if not _download_pexels_video(video_id, raw):
            return False
        return _prepare_downloaded_pexels(raw, dest, example)


def download_demo_video(key: str, *, force: bool = False) -> bool:
    from stablewalk.ui.media.demo_gait import demo_cached_file_ready, example_by_key
    from stablewalk.ui.media.utah_abnormal import ensure_utah_abnormal_demo

    ex = example_by_key(key)
    if ex is None:
        return False
    if ex.key == "abnormal":
        return ensure_utah_abnormal_demo(force=force)
    dest = demo_path(ex)
    if dest.is_file() and not force and demo_cached_file_ready(ex):
        return True
    return download_pexels_demo(ex, force=True)


def ensure_demo_video(example: DemoGaitExample) -> bool:
    from stablewalk.ui.media.demo_gait import demo_cached_file_ready
    from stablewalk.ui.media.utah_abnormal import ensure_utah_abnormal_demo

    if demo_exists(example) and demo_cached_file_ready(example):
        return True
    if example.key == "abnormal":
        return ensure_utah_abnormal_demo(force=True)
    return download_pexels_demo(example, force=True)
