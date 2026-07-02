"""
Men's walking presets for the GUI (verified URLs + optional local files).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stablewalk import config
from stablewalk.ui.media.catalog import (
    MIN_DETECTED_POSE_FRAMES,
    all_men_walk_sources,
    count_detected_poses,
    ensure_user_url_template,
    extract_pexels_id,
    load_verified_ids,
    pexels_url,
)

# Re-export for gui
__all__ = [
    "MIN_DETECTED_POSE_FRAMES",
    "VideoPreset",
    "STABLE_WALKING_PRESETS",
    "LOCAL_FILE_PRESETS",
    "CUSTOM_URL_PRESET_LABEL",
    "all_preset_labels",
    "preset_by_label",
    "list_runnable_presets",
    "preset_from_url_catalog",
    "preset_from_custom_url",
]

CUSTOM_URL_PRESET_LABEL = "(enter your own URL)"


@dataclass(frozen=True)
class VideoPreset:
    label: str
    url: str
    notes: str = ""
    verified: bool = False
    is_url: bool = True


def _build_url_presets() -> tuple[VideoPreset, ...]:
    ensure_user_url_template()
    verified_ids = set(load_verified_ids())
    presets: list[VideoPreset] = []
    seen: set[str] = set()
    for label, url in all_men_walk_sources(verified_only=False):
        if url in seen:
            continue
        seen.add(url)
        vid = extract_pexels_id(url)
        is_verified = vid in verified_ids if vid is not None else False
        presets.append(
            VideoPreset(
                label=label,
                url=url,
                notes="Men walking (URL)",
                verified=is_verified,
                is_url=True,
            )
        )
    return tuple(presets)


STABLE_WALKING_PRESETS: tuple[VideoPreset, ...] = _build_url_presets()

LOCAL_FILE_PRESETS: tuple[VideoPreset, ...] = (
    VideoPreset(
        label="Local: my_walk.mp4",
        url=str(config.INPUT_DIR / "my_walk.mp4"),
        verified=False,
        is_url=False,
    ),
    VideoPreset(
        label="Local: walking_demo.mp4",
        url=str(config.INPUT_DIR / "walking_demo.mp4"),
        verified=False,
        is_url=False,
    ),
    VideoPreset(
        label="Local: sample_walk.mp4",
        url=str(config.INPUT_DIR / "sample_walk.mp4"),
        verified=False,
        is_url=False,
    ),
)


def all_preset_labels() -> list[str]:
    labels = [p.label for p in STABLE_WALKING_PRESETS if p.verified]
    for p in STABLE_WALKING_PRESETS:
        if not p.verified and p.label not in labels:
            labels.append(p.label)
    for p in LOCAL_FILE_PRESETS:
        if Path(p.url).is_file():
            labels.append(p.label)
    labels.append(CUSTOM_URL_PRESET_LABEL)
    return labels


def preset_by_label(label: str) -> VideoPreset | None:
    if label == CUSTOM_URL_PRESET_LABEL:
        return None
    for p in STABLE_WALKING_PRESETS + LOCAL_FILE_PRESETS:
        if p.label == label:
            return p
    return None


def preset_from_custom_url(url: str, *, label: str | None = None) -> VideoPreset:
    url = url.strip()
    vid = extract_pexels_id(url)
    if label is None:
        label = f"Custom: Pexels #{vid}" if vid else "Custom URL"
    return VideoPreset(label=label, url=url, verified=False, is_url=True)


def preset_from_url_catalog(index: int) -> VideoPreset | None:
    """Preset by index in verified men's URL list (for Next video)."""
    urls = [p for p in list_runnable_presets(verified_only=True, urls_only=True)]
    if not urls:
        return None
    return urls[index % len(urls)]


def list_runnable_presets(
    *,
    verified_only: bool = True,
    urls_only: bool = False,
) -> list[VideoPreset]:
    """
    Presets for Next video.

    Default: pose-verified men's walk URLs only.
  Local files are skipped in auto-cycle (often fail pose detection).
    """
    runnable: list[VideoPreset] = []
    for p in STABLE_WALKING_PRESETS:
        if p.is_url:
            if not verified_only or p.verified:
                runnable.append(p)
    if not urls_only:
        for p in LOCAL_FILE_PRESETS:
            path = Path(p.url)
            if path.is_file() and path.stat().st_size > 50_000:
                runnable.append(p)
    return runnable
