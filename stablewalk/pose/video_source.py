"""
Video source validation and run naming for multi-video pipelines.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

from stablewalk import config
from stablewalk.pose.video import is_video_url
from stablewalk.pose.video_validation import validate_video_source

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]+")


def normalize_source(source: str) -> str:
    """Strip whitespace; expand local paths."""
    source = source.strip()
    if not source:
        return source
    if is_video_url(source):
        return source
    p = Path(source)
    if p.is_file():
        return str(p.resolve())
    alt = config.INPUT_DIR / p.name
    if alt.is_file():
        return str(alt.resolve())
    return source


def content_cache_key(source: str) -> str:
    """
    Stable key for a video source (URL string or file path + size + mtime).

    Different files / URLs never share the same key.
    """
    source = normalize_source(source)
    if is_video_url(source):
        material = source.strip()
    else:
        path = Path(source)
        stat = path.stat()
        material = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime_ns)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def derive_run_name(source: str, *, unique_session: bool = False) -> str:
    """
    Output folder name keyed by video content.

    If unique_session is True, append a timestamp so every GUI load gets a
    brand-new folder (no stale frame reuse).
    """
    import time

    digest = content_cache_key(source)[:16]
    base = f"run_{digest}"
    if unique_session:
        return f"{base}_{int(time.time() * 1000)}"
    return base


def check_source_format(source: str) -> tuple[bool, str]:
    """
    Quick format check before opening the capture.

    Returns:
        (ok, message)
    """
    source = normalize_source(source)
    if not source:
        return False, "No video source provided."

    if is_video_url(source):
        parsed = urlparse(source)
        if parsed.scheme not in ("http", "https"):
            return False, f"Unsupported URL scheme: {parsed.scheme}"
        return True, "URL format OK."

    path = Path(source)
    if not path.is_file():
        return False, f"Video file not found:\n{source}"

    ext = path.suffix.lower()
    if ext and ext not in config.VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(config.VIDEO_EXTENSIONS))
        return (
            False,
            f"Unsupported file type '{ext}'.\nSupported: {supported}",
        )
    return True, f"Local file OK ({path.name})"


def quick_validate_source(source: str) -> tuple[bool, str]:
    """
    Fast check: video opens and at least one frame is readable.

    Used for GUI / Next video so strict full-body sampling does not block loading.
    """
    import cv2

    from stablewalk.core.pipeline_reset import register_capture, release_all_captures

    source = normalize_source(source)
    release_all_captures()
    cap = cv2.VideoCapture(source)
    register_capture(cap)
    if not cap.isOpened():
        release_all_captures()
        return False, "Could not open video source."
    ret, frame = cap.read()
    release_all_captures()
    if not ret or frame is None:
        return False, "Video has no readable frames."
    return True, "OK"


def resolve_validate_mode(source: str) -> bool | str:
    """
    Validation mode for pipeline.

    - Local files: skip strict pre-check (process then show results)
    - URLs: quick open/frame check in GUI; use config default elsewhere
    """
    from stablewalk import config

    source = normalize_source(source)
    if not is_video_url(source):
        return False
    return getattr(config, "GUI_VIDEO_VALIDATE_MODE", "quick")


def validate_source(
    source: str,
    *,
    sample_count: int | None = None,
    min_valid_ratio: float | None = None,
    model_variant: str | None = None,
    mode: bool | str = True,
) -> tuple[bool, float, str]:
    """
    Validate video source.

    mode:
      - False: format check only
      - "quick": open + read one frame
      - True: full-body pose sampling (strict)
    """
    source = normalize_source(source)
    ok, msg = check_source_format(source)
    if not ok:
        return False, 0.0, msg

    if mode is False:
        return True, 1.0, msg

    if mode == "quick":
        ok, qmsg = quick_validate_source(source)
        return ok, 1.0 if ok else 0.0, qmsg if ok else qmsg

    from stablewalk.core.pipeline_reset import release_all_captures

    release_all_captures()
    return validate_video_source(
        source,
        sample_count=sample_count or config.DEFAULT_VIDEO_VALIDATION_SAMPLES,
        min_valid_ratio=min_valid_ratio or config.DEFAULT_MIN_VALID_FRAME_RATIO,
        model_variant=model_variant or config.DEFAULT_POSE_MODEL_VARIANT,
    )
