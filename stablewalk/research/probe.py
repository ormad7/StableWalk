"""Local video probe (FPS, resolution). Never downloads remote URLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from stablewalk import config

REMOTE_SCHEMES = ("http", "https", "ftp")


def is_remote_url(value: str | Path | None) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw)
    return parsed.scheme.lower() in REMOTE_SCHEMES


def probe_local_video(path: str | Path) -> dict[str, Any]:
    """Read FPS and frame size from a local file. Does not fetch URLs."""
    target = Path(path)
    if is_remote_url(path):
        raise ValueError(
            "Remote URLs are not probed or downloaded. Import a local file "
            "you are licensed to use."
        )
    if not target.is_file():
        raise FileNotFoundError(f"Video not found: {target}")
    suffix = target.suffix.lower()
    if suffix not in config.VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video type: {suffix}")
    info: dict[str, Any] = {
        "path": str(target.resolve()),
        "fps": None,
        "width": None,
        "height": None,
        "frame_count": None,
        "duration_s": None,
    }
    try:
        import cv2
    except Exception:
        return info
    cap = cv2.VideoCapture(str(target))
    try:
        if not cap.isOpened():
            return info
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        info["fps"] = fps if fps > 1e-3 else None
        info["width"] = width or None
        info["height"] = height or None
        info["frame_count"] = n or None
        if fps > 1e-3 and n > 0:
            info["duration_s"] = float(n / fps)
    finally:
        cap.release()
    return info
