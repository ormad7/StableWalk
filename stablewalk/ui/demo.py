"""
Presentation demo configuration: two walking videos for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stablewalk import config


@dataclass(frozen=True)
class DemoClip:
    """One video in the scripted demonstration."""

    label: str
    source: str
    description: str


def resolve_demo_video_b() -> str:
    """Local file preferred; else second verified Pexels URL."""
    for name in (
        "walking_demo.mp4",
        "sample_walk.mp4",
        "my_walk.mp4",
    ):
        path = config.INPUT_DIR / name
        if path.is_file() and path.stat().st_size > 5000:
            return str(path.resolve())
    return "https://www.pexels.com/download/video/6830920/"


def get_demo_clips() -> tuple[DemoClip, DemoClip]:
    """Video A = stable reference; Video B = alternate pattern."""
    return (
        DemoClip(
            label="Video A — man walking (steady)",
            source=config.DEFAULT_WALKING_VIDEO_URL,
            description="Man walking — reference gait",
        ),
        DemoClip(
            label="Video B — alternate pattern",
            source=resolve_demo_video_b(),
            description="Second walking style for comparison",
        ),
    )


# Faster processing during live demo (still enough frames for charts)
DEMO_MAX_FRAMES: int | None = 50

# Pause between demo steps (milliseconds)
DEMO_PAUSE_MS = 4000
DEMO_PLAYBACK_MS = 4500
DEMO_FINAL_HOLD_MS = 5000
