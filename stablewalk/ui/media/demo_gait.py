"""Predefined demo/comparison gait examples for the StableWalk presentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stablewalk import config


@dataclass(frozen=True)
class DemoGaitExample:
    key: str
    button_label: str
    display_name: str
    analysis_title: str
    filename: str
    source_name: str
    source_url: str
    source_attribution: str = ""
    pexels_video_id: int | None = None
    trim_start_frame: int | None = None
    note: str = ""


DEMO_GAIT_EXAMPLES: tuple[DemoGaitExample, ...] = (
    DemoGaitExample(
        key="abnormal",
        button_label="Abnormal",
        display_name="Abnormal Gait",
        analysis_title="Abnormal Gait Analysis",
        filename="abnormal_gait.mp4",
        source_name="GAVD — clinically annotated abnormal gait (pending selection)",
        source_url="https://github.com/Rahmyyy/GAVD",
        source_attribution=(
            "Source: Gait Abnormality in Video Dataset (GAVD)\n"
            "Clinical annotation + public YouTube reference — video retrieved independently"
        ),
        note=(
            "Abnormal demo must be selected from GAVD metadata using "
            "scripts/select_gavd_abnormal_candidate.py, then validated with "
            "validate_demo_candidate.py. Do not use walker-assisted or heavily occluded clips."
        ),
    ),
    DemoGaitExample(
        key="normal",
        button_label="Normal",
        display_name="Normal Gait",
        analysis_title="Normal Gait Analysis",
        filename="normal_gait.mp4",
        source_name="Health&Gait — usual gait speed (UGS) trial (pending selection)",
        source_url="https://zenodo.org/records/14039922",
        pexels_video_id=None,
        trim_start_frame=None,
        note=(
            "Normal demo uses Health&Gait UGS metadata for a controlled healthy walking trial. "
            "Install only after validate_demo_candidate.py accepts a sourced MP4."
        ),
    ),
    DemoGaitExample(
        key="athletic",
        button_label="Performance",
        display_name="Performance Gait",
        analysis_title="Performance Gait Analysis",
        filename="athletic_walking.mp4",
        source_name="Health&Gait — fast gait speed (FGS) trial (pending selection)",
        source_url="https://zenodo.org/records/14039922",
        source_attribution=(
            "Source: Health&Gait (Zafra-Palma et al., Scientific Data 2025)\n"
            "Controlled higher-speed walking (FGS) — not stock footage"
        ),
        note=(
            "Performance demo uses Health&Gait FGS metadata to select a fast-gait trial. "
            "The public Zenodo release does not redistribute raw RGB video; install only "
            "after validate_demo_candidate.py accepts a sourced MP4."
        ),
    ),
)


def demo_videos_dir() -> Path:
    return config.DEMO_VIDEOS_DIR


def demo_path(example: DemoGaitExample) -> Path:
    """Project-relative demo path resolved to an absolute Path."""
    return (config.DEMO_VIDEOS_DIR / example.filename).resolve()


def demo_exists(example: DemoGaitExample) -> bool:
    return demo_path(example).is_file()


def demo_cached_file_ready(example: DemoGaitExample, *, min_detected_rate: float = 0.25) -> bool:
    if not demo_exists(example):
        return False
    from stablewalk.ui.media.utah_abnormal import opencv_can_decode

    path = demo_path(example)
    if not opencv_can_decode(path):
        return False
    try:
        from stablewalk.ui.media.demo_validation import demo_is_ready

        return demo_is_ready(path, min_gait_detected_rate=min_detected_rate)
    except (OSError, RuntimeError, ValueError):
        return False


def demo_validation_status(example: DemoGaitExample) -> str:
    if not demo_exists(example):
        return "Demo Video: not downloaded"
    from stablewalk.ui.media.utah_abnormal import opencv_can_decode
    from stablewalk.ui.media.demo_validation import validate_demo_video

    path = demo_path(example)
    if not opencv_can_decode(path):
        return "Demo Video: cannot decode"
    return validate_demo_video(path).compact_status


def demo_stream_source(example: DemoGaitExample) -> str:
    """Always prefer validated local demo files; use project-relative paths."""
    path = demo_path(example)
    if demo_cached_file_ready(example):
        return str(path)
    if example.pexels_video_id is None:
        return str(path)
    from stablewalk.ui.media.catalog import extract_pexels_id, pexels_url

    video_id = example.pexels_video_id or extract_pexels_id(example.source_url)
    if video_id is not None:
        return pexels_url(video_id)
    return str(path)


def example_by_key(key: str) -> DemoGaitExample | None:
    for ex in DEMO_GAIT_EXAMPLES:
        if ex.key == key:
            return ex
    return None


def is_demo_video_path(path: str | Path) -> DemoGaitExample | None:
    try:
        resolved = Path(path).resolve()
        base = demo_videos_dir().resolve()
        if base not in resolved.parents and resolved.parent != base:
            return None
    except OSError:
        return None
    for ex in DEMO_GAIT_EXAMPLES:
        if resolved.name == ex.filename:
            return ex
    return None


def missing_file_message(example: DemoGaitExample) -> str:
    folder = demo_videos_dir()
    target = folder / example.filename
    if example.key == "abnormal":
        return (
            f"Abnormal demo video not found or cannot be decoded.\n\n"
            f"  {target.resolve()}\n\n"
            f"Download from University of Utah NeuroLogic Examination:\n"
            f"  {example.source_url}\n\n"
            f"Run: python scripts/download_utah_abnormal_demo.py"
        )
    return (
        f"Demo video not found.\n\n"
        f"  {target.resolve()}\n\n"
        f"Source: {example.source_url}\n\n"
        f"Run: python scripts/download_demo_videos.py"
    )


def missing_file_placeholder(example: DemoGaitExample) -> str:
    return (
        f"{example.display_name}\n\n"
        f"{demo_validation_status(example) if demo_exists(example) else 'Demo Video: not downloaded'}\n\n"
        f"Target: {demo_path(example)}\n\n"
        f"See DEMO_VIDEOS.md"
    )


def ensure_demo_video(example: DemoGaitExample) -> bool:
    from stablewalk.ui.media.demo_download import ensure_demo_video as _ensure

    return _ensure(example)
