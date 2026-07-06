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
        display_name="Abnormal / Neuropathic Gait",
        analysis_title="Abnormal / Neuropathic Gait Analysis",
        filename="abnormal_gait.mp4",
        source_name="University of Utah – NeuroLogic Examination",
        source_url="https://neurologicexam.med.utah.edu/adult/html/gait_abnormal.html",
        source_attribution=(
            "Source: University of Utah – NeuroLogic Examination\n"
            "Neuropathic Gait – Clinical Walking Example"
        ),
        note=(
            "Abnormal gait pattern: right foot dorsiflexion weakness with compensatory "
            "high stepping for foot clearance. Utah video identifier: gait_ab_10."
        ),
    ),
    DemoGaitExample(
        key="normal",
        button_label="Normal",
        display_name="Normal Gait",
        analysis_title="Normal Gait Analysis",
        filename="normal_gait.mp4",
        source_name="Pexels — man walking towards the camera",
        source_url="https://www.pexels.com/video/a-man-walking-towards-the-camera-5320110/",
        pexels_video_id=5320110,
        trim_start_frame=87,
        note="Steady full-body walking; trimmed to the continuous walking segment.",
    ),
    DemoGaitExample(
        key="athletic",
        button_label="Athletic",
        display_name="Athletic Walking",
        analysis_title="Athletic Walking Analysis",
        filename="athletic_walking.mp4",
        source_name="Pexels — a man walking on a tennis court",
        source_url="https://www.pexels.com/video/a-man-walking-on-a-tennis-court-27727783/",
        source_attribution=(
            "Source: Pexels — Lola bertoncelli\n"
            "Outdoor tennis-court walk in sportswear (rear view, full body)"
        ),
        pexels_video_id=27727783,
        trim_start_frame=None,
        note=(
            "Rear-view sportswear walking on a tennis court. Replaced Pexels 5823532 "
            "(2026-07-06) after gait-quality investigation: heel visibility 0.94 vs 0.61, "
            "HIGH step-detection confidence vs LOW, 100% pose detection."
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
