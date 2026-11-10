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
    dataset_name: str = ""
    documented_condition: str = ""
    camera_view: str = ""
    locomotion_setting: str = ""
    license_access: str = ""
    interval_description: str = ""
    pexels_video_id: int | None = None
    trim_start_frame: int | None = None
    note: str = ""


LEGACY_DEMO_FILENAMES: frozenset[str] = frozenset(
    {
        "abnormal_gait.mp4",
        "normal_gait.mp4",
        "athletic_walking.mp4",
    }
)


def is_legacy_demo_filename(name: str) -> bool:
    """True for retired demo filenames that must not be loaded as Examples."""
    return Path(name).name.lower() in {item.lower() for item in LEGACY_DEMO_FILENAMES}


DEMO_GAIT_EXAMPLES: tuple[DemoGaitExample, ...] = (
    DemoGaitExample(
        key="abnormal",
        button_label="Abnormal Gait",
        display_name="Abnormal Gait",
        analysis_title="Abnormal Gait Analysis",
        filename="abnormal_gait_v2.mp4",
        source_name="GAVD — gait_pat=parkinsons",
        source_url="https://www.youtube.com/watch?v=M-_cogKwXK4",
        source_attribution=(
            "Gait Abnormality in Video Dataset (GAVD).\n"
            "Documented label: gait_pat=parkinsons. YouTube M-_cogKwXK4."
        ),
        dataset_name="GAVD (Gait Abnormality in Video Dataset)",
        documented_condition=(
            "GAVD clinician-annotated Parkinson’s gait (gait_pat=parkinsons). "
            "YouTube title: “1.12. Vascular Parkinsonism - Parkinsonism and Related "
            "[Spring Video Atlas]”. Not dementia. Not inferred from appearance."
        ),
        license_access=(
            "GAVD annotations: MIT. Video retrieved independently from YouTube; "
            "use under YouTube Terms of Service. Dataset page: https://github.com/Rahmyyy/GAVD"
        ),
        camera_view="",
        locomotion_setting="",
        interval_description=(
            "Annotated walking sequence cljwrgl2p00243n6lwusay4pt, "
            "source frames 4625–5274"
        ),
        trim_start_frame=4625,
        note=(
            "Source label is GAVD gait_pat=parkinsons, not a diagnosis computed by StableWalk. "
            "Dementia RGB remains Dem@Care-gated and was not substituted."
        ),
    ),
    DemoGaitExample(
        key="normal",
        button_label="Normal Gait",
        display_name="Normal Gait",
        analysis_title="Normal Gait Analysis",
        filename="normal_gait_v2.mp4",
        source_name="GAVD — gait_pat=normal",
        source_url="https://www.youtube.com/watch?v=eCCYhDSDlDc",
        source_attribution=(
            "Gait Abnormality in Video Dataset (GAVD).\n"
            "Documented label: gait_pat=normal. YouTube eCCYhDSDlDc."
        ),
        dataset_name="GAVD (Gait Abnormality in Video Dataset)",
        documented_condition=(
            "GAVD clinician-annotated ordinary walking (gait_pat=normal, dataset=Normal Gait). "
            "YouTube title: “How to walk properly (Without pain)”. "
            "Not Health&Gait UGS. Not athletic. Not fast walking."
        ),
        license_access=(
            "GAVD annotations: MIT. Video retrieved independently from YouTube; "
            "use under YouTube Terms of Service. Dataset page: https://github.com/Rahmyyy/GAVD"
        ),
        camera_view="Right side",
        locomotion_setting="",
        interval_description=(
            "Annotated walking sequence cljwtgpsm000y3n6lgbj91b5h (right side), "
            "source frames 379–678 (10.01 s; shorter than the 15–30 s target)"
        ),
        note=(
            "Newly downloaded 2026-11-01. Timing is the source FPS; "
            "the clip is not looped or interpolated."
        ),
    ),
    DemoGaitExample(
        key="athletic",
        button_label="Fast Walking",
        display_name="Fast Walking",
        analysis_title="Fast Walking Analysis",
        filename="fast_walking_v2.mp4",
        source_name="Twente wearable-system validation — Subj04 walk_54",
        source_url="https://zenodo.org/records/6644593",
        source_attribution=(
            "Huawei Wang (2022). RGB data [front and side views] of the wearable "
            "system validation study. Zenodo 10.5281/zenodo.6644593.\n"
            "Trial: Subj04/walk_54_side_anonymized.avi — treadmill walking 5.4 km/h."
        ),
        dataset_name="Twente wearable-system validation RGB (Zenodo 6644593)",
        documented_condition=(
            "Documented treadmill walking at 5.4 km/h (walk_54), the fastest walking "
            "condition in Wang et al., Wearable Technologies 2023 (10.1017/wtc.2023.7). "
            "Protocol: six walking speeds 0.9–5.4 km/h, then running from 6.3 km/h. "
            "Fixed left-side camera; full body and feet visible. Not running. "
            "Not Health&Gait FGS. athletic_walking.mp4 is not used."
        ),
        license_access=(
            "Zenodo 10.5281/zenodo.6644593 is CC BY 4.0 (anonymized RGB). "
            "Companion paper: Wang, Basu, Durandau & Sartori, Wearable Technologies 4:e7, "
            "2023 (10.1017/wtc.2023.7). The Kuopio dataset linked from PMC11385067 "
            "(Zenodo 10559504) was inspected first: public files are mocap/IMU/OpenPose "
            "JSON only — no RGB."
        ),
        camera_view="Left side (fixed)",
        locomotion_setting="Treadmill",
        interval_description=(
            "Full source trial Subj04/walk_54_side_anonymized.avi "
            "(2261 frames at the original 33.000 FPS; 68.52 s). "
            "Stream-copied to MP4; not trimmed, looped, sped up, or interpolated."
        ),
        note=(
            "Source label is the protocol speed walk_54 = 5.4 km/h treadmill walking. "
            "Computed scores are not proof of fast walking. "
            "Health&Gait FGS RGB remains unpublished and was not used."
        ),
    ),
)


def demo_videos_dir() -> Path:
    return config.DEMO_VIDEOS_DIR


def demo_path(example: DemoGaitExample) -> Path:
    """Official v2 demo path. Never remaps to athletic_walking.mp4 or other legacy names."""
    name = Path(example.filename).name
    if is_legacy_demo_filename(name):
        raise ValueError(f"Legacy demo filename is not loadable: {name}")
    return (config.DEMO_VIDEOS_DIR / name).resolve()


def probe_demo_timing(path: Path) -> dict[str, float | int | str]:
    """Read container FPS, frame count, and duration from the installed file."""
    import cv2

    info: dict[str, float | int | str] = {
        "opened": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_s": 0.0,
        "width": 0,
        "height": 0,
    }
    if not path.is_file():
        return info
    cap = cv2.VideoCapture(str(path))
    opened = bool(cap.isOpened())
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    info.update(
        {
            "opened": 1 if opened else 0,
            "fps": fps,
            "frame_count": frames,
            "duration_s": (frames / fps) if fps > 1e-6 else 0.0,
            "width": width,
            "height": height,
        }
    )
    return info


def format_about_active_example(example: DemoGaitExample, *, path: Path | None = None) -> str:
    """Source metadata for About active example — no computed analysis scores."""
    video_path = path or demo_path(example)
    timing = probe_demo_timing(video_path)
    fps = float(timing["fps"])
    duration_s = float(timing["duration_s"])
    frames = int(timing["frame_count"])
    if fps > 1e-6:
        fps_line = f"{fps:.3f} FPS ({frames} frames in the installed file)"
        duration_line = f"{duration_s:.2f} s"
    else:
        fps_line = "unavailable (file missing or unreadable)"
        duration_line = "unavailable"
    lines = [
        example.display_name,
        "",
        f"Source URL: {example.source_url}",
        f"Dataset: {example.dataset_name or example.source_name}",
        f"Documented condition / activity: {example.documented_condition or '—'}",
        f"License / access: {example.license_access or 'See source URL.'}",
        f"Original FPS (installed file): {fps_line}",
        f"Duration (installed file): {duration_line}",
        f"Selected time interval: {example.interval_description or 'full installed file'}",
    ]
    if example.note:
        lines.extend(["", example.note])
    lines.extend(
        [
            "",
            "These lines are source labels only. Gait scores, cycle counts, and "
            "confidence badges come from analysis of the current load and are not "
            "hard-coded for this example.",
        ]
    )
    return "\n".join(lines)


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
    """Return only the official local v2 path — never a Pexels URL or legacy file."""
    return str(demo_path(example))


# Default tracked joint per demo category — keeps teacher comparison consistent:
# GAVD Parkinson and ordinary-walk examples use knee ROM by default.
DEMO_DEFAULT_DOF_ITEM: dict[str, str] = {
    "abnormal": "right_knee",
    "normal": "right_knee",
    "athletic": "right_knee",
}

DEMO_CATEGORY_TAGLINES: dict[str, str] = {
    "abnormal": "Source label: GAVD gait_pat=parkinsons (YouTube: Vascular Parkinsonism)",
    "normal": "Source label: GAVD gait_pat=normal · How to walk properly",
    "athletic": "Source label: Twente walk_54 · treadmill walking 5.4 km/h (side camera)",
}


@dataclass(frozen=True)
class DemoGaitInterpretation:
    """Teacher-facing gait analysis copy for each demo category."""

    category_headline: str
    movement_stability: str
    gait_quality: str
    analysis_confidence: str
    teacher_compare: str


def demo_category_tagline(key: str) -> str:
    """One-line teacher-facing description shown under the demo buttons."""
    return DEMO_CATEGORY_TAGLINES.get(key, "")


def _insufficient_data_note(*, usable_cycles: int, completeness_pct: float) -> str | None:
    """Return a warning when computed evidence is too thin — never a source label."""
    if usable_cycles < 3 or completeness_pct < 45.0:
        return (
            f"Insufficient data for a reliable gait-cycle summary "
            f"({usable_cycles} usable cycle(s), {completeness_pct:.0f}% complete)."
        )
    return None


def demo_gait_interpretation(
    key: str,
    *,
    usable_cycles: int = 0,
    detected_cycles: int = 0,
    completeness_pct: float = 0.0,
    movement_stability_score: float | None = None,
    gait_quality_score: float | None = None,
) -> DemoGaitInterpretation | None:
    """Sidebar copy: source condition first, then computed counts (no expected scores)."""
    del movement_stability_score, gait_quality_score
    thin = _insufficient_data_note(
        usable_cycles=usable_cycles, completeness_pct=completeness_pct
    )
    computed = (
        f"Computed this load: {detected_cycles} detected / {usable_cycles} usable "
        f"gait cycle(s), {completeness_pct:.0f}% complete."
    )
    if key == "abnormal":
        return DemoGaitInterpretation(
            category_headline="Abnormal Gait — Parkinson’s (GAVD gait_pat=parkinsons)",
            movement_stability=(
                "Source label is GAVD gait_pat=parkinsons. Computed torso/"
                "pelvis scores are not a diagnosis."
            ),
            gait_quality=computed if thin is None else f"{computed} {thin}",
            analysis_confidence=(
                thin
                or (
                    f"Computed evidence mass {completeness_pct:.0f}% complete. "
                    "Not a clinical score."
                )
            ),
            teacher_compare=(
                "Source contrast vs Normal: this clip is labeled parkinsons by GAVD. "
                "Do not treat cycle counts as a preset expectation."
            ),
        )
    if key == "normal":
        return DemoGaitInterpretation(
            category_headline="Normal Gait — GAVD gait_pat=normal",
            movement_stability=(
                "Source label is clinician-annotated normal walking. Computed pelvis/"
                "torso values describe this recording only."
            ),
            gait_quality=computed if thin is None else f"{computed} {thin}",
            analysis_confidence=(
                thin
                or f"Computed evidence mass {completeness_pct:.0f}% complete."
            ),
            teacher_compare=(
                "Source contrast: GAVD normal walking, no assistive device in the "
                "annotated sequence. Fast Walking is a separate Twente walk_54 trial."
            ),
        )
    if key == "athletic":
        return DemoGaitInterpretation(
            category_headline="Fast Walking — Twente walk_54 (5.4 km/h treadmill walking)",
            movement_stability=(
                "Source label is documented 5.4 km/h treadmill walking (walk_54). "
                "Computed torso/pelvis scores are not a speed measurement."
            ),
            gait_quality=computed if thin is None else f"{computed} {thin}",
            analysis_confidence=(
                thin
                or (
                    f"Computed evidence mass {completeness_pct:.0f}% complete. "
                    "Not a clinical score."
                )
            ),
            teacher_compare=(
                "Source contrast vs Normal: this clip is protocol-labeled walk_54 "
                "(5.4 km/h). Do not treat cycle counts as a preset expectation."
            ),
        )
    return None


def demo_default_dof_item(key: str) -> str:
    """Body point auto-selected when a demo category loads."""
    return DEMO_DEFAULT_DOF_ITEM.get(key, "right_knee")


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
    if is_legacy_demo_filename(resolved.name):
        return None
    for ex in DEMO_GAIT_EXAMPLES:
        if resolved.name == ex.filename:
            return ex
    return None


def missing_file_message(example: DemoGaitExample) -> str:
    target = demo_path(example)
    header = f"{example.display_name} is not available."
    body = (
        f"{header}\n\n"
        f"Expected file:\n  {target}\n\n"
        f"Dataset: {example.dataset_name or example.source_name}\n"
        f"Source: {example.source_url}\n"
        f"Documented condition / activity: {example.documented_condition or '—'}\n\n"
        f"StableWalk will not load athletic_walking.mp4, abnormal_gait.mp4, or "
        f"normal_gait.mp4 as a substitute."
    )
    if example.key == "athletic":
        body += (
            "\n\nExpected Fast Walking file is Twente Subj04 walk_54_side "
            "(Zenodo 10.5281/zenodo.6644593). athletic_walking.mp4 is not used."
        )
    return body


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
