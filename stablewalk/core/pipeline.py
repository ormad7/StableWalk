"""
End-to-end gait pipeline: video URL or file -> pose -> export JSON.

Performance: default path decodes video once (pose + optional frame cache).
Legacy two-pass path (extract JPG then re-read) available via ``cache_frames='legacy'``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from stablewalk import config
from stablewalk.models.pose_data import FrameExtractionResult
from stablewalk.core.pipeline_reset import clear_run_artifacts, release_all_captures
from stablewalk.pose.backends.pipeline_runner import (
    PoseExtractionResult,
    extract_pose_from_frames_dir,
    extract_pose_video_with_frame_cache,
)
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.pose.video import VideoProcessor, is_video_url
from stablewalk.pose.video_source import content_cache_key, derive_run_name, normalize_source, validate_source

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]
CacheFramesMode = Literal[False, True, "legacy"]


@dataclass
class PipelineResult:
    """Fresh pipeline output (never loaded from stale cache)."""

    poses_path: Path
    frames_dir: Path | None
    source: str
    run_name: str
    content_key: str
    frame_count: int
    fps: float
    width: int
    height: int
    first_frame_hash: str
    pose_backend_requested: str = "mediapipe"
    pose_backend_used: str = "mediapipe"
    pose_backend_fallback: bool = False
    pose_backend_fallback_reason: str | None = None
    smpl_motion_path: Path | None = None


def _noop_progress(message: str, fraction: float) -> None:
    pass


def _first_frame_hash(frame_paths: list[str]) -> str:
    if not frame_paths:
        return "-"
    data = Path(frame_paths[0]).read_bytes()
    return hashlib.md5(data).hexdigest()[:12]


def _save_pose_sequence(sequence, poses_path: Path) -> None:
    with PoseEstimator(
        model_variant=config.DEFAULT_POSE_MODEL_VARIANT,
        video_mode=True,
    ) as estimator:
        estimator.save_sequence(sequence, poses_path)


def _export_motion_artifacts(
    pose_result: PoseExtractionResult | None,
    run_name: str,
) -> Path | None:
    """Export smpl_motion.npz (SMPL only) under motion_reference/{run}."""
    if pose_result is None:
        return None
    from stablewalk.io.smpl_motion_export import maybe_export_smpl_motion

    run_dir = config.MOTION_REFERENCE_EXPORT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    smpl_path = maybe_export_smpl_motion(pose_result.unified_motion, run_dir)
    if smpl_path:
        logger.info("SMPL motion export → %s", smpl_path)
    for warning in pose_result.resolution.warnings:
        logger.warning(warning)
    return smpl_path


def _run_legacy_two_pass(
    source: str,
    frames_dir: Path,
    poses_path: Path,
    *,
    from_url: bool,
    max_frames: int | None,
    progress: ProgressCallback,
    pose_backend: str | None = None,
) -> tuple[FrameExtractionResult, list[str], PoseExtractionResult]:
    """Extract JPGs, then re-read each file for pose (slow, debug-friendly)."""
    progress("Extracting frames (legacy)…", 0.15)
    processor = VideoProcessor(jpeg_quality=config.DEFAULT_JPEG_QUALITY)
    if from_url:
        extraction = processor.extract_frames_from_url(
            source,
            frames_dir,
            max_frames=max_frames,
            skip_existing=False,
        )
    else:
        extraction = processor.extract_frames(
            Path(source),
            frames_dir,
            max_frames=max_frames,
            skip_existing=False,
        )
    if extraction.frame_count == 0:
        raise RuntimeError("Video returned no frames — invalid or empty video source.")

    progress("Running pose estimation (legacy)…", 0.35)
    pose_result = extract_pose_from_frames_dir(
        frames_dir,
        source_video=source,
        fps=extraction.fps,
        backend_name=pose_backend or config.POSE_BACKEND,
        max_frames=max_frames,
        enrich=True,
    )
    progress("Saving JSON export…", 0.9)
    _save_pose_sequence(pose_result.sequence, poses_path)
    return extraction, extraction.frame_paths, pose_result


def _run_single_pass(
    source: str,
    frames_dir: Path,
    poses_path: Path,
    *,
    cache_frames: bool,
    max_frames: int | None,
    progress: ProgressCallback,
    pose_backend: str | None = None,
) -> tuple[FrameExtractionResult, list[str], PoseExtractionResult]:
    """One video decode: pose inference + optional JPG cache for GUI playback."""
    progress("Processing video (single pass)…", 0.2)
    pose_result, extraction = extract_pose_video_with_frame_cache(
        source,
        frames_dir if cache_frames else None,
        backend_name=pose_backend or config.POSE_BACKEND,
        max_frames=max_frames,
        jpeg_quality=config.DEFAULT_JPEG_QUALITY,
        enrich=True,
    )
    if extraction.frame_count == 0:
        raise RuntimeError("Video returned no frames — invalid or empty video source.")
    progress("Saving JSON export…", 0.9)
    _save_pose_sequence(pose_result.sequence, poses_path)
    return extraction, extraction.frame_paths, pose_result


def _auto_export_opensim(poses_path: Path, run_name: str) -> dict[str, Path] | None:
    """Export OpenSim-compatible files after pose JSON is written."""
    try:
        from stablewalk.opensim_integration import export_from_pose_json
        from stablewalk.opensim_sdk import log_post_analysis_opensim_status

        written = export_from_pose_json(
            poses_path, config.OPENSIM_DIR / run_name, name=run_name
        )
        log_post_analysis_opensim_status(written, logger)
        return written
    except (OSError, ValueError) as exc:
        logger.error("OpenSim export failed: %s", exc)
        from stablewalk.opensim_sdk import log_post_analysis_opensim_status

        log_post_analysis_opensim_status(None, logger)
        return None


def run_gait_pipeline(
    source: str,
    *,
    run_name: str | None = None,
    max_frames: int | None = None,
    validate: bool | str = True,
    force_reprocess: bool = True,
    cache_frames: CacheFramesMode | None = None,
    on_progress: ProgressCallback | None = None,
    pose_backend: str | None = None,
) -> PipelineResult:
    """
    Process a video from URL or local path.

    Args:
        cache_frames:
            ``True`` (default): single decode; save JPGs for GUI + pose JSON.
            ``False``: single decode; pose only (faster, no frame files).
            ``"legacy"``: extract all JPGs then pose (two-pass, slowest).
        pose_backend:
            ``mediapipe``, ``smpl``, or ``auto`` (defaults to ``config.POSE_BACKEND``).
    """
    progress = on_progress or _noop_progress
    config.ensure_output_dirs()
    release_all_captures()

    backend = pose_backend or config.POSE_BACKEND

    if cache_frames is None:
        cache_frames = config.DEFAULT_CACHE_FRAMES

    source = normalize_source(source)
    from_url = is_video_url(source)
    cache_key = content_cache_key(source)
    name = run_name or derive_run_name(source, unique_session=False)

    if validate is not False:
        progress("Validating video source…", 0.05)
        passed, _, msg = validate_source(source, mode=validate)
        if not passed:
            raise RuntimeError(msg)

    use_frame_dir = cache_frames is True or cache_frames == "legacy"
    frames_dir: Path | None = config.FRAMES_DIR / name if use_frame_dir else None
    poses_path = config.POSES_DIR / f"{name}_poses.json"

    if force_reprocess:
        clear_run_artifacts(frames_dir, poses_path)

    pose_result: PoseExtractionResult | None = None
    if cache_frames == "legacy":
        extraction, frame_paths, pose_result = _run_legacy_two_pass(
            source,
            frames_dir,
            poses_path,
            from_url=from_url,
            max_frames=max_frames,
            progress=progress,
            pose_backend=backend,
        )
    else:
        extraction, frame_paths, pose_result = _run_single_pass(
            source,
            frames_dir,
            poses_path,
            cache_frames=cache_frames is True,
            max_frames=max_frames,
            progress=progress,
            pose_backend=backend,
        )

    ff_hash = _first_frame_hash(frame_paths)
    resolution = pose_result.resolution
    smpl_path = _export_motion_artifacts(pose_result, name)

    logger.info(
        "Loaded new video: %s | run=%s | frames=%d | %dx%d | backend=%s%s",
        source[:80],
        name,
        extraction.frame_count,
        extraction.width,
        extraction.height,
        resolution.used,
        " (fallback)" if resolution.fallback else "",
    )

    progress("Done.", 1.0)
    logger.info("Pipeline complete (fresh): %s", poses_path)

    export_paths = _auto_export_opensim(poses_path, name)
    if export_paths:
        logger.info(
            "OpenSim export complete for run=%s (pose backend: %s)",
            name,
            resolution.used,
        )

    return PipelineResult(
        poses_path=poses_path,
        frames_dir=frames_dir,
        source=source,
        run_name=name,
        content_key=cache_key,
        frame_count=extraction.frame_count,
        fps=extraction.fps,
        width=extraction.width,
        height=extraction.height,
        first_frame_hash=ff_hash,
        pose_backend_requested=resolution.requested,
        pose_backend_used=resolution.used,
        pose_backend_fallback=resolution.fallback,
        pose_backend_fallback_reason=resolution.fallback_reason,
        smpl_motion_path=smpl_path,
    )
