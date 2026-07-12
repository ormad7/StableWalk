"""
Pose backend resolution and unified pose extraction entry point.

Supports ``mediapipe``, ``smpl``, and ``auto`` modes with explicit fallback logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from stablewalk import config
from stablewalk.models.pose_data import FrameExtractionResult, PoseFrame, PoseSequence
from stablewalk.pose.backends.base import BackendUnavailableError, HumanMotionBackend
from stablewalk.pose.backends.canonical import human_motion_sequence_to_pose_sequence
from stablewalk.pose.backends.mediapipe_backend import MediaPipePoseBackend
from stablewalk.pose.backends.smpl_backend import SMPLPoseBackend
from stablewalk.pose.backends.smpl_validation import validate_smpl_assets
from stablewalk.pose.backends.types import HumanMotionSequence
from stablewalk.pose.backends.unified_motion import (
    ScaleInformation,
    UnifiedHumanMotion,
    human_motion_sequence_to_unified,
)
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.estimation import PoseEstimator

logger = logging.getLogger(__name__)

PoseBackendMode = Literal["mediapipe", "smpl", "auto"]


@dataclass
class BackendResolution:
    """Records which backend ran and whether fallback occurred."""

    requested: str
    used: str
    fallback: bool = False
    fallback_reason: str | None = None
    provider_name: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "used": self.used,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "provider_name": self.provider_name,
            "warnings": list(self.warnings),
        }


@dataclass
class PoseExtractionResult:
    """Outputs from a full pose extraction pass."""

    sequence: PoseSequence
    unified_motion: UnifiedHumanMotion
    resolution: BackendResolution
    human_motion_sequence: HumanMotionSequence | None = None


def normalize_backend_mode(name: str | None) -> str:
    """Normalize user/env backend string."""
    key = (name or config.POSE_BACKEND or "mediapipe").strip().lower()
    if key in ("romp", "hybrik", "wham"):
        return "smpl"
    return key


def resolve_pose_backend(
    backend_name: str | None = None,
    *,
    allow_fallback: bool | None = None,
) -> tuple[HumanMotionBackend, BackendResolution]:
    """
    Instantiate the requested pose backend.

    ``auto`` tries SMPL first, then MediaPipe with a logged warning.
    """
    requested = normalize_backend_mode(backend_name)
    fallback_ok = (
        config.POSE_BACKEND_ALLOW_FALLBACK
        if allow_fallback is None
        else allow_fallback
    )

    if requested == "auto":
        if SMPLPoseBackend.is_available():
            backend = SMPLPoseBackend()
            return backend, BackendResolution(
                requested="auto",
                used="smpl",
                provider_name=getattr(backend, "provider_name", "romp"),
            )
        validation = validate_smpl_assets()
        reason = validation.summary()
        warning = (
            f"SMPL backend unavailable ({reason}) — falling back to MediaPipe. "
            "See docs/SMPL_BACKEND_SETUP.md to enable mesh-based recovery."
        )
        logger.warning(warning)
        if not MediaPipePoseBackend.is_available():
            raise BackendUnavailableError("mediapipe", MediaPipePoseBackend.availability_reason())
        return MediaPipePoseBackend(), BackendResolution(
            requested="auto",
            used="mediapipe",
            fallback=True,
            fallback_reason=reason,
            warnings=[warning],
        )

    if requested == "smpl":
        if not SMPLPoseBackend.is_available():
            validation = validate_smpl_assets()
            msg = validation.summary()
            if fallback_ok:
                warning = f"SMPL unavailable ({msg}) — falling back to MediaPipe."
                logger.warning(warning)
                return MediaPipePoseBackend(), BackendResolution(
                    requested="smpl",
                    used="mediapipe",
                    fallback=True,
                    fallback_reason=msg,
                    warnings=[warning],
                )
            raise BackendUnavailableError("smpl", msg)
        backend = SMPLPoseBackend()
        return backend, BackendResolution(
            requested="smpl",
            used="smpl",
            provider_name=getattr(backend, "provider_name", "romp"),
        )

    if requested == "mediapipe":
        if not MediaPipePoseBackend.is_available():
            raise BackendUnavailableError("mediapipe", MediaPipePoseBackend.availability_reason())
        return MediaPipePoseBackend(), BackendResolution(requested="mediapipe", used="mediapipe")

    raise ValueError(
        f"Unknown pose backend '{requested}'. Supported: mediapipe, smpl, auto"
    )


def _human_sequence_from_backend_video(
    backend: HumanMotionBackend,
    video_path: str | Path,
    *,
    max_frames: int | None = None,
) -> HumanMotionSequence:
    return backend.process_video(str(video_path), max_frames=max_frames)


def _human_sequence_from_frames_dir(
    backend: HumanMotionBackend,
    frames_dir: Path,
    *,
    source_video: str,
    fps: float,
    max_frames: int | None = None,
) -> HumanMotionSequence:
    import cv2

    backend.reset()
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        frame_paths = sorted(frames_dir.glob("*.jpg"))
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    frames: list = []
    for i, fp in enumerate(frame_paths):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        ts = i / max(fps, 1e-6)
        frames.append(
            backend.process_frame(bgr, frame_index=i, timestamp_s=ts)
        )

    return HumanMotionSequence(
        frames=frames,
        fps=fps,
        source_video=source_video,
        backend_name=backend.name,
        coordinate_system=backend.coordinate_system,
    )


def extract_pose_from_video(
    video_path: str | Path,
    *,
    backend_name: str | None = None,
    max_frames: int | None = None,
    enrich: bool = True,
    allow_fallback: bool | None = None,
) -> PoseExtractionResult:
    """Run configured backend on a video file."""
    backend, resolution = resolve_pose_backend(backend_name, allow_fallback=allow_fallback)
    try:
        if resolution.used == "mediapipe" and not resolution.fallback:
            with PoseEstimator(
                model_variant=config.DEFAULT_POSE_MODEL_VARIANT,
                min_detection_confidence=config.DEFAULT_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.DEFAULT_MIN_TRACKING_CONFIDENCE,
                video_mode=config.DEFAULT_PIPELINE_VIDEO_MODE,
            ) as estimator:
                sequence = estimator.process_video(
                    video_path,
                    max_frames=max_frames,
                    enrich_gait=False,
                )
            hm_sequence = None
            unified = _unified_from_pose_sequence(sequence, resolution)
        else:
            hm_sequence = _human_sequence_from_backend_video(
                backend, video_path, max_frames=max_frames
            )
            sequence = human_motion_sequence_to_pose_sequence(hm_sequence)
            scale = ScaleInformation(
                notes=f"Backend {resolution.used} — mesh/landmark coordinates in meters."
            )
            unified = human_motion_sequence_to_unified(
                hm_sequence,
                scale=scale,
                provider_name=resolution.provider_name,
            )

        if enrich:
            enrich_pose_sequence(sequence)
        unified.metadata["backend_resolution"] = resolution.to_dict()
        return PoseExtractionResult(
            sequence=sequence,
            unified_motion=unified,
            resolution=resolution,
            human_motion_sequence=hm_sequence,
        )
    finally:
        if hasattr(backend, "close"):
            backend.close()


def extract_pose_from_frames_dir(
    frames_dir: Path,
    *,
    source_video: str,
    fps: float,
    backend_name: str | None = None,
    max_frames: int | None = None,
    enrich: bool = True,
    allow_fallback: bool | None = None,
) -> PoseExtractionResult:
    """Run configured backend on extracted JPG frames."""
    backend, resolution = resolve_pose_backend(backend_name, allow_fallback=allow_fallback)
    try:
        if resolution.used == "mediapipe" and not resolution.fallback:
            with PoseEstimator(
                model_variant=config.DEFAULT_POSE_MODEL_VARIANT,
                min_detection_confidence=config.DEFAULT_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.DEFAULT_MIN_TRACKING_CONFIDENCE,
                video_mode=not config.DEFAULT_POSE_IMAGE_MODE,
            ) as estimator:
                sequence = estimator.process_directory(
                    frames_dir,
                    source_video=source_video,
                    fps=fps,
                    max_frames=max_frames,
                )
            hm_sequence = None
            unified = _unified_from_pose_sequence(sequence, resolution)
        else:
            hm_sequence = _human_sequence_from_frames_dir(
                backend,
                frames_dir,
                source_video=source_video,
                fps=fps,
                max_frames=max_frames,
            )
            sequence = human_motion_sequence_to_pose_sequence(hm_sequence)
            unified = human_motion_sequence_to_unified(
                hm_sequence,
                scale=ScaleInformation(notes=f"Backend {resolution.used}"),
                provider_name=resolution.provider_name,
            )

        if enrich:
            enrich_pose_sequence(sequence)
        unified.metadata["backend_resolution"] = resolution.to_dict()
        return PoseExtractionResult(
            sequence=sequence,
            unified_motion=unified,
            resolution=resolution,
            human_motion_sequence=hm_sequence,
        )
    finally:
        if hasattr(backend, "close"):
            backend.close()


def extract_pose_video_with_frame_cache(
    video_path: str | Path,
    frames_dir: Path | None,
    *,
    backend_name: str | None = None,
    max_frames: int | None = None,
    jpeg_quality: int = 95,
    enrich: bool = True,
    allow_fallback: bool | None = None,
) -> tuple[PoseExtractionResult, FrameExtractionResult]:
    """
    Single-pass decode with optional JPG cache (GUI path).

    MediaPipe uses native ``process_video_with_frame_cache`` for parity.
    SMPL iterates decoded frames through the SMPL backend.
    """
    import cv2

    from stablewalk.pose.video import VideoReader

    backend, resolution = resolve_pose_backend(backend_name, allow_fallback=allow_fallback)

    if resolution.used == "mediapipe" and not resolution.fallback:
        with PoseEstimator(
            model_variant=config.DEFAULT_POSE_MODEL_VARIANT,
            min_detection_confidence=config.DEFAULT_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.DEFAULT_MIN_TRACKING_CONFIDENCE,
            video_mode=config.DEFAULT_PIPELINE_VIDEO_MODE,
        ) as estimator:
            sequence, extraction = estimator.process_video_with_frame_cache(
                video_path,
                frames_dir,
                max_frames=max_frames,
                jpeg_quality=jpeg_quality,
                enrich_gait=False,
            )
        if enrich:
            enrich_pose_sequence(sequence)
        unified = _unified_from_pose_sequence(sequence, resolution)
        unified.metadata["backend_resolution"] = resolution.to_dict()
        result = PoseExtractionResult(
            sequence=sequence,
            unified_motion=unified,
            resolution=resolution,
        )
        return result, extraction

    # SMPL / fallback non-native path
    video_path = str(video_path)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    frame_paths: list[str] = []
    hm_frames: list = []

    if frames_dir is not None:
        frames_dir = Path(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old in frames_dir.glob("frame_*.jpg"):
            old.unlink()

    backend.reset()
    with VideoReader(video_path) as reader:
        meta = reader.metadata
        for vf in reader.iter_frames(max_frames=max_frames):
            image_path = video_path
            if frames_dir is not None:
                out_path = frames_dir / f"frame_{vf.index:06d}.jpg"
                if not cv2.imwrite(str(out_path), vf.bgr, encode_params):
                    raise RuntimeError(f"Failed to write frame: {out_path}")
                image_path = str(out_path.resolve())
                frame_paths.append(image_path)

            hm_frames.append(
                backend.process_frame(
                    vf.bgr,
                    frame_index=vf.index,
                    timestamp_s=vf.timestamp_s,
                )
            )

        hm_sequence = HumanMotionSequence(
            frames=hm_frames,
            fps=meta.fps,
            source_video=video_path,
            backend_name=backend.name,
            coordinate_system=backend.coordinate_system,
        )
        sequence = human_motion_sequence_to_pose_sequence(hm_sequence)
        if enrich:
            enrich_pose_sequence(sequence)

        unified = human_motion_sequence_to_unified(
            hm_sequence,
            scale=ScaleInformation(notes=f"Backend {resolution.used}"),
            provider_name=resolution.provider_name,
        )
        unified.metadata["backend_resolution"] = resolution.to_dict()

        extraction = FrameExtractionResult(
            video_path=video_path,
            fps=meta.fps,
            frame_count=len(hm_frames),
            width=meta.width,
            height=meta.height,
            frame_paths=frame_paths,
        )

        result = PoseExtractionResult(
            sequence=sequence,
            unified_motion=unified,
            resolution=resolution,
            human_motion_sequence=hm_sequence,
        )
        if hasattr(backend, "close"):
            backend.close()
        return result, extraction


def _unified_from_pose_sequence(
    sequence: PoseSequence,
    resolution: BackendResolution,
) -> UnifiedHumanMotion:
    """Build unified motion from legacy MediaPipe PoseSequence."""
    from stablewalk.models.joint_registry import ROOT_JOINT_ID
    from stablewalk.pose.backends.mediapipe_backend import MEDIAPIPE_COORDINATE_SYSTEM
    from stablewalk.pose.backends.unified_motion import UnifiedMotionFrame
    import numpy as np

    frames: list[UnifiedMotionFrame] = []
    timestamps: list[float] = []

    for pf in sequence.frames:
        ts = pf.timestamp_s if pf.timestamp_s > 0 else pf.frame_index / max(sequence.fps, 1e-6)
        timestamps.append(ts)
        positions: dict[str, tuple[float, float, float]] = {}
        conf_map: dict[str, float] = {}
        for kp in pf.keypoints:
            from stablewalk.pose.backends.canonical import map_landmark_name_to_canonical

            jid = map_landmark_name_to_canonical(kp.name)
            positions[jid] = (kp.x, kp.y, kp.z)
            conf_map[jid] = kp.visibility

        pelvis = positions.get(ROOT_JOINT_ID)
        if pelvis is None:
            lh = positions.get("left_hip")
            rh = positions.get("right_hip")
            if lh and rh:
                pelvis = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2, (lh[2] + rh[2]) / 2)

        conf_vals = list(conf_map.values())
        pose_conf = float(sum(conf_vals) / len(conf_vals)) if conf_vals else 0.0

        frames.append(
            UnifiedMotionFrame(
                frame_index=pf.frame_index,
                timestamp_s=ts,
                root_position=pelvis,
                joint_positions_3d=positions,
                pose_confidence=pose_conf if pf.detected else 0.0,
                detected=pf.detected,
                metadata={"source": "mediapipe_pose_sequence"},
            )
        )

    return UnifiedHumanMotion(
        fps=float(sequence.fps),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        frames=frames,
        source_backend=resolution.used,
        coordinate_system=MEDIAPIPE_COORDINATE_SYSTEM,
        scale_information=ScaleInformation(
            scale_to_meters=1.0,
            notes="MediaPipe normalized image coordinates — not metric SMPL mesh.",
        ),
        source_video=sequence.source_video,
        provider_name="mediapipe",
        metadata={"backend_resolution": resolution.to_dict()},
    )


def get_backend_diagnostics() -> list[dict[str, Any]]:
    """Extended diagnostics including smpl and auto modes."""
    from stablewalk.pose.backends.registry import BACKEND_REGISTRY

    base = []
    for name in ("mediapipe", "smpl"):
        cls = BACKEND_REGISTRY.get(name)
        if cls is None:
            continue
        base.append(
            {
                "name": name,
                "display_name": cls.display_name,
                "available": cls.is_available(),
                "reason": cls.availability_reason(),
                "dependencies": cls.dependency_summary(),
            }
        )
    smpl_val = validate_smpl_assets().to_dict()
    base.append(
        {
            "name": "auto",
            "display_name": "Auto (SMPL → MediaPipe fallback)",
            "available": True,
            "reason": (
                "Uses SMPL when ready, else MediaPipe"
                if smpl_val.get("ready")
                else f"Will fall back to MediaPipe: {smpl_val.get('summary')}"
            ),
            "dependencies": ["smpl or mediapipe"],
            "smpl_validation": smpl_val,
        }
    )
    return base


__all__ = [
    "BackendResolution",
    "PoseExtractionResult",
    "PoseBackendMode",
    "normalize_backend_mode",
    "resolve_pose_backend",
    "extract_pose_from_video",
    "extract_pose_from_frames_dir",
    "extract_pose_video_with_frame_cache",
    "get_backend_diagnostics",
]
