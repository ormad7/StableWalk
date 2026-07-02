"""
Pose estimation from walking video or extracted frames.

Uses **MediaPipe Pose Landmarker** (Tasks API) for 2D body keypoints.
Video decoding is delegated to ``stablewalk.pose.video`` — this module
never opens raw video files except through ``VideoReader``.

Output per frame:
  - joint keypoints (x, y) in normalized image coordinates [0, 1]
  - visibility as confidence score [0, 1]
  - timestamp_s / timestamp_ms from video FPS

Exports:
  - JSON (``save_sequence_json``)
  - NumPy archive (``save_sequence_numpy``)
  - Annotated MP4 with skeleton overlay (``write_overlay_video``)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.vision import pose_landmarker as pose_landmarker_lib
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib

from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.kinematics import compute_joint_angles
from stablewalk.model_loader import get_pose_model_path
from stablewalk.models.pose_data import FrameExtractionResult, Keypoint, PoseFrame, PoseSequence
from stablewalk.pose.skeleton import draw_skeleton_on_bgr
from stablewalk.pose.validation import has_full_body_pose, meets_min_gait_dof
from stablewalk.pose.video import VideoFrame, VideoReader

logger = logging.getLogger(__name__)

PoseLandmarker = pose_landmarker_lib.PoseLandmarker
PoseLandmarkerOptions = pose_landmarker_lib.PoseLandmarkerOptions
PoseLandmarksConnections = pose_landmarker_lib.PoseLandmarksConnections
PoseLandmark = pose_landmarker_lib.PoseLandmark

LANDMARK_NAMES = [e.name.lower() for e in PoseLandmark]


@dataclass
class VideoPoseResult:
    """Outputs from processing a single walking video."""

    sequence: PoseSequence
    poses_json_path: Path | None = None
    poses_numpy_path: Path | None = None
    overlay_video_path: Path | None = None


class PoseEstimator:
    """
    Extract 2D human pose from images or video using MediaPipe.

    Parameters
    ----------
    model_variant:
        ``"lite"`` (faster) or ``"full"`` (more accurate). Requires model
        files under ``models/`` (see ``model_loader``).
    min_detection_confidence:
        Minimum score to accept a pose detection.
    video_mode:
        When True, uses MediaPipe VIDEO running mode (temporal smoothing).
        Required for ``process_video``.

    Example
    -------
    >>> with PoseEstimator(video_mode=True) as estimator:
    ...     result = estimator.process_video("data/input/my_walk.mp4")
    ...     estimator.save_sequence_json(result.sequence, "out/poses.json")
    ...     estimator.write_overlay_video(
    ...         result.sequence, "data/input/my_walk.mp4", "out/overlay.mp4"
    ...     )
    """

    def __init__(
        self,
        model_variant: str = "lite",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        *,
        video_mode: bool = False,
        require_full_body: bool = True,
    ) -> None:
        model_path = get_pose_model_path(model_variant)
        running_mode = (
            running_mode_lib.VisionTaskRunningMode.VIDEO
            if video_mode
            else running_mode_lib.VisionTaskRunningMode.IMAGE
        )

        options = PoseLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._video_mode = video_mode
        self._timestamp_ms = 0
        self._min_confidence = min_detection_confidence
        self._require_full_body = require_full_body

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def reset_timeline(self) -> None:
        """Reset video-mode timestamps (call before a new video)."""
        self._timestamp_ms = 0

    def process_frame(
        self,
        bgr: np.ndarray,
        *,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
        image_path: str = "",
    ) -> PoseFrame:
        """
        Run pose estimation on one BGR image (OpenCV format).

        Returns a ``PoseFrame`` with keypoints (x, y), visibility, and timestamps.
        """
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)

        if self._video_mode:
            self._timestamp_ms = int(round(timestamp_s * 1000))
            results = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            results = self._landmarker.detect(mp_image)

        ts_ms = int(round(timestamp_s * 1000))
        base = PoseFrame(
            frame_index=frame_index,
            image_path=image_path,
            timestamp_s=timestamp_s,
            timestamp_ms=ts_ms,
            detected=False,
        )

        if not results.pose_landmarks:
            return base

        keypoints = self._landmarks_to_keypoints(results.pose_landmarks[0])
        keypoints = self._clamp_keypoints(keypoints)
        keypoints = self._add_mid_hip(keypoints)

        if self._require_full_body and not has_full_body_pose(keypoints):
            return PoseFrame(
                frame_index=frame_index,
                image_path=image_path,
                timestamp_s=timestamp_s,
                timestamp_ms=ts_ms,
                keypoints=keypoints,
                detected=False,
                skipped=True,
            )

        joint_angles = compute_joint_angles(keypoints)
        if self._require_full_body and not meets_min_gait_dof(joint_angles):
            return PoseFrame(
                frame_index=frame_index,
                image_path=image_path,
                timestamp_s=timestamp_s,
                timestamp_ms=ts_ms,
                keypoints=keypoints,
                joint_angles=joint_angles,
                detected=False,
                skipped=True,
            )

        return PoseFrame(
            frame_index=frame_index,
            image_path=image_path,
            timestamp_s=timestamp_s,
            timestamp_ms=ts_ms,
            keypoints=keypoints,
            joint_angles=joint_angles,
            detected=True,
        )

    def process_image(
        self,
        image_path: str | Path,
        frame_index: int = 0,
        *,
        timestamp_s: float | None = None,
    ) -> PoseFrame:
        """Estimate pose on a single image file on disk."""
        image_path = Path(image_path)
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        ts = timestamp_s if timestamp_s is not None else frame_index / 30.0
        return self.process_frame(
            bgr,
            frame_index=frame_index,
            timestamp_s=ts,
            image_path=str(image_path.resolve()),
        )

    def process_video(
        self,
        video_path: str | Path,
        *,
        max_frames: int | None = None,
        enrich_gait: bool = True,
    ) -> PoseSequence:
        """
        Process a walking video file end-to-end (no intermediate frame files).

        Args:
            video_path: Local ``.mp4`` / ``.avi`` etc., or streamable URL.
            max_frames: Optional cap for quick tests.
            enrich_gait: Attach velocities, 3D skeleton, gait phases when True.

        Returns:
            ``PoseSequence`` with per-frame keypoints, confidence, and timestamps.
        """
        if not self._video_mode:
            logger.warning(
                "process_video works best with video_mode=True; "
                "enable it in PoseEstimator(..., video_mode=True)."
            )

        video_path = str(video_path)
        self.reset_timeline()
        pose_frames: list[PoseFrame] = []

        with VideoReader(video_path) as reader:
            meta = reader.metadata
            for vf in reader.iter_frames(max_frames=max_frames):
                pf = self.process_frame(
                    vf.bgr,
                    frame_index=vf.index,
                    timestamp_s=vf.timestamp_s,
                    image_path=video_path,
                )
                pose_frames.append(pf)
                if (vf.index + 1) % 50 == 0:
                    logger.info("Processed %d frames...", vf.index + 1)

        sequence = PoseSequence(
            source_video=video_path,
            fps=meta.fps,
            frames=pose_frames,
        )
        if enrich_gait:
            self._attach_gait_labels(sequence)
        self._log_summary(pose_frames)
        return sequence

    def process_video_with_frame_cache(
        self,
        video_path: str | Path,
        frames_dir: str | Path | None,
        *,
        max_frames: int | None = None,
        jpeg_quality: int = 95,
        enrich_gait: bool = True,
        frame_prefix: str = "frame",
    ) -> tuple[PoseSequence, FrameExtractionResult]:
        """
        Single-pass decode: run pose and optionally write JPG frames for the GUI.

        Avoids the legacy path of extract-all-JPGs then ``cv2.imread`` each file again.
        """
        video_path = str(video_path)
        self.reset_timeline()
        pose_frames: list[PoseFrame] = []
        frame_paths: list[str] = []
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

        if frames_dir is not None:
            frames_dir = Path(frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            for old in frames_dir.glob(f"{frame_prefix}_*.jpg"):
                old.unlink()

        with VideoReader(video_path) as reader:
            meta = reader.metadata
            for vf in reader.iter_frames(max_frames=max_frames):
                image_path = video_path
                if frames_dir is not None:
                    out_path = frames_dir / f"{frame_prefix}_{vf.index:06d}.jpg"
                    if not cv2.imwrite(str(out_path), vf.bgr, encode_params):
                        raise RuntimeError(f"Failed to write frame: {out_path}")
                    image_path = str(out_path.resolve())
                    frame_paths.append(image_path)

                pf = self.process_frame(
                    vf.bgr,
                    frame_index=vf.index,
                    timestamp_s=vf.timestamp_s,
                    image_path=image_path,
                )
                pose_frames.append(pf)
                if (vf.index + 1) % 50 == 0:
                    logger.info("Processed %d frames (single pass)...", vf.index + 1)

        sequence = PoseSequence(
            source_video=video_path,
            fps=meta.fps,
            frames=pose_frames,
        )
        if enrich_gait:
            self._attach_gait_labels(sequence)
        self._log_summary(pose_frames)

        extraction = FrameExtractionResult(
            video_path=video_path,
            output_dir=str(frames_dir.resolve()) if frames_dir else "",
            frame_count=len(pose_frames),
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
            frame_paths=frame_paths,
        )
        return sequence, extraction

    def process_directory(
        self,
        frames_dir: str | Path,
        *,
        pattern: str = "frame_*.jpg",
        source_video: str = "",
        fps: float = 30.0,
        max_frames: int | None = None,
        enrich_gait: bool = True,
    ) -> PoseSequence:
        """
        Run pose estimation on JPG frames extracted by ``VideoProcessor``.

        Timestamps are ``frame_index / fps``.
        """
        self.reset_timeline()
        frames = sorted(Path(frames_dir).glob(pattern))
        if max_frames is not None:
            frames = frames[:max_frames]

        pose_frames: list[PoseFrame] = []
        for i, frame_path in enumerate(frames):
            ts = i / max(fps, 1e-6)
            pf = self.process_image(frame_path, frame_index=i, timestamp_s=ts)
            pose_frames.append(pf)
            if (i + 1) % 50 == 0:
                logger.info("Processed %d / %d frames", i + 1, len(frames))

        sequence = PoseSequence(source_video=source_video, fps=fps, frames=pose_frames)
        if enrich_gait:
            self._attach_gait_labels(sequence)
        self._log_summary(pose_frames)
        return sequence

    # ------------------------------------------------------------------
    # Persistence & visualization
    # ------------------------------------------------------------------

    def save_sequence_json(
        self,
        sequence: PoseSequence,
        output_path: str | Path,
        *,
        pose_only: bool = False,
    ) -> Path:
        """
        Save pose sequence as JSON.

        Args:
            pose_only: If True, export only keypoints + timestamps (no gait extras).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            sequence.to_pose_tracking_dict()
            if pose_only
            else sequence.to_dict()
        )
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved pose JSON -> %s", output_path)
        return output_path

    def save_sequence(self, sequence: PoseSequence, output_path: str | Path) -> Path:
        """Alias for ``save_sequence_json`` (full gait export)."""
        return self.save_sequence_json(sequence, output_path, pose_only=False)

    def save_sequence_numpy(
        self,
        sequence: PoseSequence,
        output_path: str | Path,
    ) -> Path:
        """
        Save pose arrays to ``.npz``.

        Arrays:
          - keypoints_xy: (T, J, 2)
          - confidence: (T, J)
          - timestamps_s: (T,)
          - detected: (T,)
          - joint_names, fps, source_video (metadata)
        """
        output_path = Path(output_path)
        if output_path.suffix != ".npz":
            output_path = output_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = sequence.to_numpy_dict()
        np.savez_compressed(output_path, **arrays)
        logger.info("Saved pose NumPy -> %s", output_path)
        return output_path

    @staticmethod
    def draw_skeleton_on_frame(
        bgr: np.ndarray,
        keypoints: list[Keypoint],
        *,
        min_visibility: float = 0.5,
        highlight_dof: bool = False,
        foot_contact: dict[str, bool] | None = None,
    ) -> np.ndarray:
        """Draw skeleton on a BGR frame (delegates to ``skeleton_render``)."""
        return draw_skeleton_on_bgr(
            bgr,
            keypoints,
            min_visibility=min_visibility,
            highlight_dof=highlight_dof,
            foot_contact=foot_contact,
        )

    def write_overlay_video(
        self,
        sequence: PoseSequence,
        source_video: str | Path,
        output_path: str | Path,
        *,
        max_frames: int | None = None,
        draw_timestamp: bool = True,
    ) -> Path:
        """
        Render skeleton overlay on the source video and save as MP4.

        Frames without detection are written unchanged.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pose_by_index = {f.frame_index: f for f in sequence.frames}

        with VideoReader(source_video) as reader:
            meta = reader.metadata
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                meta.fps,
                (meta.width, meta.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {output_path}")

            try:
                for vf in reader.iter_frames(max_frames=max_frames):
                    pf = pose_by_index.get(vf.index)
                    out = vf.bgr
                    if pf and pf.keypoints and pf.detected:
                        out = self.draw_skeleton_on_frame(
                            vf.bgr,
                            pf.keypoints,
                            foot_contact=pf.foot_contact or None,
                        )
                    if draw_timestamp and pf:
                        label = f"t={pf.timestamp_s:.2f}s  conf={self._mean_confidence(pf):.2f}"
                        cv2.putText(
                            out,
                            label,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                    writer.write(out)
            finally:
                writer.release()

        logger.info("Saved overlay video -> %s", output_path)
        return output_path

    @staticmethod
    def draw_keypoints_overlay(
        image_path: str | Path,
        keypoints: list[Keypoint],
        output_path: str | Path,
        *,
        min_visibility: float = 0.5,
        highlight_dof: bool = True,
    ) -> bool:
        """Draw skeleton on a still image and save to disk."""
        from stablewalk.pose.dof import GAIT_DOF_LANDMARKS

        image_path = Path(image_path)
        output_path = Path(output_path)
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return False

        annotated = PoseEstimator.draw_skeleton_on_frame(
            bgr, keypoints, min_visibility=min_visibility
        )
        if highlight_dof:
            h, w = annotated.shape[:2]
            for kp in keypoints:
                if kp.name in GAIT_DOF_LANDMARKS and kp.visibility >= min_visibility:
                    cx, cy = int(kp.x * w), int(kp.y * h)
                    cv2.circle(annotated, (cx, cy), 7, (0, 255, 255), -1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        return True

    def process_video_and_export(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        *,
        run_name: str | None = None,
        max_frames: int | None = None,
        save_overlay: bool = True,
        save_numpy: bool = True,
        pose_only_json: bool = False,
    ) -> VideoPoseResult:
        """
        Convenience: process video, then write JSON (+ optional NPZ and overlay MP4).
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = run_name or video_path.stem

        sequence = self.process_video(video_path, max_frames=max_frames)
        json_path = self.save_sequence_json(
            sequence,
            output_dir / f"{stem}_poses.json",
            pose_only=pose_only_json,
        )
        np_path = None
        if save_numpy:
            np_path = self.save_sequence_numpy(sequence, output_dir / f"{stem}_poses.npz")
        overlay_path = None
        if save_overlay:
            overlay_path = self.write_overlay_video(
                sequence,
                video_path,
                output_dir / f"{stem}_overlay.mp4",
                max_frames=max_frames,
            )
        return VideoPoseResult(
            sequence=sequence,
            poses_json_path=json_path,
            poses_numpy_path=np_path,
            overlay_video_path=overlay_path,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _attach_gait_labels(self, sequence: PoseSequence) -> None:
        from stablewalk.pose.enrichment import enrich_pose_sequence

        events, _annotations = enrich_pose_sequence(sequence)
        sequence.gait_events_timeline = [
            {
                "frame": e.frame_index,
                "side": e.side,
                "event": e.event_type,
                "confidence": round(e.confidence, 3),
            }
            for e in events
        ]

    @staticmethod
    def _log_summary(pose_frames: list[PoseFrame]) -> None:
        detected = sum(1 for f in pose_frames if f.detected)
        skipped = sum(1 for f in pose_frames if f.skipped)
        logger.info(
            "Pose estimation complete: %d / %d frames with pose (%d skipped)",
            detected,
            len(pose_frames),
            skipped,
        )

    @staticmethod
    def _mean_confidence(frame: PoseFrame) -> float:
        if not frame.keypoints:
            return 0.0
        return sum(kp.visibility for kp in frame.keypoints) / len(frame.keypoints)

    @staticmethod
    def _clamp_keypoints(keypoints: list[Keypoint]) -> list[Keypoint]:
        return [
            Keypoint(
                name=kp.name,
                x=max(0.0, min(1.0, kp.x)),
                y=max(0.0, min(1.0, kp.y)),
                z=kp.z,
                visibility=kp.visibility,
            )
            for kp in keypoints
        ]

    @staticmethod
    def _landmarks_to_keypoints(landmarks: list) -> list[Keypoint]:
        keypoints: list[Keypoint] = []
        for i, lm in enumerate(landmarks):
            name = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"landmark_{i}"
            keypoints.append(
                Keypoint(
                    name=name,
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z or 0.0),
                    visibility=float(lm.visibility if lm.visibility is not None else 1.0),
                )
            )
        return keypoints

    @staticmethod
    def _add_mid_hip(keypoints: list[Keypoint]) -> list[Keypoint]:
        by_name = {kp.name: kp for kp in keypoints}
        left, right = by_name.get("left_hip"), by_name.get("right_hip")
        if left is None or right is None:
            return keypoints
        mid = Keypoint(
            name="mid_hip",
            x=(left.x + right.x) / 2,
            y=(left.y + right.y) / 2,
            z=(left.z + right.z) / 2,
            visibility=min(left.visibility, right.visibility),
        )
        return keypoints + [mid]
