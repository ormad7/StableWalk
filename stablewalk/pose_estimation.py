"""
Step 2: Pose estimation on extracted frames.

Uses MediaPipe Pose Landmarker (Tasks API) to detect body landmarks
and compute joint angles (14+ DOF).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.vision import pose_landmarker as pose_landmarker_lib
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib

from stablewalk.frame_enrichment import enrich_pose_sequence
from stablewalk.kinematics import compute_joint_angles
from stablewalk.pose_validation import has_full_body_pose, meets_min_gait_dof
from stablewalk.model_loader import get_pose_model_path
from stablewalk.models.pose_data import Keypoint, PoseFrame, PoseSequence

logger = logging.getLogger(__name__)

PoseLandmarker = pose_landmarker_lib.PoseLandmarker
PoseLandmarkerOptions = pose_landmarker_lib.PoseLandmarkerOptions
PoseLandmarksConnections = pose_landmarker_lib.PoseLandmarksConnections
PoseLandmark = pose_landmarker_lib.PoseLandmark

# MediaPipe landmark index → semantic name (33 landmarks)
LANDMARK_NAMES = [e.name.lower() for e in PoseLandmark]


class PoseEstimator:
    """Runs MediaPipe Pose Landmarker on images and builds pose sequences."""

    def __init__(
        self,
        model_variant: str = "lite",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        *,
        video_mode: bool = False,
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

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def process_image(
        self,
        image_path: str | Path,
        frame_index: int = 0,
    ) -> PoseFrame:
        """Estimate pose on a single image file."""
        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)

        if self._video_mode:
            self._timestamp_ms += 33  # ~30 fps between frames
            results = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            results = self._landmarker.detect(mp_image)

        if not results.pose_landmarks:
            return PoseFrame(
                frame_index=frame_index,
                image_path=str(image_path.resolve()),
                detected=False,
            )

        landmarks = results.pose_landmarks[0]
        keypoints = self._landmarks_to_keypoints(landmarks)
        keypoints = self._clamp_keypoints(keypoints)
        keypoints = self._add_mid_hip(keypoints)

        if not has_full_body_pose(keypoints):
            return PoseFrame(
                frame_index=frame_index,
                image_path=str(image_path.resolve()),
                detected=False,
                skipped=True,
            )

        joint_angles = compute_joint_angles(keypoints)
        if not meets_min_gait_dof(joint_angles):
            return PoseFrame(
                frame_index=frame_index,
                image_path=str(image_path.resolve()),
                keypoints=keypoints,
                joint_angles=joint_angles,
                detected=False,
                skipped=True,
            )

        return PoseFrame(
            frame_index=frame_index,
            image_path=str(image_path.resolve()),
            keypoints=keypoints,
            joint_angles=joint_angles,
            detected=True,
        )

    def reset_timeline(self) -> None:
        """Reset video-mode timestamps (required when processing a new video)."""
        self._timestamp_ms = 0

    def process_directory(
        self,
        frames_dir: str | Path,
        *,
        pattern: str = "frame_*.jpg",
        source_video: str = "",
        fps: float = 30.0,
        max_frames: int | None = None,
    ) -> PoseSequence:
        """Run pose estimation on all frames in a directory (frame 0 … N)."""
        self.reset_timeline()
        frames = sorted(Path(frames_dir).glob(pattern))
        if max_frames is not None:
            frames = frames[:max_frames]

        pose_frames: list[PoseFrame] = []
        for i, frame_path in enumerate(frames):
            pose_frame = self.process_image(frame_path, frame_index=i)
            pose_frames.append(pose_frame)
            if (i + 1) % 50 == 0:
                logger.info("Processed %d / %d frames", i + 1, len(frames))

        sequence = PoseSequence(source_video=source_video, fps=fps, frames=pose_frames)
        events, annotations = enrich_pose_sequence(sequence)
        sequence.gait_events_timeline = [
            {
                "frame": e.frame_index,
                "side": e.side,
                "event": e.event_type,
                "confidence": round(e.confidence, 3),
            }
            for e in events
        ]
        for frame in pose_frames:
            ann = annotations.get(frame.frame_index)
            if ann:
                frame.gait_phase = {"left": ann.phase_left, "right": ann.phase_right}
                frame.gait_events = list(ann.events)

        detected = sum(1 for f in pose_frames if f.detected)
        skipped = sum(1 for f in pose_frames if f.skipped)
        logger.info(
            "Pose estimation complete: %d / %d frames with full-body gait pose "
            "(%d skipped — no full body)",
            detected,
            len(pose_frames),
            skipped,
        )

        return sequence

    def save_sequence(self, sequence: PoseSequence, output_path: str | Path) -> Path:
        """Persist pose sequence as JSON (for Steps 3–7)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(sequence.to_dict(), f, indent=2)
        logger.info("Saved pose sequence → %s", output_path)
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
        """Draw full-body skeleton; optionally highlight gait DOF joints."""
        from stablewalk.gait_dof import GAIT_DOF_LANDMARKS
        image_path = Path(image_path)
        output_path = Path(output_path)

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return False

        h, w = image_bgr.shape[:2]
        name_to_kp = {kp.name: kp for kp in keypoints}
        index_to_kp = {
            i: name_to_kp.get(LANDMARK_NAMES[i])
            for i in range(len(LANDMARK_NAMES))
        }

        annotated = image_bgr.copy()

        for conn in PoseLandmarksConnections.POSE_LANDMARKS:
            start = index_to_kp.get(conn.start)
            end = index_to_kp.get(conn.end)
            if not start or not end:
                continue
            if start.visibility < min_visibility or end.visibility < min_visibility:
                continue
            pt1 = (int(start.x * w), int(start.y * h))
            pt2 = (int(end.x * w), int(end.y * h))
            cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

        for kp in keypoints:
            if kp.name == "mid_hip" or kp.visibility < min_visibility:
                continue
            cx, cy = int(kp.x * w), int(kp.y * h)
            if highlight_dof and kp.name in GAIT_DOF_LANDMARKS:
                cv2.circle(annotated, (cx, cy), 7, (0, 255, 255), -1)
                cv2.circle(annotated, (cx, cy), 9, (0, 200, 255), 2)
            else:
                cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        return True

    @staticmethod
    def _clamp_keypoints(keypoints: list[Keypoint]) -> list[Keypoint]:
        clamped: list[Keypoint] = []
        for kp in keypoints:
            clamped.append(
                Keypoint(
                    name=kp.name,
                    x=max(0.0, min(1.0, kp.x)),
                    y=max(0.0, min(1.0, kp.y)),
                    z=kp.z,
                    visibility=kp.visibility,
                )
            )
        return clamped

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
