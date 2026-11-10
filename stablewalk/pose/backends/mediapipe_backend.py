"""
MediaPipe Pose Landmarker backend — StableWalk default.

Wraps the existing Tasks API pipeline without changing default behaviour.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from stablewalk.model_loader import get_pose_model_path
from stablewalk.models.pose_data import Keypoint
from stablewalk.pose.backends.base import BackendUnavailableError, HumanMotionBackend
from stablewalk.pose.backends.canonical import keypoints_to_canonical_frame
from stablewalk.pose.backends.types import CoordinateSystemMetadata, HumanMotionFrame
from stablewalk.pose.validation import has_full_body_pose, meets_min_gait_dof
from stablewalk.pose.kinematics import compute_joint_angles


def _mediapipe_runtime():
    """Import MediaPipe Tasks APIs only when a backend instance is created."""
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python.core import base_options as base_options_lib
    from mediapipe.tasks.python.vision import pose_landmarker as pose_landmarker_lib
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib

    return Image, ImageFormat, base_options_lib, pose_landmarker_lib, running_mode_lib

logger = logging.getLogger(__name__)

MEDIAPIPE_COORDINATE_SYSTEM = CoordinateSystemMetadata(
    name="mediapipe_normalized_image",
    units="normalized",
    origin_description="Image top-left; x right, y down, z toward camera",
    x_axis="+x right (0–1)",
    y_axis="+y down (0–1)",
    z_axis="+z relative depth (MediaPipe)",
    notes="StableWalk canonical adapter derives pelvis-centered skeleton from landmarks.",
)


class MediaPipePoseBackend(HumanMotionBackend):
    """Production pose backend using MediaPipe Pose Landmarker."""

    name = "mediapipe"
    display_name = "MediaPipe Pose Landmarker"
    description = "Default StableWalk 2D/3D landmark tracker (CPU-friendly, no PyTorch)."

    def __init__(
        self,
        *,
        model_variant: str = "lite",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        video_mode: bool = True,
        require_full_body: bool = False,
    ) -> None:
        if not self.is_available():
            raise BackendUnavailableError(self.name, self.availability_reason())

        (
            _Image,
            _ImageFormat,
            base_options_lib,
            pose_landmarker_lib,
            running_mode_lib,
        ) = _mediapipe_runtime()
        self._mp_image_cls = _Image
        self._mp_image_format = _ImageFormat

        model_path = get_pose_model_path(model_variant)
        running_mode = (
            running_mode_lib.VisionTaskRunningMode.VIDEO
            if video_mode
            else running_mode_lib.VisionTaskRunningMode.IMAGE
        )
        options = pose_landmarker_lib.PoseLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_poses=2,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = pose_landmarker_lib.PoseLandmarker.create_from_options(options)
        self._video_mode = video_mode
        self._require_full_body = require_full_body
        self._timestamp_ms = 0
        self._subject_lock = None
        self._subject_selection = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mediapipe  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def availability_reason(cls) -> str:
        if cls.is_available():
            return "MediaPipe is installed"
        return "MediaPipe package is not installed"

    @classmethod
    def dependency_summary(cls) -> list[str]:
        return ["mediapipe", "opencv-python", "numpy"]

    @property
    def coordinate_system(self) -> CoordinateSystemMetadata:
        return MEDIAPIPE_COORDINATE_SYSTEM

    def close(self) -> None:
        self._landmarker.close()

    def reset(self) -> None:
        self._timestamp_ms = 0
        self._subject_lock = None

    def process_video(
        self,
        video_path: str,
        *,
        max_frames: int | None = None,
    ):
        from stablewalk.pose.subject_lock import selection_for_source

        if self._subject_selection is None:
            self._subject_selection = selection_for_source(video_path)
        return super().process_video(video_path, max_frames=max_frames)

    def process_frame(
        self,
        bgr: np.ndarray,
        *,
        frame_index: int,
        timestamp_s: float,
    ) -> HumanMotionFrame:
        from stablewalk.pose.estimation import PoseEstimator

        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp_image_cls(
            image_format=self._mp_image_format.SRGB, data=image_rgb
        )

        if self._video_mode:
            self._timestamp_ms = int(round(timestamp_s * 1000))
            results = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            results = self._landmarker.detect(mp_image)

        if not results.pose_landmarks:
            return HumanMotionFrame(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                joint_positions_3d={},
                landmark_confidence={},
                backend_name=self.name,
                coordinate_system=self.coordinate_system,
                detected=False,
                metadata={"rejection_reason": "no_landmarks", "n_people": 0},
            )

        from stablewalk.pose.subject_lock import pick_subject, selection_for_source

        candidates = [
            PoseEstimator._add_mid_hip(
                PoseEstimator._clamp_keypoints(
                    PoseEstimator._landmarks_to_keypoints(landmarks)
                )
            )
            for landmarks in results.pose_landmarks
        ]
        picked = pick_subject(
            candidates,
            self._subject_lock,
            selection=self._subject_selection or selection_for_source(""),
            time_s=timestamp_s,
        )
        self._subject_lock = picked.lock
        keypoints = picked.keypoints
        if keypoints is None:
            return HumanMotionFrame(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                joint_positions_3d={},
                landmark_confidence={},
                backend_name=self.name,
                coordinate_system=self.coordinate_system,
                detected=False,
                metadata={
                    "rejection_reason": picked.reason or "subject_lock_lost",
                    "n_people": len(candidates),
                    "tracking_status": picked.tracking_status,
                    "identity_locked": False,
                },
            )

        detected = True
        rejection = ""
        if self._require_full_body:
            if not has_full_body_pose(keypoints):
                detected = False
                rejection = "incomplete_body"
            else:
                angles = compute_joint_angles(keypoints)
                if not meets_min_gait_dof(angles):
                    detected = False
                    rejection = "insufficient_gait_dof"

        frame = keypoints_to_canonical_frame(
            keypoints,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            backend_name=self.name,
            coordinate_system=self.coordinate_system,
            detected=detected,
        )
        frame.metadata["n_people"] = len(candidates)
        frame.metadata["identity_locked"] = bool(picked.identity_locked)
        frame.metadata["tracking_status"] = picked.tracking_status
        if rejection:
            frame.metadata["rejection_reason"] = rejection
        return frame
