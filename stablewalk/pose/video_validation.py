"""
Pre-flight validation for video sources (URL or file) before full processing.
"""

from __future__ import annotations

import logging

import cv2
from mediapipe import Image, ImageFormat

from stablewalk.pose.estimation import PoseEstimator
from stablewalk.pose.validation import has_full_body_pose

logger = logging.getLogger(__name__)


def validate_video_source(
    source: str,
    *,
    sample_count: int = 20,
    min_valid_ratio: float = 0.25,
    model_variant: str = "full",
) -> tuple[bool, float, str]:
    """
    Sample frames from a video URL or path and check for full-body poses.

    Returns:
        (passed, valid_ratio, message)
    """
    from stablewalk.core.pipeline_reset import register_capture, release_all_captures

    release_all_captures()
    cap = cv2.VideoCapture(source)
    register_capture(cap)
    if not cap.isOpened():
        release_all_captures()
        return False, 0.0, f"Could not open video source: {source}"

    total = 0
    valid = 0

    try:
        with PoseEstimator(model_variant=model_variant, video_mode=False) as estimator:
            while total < sample_count:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
                results = estimator._landmarker.detect(mp_image)

                total += 1
                if not results.pose_landmarks:
                    continue

                landmarks = results.pose_landmarks[0]
                keypoints = estimator._add_mid_hip(
                    estimator._clamp_keypoints(estimator._landmarks_to_keypoints(landmarks))
                )
                if has_full_body_pose(keypoints):
                    valid += 1
    finally:
        release_all_captures()

    if total == 0:
        return False, 0.0, "Video source returned no readable frames."

    ratio = valid / total
    passed = ratio >= min_valid_ratio
    message = (
        f"Validation: {valid}/{total} sampled frames with full-body pose "
        f"({ratio:.0%}, need >={min_valid_ratio:.0%})."
    )
    if not passed:
        message += " Try a different URL with a clear full-body walker."
    logger.info(message)
    return passed, ratio, message
