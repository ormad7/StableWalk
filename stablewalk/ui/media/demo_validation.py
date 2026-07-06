"""Reusable validation for StableWalk demo gait videos."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import cv2

from stablewalk.pose.dof import GAIT_ANGLE_FIELDS, MIN_GAIT_DOF
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.pose.kinematics import compute_joint_angles
from stablewalk.pose.validation import has_full_body_pose, meets_min_gait_dof

logger = logging.getLogger(__name__)

FOOT_LANDMARKS = ("left_heel", "right_heel", "left_foot_index", "right_foot_index")
HIP_LANDMARKS = ("left_hip", "right_hip")
KNEE_LANDMARKS = ("left_knee", "right_knee")
ANKLE_LANDMARKS = ("left_ankle", "right_ankle")
MIN_VISIBILITY = 0.4


class ValidationGrade(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class DemoVideoValidationReport:
    video: str
    duration_s: float
    fps: float
    resolution: str
    total_frames: int
    sampled_frames: int
    pose_detection_rate: float
    hip_visibility_rate: float
    knee_visibility_rate: float
    ankle_visibility_rate: float
    foot_visibility_rate: float
    gait_detected_rate: float
    full_body_quality: ValidationGrade
    analysis_3d_suitability: ValidationGrade
    foot_clearance_suitability: ValidationGrade
    compact_status: str
    notes: tuple[str, ...] = ()

    def format_report(self) -> str:
        lines = [
            f"Video: {self.video}",
            f"Duration: {self.duration_s:.2f}s",
            f"FPS: {self.fps:.2f}",
            f"Resolution: {self.resolution}",
            f"Total Frames: {self.total_frames}",
            "",
            f"Pose Detection Rate: {self.pose_detection_rate:.1%}",
            f"Hip Visibility Rate: {self.hip_visibility_rate:.1%}",
            f"Knee Visibility Rate: {self.knee_visibility_rate:.1%}",
            f"Ankle Visibility Rate: {self.ankle_visibility_rate:.1%}",
            f"Foot Visibility Rate: {self.foot_visibility_rate:.1%}",
            f"Gait Detected Rate: {self.gait_detected_rate:.1%}",
            "",
            f"Full Body Quality: {self.full_body_quality.value}",
            f"3D Analysis Suitability: {self.analysis_3d_suitability.value}",
            f"Foot Clearance Suitability: {self.foot_clearance_suitability.value}",
            "",
            f"Compact status: {self.compact_status}",
        ]
        for note in self.notes:
            lines.append(f"Note: {note}")
        return "\n".join(lines)


def _visible(keypoints: list, name: str, min_visibility: float = MIN_VISIBILITY) -> bool:
    for kp in keypoints:
        if kp.name == name and kp.visibility >= min_visibility:
            return True
    return False


def _both_visible(keypoints: list, names: tuple[str, ...]) -> bool:
    return all(_visible(keypoints, name) for name in names)


def _grade(rate: float, *, pass_at: float, warn_at: float) -> ValidationGrade:
    if rate >= pass_at:
        return ValidationGrade.PASS
    if rate >= warn_at:
        return ValidationGrade.WARNING
    return ValidationGrade.FAIL


def validate_demo_video(
    video_path: str | Path,
    *,
    max_frames: int | None = 80,
    min_gait_detected_rate: float = 0.25,
) -> DemoVideoValidationReport:
    """
    Inspect a local demo clip before accepting it for StableWalk demos.

    Logs a detailed report and returns structured metrics for scripts and tests.
    """
    path = Path(video_path)
    if not path.is_file():
        report = DemoVideoValidationReport(
            video=str(path),
            duration_s=0.0,
            fps=0.0,
            resolution="unknown",
            total_frames=0,
            sampled_frames=0,
            pose_detection_rate=0.0,
            hip_visibility_rate=0.0,
            knee_visibility_rate=0.0,
            ankle_visibility_rate=0.0,
            foot_visibility_rate=0.0,
            gait_detected_rate=0.0,
            full_body_quality=ValidationGrade.FAIL,
            analysis_3d_suitability=ValidationGrade.FAIL,
            foot_clearance_suitability=ValidationGrade.FAIL,
            compact_status="Demo Video: file missing",
            notes=("File not found.",),
        )
        logger.warning(report.format_report())
        return report

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        report = DemoVideoValidationReport(
            video=str(path),
            duration_s=0.0,
            fps=0.0,
            resolution="unknown",
            total_frames=0,
            sampled_frames=0,
            pose_detection_rate=0.0,
            hip_visibility_rate=0.0,
            knee_visibility_rate=0.0,
            ankle_visibility_rate=0.0,
            foot_visibility_rate=0.0,
            gait_detected_rate=0.0,
            full_body_quality=ValidationGrade.FAIL,
            analysis_3d_suitability=ValidationGrade.FAIL,
            foot_clearance_suitability=ValidationGrade.FAIL,
            compact_status="Demo Video: cannot open",
            notes=("OpenCV could not open the file.",),
        )
        logger.warning(report.format_report())
        return report

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_s = (total_frames / fps) if fps > 0 else 0.0
    cap.release()

    pose_frames = hip_frames = knee_frames = ankle_frames = foot_frames = 0
    gait_detected = 0
    sampled = 0

    with PoseEstimator(video_mode=True) as estimator:
        sequence, _ = estimator.process_video_with_frame_cache(
            str(path),
            None,
            max_frames=max_frames,
        )
        sampled = len(sequence.frames)
        for frame in sequence.frames:
            keypoints = frame.keypoints or []
            if keypoints:
                pose_frames += 1
            if keypoints and _both_visible(keypoints, HIP_LANDMARKS):
                hip_frames += 1
            if keypoints and _both_visible(keypoints, KNEE_LANDMARKS):
                knee_frames += 1
            if keypoints and _both_visible(keypoints, ANKLE_LANDMARKS):
                ankle_frames += 1
            if keypoints and _both_visible(keypoints, FOOT_LANDMARKS):
                foot_frames += 1
            if frame.detected:
                gait_detected += 1

    denom = max(sampled, 1)
    pose_rate = pose_frames / denom
    hip_rate = hip_frames / denom
    knee_rate = knee_frames / denom
    ankle_rate = ankle_frames / denom
    foot_rate = foot_frames / denom
    gait_rate = gait_detected / denom

    full_body = _grade(pose_rate, pass_at=0.6, warn_at=0.35)
    if hip_rate >= 0.5 and knee_rate >= 0.5 and gait_rate >= min_gait_detected_rate:
        analysis_3d = ValidationGrade.PASS
    elif hip_rate >= 0.3 and knee_rate >= 0.3 and gait_rate >= 0.1:
        analysis_3d = ValidationGrade.WARNING
    else:
        analysis_3d = ValidationGrade.FAIL

    if foot_rate >= 0.45 and ankle_rate >= 0.5 and gait_rate >= min_gait_detected_rate:
        foot_clearance = ValidationGrade.PASS
    elif foot_rate >= 0.25 and ankle_rate >= 0.3:
        foot_clearance = ValidationGrade.WARNING
    else:
        foot_clearance = ValidationGrade.FAIL

    notes: list[str] = []
    if gait_rate < min_gait_detected_rate:
        notes.append(
            f"Gait-detected frames below target ({gait_rate:.1%} < {min_gait_detected_rate:.0%}). "
            f"MIN_GAIT_DOF={MIN_GAIT_DOF}."
        )
    if foot_rate < 0.45:
        notes.append("Foot landmarks are intermittently visible.")

    if (
        full_body == ValidationGrade.PASS
        and analysis_3d == ValidationGrade.PASS
        and foot_clearance == ValidationGrade.PASS
    ):
        compact = "Demo Video: Ready"
    elif foot_clearance == ValidationGrade.WARNING:
        compact = "Demo Video: Limited foot visibility"
    elif analysis_3d == ValidationGrade.WARNING:
        compact = "Demo Video: Limited gait detection"
    else:
        compact = "Demo Video: Not ready"

    report = DemoVideoValidationReport(
        video=str(path),
        duration_s=duration_s,
        fps=fps,
        resolution=f"{width}x{height}",
        total_frames=total_frames,
        sampled_frames=sampled,
        pose_detection_rate=pose_rate,
        hip_visibility_rate=hip_rate,
        knee_visibility_rate=knee_rate,
        ankle_visibility_rate=ankle_rate,
        foot_visibility_rate=foot_rate,
        gait_detected_rate=gait_rate,
        full_body_quality=full_body,
        analysis_3d_suitability=analysis_3d,
        foot_clearance_suitability=foot_clearance,
        compact_status=compact,
        notes=tuple(notes),
    )
    logger.info("\n%s", report.format_report())
    return report


def demo_is_ready(
    video_path: str | Path,
    *,
    max_frames: int | None = 80,
    min_gait_detected_rate: float = 0.25,
) -> bool:
    """True when a demo clip passes minimum gait-analysis readiness."""
    report = validate_demo_video(
        video_path,
        max_frames=max_frames,
        min_gait_detected_rate=min_gait_detected_rate,
    )
    return (
        report.analysis_3d_suitability != ValidationGrade.FAIL
        and report.gait_detected_rate >= min_gait_detected_rate
    )


def report_to_dict(report: DemoVideoValidationReport) -> dict:
    data = asdict(report)
    for key in (
        "full_body_quality",
        "analysis_3d_suitability",
        "foot_clearance_suitability",
    ):
        data[key] = data[key].value
    return data
