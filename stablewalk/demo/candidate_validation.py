"""
StableWalk demo candidate validation.

Evaluates whether a local video file is suitable for research-oriented gait demos.
Returns ACCEPT, ACCEPT_WITH_LIMITATIONS, or REJECT with a structured prevalidation
report suitable for DEMO_VIDEO_SOURCES.md documentation.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.biomech_stability import analyze_biomech_stability
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.analysis.gait_view_analysis import analyze_gait_view_geometry
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.ui.media.demo_validation import FOOT_LANDMARKS, MIN_VISIBILITY, _both_visible, _visible

logger = logging.getLogger(__name__)

HEEL_LANDMARKS = ("left_heel", "right_heel")
TOE_LANDMARKS = ("left_foot_index", "right_foot_index")
BODY_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

CandidateVerdict = Literal["ACCEPT", "ACCEPT_WITH_LIMITATIONS", "REJECT"]


class _Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class DemoCandidateThresholds:
    """Configurable acceptance gates for demo candidates."""

    min_pose_valid_frame_pct: float = 70.0
    min_foot_visibility_rate: float = 0.55
    min_heel_visibility_rate: float = 0.50
    min_toe_visibility_rate: float = 0.45
    min_subject_bbox_height_ratio: float = 0.35
    min_duration_s: float = 4.0
    min_usable_gait_cycles: int = 3
    min_total_heel_strikes: int = 6
    min_analysis_completeness_pct: float = 45.0
    max_camera_motion_score: float = 0.18
    min_view_confidence: float = 0.35

    # Soft limits (ACCEPT_WITH_LIMITATIONS)
    soft_pose_valid_frame_pct: float = 55.0
    soft_foot_visibility_rate: float = 0.40
    soft_usable_gait_cycles: int = 2
    soft_analysis_completeness_pct: float = 35.0


DEFAULT_DEMO_CANDIDATE_THRESHOLDS = DemoCandidateThresholds()


@dataclass
class DemoCandidateReport:
    video: str
    verdict: CandidateVerdict
    duration_s: float
    fps: float
    resolution: str
    total_frames: int
    sampled_frames: int
    pose_valid_frame_pct: float
    foot_visibility_rate: float
    heel_visibility_rate: float
    toe_visibility_rate: float
    mean_landmark_confidence: float
    subject_bbox_height_ratio: float
    camera_motion_score: float
    estimated_usable_gait_cycles: int
    detected_gait_cycles: int
    total_heel_strikes: int
    total_steps: int
    view_type: str | None
    view_confidence: float
    analysis_completeness_pct: float
    stability_confidence_badge: str
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_report(self) -> str:
        lines = [
            f"Video: {self.video}",
            f"Verdict: {self.verdict}",
            f"Duration: {self.duration_s:.2f}s @ {self.fps:.2f} FPS ({self.resolution})",
            "",
            f"Pose valid frame %: {self.pose_valid_frame_pct:.1f}",
            f"Foot visibility: {self.foot_visibility_rate:.1%}",
            f"Heel visibility: {self.heel_visibility_rate:.1%}",
            f"Toe visibility: {self.toe_visibility_rate:.1%}",
            f"Mean landmark confidence: {self.mean_landmark_confidence:.2f}",
            f"Subject bbox height / frame height: {self.subject_bbox_height_ratio:.2f}",
            f"Camera motion score: {self.camera_motion_score:.3f}",
            f"Estimated usable gait cycles: {self.estimated_usable_gait_cycles}",
            f"Detected gait cycles: {self.detected_gait_cycles}",
            f"Heel strikes: {self.total_heel_strikes}",
            f"Steps: {self.total_steps}",
            f"View: {self.view_type or 'unknown'} ({self.view_confidence:.0%})",
            f"Analysis completeness: {self.analysis_completeness_pct:.1f}%",
            f"Stability confidence badge: {self.stability_confidence_badge}",
        ]
        if self.issues:
            lines.extend(["", "Issues:"])
            lines.extend(f"  - {item}" for item in self.issues)
        if self.warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"  - {item}" for item in self.warnings)
        if self.notes:
            lines.extend(["", "Notes:"])
            lines.extend(f"  - {item}" for item in self.notes)
        return "\n".join(lines)


def _bbox_height_ratio(keypoints: list, frame_height: int) -> float | None:
    """Body span as fraction of frame height (MediaPipe y is normalized 0–1)."""
    del frame_height  # kept for call-site compatibility; y span is already normalized
    ys = [kp.y for kp in keypoints if kp.visibility >= MIN_VISIBILITY]
    if len(ys) < 4:
        return None
    return max(ys) - min(ys)


def _mean_visibility(keypoints: list, names: tuple[str, ...]) -> float | None:
    vals = [
        float(kp.visibility)
        for kp in keypoints
        if kp.name in names and kp.visibility > 0
    ]
    return statistics.mean(vals) if vals else None


def _camera_motion_score(centers: list[tuple[float, float]]) -> float:
    if len(centers) < 3:
        return 0.0
    xs = np.array([c[0] for c in centers], dtype=float)
    ys = np.array([c[1] for c in centers], dtype=float)
    dx = np.diff(xs)
    dy = np.diff(ys)
    disp = np.sqrt(dx * dx + dy * dy)
    return float(np.mean(disp))


def _evaluate_verdict(
    report: DemoCandidateReport,
    thresholds: DemoCandidateThresholds,
) -> CandidateVerdict:
    hard_fail = False
    soft_only = False

    checks: list[tuple[bool, bool, str]] = [
        (
            report.pose_valid_frame_pct >= thresholds.min_pose_valid_frame_pct,
            report.pose_valid_frame_pct >= thresholds.soft_pose_valid_frame_pct,
            f"Pose-valid frames {report.pose_valid_frame_pct:.1f}% below {thresholds.min_pose_valid_frame_pct:.0f}%",
        ),
        (
            report.foot_visibility_rate >= thresholds.min_foot_visibility_rate,
            report.foot_visibility_rate >= thresholds.soft_foot_visibility_rate,
            f"Foot visibility {report.foot_visibility_rate:.1%} below {thresholds.min_foot_visibility_rate:.0%}",
        ),
        (
            report.heel_visibility_rate >= thresholds.min_heel_visibility_rate,
            report.heel_visibility_rate >= thresholds.min_heel_visibility_rate * 0.85,
            f"Heel visibility {report.heel_visibility_rate:.1%} below {thresholds.min_heel_visibility_rate:.0%}",
        ),
        (
            report.toe_visibility_rate >= thresholds.min_toe_visibility_rate,
            report.toe_visibility_rate >= thresholds.min_toe_visibility_rate * 0.85,
            f"Toe visibility {report.toe_visibility_rate:.1%} below {thresholds.min_toe_visibility_rate:.0%}",
        ),
        (
            report.subject_bbox_height_ratio >= thresholds.min_subject_bbox_height_ratio,
            report.subject_bbox_height_ratio >= thresholds.min_subject_bbox_height_ratio * 0.85,
            (
                f"Subject scale {report.subject_bbox_height_ratio:.2f} below "
                f"{thresholds.min_subject_bbox_height_ratio:.2f} of frame height"
            ),
        ),
        (
            report.duration_s >= thresholds.min_duration_s,
            report.duration_s >= thresholds.min_duration_s * 0.85,
            f"Duration {report.duration_s:.2f}s below {thresholds.min_duration_s:.1f}s",
        ),
        (
            report.estimated_usable_gait_cycles >= thresholds.min_usable_gait_cycles,
            report.estimated_usable_gait_cycles >= thresholds.soft_usable_gait_cycles,
            (
                f"Usable gait cycles {report.estimated_usable_gait_cycles} "
                f"below {thresholds.min_usable_gait_cycles}"
            ),
        ),
        (
            report.total_heel_strikes >= thresholds.min_total_heel_strikes,
            report.total_heel_strikes >= max(4, thresholds.min_total_heel_strikes - 2),
            f"Heel strikes {report.total_heel_strikes} below {thresholds.min_total_heel_strikes}",
        ),
        (
            report.analysis_completeness_pct >= thresholds.min_analysis_completeness_pct,
            report.analysis_completeness_pct >= thresholds.soft_analysis_completeness_pct,
            (
                f"Analysis completeness {report.analysis_completeness_pct:.1f}% "
                f"below {thresholds.min_analysis_completeness_pct:.0f}%"
            ),
        ),
        (
            report.camera_motion_score <= thresholds.max_camera_motion_score,
            report.camera_motion_score <= thresholds.max_camera_motion_score * 1.35,
            f"Camera motion score {report.camera_motion_score:.3f} suggests unstable viewpoint",
        ),
    ]

    for ok, soft_ok, message in checks:
        if ok:
            continue
        if soft_ok:
            report.warnings.append(message)
            soft_only = True
        else:
            report.issues.append(message)
            hard_fail = True

    if report.view_confidence < thresholds.min_view_confidence:
        report.warnings.append(
            f"View confidence {report.view_confidence:.0%} is low for cross-video comparison"
        )
        soft_only = True

    if hard_fail:
        return "REJECT"
    if soft_only:
        return "ACCEPT_WITH_LIMITATIONS"
    return "ACCEPT"


def validate_demo_candidate(
    video_path: str | Path,
    *,
    max_frames: int | None = 180,
    thresholds: DemoCandidateThresholds | None = None,
    category: str | None = None,
) -> DemoCandidateReport:
    """
    Run StableWalk pose-quality prevalidation on a candidate demo video.

    Parameters
    ----------
    category:
        Optional demo category hint (``abnormal``, ``normal``, ``performance``).
        Does not change scoring — used only for report notes.
    """
    cfg = thresholds or DEFAULT_DEMO_CANDIDATE_THRESHOLDS
    path = Path(video_path)

    if not path.is_file():
        return DemoCandidateReport(
            video=str(path),
            verdict="REJECT",
            duration_s=0.0,
            fps=0.0,
            resolution="unknown",
            total_frames=0,
            sampled_frames=0,
            pose_valid_frame_pct=0.0,
            foot_visibility_rate=0.0,
            heel_visibility_rate=0.0,
            toe_visibility_rate=0.0,
            mean_landmark_confidence=0.0,
            subject_bbox_height_ratio=0.0,
            camera_motion_score=0.0,
            estimated_usable_gait_cycles=0,
            detected_gait_cycles=0,
            total_heel_strikes=0,
            total_steps=0,
            view_type=None,
            view_confidence=0.0,
            analysis_completeness_pct=0.0,
            stability_confidence_badge="LOW CONFIDENCE",
            issues=["Video file not found."],
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return DemoCandidateReport(
            video=str(path),
            verdict="REJECT",
            duration_s=0.0,
            fps=0.0,
            resolution="unknown",
            total_frames=0,
            sampled_frames=0,
            pose_valid_frame_pct=0.0,
            foot_visibility_rate=0.0,
            heel_visibility_rate=0.0,
            toe_visibility_rate=0.0,
            mean_landmark_confidence=0.0,
            subject_bbox_height_ratio=0.0,
            camera_motion_score=0.0,
            estimated_usable_gait_cycles=0,
            detected_gait_cycles=0,
            total_heel_strikes=0,
            total_steps=0,
            view_type=None,
            view_confidence=0.0,
            analysis_completeness_pct=0.0,
            stability_confidence_badge="LOW CONFIDENCE",
            issues=["OpenCV could not open the video file."],
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_s = (total_frames / fps) if fps > 0 else 0.0
    cap.release()

    pose_valid = foot_frames = heel_frames = toe_frames = 0
    confidences: list[float] = []
    bbox_ratios: list[float] = []
    centers: list[tuple[float, float]] = []
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
            if frame.detected:
                pose_valid += 1
            if keypoints and _both_visible(keypoints, FOOT_LANDMARKS):
                foot_frames += 1
            if keypoints and _both_visible(keypoints, HEEL_LANDMARKS):
                heel_frames += 1
            if keypoints and _both_visible(keypoints, TOE_LANDMARKS):
                toe_frames += 1
            for kp in keypoints:
                if kp.visibility > 0:
                    confidences.append(float(kp.visibility))
            ratio = _bbox_height_ratio(keypoints, height)
            if ratio is not None:
                bbox_ratios.append(ratio)
            hips = [kp for kp in keypoints if kp.name in ("left_hip", "right_hip")]
            if len(hips) >= 2:
                cx = statistics.mean(kp.x for kp in hips)
                cy = statistics.mean(kp.y for kp in hips)
                centers.append((cx, cy))

        enrich_pose_sequence(sequence)
        recording = pose_sequence_to_gait_motion(sequence)
        cycles = analyze_gait_cycles(recording)
        stability = analyze_biomech_stability(sequence, cycles=cycles)
        view_estimate, _view_profile = analyze_gait_view_geometry(sequence.frames)
        view = view_estimate

    denom = max(sampled, 1)
    report = DemoCandidateReport(
        video=str(path.resolve()),
        verdict="REJECT",
        duration_s=duration_s,
        fps=fps,
        resolution=f"{width}x{height}",
        total_frames=total_frames,
        sampled_frames=sampled,
        pose_valid_frame_pct=100.0 * pose_valid / denom,
        foot_visibility_rate=foot_frames / denom,
        heel_visibility_rate=heel_frames / denom,
        toe_visibility_rate=toe_frames / denom,
        mean_landmark_confidence=statistics.mean(confidences) if confidences else 0.0,
        subject_bbox_height_ratio=(
            statistics.mean(bbox_ratios) if bbox_ratios else 0.0
        ),
        camera_motion_score=_camera_motion_score(centers),
        estimated_usable_gait_cycles=stability.usable_gait_cycles,
        detected_gait_cycles=cycles.metrics.gait_cycle_count,
        total_heel_strikes=(
            cycles.metrics.left_heel_strike_count
            + cycles.metrics.right_heel_strike_count
        ),
        total_steps=(
            cycles.metrics.left_heel_strike_count
            + cycles.metrics.right_heel_strike_count
        ),
        view_type=view.view_type.value if view else None,
        view_confidence=view.view_confidence if view else 0.0,
        analysis_completeness_pct=stability.completeness_pct,
        stability_confidence_badge=stability.confidence_badge,
    )

    if category:
        report.notes.append(f"Category hint: {category}")

    report.verdict = _evaluate_verdict(report, cfg)
    logger.info("\n%s", report.format_report())
    return report


__all__ = [
    "CandidateVerdict",
    "DEFAULT_DEMO_CANDIDATE_THRESHOLDS",
    "DemoCandidateReport",
    "DemoCandidateThresholds",
    "validate_demo_candidate",
]
