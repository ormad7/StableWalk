"""Video quality assessment before biomechanical analysis."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from stablewalk.analysis.gait_comparison_validation import FOOT_LANDMARKS
from stablewalk.models.pose_data import PoseFrame, PoseSequence


@dataclass
class VideoQualityAssessment:
    overall_quality_score: float  # 0–100
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    kind: str = "derived"

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_quality_score": round(self.overall_quality_score, 1),
            "warnings": list(self.warnings),
            "checks": self.checks,
            "kind": self.kind,
            "note": "Heuristic video/pose quality — not clinical motion-capture QA.",
        }


def _body_truncation_score(frames: list[PoseFrame]) -> tuple[float, list[str]]:
    warnings: list[str] = []
    missing_feet = 0
    missing_head = 0
    n = 0
    for f in frames:
        if not f.detected or not f.keypoints:
            continue
        n += 1
        names = {kp.name for kp in f.keypoints if kp.visibility >= 0.3}
        if not any(x in names for x in FOOT_LANDMARKS):
            missing_feet += 1
        if "nose" not in names and "head" not in names:
            missing_head += 1
    if n == 0:
        return 0.0, ["No detected pose frames"]
    foot_pct = missing_feet / n
    head_pct = missing_head / n
    score = 100.0
    if foot_pct > 0.25:
        warnings.append(f"Missing feet in {foot_pct:.0%} of frames — contact analysis unreliable.")
        score -= 25.0
    if head_pct > 0.30:
        warnings.append(f"Head/trunk truncated in {head_pct:.0%} of frames.")
        score -= 15.0
    if foot_pct > 0.5:
        warnings.append("Body truncation: feet frequently out of frame.")
        score -= 20.0
    return max(0.0, score), warnings


def _camera_movement_score(frames: list[PoseFrame]) -> tuple[float, list[str]]:
    warnings: list[str] = []
    # Proxy: pelvis position variance in image space
    xs: list[float] = []
    for f in frames:
        if not f.detected or not f.keypoints:
            continue
        kp_map = {kp.name: kp for kp in f.keypoints}
        lh = kp_map.get("left_hip")
        rh = kp_map.get("right_hip")
        if lh and rh:
            xs.append((lh.x + rh.x) * 0.5)
    if len(xs) < 5:
        return 70.0, []
    mean_x = statistics.mean(xs)
    jitter = statistics.pstdev(xs)
    if jitter > 0.08:
        warnings.append("Possible camera movement or subject drifting in frame.")
        return max(40.0, 100.0 - jitter * 400.0), warnings
    return 95.0, warnings


def _pose_confidence_score(frames: list[PoseFrame]) -> tuple[float, list[str]]:
    warnings: list[str] = []
    confs: list[float] = []
    for f in frames:
        if not f.detected or not f.keypoints:
            continue
        confs.extend(float(kp.visibility) for kp in f.keypoints)
    if not confs:
        return 0.0, ["No pose confidence data"]
    mean_c = statistics.mean(confs)
    if mean_c < 0.45:
        warnings.append(f"Low average pose confidence ({mean_c:.0%}).")
    if mean_c < 0.35:
        warnings.append("Poor lighting or occlusion likely degrading pose estimation.")
    return min(100.0, mean_c * 110.0), warnings


def assess_video_quality(sequence: PoseSequence) -> VideoQualityAssessment:
    """Evaluate input video quality for biomechanical analysis reliability."""
    frames = sequence.frames
    total = len(frames)
    detected = sum(1 for f in frames if f.detected)
    fps = max(sequence.fps, 1e-6)

    warnings: list[str] = []
    checks: dict[str, Any] = {
        "fps": round(fps, 2),
        "frame_count": total,
        "detected_frame_pct": round(100.0 * detected / max(total, 1), 2),
    }

    score = 100.0

    if fps < 20:
        warnings.append(f"Low FPS ({fps:.1f}) — temporal gait metrics less reliable.")
        score -= 15.0
        checks["low_fps"] = True

    if detected / max(total, 1) < 0.5:
        warnings.append("Fewer than 50% valid pose frames.")
        score -= 30.0

    trunc_score, trunc_warn = _body_truncation_score(frames)
    warnings.extend(trunc_warn)
    score = min(score, trunc_score)
    checks["truncation_score"] = round(trunc_score, 1)

    cam_score, cam_warn = _camera_movement_score(frames)
    warnings.extend(cam_warn)
    score = min(score, cam_score)
    checks["camera_stability_score"] = round(cam_score, 1)

    conf_score, conf_warn = _pose_confidence_score(frames)
    warnings.extend(conf_warn)
    score = min(score, conf_score)
    checks["pose_confidence_score"] = round(conf_score, 1)

    # Motion blur proxy: large frame-to-frame landmark jumps
    jumps: list[float] = []
    prev_kp = None
    for f in frames:
        if not f.detected or not f.keypoints:
            continue
        kp_map = {kp.name: kp for kp in f.keypoints}
        ankle = kp_map.get("left_ankle")
        if ankle and prev_kp:
            jumps.append(abs(ankle.y - prev_kp))
        if ankle:
            prev_kp = ankle.y
    if jumps and statistics.mean(jumps) > 0.06:
        warnings.append("Excessive inter-frame landmark motion — possible blur or tracking loss.")
        score -= 10.0
        checks["motion_blur_suspected"] = True

    checks["occlusion_risk"] = conf_score < 55.0

    return VideoQualityAssessment(
        overall_quality_score=max(0.0, min(100.0, score)),
        warnings=warnings,
        checks=checks,
    )


__all__ = ["VideoQualityAssessment", "assess_video_quality"]
