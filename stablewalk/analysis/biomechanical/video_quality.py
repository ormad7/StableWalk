"""Video quality assessment before biomechanical analysis."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Literal

from stablewalk.analysis.gait_comparison_validation import FOOT_LANDMARKS
from stablewalk.models.pose_data import PoseFrame, PoseSequence

QualityStatus = Literal["pass", "warn", "fail"]


@dataclass
class VideoQualityItem:
    label: str
    status: QualityStatus
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "status": self.status, "detail": self.detail}


@dataclass
class VideoQualityAssessment:
    overall_quality_score: float  # 0–100
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    items: list[VideoQualityItem] = field(default_factory=list)
    summary_explanation: str = ""
    kind: str = "derived"

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_quality_score": round(self.overall_quality_score, 1),
            "warnings": list(self.warnings),
            "checks": self.checks,
            "items": [item.to_dict() for item in self.items],
            "summary_explanation": self.summary_explanation,
            "kind": self.kind,
            "note": "Heuristic video/pose quality — not clinical motion-capture QA.",
        }


def _body_truncation_score(frames: list[PoseFrame]) -> tuple[float, list[str], dict[str, Any]]:
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
        return 0.0, ["No detected pose frames"], {"feet_detected_pct": 0.0, "head_visible_pct": 0.0}

    foot_pct = 1.0 - missing_feet / n
    head_pct = 1.0 - missing_head / n
    score = 100.0
    if missing_feet / n > 0.25:
        warnings.append(f"Missing feet in {missing_feet / n:.0%} of frames — contact analysis unreliable.")
        score -= 25.0
    if missing_head / n > 0.30:
        warnings.append(f"Head/trunk truncated in {missing_head / n:.0%} of frames.")
        score -= 15.0
    if missing_feet / n > 0.5:
        warnings.append("Body truncation: feet frequently out of frame.")
        score -= 20.0
    return max(0.0, score), warnings, {
        "feet_detected_pct": round(foot_pct * 100.0, 1),
        "head_visible_pct": round(head_pct * 100.0, 1),
    }


def _camera_movement_score(frames: list[PoseFrame]) -> tuple[float, list[str], float]:
    warnings: list[str] = []
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
        return 70.0, [], 0.0
    jitter = statistics.pstdev(xs)
    if jitter > 0.08:
        warnings.append("Possible camera movement or subject drifting in frame.")
        return max(40.0, 100.0 - jitter * 400.0), warnings, jitter
    return 95.0, warnings, jitter


def _pose_confidence_score(frames: list[PoseFrame]) -> tuple[float, list[str], float]:
    warnings: list[str] = []
    confs: list[float] = []
    low_conf_joints = 0
    total_joints = 0
    for f in frames:
        if not f.detected or not f.keypoints:
            continue
        for kp in f.keypoints:
            total_joints += 1
            v = float(kp.visibility)
            confs.append(v)
            if v < 0.45:
                low_conf_joints += 1
    if not confs:
        return 0.0, ["No pose confidence data"], 0.0
    mean_c = statistics.mean(confs)
    low_pct = low_conf_joints / max(total_joints, 1)
    if mean_c < 0.45:
        warnings.append(f"Low average pose confidence ({mean_c:.0%}).")
    if mean_c < 0.35:
        warnings.append("Poor lighting or occlusion likely degrading pose estimation.")
    return min(100.0, mean_c * 110.0), warnings, low_pct


def _lighting_score(mean_conf: float) -> tuple[float, str]:
    if mean_conf >= 0.65:
        return 95.0, "Average landmark visibility is strong."
    if mean_conf >= 0.45:
        return 75.0, "Moderate landmark visibility — acceptable for gait timing."
    return max(30.0, mean_conf * 120.0), "Low landmark visibility suggests poor lighting or contrast."


def _motion_blur_proxy(frames: list[PoseFrame]) -> tuple[bool, float, str]:
    jumps: list[float] = []
    prev_kp = None
    for f in frames:
        if not f.detected or not f.keypoints:
            continue
        kp_map = {kp.name: kp for kp in f.keypoints}
        ankle = kp_map.get("left_ankle")
        if ankle and prev_kp is not None:
            jumps.append(abs(ankle.y - prev_kp))
        if ankle:
            prev_kp = ankle.y
    if not jumps:
        return False, 0.0, "Insufficient ankle samples to assess blur."
    mean_jump = statistics.mean(jumps)
    if mean_jump > 0.06:
        return True, mean_jump, f"Large inter-frame ankle motion ({mean_jump:.3f} norm units)."
    return False, mean_jump, "Inter-frame landmark motion within expected range."


def assess_video_quality(sequence: PoseSequence) -> VideoQualityAssessment:
    """Evaluate input video quality for biomechanical analysis reliability."""
    frames = sequence.frames
    total = len(frames)
    detected = sum(1 for f in frames if f.detected)
    fps = max(sequence.fps, 1e-6)

    warnings: list[str] = []
    items: list[VideoQualityItem] = []
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

    trunc_score, trunc_warn, trunc_meta = _body_truncation_score(frames)
    warnings.extend(trunc_warn)
    score = min(score, trunc_score)
    checks.update(trunc_meta)
    checks["truncation_score"] = round(trunc_score, 1)

    feet_pct = trunc_meta.get("feet_detected_pct", 0.0)
    head_pct = trunc_meta.get("head_visible_pct", 0.0)
    full_body = feet_pct >= 75.0 and head_pct >= 70.0
    items.append(
        VideoQualityItem(
            label="Full body visible",
            status="pass" if full_body else ("warn" if feet_pct >= 50 else "fail"),
            detail=f"Feet detected {feet_pct:.0f}% · head/trunk {head_pct:.0f}% of frames",
        )
    )
    items.append(
        VideoQualityItem(
            label="Feet detected",
            status="pass" if feet_pct >= 80 else ("warn" if feet_pct >= 55 else "fail"),
            detail=f"Foot landmarks visible in {feet_pct:.0f}% of detected frames",
        )
    )

    cam_score, cam_warn, jitter = _camera_movement_score(frames)
    warnings.extend(cam_warn)
    score = min(score, cam_score)
    checks["camera_stability_score"] = round(cam_score, 1)
    checks["pelvis_jitter"] = round(jitter, 4)
    static_camera = jitter <= 0.08 and cam_score >= 80.0
    items.append(
        VideoQualityItem(
            label="Static camera",
            status="pass" if static_camera else "warn",
            detail=(
                f"Pelvis image jitter σ={jitter:.3f} (lower is steadier)"
                if jitter > 0
                else "Insufficient pelvis samples for camera stability check"
            ),
        )
    )

    conf_score, conf_warn, low_conf_pct = _pose_confidence_score(frames)
    warnings.extend(conf_warn)
    score = min(score, conf_score)
    checks["pose_confidence_score"] = round(conf_score, 1)
    checks["low_confidence_joint_pct"] = round(low_conf_pct * 100.0, 1)
    mean_conf = conf_score / 110.0 if conf_score > 0 else 0.0
    light_score, light_detail = _lighting_score(mean_conf)
    score = min(score, light_score)
    items.append(
        VideoQualityItem(
            label="Good lighting",
            status="pass" if mean_conf >= 0.65 else ("warn" if mean_conf >= 0.45 else "fail"),
            detail=light_detail,
        )
    )
    low_conf_regions = low_conf_pct >= 0.35 or mean_conf < 0.45
    items.append(
        VideoQualityItem(
            label="Low confidence regions",
            status="warn" if low_conf_regions else "pass",
            detail=(
                f"{checks['low_confidence_joint_pct']:.0f}% of joint samples below 45% visibility"
                if low_conf_regions
                else "Most landmarks maintain adequate visibility"
            ),
        )
    )

    blur, blur_mag, blur_detail = _motion_blur_proxy(frames)
    if blur:
        warnings.append("Excessive inter-frame landmark motion — possible blur or tracking loss.")
        score -= 10.0
        checks["motion_blur_suspected"] = True
    checks["motion_blur_magnitude"] = round(blur_mag, 4)
    items.append(
        VideoQualityItem(
            label="Motion blur",
            status="warn" if blur else "pass",
            detail=blur_detail,
        )
    )

    checks["occlusion_risk"] = conf_score < 55.0
    overall = max(0.0, min(100.0, score))

    pass_count = sum(1 for item in items if item.status == "pass")
    warn_count = sum(1 for item in items if item.status == "warn")
    summary = (
        f"Score {overall:.0f}/100 from {pass_count} passed, {warn_count} caution checks. "
        "Derived from pose visibility, camera stability, and temporal coherence — not a clinical QA certificate."
    )

    return VideoQualityAssessment(
        overall_quality_score=overall,
        warnings=warnings,
        checks=checks,
        items=items,
        summary_explanation=summary,
    )


__all__ = ["VideoQualityAssessment", "VideoQualityItem", "assess_video_quality"]
