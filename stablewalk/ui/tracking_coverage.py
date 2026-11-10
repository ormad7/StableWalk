"""Display-only tracking coverage — not a gait score and not analysis confidence.

Separates person-detection coverage from joint visibility and from frames that
have the limb landmarks gait analysis actually uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

@dataclass(frozen=True)
class TrackingCoverage:
    total_frames: int
    detected_frames: int
    detection_pct: float
    joint_visibility_pct: float | None
    usable_gait_frames: int
    usable_gait_pct: float
    video_source_frames: int | None = None

    def hud_text(self) -> str:
        return (
            f"Detect {self.detection_pct:.0f}% ({self.detected_frames}/{self.total_frames})\n"
            f"Joints {self.joint_confidence_display()}\n"
            f"Usable {self.usable_gait_frames}"
        )

    def compact_text(self) -> str:
        return (
            f"Detect {self.detection_pct:.0f}% · "
            f"Joints {self.joint_confidence_display()} · "
            f"Usable {self.usable_gait_frames}"
        )

    def detected_display(self) -> str:
        return f"{self.detected_frames}/{self.total_frames} ({self.detection_pct:.0f}%)"

    def joint_confidence_display(self) -> str:
        if self.joint_visibility_pct is None:
            return "—"
        return f"{self.joint_visibility_pct:.0f}% vis"

    def usable_display(self) -> str:
        return f"{self.usable_gait_frames}/{self.total_frames} ({self.usable_gait_pct:.0f}%)"

    def detection_tooltip(self) -> str:
        extra = ""
        if (
            self.video_source_frames is not None
            and int(self.video_source_frames) > int(self.total_frames)
        ):
            extra = (
                f" The loaded analysis uses {self.total_frames} of "
                f"{int(self.video_source_frames)} frames in the original video file."
            )
        return (
            f"{self.detection_pct:.0f}% is person-detection coverage: "
            f"{self.detected_frames} detected frames of {self.total_frames} "
            "analyzed source frames. Detected and usable counts are separate "
            "from the source-frame count. A 100% detection rate is not accuracy "
            "and is not interchangeable with landmark visibility, spectral "
            "reliability, or biomechanical validity."
            f"{extra}"
        )

    def joint_confidence_tooltip(self) -> str:
        vis = (
            f"{self.joint_visibility_pct:.0f}%"
            if self.joint_visibility_pct is not None
            else "unavailable"
        )
        return (
            f"{vis} is the mean detector visibility of hips, knees, ankles, "
            "heels, and toes on detected frames. It is not joint-presence "
            "accuracy and is not a clinical score."
        )

    def usable_tooltip(self) -> str:
        return (
            f"{self.usable_gait_pct:.0f}% is the share of the clip with observed "
            f"hips, knees, and ankles that pass visibility, anatomical order, "
            f"and jump checks ({self.usable_gait_frames} of {self.total_frames} "
            "frames). A detected pose is not automatically usable. "
            "This is not accuracy."
        )

    def tooltip(self) -> str:
        return (
            "Tracking is not a single quality score.\n"
            f"{self.detection_tooltip()}\n"
            f"{self.joint_confidence_tooltip()}\n"
            f"{self.usable_tooltip()}\n"
            "This is not analysis confidence and is not a clinical score."
        )


def _visibility(kp: Any) -> float | None:
    raw = getattr(kp, "visibility", None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN — visibility was not stored
        return None
    return value


def measure_tracking_coverage(
    sequence: Any | None,
    pose_indices: Iterable[int] | None = None,
    *,
    video_source_frames: int | None = None,
) -> TrackingCoverage | None:
    """Compute detection / joint-visibility / usable-gait-frame counts."""
    frames = list(getattr(sequence, "frames", None) or [])
    if not frames:
        return None
    from stablewalk.pose.tracking_quality import (
        LOWER_LIMB_JOINTS,
        audit_pose_sequence,
        landmark_visibility,
    )

    total = len(frames)
    detected = 0
    vis_sum = 0.0
    vis_n = 0
    for fr in frames:
        kps = list(getattr(fr, "keypoints", None) or [])
        sk = getattr(fr, "skeleton_3d", None) or {}
        has_landmarks = bool(kps) or (isinstance(sk, dict) and bool(sk.get("joints")))
        identity_present = bool(getattr(fr, "detected", False)) or (
            bool(getattr(fr, "identity_locked", False)) and has_landmarks
        )
        if not identity_present:
            continue
        detected += 1
        for name in LOWER_LIMB_JOINTS:
            v = landmark_visibility(fr, name)
            if v is None:
                continue
            vis_sum += v
            vis_n += 1
    if pose_indices is not None:
        # pose_indices is the playback list (all stored source frames).
        # Detection counts always come from sequence.frames, not this list.
        _ = list(pose_indices)
    audit = audit_pose_sequence(sequence)
    vis_pct = (100.0 * vis_sum / vis_n) if vis_n else None
    return TrackingCoverage(
        total_frames=total,
        detected_frames=detected,
        detection_pct=100.0 * detected / max(total, 1),
        joint_visibility_pct=vis_pct,
        usable_gait_frames=audit.usable_gait_frames,
        usable_gait_pct=100.0 * audit.usable_gait_frames / max(total, 1),
        video_source_frames=video_source_frames,
    )
