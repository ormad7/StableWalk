"""Frame-level pipeline audit: source IDs, timestamps, status, pose validity."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from stablewalk.models.pose_data import (
    PoseFrame,
    PoseSequence,
    STATUS_DETECTION_FAILED,
    STATUS_FILTER_SKIP,
    STATUS_LOCK_SKIP,
    STATUS_POSE_OK,
    STATUS_UNDETECTED_UNCLASSIFIED,
)

AUDIT_COLUMNS = (
    "clip",
    "source_frame",
    "timestamp_s",
    "decode_ok",
    "in_analyzed_window",
    "processing_status",
    "inference_status",
    "pose_valid",
    "rejection_reason",
    "skipped",
    "n_people",
    "identity_locked",
    "left_contact_known",
    "right_contact_known",
    "left_unknown_reason",
    "right_unknown_reason",
    "excluded_reason",
)


@dataclass(frozen=True)
class TimelineWindows:
    analyzed_frame_count: int
    analyzed_duration_s: float
    detected_count: int
    usable_gait_frames: int
    usable_duration_s: float
    last_detected_timestamp_s: float | None
    first_detected_timestamp_s: float | None
    source_frame_count: int | None = None
    source_duration_s: float | None = None
    status_counts: dict[str, int] | None = None
    timestamp_gaps_if_detected_only: int | None = None
    cumulative_valid_duration_s: float = 0.0
    coverage: float = 0.0
    longest_valid_segment_s: float = 0.0
    n_valid_runs: int = 0
    interval_counting_convention: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_frame_count": self.source_frame_count,
            "source_duration_s": self.source_duration_s,
            "analyzed_frame_count": self.analyzed_frame_count,
            "analyzed_span_s": self.analyzed_duration_s,
            "analyzed_duration_s": self.analyzed_duration_s,
            "detected_count": self.detected_count,
            "usable_gait_frames": self.usable_gait_frames,
            "cumulative_valid_duration_s": self.cumulative_valid_duration_s,
            "coverage": self.coverage,
            "coverage_pct": round(self.coverage * 100.0, 3),
            "longest_valid_segment_s": self.longest_valid_segment_s,
            "n_valid_runs": self.n_valid_runs,
            "usable_duration_s": self.cumulative_valid_duration_s,
            "first_detected_timestamp_s": self.first_detected_timestamp_s,
            "last_detected_timestamp_s": self.last_detected_timestamp_s,
            "status_counts": dict(self.status_counts or {}),
            "timestamp_gaps_if_detected_only": self.timestamp_gaps_if_detected_only,
            "interval_counting_convention": self.interval_counting_convention,
        }


def frame_audit_row(
    frame: PoseFrame,
    *,
    clip: str = "",
    decode_ok: bool = True,
    in_analyzed_window: bool = True,
    contact_frame: Any | None = None,
    excluded_reason: str = "",
) -> dict[str, Any]:
    status = frame.processing_status()
    identity_locked = bool(getattr(frame, "identity_locked", False))
    left_known = right_known = ""
    left_reason = right_reason = ""
    if contact_frame is not None:
        left_known = bool(getattr(contact_frame, "left_contact_known", False))
        right_known = bool(getattr(contact_frame, "right_contact_known", False))
        left_reason = str(getattr(contact_frame, "left_unknown_reason", "") or "")
        right_reason = str(getattr(contact_frame, "right_unknown_reason", "") or "")
    return {
        "clip": clip,
        "source_frame": int(frame.frame_index),
        "timestamp_s": float(frame.timestamp_s),
        "decode_ok": bool(decode_ok),
        "in_analyzed_window": bool(in_analyzed_window),
        "processing_status": status,
        "inference_status": status,
        "pose_valid": bool(frame.detected),
        "rejection_reason": str(frame.rejection_reason or ""),
        "skipped": bool(frame.skipped),
        "n_people": int(frame.n_people or 0),
        "identity_locked": identity_locked,
        "left_contact_known": left_known,
        "right_contact_known": right_known,
        "left_unknown_reason": left_reason,
        "right_unknown_reason": right_reason,
        "excluded_reason": excluded_reason,
    }


def iter_frame_audit_rows(
    sequence: PoseSequence,
    *,
    clip: str = "",
    contact: Any | None = None,
) -> Iterable[dict[str, Any]]:
    contact_by_id: dict[int, Any] = {}
    if contact is not None:
        for item in getattr(contact, "per_frame", None) or ():
            try:
                contact_by_id[int(item.frame_index)] = item
            except (TypeError, ValueError, AttributeError):
                continue
    for frame in sequence.frames:
        yield frame_audit_row(
            frame,
            clip=clip,
            contact_frame=contact_by_id.get(int(frame.frame_index)),
        )


def source_only_audit_row(
    *,
    clip: str,
    source_frame: int,
    timestamp_s: float,
    excluded_reason: str,
) -> dict[str, Any]:
    """A decoded source frame that was never inferred in this analysis window."""
    return {
        "clip": clip,
        "source_frame": int(source_frame),
        "timestamp_s": float(timestamp_s),
        "decode_ok": True,
        "in_analyzed_window": False,
        "processing_status": "not_analyzed",
        "inference_status": "not_analyzed",
        "pose_valid": False,
        "rejection_reason": excluded_reason,
        "skipped": True,
        "n_people": 0,
        "identity_locked": False,
        "left_contact_known": "",
        "right_contact_known": "",
        "left_unknown_reason": "",
        "right_unknown_reason": "",
        "excluded_reason": excluded_reason,
    }


def write_frame_audit_csv(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AUDIT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in AUDIT_COLUMNS})
    return path


def _status_counts(sequence: PoseSequence) -> dict[str, int]:
    counts = {
        STATUS_POSE_OK: 0,
        STATUS_FILTER_SKIP: 0,
        STATUS_LOCK_SKIP: 0,
        STATUS_DETECTION_FAILED: 0,
        STATUS_UNDETECTED_UNCLASSIFIED: 0,
    }
    for frame in sequence.frames:
        key = frame.processing_status()
        counts[key] = counts.get(key, 0) + 1
    return counts


def timeline_windows(
    sequence: PoseSequence,
    *,
    source_frame_count: int | None = None,
    source_duration_s: float | None = None,
) -> TimelineWindows:
    from stablewalk.analysis.timeline_duration import (
        INTERVAL_COUNTING_CONVENTION,
        compute_timeline_duration_metrics,
        valid_pose_mask,
    )
    from stablewalk.pose.tracking_quality import audit_pose_sequence

    frames = list(sequence.frames)
    times = [float(f.timestamp_s) for f in frames]
    fps = max(float(sequence.fps), 1e-6)
    duration = compute_timeline_duration_metrics(
        times, valid_pose_mask(frames), fps=fps
    )
    detected = [f for f in frames if f.detected]
    det_times = [float(f.timestamp_s) for f in detected]
    audit = audit_pose_sequence(sequence)

    gaps = None
    if detected:
        from stablewalk.analysis.joint_path_trace import segment_is_connected

        gaps = 0
        for i in range(1, len(detected)):
            if not segment_is_connected(
                float(detected[i - 1].timestamp_s),
                float(detected[i].timestamp_s),
                fps=fps,
            ):
                gaps += 1

    return TimelineWindows(
        analyzed_frame_count=len(frames),
        analyzed_duration_s=duration.analyzed_span_s,
        detected_count=len(detected),
        usable_gait_frames=int(audit.usable_gait_frames),
        usable_duration_s=duration.cumulative_valid_duration_s,
        last_detected_timestamp_s=det_times[-1] if det_times else None,
        first_detected_timestamp_s=det_times[0] if det_times else None,
        source_frame_count=source_frame_count,
        source_duration_s=source_duration_s,
        status_counts=_status_counts(sequence),
        timestamp_gaps_if_detected_only=gaps,
        cumulative_valid_duration_s=duration.cumulative_valid_duration_s,
        coverage=duration.coverage,
        longest_valid_segment_s=duration.longest_valid_segment_s,
        n_valid_runs=duration.n_valid_runs,
        interval_counting_convention=INTERVAL_COUNTING_CONVENTION,
    )
