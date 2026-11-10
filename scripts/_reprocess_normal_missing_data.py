"""Re-analyze walker-seed Normal dump; export frame audit and overlap overlay.

Does not re-run MediaPipe. Identity lock and overlap-unavailable policy are
already in the walker-seed dump. This pass classifies rejection reasons,
recomputes COM / validity labels, and writes coverage before/after.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.analysis_summary import build_analysis_summary
from stablewalk.analysis.biomechanical.com_estimation import analyze_center_of_mass
from stablewalk.analysis.biomechanical.orchestrator import run_biomechanical_analysis
from stablewalk.analysis.biomechanical.video_quality import assess_video_quality
from stablewalk.analysis.foot_contact_analysis import analyze_foot_contact
from stablewalk.analysis.gait_cycle_analysis import (
    analyze_gait_cycles,
    summary_cadence_spm,
)
from stablewalk.analysis.joint_path_trace import series_with_gap_breaks
from stablewalk.analysis.result_validity import build_result_validity
from stablewalk.io.frame_audit import (
    iter_frame_audit_rows,
    timeline_windows,
    write_frame_audit_csv,
)
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.skeleton import draw_identity_on_bgr, draw_skeleton_on_bgr
from stablewalk.pose.subject_lock import pose_bbox_norm
from stablewalk.pose.video import VideoReader
from stablewalk.ui.tracking_coverage import measure_tracking_coverage

VIDEO = ROOT / "data" / "demo_videos" / "normal_gait_v2.mp4"
IDENTITY_SUMMARY = ROOT / "validation_outputs" / "normal" / "identity_probe" / "identity_summary.json"
OUT = ROOT / "validation_outputs" / "normal"
PROBE = OUT / "missing_data_probe"
CLIP_START = 130
CLIP_END = 200
STILL_FRAMES = (80, 148, 191, 220)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gap_class(frame) -> str:
    status = frame.processing_status()
    reason = str(frame.rejection_reason or "")
    if frame.detected:
        return "pose_ok"
    if status == "subject_lock_skip" or reason in {
        "subject_identity_ambiguous",
        "subject_lock_lost",
    }:
        return "identity_ambiguity"
    if reason == "incomplete_body":
        return "occlusion_or_incomplete_body"
    if reason == "insufficient_gait_dof":
        return "insufficient_gait_dof"
    if status == "filter_skip":
        return "filter_skip_other"
    if status == "detection_failed" or reason == "no_landmarks":
        return "detection_failed"
    if "low" in reason and "conf" in reason:
        return "low_confidence"
    return status or "other"


def _gap_runs(frames) -> list[dict]:
    runs: list[dict] = []
    current: dict | None = None
    for frame in frames:
        key = _gap_class(frame)
        if key == "pose_ok":
            if current is not None:
                runs.append(current)
                current = None
            continue
        idx = int(frame.frame_index)
        if current is None or current["reason"] != key:
            if current is not None:
                runs.append(current)
            current = {
                "reason": key,
                "start_frame": idx,
                "end_frame": idx,
                "n_frames": 1,
                "rejection_reason": str(frame.rejection_reason or ""),
            }
        else:
            current["end_frame"] = idx
            current["n_frames"] += 1
    if current is not None:
        runs.append(current)
    return runs


def _overlay_frame(bgr, pose_frame):
    out = bgr.copy()
    selected = bool(
        pose_frame is not None
        and pose_frame.keypoints
        and (pose_frame.detected or pose_frame.identity_locked)
    )
    if selected:
        out = draw_skeleton_on_bgr(out, pose_frame.keypoints)
    if pose_frame is not None:
        out = draw_identity_on_bgr(out, pose_frame)
    return out


def _landmark_probe(sequence) -> dict:
    by = {int(f.frame_index): f for f in sequence.frames}
    details = {}
    for idx in STILL_FRAMES:
        pf = by.get(idx)
        if pf is None:
            details[str(idx)] = None
            continue
        hip = next((kp for kp in pf.keypoints if kp.name == "mid_hip"), None)
        box = pose_bbox_norm(pf.keypoints) if pf.keypoints else None
        details[str(idx)] = {
            "detected": bool(pf.detected),
            "n_people": int(pf.n_people or 0),
            "processing_status": pf.processing_status(),
            "tracking_status": pf.tracking_status,
            "identity_locked": bool(pf.identity_locked),
            "rejection_reason": pf.rejection_reason,
            "gap_class": _gap_class(pf),
            "n_keypoints": len(pf.keypoints or []),
            "hip_x": None if hip is None else round(float(hip.x), 4),
            "subject_bbox": None if box is None else [round(v, 4) for v in box],
            "other_bboxes": [
                [round(v, 4) for v in b] for b in (pf.other_bboxes or [])
            ],
        }
    return details


def main() -> int:
    if not IDENTITY_SUMMARY.is_file():
        print("missing identity summary", IDENTITY_SUMMARY)
        return 1
    before = json.loads(IDENTITY_SUMMARY.read_text(encoding="utf-8"))
    poses_path = Path(before["poses_path"])
    if not poses_path.is_file():
        print("missing pose dump", poses_path)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    PROBE.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    sequence = load_pose_sequence(poses_path)
    frames = list(sequence.frames)
    recording = pose_sequence_to_gait_motion(sequence, detected_only=False)
    cycles = analyze_gait_cycles(recording, sequence=sequence)
    contact = analyze_foot_contact(recording, cycles=cycles)
    ba = run_biomechanical_analysis(
        recording, sequence, cycles=cycles, contact=contact
    )
    com = analyze_center_of_mass(recording, contact)
    vq = assess_video_quality(sequence)
    coverage = measure_tracking_coverage(sequence)
    validity = build_result_validity(
        biomechanical=ba,
        contact=contact,
        cycles=cycles,
        duration_s=float(frames[-1].timestamp_s - frames[0].timestamp_s) if frames else 0.0,
        n_pose_frames=len(frames),
    )
    summary = build_analysis_summary(
        source=str(VIDEO),
        biomechanical=ba,
        estimated_vgrf=None,
        contact=contact,
        cycles=cycles,
    )
    windows = timeline_windows(sequence, source_frame_count=len(frames))
    audit_path = write_frame_audit_csv(
        list(iter_frame_audit_rows(sequence, clip="normal", contact=contact)),
        OUT / "frame_audit.csv",
    )

    reason_counts = Counter(_gap_class(f) for f in frames)
    runs = _gap_runs(frames)
    long_runs = [r for r in runs if r["n_frames"] >= 8]
    long_runs.sort(key=lambda r: r["n_frames"], reverse=True)

    n_detected_presence = sum(
        1
        for f in frames
        if f.detected or (f.identity_locked and f.keypoints)
    )
    n_usable = sum(1 for f in frames if f.detected)
    n_candidates = len(getattr(cycles, "cycle_candidates", None) or [])
    n_accepted = len(getattr(cycles, "cycles", None) or [])
    n_left = sum(
        1
        for c in cycles.cycles
        if str(c.boundary_side).lower().startswith("left")
    )
    n_right = sum(
        1
        for c in cycles.cycles
        if str(c.boundary_side).lower().startswith("right")
    )

    com_y = com.positions[:, 1] if len(com.per_frame) else np.array([])
    conf = np.array([f.confidence for f in com.per_frame], dtype=float)
    if com_y.size:
        masked = com_y.copy()
        masked[conf <= 0.05] = np.nan
        t_plot, y_plot = series_with_gap_breaks(
            com.timestamps, masked, fps=com.fps
        )
        n_com_breaks = int(np.isnan(y_plot).sum())
        n_com_valid_zero = int(
            np.sum((conf > 0.05) & np.isfinite(com_y) & (np.abs(com_y) < 1e-12))
        )
        n_com_missing = int(np.sum(conf <= 0.05))
    else:
        n_com_breaks = n_com_valid_zero = n_com_missing = 0

    by = {int(f.frame_index): f for f in frames}
    clip_path = PROBE / "overlap_clip_0130_0200.mp4"
    writer = None
    if VIDEO.is_file():
        wanted = set(STILL_FRAMES)
        with VideoReader(str(VIDEO)) as reader:
            meta = reader.metadata
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            for vf in reader.iter_frames():
                pf = by.get(vf.index)
                if vf.index in wanted:
                    over = _overlay_frame(vf.bgr, pf)
                    cv2.imwrite(str(PROBE / f"overlay_{vf.index:04d}.png"), over)
                    wanted.discard(vf.index)
                if CLIP_START <= vf.index <= CLIP_END:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(clip_path),
                            fourcc,
                            meta.fps,
                            (meta.width, meta.height),
                        )
                    if writer is not None and writer.isOpened():
                        writer.write(_overlay_frame(vf.bgr, pf))
                if vf.index > CLIP_END and not wanted:
                    break
        if writer is not None:
            writer.release()

    after = {
        "n_frames": len(frames),
        "detected_frames": n_detected_presence,
        "usable_frames": n_usable,
        "pose_ok": n_usable,
        "identity_locked": sum(1 for f in frames if f.identity_locked),
        "tracking_unavailable": sum(
            1
            for f in frames
            if str(f.tracking_status) == "unavailable"
            or f.processing_status() == "subject_lock_skip"
        ),
        "candidate_cycles": n_candidates,
        "accepted_cycles": n_accepted,
        "accepted_left": n_left,
        "accepted_right": n_right,
        "summary_cadence_spm": summary_cadence_spm(cycles.metrics, cycles),
        "raw_cadence_spm": cycles.metrics.cadence_steps_per_min,
        "cadence_field": None if summary.cadence is None else summary.cadence.to_dict(),
        "pose_coverage_field": None
        if summary.pose_coverage is None
        else summary.pose_coverage.to_dict(),
        "cycle_evidence_field": None
        if summary.cycle_evidence is None
        else summary.cycle_evidence.to_dict(),
        "hud": None
        if coverage is None
        else {
            "detected_frames": coverage.detected_frames,
            "usable_gait_frames": coverage.usable_gait_frames,
            "detection_pct": coverage.detection_pct,
            "usable_gait_pct": coverage.usable_gait_pct,
        },
        "video_quality_checks": vq.checks,
        "validity_cycle": validity.cycle_detection.to_dict(),
        "com_missing_samples": n_com_missing,
        "com_valid_zeros": n_com_valid_zero,
        "com_plot_nan_breaks": n_com_breaks,
        "timeline": windows.as_dict(),
        "reason_counts": dict(reason_counts),
        "long_gap_runs": long_runs[:12],
        "frames": _landmark_probe(sequence),
    }

    payload = {
        "started_utc": started,
        "finished_utc": _utc_now(),
        "poses_path": str(poses_path),
        "mediapipe_rerun": False,
        "before": {
            "n_frames": before.get("n_frames"),
            "pose_ok": before.get("pose_ok"),
            "identity_locked": before.get("identity_locked"),
            "tracking_unavailable": before.get("tracking_unavailable"),
            "note": (
                "Walker-seed dump after subject-selection fix. "
                "pose_ok is gait-usable frames, not identity-present frames."
            ),
        },
        "after": after,
        "coverage_delta": {
            "pose_ok_before": before.get("pose_ok"),
            "pose_ok_after": n_usable,
            "detected_after": n_detected_presence,
            "usable_after": n_usable,
            "note": (
                "Tracking thresholds were not relaxed. Pose-ok coverage is "
                "unchanged. Detected now counts identity-present frames "
                "(including filter_skip with a locked walker); usable remains "
                "gait-ready pose_ok frames."
            ),
        },
        "audit_csv": str(audit_path),
        "overlay_dir": str(PROBE),
        "overlap_clip": str(clip_path) if clip_path.is_file() else None,
    }
    (PROBE / "missing_data_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (OUT / "missing_data_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["coverage_delta"], indent=2))
    print("reason_counts", json.dumps(after["reason_counts"]))
    print("long_gaps", json.dumps(long_runs[:5]))
    print("audit", audit_path)
    print("probe", PROBE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
