"""Reprocess normal_gait_v2.mp4 with walker seed and export identity diagnostics."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk.core.pipeline import run_gait_pipeline
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.skeleton import draw_identity_on_bgr, draw_skeleton_on_bgr
from stablewalk.pose.subject_lock import pose_bbox_norm, selection_for_source
from stablewalk.pose.video import VideoReader
from stablewalk.pose.video_source import derive_run_name

VIDEO = ROOT / "data" / "demo_videos" / "normal_gait_v2.mp4"
OUT = ROOT / "validation_outputs" / "normal" / "identity_probe"
MANIFEST = ROOT / "validation_outputs" / "full_rerun_manifest.json"
STILL_FRAMES = (148, 191)
CLIP_START = 130
CLIP_END = 200


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _summarize(sequence) -> dict:
    frames = list(sequence.frames)
    by = {int(f.frame_index): f for f in frames}
    locked = sum(1 for f in frames if f.identity_locked and f.detected)
    unavailable = sum(
        1
        for f in frames
        if str(f.tracking_status) == "unavailable"
        or f.processing_status() == "subject_lock_skip"
    )
    details = {}
    for idx in (80, 148, 191, 220):
        pf = by.get(idx)
        if pf is None:
            details[str(idx)] = None
            continue
        hip = next((kp for kp in pf.keypoints if kp.name == "mid_hip"), None)
        box = pose_bbox_norm(pf.keypoints) if pf.detected and pf.keypoints else None
        details[str(idx)] = {
            "detected": bool(pf.detected),
            "n_people": int(pf.n_people or 0),
            "processing_status": pf.processing_status(),
            "tracking_status": pf.tracking_status,
            "identity_locked": bool(pf.identity_locked),
            "rejection_reason": pf.rejection_reason,
            "hip_x": None if hip is None else round(float(hip.x), 4),
            "subject_bbox": None if box is None else [round(v, 4) for v in box],
            "other_bboxes": [
                [round(v, 4) for v in b] for b in (pf.other_bboxes or [])
            ],
        }
    return {
        "n_frames": len(frames),
        "pose_ok": sum(1 for f in frames if f.detected),
        "identity_locked": locked,
        "tracking_unavailable": unavailable,
        "subject_selection": dict(sequence.subject_selection or {}),
        "frames": details,
    }


def main() -> int:
    if not VIDEO.is_file():
        print("missing video", VIDEO)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    selection = selection_for_source(str(VIDEO))
    run_name = derive_run_name(str(VIDEO), unique_session=True)
    started = _utc_now()
    print("reprocessing", VIDEO.name, "run", run_name, flush=True)
    result = run_gait_pipeline(
        str(VIDEO),
        run_name=run_name,
        validate="quick",
        max_frames=None,
        force_reprocess=True,
        cache_frames=False,
        subject_selection=selection,
    )
    sequence = load_pose_sequence(result.poses_path)
    summary = _summarize(sequence)
    summary["poses_path"] = str(result.poses_path)
    summary["run_name"] = result.run_name
    summary["started_utc"] = started
    summary["finished_utc"] = _utc_now()
    (OUT / "identity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    by = {int(f.frame_index): f for f in sequence.frames}
    wanted = set(STILL_FRAMES)
    clip_path = OUT / "overlap_clip_0130_0200.mp4"
    writer = None
    with VideoReader(str(VIDEO)) as reader:
        meta = reader.metadata
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for vf in reader.iter_frames():
            pf = by.get(vf.index)
            if vf.index in wanted:
                cv2.imwrite(str(OUT / f"src_{vf.index:04d}.png"), vf.bgr)
                over = _overlay_frame(vf.bgr, pf)
                cv2.imwrite(str(OUT / f"overlay_{vf.index:04d}.png"), over)
                wanted.discard(vf.index)
            if CLIP_START <= vf.index <= CLIP_END:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(clip_path),
                        fourcc,
                        meta.fps,
                        (meta.width, meta.height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open writer {clip_path}")
                writer.write(_overlay_frame(vf.bgr, pf))
            if vf.index > CLIP_END and not wanted:
                break
    if writer is not None:
        writer.release()

    if MANIFEST.is_file():
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for rec in payload.get("pipeline") or []:
            if rec.get("filename") == VIDEO.name or rec.get("key") == "normal":
                rec["poses_path"] = str(result.poses_path)
                rec["run_name"] = result.run_name
                rec["analyzed_frame_count"] = result.frame_count
                rec["fps"] = result.fps
                rec["width"] = result.width
                rec["height"] = result.height
                rec["content_key"] = result.content_key
                rec["force_reprocess"] = True
                rec["cache_used"] = False
                rec["subject_selection"] = dict(sequence.subject_selection or {})
                rec["identity_reprocess_utc"] = summary["finished_utc"]
                rec["identity_probe_dir"] = str(OUT)
        MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("overlays", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
