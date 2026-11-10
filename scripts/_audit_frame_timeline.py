#!/usr/bin/env python3
"""Read-only audit of Normal / Fast Walking stored frames vs source video."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "validation_outputs" / "frame_timeline_probe.json"
DEMOS = (
    ("normal", "normal_gait_v2.mp4"),
    ("athletic", "fast_walking_v2.mp4"),
)


def _latest_poses(filename: str) -> Path | None:
    from stablewalk import config

    matches = []
    for path in config.POSES_DIR.glob("*_poses.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if Path(str(raw.get("source_video") or "")).name == filename:
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _video_meta(filename: str) -> dict:
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key, probe_demo_timing

    key = "athletic" if filename.startswith("fast") else "normal"
    ex = example_by_key(key)
    path = demo_path(ex) if ex else None
    info = probe_demo_timing(path) if path else {}
    info["path"] = str(path) if path else None
    info["exists"] = bool(path and path.is_file())
    return info


def main() -> int:
    from stablewalk.io.pose_loader import detected_frame_indices, load_pose_sequence
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.pose.tracking_quality import audit_pose_sequence
    from stablewalk.analysis.joint_path_trace import segment_is_connected

    report = {"examples": []}
    for key, filename in DEMOS:
        poses = _latest_poses(filename)
        video = _video_meta(filename)
        entry = {
            "key": key,
            "filename": filename,
            "pose_json": str(poses) if poses else None,
            "video": video,
        }
        if poses is None:
            entry["error"] = "no pose json"
            report["examples"].append(entry)
            continue
        seq = load_pose_sequence(poses)
        ids = [int(f.frame_index) for f in seq.frames]
        times = [float(f.timestamp_s) for f in seq.frames]
        detected = [f for f in seq.frames if f.detected and f.keypoints]
        skipped = [f for f in seq.frames if f.skipped]
        undetected = [f for f in seq.frames if not f.detected]
        holes = []
        if ids:
            want = set(range(min(ids), max(ids) + 1))
            have = set(ids)
            holes = sorted(want - have)
        rec = pose_sequence_to_gait_motion(seq, detected_only=True)
        rec_all = pose_sequence_to_gait_motion(seq, detected_only=False)
        rec_times = [float(s.time_s) for s in rec.snapshots]
        gaps = 0
        for i in range(len(rec_times) - 1):
            if not segment_is_connected(rec_times[i], rec_times[i + 1], fps=rec.fps):
                gaps += 1
        audit = audit_pose_sequence(seq)
        det_idx = detected_frame_indices(seq)
        entry.update(
            {
                "json_frame_count": len(seq.frames),
                "json_fps": seq.fps,
                "json_detected_flag": sum(1 for f in seq.frames if f.detected),
                "json_detected_with_keypoints": len(detected),
                "json_skipped": len(skipped),
                "json_undetected": len(undetected),
                "first_id": ids[0] if ids else None,
                "last_id": ids[-1] if ids else None,
                "ids_unique": len(ids) == len(set(ids)),
                "ids_contiguous": not holes,
                "missing_source_ids_in_json": len(holes),
                "missing_id_examples": holes[:12],
                "first_ts": times[0] if times else None,
                "last_ts": times[-1] if times else None,
                "span_ts": (times[-1] - times[0]) if times else None,
                "first_detected_id": detected[0].frame_index if detected else None,
                "last_detected_id": detected[-1].frame_index if detected else None,
                "first_detected_ts": detected[0].timestamp_s if detected else None,
                "last_detected_ts": detected[-1].timestamp_s if detected else None,
                "detected_span_s": (
                    detected[-1].timestamp_s - detected[0].timestamp_s if detected else None
                ),
                "recording_detected_only_n": rec.frame_count,
                "recording_detected_only_duration_s": rec.duration_s,
                "recording_all_n": rec_all.frame_count,
                "recording_all_duration_s": rec_all.duration_s,
                "pose_indices_n": len(det_idx),
                "timestamp_gaps_detected_only": gaps,
                "usable_gait_frames": audit.usable_gait_frames,
                "source_duration_if_n_over_fps": (len(seq.frames) / seq.fps) if seq.fps else None,
                "video_frame_count": video.get("frame_count"),
                "video_fps": video.get("fps"),
                "video_duration_s": video.get("duration_s"),
            }
        )
        report["examples"].append(entry)
        print(json.dumps(entry, indent=2, default=str), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
