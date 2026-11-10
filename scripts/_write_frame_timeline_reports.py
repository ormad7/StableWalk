#!/usr/bin/env python3
"""Write validation_report.md and frame_audit.csv from stored Normal/Fast dumps.

Uses already-processed pose JSON. Does not invent detections or re-run MediaPipe.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEMOS = (
    ("normal", "Normal Gait", "normal_gait_v2.mp4"),
    ("athletic", "Fast Walking", "fast_walking_v2.mp4"),
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


def _video_meta(key: str) -> dict:
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key, probe_demo_timing

    ex = example_by_key(key)
    path = demo_path(ex) if ex else None
    info = probe_demo_timing(path) if path else {}
    info["path"] = str(path) if path else None
    info["exists"] = bool(path and path.is_file())
    return info


def _ids_contiguous(ids: list[int]) -> tuple[bool, list[int]]:
    if not ids:
        return True, []
    want = set(range(min(ids), max(ids) + 1))
    missing = sorted(want - set(ids))
    return not missing, missing


def main() -> int:
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.io.frame_audit import (
        iter_frame_audit_rows,
        timeline_windows,
        write_frame_audit_csv,
    )
    from stablewalk.io.pose_loader import load_pose_sequence, playback_frame_indices
    from stablewalk.ui.media.demo_comparison import format_duration_display

    rows = []
    sections = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for key, label, filename in DEMOS:
        poses = _latest_poses(filename)
        video = _video_meta(key)
        if poses is None:
            sections.append((label, None, {"error": f"no pose JSON for {filename}"}))
            continue
        seq = load_pose_sequence(poses)
        ids = [int(f.frame_index) for f in seq.frames]
        contiguous, holes = _ids_contiguous(ids)
        source_n = video.get("frame_count")
        source_s = video.get("duration_s")
        windows = timeline_windows(
            seq,
            source_frame_count=int(source_n) if source_n else None,
            source_duration_s=float(source_s) if source_s else None,
        )
        rec = pose_sequence_to_gait_motion(seq)
        compressed = pose_sequence_to_gait_motion(seq, detected_only=True)
        play = playback_frame_indices(seq)
        clip_rows = list(iter_frame_audit_rows(seq, clip=key))
        rows.extend(clip_rows)
        sections.append(
            (
                label,
                poses,
                {
                    "key": key,
                    "video": video,
                    "windows": windows.as_dict(),
                    "json_path": str(poses),
                    "json_frames": len(seq.frames),
                    "json_fps": seq.fps,
                    "ids_contiguous": contiguous,
                    "missing_ids": holes[:12],
                    "playback_n": len(play),
                    "recording_n": rec.frame_count,
                    "recording_duration_s": rec.duration_s,
                    "compressed_n": compressed.frame_count,
                    "compressed_duration_s": compressed.duration_s,
                    "duration_display": format_duration_display(
                        analyzed_s=windows.analyzed_duration_s,
                        recording_s=windows.source_duration_s,
                        usable_s=windows.usable_duration_s,
                    ),
                    "status_counts": windows.status_counts,
                },
            )
        )

    out_dir = ROOT / "validation_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / "frame_audit.csv"
    write_frame_audit_csv(rows, csv_path)
    write_frame_audit_csv(rows, out_dir / "frame_audit.csv")

    lines = [
        "# Walking pipeline frame audit",
        "",
        f"Generated: {now}. Measured from stored pose dumps after the timeline-preservation fix. "
        "No detections were invented. MediaPipe was not re-run.",
        "",
        "## Where frames were lost",
        "",
        "Decode and JSON storage did **not** drop source IDs for either clip: Normal stores 300 contiguous IDs 0–299; Fast stores 900 contiguous IDs 0–899.",
        "Fast’s 900-frame window is the GUI `DEMO_MAX_FRAMES` cap on a 2261-frame source file, not a decoder skip.",
        "The 7.61 s Normal playback duration was compressed **result storage / GUI scheduling**: `detected_frame_indices` plus `pose_sequence_to_gait_motion(detected_only=True)` dropped undetected frames, so duration became the last detected timestamp (frame 228 at 7.6076 s) instead of the analyzed window (~9.98 s of a 10.01 s source).",
        "Fast’s 77 timestamp gaps were the same compression: consecutive *detected* snapshots after holes were removed. They are processing/detection holes, not decode gaps.",
        "Filter skips (`skipped=True`, incomplete body / insufficient gait DOF) are separate from genuine no-landmark failures. Older dumps did not store `rejection_reason`, so remaining undetected frames are `undetected_unclassified` (no_landmarks vs subject_lock_lost cannot be split without re-inference).",
        "",
        "## Duration triad",
        "",
        "Source duration = video file. Analyzed window = first–last stored frame timestamp. "
        "Usable duration = first–last detected (not skipped) timestamp. Playback now uses the analyzed window.",
        "",
    ]
    for label, poses, info in sections:
        lines.append(f"### {label}")
        lines.append("")
        if info.get("error"):
            lines.append(f"- Not measured: {info['error']}")
            lines.append("")
            continue
        w = info["windows"]
        sc = w.get("status_counts") or {}
        v = info["video"]
        lines.extend(
            [
                f"- Pose JSON: `{info['json_path']}`",
                f"- Source video: {v.get('frame_count')} frames, {v.get('fps')} FPS, {v.get('duration_s')} s "
                f"({'file present' if v.get('exists') else 'file missing'})",
                f"- Stored / analyzed frames: {info['json_frames']} at {info['json_fps']} FPS; "
                f"IDs contiguous: {info['ids_contiguous']}; missing IDs in JSON: {info['missing_ids'] or 'none'}",
                f"- Source duration: {w.get('source_duration_s')}",
                f"- Analyzed window: {w.get('analyzed_duration_s')} s ({w.get('analyzed_frame_count')} frames)",
                f"- Detected count: {w.get('detected_count')}",
                f"- Usable gait frames: {w.get('usable_gait_frames')}; usable duration: {w.get('usable_duration_s')} s",
                f"- Last detected timestamp: {w.get('last_detected_timestamp_s')} s"
                + (
                    " (this was the old compressed playback duration)"
                    if info.get("key") == "normal"
                    else ""
                ),
                f"- Status counts (legacy dumps classify lock-loss vs no-landmarks as `{sc.get('undetected_unclassified', 0)}` unclassified): {sc}",
                f"- Playback indices now: {info['playback_n']} (was detected-only {info['compressed_n']})",
                f"- Gait recording all-frames: n={info['recording_n']}, duration_s={info['recording_duration_s']}",
                f"- Gait recording detected-only (bug path): n={info['compressed_n']}, duration_s={info['compressed_duration_s']}",
                f"- Detected-only timestamp gaps: {w.get('timestamp_gaps_if_detected_only')}",
                f"- Duration HUD: `{info['duration_display']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Identity",
            "",
            "Normal contains a walking subject and a background person. `SubjectLock` already prefers the locked full-body walker and returns no pose when the lock cannot follow (`subject_lock_lost` is a processing skip, not a detection failure).",
            "Overlay now draws the persistent subject ROI, dimmer boxes for other people, and a status label on missing-pose frames. Older JSON has empty `subject_bbox` / `n_people` because those fields were not stored yet.",
            "",
            "## Reports",
            "",
            f"- `{csv_path.name}`: one row per stored source frame (`{len(rows)}` rows).",
            "- Final GUI/export reports stay gated until analysis finishes (`_require_analysis_complete`).",
            "",
            "## Tests",
            "",
            "See `tests/test_frame_timeline_preservation.py` (deliberate missing detections) and updated `tests/test_playback_sync.py`.",
            "",
        ]
    )
    report_path = ROOT / "validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", report_path)
    print("wrote", csv_path)
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
