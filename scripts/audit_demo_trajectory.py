#!/usr/bin/env python3
"""Audit Overview 3D path length / displacement / timing on the three demos.

Does not change formulas. Recomputes the same helpers the HUD uses.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/output/example_v2_verification/trajectory_timing_audit.json"
DEMOS = (
    ("abnormal", "abnormal_gait_v2.mp4"),
    ("normal", "normal_gait_v2.mp4"),
    ("athletic", "fast_walking_v2.mp4"),
)
JOINTS = ("pelvis", "right_knee", "right_ankle", "left_knee", "left_ankle")


def _latest_poses(filename: str) -> Path | None:
    from stablewalk import config

    matches = []
    for path in config.POSES_DIR.glob("*_poses.json"):
        try:
            with path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            continue
        src = str(raw.get("source_video") or "")
        if Path(src).name == filename:
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _fmt_cm(meters: float | None) -> str:
    if meters is None or not math.isfinite(meters):
        return "N/A"
    return f"{meters * 100.0:.1f} cm"


def main() -> int:
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.analysis.joint_path_trace import (
        gap_aware_prefix_lengths_m,
        polyline_length_m,
        trace_joint_path,
    )
    from stablewalk.io.pose_loader import detected_frame_indices, load_pose_sequence
    from stablewalk.models.joint_registry import ROOT_JOINT_ID
    from stablewalk.pose.reconstruction import compute_root_trajectory_point
    from stablewalk.pose.tracking_quality import audit_pose_sequence
    from stablewalk.ui.overview_path_display import cumulative_path_length_m, displacement_m
    from stablewalk.ui.viewers.dof_trajectory_3d import (
        _joint_path_with_times,
        stature_display_scale,
    )

    report: dict = {"videos": [], "definitions": {
        "path_length": (
            "Sum of Euclidean 3D segments between valid consecutive samples. "
            "A pair is consecutive only when dt <= 1.5/fps. Gaps are not connected."
        ),
        "endpoint_displacement": (
            "Euclidean distance from the first valid sample to the current/last "
            "valid sample in the same coordinate frame. Not world travel when "
            "the frame is pelvis-centered (root-relative)."
        ),
        "default_joint": "right_knee (demo_default_dof_item)",
        "default_coord": "Root-relative (hip-centered reconstructed XYZ)",
        "units": "stature-scaled centimetre estimates; not mocap-calibrated",
    }}

    for key, filename in DEMOS:
        poses_path = _latest_poses(filename)
        entry: dict = {"key": key, "filename": filename, "poses_path": None if poses_path is None else str(poses_path)}
        if poses_path is None:
            entry["error"] = "no pose JSON"
            report["videos"].append(entry)
            continue
        seq = load_pose_sequence(poses_path)
        detected = detected_frame_indices(seq)
        audit = audit_pose_sequence(seq)
        rec = pose_sequence_to_gait_motion(seq, detected_only=True)
        scale = stature_display_scale(rec)
        times = [float(s.time_s) for s in rec.snapshots]
        src_ids = [int(s.frame_index) for s in rec.snapshots]
        first_ts = times[0] if times else None
        last_ts = times[-1] if times else None
        duration = float(rec.duration_s)
        naive_duration = (len(seq.frames) - 1) / max(seq.fps, 1e-6) if seq.frames else 0.0
        compressed_duration = (rec.frame_count - 1) / max(rec.fps, 1e-6) if rec.frame_count > 1 else 0.0

        # Camera-frame pelvis path before hip centering (image-normalized, not metres).
        roots = []
        for fr in seq.frames:
            if not fr.detected or not fr.keypoints:
                roots.append((fr.frame_index, float(fr.timestamp_s), None))
                continue
            pt = compute_root_trajectory_point(list(fr.keypoints))
            roots.append((fr.frame_index, float(fr.timestamp_s), pt))

        joint_rows = {}
        for joint in JOINTS:
            jid = ROOT_JOINT_ID if joint == "pelvis" else joint
            path = _joint_path_with_times(rec, jid, float(rec.frame_count - 1), position_scale=scale)
            pts = [p for p, _t in path]
            ts = [t for _p, t in path]
            naive = polyline_length_m(pts) if pts else None
            prefixes = gap_aware_prefix_lengths_m(path, fps=rec.fps) if path else ()
            gap_len = float(prefixes[-1]) if prefixes else None
            disp = displacement_m(pts)
            hud_len = cumulative_path_length_m(pts)
            trace = trace_joint_path(rec, jid, rec.frame_count - 1, position_scale=scale)
            first = pts[0] if pts else None
            last = pts[-1] if pts else None
            joint_rows[joint] = {
                "n_samples": len(pts),
                "path_length_cm_gap_aware": None if gap_len is None else round(gap_len * 100.0, 1),
                "path_length_cm_naive_connect": None if naive is None else round(naive * 100.0, 1),
                "path_length_cm_overview_helper": None if hud_len is None else round(hud_len * 100.0, 1),
                "endpoint_displacement_cm": None if disp is None else round(disp * 100.0, 1),
                "skipped_gaps": trace.skipped_gap_count,
                "valid_segments": trace.valid_segment_count,
                "first_xyz_cm": None if first is None else [round(first.x * 100, 2), round(first.y * 100, 2), round(first.z * 100, 2)],
                "last_xyz_cm": None if last is None else [round(last.x * 100, 2), round(last.y * 100, 2), round(last.z * 100, 2)],
                "matches_screenshot_path": False,
            }

        # Screenshot match on default joint.
        target = {"abnormal": 215.0, "normal": 545.2, "athletic": 4419.6}[key]
        for joint, row in joint_rows.items():
            for field in ("path_length_cm_gap_aware", "path_length_cm_naive_connect", "path_length_cm_overview_helper"):
                val = row[field]
                if val is not None and abs(val - target) <= 0.15:
                    row["matches_screenshot_path"] = True
                    row["matched_via"] = field

        entry.update({
            "source_frames_in_pose_json": len(seq.frames),
            "detected_frames": len(detected),
            "usable_gait_frames": audit.usable_gait_frames,
            "recording_snapshots": rec.frame_count,
            "fps": seq.fps,
            "stature_display_scale": round(scale, 4),
            "first_timestamp_s": first_ts,
            "last_timestamp_s": last_ts,
            "duration_from_last_timestamp_s": round(duration, 3),
            "duration_if_index_over_fps_s": round(naive_duration, 3),
            "duration_if_compressed_index_s": round(compressed_duration, 3),
            "snapshot_frame_index_first": src_ids[0] if src_ids else None,
            "snapshot_frame_index_last": src_ids[-1] if src_ids else None,
            "timestamps_monotonic": all(times[i] <= times[i + 1] for i in range(len(times) - 1)),
            "max_dt_s": round(max((times[i + 1] - times[i] for i in range(len(times) - 1)), default=0.0), 4),
            "coordinate_system": rec.coordinate_system,
            "joints": joint_rows,
            "screenshot_path_length_cm": target,
            "screenshot_duration_s": {"abnormal": 27.03, "normal": 9.98, "athletic": 27.24}[key],
        })
        report["videos"].append(entry)
        print(
            f"{filename}: source={len(seq.frames)} detected={len(detected)} "
            f"usable={audit.usable_gait_frames} rec={rec.frame_count} "
            f"t0={first_ts:.3f} t1={last_ts:.3f} dur={duration:.3f}s "
            f"scale={scale:.3f}",
            flush=True,
        )
        for joint, row in joint_rows.items():
            mark = " <== screenshot" if row.get("matches_screenshot_path") else ""
            print(
                f"  {joint:12s} gap={row['path_length_cm_gap_aware']} "
                f"naive={row['path_length_cm_naive_connect']} "
                f"disp={row['endpoint_displacement_cm']} "
                f"gaps={row['skipped_gaps']}{mark}",
                flush=True,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
