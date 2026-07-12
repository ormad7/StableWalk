#!/usr/bin/env python3
"""Print Overview frame-index consistency for demo videos at sample frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEMO_STEMS = ("abnormal_gait", "normal_gait", "athletic_walking")


def _load(stem: str):
    from stablewalk import config
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
    from stablewalk.io.pose_loader import load_pose_sequence
    from stablewalk.pose.enrichment import enrich_pose_sequence
    from stablewalk.ui.foot_clearance_display import foot_clearance_dashboard_for_panel
    from stablewalk.ui.overview_frame_consistency import (
        assert_overview_frames_consistent,
        collect_overview_frame_indices,
    )

    path = config.POSES_DIR / f"{stem}_poses.json"
    if not path.is_file():
        return None
    sequence = load_pose_sequence(path)
    enrich_pose_sequence(sequence)
    recording = pose_sequence_to_gait_motion(sequence)
    gait = analyze_gait_cycles(recording)
    return recording, gait, foot_clearance_dashboard_for_panel, collect_overview_frame_indices, assert_overview_frames_consistent


def main() -> int:
    parser = argparse.ArgumentParser(description="Overview frame consistency audit")
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Single frame index to print (default: sample several)",
    )
    parser.add_argument(
        "--video",
        default="all",
        help="Demo stem or 'all' (default: all)",
    )
    args = parser.parse_args()

    stems = DEMO_STEMS if args.video == "all" else (args.video.replace(".mp4", ""),)
    issues = 0

    print("# Overview Frame Consistency Audit\n")
    for stem in stems:
        loaded = _load(stem)
        if loaded is None:
            print(f"## {stem}\n\nSkipped — poses not found.\n")
            continue
        (
            recording,
            gait,
            foot_panel_fn,
            collect_fn,
            assert_fn,
        ) = loaded
        print(f"## {stem}\n")
        frames = [args.frame] if args.frame is not None else [0, 10, 25, 50]
        for index in frames:
            if index < 0 or index >= recording.frame_count:
                continue
            snap = recording.snapshot_at(index)
            if snap is None:
                continue
            panel = foot_panel_fn(snap, recording, float(index))
            indices = collect_fn(
                snapshot=snap,
                gait_result=gait,
                video_frame_index=snap.frame_index,
                clearance_frame_index=snap.frame_index,
            )
            try:
                assert_fn(indices)
                status = "OK"
            except AssertionError as exc:
                status = f"MISMATCH — {exc}"
                issues += 1
            print(f"### Frame {index} — {status}")
            for line in indices.as_debug_lines():
                print(f"- {line}")
            if panel is not None:
                left = panel.left.displayed_clearance_cm
                right = panel.right.displayed_clearance_cm
                print(f"- left_clearance_cm={left}")
                print(f"- right_clearance_cm={right}")
            state = gait.frame_at(snap.frame_index)
            if state is not None:
                print(f"- left_contact={state.left_contact}")
                print(f"- right_contact={state.right_contact}")
            print()
    print(f"**Issues:** {issues}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
