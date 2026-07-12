#!/usr/bin/env python3
"""Validate foot clearance display on Abnormal / Normal / Performance demo videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.gait_comparison_validation import COMPARISON_VIDEOS
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.ui.foot_clearance_display import (
    format_foot_clearance_details,
    foot_clearance_dashboard_for_panel,
)
from stablewalk.video_processing import VideoProcessor

UI_LABELS = {
    "abnormal": "Abnormal",
    "normal": "Normal",
    "athletic": "Performance",
}


def _resolve_video(stem: str) -> Path:
    for folder in (config.DEMO_VIDEOS_DIR, config.VIDEOS_DIR, config.LEGACY_VIDEOS_DIR):
        candidate = folder / f"{stem}.mp4"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Demo video not found: {stem}.mp4")


def _load_or_estimate_poses(video_path: Path, *, use_cached: bool) -> Path:
    stem = video_path.stem
    poses_path = config.POSES_DIR / f"{stem}_poses.json"
    if use_cached and poses_path.is_file():
        return poses_path

    frames_dir = config.FRAMES_DIR / stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    processor = VideoProcessor()
    result = processor.extract_frames(str(video_path), frames_dir)
    with PoseEstimator() as est:
        seq = est.process_directory(
            frames_dir,
            source_video=str(video_path),
            fps=result.fps,
        )
        est.save_sequence(seq, poses_path)
    return poses_path


def _sample_frame_indices(frame_count: int, samples: int) -> list[int]:
    if frame_count <= 0:
        return []
    if samples <= 1 or frame_count == 1:
        return [frame_count // 2]
    step = max(1, (frame_count - 1) // (samples - 1))
    indices = list(range(0, frame_count, step))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices[:samples]


def validate_demo(
    label: str,
    stem: str,
    *,
    use_cached_poses: bool,
    samples_per_video: int,
) -> list[str]:
    lines: list[str] = []
    ui = UI_LABELS.get(label, label)
    video_path = _resolve_video(stem)
    poses_path = _load_or_estimate_poses(video_path, use_cached=use_cached_poses)
    sequence = load_pose_sequence(poses_path)
    enrich_pose_sequence(sequence)
    recording = pose_sequence_to_gait_motion(sequence)
    frame_count = recording.frame_count

    lines.append(f"## {ui} ({stem})")
    lines.append(f"Video: {video_path}")
    lines.append(f"Frames: {frame_count}")
    if frame_count == 0:
        lines.append("SKIP — no motion frames")
        lines.append("")
        return lines

    indices = _sample_frame_indices(frame_count, samples_per_video)
    prev_l: str | None = None
    prev_r: str | None = None

    for idx in indices:
        snapshot = recording.snapshots[idx]
        panel = foot_clearance_dashboard_for_panel(
            snapshot,
            recording,
            float(idx),
            prev_left_phase=prev_l,
            prev_right_phase=prev_r,
        )
        if panel is None:
            lines.append(f"Frame {idx}: no panel")
            continue
        prev_l, prev_r = panel.left_phase, panel.right_phase
        lines.append(f"### Frame {idx}")
        lines.append(
            f"L {panel.left.compact_cm} ({panel.left.compact_state}) | "
            f"R {panel.right.compact_cm} ({panel.right.compact_state})"
        )
        lines.append(panel.confidence_label)
        lines.append(
            f"LEFT: {panel.left.current_display} / {panel.left.state_display} "
            f"(max swing {panel.left.max_swing_display}, avg {panel.left.avg_swing_display})"
        )
        if panel.left.unavailable_reason:
            lines.append(f"  LEFT reason: {panel.left.unavailable_reason}")
        lines.append(
            f"RIGHT: {panel.right.current_display} / {panel.right.state_display} "
            f"(max swing {panel.right.max_swing_display}, avg {panel.right.avg_swing_display})"
        )
        if panel.right.unavailable_reason:
            lines.append(f"  RIGHT reason: {panel.right.unavailable_reason}")
        for dbg in panel.debug_lines:
            lines.append(f"  {dbg}")
        lines.append("")

    lines.append("### Details sample (last sampled frame)")
    last_panel = foot_clearance_dashboard_for_panel(
        recording.snapshots[indices[-1]],
        recording,
        float(indices[-1]),
    )
    if last_panel is not None:
        lines.append(format_foot_clearance_details(last_panel))
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate foot clearance on demo videos")
    parser.add_argument(
        "--use-cached-poses",
        action="store_true",
        help="Reuse cached pose JSON instead of re-estimating from MP4",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Frames to sample per video (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to this path (default: stdout only)",
    )
    args = parser.parse_args()

    config.ensure_output_dirs()
    report_lines = [
        "# Foot Clearance Demo Validation",
        "",
        "Samples foot clearance dashboard model on Abnormal / Normal / Performance demos.",
        "",
    ]

    for label, stem in COMPARISON_VIDEOS:
        try:
            report_lines.extend(
                validate_demo(
                    label,
                    stem,
                    use_cached_poses=args.use_cached_poses,
                    samples_per_video=max(1, args.samples),
                )
            )
        except FileNotFoundError as exc:
            report_lines.append(f"## {UI_LABELS.get(label, label)} ({stem})")
            report_lines.append(f"ERROR: {exc}")
            report_lines.append("")

    text = "\n".join(report_lines)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"\nWrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
