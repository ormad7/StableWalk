"""
Debug gait contact detection — per-frame CSV + summary for demo videos.

Usage:
  python scripts/debug_gait_contact.py abnormal_gait normal_gait athletic_walking
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.gait_cycle_analysis import (
    analyze_gait_cycles,
    classify_gait_phase,
    raw_contact_desire,
)
from stablewalk.analysis.gait_contact_debug import (
    build_contact_debug_rows,
    export_contact_debug_csv,
    print_gait_validation_report,
)
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose_estimation import PoseEstimator
from stablewalk.video_processing import VideoProcessor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("debug_gait_contact")


def _resolve_video(name: str) -> Path:
    stem = name.replace(".mp4", "")
    for folder in (config.DEMO_VIDEOS_DIR, config.VIDEOS_DIR, config.LEGACY_VIDEOS_DIR):
        for candidate in (folder / f"{stem}.mp4", folder / name):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Video not found: {name}")


def _poses_for_video(video_path: Path, *, max_frames: int | None = None) -> Path:
    run_name = video_path.stem
    poses_path = config.POSES_DIR / f"{run_name}_poses.json"
    if poses_path.is_file():
        return poses_path

    frames_dir = config.FRAMES_DIR / run_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    processor = VideoProcessor()
    result = processor.extract_frames(str(video_path), frames_dir, max_frames=max_frames)
    with PoseEstimator() as est:
        seq = est.process_directory(
            frames_dir,
            source_video=str(video_path),
            fps=result.fps,
            max_frames=max_frames,
        )
        est.save_sequence(seq, poses_path)
    return poses_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug gait contact detection")
    parser.add_argument(
        "videos",
        nargs="*",
        default=["abnormal_gait", "normal_gait", "athletic_walking"],
        help="Demo video stems (default: all three demos)",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    config.ensure_output_dirs()
    debug_dir = config.OUTPUT_DIR / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    for name in args.videos:
        video_path = _resolve_video(name)
        logger.info("Processing %s …", video_path.name)
        poses_path = _poses_for_video(video_path, max_frames=args.max_frames)
        sequence = load_pose_sequence(poses_path)
        recording = pose_sequence_to_gait_motion(sequence)
        result = analyze_gait_cycles(recording)

        stem = video_path.stem
        csv_path = debug_dir / f"{stem}_contact_debug.csv"
        export_contact_debug_csv(recording, result, csv_path)
        logger.info("Wrote %s", csv_path)
        print_gait_validation_report(stem, result, recording)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
