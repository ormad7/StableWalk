"""
Print canonical joint coordinates for selected frames.

Usage:
  python scripts/debug_coordinates.py abnormal_gait --frames 0 50 100
  python scripts/debug_coordinates.py normal_gait --audit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.coordinates.coordinate_map import (
    audit_recording_anatomy,
    coordinate_system_map_markdown,
    debug_canonical_joint_positions,
)
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose_estimation import PoseEstimator
from stablewalk.video_processing import VideoProcessor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("debug_coordinates")


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
    parser = argparse.ArgumentParser(description="Debug StableWalk canonical coordinates")
    parser.add_argument(
        "video",
        nargs="?",
        default="abnormal_gait",
        help="Demo video stem (default: abnormal_gait)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="*",
        default=[0, 30, 60, 90],
        help="Frame indices to print (default: 0 30 60 90)",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run anatomical ordering audit and print warnings",
    )
    parser.add_argument(
        "--map",
        action="store_true",
        help="Print coordinate system map markdown",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if args.map:
        print(coordinate_system_map_markdown())
        return 0

    video_path = _resolve_video(args.video)
    poses_path = _poses_for_video(video_path, max_frames=args.max_frames)
    sequence = load_pose_sequence(poses_path)
    recording = pose_sequence_to_gait_motion(sequence)

    print(debug_canonical_joint_positions(recording, args.frames))

    if args.audit:
        warnings = audit_recording_anatomy(recording)
        if warnings:
            print("=== Coordinate audit warnings ===")
            for w in warnings:
                print(f"  - {w}")
        else:
            print("=== Coordinate audit: OK (sampled frames) ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
