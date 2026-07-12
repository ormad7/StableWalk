"""
Label-blind gait comparison validation for the three StableWalk demo videos.

Runs the same analysis pipeline on each video independently. Ground-truth labels
(abnormal / normal / athletic) are used only in the comparison report output.

Outputs:
  data/output/reports/gait_comparison.csv
  data/output/reports/gait_comparison.json

Usage:
  python scripts/gait_comparison_validation.py
  python scripts/gait_comparison_validation.py --use-cached-poses
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.analysis.gait_comparison_validation import (
    COMPARISON_VIDEOS,
    VideoComparisonRecord,
    analyze_motion_label_blind,
    build_comparison_report,
    format_interpretation_text,
    format_terminal_table,
    write_comparison_csv,
    write_comparison_json,
)
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.video_processing import VideoProcessor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("gait_comparison_validation")


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


def _analyze_video(
    ground_truth_label: str,
    video_stem: str,
    *,
    use_cached_poses: bool,
    anonymize_source: bool,
) -> VideoComparisonRecord:
    """
    Analyze one video. ``ground_truth_label`` is stored for comparison only.
    """
    video_path = _resolve_video(video_stem)
    poses_path = _load_or_estimate_poses(video_path, use_cached=use_cached_poses)
    sequence = load_pose_sequence(poses_path)
    enrich_pose_sequence(sequence)

    if anonymize_source:
        sequence = copy.deepcopy(sequence)
        sequence.source_video = "anonymous_validation_clip.mp4"

    sections, stability = analyze_motion_label_blind(sequence)
    return VideoComparisonRecord(
        ground_truth_label=ground_truth_label,
        video_stem=video_stem,
        sections=sections,
        stability_result=stability,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Label-blind gait comparison validation (3 demo videos)"
    )
    parser.add_argument(
        "--use-cached-poses",
        action="store_true",
        help="Use existing data/output/poses/{stem}_poses.json when present",
    )
    parser.add_argument(
        "--anonymize-source",
        action="store_true",
        help="Replace source_video before scoring (extra filename isolation check)",
    )
    args = parser.parse_args()

    records: list[VideoComparisonRecord] = []
    for label, stem in COMPARISON_VIDEOS:
        logger.info("Analyzing %s (%s) — label not passed to scoring", stem, label)
        records.append(
            _analyze_video(
                label,
                stem,
                use_cached_poses=args.use_cached_poses,
                anonymize_source=args.anonymize_source,
            )
        )

    report = build_comparison_report(records)
    csv_path = config.REPORTS_DIR / "gait_comparison.csv"
    json_path = config.REPORTS_DIR / "gait_comparison.json"
    write_comparison_csv(report, csv_path)
    write_comparison_json(report, json_path)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    print(format_terminal_table(records))
    print(format_interpretation_text(report))
    print("")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
