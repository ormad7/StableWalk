"""
Single entry point for video → pose → enrichment → gait/advanced analysis.

Avoids duplicating pipeline logic across Streamlit, scripts, and future APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stablewalk.analysis.advanced.pipeline import AdvancedGaitReport, analyze_gait_advanced
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.analysis.report import GaitAnalysisReport
from stablewalk.models.pose_data import PoseSequence
from stablewalk.pose.estimation import PoseEstimator


@dataclass
class VideoAnalysisResult:
    sequence: PoseSequence
    overlay_path: Path | None
    gait: GaitAnalysisReport
    advanced: AdvancedGaitReport


def analyze_video_file(
    video_path: str | Path,
    *,
    work_dir: str | Path,
    body_mass_kg: float = 70.0,
    max_frames: int | None = None,
    write_overlay: bool = True,
    source_id: str = "",
) -> VideoAnalysisResult:
    """
    Decode video, estimate pose, enrich, and run standard + advanced analysis once.
    """
    video_path = Path(video_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    overlay_path: Path | None = work_dir / "overlay.mp4" if write_overlay else None

    with PoseEstimator(
        model_variant="lite",
        video_mode=True,
        require_full_body=False,
    ) as estimator:
        sequence = estimator.process_video(
            video_path,
            max_frames=max_frames,
            enrich_gait=False,
        )
        enrich_pose_sequence(sequence)
        if write_overlay and overlay_path is not None:
            estimator.write_overlay_video(sequence, video_path, overlay_path)

    sid = source_id or video_path.name
    advanced = analyze_gait_advanced(sequence, body_mass_kg=body_mass_kg, source_id=sid)

    return VideoAnalysisResult(
        sequence=sequence,
        overlay_path=overlay_path if overlay_path and overlay_path.is_file() else None,
        gait=advanced.gait,
        advanced=advanced,
    )
