"""
Overview dashboard frame-index consistency checks.

Ensures video, 3D skeleton, foot-to-floor distance, contact state, and gait phase
all reference the same analyzed recording frame at playback time t.
"""

from __future__ import annotations

from dataclasses import dataclass

from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult
from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot


@dataclass(frozen=True)
class OverviewFrameIndices:
    """Frame indices for each Overview readout at one playback instant."""

    current_video_frame: int | None
    current_skeleton_frame: int | None
    current_clearance_frame: int | None
    current_contact_frame: int | None
    current_phase_frame: int | None

    def as_debug_lines(self) -> list[str]:
        return [
            f"current_video_frame={self.current_video_frame}",
            f"current_skeleton_frame={self.current_skeleton_frame}",
            f"current_clearance_frame={self.current_clearance_frame}",
            f"current_contact_frame={self.current_contact_frame}",
            f"current_phase_frame={self.current_phase_frame}",
        ]

    def consistent(self) -> bool:
        values = [
            self.current_video_frame,
            self.current_skeleton_frame,
            self.current_clearance_frame,
            self.current_contact_frame,
            self.current_phase_frame,
        ]
        present = [v for v in values if v is not None]
        if len(present) < 2:
            return True
        return len(set(present)) == 1


def collect_overview_frame_indices(
    *,
    snapshot: SkeletonSnapshot | None,
    gait_result: GaitCycleAnalysisResult | None,
    video_frame_index: int | None = None,
    clearance_frame_index: int | None = None,
) -> OverviewFrameIndices:
    """Build frame index snapshot for debug output and consistency tests."""
    skeleton_frame = snapshot.frame_index if snapshot is not None else None
    contact_frame = None
    phase_frame = None
    if gait_result is not None and snapshot is not None:
        state = gait_result.frame_at(snapshot.frame_index)
        if state is not None:
            contact_frame = state.frame_index
            phase_frame = state.frame_index

    video = video_frame_index if video_frame_index is not None else skeleton_frame
    clearance = (
        clearance_frame_index
        if clearance_frame_index is not None
        else skeleton_frame
    )

    return OverviewFrameIndices(
        current_video_frame=video,
        current_skeleton_frame=skeleton_frame,
        current_clearance_frame=clearance,
        current_contact_frame=contact_frame,
        current_phase_frame=phase_frame,
    )


def assert_overview_frames_consistent(indices: OverviewFrameIndices) -> None:
    """Raise AssertionError when any Overview readout uses a different frame."""
    if indices.consistent():
        return
    lines = "\n".join(indices.as_debug_lines())
    raise AssertionError(
        "Overview frame indices are inconsistent:\n" + lines
    )


def log_overview_frame_consistency_if_enabled(
    gui: object,
    *,
    snapshot: SkeletonSnapshot | None,
    gait_result: GaitCycleAnalysisResult | None,
    clearance_frame_index: int | None = None,
) -> None:
    """Print frame-index debug lines when ``STABLEWALK_GUI_DEBUG`` is enabled."""
    import os

    if os.environ.get("STABLEWALK_GUI_DEBUG", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    video_frame = getattr(gui, "_overview_video_frame_index", None)
    if video_frame is None and snapshot is not None:
        video_frame = snapshot.frame_index
    clearance = clearance_frame_index
    if clearance is None:
        clearance = getattr(gui, "_foot_clearance_frame_index", None)
    indices = collect_overview_frame_indices(
        snapshot=snapshot,
        gait_result=gait_result,
        video_frame_index=video_frame,
        clearance_frame_index=clearance,
    )
    for line in indices.as_debug_lines():
        print(f"[Overview frame consistency] {line}", flush=True)
    if not indices.consistent():
        print(
            "[Overview frame consistency] WARNING: frame indices mismatch",
            flush=True,
        )


def audit_recording_frames(
    recording: GaitMotionRecording,
    gait_result: GaitCycleAnalysisResult,
    *,
    sample_every: int = 5,
) -> list[tuple[int, OverviewFrameIndices]]:
    """
    Sample frames across a recording and verify gait + skeleton alignment.

    Foot clearance uses the same ``snapshot.frame_index`` as the skeleton when
    refreshed from the live player; this audit validates the shared index path.
    """
    rows: list[tuple[int, OverviewFrameIndices]] = []
    n = recording.frame_count
    if n <= 0:
        return rows
    step = max(1, sample_every)
    for index in range(0, n, step):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        indices = collect_overview_frame_indices(
            snapshot=snap,
            gait_result=gait_result,
            video_frame_index=snap.frame_index,
            clearance_frame_index=snap.frame_index,
        )
        assert_overview_frames_consistent(indices)
        rows.append((index, indices))
    return rows
