"""
Enrich pose frames with positions, 3D-ready coords, velocities, and gait labels.
"""

from __future__ import annotations

from stablewalk.coordinates_3d import (
    build_skeleton_3d_ready,
    extract_positions,
    extract_positions_xy,
    normalize_to_hip_center,
)
from stablewalk.gait_events import GaitCycleAnnotation, GaitEvent, analyze_gait_sequence
from stablewalk.kinematics import attach_sequence_velocities
from stablewalk.models.pose_data import PoseFrame, PoseSequence
from stablewalk.skeleton_3d_model import sequence_skeleton_scale


def enrich_pose_frame(frame: PoseFrame, *, uniform_scale: float | None = None) -> None:
    """Attach position tracking and hip-centered 3D skeleton."""
    if not frame.detected or not frame.keypoints:
        frame.positions = {}
        frame.positions_xy = {}
        frame.positions_normalized = {}
        frame.skeleton_3d = {}
        return

    frame.positions = extract_positions(frame.keypoints)
    frame.positions_xy = extract_positions_xy(frame.keypoints)
    frame.skeleton_3d = build_skeleton_3d_ready(
        frame.keypoints,
        uniform_scale=uniform_scale,
    )
    frame.positions_normalized = {
        name: {"x": j["x"], "y": j["y"], "z": j["z"]}
        for name, j in frame.skeleton_3d.get("joints", {}).items()
    }


def enrich_pose_sequence(sequence: PoseSequence) -> tuple[list[GaitEvent], dict[int, GaitCycleAnnotation]]:
    """
    Full post-processing: velocities, positions, 3D skeleton, gait events.
    """
    detected_kps = [f.keypoints for f in sequence.frames if f.detected and f.keypoints]
    uniform_scale = sequence_skeleton_scale(detected_kps) if detected_kps else 1.0

    for frame in sequence.frames:
        enrich_pose_frame(frame, uniform_scale=uniform_scale)

    attach_sequence_velocities(sequence.frames, sequence.fps)
    return analyze_gait_sequence(sequence.frames)
