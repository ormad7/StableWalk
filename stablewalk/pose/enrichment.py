"""
Enrich pose frames with positions, 3D-ready coords, velocities, and gait labels.
"""

from __future__ import annotations

from stablewalk.pose.coordinates import extract_positions, extract_positions_xy
from stablewalk.pose.events import GaitCycleAnnotation, GaitEvent, analyze_gait_sequence
from stablewalk.pose.kinematics import attach_sequence_velocities
from stablewalk.models.pose_data import PoseFrame, PoseSequence
from stablewalk.pose.skeleton_3d import sequence_skeleton_scale


def attach_gait_phases_to_frames(
    sequence: PoseSequence,
    annotations: dict[int, GaitCycleAnnotation],
) -> None:
    """Write stance/swing labels and per-frame event tags onto ``PoseFrame`` objects."""
    for frame in sequence.frames:
        ann = annotations.get(frame.frame_index)
        if not ann:
            frame.gait_phase = {}
            frame.gait_events = []
            continue
        frame.gait_phase = {"left": ann.phase_left, "right": ann.phase_right}
        frame.gait_events = list(ann.events)


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
    from stablewalk.pose.reconstruction import reconstruct_frame_3d

    skel, _root = reconstruct_frame_3d(
        frame.keypoints,
        uniform_scale=uniform_scale,
        align_upright=True,
    )
    frame.skeleton_3d = skel.to_export_dict()
    frame.positions_normalized = {
        name: {"x": j.x, "y": j.y, "z": j.z}
        for name, j in skel.joints.items()
    }


def enrich_pose_sequence(sequence: PoseSequence) -> tuple[list[GaitEvent], dict[int, GaitCycleAnnotation]]:
    """
    Full post-processing: velocities, positions, 3D skeleton, gait events/phases.

    Attaches ``gait_phase`` dicts on each frame (same contract as ``PoseEstimator``).
    """
    detected_kps = [f.keypoints for f in sequence.frames if f.detected and f.keypoints]
    uniform_scale = sequence_skeleton_scale(detected_kps) if detected_kps else 1.0

    for frame in sequence.frames:
        enrich_pose_frame(frame, uniform_scale=uniform_scale)

    attach_sequence_velocities(sequence.frames, sequence.fps)

    from stablewalk.pose.contact import ContactDetector, attach_foot_contact_to_frames

    contact_result = ContactDetector().detect(sequence)
    attach_foot_contact_to_frames(sequence, contact_result)

    events, annotations = analyze_gait_sequence(sequence.frames)
    attach_gait_phases_to_frames(sequence, annotations)
    # Contact-based stance/swing kept on foot_contact; gait_phase may refine from HS/TO
    for frame in sequence.frames:
        if frame.foot_contact:
            fl = frame.foot_contact.get("left", False)
            fr = frame.foot_contact.get("right", False)
            if fl or fr:
                frame.gait_phase = {
                    "left": "stance" if fl else frame.gait_phase.get("left", "swing"),
                    "right": "stance" if fr else frame.gait_phase.get("right", "swing"),
                }
    return events, annotations
