"""Binary-contact HS–HS intervals are candidates until known stance and swing exist."""

from __future__ import annotations

from stablewalk.analysis.gait_cycle_analysis import (
    FootLandmarkSample,
    FrameContactState,
    GaitEvent,
    detect_gait_cycles,
    enumerate_cycle_candidates,
)
from stablewalk.analysis.gait_phase_classification import FLIGHT_CLEARANCE_THRESHOLD_M


def _sample(clearance: float, vis: float = 0.9) -> FootLandmarkSample:
    return FootLandmarkSample(
        heel_clearance_m=clearance,
        toe_clearance_m=clearance,
        ankle_clearance_m=clearance,
        foot_clearance_m=clearance,
        heel_velocity_m_s=0.0,
        toe_velocity_m_s=0.0,
        ankle_velocity_m_s=0.0,
        visibility=vis,
    )


def _frames_near_ground_off() -> list[FrameContactState]:
    """Left contact bit 1 for 8 frames, 0 for 12, clearance always below the flight gate."""
    frames: list[FrameContactState] = []
    for i in range(21):
        binary = 1 if i < 8 else 0
        sample = _sample(0.04)
        frames.append(
            FrameContactState(
                frame_index=i,
                time_s=i / 10.0,
                left_contact=binary,
                right_contact=0,
                phase="LEFT_STANCE" if binary else "UNCERTAIN",
                left=sample,
                right=_sample(0.04),
            )
        )
    return frames


def test_near_ground_binary_cycles_are_candidates_not_valid() -> None:
    events = [
        GaitEvent("left_heel_strike", 0, 0.0, "left"),
        GaitEvent("left_heel_strike", 20, 2.0, "left"),
    ]
    frames = _frames_near_ground_off()
    candidates = enumerate_cycle_candidates(events, per_frame=frames)
    assert len(candidates) == 1
    assert candidates[0].accepted is False
    assert "no_known_swing" in candidates[0].acceptance_reason
    cycles = detect_gait_cycles(events, duration_s=2.0, per_frame=frames)
    assert cycles == []


def test_known_stance_and_confirmed_swing_are_accepted() -> None:
    events = [
        GaitEvent("left_heel_strike", 0, 0.0, "left"),
        GaitEvent("left_heel_strike", 20, 2.0, "left"),
    ]
    frames: list[FrameContactState] = []
    for i in range(21):
        binary = 1 if i < 8 else 0
        clearance = 0.01 if binary else FLIGHT_CLEARANCE_THRESHOLD_M + 0.02
        sample = _sample(clearance)
        frames.append(
            FrameContactState(
                frame_index=i,
                time_s=i / 10.0,
                left_contact=binary,
                right_contact=0,
                phase="LEFT_STANCE" if binary else "FLIGHT",
                left=sample,
                right=_sample(clearance),
            )
        )
    candidates = enumerate_cycle_candidates(events, per_frame=frames)
    assert candidates[0].accepted is True
    assert candidates[0].n_known_stance >= 1
    assert candidates[0].n_known_swing >= 1
    cycles = detect_gait_cycles(events, duration_s=2.0, per_frame=frames)
    assert len(cycles) == 1
    assert cycles[0].boundary_side == "left"


def test_duration_only_path_preserved_without_per_frame() -> None:
    events = [
        GaitEvent("left_heel_strike", 0, 0.0, "left"),
        GaitEvent("left_heel_strike", 10, 1.0, "left"),
        GaitEvent("left_heel_strike", 20, 2.0, "left"),
    ]
    cycles = detect_gait_cycles(events, duration_s=2.0)
    assert len(cycles) == 2


def test_accepted_left_and_right_cycles_are_both_published() -> None:
    events = [
        GaitEvent("left_heel_strike", 0, 0.0, "left"),
        GaitEvent("right_heel_strike", 5, 0.5, "right"),
        GaitEvent("left_heel_strike", 10, 1.0, "left"),
        GaitEvent("right_heel_strike", 15, 1.5, "right"),
        GaitEvent("left_heel_strike", 20, 2.0, "left"),
        GaitEvent("right_heel_strike", 25, 2.5, "right"),
    ]
    frames: list[FrameContactState] = []
    for i in range(26):
        left_on = int((i % 10) < 5)
        right_on = int((i % 10) >= 5 or (i % 10) == 0)
        left_clear = 0.01 if left_on else FLIGHT_CLEARANCE_THRESHOLD_M + 0.02
        right_clear = 0.01 if right_on else FLIGHT_CLEARANCE_THRESHOLD_M + 0.02
        frames.append(
            FrameContactState(
                frame_index=i,
                time_s=i / 10.0,
                left_contact=left_on,
                right_contact=right_on,
                phase="DOUBLE_SUPPORT" if left_on and right_on else "LEFT_STANCE",
                left=_sample(left_clear),
                right=_sample(right_clear),
            )
        )
    cycles = detect_gait_cycles(events, duration_s=2.5, per_frame=frames)
    sides = [c.boundary_side for c in cycles]
    assert sides.count("left") == 2
    assert sides.count("right") == 2


def test_unsupported_pose_gap_is_rejected_not_bridged() -> None:
    events = [
        GaitEvent("left_heel_strike", 0, 0.0, "left"),
        GaitEvent("left_heel_strike", 20, 2.0, "left"),
    ]
    frames: list[FrameContactState] = []
    for i in range(21):
        binary = 1 if i < 8 else 0
        clearance = 0.01 if binary else FLIGHT_CLEARANCE_THRESHOLD_M + 0.02
        frames.append(
            FrameContactState(
                frame_index=i,
                time_s=i / 10.0,
                left_contact=binary,
                right_contact=0,
                phase="LEFT_STANCE" if binary else "FLIGHT",
                left=_sample(clearance),
                right=_sample(clearance),
            )
        )
    pose_times = [i / 10.0 for i in range(21)]
    pose_valid = [i < 6 or i >= 14 for i in range(21)]
    candidates = enumerate_cycle_candidates(
        events,
        per_frame=frames,
        pose_times=pose_times,
        pose_valid=pose_valid,
        fps=10.0,
    )
    assert len(candidates) == 1
    assert candidates[0].accepted is False
    assert "unsupported_pose_gap" in candidates[0].acceptance_reason
    assert candidates[0].longest_internal_gap_s is not None
    assert candidates[0].longest_internal_gap_s > 0.25
    cycles = detect_gait_cycles(
        events,
        duration_s=2.0,
        per_frame=frames,
        pose_times=pose_times,
        pose_valid=pose_valid,
        fps=10.0,
    )
    assert cycles == []
