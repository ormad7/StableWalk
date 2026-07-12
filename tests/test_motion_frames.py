"""Tests for global vs root-relative motion frame decomposition."""

from __future__ import annotations

import numpy as np

from stablewalk.analysis.motion_frames import (
    GaitProgressionFrame,
    estimate_gait_progression_frame,
    forward_step_displacement,
)


def test_forward_axis_from_trajectory():
    # Simulated side-view walk: large lateral (image-x) motion, small vertical bob
    t = np.linspace(0, 1, 50)
    positions = np.column_stack([
        t * 2.0 - 1.0,  # dominant axis
        0.02 * np.sin(t * 8 * np.pi),
        0.01 * np.cos(t * 4 * np.pi),
    ])
    frame = estimate_gait_progression_frame(positions)
    assert frame.confidence > 0.2
    # Forward should align with dominant displacement (X in this synthetic path)
    disp = positions[-1] - positions[0]
    disp = disp / np.linalg.norm(disp)
    assert float(np.dot(frame.forward, disp)) > 0.85


def test_forward_displacement_not_counted_as_ml_sway():
    positions = np.column_stack([
        np.linspace(-0.5, 0.5, 40),
        np.zeros(40),
        np.zeros(40),
    ])
    frame = estimate_gait_progression_frame(positions)
    origin = positions[0]
    ml = [frame.project(p - origin)[2] for p in positions]
    fwd = [frame.project(p - origin)[0] for p in positions]
    assert np.ptp(ml) < 0.05
    assert np.ptp(fwd) > 0.8


def test_forward_step_displacement_uses_gait_frame():
    from stablewalk.analysis.motion_frames import MotionFrameSeries

    positions = [np.array([i * 0.1, 0.0, 0.0]) for i in range(10)]
    series = MotionFrameSeries(
        frame_indices=list(range(10)),
        timestamps_s=[i / 30.0 for i in range(10)],
        global_pelvis=positions,
        pelvis_confidence=[1.0] * 10,
        progression=GaitProgressionFrame(
            forward=np.array([1.0, 0.0, 0.0]),
            vertical=np.array([0.0, 1.0, 0.0]),
            mediolateral=np.array([0.0, 0.0, 1.0]),
            confidence=1.0,
        ),
    )
    steps = forward_step_displacement(series, [0, 5, 9], leg_length=0.5)
    assert len(steps) == 2
    assert steps[0] > 0.9
