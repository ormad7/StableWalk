"""Synthetic gait signals with known mathematical properties (not video pose).

Used to validate autocorrelation, Welch/FFT, feature extraction, and kinematic
event detectors against ground truth. These series are not clinical recordings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from stablewalk.models.gait_motion import (
    DofSample,
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.models.joint_registry import DOF_IDS, JOINT_IDS, JOINT_PARENTS, ROOT_JOINT_ID


@dataclass(frozen=True)
class SyntheticGaitSpec:
    """Ground truth for a generated recording."""

    fps: float
    duration_s: float
    stride_hz: float
    knee_amp_left_deg: float
    knee_amp_right_deg: float
    missing_every: int | None = None

    @property
    def stride_period_s(self) -> float:
        return 1.0 / self.stride_hz

    @property
    def expected_asi_knee_amp(self) -> float:
        a, b = abs(self.knee_amp_left_deg), abs(self.knee_amp_right_deg)
        denom = a + b
        if denom <= 1e-9:
            return 0.0
        return 200.0 * abs(a - b) / denom


def make_sine_gait_recording(spec: SyntheticGaitSpec, *, seed: int = 0) -> GaitMotionRecording:
    """Hip-centred wire with sagittal knee flexion = A sin(2π f t) (+ π on the right)."""
    del seed
    n = int(round(spec.duration_s * spec.fps))
    snaps: list[SkeletonSnapshot] = []
    for i in range(n):
        t = i / spec.fps
        drop = spec.missing_every is not None and spec.missing_every > 0 and i % spec.missing_every == 0
        joints = {}
        for jid in (*JOINT_IDS, ROOT_JOINT_ID):
            phase = math.pi if jid.startswith("right_") else 0.0
            y = 0.15 * math.sin(2 * math.pi * spec.stride_hz * t + phase)
            if jid in ("left_heel", "right_heel", "left_toe", "right_toe"):
                y -= 0.42
            joints[jid] = JointSample(
                joint_id=jid,
                position=Vec3(-0.05 if "left" in jid else 0.05, y, 0.02 * math.sin(phase)),
                parent_id=JOINT_PARENTS.get(jid),
            )
        dofs = {}
        for dof in DOF_IDS:
            if drop:
                angle = None
            elif "left_knee" in dof:
                angle = spec.knee_amp_left_deg * math.sin(2 * math.pi * spec.stride_hz * t)
            elif "right_knee" in dof:
                angle = spec.knee_amp_right_deg * math.sin(
                    2 * math.pi * spec.stride_hz * t + math.pi
                )
            else:
                angle = 8.0 * math.sin(2 * math.pi * spec.stride_hz * t)
            dofs[dof] = DofSample(dof_id=dof, angle_deg=angle)
        snaps.append(
            SkeletonSnapshot(
                frame_index=i,
                time_s=t,
                joints={} if drop else joints,
                dofs=dofs,
                metadata={
                    "detected": not drop,
                    "landmark_visibility": {
                        jid: 0.1 if drop else 0.95 for jid in (*JOINT_IDS, ROOT_JOINT_ID)
                    },
                },
            )
        )
    return GaitMotionRecording(
        source="synthetic_gait",
        fps=spec.fps,
        snapshots=snaps,
        source_kind="mock",
        coordinate_system="sw_canonical_y_up",
        metadata={"knee_angle_storage": "anatomical_flexion"},
    )


def known_frequency_vector(n: int, fps: float, freq_hz: float) -> np.ndarray:
    t = np.arange(n, dtype=float) / fps
    return np.sin(2.0 * math.pi * freq_hz * t)
