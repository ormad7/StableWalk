"""
Controlled Motion Score — gait-cycle-normalized joint control quality.

High range of motion with low jerk, high repeatability, and good L/R coordination
scores well. Low ROM with irregular motion scores poorly. ROM magnitude alone
never reduces the score.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from stablewalk.analysis.gait_feature_analysis import (
    GAIT_CYCLE_SAMPLE_COUNT,
    resample_cycle_trajectory,
)
from stablewalk.analysis.stability_metrics import clamp, combine_control_scores, continuous_low_better_score
from stablewalk.models.pose_data import JointAngles, PoseFrame

MIN_CYCLE_SAMPLES = 3
MIN_CYCLES = 1
SPIKE_DEG_THRESHOLD = 18.0
SPIKE_MAD_MULTIPLIER = 4.5


@dataclass
class JointControlledMotion:
    """Per-joint or joint-pair controlled motion assessment."""

    key: str
    rom_deg: float | None = None
    repeatability_score: float | None = None
    smoothness_score: float | None = None
    lr_symmetry_score: float | None = None
    spike_resilience_score: float | None = None
    controlled_motion_score: float | None = None
    cycle_count: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "rom_deg": None if self.rom_deg is None else round(self.rom_deg, 2),
            "repeatability_score": self.repeatability_score,
            "smoothness_score": self.smoothness_score,
            "lr_symmetry_score": self.lr_symmetry_score,
            "spike_resilience_score": self.spike_resilience_score,
            "controlled_motion_score": self.controlled_motion_score,
            "cycle_count": self.cycle_count,
            "notes": list(self.notes),
        }


@dataclass
class ControlledMotionResult:
    joints: list[JointControlledMotion] = field(default_factory=list)
    overall_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "joints": [j.to_dict() for j in self.joints],
        }


def _angle_at_frame(frame: PoseFrame | None, fn: Callable[[JointAngles], float | None]) -> float | None:
    if frame is None or frame.joint_angles is None:
        return None
    return fn(frame.joint_angles)


def _extract_cycle_trajectories(
    frames: Sequence[PoseFrame],
    cycles: Sequence[Any],
    angle_fn: Callable[[JointAngles], float | None],
) -> list[np.ndarray]:
    frame_by_idx = {f.frame_index: f for f in frames}
    out: list[np.ndarray] = []
    for cycle in cycles:
        times: list[float] = []
        values: list[float] = []
        for fi in range(int(cycle.start_frame), int(cycle.end_frame) + 1):
            frame = frame_by_idx.get(fi)
            if frame is None:
                continue
            val = _angle_at_frame(frame, angle_fn)
            if val is None:
                continue
            times.append(float(frame.timestamp_s))
            values.append(float(val))
        arr = resample_cycle_trajectory(
            times,
            values,
            t_start=float(cycle.start_time_s),
            t_end=float(cycle.end_time_s),
            n_samples=GAIT_CYCLE_SAMPLE_COUNT,
        )
        if arr is not None and np.isfinite(arr).sum() >= MIN_CYCLE_SAMPLES:
            out.append(arr)
    return out


def _shape_normalize(cycle: np.ndarray) -> np.ndarray:
    ptp = float(np.ptp(cycle))
    if ptp < 1e-6:
        return np.zeros_like(cycle)
    return (cycle - float(np.min(cycle))) / ptp


def _repeatability_score(trajectories: list[np.ndarray]) -> float | None:
    if len(trajectories) < 2:
        return None
    stack = np.vstack(trajectories)
    mean_cycle = np.mean(stack, axis=0)
    rom = max(float(np.ptp(mean_cycle)), 8.0)
    rmses = [float(np.sqrt(np.mean((t - mean_cycle) ** 2))) for t in trajectories]
    norm_rmse = statistics.mean(rmses) / rom
    _, score = continuous_low_better_score(norm_rmse, steepness=5.5, reference=0.35)
    return score


def _smoothness_score(mean_cycle: np.ndarray) -> float | None:
    if mean_cycle.size < 5:
        return None
    rom = max(float(np.ptp(mean_cycle)), 8.0)
    jerk = float(np.mean(np.abs(np.diff(mean_cycle, n=2))))
    norm_j = jerk / rom
    _, score = continuous_low_better_score(norm_j, steepness=4.0, reference=0.55)
    return score


def _lr_symmetry_score(left_mean: np.ndarray, right_mean: np.ndarray) -> float | None:
    if left_mean.size != right_mean.size:
        return None
    l = _shape_normalize(left_mean)
    r = _shape_normalize(right_mean)
    rmse = float(np.sqrt(np.mean((l - r) ** 2)))
    _, score = continuous_low_better_score(rmse, steepness=5.0, reference=0.28)
    return score


def _spike_resilience_score(trajectories: list[np.ndarray]) -> float | None:
    if not trajectories:
        return None
    diffs: list[float] = []
    for traj in trajectories:
        diffs.extend(float(x) for x in np.abs(np.diff(traj)))
    if len(diffs) < 4:
        return None
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(np.asarray(diffs) - med)))
    threshold = max(SPIKE_DEG_THRESHOLD, med + SPIKE_MAD_MULTIPLIER * max(mad, 1e-6))
    spikes = sum(1 for d in diffs if d > threshold)
    frac = spikes / len(diffs)
    _, score = continuous_low_better_score(frac, steepness=8.0, reference=0.08)
    return score


def _assess_single_joint(
    key: str,
    trajectories: list[np.ndarray],
) -> JointControlledMotion:
    result = JointControlledMotion(key=key, cycle_count=len(trajectories))
    if not trajectories:
        result.notes.append("No resampled gait cycles.")
        return result

    mean_cycle = np.mean(np.vstack(trajectories), axis=0)
    result.rom_deg = float(np.ptp(mean_cycle))
    result.repeatability_score = _repeatability_score(trajectories)
    result.smoothness_score = _smoothness_score(mean_cycle)
    result.spike_resilience_score = _spike_resilience_score(trajectories)
    result.controlled_motion_score = combine_control_scores(
        [
            result.repeatability_score,
            result.smoothness_score,
            result.spike_resilience_score,
        ],
        weights=[0.40, 0.35, 0.25],
    )
    return result


def _assess_lr_pair(
    key: str,
    left_traj: list[np.ndarray],
    right_traj: list[np.ndarray],
) -> JointControlledMotion:
    all_traj = left_traj + right_traj
    result = _assess_single_joint(key, all_traj)
    if left_traj and right_traj:
        left_mean = np.mean(np.vstack(left_traj), axis=0)
        right_mean = np.mean(np.vstack(right_traj), axis=0)
        result.lr_symmetry_score = _lr_symmetry_score(left_mean, right_mean)
        result.controlled_motion_score = combine_control_scores(
            [
                result.repeatability_score,
                result.smoothness_score,
                result.lr_symmetry_score,
                result.spike_resilience_score,
            ],
            weights=[0.30, 0.25, 0.30, 0.15],
        )
    return result


def _controlled_motion_without_repeatability(joint: JointControlledMotion) -> float | None:
    """Single-cycle smoothness / symmetry / spike metrics when repeatability is unavailable."""
    if joint.lr_symmetry_score is not None:
        return combine_control_scores(
            [
                joint.smoothness_score,
                joint.lr_symmetry_score,
                joint.spike_resilience_score,
            ],
            weights=[0.35, 0.40, 0.25],
        )
    return combine_control_scores(
        [joint.smoothness_score, joint.spike_resilience_score],
        weights=[0.55, 0.45],
    )


def analyze_controlled_motion(
    frames: Sequence[PoseFrame],
    cycles: Sequence[Any],
) -> ControlledMotionResult:
    """Compute Controlled Motion Scores for knee, hip, and ankle pairs."""
    if not cycles:
        return ControlledMotionResult()

    pairs = (
        (
            "knee",
            lambda a: a.left_knee,
            lambda a: a.right_knee,
        ),
        (
            "hip",
            lambda a: a.left_hip,
            lambda a: a.right_hip,
        ),
        (
            "ankle",
            lambda a: a.left_ankle if a.left_ankle is not None else a.left_ankle_flexion,
            lambda a: a.right_ankle if a.right_ankle is not None else a.right_ankle_flexion,
        ),
    )

    joints: list[JointControlledMotion] = []
    scores: list[float] = []
    for key, fn_l, fn_r in pairs:
        left_t = _extract_cycle_trajectories(frames, cycles, fn_l)
        right_t = _extract_cycle_trajectories(frames, cycles, fn_r)
        if not left_t and not right_t:
            continue
        j = _assess_lr_pair(key, left_t, right_t)
        joints.append(j)
        if j.controlled_motion_score is not None:
            scores.append(j.controlled_motion_score)

    overall = float(statistics.mean(scores)) if scores else None
    return ControlledMotionResult(joints=joints, overall_score=overall)
