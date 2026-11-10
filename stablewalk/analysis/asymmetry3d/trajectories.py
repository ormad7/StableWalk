"""Phase-aligned left/right trajectory comparison.

Corresponding gait cycles are resampled onto a shared 0–100 % grid
(``GAIT_CYCLE_SAMPLE_COUNT`` = 101) via the existing
``resample_cycle_trajectory`` helper. Pointwise left–right differences
are computed only on those aligned curves — never on raw unaligned frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from stablewalk.analysis.gait_cycle_analysis import DetectedGaitCycle
from stablewalk.analysis.gait_feature_analysis import (
    GAIT_CYCLE_SAMPLE_COUNT,
    gait_cycle_percent_grid,
    resample_cycle_trajectory,
)
from stablewalk.analysis.asymmetry3d.indices import (
    ALIGNMENT_EVENT,
    ALIGNMENT_EVENT_LABEL,
    aligned_pearson,
    aligned_rmse,
    circular_phase_difference_pp,
    symmetry_bundle,
)
from stablewalk.analysis.asymmetry3d.kinematics import KinematicBundle

PHASE_GRID = gait_cycle_percent_grid()

PAIRED_SERIES: tuple[tuple[str, str, str, str, str], ...] = (
    # id, left_key, right_key, units, kind  kind in {dof, joint_y, joint_z}
    ("knee_flexion", "left_knee_flexion", "right_knee_flexion", "deg", "dof"),
    ("hip_flexion", "left_hip_flexion", "right_hip_flexion", "deg", "dof"),
    ("ankle_flexion", "left_ankle_flexion", "right_ankle_flexion", "deg", "dof"),
    ("heel_vertical", "left_heel", "right_heel", "body_normalized_m", "joint_y"),
    ("toe_vertical", "left_toe", "right_toe", "body_normalized_m", "joint_y"),
    ("ankle_vertical", "left_ankle", "right_ankle", "body_normalized_m", "joint_y"),
    ("knee_vertical", "left_knee", "right_knee", "body_normalized_m", "joint_y"),
    ("thigh_tilt", "left_thigh", "right_thigh", "deg", "segment"),
    ("shank_tilt", "left_shank", "right_shank", "deg", "segment"),
)


@dataclass
class AlignedTrajectoryPair:
    name: str
    units: str
    phase_pct: tuple[float, ...]
    left_mean: np.ndarray
    right_mean: np.ndarray
    left_std: np.ndarray
    right_std: np.ndarray
    n_left_cycles: int
    n_right_cycles: int
    rmse: float | None
    pearson_r: float | None
    amplitude_left: float | None
    amplitude_right: float | None
    aligned: bool
    peak_left: float | None = None
    peak_right: float | None = None
    peak_phase_left_pct: float | None = None
    peak_phase_right_pct: float | None = None
    alignment_event: str = ALIGNMENT_EVENT
    note: str = (
        "Resampled to 0–100% gait cycle (ipsilateral heel-strike) before "
        "comparison. Unaligned frames were not used."
    )

    @property
    def peak_difference(self) -> float | None:
        if self.peak_left is None or self.peak_right is None:
            return None
        return float(self.peak_left - self.peak_right)

    @property
    def timing_difference_pct(self) -> float | None:
        """Circular peak-timing difference in percentage points."""
        return circular_phase_difference_pp(
            self.peak_phase_left_pct, self.peak_phase_right_pct
        )

    @property
    def timing_difference_units(self) -> str:
        return "percentage points"

    def to_dict(self) -> dict[str, Any]:
        def _list(arr: np.ndarray) -> list[float | None]:
            return [None if not np.isfinite(v) else float(v) for v in arr]

        return {
            "name": self.name,
            "units": self.units,
            "phase_pct": list(self.phase_pct),
            "left_mean": _list(self.left_mean),
            "right_mean": _list(self.right_mean),
            "left_std": _list(self.left_std),
            "right_std": _list(self.right_std),
            "n_left_cycles": self.n_left_cycles,
            "n_right_cycles": self.n_right_cycles,
            "rmse": self.rmse,
            "pearson_r": self.pearson_r,
            "amplitude_left": self.amplitude_left,
            "amplitude_right": self.amplitude_right,
            "peak_left": self.peak_left,
            "peak_right": self.peak_right,
            "peak_phase_left_pct": self.peak_phase_left_pct,
            "peak_phase_right_pct": self.peak_phase_right_pct,
            "peak_difference": self.peak_difference,
            "timing_difference_pct": self.timing_difference_pct,
            "timing_difference_units": self.timing_difference_units,
            "amplitude_indices": symmetry_bundle(
                self.amplitude_left,
                self.amplitude_right,
                publish=self.aligned
                and self.amplitude_left is not None
                and self.amplitude_right is not None,
            ),
            "aligned": self.aligned,
            "alignment_event": self.alignment_event,
            "alignment_label": ALIGNMENT_EVENT_LABEL,
            "note": self.note,
        }


def _extract_series(kin: KinematicBundle, key: str, kind: str) -> np.ndarray:
    if kind == "dof":
        return kin.series_for_dof(key)
    if kind == "segment":
        return kin.series_for_segment(key)
    axis = 1 if kind.endswith("y") else 2
    return kin.series_for_joint(key, axis)


def _resample_cycles(
    times: np.ndarray,
    values: np.ndarray,
    cycles: Sequence[DetectedGaitCycle],
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for cycle in cycles:
        if cycle.duration_s < 0.25 or cycle.end_time_s <= cycle.start_time_s:
            continue
        # np.interp cannot use NaN; drop non-finite samples in-window.
        mask = (
            (times >= cycle.start_time_s)
            & (times <= cycle.end_time_s)
            & np.isfinite(values)
        )
        if int(mask.sum()) < 4:
            continue
        resampled = resample_cycle_trajectory(
            [float(t) for t in times[mask]],
            [float(v) for v in values[mask]],
            t_start=cycle.start_time_s,
            t_end=cycle.end_time_s,
            n_samples=GAIT_CYCLE_SAMPLE_COUNT,
        )
        if resampled is None or resampled.size != GAIT_CYCLE_SAMPLE_COUNT:
            continue
        out.append(np.asarray(resampled, dtype=float))
    return out


def _stack_mean_std(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n = GAIT_CYCLE_SAMPLE_COUNT
    if not curves:
        nan = np.full(n, np.nan)
        return nan, nan
    mat = np.vstack(curves)
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)


def _amplitude(curve: np.ndarray) -> float | None:
    finite = curve[np.isfinite(curve)]
    if finite.size < 8:
        return None
    return float(np.nanmax(finite) - np.nanmin(finite))


def _peak_and_phase(
    curve: np.ndarray,
    phase_pct: tuple[float, ...],
) -> tuple[float | None, float | None]:
    arr = np.asarray(curve, dtype=float)
    if arr.size != len(phase_pct):
        return None, None
    ok = np.isfinite(arr)
    if int(ok.sum()) < 8:
        return None, None
    idx = int(np.nanargmax(arr))
    return float(arr[idx]), float(phase_pct[idx])


def compare_phase_aligned(
    kin: KinematicBundle,
    left_cycles: Sequence[DetectedGaitCycle],
    right_cycles: Sequence[DetectedGaitCycle],
) -> list[AlignedTrajectoryPair]:
    """Compare L/R series after resampling each side's cycles to gait phase."""
    results: list[AlignedTrajectoryPair] = []
    have_sides = len(left_cycles) >= 1 and len(right_cycles) >= 1
    for name, left_key, right_key, units, kind in PAIRED_SERIES:
        left_vals = _extract_series(kin, left_key, kind)
        right_vals = _extract_series(kin, right_key, kind)
        if not have_sides:
            nan = np.full(GAIT_CYCLE_SAMPLE_COUNT, np.nan)
            results.append(
                AlignedTrajectoryPair(
                    name=name,
                    units=units,
                    phase_pct=PHASE_GRID,
                    left_mean=nan,
                    right_mean=nan,
                    left_std=nan,
                    right_std=nan,
                    n_left_cycles=len(left_cycles),
                    n_right_cycles=len(right_cycles),
                    rmse=None,
                    pearson_r=None,
                    amplitude_left=None,
                    amplitude_right=None,
                    aligned=False,
                    alignment_event=ALIGNMENT_EVENT,
                    note=(
                        "Bilateral trajectory comparison skipped: left and right "
                        "ipsilateral heel-strike cycles were not both identified. "
                        "Unaligned frames are not compared."
                    ),
                )
            )
            continue
        left_curves = _resample_cycles(kin.times_s, left_vals, left_cycles)
        right_curves = _resample_cycles(kin.times_s, right_vals, right_cycles)
        l_mean, l_std = _stack_mean_std(left_curves)
        r_mean, r_std = _stack_mean_std(right_curves)
        peak_l, phase_l = _peak_and_phase(l_mean, PHASE_GRID)
        peak_r, phase_r = _peak_and_phase(r_mean, PHASE_GRID)
        results.append(
            AlignedTrajectoryPair(
                name=name,
                units=units,
                phase_pct=PHASE_GRID,
                left_mean=l_mean,
                right_mean=r_mean,
                left_std=l_std,
                right_std=r_std,
                n_left_cycles=len(left_curves),
                n_right_cycles=len(right_curves),
                rmse=aligned_rmse(l_mean, r_mean),
                pearson_r=aligned_pearson(l_mean, r_mean),
                amplitude_left=_amplitude(l_mean),
                amplitude_right=_amplitude(r_mean),
                aligned=len(left_curves) >= 1 and len(right_curves) >= 1,
                peak_left=peak_l,
                peak_right=peak_r,
                peak_phase_left_pct=phase_l,
                peak_phase_right_pct=phase_r,
                alignment_event=ALIGNMENT_EVENT,
                note=(
                    f"{ALIGNMENT_EVENT_LABEL}. "
                    "Knee/hip/ankle series are the canonical analysis flexion "
                    "signals (anatomical knee flexion = 180° − interior HKA)."
                ),
            )
        )
    return results
