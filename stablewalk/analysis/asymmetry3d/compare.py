"""Bilateral scalar asymmetry from kinematics + gait events.

Spatial quantities (step length) are reported only when metric/view
reliability allows it. Temporal quantities remain available from events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stablewalk.analysis.asymmetry3d.events import GaitEventSummary
from stablewalk.analysis.asymmetry3d.indices import (
    ALIGNMENT_EVENT,
    ALIGNMENT_EVENT_LABEL,
    ASI_DEFINITION,
    ROBINSON_SI_DEFINITION,
    SI_NORM_DEFINITION,
    TIMING_DEFINITION,
    circular_phase_difference_pp,
    symmetry_bundle,
)
from stablewalk.analysis.asymmetry3d.kinematics import (
    KinematicBundle,
    peak_abs,
    rms,
    rom_percentile,
)
from stablewalk.analysis.asymmetry3d.trajectories import AlignedTrajectoryPair
from stablewalk.analysis.gait_cycle_analysis import DetectedGaitCycle
from stablewalk.models.gait_motion import GaitMotionRecording, Vec3

Quality = str


@dataclass
class AsymmetryScalar:
    name: str
    display: str
    units: str
    left: float | None
    right: float | None
    si_percent: float | None
    asi_percent: float | None
    si_norm: float | None
    quality: Quality
    reliable: bool
    note: str
    kind: str = "kinematic_asymmetry"
    formula: str = ""
    n_left_cycles: int = 0
    n_right_cycles: int = 0
    alignment_event: str = ALIGNMENT_EVENT
    comparison_kind: str = "scalar"
    difference: float | None = None
    difference_units: str = ""
    difference_formula: str = ""

    def unavailable_reason(self) -> str | None:
        missing_sides = [
            side
            for side, value in (("left", self.left), ("right", self.right))
            if value is None
        ]
        indices_missing = (
            self.si_percent is None
            and self.asi_percent is None
            and self.si_norm is None
        )
        if (
            not missing_sides
            and not indices_missing
            and self.quality not in {"insufficient", "unavailable"}
        ):
            return None
        if self.note:
            return self.note
        if missing_sides:
            return f"{' and '.join(missing_sides)} value(s) unavailable"
        if indices_missing:
            return "indices unavailable"
        return self.quality

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display": self.display,
            "units": self.units,
            "left": self.left,
            "right": self.right,
            "si_percent": self.si_percent,
            "asi_percent": self.asi_percent,
            "si_norm": self.si_norm,
            "quality": self.quality,
            "reliable": self.reliable,
            "note": self.note,
            "unavailable_reason": self.unavailable_reason(),
            "kind": self.kind,
            "formula": self.formula,
            "formula_id": self.name,
            "n_left_cycles": self.n_left_cycles,
            "n_right_cycles": self.n_right_cycles,
            "alignment_event": self.alignment_event,
            "comparison_kind": self.comparison_kind,
            "difference": self.difference,
            "difference_units": self.difference_units,
            "difference_formula": self.difference_formula,
        }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _mean_duration(intervals, side: str) -> float | None:
    vals = [i.duration_s for i in intervals if i.side == side]
    return _median(vals)


def _mask_cycles(times: np.ndarray, cycles: list[DetectedGaitCycle]) -> np.ndarray:
    mask = np.zeros(times.shape, dtype=bool)
    for cycle in cycles:
        mask |= (times >= cycle.start_time_s) & (times <= cycle.end_time_s)
    return mask


def _side_series(
    values: np.ndarray,
    times: np.ndarray,
    cycles: list[DetectedGaitCycle],
) -> np.ndarray:
    if not cycles:
        return np.full(0, np.nan)
    return values[_mask_cycles(times, cycles)]


def _scalar(
    name: str,
    display: str,
    units: str,
    left: float | None,
    right: float | None,
    *,
    quality: str,
    reliable: bool,
    note: str,
    allow_signed_norm: bool = False,
    kind: str = "kinematic_asymmetry",
    n_left_cycles: int = 0,
    n_right_cycles: int = 0,
    formula: str = "",
    comparison_kind: str = "scalar",
    publish_indices: bool | None = None,
) -> AsymmetryScalar:
    have_sides = left is not None and right is not None
    enough_cycles = n_left_cycles >= 1 and n_right_cycles >= 1
    insufficient = not have_sides or not enough_cycles
    if quality == "unavailable":
        quality_out = "unavailable"
        reliable_out = False
        publish = False
    elif insufficient:
        quality_out = "insufficient"
        reliable_out = False
        publish = False
    else:
        reliable_out = bool(reliable)
        quality_out = quality if reliable_out else f"{quality} · unreliable"
        publish = True if publish_indices is None else bool(publish_indices)
    bundle = symmetry_bundle(
        left, right, allow_signed_norm=allow_signed_norm, publish=publish
    )
    formula_out = formula or (
        f"{ROBINSON_SI_DEFINITION}; {ASI_DEFINITION}; {SI_NORM_DEFINITION}"
    )
    return AsymmetryScalar(
        name=name,
        display=display,
        units=units,
        left=bundle["left"],
        right=bundle["right"],
        si_percent=bundle["si_percent"],
        asi_percent=bundle["asi_percent"],
        si_norm=bundle["si_norm"],
        quality=quality_out,
        reliable=reliable_out,
        note=note,
        kind=kind,
        formula=formula_out,
        n_left_cycles=int(n_left_cycles),
        n_right_cycles=int(n_right_cycles),
        alignment_event=ALIGNMENT_EVENT,
        comparison_kind=comparison_kind,
    )


def _timing_scalar(
    name: str,
    display: str,
    left_pct: float | None,
    right_pct: float | None,
    *,
    quality: str,
    reliable: bool,
    note: str,
    n_left_cycles: int,
    n_right_cycles: int,
) -> AsymmetryScalar:
    enough = n_left_cycles >= 1 and n_right_cycles >= 1
    have = left_pct is not None and right_pct is not None
    ok = bool(reliable and enough and have)
    delta = circular_phase_difference_pp(left_pct, right_pct) if ok else None
    quality_out = quality if ok else "insufficient"
    return AsymmetryScalar(
        name=name,
        display=display,
        units="% gait cycle",
        left=None if left_pct is None else float(left_pct),
        right=None if right_pct is None else float(right_pct),
        si_percent=None,
        asi_percent=None,
        si_norm=None,
        quality=quality_out,
        reliable=ok,
        note=(
            f"{note} Δτ is percentage points on a circular 0–100% cycle "
            f"({ALIGNMENT_EVENT_LABEL}). Robinson SI is not applied — SI would "
            "be a relative percent of the mean phase, not the lag."
        ),
        kind="kinematic_asymmetry",
        formula=TIMING_DEFINITION,
        n_left_cycles=int(n_left_cycles),
        n_right_cycles=int(n_right_cycles),
        alignment_event=ALIGNMENT_EVENT,
        comparison_kind="timing",
        difference=delta,
        difference_units="percentage points",
        difference_formula="Δτ = wrap(τ_L − τ_R) ∈ [−50, +50] percentage points",
    )


def _foot_pos(snap, side: str) -> Vec3 | None:
    for jid in (f"{side}_heel", f"{side}_ankle", f"{side}_toe"):
        sample = snap.joints.get(jid)
        if sample is not None:
            return sample.position
    return None


def _body_normalized_step_lengths(
    recording: GaitMotionRecording,
    events,
) -> tuple[list[float], list[float]]:
    """Foot separation along +Z (progression) at heel strike, body-normalized m."""
    left: list[float] = []
    right: list[float] = []
    snaps = {int(s.frame_index): s for s in recording.snapshots}
    by_time = list(recording.snapshots)
    for event in events:
        if "heel_strike" not in event.event_type:
            continue
        snap = snaps.get(int(event.frame_index))
        if snap is None and by_time:
            snap = min(by_time, key=lambda s: abs(s.time_s - event.time_s))
        if snap is None:
            continue
        lf = _foot_pos(snap, "left")
        rf = _foot_pos(snap, "right")
        if lf is None or rf is None:
            continue
        sep = abs(float(lf.z) - float(rf.z))
        if not np.isfinite(sep) or sep < 1e-4:
            # Fall back to XZ Euclidean (hip-centered; still body-normalized).
            sep = float(np.hypot(lf.x - rf.x, lf.z - rf.z))
        if not np.isfinite(sep) or sep < 1e-4:
            continue
        if event.side == "left" or event.event_type.startswith("left"):
            left.append(sep)
        else:
            right.append(sep)
    return left, right


def _clearance(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return None
    # Hip-centered Y: peak height above the session floor (5th percentile).
    floor = float(np.percentile(finite, 5))
    peak = float(np.percentile(finite, 95))
    return peak - floor


def compare_bilateral(
    recording: GaitMotionRecording,
    kin: KinematicBundle,
    events: GaitEventSummary,
    aligned: list[AlignedTrajectoryPair],
    *,
    ground_reliable: bool = False,
    floor_y: float | None = None,
) -> list[AsymmetryScalar]:
    rows: list[AsymmetryScalar] = []
    times = kin.times_s
    q_kin = "estimated"
    if kin.provenance != "inferred_monocular_3d":
        q_kin = "measured"
    n_l = len(events.left_cycles)
    n_r = len(events.right_cycles)
    have_lr_cycles = n_l >= 1 and n_r >= 1
    note_kin = (
        "P5–P95 span of anatomical flexion on that side's ipsilateral "
        "heel-strike cycles (phase-local; not unaligned frames). "
        f"{ALIGNMENT_EVENT_LABEL}."
    )
    if not have_lr_cycles:
        note_kin = (
            "Left/right heel-strike cycles incomplete — scalar ROM uses the "
            "whole session and is not a phase-aligned bilateral comparison. "
            "SI / ASI / SI_norm are omitted."
        )

    def side_dof(dof_id: str, cycles) -> np.ndarray:
        series = kin.series_for_dof(dof_id)
        if cycles:
            return _side_series(series, times, list(cycles))
        return series

    reliable_angles = events.events_reliable and not events.camera_unsuitable
    pairs = (
        ("knee_flexion_rom", "Knee flexion ROM (P95−P5)", "left_knee_flexion", "right_knee_flexion", "deg"),
        ("hip_flexion_rom", "Hip flexion ROM", "left_hip_flexion", "right_hip_flexion", "deg"),
        ("ankle_flexion_rom", "Ankle flexion ROM", "left_ankle_flexion", "right_ankle_flexion", "deg"),
    )
    for name, display, l_id, r_id, units in pairs:
        left = rom_percentile(side_dof(l_id, events.left_cycles))
        right = rom_percentile(side_dof(r_id, events.right_cycles))
        rows.append(
            _scalar(
                name,
                display,
                units,
                left,
                right,
                quality=q_kin,
                reliable=reliable_angles and have_lr_cycles,
                note=note_kin,
                n_left_cycles=n_l if have_lr_cycles else 0,
                n_right_cycles=n_r if have_lr_cycles else 0,
                formula=(
                    "ROM = P95 − P5 of canonical analysis flexion (deg) on "
                    f"ipsilateral HS–HS cycles. {ROBINSON_SI_DEFINITION}. "
                    f"{ASI_DEFINITION}. {SI_NORM_DEFINITION}."
                ),
            )
        )

    def side_segment(seg_id: str, cycles) -> np.ndarray:
        series = kin.series_for_segment(seg_id)
        if cycles:
            return _side_series(series, times, list(cycles))
        return series

    for name, display, l_id, r_id in (
        ("thigh_rom", "Thigh range of motion (P95−P5)", "left_thigh", "right_thigh"),
        ("shank_rom", "Shank range of motion (P95−P5)", "left_shank", "right_shank"),
    ):
        left = rom_percentile(side_segment(l_id, events.left_cycles))
        right = rom_percentile(side_segment(r_id, events.right_cycles))
        rows.append(
            _scalar(
                name,
                display,
                "deg",
                left,
                right,
                quality=q_kin,
                reliable=reliable_angles and have_lr_cycles,
                note=(
                    "P5–P95 span of sagittal segment tilt (0° hanging down) on "
                    "ipsilateral HS–HS cycles — phase-local, not unaligned frames. "
                    f"{ALIGNMENT_EVENT_LABEL}."
                ),
                n_left_cycles=n_l if have_lr_cycles else 0,
                n_right_cycles=n_r if have_lr_cycles else 0,
                formula=(
                    "ROM = P95 − P5 of thigh/shank sagittal tilt (deg) on "
                    f"ipsilateral HS–HS cycles. {ROBINSON_SI_DEFINITION}. "
                    f"{ASI_DEFINITION}. {SI_NORM_DEFINITION}."
                ),
            )
        )

    for pair in aligned:
        if pair.name not in {
            "knee_flexion",
            "hip_flexion",
            "ankle_flexion",
            "thigh_tilt",
            "shank_tilt",
        }:
            continue
        joint = pair.name.replace("_flexion", "").replace("_", " ").title()
        rows.append(
            _scalar(
                f"{pair.name}_peak",
                f"{joint} peak angle",
                pair.units,
                pair.peak_left,
                pair.peak_right,
                quality="derived",
                reliable=pair.aligned and pair.peak_left is not None and pair.peak_right is not None,
                note="Peak of the phase-aligned (0–100%) mean flexion curve.",
                n_left_cycles=pair.n_left_cycles,
                n_right_cycles=pair.n_right_cycles,
                formula=(
                    "Peak of the ipsilateral HS–HS mean curve. "
                    f"{ROBINSON_SI_DEFINITION}. {ASI_DEFINITION}."
                ),
            )
        )
        rows.append(
            _timing_scalar(
                f"{pair.name}_peak_phase",
                f"{joint} peak timing",
                pair.peak_phase_left_pct,
                pair.peak_phase_right_pct,
                quality="derived",
                reliable=pair.aligned,
                note="Peak location on the phase-aligned mean curve.",
                n_left_cycles=pair.n_left_cycles,
                n_right_cycles=pair.n_right_cycles,
            )
        )

    for name, display, l_id, r_id in (
        ("knee_omega_rms", "Knee angular-velocity RMS", "left_knee_flexion", "right_knee_flexion"),
        ("hip_omega_rms", "Hip angular-velocity RMS", "left_hip_flexion", "right_hip_flexion"),
        ("ankle_omega_rms", "Ankle angular-velocity RMS", "left_ankle_flexion", "right_ankle_flexion"),
    ):
        left = rms(_side_series(kin.omega_for_dof(l_id), times, list(events.left_cycles)) if events.left_cycles else kin.omega_for_dof(l_id))
        right = rms(_side_series(kin.omega_for_dof(r_id), times, list(events.right_cycles)) if events.right_cycles else kin.omega_for_dof(r_id))
        rows.append(
            _scalar(
                name,
                display,
                "deg/s",
                left,
                right,
                quality="derived",
                reliable=reliable_angles and have_lr_cycles,
                note="RMS of filtered angular velocity on that side's HS–HS cycles.",
                allow_signed_norm=True,
                n_left_cycles=n_l if have_lr_cycles else 0,
                n_right_cycles=n_r if have_lr_cycles else 0,
            )
        )

    for name, display, l_id, r_id in (
        ("knee_omega_peak", "Knee peak |angular velocity|", "left_knee_flexion", "right_knee_flexion"),
        ("hip_omega_peak", "Hip peak |angular velocity|", "left_hip_flexion", "right_hip_flexion"),
    ):
        left = peak_abs(_side_series(kin.omega_for_dof(l_id), times, list(events.left_cycles)) if events.left_cycles else kin.omega_for_dof(l_id))
        right = peak_abs(_side_series(kin.omega_for_dof(r_id), times, list(events.right_cycles)) if events.right_cycles else kin.omega_for_dof(r_id))
        rows.append(
            _scalar(
                name,
                display,
                "deg/s",
                left,
                right,
                quality="derived",
                reliable=reliable_angles and have_lr_cycles,
                note="Peak absolute angular velocity on ipsilateral HS–HS cycles.",
                allow_signed_norm=True,
                n_left_cycles=n_l if have_lr_cycles else 0,
                n_right_cycles=n_r if have_lr_cycles else 0,
            )
        )

    step_l, step_r = _body_normalized_step_lengths(recording, events.heel_strikes)
    metric_ok = (
        events.spatial_reliable
        and kin.provenance == "calibrated_metric"
        and bool(step_l)
        and bool(step_r)
    )
    step_note = (
        "Step length from contralateral foot separation at heel strike. "
        "Reported as a metric value only when the session is calibrated "
        "and the camera view supports spatial gait measures."
    )
    if kin.provenance != "calibrated_metric":
        rows.append(
            _scalar(
                "step_length",
                "Step length (metric)",
                "m",
                None,
                None,
                quality="unavailable",
                reliable=False,
                note=step_note + " Coordinates are inferred / body-normalized — metric step length disabled.",
                n_left_cycles=0,
                n_right_cycles=0,
            )
        )
        rows.append(
            _scalar(
                "step_length_body_normalized",
                "Step length (body-normalized)",
                "body_normalized_m",
                _median(step_l),
                _median(step_r),
                quality="estimated",
                reliable=events.spatial_reliable and bool(step_l) and bool(step_r),
                note=step_note + " Body-normalized proxy; not laboratory metres.",
                n_left_cycles=len(step_l),
                n_right_cycles=len(step_r),
            )
        )
    else:
        rows.append(
            _scalar(
                "step_length",
                "Step length (metric)",
                "m",
                _median(step_l),
                _median(step_r),
                quality="estimated" if not metric_ok else "measured",
                reliable=metric_ok,
                note=step_note,
                n_left_cycles=len(step_l),
                n_right_cycles=len(step_r),
            )
        )

    n_stance_l = sum(1 for i in events.stance_intervals if i.side == "left")
    n_stance_r = sum(1 for i in events.stance_intervals if i.side == "right")
    n_swing_l = sum(1 for i in events.swing_intervals if i.side == "left")
    n_swing_r = sum(1 for i in events.swing_intervals if i.side == "right")
    rows.append(
        _scalar(
            "step_duration",
            "Step timing (HS to contralateral HS)",
            "s",
            _median(events.left_step_durations_s),
            _median(events.right_step_durations_s),
            quality="derived",
            reliable=events.events_reliable,
            note="HS to contralateral HS.",
            n_left_cycles=len(events.left_step_durations_s),
            n_right_cycles=len(events.right_step_durations_s),
            formula="Median step duration (s). " + ROBINSON_SI_DEFINITION,
        )
    )
    rows.append(
        _scalar(
            "stride_duration",
            "Gait cycle duration (stride HS–HS)",
            "s",
            _median(events.left_stride_durations_s),
            _median(events.right_stride_durations_s),
            quality="derived",
            reliable=events.events_reliable,
            note="Ipsilateral HS to HS.",
            n_left_cycles=len(events.left_stride_durations_s),
            n_right_cycles=len(events.right_stride_durations_s),
            formula="Median stride duration (s). " + ROBINSON_SI_DEFINITION,
        )
    )
    rows.append(
        _scalar(
            "stance_duration",
            "Stance duration",
            "s",
            _mean_duration(events.stance_intervals, "left"),
            _mean_duration(events.stance_intervals, "right"),
            quality="derived",
            reliable=events.events_reliable,
            note="Heel strike to ipsilateral toe-off.",
            n_left_cycles=n_stance_l,
            n_right_cycles=n_stance_r,
            formula="Median stance duration (s). " + ROBINSON_SI_DEFINITION,
        )
    )
    rows.append(
        _scalar(
            "swing_duration",
            "Swing duration",
            "s",
            _mean_duration(events.swing_intervals, "left"),
            _mean_duration(events.swing_intervals, "right"),
            quality="derived",
            reliable=events.events_reliable,
            note="Toe-off to next ipsilateral heel strike.",
            n_left_cycles=n_swing_l,
            n_right_cycles=n_swing_r,
            formula="Median swing duration (s). " + ROBINSON_SI_DEFINITION,
        )
    )

    heel_l = kin.series_for_joint("left_heel", 1)
    heel_r = kin.series_for_joint("right_heel", 1)
    if events.left_cycles:
        heel_l = _side_series(heel_l, times, list(events.left_cycles))
    if events.right_cycles:
        heel_r = _side_series(heel_r, times, list(events.right_cycles))
    rows.append(
        _scalar(
            "foot_clearance",
            "Foot clearance (heel vertical span)",
            kin.units,
            _clearance(heel_l),
            _clearance(heel_r),
            quality="estimated",
            reliable=events.events_reliable and have_lr_cycles,
            note=(
                "P95−P5 of heel Y on ipsilateral HS–HS cycles. Inferred "
                "monocular Y is body-normalized, not millimetre laboratory "
                "foot clearance."
            ),
            n_left_cycles=n_l if have_lr_cycles else 0,
            n_right_cycles=n_r if have_lr_cycles else 0,
        )
    )
    if ground_reliable and floor_y is not None:
        heel_l_g = kin.series_for_joint("left_heel", 1) - float(floor_y)
        heel_r_g = kin.series_for_joint("right_heel", 1) - float(floor_y)
        if events.left_cycles:
            heel_l_g = _side_series(heel_l_g, times, list(events.left_cycles))
        if events.right_cycles:
            heel_r_g = _side_series(heel_r_g, times, list(events.right_cycles))
        left_g = _clearance(np.clip(heel_l_g, 0.0, None))
        right_g = _clearance(np.clip(heel_r_g, 0.0, None))
        rows.append(
            _scalar(
                "foot_clearance_ground",
                "Foot clearance (ground plane)",
                kin.units,
                left_g,
                right_g,
                quality="estimated",
                reliable=events.events_reliable and have_lr_cycles,
                note=(
                    "P95−P5 of heel height above the estimated floor plane on "
                    "ipsilateral HS–HS cycles. Body-normalized estimated metres "
                    "— not a calibrated laboratory clearance."
                ),
                n_left_cycles=n_l if have_lr_cycles else 0,
                n_right_cycles=n_r if have_lr_cycles else 0,
            )
        )
    else:
        rows.append(
            _scalar(
                "foot_clearance_ground",
                "Foot clearance (ground plane)",
                kin.units,
                None,
                None,
                quality="unavailable",
                reliable=False,
                note=(
                    "Ground-plane clearance withheld — floor estimate is missing "
                    "or unreliable. Heel vertical span above is the fallback."
                ),
                n_left_cycles=0,
                n_right_cycles=0,
            )
        )

    for pair in aligned:
        rows.append(
            _scalar(
                f"traj_amp_{pair.name}",
                f"{pair.name.replace('_', ' ').title()} waveform amplitude",
                pair.units,
                pair.amplitude_left,
                pair.amplitude_right,
                quality="derived",
                reliable=pair.aligned,
                note=(
                    "Amplitude of the phase-aligned mean L/R curves "
                    "(0–100% ipsilateral heel-strike cycle)."
                ),
                n_left_cycles=pair.n_left_cycles,
                n_right_cycles=pair.n_right_cycles,
                formula=(
                    "max(mean curve) − min(mean curve) after HS–HS resampling. "
                    f"{ROBINSON_SI_DEFINITION}."
                ),
            )
        )
        if pair.rmse is not None:
            rows.append(
                AsymmetryScalar(
                    name=f"traj_rmse_{pair.name}",
                    display=f"{pair.name.replace('_', ' ').title()} aligned RMSE",
                    units=pair.units,
                    left=None,
                    right=None,
                    si_percent=None,
                    asi_percent=None,
                    si_norm=None,
                    quality="derived" if pair.aligned else "insufficient",
                    reliable=pair.aligned,
                    note=(
                        "RMSE of phase-aligned mean curves. This is a waveform "
                        "distance, not SI / ASI / SI_norm."
                    ),
                    kind="kinematic_asymmetry",
                    formula="RMSE = sqrt(mean((L(φ) − R(φ))²)) on the 0–100% grid",
                    n_left_cycles=pair.n_left_cycles,
                    n_right_cycles=pair.n_right_cycles,
                    alignment_event=ALIGNMENT_EVENT,
                    comparison_kind="rmse",
                    difference=pair.rmse,
                    difference_units=pair.units,
                    difference_formula="RMSE of already-aligned mean curves",
                )
            )
    return rows
