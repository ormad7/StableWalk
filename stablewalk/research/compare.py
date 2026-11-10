"""Numerical comparison of two independently analyzed gait clips.

Δ = B − A. Differences are kinematic / spectral estimates, not diagnoses.
Gait-phase resampling is used for waveform RMSE so unequal FPS/length
does not compare unaligned frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stablewalk.analysis.gait_feature_analysis import (
    GAIT_CYCLE_SAMPLE_COUNT,
    resample_cycle_trajectory,
)
from stablewalk.research.categories import CLINICAL_POLICY, display_clinical_diagnosis
from stablewalk.research.pipeline import (
    IndependentClipResult,
    slots_are_independent,
    slots_share_source_file,
)

KIND_DERIVED = "derived"


@dataclass
class ScalarDiff:
    name: str
    display: str
    units: str
    value_a: float | None
    value_b: float | None
    delta: float | None
    relative_pct: float | None
    note: str = ""
    family: str = "kinematic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display": self.display,
            "units": self.units,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "delta": self.delta,
            "relative_pct": self.relative_pct,
            "note": self.note,
            "family": self.family,
            "kind": KIND_DERIVED,
        }


@dataclass
class WaveformDiff:
    name: str
    units: str
    phase_pct: list[float]
    series_a: list[float | None]
    series_b: list[float | None]
    rmse: float | None
    aligned: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "units": self.units,
            "rmse": self.rmse,
            "aligned": self.aligned,
            "note": self.note,
            "n_samples": len(self.phase_pct),
        }


@dataclass
class ResearchComparison:
    slot_a: IndependentClipResult
    slot_b: IndependentClipResult
    independent: bool
    scalars: list[ScalarDiff] = field(default_factory=list)
    waveforms: list[WaveformDiff] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    clinical_note: str = CLINICAL_POLICY

    def scalar_by_name(self, name: str) -> ScalarDiff | None:
        for row in self.scalars:
            if row.name == name:
                return row
        return None

    def waveform_by_name(self, name: str) -> WaveformDiff | None:
        for row in self.waveforms:
            if row.name == name:
                return row
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_a": self.slot_a.to_dict(),
            "slot_b": self.slot_b.to_dict(),
            "independent": self.independent,
            "scalars": [s.to_dict() for s in self.scalars],
            "waveforms": [w.to_dict() for w in self.waveforms],
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "clinical_note": self.clinical_note,
            "kind": KIND_DERIVED,
            "delta_definition": "Δ = B − A",
        }


def _fin(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _delta(a: float | None, b: float | None) -> tuple[float | None, float | None]:
    aa, bb = _fin(a), _fin(b)
    if aa is None or bb is None:
        return None, None
    diff = bb - aa
    rel = None
    if abs(aa) > 1e-9:
        rel = 100.0 * diff / abs(aa)
    return diff, rel


def _row(
    name: str,
    display: str,
    units: str,
    a: float | None,
    b: float | None,
    *,
    family: str,
    note: str = "",
) -> ScalarDiff:
    d, rel = _delta(a, b)
    return ScalarDiff(name, display, units, _fin(a), _fin(b), d, rel, note, family)


def _cadence(result: IndependentClipResult) -> float | None:
    metrics = getattr(getattr(result, "cycles", None), "metrics", None)
    if metrics is None:
        return None
    return _fin(getattr(metrics, "cadence_steps_per_min", None))


def _stride(result: IndependentClipResult) -> float | None:
    metrics = getattr(getattr(result, "cycles", None), "metrics", None)
    if metrics is None:
        return None
    return _fin(getattr(metrics, "stride_time_s", None))


def _spectral_value(result: IndependentClipResult, key: str) -> float | None:
    features = getattr(getattr(result, "spectral", None), "features", None)
    if features is None:
        return None
    vec = getattr(features, "session_vector", None) or {}
    return _fin(vec.get(key))


def _acf_stride(result: IndependentClipResult) -> float | None:
    ac = None
    spectral = result.spectral
    if spectral is not None and hasattr(spectral, "primary_autocorr"):
        ac = spectral.primary_autocorr()
    if ac is None:
        return None
    return _fin(getattr(ac, "stride_period_s", None))


def _asi(result: IndependentClipResult, name: str) -> float | None:
    report = result.asymmetry
    if report is None:
        return None
    row = report.scalar_by_name(name)
    if row is None:
        return None
    return _fin(row.asi_percent)


def _pose_quality(result: IndependentClipResult) -> float | None:
    """Detected-frame fraction as a percent (not a clinical quality score)."""
    frac = (result.quality or {}).get("detected_fraction")
    value = _fin(frac)
    if value is None:
        return None
    return 100.0 * value


def _rom(result: IndependentClipResult, name: str) -> float | None:
    report = result.asymmetry
    if report is None:
        return None
    row = report.scalar_by_name(name)
    if row is None:
        return None
    left, right = _fin(row.left), _fin(row.right)
    if left is None or right is None:
        return left or right
    return 0.5 * (left + right)


def _mean_cycle_curve(result: IndependentClipResult, dof_id: str) -> tuple[np.ndarray | None, bool]:
    rec = result.gait_motion
    cycles = list(getattr(result.cycles, "cycles", None) or [])
    if rec is None or not cycles:
        return None, False
    times = np.array([float(s.time_s) for s in rec.snapshots], dtype=float)
    values = np.full(times.shape, np.nan)
    for i, snap in enumerate(rec.snapshots):
        dof = snap.get_dof(dof_id) if hasattr(snap, "get_dof") else None
        if dof is not None and dof.angle_deg is not None:
            values[i] = float(dof.angle_deg)
    curves: list[np.ndarray] = []
    for cycle in cycles:
        mask = (
            (times >= cycle.start_time_s)
            & (times <= cycle.end_time_s)
            & np.isfinite(values)
        )
        if int(mask.sum()) < 4:
            continue
        arr = resample_cycle_trajectory(
            [float(t) for t in times[mask]],
            [float(v) for v in values[mask]],
            t_start=cycle.start_time_s,
            t_end=cycle.end_time_s,
            n_samples=GAIT_CYCLE_SAMPLE_COUNT,
        )
        if arr is not None:
            curves.append(np.asarray(arr, dtype=float))
    if not curves:
        return None, False
    return np.nanmean(np.vstack(curves), axis=0), True


def _waveform(name: str, units: str, a: IndependentClipResult, b: IndependentClipResult, dof: str) -> WaveformDiff:
    ca, ok_a = _mean_cycle_curve(a, dof)
    cb, ok_b = _mean_cycle_curve(b, dof)
    n = GAIT_CYCLE_SAMPLE_COUNT
    pct = [i * 100.0 / (n - 1) for i in range(n)]
    aligned = bool(ok_a and ok_b and ca is not None and cb is not None)
    rmse = None
    sa: list[float | None] = [None] * n
    sb: list[float | None] = [None] * n
    if aligned and ca is not None and cb is not None:
        ok = np.isfinite(ca) & np.isfinite(cb)
        if int(ok.sum()) >= 8:
            rmse = float(np.sqrt(np.mean((ca[ok] - cb[ok]) ** 2)))
        sa = [None if not np.isfinite(v) else float(v) for v in ca]
        sb = [None if not np.isfinite(v) else float(v) for v in cb]
    return WaveformDiff(
        name=name,
        units=units,
        phase_pct=pct,
        series_a=sa,
        series_b=sb,
        rmse=rmse,
        aligned=aligned,
        note=(
            "Each video's cycles resampled to 0–100% independently, then RMSE."
            if aligned
            else "Gait-phase alignment unavailable on one or both clips."
        ),
    )


def compare_independent_clips(
    slot_a: IndependentClipResult,
    slot_b: IndependentClipResult,
) -> ResearchComparison:
    independent = slots_are_independent(slot_a, slot_b)
    same_file = slots_share_source_file(slot_a, slot_b)
    warnings: list[str] = list(slot_a.warnings) + list(slot_b.warnings)
    limitations = [
        "Monocular 3D is body-normalized and heuristic — not laboratory mocap.",
        "Δ = B − A on derived kinematic/spectral quantities, not clinical scores.",
        "Unequal FPS or duration is handled by gait-phase resampling when cycles exist.",
        CLINICAL_POLICY,
    ]
    if not independent:
        warnings.append(
            "Video A and Video B share pose or analysis objects. "
            "Comparison is blocked from being treated as two pipelines."
        )
        limitations.append("Results must not be interpreted as a two-video comparison.")
    elif same_file:
        warnings.append(
            "A and B were analyzed from the same source file. "
            "Deltas are two independent pose runs, not two recordings."
        )

    for slot, tag in ((slot_a, "A"), (slot_b, "B")):
        meta = slot.metadata
        if meta is None:
            continue
        shown = display_clinical_diagnosis(
            diagnosis=meta.clinical_diagnosis,
            verified=meta.clinical_verified,
            source=meta.clinical_source,
        )
        if meta.gait_category == "dementia_research" and shown is None:
            warnings.append(
                f"Clip {tag} is in the dementia research category but has no "
                "verified clinical diagnosis — it is not labeled as dementia."
            )

    scalars = [
        _row("cadence", "Cadence", "steps/min", _cadence(slot_a), _cadence(slot_b), family="temporal"),
        _row("stride_time", "Stride time", "s", _stride(slot_a), _stride(slot_b), family="temporal"),
        _row(
            "knee_rom",
            "Knee flexion ROM (mean L/R)",
            "deg",
            _rom(slot_a, "knee_flexion_rom"),
            _rom(slot_b, "knee_flexion_rom"),
            family="kinematic",
        ),
        _row(
            "knee_asi",
            "Knee ROM ASI",
            "%",
            _asi(slot_a, "knee_flexion_rom"),
            _asi(slot_b, "knee_flexion_rom"),
            family="asymmetry",
        ),
        _row(
            "dominant_frequency",
            "Dominant frequency",
            "Hz",
            _spectral_value(slot_a, "dominant_frequency_hz"),
            _spectral_value(slot_b, "dominant_frequency_hz"),
            family="spectral",
        ),
        _row(
            "spectral_entropy",
            "Spectral entropy",
            "1",
            _spectral_value(slot_a, "spectral_entropy"),
            _spectral_value(slot_b, "spectral_entropy"),
            family="spectral",
        ),
        _row(
            "acf_stride_lag",
            "ACF stride lag",
            "s",
            _acf_stride(slot_a),
            _acf_stride(slot_b),
            family="autocorrelation",
        ),
        _row(
            "regularity_acf",
            "ACF regularity",
            "1",
            _spectral_value(slot_a, "regularity_acf"),
            _spectral_value(slot_b, "regularity_acf"),
            family="autocorrelation",
        ),
        _row(
            "pose_quality",
            "Pose detection",
            "%",
            _pose_quality(slot_a),
            _pose_quality(slot_b),
            family="quality",
        ),
        _row("n_frames", "Analyzed frames", "1", float(slot_a.n_frames), float(slot_b.n_frames), family="quality"),
        _row("fps", "Frame rate", "Hz", slot_a.fps, slot_b.fps, family="quality"),
    ]
    waveforms = [
        _waveform("knee_flexion", "deg", slot_a, slot_b, "left_knee_flexion"),
        _waveform("hip_flexion", "deg", slot_a, slot_b, "left_hip_flexion"),
    ]
    knee_w = waveforms[0]
    if knee_w.rmse is not None:
        scalars.append(
            ScalarDiff(
                name="knee_phase_rmse",
                display="Phase-aligned knee RMSE",
                units="deg",
                value_a=None,
                value_b=knee_w.rmse,
                delta=knee_w.rmse,
                relative_pct=None,
                note=knee_w.note,
                family="joint_angle",
            )
        )

    return ResearchComparison(
        slot_a=slot_a,
        slot_b=slot_b,
        independent=independent,
        scalars=scalars,
        waveforms=waveforms,
        warnings=warnings,
        limitations=limitations,
    )
