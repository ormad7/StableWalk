"""Session research-validation snapshot from already-computed StableWalk outputs.

Reads the 16-joint wire, gait cycles, spectral/ACF report, bilateral indices,
and asymmetric moments that the current analysis pass produced. Missing or
failed products are labeled unavailable / insufficient / failed — never
fabricated, and never marked as a passed requirement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from stablewalk.analysis.asymmetry3d.indices import (
    ASI_DEFINITION,
    OVERALL_VS_KNEE_DEFINITION,
    ROBINSON_SI_DEFINITION,
    SI_NORM_DEFINITION,
    TIMING_DEFINITION,
)
from stablewalk.analysis.quality_layers import (
    LANDMARK_NOT_TRACKING,
    LAYER_BIOMECHANICAL,
    LAYER_CYCLE_DETECTION,
    LAYER_LANDMARK_CONFIDENCE,
    LAYER_POSE_COVERAGE,
    LAYER_SPECTRAL,
    biomechanical_validity_evidence,
    cycle_detection_evidence,
    layer_documentation,
    pose_coverage_evidence,
    spectral_reliability_evidence,
)
from stablewalk.analysis.gait_asymmetry_moments import (
    MIN_CONFIDENCE,
    MIN_DURATION_S,
    SCORE_THRESHOLD,
)
from stablewalk.analysis.spectral.features import FEATURE_DEFINITIONS
from stablewalk.models.gait_wireframe import (
    GAIT_WIRE_JOINT_COUNT,
    GAIT_WIRE_JOINT_IDS,
    GAIT_WIRE_VISIBILITY_MIN,
)

ValidationStatus = Literal["computed", "insufficient", "unavailable", "failed"]

# A wire frame is usable when this many canonical joints exceed visibility.
_MIN_WIRE_JOINTS_USABLE = 12


@dataclass
class ValidationItem:
    key: str
    title: str
    status: ValidationStatus
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    units: dict[str, str] = field(default_factory=dict)
    definitions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "summary": self.summary,
            "details": dict(self.details),
            "reason": self.reason,
            "unavailable_reason": self.reason if self.status != "computed" else None,
            "units": dict(self.units),
            "definitions": dict(self.definitions),
            "formula_id": self.key,
        }


@dataclass
class ResearchValidationSnapshot:
    """Compact research evidence for the Results page and the downloadable report."""

    items: list[ValidationItem] = field(default_factory=list)
    video_metadata: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    confidence: dict[str, Any] = field(default_factory=dict)
    strongest_moment: dict[str, Any] | None = None
    plot_series: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    clinical: bool = False

    def item(self, key: str) -> ValidationItem | None:
        for row in self.items:
            if row.key == key:
                return row
        return None

    def to_dict(self) -> dict[str, Any]:
        video = dict(self.video_metadata)
        return {
            "schema": "stablewalk-research-validation",
            "schema_version": "1.0",
            "kind": "derived",
            "clinical": False,
            "source": self.source,
            "video_metadata": video,
            "fps": video.get("fps"),
            "sampling_rate_hz": video.get("fps"),
            "coordinate_system": video.get("coordinate_system"),
            "preprocessing": {
                "source_kind": video.get("source_kind"),
                "pose_frame_count": video.get("pose_frame_count") or video.get("frame_count"),
            },
            "confidence": dict(self.confidence),
            "limitations": list(self.limitations),
            "strongest_moment": None if self.strongest_moment is None else dict(self.strongest_moment),
            "items": [item.to_dict() for item in self.items],
            "disclaimer": (
                "Status describes whether this session produced the listed output. "
                "It is not an acceptance of a research requirement and not a diagnosis."
            ),
        }


def _attr(session: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(session, name):
            value = getattr(session, name)
            if value is not None:
                return value
    return default


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, *, digits: int = 3, unit: str = "") -> str:
    number = _finite(value)
    if number is None:
        return "—"
    text = f"{number:.{digits}g}"
    return f"{text} {unit}".strip() if unit else text


def _series_list(value: Any) -> list[Any]:
    """Convert a stored series to a list. Numpy arrays are truthy-ambiguous."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return [x for x in value]
    except TypeError:
        return []


def _item(
    key: str,
    title: str,
    status: ValidationStatus,
    summary: str,
    *,
    details: dict[str, Any] | None = None,
    reason: str | None = None,
    units: dict[str, str] | None = None,
    definitions: dict[str, str] | None = None,
) -> ValidationItem:
    return ValidationItem(
        key=key,
        title=title,
        status=status,
        summary=summary,
        details=details or {},
        reason=reason,
        units=units or {},
        definitions=definitions or {},
    )


def _video_metadata(session: Any) -> dict[str, Any]:
    recording = _attr(session, "gait_motion")
    sequence = _attr(session, "sequence")
    source = ""
    if recording is not None:
        source = str(getattr(recording, "source", "") or "")
    if not source:
        source = str(_attr(session, "video_path", "source", default="") or "")
    meta: dict[str, Any] = {"source": source}
    if recording is not None:
        meta["fps"] = _finite(getattr(recording, "fps", None))
        meta["duration_s"] = _finite(getattr(recording, "duration_s", None))
        meta["frame_count"] = int(getattr(recording, "frame_count", 0) or 0)
        meta["coordinate_system"] = str(getattr(recording, "coordinate_system", "") or "")
        meta["source_kind"] = str(getattr(recording, "source_kind", "") or "")
        rec_meta = getattr(recording, "metadata", None) or {}
        if isinstance(rec_meta, dict):
            for key in ("detected_count", "width", "height", "video_width", "video_height"):
                if key in rec_meta:
                    meta[key] = rec_meta[key]
    if sequence is not None:
        fps = _finite(getattr(sequence, "fps", None))
        if fps is not None:
            meta.setdefault("fps", fps)
        n_frames = getattr(sequence, "n_frames", None)
        if n_frames is None:
            frames = getattr(sequence, "frames", None)
            n_frames = len(frames) if frames is not None else None
        if n_frames is not None:
            meta.setdefault("pose_frame_count", int(n_frames))
        for key in ("source_path", "video_path", "width", "height"):
            value = getattr(sequence, key, None)
            if value not in (None, ""):
                meta.setdefault(key, value)
    return {k: v for k, v in meta.items() if v not in (None, "")}


def _wire_item(session: Any) -> ValidationItem:
    wire = _attr(session, "gait_wireframe")
    recording = _attr(session, "gait_motion")
    definitions = {
        "model": (
            "Canonical 16-joint gait wire projected from MediaPipe 3D. "
            "Missing landmarks are omitted; coordinates are not invented."
        ),
        "usable_frame": (
            f"A frame is usable when at least {_MIN_WIRE_JOINTS_USABLE} of "
            f"{GAIT_WIRE_JOINT_COUNT} joints have visibility "
            f"≥ {GAIT_WIRE_VISIBILITY_MIN}."
        ),
    }
    units = {"joint_count": "count", "usable_frames": "frames"}
    if wire is None and recording is None:
        return _item(
            "wire16",
            "16-joint model",
            "unavailable",
            "16-joint gait wire was not built.",
            reason="No gait recording is loaded.",
            units=units,
            definitions=definitions,
        )
    frames = list(getattr(wire, "frames", None) or [])
    if not frames and recording is not None:
        return _item(
            "wire16",
            "16-joint model",
            "insufficient",
            "A recording is loaded, but the 16-joint wire has no frames.",
            reason="Gait wireframe frames are empty.",
            units=units,
            definitions=definitions,
        )
    usable = 0
    present: set[str] = set()
    for frame in frames:
        joints = getattr(frame, "joints", None) or {}
        keep = 0
        for jid, sample in joints.items():
            vis = float(getattr(sample, "visibility", 0.0) or 0.0)
            if vis >= GAIT_WIRE_VISIBILITY_MIN:
                keep += 1
                present.add(str(jid))
        if keep >= _MIN_WIRE_JOINTS_USABLE:
            usable += 1
    n_frames = len(frames)
    status: ValidationStatus = "computed" if usable > 0 else "insufficient"
    summary = (
        f"{GAIT_WIRE_JOINT_COUNT}-joint gait wire · {usable}/{n_frames} usable frames"
        if status == "computed"
        else f"16-joint model present, but no usable frames ({usable}/{n_frames})."
    )
    return _item(
        "wire16",
        "16-joint model",
        status,
        summary,
        details={
            "joint_count": GAIT_WIRE_JOINT_COUNT,
            "joint_ids": list(GAIT_WIRE_JOINT_IDS),
            "n_frames": n_frames,
            "usable_frames": usable,
            "min_joints_usable": _MIN_WIRE_JOINTS_USABLE,
            "visibility_min": GAIT_WIRE_VISIBILITY_MIN,
            "joints_observed": sorted(present),
        },
        reason=None if status == "computed" else "Too few visible landmarks on the 16-joint wire.",
        units=units,
        definitions=definitions,
    )


def _cycle_item(session: Any, wire_item: ValidationItem) -> ValidationItem:
    cycles = _attr(session, "_gait_cycle", "gait_cycle")
    recording = _attr(session, "gait_motion")
    moments = _attr(session, "_gait_asymmetry_moments", "gait_asymmetry_moments")
    definitions = {
        "gait_cycle": (
            "Complete ipsilateral heel-strike to heel-strike interval. "
            "Partial leading/trailing intervals are not counted."
        ),
        "usable_frames": "Detected pose frames on the gait recording (not invented).",
    }
    units = {"usable_frames": "frames", "gait_cycles": "count"}
    n_snaps = 0
    n_detected = 0
    if recording is not None:
        snaps = list(getattr(recording, "snapshots", None) or [])
        n_snaps = len(snaps)
        for snap in snaps:
            meta = getattr(snap, "metadata", None) or {}
            if meta.get("detected", True):
                n_detected += 1
    wire_usable = int((wire_item.details or {}).get("usable_frames") or 0)
    usable = wire_usable if wire_usable else n_detected
    if cycles is None:
        return _item(
            "frames_cycles",
            "Usable frames and gait cycles",
            "unavailable" if recording is None else "insufficient",
            (
                "Gait-cycle analysis was not run."
                if recording is None
                else f"{usable} usable frames · gait cycles not computed."
            ),
            details={"usable_frames": usable, "n_snapshots": n_snaps, "n_detected": n_detected},
            reason="Detected gait cycles are required; none were provided.",
            units=units,
            definitions=definitions,
        )
    cycle_list = list(getattr(cycles, "cycles", None) or [])
    n_cycles = len(cycle_list)
    n_left = int(getattr(moments, "n_left_cycles", 0) or 0) if moments is not None else 0
    n_right = int(getattr(moments, "n_right_cycles", 0) or 0) if moments is not None else 0
    if n_left == 0 and n_right == 0:
        for cycle in cycle_list:
            side = str(getattr(cycle, "boundary_side", "") or "").lower()
            if side.startswith("left"):
                n_left += 1
            elif side.startswith("right"):
                n_right += 1
    metrics = getattr(cycles, "metrics", None)
    status: ValidationStatus = "computed" if n_cycles >= 1 or n_left + n_right >= 1 else "insufficient"
    quality = str(getattr(metrics, "reliability_reason", "") or "").strip()
    if not quality:
        quality = (
            "computed — contact HS–HS intervals"
            if status == "computed"
            else "insufficient — no complete HS–HS interval"
        )
    if status == "computed":
        summary = (
            f"{usable} usable frames · event-based HS–HS (contact FSM, not ACF): "
            f"{n_left} L / {n_right} R ipsilateral cycles"
        )
        if n_cycles and n_cycles != n_left + n_right:
            summary += f" · primary same-side list {n_cycles}"
        summary += f" · {quality}"
    else:
        summary = f"{usable} usable frames · no complete event-based HS–HS cycles."
    return _item(
        "frames_cycles",
        "Usable frames and gait cycles",
        status,
        summary,
        details={
            "usable_frames": usable,
            "n_snapshots": n_snaps,
            "n_detected": n_detected,
            "gait_cycles": n_cycles,
            "n_left_cycles": n_left,
            "n_right_cycles": n_right,
            "method": "event_hs_hs_contact",
            "contact_confidence": _finite(getattr(metrics, "contact_confidence", None)) if metrics else None,
            "reliability_reason": quality,
        },
        reason=None if status == "computed" else quality or "No complete gait cycles.",
        units=units,
        definitions=definitions,
    )


def _acf_item(session: Any) -> tuple[ValidationItem, dict[str, Any]]:
    spectral = _attr(session, "_spectral_report", "spectral_report")
    worker = _attr(session, "_spectral_worker")
    definitions = {
        "acf": (
            "Normalized autocorrelation ρ(τ) of the primary spectral channel "
            "after uniform resample, linear detrend, and z-score."
        ),
        "gait_period": (
            "Lag of a positive local ACF peak in the stride window on the selected "
            "spectral channel. Independent of event-based HS–HS cycle counts. "
            "Withheld when no such peak exists or the clip is too short for ACF."
        ),
    }
    units = {"gait_period_s": "s", "rho": "dimensionless"}
    plot: dict[str, Any] = {}
    spectral_error = _attr(session, "_spectral_error")
    if spectral is None:
        pending = bool(worker is not None and getattr(worker, "is_alive", lambda: False)())
        if spectral_error:
            return (
                _item(
                    "acf",
                    "Autocorrelation and dominant period",
                    "failed",
                    "Autocorrelation failed.",
                    reason=str(spectral_error),
                    units=units,
                    definitions=definitions,
                ),
                plot,
            )
        status: ValidationStatus = "unavailable"
        reason = (
            "Spectral analysis is still running."
            if pending
            else "Autocorrelation has not been computed for this session."
        )
        return (
            _item(
                "acf",
                "Autocorrelation and dominant period",
                status,
                "Autocorrelation not computed." if not pending else "Autocorrelation pending.",
                reason=reason,
                units=units,
                definitions=definitions,
            ),
            plot,
        )
    acf = None
    getter = getattr(spectral, "primary_autocorr", None)
    if callable(getter):
        acf = getter()
    if acf is None:
        autocorr = getattr(spectral, "autocorr", None) or {}
        if isinstance(autocorr, dict) and autocorr:
            acf = next(iter(autocorr.values()))
    if acf is None:
        return (
            _item(
                "acf",
                "Autocorrelation and dominant period",
                "insufficient",
                "Spectral report has no autocorrelation result.",
                reason="Primary ACF channel is missing.",
                units=units,
                definitions=definitions,
            ),
            plot,
        )
    lags = _series_list(getattr(acf, "lags_s", None))
    rho = _series_list(getattr(acf, "rho", None))
    if lags and rho:
        plot = {
            "lags_s": [float(x) for x in lags],
            "rho": [None if v is None or not math.isfinite(float(v)) else float(v) for v in rho],
            "channel_id": str(getattr(acf, "channel_id", "") or ""),
        }
    insufficient = str(getattr(acf, "insufficient_reason", "") or "").strip()
    period = _finite(getattr(acf, "gait_period_s", None))
    if period is None:
        period = _finite(getattr(acf, "stride_period_s", None))
    reliability = str(getattr(acf, "reliability_label", "") or "")
    if insufficient or period is None:
        status = "insufficient"
        channel = str(getattr(acf, "channel_id", "") or "selected channel")
        peak_reason = (
            insufficient
            or "no local ρ(τ) peak in the stride-lag window (0.55–2.20 s, ρ > 0.05)"
        )
        summary = (
            f"Selected-channel ACF ({channel}): period unavailable — {peak_reason}. "
            "This estimator does not use event-based L/R HS–HS cycle counts."
        )
        return (
            _item(
                "acf",
                "Autocorrelation and dominant period",
                status,
                summary,
                details={
                    "channel_id": str(getattr(acf, "channel_id", "") or ""),
                    "n_samples": int(getattr(acf, "n_samples", 0) or 0),
                    "gait_period_s": None,
                    "reliability_label": reliability,
                    "insufficient_reason": insufficient or peak_reason,
                    "method": "selected_channel_acf",
                },
                reason=peak_reason,
                units=units,
                definitions=definitions,
            ),
            plot,
        )
    return (
        _item(
            "acf",
            "Autocorrelation and dominant period",
            "computed",
            f"ACF period {period:.3g} s · {reliability or 'reliability n/a'}",
            details={
                "channel_id": str(getattr(acf, "channel_id", "") or ""),
                "n_samples": int(getattr(acf, "n_samples", 0) or 0),
                "gait_period_s": period,
                "step_period_s": _finite(getattr(acf, "step_period_s", None)),
                "reliability": _finite(getattr(acf, "reliability", None)),
                "reliability_label": reliability,
            },
            units=units,
            definitions=definitions,
        ),
        plot,
    )


def _spectrum_item(session: Any) -> tuple[ValidationItem, dict[str, Any]]:
    spectral = _attr(session, "_spectral_report", "spectral_report")
    worker = _attr(session, "_spectral_worker")
    definitions = {
        "fft": "One-sided rFFT periodogram (Hann, density scaling) of the z-scored primary channel. Knee raw series are anatomical flexion (deg); ACF/FFT use the z-scored detrend.",
        "psd": "Welch PSD of the same series. Dominant frequency is searched in the walking band.",
        **{k: FEATURE_DEFINITIONS[k] for k in (
            "dominant_frequency_hz",
            "spectral_centroid_hz",
            "spectral_bandwidth_hz",
            "spectral_entropy",
            "harmonic_ratio",
        ) if k in FEATURE_DEFINITIONS},
    }
    units = {
        "frequency": "Hz",
        "psd": "units^2/Hz",
        "spectral_entropy": "0–1",
        "harmonic_ratio": "dimensionless",
    }
    plot: dict[str, Any] = {}
    spectral_error = _attr(session, "_spectral_error")
    if spectral is None:
        pending = bool(worker is not None and getattr(worker, "is_alive", lambda: False)())
        if spectral_error:
            return (
                _item(
                    "spectrum",
                    "FFT/PSD and spectral features",
                    "failed",
                    "FFT/PSD failed.",
                    reason=str(spectral_error),
                    units=units,
                    definitions=definitions,
                ),
                plot,
            )
        return (
            _item(
                "spectrum",
                "FFT/PSD and spectral features",
                "unavailable",
                "FFT/PSD pending." if pending else "FFT/PSD has not been computed.",
                reason=(
                    "Spectral analysis is still running."
                    if pending
                    else "No spectral report for this session."
                ),
                units=units,
                definitions=definitions,
            ),
            plot,
        )
    spec = None
    getter = getattr(spectral, "primary_spectrum", None)
    if callable(getter):
        spec = getter()
    if spec is None:
        spectra = getattr(spectral, "spectra", None) or {}
        if isinstance(spectra, dict) and spectra:
            spec = next(iter(spectra.values()))
    features = getattr(spectral, "features", None)
    feature_rows: list[dict[str, Any]] = []
    if features is not None:
        for record in getattr(features, "records", None) or []:
            if str(getattr(record, "scope", "") or "") not in {"session", ""}:
                continue
            available = bool(getattr(record, "available", False))
            value = _finite(getattr(record, "value", None))
            feature_rows.append(
                {
                    "name": str(getattr(record, "name", "") or ""),
                    "label": str(getattr(record, "label", "") or getattr(record, "name", "") or ""),
                    "value": value if available else None,
                    "units": str(getattr(record, "units", "") or ""),
                    "available": available,
                    "display": record.display_value() if hasattr(record, "display_value") else (
                        _fmt(value, unit=str(getattr(record, "units", "") or "")) if available else "unavailable"
                    ),
                    "definition": str(getattr(record, "definition", "") or ""),
                }
            )
    fft_f = _series_list(getattr(spec, "fft_freq_hz", None)) if spec is not None else []
    fft_p = _series_list(getattr(spec, "fft_power", None)) if spec is not None else []
    psd_f = _series_list(getattr(spec, "freq_hz", None)) if spec is not None else []
    psd_p = _series_list(getattr(spec, "psd", None)) if spec is not None else []
    has_fft = len(fft_f) >= 8 and len(fft_p) >= 8
    has_psd = len(psd_f) >= 8 and len(psd_p) >= 8
    if has_fft or has_psd:
        plot = {
            "channel_id": str(getattr(spec, "channel_id", "") or "") if spec is not None else "",
            "fft_freq_hz": [float(x) for x in fft_f] if has_fft else [],
            "fft_power": [float(x) for x in fft_p] if has_fft else [],
            "psd_freq_hz": [float(x) for x in psd_f] if has_psd else [],
            "psd": [float(x) for x in psd_p] if has_psd else [],
        }
    if spec is None or not (has_fft or has_psd):
        warns = [str(w) for w in (getattr(spectral, "warnings", None) or []) if str(w).strip()]
        return (
            _item(
                "spectrum",
                "FFT/PSD and spectral features",
                "insufficient",
                "Spectral analysis ran, but FFT/PSD series were not produced.",
                details={"features": feature_rows, "warnings": warns[:6]},
                reason=warns[0] if warns else "Primary spectrum is empty.",
                units=units,
                definitions=definitions,
            ),
            plot,
        )
    f0 = _finite(getattr(spec, "dominant_hz", None))
    entropy = _finite(getattr(spec, "spectral_entropy", None))
    centroid = _finite(getattr(spec, "spectral_centroid_hz", None))
    computed_features = [row for row in feature_rows if row.get("available")]
    summary = (
        f"FFT/PSD computed · f0 {_fmt(f0, unit='Hz')}"
        + (f" · H {_fmt(entropy)}" if entropy is not None else "")
        + (f" · {len(computed_features)} spectral features" if computed_features else "")
    )
    return (
        _item(
            "spectrum",
            "FFT/PSD and spectral features",
            "computed",
            summary,
            details={
                "channel_id": str(getattr(spec, "channel_id", "") or ""),
                "method": str(getattr(spec, "method", "") or ""),
                "dominant_hz": f0,
                "dominant_power": _finite(getattr(spec, "dominant_power", None)),
                "spectral_centroid_hz": centroid,
                "spectral_bandwidth_hz": _finite(getattr(spec, "spectral_bandwidth_hz", None)),
                "spectral_entropy": entropy,
                "harmonic_ratio": _finite(getattr(spec, "harmonic_ratio", None)),
                "n_fft_bins": len(fft_f) if has_fft else 0,
                "n_psd_bins": len(psd_f) if has_psd else 0,
                "features": feature_rows,
            },
            units=units,
            definitions=definitions,
        ),
        plot,
    )


def _symmetry_item(session: Any) -> ValidationItem:
    moments = _attr(session, "_gait_asymmetry_moments", "gait_asymmetry_moments")
    report3d = _attr(session, "_asymmetry_report", "asymmetry_report")
    definitions = {
        "si_norm": SI_NORM_DEFINITION,
        "robinson_si": ROBINSON_SI_DEFINITION,
        "asi": ASI_DEFINITION,
        "phase": "Ipsilateral heel-strike cycles are resampled to 0–100% before comparison.",
        "peak_timing": TIMING_DEFINITION,
        "overall_vs_knee": OVERALL_VS_KNEE_DEFINITION,
    }
    units = {
        "rom": "deg",
        "si_percent": "signed relative percent of (L+R)/2",
        "asi_percent": "unsigned relative percent",
        "si_norm": "0–1 similarity (1 = symmetry)",
        "peak_difference": "deg",
        "timing_difference": "percentage points (circular)",
    }
    comparisons = list(getattr(moments, "joint_comparisons", None) or []) if moments is not None else []
    if comparisons:
        rows = []
        for row in comparisons:
            rows.append(
                {
                    "name": str(getattr(row, "name", "") or ""),
                    "rom_left": _finite(getattr(row, "rom_left", None)),
                    "rom_right": _finite(getattr(row, "rom_right", None)),
                    "peak_difference": _finite(getattr(row, "peak_difference", None)),
                    "timing_difference_pct": _finite(getattr(row, "timing_difference_pct", None)),
                    "timing_difference_units": "percentage points",
                    "si_percent": _finite(getattr(row, "si_percent", None)),
                    "asi_percent": _finite(getattr(row, "asi_percent", None)),
                    "si_norm": _finite(getattr(row, "si_norm", None)),
                    "n_left_cycles": int(getattr(row, "n_left_cycles", 0) or 0),
                    "n_right_cycles": int(getattr(row, "n_right_cycles", 0) or 0),
                    "quality": str(getattr(row, "quality", "") or ""),
                }
            )
        knee = next((r for r in rows if "knee" in r["name"]), rows[0])
        si = knee.get("si_norm")
        quality = str(knee.get("quality") or "")
        if quality == "insufficient" or (si is None and knee.get("rom_left") is None):
            summary = (
                "Knee L/R comparison insufficient — SI / ASI / SI_norm omitted."
            )
        elif si is not None:
            summary = (
                f"Knee mean-curve ROM SI_norm {_fmt(si)} "
                f"(max−min of the HS–HS mean flexion curve; not overall Results symmetry). "
                f"N {knee.get('n_left_cycles')}/{knee.get('n_right_cycles')} HS–HS."
            )
            rom_row = None
            for s in list(getattr(report3d, "scalars", None) or []):
                if str(getattr(s, "name", "") or "") == "knee_flexion_rom":
                    rom_row = s
                    break
            p95 = _finite(getattr(rom_row, "si_norm", None)) if rom_row is not None else None
            if p95 is not None:
                summary += (
                    f" Asymmetry knee flexion ROM SI_norm (P95−P5) {_fmt(p95)} "
                    "uses a different ROM input and the same SI_norm formula."
                )
        else:
            summary = "Phase-normalized left/right joint comparison computed."
        return _item(
            "symmetry",
            "Left-versus-right symmetry",
            "computed",
            summary,
            details={"joints": rows, "outcome": str(getattr(moments, "outcome", "") or "")},
            units=units,
            definitions=definitions,
        )
    scalars = list(getattr(report3d, "scalars", None) or []) if report3d is not None else []
    usable = [
        s
        for s in scalars
        if bool(getattr(s, "reliable", False))
        and _finite(getattr(s, "left", None)) is not None
        and _finite(getattr(s, "right", None)) is not None
    ]
    if usable:
        knee = next((s for s in usable if "knee" in str(getattr(s, "name", ""))), usable[0])
        return _item(
            "symmetry",
            "Left-versus-right symmetry",
            "computed",
            (
                f"{getattr(knee, 'display', getattr(knee, 'name', 'L/R'))}: "
                f"L {_fmt(getattr(knee, 'left', None))}  R {_fmt(getattr(knee, 'right', None))}  "
                f"SI {_fmt(getattr(knee, 'si_percent', None))}  "
                f"ASI {_fmt(getattr(knee, 'asi_percent', None))}  "
                f"SI_norm {_fmt(getattr(knee, 'si_norm', None))} "
                f"(P95−P5 ROM when name is knee flexion ROM; not overall Results symmetry)"
            ),
            details={
                "scalars": [
                    {
                        "name": str(getattr(s, "name", "") or ""),
                        "display": str(getattr(s, "display", "") or ""),
                        "left": _finite(getattr(s, "left", None)),
                        "right": _finite(getattr(s, "right", None)),
                        "si_percent": _finite(getattr(s, "si_percent", None)),
                        "asi_percent": _finite(getattr(s, "asi_percent", None)),
                        "si_norm": _finite(getattr(s, "si_norm", None)),
                        "difference": _finite(getattr(s, "difference", None)),
                        "difference_units": str(getattr(s, "difference_units", "") or ""),
                        "n_left_cycles": int(getattr(s, "n_left_cycles", 0) or 0),
                        "n_right_cycles": int(getattr(s, "n_right_cycles", 0) or 0),
                        "quality": str(getattr(s, "quality", "") or ""),
                        "formula": str(getattr(s, "formula", "") or ""),
                    }
                    for s in usable
                    if str(getattr(s, "name", "")).startswith(("knee", "hip", "ankle"))
                    or "flexion" in str(getattr(s, "name", ""))
                ]
            },
            units=units,
            definitions=definitions,
        )
    if moments is not None and str(getattr(moments, "outcome", "") or "") == "analysis_failed":
        notes = [str(n) for n in (getattr(moments, "notes", None) or []) if str(n).strip()]
        return _item(
            "symmetry",
            "Left-versus-right symmetry",
            "failed",
            "Left–right comparison failed.",
            reason=notes[-1] if notes else "Asymmetry analysis failed.",
            units=units,
            definitions=definitions,
        )
    if moments is not None or report3d is not None:
        return _item(
            "symmetry",
            "Left-versus-right symmetry",
            "insufficient",
            "Left and right cycles could not be compared.",
            reason="Need confident left and right gait cycles for phase-normalized comparison.",
            units=units,
            definitions=definitions,
        )
    return _item(
        "symmetry",
        "Left-versus-right symmetry",
        "unavailable",
        "Left–right symmetry has not been computed.",
        reason="Asymmetry analysis has not run.",
        units=units,
        definitions=definitions,
    )


def _moments_item(session: Any) -> tuple[ValidationItem, dict[str, Any] | None]:
    moments_report = _attr(session, "_gait_asymmetry_moments", "gait_asymmetry_moments")
    definitions = {
        "score": (
            f"0–100 combined left–right difference after 0–100% phase normalization. "
            f"A moment requires score ≥ {SCORE_THRESHOLD:.0f} for ≥ {MIN_DURATION_S:.2f} s."
        ),
        "confidence": f"Frames below visibility/confidence {MIN_CONFIDENCE:.2f} are not classified.",
    }
    units = {"score": "0–100", "time_s": "s", "confidence": "0–1"}
    if moments_report is None:
        return (
            _item(
                "moments",
                "Asymmetric moments",
                "unavailable",
                "Asymmetric-moment detection has not run.",
                reason="No gait-asymmetry moment report.",
                units=units,
                definitions=definitions,
            ),
            None,
        )
    outcome = str(getattr(moments_report, "outcome", "") or "")
    rows = []
    strongest = None
    for moment in list(getattr(moments_report, "moments", None) or []):
        payload = {
            "frame_index": int(getattr(moment, "frame_index", 0) or 0),
            "time_s": _finite(getattr(moment, "time_s", None)),
            "score": _finite(getattr(moment, "score", None)),
            "confidence": _finite(getattr(moment, "confidence", None)),
            "involved_joints": list(getattr(moment, "involved_joints", None) or []),
            "severity": str(getattr(moment, "severity", "") or ""),
            "duration_s": _finite(getattr(moment, "duration_s", None)),
            "phase_pct": _finite(getattr(moment, "phase_pct", None)),
            "explanation": str(getattr(moment, "explanation", "") or ""),
        }
        contribs = []
        for item in list(getattr(moment, "contributing_measurements", None) or []):
            contribs.append(
                {
                    "measurement": str(getattr(item, "measurement", "") or ""),
                    "left_value": _finite(getattr(item, "left_value", None)),
                    "right_value": _finite(getattr(item, "right_value", None)),
                    "difference": _finite(getattr(item, "difference", None)),
                    "units": str(getattr(item, "units", "") or ""),
                    "score": _finite(getattr(item, "score", None)),
                }
            )
        payload["contributions"] = contribs
        rows.append(payload)
        if strongest is None:
            strongest = payload
    if rows:
        top = rows[0]
        joints = ", ".join(str(j).replace("_", " ") for j in top.get("involved_joints") or []) or "—"
        return (
            _item(
                "moments",
                "Asymmetric moments",
                "computed",
                (
                    f"{len(rows)} moment(s) · strongest score {_fmt(top.get('score'))} "
                    f"at {_fmt(top.get('time_s'), unit='s')} (frame {top.get('frame_index')}) · {joints}"
                ),
                details={
                    "outcome": outcome or "moments",
                    "n_moments": len(rows),
                    "score_threshold": SCORE_THRESHOLD,
                    "min_duration_s": MIN_DURATION_S,
                    "moments": rows,
                },
                units=units,
                definitions=definitions,
            ),
            strongest,
        )
    if outcome == "analysis_failed":
        notes = [str(n) for n in (getattr(moments_report, "notes", None) or []) if str(n).strip()]
        return (
            _item(
                "moments",
                "Asymmetric moments",
                "failed",
                "Asymmetric-moment detection failed.",
                reason=notes[-1] if notes else "Detection did not complete.",
                details={"outcome": outcome},
                units=units,
                definitions=definitions,
            ),
            None,
        )
    if outcome == "symmetric_gait" or bool(getattr(moments_report, "available", False)):
        return (
            _item(
                "moments",
                "Asymmetric moments",
                "computed",
                "Symmetric gait: no moment exceeded the documented threshold and duration.",
                details={
                    "outcome": "symmetric_gait",
                    "n_moments": 0,
                    "score_threshold": SCORE_THRESHOLD,
                    "min_duration_s": MIN_DURATION_S,
                    "moments": [],
                },
                units=units,
                definitions=definitions,
            ),
            None,
        )
    notes = [str(n) for n in (getattr(moments_report, "notes", None) or []) if str(n).strip()]
    return (
        _item(
            "moments",
            "Asymmetric moments",
            "insufficient",
            "Insufficient data for asymmetric-moment detection.",
            reason=notes[-1] if notes else "Left and right cycles could not be compared with enough confident frames.",
            details={"outcome": outcome or "insufficient_data", "n_moments": 0},
            units=units,
            definitions=definitions,
        ),
        None,
    )


def _reconstruction_item(
    session: Any,
    strongest: dict[str, Any] | None,
    moments_item: ValidationItem,
) -> ValidationItem:
    recording = _attr(session, "gait_motion")
    definitions = {
        "reconstruction": (
            "Existing pelvis-centered 16-joint wire at the selected video frame. "
            "Not a second 3D solver. Depth is inferred monocular 3D unless calibrated."
        )
    }
    if strongest is None:
        status = moments_item.status if moments_item.status != "computed" else "unavailable"
        reason = moments_item.reason or moments_item.summary
        return _item(
            "reconstruction3d",
            "3D reconstruction at strongest moment",
            "unavailable" if status == "computed" else status,
            "No asymmetric moment to reconstruct." if status == "computed" else reason,
            reason=reason,
            definitions=definitions,
        )
    if recording is None or not getattr(recording, "snapshots", None):
        return _item(
            "reconstruction3d",
            "3D reconstruction at strongest moment",
            "failed",
            "Moment exists, but no skeleton snapshot is available.",
            reason="Gait recording snapshots are missing.",
            details=dict(strongest),
            definitions=definitions,
        )
    frame_index = int(strongest.get("frame_index") or 0)
    snaps = list(recording.snapshots)
    snap = next((s for s in snaps if int(s.frame_index) == frame_index), None)
    if snap is None:
        target_t = _finite(strongest.get("time_s"))
        if target_t is not None:
            snap = min(snaps, key=lambda s: abs(float(s.time_s) - target_t))
    if snap is None:
        return _item(
            "reconstruction3d",
            "3D reconstruction at strongest moment",
            "failed",
            "No skeleton snapshot matches the strongest moment frame.",
            reason=f"Frame {frame_index} is not in the recording.",
            details=dict(strongest),
            definitions=definitions,
        )
    n_joints = len(getattr(snap, "joints", None) or {})
    return _item(
        "reconstruction3d",
        "3D reconstruction at strongest moment",
        "computed",
        (
            f"Full 16-joint wire at {float(snap.time_s):.2f} s "
            f"(frame {int(snap.frame_index)}) · {n_joints} joints in snapshot"
        ),
        details={
            "frame_index": int(snap.frame_index),
            "time_s": float(snap.time_s),
            "n_joints": n_joints,
            "coordinate_system": str(getattr(recording, "coordinate_system", "") or ""),
            "involved_joints": list(strongest.get("involved_joints") or []),
            "score": strongest.get("score"),
        },
        units={"time_s": "s"},
        definitions=definitions,
    )


def _confidence_item(session: Any, items: list[ValidationItem]) -> tuple[ValidationItem, dict[str, Any], list[str]]:
    summary = _attr(session, "_analysis_summary_cache", "analysis_summary")
    moments = _attr(session, "_gait_asymmetry_moments", "gait_asymmetry_moments")
    spectral = _attr(session, "_spectral_report", "spectral_report")
    recording = _attr(session, "gait_motion")
    limitations: list[str] = []
    conf: dict[str, Any] = {}
    if summary is not None:
        for key, field in (
            ("analysis", getattr(summary, "analysis_confidence", None)),
            ("tracking", getattr(summary, "tracking_confidence", None)),
            ("pipeline", getattr(summary, "pipeline_confidence", None)),
            ("video", getattr(summary, "video_quality", None)),
        ):
            if field is None:
                continue
            conf[key] = {
                "label": str(getattr(field, "label", key) or key),
                "value": str(getattr(field, "value", "") or ""),
                "available": bool(getattr(field, "available", False)),
                "reason": str(getattr(field, "reason", "") or ""),
                "tier": str(getattr(field, "tier", "") or ""),
            }
            if not getattr(field, "available", True) and getattr(field, "reason", ""):
                limitations.append(str(field.reason))
    if recording is not None:
        coord = str(getattr(recording, "coordinate_system", "") or "")
        kind = str(getattr(recording, "source_kind", "") or "")
        if "inferred" in coord.lower() or kind in {"pose_estimation", "unknown"}:
            limitations.append(
                "3D coordinates are inferred monocular estimates, not laboratory millimetres."
            )
    ba = _attr(session, "_biomech_analysis", "biomech_analysis")
    vq = getattr(ba, "video_quality", None) if ba is not None else None
    layers: dict[str, Any] = {}
    if vq is not None:
        cov = getattr(vq, "pose_coverage", None)
        land = getattr(vq, "landmark_confidence", None)
        if cov is not None:
            layers[LAYER_POSE_COVERAGE] = cov.to_dict()
        if land is not None:
            layers[LAYER_LANDMARK_CONFIDENCE] = land.to_dict()
    if LAYER_POSE_COVERAGE not in layers and recording is not None:
        snaps = list(getattr(recording, "snapshots", None) or [])
        n_det = sum(1 for s in snaps if (getattr(s, "metadata", None) or {}).get("detected", True))
        layers[LAYER_POSE_COVERAGE] = pose_coverage_evidence(
            n_frames=len(snaps), n_detected=n_det
        ).to_dict()
    cycles = _attr(session, "_gait_cycle", "gait_cycle")
    if cycles is not None:
        clist = list(getattr(cycles, "cycles", None) or [])
        metrics = getattr(cycles, "metrics", None)
        n_left = sum(1 for c in clist if str(getattr(c, "boundary_side", "")).startswith("left"))
        n_right = sum(1 for c in clist if str(getattr(c, "boundary_side", "")).startswith("right"))
        layers[LAYER_CYCLE_DETECTION] = cycle_detection_evidence(
            n_cycles=len(clist),
            n_left=n_left,
            n_right=n_right,
            contact_confidence=_finite(getattr(metrics, "contact_confidence", None)) if metrics else None,
            reason=str(getattr(metrics, "reliability_reason", "") or "") if metrics else "",
        ).to_dict()
    ac = None
    if spectral is not None:
        ac = None
        mapping = getattr(spectral, "autocorr", None) or {}
        if mapping:
            ac = next(iter(mapping.values()), None)
        if ac is not None:
            layers[LAYER_SPECTRAL] = spectral_reliability_evidence(
                duration_s=None,
                gap_fraction=None,
                n_cycles=_finite(getattr(ac, "n_stride_cycles", None)),
                insufficient_reason=getattr(ac, "insufficient_reason", None),
            ).to_dict()
            if getattr(ac, "reliability_explanation", ""):
                layers[LAYER_SPECTRAL]["explanation"] = str(ac.reliability_explanation)
    sm = getattr(ba, "stability_margin", None) if ba is not None else None
    layers[LAYER_BIOMECHANICAL] = biomechanical_validity_evidence(
        calibrated_metric="calibrated" in str(getattr(recording, "coordinate_system", "") or "").lower(),
        stability_reportable=getattr(sm, "reportable", None) if sm is not None else False,
        reason=str(getattr(sm, "experimental_note", "") or "") if sm is not None else "",
    ).to_dict()
    for layer in layers.values():
        if layer.get("status") in {"insufficient", "unavailable"} and layer.get("explanation"):
            limitations.append(f"{layer.get('layer')}: {layer.get('explanation')}")
    limitations.append("Joint moments / GRF torques are not claimed without measured force plates.")
    limitations.append("Outputs are derived movement measures, not a medical diagnosis.")
    limitations.append(LANDMARK_NOT_TRACKING)
    for item in items:
        if item.status in {"insufficient", "failed"} and item.reason:
            limitations.append(f"{item.title}: {item.reason}")
    if spectral is not None:
        for warn in list(getattr(spectral, "warnings", None) or [])[:4]:
            text = str(warn).strip()
            if text:
                limitations.append(f"Spectral: {text}")
    if moments is not None:
        for note in list(getattr(moments, "notes", None) or [])[-2:]:
            text = str(note).strip()
            if text and "not a medical" not in text.lower() and "resampled to 0" not in text.lower():
                limitations.append(text)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for line in limitations:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    available_conf = [v for v in conf.values() if v.get("available") and v.get("value")]
    if available_conf:
        status: ValidationStatus = "computed"
        summary_txt = " · ".join(f"{v['label']}: {v['value']}" for v in available_conf[:3])
        reason = None
    elif summary is None and recording is None:
        status = "unavailable"
        summary_txt = "Confidence has not been assessed."
        reason = "Load and analyze a walk first."
    else:
        status = "insufficient"
        summary_txt = "Confidence fields were not all available for this session."
        reason = "See limitations."
    item = _item(
        "confidence",
        "Confidence and limitations",
        status,
        summary_txt,
        details={"fields": conf, "limitations": unique, "layers": layers},
        reason=reason,
        definitions=layer_documentation() | {
            "confidence": (
                "Layers are pose coverage, landmark visibility, cycle detection, "
                "spectral reliability, and biomechanical validity. They are not "
                "interchangeable. Detected landmarks are not accurate tracking."
            )
        },
    )
    return item, conf, unique


def collect_research_validation(session: Any) -> ResearchValidationSnapshot:
    """Build the Results research-validation snapshot from real session objects."""
    video = _video_metadata(session)
    wire = _wire_item(session)
    cycles = _cycle_item(session, wire)
    acf, acf_plot = _acf_item(session)
    spectrum, spec_plot = _spectrum_item(session)
    symmetry = _symmetry_item(session)
    moments, strongest = _moments_item(session)
    reconstruction = _reconstruction_item(session, strongest, moments)
    early = [wire, cycles, acf, spectrum, symmetry, moments, reconstruction]
    confidence_item, conf, limitations = _confidence_item(session, early)
    items = [*early, confidence_item]
    plot_series: dict[str, Any] = {}
    if acf_plot:
        plot_series["acf"] = acf_plot
    if spec_plot:
        plot_series["spectrum"] = spec_plot
    if strongest is not None and reconstruction.status == "computed":
        plot_series["moment3d"] = {
            "frame_index": strongest.get("frame_index"),
            "time_s": strongest.get("time_s"),
            "involved_joints": list(strongest.get("involved_joints") or []),
        }
    return ResearchValidationSnapshot(
        items=items,
        video_metadata=video,
        limitations=limitations,
        confidence=conf,
        strongest_moment=strongest,
        plot_series=plot_series,
        source=str(video.get("source") or ""),
    )


__all__ = [
    "ResearchValidationSnapshot",
    "ValidationItem",
    "ValidationStatus",
    "collect_research_validation",
]
