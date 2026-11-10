"""Explainable spectral feature extraction (Erik requirement 3).

All scalars are **derived** from ``SpectralAnalysisReport`` (PSD, ACF, R(0),
lagged cross-correlation) plus optional detected gait cycles. Missing inputs
yield ``value=None`` and ``available=False`` — never a fabricated 0.

These features describe periodicity, regularity, coordination, and left–right
difference of pose time series. They are **not** clinically validated disease
indicators.

Mathematical definitions
------------------------
Let P(f) be a one-sided PSD (Welch or rFFT density) and B the gait band.

Dominant frequency
    f0 = arg max_{f in B} P(f)

Dominant spectral power
    P(f0)

Band power
    G = ∫_B P(f) df  (sum of P[i] · Δf over bins in B)

Normalized spectral entropy
    p_i = P_i / Σ P_j  for bins in B with P>0
    H = (−Σ p_i log p_i) / log(N)   in [0, 1] when N>1

Spectral centroid
    c = Σ f_i p_i

Spectral bandwidth
    √(Σ p_i (f_i − c)²)

Harmonic ratio
    P(2 f0) / P(f0) at the nearest bin to 2 f0 (if inside B)

Stride / harmonic power ratio
    ∫_{B_stride} P / ∫_{B_harm} P

ACF stride peak / lag / prominence
    From normalized ρ(k) in the configured stride-lag window

Stride duration
    lag of that peak (s)

Stride-to-stride variability
    coefficient of variation of detected cycle durations (if ≥2 cycles)

Regularity
    ρ at the stride lag (higher → more periodic). Unavailable if no peak.

R(0) eigenvalues / explained variance
    From the symmetric correlation matrix (eigh)

Cross-correlation peak lag
    arg max_τ ρ_{LR}(τ) with smallest |τ| among near-maxima

Relative phase
    wrap_180( 360° · τ_peak / T_stride )

Amplitude
    RMS of the processed (finite) samples
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from stablewalk.analysis.spectral.channels import LEFT_RIGHT_PAIRS, spec_for
from stablewalk.analysis.spectral.config import SpectralAnalysisConfig
from stablewalk.analysis.spectral.spectrum import (
    _centroid,
    _dominant_in_band,
    rfft_periodogram,
)

KIND_DERIVED = "derived"
UNAVAILABLE = "unavailable"

# Stable names for the session-level export vector (order is part of the schema).
SESSION_VECTOR_KEYS: tuple[str, ...] = (
    "dominant_frequency_hz",
    "dominant_power",
    "gait_band_power",
    "spectral_entropy",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "harmonic_ratio",
    "stride_harmonic_power_ratio",
    "acf_stride_peak",
    "acf_stride_lag_s",
    "acf_peak_prominence",
    "stride_duration_s",
    "stride_cv",
    "regularity_acf",
    "rms_amplitude",
    "leading_eigenvalue",
    "explained_variance_mode1",
    "lr_freq_delta_hz",
    "lr_entropy_delta",
    "lr_amplitude_delta",
    "lr_xcorr_lag_s",
    "lr_relative_phase_deg",
    "spectral_cadence_steps_per_min",
)


FEATURE_DEFINITIONS: dict[str, str] = {
    "dominant_frequency_hz": (
        "Dominant frequency (Hz): the walking-band frequency with the most "
        "spectral power. Often near stride rate (~0.8–1.2 Hz in typical walking)."
    ),
    "dominant_power": (
        "Peak power (z²/Hz): P(f₀) at the dominant frequency on the "
        "normalized (z-scored) signal."
    ),
    "gait_band_power": "Gait-band power (units²): sum of PSD bins in the walking band × Δf.",
    "spectral_entropy": (
        "Spectral entropy (0–1): how spread the walking-band power is. "
        "Lower = a more peaked, periodic spectrum."
    ),
    "spectral_centroid_hz": (
        "Spectral centroid (Hz): power-weighted mean frequency in the walking band."
    ),
    "spectral_bandwidth_hz": (
        "Spectral bandwidth (Hz): power-weighted RMS spread around the centroid."
    ),
    "harmonic_ratio": (
        "Harmonic ratio (dimensionless): P(2 f₀) / P(f₀) when 2 f₀ lies in the "
        "walking band. Higher = stronger first harmonic."
    ),
    "stride_harmonic_power_ratio": "Band power in the stride band / harmonic band.",
    "acf_stride_peak": "Normalized autocorrelation ρ at the stride lag (higher = more periodic).",
    "acf_stride_lag_s": "Lag of the main stride-related ACF peak (s).",
    "acf_peak_prominence": "ρ(peak) − min ρ on [0, peak] (dimensionless).",
    "stride_duration_s": "Gait period (s): ACF stride lag. Unavailable when cycles are insufficient.",
    "stride_cv": "Coefficient of variation of detected gait-cycle durations.",
    "regularity_acf": "Same as acf_stride_peak; higher → more periodic.",
    "rms_amplitude": "RMS of detrended samples in the channel’s native units (deg or m).",
    "leading_eigenvalue": "Largest eigenvalue of the joint-by-joint R(0) correlation matrix.",
    "explained_variance_mode1": "Fraction of R(0) trace explained by mode 1.",
    "principal_mode_loadings": "Absolute loadings of mode 1 on matrix channels.",
    "lr_freq_delta_hz": "Right−left dominant frequency (Hz).",
    "lr_entropy_delta": "Right−left spectral entropy.",
    "lr_amplitude_delta": "Right−left RMS amplitude (channel units).",
    "lr_xcorr_lag_s": "Lag of the L/R cross-correlation peak (s).",
    "lr_relative_phase_deg": "360° × (xcorr lag) / stride period, wrapped to ±180°.",
    "spectral_cadence_steps_per_min": (
        "Cadence (steps/min) from the selected-channel ACF: 60/T_step when an "
        "independent step peak exists; otherwise 120/T_stride (assumes two steps "
        "per stride). Not rewritten to enforce step = stride/2. Does not replace "
        "the Overview event cadence."
    ),
}


@dataclass
class FeatureRecord:
    """One explainable scalar with provenance. ``value`` is None if unavailable."""

    name: str
    label: str
    value: float | None
    units: str
    definition: str
    available: bool
    quality: str
    scope: str  # session | cycle | left | right | bilateral
    side: str | None = None
    channel_id: str | None = None
    pair_id: str | None = None
    cycle_index: int | None = None
    left_value: float | None = None
    right_value: float | None = None
    notes: list[str] = field(default_factory=list)
    kind: str = KIND_DERIVED

    def display_value(self) -> str:
        if not self.available or self.value is None or not math.isfinite(self.value):
            if self.notes:
                for note in self.notes:
                    if "insufficient cycles" in str(note).lower():
                        return "insufficient cycles"
            return UNAVAILABLE
        if self.units:
            return f"{self.value:.4g} {self.units}"
        return f"{self.value:.4g}"

    def unavailable_reason(self) -> str | None:
        if self.available and self.value is not None and math.isfinite(float(self.value)):
            return None
        if self.notes:
            joined = "; ".join(str(n) for n in self.notes if str(n).strip())
            if joined:
                return joined
        if self.quality == "insufficient":
            return "insufficient data for this feature"
        return "value is not finite"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "value": self.value,
            "units": self.units,
            "definition": self.definition,
            "formula_id": self.name,
            "available": self.available,
            "quality": self.quality,
            "scope": self.scope,
            "side": self.side,
            "channel_id": self.channel_id,
            "pair_id": self.pair_id,
            "cycle_index": self.cycle_index,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "notes": list(self.notes),
            "unavailable_reason": self.unavailable_reason(),
            "kind": self.kind,
        }


@dataclass
class SpectralFeatureSet:
    """Structured vectors: session, sides, and per-cycle (when cycles exist)."""

    records: list[FeatureRecord] = field(default_factory=list)
    session_vector: dict[str, float | None] = field(default_factory=dict)
    n_detected_cycles: int = 0
    n_valid_cycles: int = 0
    video_quality_score: float | None = None
    fps: float = 0.0
    duration_s: float = 0.0
    primary_channel_id: str = ""
    warnings: list[str] = field(default_factory=list)
    kind: str = KIND_DERIVED
    disclaimer: str = (
        "Derived spectral features from monocular pose. Not a clinically "
        "validated disease or instability indicator."
    )

    def records_for(
        self,
        *,
        scope: str | None = None,
        cycle_index: int | None = None,
    ) -> list[FeatureRecord]:
        out = list(self.records)
        if scope:
            out = [r for r in out if r.scope == scope]
        if cycle_index is not None:
            out = [r for r in out if r.cycle_index == cycle_index]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "stablewalk-spectral-features",
            "schema_version": "1.0",
            "kind": self.kind,
            "disclaimer": self.disclaimer,
            "fps": self.fps,
            "sampling_rate_hz": self.fps,
            "duration_s": self.duration_s,
            "primary_channel_id": self.primary_channel_id,
            "n_detected_cycles": self.n_detected_cycles,
            "n_valid_cycles": self.n_valid_cycles,
            "cycle_definition": (
                "Ipsilateral heel-strike to heel-strike intervals from "
                "analyze_gait_cycles; per-cycle features use those bounds."
            ),
            "video_quality_score": self.video_quality_score,
            "session_vector": dict(self.session_vector),
            "definitions": dict(FEATURE_DEFINITIONS),
            "formula_identifiers": {name: name for name in FEATURE_DEFINITIONS},
            "records": [r.to_dict() for r in self.records],
            "warnings": list(self.warnings),
        }


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _quality(available: bool, reliability: float | None = None) -> str:
    if not available:
        return "insufficient"
    if reliability is None:
        return "moderate"
    if reliability >= 0.65:
        return "moderate"
    if reliability > 0.0:
        return "low"
    return "insufficient"


def _record(
    name: str,
    value: float | None,
    *,
    units: str,
    scope: str,
    channel_id: str | None = None,
    side: str | None = None,
    pair_id: str | None = None,
    cycle_index: int | None = None,
    left_value: float | None = None,
    right_value: float | None = None,
    reliability: float | None = None,
    notes: list[str] | None = None,
    label: str | None = None,
) -> FeatureRecord:
    ok = _finite(value)
    return FeatureRecord(
        name=name,
        label=label or name.replace("_", " "),
        value=float(value) if ok else None,
        units=units,
        definition=FEATURE_DEFINITIONS.get(name, ""),
        available=ok,
        quality=_quality(ok, reliability),
        scope=scope,
        side=side,
        channel_id=channel_id,
        pair_id=pair_id,
        cycle_index=cycle_index,
        left_value=float(left_value) if _finite(left_value) else None,
        right_value=float(right_value) if _finite(right_value) else None,
        notes=list(notes or ()),
    )


def _df(freq: np.ndarray) -> float:
    if freq.size < 2:
        return 0.0
    diffs = np.diff(freq)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if diffs.size else 0.0


def _band_bins(
    freq: np.ndarray,
    psd: np.ndarray,
    band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = band
    mask = (freq >= lo) & (freq <= hi) & (freq > 0) & np.isfinite(psd)
    return freq[mask], np.clip(psd[mask], 0.0, None)


def band_power(freq: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float | None:
    f, p = _band_bins(freq, psd, band)
    if f.size == 0:
        return None
    df = _df(freq)
    if df <= 0:
        return float(np.sum(p)) if np.sum(p) > 0 else None
    val = float(np.sum(p) * df)
    return val if math.isfinite(val) else None


def spectral_entropy(freq: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float | None:
    _f, p = _band_bins(freq, psd, band)
    total = float(np.sum(p))
    if total <= 1e-18 or p.size < 2:
        return None
    probs = p / total
    probs = probs[probs > 0]
    if probs.size == 0:
        return None
    if probs.size == 1:
        return 0.0
    entropy = float(-np.sum(probs * np.log(probs)))
    return float(entropy / math.log(probs.size))


def spectral_bandwidth(freq: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float | None:
    f, p = _band_bins(freq, psd, band)
    total = float(np.sum(p))
    if total <= 1e-18 or f.size < 2:
        return None
    centroid = float(np.sum(f * p) / total)
    var = float(np.sum(p * (f - centroid) ** 2) / total)
    if var < 0:
        return None
    return math.sqrt(var)


def rms_amplitude(values: np.ndarray) -> float | None:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return None
    return float(np.sqrt(np.mean(x * x)))


def wrap_phase_deg(degrees: float) -> float:
    wrapped = (float(degrees) + 180.0) % 360.0 - 180.0
    return wrapped


def _psd_arrays(spectrum) -> tuple[np.ndarray, np.ndarray]:
    if spectrum.freq_hz.size:
        return spectrum.freq_hz, spectrum.psd
    return spectrum.fft_freq_hz, spectrum.fft_power


def _channel_frequency_features(
    spectrum,
    series,
    cfg: SpectralAnalysisConfig,
    *,
    scope: str,
    side: str | None,
    cycle_index: int | None = None,
    reliability: float | None = None,
) -> list[FeatureRecord]:
    cid = spectrum.channel_id
    freq, psd = _psd_arrays(spectrum)
    band = cfg.gait_band_hz
    recs = [
        _record(
            "dominant_frequency_hz",
            spectrum.dominant_hz,
            units="Hz",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=reliability,
        ),
        _record(
            "dominant_power",
            spectrum.dominant_power,
            units="units^2/Hz",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=reliability,
        ),
        _record(
            "gait_band_power",
            band_power(freq, psd, band),
            units="units^2",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=reliability,
        ),
        _record(
            "spectral_entropy",
            spectrum.spectral_entropy
            if getattr(spectrum, "spectral_entropy", None) is not None
            else spectral_entropy(freq, psd, band),
            units="1",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=reliability,
        ),
        _record(
            "spectral_centroid_hz",
            spectrum.spectral_centroid_hz,
            units="Hz",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=reliability,
        ),
        _record(
            "spectral_bandwidth_hz",
            spectrum.spectral_bandwidth_hz
            if getattr(spectrum, "spectral_bandwidth_hz", None) is not None
            else spectral_bandwidth(freq, psd, band),
            units="Hz",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=reliability,
        ),
        _record(
            "harmonic_ratio",
            spectrum.harmonic_ratio,
            units="1",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=reliability,
        ),
    ]
    p_stride = band_power(freq, psd, cfg.stride_band_hz)
    p_harm = band_power(freq, psd, cfg.harmonic_band_hz)
    ratio = None
    if _finite(p_stride) and _finite(p_harm) and p_harm and p_harm > 1e-18:
        ratio = p_stride / p_harm
    recs.append(
        _record(
            "stride_harmonic_power_ratio",
            ratio,
            units="1",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=reliability,
            notes=["Stride band / harmonic band power."],
        )
    )
    if series is not None:
        recs.append(
            _record(
                "rms_amplitude",
                rms_amplitude(series.processed),
                units=series.spec.units,
                scope=scope,
                channel_id=cid,
                side=side,
                cycle_index=cycle_index,
                reliability=reliability,
            )
        )
    return recs


def _channel_acf_features(
    ac,
    *,
    scope: str,
    side: str | None,
    cycle_index: int | None = None,
) -> list[FeatureRecord]:
    cid = ac.channel_id
    insuff = getattr(ac, "insufficient_reason", None)
    insuff_notes = [str(insuff)] if insuff else []
    peak = None
    for item in ac.peaks:
        if ac.stride_period_s is not None and abs(item.lag_s - ac.stride_period_s) < 1e-9:
            peak = item
            break
    if peak is None and ac.peaks:
        peak = max(ac.peaks, key=lambda p: p.value)
    rel = ac.reliability
    return [
        _record(
            "acf_stride_peak",
            None if peak is None else peak.value,
            units="1",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=rel,
        ),
        _record(
            "acf_stride_lag_s",
            ac.stride_period_s,
            units="s",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=rel,
            notes=insuff_notes,
        ),
        _record(
            "acf_peak_prominence",
            None if peak is None else peak.prominence,
            units="1",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=rel,
        ),
        _record(
            "stride_duration_s",
            ac.stride_period_s,
            units="s",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=rel,
            notes=insuff_notes,
        ),
        _record(
            "regularity_acf",
            None if peak is None else peak.value,
            units="1",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=rel,
            notes=["Normalized ACF at the stride lag."],
        ),
        _record(
            "spectral_cadence_steps_per_min",
            getattr(ac, "cadence_steps_per_min", None),
            units="steps/min",
            scope=scope,
            channel_id=cid,
            side=side,
            cycle_index=cycle_index,
            reliability=rel,
            label="cadence (spectral estimate)",
            notes=list(getattr(ac, "notes", None) or ()),
        ),
    ]


def _as_cycle_list(cycles: Any | None) -> list[Any]:
    """Accept a cycle-analysis result or a plain list of ``DetectedGaitCycle``."""
    if cycles is None:
        return []
    nested = getattr(cycles, "cycles", None)
    if nested is not None and nested is not cycles:
        return list(nested)
    if isinstance(cycles, (list, tuple)):
        return list(cycles)
    return []


def _stride_cv(cycles: Sequence[Any] | None) -> tuple[float | None, int, int]:
    if not cycles:
        return None, 0, 0
    durs = [
        float(c.duration_s)
        for c in cycles
        if getattr(c, "duration_s", 0) and float(c.duration_s) > 1e-6
    ]
    detected = len(cycles)
    valid = len(durs)
    if valid < 2:
        return None, detected, valid
    mean = float(np.mean(durs))
    if mean <= 1e-9:
        return None, detected, valid
    cv = float(np.std(durs, ddof=1) / mean)
    return cv, detected, valid


def _side_of(channel_id: str) -> str | None:
    spec = spec_for(channel_id)
    if spec is not None:
        return spec.side
    if "left_" in channel_id:
        return "left"
    if "right_" in channel_id:
        return "right"
    return "midline"


def extract_spectral_features(
    report,
    *,
    cycles: Any | None = None,
    sequence: Any | None = None,
    config: SpectralAnalysisConfig | None = None,
) -> SpectralFeatureSet:
    """Build session / side / cycle feature records from a spectral report."""
    cfg = config or getattr(report, "config", None) or SpectralAnalysisConfig()
    warnings: list[str] = []
    records: list[FeatureRecord] = []
    primary = str(getattr(report, "univariate_channel_id", "") or "")
    fps = float(report.bundle.fps) if report.bundle is not None else 0.0
    duration = float(report.bundle.metadata.get("duration_s", 0.0)) if report.bundle else 0.0

    cycle_list = _as_cycle_list(cycles)
    stride_cv, n_detected, n_valid = _stride_cv(cycle_list)

    video_q: float | None = None
    if sequence is not None:
        try:
            from stablewalk.analysis.biomechanical.video_quality import assess_video_quality

            video_q = float(assess_video_quality(sequence).overall_quality_score)
        except Exception:
            video_q = None

    if not getattr(report, "available", False):
        warnings.append("Spectral report is unavailable; no features were computed.")
        return SpectralFeatureSet(
            warnings=warnings,
            fps=fps,
            duration_s=duration,
            primary_channel_id=primary,
            n_detected_cycles=n_detected,
            n_valid_cycles=n_valid,
            video_quality_score=video_q,
        )

    ac = report.autocorr.get(primary)
    sp = report.spectra.get(primary)
    series = report.bundle.get(primary) if report.bundle else None
    rel = ac.reliability if ac is not None else None
    if ac is not None and getattr(ac, "insufficient_reason", None):
        warnings.append(str(ac.insufficient_reason))
    if n_detected > 0 and n_detected < cfg.min_cycles_for_stride:
        warnings.append(
            f"insufficient cycles: {n_detected} detected gait cycle(s); "
            f"need ≥ {cfg.min_cycles_for_stride}."
        )

    if sp is not None:
        records.extend(
            _channel_frequency_features(
                sp, series, cfg, scope="session", side=_side_of(primary), reliability=rel
            )
        )
    if ac is not None:
        records.extend(_channel_acf_features(ac, scope="session", side=_side_of(primary)))
    records.append(
        _record(
            "stride_cv",
            stride_cv,
            units="1",
            scope="session",
            channel_id=primary,
            reliability=rel,
            notes=["From detected gait-cycle durations, not from ACF."],
        )
    )
    if n_detected < 2:
        warnings.append("Fewer than two detected gait cycles; stride-to-stride CV is unavailable.")

    eig = (report.eigen or {}).get("correlation") or (report.eigen or {}).get("covariance")
    if eig is not None and eig.eigenvalues.size:
        n_modes = min(int(cfg.n_eigen_modes), int(eig.eigenvalues.size))
        records.append(
            _record(
                "leading_eigenvalue",
                float(eig.eigenvalues[0]),
                units="1",
                scope="session",
                reliability=rel,
            )
        )
        for i in range(n_modes):
            records.append(
                _record(
                    "explained_variance_mode1" if i == 0 else f"explained_variance_mode{i+1}",
                    float(eig.explained_variance[i]) if i < eig.explained_variance.size else None,
                    units="1",
                    scope="session",
                    reliability=rel,
                    label=f"explained variance mode {i+1}",
                    notes=[eig.explanation] if eig.explanation else None,
                )
            )
            if eig.eigenvectors.size and eig.channel_ids:
                loads = np.abs(eig.eigenvectors[:, i])
                top = int(np.argmax(loads))
                records.append(
                    FeatureRecord(
                        name="principal_mode_loadings",
                        label=f"mode {i+1} dominant channel",
                        value=float(loads[top]),
                        units="1",
                        definition=FEATURE_DEFINITIONS["principal_mode_loadings"],
                        available=True,
                        quality=_quality(True, rel),
                        scope="session",
                        channel_id=eig.channel_ids[top],
                        notes=[
                            f"Largest |loading| on {eig.channel_ids[top]} "
                            f"(mode {i+1} of R(0))."
                        ],
                    )
                )

    # Bilateral + per-side copies for the primary pair.
    pair_for_primary = None
    for pair_id, left_id, right_id in LEFT_RIGHT_PAIRS:
        if primary in (left_id, right_id, pair_id):
            pair_for_primary = (pair_id, left_id, right_id)
            break
    if pair_for_primary is None:
        pair_for_primary = LEFT_RIGHT_PAIRS[1]  # knee flexion default

    pair_id, left_id, right_id = pair_for_primary
    left_sp, right_sp = report.spectra.get(left_id), report.spectra.get(right_id)
    left_series = report.bundle.get(left_id) if report.bundle else None
    right_series = report.bundle.get(right_id) if report.bundle else None
    left_ac, right_ac = report.autocorr.get(left_id), report.autocorr.get(right_id)

    if left_sp is not None:
        records.extend(
            _channel_frequency_features(
                left_sp,
                left_series,
                cfg,
                scope="left",
                side="left",
                reliability=left_ac.reliability if left_ac else rel,
            )
        )
    if right_sp is not None:
        records.extend(
            _channel_frequency_features(
                right_sp,
                right_series,
                cfg,
                scope="right",
                side="right",
                reliability=right_ac.reliability if right_ac else rel,
            )
        )
    if left_ac is not None:
        records.extend(_channel_acf_features(left_ac, scope="left", side="left"))
    if right_ac is not None:
        records.extend(_channel_acf_features(right_ac, scope="right", side="right"))

    def _val(scope: str, name: str) -> float | None:
        for rec in records:
            if rec.scope == scope and rec.name == name and rec.available:
                return rec.value
        return None

    lf, rf = _val("left", "dominant_frequency_hz"), _val("right", "dominant_frequency_hz")
    le, re = _val("left", "spectral_entropy"), _val("right", "spectral_entropy")
    la, ra = _val("left", "rms_amplitude"), _val("right", "rms_amplitude")
    records.append(
        _record(
            "lr_freq_delta_hz",
            (rf - lf) if _finite(lf) and _finite(rf) else None,
            units="Hz",
            scope="bilateral",
            pair_id=pair_id,
            left_value=lf,
            right_value=rf,
            reliability=rel,
        )
    )
    records.append(
        _record(
            "lr_entropy_delta",
            (re - le) if _finite(le) and _finite(re) else None,
            units="1",
            scope="bilateral",
            pair_id=pair_id,
            left_value=le,
            right_value=re,
            reliability=rel,
        )
    )
    records.append(
        _record(
            "lr_amplitude_delta",
            (ra - la) if _finite(la) and _finite(ra) else None,
            units=left_series.spec.units if left_series is not None else "",
            scope="bilateral",
            pair_id=pair_id,
            left_value=la,
            right_value=ra,
            reliability=rel,
        )
    )

    xcorr = None
    if report.matrices is not None:
        for item in report.matrices.cross_corr:
            if {item.channel_a, item.channel_b} == {left_id, right_id}:
                xcorr = item
                break
    lag = xcorr.peak_lag_s if xcorr is not None else None
    records.append(
        _record(
            "lr_xcorr_lag_s",
            lag,
            units="s",
            scope="bilateral",
            pair_id=pair_id,
            reliability=rel,
        )
    )
    stride_t = ac.stride_period_s if ac is not None else None
    phase = None
    if _finite(lag) and _finite(stride_t) and stride_t and abs(stride_t) > 1e-9:
        phase = wrap_phase_deg(360.0 * float(lag) / float(stride_t))
    records.append(
        _record(
            "lr_relative_phase_deg",
            phase,
            units="deg",
            scope="bilateral",
            pair_id=pair_id,
            reliability=rel,
            notes=["wrap(360° × τ / T_stride); not a clinical asymmetry index."],
        )
    )

    # Per-cycle features (amplitude + coarse spectrum if the slice is long enough).
    if cycle_list and series is not None:
        times = series.times_s
        y = series.processed
        for cycle in cycle_list:
            t0, t1 = float(cycle.start_time_s), float(cycle.end_time_s)
            mask = (times >= t0) & (times < t1) & np.isfinite(y)
            sl = y[mask]
            idx = int(getattr(cycle, "cycle_index", 0))
            if sl.size < int(cfg.min_cycle_samples):
                records.append(
                    _record(
                        "rms_amplitude",
                        None,
                        units=series.spec.units,
                        scope="cycle",
                        channel_id=primary,
                        cycle_index=idx,
                        notes=["Cycle shorter than min_cycle_samples."],
                    )
                )
                continue
            records.append(
                _record(
                    "rms_amplitude",
                    rms_amplitude(sl),
                    units=series.spec.units,
                    scope="cycle",
                    channel_id=primary,
                    cycle_index=idx,
                    reliability=rel,
                )
            )
            freq_c, pxx_c = rfft_periodogram(sl, series.fps, window=cfg.fft_window)
            if freq_c.size:
                f0, p0 = _dominant_in_band(freq_c, pxx_c, cfg.gait_band_hz)
                records.append(
                    _record(
                        "dominant_frequency_hz",
                        f0,
                        units="Hz",
                        scope="cycle",
                        channel_id=primary,
                        cycle_index=idx,
                        reliability=rel,
                    )
                )
                records.append(
                    _record(
                        "dominant_power",
                        p0,
                        units="units^2/Hz",
                        scope="cycle",
                        channel_id=primary,
                        cycle_index=idx,
                        reliability=rel,
                    )
                )
                records.append(
                    _record(
                        "spectral_centroid_hz",
                        _centroid(freq_c, pxx_c, cfg.gait_band_hz),
                        units="Hz",
                        scope="cycle",
                        channel_id=primary,
                        cycle_index=idx,
                        reliability=rel,
                    )
                )
                records.append(
                    _record(
                        "spectral_entropy",
                        spectral_entropy(freq_c, pxx_c, cfg.gait_band_hz),
                        units="1",
                        scope="cycle",
                        channel_id=primary,
                        cycle_index=idx,
                        reliability=rel,
                    )
                )

    session_vector: dict[str, float | None] = {k: None for k in SESSION_VECTOR_KEYS}
    for rec in records:
        if rec.scope != "session" and rec.scope != "bilateral":
            continue
        if rec.name in session_vector:
            session_vector[rec.name] = rec.value if rec.available else None

    if video_q is not None and video_q < 50:
        warnings.append(
            f"Pose/video heuristic {video_q:.0f}/100 is a separate layer "
            "from spectral reliability (duration, gaps, cycles)."
        )
    warnings.append(
        "Features are derived from pose time series and are not disease labels."
    )

    return SpectralFeatureSet(
        records=records,
        session_vector=session_vector,
        n_detected_cycles=n_detected,
        n_valid_cycles=n_valid,
        video_quality_score=video_q,
        fps=fps,
        duration_s=duration,
        primary_channel_id=primary,
        warnings=_unique(warnings),
    )


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
