"""Normalized autocorrelation, lag peaks, and stride/step period estimates.

Peak searches are confined to walking-plausible lag windows. Results are
signal-processing estimates with a reliability score — not diagnoses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from stablewalk.analysis.quality_layers import spectral_reliability_evidence
from stablewalk.analysis.spectral.config import SpectralAnalysisConfig
from stablewalk.analysis.spectral.series import ChannelSeries


ACF_CHANNEL_PREPROCESSING = (
    "z-scored processed channel (linear detrend + unit-variance); "
    "longest finite run; mean-remove; biased Pearson ρ(k)"
)
ACF_PEAK_SELECTION_STRIDE = (
    "positive local maximum of ρ(τ) in the stride-lag window; "
    "exclusive-window assignment prefers a peak above the step-window ceiling"
)
ACF_PEAK_SELECTION_STEP = (
    "positive local maximum of ρ(τ) in the step-lag window; "
    "independent of stride (not rewritten to stride/2)"
)


@dataclass
class AutocorrPeak:
    lag_samples: int
    lag_s: float
    value: float
    prominence: float
    is_local_maximum: bool = True


@dataclass
class AutocorrResult:
    channel_id: str
    lags_s: np.ndarray
    rho: np.ndarray
    n_samples: int
    peaks: list[AutocorrPeak] = field(default_factory=list)
    stride_period_s: float | None = None
    step_period_s: float | None = None
    gait_period_s: float | None = None
    cadence_steps_per_min: float | None = None
    n_stride_cycles: float | None = None
    cadence_assumes_two_steps_per_stride: bool = False
    stride_assignment_note: str = ""
    reliability: float = 0.0
    reliability_label: str = "insufficient"
    reliability_explanation: str = ""
    quality_status: str = "insufficient"
    insufficient_reason: str | None = None
    notes: list[str] = field(default_factory=list)
    source_signal: str = ""
    time_window_s: tuple[float, float] | None = None
    preprocessing: str = ACF_CHANNEL_PREPROCESSING
    peak_selection: str = ACF_PEAK_SELECTION_STRIDE

    def to_dict(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "lags_s": [float(x) for x in self.lags_s],
            "rho": [float(x) for x in self.rho],
            "n_samples": self.n_samples,
            "peaks": [
                {
                    "lag_samples": p.lag_samples,
                    "lag_s": p.lag_s,
                    "value": p.value,
                    "prominence": p.prominence,
                    "is_local_maximum": p.is_local_maximum,
                }
                for p in self.peaks
            ],
            "stride_period_s": self.stride_period_s,
            "step_period_s": self.step_period_s,
            "gait_period_s": self.gait_period_s,
            "cadence_steps_per_min": self.cadence_steps_per_min,
            "n_stride_cycles": self.n_stride_cycles,
            "cadence_assumes_two_steps_per_stride": self.cadence_assumes_two_steps_per_stride,
            "stride_assignment_note": self.stride_assignment_note,
            "reliability": self.reliability,
            "reliability_label": self.reliability_label,
            "reliability_explanation": self.reliability_explanation,
            "quality_status": self.quality_status,
            "insufficient_reason": self.insufficient_reason,
            "notes": list(self.notes),
            "source_signal": self.source_signal,
            "time_window_s": list(self.time_window_s) if self.time_window_s else None,
            "preprocessing": self.preprocessing,
            "peak_selection": self.peak_selection,
            "units": {
                "lag": "s",
                "rho": "dimensionless",
                "gait_period": "s",
                "cadence": "steps/min",
            },
        }


def _attach_spectral_quality(result: AutocorrResult, series: ChannelSeries, *, peak_value: float | None) -> None:
    evidence = spectral_reliability_evidence(
        duration_s=float(series.duration_s),
        gap_fraction=float(series.missing_fraction),
        n_cycles=result.n_stride_cycles,
        peak_value=peak_value,
        insufficient_reason=result.insufficient_reason,
    )
    result.quality_status = evidence.status
    result.reliability_explanation = evidence.explanation
    if evidence.status == "insufficient" and not result.reliability_label:
        result.reliability_label = "insufficient"


def normalized_autocorrelation(
    values: np.ndarray,
    *,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Biased Pearson autocorrelation ρ(k) for k = 0 … max_lag.

    ρ(k) = Σ x[t] x[t+k] / Σ x[t]²  after mean removal on finite samples.
    Matches ``np.correlate`` on the centered finite series, then lag-0 normalizes.
    """
    x = np.asarray(values, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 4:
        lags = np.arange(max(max_lag, 0) + 1, dtype=int)
        return lags, np.full(lags.shape, np.nan, dtype=float)
    # Use the longest contiguous finite run so lag products stay aligned.
    run = _longest_finite_run(mask)
    x = x[run]
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    max_lag = int(max(0, min(max_lag, x.size - 1)))
    lags = np.arange(max_lag + 1, dtype=int)
    if denom <= 1e-18:
        return lags, np.full(lags.shape, np.nan, dtype=float)
    full = np.correlate(x, x, mode="full")
    mid = x.size - 1
    rho = full[mid : mid + max_lag + 1] / denom
    return lags, rho.astype(float)


def _longest_finite_run(mask: np.ndarray) -> slice:
    best_i, best_j, i = 0, 0, 0
    n = mask.size
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if (j - i) > (best_j - best_i):
            best_i, best_j = i, j
        i = j
    return slice(best_i, best_j)


def _local_maxima(values: np.ndarray) -> list[int]:
    out: list[int] = []
    if values.size < 3:
        return out
    for i in range(1, values.size - 1):
        left, mid, right = float(values[i - 1]), float(values[i]), float(values[i + 1])
        if not (math.isfinite(left) and math.isfinite(mid) and math.isfinite(right)):
            continue
        if mid >= left and mid > right:
            out.append(i)
    return out


def is_qualifying_local_maximum(
    rho: np.ndarray,
    idx: int,
    *,
    min_value: float = 0.05,
) -> bool:
    """True when ``idx`` is a positive local maximum of ρ, not a sampled point."""
    if idx <= 0 or idx >= int(np.asarray(rho).size) - 1:
        return False
    values = np.asarray(rho, dtype=float)
    mid = float(values[idx])
    left = float(values[idx - 1])
    right = float(values[idx + 1])
    if not (math.isfinite(mid) and math.isfinite(left) and math.isfinite(right)):
        return False
    if mid <= float(min_value):
        return False
    return mid >= left and mid > right


def qualifying_acf_peaks(
    lags_s: np.ndarray,
    rho: np.ndarray,
    lo: float,
    hi: float,
    *,
    min_value: float = 0.05,
) -> list[AutocorrPeak]:
    """Positive local maxima of ρ(τ) inside ``[lo, hi]`` — never subsampled points."""
    out: list[AutocorrPeak] = []
    rho_arr = np.asarray(rho, dtype=float)
    lag_arr = np.asarray(lags_s, dtype=float)
    for i in _local_maxima(rho_arr):
        if not is_qualifying_local_maximum(rho_arr, i, min_value=min_value):
            continue
        lag = float(lag_arr[i])
        value = float(rho_arr[i])
        if lo <= lag <= hi:
            out.append(
                AutocorrPeak(
                    lag_samples=int(i),
                    lag_s=lag,
                    value=value,
                    prominence=_prominence(rho_arr, i),
                    is_local_maximum=True,
                )
            )
    out.sort(key=lambda p: (p.value, p.prominence), reverse=True)
    return out


def _prominence(rho: np.ndarray, idx: int) -> float:
    left_min = float(np.min(rho[: idx + 1])) if idx > 0 else float(rho[idx])
    return float(rho[idx] - left_min)


def _best_peak_in_band(
    lags_s: np.ndarray,
    rho: np.ndarray,
    band: tuple[float, float],
    *,
    min_value: float = 0.05,
) -> AutocorrPeak | None:
    """Largest *positive* local maximum of ρ(k) inside ``band``.

    A trough in the stride window is not a stride period. Negative ρ is
    returned as no peak so feature export stays ``None`` rather than a
    fabricated duration.
    """
    peaks = qualifying_acf_peaks(lags_s, rho, band[0], band[1], min_value=min_value)
    return peaks[0] if peaks else None


def _peaks_in_window(
    lags_s: np.ndarray,
    rho: np.ndarray,
    lo: float,
    hi: float,
    *,
    min_value: float = 0.05,
) -> list[AutocorrPeak]:
    """All positive local maxima of ρ(k) inside ``[lo, hi]``."""
    return qualifying_acf_peaks(lags_s, rho, lo, hi, min_value=min_value)


def insufficient_cycles_reason(
    *,
    duration_s: float,
    period_s: float | None,
    n_detected_cycles: int = 0,
    min_cycles: int = 2,
) -> str | None:
    """Explicit gate so short clips do not emit a gait period or cadence."""
    if n_detected_cycles > 0 and n_detected_cycles < min_cycles:
        return (
            f"insufficient cycles: {n_detected_cycles} detected gait cycle(s); "
            f"need ≥ {min_cycles}."
        )
    if period_s is not None and period_s > 1e-6:
        n_est = float(duration_s) / float(period_s)
        if n_est + 1e-9 < float(min_cycles):
            return (
                f"insufficient cycles: clip spans ~{n_est:.1f} estimated cycle(s) "
                f"over {duration_s:.2f} s; need ≥ {min_cycles}."
            )
        return None
    min_duration = float(min_cycles) * 0.60
    if float(duration_s) + 1e-9 < min_duration:
        return (
            f"insufficient cycles: duration {duration_s:.2f} s is too short "
            f"for {min_cycles} gait cycles."
        )
    return None


def assign_step_and_stride_peaks(
    lags_s: np.ndarray,
    rho: np.ndarray,
    *,
    stride_lag_s: tuple[float, float],
    step_lag_s: tuple[float, float],
    min_value: float = 0.05,
) -> tuple[AutocorrPeak | None, AutocorrPeak | None, str]:
    """Pick independent step and stride peaks without a 2:1 rewrite.

    Default windows overlap at 0.55–0.90 s. Max-ρ in that overlap can be a
    step-rate lobe. If any exclusive-stride peak exists (lag strictly above
    the step-window ceiling), use that as stride. Fast gait whose true stride
    sits in the overlap still uses the overlap peak when no exclusive peak
    exists. This does not assume step = stride/2 and does not raise ρ
    thresholds.
    """
    _step_lo, step_hi = step_lag_s
    stride_lo, stride_hi = stride_lag_s
    step_peak = _best_peak_in_band(lags_s, rho, step_lag_s, min_value=min_value)
    exclusive_lo = max(stride_lo, step_hi) + 1e-9
    overlap_hi = min(stride_hi, step_hi)
    exclusive_stride = None
    if exclusive_lo < stride_hi:
        exclusive_stride = _best_peak_in_band(
            lags_s, rho, (exclusive_lo, stride_hi), min_value=min_value
        )
    overlap_stride = None
    if stride_lo <= overlap_hi:
        overlap_stride = _best_peak_in_band(
            lags_s, rho, (stride_lo, overlap_hi), min_value=min_value
        )
    if exclusive_stride is not None:
        note = (
            f"Stride assigned to exclusive-window peak at {exclusive_stride.lag_s:.3f} s "
            f"(ρ={exclusive_stride.value:.3f})."
        )
        if overlap_stride is not None:
            note += (
                f" Overlap-band peak at {overlap_stride.lag_s:.3f} s "
                f"(ρ={overlap_stride.value:.3f}) is not used as stride."
            )
        return exclusive_stride, step_peak, note
    if overlap_stride is not None:
        return (
            overlap_stride,
            step_peak,
            "No exclusive-stride ACF peak above the step-window ceiling; "
            "stride uses the overlap-band peak (possible for fast gait).",
        )
    return None, step_peak, "No positive ACF peak in the stride-lag window."


def _cadence_steps_per_min(
    *,
    step_period_s: float | None,
    stride_period_s: float | None,
) -> tuple[float | None, bool]:
    """Spectral cadence (steps/min) and whether 120/T_stride was used.

    ``60 / T_step`` is the step-rate conversion. ``120 / T_stride`` assumes
    two steps per stride and is only the fallback when no independent step
    peak exists. Does not write Overview cadence.
    """
    if step_period_s is not None and step_period_s > 1e-6:
        return float(60.0 / step_period_s), False
    if stride_period_s is not None and stride_period_s > 1e-6:
        return float(120.0 / stride_period_s), True
    return None, False


def analyze_autocorrelation(
    series: ChannelSeries,
    *,
    config: SpectralAnalysisConfig | None = None,
) -> AutocorrResult:
    cfg = config or SpectralAnalysisConfig()
    notes: list[str] = []
    norm = getattr(series, "normalized", None)
    if norm is not None and np.size(norm) and int(np.isfinite(norm).sum()) >= 4:
        y = norm
    else:
        y = series.processed
    n = int(np.isfinite(y).sum())
    max_lag = int(round(cfg.max_acf_lag_s * series.fps))
    lags_i, rho = normalized_autocorrelation(y, max_lag=max_lag)
    lags_s = lags_i.astype(float) / max(series.fps, 1e-6)
    time_window = None
    if getattr(series, "times_s", None) is not None and np.size(series.times_s) >= 2:
        time_window = (float(series.times_s[0]), float(series.times_s[-1]))
    result = AutocorrResult(
        channel_id=series.spec.channel_id,
        lags_s=lags_s,
        rho=rho,
        n_samples=n,
        notes=notes,
        source_signal=str(series.spec.channel_id),
        time_window_s=time_window,
        preprocessing=ACF_CHANNEL_PREPROCESSING,
        peak_selection=ACF_PEAK_SELECTION_STRIDE,
    )
    if n < cfg.min_samples or series.duration_s < cfg.min_duration_s:
        reason = insufficient_cycles_reason(
            duration_s=series.duration_s,
            period_s=None,
            min_cycles=cfg.min_cycles_for_stride,
        )
        if reason:
            notes.append(reason)
            result.insufficient_reason = reason
        else:
            notes.append("Insufficient samples or duration for a stride period.")
        _attach_spectral_quality(result, series, peak_value=None)
        return result
    if series.missing_fraction > cfg.max_missing_fraction:
        notes.append("Missing-data fraction is high; spectral estimates are limited.")

    stride_peak, step_peak, assign_note = assign_step_and_stride_peaks(
        lags_s,
        rho,
        stride_lag_s=cfg.stride_lag_s,
        step_lag_s=cfg.step_lag_s,
    )
    result.stride_assignment_note = assign_note
    if assign_note:
        notes.append(assign_note)
    lo_peaks = min(cfg.step_lag_s[0], cfg.stride_lag_s[0])
    hi_peaks = max(cfg.step_lag_s[1], cfg.stride_lag_s[1])
    result.peaks = [
        peak
        for peak in _peaks_in_window(lags_s, rho, lo_peaks, hi_peaks)
        if peak.is_local_maximum
        and is_qualifying_local_maximum(rho, int(peak.lag_samples))
    ]
    if step_peak is not None and stride_peak is None:
        result.peak_selection = ACF_PEAK_SELECTION_STEP
    elif step_peak is not None:
        result.peak_selection = (
            f"{ACF_PEAK_SELECTION_STRIDE}; {ACF_PEAK_SELECTION_STEP}"
        )
    candidate_stride = stride_peak.lag_s if stride_peak else None
    if candidate_stride:
        result.n_stride_cycles = series.duration_s / candidate_stride
    reason = insufficient_cycles_reason(
        duration_s=series.duration_s,
        period_s=candidate_stride,
        min_cycles=cfg.min_cycles_for_stride,
    )
    if reason:
        notes.append(reason)
        result.insufficient_reason = reason
        result.reliability_label = "insufficient"
        result.reliability = 0.0
        _attach_spectral_quality(result, series, peak_value=None)
        return result
    if stride_peak:
        result.stride_period_s = stride_peak.lag_s
        result.gait_period_s = stride_peak.lag_s
    if step_peak:
        result.step_period_s = step_peak.lag_s
    cadence, assumes_two = _cadence_steps_per_min(
        step_period_s=result.step_period_s,
        stride_period_s=result.stride_period_s,
    )
    result.cadence_steps_per_min = cadence
    result.cadence_assumes_two_steps_per_stride = assumes_two
    if assumes_two:
        notes.append(
            "Cadence used 120 / T_stride (assumes two steps per stride) because "
            "no independent ACF step peak was found."
        )
    elif cadence is not None:
        notes.append("Cadence = 60 / T_step from the ACF step-lag window.")

    if result.stride_period_s and result.step_period_s:
        ratio = result.stride_period_s / result.step_period_s
        if 1.7 <= ratio <= 2.3:
            notes.append(
                "Step lag is approximately half the stride lag — a consistency "
                "check, not a rewrite of either period."
            )
        else:
            notes.append(
                "Step and stride lags are not a 1:2 ratio; both are kept. "
                "Cadence uses 60/T_step; stride is not set to 2×step."
            )

    rel = 0.0
    if stride_peak is not None:
        rel += 0.35 * min(max(stride_peak.value, 0.0), 1.0)
        rel += 0.25 * min(max(stride_peak.prominence, 0.0), 1.0)
    if result.n_stride_cycles is not None:
        rel += 0.25 * min(result.n_stride_cycles / 6.0, 1.0)
    rel += 0.15 * max(0.0, 1.0 - series.missing_fraction)
    result.reliability = float(max(0.0, min(1.0, rel)))
    if result.reliability >= 0.65 and result.stride_period_s is not None:
        result.reliability_label = "moderate"
    elif result.stride_period_s is not None:
        result.reliability_label = "low"
    else:
        result.reliability_label = "insufficient"
        notes.append("No clear autocorrelation peak in the stride lag window.")
    notes.append(
        "Spectral reliability uses duration, gaps, contributing cycles, and ACF peak "
        "quality. It is not pose coverage, landmark visibility, or a clinical finding."
    )
    peak_val = stride_peak.value if stride_peak is not None else None
    _attach_spectral_quality(result, series, peak_value=peak_val)
    return result
