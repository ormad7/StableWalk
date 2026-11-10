"""Autocorrelation gait analysis on the existing smoothed pose series.

This module does **not** replace Spectral Analysis and does **not** write
cadence, gait-quality, or other dashboard metrics. It reads the already
smoothed / detrended channels from ``build_spectral_series`` and reports
period, regularity, and left–right lag when the data support it.

Signals
-------
left/right ankle vertical position, left/right knee angle, left/right hip
angle, and pelvis lateral translation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from stablewalk.analysis.spectral.autocorrelation import (
    AutocorrPeak,
    normalized_autocorrelation,
    _best_peak_in_band,
    _local_maxima,
    _prominence,
)
from stablewalk.analysis.spectral.config import SpectralAnalysisConfig
from stablewalk.analysis.spectral.matrices import lagged_cross_correlation
from stablewalk.analysis.spectral.series import (
    ChannelSeries,
    build_spectral_series,
    detrend_signal,
    interpolate_short_gaps,
    smooth_preserve_gait,
)
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.joint_registry import ROOT_JOINT_ID
from stablewalk.models.pose_data import PoseSequence

# Mapping from gait-ACF names → existing spectral channel ids.
GAIT_ACF_CHANNEL_IDS: dict[str, str] = {
    "left_ankle_vertical": "pos:left_ankle:y",
    "right_ankle_vertical": "pos:right_ankle:y",
    "left_knee_angle": "dof:left_knee_flexion",
    "right_knee_angle": "dof:right_knee_flexion",
    "left_hip_angle": "dof:left_hip_flexion",
    "right_hip_angle": "dof:right_hip_flexion",
    "pelvis_lateral": "pelvis:root:x",
}

GAIT_ACF_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("ankle_vertical", "left_ankle_vertical", "right_ankle_vertical"),
    ("knee_angle", "left_knee_angle", "right_knee_angle"),
    ("hip_angle", "left_hip_angle", "right_hip_angle"),
)

_STRIDE_LAG_S: tuple[float, float] = (0.60, 1.80)
_STEP_LAG_S: tuple[float, float] = (0.30, 0.90)
_PEAK_LAG_S: tuple[float, float] = (0.25, 2.50)
_MIN_PEAK_RHO = 0.12
_MIN_QUALITY_SCORE = 40.0
_MIN_DURATION_S = 2.0
_MIN_SAMPLES = 64


@dataclass
class SignalAcfResult:
    """Normalized ACF of one gait signal."""

    signal_id: str
    n_samples: int
    lags_s: list[float]
    rho: list[float | None]
    peaks: list[dict[str, float | int]] = field(default_factory=list)
    dominant_period_s: float | None = None
    gait_cycle_duration_s: float | None = None
    regularity: float | None = None
    missing_fraction: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "n_samples": self.n_samples,
            "lags_s": list(self.lags_s),
            "rho": list(self.rho),
            "peaks": list(self.peaks),
            "dominant_period_s": self.dominant_period_s,
            "gait_cycle_duration_s": self.gait_cycle_duration_s,
            "regularity": self.regularity,
            "missing_fraction": self.missing_fraction,
            "notes": list(self.notes),
        }


@dataclass
class LegOffsetResult:
    """Left–right cross-correlation offset for one bilateral pair."""

    pair_id: str
    peak_rho: float | None = None
    time_offset_s: float | None = None
    phase_offset_deg: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "peak_rho": self.peak_rho,
            "time_offset_s": self.time_offset_s,
            "phase_offset_deg": self.phase_offset_deg,
            "notes": list(self.notes),
            "units": {
                "time_offset": "s",
                "phase_offset": "deg",
                "peak_rho": "dimensionless",
            },
        }


@dataclass
class GaitAutocorrReport:
    """Session-level autocorrelation gait analysis (derived, not clinical)."""

    available: bool
    confidence: float
    confidence_label: str
    quality_score: float | None
    gait_cycle_duration_s: float | None
    dominant_period_s: float | None
    regularity_score: float | None
    signals: dict[str, SignalAcfResult] = field(default_factory=dict)
    leg_offsets: dict[str, LegOffsetResult] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    fps: float = 0.0
    duration_s: float = 0.0
    source: str = "smoothed_spectral_series"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "stablewalk-gait-acf",
            "schema_version": "1.0",
            "kind": "derived",
            "available": self.available,
            "confidence": self.confidence,
            "confidence_label": self.confidence_label,
            "quality_score": self.quality_score,
            "gait_cycle_duration_s": self.gait_cycle_duration_s,
            "dominant_period_s": self.dominant_period_s,
            "regularity_score": self.regularity_score,
            "signals": {k: v.to_dict() for k, v in self.signals.items()},
            "leg_offsets": {k: v.to_dict() for k, v in self.leg_offsets.items()},
            "notes": list(self.notes),
            "fps": self.fps,
            "duration_s": self.duration_s,
            "source": self.source,
            "disclaimer": (
                "Autocorrelation period and left–right lag are signal-processing "
                "estimates. They are not clinical gait diagnoses and do not "
                "replace cadence or gait-quality metrics."
            ),
        }


def normalize_signal(values: np.ndarray) -> np.ndarray:
    """Z-score finite samples (mean 0, std 1). Constant series become 0."""
    out = np.asarray(values, dtype=float).copy()
    mask = np.isfinite(out)
    if int(mask.sum()) < 4:
        return out
    mu = float(np.mean(out[mask]))
    sd = float(np.std(out[mask], ddof=1))
    if sd <= 1e-12:
        out[mask] = 0.0
        return out
    out[mask] = (out[mask] - mu) / sd
    return out


def detect_peaks_after_lag_zero(
    lags_s: np.ndarray,
    rho: np.ndarray,
    *,
    min_lag_s: float = _PEAK_LAG_S[0],
    max_lag_s: float = _PEAK_LAG_S[1],
    min_value: float = _MIN_PEAK_RHO,
) -> list[AutocorrPeak]:
    """Local maxima of ρ(k) for k > 0 inside the gait lag window."""
    peaks: list[AutocorrPeak] = []
    if lags_s.size == 0 or rho.size == 0:
        return peaks
    for i in _local_maxima(rho):
        if i <= 0:
            continue
        lag = float(lags_s[i])
        value = float(rho[i])
        if not math.isfinite(lag) or not math.isfinite(value):
            continue
        if lag < min_lag_s or lag > max_lag_s:
            continue
        if value <= min_value:
            continue
        peaks.append(
            AutocorrPeak(
                lag_samples=int(i),
                lag_s=lag,
                value=value,
                prominence=_prominence(rho, i),
            )
        )
    peaks.sort(key=lambda p: (p.value, p.prominence), reverse=True)
    return peaks


def _peak_dict(peak: AutocorrPeak) -> dict[str, float | int]:
    return {
        "lag_samples": peak.lag_samples,
        "lag_s": peak.lag_s,
        "value": peak.value,
        "prominence": peak.prominence,
    }


def _rho_list(rho: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for v in rho:
        fv = float(v)
        out.append(None if not math.isfinite(fv) else fv)
    return out


def _cycle_from_peaks(
    lags_s: np.ndarray,
    rho: np.ndarray,
    peaks: list[AutocorrPeak],
) -> tuple[float | None, float | None, float | None]:
    """Return (dominant_period, gait_cycle_duration, regularity)."""
    stride = _best_peak_in_band(lags_s, rho, _STRIDE_LAG_S, min_value=_MIN_PEAK_RHO)
    step = _best_peak_in_band(lags_s, rho, _STEP_LAG_S, min_value=_MIN_PEAK_RHO)
    dominant = peaks[0] if peaks else None
    dominant_period = None if dominant is None else float(dominant.lag_s)
    cycle = None
    if stride is not None:
        cycle = float(stride.lag_s)
    elif step is not None:
        doubled = float(step.lag_s) * 2.0
        if _STRIDE_LAG_S[0] <= doubled <= _STRIDE_LAG_S[1]:
            cycle = doubled
        else:
            cycle = float(step.lag_s)
    regularity = None
    source = stride or dominant
    if source is not None:
        regularity = float(max(0.0, min(1.0, source.value)))
    return dominant_period, cycle, regularity


def analyze_signal_acf(
    values: np.ndarray,
    fps: float,
    *,
    signal_id: str,
    max_lag_s: float = 2.50,
    missing_fraction: float = 0.0,
) -> SignalAcfResult:
    """Normalize one series and compute ACF peaks after lag 0."""
    notes: list[str] = []
    y = normalize_signal(values)
    n = int(np.isfinite(y).sum())
    fps = max(float(fps), 1e-6)
    max_lag = int(round(max_lag_s * fps))
    lags_i, rho = normalized_autocorrelation(y, max_lag=max_lag)
    lags_s = lags_i.astype(float) / fps
    result = SignalAcfResult(
        signal_id=signal_id,
        n_samples=n,
        lags_s=[float(x) for x in lags_s],
        rho=_rho_list(rho),
        missing_fraction=float(missing_fraction),
        notes=notes,
    )
    if n < 8 or rho.size < 3 or not np.isfinite(rho[0]):
        notes.append("Too few finite samples for autocorrelation.")
        return result
    peaks = detect_peaks_after_lag_zero(lags_s, rho)
    result.peaks = [_peak_dict(p) for p in peaks]
    dominant, cycle, regularity = _cycle_from_peaks(lags_s, rho, peaks)
    result.dominant_period_s = dominant
    result.gait_cycle_duration_s = cycle
    result.regularity = regularity
    if not peaks:
        notes.append("No meaningful autocorrelation peak after lag zero.")
    return result


def _wrap_phase_deg(time_offset_s: float, period_s: float) -> float:
    turns = float(time_offset_s) / float(period_s)
    return float((turns * 360.0 + 180.0) % 360.0 - 180.0)


def analyze_leg_offset(
    left: np.ndarray,
    right: np.ndarray,
    fps: float,
    *,
    pair_id: str,
    period_s: float | None,
    max_lag_s: float = 2.0,
) -> LegOffsetResult:
    notes: list[str] = []
    left_n = normalize_signal(left)
    right_n = normalize_signal(right)
    max_lag = int(round(max_lag_s * max(float(fps), 1e-6)))
    xc = lagged_cross_correlation(left_n, right_n, max_lag=max_lag, fps=fps)
    result = LegOffsetResult(pair_id=pair_id, notes=notes)
    if xc.peak_lag_s is None or xc.peak_value is None:
        notes.append("No left–right cross-correlation peak.")
        return result
    if not math.isfinite(xc.peak_value) or abs(float(xc.peak_value)) < 0.15:
        notes.append("Left–right correlation peak is too weak to trust an offset.")
        return result
    result.peak_rho = float(xc.peak_value)
    result.time_offset_s = float(xc.peak_lag_s)
    if period_s is not None and period_s > 1e-6:
        result.phase_offset_deg = _wrap_phase_deg(result.time_offset_s, period_s)
    else:
        notes.append("Phase offset withheld — no gait-cycle period.")
    return result


def _confidence_label(confidence: float, available: bool) -> str:
    if not available or confidence < 0.25:
        return "insufficient"
    if confidence < 0.50:
        return "low"
    if confidence < 0.75:
        return "moderate"
    return "high"


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr))


def _consensus_period(lags: list[float], *, rel_tol: float = 0.18) -> float | None:
    """Median lag only when at least two channels agree (or one very strong)."""
    if not lags:
        return None
    if len(lags) == 1:
        return None
    med = float(np.median(np.asarray(lags, dtype=float)))
    if med <= 1e-9:
        return None
    close = [float(x) for x in lags if abs(float(x) - med) / med <= rel_tol]
    if len(close) < 2:
        return None
    return float(np.median(np.asarray(close, dtype=float)))


def _withheld_report(
    *,
    fps: float,
    duration_s: float,
    quality_score: float | None,
    notes: list[str],
    signals: dict[str, SignalAcfResult] | None = None,
) -> GaitAutocorrReport:
    return GaitAutocorrReport(
        available=False,
        confidence=0.0,
        confidence_label="insufficient",
        quality_score=quality_score,
        gait_cycle_duration_s=None,
        dominant_period_s=None,
        regularity_score=None,
        signals=signals or {},
        leg_offsets={},
        notes=notes,
        fps=fps,
        duration_s=duration_s,
    )


def analyze_gait_acf_from_arrays(
    signals: Mapping[str, np.ndarray],
    fps: float,
    *,
    quality_score: float | None = 100.0,
    missing: Mapping[str, float] | None = None,
) -> GaitAutocorrReport:
    """Run gait ACF on already-smoothed named arrays (tests and callers)."""
    notes: list[str] = [
        "Signals are z-scored before autocorrelation. Results are derived, not clinical.",
    ]
    fps = max(float(fps), 1e-6)
    miss_map = dict(missing or {})
    arrays = {str(k): np.asarray(v, dtype=float) for k, v in signals.items()}
    n = max((a.size for a in arrays.values()), default=0)
    duration = float(n - 1) / fps if n >= 2 else 0.0
    q = None if quality_score is None else float(quality_score)

    if q is not None and q < _MIN_QUALITY_SCORE:
        notes.append(
            "Video quality is insufficient; autocorrelation gait estimates are withheld."
        )
        return _withheld_report(
            fps=fps, duration_s=duration, quality_score=q, notes=notes
        )
    if duration < _MIN_DURATION_S or n < _MIN_SAMPLES:
        notes.append(
            f"Clip too short for gait autocorrelation ({duration:.2f} s, {n} samples)."
        )
        return _withheld_report(
            fps=fps, duration_s=duration, quality_score=q, notes=notes
        )

    per_signal: dict[str, SignalAcfResult] = {}
    for name, values in arrays.items():
        per_signal[name] = analyze_signal_acf(
            values,
            fps,
            signal_id=name,
            missing_fraction=float(miss_map.get(name, 0.0)),
        )

    stride_lags = [
        s.gait_cycle_duration_s
        for s in per_signal.values()
        if s.gait_cycle_duration_s is not None
        and _STRIDE_LAG_S[0] <= s.gait_cycle_duration_s <= _STRIDE_LAG_S[1]
    ]
    all_dom = [
        s.dominant_period_s
        for s in per_signal.values()
        if s.dominant_period_s is not None
    ]
    cycle = _consensus_period(stride_lags)
    if cycle is None and len(stride_lags) == 1:
        only = stride_lags[0]
        strong = [
            s.regularity
            for s in per_signal.values()
            if s.gait_cycle_duration_s == only and (s.regularity or 0.0) >= 0.45
        ]
        if strong:
            cycle = only
    dominant = _consensus_period(all_dom) or cycle
    if cycle is None and dominant is not None:
        doubled = dominant * 2.0
        cycle = doubled if _STRIDE_LAG_S[0] <= doubled <= _STRIDE_LAG_S[1] else None

    regs = [s.regularity for s in per_signal.values() if s.regularity is not None]
    regularity = None if not regs else float(max(0.0, min(1.0, float(np.mean(regs)))))

    offsets: dict[str, LegOffsetResult] = {}
    for pair_id, left_id, right_id in GAIT_ACF_PAIRS:
        if left_id not in arrays or right_id not in arrays:
            continue
        offsets[pair_id] = analyze_leg_offset(
            arrays[left_id],
            arrays[right_id],
            fps,
            pair_id=pair_id,
            period_s=cycle,
        )

    n_peaks = sum(1 for s in per_signal.values() if s.gait_cycle_duration_s is not None)
    if cycle is None or regularity is None or n_peaks < 1:
        notes.append("No consistent gait-cycle peak across the analyzed signals.")
        return GaitAutocorrReport(
            available=False,
            confidence=float(regularity or 0.0) * 0.3,
            confidence_label="insufficient",
            quality_score=q,
            gait_cycle_duration_s=None,
            dominant_period_s=None,
            regularity_score=regularity,
            signals=per_signal,
            leg_offsets=offsets,
            notes=notes,
            fps=fps,
            duration_s=duration,
        )

    q_term = 1.0 if q is None else max(0.0, min(1.0, q / 100.0))
    n_cycles = duration / cycle if cycle > 1e-6 else 0.0
    confidence = (
        0.30 * q_term
        + 0.45 * regularity
        + 0.25 * min(n_cycles / 4.0, 1.0)
    )
    confidence = float(max(0.0, min(1.0, confidence)))
    available = confidence >= 0.25 and regularity >= 0.15
    if not available:
        notes.append("Confidence is too low to report a gait-cycle duration.")
        cycle_out = None
        dominant_out = None
        regularity_out = regularity
    else:
        cycle_out = cycle
        dominant_out = dominant
        regularity_out = regularity
    return GaitAutocorrReport(
        available=available,
        confidence=confidence if available else 0.0,
        confidence_label=_confidence_label(confidence, available),
        quality_score=q,
        gait_cycle_duration_s=cycle_out,
        dominant_period_s=dominant_out,
        regularity_score=regularity_out,
        signals=per_signal,
        leg_offsets=offsets if available else {},
        notes=notes,
        fps=fps,
        duration_s=duration,
    )


def _pelvis_lateral_from_recording(
    recording: GaitMotionRecording,
    times: np.ndarray,
    fps: float,
    cfg: SpectralAnalysisConfig,
) -> np.ndarray | None:
    """Image-space mid-hip X when root trajectory is unavailable."""
    src_t = np.asarray([float(s.time_s) for s in recording.snapshots], dtype=float)
    xs = np.full(src_t.shape, np.nan, dtype=float)
    for i, snap in enumerate(recording.snapshots):
        xy = (snap.metadata or {}).get("image_xy") or {}
        val = None
        for key in (ROOT_JOINT_ID, "mid_hip"):
            pt = xy.get(key)
            if isinstance(pt, (list, tuple)) and len(pt) >= 1:
                try:
                    val = float(pt[0])
                except (TypeError, ValueError):
                    val = None
                if val is not None:
                    break
        if val is None:
            lh, rh = xy.get("left_hip"), xy.get("right_hip")
            if (
                isinstance(lh, (list, tuple))
                and isinstance(rh, (list, tuple))
                and len(lh) >= 1
                and len(rh) >= 1
            ):
                val = 0.5 * (float(lh[0]) + float(rh[0]))
        if val is None:
            joint = snap.joints.get(ROOT_JOINT_ID)
            if joint is not None:
                val = float(joint.position.x)
        if val is not None and math.isfinite(val):
            xs[i] = val
    if int(np.isfinite(xs).sum()) < 8:
        return None
    if src_t.size != times.size or not np.allclose(src_t, times, atol=1e-6, equal_nan=False):
        from stablewalk.analysis.spectral.series import _resample_to_grid

        xs = _resample_to_grid(src_t, xs, times)
    max_gap = max(1, int(round(cfg.max_gap_s * fps)))
    win = max(3, int(round(cfg.smooth_window_s * fps)))
    if win % 2 == 0:
        win += 1
    filled, _mask = interpolate_short_gaps(xs, max_gap_samples=max_gap)
    processed = smooth_preserve_gait(filled, window_samples=win, polyorder=cfg.smooth_polyorder)
    processed = detrend_signal(processed, cfg.detrend)
    finite = processed[np.isfinite(processed)]
    if finite.size < 8 or float(np.std(finite, ddof=1)) <= 1e-5:
        return None
    return processed


def _quality_from_sequence(sequence: PoseSequence | None) -> tuple[float | None, list[str]]:
    if sequence is None or not getattr(sequence, "frames", None):
        return None, []
    from stablewalk.analysis.biomechanical.video_quality import assess_video_quality

    assessment = assess_video_quality(sequence)
    notes = list(assessment.warnings)
    return float(assessment.overall_quality_score), notes


def _series_array(ch: ChannelSeries | None) -> np.ndarray | None:
    if ch is None:
        return None
    y = np.asarray(ch.processed, dtype=float)
    if int(np.isfinite(y).sum()) < 8:
        return None
    return y


def analyze_gait_autocorrelation(
    recording: GaitMotionRecording,
    *,
    sequence: PoseSequence | None = None,
    config: SpectralAnalysisConfig | None = None,
    quality_score: float | None = None,
) -> GaitAutocorrReport:
    """Autocorrelation gait analysis from a recording's smoothed channels.

    Does not mutate the recording and does not update gait-quality / cadence.
    """
    cfg = config or SpectralAnalysisConfig()
    notes: list[str] = []
    q, q_notes = _quality_from_sequence(sequence)
    notes.extend(q_notes)
    if quality_score is not None:
        q = float(quality_score)

    bundle = build_spectral_series(
        recording,
        sequence=sequence,
        config=cfg,
        channel_ids=tuple(GAIT_ACF_CHANNEL_IDS.values()),
    )
    notes.extend(bundle.warnings)
    fps = float(bundle.fps)
    duration = float(bundle.metadata.get("duration_s") or 0.0)

    arrays: dict[str, np.ndarray] = {}
    missing: dict[str, float] = {}
    for name, cid in GAIT_ACF_CHANNEL_IDS.items():
        series = bundle.get(cid)
        arr = _series_array(series)
        if arr is None:
            continue
        arrays[name] = arr
        missing[name] = float(series.missing_fraction) if series is not None else 1.0

    if "pelvis_lateral" not in arrays:
        fallback = _pelvis_lateral_from_recording(
            recording, bundle.times_s, fps, cfg
        )
        if fallback is not None:
            arrays["pelvis_lateral"] = fallback
            missing["pelvis_lateral"] = float(np.mean(~np.isfinite(fallback)))
        else:
            notes.append(
                "Pelvis lateral series unavailable (need pose root trajectory or image XY)."
            )

    if q is None and arrays:
        vis = []
        for name, cid in GAIT_ACF_CHANNEL_IDS.items():
            series = bundle.get(cid)
            if series is not None and series.mean_visibility > 0:
                vis.append(series.mean_visibility)
        if vis:
            q = float(np.mean(vis) * 100.0)

    report = analyze_gait_acf_from_arrays(
        arrays, fps, quality_score=q, missing=missing
    )
    report.notes = list(dict.fromkeys([*notes, *report.notes]))
    report.duration_s = duration or report.duration_s
    report.fps = fps
    report.source = "smoothed_spectral_series"
    return report


__all__ = [
    "GAIT_ACF_CHANNEL_IDS",
    "GAIT_ACF_PAIRS",
    "GaitAutocorrReport",
    "LegOffsetResult",
    "SignalAcfResult",
    "analyze_gait_acf_from_arrays",
    "analyze_gait_autocorrelation",
    "analyze_leg_offset",
    "analyze_signal_acf",
    "detect_peaks_after_lag_zero",
    "normalize_signal",
]
