"""Spectral gait analysis on existing normalized, smoothed joint signals.

Uses FFT and Welch PSD. Does **not** replace ``analyze_spectral`` and does
**not** write cadence, gait-quality, or other dashboard metrics.

Signals
-------
ankle height, knee angle, hip angle, pelvis movement, foot-to-ground distance.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from stablewalk.analysis.gait_acf import (
    normalize_signal,
    _pelvis_lateral_from_recording,
    _series_array,
)
from stablewalk.analysis.spectral.config import SpectralAnalysisConfig
from stablewalk.analysis.spectral.features import band_power, spectral_entropy, wrap_phase_deg
from stablewalk.analysis.spectral.matrices import lagged_cross_correlation
from stablewalk.analysis.spectral.series import (
    build_spectral_series,
    detrend_signal,
    interpolate_short_gaps,
    smooth_preserve_gait,
)
from stablewalk.analysis.spectral.spectrum import (
    _dominant_in_band,
    choose_psd_nperseg,
    rfft_periodogram,
    welch_psd,
)
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.pose_data import PoseSequence

# Spectral-series channel ids for the requested gait signals.
GAIT_SPEC_CHANNEL_IDS: dict[str, str] = {
    "left_ankle_height": "pos:left_ankle:y",
    "right_ankle_height": "pos:right_ankle:y",
    "left_knee_angle": "dof:left_knee_flexion",
    "right_knee_angle": "dof:right_knee_flexion",
    "left_hip_angle": "dof:left_hip_flexion",
    "right_hip_angle": "dof:right_hip_flexion",
    "pelvis_lateral": "pelvis:root:x",
    "pelvis_vertical": "pelvis:root:y",
    "left_heel_height": "pos:left_heel:y",
    "right_heel_height": "pos:right_heel:y",
}

GAIT_SPEC_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("ankle_height", "left_ankle_height", "right_ankle_height"),
    ("knee_angle", "left_knee_angle", "right_knee_angle"),
    ("hip_angle", "left_hip_angle", "right_hip_angle"),
    ("foot_ground", "left_foot_ground", "right_foot_ground"),
)

# Walking-relevant bands (Hz). High band is clipped to Nyquist at runtime.
LOW_BAND_HZ: tuple[float, float] = (0.40, 1.50)
MID_BAND_HZ: tuple[float, float] = (1.50, 3.00)
HIGH_BAND_HZ: tuple[float, float] = (3.00, 8.00)
GAIT_BAND_HZ: tuple[float, float] = (0.40, 4.00)

_MIN_SAMPLES = 64
_MIN_DURATION_S = 2.0
_MIN_FINITE = 8


@dataclass
class SignalSpectrumResult:
    """FFT + Welch features for one gait signal."""

    signal_id: str
    n_samples: int
    fps: float
    missing_fraction: float = 0.0
    dominant_frequency_hz: float | None = None
    dominant_period_s: float | None = None
    spectral_energy: float | None = None
    spectral_entropy: float | None = None
    harmonic_ratio: float | None = None
    low_band_power: float | None = None
    mid_band_power: float | None = None
    high_band_power: float | None = None
    freq_hz: list[float] = field(default_factory=list)
    psd: list[float | None] = field(default_factory=list)
    fft_freq_hz: list[float] = field(default_factory=list)
    fft_power: list[float | None] = field(default_factory=list)
    method: str = "insufficient"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "n_samples": self.n_samples,
            "fps": self.fps,
            "missing_fraction": self.missing_fraction,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "dominant_period_s": self.dominant_period_s,
            "spectral_energy": self.spectral_energy,
            "spectral_entropy": self.spectral_entropy,
            "harmonic_ratio": self.harmonic_ratio,
            "low_band_power": self.low_band_power,
            "mid_band_power": self.mid_band_power,
            "high_band_power": self.high_band_power,
            "freq_hz": list(self.freq_hz),
            "psd": list(self.psd),
            "fft_freq_hz": list(self.fft_freq_hz),
            "fft_power": list(self.fft_power),
            "method": self.method,
            "notes": list(self.notes),
            "units": {
                "frequency": "Hz",
                "period": "s",
                "power": "normalized_units^2",
                "psd": "normalized_units^2/Hz",
                "entropy": "dimensionless",
            },
        }


@dataclass
class LeftRightPhaseResult:
    pair_id: str
    time_offset_s: float | None = None
    phase_difference_deg: float | None = None
    peak_rho: float | None = None
    method: str = "unavailable"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "time_offset_s": self.time_offset_s,
            "phase_difference_deg": self.phase_difference_deg,
            "peak_rho": self.peak_rho,
            "method": self.method,
            "notes": list(self.notes),
            "units": {"time_offset": "s", "phase_difference": "deg"},
        }


@dataclass
class GaitSpectrumReport:
    """Session-level spectral gait product (derived, exportable)."""

    available: bool
    fps: float
    duration_s: float
    timestamps_s: list[float] = field(default_factory=list)
    signals: dict[str, SignalSpectrumResult] = field(default_factory=dict)
    phase_offsets: dict[str, LeftRightPhaseResult] = field(default_factory=dict)
    dominant_frequency_hz: float | None = None
    dominant_period_s: float | None = None
    notes: list[str] = field(default_factory=list)
    source: str = "smoothed_spectral_series"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "stablewalk-gait-spectrum",
            "schema_version": "1.0",
            "kind": "derived",
            "available": self.available,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "timestamps_s": list(self.timestamps_s),
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "dominant_period_s": self.dominant_period_s,
            "signals": {k: v.to_dict() for k, v in self.signals.items()},
            "phase_offsets": {k: v.to_dict() for k, v in self.phase_offsets.items()},
            "notes": list(self.notes),
            "source": self.source,
            "disclaimer": (
                "FFT/Welch gait features are signal-processing estimates. "
                "They are not clinical diagnoses and do not replace cadence "
                "or gait-quality metrics."
            ),
        }


def _finite_list(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for v in values:
        fv = float(v)
        out.append(None if not math.isfinite(fv) else fv)
    return out


def _remove_dc(values: np.ndarray) -> np.ndarray:
    """Z-score finite samples (mean 0). Constant series stay 0."""
    return normalize_signal(values)


def _high_band(fps: float) -> tuple[float, float]:
    nyquist = 0.5 * max(float(fps), 1e-6)
    hi = min(HIGH_BAND_HZ[1], max(HIGH_BAND_HZ[0] + 0.05, nyquist - 1e-6))
    return (HIGH_BAND_HZ[0], hi)


def _harmonic_ratio(
    freq: np.ndarray,
    psd: np.ndarray,
    f0: float,
    fps: float,
    nperseg: int,
) -> float | None:
    if f0 <= 0 or freq.size == 0 or psd.size == 0:
        return None
    j0 = int(np.argmin(np.abs(freq - f0)))
    p0 = float(psd[j0])
    if not math.isfinite(p0) or p0 <= 1e-18:
        return None
    target = 2.0 * f0
    j = int(np.argmin(np.abs(freq - target)))
    tol = max(0.15, 0.5 * fps / max(nperseg, 8))
    if abs(float(freq[j]) - target) > tol:
        return None
    p2 = float(psd[j])
    if not math.isfinite(p2) or p2 < 0:
        return None
    return float(p2 / p0)


def _cross_phase_deg(
    left: np.ndarray,
    right: np.ndarray,
    fps: float,
    f0: float,
) -> float | None:
    """Phase(right) − phase(left) at the nearest FFT bin to ``f0``."""
    n = min(left.size, right.size)
    if n < _MIN_FINITE or f0 <= 0:
        return None
    x = left[:n].copy()
    y = right[:n].copy()
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < _MIN_FINITE:
        return None
    # Longest contiguous run so FFT bins stay aligned with fps.
    valid = np.flatnonzero(mask)
    # Use full series with NaN→0 only on a contiguous finite block.
    i0, i1 = int(valid[0]), int(valid[-1]) + 1
    block_x = x[i0:i1]
    block_y = y[i0:i1]
    ok = np.isfinite(block_x) & np.isfinite(block_y)
    if int(ok.sum()) < n * 0.6:
        block_x = block_x[ok]
        block_y = block_y[ok]
    else:
        block_x = np.where(ok, block_x, 0.0)
        block_y = np.where(ok, block_y, 0.0)
    if block_x.size < _MIN_FINITE:
        return None
    block_x = block_x - float(np.mean(block_x))
    block_y = block_y - float(np.mean(block_y))
    win = np.hanning(block_x.size).astype(float)
    spec_l = np.fft.rfft(block_x * win)
    spec_r = np.fft.rfft(block_y * win)
    freq = np.fft.rfftfreq(block_x.size, d=1.0 / max(fps, 1e-6))
    j = int(np.argmin(np.abs(freq - f0)))
    cross = spec_r[j] * np.conj(spec_l[j])
    if abs(cross) <= 1e-18:
        return None
    return wrap_phase_deg(float(np.degrees(np.angle(cross))))


def analyze_signal_spectrum(
    values: np.ndarray,
    fps: float,
    *,
    signal_id: str,
    missing_fraction: float = 0.0,
    config: SpectralAnalysisConfig | None = None,
) -> SignalSpectrumResult:
    """DC-removed FFT + Welch features for one series."""
    cfg = config or SpectralAnalysisConfig()
    notes: list[str] = []
    fps = max(float(fps), 1e-6)
    y = _remove_dc(np.asarray(values, dtype=float))
    n = int(np.isfinite(y).sum())
    result = SignalSpectrumResult(
        signal_id=signal_id,
        n_samples=n,
        fps=fps,
        missing_fraction=float(missing_fraction),
        notes=notes,
    )
    if n < _MIN_FINITE:
        notes.append("Too few finite samples for a spectrum.")
        return result

    if cfg.welch_nperseg is not None:
        nperseg = max(16, min(int(cfg.welch_nperseg), max(n, 16)))
        noverlap = cfg.welch_noverlap if cfg.welch_noverlap is not None else nperseg // 2
    else:
        nperseg, noverlap, _n_seg = choose_psd_nperseg(n, fps)
    freq, pxx, method = welch_psd(
        y, fps, nperseg=nperseg, noverlap=noverlap, window=cfg.fft_window
    )
    fft_f, fft_p = rfft_periodogram(y, fps, window=cfg.fft_window)
    result.method = method if freq.size else ("rfft" if fft_f.size else "insufficient")
    result.freq_hz = [float(x) for x in freq]
    result.psd = _finite_list(pxx)
    result.fft_freq_hz = [float(x) for x in fft_f]
    result.fft_power = _finite_list(fft_p)

    use_f, use_p = (freq, pxx) if freq.size else (fft_f, fft_p)
    if use_f.size == 0 or use_p.size == 0:
        notes.append("Spectrum empty after DC removal.")
        return result

    f0, _p0 = _dominant_in_band(use_f, use_p, GAIT_BAND_HZ)
    result.dominant_frequency_hz = f0
    if f0 is not None and f0 > 1e-9:
        result.dominant_period_s = float(1.0 / f0)
    else:
        notes.append("No spectral peak inside the walking band.")

    result.spectral_energy = band_power(use_f, use_p, GAIT_BAND_HZ)
    result.spectral_entropy = spectral_entropy(use_f, use_p, GAIT_BAND_HZ)
    result.low_band_power = band_power(use_f, use_p, LOW_BAND_HZ)
    result.mid_band_power = band_power(use_f, use_p, MID_BAND_HZ)
    result.high_band_power = band_power(use_f, use_p, _high_band(fps))
    if f0 is not None:
        result.harmonic_ratio = _harmonic_ratio(use_f, use_p, f0, fps, nperseg)
    return result


def analyze_lr_phase(
    left: np.ndarray,
    right: np.ndarray,
    fps: float,
    *,
    pair_id: str,
    period_s: float | None,
    f0: float | None,
) -> LeftRightPhaseResult:
    notes: list[str] = []
    left_n = _remove_dc(left)
    right_n = _remove_dc(right)
    max_lag = int(round(2.0 * max(float(fps), 1e-6)))
    xc = lagged_cross_correlation(left_n, right_n, max_lag=max_lag, fps=fps)
    result = LeftRightPhaseResult(pair_id=pair_id, notes=notes)
    fft_phase = None
    if f0 is not None:
        fft_phase = _cross_phase_deg(left_n, right_n, fps, f0)
    if xc.peak_lag_s is not None and xc.peak_value is not None and abs(xc.peak_value) >= 0.15:
        result.time_offset_s = float(xc.peak_lag_s)
        result.peak_rho = float(xc.peak_value)
        if period_s is not None and period_s > 1e-6:
            result.phase_difference_deg = wrap_phase_deg(
                360.0 * float(xc.peak_lag_s) / float(period_s)
            )
            result.method = "cross_correlation"
        elif fft_phase is not None:
            result.phase_difference_deg = fft_phase
            result.method = "fft_cross_spectrum"
    elif fft_phase is not None:
        result.phase_difference_deg = fft_phase
        result.method = "fft_cross_spectrum"
        if f0 is not None and f0 > 1e-9:
            result.time_offset_s = float(fft_phase / 360.0 / f0)
    else:
        notes.append("Left–right phase unavailable (weak correlation and empty cross-spectrum).")
        result.method = "unavailable"
    if fft_phase is not None and result.phase_difference_deg is None:
        result.phase_difference_deg = fft_phase
    return result


def _consensus_hz(values: list[float], *, rel_tol: float = 0.20) -> float | None:
    if len(values) < 2:
        return None
    med = float(np.median(np.asarray(values, dtype=float)))
    if med <= 1e-9:
        return None
    close = [float(v) for v in values if abs(float(v) - med) / med <= rel_tol]
    if len(close) < 2:
        return None
    return float(np.median(np.asarray(close, dtype=float)))


def analyze_gait_spectrum_from_arrays(
    signals: Mapping[str, np.ndarray],
    fps: float,
    *,
    times_s: np.ndarray | None = None,
    missing: Mapping[str, float] | None = None,
    config: SpectralAnalysisConfig | None = None,
) -> GaitSpectrumReport:
    """Spectral gait features from named, already-smoothed arrays."""
    notes: list[str] = [
        "DC removed (z-score) before FFT/Welch. Results are derived, not clinical.",
    ]
    cfg = config or SpectralAnalysisConfig()
    fps = max(float(fps), 1e-6)
    arrays = {str(k): np.asarray(v, dtype=float) for k, v in signals.items()}
    n = max((a.size for a in arrays.values()), default=0)
    if times_s is not None and np.asarray(times_s).size >= 2:
        ts = np.asarray(times_s, dtype=float)
        duration = float(ts[-1] - ts[0])
        timestamps = [float(x) for x in ts]
    else:
        duration = float(n - 1) / fps if n >= 2 else 0.0
        timestamps = [i / fps for i in range(n)]

    if duration < _MIN_DURATION_S or n < _MIN_SAMPLES:
        notes.append(
            f"Clip too short for spectral gait analysis ({duration:.2f} s, {n} samples)."
        )
        return GaitSpectrumReport(
            available=False,
            fps=fps,
            duration_s=duration,
            timestamps_s=timestamps,
            notes=notes,
        )

    miss_map = dict(missing or {})
    per_signal: dict[str, SignalSpectrumResult] = {}
    for name, values in arrays.items():
        per_signal[name] = analyze_signal_spectrum(
            values,
            fps,
            signal_id=name,
            missing_fraction=float(miss_map.get(name, 0.0)),
            config=cfg,
        )

    def _gait_like(sig: SignalSpectrumResult) -> bool:
        if sig.dominant_frequency_hz is None or sig.dominant_frequency_hz <= 0:
            return False
        if sig.spectral_entropy is None or sig.spectral_entropy > 0.80:
            return False
        low = float(sig.low_band_power or 0.0)
        mid = float(sig.mid_band_power or 0.0)
        high = float(sig.high_band_power or 0.0)
        tot = low + mid + high
        return tot > 0.0 and (low / tot) >= 0.28

    freqs = [
        s.dominant_frequency_hz
        for s in per_signal.values()
        if s.dominant_frequency_hz is not None and _gait_like(s)
    ]
    f0 = _consensus_hz(freqs)
    if f0 is None and len(freqs) == 1:
        only = next(s for s in per_signal.values() if s.dominant_frequency_hz == freqs[0])
        if only.spectral_entropy is not None and only.spectral_entropy < 0.50:
            f0 = freqs[0]
    ents = [
        s.spectral_entropy
        for s in per_signal.values()
        if s.spectral_entropy is not None
    ]
    if f0 is not None and ents and float(np.median(ents)) > 0.82:
        f0 = None
        notes.append("Spectrum is broadband; dominant gait frequency withheld.")
    period = float(1.0 / f0) if f0 is not None else None

    phases: dict[str, LeftRightPhaseResult] = {}
    for pair_id, left_id, right_id in GAIT_SPEC_PAIRS:
        if left_id not in arrays or right_id not in arrays:
            continue
        phases[pair_id] = analyze_lr_phase(
            arrays[left_id],
            arrays[right_id],
            fps,
            pair_id=pair_id,
            period_s=period,
            f0=f0,
        )

    available = f0 is not None
    if not available:
        notes.append("No consistent dominant frequency across gait signals.")
        phases = {}
        period = None
    return GaitSpectrumReport(
        available=available,
        fps=fps,
        duration_s=duration,
        timestamps_s=timestamps,
        signals=per_signal,
        phase_offsets=phases,
        dominant_frequency_hz=f0,
        dominant_period_s=period,
        notes=notes,
    )


def _heel_ground_distance(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float).copy()
    finite = y[np.isfinite(y)]
    if finite.size < _MIN_FINITE:
        return y
    floor = float(np.percentile(finite, 10.0))
    out = y - floor
    out[~np.isfinite(y)] = np.nan
    return out


def _clearance_from_gait_cycle(
    gait_cycle: Any,
    times: np.ndarray,
    side: str,
) -> np.ndarray | None:
    frames = getattr(gait_cycle, "per_frame", None) or []
    if not frames:
        return None
    src_t = np.asarray([float(fr.time_s) for fr in frames], dtype=float)
    vals = np.full(src_t.shape, np.nan, dtype=float)
    for i, fr in enumerate(frames):
        sample = getattr(fr, side, None)
        dist = getattr(sample, "foot_clearance_m", None) if sample is not None else None
        if dist is None:
            continue
        try:
            vals[i] = float(dist)
        except (TypeError, ValueError):
            continue
    if int(np.isfinite(vals).sum()) < _MIN_FINITE:
        return None
    from stablewalk.analysis.spectral.series import _resample_to_grid

    return _resample_to_grid(src_t, vals, times)


def _process_like_series(
    raw: np.ndarray,
    fps: float,
    cfg: SpectralAnalysisConfig,
) -> np.ndarray:
    max_gap = max(1, int(round(cfg.max_gap_s * fps)))
    win = max(3, int(round(cfg.smooth_window_s * fps)))
    if win % 2 == 0:
        win += 1
    filled, _mask = interpolate_short_gaps(raw, max_gap_samples=max_gap)
    processed = smooth_preserve_gait(filled, window_samples=win, polyorder=cfg.smooth_polyorder)
    return detrend_signal(processed, cfg.detrend)


def analyze_gait_spectrum(
    recording: GaitMotionRecording,
    *,
    sequence: PoseSequence | None = None,
    gait_cycle: Any | None = None,
    config: SpectralAnalysisConfig | None = None,
) -> GaitSpectrumReport:
    """Spectral gait analysis from a recording's smoothed channels.

    Uses ``recording.fps`` and snapshot timestamps. Does not mutate metrics.
    """
    cfg = config or SpectralAnalysisConfig()
    notes: list[str] = []
    fps = max(float(recording.fps), 1e-6)
    bundle = build_spectral_series(
        recording,
        sequence=sequence,
        config=cfg,
        channel_ids=tuple(GAIT_SPEC_CHANNEL_IDS.values()),
    )
    notes.extend(bundle.warnings)
    times = bundle.times_s
    arrays: dict[str, np.ndarray] = {}
    missing: dict[str, float] = {}

    for name, cid in GAIT_SPEC_CHANNEL_IDS.items():
        if name.endswith("_heel_height"):
            continue
        series = bundle.get(cid)
        arr = _series_array(series)
        if arr is None:
            continue
        arrays[name] = arr
        missing[name] = float(series.missing_fraction) if series is not None else 1.0

    if "pelvis_lateral" not in arrays:
        fallback = _pelvis_lateral_from_recording(recording, times, fps, cfg)
        if fallback is not None:
            arrays["pelvis_lateral"] = fallback
            missing["pelvis_lateral"] = float(np.mean(~np.isfinite(fallback)))
        else:
            notes.append("Pelvis lateral series unavailable.")

    for side, key in (("left", "left_foot_ground"), ("right", "right_foot_ground")):
        aligned = None
        from_cycle = False
        if gait_cycle is not None:
            aligned = _clearance_from_gait_cycle(gait_cycle, times, side)
            from_cycle = aligned is not None
        if aligned is None:
            heel = _series_array(bundle.get(f"pos:{side}_heel:y"))
            ankle = _series_array(bundle.get(f"pos:{side}_ankle:y"))
            src = heel if heel is not None else ankle
            if src is not None:
                aligned = _heel_ground_distance(src)
        if aligned is None:
            notes.append(f"{side} foot-to-ground distance unavailable.")
            continue
        if from_cycle:
            aligned = _process_like_series(aligned, fps, cfg)
        arrays[key] = aligned
        missing[key] = float(np.mean(~np.isfinite(arrays[key])))

    report = analyze_gait_spectrum_from_arrays(
        arrays,
        fps,
        times_s=times,
        missing=missing,
        config=cfg,
    )
    report.notes = list(dict.fromkeys([*notes, *report.notes]))
    report.source = "smoothed_spectral_series"
    if times.size:
        report.timestamps_s = [float(x) for x in times]
        if times.size >= 2:
            report.duration_s = float(times[-1] - times[0])
    report.fps = fps
    return report


def export_gait_spectrum_json(report: GaitSpectrumReport, path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def export_gait_spectrum_csv(report: GaitSpectrumReport, path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[Any]] = []

    def section(title: str, headers: list[str], data: list[list[Any]]) -> None:
        rows.append([f"# {title}"])
        rows.append(headers)
        rows.extend(data)
        rows.append([])

    section(
        "summary",
        ["key", "value"],
        [
            ["fps", report.fps],
            ["duration_s", report.duration_s],
            ["available", "yes" if report.available else "no"],
            ["dominant_frequency_hz", report.dominant_frequency_hz or ""],
            ["dominant_period_s", report.dominant_period_s or ""],
            ["kind", "derived"],
        ],
    )
    feat_rows: list[list[Any]] = []
    for sid, sig in report.signals.items():
        feat_rows.append(
            [
                sid,
                sig.n_samples,
                sig.dominant_frequency_hz if sig.dominant_frequency_hz is not None else "",
                sig.dominant_period_s if sig.dominant_period_s is not None else "",
                sig.spectral_energy if sig.spectral_energy is not None else "",
                sig.spectral_entropy if sig.spectral_entropy is not None else "",
                sig.harmonic_ratio if sig.harmonic_ratio is not None else "",
                sig.low_band_power if sig.low_band_power is not None else "",
                sig.mid_band_power if sig.mid_band_power is not None else "",
                sig.high_band_power if sig.high_band_power is not None else "",
                sig.method,
            ]
        )
    section(
        "signal_features",
        [
            "signal_id",
            "n_samples",
            "dominant_frequency_hz",
            "dominant_period_s",
            "spectral_energy",
            "spectral_entropy",
            "harmonic_ratio",
            "low_band_power",
            "mid_band_power",
            "high_band_power",
            "method",
        ],
        feat_rows,
    )
    phase_rows = [
        [
            pid,
            ph.time_offset_s if ph.time_offset_s is not None else "",
            ph.phase_difference_deg if ph.phase_difference_deg is not None else "",
            ph.peak_rho if ph.peak_rho is not None else "",
            ph.method,
        ]
        for pid, ph in report.phase_offsets.items()
    ]
    section(
        "left_right_phase",
        ["pair_id", "time_offset_s", "phase_difference_deg", "peak_rho", "method"],
        phase_rows,
    )
    psd_rows: list[list[Any]] = []
    for sid, sig in report.signals.items():
        for f, p in zip(sig.freq_hz, sig.psd):
            psd_rows.append([sid, f, "" if p is None else p, sig.method])
    section("welch_psd", ["signal_id", "freq_hz", "psd", "method"], psd_rows)
    ts_rows = [[i, t] for i, t in enumerate(report.timestamps_s)]
    section("timestamps", ["index", "time_s"], ts_rows)
    section("notes", ["message"], [[n] for n in report.notes] or [["none"]])

    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    return dest


__all__ = [
    "GAIT_SPEC_CHANNEL_IDS",
    "GAIT_SPEC_PAIRS",
    "GaitSpectrumReport",
    "LeftRightPhaseResult",
    "SignalSpectrumResult",
    "analyze_gait_spectrum",
    "analyze_gait_spectrum_from_arrays",
    "analyze_signal_spectrum",
    "export_gait_spectrum_csv",
    "export_gait_spectrum_json",
]
