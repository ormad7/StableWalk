"""FFT periodograms, Welch PSD, harmonics, and left/right frequency compare.

Dominant-frequency search is restricted to a walking band so camera shake
and slow drift are not reported as gait. Outputs are derived estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stablewalk.analysis.spectral.config import SpectralAnalysisConfig
from stablewalk.analysis.spectral.series import ChannelSeries, SpectralSeriesBundle


def _hann(n: int, *, periodic: bool = False) -> np.ndarray:
    """Hann window. ``periodic=True`` matches SciPy Welch (fftbins)."""
    if n <= 1:
        return np.ones(max(n, 1), dtype=float)
    if periodic:
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=float) / n)
    return np.hanning(n).astype(float)


def rfft_periodogram(
    values: np.ndarray,
    fps: float,
    *,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum (density scaling, V²/Hz) of a finite run."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = x - float(np.mean(x))
    win = _hann(n) if window == "hann" else np.ones(n, dtype=float)
    spec = np.fft.rfft(x * win)
    freq = np.fft.rfftfreq(n, d=1.0 / max(fps, 1e-6))
    scale = 1.0 / (max(fps, 1e-6) * float(np.sum(win * win)))
    pxx = (np.abs(spec) ** 2) * scale
    if n % 2 == 0:
        pxx[1:-1] *= 2.0
    else:
        pxx[1:] *= 2.0
    return freq.astype(float), pxx.astype(float)


def welch_psd_numpy(
    values: np.ndarray,
    fps: float,
    *,
    nperseg: int,
    noverlap: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD implemented with ``numpy.fft.rfft`` (SciPy-compatible scaling)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    nperseg = int(max(8, min(nperseg, n)))
    noverlap = int(max(0, min(noverlap, nperseg - 1)))
    step = nperseg - noverlap
    win = _hann(nperseg, periodic=True) if window == "hann" else np.ones(nperseg, dtype=float)
    acc = None
    count = 0
    for start in range(0, n - nperseg + 1, step):
        seg = x[start : start + nperseg]
        seg = seg - float(np.mean(seg))
        spec = np.fft.rfft(seg * win)
        p = np.abs(spec) ** 2
        acc = p if acc is None else acc + p
        count += 1
    if acc is None or count == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    pxx = acc / count
    freq = np.fft.rfftfreq(nperseg, d=1.0 / max(fps, 1e-6))
    scale = 1.0 / (max(fps, 1e-6) * float(np.sum(win * win)))
    pxx = pxx * scale
    if nperseg % 2 == 0:
        pxx[1:-1] *= 2.0
    else:
        pxx[1:] *= 2.0
    return freq.astype(float), pxx.astype(float)


def welch_psd(
    values: np.ndarray,
    fps: float,
    *,
    nperseg: int,
    noverlap: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Prefer ``scipy.signal.welch`` when installed; else the numpy equivalent."""
    x = np.asarray(values, dtype=float)
    finite = x[np.isfinite(x)]
    try:
        from scipy.signal import welch as scipy_welch

        if finite.size < 8:
            return np.asarray([]), np.asarray([]), "insufficient"
        nperseg = int(max(8, min(nperseg, finite.size)))
        noverlap = int(max(0, min(noverlap, nperseg - 1)))
        freq, pxx = scipy_welch(
            finite,
            fs=fps,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )
        return np.asarray(freq, dtype=float), np.asarray(pxx, dtype=float), "scipy.signal.welch"
    except Exception:
        freq, pxx = welch_psd_numpy(
            finite, fps, nperseg=nperseg, noverlap=noverlap, window=window
        )
        return freq, pxx, "numpy_welch"


_MIN_COH_SEGMENTS = 4
_MIN_COH_NPERSEG = 32
_MIN_COH_DURATION_S = 6.0
_LIMITED_DURATION_S = 8.0
_MAX_NPERSEG = 2048


def choose_psd_nperseg(
    n: int,
    fps: float,
    *,
    target_window_s: float = 4.0,
    long_window_s: float = 8.0,
    long_duration_s: float = 16.0,
    max_nperseg: int = _MAX_NPERSEG,
) -> tuple[int, int, int]:
    """Welch length that keeps usable Δf and extra segments on long clips.

    Returns ``(nperseg, noverlap, n_segments)``. Fewer than two segments
    means the full-length rFFT should be the primary PSD.
    """
    n = int(max(0, n))
    fps = float(max(fps, 1e-6))
    duration = n / fps if n else 0.0
    if n < 16:
        return max(n, 8), 0, 1 if n >= 8 else 0
    window_s = long_window_s if duration >= long_duration_s else target_window_s
    target = int(round(window_s * fps))
    max_two = int(2 * n / 3)
    nperseg = min(max(16, target), max_two, n, int(max_nperseg))
    if nperseg % 2:
        nperseg -= 1
    nperseg = max(16, nperseg)
    noverlap = nperseg // 2
    count, nperseg, noverlap = _welch_segment_count(n, nperseg, noverlap)
    return nperseg, noverlap, count


def _prefer_rfft(n: int, fps: float, nperseg: int, n_seg: int) -> bool:
    """Use the full-length periodogram when Welch would coarsen Δf for little gain."""
    if n_seg < 2 or nperseg <= 0 or n < 8:
        return True
    if n_seg >= 3:
        return False
    df_welch = float(fps) / float(nperseg)
    df_full = float(fps) / float(max(n, 1))
    return df_welch > df_full * 1.25


def _welch_segment_count(n: int, nperseg: int, noverlap: int) -> tuple[int, int, int]:
    nperseg = int(max(8, min(nperseg, max(n, 8))))
    noverlap = int(max(0, min(noverlap, nperseg - 1)))
    step = nperseg - noverlap
    if n < nperseg or step <= 0:
        return 0, nperseg, noverlap
    return int(1 + (n - nperseg) // step), nperseg, noverlap


def coherence_nperseg(
    n: int,
    *,
    preferred: int = 256,
    min_segments: int = _MIN_COH_SEGMENTS,
    min_nperseg: int = _MIN_COH_NPERSEG,
) -> tuple[int, int, int]:
    """Choose a Welch length that can support ``min_segments`` windows."""
    n = int(n)
    nperseg = int(max(min_nperseg, min(preferred, n)))
    if nperseg % 2:
        nperseg -= 1
    nperseg = max(min_nperseg, nperseg)
    noverlap = nperseg // 2
    count, nperseg, noverlap = _welch_segment_count(n, nperseg, noverlap)
    if count >= min_segments:
        return nperseg, noverlap, count
    # 50% overlap: nperseg * (1 + (k-1)/2) <= n  →  nperseg <= 2n/(k+1)
    k = max(2, int(min_segments))
    nperseg = int(2 * n / (k + 1))
    if nperseg % 2:
        nperseg -= 1
    nperseg = int(max(min_nperseg, min(nperseg, n)))
    noverlap = nperseg // 2
    count, nperseg, noverlap = _welch_segment_count(n, nperseg, noverlap)
    return nperseg, noverlap, count


def welch_csd_numpy(
    x: np.ndarray,
    y: np.ndarray,
    fps: float,
    *,
    nperseg: int,
    noverlap: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """One-sided Welch Sxx, Syy, Sxy. Same scaling as ``welch_psd_numpy``."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    n = xa.size
    if n < 8:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty, 0
    nperseg = int(max(8, min(nperseg, n)))
    noverlap = int(max(0, min(noverlap, nperseg - 1)))
    step = nperseg - noverlap
    win = _hann(nperseg, periodic=True) if window == "hann" else np.ones(nperseg, dtype=float)
    acc_xx = None
    acc_yy = None
    acc_xy = None
    count = 0
    for start in range(0, n - nperseg + 1, step):
        sx = xa[start : start + nperseg] - float(np.mean(xa[start : start + nperseg]))
        sy = ya[start : start + nperseg] - float(np.mean(ya[start : start + nperseg]))
        X = np.fft.rfft(sx * win)
        Y = np.fft.rfft(sy * win)
        pxx = np.abs(X) ** 2
        pyy = np.abs(Y) ** 2
        pxy = X * np.conjugate(Y)
        acc_xx = pxx if acc_xx is None else acc_xx + pxx
        acc_yy = pyy if acc_yy is None else acc_yy + pyy
        acc_xy = pxy if acc_xy is None else acc_xy + pxy
        count += 1
    if acc_xx is None or count == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty, 0
    scale = 1.0 / (max(fps, 1e-6) * float(np.sum(win * win)))
    sxx = (acc_xx / count) * scale
    syy = (acc_yy / count) * scale
    sxy = (acc_xy / count) * scale
    if nperseg % 2 == 0:
        sxx[1:-1] *= 2.0
        syy[1:-1] *= 2.0
        sxy[1:-1] *= 2.0
    else:
        sxx[1:] *= 2.0
        syy[1:] *= 2.0
        sxy[1:] *= 2.0
    freq = np.fft.rfftfreq(nperseg, d=1.0 / max(fps, 1e-6))
    return freq.astype(float), sxx.astype(float), syy.astype(float), np.asarray(sxy), count


@dataclass
class CoherenceResult:
    channel_a: str
    channel_b: str
    freq_hz: np.ndarray
    coherence: np.ndarray
    n_segments: int
    n_paired: int
    method: str
    sufficient: bool
    mean_band: float | None = None
    peak_band: float | None = None
    peak_hz: float | None = None
    notes: list[str] = field(default_factory=list)


def magnitude_squared_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fps: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    min_segments: int = _MIN_COH_SEGMENTS,
    min_duration_s: float = _MIN_COH_DURATION_S,
    band: tuple[float, float] | None = None,
    channel_a: str = "",
    channel_b: str = "",
) -> CoherenceResult:
    """|P_xy(f)|² / (P_xx(f) P_yy(f)) on pairwise-complete samples.

    Withheld when fewer than ``min_segments`` Welch windows are available.
    This is not time-domain cross-correlation.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n_in = min(xa.size, ya.size)
    xa = xa[:n_in]
    ya = ya[:n_in]
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    n = int(xa.size)
    notes: list[str] = [
        "Magnitude-squared coherence is a frequency-domain association, "
        "not time-domain cross-correlation."
    ]
    empty = CoherenceResult(
        channel_a=channel_a,
        channel_b=channel_b,
        freq_hz=np.asarray([], dtype=float),
        coherence=np.asarray([], dtype=float),
        n_segments=0,
        n_paired=n,
        method="insufficient",
        sufficient=False,
        notes=notes,
    )
    duration_s = n / max(float(fps), 1e-6)
    if n < _MIN_COH_NPERSEG * 2:
        empty.notes.append("Too few paired samples for coherence.")
        return empty
    if duration_s + 1e-9 < float(min_duration_s):
        empty.notes.append(
            f"Clip is {duration_s:.2f} s; need ≥ {float(min_duration_s):.1f} s "
            f"and ≥ {min_segments} Welch segments for a supported coherence estimate."
        )
        return empty
    if nperseg is None:
        nperseg, noverlap, n_seg = coherence_nperseg(n, min_segments=min_segments)
    else:
        nperseg = int(nperseg)
        noverlap = nperseg // 2 if noverlap is None else int(noverlap)
        n_seg, nperseg, noverlap = _welch_segment_count(n, nperseg, noverlap)
    sufficient = n_seg >= min_segments
    if n_seg < 2:
        empty.n_segments = n_seg
        empty.notes.append(
            f"Only {n_seg} Welch segment(s); need ≥ {min_segments} for a "
            "supported coherence estimate."
        )
        return empty
    if not sufficient:
        notes.append(
            f"Only {n_seg} Welch segment(s); need ≥ {min_segments} for a "
            "supported coherence estimate."
        )
        empty.n_segments = n_seg
        empty.notes = notes
        return empty
    method = "numpy_coherence"
    freq = np.asarray([], dtype=float)
    coh = np.asarray([], dtype=float)
    try:
        from scipy.signal import coherence as scipy_coherence

        freq, coh = scipy_coherence(
            xa,
            ya,
            fs=fps,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        freq = np.asarray(freq, dtype=float)
        coh = np.clip(np.asarray(coh, dtype=float), 0.0, 1.0)
        method = "scipy.signal.coherence"
    except Exception:
        freq, sxx, syy, sxy, n_seg = welch_csd_numpy(
            xa, ya, fps, nperseg=nperseg, noverlap=noverlap, window=window
        )
        if freq.size == 0 or sxx.size == 0:
            empty.n_segments = n_seg
            empty.notes.append("Coherence estimate failed.")
            return empty
        denom = np.clip(sxx, 0.0, None) * np.clip(syy, 0.0, None)
        coh = np.full(freq.shape, np.nan, dtype=float)
        ok = denom > 1e-18
        coh[ok] = np.clip(np.abs(sxy[ok]) ** 2 / denom[ok], 0.0, 1.0)
        method = "numpy_coherence"
    lo, hi = band if band is not None else (0.40, 4.00)
    band_mask = (freq >= lo) & (freq <= hi) & (freq > 0) & np.isfinite(coh)
    mean_band = float(np.nanmean(coh[band_mask])) if np.any(band_mask) else None
    peak_band = None
    peak_hz = None
    if np.any(band_mask):
        i = int(np.nanargmax(coh[band_mask]))
        peak_band = float(coh[band_mask][i])
        peak_hz = float(freq[band_mask][i])
    return CoherenceResult(
        channel_a=channel_a,
        channel_b=channel_b,
        freq_hz=freq,
        coherence=coh,
        n_segments=int(n_seg),
        n_paired=n,
        method=method,
        sufficient=True,
        mean_band=mean_band,
        peak_band=peak_band,
        peak_hz=peak_hz,
        notes=notes,
    )


@dataclass
class ChannelSpectrum:
    channel_id: str
    freq_hz: np.ndarray
    psd: np.ndarray
    fft_freq_hz: np.ndarray
    fft_power: np.ndarray
    method: str
    dominant_hz: float | None = None
    dominant_power: float | None = None
    spectral_centroid_hz: float | None = None
    spectral_bandwidth_hz: float | None = None
    spectral_entropy: float | None = None
    band_power_fraction: float | None = None
    harmonic_hz: list[float] = field(default_factory=list)
    harmonic_ratio: float | None = None
    quality_status: str = "insufficient"
    n_segments: int = 0
    resolution_hz: float | None = None
    band_energy: float | None = None
    total_energy: float | None = None
    duration_s: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "freq_hz": [float(x) for x in self.freq_hz],
            "psd": [float(x) for x in self.psd],
            "fft_freq_hz": [float(x) for x in self.fft_freq_hz],
            "fft_power": [float(x) for x in self.fft_power],
            "method": self.method,
            "dominant_hz": self.dominant_hz,
            "dominant_power": self.dominant_power,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "spectral_bandwidth_hz": self.spectral_bandwidth_hz,
            "spectral_entropy": self.spectral_entropy,
            "band_power_fraction": self.band_power_fraction,
            "harmonic_hz": list(self.harmonic_hz),
            "harmonic_ratio": self.harmonic_ratio,
            "quality_status": self.quality_status,
            "n_segments": self.n_segments,
            "resolution_hz": self.resolution_hz,
            "band_energy": self.band_energy,
            "total_energy": self.total_energy,
            "duration_s": self.duration_s,
            "notes": list(self.notes),
            "units": {
                "frequency": "Hz",
                "psd": "units^2/Hz",
                "energy": "units^2",
            },
        }


@dataclass
class LeftRightSpectrumCompare:
    pair_id: str
    left_id: str
    right_id: str
    left_dominant_hz: float | None
    right_dominant_hz: float | None
    delta_hz: float | None
    left_centroid_hz: float | None
    right_centroid_hz: float | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "left_dominant_hz": self.left_dominant_hz,
            "right_dominant_hz": self.right_dominant_hz,
            "delta_hz": self.delta_hz,
            "left_centroid_hz": self.left_centroid_hz,
            "right_centroid_hz": self.right_centroid_hz,
            "notes": list(self.notes),
            "kind": "derived",
        }


def _band_mask(freq: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return (freq >= lo) & (freq <= hi) & (freq > 0)


def _dominant_in_band(freq: np.ndarray, pxx: np.ndarray, band: tuple[float, float]) -> tuple[float | None, float | None]:
    mask = _band_mask(freq, band)
    if not np.any(mask):
        return None, None
    f = freq[mask]
    p = pxx[mask]
    if p.size == 0 or not np.any(np.isfinite(p)):
        return None, None
    i = int(np.nanargmax(p))
    return float(f[i]), float(p[i])


def _centroid(freq: np.ndarray, pxx: np.ndarray, band: tuple[float, float]) -> float | None:
    mask = _band_mask(freq, band) & np.isfinite(pxx)
    if not np.any(mask):
        return None
    w = np.clip(pxx[mask], 0.0, None)
    s = float(np.sum(w))
    if s <= 1e-18:
        return None
    return float(np.sum(freq[mask] * w) / s)


def _bandwidth(freq: np.ndarray, pxx: np.ndarray, band: tuple[float, float]) -> float | None:
    mask = _band_mask(freq, band) & np.isfinite(pxx)
    if not np.any(mask):
        return None
    w = np.clip(pxx[mask], 0.0, None)
    s = float(np.sum(w))
    if s <= 1e-18 or int(np.count_nonzero(mask)) < 2:
        return None
    f = freq[mask]
    c = float(np.sum(f * w) / s)
    var = float(np.sum(w * (f - c) ** 2) / s)
    if var < 0.0:
        return None
    return float(np.sqrt(var))


def _median_df(freq: np.ndarray) -> float:
    diffs = np.diff(np.asarray(freq, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if diffs.size else 0.0


def _integrated_power(
    freq: np.ndarray,
    pxx: np.ndarray,
    band: tuple[float, float] | None = None,
) -> float | None:
    """∫ P(f) df. ``band=None`` integrates the full positive-frequency PSD."""
    freq = np.asarray(freq, dtype=float)
    pxx = np.asarray(pxx, dtype=float)
    if freq.size == 0 or pxx.size != freq.size:
        return None
    if band is None:
        mask = (freq > 0) & np.isfinite(pxx)
    else:
        mask = _band_mask(freq, band) & np.isfinite(pxx)
    p = np.clip(pxx[mask], 0.0, None)
    if p.size == 0:
        return None
    df = _median_df(freq)
    val = float(np.sum(p)) if df <= 0.0 else float(np.sum(p) * df)
    return val if np.isfinite(val) else None


def _entropy(freq: np.ndarray, pxx: np.ndarray, band: tuple[float, float]) -> float | None:
    mask = _band_mask(freq, band) & np.isfinite(pxx)
    w = np.clip(pxx[mask], 0.0, None) if np.any(mask) else np.asarray([], dtype=float)
    total = float(np.sum(w))
    if total <= 1e-18 or w.size < 2:
        return None
    probs = w / total
    probs = probs[probs > 0]
    if probs.size == 0:
        return None
    if probs.size == 1:
        return 0.0
    entropy = float(-np.sum(probs * np.log(probs)))
    return float(entropy / np.log(probs.size))


def analyze_channel_spectrum(
    series: ChannelSeries,
    *,
    config: SpectralAnalysisConfig | None = None,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> ChannelSpectrum:
    cfg = config or SpectralAnalysisConfig()
    norm = getattr(series, "normalized", None)
    if norm is not None and np.size(norm) and int(np.isfinite(norm).sum()) >= 8:
        y = norm
    else:
        y = series.processed
    n = int(np.isfinite(y).sum())
    notes: list[str] = []
    duration = float(getattr(series, "duration_s", 0.0) or 0.0)
    if duration <= 0.0 and series.fps > 0:
        duration = n / float(series.fps)
    chosen_nperseg, chosen_noverlap, n_seg = choose_psd_nperseg(n, series.fps)
    if cfg.welch_nperseg is not None:
        chosen_nperseg = int(max(8, min(int(cfg.welch_nperseg), max(n, 8))))
        chosen_noverlap = (
            int(cfg.welch_noverlap)
            if cfg.welch_noverlap is not None
            else chosen_nperseg // 2
        )
        n_seg, chosen_nperseg, chosen_noverlap = _welch_segment_count(
            n, chosen_nperseg, chosen_noverlap
        )
    if nperseg is not None:
        chosen_nperseg = int(max(8, min(int(nperseg), max(n, 8))))
        chosen_noverlap = (
            int(noverlap) if noverlap is not None else chosen_nperseg // 2
        )
        n_seg, chosen_nperseg, chosen_noverlap = _welch_segment_count(
            n, chosen_nperseg, chosen_noverlap
        )
    freq, pxx, welch_method = welch_psd(
        y,
        series.fps,
        nperseg=chosen_nperseg,
        noverlap=chosen_noverlap,
        window=cfg.fft_window,
    )
    fft_f, fft_p = rfft_periodogram(y, series.fps, window=cfg.fft_window)
    use_rfft = _prefer_rfft(n, series.fps, chosen_nperseg, n_seg)
    if use_rfft and fft_f.size:
        use_f, use_p = fft_f, fft_p
        method = "rfft_periodogram"
        n_seg = 1
        df = _median_df(fft_f) or (series.fps / max(n, 1))
    else:
        use_f, use_p = (freq, pxx) if freq.size else (fft_f, fft_p)
        method = welch_method if freq.size else ("rfft_periodogram" if fft_f.size else "insufficient")
        df = _median_df(use_f)
        if df <= 0.0:
            df = series.fps / max(chosen_nperseg if not use_rfft else n, 1)
    spec = ChannelSpectrum(
        channel_id=series.spec.channel_id,
        freq_hz=use_f,
        psd=use_p,
        fft_freq_hz=fft_f,
        fft_power=fft_p,
        method=method,
        n_segments=int(n_seg),
        resolution_hz=float(df) if df > 0 else None,
        duration_s=duration,
        notes=notes,
    )
    spec.band_energy = _integrated_power(use_f, use_p, cfg.gait_band_hz)
    spec.total_energy = _integrated_power(use_f, use_p, None)
    if spec.total_energy is not None and spec.total_energy > 1e-18 and spec.band_energy is not None:
        spec.band_power_fraction = float(spec.band_energy / spec.total_energy)
    too_short = n < cfg.min_samples or duration + 1e-9 < cfg.min_duration_s
    if use_f.size < 4 or use_p.size < 4:
        spec.quality_status = "insufficient"
        notes.append("Too few frequency bins for a spectrum.")
        return spec
    if too_short:
        spec.quality_status = "insufficient"
        notes.append(
            f"Clip is {duration:.2f} s with {n} finite samples — too short for "
            "a stable dominant frequency or gait period. PSD is shown; "
            "frequency metrics stay blank."
        )
        spec.spectral_centroid_hz = _centroid(use_f, use_p, cfg.gait_band_hz)
        spec.spectral_bandwidth_hz = _bandwidth(use_f, use_p, cfg.gait_band_hz)
        spec.spectral_entropy = _entropy(use_f, use_p, cfg.gait_band_hz)
        return spec
    f0, p0 = _dominant_in_band(use_f, use_p, cfg.gait_band_hz)
    spec.dominant_hz = f0
    spec.dominant_power = p0
    spec.spectral_centroid_hz = _centroid(use_f, use_p, cfg.gait_band_hz)
    spec.spectral_bandwidth_hz = _bandwidth(use_f, use_p, cfg.gait_band_hz)
    spec.spectral_entropy = _entropy(use_f, use_p, cfg.gait_band_hz)
    if spec.band_power_fraction is not None and spec.band_power_fraction < 0.15:
        notes.append(
            "Most spectral energy lies outside the walking band; dominant "
            "frequency may reflect noise or slow drift rather than gait."
        )
    # Reliability comes from duration and Welch segment count — not from
    # FFT bin spacing Δf, which is a grid resolution, not confidence.
    limited = int(n_seg) < 3 or duration + 1e-9 < _LIMITED_DURATION_S
    spec.quality_status = "limited" if limited else "computed"
    if limited:
        notes.append(
            f"Spectral reliability is limited ({duration:.2f} s, "
            f"{int(n_seg)} Welch segment(s)). Dominant frequency, if shown, "
            "is a descriptive peak — not a confidence-weighted measurement. "
            "Δf is FFT bin spacing, not reliability."
        )
    if f0 and f0 > 0:
        harmonics: list[float] = []
        p_h = []
        for k in (2, 3):
            target = k * f0
            if target > cfg.gait_band_hz[1]:
                continue
            src_f = use_f
            src_p = use_p
            if src_f.size == 0:
                continue
            j = int(np.argmin(np.abs(src_f - target)))
            if abs(float(src_f[j]) - target) <= max(0.15, 0.5 * series.fps / max(chosen_nperseg, 8)):
                harmonics.append(float(src_f[j]))
                p_h.append(float(src_p[j]))
        spec.harmonic_hz = harmonics
        if p0 and p0 > 1e-18 and p_h:
            spec.harmonic_ratio = float(p_h[0] / p0)
        notes.append(
            f"Dominant frequency searched in {cfg.gait_band_hz[0]:.1f}–{cfg.gait_band_hz[1]:.1f} Hz."
        )
    else:
        spec.dominant_hz = None
        spec.dominant_power = None
        notes.append("No spectral peak inside the walking frequency band.")
    return spec


def compare_left_right_spectra(
    spectra: dict[str, ChannelSpectrum],
    pairs: tuple[tuple[str, str, str], ...],
) -> list[LeftRightSpectrumCompare]:
    out: list[LeftRightSpectrumCompare] = []
    for pair_id, left_id, right_id in pairs:
        sl, sr = spectra.get(left_id), spectra.get(right_id)
        if sl is None or sr is None:
            continue
        delta = None
        notes = [
            "Left/right frequency difference is a kinematic observation, not a diagnosis."
        ]
        if sl.dominant_hz is not None and sr.dominant_hz is not None:
            delta = float(sr.dominant_hz - sl.dominant_hz)
        out.append(
            LeftRightSpectrumCompare(
                pair_id=pair_id,
                left_id=left_id,
                right_id=right_id,
                left_dominant_hz=sl.dominant_hz,
                right_dominant_hz=sr.dominant_hz,
                delta_hz=delta,
                left_centroid_hz=sl.spectral_centroid_hz,
                right_centroid_hz=sr.spectral_centroid_hz,
                notes=notes,
            )
        )
    return out


def analyze_spectra(
    bundle: SpectralSeriesBundle,
    channel_ids: list[str],
    *,
    config: SpectralAnalysisConfig | None = None,
) -> dict[str, ChannelSpectrum]:
    cfg = config or SpectralAnalysisConfig()
    out: dict[str, ChannelSpectrum] = {}
    for cid in channel_ids:
        series = bundle.get(cid)
        if series is None:
            continue
        out[cid] = analyze_channel_spectrum(series, config=cfg)
    return out
