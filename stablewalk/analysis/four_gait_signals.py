"""Four thigh/shank segment-angle signals and their zero-lag Pearson matrix.

Reuses ``build_spectral_series`` (resample, gap fill, Savitzky–Golay) and
``pearson_overlap`` / ``lagged_matrix`` / ``lagged_cross_correlation``. Does
not add a second smoother or a new correlation algorithm.

Zero-lag r is a same-time association. Left and right limbs are typically
out of phase in walking, so a low or negative bilateral coefficient is not
a finding of gait instability.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from stablewalk.analysis.spectral.autocorrelation import (
    AutocorrResult,
    analyze_autocorrelation,
)
from stablewalk.analysis.spectral.channels import (
    CHANNEL_BY_ID,
    FOUR_GAIT_ANGLE_UNITS,
    FOUR_GAIT_ANGLE_YLABEL,
    FOUR_GAIT_CHANNEL_IDS,
    SEGMENT_SPECS,
    four_gait_joints,
)
from stablewalk.analysis.spectral.config import DEFAULT_GAIT_BAND_HZ, default_config
from stablewalk.analysis.spectral.matrices import (
    CrossCorrSeries,
    lagged_cross_correlation,
    lagged_matrix,
    pearson_overlap,
)
from stablewalk.analysis.spectral.series import (
    ChannelSeries,
    build_spectral_series,
    detrend_signal,
    normalize_unit_variance,
)
from stablewalk.analysis.spectral.spectrum import (
    ChannelSpectrum,
    CoherenceResult,
    analyze_channel_spectrum,
    choose_psd_nperseg,
    magnitude_squared_coherence,
)
from stablewalk.analysis.gait_timing import GaitTimingResult, compose_gait_timing
from stablewalk.models.gait_motion import GaitMotionRecording

SignalMode = Literal["raw", "filtered"]

FOUR_GAIT_LABELS: tuple[str, ...] = tuple(spec.label for spec in SEGMENT_SPECS)
FOUR_GAIT_SHORT_LABELS: tuple[str, ...] = (
    "Left Thigh",
    "Right Thigh",
    "Left Shank",
    "Right Shank",
)

# (name, i, j, display) — highlighted comparisons requested in the UI.
FOCUS_PAIRS: tuple[tuple[str, int, int, str], ...] = (
    ("thigh_lr", 0, 1, "Left Thigh vs Right Thigh"),
    ("shank_lr", 2, 3, "Left Shank vs Right Shank"),
    ("left_intra", 0, 2, "Left Thigh vs Left Shank"),
    ("right_intra", 1, 3, "Right Thigh vs Right Shank"),
)

INTERPRETATION = (
    "Zero-lag Pearson r is the same-time linear association of the four "
    "segment angles (−1 to 1). During walking the left and right limbs are "
    "typically out of phase, so bilateral r (thigh–thigh, shank–shank) is "
    "often low or negative even in regular gait — that is not instability. "
    "Use the xcorr peak lag for the left–right phase shift. Ipsilateral "
    "thigh–shank r at lag 0 is simultaneous coupling only. Blank cells mean "
    "too few paired samples or a constant series. Not a diagnosis."
)

SPECTRAL_COMPARE_INTERPRETATION = (
    "The 4×4 matrix is Pearson r of unit-energy PSD shapes on shared "
    "walking-band frequency bins — a mathematical shape comparison, not a "
    "visual overlay. It is not time-domain C(lag) or cross-correlation. "
    "Left–right thigh, left–right shank, and ipsilateral thigh–shank "
    "overlays use the same PSD estimates. Close dominant frequencies are "
    "a kinematic observation only and do not prove symmetric gait. "
    "Magnitude-squared coherence is shown only when enough Welch segments "
    "are available."
)

SPECTRUM_INTERPRETATION = (
    "PSD of the full-frame uniform FPS series (Hann, mean removed, linear "
    "detrend) — not 0.25 s / 0.5 s summary samples. Longer clips use Welch "
    "averaging; short clips use the full-length rFFT so Δf is not coarsened. "
    "Dominant frequency is the walking-band peak (0.40–4.00 Hz) and is left "
    "blank when the clip is too short. Spectral entropy is normalized to "
    "[0, 1] in that band. Energy is ∫ P(f) df over the band. Δf is FFT bin "
    "spacing, not measurement confidence. Short recordings still show the "
    "PSD and descriptive metrics with a reliability warning. This is a "
    "kinematic spectrum, not a clinical finding."
)

LAG_INTERPRETATION = (
    "Each panel is normalized autocorrelation ρ(τ) of one thigh/shank series, "
    "or left–right cross-correlation. ACF stride peaks are searched in "
    "0.55–2.20 s (physiologically plausible walking). The reported gait "
    "period is that ACF peak only when the clip spans enough cycles and the "
    "peak is positive — heel-strike intervals are an independent check and "
    "are never copied into the ACF. Best left–right lag is the xcorr "
    "maximum in seconds. A low or negative left–right r at τ = 0 is walking "
    "antiphase — not a finding of instability. Short clips, weak "
    "periodicity, or peaks outside the search window stay blank. Not a "
    "diagnosis."
)

MATRIX_INTERPRETATION = (
    "The large 4×4 heatmap is the zero-lag Pearson matrix C(0) of Left/Right "
    "Thigh (hip→knee) and Left/Right Shank (knee→ankle). Values are "
    "dimensionless r in [−1, +1]. C(0) is symmetric; a diagonal cell is 1 "
    "only when that series has variance and enough paired samples, otherwise "
    "it stays blank. The separate C(τ) matrix uses the lag slider "
    "(τ = samples / FPS) and is not symmetric for τ ≠ 0. Negative C(0) "
    "between opposing legs is the usual walking antiphase — not proof of "
    "asymmetric gait. Blank cells are not filled in. Not a diagnosis."
)

# Left–right xcorr pairs (name, left_id, right_id, display).
LR_XCORR_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("thigh_lr", "seg:left_thigh", "seg:right_thigh", "Left Thigh vs Right Thigh"),
    ("shank_lr", "seg:left_shank", "seg:right_shank", "Left Shank vs Right Shank"),
)

# Ipsilateral thigh–shank xcorr (phase-matched, not simultaneous).
INTRA_XCORR_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("left_intra", "seg:left_thigh", "seg:left_shank", "Left Thigh vs Left Shank"),
    ("right_intra", "seg:right_thigh", "seg:right_shank", "Right Thigh vs Right Shank"),
)

XCORR_PAIRS: tuple[tuple[str, str, str, str], ...] = LR_XCORR_PAIRS + INTRA_XCORR_PAIRS

# Spectral comparison pairs (name, id_a, id_b, display).
COMPARE_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("thigh_lr", "seg:left_thigh", "seg:right_thigh", "Left Thigh vs Right Thigh"),
    ("shank_lr", "seg:left_shank", "seg:right_shank", "Left Shank vs Right Shank"),
    ("left_intra", "seg:left_thigh", "seg:left_shank", "Left Thigh vs Left Shank"),
    ("right_intra", "seg:right_thigh", "seg:right_shank", "Right Thigh vs Right Shank"),
)

_MIN_PAIRED = 4
_PERIOD_REL_TOL = 0.25
_PEAK_REL = 0.20
_MAX_PEAKS = 5


def _psd_arrays(spec: ChannelSpectrum) -> tuple[np.ndarray, np.ndarray]:
    if spec.freq_hz.size:
        return np.asarray(spec.freq_hz, dtype=float), np.asarray(spec.psd, dtype=float)
    return np.asarray(spec.fft_freq_hz, dtype=float), np.asarray(spec.fft_power, dtype=float)


def detect_spectral_peaks(
    spec: ChannelSpectrum,
    *,
    band: tuple[float, float] = DEFAULT_GAIT_BAND_HZ,
    min_rel: float = _PEAK_REL,
    max_peaks: int = _MAX_PEAKS,
) -> list[tuple[float, float]]:
    """Local PSD maxima in the walking band, strongest first."""
    freq, psd = _psd_arrays(spec)
    if freq.size < 3 or psd.size != freq.size:
        return []
    lo, hi = band
    mask = (freq >= lo) & (freq <= hi) & (freq > 0) & np.isfinite(psd)
    if int(mask.sum()) < 3:
        return []
    f = freq[mask]
    p = np.clip(psd[mask], 0.0, None)
    peak_h = float(np.max(p))
    if peak_h <= 1e-18:
        return []
    floor = float(min_rel) * peak_h
    found: list[tuple[float, float]] = []
    for i in range(1, int(p.size) - 1):
        if p[i] < floor:
            continue
        if p[i] >= p[i - 1] and p[i] > p[i + 1]:
            found.append((float(f[i]), float(p[i])))
    found.sort(key=lambda item: item[1], reverse=True)
    found = found[: int(max_peaks)]
    return found


def spectral_band_power(
    spec: ChannelSpectrum,
    band: tuple[float, float] = DEFAULT_GAIT_BAND_HZ,
) -> float | None:
    """∫ P(f) df over the walking band (sum of bins × Δf)."""
    stored = getattr(spec, "band_energy", None)
    if stored is not None and np.isfinite(float(stored)):
        return float(stored)
    freq, psd = _psd_arrays(spec)
    if freq.size == 0 or psd.size != freq.size:
        return None
    lo, hi = band
    mask = (freq >= lo) & (freq <= hi) & (freq > 0) & np.isfinite(psd)
    p = np.clip(psd[mask], 0.0, None)
    if p.size == 0:
        return None
    diffs = np.diff(freq)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    df = float(np.median(diffs)) if diffs.size else 0.0
    val = float(np.sum(p)) if df <= 0.0 else float(np.sum(p) * df)
    return val if np.isfinite(val) else None


def _spectrum_on_values(
    series: ChannelSeries,
    values: np.ndarray,
    cfg: Any,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> ChannelSpectrum:
    """Welch + rFFT on a linearly detrended, z-scored full-frame copy."""
    y = np.asarray(values, dtype=float)
    detrended = detrend_signal(y, "linear")
    clone = replace(
        series,
        processed=detrended,
        normalized=normalize_unit_variance(detrended),
        valid_mask=np.isfinite(detrended),
        missing_fraction=float(np.mean(~np.isfinite(y))) if y.size else 1.0,
    )
    return analyze_channel_spectrum(
        clone, config=cfg, nperseg=nperseg, noverlap=noverlap
    )


def _spectrum_quality_notes(spec: ChannelSpectrum, label: str) -> list[str]:
    notes: list[str] = []
    status = str(getattr(spec, "quality_status", "") or "")
    if status in {"insufficient", "limited"}:
        notes.append(
            f"{label}: spectral quality {status} — descriptive only, not a "
            "reliable classification."
        )
    for note in spec.notes:
        text = str(note).strip()
        if not text:
            continue
        if "searched in" in text.lower():
            continue
        notes.append(f"{label}: {text}")
    return notes


@dataclass
class DominantFrequencyCompare:
    pair_name: str
    label: str
    channel_a: str
    channel_b: str
    f_a_hz: float | None
    f_b_hz: float | None
    delta_hz: float | None
    delta_pct: float | None
    notes: list[str] = field(default_factory=list)


def common_psd_frequency(
    spectra: dict[str, ChannelSpectrum],
    channel_ids: tuple[str, ...] = FOUR_GAIT_CHANNEL_IDS,
) -> np.ndarray:
    """Shared frequency axis: exact match, else the coarsest grid (no upsampling)."""
    grids: list[np.ndarray] = []
    for cid in channel_ids:
        spec = spectra.get(cid)
        if spec is None:
            continue
        freq, _psd = _psd_arrays(spec)
        if freq.size:
            grids.append(np.asarray(freq, dtype=float))
    if not grids:
        return np.asarray([], dtype=float)
    ref = grids[0]
    if all(
        g.size == ref.size and np.allclose(g, ref, atol=1e-9, rtol=0.0) for g in grids
    ):
        return ref
    grids.sort(key=lambda g: (g.size, float(np.median(np.diff(g))) if g.size > 1 else 0.0))
    return grids[0]


def align_psd_shapes(
    spectra: dict[str, ChannelSpectrum],
    *,
    band: tuple[float, float] = DEFAULT_GAIT_BAND_HZ,
    channel_ids: tuple[str, ...] = FOUR_GAIT_CHANNEL_IDS,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Unit-energy PSD shapes on a common walking-band frequency grid."""
    freq = common_psd_frequency(spectra, channel_ids)
    if freq.size == 0:
        return freq, [np.asarray([], dtype=float) for _ in channel_ids]
    lo, hi = band
    mask = (freq >= lo) & (freq <= hi) & (freq > 0)
    grid = freq[mask]
    shapes: list[np.ndarray] = []
    for cid in channel_ids:
        spec = spectra.get(cid)
        if spec is None or grid.size == 0:
            shapes.append(np.full(grid.size, np.nan, dtype=float))
            continue
        src_f, src_p = _psd_arrays(spec)
        if src_f.size == 0:
            shapes.append(np.full(grid.size, np.nan, dtype=float))
            continue
        if src_f.size == freq.size and np.allclose(src_f, freq, atol=1e-9, rtol=0.0):
            p = np.clip(np.asarray(src_p[mask], dtype=float), 0.0, None)
        else:
            p = np.clip(
                np.interp(grid, src_f, np.clip(src_p, 0.0, None), left=np.nan, right=np.nan),
                0.0,
                None,
            )
        total = float(np.nansum(p))
        if total <= 1e-18 or int(np.isfinite(p).sum()) < _MIN_PAIRED:
            shapes.append(np.full(grid.size, np.nan, dtype=float))
        else:
            shapes.append(p / total)
    return grid, shapes


def spectral_shape_correlation(
    spectra: dict[str, ChannelSpectrum],
    *,
    band: tuple[float, float] = DEFAULT_GAIT_BAND_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4×4 Pearson r of unit-energy PSD shapes. Not time-domain C(lag)."""
    grid, shapes = align_psd_shapes(spectra, band=band)
    if not shapes or grid.size == 0:
        empty = np.full((4, 4), np.nan, dtype=float)
        return empty, np.zeros((4, 4), dtype=int), grid
    corr, counts = pairwise_pearson(shapes)
    return corr, counts, grid


def compare_dominant_frequencies(
    spectra: dict[str, ChannelSpectrum],
) -> dict[str, DominantFrequencyCompare]:
    """Δf (Hz and %) for the four focus pairs. Similar f0 ≠ symmetry."""
    out: dict[str, DominantFrequencyCompare] = {}
    note = "Similar dominant frequencies are not proof of symmetric gait."
    for name, id_a, id_b, label in COMPARE_PAIRS:
        sa = spectra.get(id_a)
        sb = spectra.get(id_b)
        fa = None if sa is None else sa.dominant_hz
        fb = None if sb is None else sb.dominant_hz
        delta = None
        pct = None
        notes = [note]
        if fa is None or fb is None:
            notes.append("Dominant frequency unavailable on one or both channels.")
        elif not (np.isfinite(float(fa)) and np.isfinite(float(fb))):
            notes.append("Dominant frequency is not finite.")
        else:
            delta = float(fb) - float(fa)
            mid = 0.5 * (float(fa) + float(fb))
            if abs(mid) > 1e-9:
                pct = 100.0 * delta / mid
        out[name] = DominantFrequencyCompare(
            pair_name=name,
            label=label,
            channel_a=id_a,
            channel_b=id_b,
            f_a_hz=None if fa is None else float(fa),
            f_b_hz=None if fb is None else float(fb),
            delta_hz=delta,
            delta_pct=pct,
            notes=notes,
        )
    return out


def _coherence_pairs(
    series: dict[str, np.ndarray],
    *,
    fps: float,
    window: str,
    band: tuple[float, float],
) -> dict[str, CoherenceResult]:
    out: dict[str, CoherenceResult] = {}
    for name, id_a, id_b, _label in COMPARE_PAIRS:
        xa = detrend_signal(np.asarray(series.get(id_a), dtype=float), "linear")
        xb = detrend_signal(np.asarray(series.get(id_b), dtype=float), "linear")
        out[name] = magnitude_squared_coherence(
            normalize_unit_variance(xa),
            normalize_unit_variance(xb),
            fps,
            window=window,
            band=band,
            channel_a=id_a,
            channel_b=id_b,
        )
    return out


@dataclass
class FourGaitSignalsReport:
    """Display-ready four-signal bundle plus raw and filtered R(0) matrices."""

    available: bool
    times_s: np.ndarray
    fps: float
    channel_ids: tuple[str, ...] = FOUR_GAIT_CHANNEL_IDS
    labels: tuple[str, ...] = FOUR_GAIT_SHORT_LABELS
    raw: dict[str, np.ndarray] = field(default_factory=dict)
    filtered: dict[str, np.ndarray] = field(default_factory=dict)
    correlation_raw: np.ndarray = field(
        default_factory=lambda: np.full((4, 4), np.nan, dtype=float)
    )
    correlation_filtered: np.ndarray = field(
        default_factory=lambda: np.full((4, 4), np.nan, dtype=float)
    )
    n_paired_raw: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 4), dtype=int)
    )
    n_paired_filtered: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 4), dtype=int)
    )
    warnings: list[str] = field(default_factory=list)
    interpretation: str = INTERPRETATION
    coordinate_system: str = ""
    angle_definition: str = (
        "Segment sagittal tilt from downward vertical (deg); "
        "0° ≈ hanging down, positive = distal more anterior. "
        "Thigh = hip→knee, shank = knee→ankle. Same definition on all four traces."
    )
    angle_units: str = FOUR_GAIT_ANGLE_UNITS
    angle_ylabel: str = FOUR_GAIT_ANGLE_YLABEL
    autocorr_raw: dict[str, AutocorrResult] = field(default_factory=dict)
    autocorr_filtered: dict[str, AutocorrResult] = field(default_factory=dict)
    xcorr_raw: dict[str, CrossCorrSeries] = field(default_factory=dict)
    xcorr_filtered: dict[str, CrossCorrSeries] = field(default_factory=dict)
    data_raw: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=float))
    data_filtered: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=float))
    max_lag_samples: int = 0
    gait_period_s: float | None = None
    gait_period_reason: str = ""
    event_stride_period_s: float | None = None
    event_stride_n: int = 0
    acf_event_note: str = ""
    gait_period_channel_ids: tuple[str, ...] = ()
    timing: GaitTimingResult | None = None
    spectrum_raw: dict[str, ChannelSpectrum] = field(default_factory=dict)
    spectrum_filtered: dict[str, ChannelSpectrum] = field(default_factory=dict)
    spectrum_peaks_raw: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    spectrum_peaks_filtered: dict[str, list[tuple[float, float]]] = field(
        default_factory=dict
    )
    band_power_raw: dict[str, float | None] = field(default_factory=dict)
    band_power_filtered: dict[str, float | None] = field(default_factory=dict)
    spectral_corr_raw: np.ndarray = field(
        default_factory=lambda: np.full((4, 4), np.nan, dtype=float)
    )
    spectral_corr_filtered: np.ndarray = field(
        default_factory=lambda: np.full((4, 4), np.nan, dtype=float)
    )
    n_spectral_bins_raw: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 4), dtype=int)
    )
    n_spectral_bins_filtered: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 4), dtype=int)
    )
    freq_compare_raw: dict[str, DominantFrequencyCompare] = field(default_factory=dict)
    freq_compare_filtered: dict[str, DominantFrequencyCompare] = field(
        default_factory=dict
    )
    coherence_raw: dict[str, CoherenceResult] = field(default_factory=dict)
    coherence_filtered: dict[str, CoherenceResult] = field(default_factory=dict)

    def series_for(self, mode: SignalMode) -> dict[str, np.ndarray]:
        return self.filtered if mode == "filtered" else self.raw

    def correlation_for(self, mode: SignalMode) -> np.ndarray:
        return self.correlation_filtered if mode == "filtered" else self.correlation_raw

    def n_paired_for(self, mode: SignalMode) -> np.ndarray:
        return self.n_paired_filtered if mode == "filtered" else self.n_paired_raw

    def autocorr_for(self, mode: SignalMode) -> dict[str, AutocorrResult]:
        return self.autocorr_filtered if mode == "filtered" else self.autocorr_raw

    def xcorr_for(self, mode: SignalMode) -> dict[str, CrossCorrSeries]:
        return self.xcorr_filtered if mode == "filtered" else self.xcorr_raw

    def data_for(self, mode: SignalMode) -> np.ndarray:
        return self.data_filtered if mode == "filtered" else self.data_raw

    def spectrum_for(self, mode: SignalMode) -> dict[str, ChannelSpectrum]:
        return self.spectrum_filtered if mode == "filtered" else self.spectrum_raw

    def spectrum_peaks_for(self, mode: SignalMode) -> dict[str, list[tuple[float, float]]]:
        return (
            self.spectrum_peaks_filtered if mode == "filtered" else self.spectrum_peaks_raw
        )

    def band_power_for(self, mode: SignalMode) -> dict[str, float | None]:
        return self.band_power_filtered if mode == "filtered" else self.band_power_raw

    def spectral_corr_for(self, mode: SignalMode) -> np.ndarray:
        return self.spectral_corr_filtered if mode == "filtered" else self.spectral_corr_raw

    def n_spectral_bins_for(self, mode: SignalMode) -> np.ndarray:
        return (
            self.n_spectral_bins_filtered if mode == "filtered" else self.n_spectral_bins_raw
        )

    def freq_compare_for(self, mode: SignalMode) -> dict[str, DominantFrequencyCompare]:
        return self.freq_compare_filtered if mode == "filtered" else self.freq_compare_raw

    def coherence_for(self, mode: SignalMode) -> dict[str, CoherenceResult]:
        return self.coherence_filtered if mode == "filtered" else self.coherence_raw


def pairwise_pearson(
    columns: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete Pearson C(0). Same overlap formula as C(τ).

    Non-constant series have r_ii = 1. Constant or short pairs stay NaN
    off-diagonal. The matrix is symmetric.
    """
    n = len(columns)
    corr = np.full((n, n), np.nan, dtype=float)
    counts = np.zeros((n, n), dtype=int)
    if n == 0:
        return corr, counts
    lengths = {int(np.asarray(col).size) for col in columns}
    if len(lengths) != 1:
        return corr, counts
    data = np.column_stack([np.asarray(col, dtype=float) for col in columns])
    if data.size == 0:
        return corr, counts
    for i in range(n):
        for j in range(n):
            r, count = pearson_overlap(data[:, i], data[:, j], min_paired=_MIN_PAIRED)
            counts[i, j] = count
            # Use the overlap Pearson as-is. Constant or short pairs stay NaN,
            # including on the diagonal — do not invent r = 1.
            corr[i, j] = r
    return corr, counts


def _stack_channels(series: dict[str, np.ndarray], n_times: int) -> np.ndarray:
    cols: list[np.ndarray] = []
    for cid in FOUR_GAIT_CHANNEL_IDS:
        col = np.asarray(series.get(cid), dtype=float)
        if col.size != n_times:
            col = np.full(n_times, np.nan, dtype=float)
        cols.append(col)
    if not cols or n_times <= 0:
        return np.zeros((0, 4), dtype=float)
    return np.column_stack(cols)


def _acf_on_values(series: ChannelSeries, values: np.ndarray, cfg: Any) -> AutocorrResult:
    y = np.asarray(values, dtype=float)
    clone = replace(
        series,
        processed=y,
        normalized=normalize_unit_variance(y),
        valid_mask=np.isfinite(y),
        missing_fraction=float(np.mean(~np.isfinite(y))) if y.size else 1.0,
    )
    return analyze_autocorrelation(clone, config=cfg)


def _xcorr_pairs(
    series: dict[str, np.ndarray],
    *,
    fps: float,
    max_lag: int,
) -> dict[str, CrossCorrSeries]:
    out: dict[str, CrossCorrSeries] = {}
    for name, left_id, right_id, _label in XCORR_PAIRS:
        left = np.asarray(series.get(left_id), dtype=float)
        right = np.asarray(series.get(right_id), dtype=float)
        xc = lagged_cross_correlation(left, right, max_lag=max_lag, fps=fps)
        xc.channel_a = left_id
        xc.channel_b = right_id
        out[name] = xc
    return out


def consensus_gait_period_s(
    results: list[AutocorrResult],
) -> tuple[float | None, str]:
    """Median of reliable ACF stride lags, or None when evidence is weak."""
    usable: list[float] = []
    withheld = 0
    for ac in results:
        if ac.gait_period_s is None or not np.isfinite(float(ac.gait_period_s)):
            withheld += 1
            continue
        if ac.insufficient_reason or ac.reliability_label == "insufficient":
            withheld += 1
            continue
        usable.append(float(ac.gait_period_s))
    if not usable:
        return None, (
            "Gait period withheld — no reliable ACF peak, or the clip is too "
            "short / poorly periodic."
        )
    med = float(np.median(np.asarray(usable, dtype=float)))
    if len(usable) >= 2:
        spread = float(max(usable) - min(usable))
        if med > 1e-9 and spread > _PERIOD_REL_TOL * med:
            return None, (
                "Gait period withheld — reliable channel periods disagree "
                f"({min(usable):.2f}–{max(usable):.2f} s)."
            )
    return med, ""


_EVENT_STRIDE_BAND_S = (0.50, 2.50)
_ACF_EVENT_REL_TOL = 0.25


def event_stride_period_s(cycles: Any | None) -> tuple[float | None, int, str]:
    """Median ipsilateral heel-strike interval. Independent of ACF."""
    if cycles is None:
        return None, 0, "No independent heel-strike result was supplied."
    events = list(getattr(cycles, "events", None) or [])
    if not events:
        return None, 0, "Heel-strike events were not available."
    intervals: list[float] = []
    for side in ("left", "right"):
        times: list[float] = []
        for event in events:
            etype = str(getattr(event, "event_type", "") or "")
            eside = str(getattr(event, "side", "") or "")
            if etype == f"{side}_heel_strike" or (
                eside == side and "heel_strike" in etype
            ):
                try:
                    times.append(float(event.time_s))
                except (TypeError, ValueError):
                    continue
        times.sort()
        for prev, cur in zip(times, times[1:]):
            dt = float(cur - prev)
            if _EVENT_STRIDE_BAND_S[0] <= dt <= _EVENT_STRIDE_BAND_S[1]:
                intervals.append(dt)
    if not intervals:
        return None, 0, "Heel-strike events did not yield a complete ipsilateral interval."
    return float(np.median(np.asarray(intervals, dtype=float))), len(intervals), ""


def compare_acf_and_event_periods(
    acf_period_s: float | None,
    event_period_s: float | None,
    *,
    n_event_intervals: int = 0,
) -> str:
    """Describe ACF vs heel-strike agreement without substituting either value."""
    acf_ok = acf_period_s is not None and np.isfinite(float(acf_period_s))
    ev_ok = event_period_s is not None and np.isfinite(float(event_period_s))
    if not acf_ok and ev_ok:
        return (
            f"Heel-strike cycles exist ({n_event_intervals} interval(s), "
            f"median {float(event_period_s):.2f} s), but ACF has no reliable "
            "stride peak. The event period is not used as an ACF measurement."
        )
    if acf_ok and not ev_ok:
        return (
            f"ACF period {float(acf_period_s):.2f} s has no independent "
            "heel-strike interval to cross-check."
        )
    if not acf_ok and not ev_ok:
        return "ACF period and heel-strike intervals are both unavailable."
    acf = float(acf_period_s)
    event = float(event_period_s)
    delta = abs(acf - event)
    rel = delta / event if event > 1e-9 else float("inf")
    if rel <= _ACF_EVENT_REL_TOL:
        return (
            f"ACF period {acf:.2f} s agrees with heel-strike intervals "
            f"({event:.2f} s; {n_event_intervals} interval(s)). "
            "This is an independent check — not a substituted ACF value."
        )
    return (
        f"ACF period {acf:.2f} s differs from heel-strike intervals "
        f"({event:.2f} s; {n_event_intervals} interval(s)). "
        "ACF is not replaced by the event period."
    )


def correlation_at_lag(
    report: FourGaitSignalsReport,
    mode: SignalMode,
    lag_samples: int,
) -> np.ndarray:
    """4×4 Pearson C(lag) at an integer sample offset. Uses recording FPS."""
    data = np.asarray(report.data_for(mode), dtype=float)
    if data.ndim != 2 or data.shape[1] != 4 or data.shape[0] < _MIN_PAIRED:
        return np.full((4, 4), np.nan, dtype=float)
    tau = int(lag_samples)
    if tau == 0:
        return np.asarray(report.correlation_for(mode), dtype=float)
    mat = lagged_matrix(data, tau, float(report.fps))
    corr = np.asarray(mat.correlation, dtype=float)
    if corr.shape != (4, 4):
        return np.full((4, 4), np.nan, dtype=float)
    return corr


def lag_seconds(report: FourGaitSignalsReport, lag_samples: int) -> float:
    return float(lag_samples) / max(float(report.fps), 1e-6)


def _laterality_warning(recording: GaitMotionRecording) -> str | None:
    """Flag a possible left/right landmark swap — not facing-camera image +X."""
    from stablewalk.analysis.asymmetry3d.laterality import assess_recording_laterality

    assessment = assess_recording_laterality(recording)
    if assessment.status in {"fail", "warn"}:
        return assessment.detail
    return None


def analyze_four_gait_signals(
    recording: GaitMotionRecording,
    *,
    sequence: Any | None = None,
    cycles: Any | None = None,
) -> FourGaitSignalsReport:
    """Build L/R thigh and shank angles with the existing spectral preprocessor.

    Filtered series use gap fill + Savitzky–Golay only (no linear detrend) so
    the displayed Y-axis stays in degrees.
    """
    cfg = default_config(detrend="none")
    bundle = build_spectral_series(
        recording,
        sequence=sequence,
        config=cfg,
        channel_ids=FOUR_GAIT_CHANNEL_IDS,
    )
    warnings = list(bundle.warnings)
    raw: dict[str, np.ndarray] = {}
    filtered: dict[str, np.ndarray] = {}
    n_ok = 0
    for cid in FOUR_GAIT_CHANNEL_IDS:
        series = bundle.get(cid)
        if series is None:
            raw[cid] = np.asarray([], dtype=float)
            filtered[cid] = np.asarray([], dtype=float)
            warnings.append(f"{CHANNEL_BY_ID[cid].label} was not extracted.")
            continue
        raw[cid] = np.asarray(series.raw, dtype=float)
        filtered[cid] = np.asarray(series.processed, dtype=float)
        warnings.extend(series.warnings)
        if int(np.isfinite(filtered[cid]).sum()) >= _MIN_PAIRED:
            n_ok += 1
    times = np.asarray(bundle.times_s, dtype=float)
    corr_raw, n_raw = pairwise_pearson([raw[cid] for cid in FOUR_GAIT_CHANNEL_IDS])
    corr_filt, n_filt = pairwise_pearson(
        [filtered[cid] for cid in FOUR_GAIT_CHANNEL_IDS]
    )
    if n_ok < 2:
        warnings.append(
            "Need at least two thigh/shank series with synchronized samples."
        )
    laterality = _laterality_warning(recording)
    if laterality:
        warnings.append(laterality)
    for cid in FOUR_GAIT_CHANNEL_IDS:
        prox, dist = four_gait_joints(cid)
        col = np.asarray(filtered.get(cid), dtype=float)
        if col.size and int(np.isfinite(col).sum()) < _MIN_PAIRED:
            warnings.append(
                f"{CHANNEL_BY_ID[cid].label} needs 3D {prox} and {dist} "
                f"with a usable segment length."
            )
    fps = float(bundle.fps)
    max_lag = int(round(cfg.max_acf_lag_s * max(fps, 1e-6)))
    if times.size:
        max_lag = int(max(0, min(max_lag, int(times.size) - 2)))
    acf_raw: dict[str, AutocorrResult] = {}
    acf_filt: dict[str, AutocorrResult] = {}
    for cid in FOUR_GAIT_CHANNEL_IDS:
        series = bundle.get(cid)
        if series is None:
            continue
        acf_filt[cid] = analyze_autocorrelation(series, config=cfg)
        acf_raw[cid] = _acf_on_values(series, raw[cid], cfg)
    xcorr_raw = _xcorr_pairs(raw, fps=fps, max_lag=max_lag)
    xcorr_filt = _xcorr_pairs(filtered, fps=fps, max_lag=max_lag)
    period, period_reason = consensus_gait_period_s(list(acf_filt.values()))
    gait_period_channel_ids = tuple(
        ac.channel_id
        for ac in acf_filt.values()
        if ac.gait_period_s is not None
        and np.isfinite(float(ac.gait_period_s))
        and not ac.insufficient_reason
        and ac.reliability_label != "insufficient"
    )
    event_period, event_n, event_reason = event_stride_period_s(cycles)
    acf_event_note = compare_acf_and_event_periods(
        period, event_period, n_event_intervals=event_n
    )
    if period_reason:
        warnings.append(period_reason)
    if cycles is not None and acf_event_note:
        warnings.append(acf_event_note)
    spec_cfg = default_config()
    spec_raw: dict[str, ChannelSpectrum] = {}
    spec_filt: dict[str, ChannelSpectrum] = {}
    peaks_raw: dict[str, list[tuple[float, float]]] = {}
    peaks_filt: dict[str, list[tuple[float, float]]] = {}
    power_raw: dict[str, float | None] = {}
    power_filt: dict[str, float | None] = {}
    n_finite_filt = [
        int(np.isfinite(np.asarray(filtered.get(cid), dtype=float)).sum())
        for cid in FOUR_GAIT_CHANNEL_IDS
    ]
    n_finite_raw = [
        int(np.isfinite(np.asarray(raw.get(cid), dtype=float)).sum())
        for cid in FOUR_GAIT_CHANNEL_IDS
    ]
    n_shared_filt = min((c for c in n_finite_filt if c > 0), default=0)
    n_shared_raw = min((c for c in n_finite_raw if c > 0), default=0)
    nperseg_f, noverlap_f, _seg_f = choose_psd_nperseg(n_shared_filt, fps)
    nperseg_r, noverlap_r, _seg_r = choose_psd_nperseg(n_shared_raw, fps)
    duration_s = float(times[-1] - times[0]) if times.size >= 2 else 0.0
    if duration_s + 1e-9 < 8.0:
        warnings.append(
            f"Clip is {duration_s:.2f} s — spectral reliability is limited "
            "and insufficient for a stable gait-frequency classification. "
            "PSD, entropy, energy, peaks, and shape r are shown as "
            "descriptive results. Dominant frequencies, if shown, are "
            "coarse. Coherence stays blank unless enough Welch segments "
            "exist. Values are not invented."
        )
    for cid in FOUR_GAIT_CHANNEL_IDS:
        series = bundle.get(cid)
        if series is None:
            continue
        spec_filt[cid] = _spectrum_on_values(
            series, filtered[cid], spec_cfg, nperseg=nperseg_f, noverlap=noverlap_f
        )
        spec_raw[cid] = _spectrum_on_values(
            series, raw[cid], spec_cfg, nperseg=nperseg_r, noverlap=noverlap_r
        )
        peaks_filt[cid] = detect_spectral_peaks(
            spec_filt[cid], band=spec_cfg.gait_band_hz
        )
        peaks_raw[cid] = detect_spectral_peaks(
            spec_raw[cid], band=spec_cfg.gait_band_hz
        )
        power_filt[cid] = spectral_band_power(
            spec_filt[cid], spec_cfg.gait_band_hz
        )
        power_raw[cid] = spectral_band_power(spec_raw[cid], spec_cfg.gait_band_hz)
    label_by_id = dict(zip(FOUR_GAIT_CHANNEL_IDS, FOUR_GAIT_SHORT_LABELS))
    for cid, spec in spec_filt.items():
        warnings.extend(_spectrum_quality_notes(spec, label_by_id.get(cid, cid)))
    spec_corr_raw, n_spec_raw, _grid_raw = spectral_shape_correlation(
        spec_raw, band=spec_cfg.gait_band_hz
    )
    spec_corr_filt, n_spec_filt, _grid_filt = spectral_shape_correlation(
        spec_filt, band=spec_cfg.gait_band_hz
    )
    freq_cmp_raw = compare_dominant_frequencies(spec_raw)
    freq_cmp_filt = compare_dominant_frequencies(spec_filt)
    coh_raw = _coherence_pairs(
        raw, fps=fps, window=spec_cfg.fft_window, band=spec_cfg.gait_band_hz
    )
    coh_filt = _coherence_pairs(
        filtered, fps=fps, window=spec_cfg.fft_window, band=spec_cfg.gait_band_hz
    )
    if spec_filt and all(
        not item.sufficient for item in coh_filt.values()
    ) and coh_filt:
        warnings.append(
            "Magnitude-squared coherence withheld — too few Welch segments "
            "for a supported estimate."
        )
    report = FourGaitSignalsReport(
        available=n_ok >= 1 and times.size >= 2,
        times_s=times,
        fps=fps,
        raw=raw,
        filtered=filtered,
        correlation_raw=corr_raw,
        correlation_filtered=corr_filt,
        n_paired_raw=n_raw,
        n_paired_filtered=n_filt,
        warnings=warnings,
        coordinate_system=str(bundle.coordinate_system or ""),
        autocorr_raw=acf_raw,
        autocorr_filtered=acf_filt,
        xcorr_raw=xcorr_raw,
        xcorr_filtered=xcorr_filt,
        data_raw=_stack_channels(raw, int(times.size)),
        data_filtered=_stack_channels(filtered, int(times.size)),
        max_lag_samples=max_lag,
        gait_period_s=period,
        gait_period_reason=period_reason,
        event_stride_period_s=event_period,
        event_stride_n=event_n,
        acf_event_note=acf_event_note,
        gait_period_channel_ids=gait_period_channel_ids,
        spectrum_raw=spec_raw,
        spectrum_filtered=spec_filt,
        spectrum_peaks_raw=peaks_raw,
        spectrum_peaks_filtered=peaks_filt,
        band_power_raw=power_raw,
        band_power_filtered=power_filt,
        spectral_corr_raw=spec_corr_raw,
        spectral_corr_filtered=spec_corr_filt,
        n_spectral_bins_raw=n_spec_raw,
        n_spectral_bins_filtered=n_spec_filt,
        freq_compare_raw=freq_cmp_raw,
        freq_compare_filtered=freq_cmp_filt,
        coherence_raw=coh_raw,
        coherence_filtered=coh_filt,
    )
    report.timing = compose_gait_timing(
        duration_s=duration_s,
        fps=fps,
        session_period_s=period,
        session_reason=period_reason,
        session_channel_ids=gait_period_channel_ids,
        event_stride_s=event_period,
        event_stride_n=event_n,
        acf_event_note=acf_event_note,
        cycles=cycles,
        stride_window_s=cfg.stride_lag_s,
        step_window_s=cfg.step_lag_s,
    )
    return report
