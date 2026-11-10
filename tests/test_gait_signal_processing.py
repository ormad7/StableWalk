"""Controlled synthetic-signal tests for correlation and spectral math.

These cases feed 1-D numpy arrays into the existing processors. They do not
analyze video, do not write into the GUI report cache, and do not replace
results computed from a loaded recording.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stablewalk.analysis.four_gait_signals import (
    FOUR_GAIT_CHANNEL_IDS,
    compare_dominant_frequencies,
    detect_spectral_peaks,
    pairwise_pearson,
    spectral_shape_correlation,
)
from stablewalk.analysis.spectral.autocorrelation import (
    _best_peak_in_band,
    analyze_autocorrelation,
    assign_step_and_stride_peaks,
    is_qualifying_local_maximum,
    normalized_autocorrelation,
)
from stablewalk.analysis.spectral.channels import CHANNEL_BY_ID
from stablewalk.analysis.spectral.config import (
    DEFAULT_GAIT_BAND_HZ,
    DEFAULT_STEP_LAG_S,
    DEFAULT_STRIDE_LAG_S,
)
from stablewalk.analysis.spectral.matrices import (
    lagged_cross_correlation,
    lagged_matrix,
    pearson_overlap,
)
from stablewalk.analysis.spectral.series import (
    ChannelSeries,
    interpolate_short_gaps,
    normalize_unit_variance,
)
from stablewalk.analysis.spectral.spectrum import (
    ChannelSpectrum,
    analyze_channel_spectrum,
    choose_psd_nperseg,
    rfft_periodogram,
    welch_psd,
)


def _sine(freq_hz: float, *, fps: float = 50.0, duration_s: float = 8.0, phase: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(duration_s * fps))
    times = np.arange(n, dtype=float) / fps
    values = np.sin(2.0 * math.pi * freq_hz * times + phase)
    return times, values


def _series(values: np.ndarray, *, fps: float, channel_id: str = "seg:left_thigh") -> ChannelSeries:
    y = np.asarray(values, dtype=float)
    times = np.arange(y.size, dtype=float) / fps
    return ChannelSeries(
        spec=CHANNEL_BY_ID[channel_id],
        times_s=times,
        raw=y.copy(),
        processed=y.copy(),
        valid_mask=np.isfinite(y),
        visibility=np.ones_like(y),
        missing_fraction=float(np.mean(~np.isfinite(y))) if y.size else 1.0,
        interpolated_fraction=0.0,
        fps=fps,
        coordinate_system="synthetic_signal",
        normalized=normalize_unit_variance(y),
    )


def _spectrum(values: np.ndarray, *, fps: float, channel_id: str) -> ChannelSpectrum:
    freq, psd, method = welch_psd(values, fps, nperseg=256, noverlap=128)
    fft_f, fft_p = rfft_periodogram(values, fps)
    peak_f = peak_p = None
    if freq.size and psd.size:
        lo, hi = DEFAULT_GAIT_BAND_HZ
        mask = (freq >= lo) & (freq <= hi) & (freq > 0) & np.isfinite(psd)
        if np.any(mask):
            i = int(np.nanargmax(psd[mask]))
            peak_f = float(freq[mask][i])
            peak_p = float(psd[mask][i])
    return ChannelSpectrum(
        channel_id=channel_id,
        freq_hz=freq,
        psd=psd,
        fft_freq_hz=fft_f,
        fft_power=fft_p,
        method=method,
        dominant_hz=peak_f,
        dominant_power=peak_p,
    )


# --- 1. Zero-lag correlation is symmetric ---------------------------------


def test_zero_lag_correlation_matrix_is_symmetric() -> None:
    _t, a = _sine(1.0)
    _t, b = _sine(1.0, phase=0.4)
    _t, c = _sine(1.5, phase=1.1)
    _t, d = _sine(2.0, phase=-0.3)
    data = np.column_stack([a, b, c, d])
    mat = lagged_matrix(data, 0, 50.0)
    assert mat.symmetric is True
    assert mat.tau_samples == 0
    corr, _n = pairwise_pearson([a, b, c, d])
    np.testing.assert_allclose(corr, corr.T, equal_nan=True, atol=1e-12)
    np.testing.assert_allclose(mat.correlation, mat.correlation.T, equal_nan=True, atol=1e-12)
    np.testing.assert_allclose(mat.correlation, corr, equal_nan=True, atol=1e-12)
    for i in range(4):
        assert abs(float(mat.correlation[i, i]) - 1.0) < 1e-12


def test_c0_uses_same_overlap_pearson_as_lagged_entries() -> None:
    fps = 50.0
    _t, a = _sine(1.0, fps=fps)
    _t, b = _sine(1.0, fps=fps, phase=0.6)
    a = a.copy()
    b = b.copy()
    a[::6] = np.nan
    b[1::8] = np.nan
    data = np.column_stack([a, b])
    mat = lagged_matrix(data, 0, fps)
    r, n = pearson_overlap(a, b)
    assert n >= 4
    assert abs(float(mat.correlation[0, 1]) - r) < 1e-12
    assert abs(float(mat.correlation[1, 0]) - r) < 1e-12
    assert abs(float(mat.correlation[0, 0]) - 1.0) < 1e-12
    assert abs(float(mat.correlation[1, 1]) - 1.0) < 1e-12
    assert mat.symmetric is True


def test_nonzero_lag_matrix_is_not_assumed_symmetric() -> None:
    _t, a = _sine(1.0)
    b = np.roll(a, 8)
    data = np.column_stack([a, b, a + 0.1, b - 0.1])
    mat = lagged_matrix(data, 8, 50.0)
    assert mat.symmetric is False
    assert mat.tau_s == pytest_approx_lag(8, 50.0)


def pytest_approx_lag(samples: int, fps: float) -> float:
    return float(samples) / fps


# --- 2. Diagonal is 1 for valid non-constant series -----------------------


def test_correlation_diagonal_is_one_for_nonconstant_signals() -> None:
    _t, a = _sine(1.0)
    _t, b = _sine(1.2, phase=0.7)
    corr, n = pairwise_pearson([a, b, a + 0.25 * b])
    assert n[0, 0] == a.size
    for i in range(3):
        assert abs(float(corr[i, i]) - 1.0) < 1e-12


def test_constant_series_keeps_identity_and_nans_off_diagonal() -> None:
    x = np.ones(80)
    y = np.linspace(-1.0, 1.0, 80)
    corr, n = pairwise_pearson([x, y])
    assert n[0, 0] == 80
    assert not np.isfinite(corr[0, 0])
    assert abs(float(corr[1, 1]) - 1.0) < 1e-12
    assert not np.isfinite(corr[0, 1])
    assert not np.isfinite(corr[1, 0])


# --- 3. Autocorrelation recovers a known period ---------------------------


def test_autocorrelation_recovers_one_second_period() -> None:
    fps = 50.0
    _t, y = _sine(1.0, fps=fps, duration_s=8.0)
    max_lag = int(round(2.5 * fps))
    lags, rho = normalized_autocorrelation(y, max_lag=max_lag)
    lags_s = lags.astype(float) / fps
    assert abs(float(rho[0]) - 1.0) < 1e-9
    peak = _best_peak_in_band(lags_s, rho, DEFAULT_STRIDE_LAG_S)
    assert peak is not None
    assert abs(float(peak.lag_s) - 1.0) < 0.04

    ac = analyze_autocorrelation(_series(y, fps=fps))
    assert ac.gait_period_s is not None
    assert abs(float(ac.gait_period_s) - 1.0) < 0.08
    assert ac.insufficient_reason is None
    assert ac.cadence_assumes_two_steps_per_stride is True
    assert ac.cadence_steps_per_min is not None
    assert abs(float(ac.cadence_steps_per_min) - 120.0) < 8.0


def test_short_sine_withholds_acf_period() -> None:
    fps = 50.0
    _t, y = _sine(1.0, fps=fps, duration_s=1.1)
    ac = analyze_autocorrelation(_series(y, fps=fps))
    assert ac.gait_period_s is None
    assert ac.insufficient_reason is not None


def test_slow_stride_peak_is_inside_plausible_window() -> None:
    fps = 50.0
    _t, y = _sine(0.50, fps=fps, duration_s=8.0)
    lags, rho = normalized_autocorrelation(y, max_lag=int(round(2.5 * fps)))
    lags_s = lags.astype(float) / fps
    peak = _best_peak_in_band(lags_s, rho, DEFAULT_STRIDE_LAG_S)
    assert peak is not None
    assert abs(float(peak.lag_s) - 2.0) < 0.08
    ac = analyze_autocorrelation(_series(y, fps=fps))
    assert ac.gait_period_s is not None
    assert abs(float(ac.gait_period_s) - 2.0) < 0.12
    assert ac.insufficient_reason is None
    assert ac.reliability_label in {"low", "moderate"}


def test_demo_length_slow_walk_reports_acf_period_when_cycles_exist() -> None:
    fps = 25.0
    _t, y = _sine(0.50, fps=fps, duration_s=4.76)
    ac = analyze_autocorrelation(_series(y, fps=fps))
    assert abs(float(ac.n_samples) / fps - 4.76) < 0.08
    assert ac.gait_period_s is not None
    assert abs(float(ac.gait_period_s) - 2.0) < 0.16
    assert ac.n_stride_cycles is not None
    assert float(ac.n_stride_cycles) >= 2.0


def test_fundamental_plus_harmonic_keeps_stride_and_step_independent() -> None:
    """Step harmonic must be strong enough for ρ(0.5) > 0; stride stays at 1 s.

    For x = cos(2πt) + B cos(4πt), ρ(0.5) is positive only when B² > 1.
    A weak 2 Hz add-on is a trough at 0.5 s, not a step period.
    """
    fps = 50.0
    duration_s = 10.0
    t = np.arange(0.0, duration_s, 1.0 / fps)
    y = np.cos(2.0 * math.pi * t) + 1.25 * np.cos(4.0 * math.pi * t)
    ac = analyze_autocorrelation(_series(y, fps=fps))
    assert ac.gait_period_s is not None
    assert abs(float(ac.gait_period_s) - 1.0) < 0.08
    assert ac.step_period_s is not None
    assert abs(float(ac.step_period_s) - 0.5) < 0.08
    assert ac.cadence_assumes_two_steps_per_stride is False
    assert ac.cadence_steps_per_min is not None
    assert abs(float(ac.cadence_steps_per_min) - 120.0) < 10.0
    assert ac.n_stride_cycles is not None
    assert ac.n_stride_cycles == pytest.approx(
        duration_s / float(ac.gait_period_s), rel=0.03
    )


def _periodic_pulses(
    times: np.ndarray,
    period_s: float,
    locations: tuple[float, ...],
    amps: tuple[float, ...],
    width_s: float = 0.05,
) -> np.ndarray:
    y = np.zeros_like(times, dtype=float)
    for loc, amp in zip(locations, amps):
        phase = np.mod(times - loc, period_s)
        phase = np.minimum(phase, period_s - phase)
        y += amp * np.exp(-0.5 * (phase / width_s) ** 2)
    return y


def test_overlap_lobe_is_not_stride_when_exclusive_peak_exists() -> None:
    fps = 50.0
    duration_s = 12.8
    stride_s = 1.60
    t = np.arange(0.0, duration_s, 1.0 / fps)
    y = _periodic_pulses(t, stride_s, (0.0, 0.64), (1.0, 0.85))
    ac = analyze_autocorrelation(_series(y, fps=fps))
    assert ac.stride_period_s is not None
    assert float(ac.stride_period_s) > 0.90
    assert abs(float(ac.stride_period_s) - stride_s) < 0.12
    assert ac.step_period_s is not None
    assert float(ac.step_period_s) <= 0.90
    assert ac.n_stride_cycles is not None
    assert ac.n_stride_cycles == pytest.approx(
        duration_s / float(ac.stride_period_s), rel=0.03
    )


def test_asymmetric_duty_does_not_force_step_half_of_stride() -> None:
    fps = 50.0
    duration_s = 16.0
    stride_s = 1.60
    on_s = 0.54
    t = np.arange(0.0, duration_s, 1.0 / fps)
    y = np.where(np.mod(t, stride_s) < on_s, 1.0, -0.4).astype(float)
    ac = analyze_autocorrelation(_series(y, fps=fps))
    assert ac.stride_period_s is not None
    assert abs(float(ac.stride_period_s) - stride_s) < 0.12
    if ac.step_period_s is not None:
        assert ac.cadence_assumes_two_steps_per_stride is False
        assert ac.cadence_steps_per_min == 60.0 / float(ac.step_period_s)
        assert abs(float(ac.stride_period_s) - 2.0 * float(ac.step_period_s)) > 0.08


def test_assign_prefers_exclusive_stride_over_stronger_overlap_lobe() -> None:
    lags_s = np.linspace(0.0, 2.20, 221)
    rho = np.zeros_like(lags_s)
    rho[0] = 1.0
    for lag, val in ((0.542, 0.348), (0.667, 0.307), (1.001, 0.172), (1.543, 0.140)):
        i = int(np.argmin(np.abs(lags_s - lag)))
        rho[i] = val
        if i > 0:
            rho[i - 1] = val * 0.55
        if i + 1 < rho.size:
            rho[i + 1] = val * 0.45
    stride, step, note = assign_step_and_stride_peaks(
        lags_s,
        rho,
        stride_lag_s=DEFAULT_STRIDE_LAG_S,
        step_lag_s=DEFAULT_STEP_LAG_S,
    )
    assert step is not None
    assert abs(float(step.lag_s) - 0.542) < 0.02
    assert stride is not None
    assert abs(float(stride.lag_s) - 1.001) < 0.02
    assert "exclusive" in note.lower()


# --- 4. Cross-correlation recovers a known delay --------------------------


def test_cross_correlation_recovers_known_delay() -> None:
    fps = 50.0
    delay_s = 0.20
    shift = int(round(delay_s * fps))
    _t, x = _sine(1.0, fps=fps, duration_s=8.0)
    y = np.roll(x, shift)
    xc = lagged_cross_correlation(x, y, max_lag=int(round(1.0 * fps)), fps=fps)
    assert xc.peak_lag_s is not None
    assert abs(float(xc.peak_lag_s) - delay_s) < 1.0 / fps + 1e-9
    assert xc.peak_value is not None
    assert float(xc.peak_value) > 0.95
    dt = float(xc.lags_s[1] - xc.lags_s[0])
    assert abs(dt - 1.0 / fps) < 1e-12


def test_antiphase_xcorr_peaks_at_half_period_not_zero() -> None:
    fps = 50.0
    _t, left = _sine(1.0, fps=fps, duration_s=8.0)
    right = -left
    xc = lagged_cross_correlation(left, right, max_lag=int(round(1.2 * fps)), fps=fps)
    assert xc.peak_lag_s is not None
    assert abs(float(xc.peak_lag_s)) > 0.40
    assert abs(abs(float(xc.peak_lag_s)) - 0.5) < 1.0 / fps + 1e-9
    assert xc.peak_value is not None
    assert float(xc.peak_value) > 0.95
    # Simultaneous r is strongly negative; that is antiphase, not a failure.
    r0, _n = pearson_overlap(left, right)
    assert r0 < -0.95
    zero = int(np.argmin(np.abs(xc.lags_s)))
    assert float(xc.rho[zero]) < 0.0


def test_intra_limb_xcorr_recovers_thigh_shank_delay() -> None:
    fps = 50.0
    delay_s = 0.10
    shift = int(round(delay_s * fps))
    _t, thigh = _sine(1.0, fps=fps, duration_s=8.0)
    shank = np.roll(thigh, shift)
    xc = lagged_cross_correlation(thigh, shank, max_lag=int(round(1.0 * fps)), fps=fps)
    assert xc.peak_lag_s is not None
    assert abs(float(xc.peak_lag_s) - delay_s) < 1.0 / fps + 1e-9
    assert xc.peak_value is not None
    assert float(xc.peak_value) > 0.95


# --- 5. FFT / Welch recover a known frequency -----------------------------


def test_rfft_and_welch_recover_known_frequency() -> None:
    fps = 50.0
    freq_hz = 1.25
    _t, y = _sine(freq_hz, fps=fps, duration_s=8.0)
    fft_f, fft_p = rfft_periodogram(y, fps)
    welch_f, welch_p, method = welch_psd(y, fps, nperseg=256, noverlap=128)
    assert fft_f.size > 8
    df = float(np.median(np.diff(fft_f)))
    assert abs(df - fps / y.size) < 1e-9
    lo, hi = DEFAULT_GAIT_BAND_HZ
    mask = (fft_f >= lo) & (fft_f <= hi)
    peak = float(fft_f[mask][int(np.argmax(fft_p[mask]))])
    assert abs(peak - freq_hz) < 0.08
    wmask = (welch_f >= lo) & (welch_f <= hi)
    wpeak = float(welch_f[wmask][int(np.argmax(welch_p[wmask]))])
    assert abs(wpeak - freq_hz) < 0.12
    assert method in {"scipy.signal.welch", "numpy_welch"}


def test_welch_nperseg_grows_with_long_recordings() -> None:
    short_n, _ov, short_seg = choose_psd_nperseg(143, 30.0)
    long_n, _ov2, long_seg = choose_psd_nperseg(1000, 50.0)
    assert short_n <= 143
    assert long_seg >= 3
    assert long_n > short_n
    assert 50.0 / long_n <= 0.20


def test_spectral_peaks_are_local_maxima_not_argmax_fallback() -> None:
    freq = np.linspace(0.4, 4.0, 20)
    spec = ChannelSpectrum(
        channel_id="seg:left_thigh",
        freq_hz=freq,
        psd=np.linspace(1.0, 0.1, 20),
        fft_freq_hz=freq,
        fft_power=np.linspace(1.0, 0.1, 20),
        method="synthetic",
        dominant_hz=0.4,
        dominant_power=1.0,
    )
    assert detect_spectral_peaks(spec) == []


# --- 6. Spectral-shape correlation is consistent --------------------------


def test_spectral_shape_correlation_is_symmetric_with_unit_diagonal() -> None:
    fps = 50.0
    _t, a = _sine(1.0, fps=fps, duration_s=8.0)
    _t, b = _sine(1.0, fps=fps, duration_s=8.0, phase=1.3)
    _t, c = _sine(2.0, fps=fps, duration_s=8.0)
    _t, d = _sine(2.0, fps=fps, duration_s=8.0, phase=0.5)
    spectra = {
        FOUR_GAIT_CHANNEL_IDS[0]: _spectrum(a, fps=fps, channel_id=FOUR_GAIT_CHANNEL_IDS[0]),
        FOUR_GAIT_CHANNEL_IDS[1]: _spectrum(b, fps=fps, channel_id=FOUR_GAIT_CHANNEL_IDS[1]),
        FOUR_GAIT_CHANNEL_IDS[2]: _spectrum(c, fps=fps, channel_id=FOUR_GAIT_CHANNEL_IDS[2]),
        FOUR_GAIT_CHANNEL_IDS[3]: _spectrum(d, fps=fps, channel_id=FOUR_GAIT_CHANNEL_IDS[3]),
    }
    corr, counts, grid = spectral_shape_correlation(spectra)
    assert corr.shape == (4, 4)
    np.testing.assert_allclose(corr, corr.T, equal_nan=True, atol=1e-10)
    for i in range(4):
        assert abs(float(corr[i, i]) - 1.0) < 1e-6
        assert int(counts[i, i]) >= 4
    assert grid.size >= 4
    # Same-frequency pair is more similar than 1 Hz vs 2 Hz.
    assert float(corr[0, 1]) > 0.85
    assert float(corr[0, 2]) < float(corr[0, 1])


def test_identical_spectra_have_near_one_shape_r() -> None:
    fps = 50.0
    _t, y = _sine(1.0, fps=fps, duration_s=8.0)
    spec = _spectrum(y, fps=fps, channel_id=FOUR_GAIT_CHANNEL_IDS[0])
    spectra = {cid: spec for cid in FOUR_GAIT_CHANNEL_IDS}
    corr, _n, _g = spectral_shape_correlation(spectra)
    np.testing.assert_allclose(corr, np.ones((4, 4)), atol=1e-9)


def test_antiphase_sines_have_high_shape_r_and_negative_time_r() -> None:
    fps = 50.0
    _t, a = _sine(1.0, fps=fps, duration_s=8.0, phase=0.0)
    _t, b = _sine(1.0, fps=fps, duration_s=8.0, phase=math.pi)
    time_r, _n = pairwise_pearson([a, b])
    assert float(time_r[0, 1]) < -0.95
    spectra = {
        FOUR_GAIT_CHANNEL_IDS[0]: _spectrum(a, fps=fps, channel_id=FOUR_GAIT_CHANNEL_IDS[0]),
        FOUR_GAIT_CHANNEL_IDS[1]: _spectrum(b, fps=fps, channel_id=FOUR_GAIT_CHANNEL_IDS[1]),
        FOUR_GAIT_CHANNEL_IDS[2]: _spectrum(a, fps=fps, channel_id=FOUR_GAIT_CHANNEL_IDS[2]),
        FOUR_GAIT_CHANNEL_IDS[3]: _spectrum(b, fps=fps, channel_id=FOUR_GAIT_CHANNEL_IDS[3]),
    }
    corr, _counts, _grid = spectral_shape_correlation(spectra)
    assert float(corr[0, 1]) > 0.90
    assert abs(float(corr[0, 1]) - float(time_r[0, 1])) > 1.5


def test_spectrum_quality_is_not_fft_bin_confidence() -> None:
    fps = 25.0
    _t, y = _sine(1.0, fps=fps, duration_s=9.0)
    spec = analyze_channel_spectrum(_series(y, fps=fps))
    assert spec.resolution_hz is not None
    assert float(spec.resolution_hz) > 0.20
    assert spec.quality_status == "computed"
    assert spec.dominant_hz is not None
    assert abs(float(spec.dominant_hz) - 1.0) < 0.25
    joined = " ".join(spec.notes).lower()
    assert "limited" not in joined
    assert "confidence" not in joined


def test_dominant_frequency_compare_reports_known_delta() -> None:
    fps = 50.0
    _t, a = _sine(1.0, fps=fps, duration_s=8.0)
    _t, b = _sine(1.5, fps=fps, duration_s=8.0)
    spectra = {
        FOUR_GAIT_CHANNEL_IDS[0]: _spectrum(a, fps=fps, channel_id=FOUR_GAIT_CHANNEL_IDS[0]),
        FOUR_GAIT_CHANNEL_IDS[1]: _spectrum(b, fps=fps, channel_id=FOUR_GAIT_CHANNEL_IDS[1]),
        FOUR_GAIT_CHANNEL_IDS[2]: _spectrum(a, fps=fps, channel_id=FOUR_GAIT_CHANNEL_IDS[2]),
        FOUR_GAIT_CHANNEL_IDS[3]: _spectrum(b, fps=fps, channel_id=FOUR_GAIT_CHANNEL_IDS[3]),
    }
    compares = compare_dominant_frequencies(spectra)
    thigh = compares["thigh_lr"]
    assert thigh.f_a_hz is not None and thigh.f_b_hz is not None
    assert abs(float(thigh.f_a_hz) - 1.0) < 0.12
    assert abs(float(thigh.f_b_hz) - 1.5) < 0.12
    assert thigh.delta_hz is not None
    assert abs(float(thigh.delta_hz) - 0.5) < 0.20
    assert any("not proof of symmetric" in n.lower() for n in thigh.notes)


# --- 7. Frequency and time units ------------------------------------------


def test_frequency_axis_and_lag_timestamps_use_recording_fps() -> None:
    fps = 40.0
    n = 400
    times = np.arange(n, dtype=float) / fps
    y = np.sin(2.0 * math.pi * 1.0 * times)
    freq, _p = rfft_periodogram(y, fps)
    assert abs(float(freq[0])) < 1e-12
    assert abs(float(freq[-1]) - fps / 2.0) < 1e-9
    assert np.all(np.diff(freq) > 0)
    mat = lagged_matrix(np.column_stack([y, np.roll(y, 5)]), 5, fps)
    assert abs(mat.tau_s - 5.0 / fps) < 1e-12
    xc = lagged_cross_correlation(y, np.roll(y, 5), max_lag=20, fps=fps)
    assert abs(float(xc.lags_s[1] - xc.lags_s[0]) - 1.0 / fps) < 1e-12


# --- 8. Missing data and low-confidence handling --------------------------


def test_pairwise_pearson_ignores_nan_pairs() -> None:
    x = np.linspace(-1.0, 1.0, 60)
    y = x.copy()
    drop = np.zeros(60, dtype=bool)
    drop[::5] = True
    x[drop] = np.nan
    y[drop] = np.nan
    corr, n = pairwise_pearson([x, y])
    expected = int(np.sum(~drop))
    assert n[0, 1] == expected
    assert expected < 60
    assert abs(float(corr[0, 1]) - 1.0) < 1e-9

    staggered_x = np.linspace(-1.0, 1.0, 60)
    staggered_y = staggered_x.copy()
    staggered_y[::5] = np.nan
    staggered_x[1::7] = np.nan
    corr_s, n_s = pairwise_pearson([staggered_x, staggered_y])
    assert n_s[0, 1] == int(
        np.sum(np.isfinite(staggered_x) & np.isfinite(staggered_y))
    )
    assert n_s[0, 1] < 60
    assert np.isfinite(corr_s[0, 1])
    # Overlap Pearson uses σ from the shared finite window, so identical
    # overlapping samples have r = 1 even when holes are staggered.
    assert abs(float(corr_s[0, 1]) - 1.0) < 1e-12


def test_too_few_paired_samples_are_nan() -> None:
    x = np.array([1.0, np.nan, 2.0, np.nan, 3.0, np.nan])
    y = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
    corr, n = pairwise_pearson([x, y])
    assert n[0, 1] < 4
    assert not np.isfinite(corr[0, 1])


def test_short_gaps_fill_long_gaps_stay_nan() -> None:
    y = np.array([0.0, 1.0, np.nan, 3.0, 4.0, np.nan, np.nan, np.nan, 8.0], dtype=float)
    filled, mask = interpolate_short_gaps(y, max_gap_samples=1)
    assert mask[2]
    assert abs(float(filled[2]) - 2.0) < 1e-9
    assert not mask[5]
    assert not np.isfinite(filled[5])
    assert not np.isfinite(filled[6])
    assert not np.isfinite(filled[7])


def test_high_missing_fraction_limits_acf() -> None:
    fps = 50.0
    _t, y = _sine(1.0, fps=fps, duration_s=8.0)
    y[::2] = np.nan
    series = _series(y, fps=fps)
    assert series.missing_fraction > 0.45
    ac = analyze_autocorrelation(series)
    notes = " ".join(ac.notes).lower()
    limited = (
        ac.gait_period_s is None
        or "missing" in notes
        or ac.reliability_label == "insufficient"
        or ac.insufficient_reason is not None
    )
    assert limited


def test_acf_peaks_are_qualifying_local_maxima_not_sampled_points() -> None:
    fps = 50.0
    t = np.arange(0.0, 10.0, 1.0 / fps)
    y = np.cos(2.0 * math.pi * t) + 1.25 * np.cos(4.0 * math.pi * t)
    ac = analyze_autocorrelation(_series(y, fps=fps))
    assert ac.peaks
    sampled = set(np.linspace(1, ac.rho.size - 2, 12, dtype=int))
    peak_idx = {int(p.lag_samples) for p in ac.peaks}
    assert not sampled <= peak_idx
    for peak in ac.peaks:
        idx = int(peak.lag_samples)
        assert peak.is_local_maximum
        assert is_qualifying_local_maximum(ac.rho, idx)
        assert float(ac.rho[idx]) == pytest.approx(float(peak.value))
        if idx > 0:
            assert float(ac.rho[idx]) >= float(ac.rho[idx - 1])
        if idx + 1 < ac.rho.size:
            assert float(ac.rho[idx]) > float(ac.rho[idx + 1])
    assert ac.source_signal
    assert ac.time_window_s is not None
    assert ac.preprocessing
    assert "local maximum" in ac.peak_selection.lower()


def test_aperiodic_noise_is_unavailable_or_low_confidence() -> None:
    fps = 50.0
    rng = np.random.default_rng(0)
    y = rng.normal(size=int(8.0 * fps))
    ac = analyze_autocorrelation(_series(y, fps=fps))
    assert ac.reliability_label in {"insufficient", "low"}
    if ac.reliability_label == "insufficient":
        assert ac.gait_period_s is None or ac.insufficient_reason
    else:
        assert ac.reliability < 0.65
    if ac.gait_period_s is not None:
        for peak in ac.peaks:
            assert is_qualifying_local_maximum(ac.rho, int(peak.lag_samples))


def test_drifting_period_is_not_forced_to_a_high_confidence_stride() -> None:
    fps = 50.0
    duration_s = 8.0
    t = np.arange(0.0, duration_s, 1.0 / fps)
    inst_hz = 0.45 + 2.2 * (t / duration_s)
    phase = 2.0 * math.pi * np.cumsum(inst_hz) / fps
    y = np.sin(phase)
    ac = analyze_autocorrelation(_series(y, fps=fps))
    assert ac.reliability_label in {"insufficient", "low"}
    if ac.gait_period_s is not None:
        assert ac.reliability < 0.65


def test_periodic_sine_keeps_existing_high_periodicity_behavior() -> None:
    fps = 50.0
    ac = analyze_autocorrelation(_series(_sine(1.0, fps=fps, duration_s=8.0)[1], fps=fps))
    assert ac.gait_period_s is not None
    assert abs(float(ac.gait_period_s) - 1.0) < 0.08
    assert ac.insufficient_reason is None


# --- 9 / 10 live in test_gait_correlation_section.py (cache + GUI wiring) -
