"""Unit tests for spectral / autocorrelation analysis (Erik requirement 2)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from stablewalk.analysis.spectral.autocorrelation import (
    analyze_autocorrelation,
    normalized_autocorrelation,
    _best_peak_in_band,
)
from stablewalk.analysis.spectral.config import SpectralAnalysisConfig
from stablewalk.analysis.spectral.eigen import NonsymmetricEigError, eigh_symmetric
from stablewalk.analysis.spectral.export import export_spectral_csv, export_spectral_json
from stablewalk.analysis.spectral.matrices import (
    build_matrix_bundle,
    lagged_cross_correlation,
    lagged_matrix,
)
from stablewalk.analysis.spectral.report import analyze_spectral
from stablewalk.analysis.spectral.series import (
    build_spectral_series,
    interpolate_short_gaps,
    smooth_preserve_gait,
)
from stablewalk.analysis.spectral.spectrum import rfft_periodogram, welch_psd, welch_psd_numpy
from stablewalk.models.gait_motion import (
    DofSample,
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.models.joint_registry import DOF_IDS, JOINT_IDS, JOINT_PARENTS, ROOT_JOINT_ID


def _sine_recording(
    *,
    fps: float = 50.0,
    duration_s: float = 8.0,
    freq_hz: float = 1.0,
    noise: float = 0.0,
    drop_every: int | None = None,
    irregular: bool = False,
    seed: int = 0,
) -> GaitMotionRecording:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fps))
    snaps: list[SkeletonSnapshot] = []
    for i in range(n):
        t = i / fps
        if irregular:
            val = float(rng.normal())
        else:
            val = math.sin(2 * math.pi * freq_hz * t)
            if noise:
                val += float(rng.normal() * noise)
        if drop_every and i % drop_every == 0:
            detected = False
            vis = 0.1
            angle = None
        else:
            detected = True
            vis = 0.95
            angle = val * 20.0 + 40.0
        joints = {}
        for jid in (*JOINT_IDS, ROOT_JOINT_ID):
            phase = 0.0
            if jid.startswith("right_"):
                phase = math.pi
            y = 0.2 * math.sin(2 * math.pi * freq_hz * t + phase)
            if jid in ("left_heel", "right_heel", "left_toe", "right_toe"):
                y -= 0.4
            joints[jid] = JointSample(
                joint_id=jid,
                position=Vec3(0.05 if "right" in jid else -0.05, y, 0.0),
                parent_id=JOINT_PARENTS.get(jid),
            )
        dofs = {}
        for dof in DOF_IDS:
            a = angle
            if a is not None and "right_" in dof:
                a = -a
            dofs[dof] = DofSample(dof_id=dof, angle_deg=a)
        snaps.append(
            SkeletonSnapshot(
                frame_index=i,
                time_s=t,
                joints=joints if detected else {},
                dofs=dofs,
                metadata={
                    "detected": detected,
                    "landmark_visibility": {
                        jid: vis for jid in (*JOINT_IDS, ROOT_JOINT_ID)
                    },
                },
            )
        )
    return GaitMotionRecording(
        source="synthetic",
        fps=fps,
        snapshots=snaps,
        source_kind="mock",
        coordinate_system="sw_canonical_y_up",
    )


def test_normalized_acf_matches_numpy_correlate() -> None:
    fps = 50.0
    t = np.arange(0, 6.0, 1.0 / fps)
    x = np.sin(2 * math.pi * 1.25 * t)
    x = x - x.mean()
    lags, rho = normalized_autocorrelation(x, max_lag=80)
    full = np.correlate(x, x, mode="full")
    mid = x.size - 1
    expected = full[mid : mid + 81] / np.dot(x, x)
    np.testing.assert_allclose(rho, expected, atol=1e-10)
    lag_period = int(round(0.8 * fps))
    # Finite-length biased ACF of a sine is a tapered cosine; the period lag
    # is a local maximum, not exactly 1.
    assert rho[lag_period] > rho[lag_period - 2]
    assert rho[lag_period] > rho[lag_period + 2]
    assert rho[lag_period] > 0.8


def test_sine_spectrum_dominant_frequency() -> None:
    fps = 50.0
    t = np.arange(0, 8.0, 1.0 / fps)
    x = np.sin(2 * math.pi * 1.0 * t)
    freq, pxx = rfft_periodogram(x, fps)
    band = (freq >= 0.4) & (freq <= 4.0)
    f0 = float(freq[band][np.argmax(pxx[band])])
    bin_hz = fps / x.size
    assert abs(f0 - 1.0) <= bin_hz + 1e-9


def test_numpy_welch_matches_scipy_when_available() -> None:
    rng = np.random.default_rng(1)
    fps = 40.0
    x = np.sin(2 * math.pi * 1.2 * np.arange(400) / fps) + 0.05 * rng.normal(size=400)
    nperseg, noverlap = 128, 64
    f_np, p_np = welch_psd_numpy(x, fps, nperseg=nperseg, noverlap=noverlap)
    try:
        from scipy.signal import welch
    except Exception:
        pytest.skip("scipy not installed")
    f_sp, p_sp = welch(x, fs=fps, window="hann", nperseg=nperseg, noverlap=noverlap, scaling="density")
    np.testing.assert_allclose(f_np, f_sp, atol=1e-10)
    np.testing.assert_allclose(p_np, p_sp, rtol=1e-5, atol=1e-8)
    _f, _p, method = welch_psd(x, fps, nperseg=nperseg, noverlap=noverlap)
    assert method in {"scipy.signal.welch", "numpy_welch"}


def test_short_gaps_interpolated_long_gaps_not_extrapolated() -> None:
    y = np.array([0.0, 1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0, np.nan])
    filled, mask = interpolate_short_gaps(y, max_gap_samples=1)
    assert filled[2] == pytest.approx(2.0)
    assert mask[2]
    assert np.isnan(filled[4])  # 4-sample interior gap remains
    assert np.isnan(filled[-1])  # trailing NaN not extrapolated
    assert np.isnan(interpolate_short_gaps(np.array([np.nan, 1.0, 2.0]), max_gap_samples=5)[0][0])


def test_smoothing_does_not_shift_1hz_peak() -> None:
    fps = 50.0
    t = np.arange(0, 6.0, 1 / fps)
    x = np.sin(2 * math.pi * 1.0 * t)
    y = smooth_preserve_gait(x, window_samples=7, polyorder=3)
    freq, pxx = rfft_periodogram(y, fps)
    band = (freq >= 0.4) & (freq <= 4.0)
    f0 = float(freq[band][np.argmax(pxx[band])])
    assert abs(f0 - 1.0) <= fps / t.size + 1e-9


def test_acf_stride_peak_rejects_negative_rho() -> None:
    lags = np.linspace(0.0, 2.5, 126)
    rho = np.full(lags.shape, -0.25)
    rho[0] = 1.0
    idx = int(np.argmin(np.abs(lags - 1.04)))
    rho[idx - 1] = -0.22
    rho[idx] = -0.08
    rho[idx + 1] = -0.22
    assert _best_peak_in_band(lags, rho, (0.60, 1.80)) is None
    rho[idx] = 0.42
    peak = _best_peak_in_band(lags, rho, (0.60, 1.80))
    assert peak is not None
    assert peak.value == pytest.approx(0.42)
    assert abs(peak.lag_s - 1.04) <= 0.04


def test_recording_stride_from_sine_knees() -> None:
    rec = _sine_recording(freq_hz=1.0, duration_s=8.0, fps=50.0)
    report = analyze_spectral(rec)
    assert report.available
    ac = report.primary_autocorr()
    assert ac is not None
    assert ac.stride_period_s is not None
    assert abs(ac.stride_period_s - 1.0) <= 0.06
    spec = report.primary_spectrum()
    assert spec is not None
    assert spec.dominant_hz is not None
    assert abs(spec.dominant_hz - 1.0) <= 0.15
    assert ac.reliability_label in {"low", "moderate"}


def test_insufficient_clip_warns_and_skips_stride() -> None:
    rec = _sine_recording(duration_s=0.6, fps=30.0)
    cfg = SpectralAnalysisConfig(min_duration_s=2.0, min_samples=64)
    report = analyze_spectral(rec, config=cfg)
    assert any("shorter" in w.lower() or "insufficient" in w.lower() for w in report.warnings)
    ac = report.primary_autocorr()
    assert ac is not None
    assert ac.reliability_label == "insufficient"
    assert ac.stride_period_s is None


def test_irregular_signal_has_low_reliability() -> None:
    rec = _sine_recording(duration_s=6.0, fps=40.0, irregular=True)
    report = analyze_spectral(rec)
    ac = report.primary_autocorr()
    assert ac is not None
    assert ac.reliability < 0.65 or ac.stride_period_s is None


def test_noisy_sine_still_finds_frequency() -> None:
    rec = _sine_recording(freq_hz=1.1, duration_s=10.0, fps=50.0, noise=0.08)
    report = analyze_spectral(rec)
    spec = report.primary_spectrum()
    assert spec is not None and spec.dominant_hz is not None
    assert abs(spec.dominant_hz - 1.1) <= 0.2


def test_missing_joint_is_nan_not_zero_fill() -> None:
    rec = _sine_recording(duration_s=3.0, fps=30.0)
    rec.snapshots[5].joints.pop("left_knee", None)
    rec.snapshots[5].dofs["left_knee_flexion"] = DofSample(
        dof_id="left_knee_flexion", angle_deg=None
    )
    rec.snapshots[5].metadata["landmark_visibility"]["left_knee"] = 0.0
    bundle = build_spectral_series(rec, channel_ids=["dof:left_knee_flexion"])
    series = bundle.get("dof:left_knee_flexion")
    assert series is not None
    # Raw grid may interpolate a 1-sample hole; the source sample itself is missing.
    assert np.isnan(series.raw).sum() >= 0
    assert not np.all(series.raw == 0.0)


def test_r0_is_symmetric_and_eigh_matches_numpy() -> None:
    rec = _sine_recording(duration_s=6.0, fps=40.0)
    report = analyze_spectral(rec)
    assert report.matrices is not None and report.matrices.r0 is not None
    corr = report.matrices.r0.correlation
    np.testing.assert_allclose(corr, corr.T, atol=1e-10)
    eig = report.eigen["correlation"]
    work = np.nan_to_num(0.5 * (corr + corr.T), nan=0.0)
    w, _v = np.linalg.eigh(work)
    np.testing.assert_allclose(np.sort(eig.eigenvalues), np.sort(w), atol=1e-8)


def test_lagged_matrix_obeys_time_reversal_transpose() -> None:
    """R(τ) = R(−τ)^T for lagged covariance; τ ≠ 0 is not symmetric."""
    fps = 50.0
    t = np.arange(0, 8.0, 1 / fps)
    data = np.column_stack(
        [
            np.sin(2 * np.pi * 1.0 * t),
            np.sin(2 * np.pi * 1.0 * t - 0.35),
            0.4 * np.sin(2 * np.pi * 2.0 * t) + 0.1 * t,
        ]
    )
    tau = 12
    r_pos = lagged_matrix(data, tau=tau, fps=fps)
    r_neg = lagged_matrix(data, tau=-tau, fps=fps)
    assert r_pos.symmetric is False
    assert r_neg.symmetric is False
    np.testing.assert_allclose(r_pos.covariance, r_neg.covariance.T, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(r_pos.correlation, r_neg.correlation.T, atol=1e-10, equal_nan=True)
    assert not np.allclose(r_pos.covariance, r_pos.covariance.T, atol=1e-6, equal_nan=True)


def test_lagged_matrix_is_not_fed_to_eigh() -> None:
    data = np.column_stack(
        [
            np.sin(2 * np.pi * 1.0 * np.arange(200) / 50.0),
            np.sin(2 * np.pi * 1.0 * (np.arange(200) / 50.0 - 0.2)),
        ]
    )
    lagged = lagged_matrix(data, tau=10, fps=50.0)
    assert lagged.symmetric is False
    with pytest.raises(NonsymmetricEigError):
        eigh_symmetric(
            lagged.covariance,
            channel_ids=["a", "b"],
            source="R(tau)",
            explanation="should fail",
            atol=1e-12,
        )


def test_left_right_cross_corr_peaks_at_half_period() -> None:
    fps = 50.0
    t = np.arange(0, 8.0, 1 / fps)
    left = np.sin(2 * np.pi * 1.0 * t)
    right = np.sin(2 * np.pi * 1.0 * t + math.pi)  # opposite phase ≈ half stride
    xc = lagged_cross_correlation(left, right, max_lag=int(2.0 * fps), fps=fps)
    assert xc.peak_lag_s is not None
    # Opposite-phase sinusoids peak at ±0.5 s
    assert abs(abs(xc.peak_lag_s) - 0.5) <= 0.06


def test_identical_channels_unit_correlation() -> None:
    rec = _sine_recording(duration_s=5.0, fps=40.0)
    bundle = build_spectral_series(
        rec, channel_ids=["dof:left_knee_flexion", "dof:left_hip_flexion"]
    )
    # Overwrite hip with a copy of knee processed values via matrix_block after cloning
    knee = bundle.channels["dof:left_knee_flexion"].processed.copy()
    bundle.channels["dof:left_hip_flexion"].processed[:] = knee
    bundle.channels["dof:left_hip_flexion"].raw[:] = knee
    mats = build_matrix_bundle(
        bundle,
        ["dof:left_knee_flexion", "dof:left_hip_flexion"],
        config=SpectralAnalysisConfig(zscore_for_matrices=True),
    )
    assert mats.r0 is not None
    r = mats.r0.correlation[0, 1]
    assert r == pytest.approx(1.0, abs=1e-6)


def test_export_json_and_csv(tmp_path: Path) -> None:
    rec = _sine_recording(duration_s=4.0, fps=30.0)
    report = analyze_spectral(rec)
    js = export_spectral_json(report, tmp_path / "spectral.json")
    csv_path = export_spectral_csv(report, tmp_path / "spectral.csv")
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["schema"] == "stablewalk-spectral-analysis"
    assert payload["kind"] == "derived"
    assert "disclaimer" in payload
    text = csv_path.read_text(encoding="utf-8")
    assert "# parameters" in text
    assert "# autocorrelation" in text
    assert "# welch_psd" in text
    assert "# session_vector" in text
    assert "dominant_frequency_hz" in text
    assert "# r_tau_correlation" in text or "r_tau_correlation_tau_" in text
    assert "# left_right_cross_correlation" in text


def test_empty_recording_unavailable() -> None:
    rec = GaitMotionRecording(source="", fps=30.0, snapshots=[])
    report = analyze_spectral(rec)
    assert report.available is False
    assert report.warnings


def test_processed_signal_is_detrended_and_normalized() -> None:
    rec = _sine_recording(freq_hz=1.0, duration_s=6.0, fps=50.0)
    bundle = build_spectral_series(rec)
    ch = bundle.get("dof:left_knee_flexion")
    assert ch is not None
    y = ch.normalized[np.isfinite(ch.normalized)]
    assert y.size >= 64
    assert abs(float(np.mean(y))) < 0.08
    assert abs(float(np.std(y)) - 1.0) < 0.08
    native = ch.processed[np.isfinite(ch.processed)]
    assert float(np.std(native)) > 1.5  # still in degrees, not z-scored


def test_insufficient_cycles_omits_period_and_cadence() -> None:
    from stablewalk.analysis.spectral.autocorrelation import insufficient_cycles_reason

    cfg = SpectralAnalysisConfig(
        min_duration_s=1.0, min_samples=32, min_cycles_for_stride=2
    )
    rec = _sine_recording(freq_hz=1.0, duration_s=1.5, fps=50.0)
    report = analyze_spectral(rec, config=cfg)
    ac = report.primary_autocorr()
    assert ac is not None
    assert ac.stride_period_s is None
    assert ac.cadence_steps_per_min is None
    assert ac.gait_period_s is None
    assert ac.insufficient_reason is not None
    assert "insufficient cycles" in ac.insufficient_reason.lower()
    assert any("insufficient cycles" in w.lower() for w in report.warnings)
    assert ac.rho.size > 10
    reason = insufficient_cycles_reason(
        duration_s=1.5, period_s=1.0, min_cycles=2
    )
    assert reason is not None and "insufficient cycles" in reason.lower()


def test_sine_period_cadence_and_spectral_shape() -> None:
    rec = _sine_recording(freq_hz=1.0, duration_s=8.0, fps=50.0)
    report = analyze_spectral(rec)
    ac = report.primary_autocorr()
    sp = report.primary_spectrum()
    assert ac is not None and sp is not None
    assert ac.insufficient_reason is None
    assert ac.gait_period_s == pytest.approx(1.0, abs=0.08)
    assert ac.cadence_steps_per_min == pytest.approx(120.0, abs=15.0)
    assert ac.peaks
    assert sp.dominant_hz == pytest.approx(1.0, abs=0.15)
    assert sp.dominant_power is not None and sp.dominant_power > 0
    assert sp.spectral_centroid_hz == pytest.approx(1.0, abs=0.30)
    assert sp.spectral_bandwidth_hz is not None
    assert sp.spectral_entropy is not None
    assert sp.spectral_entropy < 0.55
    noise = analyze_spectral(
        _sine_recording(duration_s=8.0, fps=50.0, irregular=True)
    )
    nsp = noise.primary_spectrum()
    assert nsp is not None and nsp.spectral_entropy is not None
    assert nsp.spectral_entropy > sp.spectral_entropy


def test_irregular_timestamps_resample_keeps_frequency() -> None:
    rec = _sine_recording(freq_hz=1.0, duration_s=8.0, fps=50.0)
    for i, snap in enumerate(rec.snapshots):
        snap.time_s = i / 50.0 + 0.004 * math.sin(i * 0.7)
    report = analyze_spectral(rec)
    spec = report.primary_spectrum()
    assert spec is not None
    assert spec.dominant_hz == pytest.approx(1.0, abs=0.20)
