"""Tests for spectral feature extraction (Erik requirement 3)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from stablewalk.analysis.gait_cycle_analysis import DetectedGaitCycle, GaitCycleAnalysisResult
from stablewalk.analysis.spectral.config import SpectralAnalysisConfig
from stablewalk.analysis.spectral.export import export_spectral_csv, export_spectral_json
from stablewalk.analysis.spectral.features import (
    SESSION_VECTOR_KEYS,
    band_power,
    rms_amplitude,
    spectral_bandwidth,
    spectral_entropy,
    wrap_phase_deg,
)
from stablewalk.analysis.spectral.report import analyze_spectral
from stablewalk.models.gait_motion import GaitMotionRecording

from test_spectral_analysis import _sine_recording


def _by_name(records, name, *, scope="session"):
    return [r for r in records if r.name == name and r.scope == scope]


def test_spectral_entropy_matches_numpy_shannon_definition() -> None:
    freq = np.linspace(0.0, 10.0, 101)
    psd = np.exp(-((freq - 1.0) ** 2) / (2 * 0.15**2))
    band = (0.4, 4.0)
    got = spectral_entropy(freq, psd, band)
    mask = (freq >= band[0]) & (freq <= band[1]) & np.isfinite(psd) & (psd > 0)
    p = psd[mask]
    probs = p / np.sum(p)
    probs = probs[probs > 0]
    expected = float(-np.sum(probs * np.log(probs)) / math.log(probs.size))
    assert got is not None
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_band_power_matches_numpy_bin_sum() -> None:
    freq = np.linspace(0.0, 8.0, 161)
    psd = np.ones_like(freq)
    band = (1.0, 3.0)
    got = band_power(freq, psd, band)
    mask = (freq >= band[0]) & (freq <= band[1]) & np.isfinite(psd)
    df = float(np.mean(np.diff(freq)))
    expected = float(np.sum(psd[mask]) * df)
    assert got is not None
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_sine_dominant_frequency_and_entropy_vs_noise() -> None:
    sine = analyze_spectral(_sine_recording(freq_hz=1.0, duration_s=8.0, fps=50.0))
    noise = analyze_spectral(_sine_recording(duration_s=8.0, fps=50.0, irregular=True))
    assert sine.features is not None and noise.features is not None
    f0 = _by_name(sine.features.records, "dominant_frequency_hz")[0]
    assert f0.available
    assert abs(f0.value - 1.0) <= 0.15
    h_sine = _by_name(sine.features.records, "spectral_entropy")[0]
    h_noise = _by_name(noise.features.records, "spectral_entropy")[0]
    assert h_sine.available and h_noise.available
    assert h_sine.value < h_noise.value


def test_sampling_rate_does_not_shift_known_frequency() -> None:
    a = analyze_spectral(_sine_recording(freq_hz=1.2, duration_s=8.0, fps=30.0))
    b = analyze_spectral(_sine_recording(freq_hz=1.2, duration_s=8.0, fps=50.0))
    fa = _by_name(a.features.records, "dominant_frequency_hz")[0]
    fb = _by_name(b.features.records, "dominant_frequency_hz")[0]
    assert fa.available and fb.available
    assert abs(fa.value - 1.2) <= 0.2
    assert abs(fb.value - 1.2) <= 0.15


def test_short_clip_marks_features_unavailable() -> None:
    cfg = SpectralAnalysisConfig(min_duration_s=2.0, min_samples=64)
    report = analyze_spectral(_sine_recording(duration_s=0.5, fps=30.0), config=cfg)
    assert report.features is not None
    stride = _by_name(report.features.records, "stride_duration_s")
    if stride:
        assert stride[0].available is False
        assert stride[0].value is None
        assert stride[0].quality == "insufficient"


def test_missing_landmarks_do_not_fabricate_amplitude() -> None:
    rec = _sine_recording(duration_s=4.0, fps=30.0, drop_every=2)
    report = analyze_spectral(rec)
    assert report.features is not None
    for rec_feat in report.features.records:
        if not rec_feat.available:
            assert rec_feat.value is None


def test_session_vector_keys_stable_and_nullable() -> None:
    report = analyze_spectral(_sine_recording(freq_hz=1.0, duration_s=6.0, fps=40.0))
    vec = report.features.session_vector
    assert tuple(vec) == SESSION_VECTOR_KEYS
    assert vec["dominant_frequency_hz"] is not None
    # Irregular / missing cycle CV must stay None, not 0.
    rec = _sine_recording(duration_s=3.0, fps=30.0)
    report2 = analyze_spectral(rec)
    assert report2.features.session_vector["stride_cv"] is None


def test_left_right_phase_for_opposite_sines() -> None:
    report = analyze_spectral(_sine_recording(freq_hz=1.0, duration_s=8.0, fps=50.0))
    feats = report.features
    lag = [r for r in feats.records if r.name == "lr_xcorr_lag_s"][0]
    phase = [r for r in feats.records if r.name == "lr_relative_phase_deg"][0]
    assert lag.available
    # Antiphase 1 Hz peaks at odd half-seconds (±0.5, ±1.5, ±2.5). Global max ρ
    # may land on a longer alias; the half-cycle remainder must still be 0.5 s.
    assert abs((abs(float(lag.value)) % 1.0) - 0.5) <= 0.08
    assert phase.available
    assert abs(abs(phase.value) - 180.0) <= 25.0


def test_cycle_features_from_detected_cycles() -> None:
    rec = _sine_recording(freq_hz=1.0, duration_s=6.0, fps=50.0)
    cycles = GaitCycleAnalysisResult(
        cycles=[
            DetectedGaitCycle(0, 0, 50, 0.0, 1.0, 1.0, "left"),
            DetectedGaitCycle(1, 50, 100, 1.0, 2.0, 1.0, "left"),
            DetectedGaitCycle(2, 100, 150, 2.0, 3.05, 1.05, "left"),
        ]
    )
    report = analyze_spectral(rec, cycles=cycles)
    assert report.features.n_detected_cycles == 3
    assert report.features.n_valid_cycles == 3
    cv = _by_name(report.features.records, "stride_cv")[0]
    assert cv.available
    assert cv.value is not None and cv.value < 0.1
    cycle_rms = [r for r in report.features.records if r.scope == "cycle" and r.name == "rms_amplitude"]
    assert len(cycle_rms) == 3
    assert all(r.available for r in cycle_rms)


def test_stride_cv_accepts_plain_cycle_list() -> None:
    rec = _sine_recording(freq_hz=1.0, duration_s=6.0, fps=50.0)
    cycles = [
        DetectedGaitCycle(0, 0, 50, 0.0, 1.0, 1.0, "left"),
        DetectedGaitCycle(1, 50, 100, 1.0, 2.0, 1.02, "left"),
    ]
    report = analyze_spectral(rec, cycles=cycles)
    assert report.features.n_detected_cycles == 2
    cv = _by_name(report.features.records, "stride_cv")[0]
    assert cv.available
    assert cv.value is not None


def test_entropy_bandwidth_helpers() -> None:
    freq = np.linspace(0.0, 5.0, 51)
    psd = np.zeros_like(freq)
    psd[10] = 1.0  # 1 Hz if df=0.1
    band = (0.4, 4.0)
    # Make a narrow peak at 1.0 Hz
    psd[:] = 0
    idx = int(np.argmin(np.abs(freq - 1.0)))
    psd[idx] = 10.0
    h = spectral_entropy(freq, psd, band)
    bw = spectral_bandwidth(freq, psd, band)
    p = band_power(freq, psd, band)
    assert h is not None and h < 0.2
    assert bw is not None and bw < 0.3
    assert p is not None and p > 0
    assert rms_amplitude(np.array([3.0, 4.0])) == pytest.approx(math.sqrt(12.5))
    assert wrap_phase_deg(190.0) == pytest.approx(-170.0)


def test_noisy_sine_keeps_frequency() -> None:
    report = analyze_spectral(
        _sine_recording(freq_hz=1.0, duration_s=10.0, fps=50.0, noise=0.1)
    )
    f0 = _by_name(report.features.records, "dominant_frequency_hz")[0]
    assert f0.available
    assert abs(f0.value - 1.0) <= 0.2


def test_empty_recording_has_no_invented_vector() -> None:
    report = analyze_spectral(GaitMotionRecording(source="", fps=30.0, snapshots=[]))
    assert report.features is not None
    assert report.features.session_vector == {} or all(
        v is None for v in report.features.session_vector.values()
    )
    assert any("unavailable" in w.lower() or "no gait" in w.lower() for w in report.warnings)


def test_feature_export_includes_definitions(tmp_path: Path) -> None:
    report = analyze_spectral(_sine_recording(duration_s=4.0, fps=30.0, freq_hz=1.0))
    js = export_spectral_json(report, tmp_path / "out.json")
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["features"]["schema"] == "stablewalk-spectral-features"
    assert "dominant_frequency_hz" in payload["features"]["definitions"]
    assert payload["features"]["kind"] == "derived"
    csv_text = export_spectral_csv(report, tmp_path / "out.csv").read_text(encoding="utf-8")
    assert "# spectral_features" in csv_text
    assert "# session_vector" in csv_text
    assert "dominant_frequency_hz" in csv_text
    assert "not a clinical" in payload["features"]["disclaimer"].lower()
    assert payload["features"]["session_vector"]["dominant_frequency_hz"] is not None
    assert "mean_visibility" in payload["channels"][report.univariate_channel_id]


def test_harmonic_ratio_on_two_tone_sine() -> None:
    rec = _sine_recording(freq_hz=1.0, duration_s=8.0, fps=50.0)
    fps = 50.0
    for i, snap in enumerate(rec.snapshots):
        t = i / fps
        val = math.sin(2 * math.pi * t) + 0.6 * math.sin(4 * math.pi * t)
        angle = val * 20.0 + 40.0
        for dof_id, sample in snap.dofs.items():
            sample.angle_deg = -angle if "right_" in dof_id else angle
    report = analyze_spectral(rec)
    hr = _by_name(report.features.records, "harmonic_ratio")[0]
    assert hr.available
    assert hr.value is not None and hr.value > 0.15
    cad = _by_name(report.features.records, "spectral_cadence_steps_per_min")[0]
    assert cad.available
    assert abs(float(cad.value) - 120.0) <= 20.0
    assert "steps/min" in cad.display_value()


def test_insufficient_cycles_features_are_explicit() -> None:
    cfg = SpectralAnalysisConfig(
        min_duration_s=1.0, min_samples=32, min_cycles_for_stride=2
    )
    rec = _sine_recording(freq_hz=1.0, duration_s=1.5, fps=50.0)
    report = analyze_spectral(rec, config=cfg)
    assert report.features is not None
    stride = _by_name(report.features.records, "stride_duration_s")[0]
    cad = _by_name(report.features.records, "spectral_cadence_steps_per_min")[0]
    assert stride.available is False
    assert stride.value is None
    assert "insufficient cycles" in stride.display_value().lower()
    assert cad.available is False
    assert cad.value is None
    vec = report.features.session_vector
    assert vec["stride_duration_s"] is None
    assert vec["spectral_cadence_steps_per_min"] is None
    assert any("insufficient cycles" in w.lower() for w in report.warnings)


def test_repeated_analysis_is_reproducible() -> None:
    rec = _sine_recording(freq_hz=1.0, duration_s=6.0, fps=50.0, seed=7)
    a = analyze_spectral(rec)
    b = analyze_spectral(rec)
    assert a.features is not None and b.features is not None
    assert a.features.session_vector == b.features.session_vector
    knee = a.bundle.get("dof:left_knee_flexion")
    assert knee is not None
    assert 0.0 <= knee.mean_visibility <= 1.0
    assert knee.mean_visibility >= 0.5
