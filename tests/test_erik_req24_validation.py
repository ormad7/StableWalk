"""Executed validation for Erik requirements 2–4 (synthetic GT + demo poses)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.asymmetry3d.report import analyze_asymmetry3d
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.analysis.opensim_id_readiness import assess_opensim_id_readiness
from stablewalk.analysis.spectral.report import analyze_spectral
from stablewalk.analysis.spectral.spectrum import rfft_periodogram, welch_psd
from stablewalk.analysis.synthetic_gait import SyntheticGaitSpec, known_frequency_vector, make_sine_gait_recording
from stablewalk.io.pose_loader import load_pose_sequence


def test_synthetic_sine_recovers_known_frequency_fft_and_welch() -> None:
    fps, f0, n = 50.0, 1.25, 400
    x = known_frequency_vector(n, fps, f0)
    freq, pxx = rfft_periodogram(x, fps)
    band = (freq >= 0.4) & (freq <= 4.0)
    hat = float(freq[band][np.argmax(pxx[band])])
    assert abs(hat - f0) <= fps / n + 1e-9
    fw, pw, method = welch_psd(x, fps, nperseg=128, noverlap=64)
    assert method in {"scipy.signal.welch", "numpy_welch"}
    wb = (fw >= 0.4) & (fw <= 4.0)
    hat_w = float(fw[wb][np.argmax(pw[wb])])
    assert abs(hat_w - f0) <= 0.12


def test_synthetic_recording_features_match_spec() -> None:
    spec = SyntheticGaitSpec(
        fps=50.0,
        duration_s=8.0,
        stride_hz=1.0,
        knee_amp_left_deg=20.0,
        knee_amp_right_deg=20.0,
    )
    rec = make_sine_gait_recording(spec)
    report = analyze_spectral(rec)
    assert report.available
    vec = report.features.session_vector
    assert vec["dominant_frequency_hz"] is not None
    assert abs(float(vec["dominant_frequency_hz"]) - spec.stride_hz) <= 0.15
    assert vec["spectral_entropy"] is not None
    assert 0.0 <= float(vec["spectral_entropy"]) <= 1.0
    ac = report.primary_autocorr()
    assert ac is not None and ac.stride_period_s is not None
    assert abs(ac.stride_period_s - spec.stride_period_s) <= 0.08
    assert ac.peaks and ac.peaks[0].value > 0.05
    assert report.matrices is not None and report.matrices.r0 is not None
    r0 = report.matrices.r0.correlation
    assert r0.shape[0] == r0.shape[1]
    assert np.allclose(r0, r0.T, equal_nan=True, atol=1e-8)
    assert "correlation" in report.eigen


def test_synthetic_missing_samples_do_not_zero_fill() -> None:
    spec = SyntheticGaitSpec(
        fps=40.0,
        duration_s=6.0,
        stride_hz=1.0,
        knee_amp_left_deg=18.0,
        knee_amp_right_deg=18.0,
        missing_every=7,
    )
    rec = make_sine_gait_recording(spec)
    report = analyze_spectral(rec)
    series = report.bundle.get(report.univariate_channel_id)
    assert series is not None
    assert series.missing_fraction > 0.0
    assert not np.all(np.nan_to_num(series.raw, nan=0.0) == series.raw) or np.isnan(series.raw).any()
    for feat in report.features.records:
        if not feat.available:
            assert feat.value is None


def _pose_rec(stem: str):
    path = config.POSES_DIR / f"{stem}_poses.json"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    return pose_sequence_to_gait_motion(load_pose_sequence(path))


def test_normal_gait_pose_spectral_and_events() -> None:
    rec = _pose_rec("normal_gait")
    cycles = analyze_gait_cycles(rec)
    spec = analyze_spectral(rec, cycles=cycles)
    asym = analyze_asymmetry3d(rec, cycles=cycles)
    assert rec.frame_count >= 60
    assert spec.available
    vec = spec.features.session_vector
    assert vec["dominant_frequency_hz"] is not None
    assert vec["spectral_entropy"] is not None
    if vec["acf_stride_peak"] is not None:
        assert float(vec["acf_stride_peak"]) > 0.0
    id_report = assess_opensim_id_readiness(run_name="normal_gait")
    assert id_report.ready_for_traditional_id is False
    assert asym is not None
    if cycles.metrics.gait_cycle_count < 2:
        assert cycles.event_sufficiency in ("input_insufficient", "processing_failure")
        assert cycles.metrics.cadence_steps_per_min is None


def test_abnormal_gait_does_not_fabricate_cadence_when_insufficient() -> None:
    rec = _pose_rec("abnormal_gait")
    cycles = analyze_gait_cycles(rec)
    spec = analyze_spectral(rec, cycles=cycles)
    assert spec.available
    for feat in spec.features.records:
        if not feat.available:
            assert feat.value is None
    if cycles.metrics.gait_cycle_count < 2 or not cycles.metrics.metrics_reliable:
        assert cycles.metrics.cadence_steps_per_min is None
        assert cycles.event_sufficiency in ("input_insufficient", "processing_failure")
        assert cycles.event_sufficiency_reason
    assert assess_opensim_id_readiness(run_name="abnormal_gait").ready_for_traditional_id is False


def test_synthetic_asymmetric_gait_reports_nonzero_knee_asi() -> None:
    spec = SyntheticGaitSpec(
        fps=50.0,
        duration_s=8.0,
        stride_hz=1.0,
        knee_amp_left_deg=20.0,
        knee_amp_right_deg=40.0,
    )
    rec = make_sine_gait_recording(spec)
    report = analyze_asymmetry3d(rec)
    knee = report.scalar_by_name("knee_flexion_rom")
    assert knee is not None
    assert knee.left is not None and knee.right is not None
    assert knee.right > knee.left
    assert knee.asi_percent is not None
    assert abs(float(knee.asi_percent) - spec.expected_asi_knee_amp) < 25.0
    id_report = assess_opensim_id_readiness(run_name="synthetic_asymmetric")
    assert id_report.ready_for_traditional_id is False
