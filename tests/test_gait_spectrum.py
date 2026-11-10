"""Spectral gait analysis: known sines, irregular signals, export, short clips."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from stablewalk.analysis.gait_spectrum import (
    analyze_gait_spectrum,
    analyze_gait_spectrum_from_arrays,
    analyze_signal_spectrum,
    export_gait_spectrum_csv,
    export_gait_spectrum_json,
)
from stablewalk.models.gait_motion import (
    DofSample,
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.models.joint_registry import DOF_IDS, JOINT_IDS, JOINT_PARENTS, ROOT_JOINT_ID


def _periodic_arrays(
    *,
    fps: float = 50.0,
    duration_s: float = 8.0,
    freq_hz: float = 1.0,
    noise: float = 0.0,
    irregular: bool = False,
    dc: float = 0.0,
    seed: int = 0,
    lr_phase: float = math.pi,
) -> tuple[dict[str, np.ndarray], float, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fps))
    t = np.arange(n, dtype=float) / fps
    if irregular:
        left = rng.normal(size=n)
        right = rng.normal(size=n)
        pelvis = rng.normal(size=n)
    else:
        left = np.sin(2 * math.pi * freq_hz * t) + dc
        right = np.sin(2 * math.pi * freq_hz * t + lr_phase) + dc
        pelvis = 0.12 * np.sin(2 * math.pi * freq_hz * t) + dc
        if noise:
            left = left + rng.normal(scale=noise, size=n)
            right = right + rng.normal(scale=noise, size=n)
            pelvis = pelvis + rng.normal(scale=noise * 0.3, size=n)
    signals = {
        "left_ankle_height": left.copy(),
        "right_ankle_height": right.copy(),
        "left_knee_angle": 20.0 * (left - dc) + 40.0 + dc,
        "right_knee_angle": 20.0 * (right - dc) + 40.0 + dc,
        "left_hip_angle": 12.0 * (left - dc) + 8.0,
        "right_hip_angle": 12.0 * (right - dc) + 8.0,
        "pelvis_lateral": pelvis,
        "pelvis_vertical": 0.04 * (left - dc),
        "left_foot_ground": np.clip(left - dc, -0.2, 0.4),
        "right_foot_ground": np.clip(right - dc, -0.2, 0.4),
    }
    return signals, fps, t


def test_sine_dominant_frequency_and_period() -> None:
    signals, fps, times = _periodic_arrays(freq_hz=1.0, duration_s=8.0, fps=50.0)
    report = analyze_gait_spectrum_from_arrays(signals, fps, times_s=times)
    assert report.available is True
    assert report.fps == pytest.approx(50.0)
    assert abs(report.timestamps_s[10] - 10 / 50.0) < 1e-9
    assert report.dominant_frequency_hz is not None
    assert abs(report.dominant_frequency_hz - 1.0) <= 0.12
    assert report.dominant_period_s is not None
    assert abs(report.dominant_period_s - 1.0) <= 0.15
    knee = report.signals["left_knee_angle"]
    assert knee.spectral_energy is not None and knee.spectral_energy > 0
    assert knee.spectral_entropy is not None
    assert knee.low_band_power is not None
    assert knee.mid_band_power is not None
    assert knee.high_band_power is not None
    assert knee.low_band_power > knee.high_band_power


def test_dc_offset_does_not_shift_frequency() -> None:
    a, fps, t = _periodic_arrays(freq_hz=1.25, dc=0.0, seed=2)
    b, _, _ = _periodic_arrays(freq_hz=1.25, dc=15.0, seed=2)
    ra = analyze_gait_spectrum_from_arrays(a, fps, times_s=t)
    rb = analyze_gait_spectrum_from_arrays(b, fps, times_s=t)
    assert ra.dominant_frequency_hz is not None
    assert rb.dominant_frequency_hz is not None
    assert abs(ra.dominant_frequency_hz - 1.25) <= 0.15
    assert abs(rb.dominant_frequency_hz - ra.dominant_frequency_hz) <= 0.08


def test_sampling_rate_from_timestamps() -> None:
    for fps in (30.0, 50.0):
        signals, f, times = _periodic_arrays(freq_hz=1.0, duration_s=8.0, fps=fps)
        report = analyze_gait_spectrum_from_arrays(signals, f, times_s=times)
        assert report.available
        assert abs(report.dominant_frequency_hz - 1.0) <= 0.18
        dt = np.diff(report.timestamps_s)
        assert np.allclose(dt, 1.0 / fps, atol=1e-9)


def test_irregular_signals_have_higher_entropy_and_no_cycle() -> None:
    sine, fps, t = _periodic_arrays(freq_hz=1.0, duration_s=8.0, irregular=False)
    noise, _, _ = _periodic_arrays(duration_s=8.0, irregular=True, seed=4)
    sine_r = analyze_gait_spectrum_from_arrays(sine, fps, times_s=t)
    noise_r = analyze_gait_spectrum_from_arrays(noise, fps, times_s=t)
    assert sine_r.available is True
    assert noise_r.available is False
    assert noise_r.dominant_frequency_hz is None
    assert noise_r.dominant_period_s is None
    h_sine = sine_r.signals["left_knee_angle"].spectral_entropy
    h_noise = noise_r.signals["left_knee_angle"].spectral_entropy
    assert h_sine is not None and h_noise is not None
    assert h_sine < h_noise


def test_left_right_phase_near_half_cycle() -> None:
    signals, fps, t = _periodic_arrays(freq_hz=1.0, duration_s=8.0, lr_phase=math.pi)
    report = analyze_gait_spectrum_from_arrays(signals, fps, times_s=t)
    knee = report.phase_offsets["knee_angle"]
    assert knee.phase_difference_deg is not None
    assert abs(abs(knee.phase_difference_deg) - 180.0) <= 30.0
    assert knee.time_offset_s is not None
    assert abs(abs(knee.time_offset_s) - 0.5) <= 0.10


def test_short_and_missing_signals_are_safe() -> None:
    short, fps, t = _periodic_arrays(duration_s=0.6, fps=30.0)
    report = analyze_gait_spectrum_from_arrays(short, fps, times_s=t)
    assert report.available is False
    assert report.dominant_frequency_hz is None
    y = np.full(80, np.nan)
    spec = analyze_signal_spectrum(y, 50.0, signal_id="empty")
    assert spec.dominant_frequency_hz is None
    assert spec.n_samples == 0
    y2 = np.sin(2 * np.pi * np.arange(200) / 50.0)
    y2[10:18] = np.nan
    spec2 = analyze_signal_spectrum(y2, 50.0, signal_id="gaps")
    assert spec2.n_samples == 192


def test_export_json_and_csv(tmp_path) -> None:
    signals, fps, t = _periodic_arrays(duration_s=6.0, fps=40.0)
    report = analyze_gait_spectrum_from_arrays(signals, fps, times_s=t)
    js = export_gait_spectrum_json(report, tmp_path / "gait_spec.json")
    csv_path = export_gait_spectrum_csv(report, tmp_path / "gait_spec.csv")
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["schema"] == "stablewalk-gait-spectrum"
    assert payload["kind"] == "derived"
    assert "dominant_frequency_hz" in payload
    assert "signals" in payload
    text = csv_path.read_text(encoding="utf-8")
    assert "# summary" in text
    assert "# signal_features" in text
    assert "# welch_psd" in text
    assert "# timestamps" in text
    assert "spectral_entropy" in text
    assert "left_right_phase" in text


def _sine_recording(*, fps: float = 50.0, duration_s: float = 8.0, freq_hz: float = 1.0) -> GaitMotionRecording:
    n = int(round(duration_s * fps))
    snaps: list[SkeletonSnapshot] = []
    for i in range(n):
        t = i / fps
        val = math.sin(2 * math.pi * freq_hz * t)
        joints = {}
        for jid in (*JOINT_IDS, ROOT_JOINT_ID):
            phase = math.pi if jid.startswith("right_") else 0.0
            y = 0.25 * math.sin(2 * math.pi * freq_hz * t + phase)
            joints[jid] = JointSample(
                joint_id=jid,
                position=Vec3(0.05 if "right" in jid else -0.05, y, 0.0),
                parent_id=JOINT_PARENTS.get(jid),
            )
        dofs = {}
        for dof in DOF_IDS:
            a = val * 20.0 + 40.0
            if "right_" in dof:
                a = -a
            dofs[dof] = DofSample(dof_id=dof, angle_deg=a)
        snaps.append(
            SkeletonSnapshot(
                frame_index=i,
                time_s=t,
                joints=joints,
                dofs=dofs,
                metadata={
                    "detected": True,
                    "landmark_visibility": {jid: 0.95 for jid in (*JOINT_IDS, ROOT_JOINT_ID)},
                    "image_xy": {
                        ROOT_JOINT_ID: [0.5 + 0.02 * val, 0.55],
                        "left_hip": [0.47, 0.55],
                        "right_hip": [0.53, 0.55],
                    },
                },
            )
        )
    return GaitMotionRecording(
        source="synthetic_gait_spectrum",
        fps=fps,
        snapshots=snaps,
        source_kind="mock",
        coordinate_system="sw_canonical_y_up",
    )


def test_recording_uses_video_fps_and_smoothed_series() -> None:
    rec = _sine_recording(fps=50.0, duration_s=8.0, freq_hz=1.0)
    report = analyze_gait_spectrum(rec)
    assert report.fps == pytest.approx(50.0)
    assert report.source == "smoothed_spectral_series"
    assert report.available is True
    assert report.dominant_frequency_hz is not None
    assert abs(report.dominant_frequency_hz - 1.0) <= 0.15
    assert "left_ankle_height" in report.signals
    assert "left_foot_ground" in report.signals
    assert "left_knee_angle" in report.signals
    dt = np.diff(report.timestamps_s)
    assert np.allclose(dt, 1.0 / rec.fps, atol=1e-6)
    assert rec.snapshots[0].dofs["left_knee_flexion"].angle_deg is not None
    payload = report.to_dict()
    assert payload["kind"] == "derived"
    assert "not clinical" in payload["disclaimer"].lower() or "diagnos" in payload["disclaimer"].lower()
