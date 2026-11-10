"""Autocorrelation gait analysis: periodic, noisy, and irregular signals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stablewalk.analysis.gait_acf import (
    GAIT_ACF_CHANNEL_IDS,
    analyze_gait_acf_from_arrays,
    analyze_gait_autocorrelation,
    detect_peaks_after_lag_zero,
    normalize_signal,
)
from stablewalk.analysis.spectral.autocorrelation import normalized_autocorrelation
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
    seed: int = 0,
    lr_phase: float = math.pi,
) -> tuple[dict[str, np.ndarray], float]:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fps))
    t = np.arange(n, dtype=float) / fps
    if irregular:
        left = rng.normal(size=n)
        right = rng.normal(size=n)
        pelvis = rng.normal(size=n)
    else:
        left = np.sin(2 * math.pi * freq_hz * t)
        right = np.sin(2 * math.pi * freq_hz * t + lr_phase)
        pelvis = 0.15 * np.sin(2 * math.pi * freq_hz * t)
        if noise:
            left = left + rng.normal(scale=noise, size=n)
            right = right + rng.normal(scale=noise, size=n)
            pelvis = pelvis + rng.normal(scale=noise * 0.4, size=n)
    signals = {
        "left_ankle_vertical": left.copy(),
        "right_ankle_vertical": right.copy(),
        "left_knee_angle": 20.0 * left + 40.0,
        "right_knee_angle": 20.0 * right + 40.0,
        "left_hip_angle": 12.0 * left + 8.0,
        "right_hip_angle": 12.0 * right + 8.0,
        "pelvis_lateral": pelvis,
    }
    return signals, fps


def test_normalize_signal_is_zscored() -> None:
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    z = normalize_signal(y)
    assert abs(float(np.mean(z))) < 1e-12
    assert float(np.std(z, ddof=1)) == pytest.approx(1.0)
    const = normalize_signal(np.ones(12) * 5.0)
    np.testing.assert_allclose(const, 0.0, atol=1e-12)


def test_peaks_exclude_lag_zero() -> None:
    fps = 50.0
    t = np.arange(0, 6.0, 1.0 / fps)
    x = np.sin(2 * math.pi * 1.0 * t)
    x = normalize_signal(x)
    lags, rho = normalized_autocorrelation(x, max_lag=int(2.5 * fps))
    lags_s = lags.astype(float) / fps
    assert rho[0] == pytest.approx(1.0)
    peaks = detect_peaks_after_lag_zero(lags_s, rho)
    assert peaks
    assert all(p.lag_samples > 0 and p.lag_s >= 0.25 for p in peaks)
    best = peaks[0]
    assert abs(best.lag_s - 1.0) <= 0.06
    assert best.value > 0.7


def test_periodic_signals_recover_cycle_and_leg_offset() -> None:
    signals, fps = _periodic_arrays(freq_hz=1.0, duration_s=8.0, fps=50.0)
    report = analyze_gait_acf_from_arrays(signals, fps, quality_score=90.0)
    assert report.available is True
    assert report.gait_cycle_duration_s is not None
    assert abs(report.gait_cycle_duration_s - 1.0) <= 0.06
    assert report.dominant_period_s is not None
    assert abs(report.dominant_period_s - 1.0) <= 0.08
    assert report.regularity_score is not None
    assert report.regularity_score > 0.55
    assert report.confidence > 0.5
    assert report.confidence_label in {"moderate", "high"}
    knee = report.leg_offsets["knee_angle"]
    assert knee.time_offset_s is not None
    assert abs(abs(knee.time_offset_s) - 0.5) <= 0.08
    assert knee.phase_offset_deg is not None
    assert abs(abs(knee.phase_offset_deg) - 180.0) <= 25.0
    hip = report.leg_offsets["hip_angle"]
    assert hip.time_offset_s is not None
    assert abs(abs(hip.time_offset_s) - 0.5) <= 0.08
    assert all(s.n_samples > 0 for s in report.signals.values())
    assert "left_ankle_vertical" in report.signals
    assert "pelvis_lateral" in report.signals


def test_noisy_periodic_still_recovers_period() -> None:
    clean, fps = _periodic_arrays(freq_hz=1.0, duration_s=8.0, noise=0.0, seed=1)
    noisy, _ = _periodic_arrays(freq_hz=1.0, duration_s=8.0, noise=0.35, seed=1)
    clean_r = analyze_gait_acf_from_arrays(clean, fps, quality_score=88.0)
    noisy_r = analyze_gait_acf_from_arrays(noisy, fps, quality_score=88.0)
    assert noisy_r.available is True
    assert noisy_r.gait_cycle_duration_s is not None
    assert abs(noisy_r.gait_cycle_duration_s - 1.0) <= 0.10
    assert noisy_r.regularity_score is not None
    assert clean_r.regularity_score is not None
    assert noisy_r.regularity_score < clean_r.regularity_score
    assert noisy_r.confidence <= clean_r.confidence + 1e-9


def test_irregular_signals_do_not_invent_a_cycle() -> None:
    signals, fps = _periodic_arrays(duration_s=8.0, irregular=True, seed=3)
    report = analyze_gait_acf_from_arrays(signals, fps, quality_score=85.0)
    assert report.available is False
    assert report.gait_cycle_duration_s is None
    assert report.dominant_period_s is None
    assert report.confidence_label == "insufficient"
    assert report.leg_offsets == {} or report.available is False


def test_low_video_quality_withholds_results() -> None:
    signals, fps = _periodic_arrays(freq_hz=1.0, duration_s=8.0)
    report = analyze_gait_acf_from_arrays(signals, fps, quality_score=25.0)
    assert report.available is False
    assert report.gait_cycle_duration_s is None
    assert report.dominant_period_s is None
    assert report.regularity_score is None
    assert report.leg_offsets == {}
    assert any("insufficient" in n.lower() or "quality" in n.lower() for n in report.notes)


def test_short_clip_is_insufficient() -> None:
    signals, fps = _periodic_arrays(duration_s=0.8, fps=30.0)
    report = analyze_gait_acf_from_arrays(signals, fps, quality_score=90.0)
    assert report.available is False
    assert report.gait_cycle_duration_s is None


def _sine_recording(
    *,
    fps: float = 50.0,
    duration_s: float = 8.0,
    freq_hz: float = 1.0,
) -> GaitMotionRecording:
    n = int(round(duration_s * fps))
    snaps: list[SkeletonSnapshot] = []
    for i in range(n):
        t = i / fps
        val = math.sin(2 * math.pi * freq_hz * t)
        joints = {}
        for jid in (*JOINT_IDS, ROOT_JOINT_ID):
            phase = math.pi if jid.startswith("right_") else 0.0
            y = 0.2 * math.sin(2 * math.pi * freq_hz * t + phase)
            x = 0.05 if "right" in jid else -0.05
            joints[jid] = JointSample(
                joint_id=jid,
                position=Vec3(x, y, 0.0),
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
                    "landmark_visibility": {
                        jid: 0.95 for jid in (*JOINT_IDS, ROOT_JOINT_ID)
                    },
                    "image_xy": {
                        ROOT_JOINT_ID: [0.5 + 0.02 * val, 0.55],
                        "left_hip": [0.47, 0.55],
                        "right_hip": [0.53, 0.55],
                    },
                },
            )
        )
    return GaitMotionRecording(
        source="synthetic_gait_acf",
        fps=fps,
        snapshots=snaps,
        source_kind="mock",
        coordinate_system="sw_canonical_y_up",
    )


def test_recording_uses_smoothed_series_without_changing_metrics() -> None:
    rec = _sine_recording()
    report = analyze_gait_autocorrelation(rec, quality_score=90.0)
    assert set(GAIT_ACF_CHANNEL_IDS) >= {
        "left_ankle_vertical",
        "right_ankle_vertical",
        "left_knee_angle",
        "right_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
        "pelvis_lateral",
    }
    assert report.available is True
    assert report.gait_cycle_duration_s is not None
    assert abs(report.gait_cycle_duration_s - 1.0) <= 0.08
    assert report.source == "smoothed_spectral_series"
    assert "left_knee_angle" in report.signals
    assert rec.snapshots[0].dofs["left_knee_flexion"].angle_deg is not None
    # Recording is not mutated and gait-quality modules are not invoked.
    assert report.to_dict()["kind"] == "derived"
    assert "not clinical" in report.to_dict()["disclaimer"].lower() or "diagnos" in report.to_dict()["disclaimer"].lower()
