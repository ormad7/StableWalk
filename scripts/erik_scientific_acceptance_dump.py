"""Dump requirement 2–4 numbers and Spectral Analysis figures for the acceptance doc."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "erik_acceptance"
OUT.mkdir(parents=True, exist_ok=True)


def _clip(stem: str) -> dict:
    from stablewalk import config
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.analysis.asymmetry3d.report import analyze_asymmetry3d
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
    from stablewalk.analysis.opensim_id_readiness import assess_opensim_id_readiness
    from stablewalk.analysis.spectral.report import analyze_spectral
    from stablewalk.io.pose_loader import load_pose_sequence
    from stablewalk.ui.viewers.spectral_charts import render_spectral_figure

    poses = config.POSES_DIR / f"{stem}_poses.json"
    seq = load_pose_sequence(poses)
    rec = pose_sequence_to_gait_motion(seq)
    cycles = analyze_gait_cycles(rec)
    spec = analyze_spectral(rec, sequence=seq, cycles=cycles)
    asym = analyze_asymmetry3d(rec, cycles=cycles, sequence=seq)
    idr = assess_opensim_id_readiness(run_name=stem)

    fig, axes = plt.subplots(3, 2, figsize=(11.2, 8.4), dpi=110)
    render_spectral_figure(
        fig,
        (
            axes[0, 0],
            axes[0, 1],
            axes[1, 0],
            axes[1, 1],
            axes[2, 0],
            axes[2, 1],
        ),
        spec,
        spec.univariate_channel_id,
    )
    fig.savefig(OUT / f"{stem}_spectral.png", facecolor="white")
    plt.close(fig)

    eig = spec.eigen.get("correlation")
    xc0 = None
    if spec.matrices and spec.matrices.cross_corr:
        xc = spec.matrices.cross_corr[0]
        xc0 = {"a": xc.channel_a, "b": xc.channel_b, "peak_lag_s": xc.peak_lag_s, "peak_rho": xc.peak_value}
    ch = spec.bundle.get(spec.univariate_channel_id)
    vec = spec.features.session_vector if spec.features else {}
    kin = asym.kinematics if asym is not None else None
    return {
        "stem": stem,
        "frames": rec.frame_count,
        "fps": rec.fps,
        "duration_s": rec.duration_s,
        "coordinate_system": spec.bundle.coordinate_system,
        "missing_fraction": None if ch is None else ch.missing_fraction,
        "interpolated_fraction": None if ch is None else ch.interpolated_fraction,
        "visibility_min": spec.config.visibility_min,
        "max_gap_s": spec.config.max_gap_s,
        "gait_band_hz": list(spec.config.gait_band_hz),
        "stride_lag_s": list(spec.config.stride_lag_s),
        "r_tau_s": [m.tau_s for m in (spec.matrices.lagged if spec.matrices else [])],
        "session_vector": {k: (None if v is None else float(v)) for k, v in vec.items()},
        "leading_eig": None if eig is None or eig.eigenvalues.size == 0 else float(eig.eigenvalues[0]),
        "r0_shape": None if spec.matrices is None or spec.matrices.r0 is None else list(spec.matrices.r0.correlation.shape),
        "xcorr0": xc0,
        "psd_method": None if spec.primary_spectrum() is None else spec.primary_spectrum().method,
        "acf_stride_s": None if spec.primary_autocorr() is None else spec.primary_autocorr().stride_period_s,
        "cycle_count": cycles.metrics.gait_cycle_count,
        "cadence": cycles.metrics.cadence_steps_per_min,
        "event_sufficiency": cycles.event_sufficiency,
        "event_sufficiency_reason": cycles.event_sufficiency_reason,
        "metrics_reliable": cycles.metrics.metrics_reliable,
        "provenance": None if kin is None else kin.provenance,
        "units": None if kin is None else kin.units,
        "opensim_id_ready": idr.ready_for_traditional_id,
        "measured_moments": False if asym is None else bool(asym.moments.measured_available),
        "estimated_moments": False if asym is None else bool(asym.moments.estimated_available),
        "scalar_names": [s.name for s in (asym.scalars if asym else [])][:12],
        "figure": str((OUT / f"{stem}_spectral.png").relative_to(ROOT)),
    }


def _synthetic() -> dict:
    from stablewalk.analysis.spectral.report import analyze_spectral
    from stablewalk.analysis.spectral.spectrum import rfft_periodogram, welch_psd
    from stablewalk.analysis.asymmetry3d.report import analyze_asymmetry3d
    from stablewalk.analysis.synthetic_gait import (
        SyntheticGaitSpec,
        known_frequency_vector,
        make_sine_gait_recording,
    )

    fps, f0, n = 50.0, 1.25, 400
    x = known_frequency_vector(n, fps, f0)
    freq, pxx = rfft_periodogram(x, fps)
    band = (freq >= 0.4) & (freq <= 4.0)
    fft_hat = float(freq[band][np.argmax(pxx[band])])
    fw, pw, method = welch_psd(x, fps, nperseg=128, noverlap=64)
    wb = (fw >= 0.4) & (fw <= 4.0)
    welch_hat = float(fw[wb][np.argmax(pw[wb])])

    spec = SyntheticGaitSpec(
        fps=50.0, duration_s=8.0, stride_hz=1.0,
        knee_amp_left_deg=20.0, knee_amp_right_deg=40.0,
    )
    rec = make_sine_gait_recording(spec)
    spectral = analyze_spectral(rec)
    asym = analyze_asymmetry3d(rec)
    knee = asym.scalar_by_name("knee_flexion_rom")
    ac = spectral.primary_autocorr()
    return {
        "fft_peak_hz": fft_hat,
        "fft_error_hz": abs(fft_hat - f0),
        "welch_peak_hz": welch_hat,
        "welch_error_hz": abs(welch_hat - f0),
        "welch_method": method,
        "acf_stride_s": None if ac is None else ac.stride_period_s,
        "session_f0": spectral.features.session_vector.get("dominant_frequency_hz"),
        "session_entropy": spectral.features.session_vector.get("spectral_entropy"),
        "knee_asi": None if knee is None else knee.asi_percent,
        "expected_asi": spec.expected_asi_knee_amp,
    }


def main() -> None:
    payload = {
        "synthetic": _synthetic(),
        "clips": {},
    }
    for stem in ("normal_gait", "abnormal_gait"):
        payload["clips"][stem] = _clip(stem)
    dest = ROOT / "docs" / "_erik_acceptance_dump.json"
    dest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str)[:8000])
    print("wrote", dest)


if __name__ == "__main__":
    main()
