"""Thigh/shank segment angles and their zero-lag Pearson matrix."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from stablewalk.analysis.four_gait_signals import (
    COMPARE_PAIRS,
    FOCUS_PAIRS,
    FOUR_GAIT_CHANNEL_IDS,
    INTERPRETATION,
    LAG_INTERPRETATION,
    MATRIX_INTERPRETATION,
    SPECTRAL_COMPARE_INTERPRETATION,
    compare_acf_and_event_periods,
    event_stride_period_s,
    SPECTRUM_INTERPRETATION,
    XCORR_PAIRS,
    align_psd_shapes,
    analyze_four_gait_signals,
    consensus_gait_period_s,
    correlation_at_lag,
    detect_spectral_peaks,
    pairwise_pearson,
    spectral_shape_correlation,
)
from stablewalk.analysis.spectral.channels import CHANNEL_BY_ID, SEGMENT_SPECS
from stablewalk.analysis.spectral.series import segment_sagittal_tilt_deg
from stablewalk.models.gait_motion import (
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.models.joint_registry import JOINT_IDS, JOINT_PARENTS, ROOT_JOINT_ID
from stablewalk.ui.tk.dashboard_notebook import TAB_MOTION
from stablewalk.ui.tk.playback_active_tab import TAB_PLAYBACK_GROUPS


class _Pos:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


def test_segment_angle_hanging_down_is_zero() -> None:
    hip = _Pos(0.0, 0.0, 0.0)
    knee = _Pos(0.0, -0.4, 0.0)
    assert abs(segment_sagittal_tilt_deg(hip, knee)) < 1e-9


def test_segment_angle_anterior_is_positive() -> None:
    hip = _Pos(0.0, 0.0, 0.0)
    knee = _Pos(0.0, -0.4, 0.2)
    assert segment_sagittal_tilt_deg(hip, knee) > 0.0


def test_pairwise_pearson_identical_is_one() -> None:
    x = np.linspace(-1.0, 1.0, 40)
    corr, n = pairwise_pearson([x, x.copy()])
    assert n[0, 1] == 40
    assert abs(float(corr[0, 1]) - 1.0) < 1e-9


def test_pairwise_pearson_constant_is_nan() -> None:
    x = np.ones(30)
    y = np.linspace(0.0, 1.0, 30)
    corr, n = pairwise_pearson([x, y])
    assert n[0, 1] == 30
    assert not np.isfinite(corr[0, 1])
    assert not np.isfinite(corr[0, 0])
    assert abs(float(corr[1, 1]) - 1.0) < 1e-9


def test_pairwise_pearson_skips_short_or_missing() -> None:
    x = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    corr, n = pairwise_pearson([x, y])
    assert n[0, 1] == 2
    assert not np.isfinite(corr[0, 1])


def test_pairwise_pearson_antiphase_is_negative() -> None:
    t = np.linspace(0.0, 2.0 * math.pi, 80)
    left = np.sin(t)
    right = np.sin(t + math.pi)
    corr, _n = pairwise_pearson([left, right])
    assert float(corr[0, 1]) < -0.95


def _walking_recording(*, fps: float = 50.0, duration_s: float = 4.0) -> GaitMotionRecording:
    n = int(round(duration_s * fps))
    snaps: list[SkeletonSnapshot] = []
    for i in range(n):
        t = i / fps
        phase_l = 2.0 * math.pi * t
        phase_r = phase_l + math.pi
        joints = {}
        for jid in (*JOINT_IDS, ROOT_JOINT_ID):
            side_phase = phase_r if jid.startswith("right_") else phase_l
            if "hip" in jid:
                pos = Vec3(-0.1 if "left" in jid else 0.1, 0.0, 0.0)
            elif "knee" in jid:
                pos = Vec3(
                    -0.1 if "left" in jid else 0.1,
                    -0.40,
                    0.12 * math.sin(side_phase),
                )
            elif "ankle" in jid or "heel" in jid or "toe" in jid or "foot" in jid:
                pos = Vec3(
                    -0.1 if "left" in jid else 0.1,
                    -0.82,
                    0.12 * math.sin(side_phase) + 0.10 * math.sin(side_phase + 0.4),
                )
            else:
                pos = Vec3(0.0, 0.2, 0.0)
            joints[jid] = JointSample(
                joint_id=jid,
                position=pos,
                parent_id=JOINT_PARENTS.get(jid),
            )
        snaps.append(
            SkeletonSnapshot(
                frame_index=i,
                time_s=t,
                joints=joints,
                dofs={},
                metadata={
                    "detected": True,
                    "landmark_visibility": {
                        jid: 0.95 for jid in (*JOINT_IDS, ROOT_JOINT_ID)
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


def test_analyze_four_gait_signals_has_four_channels() -> None:
    report = analyze_four_gait_signals(_walking_recording())
    assert report.available
    assert tuple(report.channel_ids) == FOUR_GAIT_CHANNEL_IDS
    for cid in FOUR_GAIT_CHANNEL_IDS:
        assert cid in CHANNEL_BY_ID
        assert report.filtered[cid].size == report.times_s.size
        assert int(np.isfinite(report.filtered[cid]).sum()) >= 20
    assert report.correlation_filtered.shape == (4, 4)
    # Antiphase L/R thighs should not look like r ≈ +1 at lag 0.
    assert float(report.correlation_filtered[0, 1]) < 0.2
    assert "instability" in INTERPRETATION.lower()
    assert "not a diagnosis" in INTERPRETATION.lower()
    assert "not a diagnosis" in MATRIX_INTERPRETATION.lower()
    assert "c(0)" in MATRIX_INTERPRETATION.lower()
    assert "c(τ)" in MATRIX_INTERPRETATION.lower()
    assert "antiphase" in MATRIX_INTERPRETATION.lower()
    assert "not" in LAG_INTERPRETATION.lower() and "instability" in LAG_INTERPRETATION.lower()
    c0 = np.asarray(report.correlation_filtered, dtype=float)
    np.testing.assert_allclose(c0, c0.T, equal_nan=True, atol=1e-12)
    for i in range(4):
        assert abs(float(c0[i, i]) - 1.0) < 1e-12
    assert set(report.xcorr_filtered) == {name for name, *_rest in XCORR_PAIRS}


def test_focus_pairs_cover_requested_comparisons() -> None:
    labels = {lab for _n, _i, _j, lab in FOCUS_PAIRS}
    assert "Left Thigh vs Right Thigh" in labels
    assert "Left Shank vs Right Shank" in labels
    assert "Left Thigh vs Left Shank" in labels
    assert "Right Thigh vs Right Shank" in labels
    assert len(SEGMENT_SPECS) == 4


def test_motion_tab_includes_four_gait_playback_group() -> None:
    assert "four_gait" in TAB_PLAYBACK_GROUPS[TAB_MOTION]


def test_acf_and_xcorr_use_recording_fps() -> None:
    report = analyze_four_gait_signals(_walking_recording(fps=50.0, duration_s=6.0))
    assert report.available
    assert abs(report.fps - 50.0) < 1e-9
    ac = report.autocorr_filtered.get("seg:left_thigh")
    assert ac is not None
    assert ac.lags_s.size >= 2
    dt = float(ac.lags_s[1] - ac.lags_s[0])
    assert abs(dt - 1.0 / 50.0) < 1e-9
    xc = report.xcorr_filtered.get("thigh_lr")
    assert xc is not None
    assert xc.lags_s[0] < 0.0 < xc.lags_s[-1]
    assert xc.peak_value is not None
    # 1 Hz antiphase L/R → peak lag near half-cycle when a period is reported.
    if report.gait_period_s is not None:
        assert 0.70 <= float(report.gait_period_s) <= 1.30
        assert xc.peak_lag_s is not None
        half = abs(abs(float(xc.peak_lag_s)) - 0.5 * float(report.gait_period_s))
        assert half < 0.20


def test_short_clip_withholds_gait_period() -> None:
    report = analyze_four_gait_signals(_walking_recording(fps=25.0, duration_s=1.2))
    assert report.gait_period_s is None
    assert "withheld" in report.gait_period_reason.lower()


def test_event_stride_is_not_copied_into_acf() -> None:
    cycles = SimpleNamespace(
        events=[
            SimpleNamespace(event_type="left_heel_strike", side="left", time_s=0.20),
            SimpleNamespace(event_type="left_heel_strike", side="left", time_s=1.20),
            SimpleNamespace(event_type="right_heel_strike", side="right", time_s=0.70),
            SimpleNamespace(event_type="right_heel_strike", side="right", time_s=1.70),
        ]
    )
    event_p, n_iv, _reason = event_stride_period_s(cycles)
    assert n_iv == 2
    assert event_p is not None
    assert abs(float(event_p) - 1.0) < 1e-9
    report = analyze_four_gait_signals(
        _walking_recording(fps=25.0, duration_s=1.2), cycles=cycles
    )
    assert report.gait_period_s is None
    assert report.event_stride_period_s is not None
    assert abs(float(report.event_stride_period_s) - 1.0) < 1e-9
    note = report.acf_event_note.lower()
    assert "not used as an acf" in note
    assert "heel-strike" in note
    assert "not used as an acf" in compare_acf_and_event_periods(
        None, 1.04, n_event_intervals=2
    ).lower()
    agree = compare_acf_and_event_periods(1.00, 1.04, n_event_intervals=2).lower()
    assert "agrees" in agree
    disagree = compare_acf_and_event_periods(1.00, 1.80, n_event_intervals=2).lower()
    assert "differs" in disagree
    assert "not replaced" in disagree
    assert "heel-strike" in LAG_INTERPRETATION.lower()


def test_c_lag_zero_matches_r0() -> None:
    report = analyze_four_gait_signals(_walking_recording())
    c0 = correlation_at_lag(report, "filtered", 0)
    np.testing.assert_allclose(c0, report.correlation_filtered, equal_nan=True)
    c_half = correlation_at_lag(report, "filtered", int(round(0.5 * report.fps)))
    assert c_half.shape == (4, 4)


def test_consensus_period_disagreement_is_withheld() -> None:
    from stablewalk.analysis.spectral.autocorrelation import AutocorrResult

    a = AutocorrResult(
        channel_id="a",
        lags_s=np.asarray([0.0]),
        rho=np.asarray([1.0]),
        n_samples=80,
        gait_period_s=0.80,
        reliability_label="moderate",
    )
    b = AutocorrResult(
        channel_id="b",
        lags_s=np.asarray([0.0]),
        rho=np.asarray([1.0]),
        n_samples=80,
        gait_period_s=1.40,
        reliability_label="moderate",
    )
    period, reason = consensus_gait_period_s([a, b])
    assert period is None
    assert "disagree" in reason.lower()


def test_empty_recording_is_unavailable() -> None:
    rec = GaitMotionRecording(
        source="empty",
        fps=25.0,
        snapshots=[],
        source_kind="mock",
    )
    report = analyze_four_gait_signals(rec)
    assert report.available is False


def test_spectrum_uses_full_frame_fps_not_summaries() -> None:
    fps = 50.0
    duration_s = 6.0
    report = analyze_four_gait_signals(_walking_recording(fps=fps, duration_s=duration_s))
    assert report.available
    assert abs(report.fps - fps) < 1e-9
    assert report.times_s.size == int(round(duration_s * fps))
    dt = float(np.median(np.diff(report.times_s)))
    assert abs(dt - 1.0 / fps) < 1e-9
    n_quarter = int(round(duration_s / 0.25))
    assert report.times_s.size > n_quarter * 4
    assert "0.25" in SPECTRUM_INTERPRETATION
    for cid in FOUR_GAIT_CHANNEL_IDS:
        spec = report.spectrum_filtered[cid]
        assert spec.freq_hz.size > 8 or spec.fft_freq_hz.size > 8
        assert spec.dominant_hz is not None
        assert abs(float(spec.dominant_hz) - 1.0) < 0.30
        entropy = spec.spectral_entropy
        assert entropy is not None
        assert 0.0 <= float(entropy) <= 1.0
        power = report.band_power_filtered[cid]
        assert power is not None and float(power) > 0.0
        peaks = report.spectrum_peaks_filtered[cid]
        assert peaks
        assert any(abs(hz - 1.0) < 0.30 for hz, _p in peaks)
        assert spec.method in {"scipy.signal.welch", "numpy_welch", "rfft_periodogram"} or (
            "welch" in spec.method.lower() or "rfft" in spec.method.lower()
        )
        assert spec.band_energy is not None and float(spec.band_energy) > 0.0
        assert spec.quality_status in {"computed", "limited", "insufficient"}
        raw_spec = report.spectrum_raw[cid]
        assert raw_spec.dominant_hz is not None


def test_short_clip_spectrum_quality_warning() -> None:
    report = analyze_four_gait_signals(_walking_recording(fps=25.0, duration_s=1.2))
    assert any(
        "too few" in w.lower()
        or "shorter" in w.lower()
        or "only " in w.lower()
        or "too short" in w.lower()
        or "limited" in w.lower()
        for w in report.warnings
    )
    spec = report.spectrum_filtered.get("seg:left_thigh")
    assert spec is not None
    assert spec.notes
    assert spec.quality_status == "insufficient"
    assert spec.dominant_hz is None
    assert report.gait_period_s is None


def test_demo_length_clip_does_not_invent_frequency_results() -> None:
    report = analyze_four_gait_signals(_walking_recording(fps=30.0, duration_s=4.76))
    assert any("limited" in w.lower() for w in report.warnings)
    assert abs(float(report.times_s[-1] - report.times_s[0]) - 4.76) < 0.08
    spec = report.spectrum_filtered["seg:left_thigh"]
    assert spec.quality_status in {"limited", "insufficient"}
    assert spec.quality_status != "computed"
    assert spec.freq_hz.size > 4 or spec.fft_freq_hz.size > 4
    assert spec.spectral_entropy is not None
    assert spec.band_energy is not None and float(spec.band_energy) > 0.0
    joined = " ".join(spec.notes + report.warnings).lower()
    assert "bin spacing" in joined or "descriptive" in joined
    assert "not a confidence" in joined or "not invented" in joined
    if spec.quality_status == "insufficient":
        assert spec.dominant_hz is None
    coh = report.coherence_filtered["thigh_lr"]
    assert coh.sufficient is False
    if report.gait_period_s is None:
        assert report.gait_period_reason


def test_long_clip_uses_shared_welch_bins_and_recovers_frequency() -> None:
    report = analyze_four_gait_signals(_walking_recording(fps=50.0, duration_s=16.0))
    grids = []
    for cid in FOUR_GAIT_CHANNEL_IDS:
        spec = report.spectrum_filtered[cid]
        assert spec.dominant_hz is not None
        assert abs(float(spec.dominant_hz) - 1.0) < 0.20
        assert spec.quality_status in {"computed", "limited"}
        assert spec.band_energy is not None
        grids.append(np.asarray(spec.freq_hz if spec.freq_hz.size else spec.fft_freq_hz))
    ref = grids[0]
    for g in grids[1:]:
        assert g.size == ref.size
        np.testing.assert_allclose(g, ref, atol=1e-9)
    corr = np.asarray(report.spectral_corr_filtered)
    np.testing.assert_allclose(corr, corr.T, equal_nan=True, atol=1e-10)
    for i in range(4):
        assert abs(float(corr[i, i]) - 1.0) < 1e-6
    assert report.coherence_filtered["thigh_lr"].sufficient
    assert report.coherence_filtered["left_intra"].sufficient


def test_irregular_timestamps_are_resampled_before_fft() -> None:
    rec = _walking_recording(fps=50.0, duration_s=6.0)
    for i, snap in enumerate(rec.snapshots):
        snap.time_s = i / 50.0 + 0.006 * (1.0 if i % 2 else -1.0)
    report = analyze_four_gait_signals(rec)
    dt = float(np.median(np.diff(report.times_s)))
    assert abs(dt - 1.0 / 50.0) < 1e-9
    spec = report.spectrum_filtered["seg:left_thigh"]
    assert spec.dominant_hz is not None
    assert abs(float(spec.dominant_hz) - 1.0) < 0.35

    spaced = _walking_recording(fps=50.0, duration_s=6.0)
    for i, snap in enumerate(spaced.snapshots):
        snap.time_s = i / 35.0
    spaced_report = analyze_four_gait_signals(spaced)
    assert any(
        "resampl" in w.lower() or "spacing" in w.lower()
        for w in spaced_report.warnings
    )
    spaced_dt = float(np.median(np.diff(spaced_report.times_s)))
    assert abs(spaced_dt - 1.0 / 50.0) < 1e-9


def test_spectrum_figure_uses_hz_axis() -> None:
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.four_gait_charts import (
        build_signal_axes,
        render_four_gait_spectrum_figure,
    )

    report = analyze_four_gait_signals(_walking_recording(fps=50.0, duration_s=6.0))
    fig = Figure()
    axes = build_signal_axes(fig, combined=False)
    assert len(axes) == 4
    render_four_gait_spectrum_figure(fig, axes, report, "filtered")
    titles = " ".join(ax.get_title() or "" for ax in axes)
    assert "Left Thigh" in titles
    assert "Right Thigh" in titles
    assert "Left Shank" in titles
    assert "Right Shank" in titles
    for ax in axes:
        assert "Hz" in (ax.get_xlabel() or "")
        assert ax.lines
        assert any("f" in (t.get_text() or "") for t in ax.texts)


def test_spectral_shape_r_is_not_time_domain_r() -> None:
    report = analyze_four_gait_signals(_walking_recording(fps=50.0, duration_s=6.0))
    time_r = float(report.correlation_filtered[0, 1])
    shape_r = float(report.spectral_corr_filtered[0, 1])
    assert time_r < 0.2
    assert shape_r > 0.7
    assert "not time-domain" in SPECTRAL_COMPARE_INTERPRETATION.lower()
    assert "do not prove symmetric gait" in SPECTRAL_COMPARE_INTERPRETATION.lower()
    for i in range(4):
        assert abs(float(report.spectral_corr_filtered[i, i]) - 1.0) < 1e-6


def test_dominant_frequency_delta_hz_and_percent() -> None:
    report = analyze_four_gait_signals(_walking_recording(fps=50.0, duration_s=6.0))
    names = {name for name, _a, _b, _lab in COMPARE_PAIRS}
    assert names == {"thigh_lr", "shank_lr", "left_intra", "right_intra"}
    for name, _a, _b, _lab in COMPARE_PAIRS:
        cmp = report.freq_compare_filtered[name]
        assert cmp.f_a_hz is not None and cmp.f_b_hz is not None
        assert cmp.delta_hz is not None
        assert cmp.delta_pct is not None
        assert abs(float(cmp.delta_hz)) < 0.30
        assert abs(float(cmp.delta_pct)) < 35.0
        assert any("not proof of symmetric" in n.lower() for n in cmp.notes)


def test_coherence_supported_on_long_clip_withheld_on_short() -> None:
    long = analyze_four_gait_signals(_walking_recording(fps=50.0, duration_s=8.0))
    thigh = long.coherence_filtered["thigh_lr"]
    assert thigh.sufficient
    assert thigh.freq_hz.size > 4
    assert thigh.mean_band is not None
    assert 0.0 <= float(thigh.mean_band) <= 1.0
    assert "cross-correlation" in " ".join(thigh.notes).lower()

    short = analyze_four_gait_signals(_walking_recording(fps=25.0, duration_s=1.2))
    short_c = short.coherence_filtered.get("thigh_lr")
    assert short_c is not None
    assert short_c.sufficient is False
    assert any("withheld" in w.lower() or "segment" in w.lower() or "few" in w.lower()
               for w in short.warnings + short_c.notes)


def test_compare_figure_uses_hz_and_has_pair_overlays() -> None:
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.four_gait_charts import (
        build_compare_axes,
        render_four_gait_compare_figure,
    )

    report = analyze_four_gait_signals(_walking_recording(fps=50.0, duration_s=8.0))
    fig = Figure()
    axes = build_compare_axes(fig)
    assert len(axes) == 5
    render_four_gait_compare_figure(fig, axes, report, "filtered")
    heat_title = (axes[0].get_title() or "").lower()
    assert "shape r" in heat_title
    assert axes[0].images
    overlay_titles = " ".join(ax.get_title() or "" for ax in axes[1:])
    assert "Left Thigh vs Right Thigh" in overlay_titles
    assert "Left Shank vs Right Shank" in overlay_titles
    assert "Left Thigh vs Left Shank" in overlay_titles
    assert "Right Thigh vs Right Shank" in overlay_titles
    for ax in axes[1:]:
        assert "Hz" in (ax.get_xlabel() or "")
        assert ax.lines


def test_limited_clip_draws_descriptive_spectra_not_blank() -> None:
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.four_gait_charts import (
        build_compare_axes,
        build_signal_axes,
        render_four_gait_compare_figure,
        render_four_gait_spectrum_figure,
    )

    report = analyze_four_gait_signals(_walking_recording(fps=25.0, duration_s=4.76))
    spec = report.spectrum_filtered["seg:left_thigh"]
    assert spec.quality_status in {"limited", "insufficient"}
    assert spec.quality_status != "computed"
    assert spec.freq_hz.size > 4 or spec.fft_freq_hz.size > 4
    assert spec.spectral_entropy is not None
    assert spec.band_energy is not None
    assert report.coherence_filtered["thigh_lr"].sufficient is False
    assert any("limited" in w.lower() for w in report.warnings)

    fig = Figure()
    axes = build_signal_axes(fig, combined=False)
    render_four_gait_spectrum_figure(fig, axes, report, "filtered")
    assert len(axes) == 4
    for ax in axes:
        assert ax.lines or ax.texts
        assert "Hz" in (ax.get_xlabel() or "")

    fig2 = Figure()
    caxes = build_compare_axes(fig2)
    render_four_gait_compare_figure(fig2, caxes, report, "filtered")
    assert caxes[0].images
    texts = " ".join(t.get_text() for ax in caxes[1:] for t in ax.texts)
    assert "MSC withheld" in texts
    for ax in caxes[1:]:
        assert ax.lines or ax.texts


def test_spectral_shape_r_matches_pearson_of_aligned_shapes() -> None:
    report = analyze_four_gait_signals(_walking_recording(fps=50.0, duration_s=8.0))
    spectra = report.spectrum_filtered
    _grid, shapes = align_psd_shapes(spectra)
    corr, _n, _g = spectral_shape_correlation(spectra)
    expected, _counts = pairwise_pearson(shapes)
    np.testing.assert_allclose(corr, expected, equal_nan=True, atol=1e-12)
    time_r = float(report.correlation_filtered[0, 1])
    shape_r = float(corr[0, 1])
    assert np.isfinite(shape_r)
    assert abs(shape_r - time_r) > 0.3


def test_zero_lag_matrix_is_symmetric_with_unit_diagonal() -> None:
    report = analyze_four_gait_signals(_walking_recording(fps=50.0, duration_s=4.0))
    assert report.available
    c0 = np.asarray(report.correlation_filtered, dtype=float)
    assert c0.shape == (4, 4)
    np.testing.assert_allclose(c0, c0.T, equal_nan=True, atol=1e-12)
    for i in range(4):
        assert abs(float(c0[i, i]) - 1.0) < 1e-12
        assert -1.0 - 1e-9 <= float(c0[i, 1 if i == 0 else 0]) <= 1.0 + 1e-9
    # Opposing thighs are antiphase in the synthetic walk.
    assert float(c0[0, 1]) < 0.0
    assert float(c0[2, 3]) < 0.0
    c0_again = correlation_at_lag(report, "filtered", 0)
    np.testing.assert_allclose(c0_again, c0, equal_nan=True)


def test_ctau_differs_from_zero_lag_at_half_cycle() -> None:
    report = analyze_four_gait_signals(_walking_recording(fps=50.0, duration_s=4.0))
    c0 = correlation_at_lag(report, "filtered", 0)
    half = int(round(0.5 * report.fps))
    ctau = correlation_at_lag(report, "filtered", half)
    assert ctau.shape == (4, 4)
    assert np.isfinite(c0[0, 1]) and np.isfinite(ctau[0, 1])
    assert float(c0[0, 1]) < 0.0
    assert float(ctau[0, 1]) > float(c0[0, 1]) + 0.4


def test_insufficient_pairs_are_not_invented() -> None:
    x = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    corr, n = pairwise_pearson([x, y])
    assert n[0, 1] == 2
    assert not np.isfinite(corr[0, 1])
    const = np.ones(20)
    vary = np.linspace(-1.0, 1.0, 20)
    corr_c, _n = pairwise_pearson([const, vary])
    assert not np.isfinite(corr_c[0, 0])
    assert not np.isfinite(corr_c[0, 1])
    assert abs(float(corr_c[1, 1]) - 1.0) < 1e-12


def test_matrix_figure_has_title_and_r_labels() -> None:
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.four_gait_charts import (
        build_matrix_axes,
        draw_correlation_heatmap,
        render_four_gait_matrix_figure,
    )

    report = analyze_four_gait_signals(_walking_recording(fps=50.0, duration_s=4.0))
    fig = Figure()
    axes = build_matrix_axes(fig)
    assert len(axes) == 1
    render_four_gait_matrix_figure(fig, axes, report, "filtered", lag_samples=12)
    title = axes[0].get_title() or ""
    assert "zero-lag" in title.lower()
    assert "c(0)" in title.lower()
    assert "c(τ)" not in title.lower()
    labels = [tick.get_text() for tick in axes[0].get_xticklabels()]
    assert labels == ["Left Thigh", "Right Thigh", "Left Shank", "Right Shank"]
    texts = [t.get_text() for t in axes[0].texts]
    assert any(txt.startswith("-") or txt[0].isdigit() for txt in texts)
    patches = [p for p in axes[0].patches if p.get_linewidth() >= 1.5]
    assert len(patches) >= 4
    lag_fig = Figure()
    lag_ax = lag_fig.add_subplot(111)
    draw_correlation_heatmap(lag_ax, report, "filtered", lag_samples=10, zero_lag=False)
    lag_title = lag_ax.get_title() or ""
    assert "c(τ)" in lag_title.lower()
    assert "zero-lag" not in lag_title.lower()


def test_corr_figure_has_four_acf_and_lr_xcorr_axes() -> None:
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.four_gait_charts import (
        build_corr_axes,
        render_four_gait_corr_figure,
    )

    report = analyze_four_gait_signals(_walking_recording(fps=50.0, duration_s=6.0))
    fig = Figure()
    axes = build_corr_axes(fig)
    assert len(axes) == 6
    render_four_gait_corr_figure(fig, axes, report, "filtered")
    for ax in axes:
        assert "τ" in (ax.get_xlabel() or "") or "Lag" in (ax.get_xlabel() or "")
    acf_titles = " ".join(ax.get_title() or "" for ax in axes[:4])
    assert "Left Thigh" in acf_titles
    assert "Right Shank" in acf_titles
    xcorr_titles = " ".join(ax.get_title() or "" for ax in axes[4:])
    assert "Left Thigh vs Right Thigh" in xcorr_titles
    assert "Left Shank vs Right Shank" in xcorr_titles
    assert "Left Thigh vs Left Shank" not in xcorr_titles
    thigh = report.xcorr_filtered["thigh_lr"]
    assert thigh.peak_lag_s is not None
    assert abs(float(thigh.peak_lag_s)) > 0.15


def test_four_gait_csv_export_has_signals_and_matrices(tmp_path) -> None:
    from stablewalk.analysis.four_gait_export import export_four_gait_csv

    report = analyze_four_gait_signals(_walking_recording(fps=50.0, duration_s=4.0))
    dest = export_four_gait_csv(report, tmp_path / "gait.csv", mode="filtered")
    text = dest.read_text(encoding="utf-8")
    assert "# signals_deg" in text
    assert "# correlation_r0" in text
    assert "# xcorr_peaks" in text
    assert "Left Thigh vs Right Thigh" in text
    assert "Left Thigh vs Left Shank" in text
    assert "# spectral_shape_correlation" in text
    assert "# frequency_metrics" in text
    assert "time_s" in text
    assert "not a clinical diagnosis" in text
    assert "not C(lag)" in text or "not spectral shape" in text
