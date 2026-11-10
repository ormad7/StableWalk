"""Shared gait-timing model: session vs channel, estimated vs detected."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.analysis.gait_timing import compose_gait_timing, merge_gait_timing
from stablewalk.ui.tk.dashboard_spectral import _provenance_meaning


def test_compose_keeps_session_channel_estimated_and_detected_distinct() -> None:
    acf = SimpleNamespace(
        channel_id="dof:left_knee_flexion",
        stride_period_s=1.001,
        step_period_s=0.542208,
        reliability_label="low",
        stride_assignment_note="exclusive-window peak",
        notes=["Cadence = 60 / T_step from the ACF step-lag window."],
        cadence_assumes_two_steps_per_stride=False,
    )
    cycles = SimpleNamespace(
        cycles=[
            SimpleNamespace(duration_s=1.418),
            SimpleNamespace(duration_s=0.417),
            SimpleNamespace(duration_s=2.920),
        ],
        metrics=SimpleNamespace(
            cadence_steps_per_min=None,
            step_time_s=None,
            stride_time_s=1.418,
        ),
    )
    timing = compose_gait_timing(
        duration_s=27.027,
        fps=23.976,
        channel_acf=acf,
        session_period_s=1.584917,
        session_reason="",
        session_channel_ids=("seg:left_shank", "seg:right_shank"),
        event_stride_s=1.418,
        event_stride_n=1,
        cycles=cycles,
    )
    assert timing.session_stride.kind == "session_acf_stride"
    assert timing.channel_stride.kind == "channel_acf_stride"
    assert timing.channel_step.kind == "channel_acf_step"
    assert timing.event_stride.kind == "event_hs_hs"
    assert timing.session_stride.value_s == 1.584917
    assert timing.channel_stride.value_s == 1.001
    assert timing.channel_step.value_s == 0.542208
    assert timing.cadence_spectral.method == "sixty_over_step"
    assert timing.cadence_spectral.assumes_two_steps_per_stride is False
    assert abs(float(timing.cadence_spectral.steps_per_min) - 60.0 / 0.542208) < 1e-6
    for estimate in (
        timing.session_stride,
        timing.channel_stride,
        timing.channel_step,
        timing.event_stride,
    ):
        prov = estimate.provenance
        assert prov.source_signal
        assert prov.time_window_s == (0.0, 27.027)
        assert prov.preprocessing
        assert prov.peak_selection
        assert prov.confidence_label
    assert "left_knee" in timing.channel_stride.provenance.source_signal
    assert "local maximum" in timing.channel_stride.provenance.peak_selection.lower()
    assert "60 / T_step" in timing.cadence_spectral.provenance.peak_selection
    assert timing.cadence_spectral.provenance.source_signal == timing.channel_step.provenance.source_signal
    assert abs(float(timing.session_stride.value_s) - float(timing.channel_stride.value_s)) > 0.2
    assert any("differs" in n.lower() for n in timing.consistency_notes)
    est = timing.cycles.estimated_from_channel_stride
    assert est is not None
    assert abs(est - 27.027 / 1.001) < 1e-9
    assert abs(float(timing.cycles.estimated_from_session_stride) - 27.027 / 1.584917) < 1e-9
    assert timing.cycles.detected == 3
    assert timing.cycles.valid == 3
    assert timing.cycles.estimated_from_channel_stride != float(timing.cycles.detected)
    assert any("not half" in n.lower() or "ratio" in n.lower() for n in timing.consistency_notes)


def test_cadence_fallback_is_labeled_two_steps_assumption() -> None:
    acf = SimpleNamespace(
        channel_id="seg:left_thigh",
        stride_period_s=1.0,
        step_period_s=None,
        reliability_label="moderate",
        stride_assignment_note="",
        notes=[],
    )
    timing = compose_gait_timing(duration_s=8.0, fps=50.0, channel_acf=acf)
    assert timing.cadence_spectral.method == "one_twenty_over_stride_assumes_two_steps"
    assert timing.cadence_spectral.assumes_two_steps_per_stride is True
    assert abs(float(timing.cadence_spectral.steps_per_min) - 120.0) < 1e-9


def test_merge_prefers_four_gait_session_and_spectral_channel() -> None:
    spectral = SimpleNamespace(
        primary_autocorr=lambda: SimpleNamespace(
            channel_id="dof:left_knee_flexion",
            stride_period_s=1.001,
            step_period_s=0.542,
            reliability_label="low",
            stride_assignment_note="exclusive",
            notes=[],
        ),
        timing=SimpleNamespace(duration_s=27.027, fps=23.976, session_stride=SimpleNamespace(value_s=None), event_stride=SimpleNamespace(value_s=None)),
    )
    four = SimpleNamespace(
        gait_period_s=1.585,
        gait_period_reason="",
        gait_period_channel_ids=("seg:left_shank", "seg:right_shank"),
        event_stride_period_s=1.418,
        event_stride_n=1,
        acf_event_note="ACF vs event within 25%",
        times_s=None,
        fps=23.976,
        timing=None,
    )
    merged = merge_gait_timing(spectral, four, None)
    assert merged is not None
    assert merged.session_stride.value_s == 1.585
    assert merged.channel_stride.value_s == 1.001
    assert merged.channel_stride.channel_id == "dof:left_knee_flexion"
    assert merged.event_stride.value_s == 1.418


def test_provenance_meaning_exposes_source_window_method_and_confidence() -> None:
    timing = compose_gait_timing(
        duration_s=27.027,
        fps=23.976,
        channel_acf=SimpleNamespace(
            channel_id="dof:left_knee_flexion",
            stride_period_s=1.001,
            step_period_s=0.542,
            reliability=0.42,
            reliability_label="low",
            preprocessing="z-scored processed channel",
            peak_selection="positive local maximum of ρ(τ)",
            time_window_s=(0.0, 27.027),
            notes=[],
        ),
        session_period_s=1.585,
        session_channel_ids=("seg:left_shank", "seg:right_shank"),
    )
    text = _provenance_meaning("Channel stride.", timing.channel_stride.provenance)
    assert "Source: dof:left_knee_flexion" in text
    assert "Time window: 0.000–27.027 s" in text
    assert "Preprocess:" in text
    assert "Peak selection:" in text
    assert "Confidence: 0.42 (low)" in text
    differ = " ".join(timing.consistency_notes).lower()
    assert "differs" in differ
    assert "1.585" in differ or "1.001" in differ
    assert timing.cycles.notes[0].lower().startswith("estimated cycle count")
    assert "detected" in timing.cycles.notes[1].lower()
    assert "hs" in timing.cycles.notes[1].lower()
