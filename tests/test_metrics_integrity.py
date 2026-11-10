"""Scientific correctness and data-integrity checks for reported session metrics.

Does not retune normal_gait.mp4 to look better. Numbers must match formulas.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.analysis_summary import (
    HIGH_CONFIDENCE_DEFINITION,
    HIGH_MIN_CYCLES,
    HIGH_MIN_DURATION_S,
    _analysis_confidence,
    build_analysis_summary,
)
from stablewalk.analysis.angle_filter import despike_angle_series
from stablewalk.analysis.biomechanical.gait_quality_score import compute_gait_quality_score
from stablewalk.analysis.biomechanical.orchestrator import run_biomechanical_analysis
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.ui.overview_clinical_summary import (
    extract_confidence_visualization,
    primary_finding_inputs,
)


def _autosave_pose() -> Path | None:
    video = config.DEMO_VIDEOS_DIR / "normal_gait.mp4"
    if not video.is_file():
        return None
    named = config.POSES_DIR / "normal_gait_poses.json"
    if named.is_file():
        return named
    autosave = config.SESSION_EXPORT_DIR / "autosave" / "stablewalk_autosave" / "poses.json"
    if not autosave.is_file():
        return None
    meta = config.SESSION_EXPORT_DIR / "autosave" / "stablewalk_autosave" / "session_metadata.json"
    if meta.is_file():
        data = json.loads(meta.read_text(encoding="utf-8"))
        src = str(data.get("video_source") or "")
        if video.name not in src.replace("\\", "/"):
            return None
    return autosave


def _load_normal_gait():
    path = _autosave_pose()
    if path is None:
        pytest.skip("No pose JSON for normal_gait.mp4")
    seq = load_pose_sequence(path)
    rec = pose_sequence_to_gait_motion(seq)
    return seq, rec


def test_high_confidence_definition_is_documented() -> None:
    assert HIGH_MIN_CYCLES == 3
    assert HIGH_MIN_DURATION_S == 6.0
    assert "pose tracking" in HIGH_CONFIDENCE_DEFINITION.lower()


def test_two_cycles_and_short_clip_cannot_be_high_confidence() -> None:
    ba = SimpleNamespace(
        video_quality=SimpleNamespace(overall_quality_score=95.0),
        gait_quality=SimpleNamespace(confidence=0.95),
        stability_margin=SimpleNamespace(reportable=False),
    )
    cycles = SimpleNamespace(
        metrics=SimpleNamespace(
            gait_cycle_count=2,
            contact_confidence=1.0,
            confidence_tier="HIGH",
        ),
        events=[SimpleNamespace(time_s=0.04), SimpleNamespace(time_s=4.68)],
        cycles=[],
    )
    contact = SimpleNamespace(metrics=SimpleNamespace(contact_confidence=1.0))
    level, detail, _score = _analysis_confidence(ba, cycles, contact)
    assert level != "High"
    assert level == "Moderate"
    assert "cycle" in detail.lower()
    assert "pose tracking" in detail.lower()
    assert "95" in detail


def test_despike_flags_isolated_jump_and_keeps_sustained_change() -> None:
    series = np.array([10.0, 12.0, 95.0, 13.0, 14.0], dtype=float)
    cleaned, n = despike_angle_series(series, max_jump_deg=40.0, return_n=True)
    assert n == 1
    assert cleaned[2] == pytest.approx(12.5)
    limp = np.array([10.0, 40.0, 55.0, 50.0, 20.0], dtype=float)
    kept, n_kept = despike_angle_series(limp, max_jump_deg=40.0, return_n=True)
    assert n_kept == 0
    assert kept[2] == pytest.approx(55.0)


def test_gait_quality_unavailable_is_not_zero() -> None:
    gq = compute_gait_quality_score()
    assert gq.available is False
    assert gq.score is None


def test_normal_gait_metrics_are_reproducible_and_formula_backed() -> None:
    seq, rec = _load_normal_gait()
    assert rec.fps == pytest.approx(25.0, abs=0.6)
    assert rec.frame_count == 120

    first = run_biomechanical_analysis(rec, seq)
    second = run_biomechanical_analysis(rec, seq)
    c1 = first.cycles.metrics.cadence_steps_per_min if first.cycles else None
    c2 = second.cycles.metrics.cadence_steps_per_min if second.cycles else None
    assert c1 is not None and c2 is not None
    assert c1 == pytest.approx(c2, rel=0, abs=1e-9)
    s1 = first.symmetry.overall_symmetry_pct.value if first.symmetry else None
    s2 = second.symmetry.overall_symmetry_pct.value if second.symmetry else None
    assert s1 is not None and s2 is not None
    assert s1 == pytest.approx(s2, rel=0, abs=1e-9)
    assert first.gait_quality.score == pytest.approx(second.gait_quality.score, abs=1e-9)

    events = list(first.cycles.events)
    left_hs = [e for e in events if e.event_type == "left_heel_strike"]
    right_hs = [e for e in events if e.event_type == "right_heel_strike"]
    assert len(left_hs) >= 2 and len(right_hs) >= 2
    # Cadence is step frequency (steps/min), not stride frequency (strides/min).
    stride = first.cycles.metrics.stride_time_s
    step = first.cycles.metrics.step_time_s
    cadence = first.cycles.metrics.cadence_steps_per_min
    assert stride is not None and cadence is not None
    if step and step > 1e-6:
        assert cadence == pytest.approx(60.0 / step, rel=1e-6)
    else:
        assert cadence == pytest.approx(120.0 / stride, rel=1e-6)
    stride_freq_per_min = 60.0 / stride
    assert cadence == pytest.approx(2.0 * stride_freq_per_min, rel=0.08)

    # Short recording: 120 frames at 25 FPS ≈ 4.8 s, 2 cycles — thin evidence.
    assert first.cycles.metrics.gait_cycle_count == 2
    duration = rec.snapshots[-1].time_s - rec.snapshots[0].time_s
    assert duration < HIGH_MIN_DURATION_S

    summary = build_analysis_summary(
        source="normal_gait.mp4",
        biomechanical=first,
        estimated_vgrf=None,
        contact=first.contact,
        cycles=first.cycles,
    )
    assert summary.analysis_confidence.value != "High"
    assert summary.stability_margin.available is False
    assert summary.stability_margin.value == "Not available"
    assert "5.83" not in (summary.stability_margin.value or "")
    assert first.stability_margin is not None
    assert first.stability_margin.reportable is False

    overall = first.symmetry.overall_symmetry_pct
    assert overall is not None and overall.unit == "%"
    assert "weighted" in (overall.note or "").lower()
    explain = (first.gait_quality.explanation or "").lower()
    assert "composite" in explain
    assert "diagnosis" not in explain

    plane = getattr(first.cycles, "ground_plane", None) if first.cycles else None
    assert plane is not None
    assert getattr(plane, "scale_mode", "body_normalized") == "body_normalized"
    fc = first.features.features if first.features else None
    if fc is not None and getattr(fc, "foot_clearance_left_m", None) is not None:
        assert fc.foot_clearance_left_m >= 0.0
        # Body-normalized metres converted ×100 for display — not tape-measure cm.


def test_overview_does_not_pass_unreportable_bos_into_findings() -> None:
    gui = SimpleNamespace(
        _biomech_analysis=SimpleNamespace(
            gait_quality=None,
            symmetry=None,
            gait_metrics=None,
            stability_margin=SimpleNamespace(
                reportable=False,
                stable_pct=5.83,
                mean_margin_m=0.008,
            ),
            video_quality=None,
        ),
        _gait_cycle=None,
        _stability_result=None,
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (2, 2),
    )
    inputs = primary_finding_inputs(gui)
    assert inputs["stable_pct"] is None
    assert inputs["stability_margin_m"] is None


def test_overview_confidence_two_cycles_is_not_high() -> None:
    ba = SimpleNamespace(
        video_quality=SimpleNamespace(
            checks={"pose_confidence_score": 95.0},
            overall_quality_score=95.0,
        ),
        gait_quality=SimpleNamespace(confidence=0.95, score=84.0, components={}),
        stability_margin=SimpleNamespace(
            reportable=False, valid_frame_ratio=1.0, stable_pct=5.83
        ),
    )
    ac = SimpleNamespace(
        level="HIGH",
        completeness_pct=57.0,
        evidence_index=0.74,
        pose_valid_frame_pct=100.0,
    )
    gui = SimpleNamespace(
        _biomech_analysis=ba,
        _stability_result=SimpleNamespace(
            gait_summary=SimpleNamespace(analysis_confidence=ac),
            usable_gait_cycles=2,
        ),
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (2, 2),
    )
    conf = extract_confidence_visualization(gui)
    assert conf["display"] != "High"
    assert conf["pose_level"] == "Landmark visibility high"
    names = " ".join(conf.get("signal_names") or [])
    assert "tracking" not in names


def test_new_video_reset_clears_dependent_reports() -> None:
    text = (
        Path(__file__).resolve().parents[1] / "stablewalk/ui/tk/app.py"
    ).read_text(encoding="utf-8")
    reset = text.split("def _reset_gait_cycle_panel", 1)[1].split("def ", 1)[0]
    assert "self._spectral_report = None" in reset
    assert "self._asymmetry_report = None" in reset
    assert "self._analysis_summary_cache = None" in reset
    assert "self._biomech_analysis = None" in reset


def test_cadence_events_match_recomputed_cycles() -> None:
    _seq, rec = _load_normal_gait()
    a = analyze_gait_cycles(rec)
    b = analyze_gait_cycles(rec)
    assert [(e.event_type, round(e.time_s, 4), e.side) for e in a.events] == [
        (e.event_type, round(e.time_s, 4), e.side) for e in b.events
    ]
    assert a.metrics.cadence_steps_per_min == pytest.approx(
        b.metrics.cadence_steps_per_min, abs=1e-12
    )
