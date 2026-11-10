"""Tests for the Compare workflow validation and presentation rules."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from stablewalk.analysis.session_compare import CompareMetrics, compare_session_metrics
from stablewalk.ui.tk.compare_workflow import (
    PRIMARY_COMPARE_METRICS,
    CompareUiState,
    build_cached_comparison,
    build_comparison_rows,
    build_metric_row,
    candidate_from_bundle_metadata,
    comparison_fingerprint,
    is_notable_difference,
    joint_curves_ready,
    notable_difference_label,
    snapshot_is_analyzed,
    validate_compare_slots,
)
from stablewalk.ui.tk.session_cache import (
    AnalyzedSessionSnapshot,
    SessionCompareCache,
)


def _snap(
    key: str,
    *,
    n_frames: int = 30,
    gait_quality: float | None = 70.0,
    cadence: float | None = 100.0,
    speed: float | None = 1.1,
    confidence: str = "derived",
    fps: float = 30.0,
    knee_y: list | None = None,
) -> AnalyzedSessionSnapshot:
    return AnalyzedSessionSnapshot(
        key=key,
        label=key.title(),
        source=f"{key}.mp4",
        video_path=f"{key}.mp4",
        fps=fps,
        n_frames=n_frames,
        sequence=object(),  # truthy for ready
        metrics=CompareMetrics(
            label=key.title(),
            session_key=key,
            gait_quality=gait_quality,
            cadence_spm=cadence,
            walking_speed_m_s=speed,
            symmetry_pct=80.0,
            stability_margin_m=0.04,
            knee_rom_deg=55.0,
            duration_s=n_frames / fps if fps else None,
        ),
        knee_y=knee_y if knee_y is not None else [10.0, 20.0, 30.0, 25.0],
        confidence_label=confidence,
        valid_frame_count=n_frames,
        pipeline_complete=True,
    )


def test_same_session_selected_twice_blocks_compare() -> None:
    cache = SessionCompareCache()
    snap = _snap("normal")
    cache.pin(snap)
    cache.set_slot("left", "normal")
    cache.set_slot("right", "normal")
    v = validate_compare_slots(cache)
    assert v.state == CompareUiState.IDENTICAL
    assert v.can_reload is False
    assert v.can_compare is False
    assert "same session" in v.message.lower() or "identical" in v.message.lower()


def test_missing_session_b_offers_analyze_action() -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.set_slot("left", "normal")
    cache.set_slot("right", "abnormal")
    v = validate_compare_slots(cache)
    assert v.state == CompareUiState.INCOMPLETE
    assert v.left_ready is True
    assert v.right_ready is False
    assert v.action == "analyze_session_b"
    assert "Analyze Session B" in v.action_label
    assert v.can_reload is False


def test_different_frame_rates_still_comparable() -> None:
    cache = SessionCompareCache()
    a = _snap("normal", fps=24.0, n_frames=48)
    b = _snap("abnormal", fps=60.0, n_frames=120, gait_quality=50.0)
    cache.pin(a)
    cache.pin(b)
    cache.set_slot("left", "normal")
    cache.set_slot("right", "abnormal")
    v = validate_compare_slots(cache)
    assert v.state == CompareUiState.READY
    assert v.can_compare is True
    assert v.can_reload is True


def test_different_video_lengths_still_comparable() -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("normal", n_frames=20))
    cache.pin(_snap("abnormal", n_frames=200, gait_quality=55.0))
    cache.set_slot("left", "normal")
    cache.set_slot("right", "abnormal")
    assert validate_compare_slots(cache).can_compare is True
    cached = build_cached_comparison(cache)
    assert cached.result is not None
    assert len(cached.primary_rows) == len(PRIMARY_COMPARE_METRICS)


def test_unavailable_speed_shows_reason_not_bare_na() -> None:
    left = CompareMetrics(label="A", walking_speed_m_s=None, gait_quality=70.0)
    right = CompareMetrics(label="B", walking_speed_m_s=1.2, gait_quality=60.0)
    row = build_metric_row("Speed", left, right)
    assert row.available_both is False
    assert row.delta is None
    assert row.display_delta == "—"
    assert "Unavailable" not in row.reason
    assert "scale" in row.reason.lower() or "reportable" in row.reason.lower()
    # Both sessions exist; a missing speed is a metric gap, not a missing session.
    assert "Missing in Session A" not in row.reason


def test_low_confidence_session_still_valid_when_kpis_present() -> None:
    cache = SessionCompareCache()
    low = _snap("normal", confidence="low")
    other = _snap("abnormal", gait_quality=40.0)
    cache.pin(low)
    cache.pin(other)
    cache.set_slot("left", "normal")
    cache.set_slot("right", "abnormal")
    ok, _ = snapshot_is_analyzed(low)
    assert ok is True
    assert validate_compare_slots(cache).can_compare is True


def test_partially_completed_pipeline_is_incomplete() -> None:
    snap = AnalyzedSessionSnapshot(
        key="partial",
        label="Partial",
        n_frames=10,
        sequence=object(),
        metrics=CompareMetrics(label="Partial"),  # no primary KPIs
        pipeline_complete=False,
    )
    ok, reason = snapshot_is_analyzed(snap)
    assert ok is False
    assert "incomplete" in reason.lower() or "kpi" in reason.lower()

    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.pin(snap)
    cache.set_slot("left", "normal")
    cache.set_slot("right", "partial")
    v = validate_compare_slots(cache)
    assert v.state == CompareUiState.INCOMPLETE
    assert v.action == "analyze_session_b"


def test_delta_is_b_minus_a() -> None:
    left = CompareMetrics(label="A", gait_quality=80.0, cadence_spm=100.0)
    right = CompareMetrics(label="B", gait_quality=60.0, cadence_spm=110.0)
    row = build_metric_row("Gait Quality", left, right)
    assert row.delta == pytest.approx(-20.0)
    cad = build_metric_row("Cadence", left, right)
    assert cad.delta == pytest.approx(10.0)
    assert cad.directionality == "higher_better"
    assert cad.tone == "better"  # B higher cadence = improvement
    assert is_notable_difference("Cadence", cad.delta)
    assert "Notable" in (cad.reason or "")
    assert "Notable" in notable_difference_label("Cadence", 10.0)
    assert is_notable_difference("Cadence", 1.0) is False


def test_small_delta_is_not_notable() -> None:
    left = CompareMetrics(label="A", cadence_spm=100.0, walking_speed_m_s=1.20)
    right = CompareMetrics(label="B", cadence_spm=101.0, walking_speed_m_s=1.22)
    cad = build_metric_row("Cadence", left, right)
    assert cad.delta == pytest.approx(1.0)
    assert cad.reason == ""
    assert is_notable_difference("Speed", 0.01) is False


def test_positive_delta_labeled_improvement_for_primary_metrics() -> None:
    left = CompareMetrics(label="A", cadence_spm=90.0, walking_speed_m_s=1.0)
    right = CompareMetrics(label="B", cadence_spm=120.0, walking_speed_m_s=1.3)
    result = compare_session_metrics(left, right)
    by_name = {d.name: d for d in result.diffs}
    assert by_name["Cadence difference"].tone_right == "better"
    assert by_name["Walking speed"].tone_right == "better"
    left.symmetry_pct = 70.0
    right.symmetry_pct = 90.0
    result2 = compare_session_metrics(left, right)
    assert {d.name: d for d in result2.diffs}["Symmetry"].tone_right == "better"


def test_comparison_cache_reused_for_same_fingerprint() -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.pin(_snap("abnormal", gait_quality=50.0))
    cache.set_slot("left", "normal")
    cache.set_slot("right", "abnormal")
    first = build_cached_comparison(cache)
    second = build_cached_comparison(cache, previous=first)
    assert second is first
    # Playback-style re-entry must not change fingerprint when sessions unchanged.
    fp = comparison_fingerprint(
        cache.left(),
        cache.right(),
        left_key=cache.left_key,
        right_key=cache.right_key,
    )
    assert first.fingerprint == fp
    cache.right().metrics.symmetry_pct = 77.0  # type: ignore[union-attr]
    changed = comparison_fingerprint(
        cache.left(),
        cache.right(),
        left_key=cache.left_key,
        right_key=cache.right_key,
    )
    assert changed != fp


def test_joint_curves_require_both_valid_trajectories() -> None:
    a = _snap("normal", knee_y=[1.0, 2.0])  # too short
    b = _snap("abnormal", knee_y=[1.0, 2.0, 3.0, 4.0])
    ok, reason = joint_curves_ready(a, b)
    assert ok is False
    assert "incomplete" in reason.lower() or "trajectories" in reason.lower()

    a2 = _snap("normal", knee_y=[1.0, 2.0, 3.0, 4.0])
    ok2, _ = joint_curves_ready(a2, b)
    assert ok2 is True


def test_bundle_without_results_not_analyzed(tmp_path: Path) -> None:
    bundle = tmp_path / "session_x"
    bundle.mkdir()
    (bundle / "session_metadata.json").write_text(
        '{"display_name":"Demo","video_source":"clip.mp4","frame_count":0}',
        encoding="utf-8",
    )
    cand = candidate_from_bundle_metadata(bundle)
    assert cand.is_analyzed is False
    assert cand.incomplete_reason


def test_primary_rows_match_primary_metric_list() -> None:
    left = CompareMetrics(
        label="A",
        gait_quality=70.0,
        cadence_spm=100.0,
        symmetry_pct=80.0,
        stability_margin_m=0.05,
        knee_rom_deg=50.0,
        walking_speed_m_s=1.1,
        step_length_m=0.6,
    )
    right = CompareMetrics(
        label="B",
        gait_quality=60.0,
        cadence_spm=90.0,
        symmetry_pct=70.0,
        stability_margin_m=0.02,
        knee_rom_deg=40.0,
        walking_speed_m_s=None,
        step_length_m=0.5,
    )
    primary = build_comparison_rows(left, right, names=PRIMARY_COMPARE_METRICS)
    assert len(primary) == len(PRIMARY_COMPARE_METRICS)
    assert len(primary) == 6
    assert all(r.name in PRIMARY_COMPARE_METRICS for r in primary)


def test_demo_delta_rows_are_stability_symmetry_cadence_rom() -> None:
    from stablewalk.ui.tk.compare_workflow import (
        DEMO_DELTA_METRICS,
        DEMO_DELTA_TITLES,
        assign_demo_comparison_slots,
        build_demo_delta_rows,
    )

    assert DEMO_DELTA_METRICS == ("Stability", "Symmetry", "Cadence", "Joint ROM")
    assert DEMO_DELTA_TITLES["Joint ROM"] == "Knee ROM (session span)"
    assert DEMO_DELTA_TITLES["Stability"] == "Mean COM–BoS Margin"
    left = CompareMetrics(
        label="Normal",
        cadence_spm=110.0,
        symmetry_pct=92.0,
        stability_margin_m=0.05,
        knee_rom_deg=60.0,
    )
    right = CompareMetrics(
        label="Abnormal",
        cadence_spm=80.0,
        symmetry_pct=70.0,
        stability_margin_m=0.02,
        knee_rom_deg=40.0,
    )
    rows = build_demo_delta_rows(left, right)
    by_name = {r.name: r for r in rows}
    assert set(by_name) == set(DEMO_DELTA_METRICS)
    assert by_name["Cadence"].delta == pytest.approx(-30.0)
    assert by_name["Symmetry"].delta == pytest.approx(-22.0)
    assert by_name["Stability"].delta == pytest.approx(-0.03)
    assert by_name["Joint ROM"].delta == pytest.approx(-20.0)
    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.pin(_snap("abnormal"))
    left_key, right_key = assign_demo_comparison_slots(cache)
    assert left_key == "normal"
    assert right_key == "abnormal"
    assert cache.left_key == "normal"
    assert cache.right_key == "abnormal"


def test_pin_normal_vs_abnormal_uses_demo_snapshots(monkeypatch) -> None:
    from stablewalk.ui.tk import session_cache as sc

    cache = SessionCompareCache()
    calls: list[str] = []

    def _fake_build(key: str):
        calls.append(key)
        return _snap(key)

    monkeypatch.setattr(sc, "try_build_demo_snapshot", _fake_build)
    missing = sc.pin_normal_vs_abnormal(cache)
    assert missing == []
    assert calls == ["normal", "abnormal"]
    assert cache.get("normal") is not None
    assert cache.get("abnormal") is not None
    assert cache.left_key == "normal"
    assert cache.right_key == "abnormal"

