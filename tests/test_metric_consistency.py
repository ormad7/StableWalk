"""Central metric definition / status consistency tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stablewalk.analysis.biomechanical.gait_quality_score import compute_gait_quality_score
from stablewalk.analysis.biomechanical.types import GaitQualityScore
from stablewalk.metrics.definitions import (
    GQ_NORMAL_MIN,
    METRIC_DEFINITIONS,
    STABLE_PCT_BORDERLINE_MIN,
    STABLE_PCT_NORMAL_MIN,
    get_metric_definition,
)
from stablewalk.metrics.session_view import (
    build_session_metric_snapshots,
    detect_metric_contradictions,
    get_snapshot,
)
from stablewalk.metrics.status import (
    DisplayStatus,
    ValidityState,
    build_metric_status,
    engine_stability_state_status,
    is_genuine_zero,
    never_zero_from_missing,
)
from stablewalk.ui.overview_clinical_summary import classify_overview_metric, status_label
from stablewalk.ui.summary_metric_style import interpret_summary_metric, metric_visual_style
from stablewalk.analysis.analysis_summary import SummaryField
from stablewalk.ui.scientific_labels import TIER_DERIVED


def test_registry_covers_primary_kpis() -> None:
    for key in (
        "gait_quality_composite",
        "stable_pct",
        "symmetry",
        "cadence",
        "analysis_confidence",
        "gait_cycles",
    ):
        assert key in METRIC_DEFINITIONS
        assert get_metric_definition(key) is not None


def test_unavailable_never_becomes_zero() -> None:
    st = build_metric_status("stable_pct", value=None, available=False)
    assert st.validity == ValidityState.UNAVAILABLE
    assert st.display_value == "—"
    assert st.value is None
    assert never_zero_from_missing(st)
    assert not is_genuine_zero(st)


def test_genuine_zero_stable_pct_is_allowed() -> None:
    st = build_metric_status("stable_pct", value=0.0, available=True)
    assert st.validity == ValidityState.VALID
    assert st.value == 0.0
    assert st.display_value in ("0", "0.0", "0.00")
    assert is_genuine_zero(st)
    assert st.display_status == DisplayStatus.REDUCED


def test_gait_quality_insufficient_is_not_zero() -> None:
    gq = compute_gait_quality_score()
    assert gq.available is False
    assert gq.score is None
    view = build_session_metric_snapshots(
        biomechanical=SimpleNamespace(
            gait_quality=gq,
            stability_margin=None,
            symmetry=None,
            gait_metrics=None,
        )
    )
    snap = view.require("gait_quality_composite")
    assert snap.validity in (
        ValidityState.UNAVAILABLE,
        ValidityState.INSUFFICIENT_DATA,
    )
    assert snap.value is None
    assert snap.display_value == "—"
    assert never_zero_from_missing(snap.status)


def test_overview_and_results_share_gait_quality_bands() -> None:
    assert classify_overview_metric(
        "gait_quality", display="71/100", gait_quality_score=71.0
    ) == "normal"
    field = SummaryField("", "71/100", TIER_DERIVED, True)
    assert interpret_summary_metric("gait_quality", field) == "normal"
    assert metric_visual_style("normal").status_label == DisplayStatus.ACCEPTABLE.value
    assert status_label("normal") == DisplayStatus.ACCEPTABLE.value
    from stablewalk.ui.summary_metric_style import kpi_status_label

    assert kpi_status_label("normal", key="gait_quality") == DisplayStatus.UNVALIDATED_HEURISTIC.value
    assert kpi_status_label("normal", key="symmetry") == DisplayStatus.UNVALIDATED_HEURISTIC.value


def test_engine_stable_is_not_labelled_acceptable() -> None:
    st = engine_stability_state_status("Stable", available=True, confidence=0.8)
    assert st.value == "Stable"
    assert st.display_status == DisplayStatus.HEURISTIC
    assert st.display_status != DisplayStatus.ACCEPTABLE


def test_stability_bands_shared_across_surfaces() -> None:
    assert classify_overview_metric(
        "stability_state", display="0%", stable_pct=0.0
    ) == "abnormal"
    field = SummaryField("", "Unstable (0% stable frames)", TIER_DERIVED, True)
    assert interpret_summary_metric("stability_margin", field) == "abnormal"
    st = build_metric_status("stable_pct", value=0.0, available=True)
    assert st.display_status == DisplayStatus.REDUCED


def test_gq_vs_stable_pct_contradiction_is_explained_not_fudged() -> None:
    """GQ≈71 Heuristic with stable_pct=0 is a real formula divergence."""
    ba = SimpleNamespace(
        gait_quality=GaitQualityScore(
            score=71.0,
            confidence=0.8,
            dominant_factors=["strong symmetry"],
            explanation="Composite Gait Quality 71/100 (derived).",
            components={
                "stability": 80.0,
                "symmetry": 90.0,
                "stability_margin": 0.0,
            },
            available=True,
        ),
        stability_margin=SimpleNamespace(
            stable_pct=0.0,
            mean_margin_m=-0.01,
            valid_frame_ratio=0.9,
            confidence=0.7,
            warnings=[],
            per_frame=[],
        ),
        symmetry=SimpleNamespace(
            overall_symmetry_pct=SimpleNamespace(value=90.0, confidence=0.9)
        ),
        gait_metrics=None,
    )
    view = build_session_metric_snapshots(biomechanical=ba, session_label="normal_gait.mp4")
    gq = view.require("gait_quality_composite")
    stab = view.require("stable_pct")
    assert float(gq.value) >= GQ_NORMAL_MIN
    assert gq.display_status == DisplayStatus.HEURISTIC
    assert float(stab.value) < STABLE_PCT_BORDERLINE_MIN
    assert stab.display_status == DisplayStatus.REDUCED
    notes = detect_metric_contradictions(view.snapshots)
    assert notes
    assert "different formulas" in notes[0].lower() or "weighted" in notes[0].lower()
    # Values must not be altered to force agreement.
    assert float(gq.value) == 71.0
    assert float(stab.value) == 0.0


def test_missing_frames_marks_stable_pct_unavailable() -> None:
    ba = SimpleNamespace(
        gait_quality=None,
        stability_margin=SimpleNamespace(
            stable_pct=None,
            mean_margin_m=None,
            valid_frame_ratio=0.2,
            confidence=0.1,
            warnings=["Stability margin unavailable: only 20% of frames..."],
            per_frame=[],
        ),
        symmetry=None,
        gait_metrics=None,
    )
    view = build_session_metric_snapshots(biomechanical=ba)
    snap = view.require("stable_pct")
    assert snap.value is None
    assert snap.display_value == "—"
    assert snap.validity in (
        ValidityState.UNAVAILABLE,
        ValidityState.INSUFFICIENT_DATA,
    )


def test_low_confidence_validity() -> None:
    st = build_metric_status(
        "symmetry", value=80.0, available=True, confidence=0.1, low_confidence=True
    )
    assert st.validity == ValidityState.LOW_CONFIDENCE
    assert st.display_status == DisplayStatus.LOW_CONFIDENCE


def test_no_gait_cycle_snapshot() -> None:
    view = build_session_metric_snapshots(
        biomechanical=SimpleNamespace(
            gait_quality=None,
            stability_margin=None,
            symmetry=None,
            gait_metrics=None,
        ),
        cycles=SimpleNamespace(cycles=[], metrics=SimpleNamespace()),
    )
    snap = view.require("gait_cycles")
    assert snap.raw_value == 0
    assert snap.value == 0
    assert is_genuine_zero(snap.status)


def test_short_video_cadence_unavailable() -> None:
    view = build_session_metric_snapshots(
        biomechanical=SimpleNamespace(
            gait_quality=None,
            stability_margin=None,
            symmetry=None,
            gait_metrics=SimpleNamespace(
                cadence=SimpleNamespace(value=None, confidence=0.0, note="N/A")
            ),
        ),
        cycles=None,
    )
    snap = view.require("cadence")
    assert snap.value is None
    assert snap.display_value == "—"


def test_missing_opensim_does_not_invent_metrics() -> None:
    view = build_session_metric_snapshots(
        biomechanical=None,
        opensim_available=False,
        session_label="no-opensim",
    )
    for key in ("gait_quality_composite", "stable_pct", "symmetry"):
        snap = view.require(key)
        assert snap.value is None
        assert snap.display_value == "—"


def test_gait_summary_does_not_fill_quality_when_biomech_score_missing() -> None:
    """Coordination-group score is a different algorithm — do not alias it."""
    view = build_session_metric_snapshots(
        biomechanical=None,
        gait_summary=SimpleNamespace(
            gait_quality=GaitQualityScore(
                score=71.0,
                confidence=0.8,
                dominant_factors=[],
                explanation="ok",
                components={"symmetry": 71.0},
                available=True,
            )
        ),
    )
    snap = view.require("gait_quality_composite")
    assert snap.value is None
    assert snap.display_value in ("—", "N/A")


def test_alias_resolution_consistency() -> None:
    view = build_session_metric_snapshots(
        biomechanical=SimpleNamespace(
            gait_quality=GaitQualityScore(
                score=71.0,
                confidence=0.8,
                dominant_factors=[],
                explanation="ok",
                components={"symmetry": 71.0},
                available=True,
            ),
            stability_margin=None,
            symmetry=None,
            gait_metrics=None,
        )
    )
    assert get_snapshot(view.snapshots, "gait_quality") is not None
    assert get_snapshot(view.snapshots, "overall_gait_quality") is not None
    assert get_snapshot(view.snapshots, "gait_quality").value == 71.0


def test_export_dict_preserves_provenance() -> None:
    view = build_session_metric_snapshots(
        biomechanical=SimpleNamespace(
            gait_quality=GaitQualityScore(
                score=71.0,
                confidence=0.8,
                dominant_factors=["strong symmetry"],
                explanation="ok",
                components={"symmetry": 90.0, "stability_margin": 0.0},
                available=True,
            ),
            stability_margin=SimpleNamespace(
                stable_pct=0.0,
                mean_margin_m=-0.02,
                valid_frame_ratio=0.8,
                confidence=0.6,
                warnings=[],
                per_frame=[],
            ),
            symmetry=None,
            gait_metrics=None,
        ),
        session_label="normal_gait.mp4",
    )
    payload = view.export_dict()
    assert payload["session_label"] == "normal_gait.mp4"
    gq = payload["metrics"]["gait_quality_composite"]
    assert gq["value"] == 71.0
    assert gq["source"] == "derived"
    assert gq["threshold_source"] == "heuristic"
    assert gq["raw_value"] == 71.0
    stab = payload["metrics"]["stable_pct"]
    assert stab["value"] == 0.0
    assert stab["validity"] == "valid"
    assert payload["contradictions"]


def test_stable_pct_border_constants_match_classifier() -> None:
    assert (
        build_metric_status(
            "stable_pct", value=STABLE_PCT_NORMAL_MIN, available=True
        ).display_status
        == DisplayStatus.HEURISTIC
    )
    assert (
        build_metric_status(
            "stable_pct", value=STABLE_PCT_BORDERLINE_MIN, available=True
        ).display_status
        == DisplayStatus.BORDERLINE
    )
