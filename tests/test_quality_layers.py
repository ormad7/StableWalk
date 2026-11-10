"""Quality layers stay distinct; missing physical inputs stay unavailable."""

from __future__ import annotations

from stablewalk.analysis.biomechanical.gait_quality_score import compute_gait_quality_score
from stablewalk.analysis.quality_layers import (
    FORBIDDEN_QUALITY_WORDS,
    GAIT_QUALITY_KIND,
    LANDMARK_NOT_TRACKING,
    biomechanical_validity_evidence,
    gait_quality_confidence,
    landmark_confidence_evidence,
    pose_coverage_evidence,
    spectral_reliability_evidence,
    status_from_evidence,
)
from stablewalk.analysis.spectral.autocorrelation import AutocorrResult
import numpy as np


def test_pose_coverage_is_not_landmark_accuracy() -> None:
    ev = pose_coverage_evidence(n_frames=100, n_detected=95)
    assert ev.status == "computed"
    assert ev.coverage_pct == 95.0
    assert "not a claim of accurate tracking" in ev.explanation.lower()
    assert ev.layer == "pose_coverage"


def test_landmark_visibility_is_not_tracking() -> None:
    ev = landmark_confidence_evidence(mean_visibility=0.82, n_samples=200)
    assert ev.status == "computed"
    assert LANDMARK_NOT_TRACKING in ev.explanation
    low = landmark_confidence_evidence(mean_visibility=0.25, n_samples=50)
    assert low.status == "insufficient"


def test_spectral_layer_uses_duration_gaps_and_cycles() -> None:
    good = spectral_reliability_evidence(
        duration_s=8.0, gap_fraction=0.05, n_cycles=6.0, peak_value=0.7
    )
    assert good.status == "computed"
    assert "not pose coverage" in good.explanation.lower()
    short = spectral_reliability_evidence(
        duration_s=0.4, gap_fraction=0.5, n_cycles=0, insufficient_reason="too short"
    )
    assert short.status == "insufficient"


def test_physical_stability_stays_unavailable() -> None:
    ev = biomechanical_validity_evidence(stability_reportable=False)
    assert ev.status == "unavailable"
    assert ev.layer == "biomechanical"


def test_gait_score_is_heuristic_and_not_invented() -> None:
    empty = compute_gait_quality_score()
    assert empty.available is False
    assert empty.score is None
    assert empty.kind == GAIT_QUALITY_KIND
    assert "heuristic" in empty.explanation.lower()
    blob = empty.to_dict()
    assert blob["kind"] == "heuristic"
    assert "validated" in blob["limitations"].lower()


def test_gait_confidence_does_not_default_missing_contact() -> None:
    conf = gait_quality_confidence(n_active=2, input_confidences=[])
    assert 0.0 < conf < 0.5
    assert gait_quality_confidence(n_active=0, input_confidences=[]) == 0.0


def test_status_vocabulary_avoids_reliable_acceptable() -> None:
    status = status_from_evidence(have_inputs=True, n_cycles=4, usable_duration_s=6.0)
    assert status == "computed"
    assert status not in FORBIDDEN_QUALITY_WORDS
    limited = status_from_evidence(have_inputs=True, n_cycles=1, usable_duration_s=2.0)
    assert limited == "limited"


def test_autocorr_result_exports_quality_status() -> None:
    ac = AutocorrResult(
        channel_id="knee",
        lags_s=np.array([0.0, 0.1]),
        rho=np.array([1.0, 0.2]),
        n_samples=20,
        quality_status="limited",
        reliability_explanation="gap fraction 20%",
    )
    payload = ac.to_dict()
    assert payload["quality_status"] == "limited"
    assert payload["reliability_explanation"]
