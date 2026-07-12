"""Tests for stability result validity and comparison reliability."""

from __future__ import annotations

import pytest

from stablewalk import config
from stablewalk.analysis.biomech_stability import analyze_biomech_stability
from stablewalk.analysis.stability_validity import (
    VALIDITY_CONFIGURATION_DOC,
    assess_stability_result_validity,
    compare_stability_results,
)
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.enrichment import enrich_pose_sequence


@pytest.fixture(scope="module")
def demo_results() -> dict[str, object]:
    stems = {
        "abnormal": "abnormal_gait",
        "normal": "normal_gait",
        "athletic": "athletic_walking",
    }
    out = {}
    for label, stem in stems.items():
        path = config.POSES_DIR / f"{stem}_poses.json"
        if not path.is_file():
            pytest.skip(f"{stem}_poses.json not available")
        seq = load_pose_sequence(path)
        enrich_pose_sequence(seq)
        out[label] = analyze_biomech_stability(seq)
    return out


def test_validity_configuration_documented():
    assert "evidence index" in VALIDITY_CONFIGURATION_DOC.lower()
    assert "comparison tolerance" in VALIDITY_CONFIGURATION_DOC.lower()


def test_demo_abnormal_insufficient_data(demo_results):
    result = demo_results["abnormal"]
    validity = result.validity
    assert validity is not None
    assert validity.status == "INSUFFICIENT_DATA"
    assert validity.comparable_score == "NO"
    assert validity.show_partial_estimate


def test_demo_normal_insufficient_data(demo_results):
    result = demo_results["normal"]
    validity = result.validity
    assert validity is not None
    assert validity.status == "INSUFFICIENT_DATA"
    assert validity.comparable_score == "NO"


def test_demo_athletic_provisional(demo_results):
    result = demo_results["athletic"]
    validity = result.validity
    assert validity is not None
    assert validity.status == "PROVISIONAL"
    assert validity.comparable_score == "LIMITED"


def test_abnormal_vs_normal_not_comparable(demo_results):
    comparison = compare_stability_results(
        demo_results["abnormal"],
        demo_results["normal"],
        label_a="abnormal",
        label_b="normal",
    )
    assert comparison.verdict == "NOT_COMPARABLE"
    assert "not a meaningful ranking" in comparison.explanation


def test_normal_vs_athletic_not_comparable_or_no_reliable_difference(demo_results):
    comparison = compare_stability_results(
        demo_results["normal"],
        demo_results["athletic"],
        label_a="normal",
        label_b="athletic",
    )
    assert comparison.verdict in ("NOT_COMPARABLE", "NO_RELIABLE_DIFFERENCE")
    assert "meaningful ranking" in comparison.explanation


def test_does_not_force_abnormal_greater_than_normal_ranking(demo_results):
    """Abnormal score may exceed normal numerically, but comparison must gate ranking."""
    abnormal = demo_results["abnormal"]
    normal = demo_results["normal"]
    assert abnormal.score > normal.score
    comparison = compare_stability_results(
        abnormal, normal, label_a="abnormal", label_b="normal"
    )
    assert comparison.verdict == "NOT_COMPARABLE"


def test_stability_result_includes_validity_dict(demo_results):
    payload = demo_results["normal"].to_dict()
    assert "validity" in payload
    assert payload["validity"]["status"] == "INSUFFICIENT_DATA"
