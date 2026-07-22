"""Tests for three-pillar gait analysis semantics."""

from __future__ import annotations

import pytest

from stablewalk import config
from stablewalk.analysis.biomech_stability import analyze_biomech_stability
from stablewalk.analysis.gait_analysis_summary import (
    DOMAIN_SEMANTICS_DOC,
    GAIT_QUALITY_DOMAIN_KEYS,
    MOVEMENT_STABILITY_DOMAIN_KEYS,
    build_gait_analysis_summary,
    generate_gait_analysis_explanation,
)
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.enrichment import enrich_pose_sequence

FORBIDDEN_LABELS = ("abnormal", "normal", "athletic", "pathology", "neuropathic")


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


def test_domain_semantics_documented():
    assert "Movement Stability" in DOMAIN_SEMANTICS_DOC
    assert "Gait Quality" in DOMAIN_SEMANTICS_DOC
    assert "pelvis_stability" in DOMAIN_SEMANTICS_DOC
    assert "temporal_symmetry" in DOMAIN_SEMANTICS_DOC


def test_domain_partitions_are_disjoint_and_complete():
    all_keys = MOVEMENT_STABILITY_DOMAIN_KEYS | GAIT_QUALITY_DOMAIN_KEYS
    assert len(all_keys) == 8
    assert not MOVEMENT_STABILITY_DOMAIN_KEYS & GAIT_QUALITY_DOMAIN_KEYS


def test_gait_summary_attached_to_result(demo_results):
    for result in demo_results.values():
        assert result.gait_summary is not None
        assert result.gait_summary.movement_stability.label == "Movement Stability"
        assert result.gait_summary.gait_quality.label == "Gait Quality (derived)"


def test_explanation_uses_no_demo_labels(demo_results):
    for result in demo_results.values():
        summary = result.gait_summary or build_gait_analysis_summary(result)
        text = summary.explanation.lower()
        for label in FORBIDDEN_LABELS:
            assert label not in text


def test_abnormal_can_have_high_movement_low_gait_quality(demo_results):
    """Walker-assisted gait may show controlled motion without good gait quality."""
    result = demo_results["abnormal"]
    summary = result.gait_summary
    assert summary is not None
    ms = summary.movement_stability.score
    gq = summary.gait_quality.score
    assert ms is not None
    assert gq is None or ms > gq or summary.gait_quality.completeness_pct < 30


def test_analysis_confidence_is_categorical(demo_results):
    for result in demo_results.values():
        summary = result.gait_summary
        assert summary is not None
        assert summary.analysis_confidence.level in (
            "HIGH", "MODERATE", "LOW", "INSUFFICIENT"
        )


def test_legacy_score_preserved(demo_results):
    for result in demo_results.values():
        summary = result.gait_summary
        assert summary is not None
        assert summary.legacy_composite_score == pytest.approx(result.score)


def test_generate_explanation_from_measured_evidence(demo_results):
    result = demo_results["abnormal"]
    summary = build_gait_analysis_summary(result)
    text = generate_gait_analysis_explanation(summary, result)
    assert len(text) > 20
    assert any(
        phrase in text.lower()
        for phrase in ("controlled", "gait quality", "cycle", "assessed")
    )
