"""Results validity layers replace a single Analysis Confidence headline."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.analysis.analysis_summary import _analysis_confidence, build_analysis_summary
from stablewalk.analysis.foot_contact_analysis import (
    CONTACT_UNKNOWN_REASON_LABELS,
    FootContactAnalysisResult,
    FootContactFrame,
    contact_unknown_reason,
    iter_contact_frame_evidence,
)
from stablewalk.analysis.gait_cycle_analysis import FootLandmarkSample
from stablewalk.analysis.gait_phase_classification import FLIGHT_CLEARANCE_THRESHOLD_M
from stablewalk.analysis.quality_layers import UNEQUAL_CONTACT_COVERAGE_NOTE
from stablewalk.analysis.result_validity import (
    build_result_validity,
    contact_coverage_blocks_laterality,
    gait_quality_withhold_reason,
    laterality_withhold_reason,
    result_validity_from_export_dict,
)
from stablewalk.io.contact_evidence_export import contact_evidence_summary


def _sample(vis: float, clearance: float | None) -> FootLandmarkSample:
    return FootLandmarkSample(
        heel_clearance_m=clearance,
        toe_clearance_m=clearance,
        ankle_clearance_m=clearance,
        foot_clearance_m=clearance,
        heel_velocity_m_s=0.0,
        toe_velocity_m_s=0.0,
        ankle_velocity_m_s=0.0,
        visibility=vis,
    )


def _frame(i: int, *, left_known: bool, right_known: bool = True, reason: str = "") -> FootContactFrame:
    return FootContactFrame(
        frame_index=i,
        time_s=i / 30.0,
        left_contact_probability=0.8 if left_known else 0.2,
        right_contact_probability=0.9,
        left_contact_binary=1 if left_known else 0,
        right_contact_binary=1,
        left_heel_strike=0,
        right_heel_strike=0,
        left_toe_off=0,
        right_toe_off=0,
        left_foot_substate="mid_stance" if left_known else "swing",
        right_foot_substate="mid_stance",
        macro_phase="stance" if left_known else "uncertain",
        left_confidence=0.8,
        right_confidence=0.8,
        left_contact_known=left_known,
        right_contact_known=right_known,
        left_unknown_reason="" if left_known else reason or "not_confirmed_flight",
        right_unknown_reason="",
    )


def test_near_ground_off_is_unknown_not_swing() -> None:
    sample = _sample(0.9, FLIGHT_CLEARANCE_THRESHOLD_M * 0.4)
    reason = contact_unknown_reason(sample, 0, probability=0.2)
    assert reason == "not_confirmed_flight"
    assert reason in CONTACT_UNKNOWN_REASON_LABELS


def test_low_visibility_is_unknown() -> None:
    sample = _sample(0.1, 0.2)
    assert contact_unknown_reason(sample, 1, probability=0.9) == "low_foot_visibility"


def test_unilateral_cycles_cannot_be_high_confidence() -> None:
    ba = SimpleNamespace(
        video_quality=SimpleNamespace(overall_quality_score=95.0),
        gait_quality=SimpleNamespace(confidence=0.95),
        stability_margin=SimpleNamespace(reportable=False),
    )
    cycles = SimpleNamespace(
        metrics=SimpleNamespace(
            gait_cycle_count=3,
            contact_confidence=0.9,
            confidence_tier="HIGH",
        ),
        events=[SimpleNamespace(time_s=0.0), SimpleNamespace(time_s=8.0)],
        cycles=[
            SimpleNamespace(boundary_side="right", start_time_s=0.0, end_time_s=2.0),
            SimpleNamespace(boundary_side="right", start_time_s=2.0, end_time_s=4.0),
            SimpleNamespace(boundary_side="right", start_time_s=4.0, end_time_s=6.0),
        ],
    )
    contact = SimpleNamespace(metrics=SimpleNamespace(contact_confidence=0.9))
    level, detail, _score = _analysis_confidence(ba, cycles, contact)
    assert level != "High"
    assert "0 L / 3 R" in detail
    assert "laterality" in detail.lower()


def test_validity_splits_pose_contact_spectral_biomech() -> None:
    frames = [_frame(i, left_known=i < 3, reason="not_confirmed_flight") for i in range(12)]
    contact = FootContactAnalysisResult(per_frame=frames, fps=30.0)
    cycles = SimpleNamespace(
        metrics=SimpleNamespace(
            gait_cycle_count=3,
            contact_confidence=0.8,
            reliability_reason="",
        ),
        cycles=[
            SimpleNamespace(boundary_side="right"),
            SimpleNamespace(boundary_side="right"),
            SimpleNamespace(boundary_side="right"),
        ],
    )
    spectral = SimpleNamespace(
        univariate_channel_id="right_knee",
        autocorr={
            "right_knee": SimpleNamespace(
                stride_period_s=None,
                peaks=[],
                n_stride_cycles=1.2,
                insufficient_reason="Clip is too short for a reliable ACF stride peak.",
                reliability_label="low",
                reliability_explanation="Spectral reliability is low.",
                gap_fraction=0.2,
            )
        },
    )
    ba = SimpleNamespace(
        video_quality=SimpleNamespace(
            pose_coverage=None,
            checks={"detected_frame_pct": 98.0, "pose_coverage_pct": 98.0},
        ),
        stability_margin=SimpleNamespace(
            reportable=False,
            experimental_note="Physical COM–BoS is not reportable from monocular pose.",
        ),
    )
    report = build_result_validity(
        biomechanical=ba,
        contact=contact,
        cycles=cycles,
        spectral=spectral,
        duration_s=4.0,
    )
    assert report.pose_coverage.status == "computed"
    assert report.contact_left.coverage_pct == 25.0
    assert report.contact_right.coverage_pct == 100.0
    assert report.contact_coverage.extras["unequal"] is True
    assert report.spectral.status == "insufficient"
    assert report.biomechanical.status == "unavailable"
    assert contact_coverage_blocks_laterality(report)
    headline = "\n".join(report.headline_lines())
    assert "Analysis Confidence: High" not in headline
    assert "0 L / 3 R" in headline or "0 left / 3 right" in headline.lower() or "0 L / 3 R" in headline
    assert UNEQUAL_CONTACT_COVERAGE_NOTE in headline
    assert "Physical COM–BoS" in report.biomechanical.explanation


def test_results_summary_withholds_symmetry_when_coverage_unequal() -> None:
    frames = [_frame(i, left_known=False) for i in range(10)]
    contact = FootContactAnalysisResult(per_frame=frames, fps=30.0)
    ba = SimpleNamespace(
        gait_metrics=None,
        gait_quality=None,
        symmetry=SimpleNamespace(
            overall_symmetry_pct=SimpleNamespace(value=72.0, note="weighted"),
            warnings=[],
        ),
        stability_margin=SimpleNamespace(reportable=False, experimental_note="unreportable", per_frame=[]),
        center_of_mass=None,
        joint_rom=None,
        video_quality=None,
    )
    cycles = SimpleNamespace(
        metrics=SimpleNamespace(
            cadence_steps_per_min=None,
            contact_confidence=0.8,
            gait_cycle_count=3,
            confidence_tier="HIGH",
            reliability_reason="",
        ),
        cycles=[SimpleNamespace(boundary_side="right") for _ in range(3)],
        events=[],
    )
    summary = build_analysis_summary(
        source="abnormal",
        biomechanical=ba,
        estimated_vgrf=None,
        contact=contact,
        cycles=cycles,
    )
    assert summary.symmetry is not None
    assert summary.symmetry.available is False
    assert "laterality" in summary.symmetry.reason.lower()
    assert summary.result_validity is not None
    text = "\n".join(summary.result_validity.headline_lines())
    assert "High" not in text.split("Pose coverage", 1)[0] or "Pose coverage" in text
    assert "Analysis Confidence: High" not in text


def test_low_acf_label_is_not_computed_from_clip_duration() -> None:
    frames = [_frame(i, left_known=True) for i in range(8)]
    contact = FootContactAnalysisResult(per_frame=frames, fps=30.0)
    spectral = SimpleNamespace(
        univariate_channel_id="left_knee",
        autocorr={
            "left_knee": SimpleNamespace(
                stride_period_s=1.0,
                peaks=[SimpleNamespace(lag_s=1.0, value=0.17)],
                n_stride_cycles=27.0,
                insufficient_reason=None,
                reliability_label="low",
                reliability=0.46,
                reliability_explanation="Spectral reliability from duration 27.00 s.",
                gap_fraction=0.02,
            )
        },
    )
    report = build_result_validity(
        biomechanical=SimpleNamespace(
            video_quality=SimpleNamespace(checks={"pose_coverage_pct": 99.0}),
            stability_margin=SimpleNamespace(reportable=False, experimental_note="unreportable"),
        ),
        contact=contact,
        cycles=SimpleNamespace(metrics=SimpleNamespace(gait_cycle_count=3), cycles=[]),
        spectral=spectral,
        duration_s=27.0,
    )
    assert report.spectral.status == "limited"
    assert report.spectral.extras["reliability_label"] == "low"
    assert "low" in report.spectral.explanation.lower()


def test_contact_evidence_export_counts_unknown_reasons() -> None:
    frames = [
        _frame(0, left_known=False, reason="low_foot_visibility"),
        _frame(1, left_known=False, reason="not_confirmed_flight"),
        _frame(2, left_known=True),
    ]
    contact = FootContactAnalysisResult(per_frame=frames, fps=30.0)
    rows = iter_contact_frame_evidence(contact)
    assert len(rows) == 6
    left = [r for r in rows if r["side"] == "left"]
    assert left[0]["unknown_reason"] == "low_foot_visibility"
    assert left[1]["unknown_reason"] == "not_confirmed_flight"
    assert left[2]["known"] is True
    summary = contact_evidence_summary(rows)
    assert summary["by_side"]["left"]["n_unknown"] == 2
    assert summary["by_side"]["right"]["n_unknown"] == 0
    assert "laterality" in summary["note"].lower()


def _normal_like_validity() -> object:
    return result_validity_from_export_dict(
        {
            "validity": {
                "pose_coverage": {"layer": "pose_coverage", "status": "limited"},
                "contact_left": {
                    "layer": "contact_coverage",
                    "status": "insufficient",
                    "coverage_pct": 20.0,
                },
                "contact_right": {
                    "layer": "contact_coverage",
                    "status": "insufficient",
                    "coverage_pct": 37.0,
                },
                "contact_coverage": {
                    "layer": "contact_coverage",
                    "status": "insufficient",
                    "extras": {"left_pct": 20.0, "right_pct": 37.0, "unequal": False},
                },
                "spectral": {"layer": "spectral", "status": "insufficient"},
                "biomechanical": {"layer": "biomechanical", "status": "unavailable"},
                "cycle_detection": {
                    "layer": "cycle_detection",
                    "status": "computed",
                    "n_cycles": 4,
                    "extras": {"n_left": 4, "n_right": 0},
                },
            }
        }
    )


def test_normal_like_inputs_withhold_unvalidated_heuristics() -> None:
    report = _normal_like_validity()
    laterality = laterality_withhold_reason(report)
    quality = gait_quality_withhold_reason(report)
    assert laterality
    assert quality
    assert "not a gait-laterality" in laterality.lower()
    assert "4 L / 0 R" in laterality or "4 L / 0 R" in quality
    assert "normal" not in laterality.lower()
    assert "normal" not in quality.lower()


def test_filename_does_not_raise_normal_quality() -> None:
    frames = [_frame(i, left_known=i % 5 == 0, right_known=i % 3 == 0) for i in range(20)]
    contact = FootContactAnalysisResult(per_frame=frames, fps=30.0)
    ba = SimpleNamespace(
        gait_metrics=None,
        gait_quality=SimpleNamespace(
            score=68.0,
            available=True,
            explanation="renormalized composite",
            components={"symmetry": 70.0},
            dominant_factors=[],
            confidence=0.4,
        ),
        symmetry=SimpleNamespace(
            overall_symmetry_pct=SimpleNamespace(value=61.0, note="weighted"),
            warnings=[],
        ),
        stability_margin=SimpleNamespace(
            reportable=False, experimental_note="unreportable", per_frame=[]
        ),
        center_of_mass=None,
        joint_rom=None,
        video_quality=None,
    )
    cycles = SimpleNamespace(
        metrics=SimpleNamespace(
            cadence_steps_per_min=None,
            contact_confidence=0.4,
            gait_cycle_count=4,
            confidence_tier="LOW",
            reliability_reason="",
        ),
        cycles=[SimpleNamespace(boundary_side="left") for _ in range(4)],
        events=[],
    )
    spectral = SimpleNamespace(
        univariate_channel_id="left_knee",
        autocorr={
            "left_knee": SimpleNamespace(
                stride_period_s=None,
                peaks=[],
                n_stride_cycles=0.0,
                insufficient_reason="short series",
                reliability_label="insufficient",
                reliability=0.13,
                reliability_explanation="insufficient",
                gap_fraction=0.12,
            )
        },
    )
    as_normal = build_analysis_summary(
        source="normal_gait.mp4",
        biomechanical=ba,
        estimated_vgrf=None,
        contact=contact,
        cycles=cycles,
        spectral=spectral,
    )
    as_other = build_analysis_summary(
        source="unlabeled_clip.mp4",
        biomechanical=ba,
        estimated_vgrf=None,
        contact=contact,
        cycles=cycles,
        spectral=spectral,
    )
    assert as_normal.overall_gait_quality is not None
    assert as_normal.overall_gait_quality.available is False
    assert as_normal.symmetry is not None
    assert as_normal.symmetry.available is False
    assert as_normal.overall_gait_quality.value == as_other.overall_gait_quality.value
    assert as_normal.symmetry.value == as_other.symmetry.value
    assert as_normal.contact_coverage_left is not None
    assert as_normal.cycle_evidence is not None
    assert "L /" in str(as_normal.cycle_evidence.value)
    assert "Borderline" not in str(as_normal.overall_gait_quality.value)
    assert "filename" not in (as_normal.overall_gait_quality.reason or "").lower()
