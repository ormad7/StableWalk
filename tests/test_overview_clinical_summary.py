"""Overview clinical summary helpers — no invented values."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.ui.overview_clinical_summary import (
    NOT_AVAILABLE,
    build_clinical_summary_bullets,
    build_key_findings,
    build_key_limitations,
    build_main_findings,
    build_overview_primary_finding,
    build_system_notes,
    classify_overview_metric,
    extract_confidence_visualization,
    extract_overview_summary_fields,
    format_gait_quality_session_details,
    format_symmetry_session_details,
    na_display,
    primary_finding_inputs,
    status_label,
    status_label_for_overview,
)


def test_na_display_never_guesses_a_number() -> None:
    text, tip = na_display("cadence not computed")
    assert text == NOT_AVAILABLE
    assert "cadence" in tip


def test_clinical_bullets_reconcile_gq_vs_reduced_stability() -> None:
    bullets = build_clinical_summary_bullets(
        usable_cycles=2,
        detected_cycles=2,
        cadence_spm=63.0,
        symmetry_pct=88.0,
        stable_pct=40.0,
        stability_margin_m=None,
        analysis_confidence_level="MODERATE",
        gait_quality_score=71.0,
    )
    assert 1 <= len(bullets) <= 5
    joined = " ".join(bullets).lower()
    assert any("two usable gait cycles" in b.lower() for b in bullets)
    assert "heuristic gait-quality descriptor: 71/100" in joined
    assert "pose-derived stability proxy is in the lower band" in joined
    assert "does not validate physical stability" in joined
    assert all(len(b) < 140 for b in bullets)
    assert all("\n" not in b for b in bullets)
    # Must not juxtapose bare clinical Normal with Unstable.
    assert not any("unstable" in b.lower() for b in bullets)
    assert not any("healthy" in b.lower() for b in bullets)


def test_status_labels_use_overview_vocabulary() -> None:
    assert status_label("normal") == "Acceptable"
    assert status_label("borderline") == "Borderline"
    assert status_label("abnormal") == "Reduced"
    assert status_label("unavailable") == "Not available"
    assert status_label_for_overview("stability_state", "abnormal") == "Reduced"
    assert status_label_for_overview("gait_quality", "normal") == "Unvalidated heuristic"
    assert status_label_for_overview("symmetry", "normal") == "Unvalidated heuristic"
    assert status_label_for_overview("gait_quality", "borderline") == "Unvalidated heuristic"
    assert status_label_for_overview("symmetry", "borderline") == "Unvalidated heuristic"
    assert status_label_for_overview("video_quality", "normal") == "Not accuracy"
    assert status_label_for_overview("tracking_confidence", "normal") == "Not accuracy"
    assert status_label_for_overview("analysis_confidence", "abnormal") == "Low confidence"
    assert status_label_for_overview("cadence", "abnormal", cadence_spm=150.0) == "Elevated"
    assert status_label_for_overview("cadence", "abnormal", cadence_spm=40.0) == "Reduced"


def test_primary_finding_uses_only_provided_fields() -> None:
    text = build_overview_primary_finding(
        usable_cycles=2,
        detected_cycles=2,
        cadence_spm=110.0,
        symmetry_pct=88.0,
        stable_pct=40.0,
        stability_margin_m=None,
        analysis_confidence_level="MODERATE",
        gait_quality_score=72.0,
    )
    assert "Two usable gait cycles" in text
    assert "heuristic gait-quality" in text.lower()
    assert "72/100" in text


def test_primary_finding_empty_session() -> None:
    text = build_overview_primary_finding(
        usable_cycles=None,
        detected_cycles=None,
        cadence_spm=None,
        symmetry_pct=None,
        stable_pct=None,
        stability_margin_m=None,
        analysis_confidence_level=None,
        gait_quality_score=None,
    )
    assert "load and analyze" in text.lower()


def test_main_findings_match_clinical_report_style() -> None:
    findings = build_main_findings(
        usable_cycles=2,
        detected_cycles=2,
        cadence_spm=110.0,
        symmetry_pct=88.0,
        stable_pct=40.0,
        stability_margin_m=None,
        analysis_confidence_level="MODERATE",
        gait_quality_score=71.0,
        tracking_score=80.0,
    )
    icons = {f[0] for f in findings}
    texts = " ".join(f[1] for f in findings).lower()
    assert "✓" in icons
    assert "⚠" in icons
    assert "landmark visibility" in texts
    assert "not tracking accuracy" in texts
    assert "pose-derived estimation" in texts


def test_key_findings_alias_matches_main_findings() -> None:
    kwargs = dict(
        usable_cycles=2,
        detected_cycles=2,
        cadence_spm=110.0,
        symmetry_pct=88.0,
        stable_pct=40.0,
        stability_margin_m=None,
        analysis_confidence_level="MODERATE",
        gait_quality_score=71.0,
        tracking_score=80.0,
    )
    assert build_key_findings(**kwargs) == build_main_findings(**kwargs)


def test_key_limitations_are_evidence_based() -> None:
    notes = build_key_limitations(
        tracking_score=40.0,
        pose_confidence_level="Low",
        walking_speed_available=False,
        opensim_enabled=True,
        opensim_ik_completed=False,
        monocular=True,
    )
    joined = " ".join(notes).lower()
    assert "monocular" in joined
    assert "pose-derived" in joined
    assert "approximate" in joined
    assert "low-confidence" in joined or "low" in joined
    assert "walking speed" in joined
    assert "opensim" in joined
    assert all(len(n) < 120 for n in notes)


def test_system_notes_alias_still_works() -> None:
    notes = build_system_notes(
        pose_confidence_level="High",
        opensim_enabled=True,
        monocular=True,
    )
    assert any("Monocular" in n for n in notes)


def test_classify_overview_metric_color_bands() -> None:
    assert (
        classify_overview_metric("gait_quality", display="71/100", gait_quality_score=71)
        == "normal"
    )
    assert (
        classify_overview_metric("symmetry", display="50%", symmetry_pct=50.0)
        == "abnormal"
    )
    assert (
        classify_overview_metric("cadence", display="110", cadence_spm=110.0) == "normal"
    )


def test_extract_summary_fields_unavailable_without_analysis() -> None:
    gui = SimpleNamespace(
        _biomech_analysis=None,
        _gait_cycle=None,
        _stability_result=None,
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (None, None),
    )
    fields = extract_overview_summary_fields(gui)
    assert fields["gait_quality"][0] == NOT_AVAILABLE
    assert "not a 0" in fields["gait_quality"][1].lower()
    assert fields["cadence"][0] == NOT_AVAILABLE
    inputs = primary_finding_inputs(gui)
    assert inputs["cadence_spm"] is None


def test_overview_rejects_unavailable_gait_quality_placeholder_score() -> None:
    unavailable_gq = SimpleNamespace(
        score=0.0,
        available=False,
        components={},
        dominant_factors=["insufficient data"],
    )
    gui = SimpleNamespace(
        _biomech_analysis=SimpleNamespace(
            gait_quality=unavailable_gq,
            symmetry=None,
            gait_metrics=None,
            stability_margin=None,
            video_quality=None,
        ),
        _gait_cycle=None,
        _stability_result=None,
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (None, None),
    )

    fields = extract_overview_summary_fields(gui)
    assert fields["gait_quality"][0] == NOT_AVAILABLE
    assert primary_finding_inputs(gui)["gait_quality_score"] is None
    conf = extract_confidence_visualization(gui)
    assert conf["available"] is False
    assert conf["pct"] is None


def test_confidence_does_not_invent_mapped_percentages() -> None:
    """Categorical HIGH/MODERATE/LOW must not become decorative 85/65/35%."""
    ac = SimpleNamespace(
        level="HIGH",
        completeness_pct=None,
        evidence_index=None,
        pose_valid_frame_pct=None,
    )
    summary = SimpleNamespace(analysis_confidence=ac)
    result = SimpleNamespace(gait_summary=summary, usable_gait_cycles=None)
    gui = SimpleNamespace(
        _biomech_analysis=None,
        _stability_result=result,
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (None, None),
    )
    conf = extract_confidence_visualization(gui)
    assert conf["pct"] is None
    assert conf["display"] == "High"
    assert "mapped" not in (conf.get("tip") or "").lower()


def test_confidence_uses_real_quality_signals() -> None:
    ba = SimpleNamespace(
        video_quality=SimpleNamespace(
            checks={"pose_confidence_score": 80.0},
            overall_quality_score=None,
        ),
        gait_quality=SimpleNamespace(confidence=0.7, score=70.0, components={}),
        stability_margin=SimpleNamespace(valid_frame_ratio=0.9, stable_pct=60.0),
    )
    gui = SimpleNamespace(
        _biomech_analysis=ba,
        _stability_result=None,
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (2, 2),
    )
    conf = extract_confidence_visualization(gui)
    assert conf["available"] is True
    assert conf["pct"] is not None
    assert 40.0 <= float(conf["pct"]) <= 100.0
    # Two cycles cannot display High; pose tracking is pose_level, not the mean.
    assert conf["display"] != "High"
    assert conf["pose_level"] == "Landmark visibility high"
    names = " ".join(conf.get("signal_names") or [])
    assert "cycle" in names
    assert "tracking" not in names


def test_unreportable_monocular_stability_is_not_a_kpi_percent() -> None:
    ba = SimpleNamespace(
        gait_quality=None,
        symmetry=None,
        gait_metrics=None,
        stability_margin=SimpleNamespace(
            stable_pct=5.83,
            reportable=False,
            experimental_note=(
                "Monocular COM–BoS is experimental and is not a physical stability score."
            ),
            per_frame=[],
            valid_frame_ratio=1.0,
        ),
        video_quality=None,
    )
    gui = SimpleNamespace(
        _biomech_analysis=ba,
        _gait_cycle=None,
        _stability_result=None,
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (2, 2),
    )
    fields = extract_overview_summary_fields(gui)
    display, tip = fields["stability_state"]
    assert display == NOT_AVAILABLE
    assert "5.83" not in display
    assert "experimental" in tip.lower() or "monocular" in tip.lower()


def test_stability_overview_does_not_use_unsupported_classification() -> None:
    ba = SimpleNamespace(
        gait_quality=None,
        symmetry=None,
        gait_metrics=None,
        stability_margin=SimpleNamespace(
            stable_pct=42.0,
            per_frame=[
                SimpleNamespace(frame_index=0, stability_state="Unstable"),
            ],
            valid_frame_ratio=0.8,
        ),
        video_quality=None,
    )
    gui = SimpleNamespace(
        _biomech_analysis=ba,
        _gait_cycle=None,
        _stability_result=None,
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (None, None),
    )
    fields = extract_overview_summary_fields(gui)
    display, tip = fields["stability_state"]
    assert display == NOT_AVAILABLE
    assert "42%" not in display
    assert "unstable" not in display.lower()
    blob = tip.lower()
    assert "0.04" in blob or "classification" in blob or "unvalidated" in blob
    assert "unstable" not in display.lower()


def test_gait_quality_session_details_lists_components() -> None:
    ba = SimpleNamespace(
        gait_quality=SimpleNamespace(
            score=71.0,
            confidence=0.66,
            components={"symmetry": 88.0, "stability_margin": 40.0},
            dominant_factors=["low stability margin"],
            explanation="Composite Gait Quality 71/100 (derived).",
        )
    )
    gui = SimpleNamespace(_biomech_analysis=ba)
    text = format_gait_quality_session_details(gui)
    assert text is not None
    assert "Score: 71/100" in text
    assert "symmetry: 88.0" in text
    assert "Missing components" in text
    assert "stability" in text  # listed as missing
    assert "Limiting" in text or "low stability" in text
    assert "Threshold source" in text
    assert "heuristic" in text.lower()
    assert "Formula:" in text


def test_symmetry_session_details_lists_formula_components_and_threshold() -> None:
    from stablewalk.analysis.biomechanical.types import MetricWithConfidence

    ba = SimpleNamespace(
        symmetry=SimpleNamespace(
            overall_symmetry_pct=MetricWithConfidence(
                value=81.0, unit="%", kind="derived", confidence=0.7, note=""
            ),
            stance_symmetry=MetricWithConfidence(
                value=0.91, unit="ratio", kind="derived", confidence=0.7, note="stance"
            ),
            swing_symmetry=MetricWithConfidence(
                value=0.88, unit="ratio", kind="derived", confidence=0.7, note="swing"
            ),
            step_length_symmetry=None,
            stride_length_symmetry=None,
            cadence_consistency=None,
            knee_rom_symmetry=None,
            hip_rom_symmetry=None,
            ankle_rom_symmetry=None,
            warnings=["Gait symmetry computed at reduced reliability: short clip."],
        )
    )
    text = format_symmetry_session_details(SimpleNamespace(_biomech_analysis=ba))
    assert text is not None
    assert "Overall: 81%" in text
    assert "Formula:" in text
    assert "stance_symmetry" in text
    assert "Missing inputs" in text
    assert "knee_rom_symmetry" in text
    assert "Threshold source" in text
    assert "heuristic" in text.lower()
    assert "Normal / Abnormal / Fast Walking" in text


def test_scored_gait_quality_tooltip_stays_heuristic_and_independent() -> None:
    gui = SimpleNamespace(
        _biomech_analysis=SimpleNamespace(
            gait_quality=SimpleNamespace(
                score=71.0,
                available=True,
                components={"symmetry": 88.0},
            ),
            symmetry=None,
            gait_metrics=None,
            stability_margin=None,
            video_quality=None,
        ),
        _gait_cycle=None,
        _stability_result=None,
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (2, 2),
    )
    display, tip = extract_overview_summary_fields(gui)["gait_quality"]
    assert display == "71/100"
    assert "heuristic" in tip.lower()
    assert "source video labels" in tip.lower()
    assert "clinical" in tip.lower()


def test_overview_does_not_use_gait_analysis_summary_as_gait_quality() -> None:
    """Coordination-group score must not appear as session Gait Quality."""
    alt = SimpleNamespace(
        score=26.0,
        available=True,
        components={"temporal_symmetry": 26.0},
        dominant_factors=[],
    )
    gui = SimpleNamespace(
        _biomech_analysis=SimpleNamespace(
            gait_quality=None,
            symmetry=None,
            gait_metrics=None,
            stability_margin=None,
            video_quality=None,
        ),
        _gait_cycle=None,
        _stability_result=SimpleNamespace(
            gait_summary=SimpleNamespace(gait_quality=alt)
        ),
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (2, 2),
    )
    fields = extract_overview_summary_fields(gui)
    assert fields["gait_quality"][0] == NOT_AVAILABLE
    assert "26" not in fields["gait_quality"][0]


def test_symmetry_na_after_analysis_uses_warning_not_not_computed() -> None:
    """Analysis ran but overall % is missing — keep N/A with the real reason."""
    ba = SimpleNamespace(
        gait_quality=None,
        symmetry=SimpleNamespace(
            overall_symmetry_pct=None,
            warnings=[
                "Overall symmetry unavailable: fewer than two valid bilateral components."
            ],
        ),
        gait_metrics=None,
        stability_margin=None,
        video_quality=None,
    )
    gui = SimpleNamespace(
        _biomech_analysis=ba,
        _gait_cycle=object(),
        _stability_result=None,
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (1, 1),
    )
    display, tip = extract_overview_summary_fields(gui)["symmetry"]
    assert display == NOT_AVAILABLE
    assert "not computed" not in tip.lower()
    assert "bilateral" in tip.lower()
    assert not any(ch.isdigit() for ch in display)


def test_traj_column_weights_match_lab_dashboard() -> None:
    from stablewalk.ui.tk.dashboard_sections import (
        SEC1_TRAJ_PATH_WEIGHT,
        SEC1_TRAJ_SKELETON_WEIGHT,
        SEC1_TRAJ_VIDEO_WEIGHT,
    )

    assert (SEC1_TRAJ_VIDEO_WEIGHT, SEC1_TRAJ_SKELETON_WEIGHT, SEC1_TRAJ_PATH_WEIGHT) == (
        40,
        30,
        30,
    )


def test_overview_withholds_normal_filename_heuristics_when_inputs_insufficient() -> None:
    from stablewalk.analysis.result_validity import result_validity_from_export_dict

    report = result_validity_from_export_dict(
        {
            "validity": {
                "pose_coverage": {"layer": "pose_coverage", "status": "limited"},
                "contact_left": {"layer": "contact_coverage", "status": "insufficient"},
                "contact_right": {"layer": "contact_coverage", "status": "insufficient"},
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
    ba = SimpleNamespace(
        gait_quality=SimpleNamespace(
            score=68.0,
            available=True,
            components={"symmetry": 70.0},
            explanation="renormalized",
        ),
        symmetry=SimpleNamespace(
            overall_symmetry_pct=SimpleNamespace(value=61.0, confidence=0.4),
            warnings=[],
        ),
        gait_metrics=None,
        stability_margin=None,
        video_quality=None,
    )
    gui = SimpleNamespace(
        _biomech_analysis=ba,
        _gait_cycle=None,
        _foot_contact=None,
        _spectral_report=None,
        _analysis_summary_cache=SimpleNamespace(result_validity=report),
        _stability_result=None,
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (4, 4),
    )
    fields = extract_overview_summary_fields(gui)
    assert fields["gait_quality"][0] == NOT_AVAILABLE
    assert "withheld" in fields["gait_quality"][1].lower()
    assert fields["symmetry"][0] == NOT_AVAILABLE
    assert "laterality" in fields["symmetry"][1].lower()
    assert fields["contact_coverage_left"][0] == "20% known"
    assert fields["contact_coverage_right"][0] == "37% known"
    assert fields["cycle_evidence"][0] == "4 L / 0 R accepted"
    assert fields["gait_cycles"][0] == "4 L / 0 R accepted"
    inputs = primary_finding_inputs(gui)
    assert inputs["gait_quality_score"] is None
    assert inputs["symmetry_pct"] is None
