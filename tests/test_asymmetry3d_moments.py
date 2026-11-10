"""Kinematic vs estimated vs measured joint-moment distinction (req 4D)."""

from __future__ import annotations

import math

from stablewalk.analysis.asymmetry3d.moments import (
    KIND_ESTIMATED,
    KIND_KINEMATIC,
    KIND_MEASURED,
    ESTIMATED_MOMENT_ASSUMPTIONS,
    proximal_inertia_kg_m2,
)
from stablewalk.analysis.asymmetry3d.report import analyze_asymmetry3d
from stablewalk.analysis.opensim_id_readiness import OPENSIM_ID_REQUIRES_MEASURED_FORCES

from test_asymmetry3d import _walking_recording


def test_moments_disabled_without_mass() -> None:
    rec, cycles = _walking_recording(n_strides=4)
    report = analyze_asymmetry3d(rec, cycles=cycles, subject_mass_kg=None)
    mom = report.moments
    assert mom is not None
    assert mom.estimated_available is False
    assert "mass" in mom.estimated_disabled_reason.lower()
    assert mom.measured_available is False
    assert "force" in mom.measured_disabled_reason.lower() or "GRF" in mom.measured_disabled_reason
    assert mom.kinematic
    assert mom.kinematic[0].kind == KIND_KINEMATIC


def test_estimated_moments_require_mass_and_are_labeled() -> None:
    rec, cycles = _walking_recording(n_strides=4)
    report = analyze_asymmetry3d(
        rec,
        cycles=cycles,
        subject_mass_kg=70.0,
        subject_height_m=1.70,
    )
    mom = report.moments
    assert mom is not None
    assert mom.estimated_available is True
    assert mom.measured_available is False
    assert mom.subject_mass_kg == 70.0
    knee = [s for s in mom.estimated if s.joint == "knee"]
    assert len(knee) == 2
    assert all(s.kind == KIND_ESTIMATED for s in knee)
    assert all(s.available for s in knee)
    i_knee = proximal_inertia_kg_m2("knee", mass_kg=70.0, height_m=1.70)
    assert i_knee is not None and i_knee > 0
    for s in knee:
        assert math.isclose(s.inertia_kg_m2, i_knee, rel_tol=1e-9)
        finite = s.moment_nm[s.moment_nm == s.moment_nm]
        assert finite.size >= 8
    assert "Dempster" in ESTIMATED_MOMENT_ASSUMPTIONS
    assert KIND_MEASURED in mom.to_dict()["kinds"]


def test_measured_id_not_claimed_without_grf() -> None:
    rec, cycles = _walking_recording(n_strides=3)
    report = analyze_asymmetry3d(rec, cycles=cycles, subject_mass_kg=70.0)
    mom = report.moments
    assert mom is not None
    assert mom.measured_available is False
    assert OPENSIM_ID_REQUIRES_MEASURED_FORCES[:40] in mom.measured_disabled_reason or "GRF" in mom.measured_disabled_reason
    blob = mom.to_dict()
    assert blob["kinds"][KIND_KINEMATIC]
    assert "statistical" in blob["kinds"][KIND_KINEMATIC].lower()


def test_symmetric_paths_have_small_kinematic_moment_delta() -> None:
    rec, cycles = _walking_recording(left_amp=40.0, right_amp=40.0, n_strides=5)
    report = analyze_asymmetry3d(rec, cycles=cycles)
    traces = [k for k in report.moments.kinematic if k.joint in ("knee", "ankle", "heel")]
    assert traces
    for row in traces:
        if row.indices_trace.get("asi_percent") is None:
            continue
        assert row.indices_trace["asi_percent"] < 35.0
