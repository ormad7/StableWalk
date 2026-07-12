"""Tests for OpenSim Inverse Dynamics readiness audit."""

from __future__ import annotations

from stablewalk.analysis.opensim_id_readiness import (
    OPENSIM_ID_REQUIRES_MEASURED_FORCES,
    assess_opensim_id_readiness,
)


def test_id_readiness_not_ready_without_external_loads(tmp_path, monkeypatch) -> None:
    from stablewalk import config

    run_name = "test_run"
    run_dir = tmp_path / "opensim" / run_name
    run_dir.mkdir(parents=True)
    (run_dir / f"{run_name}.mot").write_text("name test\n", encoding="utf-8")
    monkeypatch.setattr(config, "OPENSIM_DIR", tmp_path / "opensim")

    report = assess_opensim_id_readiness(run_name, body_mass_kg=70.0)

    assert report.ready_for_traditional_id is False
    assert report.run_name == run_name
    names = {c.name for c in report.checks}
    assert "external_loads_xml" in names
    assert "ground_reaction_forces" in names
    assert "center_of_pressure" in names
    assert any("contact masks" in w.lower() or "external loads" in w.lower() for w in report.warnings)
    assert "measured" in OPENSIM_ID_REQUIRES_MEASURED_FORCES.lower()


def test_id_readiness_ik_mot_partial_status(tmp_path, monkeypatch) -> None:
    from stablewalk import config

    run_name = "ik_run"
    run_dir = tmp_path / "opensim" / run_name
    run_dir.mkdir(parents=True)
    (run_dir / f"{run_name}_ik.mot").write_text("Coordinates\n", encoding="utf-8")
    monkeypatch.setattr(config, "OPENSIM_DIR", tmp_path / "opensim")

    report = assess_opensim_id_readiness(run_name)
    kin = next(c for c in report.checks if c.name == "joint_kinematics")
    assert kin.status == "present"
