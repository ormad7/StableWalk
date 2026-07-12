"""
OpenSim Inverse Dynamics readiness audit for StableWalk.

Traditional OpenSim Inverse Dynamics (ID) solves for joint moments and muscle
forces given:

  1. A **subject-specific scaled** musculoskeletal model (``ScaleTool`` output)
  2. **Joint kinematics** from Inverse Kinematics (``InverseKinematicsTool``)
  3. **Measured external loads** — typically force-plate ground reaction forces
     (Fx, Fy, Fz) and center of pressure (COP), referenced in ``ExternalLoads.xml``

StableWalk currently exports video-derived kinematics and can run experimental IK.
It does **not** provide measured GRF/COP or an ``ExternalLoads`` configuration.
Feeding invented forces into OpenSim ID would produce non-physical results and must
not be done.

Use :func:`assess_opensim_id_readiness` to report what is present vs missing before
any future ID integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from stablewalk import config

logger = logging.getLogger(__name__)

ReadinessStatus = Literal["present", "partial", "missing", "not_applicable"]

# OpenSim ID generally requires measured kinetics — not video proxies.
OPENSIM_ID_REQUIRES_MEASURED_FORCES = """
OpenSim Inverse Dynamics requires measured external forces (force plates or
validated instrumented walkways) referenced through ExternalLoads.xml.

Video-derived vertical GRF proxies and foot contact masks are **not** substitutes
for measured kinetics. Do not pass estimated or simulated forces into
InverseDynamicsTool as if they were instrumented data.
""".strip()


@dataclass(frozen=True)
class IDRequirementCheck:
    """Status of one Inverse Dynamics prerequisite."""

    name: str
    status: ReadinessStatus
    detail: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "path": self.path,
        }


@dataclass
class OpenSimIDReadinessReport:
    """
    Audit of data available for traditional OpenSim Inverse Dynamics.

    ``ready_for_traditional_id`` is True only when scaled model, IK kinematics,
    subject mass in the model, and measured external loads (GRF + COP + XML) are
    all present. StableWalk sessions typically fail on external loads.
    """

    run_name: str
    checks: list[IDRequirementCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ready_for_traditional_id: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "ready_for_traditional_id": self.ready_for_traditional_id,
            "checks": [c.to_dict() for c in self.checks],
            "warnings": list(self.warnings),
            "note": OPENSIM_ID_REQUIRES_MEASURED_FORCES,
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"OpenSim ID readiness ({self.run_name}): "
            f"{'READY' if self.ready_for_traditional_id else 'NOT READY'}",
        ]
        for check in self.checks:
            lines.append(f"  [{check.status.upper():7}] {check.name}: {check.detail}")
        return lines


def _run_dir(run_name: str) -> Path:
    return config.OPENSIM_DIR / run_name


def _find_scaled_model(model_path: Path | None) -> IDRequirementCheck:
    if model_path is None:
        demo = config.OPENSIM_MODELS_DIR / "Gait2392_Pipeline" / "subject01_simbody.osim"
        if demo.is_file():
            return IDRequirementCheck(
                name="scaled_opensim_model",
                status="partial",
                detail=(
                    "Demo pre-scaled model available; no subject-specific ScaleTool output "
                    "from this session."
                ),
                path=str(demo),
            )
        return IDRequirementCheck(
            name="scaled_opensim_model",
            status="missing",
            detail="No scaled .osim model path provided and demo model not found.",
        )
    if not model_path.is_file():
        return IDRequirementCheck(
            name="scaled_opensim_model",
            status="missing",
            detail=f"Model file not found: {model_path}",
            path=str(model_path),
        )
    name_lower = model_path.name.lower()
    if "unscaled" in name_lower or "generic" in name_lower:
        return IDRequirementCheck(
            name="scaled_opensim_model",
            status="partial",
            detail="Model path looks like an unscaled template, not ScaleTool output.",
            path=str(model_path),
        )
    return IDRequirementCheck(
        name="scaled_opensim_model",
        status="present",
        detail="Scaled musculoskeletal model file found.",
        path=str(model_path),
    )


def _check_subject_mass(body_mass_kg: float | None) -> IDRequirementCheck:
    if body_mass_kg is None or body_mass_kg <= 0:
        return IDRequirementCheck(
            name="subject_mass",
            status="missing",
            detail="Subject mass not configured for OpenSim model properties.",
        )
    return IDRequirementCheck(
        name="subject_mass",
        status="partial",
        detail=(
            f"Estimated mass {body_mass_kg:.1f} kg available for pose-based analysis only; "
            "not written into OpenSim model mass properties."
        ),
    )


def _check_joint_kinematics(run_name: str) -> IDRequirementCheck:
    run_dir = _run_dir(run_name)
    ik_mot = run_dir / f"{run_name}_ik.mot"
    pose_mot = run_dir / f"{run_name}.mot"
    if ik_mot.is_file():
        return IDRequirementCheck(
            name="joint_kinematics",
            status="present",
            detail="OpenSim IK generalized coordinates available.",
            path=str(ik_mot),
        )
    if pose_mot.is_file():
        return IDRequirementCheck(
            name="joint_kinematics",
            status="partial",
            detail="Pose-derived .mot exported; OpenSim IK .mot not yet run.",
            path=str(pose_mot),
        )
    return IDRequirementCheck(
        name="joint_kinematics",
        status="missing",
        detail="No .mot kinematics found under OPENSIM_DIR for this run.",
        path=str(run_dir),
    )


def _check_external_loads(run_name: str) -> IDRequirementCheck:
    run_dir = _run_dir(run_name)
    for pattern in ("ExternalLoads.xml", "external_loads.xml", "*ExternalLoads*.xml"):
        matches = list(run_dir.glob(pattern))
        if matches:
            return IDRequirementCheck(
                name="external_loads_xml",
                status="present",
                detail="ExternalLoads XML found (verify forces are measured, not estimated).",
                path=str(matches[0]),
            )
    return IDRequirementCheck(
        name="external_loads_xml",
        status="missing",
        detail="No ExternalLoads.xml — traditional OpenSim ID cannot run.",
        path=str(run_dir),
    )


def _check_measured_grf(run_name: str) -> IDRequirementCheck:
    run_dir = _run_dir(run_name)
    for pattern in ("*GRF*.mot", "*grf*.sto", "*force*.mot", "*Force*.sto"):
        matches = list(run_dir.glob(pattern))
        if matches:
            return IDRequirementCheck(
                name="ground_reaction_forces",
                status="partial",
                detail="Force storage file found — confirm data are measured force-plate GRF.",
                path=str(matches[0]),
            )
    return IDRequirementCheck(
        name="ground_reaction_forces",
        status="missing",
        detail=(
            "No measured force-plate GRF file. StableWalk pose-based GRF is not OpenSim ID input."
        ),
    )


def _check_cop(run_name: str) -> IDRequirementCheck:
    run_dir = _run_dir(run_name)
    for pattern in ("*COP*.mot", "*cop*.sto", "*center_of_pressure*"):
        matches = list(run_dir.glob(pattern))
        if matches:
            return IDRequirementCheck(
                name="center_of_pressure",
                status="partial",
                detail="COP storage found — confirm instrumented measurement.",
                path=str(matches[0]),
            )
    return IDRequirementCheck(
        name="center_of_pressure",
        status="missing",
        detail="No center-of-pressure data for OpenSim external loads.",
    )


def assess_opensim_id_readiness(
    run_name: str,
    *,
    model_path: Path | str | None = None,
    body_mass_kg: float | None = 70.0,
) -> OpenSimIDReadinessReport:
    """
    Inspect StableWalk + OpenSim artifacts for traditional Inverse Dynamics readiness.

    Returns an honest report. ``ready_for_traditional_id`` remains False unless
    measured external loads and ExternalLoads XML are present alongside IK output.
    """
    model = Path(model_path) if model_path else None
    checks = [
        _find_scaled_model(model),
        _check_subject_mass(body_mass_kg),
        _check_joint_kinematics(run_name),
        _check_external_loads(run_name),
        _check_measured_grf(run_name),
        _check_cop(run_name),
    ]

    warnings: list[str] = []
    if not any(c.status == "present" for c in checks if c.name == "ground_reaction_forces"):
        warnings.append(
            "Video-derived contact masks and pose-based GRF proxies must not be used "
            "as OpenSim Inverse Dynamics external loads."
        )
    if not any(c.status == "present" for c in checks if c.name == "external_loads_xml"):
        warnings.append(OPENSIM_ID_REQUIRES_MEASURED_FORCES.split("\n")[0])

    ready = all(
        c.status == "present"
        for c in checks
        if c.name
        in (
            "scaled_opensim_model",
            "joint_kinematics",
            "external_loads_xml",
            "ground_reaction_forces",
            "center_of_pressure",
        )
    )

    report = OpenSimIDReadinessReport(
        run_name=run_name,
        checks=checks,
        warnings=warnings,
        ready_for_traditional_id=ready,
    )
    for line in report.summary_lines():
        logger.debug(line)
    return report


__all__ = [
    "IDRequirementCheck",
    "OpenSimIDReadinessReport",
    "OPENSIM_ID_REQUIRES_MEASURED_FORCES",
    "ReadinessStatus",
    "assess_opensim_id_readiness",
]
