"""Joint-moment paths: kinematic statistics vs estimated vs measured ID.

Three kinds are distinguished explicitly:

1. ``kinematic_asymmetry`` — statistical 1st/2nd moments of 3D joint
   trajectories (means, covariances). Not torques.
2. ``estimated_joint_moment`` — optional sagittal ``M ≈ I α`` using
   Dempster/Winter segment inertia. Model-dependent and experimental.
3. ``measured_joint_moment`` — OpenSim Inverse Dynamics with measured GRF.
   Disabled unless ``assess_opensim_id_readiness`` reports ready. StableWalk
   sessions typically lack force plates; this path stays unavailable.

Never pass video vGRF proxies into InverseDynamicsTool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stablewalk.analysis.asymmetry3d.indices import symmetry_bundle
from stablewalk.analysis.asymmetry3d.kinematics import KinematicBundle, PAIRED_JOINTS
from stablewalk.analysis.opensim_id_readiness import (
    OPENSIM_ID_REQUIRES_MEASURED_FORCES,
    OpenSimIDReadinessReport,
    assess_opensim_id_readiness,
)
from stablewalk.coordinates.coordinate_map import DEFAULT_SUBJECT_HEIGHT_M

KIND_KINEMATIC = "kinematic_asymmetry"
KIND_ESTIMATED = "estimated_joint_moment"
KIND_MEASURED = "measured_joint_moment"

# Winter / Dempster sagittal inertia about the proximal joint.
# I_prox = m * L^2 * (k_com^2 + com_prox^2)
# mass_frac of body mass; length_frac of stature.
SEGMENT_INERTIA: dict[str, dict[str, float]] = {
    "hip": {
        "segment": "thigh",
        "mass_frac": 0.1000,
        "length_frac": 0.245,
        "k_com": 0.323,
        "com_prox": 0.433,
        "dof": "hip_flexion",
    },
    "knee": {
        "segment": "shank",
        "mass_frac": 0.0465,
        "length_frac": 0.246,
        "k_com": 0.302,
        "com_prox": 0.433,
        "dof": "knee_flexion",
    },
    "ankle": {
        "segment": "foot",
        "mass_frac": 0.0145,
        "length_frac": 0.152,
        "k_com": 0.475,
        "com_prox": 0.500,
        "dof": "ankle_flexion",
    },
}

ESTIMATED_MOMENT_ASSUMPTIONS = """
Estimated sagittal joint moments use a single rigid-segment planar model:
  M_i(t) = I_i * α_i(t)
  I_i = m_i * L_i² * (k_com² + d_com²)
with Dempster/Winter (Winter 2009) mass and radius-of-gyration fractions.
Required inputs: subject mass (kg) and stature (m). Ground reaction forces
are NOT used. This is not OpenSim Inverse Dynamics and is not a measured
joint moment. Disable when mass is missing. Hip-centered monocular α is
noisy; treat peaks as experimental.
""".strip()


@dataclass
class TrajectoryMoments:
    joint: str
    left_mean_xyz: tuple[float | None, float | None, float | None]
    right_mean_xyz: tuple[float | None, float | None, float | None]
    left_cov_trace: float | None
    right_cov_trace: float | None
    delta_mean_norm: float | None
    indices_trace: dict[str, float | None]
    kind: str = KIND_KINEMATIC

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "left_mean_xyz": list(self.left_mean_xyz),
            "right_mean_xyz": list(self.right_mean_xyz),
            "left_cov_trace": self.left_cov_trace,
            "right_cov_trace": self.right_cov_trace,
            "delta_mean_norm": self.delta_mean_norm,
            "indices_trace": dict(self.indices_trace),
            "kind": self.kind,
            "note": "Statistical moments of 3D trajectories, not joint torques.",
        }


@dataclass
class EstimatedMomentSeries:
    joint: str
    side: str
    times_s: np.ndarray
    moment_nm: np.ndarray
    inertia_kg_m2: float
    available: bool
    kind: str = KIND_ESTIMATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "side": self.side,
            "inertia_kg_m2": self.inertia_kg_m2,
            "rms_nm": _rms(self.moment_nm) if self.available else None,
            "peak_nm": _peak(self.moment_nm) if self.available else None,
            "available": self.available,
            "kind": self.kind,
        }


@dataclass
class MomentReport:
    kinematic: list[TrajectoryMoments] = field(default_factory=list)
    estimated: list[EstimatedMomentSeries] = field(default_factory=list)
    estimated_available: bool = False
    estimated_disabled_reason: str = ""
    measured_available: bool = False
    measured_disabled_reason: str = OPENSIM_ID_REQUIRES_MEASURED_FORCES
    opensim_id: dict[str, Any] | None = None
    subject_mass_kg: float | None = None
    subject_height_m: float | None = None
    assumptions: str = ESTIMATED_MOMENT_ASSUMPTIONS
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kinematic": [k.to_dict() for k in self.kinematic],
            "estimated": [e.to_dict() for e in self.estimated],
            "estimated_available": self.estimated_available,
            "estimated_disabled_reason": self.estimated_disabled_reason,
            "measured_available": self.measured_available,
            "measured_disabled_reason": self.measured_disabled_reason,
            "opensim_id": self.opensim_id,
            "subject_mass_kg": self.subject_mass_kg,
            "subject_height_m": self.subject_height_m,
            "assumptions": self.assumptions,
            "warnings": list(self.warnings),
            "kinds": {
                KIND_KINEMATIC: "L/R statistical moments of 3D joint paths",
                KIND_ESTIMATED: "Model-dependent I*α sagittal proxy",
                KIND_MEASURED: "OpenSim ID with measured GRF (typically unavailable)",
            },
        }


def _rms(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    return float(np.sqrt(np.mean(finite * finite)))


def _peak(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    return float(np.max(np.abs(finite)))


def _xyz_moments(points: np.ndarray) -> tuple[tuple[float | None, float | None, float | None], float | None]:
    ok = np.all(np.isfinite(points), axis=1)
    pts = points[ok]
    if pts.shape[0] < 4:
        return (None, None, None), None
    mean = np.mean(pts, axis=0)
    cov = np.cov(pts, rowvar=False)
    trace = float(np.trace(cov)) if cov.size else None
    return (float(mean[0]), float(mean[1]), float(mean[2])), trace


def kinematic_trajectory_moments(kin: KinematicBundle) -> list[TrajectoryMoments]:
    rows: list[TrajectoryMoments] = []
    for left_id, right_id in PAIRED_JOINTS:
        li = kin.joint_index(left_id)
        ri = kin.joint_index(right_id)
        if li is None or ri is None:
            continue
        l_mean, l_tr = _xyz_moments(kin.positions[:, li, :])
        r_mean, r_tr = _xyz_moments(kin.positions[:, ri, :])
        delta = None
        if all(v is not None for v in l_mean) and all(v is not None for v in r_mean):
            delta = math.sqrt(
                (l_mean[0] - r_mean[0]) ** 2
                + (l_mean[1] - r_mean[1]) ** 2
                + (l_mean[2] - r_mean[2]) ** 2
            )
        rows.append(
            TrajectoryMoments(
                joint=left_id.replace("left_", ""),
                left_mean_xyz=l_mean,
                right_mean_xyz=r_mean,
                left_cov_trace=l_tr,
                right_cov_trace=r_tr,
                delta_mean_norm=delta,
                indices_trace=symmetry_bundle(l_tr, r_tr),
            )
        )
    return rows


def proximal_inertia_kg_m2(
    joint: str,
    *,
    mass_kg: float,
    height_m: float,
) -> float | None:
    spec = SEGMENT_INERTIA.get(joint)
    if spec is None or mass_kg <= 0 or height_m <= 0:
        return None
    m_i = spec["mass_frac"] * mass_kg
    L = spec["length_frac"] * height_m
    k = spec["k_com"]
    d = spec["com_prox"]
    return float(m_i * (L ** 2) * (k * k + d * d))


def _estimate_series(
    kin: KinematicBundle,
    joint: str,
    side: str,
    inertia: float,
) -> EstimatedMomentSeries:
    dof = f"{side}_{SEGMENT_INERTIA[joint]['dof']}"
    idx = kin.dof_index(dof)
    times = kin.times_s
    if idx is None:
        return EstimatedMomentSeries(
            joint=joint,
            side=side,
            times_s=times,
            moment_nm=np.full(times.shape, np.nan),
            inertia_kg_m2=inertia,
            available=False,
        )
    alpha_rad = np.deg2rad(kin.alpha_deg_s2[:, idx])
    moment = inertia * alpha_rad
    return EstimatedMomentSeries(
        joint=joint,
        side=side,
        times_s=times,
        moment_nm=moment,
        inertia_kg_m2=inertia,
        available=bool(np.isfinite(moment).sum() >= 8),
    )


def build_moment_report(
    kin: KinematicBundle,
    *,
    subject_mass_kg: float | None = None,
    subject_height_m: float | None = None,
    opensim_run_name: str | None = None,
) -> MomentReport:
    warnings: list[str] = [
        "Kinematic trajectory moments are statistical, not inverse-dynamics torques."
    ]
    height = float(subject_height_m) if subject_height_m else DEFAULT_SUBJECT_HEIGHT_M
    mass = float(subject_mass_kg) if subject_mass_kg and subject_mass_kg > 0 else None

    id_report: OpenSimIDReadinessReport | None = None
    if opensim_run_name:
        try:
            id_report = assess_opensim_id_readiness(opensim_run_name)
        except Exception as exc:
            warnings.append(f"OpenSim ID readiness check failed: {exc}")

    measured_ready = bool(id_report.ready_for_traditional_id) if id_report else False
    measured_reason = OPENSIM_ID_REQUIRES_MEASURED_FORCES
    if id_report is not None and not measured_ready:
        missing = [c.name for c in id_report.checks if c.status == "missing"]
        measured_reason = (
            "OpenSim Inverse Dynamics not ready ("
            + (", ".join(missing) if missing else "requirements unmet")
            + "). "
            + OPENSIM_ID_REQUIRES_MEASURED_FORCES
        )
    if measured_ready:
        measured_reason = (
            "ID prerequisites flagged ready, but StableWalk does not execute "
            "InverseDynamicsTool here. No measured joint-moment series is attached."
        )
        warnings.append(measured_reason)

    estimated: list[EstimatedMomentSeries] = []
    estimated_ok = mass is not None
    disabled = ""
    if not estimated_ok:
        disabled = (
            "Estimated joint moments disabled: subject mass (kg) is required. "
            "Measured GRF is also absent, so OpenSim ID is not used."
        )
        warnings.append(disabled)
    else:
        warnings.append(
            "Estimated sagittal moments are I*α with Dempster/Winter inertia — "
            "experimental and model-dependent."
        )
        for joint in SEGMENT_INERTIA:
            inertia = proximal_inertia_kg_m2(joint, mass_kg=mass, height_m=height)
            if inertia is None:
                continue
            estimated.append(_estimate_series(kin, joint, "left", inertia))
            estimated.append(_estimate_series(kin, joint, "right", inertia))

    return MomentReport(
        kinematic=kinematic_trajectory_moments(kin),
        estimated=estimated,
        estimated_available=estimated_ok and any(s.available for s in estimated),
        estimated_disabled_reason=disabled,
        measured_available=False,
        measured_disabled_reason=measured_reason,
        opensim_id=id_report.to_dict() if id_report else None,
        subject_mass_kg=mass,
        subject_height_m=height if mass is not None else None,
        warnings=warnings,
    )
