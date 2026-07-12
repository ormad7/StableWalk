"""
Estimated / Simulated Virtual Ground Reaction Force (vGRF) — research architecture.

Scientific scope
----------------
Monocular video does **not** measure ground reaction forces. StableWalk uses the
terminology **Estimated Virtual Ground Reaction Force** or **Simulated Ground
Reaction Force** for physics-based or learned proxies intended for comparative
research — not instrumented kinetics.

Foot contact detection (clearance / velocity heuristics) indicates *when* a foot
is likely on the ground. It is **not** force data and must not be presented as GRF.

Traditional OpenSim Inverse Dynamics requires measured external loads; see
``opensim_id_readiness`` and ``docs/VIRTUAL_GRF.md``.

Future Real-to-Sim pipeline (planned)
-------------------------------------
Video → Pose/HMR → Canonical 3D Motion → Gait Contact Mask → Motion Retargeting
→ Physics-Based Humanoid Simulation (e.g. Isaac Lab) → Contact Forces → vGRF Analysis

Isaac Lab ``ContactSensor`` or physics contact-force APIs may read simulated
foot–ground interaction in a **separate** simulation environment. Isaac Lab is
not installed in the OpenSim Python environment.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult, GaitEvent
from stablewalk.analysis.gait_feature_analysis import (
    BodySegmentDimensions,
    CycleConsistencyResult,
    GaitFeatureAnalysisResult,
)
from stablewalk.models.gait_motion import GaitMotionRecording, Vec3

logger = logging.getLogger(__name__)

G = 9.81

SCIENTIFIC_DISCLAIMER = (
    "Estimated Virtual Ground Reaction Force (vGRF) — not measured kinetics. "
    "Monocular video cannot directly measure ground reaction forces. "
    "Foot contact masks indicate timing only, not force magnitude. "
    "Do not use vGRF output as OpenSim Inverse Dynamics external loads."
)

VGRF_TERMINOLOGY = "Estimated Virtual Ground Reaction Force"


class VirtualForceMethod(str, Enum):
    """How vGRF was produced."""

    UNAVAILABLE = "unavailable"
    PHYSICS_SIMULATION = "physics_simulation"  # future: Isaac Lab / MuJoCo
    LEARNED = "learned"  # future: ML estimator
    LEGACY_POSE_PROXY = "legacy_pose_proxy"  # existing GRFAnalyzer (separate module)


@dataclass(frozen=True)
class VirtualForceEstimatorInput:
    """
    Data contract for virtual GRF estimators.

    All arrays are time-aligned to ``timestamps`` (seconds).
    """

    timestamps: np.ndarray
    fps: float
    left_contact_mask: np.ndarray
    right_contact_mask: np.ndarray
    contact_events: tuple[GaitEvent, ...]
    body_mass_kg: float
    body_dimensions: BodySegmentDimensions
    root_positions: np.ndarray  # (N, 3) pelvis/root trajectory
    root_orientations: np.ndarray | None  # (N, 4) quat wxyz or None
    joint_positions: dict[str, np.ndarray]  # joint_id -> (N, 3)
    joint_rotations: dict[str, np.ndarray] | None  # optional, estimator-specific
    gait_cycle_kinematics: CycleConsistencyResult | None
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VirtualForceResult:
    """
    Output of a :class:`VirtualForceEstimator`.

    ``vgrf_*_bw`` = vertical force / (mass * 9.81) — body-weight normalized.
    """

    method: VirtualForceMethod
    confidence: float  # 0..1
    scientific_disclaimer: str = SCIENTIFIC_DISCLAIMER
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    left_vgrf_n: np.ndarray = field(default_factory=lambda: np.array([]))
    right_vgrf_n: np.ndarray = field(default_factory=lambda: np.array([]))
    left_vgrf_bw: np.ndarray = field(default_factory=lambda: np.array([]))
    right_vgrf_bw: np.ndarray = field(default_factory=lambda: np.array([]))
    left_force_3d: np.ndarray | None = None  # (N, 3) optional
    right_force_3d: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)
    available: bool = False

    @property
    def estimation_method_label(self) -> str:
        return self.method.value.replace("_", " ").title()

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminology": VGRF_TERMINOLOGY,
            "method": self.method.value,
            "confidence": self.confidence,
            "available": self.available,
            "scientific_disclaimer": self.scientific_disclaimer,
            "sample_count": int(len(self.timestamps)),
            "notes": list(self.notes),
        }


class VirtualForceEstimator(ABC):
    """Interface for physics-based or learned virtual GRF estimation."""

    @property
    @abstractmethod
    def method(self) -> VirtualForceMethod:
        """Estimator identifier."""

    @abstractmethod
    def estimate(self, data: VirtualForceEstimatorInput) -> VirtualForceResult:
        """Produce vGRF trajectories from kinematic + contact inputs."""


class UnavailableForceEstimator(VirtualForceEstimator):
    """Default placeholder — no vGRF engine configured."""

    @property
    def method(self) -> VirtualForceMethod:
        return VirtualForceMethod.UNAVAILABLE

    def estimate(self, data: VirtualForceEstimatorInput) -> VirtualForceResult:
        return VirtualForceResult(
            method=self.method,
            confidence=0.0,
            timestamps=np.asarray(data.timestamps, dtype=float),
            available=False,
            notes=[
                "Virtual force estimation is not configured.",
                "Export stablewalk_motion.npz and run a future Isaac Lab pipeline.",
                "Foot contact masks are available for timing only — not force magnitude.",
            ],
        )


class PhysicsSimulationForceEstimator(VirtualForceEstimator):
    """
    Placeholder for Isaac Lab / physics humanoid simulation integration.

    Planned workflow:
      1. Load ``stablewalk_motion.npz`` retargeting reference
      2. Spawn articulated humanoid in Isaac Lab (separate env — not OpenSim)
      3. Replay / track retargeted motion with PD or imitation controller
      4. Read ``ContactSensor`` or ``get_contact_forces`` per foot
      5. Map simulated contact forces to vGRF trajectories

    See ``docs/VIRTUAL_GRF.md`` and ``isaac_lab_integration`` module stub.
    """

    @property
    def method(self) -> VirtualForceMethod:
        return VirtualForceMethod.PHYSICS_SIMULATION

    def estimate(self, data: VirtualForceEstimatorInput) -> VirtualForceResult:
        from stablewalk.analysis.isaac_lab_integration import ISAAC_LAB_PIPELINE_NOTE

        return VirtualForceResult(
            method=self.method,
            confidence=0.0,
            timestamps=np.asarray(data.timestamps, dtype=float),
            available=False,
            notes=[
                "PhysicsSimulationForceEstimator is a research placeholder.",
                ISAAC_LAB_PIPELINE_NOTE,
                "Install Isaac Lab in a separate conda env — not bundled with OpenSim.",
            ],
        )


class LearnedForceEstimator(VirtualForceEstimator):
    """Placeholder for a future learned vGRF model (e.g. video-to-force network)."""

    @property
    def method(self) -> VirtualForceMethod:
        return VirtualForceMethod.LEARNED

    def estimate(self, data: VirtualForceEstimatorInput) -> VirtualForceResult:
        return VirtualForceResult(
            method=self.method,
            confidence=0.0,
            timestamps=np.asarray(data.timestamps, dtype=float),
            available=False,
            notes=[
                "LearnedForceEstimator is not trained or deployed.",
                "Requires labeled force-plate datasets for supervised learning.",
            ],
        )


DEFAULT_VIRTUAL_FORCE_ESTIMATOR: VirtualForceEstimator = UnavailableForceEstimator()


def build_virtual_force_input(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult,
    *,
    body_mass_kg: float = 70.0,
    body_dimensions: BodySegmentDimensions | None = None,
    gait_features: GaitFeatureAnalysisResult | None = None,
) -> VirtualForceEstimatorInput:
    """Assemble the vGRF estimator input contract from gait analysis artifacts."""
    from stablewalk.analysis.gait_feature_analysis import estimate_body_segment_dimensions
    from stablewalk.models.joint_registry import JOINT_IDS, ROOT_JOINT_ID

    dimensions = body_dimensions or estimate_body_segment_dimensions(recording)
    cycle_kin = gait_features.cycle_consistency if gait_features else None

    n = len(recording.snapshots)
    timestamps = np.array([s.time_s for s in recording.snapshots], dtype=float)
    root_positions = np.zeros((n, 3), dtype=float)
    joint_positions: dict[str, np.ndarray] = {}

    for jid in (ROOT_JOINT_ID, *JOINT_IDS):
        joint_positions[jid] = np.full((n, 3), np.nan, dtype=float)

    for i, snap in enumerate(recording.snapshots):
        pelvis = _pelvis_from_snapshot(snap)
        if pelvis is not None:
            root_positions[i] = (pelvis.x, pelvis.y, pelvis.z)
        for jid, arr in joint_positions.items():
            joint = snap.joints.get(jid)
            if joint is not None:
                arr[i] = (joint.position.x, joint.position.y, joint.position.z)

    left_mask = np.zeros(n, dtype=np.int8)
    right_mask = np.zeros(n, dtype=np.int8)
    for state in cycles.per_frame:
        idx = next(
            (i for i, s in enumerate(recording.snapshots) if s.frame_index == state.frame_index),
            None,
        )
        if idx is not None:
            left_mask[idx] = int(state.left_contact)
            right_mask[idx] = int(state.right_contact)

    return VirtualForceEstimatorInput(
        timestamps=timestamps,
        fps=float(recording.fps),
        left_contact_mask=left_mask,
        right_contact_mask=right_mask,
        contact_events=tuple(cycles.events),
        body_mass_kg=body_mass_kg,
        body_dimensions=dimensions,
        root_positions=root_positions,
        root_orientations=None,
        joint_positions=joint_positions,
        joint_rotations=None,
        gait_cycle_kinematics=cycle_kin,
        source_id=recording.source or "",
        metadata={
            "coordinate_system": recording.coordinate_system,
            "contact_confidence": cycles.metrics.contact_confidence,
        },
    )


def estimate_virtual_grf(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult,
    *,
    estimator: VirtualForceEstimator | None = None,
    body_mass_kg: float = 70.0,
    gait_features: GaitFeatureAnalysisResult | None = None,
) -> VirtualForceResult:
    """Run the configured virtual force estimator (default: unavailable placeholder)."""
    est = estimator or DEFAULT_VIRTUAL_FORCE_ESTIMATOR
    data = build_virtual_force_input(
        recording,
        cycles,
        body_mass_kg=body_mass_kg,
        gait_features=gait_features,
    )
    result = est.estimate(data)
    if result.available and len(result.left_vgrf_n):
        bw = max(body_mass_kg * G, 1e-6)
        result.left_vgrf_bw = result.left_vgrf_n / bw
        result.right_vgrf_bw = result.right_vgrf_n / bw
    return result


def _pelvis_from_snapshot(snap) -> Vec3 | None:
    lh = snap.joints.get("left_hip")
    rh = snap.joints.get("right_hip")
    if lh is None or rh is None:
        return None
    return Vec3(
        (lh.position.x + rh.position.x) * 0.5,
        (lh.position.y + rh.position.y) * 0.5,
        (lh.position.z + rh.position.z) * 0.5,
    )


__all__ = [
    "G",
    "SCIENTIFIC_DISCLAIMER",
    "VGRF_TERMINOLOGY",
    "VirtualForceMethod",
    "VirtualForceEstimatorInput",
    "VirtualForceResult",
    "VirtualForceEstimator",
    "UnavailableForceEstimator",
    "PhysicsSimulationForceEstimator",
    "LearnedForceEstimator",
    "DEFAULT_VIRTUAL_FORCE_ESTIMATOR",
    "build_virtual_force_input",
    "estimate_virtual_grf",
]
