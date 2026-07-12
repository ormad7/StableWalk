"""
Isaac Lab integration placeholders for StableWalk Real-to-Sim vGRF research.

Isaac Lab is **not** installed automatically in the OpenSim / MediaPipe environment.
Use a separate conda environment when NVIDIA Isaac Sim + Isaac Lab are available.

Planned pipeline
----------------
1. Export ``stablewalk_motion.npz`` via ``python main.py --export-motion-reference``
2. Retarget canonical joint trajectories to a humanoid URDF/USD in Isaac Lab
3. Run physics simulation with foot–ground contact
4. Read contact forces via ``ContactSensor`` or Isaac Lab contact-force APIs
5. Feed simulated forces into ``PhysicsSimulationForceEstimator``

Reference (future):
  - ``isaaclab.sensors.ContactSensor``
  - Articulation contact force queries on foot bodies
"""

from __future__ import annotations

ISAAC_LAB_AVAILABLE = False
ISAAC_LAB_PIPELINE_NOTE = (
    "Future Isaac Lab step: load stablewalk_motion.npz → retarget → simulate → "
    "read ContactSensor foot-ground forces. Requires separate Isaac Sim install."
)

MOTION_REFERENCE_FILENAME = "stablewalk_motion.npz"


def check_isaac_lab_available() -> tuple[bool, str]:
    """
    Probe for Isaac Lab without importing heavy dependencies at module load.

    Returns ``(False, message)`` in the default StableWalk environment.
    """
    try:
        import isaaclab  # noqa: F401

        return True, "Isaac Lab import succeeded (experimental — not validated)."
    except ImportError:
        return False, (
            "Isaac Lab not installed. Use a dedicated simulation environment; "
            "do not merge with the OpenSim Python stack."
        )


__all__ = [
    "ISAAC_LAB_AVAILABLE",
    "ISAAC_LAB_PIPELINE_NOTE",
    "MOTION_REFERENCE_FILENAME",
    "check_isaac_lab_available",
]
