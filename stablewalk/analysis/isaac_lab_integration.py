"""
Isaac Lab integration for StableWalk Real-to-Sim pipeline.

Isaac Lab is **not** installed in the default StableWalk environment.
Use a separate conda environment when NVIDIA Isaac Sim + Isaac Lab are available.

4-stage pipeline (see ``docs/REAL_TO_SIM_PIPELINE.md``)
---------------------------------------------------------
1. Perception   — ``stablewalk/real_to_sim/gait_style_extraction.py``
2. Retargeting  — ``stablewalk/real_to_sim/retargeting.py`` + ``models/real_to_sim/``
3. Simulation   — ``stablewalk/real_to_sim/amp_reference_export.py`` → Isaac Lab AMP
4. Physics      — ``ContactSensor`` foot forces → ``virtual_grf.py``

Offline entry point::

    python main.py --real-to-sim walk_stream
"""

from __future__ import annotations

from pathlib import Path

ISAAC_LAB_AVAILABLE = False
ISAAC_LAB_PIPELINE_NOTE = (
    "Isaac Lab step: load amp_reference_motion.npz → spawn Unitree G1 → "
    "run AMP training → read ContactSensor foot-ground forces. "
    "Requires separate Isaac Sim install."
)

MOTION_REFERENCE_FILENAME = "stablewalk_motion.npz"
AMP_REFERENCE_FILENAME = "amp_reference_motion.npz"

RETARGET_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "real_to_sim" / "unitree_g1_retarget.json"
)

PIPELINE_STAGES = (
    {
        "id": "1_perception",
        "name": "Perception",
        "tool": "MediaPipe / ROMP / HybrIK",
        "output": "Gait style fingerprint + stablewalk_motion.npz",
        "stablewalk_module": "stablewalk.real_to_sim.gait_style_extraction",
    },
    {
        "id": "2_retargeting",
        "name": "Retargeting",
        "tool": "Uniform scale + GMR (Isaac Lab)",
        "output": "Scaled joint trajectories for Unitree G1/H1",
        "stablewalk_module": "stablewalk.real_to_sim.retargeting",
    },
    {
        "id": "3_simulation",
        "name": "Simulation (AMP)",
        "tool": "Isaac Lab + Adversarial Motion Priors",
        "output": "RL policy mimicking video gait style",
        "stablewalk_module": "stablewalk.real_to_sim.amp_reference_export",
    },
    {
        "id": "4_physics",
        "name": "Physics & vGRF",
        "tool": "ContactSensor / PhysX contact forces",
        "output": "Virtual ground reaction forces + contact-sync reward",
        "stablewalk_module": "stablewalk.analysis.virtual_grf",
    },
)


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
            "Isaac Lab not installed. Export AMP reference with "
            "`python main.py --real-to-sim` and train in a dedicated simulation env."
        )


def generate_isaac_lab_runner_notes() -> str:
    """Return setup notes for Isaac Lab AMP training (text only)."""
    return """
Isaac Lab AMP training (separate environment)
-----------------------------------------------
1. Install Isaac Sim + Isaac Lab (not in StableWalk OpenSim env).
2. Copy ``data/output/motion_reference/{run}/amp_reference_motion.npz``.
3. Load Unitree G1 URDF from ``models/real_to_sim/unitree_g1_retarget.json``.
4. Configure ContactSensor on foot links (L_FOOT, R_FOOT).
5. Run AMP training with reference motion clip.
6. Use contact_sync_reward to align video contact masks with simulated GRF.

Reference: Minimal G1 AMP repository (Isaac Lab manager-based wrapper).
""".strip()


__all__ = [
    "AMP_REFERENCE_FILENAME",
    "ISAAC_LAB_AVAILABLE",
    "ISAAC_LAB_PIPELINE_NOTE",
    "MOTION_REFERENCE_FILENAME",
    "PIPELINE_STAGES",
    "RETARGET_CONFIG_PATH",
    "check_isaac_lab_available",
    "generate_isaac_lab_runner_notes",
]
