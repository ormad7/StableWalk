"""
Real-to-Sim gait pipeline — video gait style → motion reference → retargeting → simulation.

Implements the 4-stage workflow described in ``docs/REAL_TO_SIM_PIPELINE.md``:

1. Perception — extract gait style fingerprint from video pose
2. Retargeting — scale human motion to a humanoid morphology (offline)
3. Simulation — AMP / Isaac Lab reference export (requires separate env)
4. Physics — contact-mask sync reward + virtual GRF estimation
"""

from stablewalk.real_to_sim.amp_reference_export import (
    AMPReferenceExportResult,
    export_amp_reference,
)
from stablewalk.real_to_sim.contact_sync_reward import (
    contact_force_sync_reward,
    summarize_contact_sync,
)
from stablewalk.real_to_sim.gait_style_extraction import (
    GaitStyleFingerprint,
    extract_gait_style_fingerprint,
)
from stablewalk.real_to_sim.motion_reference_loader import (
    load_motion_reference,
    validate_motion_reference,
)
from stablewalk.real_to_sim.pipeline import (
    RealToSimPipelineReport,
    run_real_to_sim_pipeline,
)
from stablewalk.real_to_sim.retargeting import (
    RetargetConfig,
    load_retarget_config,
    retarget_motion_reference,
)

__all__ = [
    "AMPReferenceExportResult",
    "GaitStyleFingerprint",
    "RealToSimPipelineReport",
    "RetargetConfig",
    "contact_force_sync_reward",
    "export_amp_reference",
    "extract_gait_style_fingerprint",
    "load_motion_reference",
    "load_retarget_config",
    "retarget_motion_reference",
    "run_real_to_sim_pipeline",
    "summarize_contact_sync",
    "validate_motion_reference",
]
