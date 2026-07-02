"""
Presentation / professor demo configuration for StableWalk.

Loads immediately on startup in a stopped state (no video or pose JSON required) with
pre-selected degrees of freedom and a synthetic walking sequence that drives every
dashboard panel: skeleton, tables, trajectories, and step preview. Press Play to animate.
"""

from __future__ import annotations

from stablewalk.data.mock_gait import MockGaitConfig, generate_mock_gait
from stablewalk.models.gait_motion import GaitMotionRecording

# Shown in the sidebar “About” panel
DEMO_EXPLANATION = (
    "This system visualizes human gait using skeleton-based motion data. "
    "The user can inspect degrees of freedom, track joint positions over time, "
    "view 3D trajectories, and prepare the data for OpenSim-style biomechanical analysis."
)

# Default DOF highlights for the live walk-through
DEFAULT_DEMO_DOF_IDS: frozenset[str] = frozenset({"right_hip", "left_shoulder"})

PRESENTATION_VIDEO_CAPTION = (
    "Demo mode — synthetic walking skeleton\n"
    "For real video analysis, load or analyze a walking video from the toolbar."
)

PRESENTATION_SESSION_SUMMARY = "Presentation walk — synthetic gait cycle"
PRESENTATION_GAIT_HINT = "Demo selection: Right Hip + Left Shoulder"
PRESENTATION_STABILITY_LABEL = "Ready · skeleton + DOF + OpenSim-ready export"

# Slightly longer walk for smoother demo loops
PRESENTATION_GAIT_CONFIG = MockGaitConfig(
    fps=30.0,
    duration_s=5.0,
    cadence_hz=1.0,
    stride_length_m=0.58,
    pelvis_height_m=0.95,
    arm_swing_deg=24.0,
)

def generate_presentation_recording() -> GaitMotionRecording:
    """Build the canonical in-app walking sequence for live demos."""
    recording = generate_mock_gait(
        PRESENTATION_GAIT_CONFIG,
        source_label="mock://presentation_walk",
    )
    recording.build_time_series()
    return recording
