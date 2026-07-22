"""
Canonical scientific terminology for StableWalk GUI, charts, and exports.

Use these strings for user-facing copy so measured / estimated / derived /
calculated tiers stay consistent across tabs and reports.
"""

from __future__ import annotations

# ── Data tier definitions (also embedded in JSON exports) ─────────────────────
TIER_MEASURED = "measured"
TIER_ESTIMATED = "estimated"
TIER_DERIVED = "derived"
TIER_CALCULATED = "calculated"

TIER_DEFINITIONS: dict[str, str] = {
    TIER_MEASURED: (
        "Instrumented or direct sensor data (force plate, mocap, OpenSim IK after calibration)."
    ),
    TIER_ESTIMATED: (
        "Pose-derived proxy using anthropometric or heuristic models (monocular video)."
    ),
    TIER_DERIVED: (
        "Computed from estimated inputs (symmetry index, stability margin, gait quality)."
    ),
    TIER_CALCULATED: (
        "Deterministic post-processing of timestamps or geometry (cadence, phase %, finite differences)."
    ),
}

# ── Confidence (scoped labels — always title case) ───────────────────────────
LABEL_ANALYSIS_CONFIDENCE = "Analysis Confidence"
LABEL_TRACKING_CONFIDENCE = "Tracking Confidence"
LABEL_PIPELINE_CONFIDENCE = "Pipeline Confidence"
LABEL_CONTACT_CONFIDENCE = "Contact Confidence"
LABEL_FOOT_CLEARANCE_CONFIDENCE = "Foot Clearance Confidence"
LABEL_TRAJECTORY_CONFIDENCE = "Trajectory Confidence"
LABEL_VIEW_CONFIDENCE = "View Confidence"
LABEL_DOMAIN_CONFIDENCE = "Domain Confidence"

# ── Force / contact ─────────────────────────────────────────────────────────
LABEL_VIRTUAL_GRF_FULL = "Estimated Virtual GRF (vGRF)"
LABEL_VIRTUAL_GRF_SHORT = "vGRF (est.)"
LABEL_VIRTUAL_GRF_PANEL = "Estimated Virtual GRF"
LABEL_VIRTUAL_GRF_UNAVAILABLE = "Estimated Virtual GRF unavailable"
LABEL_VIRTUAL_GRF_STATUS_PREFIX = "Estimated Virtual GRF:"
LABEL_VIRTUAL_GRF_UNAVAILABLE_MSG = (
    "Estimated Virtual GRF unavailable (need pose sequence)"
)
LABEL_FOOT_CONTACT = "Foot Contact (estimated timing)"
LABEL_FOOT_CONTACT_TIMELINE = "Foot Contact Timeline (estimated)"
LABEL_CONTACT_PATTERN = "Contact Pattern"
LABEL_CONTACT_STATE = "Contact State"
LABEL_CONTACT_NOT_FORCE = "Foot contact = timing only; not measured kinetics."

# ── Center of mass / support ──────────────────────────────────────────────────
LABEL_COM_FULL = "Center of Mass (estimated)"
LABEL_COM_SHORT = "COM (est.)"
LABEL_BOS = "Base of Support (estimated)"
LABEL_STABILITY_MARGIN = "Stability Margin (derived)"
LABEL_STABILITY_STATE = "Stability State (derived)"

# ── Composite scores ──────────────────────────────────────────────────────────
LABEL_MOVEMENT_STABILITY = "Movement Stability (derived score)"
LABEL_GAIT_QUALITY = "Gait Quality (derived)"
LABEL_GAIT_COORDINATION = "Gait Coordination (derived)"
LABEL_COMPOSITE_GAIT_QUALITY = "Composite Gait Quality (derived)"
LABEL_CADENCE = "Cadence (calculated)"
LABEL_CADENCE_SHORT = "Cadence (calc.)"
UNIT_CADENCE = "steps/min"
LABEL_WALKING_SPEED = "Walking Speed (estimated)"
UNIT_WALKING_SPEED = "m/s"
LABEL_JOINT_ROM = "Joint ROM (estimated)"
LABEL_JOINT_ROM_SUMMARY = "Joint ROM Summary (estimated)"
LABEL_LEFT_ROM = "Left ROM"
LABEL_RIGHT_ROM = "Right ROM"
LABEL_ROM_ASYMMETRY = "ROM asymmetry"
LABEL_PATH_SPAN = "Path span"
LABEL_GAIT_SYMMETRY = "Gait Symmetry (derived)"
LABEL_SYMMETRY = "Symmetry (derived)"
LABEL_VIDEO_QUALITY = "Video Quality (derived heuristics)"

# ── Tab / section headers ─────────────────────────────────────────────────────
LABEL_BIOMECH_TAB_HEADER = "Estimated Biomechanical Parameters (from pose)"
LABEL_BIOMECH_TIMELINE = "Biomechanics Timeline"
LABEL_CONTACT_VGRF_SECTION = "Foot Contact & Estimated Virtual GRF"

# ── Chart titles ──────────────────────────────────────────────────────────────
CHART_COM_HEIGHT = "Center of Mass — height (body-normalized, estimated)"
CHART_STABILITY_MARGIN = "Stability Margin (derived)"
CHART_CONTACT_EVENTS = "Contact Timing & Gait Events (derived)"
CHART_GAIT_METRICS = "Gait Metrics Summary (derived)"
CHART_GAIT_PHASE = "Gait Phase Timeline (derived)"
CHART_VGRF = LABEL_VIRTUAL_GRF_FULL

# ── Disclaimers ───────────────────────────────────────────────────────────────
DISCLAIMER_VGRF = "Not force-plate or PhysX — pose-based estimate."
DISCLAIMER_MONOCULAR = "Monocular video; absolute meters are approximate."
DISCLAIMER_NOT_MEASURED = "Not available as measured data from video alone."

# ── Overlay toggles (3D skeleton) ───────────────────────────────────────────
OVERLAY_COM = "COM (est.)"
OVERLAY_BOS = "BoS (est.)"
OVERLAY_COM_VEL = "COM Vel"


TIER_DISPLAY: dict[str, str] = {
    TIER_MEASURED: "measured",
    TIER_ESTIMATED: "estimated",
    TIER_DERIVED: "derived",
    TIER_CALCULATED: "calculated",
}


def format_tier_badge(tier: str) -> str:
    """Short tier label for metric cards and sidebars."""
    return TIER_DISPLAY.get(tier, tier)


def format_tier_suffix(tier: str) -> str:
    """Parenthetical tier suffix for inline metric readouts."""
    label = format_tier_badge(tier)
    return f" ({label})" if label else ""


def format_metric_title(base: str, tier: str | None = None) -> str:
    """Metric card title with optional tier suffix, e.g. Walking Speed (estimated)."""
    if tier:
        return f"{base}{format_tier_suffix(tier)}"
    return base


def format_confidence_label(scope: str, level: str) -> str:
    """Scoped confidence readout, e.g. Contact Confidence: High."""
    return f"{scope}: {level}"


def format_walking_speed_value(speed_m_s: float) -> str:
    """Canonical walking-speed value string."""
    return f"{speed_m_s:.2f} {UNIT_WALKING_SPEED} (estimated)"


def format_walking_speed_value_numeric(speed_m_s: float) -> str:
    """Walking-speed value when the metric title already includes the tier."""
    return f"{speed_m_s:.2f} {UNIT_WALKING_SPEED}"


def format_cadence_value(cadence_steps_per_min: float) -> str:
    """Canonical cadence value string."""
    return f"{cadence_steps_per_min:.0f} {UNIT_CADENCE}"


def export_terminology_block() -> dict[str, str]:
    """Standard terminology block for JSON/NPZ sidecars."""
    return dict(TIER_DEFINITIONS)


__all__ = [
    "CHART_COM_HEIGHT",
    "CHART_CONTACT_EVENTS",
    "CHART_GAIT_METRICS",
    "CHART_GAIT_PHASE",
    "CHART_STABILITY_MARGIN",
    "CHART_VGRF",
    "DISCLAIMER_MONOCULAR",
    "DISCLAIMER_NOT_MEASURED",
    "DISCLAIMER_VGRF",
    "LABEL_ANALYSIS_CONFIDENCE",
    "LABEL_BIOMECH_TAB_HEADER",
    "LABEL_BIOMECH_TIMELINE",
    "LABEL_BOS",
    "LABEL_CADENCE",
    "LABEL_CADENCE_SHORT",
    "LABEL_COM_FULL",
    "LABEL_COM_SHORT",
    "LABEL_CONTACT_CONFIDENCE",
    "LABEL_CONTACT_NOT_FORCE",
    "LABEL_CONTACT_PATTERN",
    "LABEL_CONTACT_STATE",
    "LABEL_CONTACT_VGRF_SECTION",
    "LABEL_DOMAIN_CONFIDENCE",
    "LABEL_FOOT_CLEARANCE_CONFIDENCE",
    "LABEL_FOOT_CONTACT",
    "LABEL_FOOT_CONTACT_TIMELINE",
    "LABEL_GAIT_COORDINATION",
    "LABEL_GAIT_QUALITY",
    "LABEL_GAIT_SYMMETRY",
    "LABEL_COMPOSITE_GAIT_QUALITY",
    "LABEL_JOINT_ROM",
    "LABEL_JOINT_ROM_SUMMARY",
    "LABEL_LEFT_ROM",
    "LABEL_MOVEMENT_STABILITY",
    "LABEL_PATH_SPAN",
    "LABEL_PIPELINE_CONFIDENCE",
    "LABEL_RIGHT_ROM",
    "LABEL_ROM_ASYMMETRY",
    "LABEL_STABILITY_MARGIN",
    "LABEL_STABILITY_STATE",
    "LABEL_SYMMETRY",
    "LABEL_TRACKING_CONFIDENCE",
    "LABEL_TRAJECTORY_CONFIDENCE",
    "LABEL_VIDEO_QUALITY",
    "LABEL_VIEW_CONFIDENCE",
    "LABEL_VIRTUAL_GRF_FULL",
    "LABEL_VIRTUAL_GRF_PANEL",
    "LABEL_VIRTUAL_GRF_SHORT",
    "LABEL_VIRTUAL_GRF_STATUS_PREFIX",
    "LABEL_VIRTUAL_GRF_UNAVAILABLE",
    "LABEL_VIRTUAL_GRF_UNAVAILABLE_MSG",
    "LABEL_WALKING_SPEED",
    "OVERLAY_BOS",
    "OVERLAY_COM",
    "OVERLAY_COM_VEL",
    "TIER_CALCULATED",
    "TIER_DEFINITIONS",
    "TIER_DERIVED",
    "TIER_DISPLAY",
    "TIER_ESTIMATED",
    "TIER_MEASURED",
    "UNIT_CADENCE",
    "UNIT_WALKING_SPEED",
    "export_terminology_block",
    "format_cadence_value",
    "format_confidence_label",
    "format_metric_title",
    "format_tier_badge",
    "format_tier_suffix",
    "format_walking_speed_value",
    "format_walking_speed_value_numeric",
]
