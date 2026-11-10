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
LABEL_POSE_COVERAGE = "Pose coverage"
LABEL_CONTACT_COVERAGE = "Contact coverage"
LABEL_CONTACT_COVERAGE_LEFT = "Left contact coverage"
LABEL_CONTACT_COVERAGE_RIGHT = "Right contact coverage"
LABEL_CYCLE_EVIDENCE = "Cycle evidence"
LABEL_SPECTRAL_RELIABILITY = "Spectral reliability"
LABEL_BIOMECHANICAL_VALIDITY = "Biomechanical validity"
LABEL_TRACKING_CONFIDENCE = "Tracking Confidence"
LABEL_PIPELINE_CONFIDENCE = "Pipeline Confidence"
LABEL_CONTACT_CONFIDENCE = "Contact Confidence"
LABEL_FOOT_CLEARANCE_CONFIDENCE = "Foot Clearance Confidence"
LABEL_FOOT_CLEARANCE = "Estimated Foot Clearance"
LABEL_FOOT_CLEARANCE_CURRENT = "Current foot clearance"
LABEL_FOOT_CLEARANCE_MIN_SWING = "Minimum swing clearance"
LABEL_FOOT_CLEARANCE_MAX_SWING = "Maximum swing clearance"
LABEL_FOOT_CLEARANCE_SUBTITLE = (
    "Monocular pose estimate · stance-foot ground plane · cm from body-scale calibration"
)
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
LABEL_GAIT_QUALITY = "Gait Quality (heuristic descriptor)"
LABEL_GAIT_COORDINATION = "Gait Coordination (derived)"
LABEL_COMPOSITE_GAIT_QUALITY = "Composite Gait Quality (derived)"
LABEL_CADENCE = "Cadence (calculated)"
LABEL_CADENCE_SHORT = "Cadence (calc.)"
UNIT_CADENCE = "steps/min"
LABEL_WALKING_SPEED = "Walking Speed (estimated)"
UNIT_WALKING_SPEED = "m/s"
LABEL_JOINT_ROM = "Full-session robust ROM (estimated)"
LABEL_JOINT_ROM_SUMMARY = "Full-session robust ROM summary (estimated)"
LABEL_LEFT_ROM = "Left-cycle ROM"
LABEL_RIGHT_ROM = "Right-cycle ROM"
LABEL_ROM_ASYMMETRY = "ROM asymmetry"
LABEL_PATH_SPAN = "Path span"
LABEL_SPECTRAL_ANALYSIS = "Spectral Analysis (derived)"
LABEL_AUTOCORRELATION = "Normalized autocorrelation (derived)"
LABEL_WELCH_PSD = "Welch power spectral density (derived)"
DISCLAIMER_SPECTRAL = (
    "Derived from monocular pose time series. Reliability is a data-quality "
    "score — not a clinical instability diagnosis."
)
LABEL_GAIT_SYMMETRY = "Gait Symmetry (derived)"
LABEL_SYMMETRY = "Symmetry (derived)"
LABEL_VIDEO_QUALITY = "Pose / video heuristics (not tracking accuracy)"

# ── Tab / section headers ─────────────────────────────────────────────────────
LABEL_BIOMECH_TAB_HEADER = "Estimated Biomechanical Parameters (from pose)"
LABEL_BIOMECH_TIMELINE = "Biomechanics Timeline"
LABEL_CONTACT_VGRF_SECTION = "Foot Contact & Estimated Virtual GRF"

# ── Chart titles ──────────────────────────────────────────────────────────────
CHART_COM_HEIGHT = "COM Vertical Position (est., pelvis-relative, approx. BH)"
CHART_STABILITY_MARGIN = "COM–BoS signed distance (unvalidated preview, m)"
CHART_CONTACT_EVENTS = "Contact Timing (estimated)"
CHART_GAIT_METRICS = "Gait Phase (derived)"
CHART_GAIT_PHASE = "Gait Phase Timeline (derived)"
CHART_VGRF = LABEL_VIRTUAL_GRF_FULL

# ── Disclaimers ───────────────────────────────────────────────────────────────
DISCLAIMER_VGRF = (
    "Not force-plate or PhysX — pose-based estimate. "
    "Unknown/unreliable contact is withheld, not plotted as zero."
)
DISCLAIMER_MONOCULAR = "Monocular video; absolute meters are approximate."
DISCLAIMER_NOT_MEASURED = "Not available as measured data from video alone."
DISCLAIMER_OVERVIEW_VISIBLE = (
    "Research and educational output based on monocular pose estimation. "
    "Not a clinical diagnosis or medical device measurement."
)
DISCLAIMER_RESEARCH = (
    "Pose-estimated biomechanics for research and education — "
    "not a clinical diagnosis or medical device output."
)
DISCLAIMER_COMPARE = (
    "Δ = Session B − Session A (pose-estimated). "
    "Color shows relative direction; “Notable” marks differences above "
    "display thresholds — not formal statistical significance (p-values)."
)

# Shared units / coordinate legend for Overview path, Motion, and Biomechanics.
BH_DEFINITION = (
    "BH = body height. COM charts plot pelvis-relative, hip-centered vertical "
    "position (approx. BH). The skeleton is body-normalized, but the measured "
    "head–ankle span is often not 1.0, so one chart unit is not one standing "
    "height and is not floor-absolute COM (~0.55 BH). Physical metres use "
    "assumed stature (default 1.70 m) divided by that span and are not "
    "tape-calibrated."
)
UNITS_COORD_LEGEND = (
    "Coords: Root-relative (pelvis origin), not world · Path: cm (stature-scaled estimate, not mocap) · "
    "Angles: degrees (est. DOF/flexion as stored) · ω: °/s (derived) · "
    "COM: body heights (BH; pelvis-relative hip-centered Y, not floor-absolute height) · "
    "Margin: m (derived) · Speed: m/s (est.) · "
    f"{DISCLAIMER_MONOCULAR}"
)

# Help → Analysis methods (thesis defense narrative).
METHODS_OVERVIEW = (
    "StableWalk recovers 2D/3D joint landmarks from monocular video "
    "(MediaPipe Pose by default; optional SMPL backend), then derives "
    "gait timing, COM/BoS proxies, stability margins, and contact events "
    "from those landmarks. Estimated Virtual GRF is a pose-based "
    "heuristic — not instrumented kinetics. OpenSim export (.trc / .mot) "
    "and optional IK are available when the OpenSim SDK is installed; "
    "dashboard scores do not require OpenSim IK.\n\n"
    f"{BH_DEFINITION}"
)

# ── Overlay toggles (3D skeleton) ───────────────────────────────────────────
OVERLAY_COM = "COM (est.)"
OVERLAY_BOS = "BoS (est.)"
OVERLAY_COM_VEL = "COM Vel"
OVERLAY_GROUND = "Estimated ground"


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
    "DISCLAIMER_COMPARE",
    "DISCLAIMER_MONOCULAR",
    "DISCLAIMER_NOT_MEASURED",
    "DISCLAIMER_OVERVIEW_VISIBLE",
    "DISCLAIMER_RESEARCH",
    "DISCLAIMER_VGRF",
    "BH_DEFINITION",
    "METHODS_OVERVIEW",
    "UNITS_COORD_LEGEND",
    "LABEL_ANALYSIS_CONFIDENCE",
    "LABEL_BIOMECHANICAL_VALIDITY",
    "LABEL_CONTACT_COVERAGE",
    "LABEL_CONTACT_COVERAGE_LEFT",
    "LABEL_CONTACT_COVERAGE_RIGHT",
    "LABEL_CYCLE_EVIDENCE",
    "LABEL_POSE_COVERAGE",
    "LABEL_SPECTRAL_RELIABILITY",
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
    "LABEL_FOOT_CLEARANCE",
    "LABEL_FOOT_CLEARANCE_CONFIDENCE",
    "LABEL_FOOT_CLEARANCE_CURRENT",
    "LABEL_FOOT_CLEARANCE_MAX_SWING",
    "LABEL_FOOT_CLEARANCE_MIN_SWING",
    "LABEL_FOOT_CLEARANCE_SUBTITLE",
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
    "OVERLAY_GROUND",
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
