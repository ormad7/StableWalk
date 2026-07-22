"""
Professional scientific tooltips for StableWalk GUI metrics.

Each entry covers: meaning, calculation, units, normal range, clinical significance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricTooltip:
    """Structured tooltip copy for a gait / biomechanics metric."""

    title: str
    meaning: str
    calculation: str
    units: str
    normal: str
    clinical: str

    def format(self) -> str:
        return (
            f"{self.title}\n\n"
            f"Meaning: {self.meaning}\n\n"
            f"Calculation: {self.calculation}\n\n"
            f"Units: {self.units}\n\n"
            f"Normal: {self.normal}\n\n"
            f"Clinical: {self.clinical}"
        )


def format_metric_tooltip(
    title: str,
    *,
    meaning: str,
    calculation: str,
    units: str,
    normal: str,
    clinical: str,
) -> str:
    return MetricTooltip(
        title=title,
        meaning=meaning,
        calculation=calculation,
        units=units,
        normal=normal,
        clinical=clinical,
    ).format()


# ── Canonical metric tooltips ───────────────────────────────────────────────

_COM = MetricTooltip(
    title="Center of Mass (COM)",
    meaning=(
        "Estimated whole-body mass centroid representing the resultant "
        "location of segmental masses during gait."
    ),
    calculation=(
        "Weighted average of pose landmarks using anthropometric segment "
        "mass fractions (monocular / root-relative estimate)."
    ),
    units="meters (body-scaled) or normalized height",
    normal=(
        "Near midline; vertical oscillation typically ~2–5 cm at preferred "
        "walking speed in healthy adults."
    ),
    clinical=(
        "Excess lateral excursion, sudden drops, or asymmetric COM paths "
        "suggest instability, compensatory strategies, or unilateral deficit."
    ),
)

_BOS = MetricTooltip(
    title="Base of Support (BoS)",
    meaning=(
        "Support polygon formed by contacting foot (or feet) regions on the "
        "estimated ground plane."
    ),
    calculation=(
        "Convex hull / footprint polygon from heel–toe contact landmarks "
        "while stance is detected."
    ),
    units="meters² (area) or planar coordinates (m)",
    normal=(
        "BoS expands in double support and narrows in single support; COM "
        "projection should remain inside or near the BoS during stable gait."
    ),
    clinical=(
        "Narrow or asymmetric BoS and COM outside BoS increase fall risk "
        "and indicate reduced dynamic stability."
    ),
)

_GRF = MetricTooltip(
    title="Estimated Virtual Ground Reaction Force (vGRF)",
    meaning=(
        "Pose-based proxy for the vertical ground reaction force acting on "
        "the stance limb(s)."
    ),
    calculation=(
        "Contact-synchronized estimate from COM vertical acceleration and "
        "body-mass scaling (not a force-plate measurement)."
    ),
    units="newtons (N) or body-weight multiples (BW)",
    normal=(
        "Healthy walking often shows a double-hump vertical GRF pattern of "
        "roughly 1.0–1.2 BW peaks at preferred speed."
    ),
    clinical=(
        "Reduced peaks, loss of double-hump shape, or side asymmetry may "
        "reflect pain avoidance, weakness, or abnormal loading."
    ),
)

_ROM = MetricTooltip(
    title="Range of Motion (ROM)",
    meaning=(
        "Angular excursion of a joint over the stride or analysis window "
        "(peak flexion minus peak extension)."
    ),
    calculation=(
        "Peak-to-peak joint angle from pose-derived segment vectors "
        "(estimated kinematics)."
    ),
    units="degrees (°)",
    normal=(
        "Typical adult walking: knee flexion ROM ~60–70°; hip flexion–"
        "extension ROM ~40–50°; ankle ~20–30° (speed-dependent)."
    ),
    clinical=(
        "Reduced ROM suggests stiffness, pain, or compensation; excessive "
        "or asymmetric ROM may indicate instability or motor control deficit."
    ),
)

_CADENCE = MetricTooltip(
    title="Cadence",
    meaning="Step rate — number of footfalls per minute.",
    calculation=(
        "Steps counted from heel-strike (or contact) events divided by "
        "elapsed gait time, scaled to steps/min."
    ),
    units="steps/min",
    normal=(
        "Preferred adult walking cadence is commonly ~100–120 steps/min "
        "(higher with faster speeds)."
    ),
    clinical=(
        "Low cadence with short steps may indicate cautious gait; "
        "unusually high cadence can compensate for reduced step length."
    ),
)

_SYMMETRY = MetricTooltip(
    title="Gait Symmetry",
    meaning=(
        "Left–right similarity of temporal and/or spatial gait features "
        "(stance, swing, ROM, or kinematics)."
    ),
    calculation=(
        "Derived index comparing bilateral metrics (e.g. stance-time or "
        "ROM ratios), often expressed as percent symmetry or asymmetry."
    ),
    units="percent (%) or unitless symmetry index",
    normal=(
        "Healthy gait is near-symmetric; mild asymmetry (<5–10%) is common; "
        "larger differences warrant clinical attention."
    ),
    clinical=(
        "Marked asymmetry is associated with unilateral pathology, "
        "compensatory loading, and elevated injury or fall risk."
    ),
)

_STABILITY = MetricTooltip(
    title="Movement Stability",
    meaning=(
        "Composite estimate of postural control during walking from pelvis, "
        "trunk, and COM/BoS kinematics."
    ),
    calculation=(
        "Derived score from COM–BoS relationship, pelvic/trunk motion "
        "smoothness, and related root-relative features."
    ),
    units="score (0–100) or margin in meters",
    normal=(
        "Higher scores / positive stability margins indicate COM remains "
        "well controlled relative to the support base."
    ),
    clinical=(
        "Low stability scores or negative margins suggest dynamic "
        "instability and may correlate with fall risk or ataxia."
    ),
)

_HEEL = MetricTooltip(
    title="Heel Strike (Initial Contact)",
    meaning=(
        "Gait event when the heel (or leading foot) first contacts the "
        "ground, starting stance."
    ),
    calculation=(
        "Detected from foot kinematics / contact heuristics (height, "
        "velocity, and phase constraints) — estimated timing, not pressure."
    ),
    units="time (s) or gait-cycle percent (%GC)",
    normal=(
        "Defines 0% of the gait cycle; occurs once per limb per stride in "
        "typical heel-toe walking."
    ),
    clinical=(
        "Absent heel contact, toe-first landing, or irregular timing may "
        "indicate equinus, drop-foot, or pain-avoidance patterns."
    ),
)

_TOE = MetricTooltip(
    title="Toe Off (Pre-swing / Terminal Stance end)",
    meaning=(
        "Gait event when the stance foot leaves the ground, beginning swing."
    ),
    calculation=(
        "Estimated from loss of foot–ground contact using landmark height "
        "and velocity thresholds synchronized to the gait cycle."
    ),
    units="time (s) or gait-cycle percent (%GC)",
    normal=(
        "Typically near ~60% of the gait cycle in healthy walking "
        "(stance ≈ 60%, swing ≈ 40%)."
    ),
    clinical=(
        "Early toe-off shortens stance; delayed toe-off prolongs double "
        "support — both are common in weak, painful, or cautious gait."
    ),
)

_CLEARANCE = MetricTooltip(
    title="Foot Clearance",
    meaning=(
        "Vertical distance between the swing foot (toe/heel/ankle landmark) "
        "and the estimated floor plane."
    ),
    calculation=(
        "Minimum vertical clearance during mid-swing relative to a "
        "body-scaled ground reference from pose."
    ),
    units="centimeters (cm)",
    normal=(
        "Healthy mid-swing toe clearance is often on the order of ~1–2 cm "
        "(highly sensitive to measurement method)."
    ),
    clinical=(
        "Insufficient clearance increases trip risk; excessive clearance "
        "may reflect compensatory steppage or circumduction."
    ),
)

_JOINT_ANGLE = MetricTooltip(
    title="Joint Angle",
    meaning=(
        "Instantaneous orientation between adjacent body segments at a "
        "selected joint (e.g. knee flexion)."
    ),
    calculation=(
        "Angle between segment vectors from 2D/3D pose landmarks; "
        "convention depends on joint (flexion positive for knee)."
    ),
    units="degrees (°)",
    normal=(
        "Healthy knee flexion peaks near ~60–70° in swing; stance knee "
        "remains near extension with a small loading-response flex."
    ),
    clinical=(
        "Limited peak flexion, extension deficit, or side asymmetry can "
        "indicate pathology, pain, or surgical constraint."
    ),
)

_PIPELINE = MetricTooltip(
    title="Pipeline Status",
    meaning=(
        "End-to-end Real-to-Sim processing checklist: pose, biomechanics, "
        "forces, OpenSim, and export readiness."
    ),
    calculation=(
        "Derived from completed artifacts and stage validators (poses "
        "present, gait metrics computed, OpenSim export/IK state, etc.)."
    ),
    units="stage status (completed / partial / unavailable)",
    normal=(
        "After a successful Analyze run, core pose and biomechanics stages "
        "should show Completed; OpenSim/IK depend on SDK and model setup."
    ),
    clinical=(
        "Incomplete stages reduce confidence in downstream clinical "
        "metrics — verify pose quality before interpreting kinetics or IK."
    ),
)

METRIC_TOOLTIPS: dict[str, MetricTooltip] = {
    "com": _COM,
    "bos": _BOS,
    "grf": _GRF,
    "vgrf": _GRF,
    "rom": _ROM,
    "joint_rom": _ROM,
    "cadence": _CADENCE,
    "symmetry": _SYMMETRY,
    "stability": _STABILITY,
    "stability_margin": _STABILITY,
    "stability_state": _STABILITY,
    "movement_stability": _STABILITY,
    "heel_strike": _HEEL,
    "toe_off": _TOE,
    "foot_clearance": _CLEARANCE,
    "joint_angle": _JOINT_ANGLE,
    "pipeline_status": _PIPELINE,
    "gait_quality": MetricTooltip(
        title="Gait Quality",
        meaning=(
            "Composite score of timing regularity, symmetry, contact quality, "
            "and cycle repeatability."
        ),
        calculation=(
            "Derived aggregation of available gait-cycle and symmetry features "
            "when sufficient strides are detected."
        ),
        units="score (0–100)",
        normal="Higher scores indicate steadier, more repeatable walking patterns.",
        clinical=(
            "Low scores flag irregular or poorly tracked gait and should be "
            "interpreted with analysis confidence."
        ),
    ),
    "walking_speed": MetricTooltip(
        title="Walking Speed",
        meaning="Estimated forward progression speed of the walker.",
        calculation=(
            "Derived from root/COM displacement over time with body-scale "
            "calibration (monocular estimate)."
        ),
        units="meters per second (m/s)",
        normal=(
            "Comfortable adult walking speed is typically ~1.2–1.4 m/s "
            "(varies with age and setting)."
        ),
        clinical=(
            "Slowed gait speed is a robust marker of frailty, pain, and "
            "neuromuscular impairment."
        ),
    ),
    "knee_motion": MetricTooltip(
        title="Knee Motion",
        meaning="Knee flexion–extension angle time series during walking.",
        calculation=(
            "Angle between thigh and shank segments from pose landmarks; "
            "left/right traces compared over time or gait cycles."
        ),
        units="degrees (°)",
        normal=(
            "Stance: near extension with small flexion wave; swing: peak "
            "flexion ~60–70° in healthy adults."
        ),
        clinical=(
            "Reduced peak flexion or extension lag may indicate stiffness, "
            "quadriceps weakness, or post-surgical constraint."
        ),
    ),
    "gait_cycle": MetricTooltip(
        title="Gait Cycle",
        meaning=(
            "One stride from heel strike to the next ipsilateral heel strike, "
            "partitioned into stance and swing phases."
        ),
        calculation=(
            "Phase segmentation from estimated contact events and temporal "
            "normalization to percent gait cycle."
        ),
        units="percent gait cycle (%GC) or seconds (s)",
        normal="Stance ≈ 60% and swing ≈ 40% at typical walking speeds.",
        clinical=(
            "Altered phase timing (e.g. prolonged double support) is common in "
            "cautious, painful, or balance-impaired gait."
        ),
    ),
    "analysis_confidence": MetricTooltip(
        title="Analysis Confidence",
        meaning=(
            "Overall trustworthiness of the derived metrics given tracking "
            "quality and gait evidence."
        ),
        calculation=(
            "Heuristic from pose detection rate, foot visibility, cycle count, "
            "and related quality checks."
        ),
        units="categorical (High / Moderate / Low) or score",
        normal="High confidence requires sufficient detected strides and stable tracking.",
        clinical=(
            "Low confidence means values are exploratory — avoid strong "
            "clinical conclusions until tracking improves."
        ),
    ),
    "joint_movement_3d": MetricTooltip(
        title="3D Joint Movement Path",
        meaning="Spatial trajectory of a selected joint relative to the pelvis/root.",
        calculation=(
            "Time-ordered joint positions in root-relative coordinates; "
            "travel, smoothness, and deviation summarize path quality."
        ),
        units="meters (body-scaled) or centimeters",
        normal="Repeatable, smooth loops or arcs across strides for the same joint.",
        clinical=(
            "Erratic or asymmetric paths suggest poor motor control, "
            "compensation, or tracking noise."
        ),
    ),
}


def get_metric_tooltip(key: str) -> str | None:
    """Return formatted tooltip text for *key*, or None if unknown."""
    tip = METRIC_TOOLTIPS.get(key)
    if tip is None:
        return None
    return tip.format()


def combine_metric_tooltip(base: str | None, detail: str | None) -> str:
    """Merge canonical science tip with live value/reason detail."""
    base = (base or "").strip()
    detail = (detail or "").strip()
    if base and detail and detail not in base:
        return f"{base}\n\nCurrent: {detail}"
    return base or detail or "—"


def metric_help_body(key: str) -> str:
    """Body text for METRIC_HELP / ? icons (same content as hover tooltips)."""
    tip = METRIC_TOOLTIPS.get(key)
    if tip is None:
        return ""
    return (
        f"Meaning: {tip.meaning}\n\n"
        f"Calculation: {tip.calculation}\n\n"
        f"Units: {tip.units}\n\n"
        f"Normal: {tip.normal}\n\n"
        f"Clinical: {tip.clinical}"
    )


# Widget attribute → tooltip key for one-shot GUI wiring
METRIC_WIDGET_TOOLTIPS: tuple[tuple[str, str], ...] = (
    ("lbl_biomech_symmetry", "symmetry"),
    ("lbl_biomech_stability_margin", "stability"),
    ("lbl_biomech_stability_state", "stability"),
    ("lbl_biomech_cadence", "cadence"),
    ("lbl_biomech_rom", "rom"),
    ("lbl_biomech_gait_quality", "gait_quality"),
    ("lbl_biomech_walking_speed", "walking_speed"),
    ("lbl_physics_force_status", "grf"),
    ("lbl_physics_force_method", "grf"),
    ("lbl_physics_force_note", "grf"),
    ("lbl_summary_ms_value", "stability"),
    ("lbl_movement_stability", "stability"),
    ("lbl_stab_score", "stability"),
    ("lbl_stab_category", "stability"),
    ("lbl_stab_headline", "stability"),
    ("lbl_ground_clearance_left", "foot_clearance"),
    ("lbl_ground_clearance_right", "foot_clearance"),
    ("lbl_foot_clearance_confidence", "foot_clearance"),
    ("lbl_foot_left_current", "foot_clearance"),
    ("lbl_foot_right_current", "foot_clearance"),
    ("lbl_knee_summary_compact", "joint_angle"),
    ("lbl_joint_motion_hint", "joint_angle"),
    ("lbl_joint_movement_title", "joint_movement_3d"),
    ("lbl_advanced_temporal", "symmetry"),
    ("_section_pipeline_status", "pipeline_status"),
    ("_section_pipeline_status_advanced", "pipeline_status"),
    ("_section_pipeline_status_summary", "pipeline_status"),
)


def attach_metric_tooltip(widget: Any, key: str, *, title: str | None = None) -> None:
    """Attach the professional tooltip for *key* to *widget*."""
    del title  # title is embedded in the canonical tip text
    from stablewalk.ui.theme import create_tooltip

    tip = get_metric_tooltip(key)
    if not tip:
        return
    create_tooltip(widget, tip, wraplength=340)


def attach_professional_metric_tooltips(gui: Any) -> None:
    """Wire scientific tooltips onto known metric widgets (idempotent)."""
    from stablewalk.ui.theme import create_tooltip

    for attr, key in METRIC_WIDGET_TOOLTIPS:
        widget = getattr(gui, attr, None)
        if widget is None:
            continue
        tip = get_metric_tooltip(key)
        if tip:
            create_tooltip(widget, tip, wraplength=340)

    fields = getattr(gui, "_summary_field_labels", None) or {}
    for key, lbl in fields.items():
        tip = get_metric_tooltip(key)
        if tip and lbl is not None:
            create_tooltip(lbl, tip, wraplength=340)

    events = getattr(gui, "_summary_event_labels", None) or {}
    for key, pair in events.items():
        tip = get_metric_tooltip(key)
        if not tip:
            continue
        status_lbl, detail_lbl = pair
        create_tooltip(status_lbl, tip, wraplength=340)
        create_tooltip(detail_lbl, tip, wraplength=340)

    for attr, key in (
        ("lbl_biomech_symmetry", "symmetry"),
        ("lbl_biomech_cadence", "cadence"),
        ("lbl_biomech_stability_margin", "stability"),
        ("lbl_biomech_rom", "rom"),
    ):
        lbl = getattr(gui, attr, None)
        if lbl is None:
            continue
        tip = get_metric_tooltip(key)
        if not tip:
            continue
        try:
            parent = lbl.master
            for child in parent.winfo_children():
                create_tooltip(child, tip, wraplength=340)
        except Exception:
            pass


__all__ = [
    "METRIC_TOOLTIPS",
    "METRIC_WIDGET_TOOLTIPS",
    "MetricTooltip",
    "attach_metric_tooltip",
    "attach_professional_metric_tooltips",
    "combine_metric_tooltip",
    "format_metric_tooltip",
    "get_metric_tooltip",
    "metric_help_body",
]
