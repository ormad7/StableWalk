"""
Classification of all stability v2 features by measurement intent.

MAGNITUDE — absolute level (must NOT directly reduce stability score).
ASYMMETRY — left-right or phase differences.
VARIABILITY — frame-to-frame or step-to-step inconsistency.
SMOOTHNESS — jerk / acceleration irregularity.
REPEATABILITY — cycle-to-cycle trajectory consistency.
CONTACT_TIMING — stance/swing/heel-strike sequencing.
CONTROL — composite control-quality constructs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

FeatureClass = Literal[
    "MAGNITUDE",
    "ASYMMETRY",
    "VARIABILITY",
    "SMOOTHNESS",
    "REPEATABILITY",
    "CONTACT_TIMING",
    "CONTROL",
]


class FeatureIntent(str, Enum):
    MAGNITUDE = "MAGNITUDE"
    ASYMMETRY = "ASYMMETRY"
    VARIABILITY = "VARIABILITY"
    SMOOTHNESS = "SMOOTHNESS"
    REPEATABILITY = "REPEATABILITY"
    CONTACT_TIMING = "CONTACT_TIMING"
    CONTROL = "CONTROL"


@dataclass(frozen=True)
class FeatureSpec:
    domain: str
    feature: str
    intent: FeatureIntent
    penalizes_magnitude: bool = False
    notes: str = ""


STABILITY_FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    # Temporal symmetry
    FeatureSpec("temporal_symmetry", "stance_duration_symmetry", FeatureIntent.ASYMMETRY, notes="L/R stance ratio"),
    FeatureSpec("temporal_symmetry", "swing_duration_symmetry", FeatureIntent.ASYMMETRY),
    FeatureSpec("temporal_symmetry", "step_time_asymmetry", FeatureIntent.ASYMMETRY),
    FeatureSpec("temporal_symmetry", "raw_temporal_asymmetry", FeatureIntent.ASYMMETRY, notes="Combined raw asymmetry"),
    FeatureSpec("temporal_symmetry", "normalized_temporal_penalty", FeatureIntent.VARIABILITY),
    FeatureSpec("temporal_symmetry", "final_temporal_score", FeatureIntent.CONTROL),
    # Spatial symmetry
    FeatureSpec("spatial_symmetry", "norm_step_length_asymmetry", FeatureIntent.ASYMMETRY, notes="Not step length magnitude"),
    FeatureSpec("spatial_symmetry", "norm_stride_length_asymmetry", FeatureIntent.ASYMMETRY),
    FeatureSpec("spatial_symmetry", "foot_progression_asymmetry", FeatureIntent.ASYMMETRY),
    FeatureSpec("spatial_symmetry", "norm_step_length_left_mean", FeatureIntent.MAGNITUDE, penalizes_magnitude=False),
    FeatureSpec("spatial_symmetry", "norm_step_length_right_mean", FeatureIntent.MAGNITUDE, penalizes_magnitude=False),
    # Pelvis stability (progression-relative)
    FeatureSpec("pelvis_stability", "mediolateral_sway_ratio", FeatureIntent.VARIABILITY, notes="Gait-frame ML sway"),
    FeatureSpec("pelvis_stability", "vertical_oscillation_normalized", FeatureIntent.VARIABILITY),
    FeatureSpec("pelvis_stability", "mediolateral_acceleration_cv", FeatureIntent.SMOOTHNESS),
    FeatureSpec("pelvis_stability", "lateral_jerk_normalized", FeatureIntent.SMOOTHNESS),
    FeatureSpec("pelvis_stability", "pelvis_forward_displacement", FeatureIntent.MAGNITUDE, penalizes_magnitude=False),
    FeatureSpec("pelvis_stability", "global_pelvis_displacement_m", FeatureIntent.MAGNITUDE, penalizes_magnitude=False),
    # Trunk stability
    FeatureSpec("trunk_stability", "root_relative_ml_sway_ratio", FeatureIntent.VARIABILITY),
    FeatureSpec("trunk_stability", "trunk_lean_variation_deg", FeatureIntent.VARIABILITY),
    FeatureSpec("trunk_stability", "upper_body_oscillation", FeatureIntent.VARIABILITY),
    # Foot clearance
    FeatureSpec("foot_clearance", "max_swing_clearance_asymmetry", FeatureIntent.ASYMMETRY),
    FeatureSpec("foot_clearance", "clearance_cycle_consistency", FeatureIntent.REPEATABILITY),
    FeatureSpec("foot_clearance", "clearance_outlier_fraction", FeatureIntent.VARIABILITY),
    FeatureSpec("foot_clearance", "toe_drag_frames", FeatureIntent.CONTACT_TIMING),
    FeatureSpec("foot_clearance", "max_clearance_left", FeatureIntent.MAGNITUDE, penalizes_magnitude=False),
    FeatureSpec("foot_clearance", "max_clearance_right", FeatureIntent.MAGNITUDE, penalizes_magnitude=False),
    # Joint motion / controlled motion
    FeatureSpec("joint_smoothness", "controlled_motion_knee", FeatureIntent.CONTROL),
    FeatureSpec("joint_smoothness", "controlled_motion_hip", FeatureIntent.CONTROL),
    FeatureSpec("joint_smoothness", "controlled_motion_ankle", FeatureIntent.CONTROL),
    FeatureSpec("joint_smoothness", "cycle_repeatability", FeatureIntent.REPEATABILITY),
    FeatureSpec("joint_smoothness", "cycle_smoothness", FeatureIntent.SMOOTHNESS),
    FeatureSpec("joint_smoothness", "lr_phase_symmetry", FeatureIntent.ASYMMETRY),
    FeatureSpec("joint_smoothness", "spike_resilience", FeatureIntent.SMOOTHNESS),
    FeatureSpec("joint_smoothness", "rom_deg", FeatureIntent.MAGNITUDE, penalizes_magnitude=False),
    # Cycle consistency
    FeatureSpec("cycle_consistency", "cycle_repeatability_score", FeatureIntent.REPEATABILITY),
    FeatureSpec("cycle_consistency", "left_right_knee_phase_consistency", FeatureIntent.ASYMMETRY),
    FeatureSpec("cycle_consistency", "cycle_duration_cv", FeatureIntent.VARIABILITY),
    FeatureSpec("cycle_consistency", "mean_cycle_rmse", FeatureIntent.REPEATABILITY),
    # Contact pattern
    FeatureSpec("contact_pattern", "left_stance_duration_cv", FeatureIntent.VARIABILITY),
    FeatureSpec("contact_pattern", "right_stance_duration_cv", FeatureIntent.VARIABILITY),
    FeatureSpec("contact_pattern", "contact_toggle_rate_hz", FeatureIntent.CONTACT_TIMING),
    FeatureSpec("contact_pattern", "double_support_fraction", FeatureIntent.CONTACT_TIMING),
    FeatureSpec("contact_pattern", "heel_strike_alternation", FeatureIntent.CONTACT_TIMING),
)


MAGNITUDE_FEATURES_NOT_PENALIZED = tuple(
    f.feature
    for f in STABILITY_FEATURE_REGISTRY
    if f.intent == FeatureIntent.MAGNITUDE and not f.penalizes_magnitude
)


def registry_report_lines() -> list[str]:
    """Human-readable classification table."""
    lines = [
        f"{'Domain':<22}{'Feature':<36}{'Class':<18}{'Penalizes mag?'}",
        "-" * 90,
    ]
    for spec in STABILITY_FEATURE_REGISTRY:
        pen = "yes" if spec.penalizes_magnitude else "no"
        lines.append(f"{spec.domain:<22}{spec.feature:<36}{spec.intent.value:<18}{pen}")
    return lines


def registry_by_domain() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for spec in STABILITY_FEATURE_REGISTRY:
        out.setdefault(spec.domain, []).append(
            {
                "feature": spec.feature,
                "class": spec.intent.value,
                "penalizes_magnitude": spec.penalizes_magnitude,
                "notes": spec.notes,
            }
        )
    return out
