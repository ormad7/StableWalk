"""
Central configuration for the redesigned gait stability score (v2).

All thresholds and weights live here — domain scorers read from
``DEFAULT_STABILITY_CONFIG`` rather than hard-coding magic numbers.

Legacy v1 constants remain in ``biomech_stability.py`` for reference only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stablewalk.analysis.gait_evidence import (
    DEFAULT_GAIT_EVIDENCE_THRESHOLDS,
    GaitEvidenceThresholds,
)


@dataclass(frozen=True)
class ThresholdBand:
    """
    Map a normalized feature to 0–100.

    For *lower_is_better* metrics: value <= good -> 100, value >= poor -> 0.
    For *higher_is_better* metrics: invert good/poor semantics via ``higher_is_better``.
    """

    good: float
    poor: float
    higher_is_better: bool = False

    def score(self, value: float) -> float:
        if self.higher_is_better:
            if value >= self.good:
                return 100.0
            if value <= self.poor:
                return 0.0
            frac = (value - self.poor) / (self.good - self.poor)
        else:
            if value <= self.good:
                return 100.0
            if value >= self.poor:
                return 0.0
            frac = (value - self.good) / (self.poor - self.good)
        return max(0.0, min(100.0, 100.0 * (1.0 - frac)))


@dataclass(frozen=True)
class StabilityScoreConfig:
    """Weights, classification cutoffs, and feature thresholds for stability v2."""

    # Final classification (derived from motion only — no gait-type labels).
    stable_min: float = 70.0
    moderate_min: float = 45.0

    # Sub-score weights (must sum to 1.0).
    weight_temporal_symmetry: float = 0.14
    weight_spatial_symmetry: float = 0.12
    weight_pelvis_stability: float = 0.14
    weight_trunk_stability: float = 0.10
    weight_foot_clearance: float = 0.14
    weight_joint_smoothness: float = 0.14
    weight_cycle_consistency: float = 0.12
    weight_contact_pattern: float = 0.10

    min_frames: int = 8
    min_cycles_for_cycle_score: int = 2
    gait_evidence: GaitEvidenceThresholds = field(default_factory=lambda: DEFAULT_GAIT_EVIDENCE_THRESHOLDS)

    # Temporal symmetry — L/R ratio asymmetry (1.0 = perfect; lower = worse).
    temporal_stance_asymmetry: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.92, poor=0.55)
    )
    temporal_swing_asymmetry: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.92, poor=0.55)
    )
    temporal_step_time_asymmetry: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.90, poor=0.50)
    )

    # Spatial symmetry — normalized L/R differences (lower is better).
    spatial_step_length_asymmetry: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.08, poor=0.35)
    )
    spatial_stride_length_asymmetry: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.08, poor=0.35)
    )
    spatial_foot_progression_asymmetry: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.10, poor=0.40)
    )

    # Pelvis stability — body-normalized irregularity (lower is better).
    pelvis_lateral_sway: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.06, poor=0.20)
    )
    pelvis_vertical_oscillation_cv: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.12, poor=0.45)
    )
    pelvis_velocity_inconsistency: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.15, poor=0.55)
    )
    pelvis_jerk_normalized: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.8, poor=3.5)
    )

    # Trunk stability.
    trunk_lateral_sway: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.05, poor=0.18)
    )
    trunk_lean_variation_deg: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=3.0, poor=12.0)
    )
    trunk_upper_oscillation: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.06, poor=0.22)
    )

    # Foot clearance (body-scale meters).
    clearance_asymmetry_ratio: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.08, poor=0.45)
    )
    clearance_swing_variability: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.18, poor=0.55)
    )
    toe_drag_clearance_m: float = 0.015
    toe_drag_penalty_per_event: float = 12.0

    # Joint smoothness — irregularity not magnitude.
    joint_jerk_normalized: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.45, poor=1.35)
    )
    joint_velocity_cv: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.20, poor=0.65)
    )
    jerk_rom_floor_deg: float = 8.0

    # Gait cycle consistency.
    cycle_duration_cv: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.06, poor=0.30)
    )
    cycle_shape_similarity: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.82, poor=0.45, higher_is_better=True)
    )

    # Contact pattern.
    contact_stance_cv: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.10, poor=0.40)
    )
    contact_toggle_rate_hz: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.8, poor=4.0)
    )
    double_support_deviation: ThresholdBand = field(
        default_factory=lambda: ThresholdBand(good=0.08, poor=0.25)
    )
    expected_double_support_fraction: float = 0.14

    # Pose quality gates confidence — does not dominate the final score.
    pose_quality_weight_cap: float = 0.0  # v2: no separate pose-quality bucket

    def sub_score_weights(self) -> dict[str, float]:
        return {
            "temporal_symmetry": self.weight_temporal_symmetry,
            "spatial_symmetry": self.weight_spatial_symmetry,
            "pelvis_stability": self.weight_pelvis_stability,
            "trunk_stability": self.weight_trunk_stability,
            "foot_clearance": self.weight_foot_clearance,
            "joint_smoothness": self.weight_joint_smoothness,
            "cycle_consistency": self.weight_cycle_consistency,
            "contact_pattern": self.weight_contact_pattern,
        }

    def validate(self) -> None:
        total = sum(self.sub_score_weights().values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Stability sub-score weights must sum to 1.0, got {total:.4f}")


DEFAULT_STABILITY_CONFIG = StabilityScoreConfig()

# Document legacy v1 for auditors (see biomech_stability.py for implementation).
LEGACY_V1_DOCUMENTATION = """
Legacy stability score (v1) — superseded by stability_scoring.py

Six weighted groups (renormalized if missing):
  symmetry 22%, step_consistency 24%, body_stability 14%,
  range_of_motion 22%, trajectory_smoothness 11%, pose_quality 7%.

Key features and direction (lower is better unless noted):
  - L/R knee/hip/ankle mean abs angle diff (deg) -> symmetry
  - Same-side step interval CV -> step_consistency
  - Pelvis lateral sway / shoulder width -> body_stability
  - L/R ROM ratio difference -> range_of_motion
  - |Δ²θ|/ROM joint jerk -> trajectory_smoothness (lower better)
  - Foot visibility + frame coverage -> pose_quality (higher better)

Normalization: shoulder width, body height (image-normalized 0-1 coords).
Duration/FPS: step coverage penalizes short clips; CV uses actual FPS.
Correlated pairs: symmetry↔ROM (both L/R angle), step_consistency↔symmetry (timing),
  body_stability low sway + low ROM bias (restricted gait scored as stable).
Known issue: similar scores (~50-63) across abnormal/normal/athletic demos.
"""
