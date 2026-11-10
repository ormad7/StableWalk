"""
Canonical Metric Details for StableWalk — formulas from implemented code only.

Each entry documents the mathematical definition as written in analysis modules.
Do not invent equations here; update entries when the source code changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDetails:
    """Structured scientific documentation for one computed metric."""

    key: str
    title: str
    mathematical_definition: str
    coordinate_system: str
    units: str
    assumptions: str
    limitations: str
    confidence_estimation: str
    source: str  # file:function as implemented

    def format_text(self) -> str:
        return (
            f"{self.title}\n"
            f"{'=' * len(self.title)}\n\n"
            f"Mathematical definition\n"
            f"{self.mathematical_definition}\n\n"
            f"Coordinate system\n"
            f"{self.coordinate_system}\n\n"
            f"Units\n"
            f"{self.units}\n\n"
            f"Assumptions\n"
            f"{self.assumptions}\n\n"
            f"Limitations\n"
            f"{self.limitations}\n\n"
            f"Confidence estimation\n"
            f"{self.confidence_estimation}\n\n"
            f"Implementation\n"
            f"{self.source}"
        )


# Keys used by KPI cards / charts that map to a catalog entry.
_KEY_ALIASES: dict[str, str] = {
    "stability": "stability_margin",
    "grf": "vgrf",
    "estimated_vgrf": "vgrf",
    "virtual_grf": "vgrf",
    "bos": "base_of_support",
    "joint_rom": "rom",
    "joint_angle": "rom",
    "knee_motion": "rom",
    "walking_speed_m_s": "walking_speed",
    "cadence_spm": "cadence",
    "symmetry_pct": "symmetry",
    "gait_quality_overview": "gait_coordination",
    "heel_strikes": "heel_strike",
    "stance": "stance_pct",
    "swing": "swing_pct",
    "tracking_confidence": "tracking_confidence",
    "pipeline_confidence": "pipeline_confidence",
    "video_quality": "video_quality",
    "joint_movement_3d": "trajectory_path",
    "biomechanics": "biomechanics_timeline",
    "contact_gait": "heel_strike",
    "contact_vgrf": "vgrf",
    "knee": "rom",
    "knee_flexion": "rom",
    "joint_motion": "rom",
    "joint_motion_analysis": "rom",
    "compare_knee": "rom",
}


def _d(
    key: str,
    title: str,
    *,
    definition: str,
    coords: str,
    units: str,
    assumptions: str,
    limitations: str,
    confidence: str,
    source: str,
) -> MetricDetails:
    return MetricDetails(
        key=key,
        title=title,
        mathematical_definition=definition,
        coordinate_system=coords,
        units=units,
        assumptions=assumptions,
        limitations=limitations,
        confidence_estimation=confidence,
        source=source,
    )


METRIC_DETAILS: dict[str, MetricDetails] = {
    "movement_stability": _d(
        "movement_stability",
        "Movement Stability (derived score)",
        definition=(
            "Domain-group score over pelvis_stability, trunk_stability, and "
            "joint_smoothness.\n\n"
            "score = Σ(score_i · weight_i · confidence_i) / Σ(weight_i · confidence_i)\n"
            "over domains with availability ≠ UNAVAILABLE and score ≠ None.\n\n"
            "completeness_pct = 100 · Σ(weight·confidence)_active / Σ(weight)_group"
        ),
        coords=(
            "Derived score (dimensionless). Domain features use gait-frame / "
            "root-relative pelvis and joint kinematics (Y-up pose)."
        ),
        units="0–100 score",
        assumptions=(
            "Domain scorers in stability_scoring.py are available; weights come "
            "from the stability configuration. Interprets torso/pelvis control "
            "and joint smoothness, not clinical diagnosis."
        ),
        limitations=(
            "Not a force-plate or instrumented balance measure. Incomplete domains "
            "reduce completeness_pct and can leave the score None."
        ),
        confidence=(
            "Uses each domain's confidence in the weighted mean. "
            "Group completeness_pct reflects weighted confidence coverage."
        ),
        source="gait_analysis_summary.py:_group_score (+ stability_scoring domain scorers)",
    ),
    "gait_coordination": _d(
        "gait_coordination",
        "Gait Coordination (Overview derived score)",
        definition=(
            "Domain-group score over temporal_symmetry, spatial_symmetry, "
            "foot_clearance, cycle_consistency, and contact_pattern.\n\n"
            "score = Σ(score_i · weight_i · confidence_i) / Σ(weight_i · confidence_i)\n\n"
            "Distinct from Biomechanics Composite Gait Quality "
            "(gait_quality_score.py), which uses a different component set."
        ),
        coords="Derived score from spatiotemporal and contact domains.",
        units="0–100 score",
        assumptions=(
            "Requires scored symmetry / clearance / cycle / contact domains from "
            "stability_scoring.py."
        ),
        limitations=(
            "Overview label may share the key name with Composite Gait Quality "
            "elsewhere; algorithms differ (see gait_quality_score docstring)."
        ),
        confidence="Weighted by domain confidence; completeness_pct as in _group_score.",
        source="gait_analysis_summary.py:_group_score",
    ),
    "gait_quality": _d(
        "gait_quality",
        "Gait Quality (heuristic descriptor)",
        definition=(
            "Heuristic session descriptor — not a validated gait-quality instrument.\n"
            "Weighted mean of available components (weights renormalized over active):\n"
            "  stability            0.22\n"
            "  stability_margin     0.15  (= stable_pct; omitted if unreportable)\n"
            "  symmetry             0.20\n"
            "  cadence_consistency  0.12  (= cadence_consistency.value × 100)\n"
            "  cycle_consistency    0.12  (= gait_cycle_consistency × 100)\n"
            "  contact_reliability  0.10  (= contact_confidence × 100)\n"
            "  motion_quality       0.09  (= joint_smoothness when available)\n\n"
            "score = Σ(component_k · weight_k) / Σ(weight_k) for active k\n\n"
            "Separate from pose coverage, landmark visibility, cycle-detection "
            "confidence, spectral reliability, and physical stability."
        ),
        coords="Heuristic composite from pose-estimated proxies.",
        units="0–100 heuristic score",
        assumptions=(
            "Uses StabilityResult, reportable StabilityMarginAnalysis.stable_pct, "
            "SymmetryAnalysis, cycle consistency, and foot-contact confidence when present."
        ),
        limitations=(
            "Not a validated clinical instrument. Missing components are omitted "
            "and weights renormalized, so a high number can mean few inputs. "
            "Unreportable COM–BoS stays unavailable. Display bands (≥60 / ≥45) "
            "are heuristic, not clinical normality, and independent of source "
            "video labels (Normal / Abnormal / Fast Walking)."
        ),
        confidence=(
            "0.5·(n_active/n_intended) + 0.5·mean(real contact/completeness scores). "
            "Missing contact is not defaulted to 0.5."
        ),
        source="biomechanical/gait_quality_score.py:compute_gait_quality_score",
    ),
    "pelvis_stability": _d(
        "pelvis_stability",
        "Pelvis Stability (heuristic domain)",
        definition=(
            "Gait-frame pelvis domain used by Movement Stability / Advanced.\n"
            "score = mean of available sub-scores:\n"
            "  • mediolateral sway / hip width (lower better)\n"
            "  • vertical oscillation / hip width (lower better)\n"
            "  • mediolateral acceleration (or velocity) CV (lower better)\n"
            "  • lateral jerk / hip width (lower better)\n\n"
            "Each raw feature is mapped through a heuristic scoring band in "
            "score_pelvis_stability. Missing submetrics are omitted from the mean.\n"
            "This is not the Composite Gait Quality score and not a clinical "
            "pelvis-stability grade."
        ),
        coords="Gait-frame pelvis (Y-up); sway normalized by hip width.",
        units="0–100 heuristic domain score",
        assumptions="Enough tracked pelvis frames (min_frames / clip length).",
        limitations=(
            "Pose-derived heuristic. No documented clinical Acceptable cut-off. "
            "Not interchangeable with COM–BoS margin or gait-quality 92.5."
        ),
        confidence="Domain availability/confidence from StabilityResult.metric.",
        source="stability_scoring.py:score_pelvis_stability",
    ),
    "analysis_confidence": _d(
        "analysis_confidence",
        "Analysis Confidence",
        definition=(
            "Results no longer treat a single Analysis Confidence as the session "
            "headline. Pose coverage, per-foot contact coverage, spectral "
            "reliability, and biomechanical validity are reported separately.\n\n"
            "The legacy composite remains in exports only. High pose coverage "
            "must not be read as High spectral reliability or physical stability."
        ),
        coords="Meta-metric (not a spatial quantity).",
        units="0–100% data-quality aggregate, or categorical level text",
        assumptions="At least one pose/cycle quality signal is available.",
        limitations=(
            "Measures data quality for interpretability — not clinical certainty "
            "or gait performance. OpenSim incompleteness does not zero pose-only "
            "confidence."
        ),
        confidence="This metric is itself the data-quality readout.",
        source=(
            "overview_clinical_summary.py:extract_confidence_visualization; "
            "gait_analysis_summary.py:_build_analysis_confidence (categorical level)"
        ),
    ),
    "cadence": _d(
        "cadence",
        "Cadence (calculated)",
        definition=(
            "From complete same-side heel-strike cycles with reliable contact:\n"
            "  step_time = mean(alternating left/right HS intervals)\n"
            "  stride_time = mean(cycle durations)\n"
            "  cadence = 60 / step_time   if step_time > 0\n"
            "          = 120 / stride_time otherwise"
        ),
        coords="Temporal (event times in seconds).",
        units="steps/min",
        assumptions=(
            f"Requires ≥2 complete cycles (MIN_CYCLES_FOR_CADENCE), "
            f"contact_confidence ≥ 0.48 (MIN_RELIABLE_CONTACT_CONFIDENCE), "
            "and confidence_tier ≠ LOW_CONFIDENCE."
        ),
        limitations=(
            "Heel strikes are estimated from pose contact timing, not force plates. "
            "Unreliable sessions leave cadence unset (metrics_reliable=False)."
        ),
        confidence=(
            "Gated by contact_confidence and confidence_tier; stored on "
            "GaitTemporalMetrics.contact_confidence."
        ),
        source="gait_cycle_analysis.py:compute_temporal_metrics",
    ),
    "walking_speed": _d(
        "walking_speed",
        "Walking Speed (estimated)",
        definition=(
            "Best of three monocular estimators (highest confidence wins):\n"
            "  1) Global pelvis forward speed: median |Δfwd| · meters_per_unit / Δt\n"
            "  2) Cadence × step length (or stride/2) in metres\n"
            "  3) Image-plane pelvis drift scaled by subject stature\n\n"
            "Plausible band: 0.25–3.5 m/s. Reported only if confidence ≥ MIN_REPORT_CONFIDENCE "
            "(0.35) after conf = max(base·0.85, min(0.95, best.confidence))."
        ),
        coords=(
            "Global / stature-scaled metres where possible; body-normalized pose "
            "converted with subject_height_m (default 1.70 m)."
        ),
        units="m/s",
        assumptions=(
            "Subject height (or default) scales normalized pose to metres. "
            "Hip-centered COM/ankle velocity methods are excluded by design."
        ),
        limitations=(
            "Monocular absolute scale is approximate. Returns None when no method "
            "is plausible or confidence is below the report gate."
        ),
        confidence="Per-method confidence; best candidate selected; clipped report gate.",
        source="biomechanical/walking_speed.py:estimate_walking_speed",
    ),
    "symmetry": _d(
        "symmetry",
        "Gait Symmetry (derived)",
        definition=(
            "Component indices (when metrics_reliable):\n"
            "  stance/swing:  symmetry_ratio = min(L,R) / max(L,R)\n"
            "  step/stride/ROM: symmetry_index = 2·min(L,R) / (L+R+eps)\n"
            "  cadence_consistency = 1 − CV(cycle duration)\n\n"
            "overall_symmetry_pct = 100 · Σ(index·w) / Σ(w)\n"
            "weights: stride 1.4, step 1.3, stance 1.2, swing 1.2, "
            "cadence_consistency 1.1, knee ROM 1.0, hip 0.9, ankle 0.9\n"
            "(needs ≥2 valid components).\n\n"
            "This composite is not interchangeable with knee SI_norm on the "
            "Asymmetry page, nor with signed Robinson SI or absolute ASI. "
            "A symmetric knee can sit inside an asymmetric overall score, "
            "and the reverse is also true."
        ),
        coords="Derived from bilateral temporal and spatial gait features.",
        units="Overall as %; component indices in [0, 1]",
        assumptions="Reliable gait cycles (metrics_reliable) and bilateral feature availability.",
        limitations=(
            "Unavailable when cycles are unreliable or fewer than two components. "
            "Display bands (≥70% / ≥55%) are heuristic, not clinical laterality "
            "or normality, and independent of source video labels "
            "(Normal / Abnormal / Fast Walking)."
        ),
        confidence=(
            "Base = cycles.metrics.contact_confidence; components scale by "
            "0.9–0.75 factors as coded in analyze_symmetry."
        ),
        source="biomechanical/symmetry_metrics.py:analyze_symmetry",
    ),
    "stability_margin": _d(
        "stability_margin",
        "Stability Margin (derived)",
        definition=(
            "Per frame: project COM to (x, z). Contact feet are placed on the COM "
            "support plane per foot so inter-foot step length is not treated as BoS "
            "depth. Align remaining polygon mean Z to COM Z so visualization "
            "DEPTH_LAYERS (feet behind trunk) are not treated as lean; "
            "if remaining AP span is below a measured foot length (~DEPTH_LAYERS "
            "heel–toe of 1 cm), collapse to the mediolateral interval and use the "
            "two-point thickened-foot model (5 cm physical half-width). Then compute "
            "signed distance in metres (_signed_margin_to_polygon).\n"
            "State (_classify_margin):\n"
            "  Stable if margin_m ≥ 0.04\n"
            "  Reduced Stability if margin_m ≥ 0.0\n"
            "  Unstable otherwise\n"
            "  Unavailable if margin missing, carried-stance fallback, or frame conf < 0.35\n\n"
            "mean_margin_m = mean(valid margins)\n"
            "stable_pct = 100 · (# Stable) / (# valid margins)\n"
            "Session metrics cleared if valid_frame_ratio < 0.50."
        ),
        coords=(
            "Horizontal X–Z plane; COM projection (position[0], position[2]); "
            "Y-up pose. Margin converted to metres via meters_per_unit."
        ),
        units="m (margin); % (stable_pct); categorical state",
        assumptions=(
            "Estimated COM and foot BoS polygons; thresholds ~ fraction of typical "
            "foot length (~0.25 m body scale) as noted in stability_margin.py."
        ),
        limitations=(
            "Pose-estimated COM/BoS — not force-plate extrapolated CoM or measured "
            "support. Invalid if too few confident frames."
        ),
        confidence=(
            "Per-frame conf = min(COM conf, BoS conf); session confidence = mean of "
            "valid-frame confidences (≥ 0.35)."
        ),
        source="biomechanical/stability_margin.py:analyze_stability_margin",
    ),
    "stability_state": _d(
        "stability_state",
        "Stability State (derived)",
        definition=(
            "Classification of stability_margin_m via _classify_margin:\n"
            "  ≥ 0.04 m → Stable\n"
            "  ≥ 0.0 m  → Reduced Stability\n"
            "  < 0.0 m  → Unstable\n"
            "  else     → Unavailable"
        ),
        coords="Same as Stability Margin (COM vs BoS in X–Z).",
        units="Categorical",
        assumptions="Uses the same margin thresholds as analyze_stability_margin.",
        limitations="Thresholds are fixed heuristics for pose-estimated margins.",
        confidence="Unavailable when per-frame confidence < 0.35.",
        source="biomechanical/stability_margin.py:_classify_margin",
    ),
    "com": _d(
        "com",
        "Center of Mass (estimated)",
        definition=(
            "Segment COM along proximal→distal: r = p + t·(d − p) with t = com_frac.\n"
            "Whole-body: COM = Σ(m_i · r_i) / Σ m_i\n"
            "Mass fractions (code _SEGMENT_SPECS): head 0.081, trunk_upper 0.150, "
            "trunk_lower 0.347, thighs 0.100 each, shanks 0.0465 each, feet 0.0145 each, "
            "arms 0.028 each.\n"
            "Fallback: mid-hip if no segments.\n"
            "Kinematics: v[i]=(x[i]−x[i−1])/dt ; a[i]=(v[i]−v[i−1])/dt\n"
            "Charts often plot COM height as body-normalized Y (BH).\n"
            "BH = body height: pelvis-relative, hip-centered vertical position. "
            "The measured head–ankle span is often not 1.0, so this is not "
            "floor-absolute COM height and is not tape-calibrated metres."
        ),
        coords=(
            "Hip-/root-relative skeleton; Y-up. Horizontal projection for BoS uses (X, Z)."
        ),
        units="Pose units / body heights (BH) on charts; stature-scaled m/cm in summaries",
        assumptions="de Leva-style segment mass fractions; joints mark segment endpoints.",
        limitations=(
            "Estimated from monocular landmarks — not force-plate or OpenSim ID COM."
        ),
        confidence=(
            "mean(segment confidences) · min(1, n_segments / n_specs); mid-hip fallback 0.35."
        ),
        source="biomechanical/com_estimation.py:estimate_frame_com",
    ),
    "base_of_support": _d(
        "base_of_support",
        "Base of Support (estimated)",
        definition=(
            "Foot landmarks projected to (x, z); convex hull polygon; area via shoelace. "
            "Support type from left/right contact state."
        ),
        coords="Horizontal X–Z plane in pose coordinates (optionally scaled to m).",
        units="m² (when scaled) or body-normalized area",
        assumptions="Contact-detected heel/toe landmarks define the support polygon.",
        limitations="Estimated footprints from video pose — not pressure mats.",
        confidence="Averaged from contact confidence on supporting feet.",
        source="biomechanical/base_of_support.py:analyze_base_of_support",
    ),
    "vgrf": _d(
        "vgrf",
        "Estimated Virtual GRF (vGRF)",
        definition=(
            "BW = m · g with g = 9.81, default m = 70 kg.\n"
            "Vertical height series h(t) (prefer global pelvis stature-scaled metres).\n"
            "Acceleration a_z from finite differences (5-point box smooth when n≥5).\n"
            "  F_total = m · (a_z + g), clipped to [0, 2.5·BW]\n"
            "Allocated to left/right by contact binaries and contact probabilities.\n"
            "vGRF_BW = F / BW."
        ),
        coords=(
            "Vertical SI metres (prefer global pelvis before hip-centering); "
            "timing from foot-contact frames."
        ),
        units="N (force) and body-weight multiples (BW)",
        assumptions=(
            "Body mass (default 70 kg) and stature scaling; contact masks define stance."
        ),
        limitations=(
            "Code note: not measured kinetics or PhysX. Hip-centered fallback weakens "
            "vertical oscillation."
        ),
        confidence=(
            "Per frame: min(1, contact_confidence·0.5 + left_conf·left_prob·0.25 + "
            "right_conf·right_prob·0.25)."
        ),
        source="estimated_vgrf_analysis.py:analyze_estimated_vgrf",
    ),
    "rom": _d(
        "rom",
        "Joint ROM (estimated)",
        definition=(
            "Knee/hip flexion angle = 180° − interior joint angle; ankle keeps interior.\n"
            "Per gait cycle: robust span (P10–P90 with MAD gate); rom = hi − lo.\n"
            "Session ROM = mean across qualifying cycles."
        ),
        coords="Pose joint vectors in the tracked skeleton frame (root-relative).",
        units="degrees (°)",
        assumptions="Adequate landmark coverage per cycle (≥60% coverage in analyzer).",
        limitations="Not a clinical goniometer; monocular depth noise affects angles.",
        confidence="Uses contact confidence; soft-max rules can lower confidence.",
        source="biomechanical/joint_rom.py:analyze_joint_rom",
    ),
    "heel_strike": _d(
        "heel_strike",
        "Heel Strike (estimated timing)",
        definition=(
            "Rising edge of binary foot-contact after clearance + velocity Schmitt "
            "trigger (detect_gait_events / foot contact pipeline).\n"
            "Gait cycles = consecutive same-side heel strikes."
        ),
        coords="Temporal events on the contact timeline (seconds / frames).",
        units="time (s); counts",
        assumptions="Pose-based clearance and foot velocity approximate ground contact.",
        limitations="Timing only — not measured ground reaction or pressure onset.",
        confidence="assess_contact_confidence (plane, visibility, duration, HS count).",
        source="gait_cycle_analysis.py:detect_gait_events; foot_contact_analysis.py",
    ),
    "toe_off": _d(
        "toe_off",
        "Toe Off (estimated timing)",
        definition=(
            "Falling edge of binary foot-contact (exit when clearance / displayed "
            "clearance exceeds exit thresholds)."
        ),
        coords="Temporal events on the contact timeline.",
        units="time (s); counts",
        assumptions="Same contact Schmitt model as heel strike.",
        limitations="Estimated timing only; not kinetics.",
        confidence="Shared contact_confidence with heel-strike detection.",
        source="gait_cycle_analysis.py:detect_gait_events",
    ),
    "stance_pct": _d(
        "stance_pct",
        "Stance % (calculated)",
        definition=(
            "Per complete HS–HS cycle: integrate contact time for each side.\n"
            "swing_side = max(cycle.duration_s − stance_side, 0)\n"
            "stance_% = mean(stance / duration × 100) across usable cycles "
            "(bilateral mean of L/R used in UI)."
        ),
        coords="Temporal fraction of same-side heel-strike cycles.",
        units="% of gait cycle",
        assumptions="Reliable complete cycles and contact masks.",
        limitations="Depends on estimated contact timing.",
        confidence="Requires metrics_reliable (cycle count + contact confidence gates).",
        source="gait_cycle_analysis.py:compute_temporal_metrics",
    ),
    "swing_pct": _d(
        "swing_pct",
        "Swing % (calculated)",
        definition=(
            "swing_side = max(duration − stance_side, 0); "
            "swing_% = mean(swing / duration × 100). "
            "UI often shows bilateral mean of left/right."
        ),
        coords="Temporal fraction of gait cycles.",
        units="% of gait cycle",
        assumptions="Same as Stance %.",
        limitations="Same contact-estimation limits as Stance %.",
        confidence="Same reliability gate as temporal metrics.",
        source="gait_cycle_analysis.py:compute_temporal_metrics",
    ),
    "double_support": _d(
        "double_support",
        "Double Support (estimated)",
        definition=(
            "Within each complete cycle: time both feet in contact "
            "(_cycle_support_durations). double_support_% = mean(ds / duration × 100)."
        ),
        coords="Temporal fraction of gait cycles.",
        units="% of gait cycle; event detection counts pulses/frames",
        assumptions="Binary L/R contact from pose.",
        limitations="Estimated contact — not force-plate double support.",
        confidence="Contact confidence / Summary event detection counts.",
        source="gait_cycle_analysis.py:_cycle_support_durations",
    ),
    "single_support": _d(
        "single_support",
        "Single Support (estimated, ipsilateral)",
        definition=(
            "Ipsilateral mean: 0.5·(left_only + right_only) per cycle, then % of cycle. "
            "Comment in code: typically ~40%; not total exclusive single-support time."
        ),
        coords="Temporal fraction of gait cycles.",
        units="% of gait cycle",
        assumptions="Binary contact states per frame.",
        limitations="Ipsilateral averaging definition — read UI label carefully.",
        confidence="Same temporal reliability gates.",
        source="gait_cycle_analysis.py:compute_temporal_metrics",
    ),
    "step_length": _d(
        "step_length",
        "Step Length (estimated)",
        definition=(
            "At heel strike: foot separation along the dominant progression axis "
            "(X or Z by median). Mean of left/right steps × meters_per_unit."
        ),
        coords="Horizontal pose axes → metres via stature/leg-length scale.",
        units="m (UI often cm)",
        assumptions="Detectable alternating HS and foot landmarks.",
        limitations="Monocular scale; progression axis chosen from data statistics.",
        confidence="Typically contact_confidence × 0.85 in advanced metrics wiring.",
        source="gait_feature_analysis.py:_step_lengths_from_foot_events",
    ),
    "stride_length": _d(
        "stride_length",
        "Stride Length (estimated)",
        definition=(
            "stride ≈ step_L + step_R (or 2× one side when only one available), "
            "then × meters_per_unit. Normalized forms divide by leg length."
        ),
        coords="Same horizontal progression frame as step length.",
        units="m",
        assumptions="Step lengths available from foot events.",
        limitations="Same monocular scale limits as step length.",
        confidence="Tied to feature/contact confidence.",
        source="gait_feature_analysis.py:compute_normalized_gait_features",
    ),
    "step_width": _d(
        "step_width",
        "Step Width (estimated)",
        definition=(
            "Median ankle horizontal separation; width = smaller of |Δx| / |Δz| "
            "components × meters_per_unit (orthogonal to progression)."
        ),
        coords="Horizontal plane.",
        units="m",
        assumptions="Tracked ankles across steps.",
        limitations="Estimated from pose; not pressure-mat base of support width.",
        confidence="Often contact_confidence × 0.75 in advanced metrics.",
        source="biomechanical/advanced_gait_metrics.py:_mean_step_width",
    ),
    "foot_clearance": _d(
        "foot_clearance",
        "Foot Clearance (estimated)",
        definition=(
            "ground_distance_m = point_Y − floor_y\n"
            "clearance = max(0, distance)\n"
            "displayed clearance = min(heel, toe) clearance "
            "(ankle excluded from displayed minimum)."
        ),
        coords="Y-up vs estimated ground plane.",
        units="m (UI cm)",
        assumptions="Estimated floor plane and foot landmarks.",
        limitations="Monocular; domain score emphasizes consistency, not absolute height.",
        confidence="Contact / visibility confidence on clearance samples.",
        source="ground_reference.py:foot_clearance_m; displayed_foot_clearance_m",
    ),
    "video_quality": _d(
        "video_quality",
        "Pose / video heuristics (not tracking accuracy)",
        definition=(
            "Two separate layers plus a heuristic aggregate:\n"
            "  pose_coverage = detected_frames / n_frames (presence only)\n"
            "  landmark_confidence = mean detector visibility "
            "(not tracking accuracy)\n"
            "Aggregate starts at 100 and is reduced by truncation, camera jitter, "
            "FPS, and detection-rate penalties."
        ),
        coords="Image / detection heuristics (not skeletal accuracy).",
        units="coverage % · visibility 0–1 · optional 0–100 aggregate",
        assumptions="Access to frame stats and pose detection rates.",
        limitations=(
            "Detected landmarks ≠ accurate tracking. This is not spectral, "
            "cycle, or biomechanical quality."
        ),
        confidence="Layer statuses are computed / limited / insufficient / unavailable.",
        source="video_quality.py:assess_video_quality",
    ),
    "tracking_confidence": _d(
        "tracking_confidence",
        "Tracking Confidence",
        definition=(
            "Overview Tracking reports three separate quantities: person-detection "
            "coverage, mean joint visibility on detected frames, and usable "
            "gait-analysis frames. It is not a single 100% score."
        ),
        coords="Pose landmark visibility.",
        units="/100 or %",
        assumptions="MediaPipe (or backend) visibility available.",
        limitations="Visibility ≠ biomechanical accuracy.",
        confidence="This field is the tracking confidence estimate.",
        source="analysis_summary.py:_tracking_confidence_field",
    ),
    "pipeline_confidence": _d(
        "pipeline_confidence",
        "Pipeline Confidence",
        definition=(
            "Mean of stage scores: completed=1.0, partial=0.55, unavailable=0.0, "
            "then ×100. Levels: High ≥75 / Moderate ≥45 / Low."
        ),
        coords="Pipeline stage status (engineering meta-metric).",
        units="level + %",
        assumptions="Pipeline status diagram/groups available.",
        limitations="Does not quantify biomechanical correctness.",
        confidence="Derived from stage completion only.",
        source="analysis_summary.py:_pipeline_confidence_field",
    ),
    "trajectory_path": _d(
        "trajectory_path",
        "3D Joint Movement Path",
        definition=(
            "For selected joint positions p_i:\n"
            "  total_travel = Σ ‖Δp‖\n"
            "  smoothness from path/direct ratio "
            "(≤1.2 High, ≤1.55 Moderate, else Low)\n"
            "  max_deviation = max |out_of_plane − mean(out_of_plane)|"
        ),
        coords=(
            "Root-relative (pelvis origin) by default; optional global mode. "
            "X lateral, Y up, Z forward. UI path ticks often in cm."
        ),
        units="m (display often cm); smoothness categorical",
        assumptions="Tracked joint series for the selected DOF.",
        limitations="Path is joint motion relative to root — not whole-body walking translation.",
        confidence="Smoothness category acts as a qualitative path-confidence cue.",
        source="dashboard_interpretability.py:compute_trajectory_path_metrics",
    ),
    "biomechanics_timeline": _d(
        "biomechanics_timeline",
        "Biomechanics Timeline (stacked charts)",
        definition=(
            "Display stack of implemented series:\n"
            "  • COM Vertical Position — pelvis-relative, approximately "
            "body-height-normalized Y from estimate_frame_com\n"
            "  • Stability Margin — margin_m from analyze_stability_margin\n"
            "  • Contact Timing — estimated HS/TO / contact binaries\n"
            "  • Gait Phase — classify_gait_phase from L/R contact\n\n"
            "See individual Metric Details for each series' equations."
        ),
        coords="Shared timebase (s); COM in BH; margin in m; contact as timing.",
        units="Mixed (BH, m, timing, phase labels)",
        assumptions="BiomechanicalAnalysisResult populated from pose sequence.",
        limitations="All series are pose-estimated; see DISCLAIMER_MONOCULAR.",
        confidence=(
            "Cyan low-confidence wash where per-frame confidence < 0.55 "
            "(draw_confidence_overlay)."
        ),
        source="ui/viewers/biomechanics_charts.py (+ analysis modules above)",
    ),
}


def resolve_metric_key(key: str | None) -> str | None:
    if not key:
        return None
    k = str(key).strip()
    if not k:
        return None
    if k in METRIC_DETAILS:
        return k
    return _KEY_ALIASES.get(k)


def get_metric_details(key: str | None) -> MetricDetails | None:
    """Return MetricDetails for *key*, resolving common UI aliases."""
    resolved = resolve_metric_key(key)
    if resolved is None:
        return None
    return METRIC_DETAILS.get(resolved)


def list_metric_details() -> list[MetricDetails]:
    """All catalog entries in stable title order."""
    return sorted(METRIC_DETAILS.values(), key=lambda d: d.title.lower())


def metric_details_keys() -> tuple[str, ...]:
    return tuple(sorted(METRIC_DETAILS.keys()))


__all__ = [
    "METRIC_DETAILS",
    "MetricDetails",
    "get_metric_details",
    "list_metric_details",
    "metric_details_keys",
    "resolve_metric_key",
]
