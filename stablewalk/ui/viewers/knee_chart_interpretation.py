"""
Plain-language knee motion summary and diagnostic reporting for the gait chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from stablewalk.ui.viewers.knee_angle_chart import (
    ANGLE_CONVENTION_SUMMARY,
    KneeAngleSeries,
    knee_flexion_rom_deg,
    largest_frame_jump_deg,
)

if TYPE_CHECKING:
    from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult
    from stablewalk.analysis.gait_feature_analysis import GaitFeatureAnalysisResult

ChartMode = Literal["video_time", "gait_cycle_pct"]

MIN_CYCLES_FOR_CYCLE_MODE = 2
MIN_CONTACT_CONFIDENCE_FOR_PHASE_REGIONS = 0.48

GAIT_PHASE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("Early stance", 0.0, 10.0),
    ("Mid stance", 10.0, 30.0),
    ("Late stance", 30.0, 60.0),
    ("Early swing", 60.0, 75.0),
    ("Late swing", 75.0, 100.0),
)


@dataclass
class KneeMotionSummary:
    """Interpretation metrics shown beside / below the knee chart."""

    left_rom_deg: float | None
    right_rom_deg: float | None
    rom_asymmetry_pct: float | None
    cycle_repeatability: str | None
    largest_difference_phase: str | None
    plain_language: str
    angle_source_label: str
    angle_convention: str
    chart_mode: ChartMode
    gap_note: str | None = None


def usable_knee_cycle_count(gait_features: GaitFeatureAnalysisResult | None) -> int:
    """Number of complete resampled knee cycles available for cycle-normalized plots."""
    if gait_features is None:
        return 0
    cc = gait_features.cycle_consistency
    counts: list[int] = []
    for key in ("left_knee_angle", "right_knee_angle"):
        traj = cc.trajectories.get(key)
        if traj is not None and traj.per_cycle:
            counts.append(len(traj.per_cycle))
    return max(counts) if counts else 0


def cycle_mode_is_available(gait_features: GaitFeatureAnalysisResult | None) -> bool:
    return usable_knee_cycle_count(gait_features) >= MIN_CYCLES_FOR_CYCLE_MODE


def _repeatability_label(score: float | None, *, std_ratio: float | None = None) -> str | None:
    if score is not None:
        if score >= 70.0:
            return "High"
        if score >= 40.0:
            return "Moderate"
        return "Low"
    if std_ratio is not None:
        if std_ratio <= 0.12:
            return "High"
        if std_ratio <= 0.25:
            return "Moderate"
        return "Low"
    return None


def _largest_lr_difference_phase(
    gait_features: GaitFeatureAnalysisResult | None,
) -> str | None:
    if gait_features is None:
        return None
    cc = gait_features.cycle_consistency
    left = cc.trajectories.get("left_knee_angle")
    right = cc.trajectories.get("right_knee_angle")
    if left is None or right is None:
        return None
    l_mean = np.asarray(left.mean, dtype=float)
    r_mean = np.asarray(right.mean, dtype=float)
    if cc.angle_source == "mediapipe_angles":
        l_mean = 180.0 - l_mean
        r_mean = 180.0 - r_mean
    if l_mean.size != r_mean.size:
        return None
    diff = np.abs(l_mean - r_mean)
    best_phase = None
    best_val = -1.0
    for name, lo, hi in GAIT_PHASE_BUCKETS:
        i0 = int(round(lo))
        i1 = int(round(hi)) + 1
        i0 = max(0, min(i0, len(diff) - 1))
        i1 = max(i0 + 1, min(i1, len(diff)))
        region = diff[i0:i1]
        if region.size == 0:
            continue
        val = float(np.mean(region))
        if val > best_val:
            best_val = val
            best_phase = name
    return best_phase


def _gap_note(series: KneeAngleSeries) -> str | None:
    nan_pct = float(series.metadata.get("nan_pct", 0.0))
    if nan_pct < 5.0:
        return None
    return (
        "Gaps in the knee traces occur when pose landmarks are missing or unreliable "
        "for one or more frames; segments are not separate gait cycles."
    )


def _plain_language(
    *,
    left_rom: float | None,
    right_rom: float | None,
    asym_pct: float | None,
    repeatability: str | None,
    largest_phase: str | None,
    gap_note: str | None,
    cycle_count: int,
    mode: ChartMode,
) -> str:
    parts: list[str] = []

    if left_rom is not None and right_rom is not None:
        if asym_pct is not None and asym_pct <= 8.0:
            parts.append(
                "Left and right knee flexion ranges are similar across the analyzed video."
            )
        elif asym_pct is not None:
            parts.append(
                f"Left and right knee flexion ranges differ by about {asym_pct:.1f}% "
                "(range of motion asymmetry)."
            )
        else:
            parts.append("Left and right knee flexion were measured across the analyzed video.")

    if mode == "gait_cycle_pct" and cycle_count >= MIN_CYCLES_FOR_CYCLE_MODE:
        if repeatability == "High":
            parts.append(
                "Knee motion is repeatable across detected gait cycles with low cycle-to-cycle spread."
            )
        elif repeatability == "Moderate":
            parts.append(
                "Knee motion shows moderate variation between gait cycles."
            )
        elif repeatability == "Low":
            parts.append(
                "Knee flexion varies substantially between cycles, reducing movement consistency."
            )
        if largest_phase:
            parts.append(f"The largest left–right difference tends to occur during {largest_phase.lower()}.")
    elif mode == "video_time":
        if gap_note:
            parts.append(gap_note)

    if not parts:
        return "Insufficient knee flexion data to summarize motion."

    return " ".join(parts)


def build_knee_motion_summary(
    series: KneeAngleSeries | None,
    gait_features: GaitFeatureAnalysisResult | None,
    *,
    chart_mode: ChartMode,
) -> KneeMotionSummary:
    left_rom = right_rom = asym_pct = None
    repeatability = largest_phase = None
    gap_note = None
    source_label = "Pose-derived"
    convention = ANGLE_CONVENTION_SUMMARY

    if series is not None:
        left_rom = knee_flexion_rom_deg(series.left_deg)
        right_rom = knee_flexion_rom_deg(series.right_deg)
        if left_rom is not None and right_rom is not None:
            mean_rom = (left_rom + right_rom) / 2.0
            if mean_rom > 1e-6:
                asym_pct = 100.0 * abs(left_rom - right_rom) / mean_rom
        gap_note = _gap_note(series)
        source_label = (
            "OpenSim IK" if series.source == "opensim_ik" else "Pose-derived"
        )
        convention = series.angle_definition

    cc = gait_features.cycle_consistency if gait_features else None
    cycle_count = usable_knee_cycle_count(gait_features)
    std_ratio = None
    if chart_mode == "gait_cycle_pct" and cc is not None:
        if cc.angle_source == "opensim_ik":
            source_label = "OpenSim IK"
            convention = "OpenSim IK knee flexion (deg); 0 deg = extension"
        elif cc.angle_source == "mediapipe_angles":
            source_label = "Pose-derived"
            convention = "Pose interior angle -> flexion (180 deg - theta); 0 deg = extension"
        repeatability = _repeatability_label(cc.cycle_repeatability_score)
        largest_phase = _largest_lr_difference_phase(gait_features)
        if left_rom is None or right_rom is None:
            for key, rom_attr in (
                ("left_knee_angle", "left_rom"),
                ("right_knee_angle", "right_rom"),
            ):
                traj = cc.trajectories.get(key)
                if traj is None:
                    continue
                arr = np.asarray(traj.mean, dtype=float)
                if cc.angle_source == "mediapipe_angles":
                    arr = 180.0 - arr
                if np.any(np.isfinite(arr)):
                    val = float(np.nanmax(arr) - np.nanmin(arr))
                    if key.startswith("left"):
                        left_rom = val
                    else:
                        right_rom = val
        if left_rom is not None and right_rom is not None:
            mean_rom = (left_rom + right_rom) / 2.0
            if mean_rom > 1e-6:
                asym_pct = 100.0 * abs(left_rom - right_rom) / mean_rom
        left_t = cc.trajectories.get("left_knee_angle") if cc else None
        if left_t is not None and left_t.std and left_t.mean:
            m = np.asarray(left_t.mean, dtype=float)
            s = np.asarray(left_t.std, dtype=float)
            if cc and cc.angle_source == "mediapipe_angles":
                pass  # std unchanged under affine 180-x
            rom = float(np.nanmax(m) - np.nanmin(m)) if np.any(np.isfinite(m)) else 0.0
            if rom > 1e-6:
                std_ratio = float(np.nanmean(s) / rom)
                if repeatability is None:
                    repeatability = _repeatability_label(None, std_ratio=std_ratio)

    if chart_mode == "gait_cycle_pct" and cycle_count < MIN_CYCLES_FOR_CYCLE_MODE:
        plain = (
            "Insufficient complete gait cycles for cycle-normalized analysis. "
            "Switch to Video Time to inspect the full recording, or record a longer walk "
            "with clearer foot contact."
        )
        if left_rom is not None and right_rom is not None:
            plain += (
                f" Video-time ROM: left {left_rom:.0f}°, right {right_rom:.0f}°."
            )
    else:
        plain = _plain_language(
            left_rom=left_rom,
            right_rom=right_rom,
            asym_pct=asym_pct,
            repeatability=repeatability,
            largest_phase=largest_phase,
            gap_note=gap_note,
            cycle_count=cycle_count,
            mode=chart_mode,
        )

    return KneeMotionSummary(
        left_rom_deg=left_rom,
        right_rom_deg=right_rom,
        rom_asymmetry_pct=asym_pct,
        cycle_repeatability=repeatability,
        largest_difference_phase=largest_phase,
        plain_language=plain,
        angle_source_label=source_label,
        angle_convention=convention,
        chart_mode=chart_mode,
        gap_note=gap_note,
    )


def format_interpretation_panel(summary: KneeMotionSummary) -> str:
    lines = ["KNEE MOTION SUMMARY", ""]
    if summary.left_rom_deg is not None:
        lines.append(f"Left ROM: {summary.left_rom_deg:.0f}°")
    else:
        lines.append("Left ROM: —")
    if summary.right_rom_deg is not None:
        lines.append(f"Right ROM: {summary.right_rom_deg:.0f}°")
    else:
        lines.append("Right ROM: —")
    if summary.rom_asymmetry_pct is not None:
        lines.append(f"ROM asymmetry: {summary.rom_asymmetry_pct:.1f}%")
    if summary.cycle_repeatability and summary.chart_mode == "gait_cycle_pct":
        lines.append(f"Cycle repeatability: {summary.cycle_repeatability}")
    if summary.largest_difference_phase and summary.chart_mode == "gait_cycle_pct":
        lines.append(f"Largest difference: {summary.largest_difference_phase}")
    lines.append("")
    lines.append(summary.plain_language)
    return "\n".join(lines)


def format_diagnostic_report(series: KneeAngleSeries) -> str:
    """Per-video diagnostic report for angle convention and data quality."""
    m = series.metadata
    lines = [
        "KNEE ANGLE DIAGNOSTIC REPORT",
        "=" * 40,
        f"Angle source:          {series.source}",
        f"Angle convention:      {series.angle_definition}",
        f"Convention note:       {ANGLE_CONVENTION_SUMMARY}",
        f"Left valid samples:    {m.get('left_valid', 0)}",
        f"Right valid samples:   {m.get('right_valid', 0)}",
        f"Left min/max/ROM:      {m.get('left_min', 'N/A')} / {m.get('left_max', 'N/A')} / {m.get('left_rom', 'N/A')} deg",
        f"Right min/max/ROM:     {m.get('right_min', 'N/A')} / {m.get('right_max', 'N/A')} / {m.get('right_rom', 'N/A')} deg",
        f"NaN percentage:        {m.get('nan_pct', 0.0):.1f}%",
        f"Largest L frame jump:  {m.get('left_max_jump', 'N/A')} deg",
        f"Largest R frame jump:  {m.get('right_max_jump', 'N/A')} deg",
        f"Graph X range:         {m.get('x_min', 'N/A')} - {m.get('x_max', 'N/A')} s",
        f"Graph Y range:         {m.get('y_min', 'N/A')} - {m.get('y_max', 'N/A')} deg",
    ]
    if m.get("ik_mot_path"):
        lines.append(f"IK MOT path:           {m.get('ik_mot_path')}")
        lines.append(f"IK quality OK:         {m.get('ik_quality_ok', 'N/A')}")
    return "\n".join(lines)
