"""Per-frame gait contact debug export (CSV + validation report)."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from stablewalk.analysis.gait_cycle_analysis import (
    GaitCycleAnalysisResult,
    _joint_position,
    _lowest_clearance,
    calibrate_contact_thresholds,
    classify_gait_phase,
    estimate_leg_length_m,
    raw_contact_desire,
    resolve_vertical_axis,
)
from stablewalk.analysis.ground_reference import (
    estimate_ground_plane,
    vertical_coordinate,
)
from stablewalk.models.gait_motion import GaitMotionRecording

logger = logging.getLogger(__name__)

DEBUG_CSV_COLUMNS = [
    "frame_index",
    "timestamp_s",
    "vertical_axis",
    "floor_y_m",
    "left_heel_y_m",
    "right_heel_y_m",
    "left_toe_y_m",
    "right_toe_y_m",
    "left_ankle_y_m",
    "right_ankle_y_m",
    "left_heel_clearance_m",
    "right_heel_clearance_m",
    "left_toe_clearance_m",
    "right_toe_clearance_m",
    "left_heel_clearance_cm",
    "right_heel_clearance_cm",
    "left_heel_visibility",
    "right_heel_visibility",
    "left_toe_visibility",
    "right_toe_visibility",
    "left_ankle_visibility",
    "right_ankle_visibility",
    "left_vertical_velocity_m_s",
    "right_vertical_velocity_m_s",
    "raw_left_contact",
    "raw_right_contact",
    "smoothed_left_contact",
    "smoothed_right_contact",
    "gait_phase",
    "leg_length_m",
    "entry_clearance_m",
    "exit_clearance_m",
    "entry_clearance_normalized",
]


def build_contact_debug_rows(
    recording: GaitMotionRecording,
    result: GaitCycleAnalysisResult,
) -> list[dict[str, Any]]:
    """Build per-frame debug rows aligned with gait cycle analysis output."""
    plane = result.ground_plane or estimate_ground_plane(
        recording, float(max(recording.frame_count - 1, 0))
    )
    axis = resolve_vertical_axis(plane)
    floor_y = plane.floor_y if plane else None
    thresholds = result.contact_thresholds
    by_frame = {s.frame_index: s for s in result.per_frame}

    rows: list[dict[str, Any]] = []
    left_state = 0
    right_state = 0

    for index in range(recording.frame_count):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        state = by_frame.get(index)

        def _vis(jid: str) -> float | None:
            vmap = snap.metadata.get("landmark_visibility")
            if isinstance(vmap, dict) and jid in vmap:
                return float(vmap[jid])
            _, v = _joint_position(snap, jid)
            return v if v > 0 else None

        def _y(jid: str) -> float | None:
            pos, _ = _joint_position(snap, jid)
            return vertical_coordinate(pos, axis=axis) if pos else None

        raw_l = raw_r = None
        if state is not None and thresholds is not None:
            if state.left.foot_clearance_m is not None and state.left.visibility >= 0.35:
                raw_l = raw_contact_desire(
                    state.left, currently_in_contact=bool(left_state), thresholds=thresholds
                )
                if raw_l is not None:
                    left_state = int(raw_l)
            if state.right.foot_clearance_m is not None and state.right.visibility >= 0.35:
                raw_r = raw_contact_desire(
                    state.right, currently_in_contact=bool(right_state), thresholds=thresholds
                )
                if raw_r is not None:
                    right_state = int(raw_r)

        l_vel = None
        r_vel = None
        if state is not None:
            vels_l = [
                v
                for v in (
                    state.left.heel_velocity_m_s,
                    state.left.toe_velocity_m_s,
                    state.left.ankle_velocity_m_s,
                )
                if v is not None
            ]
            vels_r = [
                v
                for v in (
                    state.right.heel_velocity_m_s,
                    state.right.toe_velocity_m_s,
                    state.right.ankle_velocity_m_s,
                )
                if v is not None
            ]
            if vels_l:
                l_vel = max(abs(v) for v in vels_l)
            if vels_r:
                r_vel = max(abs(v) for v in vels_r)

        lh_c, lt_c, la_c, lf_c = (
            (state.left.heel_clearance_m, state.left.toe_clearance_m, state.left.ankle_clearance_m, state.left.foot_clearance_m)
            if state
            else (None, None, None, None)
        )
        rh_c, rt_c, ra_c, rf_c = (
            (state.right.heel_clearance_m, state.right.toe_clearance_m, state.right.ankle_clearance_m, state.right.foot_clearance_m)
            if state
            else (None, None, None, None)
        )

        rows.append(
            {
                "frame_index": index,
                "timestamp_s": round(snap.time_s, 4),
                "vertical_axis": axis,
                "floor_y_m": round(floor_y, 5) if floor_y is not None else "",
                "left_heel_y_m": _y("left_heel"),
                "right_heel_y_m": _y("right_heel"),
                "left_toe_y_m": _y("left_toe"),
                "right_toe_y_m": _y("right_toe"),
                "left_ankle_y_m": _y("left_ankle"),
                "right_ankle_y_m": _y("right_ankle"),
                "left_heel_clearance_m": lh_c,
                "right_heel_clearance_m": rh_c,
                "left_toe_clearance_m": lt_c,
                "right_toe_clearance_m": rt_c,
                "left_foot_clearance_m": lf_c,
                "right_foot_clearance_m": rf_c,
                "left_heel_clearance_cm": round(lh_c * 100, 2) if lh_c is not None else "",
                "right_heel_clearance_cm": round(rh_c * 100, 2) if rh_c is not None else "",
                "left_heel_visibility": _vis("left_heel"),
                "right_heel_visibility": _vis("right_heel"),
                "left_toe_visibility": _vis("left_toe"),
                "right_toe_visibility": _vis("right_toe"),
                "left_ankle_visibility": _vis("left_ankle"),
                "right_ankle_visibility": _vis("right_ankle"),
                "left_vertical_velocity_m_s": l_vel,
                "right_vertical_velocity_m_s": r_vel,
                "raw_left_contact": raw_l,
                "raw_right_contact": raw_r,
                "smoothed_left_contact": state.left_contact if state else "",
                "smoothed_right_contact": state.right_contact if state else "",
                "gait_phase": state.phase if state else "",
                "leg_length_m": thresholds.leg_length_m if thresholds else "",
                "entry_clearance_m": thresholds.entry_clearance_m if thresholds else "",
                "exit_clearance_m": thresholds.exit_clearance_m if thresholds else "",
                "entry_clearance_normalized": thresholds.entry_normalized if thresholds else "",
            }
        )
    return rows


def export_contact_debug_csv(
    recording: GaitMotionRecording,
    result: GaitCycleAnalysisResult,
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_contact_debug_rows(recording, result)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DEBUG_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_gait_validation_report(
    name: str,
    result: GaitCycleAnalysisResult,
    recording: GaitMotionRecording,
) -> None:
    """Print summary metrics and validation warnings for one video."""
    m = result.metrics
    n = len(result.per_frame)
    left_c = sum(s.left_contact for s in result.per_frame)
    right_c = sum(s.right_contact for s in result.per_frame)
    double = sum(1 for s in result.per_frame if s.left_contact and s.right_contact)
    uncertain = sum(1 for s in result.per_frame if s.phase in ("UNCERTAIN", "FLIGHT", "FLIGHT_OR_UNCERTAIN"))

    print(f"\n{'=' * 60}")
    print(f"GAIT CONTACT REPORT — {name}")
    print(f"{'=' * 60}")
    if result.ground_plane:
        print(
            f"Floor Y: {result.ground_plane.floor_y:.4f} m  "
            f"(body span {result.ground_plane.body_vertical_span:.3f} m, "
            f"scale={result.ground_plane.scale_mode})"
        )
    if result.contact_thresholds:
        t = result.contact_thresholds
        print(
            f"Thresholds — entry {t.entry_clearance_m*100:.1f} cm "
            f"({t.entry_normalized:.2f}× leg), exit {t.exit_clearance_m*100:.1f} cm "
            f"({t.exit_normalized:.2f}× leg)"
        )
        print(
            f"Clearance distribution p10/p50/p90: "
            f"{t.clearance_p10_m*100:.1f} / {t.clearance_p50_m*100:.1f} / "
            f"{t.clearance_p90_m*100:.1f} cm"
        )
    print(f"Frames analyzed:              {n}")
    print(f"Frames with left contact:       {left_c}")
    print(f"Frames with right contact:      {right_c}")
    print(f"Frames with double support:     {double}")
    print(f"Frames classified uncertain:    {uncertain}")
    print(f"Left heel strikes:              {m.left_heel_strike_count}")
    print(f"Right heel strikes:             {m.right_heel_strike_count}")
    print(f"Left toe offs:                  {m.left_toe_off_count}")
    print(f"Right toe offs:                 {m.right_toe_off_count}")
    print(f"Complete gait cycles:           {m.gait_cycle_count}")
    cadence = f"{m.cadence_steps_per_min:.1f}" if m.cadence_steps_per_min else "—"
    print(f"Cadence (steps/min):            {cadence}")
    l_stance = f"{m.left_stance_time_s:.3f}s" if m.left_stance_time_s is not None else "—"
    r_stance = f"{m.right_stance_time_s:.3f}s" if m.right_stance_time_s is not None else "—"
    print(f"Left stance duration (mean):    {l_stance}")
    print(f"Right stance duration (mean):   {r_stance}")
    sym = (
        f"{m.left_right_stance_symmetry:.0%}"
        if m.left_right_stance_symmetry is not None
        else "—"
    )
    print(f"Stance symmetry:                {sym}")
    print(
        f"Contact confidence:             {m.contact_confidence:.2f} ({m.confidence_tier})"
    )
    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  • {w}")
    print(f"{'=' * 60}\n")
