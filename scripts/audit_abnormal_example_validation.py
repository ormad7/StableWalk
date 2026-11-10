#!/usr/bin/env python3
"""Read-only audit of abnormal_gait_v2: vGRF gaps, COM/BoS frames, ACF peaks.

Does not change analysis formulas or thresholds.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/output/example_v2_verification/abnormal_example_validation.json"


def _latest_poses(filename: str) -> Path | None:
    from stablewalk import config

    matches = []
    for path in config.POSES_DIR.glob("*_poses.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if Path(str(raw.get("source_video") or "")).name == filename:
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _runs(mask: np.ndarray, times: np.ndarray) -> list[dict]:
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append(
            {
                "start_s": float(times[i]),
                "end_s": float(times[j - 1]),
                "frames": int(j - i),
            }
        )
        i = j
    return runs


def _nan_stats(arr: np.ndarray, times: np.ndarray, *, name: str) -> dict:
    finite = np.isfinite(arr)
    zeros = finite & (np.abs(arr) < 1e-12)
    positive = finite & (arr > 1e-12)
    return {
        "name": name,
        "n": int(len(arr)),
        "n_finite": int(finite.sum()),
        "n_nan": int((~finite).sum()),
        "n_zero": int(zeros.sum()),
        "n_positive": int(positive.sum()),
        "finite_frac": float(finite.mean()) if len(arr) else 0.0,
        "first_finite_s": float(times[finite][0]) if finite.any() else None,
        "last_finite_s": float(times[finite][-1]) if finite.any() else None,
        "finite_runs": _runs(finite, times),
        "nan_runs": _runs(~finite, times)[:12],
        "peak": float(np.nanmax(arr)) if finite.any() else None,
    }


def main() -> int:
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.analysis.asymmetry3d.laterality import assess_recording_laterality
    from stablewalk.analysis.biomechanical.orchestrator import run_biomechanical_analysis
    from stablewalk.analysis.biomechanical.walking_speed import (
        _meters_per_normalized_unit,
        _normalized_span_m,
    )
    from stablewalk.analysis.estimated_vgrf_analysis import analyze_estimated_vgrf
    from stablewalk.analysis.foot_contact_analysis import analyze_foot_contact
    from stablewalk.analysis.four_gait_signals import analyze_four_gait_signals
    from stablewalk.analysis.gait_cycle_analysis import MIN_FOOT_VISIBILITY, analyze_gait_cycles
    from stablewalk.analysis.gait_phase_classification import FLIGHT_CLEARANCE_THRESHOLD_M
    from stablewalk.analysis.spectral.report import analyze_spectral
    from stablewalk.config import DEFAULT_SUBJECT_HEIGHT_M
    from stablewalk.coordinates.coordinate_map import TARGET_SKELETON_HEIGHT
    from stablewalk.io.pose_loader import load_pose_sequence
    from stablewalk.pose.tracking_quality import audit_pose_sequence

    poses = _latest_poses("abnormal_gait_v2.mp4")
    report: dict = {
        "example": "abnormal_gait_v2.mp4",
        "pose_json": str(poses) if poses else None,
        "thresholds_untouched": True,
    }
    if poses is None:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("No pose JSON", flush=True)
        return 1

    seq = load_pose_sequence(poses)
    rec = pose_sequence_to_gait_motion(seq, detected_only=True)
    cycles = analyze_gait_cycles(rec)
    contact = analyze_foot_contact(rec, cycles=cycles)
    vgrf = analyze_estimated_vgrf(rec, contact, sequence=seq, body_mass_kg=70.0)
    biomech = run_biomechanical_analysis(rec, seq, cycles=cycles, contact=contact)
    spectral = analyze_spectral(rec, sequence=seq, cycles=cycles)
    four = analyze_four_gait_signals(rec, cycles=cycles)
    laterality = assess_recording_laterality(rec)
    try:
        track = audit_pose_sequence(seq)
    except Exception as exc:
        track = None
        report["tracking_error"] = str(exc)

    times = np.asarray(contact.timestamps, dtype=float)
    left_known = contact.left_contact_known_mask
    right_known = contact.right_contact_known_mask
    left_bin = contact.left_contact_binary.astype(bool)
    right_bin = contact.right_contact_binary.astype(bool)
    left_prob = contact.left_contact_probability
    right_prob = contact.right_contact_probability

    n = len(contact.per_frame)
    confirmed_swing_l = 0
    unknown_off_l = 0
    for i in range(n):
        if not left_bin[i] and left_known[i]:
            confirmed_swing_l += 1
        if not left_bin[i] and not left_known[i]:
            unknown_off_l += 1

    left_force = np.asarray(vgrf.left_vgrf_bw, dtype=float)
    right_force = np.asarray(vgrf.right_vgrf_bw, dtype=float)
    # Allocation checks (do not invent): unknown → NaN; confirmed swing → 0.
    alloc_ok = True
    alloc_failures = []
    for i, frame in enumerate(contact.per_frame):
        if i >= len(left_force):
            break
        lv, rv = left_force[i], right_force[i]
        if not frame.left_contact_known:
            if not math.isnan(float(lv)):
                alloc_ok = False
                alloc_failures.append({"i": i, "side": "left", "got": float(lv), "want": "nan"})
        elif not left_bin[i]:
            if not (math.isfinite(float(lv)) and abs(float(lv)) < 1e-12):
                alloc_ok = False
                alloc_failures.append({"i": i, "side": "left", "got": float(lv), "want": 0.0})
        if not frame.right_contact_known:
            if not math.isnan(float(rv)):
                alloc_ok = False
                alloc_failures.append({"i": i, "side": "right", "got": float(rv), "want": "nan"})
        elif not right_bin[i]:
            if not (math.isfinite(float(rv)) and abs(float(rv)) < 1e-12):
                alloc_ok = False
                alloc_failures.append({"i": i, "side": "right", "got": float(rv), "want": 0.0})

    mpu = _meters_per_normalized_unit(rec, subject_height_m=DEFAULT_SUBJECT_HEIGHT_M)
    span = _normalized_span_m(rec)
    com = biomech.center_of_mass
    com_y = com.positions[:, 1] if com.per_frame else np.array([])
    margin = biomech.stability_margin
    margins = [
        f.stability_margin_m
        for f in (margin.per_frame if margin else [])
        if f.stability_margin_m is not None and math.isfinite(f.stability_margin_m)
    ]

    primary = spectral.univariate_channel_id
    ac = spectral.autocorr.get(primary)
    peaks = []
    if ac is not None:
        for p in ac.peaks or []:
            peaks.append(
                {
                    "lag_s": float(p.lag_s),
                    "rho": float(p.value),
                    "prominence": float(p.prominence),
                    "selected_stride": bool(
                        ac.stride_period_s is not None
                        and abs(float(p.lag_s) - float(ac.stride_period_s)) < 1e-9
                    ),
                    "selected_step": bool(
                        ac.step_period_s is not None
                        and abs(float(p.lag_s) - float(ac.step_period_s)) < 1e-9
                    ),
                    "in_step_window": 0.30 <= float(p.lag_s) <= 0.90,
                    "in_stride_window": 0.55 <= float(p.lag_s) <= 2.20,
                    "exclusive_stride": float(p.lag_s) > 0.90 + 1e-9,
                }
            )

    cycle_list = list(getattr(cycles, "cycles", None) or [])
    hs = [
        {
            "type": e.event_type,
            "side": e.side,
            "time_s": float(e.time_s),
        }
        for e in (getattr(cycles, "events", None) or [])
        if "heel_strike" in str(getattr(e, "event_type", ""))
    ]

    report.update(
        {
            "clip": {
                "n_pose_frames": len(seq.frames),
                "n_detected": sum(1 for f in seq.frames if f.detected),
                "recording_frames": rec.frame_count,
                "duration_s": float(rec.duration_s),
                "fps": float(seq.fps),
                "cam_view": "front (GAVD vascular parkinsonism)",
            },
            "identity": {
                "laterality_status": laterality.status,
                "laterality_facing": laterality.facing,
                "laterality_swapped": bool(laterality.swapped),
                "laterality_consistent": bool(laterality.consistent),
                "laterality_detail": laterality.detail,
                "tracking_laterality_flips": list(getattr(track, "laterality_flips", []) or [])
                if track is not None
                else None,
                "identity_switches": list(getattr(track, "identity_switches", []) or [])
                if track is not None
                else None,
            },
            "contact": {
                "n_frames": n,
                "left_known": int(left_known.sum()),
                "right_known": int(right_known.sum()),
                "left_unknown": int((~left_known).sum()),
                "right_unknown": int((~right_known).sum()),
                "left_contact_known_on": int((left_known & left_bin).sum()),
                "right_contact_known_on": int((right_known & right_bin).sum()),
                "left_confirmed_swing": confirmed_swing_l,
                "left_unknown_off": unknown_off_l,
                "left_prob_nan": int(np.isnan(left_prob).sum()) if len(left_prob) else 0,
                "right_prob_nan": int(np.isnan(right_prob).sum()) if len(right_prob) else 0,
                "flight_clearance_m": float(FLIGHT_CLEARANCE_THRESHOLD_M),
                "min_foot_visibility": float(MIN_FOOT_VISIBILITY),
                "unknown_left_runs": _runs(~left_known, times)[:16],
                "unknown_right_runs": _runs(~right_known, times)[:16],
                "warnings": list(contact.warnings or []),
            },
            "vgrf": {
                "available": bool(vgrf.available),
                "notes": list(vgrf.notes or []),
                "left": _nan_stats(left_force, np.asarray(vgrf.timestamps, dtype=float), name="left_bw"),
                "right": _nan_stats(right_force, np.asarray(vgrf.timestamps, dtype=float), name="right_bw"),
                "allocation_preserves_nan_gaps": alloc_ok,
                "allocation_failures": alloc_failures[:20],
                "metrics": vgrf.metrics.to_dict() if vgrf.metrics else {},
            },
            "com_bos": {
                "skeleton_frame": "hip-centered reconstructed XYZ, +Y up",
                "target_skeleton_height_units": float(TARGET_SKELETON_HEIGHT),
                "default_subject_height_m": float(DEFAULT_SUBJECT_HEIGHT_M),
                "measured_normalized_span": float(span),
                "meters_per_normalized_unit": float(mpu),
                "physical_scale_calibrated": False,
                "com_y_units": "pelvis-relative hip-centered Y; labeled approx. BH because span is normalized ~1.0",
                "com_n": int(len(com_y)),
                "com_y_min": float(np.nanmin(com_y)) if com_y.size else None,
                "com_y_max": float(np.nanmax(com_y)) if com_y.size else None,
                "com_y_mean": float(np.nanmean(com_y)) if com_y.size else None,
                "com_y_std": float(np.nanstd(com_y)) if com_y.size else None,
                "illustrative_bh_band": [-0.03, 0.03],
                "balance_unit": "m",
                "balance_frame": "pelvis-centred body-normalized XZ; BoS mean Z aligned to COM Z; metres",
                "meters_per_unit_applied_to_margin": True,
                "n_margin_finite": len(margins),
                "margin_min_m": float(min(margins)) if margins else None,
                "margin_max_m": float(max(margins)) if margins else None,
                "margin_mean_m": float(sum(margins) / len(margins)) if margins else None,
                "margin_validity": getattr(margin, "validity", None),
                "margin_warnings": list(getattr(margin, "warnings", None) or []),
                "compatible_frames": True,
                "compatible_note": (
                    "COM Y chart stays in body-normalized hip-centered units. "
                    "COM–BoS signed distance multiplies XZ by meters_per_unit "
                    f"({mpu:.4f} m/unit from stature {DEFAULT_SUBJECT_HEIGHT_M:.2f} m / span {span:.4f}). "
                    "Not a tape-calibrated physical scale."
                ),
            },
            "spectral": {
                "primary_channel": primary,
                "reliability": None if ac is None else ac.reliability,
                "reliability_label": None if ac is None else ac.reliability_label,
                "quality_status": None if ac is None else ac.quality_status,
                "insufficient_reason": None if ac is None else ac.insufficient_reason,
                "peak_selection": None if ac is None else ac.peak_selection,
                "stride_assignment_note": None if ac is None else ac.stride_assignment_note,
                "accepted_stride_s": None if ac is None else ac.stride_period_s,
                "accepted_step_s": None if ac is None else ac.step_period_s,
                "accepted_gait_period_s": None if ac is None else ac.gait_period_s,
                "cadence_steps_per_min": None if ac is None else ac.cadence_steps_per_min,
                "cadence_assumes_two_steps": None
                if ac is None
                else ac.cadence_assumes_two_steps_per_stride,
                "n_stride_cycles_duration_over_period": None
                if ac is None
                else ac.n_stride_cycles,
                "selected_peak_prominence": next(
                    (p["prominence"] for p in peaks if p["selected_stride"]), None
                ),
                "selected_peak_rho": next((p["rho"] for p in peaks if p["selected_stride"]), None),
                "candidates": peaks,
                "notes": list(ac.notes or []) if ac is not None else [],
                "acceptance": {
                    "min_rho": 0.05,
                    "must_be_positive_local_max": True,
                    "stride_window_s": [0.55, 2.20],
                    "step_window_s": [0.30, 0.90],
                    "exclusive_stride_prefers_lag_gt_step_hi": True,
                    "min_cycles": 2,
                    "reliability_moderate_if": "score >= 0.65 and stride accepted",
                    "reliability_formula": (
                        "0.35*clip(rho)+0.25*clip(prominence)+0.25*min(n_cycles/6,1)"
                        "+0.15*(1-missing_frac)"
                    ),
                },
            },
            "gait_estimates": {
                "computed_channel_stride_s": None if ac is None else ac.stride_period_s,
                "computed_channel_step_s": None if ac is None else ac.step_period_s,
                "session_acf_stride_s": four.gait_period_s,
                "session_reason": four.gait_period_reason,
                "event_stride_s": four.event_stride_period_s,
                "event_stride_n": four.event_stride_n,
                "acf_event_note": four.acf_event_note,
                "detected_cycles": len(cycle_list),
                "valid_cycles": sum(
                    1 for c in cycle_list if getattr(c, "duration_s", 0) and c.duration_s > 1e-6
                ),
                "heel_strikes": hs,
                "event_cadence": getattr(getattr(cycles, "metrics", None), "cadence_steps_per_min", None),
                "kind_note": (
                    "Channel ACF stride/step are computed candidates on the primary series. "
                    "Session ACF is the median of contributing thigh/shank strides. "
                    "Event HS–HS is independent and is not substituted into ACF."
                ),
            },
        }
    )

    from stablewalk.analysis.foot_contact_analysis import iter_contact_frame_evidence
    from stablewalk.analysis.result_validity import build_result_validity
    from stablewalk.io.contact_evidence_export import (
        contact_evidence_summary,
        write_contact_frame_evidence,
    )

    validity = build_result_validity(
        biomechanical=biomech,
        contact=contact,
        cycles=cycles,
        spectral=spectral,
        duration_s=float(rec.duration_s),
        n_pose_frames=len(seq.frames),
        coordinate_system=str(getattr(rec, "coordinate_system", "") or ""),
    )
    report["result_validity"] = validity.to_dict()
    evidence_csv = OUT.parent / "abnormal_contact_frame_evidence.csv"
    write_contact_frame_evidence(contact, evidence_csv)
    evidence_rows = iter_contact_frame_evidence(contact)
    evidence_summary = contact_evidence_summary(evidence_rows)
    report["contact"]["unknown_reason_counts"] = evidence_summary["by_side"]
    report["contact_evidence_csv"] = str(evidence_csv)
    report["contact_evidence_json"] = str(evidence_csv.with_suffix(".json"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "pose": report["pose_json"],
        "identity": report["identity"],
        "result_validity": {
            "headline": validity.headline_lines(),
            "pose": validity.pose_coverage.status,
            "contact": validity.contact_coverage.status,
            "spectral": validity.spectral.status,
            "biomech": validity.biomechanical.status,
            "cycles": validity.cycle_detection.extras,
        },
        "contact_unknown": {
            "left": report["contact"]["left_unknown"],
            "right": report["contact"]["right_unknown"],
            "left_unknown_off": report["contact"]["left_unknown_off"],
            "left_confirmed_swing": report["contact"]["left_confirmed_swing"],
            "unknown_reason_counts": report["contact"]["unknown_reason_counts"],
        },
        "vgrf_left": report["vgrf"]["left"],
        "vgrf_right": report["vgrf"]["right"],
        "alloc_ok": report["vgrf"]["allocation_preserves_nan_gaps"],
        "com": {
            "y_min": report["com_bos"]["com_y_min"],
            "y_max": report["com_bos"]["com_y_max"],
            "mpu": report["com_bos"]["meters_per_normalized_unit"],
            "span": report["com_bos"]["measured_normalized_span"],
            "validity": report["com_bos"]["margin_validity"],
            "n_margin": report["com_bos"]["n_margin_finite"],
        },
        "spectral": {
            "channel": primary,
            "label": report["spectral"]["reliability_label"],
            "rel": report["spectral"]["reliability"],
            "stride": report["spectral"]["accepted_stride_s"],
            "step": report["spectral"]["accepted_step_s"],
            "prominence": report["spectral"]["selected_peak_prominence"],
            "rho": report["spectral"]["selected_peak_rho"],
            "note": report["spectral"]["stride_assignment_note"],
        },
        "session": {
            "acf": four.gait_period_s,
            "event": four.event_stride_period_s,
            "cycles": report["gait_estimates"]["detected_cycles"],
        },
    }, indent=2, default=str), flush=True)
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
