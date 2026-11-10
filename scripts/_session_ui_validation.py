#!/usr/bin/env python3
"""Headless verification of the current UI-only session (no analysis retuning).

Exports JSON/CSV for Abnormal / Normal / Fast Walking into validation_outputs/
and records quality-layer, playhead-marker, GRF/Contact note, and layout checks.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "validation_outputs"
EXAMPLES = (
    ("abnormal", "Abnormal Gait", "abnormal_gait_v2.mp4"),
    ("normal", "Normal Gait", "normal_gait_v2.mp4"),
    ("athletic", "Fast Walking", "fast_walking_v2.mp4"),
)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    return str(obj)


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


def _marker_xy(ax):
    artists = getattr(ax, "_stablewalk_playhead_artists", None)
    if not artists or len(artists) < 4:
        return None
    cap = artists[3]
    if cap is None or not cap.get_visible():
        return None
    off = np.asarray(cap.get_offsets(), dtype=float)
    if off.size < 2:
        return None
    return float(off[0, 0]), float(off[0, 1])


def _check_playhead_on_series(ax, time_s: float, series_y: float | None) -> dict:
    xy = _marker_xy(ax)
    if series_y is None:
        return {
            "time_s": time_s,
            "series_y": None,
            "marker_visible": xy is not None,
            "ok": xy is None,
            "detail": "missing sample must hide marker",
        }
    if xy is None:
        return {
            "time_s": time_s,
            "series_y": series_y,
            "marker_visible": False,
            "ok": False,
            "detail": "finite sample but marker hidden",
        }
    mx, my = xy
    ok = abs(mx - time_s) < 1e-6 and abs(my - series_y) < 1e-6
    return {
        "time_s": time_s,
        "series_y": series_y,
        "marker_x": mx,
        "marker_y": my,
        "ok": ok,
        "detail": "marker on displayed sample" if ok else "marker off series",
    }


def _cycle_side_counts(cycles) -> dict:
    clist = list(getattr(cycles, "cycles", None) or [])
    n_left = sum(
        1
        for c in clist
        if str(getattr(c, "boundary_side", "")).lower().startswith("left")
    )
    n_right = sum(
        1
        for c in clist
        if str(getattr(c, "boundary_side", "")).lower().startswith("right")
    )
    return {"n_cycles": len(clist), "n_left": n_left, "n_right": n_right}


def analyze_example(key: str, label: str, filename: str) -> dict:
    from matplotlib.figure import Figure

    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.analysis.biomechanical.orchestrator import run_biomechanical_analysis
    from stablewalk.analysis.estimated_vgrf_analysis import analyze_estimated_vgrf
    from stablewalk.analysis.foot_contact_analysis import analyze_foot_contact
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
    from stablewalk.analysis.result_validity import build_result_validity
    from stablewalk.analysis.spectral.report import analyze_spectral
    from stablewalk.data_export import export_analysis
    from stablewalk.io.contact_evidence_export import (
        contact_evidence_summary,
        write_contact_frame_evidence,
    )
    from stablewalk.io.pose_loader import load_pose_sequence
    from stablewalk.ui.viewers.biomechanics_charts import (
        BIOMECH_SERIES_COM,
        BIOMECH_SERIES_STABILITY,
        draw_biomechanics_dashboard,
    )
    from stablewalk.ui.viewers.chart_playhead import sample_displayed_series
    from stablewalk.ui.viewers.contact_gait_viewer import (
        CONTACT_PANEL_CONTACT,
        CONTACT_PANEL_VGRF,
        contact_panel_note,
        draw_contact_gait_dashboard,
    )
    from stablewalk.ui.playback_state import PlaybackState, sample_sync_frame_indices
    from stablewalk.ui.viewers.chart_playhead import (
        PLAYHEAD_VALUE_SOURCE_ATTR,
        playhead_from_playback as _cursor,
    )

    dest = OUT / key
    dest.mkdir(parents=True, exist_ok=True)
    poses = _latest_poses(filename)
    rec: dict = {
        "key": key,
        "display_name": label,
        "filename": filename,
        "pose_json": str(poses) if poses else None,
    }
    if poses is None:
        rec["error"] = "No pose JSON found for this example"
        return rec

    t0 = time.perf_counter()
    seq = load_pose_sequence(poses)
    recording = pose_sequence_to_gait_motion(seq, detected_only=False)
    cycles = analyze_gait_cycles(recording)
    contact = analyze_foot_contact(recording, cycles=cycles)
    vgrf = analyze_estimated_vgrf(recording, contact, sequence=seq, body_mass_kg=70.0)
    biomech = run_biomechanical_analysis(recording, seq, cycles=cycles, contact=contact)
    spectral = analyze_spectral(recording, sequence=seq, cycles=cycles)
    validity = build_result_validity(
        biomechanical=biomech,
        contact=contact,
        cycles=cycles,
        spectral=spectral,
        duration_s=float(recording.duration_s),
        n_pose_frames=len(seq.frames),
        coordinate_system=str(getattr(recording, "coordinate_system", "") or ""),
    )
    rec["analysis_s"] = round(time.perf_counter() - t0, 2)

    json_path = export_analysis(seq, dest / f"{key}_analysis.json")
    csv_path = export_analysis(seq, dest / f"{key}_analysis.csv")
    evidence_csv = write_contact_frame_evidence(
        contact, dest / f"{key}_contact_evidence.csv", also_json=True
    )
    evidence_rows = []
    evidence_json = evidence_csv.with_suffix(".json")
    if evidence_json.is_file():
        evidence_rows = json.loads(evidence_json.read_text(encoding="utf-8")).get(
            "rows", []
        )
    rec["exports"] = {
        "analysis_json": str(json_path),
        "analysis_csv": str(csv_path),
        "contact_evidence_csv": str(evidence_csv),
        "contact_evidence_json": str(evidence_json),
        "analysis_json_bytes": json_path.stat().st_size,
        "analysis_csv_bytes": csv_path.stat().st_size,
    }

    left_force = np.asarray(vgrf.left_vgrf_bw, dtype=float)
    right_force = np.asarray(vgrf.right_vgrf_bw, dtype=float)
    gap_ok = True
    gap_failures = 0
    for i, frame in enumerate(contact.per_frame):
        if i >= len(left_force):
            break
        if not frame.left_contact_known and not math.isnan(float(left_force[i])):
            gap_ok = False
            gap_failures += 1
        if not frame.right_contact_known and not math.isnan(float(right_force[i])):
            gap_ok = False
            gap_failures += 1

    rec["clip"] = {
        "n_pose_frames": len(seq.frames),
        "n_detected": sum(1 for f in seq.frames if f.detected),
        "recording_frames": recording.frame_count,
        "duration_s": float(recording.duration_s),
        "fps": float(seq.fps),
    }
    rec["cycles"] = _cycle_side_counts(cycles)
    rec["contact_coverage"] = contact_evidence_summary(evidence_rows) if evidence_rows else {}
    rec["validity"] = validity.to_dict()
    rec["headline"] = validity.headline_lines()
    rec["vgrf_unknown_are_nan"] = {"ok": gap_ok, "failures": gap_failures}

    # Playhead: canonical 1-based display frame vs committed 0-based index.
    n = max(1, recording.frame_count)
    fps = float(seq.fps) or 25.0
    sync_idxs = sample_sync_frame_indices(n)
    sync_checks = []
    for idx in sync_idxs:
        t_s = idx / fps
        state = PlaybackState(
            current_frame_index=idx,
            current_time_seconds=t_s,
            total_frames=n,
            fps=fps,
            duration_seconds=float(recording.duration_s),
        )
        cursor = _cursor(state)
        sync_checks.append(
            {
                "list_index": idx,
                "display_frame": state.display_frame_number,
                "cursor_frame": cursor.frame_index,
                "time_s": t_s,
                "ok": cursor.frame_index == idx + 1 and cursor.list_pos == idx,
            }
        )
    rec["playback_frame_labels"] = sync_checks

    # COM / Balance / GRF / Contact markers at a mid-clip timestamp.
    mid = sync_idxs[len(sync_idxs) // 2]
    t_mid = mid / fps
    marker_checks = []
    fig = Figure(figsize=(8.0, 4.2), dpi=100)
    draw_biomechanics_dashboard(
        fig, biomech, playhead_time_s=t_mid, playhead_frame=mid + 1, series=BIOMECH_SERIES_COM
    )
    ax = fig.axes[0] if fig.axes else None
    if ax is not None:
        src = getattr(ax, PLAYHEAD_VALUE_SOURCE_ATTR, None)
        series_y = None
        if isinstance(src, (tuple, list)) and len(src) >= 2:
            series_y = sample_displayed_series(src[0], src[1], t_mid)
        marker_checks.append({"panel": "COM", **_check_playhead_on_series(ax, t_mid, series_y)})
    fig.clear()

    draw_biomechanics_dashboard(
        fig,
        biomech,
        playhead_time_s=t_mid,
        playhead_frame=mid + 1,
        series=BIOMECH_SERIES_STABILITY,
    )
    ax = fig.axes[0] if fig.axes else None
    if ax is not None:
        from stablewalk.ui.viewers.chart_playhead import PLAYHEAD_VALUE_SOURCE_ATTR

        src = getattr(ax, PLAYHEAD_VALUE_SOURCE_ATTR, None)
        if src is None:
            marker_checks.append(
                {
                    "panel": "Balance",
                    "ok": True,
                    "detail": "no playhead series (unavailable or empty-state)",
                    "empty_title": ax.get_title(),
                }
            )
        else:
            series_y = sample_displayed_series(src[0], src[1], t_mid)
            marker_checks.append(
                {"panel": "Balance", **_check_playhead_on_series(ax, t_mid, series_y)}
            )
    fig.clear()

    draw_contact_gait_dashboard(
        fig,
        contact,
        vgrf,
        panel=CONTACT_PANEL_VGRF,
        playhead_time_s=t_mid,
        playhead_frame=mid + 1,
    )
    grf_note = contact_panel_note(CONTACT_PANEL_VGRF, contact=contact, vgrf=vgrf)
    fig_note = str(getattr(fig, "_sw_panel_note", "") or "")
    ax = fig.axes[0] if fig.axes else None
    figure_text = " ".join(t.get_text() or "" for t in fig.texts)
    if ax is not None:
        src = getattr(ax, PLAYHEAD_VALUE_SOURCE_ATTR, None)
        series_y = (
            sample_displayed_series(src[0], src[1], t_mid)
            if isinstance(src, (tuple, list)) and len(src) >= 2
            else None
        )
        marker_checks.append({"panel": "GRF", **_check_playhead_on_series(ax, t_mid, series_y)})
    rec["grf_note"] = {
        "text": grf_note,
        "fig_note": fig_note,
        "has_peak": "peak" in grf_note.lower() or "BW" in grf_note,
        "on_axes_peak": "L peak" in figure_text,
    }
    fig.clear()

    draw_contact_gait_dashboard(
        fig,
        contact,
        vgrf,
        panel=CONTACT_PANEL_CONTACT,
        playhead_time_s=t_mid,
        playhead_frame=mid + 1,
    )
    contact_note = contact_panel_note(CONTACT_PANEL_CONTACT, contact=contact, vgrf=vgrf)
    fig_note = str(getattr(fig, "_sw_panel_note", "") or "")
    figure_text = " ".join(t.get_text() or "" for t in fig.texts)
    axes_text = " ".join(t.get_text() or "" for t in (fig.axes[0].texts if fig.axes else []))
    rec["contact_note"] = {
        "text": contact_note,
        "fig_note": fig_note,
        "leaks_grf_peak": (
            "L peak" in fig_note
            or "L peak" in figure_text
            or "L peak" in axes_text
            or "BW" in figure_text
        ),
        "matches_panel_note": fig_note == contact_note,
    }
    rec["marker_checks"] = marker_checks
    rec["ok_markers"] = all(item.get("ok", False) for item in marker_checks)
    rec["ok_contact_isolation"] = not rec["contact_note"]["leaks_grf_peak"]
    rec["ok_playback_labels"] = all(item.get("ok", False) for item in sync_checks)

    summary_path = dest / f"{key}_validity.json"
    summary_path.write_text(
        json.dumps(_jsonable({"headline": rec["headline"], "validity": rec["validity"], "cycles": rec["cycles"], "contact_coverage": rec["contact_coverage"]}), indent=2),
        encoding="utf-8",
    )
    rec["exports"]["validity_json"] = str(summary_path)
    return rec


def layout_checks() -> dict:
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.dof_trajectory_3d import (
        _overview_dof_figure_margins,
        _single_dof_figure_margins,
    )
    from stablewalk.ui.viewers.four_gait_charts import layout_four_gait_figure
    from stablewalk.opensim_ik_status import (
        ik_mot_unavailable_reason,
        resolve_opensim_ik_status,
        resolve_opensim_pipeline_status,
        format_opensim_status_pair,
    )

    ov = _overview_dof_figure_margins(3.0, 3.0, dpi=100.0)
    single = _single_dof_figure_margins(6.5, 5.2, dpi=100.0)
    fig = Figure(figsize=(7.2, 5.0), dpi=100)
    axes = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
    layout_four_gait_figure(fig)
    titles_ok = all(ax.get_position().y1 <= 0.92 and ax.get_position().y0 >= 0.12 for ax in axes)

    export_only = resolve_opensim_ik_status(export_completed=True, mapped_trc_ready=True)
    pipeline = resolve_opensim_pipeline_status(export_completed=True, mapped_trc_ready=True)
    pair = format_opensim_status_pair(pipeline, export_only)
    leftover = ik_mot_unavailable_reason(export_only, ik_output_on_disk=True)
    return {
        "overview_margins_span": ov[3] - ov[1],
        "overview_not_flush": ov[3] <= 0.93 and (ov[3] - ov[1]) > 0.65,
        "motion_area": (single[2] - single[0]) * (single[3] - single[1]),
        "four_motion_title_room": titles_ok,
        "opensim_pair_short": pair.short_label,
        "ik_leftover_reason": leftover,
        "export_is_not_ik": "IK" in leftover or "Inverse Kinematics" in leftover,
    }


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "examples": [],
        "layout": layout_checks(),
        "gui": {"attempted": False, "available": False, "detail": "GUI driver is a separate process"},
    }
    failures = []
    for key, label, filename in EXAMPLES:
        print(f"Analyzing {label}...", flush=True)
        rec = analyze_example(key, label, filename)
        report["examples"].append(rec)
        if rec.get("error"):
            failures.append(f"{key}: {rec['error']}")
            continue
        if not rec.get("ok_markers", True):
            failures.append(f"{key}: playhead marker off series")
        if not rec.get("ok_contact_isolation", True):
            failures.append(f"{key}: GRF peak note leaked onto Contact")
        if not rec.get("ok_playback_labels", True):
            failures.append(f"{key}: display frame labels disagree")
        if not rec.get("vgrf_unknown_are_nan", {}).get("ok", True):
            failures.append(f"{key}: unknown contact filled vGRF instead of NaN")
        print(
            f"  frames={rec['clip']['recording_frames']} "
            f"cycles L/R={rec['cycles']['n_left']}/{rec['cycles']['n_right']} "
            f"spectral={rec['validity']['spectral']['status']} "
            f"biomech={rec['validity']['biomechanical']['status']}",
            flush=True,
        )
    report["failures"] = failures
    (OUT / "session_validation.json").write_text(
        json.dumps(_jsonable(report), indent=2), encoding="utf-8"
    )
    print(f"Wrote {OUT / 'session_validation.json'}", flush=True)
    if failures:
        print("FAILURES:", *failures, sep="\n - ", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
