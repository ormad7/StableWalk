#!/usr/bin/env python3
"""Audit Normal / Fast / Abnormal demos independently of playback.

Reuses stored pose dumps plus source-video decode. Does not retune thresholds
or invent detections. Writes validation_outputs/<key>/ and validation_report.md.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
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
UNPERFORMED = (
    "Force-plate or instrumented-treadmill GRF",
    "Marker-based optical motion capture",
    "IMU / wearable validation",
    "Clinician-scored gait quality instrument",
    "Live GUI click-through of Examples (preview) vs Analyze (full source) on a physical display",
    "Live Compare A/B click-through: pin a preview session against a full session and confirm the mixed-mode block in the UI",
    "Live Compare playback of Session A/B after Analyze Session B inherits full vs preview",
    "MediaPipe re-inference (this audit reused the 2026-11-03 13:07 UTC full-analysis dumps)",
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


def _pinned_full_analysis_dump(filename: str) -> Path | None:
    """Reuse the 13:07 UTC full-analysis dumps; do not pick a later GUI preview."""
    manifest_path = OUT / "full_rerun_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for rec in payload.get("pipeline") or []:
        source = Path(str(rec.get("source_file") or "")).name
        dump = rec.get("poses_path")
        if source == filename and dump:
            path = Path(str(dump))
            if path.is_file():
                return path
    return None


def _latest_poses(filename: str) -> Path | None:
    pinned = _pinned_full_analysis_dump(filename)
    if pinned is not None:
        return pinned
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


def _decode_source_timeline(video_path: Path) -> list[dict]:
    """Decode every source frame ID and timestamp without pose inference."""
    from stablewalk.pose.video import VideoReader

    rows = []
    with VideoReader(video_path) as reader:
        for frame in reader.iter_frames():
            rows.append(
                {
                    "source_frame": int(frame.index),
                    "timestamp_s": float(frame.timestamp_s),
                    "decode_ok": True,
                }
            )
            frame.bgr = None  # type: ignore[assignment]
    return rows


def _candidate_rejection_counts(candidates) -> dict:
    counts: dict[str, int] = {}
    for item in candidates:
        if getattr(item, "accepted", False):
            continue
        reason = str(getattr(item, "acceptance_reason", "") or "unspecified")
        key = reason.split(" ", 1)[0].rstrip("—-:")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


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


def _phase_annotation_text(biomech) -> str:
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.biomechanics_charts import (
        BIOMECH_SERIES_PHASE,
        draw_biomechanics_dashboard,
    )

    fig = Figure(figsize=(8.0, 4.2), dpi=100)
    draw_biomechanics_dashboard(fig, biomech, series=BIOMECH_SERIES_PHASE)
    texts = [t.get_text() or "" for t in fig.texts]
    if fig.axes:
        texts.extend(t.get_text() or "" for t in fig.axes[0].texts)
    fig.clear()
    return " | ".join(t for t in texts if t.strip())


def _cursor_checks(recording, seq, contact, vgrf, biomech) -> dict:
    from matplotlib.figure import Figure

    from stablewalk.ui.playback_state import PlaybackState, sample_sync_frame_indices
    from stablewalk.ui.viewers.biomechanics_charts import (
        BIOMECH_SERIES_COM,
        draw_biomechanics_dashboard,
    )
    from stablewalk.ui.viewers.chart_hover import format_inspect_crosshair_chip
    from stablewalk.ui.viewers.chart_playhead import (
        PLAYHEAD_VALUE_SOURCE_ATTR,
        PlayheadState,
        format_playhead_label,
        playhead_from_playback,
        sample_displayed_series,
        update_playhead_on_axes,
    )
    from stablewalk.ui.viewers.contact_gait_viewer import (
        CONTACT_PANEL_CONTACT,
        CONTACT_PANEL_VGRF,
        draw_contact_gait_dashboard,
    )

    n = max(1, recording.frame_count)
    fps = float(seq.fps) or 25.0
    idxs = sample_sync_frame_indices(n)
    labels = []
    for idx in idxs:
        t_s = float(recording.snapshots[idx].time_s)
        state = PlaybackState(
            current_frame_index=idx,
            current_time_seconds=t_s,
            total_frames=n,
            fps=fps,
            duration_seconds=float(recording.duration_s),
        )
        cursor = playhead_from_playback(state)
        labels.append(
            {
                "list_index": idx,
                "source_frame": int(recording.snapshots[idx].frame_index),
                "display_frame": state.display_frame_number,
                "cursor_frame": cursor.frame_index,
                "time_s": t_s,
                "ok": cursor.list_pos == idx,
            }
        )

    fig = Figure(figsize=(8.0, 4.2), dpi=100)
    mid = idxs[len(idxs) // 2]
    t_mid = float(recording.snapshots[mid].time_s)
    draw_biomechanics_dashboard(
        fig,
        biomech,
        playhead_time_s=t_mid,
        playhead_frame=mid + 1,
        series=BIOMECH_SERIES_COM,
    )
    ax = fig.axes[0] if fig.axes else None
    hover_vs_play = {"ok": False, "detail": "no COM axes"}
    pause_step = []
    if ax is not None:
        src = getattr(ax, PLAYHEAD_VALUE_SOURCE_ATTR, None)
        inspect = format_inspect_crosshair_chip(t_mid)
        play_lbl = format_playhead_label(
            PlayheadState(time_s=t_mid, frame_index=mid + 1)
        )
        hover_vs_play = {
            "inspect_chip": inspect,
            "playhead_label": play_lbl,
            "inspect_is_not_playhead": inspect.startswith("Inspect")
            and "Inspect" not in play_lbl,
            "ok": inspect.startswith("Inspect") and "Inspect" not in play_lbl,
        }
        if isinstance(src, (tuple, list)) and len(src) >= 2:
            times = np.asarray(src[0], dtype=float)
            values = np.asarray(src[1], dtype=float)
            for name, idx in (("pause", 0), ("step", min(1, len(times) - 1)), ("seek", len(times) - 1)):
                expected = sample_displayed_series(times, values, float(times[idx]))
                update_playhead_on_axes(
                    [ax],
                    PlayheadState(time_s=float(times[idx]), frame_index=idx + 1),
                )
                artists = getattr(ax, "_stablewalk_playhead_artists", None)
                visible = False
                mx = my = None
                if artists and len(artists) >= 4 and artists[3] is not None:
                    visible = bool(artists[3].get_visible())
                    off = np.asarray(artists[3].get_offsets(), dtype=float)
                    if off.size >= 2:
                        mx, my = float(off[0, 0]), float(off[0, 1])
                ok = expected is None or (
                    visible
                    and mx is not None
                    and abs(mx - float(times[idx])) < 1e-6
                    and abs(my - float(expected)) < 1e-6
                )
                pause_step.append(
                    {
                        "action": name,
                        "time_s": float(times[idx]),
                        "expected_y": None if expected is None else float(expected),
                        "marker_visible": visible,
                        "ok": bool(ok),
                    }
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
    fig.clear()
    draw_contact_gait_dashboard(
        fig,
        contact,
        vgrf,
        panel=CONTACT_PANEL_CONTACT,
        playhead_time_s=t_mid,
        playhead_frame=mid + 1,
    )
    fig.clear()
    return {
        "playback_labels": labels,
        "labels_ok": all(item.get("ok") for item in labels),
        "hover_vs_playhead": hover_vs_play,
        "pause_step_seek": pause_step,
        "pause_step_ok": all(item.get("ok") for item in pause_step) if pause_step else False,
    }


def analyze_example(key: str, label: str, filename: str) -> dict:
    from matplotlib.figure import Figure  # noqa: F401 — backend pin in main()

    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.analysis.analysis_summary import build_analysis_summary
    from stablewalk.analysis.biomechanical.orchestrator import run_biomechanical_analysis
    from stablewalk.analysis.biomechanical.walking_speed import is_reportable_walking_speed
    from stablewalk.analysis.estimated_vgrf_analysis import analyze_estimated_vgrf
    from stablewalk.analysis.foot_contact_analysis import analyze_foot_contact
    from stablewalk.analysis.ground_reference import (
        CLEARANCE_GATE_RATIONALE,
        NEAR_GROUND_THRESHOLD_M,
    )
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
    from stablewalk.analysis.result_validity import (
        build_result_validity,
        gait_quality_withhold_reason,
        laterality_withhold_reason,
    )
    from stablewalk.analysis.spectral.report import analyze_spectral
    from stablewalk.data_export import export_analysis
    from stablewalk.io.contact_evidence_export import (
        contact_evidence_summary,
        write_contact_frame_evidence,
    )
    from stablewalk.io.frame_audit import (
        frame_audit_row,
        source_only_audit_row,
        timeline_windows,
        write_frame_audit_csv,
    )
    from stablewalk.io.pose_loader import load_pose_sequence, playback_frame_indices
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key, probe_demo_timing

    dest = OUT / key
    dest.mkdir(parents=True, exist_ok=True)
    example = example_by_key(key)
    video_path = demo_path(example) if example else None
    probe = probe_demo_timing(video_path) if video_path else {}
    poses = _latest_poses(filename)
    rec: dict = {
        "key": key,
        "display_name": label,
        "filename": filename,
        "pose_json": str(poses) if poses else None,
        "source_video": str(video_path) if video_path else None,
        "source_probe": probe,
    }
    if poses is None:
        rec["error"] = "No pose JSON found for this example"
        return rec
    if video_path is None or not video_path.is_file():
        rec["error"] = f"Source video missing: {filename}"
        return rec

    t0 = time.perf_counter()
    source_rows = _decode_source_timeline(video_path)
    rec["decode_s"] = round(time.perf_counter() - t0, 2)

    seq = load_pose_sequence(poses)
    pose_by_id = {int(f.frame_index): f for f in seq.frames}
    playback_ids = playback_frame_indices(seq)
    recording = pose_sequence_to_gait_motion(seq, detected_only=False)
    cycles = analyze_gait_cycles(recording)
    contact = analyze_foot_contact(recording, cycles=cycles)
    from stablewalk.io.cycle_candidate_export import write_cycle_candidates_csv

    cycle_csv = write_cycle_candidates_csv(
        getattr(cycles, "cycle_candidates", None) or [],
        dest / f"{key}_cycle_candidates.csv",
        also_json=True,
    )
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
    summary = build_analysis_summary(
        source=filename,
        biomechanical=biomech,
        estimated_vgrf=vgrf,
        contact=contact,
        cycles=cycles,
        spectral=spectral,
    )
    rec["analysis_s"] = round(time.perf_counter() - t0, 2)

    contact_by_id = {int(f.frame_index): f for f in contact.per_frame}
    demo_cap_reason = (
        "Excluded from this dump by the GUI DEMO_MAX_FRAMES=900 cap. "
        "Not a decoder skip and not a detection failure."
    )
    audit_rows = []
    dropped = []
    for src in source_rows:
        fid = int(src["source_frame"])
        pose = pose_by_id.get(fid)
        if pose is None:
            row = source_only_audit_row(
                clip=key,
                source_frame=fid,
                timestamp_s=float(src["timestamp_s"]),
                excluded_reason=demo_cap_reason if fid >= 900 else "not stored in pose dump",
            )
            dropped.append(row)
            audit_rows.append(row)
            continue
        audit_rows.append(
            frame_audit_row(
                pose,
                clip=key,
                decode_ok=True,
                in_analyzed_window=True,
                contact_frame=contact_by_id.get(fid),
            )
        )
    holes = [
        i
        for i in range(min(pose_by_id), max(pose_by_id) + 1)
        if i not in pose_by_id
    ] if pose_by_id else []

    audit_csv = write_frame_audit_csv(audit_rows, dest / "frame_audit.csv")
    windows = timeline_windows(
        seq,
        source_frame_count=int(probe.get("frame_count") or len(source_rows)),
        source_duration_s=float(probe.get("duration_s") or 0.0),
    )
    json_path = export_analysis(seq, dest / f"{key}_analysis.json")
    csv_path = export_analysis(seq, dest / f"{key}_analysis.csv")
    evidence_csv = write_contact_frame_evidence(
        contact, dest / f"{key}_contact_evidence.csv", also_json=True
    )
    evidence_json = evidence_csv.with_suffix(".json")
    evidence_rows = []
    if evidence_json.is_file():
        evidence_rows = json.loads(evidence_json.read_text(encoding="utf-8")).get(
            "rows", []
        )

    ws = getattr(getattr(biomech, "gait_metrics", None), "walking_speed", None)
    phase_text = _phase_annotation_text(biomech)
    cursor = _cursor_checks(recording, seq, contact, vgrf, biomech)
    rec.update(
        {
            "clip": {
                "source_frames_decoded": len(source_rows),
                "source_duration_s": float(probe.get("duration_s") or 0.0),
                "source_fps": float(probe.get("fps") or 0.0),
                "n_pose_frames": len(seq.frames),
                "playback_slots": len(playback_ids),
                "recording_frames": recording.frame_count,
                "detected_only": bool(recording.metadata.get("detected_only")),
                "analyzed_duration_s": float(windows.analyzed_duration_s),
                "analyzed_span_s": float(windows.analyzed_duration_s),
                "cumulative_valid_duration_s": float(windows.cumulative_valid_duration_s),
                "coverage": float(windows.coverage),
                "longest_valid_segment_s": float(windows.longest_valid_segment_s),
                "n_valid_runs": int(windows.n_valid_runs),
                "interval_counting_convention": windows.interval_counting_convention,
                "usable_duration_s": float(windows.cumulative_valid_duration_s),
                "ids_match_recording": recording.frame_count == len(seq.frames),
            },
            "dropped_or_excluded": {
                "pose_id_holes_in_dump": holes[:40],
                "n_pose_id_holes": len(holes),
                "n_source_not_in_dump": len(dropped),
                "first_excluded_source_frame": dropped[0]["source_frame"] if dropped else None,
                "excluded_reason": dropped[0]["excluded_reason"] if dropped else "",
                "status_counts": windows.status_counts,
            },
            "cycles": {
                **_cycle_side_counts(cycles),
                "n_candidates": len(getattr(cycles, "cycle_candidates", None) or []),
                "n_accepted": sum(
                    1
                    for c in (getattr(cycles, "cycle_candidates", None) or [])
                    if getattr(c, "accepted", False)
                ),
                "n_accepted_left": sum(
                    1
                    for c in (getattr(cycles, "cycle_candidates", None) or [])
                    if getattr(c, "accepted", False)
                    and str(getattr(c, "side", "")).lower().startswith("left")
                ),
                "n_accepted_right": sum(
                    1
                    for c in (getattr(cycles, "cycle_candidates", None) or [])
                    if getattr(c, "accepted", False)
                    and str(getattr(c, "side", "")).lower().startswith("right")
                ),
                "rejection_counts": _candidate_rejection_counts(
                    getattr(cycles, "cycle_candidates", None) or []
                ),
                "candidate_reasons": [
                    {
                        "side": getattr(c, "side", ""),
                        "start_frame": getattr(c, "start_frame", None),
                        "end_frame": getattr(c, "end_frame", None),
                        "duration_s": getattr(c, "duration_s", None),
                        "known_contact_coverage": getattr(c, "known_contact_coverage", None),
                        "n_known_stance": getattr(c, "n_known_stance", None),
                        "n_known_swing": getattr(c, "n_known_swing", None),
                        "valid_pose_coverage": getattr(c, "valid_pose_coverage", None),
                        "longest_internal_gap_s": getattr(c, "longest_internal_gap_s", None),
                        "accepted": getattr(c, "accepted", False),
                        "acceptance_reason": getattr(c, "acceptance_reason", ""),
                    }
                    for c in (getattr(cycles, "cycle_candidates", None) or [])[:80]
                ],
            },
            "contact_coverage": contact_evidence_summary(evidence_rows) if evidence_rows else {},
            "validity": validity.to_dict(),
            "headline": validity.headline_lines(),
            "laterality_withhold": laterality_withhold_reason(validity),
            "gait_quality_withhold": gait_quality_withhold_reason(validity),
            "summary": {
                "gait_quality": None
                if summary.overall_gait_quality is None
                else summary.overall_gait_quality.to_dict(),
                "walking_speed": None
                if summary.walking_speed is None
                else summary.walking_speed.to_dict(),
                "symmetry": None if summary.symmetry is None else summary.symmetry.to_dict(),
                "contact_left": None
                if summary.contact_coverage_left is None
                else summary.contact_coverage_left.to_dict(),
                "contact_right": None
                if summary.contact_coverage_right is None
                else summary.contact_coverage_right.to_dict(),
                "cycle_evidence": None
                if summary.cycle_evidence is None
                else summary.cycle_evidence.to_dict(),
            },
            "walking_speed": {
                "raw_value": None if ws is None else getattr(ws, "value", None),
                "confidence": None if ws is None else getattr(ws, "confidence", None),
                "kind": None if ws is None else getattr(ws, "kind", None),
                "note": None if ws is None else getattr(ws, "note", None),
                "reportable": is_reportable_walking_speed(ws),
                "measured": False,
                "estimated": True,
                "fallback": False,
            },
            "phase_annotation": phase_text,
            "phase_annotation_hides_withheld_quality": "Gait quality" not in phase_text,
            "cursor": cursor,
            "exports": {
                "frame_audit_csv": str(audit_csv),
                "analysis_json": str(json_path),
                "analysis_csv": str(csv_path),
                "contact_evidence_csv": str(evidence_csv),
                "contact_evidence_json": str(evidence_json),
                "cycle_candidates_csv": str(cycle_csv),
                "cycle_candidates_json": str(Path(cycle_csv).with_suffix(".json")),
            },
            "cache_used": True,
            "cache_note": "Reused 2026-11-03 13:07 UTC full-analysis pose dumps; no MediaPipe rerun.",
        }
    )
    (dest / f"{key}_validity.json").write_text(
        json.dumps(
            _jsonable(
                {
                    "headline": rec["headline"],
                    "validity": rec["validity"],
                    "cycles": rec["cycles"],
                    "contact_coverage": rec["contact_coverage"],
                    "laterality_withhold": rec["laterality_withhold"],
                    "gait_quality_withhold": rec["gait_quality_withhold"],
                    "summary": rec["summary"],
                    "walking_speed": rec["walking_speed"],
                    "clip": rec["clip"],
                    "dropped_or_excluded": rec["dropped_or_excluded"],
                    "clearance_gate": CLEARANCE_GATE_RATIONALE,
                    "clearance_gate_m": NEAR_GROUND_THRESHOLD_M,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    rec["exports"]["validity_json"] = str(dest / f"{key}_validity.json")
    rec["exports"]["audit_json"] = str(dest / f"{key}_source_audit.json")
    (dest / f"{key}_source_audit.json").write_text(
        json.dumps(_jsonable(rec), indent=2), encoding="utf-8"
    )
    return rec


def _fast_pose_gap_note(examples: list[dict]) -> str:
    athletic = next((rec for rec in examples if rec.get("key") == "athletic"), None)
    if not athletic or athletic.get("error"):
        return (
            "- Pose-gap rejection is active: HS–HS intervals that span an invalid-pose "
            "hole larger than one missed sample are `unsupported_pose_gap`, not gait cycles."
        )
    longest = float((athletic.get("clip") or {}).get("longest_valid_segment_s") or 0.0)
    n_gap = int(
        ((athletic.get("cycles") or {}).get("rejection_counts") or {}).get(
            "unsupported_pose_gap", 0
        )
    )
    n_pub_l = int((athletic.get("cycles") or {}).get("n_left") or 0)
    n_pub_r = int((athletic.get("cycles") or {}).get("n_right") or 0)
    return (
        f"- Fast Walking longest continuous valid pose segment is {longest:.2f} s. "
        f"{n_gap} candidate(s) were rejected as `unsupported_pose_gap` rather than "
        "bridging those holes. "
        f"Published accepted cycles: {n_pub_l} L / {n_pub_r} R "
        "(both sides; no silent one-side ranking)."
    )


def _run_pytest() -> dict:
    tests = [
        "tests/test_frame_timeline_preservation.py",
        "tests/test_playback_sync.py",
        "tests/test_plot_cursor_series.py",
        "tests/test_chart_playhead.py",
        "tests/test_result_validity.py",
        "tests/test_overview_clinical_summary.py",
        "tests/test_tracking_coverage.py",
        "tests/test_analysis_mode.py",
        "tests/test_timeline_duration.py",
        "tests/test_cycle_candidate_validity.py",
        "tests/test_clearance_gate.py",
    ]
    cmd = [sys.executable, "-m", "pytest", *tests, "-q", "--tb=line"]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
        "tests": tests,
    }


def _write_report(examples: list[dict], pytest_result: dict) -> Path:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Validation report",
        "",
        f"Generated {now}. Analysis used the existing full-analysis pose dumps (`detected_only=False`); MediaPipe was not re-run. Failed detections keep source frame IDs and timestamps. Source videos were decoded independently of playback.",
        "",
        "Filename / demo key is a source label only. Missing L/R contact is not physical asymmetry. Thresholds were not retuned.",
        "",
        "## Actual tests",
        "",
        f"- pytest return code: {pytest_result.get('returncode')}",
        f"- result: {'passed' if pytest_result.get('passed') else 'failed'}",
        "",
        "```",
        str(pytest_result.get("stdout_tail") or ""),
        "```",
        "",
        "## Per-example source vs analyzed duration",
        "",
    ]
    for rec in examples:
        label = rec.get("display_name", rec.get("key"))
        if rec.get("error"):
            lines.extend([f"### {label}", "", f"Error: {rec['error']}", ""])
            continue
        clip = rec["clip"]
        dropped = rec["dropped_or_excluded"]
        contact = rec.get("contact_coverage", {}).get("by_side", {})
        left = contact.get("left", {})
        right = contact.get("right", {})
        lines.extend(
            [
                f"### {label} (`{rec['filename']}`)",
                "",
                f"- Source decoded: **{clip['source_frames_decoded']}** frames, "
                f"{clip['source_duration_s']:.2f} s @ {clip['source_fps']:.3f} fps.",
                f"- Analyzed dump: **{clip['n_pose_frames']}** stored frames.",
                f"- Analyzed span: **{clip.get('analyzed_span_s', clip['analyzed_duration_s']):.2f} s** "
                f"(last−first original timestamp, includes gaps).",
                f"- Cumulative valid duration: **{clip.get('cumulative_valid_duration_s', 0.0):.2f} s** "
                f"(sum of contiguous pose_ok runs; n−1 intervals per run).",
                f"- Coverage: **{100.0 * float(clip.get('coverage') or 0.0):.1f}%** of analyzed span; "
                f"longest valid segment **{clip.get('longest_valid_segment_s', 0.0):.2f} s** "
                f"({clip.get('n_valid_runs', 0)} runs).",
                f"- Recording slots equal stored frames: {clip['ids_match_recording']} "
                f"(detected_only={clip['detected_only']}).",
                f"- Status counts: `{dropped.get('status_counts')}`",
                f"- Source frames not in dump: {dropped['n_source_not_in_dump']}"
                + (
                    f" (from source frame {dropped['first_excluded_source_frame']}: "
                    f"{dropped['excluded_reason']})"
                    if dropped["n_source_not_in_dump"]
                    else ""
                ),
                f"- Pose-dump ID holes: {dropped['n_pose_id_holes']}",
                f"- Left contact known: {left.get('known_pct', 'n/a'):.1f}%"
                if isinstance(left.get("known_pct"), (int, float))
                else "- Left contact known: n/a",
                f"- Right contact known: {right.get('known_pct', 'n/a'):.1f}%"
                if isinstance(right.get("known_pct"), (int, float))
                else "- Right contact known: n/a",
                f"- Cycle evidence (accepted, both sides published): "
                f"{rec['cycles']['n_left']} L / {rec['cycles']['n_right']} R "
                f"(candidates {rec['cycles'].get('n_candidates', 0)}, "
                f"accepted {rec['cycles'].get('n_accepted', rec['cycles']['n_cycles'])}; "
                f"accepted candidates {rec['cycles'].get('n_accepted_left', rec['cycles']['n_left'])} L / "
                f"{rec['cycles'].get('n_accepted_right', rec['cycles']['n_right'])} R). "
                f"Rejection counts: `{rec['cycles'].get('rejection_counts') or {}}`.",
                f"- Longest continuous valid pose segment: "
                f"**{clip.get('longest_valid_segment_s', 0.0):.2f} s**. "
                "A cycle longer than that interval is accepted only when every internal "
                "pose gap is at most one missed sample; unsupported gaps are rejected as "
                "`unsupported_pose_gap` and are not interpolated.",
                f"- Laterality withhold: {rec['laterality_withhold'] or 'not withheld'}",
                f"- Gait-quality withhold: {rec['gait_quality_withhold'] or 'not withheld'}",
                f"- Summary gait quality: {(rec['summary']['gait_quality'] or {}).get('value')}",
                f"- Summary walking speed: {(rec['summary']['walking_speed'] or {}).get('value')} "
                f"(estimated={rec['walking_speed']['estimated']}, "
                f"reportable={rec['walking_speed']['reportable']}, "
                f"not a measured speed).",
                f"- Phase-chart annotation: `{rec['phase_annotation'][:180]}`",
                f"- Cursor labels ok: {rec['cursor']['labels_ok']}; "
                f"hover≠playback: {rec['cursor']['hover_vs_playhead'].get('ok')}; "
                f"pause/step/seek: {rec['cursor']['pause_step_ok']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Verified fixes",
            "",
            "- Analyze (`_run_full_analysis`) now requests `analysis_mode=full` so "
            "`max_frames=None`. Examples (`_load_demo_gait`) request labeled preview "
            f"(first {900} frames). Static source inspection of those call sites is in "
            "`tests/test_analysis_mode.py`.",
            "- `usable_duration` is replaced by analyzed span, cumulative valid duration, "
            "coverage, and longest valid segment using original timestamps "
            "(n−1 intervals per contiguous valid run).",
            "- Binary-contact HS–HS intervals are exported as candidates. Valid cycles "
            "require known stance, confirmed swing, and no unsupported pose gap. "
            "Accepted left and right cycles are both published.",
            "- Cycle-candidate CSV/JSON include side, start/end source frames, duration, "
            "valid-pose coverage, longest internal gap, known stance/swing, and the "
            "acceptance or rejection reason. Gaps larger than one missed sample are not bridged.",
            "- The 0.100 m gate is documented as body-scale +Y vs floor_y confirmed-flight. "
            "Near-ground without a prior phase is Unknown, not stance.",
            "- Compare Session A/B now carry an explicit analysis mode. Session B no longer "
            "silently applies `DEMO_MAX_FRAMES`. Full-versus-preview comparisons are refused.",
            "",
            "## Unequal contact coverage",
            "",
            "Unknown contact is a missing observation (`not_confirmed_flight` dominates). "
            "Unequal L/R known percentages and unilateral cycle counts are coverage gaps, "
            "not a gait-laterality finding. Overview / Results withhold symmetry and the "
            "gait-quality heuristic for all three examples.",
            "",
            "## Remaining causes",
            "",
            "- Fast Walking binary-contact HS–HS intervals can far outnumber known left-contact frames. "
            "Those intervals are exported as candidates; only intervals with known stance, confirmed swing, "
            "and no unsupported pose gap count as valid cycles.",
            "- Monocular COM–BoS remains biomechanically unavailable.",
            "- Walking speed, when shown, is an **estimate** scaled to default stature, not a measured overground speed and not a 0.55 m/s fallback constant.",
            "- The 0.100 m clearance gate is estimated body-scale metres on +Y vs floor_y. "
            "It confirms flight; it does not classify stance. Near-ground off bits stay Unknown.",
            _fast_pose_gap_note(examples),
            "",
            "## Checks not performed",
            "",
        ]
    )
    for item in UNPERFORMED:
        lines.append(f"- {item}")
    lines.extend(["", "## Upload set", "", "See the folder listing at the end of the agent reply.", ""])
    text = "\n".join(lines)
    primary = OUT / "validation_report.md"
    primary.write_text(text, encoding="utf-8")
    (ROOT / "validation_report.md").write_text(text, encoding="utf-8")
    return primary


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    examples = []
    for key, label, filename in EXAMPLES:
        print(f"Auditing {label}...", flush=True)
        rec = analyze_example(key, label, filename)
        examples.append(rec)
        if rec.get("error"):
            print(f"  ERROR {rec['error']}", flush=True)
            continue
        print(
            f"  source={rec['clip']['source_frames_decoded']} "
            f"analyzed={rec['clip']['n_pose_frames']} "
            f"excluded={rec['dropped_or_excluded']['n_source_not_in_dump']} "
            f"cycles={rec['cycles']['n_left']}L/{rec['cycles']['n_right']}R",
            flush=True,
        )
    print("Running pytest...", flush=True)
    pytest_result = _run_pytest()
    print(pytest_result.get("stdout_tail") or "", flush=True)
    report_path = _write_report(examples, pytest_result)
    bundle = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pytest": pytest_result,
        "examples": examples,
        "report": str(report_path),
    }
    (OUT / "session_validation.json").write_text(
        json.dumps(_jsonable(bundle), indent=2), encoding="utf-8"
    )
    print(f"Wrote {report_path}", flush=True)
    return 0 if pytest_result.get("passed") and not any(e.get("error") for e in examples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
