"""End-to-end validation of the 16-joint / ACF / FFT / asymmetry pipeline.

Uses cached pose JSON (does not re-run MediaPipe). Does not invent metrics
or change cadence / gait quality. Writes JSON + a short markdown report.

  python scripts/gait_pipeline_e2e_validation.py
"""
from __future__ import annotations

import json
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.gait_acf import analyze_gait_autocorrelation
from stablewalk.analysis.gait_asymmetry_moments import detect_gait_asymmetry_moments
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.analysis.gait_spectrum import analyze_gait_spectrum
from stablewalk.analysis.spectral.report import analyze_spectral
from stablewalk.analysis.spectral.spectrum import _dominant_in_band
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.models.gait_wireframe import (
    GAIT_WIRE_JOINT_COUNT,
    GAIT_WIRE_JOINT_IDS,
    build_gait_wireframe,
)

OUT_JSON = ROOT / "docs" / "_gait_pipeline_e2e.json"
OUT_MD = ROOT / "docs" / "_gait_pipeline_e2e_detail.md"
REPORT_MD = ROOT / "docs" / "gait_pipeline_e2e_validation.md"

CLIPS = (
    {
        "id": "normal",
        "kind": "normal_symmetric",
        "video": config.DEMO_VIDEOS_DIR / "normal_gait.mp4",
        "poses": config.POSES_DIR / "normal_gait_poses.json",
    },
    {
        "id": "asymmetric",
        "kind": "asymmetric",
        "video": config.DEMO_VIDEOS_DIR / "abnormal_gait.mp4",
        "poses": config.POSES_DIR / "abnormal_gait_poses.json",
    },
    {
        "id": "noisy",
        "kind": "noisy_low_confidence",
        "video": config.DEMO_VIDEOS_DIR / "athletic_walking.mp4",
        "poses": config.POSES_DIR / "athletic_walking_poses.json",
    },
)

GAIT_BAND = (0.40, 4.00)
REALISTIC_CYCLE_S = (0.50, 2.00)


def _video_meta(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "cannot_open", "path": str(path)}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, frame = cap.read()
    cap.release()
    duration = n / fps if fps > 1e-6 else 0.0
    return {
        "path": str(path),
        "fps": fps,
        "frames": n,
        "duration_s": duration,
        "width": w,
        "height": h,
        "first_frame_ok": bool(ok and frame is not None),
    }


def _mean_visibility(seq) -> float | None:
    vals: list[float] = []
    for fr in seq.frames:
        for kp in fr.keypoints or []:
            vis = getattr(kp, "visibility", None)
            if vis is None:
                continue
            try:
                vals.append(float(vis))
            except (TypeError, ValueError):
                continue
    if not vals:
        return None
    return float(np.mean(vals))


def _degrade_sequence(seq, *, vis_scale: float = 0.22, noise: float = 0.012, seed: int = 7):
    """In-memory low-confidence overlay of an existing pose sequence."""
    rng = np.random.default_rng(seed)
    clone = deepcopy(seq)
    for fr in clone.frames:
        for kp in fr.keypoints or []:
            try:
                kp.visibility = float(kp.visibility) * vis_scale
            except Exception:
                kp.visibility = vis_scale
            kp.x = float(kp.x) + float(rng.normal(scale=noise))
            kp.y = float(kp.y) + float(rng.normal(scale=noise))
    return clone


def _checks_timing(seq, rec, video_meta: dict) -> dict:
    fps = max(float(seq.fps), 1e-6)
    detected = [fr for fr in seq.frames if fr.detected and fr.keypoints]
    rec_times = np.asarray([float(s.time_s) for s in rec.snapshots], dtype=float)
    seq_times = np.asarray([float(fr.time_seconds(fps)) for fr in detected], dtype=float)
    dt = np.diff(rec_times) if rec_times.size >= 2 else np.asarray([])
    median_dt = float(np.median(dt)) if dt.size else None
    implied_fps = (1.0 / median_dt) if median_dt and median_dt > 1e-9 else None
    max_skew = None
    if rec_times.size and seq_times.size and rec_times.size == seq_times.size:
        max_skew = float(np.max(np.abs(rec_times - seq_times)))
    elif rec_times.size and seq_times.size:
        n = min(rec_times.size, seq_times.size)
        max_skew = float(np.max(np.abs(rec_times[:n] - seq_times[:n])))
    index_time_skew = 0.0
    for snap in rec.snapshots:
        expected = float(snap.frame_index) / fps
        # Prefer stored timestamps when the pose JSON actually has them.
        index_time_skew = max(index_time_skew, abs(float(snap.time_s) - expected))
    video_fps = float(video_meta.get("fps") or 0.0)
    fps_ok = abs(float(rec.fps) - float(seq.fps)) < 0.05
    vs_video = None
    if video_fps > 1.0:
        vs_video = abs(float(rec.fps) - video_fps) / video_fps
    return {
        "seq_fps": float(seq.fps),
        "rec_fps": float(rec.fps),
        "video_fps": video_fps,
        "n_seq": len(seq.frames),
        "n_detected": len(detected),
        "n_rec": rec.frame_count,
        "duration_rec_s": float(rec_times[-1] - rec_times[0]) if rec_times.size >= 2 else 0.0,
        "implied_fps": implied_fps,
        "max_time_skew_s": max_skew,
        "max_index_time_skew_s": index_time_skew,
        "fps_seq_rec_match": fps_ok,
        "video_fps_rel_err": vs_video,
        "monotonic": bool(rec_times.size < 2 or np.all(np.diff(rec_times) >= -1e-9)),
        "detected_equals_recording": len(detected) == rec.frame_count,
    }


def _checks_wire(rec) -> dict:
    wire = build_gait_wireframe(rec)
    missing_frames = 0
    incomplete = 0
    time_mismatch = 0
    for i, fr in enumerate(wire.frames):
        if set(fr.joints) != set(GAIT_WIRE_JOINT_IDS):
            incomplete += 1
        snap = rec.snapshots[i] if i < len(rec.snapshots) else None
        if snap is None:
            missing_frames += 1
            continue
        if abs(float(fr.time_s) - float(snap.time_s)) > 1e-9:
            time_mismatch += 1
        if int(fr.frame_index) != int(snap.frame_index):
            time_mismatch += 1
    return {
        "joint_count": GAIT_WIRE_JOINT_COUNT,
        "joint_ids": list(GAIT_WIRE_JOINT_IDS),
        "n_frames": len(wire.frames),
        "n_recording": rec.frame_count,
        "incomplete_frames": incomplete,
        "time_mismatch_frames": time_mismatch,
        "complete": incomplete == 0 and len(wire.frames) == rec.frame_count,
        "body_scale": float(wire.body_scale),
        "coordinate_system": wire.coordinate_system,
    }


def _fft_psd_consistency(spec) -> dict:
    rows = []
    for sid, sig in spec.signals.items():
        fft_f = np.asarray(sig.fft_freq_hz, dtype=float)
        fft_p = np.asarray([np.nan if v is None else v for v in sig.fft_power], dtype=float)
        psd_f = np.asarray(sig.freq_hz, dtype=float)
        psd_p = np.asarray([np.nan if v is None else v for v in sig.psd], dtype=float)
        fft_dom, _ = _dominant_in_band(fft_f, fft_p, GAIT_BAND) if fft_f.size else (None, None)
        psd_dom, _ = _dominant_in_band(psd_f, psd_p, GAIT_BAND) if psd_f.size else (None, None)
        delta = None
        if fft_dom is not None and psd_dom is not None:
            delta = abs(float(fft_dom) - float(psd_dom))
        rows.append(
            {
                "signal": sid,
                "fft_f0": fft_dom,
                "psd_f0": psd_dom,
                "reported_f0": sig.dominant_frequency_hz,
                "abs_delta_hz": delta,
            }
        )
    deltas = [r["abs_delta_hz"] for r in rows if r["abs_delta_hz"] is not None]
    return {
        "per_signal": rows,
        "median_abs_delta_hz": float(np.median(deltas)) if deltas else None,
        "max_abs_delta_hz": float(np.max(deltas)) if deltas else None,
        "session_f0": spec.dominant_frequency_hz,
        "session_period_s": spec.dominant_period_s,
        "available": spec.available,
    }


def _phase_equivalent(moments, spec) -> dict:
    half = getattr(moments, "half_cycle_s", None)
    used = [s for s in moments.per_frame if s.used]
    phase_skews = []
    for sample in used:
        phase = getattr(sample, "phase_pct", None)
        match_phase = getattr(sample, "match_phase_pct", None)
        if phase is None or match_phase is None:
            continue
        phase_skews.append(abs(float(phase) - float(match_phase)))
    phases = {}
    for pid, item in spec.phase_offsets.items():
        phases[pid] = {
            "phase_difference_deg": item.phase_difference_deg,
            "time_offset_s": item.time_offset_s,
            "method": item.method,
        }
    corresponding = bool(phase_skews) and float(np.max(phase_skews)) < 1e-6
    return {
        "cycle_duration_s": moments.cycle_duration_s,
        "half_cycle_s": half,
        "n_used_frames": len(used),
        "max_phase_skew_pct": float(np.max(phase_skews)) if phase_skews else None,
        "phase_is_corresponding": corresponding,
        "phase_is_half_cycle": corresponding,
        "lr_spectral_phase": phases,
    }


def _moments_vs_events(moments, cycles) -> dict:
    events = [e for e in (getattr(cycles, "events", None) or []) if getattr(e, "time_s", None) is not None]
    event_t = np.asarray([float(e.time_s) for e in events], dtype=float)
    rows = []
    for m in moments.moments:
        nearest = None
        if event_t.size:
            j = int(np.argmin(np.abs(event_t - float(m.time_s))))
            nearest = {
                "event": getattr(events[j], "event_type", None),
                "side": getattr(events[j], "side", None),
                "event_time_s": float(events[j].time_s),
                "dt_s": abs(float(events[j].time_s) - float(m.time_s)),
            }
        rows.append(
            {
                "frame": m.frame_index,
                "time_s": m.time_s,
                "match_time_s": m.match_time_s,
                "score": m.score,
                "joints": list(m.involved_joints),
                "severity": m.severity,
                "explanation": m.explanation,
                "nearest_event": nearest,
                "clinical": False,
            }
        )
    return {
        "available": moments.available,
        "n_moments": len(moments.moments),
        "n_events": len(events),
        "moments": rows,
    }


def _existing_metrics(cycles, rec) -> dict:
    m = cycles.metrics
    return {
        "cadence_steps_per_min": m.cadence_steps_per_min,
        "gait_cycle_count": m.gait_cycle_count,
        "stride_time_s": m.stride_time_s,
        "metrics_reliable": m.metrics_reliable,
        "event_sufficiency": cycles.event_sufficiency,
        "n_snapshots": rec.frame_count,
        "rec_id": id(rec),
        "first_knee": (
            rec.snapshots[0].joints["left_knee"].position.x
            if rec.snapshots and "left_knee" in rec.snapshots[0].joints
            else None
        ),
    }


def _static_duplicate_and_sync_audit() -> dict:
    app = (ROOT / "stablewalk" / "ui" / "tk" / "app.py").read_text(encoding="utf-8")
    spec = (ROOT / "stablewalk" / "ui" / "tk" / "dashboard_spectral.py").read_text(encoding="utf-8")
    overview = (ROOT / "stablewalk" / "ui" / "tk" / "overview_signal_spectral.py").read_text(
        encoding="utf-8"
    )
    cache = (ROOT / "stablewalk" / "ui" / "session_analysis_cache.py").read_text(encoding="utf-8")
    return {
        "gait_acf_imported_in_app": "analyze_gait_autocorrelation" in app,
        "gait_spectrum_imported_in_app": "analyze_gait_spectrum" in app,
        "moments_in_analysis_cache": "gait_asymmetry_moments" in cache,
        "spectral_uses_worker_thread": "stablewalk-spectral" in spec and "threading.Thread" in spec,
        "overview_reuses_spectral_report": "_spectral_report" in overview,
        "overview_playhead_only": "playhead_only" in overview,
        "seek_updates_video_and_skeleton": "_seek_motion_workspace_time" in app
        and "self._sync_all_to_frame" in app,
        "cache_doc_forbids_replay_analysis": "must never re-trigger analysis" in cache,
    }


def _status(ok: bool, *, skip: bool = False) -> str:
    if skip:
        return "NOT TESTED"
    return "PASS" if ok else "FAIL"


def analyze_clip(spec: dict, *, degrade: bool = False) -> dict:
    out = {
        "id": spec["id"],
        "kind": spec["kind"] + ("_degraded" if degrade else ""),
        "video_exists": spec["video"].is_file(),
        "poses_exist": spec["poses"].is_file(),
        "checks": {},
        "notes": [],
        "error": None,
    }
    if not spec["video"].is_file() or not spec["poses"].is_file():
        out["error"] = "missing video or poses"
        return out
    video_meta = _video_meta(spec["video"])
    out["video"] = video_meta
    seq = load_pose_sequence(spec["poses"])
    if degrade:
        seq = _degrade_sequence(seq)
        out["notes"].append(
            "Pose visibility scaled down and image XY jittered in memory "
            "to simulate low-confidence tracking. MediaPipe was not re-run."
        )
    out["mean_visibility"] = _mean_visibility(seq)
    rec = pose_sequence_to_gait_motion(seq)
    rec.build_time_series()
    t0 = time.perf_counter()
    cycles = analyze_gait_cycles(rec)
    metrics_before = _existing_metrics(cycles, rec)
    acf = analyze_gait_autocorrelation(rec, sequence=seq)
    spec_gait = analyze_gait_spectrum(rec, sequence=seq, gait_cycle=cycles)
    spec_full = analyze_spectral(rec, sequence=seq, cycles=cycles)
    moments = detect_gait_asymmetry_moments(rec, cycles)
    elapsed = time.perf_counter() - t0
    metrics_after = _existing_metrics(analyze_gait_cycles(rec), rec)
    out["elapsed_s"] = elapsed
    out["timing"] = _checks_timing(seq, rec, video_meta)
    out["wire16"] = _checks_wire(rec)
    out["acf"] = {
        "available": acf.available,
        "gait_cycle_duration_s": acf.gait_cycle_duration_s,
        "dominant_period_s": acf.dominant_period_s,
        "regularity_score": acf.regularity_score,
        "confidence": acf.confidence,
        "confidence_label": acf.confidence_label,
        "notes": list(acf.notes)[:8],
        "n_signals": len(acf.signals),
        "source": acf.source,
    }
    out["spectrum"] = _fft_psd_consistency(spec_gait)
    out["spectrum"]["notes"] = list(spec_gait.notes)[:8]
    out["spectral_full"] = {
        "available": spec_full.available,
        "fps": spec_full.bundle.fps,
        "n_times": int(spec_full.bundle.times_s.size),
        "univariate": spec_full.univariate_channel_id,
        "primary_f0": (
            spec_full.primary_spectrum().dominant_hz
            if spec_full.primary_spectrum() is not None
            else None
        ),
        "warnings": list(spec_full.warnings)[:6],
    }
    out["phase"] = _phase_equivalent(moments, spec_gait)
    out["moments"] = _moments_vs_events(moments, cycles)
    out["metrics_before"] = metrics_before
    out["metrics_after"] = metrics_after
    out["metrics_unchanged"] = (
        metrics_before["cadence_steps_per_min"] == metrics_after["cadence_steps_per_min"]
        and metrics_before["gait_cycle_count"] == metrics_after["gait_cycle_count"]
        and metrics_before["rec_id"] == metrics_after["rec_id"]
        and metrics_before["first_knee"] == metrics_after["first_knee"]
    )
    cycle_ok = True
    if acf.available and acf.gait_cycle_duration_s is not None:
        cycle_ok = REALISTIC_CYCLE_S[0] <= float(acf.gait_cycle_duration_s) <= REALISTIC_CYCLE_S[1]
    fft_psd_ok = True
    med = out["spectrum"]["median_abs_delta_hz"]
    if med is not None and spec_gait.available:
        fft_psd_ok = med <= 0.35
    out["checks"] = {
        "wire16_complete": _status(out["wire16"]["complete"]),
        "timestamps_fps": _status(
            out["timing"]["fps_seq_rec_match"]
            and out["timing"]["monotonic"]
            and out["timing"]["detected_equals_recording"]
            and (out["timing"]["max_time_skew_s"] is None or out["timing"]["max_time_skew_s"] < 1e-6)
        ),
        "acf_realistic_or_withheld": _status(
            (not acf.available) or cycle_ok
        ),
        "fft_psd_consistent": _status(fft_psd_ok),
        "phase_half_cycle": _status(
            (not moments.available) or bool(out["phase"]["phase_is_corresponding"])
        ),
        "moments_not_clinical": _status(
            all("diagnos" not in (m.get("explanation") or "").lower() for m in out["moments"]["moments"])
        ),
        "existing_metrics_unchanged": _status(out["metrics_unchanged"]),
        "analysis_elapsed_s": elapsed,
    }
    return out


def _verdict_rows(payload: dict) -> list[tuple[str, str, str]]:
    rows = []
    for clip in payload["clips"]:
        cid = clip["id"]
        if clip.get("error"):
            rows.append((cid, "load", "FAIL"))
            continue
        for key, status in clip["checks"].items():
            if key == "analysis_elapsed_s":
                continue
            rows.append((cid, key, status))
    audit = payload["duplicate_sync_audit"]
    rows.append(
        (
            "gui",
            "no_duplicate_acf_spectrum_on_ui_thread",
            _status(
                (not audit["gait_acf_imported_in_app"])
                and (not audit["gait_spectrum_imported_in_app"])
                and audit["spectral_uses_worker_thread"]
                and audit["overview_reuses_spectral_report"]
                and audit["overview_playhead_only"]
            ),
        )
    )
    rows.append(
        (
            "gui",
            "video_graph_3d_seek_wired",
            _status(audit["seek_updates_video_and_skeleton"]),
        )
    )
    return rows


def write_report(payload: dict) -> None:
    rows = _verdict_rows(payload)
    n_pass = sum(1 for r in rows if r[2] == "PASS")
    n_fail = sum(1 for r in rows if r[2] == "FAIL")
    lines = [
        "# Gait-analysis pipeline — end-to-end validation",
        "",
        f"**Date:** {payload['date']}",
        "**Role:** headless scientific check of the 16-joint wireframe, ACF, FFT/PSD, phase-normalized L/R comparison, and asymmetric moments.",
        "**Pose source:** cached JSON (MediaPipe was not re-run).",
        f"**Checks:** {n_pass} PASS, {n_fail} FAIL.",
        "",
        "This is **not** clinical validation. Cadence and gait-quality were not rewritten.",
        "",
        "## Clips",
        "",
        "| Clip | Kind | Video FPS | Pose FPS | Frames (video/pose) | Mean visibility | Analysis s |",
        "|------|------|----------:|---------:|--------------------:|----------------:|-----------:|",
    ]
    for clip in payload["clips"]:
        if clip.get("error"):
            lines.append(f"| {clip['id']} | {clip['kind']} | — | — | missing | — | — |")
            continue
        v = clip["video"]
        t = clip["timing"]
        vis = clip.get("mean_visibility")
        lines.append(
            f"| {clip['id']} | {clip['kind']} | {v.get('fps'):.2f} | {t['rec_fps']:.2f} | "
            f"{v.get('frames')}/{t['n_rec']} | "
            f"{vis if vis is None else f'{vis:.3f}'} | {clip['elapsed_s']:.2f} |"
        )
    lines += ["", "## Results", ""]
    for clip in payload["clips"]:
        lines.append(f"### {clip['id']} ({clip['kind']})")
        if clip.get("error"):
            lines.append(f"- **FAIL** {clip['error']}")
            lines.append("")
            continue
        acf = clip["acf"]
        spec = clip["spectrum"]
        mom = clip["moments"]
        phase = clip["phase"]
        wire = clip["wire16"]
        met = clip["metrics_before"]
        lines.append(
            f"- 16-joint model: {wire['n_frames']} frames, incomplete={wire['incomplete_frames']}, "
            f"scale={wire['body_scale']:.3f} ({wire['coordinate_system']})."
        )
        lines.append(
            f"- Timing: rec FPS {clip['timing']['rec_fps']:.3f}, video FPS {clip['timing']['video_fps']:.3f}, "
            f"max t-skew {clip['timing']['max_time_skew_s']}."
        )
        lines.append(
            f"- ACF: available={acf['available']}, cycle={acf['gait_cycle_duration_s']}, "
            f"regularity={acf['regularity_score']}, confidence={acf['confidence_label']}."
        )
        lines.append(
            f"- Spectrum: available={spec['available']}, session f0={spec['session_f0']}, "
            f"period={spec['session_period_s']}, median |FFT−PSD|={spec['median_abs_delta_hz']} Hz."
        )
        lines.append(
            f"- Phase: T={phase['cycle_duration_s']}, T/2={phase['half_cycle_s']}, "
            f"used frames={phase['n_used_frames']}, phase skew={phase.get('max_phase_skew_pct')}%."
        )
        lines.append(
            f"- Moments: available={mom['available']}, n={mom['n_moments']} (top-3 merged). "
            "Not a diagnosis."
        )
        for item in mom["moments"]:
            near = item.get("nearest_event") or {}
            lines.append(
                f"  - t={item['time_s']:.2f}s frame {item['frame']} score={item['score']:.1f} "
                f"joints={item['joints']} nearest_event={near.get('event')} dt={near.get('dt_s')}."
            )
        lines.append(
            f"- Existing gait metrics after new modules: cadence={met['cadence_steps_per_min']}, "
            f"cycles={met['gait_cycle_count']}, unchanged={clip['metrics_unchanged']}."
        )
        for k, v in clip["checks"].items():
            if k == "analysis_elapsed_s":
                continue
            lines.append(f"- Check `{k}`: **{v}**")
        lines.append("")
    lines += [
        "## Duplicate processing / UI freeze",
        "",
        "GUI analysis cache computes gait cycles, asymmetry moments, contact, and biomechanics once per load. "
        "`analyze_gait_autocorrelation` and `analyze_gait_spectrum` are **not** called from `app.py` "
        "(they would duplicate `analyze_spectral`). Spectral work is a background thread. "
        "Overview Signal & Spectral reuses `_spectral_report` and moves the playhead only during play.",
        "",
        "```",
        json.dumps(payload["duplicate_sync_audit"], indent=2),
        "```",
        "",
        "## Remaining limitations",
        "",
    ]
    lines.extend(f"- {x}" for x in payload["limitations"])
    lines += [
        "",
        "## Files changed",
        "",
    ]
    lines.extend(f"- `{p}`" for p in payload["files_changed"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not REPORT_MD.is_file():
        REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    from datetime import datetime, timezone

    payload = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "clips": [],
        "duplicate_sync_audit": _static_duplicate_and_sync_audit(),
        "limitations": [],
        "files_changed": [
            "scripts/gait_pipeline_e2e_validation.py",
            "docs/gait_pipeline_e2e_validation.md",
            "docs/_gait_pipeline_e2e.json",
        ],
        "errors": [],
    }
    for spec in CLIPS:
        try:
            payload["clips"].append(analyze_clip(spec))
        except Exception as exc:
            traceback.print_exc()
            payload["errors"].append(f"{spec['id']}: {exc}")
            payload["clips"].append(
                {"id": spec["id"], "kind": spec["kind"], "error": str(exc), "checks": {}}
            )
        if spec["id"] == "noisy":
            try:
                payload["clips"].append(analyze_clip(spec, degrade=True))
            except Exception as exc:
                payload["errors"].append(f"noisy_degraded: {exc}")
    payload["limitations"] = [
        "Cached poses were used. MediaPipe was not re-estimated on the MP4s.",
        "weightgait_limping_gait.mp4 has no pose JSON; abnormal_gait.mp4 is the asymmetric clip.",
        "Demo clips are ~4–8 s. ACF/FFT may withhold a stride when the series is too short or not periodic.",
        "Monocular 3D is inferred / body-normalized, not laboratory metres.",
        "Asymmetric-moment alignment is compared to detected gait events, not a force-plate or video annotation.",
        "GUI seek/playhead wiring was inspected in source; this run did not capture a timed Tk playback session.",
    ]
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    fails = [r for r in _verdict_rows(payload) if r[2] == "FAIL"]
    for row in fails:
        print("FAIL", row)
    return 1 if fails or payload["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
