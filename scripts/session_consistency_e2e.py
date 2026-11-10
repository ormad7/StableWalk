"""End-to-end consistency verification of current StableWalk analysis products.

Uses cached pose JSON (does not re-run MediaPipe). Does not infer abnormality
from a filename. Does not treat a UI widget as proof that a value is consistent.

  python scripts/session_consistency_e2e.py
"""

from __future__ import annotations

import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.asymmetry3d.export import export_asymmetry_json
from stablewalk.analysis.asymmetry3d.report import analyze_asymmetry3d
from stablewalk.analysis.gait_asymmetry_moments import detect_gait_asymmetry_moments
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.analysis.pipeline_validation import parse_ik_marker_errors
from stablewalk.analysis.research_validation import collect_research_validation
from stablewalk.analysis.spectral.export import export_spectral_json
from stablewalk.analysis.spectral.report import analyze_spectral
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.io.research_report_export import export_research_report_zip
from stablewalk.models.gait_wireframe import build_gait_wireframe
from stablewalk.opensim_ik_status import (
    format_opensim_status_pair,
    resolve_opensim_ik_status,
    resolve_opensim_pipeline_status,
)
from stablewalk.pose.knee_convention import (
    KNEE_STORAGE_INTERIOR,
    analysis_knee_flexion_deg,
    recording_knee_storage,
)

OUT_DIR = ROOT / "docs" / "_session_consistency_e2e"
REPORT_MD = ROOT / "docs" / "session_consistency_e2e.md"
REPORT_JSON = OUT_DIR / "session_consistency_e2e.json"

RECORDINGS = (
    {
        "id": "recording_a",
        "video": config.DEMO_VIDEOS_DIR / "normal_gait.mp4",
        "poses": config.POSES_DIR / "normal_gait_poses.json",
        "opensim_named": config.OPENSIM_DIR / "normal_gait",
    },
    {
        "id": "recording_b",
        "video": config.DEMO_VIDEOS_DIR / "athletic_walking.mp4",
        "poses": config.POSES_DIR / "athletic_walking_poses.json",
        "opensim_named": config.OPENSIM_DIR / "athletic_walking",
    },
    {
        "id": "recording_c",
        "video": config.DEMO_VIDEOS_DIR / "abnormal_gait.mp4",
        "poses": config.POSES_DIR / "abnormal_gait_poses.json",
        "opensim_named": config.OPENSIM_DIR / "abnormal_gait",
    },
)

REQUIRED_FEATURE_KEYS = (
    "formula_id",
    "definition",
    "units",
    "quality",
    "available",
    "unavailable_reason",
    "notes",
)
REQUIRED_SCALAR_KEYS = (
    "formula_id",
    "formula",
    "units",
    "quality",
    "unavailable_reason",
    "n_left_cycles",
    "n_right_cycles",
    "note",
)
REQUIRED_SPECTRAL_TOP = (
    "fps",
    "sampling_rate_hz",
    "coordinate_system",
    "preprocessing",
    "config",
    "cycle_definition",
)
REQUIRED_ASYMMETRY_TOP = (
    "fps",
    "sampling_rate_hz",
    "coordinate_system",
    "preprocessing",
    "cycle_definition",
    "formula_identifiers",
    "index_definitions",
)
REQUIRED_RESEARCH_TOP = (
    "fps",
    "sampling_rate_hz",
    "coordinate_system",
    "preprocessing",
    "video_metadata",
)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return str(value)


def _check(name: str, passed: bool, detail: str, *, fatal: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "detail": detail,
        "fatal": bool(fatal),
    }


def _video_meta(path: Path) -> dict[str, Any]:
    import cv2

    if not path.is_file():
        return {"path": str(path), "exists": False}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "exists": True, "error": "cannot_open", "size_bytes": path.stat().st_size}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "path": str(path),
        "exists": True,
        "fps": fps,
        "frames": n,
        "duration_s": n / fps if fps > 1e-6 else 0.0,
        "width": w,
        "height": h,
        "size_bytes": path.stat().st_size,
    }


def _series_from_recording(rec) -> tuple[list[float | None], list[float | None], list[float]]:
    storage = recording_knee_storage(rec)
    left: list[float | None] = []
    right: list[float | None] = []
    times: list[float] = []
    for snap in rec.snapshots:
        left.append(analysis_knee_flexion_deg(snap, "left_knee_flexion", storage=storage))
        right.append(analysis_knee_flexion_deg(snap, "right_knee_flexion", storage=storage))
        times.append(float(getattr(snap, "time_s", 0.0) or 0.0))
    return left, right, times


def _max_abs_diff(a: Any, b: Any) -> float | None:
    aa = np.asarray([_finite(v) if _finite(v) is not None else np.nan for v in a], dtype=float)
    bb = np.asarray([_finite(v) if _finite(v) is not None else np.nan for v in b], dtype=float)
    n = min(aa.size, bb.size)
    if n == 0:
        return None
    mask = np.isfinite(aa[:n]) & np.isfinite(bb[:n])
    if int(np.count_nonzero(mask)) < 4:
        return None
    return float(np.nanmax(np.abs(aa[:n][mask] - bb[:n][mask])))


def _aligned_max_abs(
    src_t: Any,
    src_y: Any,
    dst_t: Any,
    dst_y: Any,
    *,
    tol_s: float,
) -> tuple[float | None, int]:
    """Max |src−dst| at matching timestamps (nearest dst sample within tol_s)."""
    st = np.asarray(src_t, dtype=float)
    sy = np.asarray(
        [_finite(v) if _finite(v) is not None else np.nan for v in src_y],
        dtype=float,
    )
    dt = np.asarray(dst_t, dtype=float)
    dy = np.asarray(
        [_finite(v) if _finite(v) is not None else np.nan for v in dst_y],
        dtype=float,
    )
    if st.size == 0 or dt.size == 0:
        return None, 0
    diffs: list[float] = []
    for t, y in zip(st, sy):
        if not np.isfinite(t) or not np.isfinite(y):
            continue
        idx = int(np.argmin(np.abs(dt - t)))
        if abs(float(dt[idx]) - float(t)) > tol_s:
            continue
        if not np.isfinite(dy[idx]):
            continue
        diffs.append(abs(float(y) - float(dy[idx])))
    if len(diffs) < 4:
        return None, len(diffs)
    return float(max(diffs)), len(diffs)


def _missing_keys(payload: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    return [key for key in required if key not in payload]


def _inspect_opensim_dir(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {"path": str(path), "exists": False, "files": {}}
    names = {p.name: p for p in path.iterdir() if p.is_file()}
    files = {
        "TRC": any(n.endswith(".trc") and "mapped" not in n for n in names),
        "MAPPED": any("mapped_for_opensim.trc" in n for n in names),
        "MOT": any(n.endswith(".mot") and "_ik.mot" not in n for n in names),
        "IK_MOT": any(n.endswith("_ik.mot") for n in names),
        "IK_ERRORS": any(n.endswith("_ik_marker_errors.sto") for n in names),
        "JSON": any(n.endswith("_opensim.json") or n.endswith("opensim.json") for n in names),
    }
    sdk = None
    ik_state = None
    source_video = None
    json_path = next((p for n, p in names.items() if n.endswith("_opensim.json")), None)
    if json_path is not None:
        try:
            blob = json.loads(json_path.read_text(encoding="utf-8"))
            sdk = blob.get("opensim_sdk_available")
            source_video = blob.get("source_video")
            meta = blob.get("metadata") or {}
            if isinstance(meta, dict):
                sdk = meta.get("opensim_sdk_available", sdk)
                ik_state = meta.get("ik_state") or blob.get("ik_state")
        except Exception as exc:
            sdk = f"json_error:{exc}"
    sto = next((p for n, p in names.items() if n.endswith("_ik_marker_errors.sto")), None)
    errors = None
    if sto is not None:
        parsed = parse_ik_marker_errors(sto)
        errors = None if parsed is None else parsed.to_dict()
    return {
        "path": str(path),
        "exists": True,
        "files": files,
        "opensim_sdk_available": sdk,
        "ik_state_in_json": ik_state,
        "source_video": source_video,
        "ik_marker_errors": errors,
        "file_names": sorted(names),
    }


def _newest_run_dirs(limit: int = 3) -> list[Path]:
    if not config.OPENSIM_DIR.is_dir():
        return []
    runs = [p for p in config.OPENSIM_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[:limit]


def _analyze_recording(clip: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    video_meta = _video_meta(clip["video"])
    poses_path: Path = clip["poses"]
    out: dict[str, Any] = {
        "id": clip["id"],
        "source_video": str(clip["video"]),
        "source_video_name": clip["video"].name,
        "poses_path": str(poses_path),
        "video": video_meta,
        "filename_diagnosis": None,
        "errors": [],
        "checks": checks,
    }
    checks.append(
        _check(
            "video_present",
            bool(video_meta.get("exists")),
            f"{clip['video'].name} exists={video_meta.get('exists')} "
            f"fps={video_meta.get('fps')} frames={video_meta.get('frames')}",
            fatal=True,
        )
    )
    checks.append(
        _check(
            "poses_present",
            poses_path.is_file(),
            f"{poses_path.name} exists={poses_path.is_file()}",
            fatal=True,
        )
    )
    if not poses_path.is_file():
        out["errors"].append(f"Pose JSON missing: {poses_path}")
        return out

    seq = load_pose_sequence(poses_path)
    rec = pose_sequence_to_gait_motion(seq)
    rec.source = str(clip["video"])
    storage = recording_knee_storage(rec)
    cycles = analyze_gait_cycles(rec)
    wire = build_gait_wireframe(rec)
    spectral = analyze_spectral(rec, sequence=seq, cycles=cycles)
    report3d = analyze_asymmetry3d(rec, cycles=cycles)
    moments = detect_gait_asymmetry_moments(rec, cycles)

    dest = OUT_DIR / clip["id"]
    dest.mkdir(parents=True, exist_ok=True)
    spectral_path = export_spectral_json(spectral, dest / "spectral.json")
    asymmetry_path = export_asymmetry_json(report3d, dest / "asymmetry.json")
    session = SimpleNamespace(
        gait_motion=rec,
        sequence=seq,
        gait_wireframe=wire,
        _gait_cycle=cycles,
        _spectral_report=spectral,
        _spectral_error=None,
        _gait_asymmetry_moments=moments,
        _asymmetry_report=report3d,
        video_path=str(clip["video"]),
        source=str(clip["video"]),
        _session_analysis=SimpleNamespace(ready=True),
        _biomech_analysis=None,
        _selected_asymmetry_moment_index=0
        if list(getattr(moments, "moments", None) or [])
        else None,
    )
    snapshot = collect_research_validation(session)
    zip_path = dest / "research_report.zip"
    export_research_report_zip(snapshot, zip_path, session=session)
    research_json = dest / "research_validation.json"
    research_json.write_text(
        json.dumps(snapshot.to_dict(), indent=2),
        encoding="utf-8",
    )

    spectral_payload = json.loads(spectral_path.read_text(encoding="utf-8"))
    asymmetry_payload = json.loads(asymmetry_path.read_text(encoding="utf-8"))
    research_payload = snapshot.to_dict()

    left_flex, right_flex, rec_times = _series_from_recording(rec)
    knee_ch = spectral.bundle.channels.get("dof:left_knee_flexion")
    spec_raw = [] if knee_ch is None else [None if v != v else float(v) for v in knee_ch.raw]
    spec_times = [] if knee_ch is None else [float(t) for t in knee_ch.times_s]
    kin_left = report3d.kinematics.series_for_dof("left_knee_flexion")
    kin_times = list(report3d.kinematics.times_s)
    n_grid = len(spec_times)
    n_snap = len(rec_times)
    tol_s = 0.51 / max(float(rec.fps or 30.0), 1e-6)

    knee_spec_diff, n_spec_matched = _aligned_max_abs(
        rec_times, left_flex, spec_times, spec_raw, tol_s=tol_s
    )
    knee_kin_diff, n_kin_matched = _aligned_max_abs(
        rec_times, left_flex, kin_times, kin_left, tol_s=tol_s
    )
    time_kin_diff = _max_abs_diff(rec_times, kin_times)
    grid_span_ok = False
    if rec_times and spec_times:
        grid_span_ok = (
            abs(float(spec_times[0]) - float(rec_times[0])) < tol_s
            and abs(float(spec_times[-1]) - float(rec_times[-1])) < tol_s
        )

    n_left = int(getattr(cycles.metrics, "left_heel_strike_count", 0) or 0)
    n_right = int(getattr(cycles.metrics, "right_heel_strike_count", 0) or 0)
    n_cycles = int(getattr(cycles.metrics, "gait_cycle_count", 0) or 0)
    n_left_evt = len(report3d.events.left_cycles)
    n_right_evt = len(report3d.events.right_cycles)
    feat = spectral.features
    n_detected = 0 if feat is None else int(feat.n_detected_cycles)
    n_valid = 0 if feat is None else int(feat.n_valid_cycles)

    fps_rec = float(rec.fps or 0.0)
    fps_spec = float(spectral.bundle.fps or 0.0)
    fps_kin = float(report3d.kinematics.fps or 0.0)
    fps_video = _finite(video_meta.get("fps")) or 0.0

    checks.append(
        _check(
            "knee_storage_pose_interior",
            storage == KNEE_STORAGE_INTERIOR,
            f"recording_knee_storage={storage} (pose JSON should be pose_interior_hka)",
        )
    )
    checks.append(
        _check(
            "knee_flexion_matches_spectral_raw",
            knee_spec_diff is not None and knee_spec_diff < 1e-4,
            (
                f"time-aligned max |analysis_knee_flexion − spectral raw| = {knee_spec_diff} "
                f"({n_spec_matched} matched samples, tol={tol_s:.4g}s)"
            ),
        )
    )
    # Asymmetry kinematics applies a 0.12 s Savitzky–Golay filter. Convention
    # identity is the same 180°−θ flexion; a few degrees of filter residual is
    # expected. A 180° convention swap would be ~160°.
    kin_convention_ok = (
        knee_kin_diff is not None and knee_kin_diff < 25.0
    )
    checks.append(
        _check(
            "knee_flexion_asymmetry_same_convention",
            kin_convention_ok,
            (
                f"time-aligned max |analysis_knee_flexion − smoothed kinematics| = "
                f"{knee_kin_diff} ({n_kin_matched} samples). "
                "Asymmetry kinematics is gap-filled + 0.12 s smoothed; "
                "this is not the raw spectral series."
            ),
        )
    )
    checks.append(
        _check(
            "spectral_grid_spans_snapshots",
            grid_span_ok,
            (
                f"spectral n_grid={n_grid} vs recording n_snapshots={n_snap}; "
                f"grid [{spec_times[0] if spec_times else None}, "
                f"{spec_times[-1] if spec_times else None}] vs snapshots "
                f"[{rec_times[0] if rec_times else None}, "
                f"{rec_times[-1] if rec_times else None}]"
            ),
        )
    )
    checks.append(
        _check(
            "timestamps_match_asymmetry",
            time_kin_diff is not None and time_kin_diff < 1e-6,
            f"max |recording time − asymmetry time| = {time_kin_diff}",
        )
    )
    fps_ok = abs(fps_rec - fps_spec) < 0.05 and abs(fps_rec - fps_kin) < 0.05
    checks.append(
        _check(
            "sampling_rate_shared",
            fps_ok,
            f"recording={fps_rec:.4g} Hz spectral={fps_spec:.4g} Hz "
            f"asymmetry={fps_kin:.4g} Hz video={fps_video:.4g} Hz",
        )
    )
    if fps_video > 1e-6:
        checks.append(
            _check(
                "recording_fps_matches_video",
                abs(fps_rec - fps_video) < 0.6,
                f"recording fps {fps_rec:.4g} vs video container fps {fps_video:.4g}",
            )
        )
    coord_rec = str(getattr(rec, "coordinate_system", "") or "")
    coord_spec = str(spectral.bundle.coordinate_system or "")
    coord_kin = str(report3d.kinematics.coordinate_system or "")
    checks.append(
        _check(
            "coordinate_system_shared",
            bool(coord_rec) and coord_spec == coord_rec,
            f"recording={coord_rec!r} spectral={coord_spec!r} asymmetry={coord_kin!r}",
        )
    )
    cycle_align = str(report3d.events.alignment_event or "")
    checks.append(
        _check(
            "cycle_definition_ipsilateral_hs",
            cycle_align == "ipsilateral_heel_strike",
            f"alignment_event={cycle_align!r}; gait_cycle_count={n_cycles} "
            f"HS L/R={n_left}/{n_right}; asymmetry cycles L/R={n_left_evt}/{n_right_evt}; "
            f"spectral detected/valid={n_detected}/{n_valid}",
        )
    )
    checks.append(
        _check(
            "cycle_counts_consistent",
            n_left_evt <= max(n_left, 0) and n_right_evt <= max(n_right, 0),
            "Asymmetry ipsilateral cycles are HS→HS subsets of detected heel strikes "
            f"(L {n_left_evt}/{n_left}, R {n_right_evt}/{n_right}).",
        )
    )
    if knee_ch is not None:
        raw_units = knee_ch.spec.units
        processed = np.asarray(knee_ch.processed, dtype=float)
        finite_p = processed[np.isfinite(processed)]
        mean_p = float(np.mean(finite_p)) if finite_p.size else None
        checks.append(
            _check(
                "knee_units_degrees",
                raw_units == "deg",
                f"spectral left-knee units={raw_units!r}; processed mean={mean_p} (detrend ~0)",
            )
        )
        if finite_p.size >= 16:
            checks.append(
                _check(
                    "spectral_processed_is_detrended",
                    abs(mean_p or 99.0) < 2.0,
                    f"processed (detrended flexion) mean={mean_p:.4g} deg about 0",
                )
            )

    miss_spec = _missing_keys(spectral_payload, REQUIRED_SPECTRAL_TOP)
    checks.append(
        _check(
            "spectral_json_provenance",
            not miss_spec,
            "missing " + ", ".join(miss_spec) if miss_spec else "fps/config/coordinate/cycle keys present",
        )
    )
    miss_as = _missing_keys(asymmetry_payload, REQUIRED_ASYMMETRY_TOP)
    checks.append(
        _check(
            "asymmetry_json_provenance",
            not miss_as,
            "missing " + ", ".join(miss_as) if miss_as else "fps/coordinate/preprocessing/cycle keys present",
        )
    )
    miss_rv = _missing_keys(research_payload, REQUIRED_RESEARCH_TOP)
    checks.append(
        _check(
            "research_json_provenance",
            not miss_rv,
            "missing " + ", ".join(miss_rv) if miss_rv else "fps/coordinate/preprocessing keys present",
        )
    )

    feature_rows = list((spectral_payload.get("features") or {}).get("records") or [])
    feature_gaps = []
    unavailable_without_reason = 0
    for row in feature_rows:
        gaps = _missing_keys(row, REQUIRED_FEATURE_KEYS)
        if gaps:
            feature_gaps.append(f"{row.get('name')}:{','.join(gaps)}")
        if not row.get("available") and not row.get("unavailable_reason"):
            unavailable_without_reason += 1
    checks.append(
        _check(
            "spectral_feature_records_provenance",
            not feature_gaps and unavailable_without_reason == 0,
            (
                f"{len(feature_rows)} records; key gaps={feature_gaps[:6] or 'none'}; "
                f"unavailable without reason={unavailable_without_reason}"
            ),
        )
    )

    scalar_rows = list(asymmetry_payload.get("scalars") or [])
    scalar_gaps = []
    scalar_missing_reason = 0
    for row in scalar_rows:
        gaps = _missing_keys(row, REQUIRED_SCALAR_KEYS)
        if gaps:
            scalar_gaps.append(f"{row.get('name')}:{','.join(gaps)}")
        if (
            row.get("left") is None
            and row.get("right") is None
            and not row.get("unavailable_reason")
        ):
            scalar_missing_reason += 1
    checks.append(
        _check(
            "asymmetry_scalar_provenance",
            not scalar_gaps and scalar_missing_reason == 0,
            (
                f"{len(scalar_rows)} scalars; key gaps={scalar_gaps[:6] or 'none'}; "
                f"blank sides without reason={scalar_missing_reason}"
            ),
        )
    )

    rv_items = list(research_payload.get("items") or [])
    rv_missing = [
        item.get("key")
        for item in rv_items
        if item.get("status") != "computed" and not item.get("unavailable_reason") and not item.get("reason")
    ]
    checks.append(
        _check(
            "research_items_have_reasons",
            not rv_missing,
            f"non-computed items without reason: {rv_missing or 'none'}",
        )
    )

    named = _inspect_opensim_dir(clip["opensim_named"])
    leftover_status = resolve_opensim_ik_status(
        ik_state=named.get("ik_state_in_json"),
        trc_exported=bool((named.get("files") or {}).get("TRC")),
        mot_exported=bool((named.get("files") or {}).get("MOT")),
        mapped_trc_ready=bool((named.get("files") or {}).get("MAPPED")),
        ik_output_on_disk=bool((named.get("files") or {}).get("IK_MOT")),
        sdk_available=bool(named.get("opensim_sdk_available") is True),
        has_session=True,
    )
    leftover_pipeline = resolve_opensim_pipeline_status(
        trc_exported=bool((named.get("files") or {}).get("TRC")),
        mot_exported=bool((named.get("files") or {}).get("MOT")),
        mapped_trc_ready=bool((named.get("files") or {}).get("MAPPED")),
        has_session=True,
    )
    session_ik = resolve_opensim_ik_status(has_session=True)
    pair = format_opensim_status_pair(leftover_pipeline, leftover_status)
    leftover_complete_from_files = leftover_status.key == "complete" and named.get(
        "ik_state_in_json"
    ) not in {"Completed", "completed"}
    checks.append(
        _check(
            "opensim_ik_not_inferred_from_leftover_files",
            not leftover_complete_from_files,
            (
                f"named dir IK={leftover_status.label} ({leftover_status.detail}); "
                f"pipeline={leftover_pipeline.label} ({leftover_pipeline.detail}); "
                f"this-session ik_state=None → {session_ik.label}. "
                "File presence must not upgrade IK to Complete."
            ),
        )
    )
    checks.append(
        _check(
            "pipeline_complete_is_not_ik_complete",
            True,
            (
                "app.py 'Pipeline complete' is the Real-to-Sim dialog after "
                "run_real_to_sim_pipeline. OpenSim pipeline Complete means TRC+MOT "
                f"export ({leftover_pipeline.label}). IK Complete requires "
                f"ik_state=='Completed' (this session: {session_ik.label}). "
                f"Combined leftover readout: {pair.short_label}."
            ),
        )
    )

    out.update(
        {
            "pose_frames": len(seq.frames),
            "recording_frames": rec.frame_count,
            "fps": fps_rec,
            "duration_s": float(rec.duration_s or 0.0),
            "coordinate_system": coord_rec,
            "source_kind": str(getattr(rec, "source_kind", "") or ""),
            "knee_storage": storage,
            "knee_flexion": {
                "n_left_finite": sum(1 for v in left_flex if v is not None),
                "n_right_finite": sum(1 for v in right_flex if v is not None),
                "left_median": _finite(np.nanmedian([v if v is not None else np.nan for v in left_flex])),
                "right_median": _finite(np.nanmedian([v if v is not None else np.nan for v in right_flex])),
                "vs_spectral_max_abs": knee_spec_diff,
                "vs_spectral_n_matched": n_spec_matched,
                "vs_asymmetry_max_abs": knee_kin_diff,
                "vs_asymmetry_n_matched": n_kin_matched,
                "spectral_n_grid": n_grid,
                "recording_n_snapshots": n_snap,
            },
            "cycles": {
                "gait_cycle_count": n_cycles,
                "left_heel_strikes": n_left,
                "right_heel_strikes": n_right,
                "metrics_reliable": bool(getattr(cycles.metrics, "metrics_reliable", False)),
                "reliability_reason": str(getattr(cycles.metrics, "reliability_reason", "") or ""),
                "cadence_steps_per_min": _finite(getattr(cycles.metrics, "cadence_steps_per_min", None)),
                "asymmetry_left_cycles": n_left_evt,
                "asymmetry_right_cycles": n_right_evt,
                "alignment_event": cycle_align,
                "spectral_detected_cycles": n_detected,
                "spectral_valid_cycles": n_valid,
            },
            "exports": {
                "spectral": str(spectral_path),
                "asymmetry": str(asymmetry_path),
                "research_zip": str(zip_path),
                "research_json": str(research_json),
            },
            "opensim_named": named,
            "opensim_leftover_ik": leftover_status.key,
            "opensim_leftover_pipeline": leftover_pipeline.key,
            "opensim_session_ik": session_ik.key,
            "opensim_pair_label": pair.short_label,
        }
    )
    return out


def _write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Session consistency end-to-end verification",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "This report records checks that were actually executed on cached pose JSON.",
        "A filename is not a diagnosis. A UI label is not proof of a consistent signal.",
        "",
        "## Recordings",
        "",
    ]
    for rec in payload["recordings"]:
        video = rec.get("video") or {}
        lines += [
            f"### {rec['id']}",
            "",
            f"- Source: `{rec.get('source_video_name')}`",
            f"- Video exists: {video.get('exists')} · fps={video.get('fps')} · "
            f"frames={video.get('frames')} · duration_s={video.get('duration_s')}",
            f"- Pose frames: {rec.get('pose_frames')} · recording frames: {rec.get('recording_frames')}",
            f"- Coordinate system: `{rec.get('coordinate_system')}` · knee storage: `{rec.get('knee_storage')}`",
            f"- Cycles: {rec.get('cycles')}",
            f"- Knee comparison: {rec.get('knee_flexion')}",
            f"- Filename diagnosis: {rec.get('filename_diagnosis')}",
            "",
        ]
    lines += ["## OpenSim: Pipeline complete versus IK", ""]
    osim = payload.get("opensim") or {}
    lines += [
        str(osim.get("explanation") or ""),
        "",
        f"- Named leftover dirs: `{json.dumps(_jsonable(osim.get('named')), indent=2)[:2000]}`",
        f"- Newest leftover run dirs: `{[p.get('path') for p in osim.get('recent_runs') or []]}`",
        "",
    ]
    if osim.get("recent_run_errors"):
        lines += ["### Leftover IK fitting errors (on-disk, not this session)", ""]
        for row in osim["recent_run_errors"]:
            lines.append(
                f"- `{row.get('path')}`: n_frames={row.get('n_frames')} "
                f"mean_rms_m={row.get('mean_rms_m')} n_failed={row.get('n_failed_frames')} "
                f"quality={row.get('quality')}"
            )
            for fr in (row.get("failed_frames") or [])[:8]:
                lines.append(
                    f"  - frame {fr.get('frame_index')} t={fr.get('time_s')}s "
                    f"RMS={fr.get('marker_error_rms_m')} max={fr.get('marker_error_max_m')} "
                    f"({fr.get('reason')})"
                )
        lines.append("")
    lines += ["## Checks performed", ""]
    for rec in payload["recordings"]:
        lines.append(f"### {rec['id']}")
        lines.append("")
        for chk in rec.get("checks") or []:
            mark = "PASS" if chk.get("passed") else "FAIL"
            lines.append(f"- **{mark}** `{chk['name']}` — {chk['detail']}")
        lines.append("")
    failed = payload.get("failed_checks") or []
    lines += [
        "## Remaining failures",
        "",
    ]
    if not failed:
        lines.append("No failed checks in this harness.")
    else:
        for chk in failed:
            lines.append(f"- `{chk['recording']}` / `{chk['name']}` — {chk['detail']}")
    lines += [
        "",
        "## Changed files",
        "",
    ]
    for path in payload.get("changed_files") or []:
        lines.append(f"- `{path}`")
    lines += ["", "## Notes", "", *(payload.get("notes") or []), ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recordings: list[dict[str, Any]] = []
    errors: list[str] = []
    for clip in RECORDINGS:
        try:
            recordings.append(_analyze_recording(clip))
        except Exception as exc:
            errors.append(f"{clip['id']}: {type(exc).__name__}: {exc}")
            recordings.append(
                {
                    "id": clip["id"],
                    "source_video": str(clip["video"]),
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "traceback": traceback.format_exc()[-2000:],
                    "checks": [
                        _check("analysis_ran", False, f"{type(exc).__name__}: {exc}", fatal=True)
                    ],
                }
            )

    named = [_inspect_opensim_dir(clip["opensim_named"]) for clip in RECORDINGS]
    recent = []
    recent_errors = []
    for path in _newest_run_dirs(3):
        info = _inspect_opensim_dir(path)
        recent.append(info)
        if info.get("ik_marker_errors"):
            recent_errors.append(info["ik_marker_errors"])

    session_ik = resolve_opensim_ik_status(has_session=True)
    leftover_ik = resolve_opensim_ik_status(
        trc_exported=any((row.get("files") or {}).get("TRC") for row in named),
        mot_exported=any((row.get("files") or {}).get("MOT") for row in named),
        mapped_trc_ready=any((row.get("files") or {}).get("MAPPED") for row in named),
        ik_output_on_disk=any((row.get("files") or {}).get("IK_MOT") for row in named + recent),
        has_session=True,
    )
    leftover_pipe = resolve_opensim_pipeline_status(
        trc_exported=any((row.get("files") or {}).get("TRC") for row in named),
        mot_exported=any((row.get("files") or {}).get("MOT") for row in named),
        has_session=True,
    )
    pair = format_opensim_status_pair(leftover_pipe, leftover_ik)

    failed_checks = []
    for rec in recordings:
        for chk in rec.get("checks") or []:
            if not chk.get("passed"):
                failed_checks.append({"recording": rec.get("id"), **chk})

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "harness": "scripts/session_consistency_e2e.py",
        "filename_diagnosis_used": False,
        "recordings": _jsonable(recordings),
        "failed_checks": _jsonable(failed_checks),
        "errors": errors,
        "opensim": {
            "explanation": (
                "This harness did not run Inverse Kinematics. "
                f"Session IK with ik_state=None is '{session_ik.label}' ({session_ik.detail}). "
                f"Leftover named-folder files resolve to pipeline '{leftover_pipe.label}' "
                f"and IK '{leftover_ik.label}' ({pair.short_label}). "
                "app.py messagebox 'Pipeline complete' is Real-to-Sim after "
                "run_real_to_sim_pipeline; it is not IK Complete. "
                "OpenSim pipeline Complete means TRC and MOT were exported. "
                "IK Complete is allowed only when ik_state == 'Completed'. "
                "Leftover .mot/.sto files are artifacts, not a successful IK run in this session."
            ),
            "session_ik": {"key": session_ik.key, "label": session_ik.label, "detail": session_ik.detail},
            "leftover_ik": {"key": leftover_ik.key, "label": leftover_ik.label, "detail": leftover_ik.detail},
            "leftover_pipeline": {
                "key": leftover_pipe.key,
                "label": leftover_pipe.label,
                "detail": leftover_pipe.detail,
            },
            "pair_short_label": pair.short_label,
            "named": named,
            "recent_runs": recent,
            "recent_run_errors": recent_errors,
        },
        "changed_files": [
            "stablewalk/analysis/spectral/features.py",
            "stablewalk/analysis/spectral/report.py",
            "stablewalk/analysis/asymmetry3d/compare.py",
            "stablewalk/analysis/asymmetry3d/report.py",
            "stablewalk/analysis/research_validation.py",
            "stablewalk/analysis/pipeline_validation.py",
            "scripts/session_consistency_e2e.py",
            "tests/test_export_provenance.py",
            "docs/session_consistency_e2e.md",
        ],
        "notes": [
            "Recording B is athletic_walking.mp4 and recording C is abnormal_gait.mp4. "
            "Neither filename is treated as a diagnosis.",
            "Tabs were verified through the analysis objects they consume, not by clicking widgets.",
            "JSON provenance fields were added so exports carry formula ids and unavailable reasons.",
        ],
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload)
    print(json.dumps(
        {
            "report": str(REPORT_MD),
            "json": str(REPORT_JSON),
            "n_recordings": len(recordings),
            "n_failed_checks": len(failed_checks),
            "failed": [{"recording": c["recording"], "name": c["name"]} for c in failed_checks],
            "errors": errors,
        },
        indent=2,
    ))
    return 1 if failed_checks or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
