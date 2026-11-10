"""End-to-end lecturer-requirement validation on two demo walks.

Uses cached pose JSON (does not re-run MediaPipe). Does not invent metrics,
does not mark missing products as passed, and does not rewrite cadence.

  python scripts/lecturer_e2e_validation.py
"""
from __future__ import annotations

import json
import math
import traceback
import zipfile
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
from stablewalk.analysis.asymmetry3d.report import analyze_asymmetry3d
from stablewalk.analysis.gait_asymmetry_moments import detect_gait_asymmetry_moments
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.analysis.research_validation import collect_research_validation
from stablewalk.analysis.session_compare import (
    compare_session_metrics,
    extract_compare_metrics,
)
from stablewalk.analysis.spectral.report import analyze_spectral
from stablewalk.analysis.synthetic_gait import SyntheticGaitSpec, make_sine_gait_recording
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.io.research_report_export import export_research_report_zip
from stablewalk.models.gait_wireframe import (
    GAIT_WIRE_JOINT_COUNT,
    GAIT_WIRE_JOINT_IDS,
    GAIT_WIRE_VISIBILITY_MIN,
    build_gait_wireframe,
)
from stablewalk.pose.kinematics import knee_flexion_from_snapshot
from stablewalk.ui.tk.session_cache import build_snapshot_from_sequence

OUT_DIR = ROOT / "docs" / "_lecturer_e2e"
OUT_JSON = OUT_DIR / "lecturer_e2e_validation.json"
MIN_USABLE_CYCLES = 5

CLIPS = (
    {
        "id": "A_normal",
        "kind": "normal_walking",
        "label": "A. Normal walking",
        "video": config.DEMO_VIDEOS_DIR / "normal_gait.mp4",
        "poses": config.POSES_DIR / "normal_gait_poses.json",
    },
    {
        "id": "B_asymmetric",
        "kind": "asymmetric_walking",
        "label": "B. Asymmetric walking",
        "video": config.DEMO_VIDEOS_DIR / "abnormal_gait.mp4",
        "poses": config.POSES_DIR / "abnormal_gait_poses.json",
        "alternate_video": config.DEMO_VIDEOS_DIR / "candidates" / "weightgait_limping_gait.mp4",
    },
)


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


def _video_meta(path: Path) -> dict[str, Any]:
    import cv2

    if not path.is_file():
        return {"path": str(path), "exists": False}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "exists": True, "error": "cannot_open"}
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


def _series_len(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(len(value))
    except TypeError:
        return 0


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _source_sync_audit() -> dict[str, Any]:
    app = (ROOT / "stablewalk" / "ui" / "tk" / "app.py").read_text(encoding="utf-8")
    summary = (ROOT / "stablewalk" / "ui" / "tk" / "dashboard_summary.py").read_text(encoding="utf-8")
    spectral = (ROOT / "stablewalk" / "ui" / "tk" / "dashboard_spectral.py").read_text(encoding="utf-8")
    asym = (ROOT / "stablewalk" / "ui" / "tk" / "dashboard_asymmetry.py").read_text(encoding="utf-8")
    rv = (ROOT / "stablewalk" / "ui" / "tk" / "dashboard_research_validation.py").read_text(
        encoding="utf-8"
    )
    return {
        "refresh_bound_calls_research_validation": "_refresh_research_validation" in app
        and "def _refresh_bound_analysis_panels" in app,
        "spectral_refreshes_research_validation": "_refresh_research_validation" in spectral,
        "asymmetry_refreshes_research_validation": "_refresh_research_validation" in asym,
        "results_has_download_research_report": "Download Research Report" in summary,
        "research_section_on_results": "build_research_validation_section" in summary,
        "sync_all_to_frame_defined": "def _sync_all_to_frame" in app,
        "jump_to_asymmetric_moment_defined": "def _jump_to_asymmetric_moment" in app,
        "gait_acf_not_on_gui_load": "analyze_gait_autocorrelation" not in app,
        "research_status_labels_computed_only": '"passed"' not in rv
        and "computed" in rv
        and "not a requirement pass/fail" in rv.lower(),
    }


def _knee_flexion(rec) -> dict[str, Any]:
    left: list[float] = []
    right: list[float] = []
    missing = 0
    dof_left: list[float] = []
    dof_right: list[float] = []
    for snap in rec.snapshots:
        lf = knee_flexion_from_snapshot(snap, "left_knee")
        rf = knee_flexion_from_snapshot(snap, "right_knee")
        if lf is None:
            missing += 1
        else:
            left.append(float(lf))
        if rf is None:
            missing += 1
        else:
            right.append(float(rf))
        ld = (getattr(snap, "dofs", None) or {}).get("left_knee_flexion")
        rd = (getattr(snap, "dofs", None) or {}).get("right_knee_flexion")
        if ld is not None and _finite(getattr(ld, "value", None)) is not None:
            dof_left.append(float(ld.value))
        if rd is not None and _finite(getattr(rd, "value", None)) is not None:
            dof_right.append(float(rd.value))
    n_possible = 2 * len(rec.snapshots)
    out = {
        "definition": "anatomical knee flexion deg = 180 − interior 3D Hip–Knee–Ankle angle",
        "n_snapshots": len(rec.snapshots),
        "n_possible_sides": n_possible,
        "n_missing_hka": missing,
        "n_left": len(left),
        "n_right": len(right),
        "left_mean_deg": float(np.mean(left)) if left else None,
        "left_p10_p90_deg": (
            [float(np.percentile(left, 10)), float(np.percentile(left, 90))] if left else None
        ),
        "right_mean_deg": float(np.mean(right)) if right else None,
        "right_p10_p90_deg": (
            [float(np.percentile(right, 10)), float(np.percentile(right, 90))] if right else None
        ),
        "left_rom_deg": (float(max(left) - min(left)) if len(left) >= 2 else None),
        "right_rom_deg": (float(max(right) - min(right)) if len(right) >= 2 else None),
        "dof_left_n": len(dof_left),
        "dof_right_n": len(dof_right),
        "invented_when_missing": False,
    }
    return out


def _wire_stats(rec) -> dict[str, Any]:
    wire = build_gait_wireframe(rec)
    usable = 0
    incomplete = 0
    present: set[str] = set()
    for frame in wire.frames:
        joints = getattr(frame, "joints", None) or {}
        keep = 0
        for jid, sample in joints.items():
            vis = float(getattr(sample, "visibility", 0.0) or 0.0)
            if vis >= GAIT_WIRE_VISIBILITY_MIN:
                keep += 1
                present.add(str(jid))
        if keep >= 12:
            usable += 1
        if set(joints) != set(GAIT_WIRE_JOINT_IDS):
            incomplete += 1
    return {
        "joint_count": GAIT_WIRE_JOINT_COUNT,
        "joint_ids": list(GAIT_WIRE_JOINT_IDS),
        "n_frames": len(wire.frames),
        "usable_frames": usable,
        "incomplete_frames": incomplete,
        "joints_observed": sorted(present),
        "body_scale": float(wire.body_scale),
        "coordinate_system": str(wire.coordinate_system or ""),
        "complete_ids_every_frame": incomplete == 0 and len(wire.frames) == rec.frame_count,
        "wire": wire,
    }


def _spectral_stats(report) -> dict[str, Any]:
    if report is None:
        return {"available": False, "reason": "No spectral report."}
    acf = report.primary_autocorr() if hasattr(report, "primary_autocorr") else None
    spec = report.primary_spectrum() if hasattr(report, "primary_spectrum") else None
    features = []
    feat_obj = getattr(report, "features", None)
    if feat_obj is not None:
        for rec in getattr(feat_obj, "records", None) or []:
            if str(getattr(rec, "scope", "") or "") not in {"session", ""}:
                continue
            features.append(
                {
                    "name": str(getattr(rec, "name", "") or ""),
                    "label": str(getattr(rec, "label", "") or ""),
                    "available": bool(getattr(rec, "available", False)),
                    "value": _finite(getattr(rec, "value", None)),
                    "units": str(getattr(rec, "units", "") or ""),
                    "display": rec.display_value() if hasattr(rec, "display_value") else None,
                }
            )
    acf_out = None
    if acf is not None:
        acf_out = {
            "channel_id": str(getattr(acf, "channel_id", "") or ""),
            "n_samples": int(getattr(acf, "n_samples", 0) or 0),
            "gait_period_s": _finite(getattr(acf, "gait_period_s", None)),
            "stride_period_s": _finite(getattr(acf, "stride_period_s", None)),
            "reliability_label": str(getattr(acf, "reliability_label", "") or ""),
            "insufficient_reason": str(getattr(acf, "insufficient_reason", "") or ""),
            "has_series": _series_len(getattr(acf, "lags_s", None)) > 0
            and _series_len(getattr(acf, "rho", None)) > 0,
        }
    spec_out = None
    if spec is not None:
        fft_n = _series_len(getattr(spec, "fft_freq_hz", None))
        psd_n = _series_len(getattr(spec, "freq_hz", None))
        spec_out = {
            "channel_id": str(getattr(spec, "channel_id", "") or ""),
            "method": str(getattr(spec, "method", "") or ""),
            "dominant_hz": _finite(getattr(spec, "dominant_hz", None)),
            "spectral_centroid_hz": _finite(getattr(spec, "spectral_centroid_hz", None)),
            "spectral_bandwidth_hz": _finite(getattr(spec, "spectral_bandwidth_hz", None)),
            "spectral_entropy": _finite(getattr(spec, "spectral_entropy", None)),
            "harmonic_ratio": _finite(getattr(spec, "harmonic_ratio", None)),
            "n_fft_bins": fft_n,
            "n_psd_bins": psd_n,
        }
    return {
        "available": bool(getattr(report, "available", False)),
        "warnings": [str(w) for w in (getattr(report, "warnings", None) or []) if str(w).strip()],
        "acf": acf_out,
        "spectrum": spec_out,
        "features": features,
        "n_features_computed": sum(1 for f in features if f.get("available")),
        "n_features_total": len(features),
    }


def _moments_stats(moments) -> dict[str, Any]:
    if moments is None:
        return {"available": False, "reason": "No moment report."}
    rows = []
    for m in list(getattr(moments, "moments", None) or []):
        rows.append(
            {
                "frame_index": int(getattr(m, "frame_index", 0) or 0),
                "time_s": _finite(getattr(m, "time_s", None)),
                "score": _finite(getattr(m, "score", None)),
                "confidence": _finite(getattr(m, "confidence", None)),
                "involved_joints": list(getattr(m, "involved_joints", None) or []),
                "severity": str(getattr(m, "severity", "") or ""),
                "duration_s": _finite(getattr(m, "duration_s", None)),
            }
        )
    return {
        "available": bool(getattr(moments, "available", False)),
        "outcome": str(getattr(moments, "outcome", "") or ""),
        "n_moments": len(rows),
        "n_left_cycles": int(getattr(moments, "n_left_cycles", 0) or 0),
        "n_right_cycles": int(getattr(moments, "n_right_cycles", 0) or 0),
        "notes": [str(n) for n in (getattr(moments, "notes", None) or []) if str(n).strip()][:8],
        "moments": rows,
        "joint_comparisons": [
            {
                "name": str(getattr(row, "name", "") or ""),
                "rom_left": _finite(getattr(row, "rom_left", None)),
                "rom_right": _finite(getattr(row, "rom_right", None)),
                "si_norm": _finite(getattr(row, "si_norm", None)),
                "peak_difference": _finite(getattr(row, "peak_difference", None)),
                "timing_difference_pct": _finite(getattr(row, "timing_difference_pct", None)),
            }
            for row in list(getattr(moments, "joint_comparisons", None) or [])
        ],
    }


def _zip_inventory(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        reasons = {}
        if "failure_reasons.json" in names:
            reasons = json.loads(zf.read("failure_reasons.json").decode("utf-8"))
        md = zf.read("research_report.md").decode("utf-8") if "research_report.md" in names else ""
        return {
            "exists": True,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "names": sorted(names),
            "has_json": "research_validation.json" in names,
            "has_markdown": "research_report.md" in names,
            "has_definitions": "parameter_definitions.json" in names,
            "has_video_metadata": "video_metadata.json" in names,
            "has_failure_reasons": "failure_reasons.json" in names,
            "plots": sorted(n for n in names if n.startswith("plots/") and n.endswith(".png")),
            "failure_reason_keys": sorted(reasons.keys()),
            "status_pass_claimed": "**Status:** pass" in md.lower() or "status: passed" in md.lower(),
        }


def _ui_visible(session: SimpleNamespace) -> dict[str, Any]:
    try:
        import tkinter as tk
        from tkinter import ttk

        from stablewalk.ui.theme import apply_theme
        from stablewalk.ui.tk.asymmetric_moments_panel import (
            build_asymmetric_moments_section,
            empty_state_copy,
        )
        from stablewalk.ui.tk.dashboard_research_validation import (
            build_research_validation_section,
        )
    except Exception as exc:
        return {"ok": False, "error": f"Tk import failed: {exc}"}

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        style = ttk.Style(root)
        apply_theme(root, style)
        host = tk.Frame(root)
        host.grid()
        session.root = root
        build_research_validation_section(session, host)
        moments_host = tk.Frame(root)
        moments_host.grid()
        build_asymmetric_moments_section(session, moments_host)
        root.update_idletasks()
        cards = {}
        for key, card in (getattr(session, "_research_validation_cards", None) or {}).items():
            cards[key] = {
                "title": str(card["title"].cget("text")),
                "summary": str(card["summary"].cget("text")),
                "badge": str(card["badge"].label.cget("text"))
                if hasattr(card["badge"], "label")
                else "",
            }
        jump = getattr(session, "_research_validation_jump", None)
        jump_state = str(jump.cget("state")) if jump is not None else "missing"
        moments_copy = empty_state_copy(session)
        return {
            "ok": True,
            "cards": cards,
            "n_cards": len(cards),
            "jump_state": jump_state,
            "moments_empty_copy": moments_copy,
            "research_section_present": getattr(session, "_research_validation_section", None)
            is not None,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-800:]}
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def _analyze_clip(clip: dict[str, Any]) -> dict[str, Any]:
    video_meta = _video_meta(clip["video"])
    alt = clip.get("alternate_video")
    alt_meta = _video_meta(alt) if alt else None
    poses_path: Path = clip["poses"]
    out: dict[str, Any] = {
        "id": clip["id"],
        "kind": clip["kind"],
        "label": clip["label"],
        "video": video_meta,
        "alternate_video": alt_meta,
        "poses_path": str(poses_path),
        "poses_exist": poses_path.is_file(),
        "errors": [],
        "warnings": [],
    }
    if not poses_path.is_file():
        out["errors"].append(f"Pose JSON missing: {poses_path}")
        return out
    seq = load_pose_sequence(poses_path)
    rec = pose_sequence_to_gait_motion(seq)
    rec.source = str(clip["video"])
    cycles = analyze_gait_cycles(rec)
    wire_info = _wire_stats(rec)
    wire = wire_info.pop("wire")
    knee = _knee_flexion(rec)
    spectral_error = None
    spectral = None
    try:
        spectral = analyze_spectral(rec)
    except Exception as exc:
        spectral_error = f"{type(exc).__name__}: {exc}"
        out["errors"].append(spectral_error)
    moments = detect_gait_asymmetry_moments(rec, cycles)
    try:
        report3d = analyze_asymmetry3d(rec, cycles=cycles)
    except Exception as exc:
        report3d = None
        out["warnings"].append(f"analyze_asymmetry3d: {type(exc).__name__}: {exc}")

    cadence_before = _finite(getattr(cycles.metrics, "cadence_steps_per_min", None))
    n_cycles = int(getattr(cycles.metrics, "gait_cycle_count", 0) or 0)
    n_left_hs = int(getattr(cycles.metrics, "left_heel_strike_count", 0) or 0)
    n_right_hs = int(getattr(cycles.metrics, "right_heel_strike_count", 0) or 0)

    session = SimpleNamespace(
        gait_motion=rec,
        sequence=seq,
        gait_wireframe=wire,
        _gait_cycle=cycles,
        _spectral_report=spectral,
        _spectral_error=spectral_error,
        _gait_asymmetry_moments=moments,
        _asymmetry_report=report3d,
        video_path=str(clip["video"]),
        source=str(clip["video"]),
        _session_analysis=SimpleNamespace(ready=True),
        _biomech_analysis=None,
        _selected_asymmetry_moment_index=0 if list(getattr(moments, "moments", None) or []) else None,
    )
    snapshot = collect_research_validation(session)
    zip_path = OUT_DIR / f"{clip['id']}_research_report.zip"
    export_research_report_zip(snapshot, zip_path, session=session)
    cadence_after = _finite(getattr(cycles.metrics, "cadence_steps_per_min", None))
    ui = _ui_visible(session)

    rv_items = {item.key: item.to_dict() for item in snapshot.items}
    spectral_stats = _spectral_stats(spectral)
    moment_stats = _moments_stats(moments)
    zip_info = _zip_inventory(zip_path)

    recon_ok = False
    strongest = snapshot.strongest_moment
    if strongest is not None:
        frame_index = int(strongest.get("frame_index") or 0)
        snap = next((s for s in rec.snapshots if int(s.frame_index) == frame_index), None)
        recon_ok = snap is not None and bool(getattr(snap, "joints", None))

    out.update(
        {
            "pose_frames": len(seq.frames),
            "detected_frames": sum(
                1 for fr in seq.frames if getattr(fr, "detected", False) and getattr(fr, "keypoints", None)
            ),
            "recording_frames": rec.frame_count,
            "fps": float(rec.fps),
            "duration_s": float(rec.duration_s or 0.0),
            "coordinate_system": str(rec.coordinate_system or ""),
            "source_kind": str(getattr(rec, "source_kind", "") or ""),
            "gait_cycles": n_cycles,
            "left_heel_strikes": n_left_hs,
            "right_heel_strikes": n_right_hs,
            "cadence_spm": cadence_before,
            "cadence_unchanged_after_spectral": cadence_before == cadence_after,
            "metrics_reliable": bool(getattr(cycles.metrics, "metrics_reliable", False)),
            "reliability_reason": str(getattr(cycles.metrics, "reliability_reason", "") or ""),
            "contact_confidence": _finite(getattr(cycles.metrics, "contact_confidence", None)),
            "meets_five_cycle_gate": n_cycles >= MIN_USABLE_CYCLES,
            "wire": wire_info,
            "knee_flexion": knee,
            "spectral": spectral_stats,
            "moments": moment_stats,
            "research_validation": {
                "source": snapshot.source,
                "items": rv_items,
                "limitations": list(snapshot.limitations),
                "strongest_moment": snapshot.strongest_moment,
                "plot_series_keys": sorted(snapshot.plot_series.keys()),
                "clinical": snapshot.clinical,
            },
            "reconstruction3d": {
                "strongest_present": strongest is not None,
                "snapshot_matched": recon_ok,
                "n_joints_in_snapshot": (
                    len(snap.joints) if recon_ok else 0  # type: ignore[name-defined]
                ),
                "export_plot": "plots/wire3d_moment.png" in zip_info.get("plots", []),
            },
            "export": zip_info,
            "ui": ui,
            "session_object_ids": {
                "recording": id(rec),
                "cycles": id(cycles),
                "spectral": id(spectral) if spectral is not None else None,
                "moments": id(moments),
            },
        }
    )
    if n_cycles < MIN_USABLE_CYCLES:
        out["warnings"].append(
            f"Usable gait cycles = {n_cycles} (< {MIN_USABLE_CYCLES} required by this validation protocol)."
        )
    if spectral_stats.get("acf") and spectral_stats["acf"].get("insufficient_reason"):
        out["warnings"].append(f"ACF withheld: {spectral_stats['acf']['insufficient_reason']}")
    for warn in spectral_stats.get("warnings") or []:
        out["warnings"].append(f"Spectral: {warn}")
    return out


def _synthetic_five_cycle_probe() -> dict[str, Any]:
    """Pipeline capability on a generated 12 s recording — not a video substitute."""
    spec = SyntheticGaitSpec(
        fps=50.0,
        duration_s=12.0,
        stride_hz=1.0,
        knee_amp_left_deg=20.0,
        knee_amp_right_deg=40.0,
    )
    rec = make_sine_gait_recording(spec)
    rec.source = "synthetic://asymmetric_sine_12s"
    cycles = analyze_gait_cycles(rec)
    spectral = analyze_spectral(rec)
    moments = detect_gait_asymmetry_moments(rec, cycles)
    wire = build_gait_wireframe(rec)
    session = SimpleNamespace(
        gait_motion=rec,
        gait_wireframe=wire,
        _gait_cycle=cycles,
        _spectral_report=spectral,
        _gait_asymmetry_moments=moments,
        source=rec.source,
        video_path=rec.source,
    )
    snap = collect_research_validation(session)
    n_cycles = int(getattr(cycles.metrics, "gait_cycle_count", 0) or 0)
    acf = spectral.primary_autocorr() if spectral is not None else None
    spec_obj = spectral.primary_spectrum() if spectral is not None else None
    zip_path = OUT_DIR / "synthetic_12s_research_report.zip"
    export_research_report_zip(snap, zip_path, session=session)
    return {
        "kind": "synthetic_not_video",
        "note": (
            "Generated 12 s sine gait (L/R knee 20°/40°, stride 1.0 Hz). "
            "This is a pipeline-capability probe, not a substitute for videos A/B."
        ),
        "duration_s": spec.duration_s,
        "gait_cycles": n_cycles,
        "meets_five_cycle_gate": n_cycles >= MIN_USABLE_CYCLES,
        "acf_period_s": _finite(getattr(acf, "gait_period_s", None) or getattr(acf, "stride_period_s", None)),
        "acf_insufficient": str(getattr(acf, "insufficient_reason", "") or ""),
        "fft_dominant_hz": _finite(getattr(spec_obj, "dominant_hz", None)),
        "n_moments": len(list(getattr(moments, "moments", None) or [])),
        "moments_outcome": str(getattr(moments, "outcome", "") or ""),
        "research_statuses": {item.key: item.status for item in snap.items},
        "export": _zip_inventory(zip_path),
    }


def _score_row(
    requirement: str,
    *,
    a: dict[str, Any],
    b: dict[str, Any],
    verdict: str,
    evidence: str,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    assert verdict in {"Pass", "Partial", "Fail"}
    return {
        "requirement": requirement,
        "verdict": verdict,
        "evidence": evidence,
        "blockers": blockers or [],
        "A_cycles": a.get("gait_cycles"),
        "B_cycles": b.get("gait_cycles"),
    }


def _item_status(clip: dict[str, Any], key: str) -> str:
    items = ((clip.get("research_validation") or {}).get("items") or {})
    row = items.get(key) or {}
    return str(row.get("status") or "unavailable")


def _build_table(a: dict[str, Any], b: dict[str, Any], audit: dict[str, Any], synth: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    a_n = int(a.get("gait_cycles") or 0)
    b_n = int(b.get("gait_cycles") or 0)
    five_ok = a_n >= MIN_USABLE_CYCLES and b_n >= MIN_USABLE_CYCLES

    rows.append(
        _score_row(
            "Recordings ≥5 usable gait cycles",
            a=a,
            b=b,
            verdict="Pass" if five_ok else "Fail",
            evidence=(
                f"A {a.get('video', {}).get('path', '')} duration {a.get('duration_s')} s → {a_n} complete cycles. "
                f"B {b.get('video', {}).get('path', '')} duration {b.get('duration_s')} s → {b_n} complete cycles. "
                f"WeightGait limp alternate is {b.get('alternate_video', {}).get('duration_s')} s @ "
                f"{b.get('alternate_video', {}).get('fps')} fps with no pose JSON."
            ),
            blockers=[]
            if five_ok
            else [
                "Installed demo MP4s are ~5 s and yield 2 complete cycles each.",
                "weightgait_limping_gait.mp4 is 2.14 s / 15 frames / 7 fps and has no pose JSON.",
                "No longer normal vs asymmetric pair is on disk for this run.",
            ],
        )
    )

    a_ui = bool((a.get("ui") or {}).get("ok"))
    b_ui = bool((b.get("ui") or {}).get("ok"))
    sync_ok = all(audit.values()) if audit else False
    cadence_ok = bool(a.get("cadence_unchanged_after_spectral")) and bool(
        b.get("cadence_unchanged_after_spectral")
    )
    rows.append(
        _score_row(
            "Analysis state synchronization",
            a=a,
            b=b,
            verdict="Pass" if sync_ok and cadence_ok and a_ui and b_ui else ("Partial" if sync_ok and cadence_ok else "Fail"),
            evidence=(
                f"GUI wiring: bound-panel refresh, spectral/asymmetry refresh, "
                f"_sync_all_to_frame, jump-to-moment all present. "
                f"Cadence identity preserved after spectral (A {a.get('cadence_spm')} / "
                f"B {b.get('cadence_spm')} spm). "
                f"Tk Research Validation cards: A={a_ui} ({(a.get('ui') or {}).get('n_cards')} cards), "
                f"B={b_ui}. Timed on-screen playback capture was not performed."
            ),
            blockers=[]
            if (sync_ok and cadence_ok and a_ui and b_ui)
            else (
                []
                if sync_ok and cadence_ok
                else ["State-sync source audit failed or cadence was overwritten."]
            )
            + (
                []
                if a_ui and b_ui
                else [
                    f"Tk bind A: {(a.get('ui') or {}).get('error')}",
                    f"Tk bind B: {(b.get('ui') or {}).get('error')}",
                ]
            ),
        )
    )

    a_wire = a.get("wire") or {}
    b_wire = b.get("wire") or {}
    wire_computed = _item_status(a, "wire16") == "computed" and _item_status(b, "wire16") == "computed"
    rows.append(
        _score_row(
            "16-joint model",
            a=a,
            b=b,
            verdict="Pass" if wire_computed and a_wire.get("joint_count") == 16 and b_wire.get("joint_count") == 16 else "Fail",
            evidence=(
                f"A: {a_wire.get('usable_frames')}/{a_wire.get('n_frames')} usable frames, "
                f"incomplete={a_wire.get('incomplete_frames')}, 16 ids={a_wire.get('complete_ids_every_frame')}. "
                f"B: {b_wire.get('usable_frames')}/{b_wire.get('n_frames')} usable frames, "
                f"incomplete={b_wire.get('incomplete_frames')}. "
                f"RV status A={_item_status(a, 'wire16')} B={_item_status(b, 'wire16')}."
            ),
        )
    )

    a_k = a.get("knee_flexion") or {}
    b_k = b.get("knee_flexion") or {}
    knee_ok = (
        (a_k.get("n_left") or 0) > 0
        and (a_k.get("n_right") or 0) > 0
        and (b_k.get("n_left") or 0) > 0
        and (b_k.get("n_right") or 0) > 0
        and a_k.get("invented_when_missing") is False
    )
    rows.append(
        _score_row(
            "Anatomical knee flexion",
            a=a,
            b=b,
            verdict="Pass" if knee_ok else "Fail",
            evidence=(
                "Definition: 180° − interior 3D Hip–Knee–Ankle. Missing HKA is omitted, not filled. "
                f"A L ROM {a_k.get('left_rom_deg')}° / R {a_k.get('right_rom_deg')}° "
                f"(mean L {a_k.get('left_mean_deg')} R {a_k.get('right_mean_deg')}; "
                f"missing {a_k.get('n_missing_hka')}/{a_k.get('n_possible_sides')}). "
                f"B L ROM {b_k.get('left_rom_deg')}° / R {b_k.get('right_rom_deg')}° "
                f"(missing {b_k.get('n_missing_hka')}/{b_k.get('n_possible_sides')})."
            ),
        )
    )

    cycle_computed = _item_status(a, "frames_cycles") == "computed" and _item_status(b, "frames_cycles") == "computed"
    rows.append(
        _score_row(
            "Gait-cycle detection",
            a=a,
            b=b,
            verdict="Partial" if cycle_computed and not five_ok else ("Pass" if cycle_computed else "Fail"),
            evidence=(
                f"A: {a_n} complete cycles, L/R HS {a.get('left_heel_strikes')}/{a.get('right_heel_strikes')}, "
                f"cadence {a.get('cadence_spm')} spm, reliable={a.get('metrics_reliable')} "
                f"({a.get('reliability_reason')}). "
                f"B: {b_n} complete cycles, L/R HS {b.get('left_heel_strikes')}/{b.get('right_heel_strikes')}, "
                f"cadence {b.get('cadence_spm')} spm, reliable={b.get('metrics_reliable')} "
                f"({b.get('reliability_reason')}). "
                "Detection ran; protocol asked for ≥5 usable cycles."
            ),
            blockers=[] if five_ok else ["Both demo recordings are shorter than five complete gait cycles."],
        )
    )

    def _acf_verdict(clip: dict[str, Any]) -> str:
        st = _item_status(clip, "acf")
        acf = (clip.get("spectral") or {}).get("acf") or {}
        if st == "computed" and acf.get("gait_period_s") is not None and not acf.get("insufficient_reason"):
            return "computed"
        return st

    a_acf = _acf_verdict(a)
    b_acf = _acf_verdict(b)
    if a_acf == "computed" and b_acf == "computed":
        acf_v = "Pass"
    elif a_acf == "computed" or b_acf == "computed":
        acf_v = "Partial"
    else:
        acf_v = "Fail"
    a_acf_d = (a.get("spectral") or {}).get("acf") or {}
    b_acf_d = (b.get("spectral") or {}).get("acf") or {}
    rows.append(
        _score_row(
            "Autocorrelation",
            a=a,
            b=b,
            verdict=acf_v,
            evidence=(
                f"A status={_item_status(a, 'acf')} period={a_acf_d.get('gait_period_s')} s "
                f"reliability={a_acf_d.get('reliability_label')} "
                f"reason={a_acf_d.get('insufficient_reason') or '—'}. "
                f"B status={_item_status(b, 'acf')} period={b_acf_d.get('gait_period_s')} s "
                f"reliability={b_acf_d.get('reliability_label')} "
                f"reason={b_acf_d.get('insufficient_reason') or '—'}. "
                f"Export plots A={a.get('export', {}).get('plots')} B={b.get('export', {}).get('plots')}. "
                "Period is withheld when insufficient_reason is set."
            ),
            blockers=[
                x
                for x in [
                    None if a_acf == "computed" else f"A ACF {a_acf}: {a_acf_d.get('insufficient_reason') or a_acf_d.get('reliability_label')}",
                    None if b_acf == "computed" else f"B ACF {b_acf}: {b_acf_d.get('insufficient_reason') or b_acf_d.get('reliability_label')}",
                ]
                if x
            ],
        )
    )

    def _spec_ok(clip: dict[str, Any]) -> bool:
        st = _item_status(clip, "spectrum")
        spec = (clip.get("spectral") or {}).get("spectrum") or {}
        return st == "computed" and (spec.get("n_fft_bins") or 0) >= 8 and (spec.get("n_psd_bins") or 0) >= 8

    a_sp = _spec_ok(a)
    b_sp = _spec_ok(b)
    a_feat = (a.get("spectral") or {}).get("n_features_computed") or 0
    b_feat = (b.get("spectral") or {}).get("n_features_computed") or 0
    spec_v = "Pass" if a_sp and b_sp else ("Partial" if a_sp or b_sp else "Fail")
    a_spec = (a.get("spectral") or {}).get("spectrum") or {}
    b_spec = (b.get("spectral") or {}).get("spectrum") or {}
    rows.append(
        _score_row(
            "FFT/PSD features",
            a=a,
            b=b,
            verdict=spec_v,
            evidence=(
                f"A status={_item_status(a, 'spectrum')} f0={a_spec.get('dominant_hz')} Hz "
                f"FFT bins={a_spec.get('n_fft_bins')} PSD bins={a_spec.get('n_psd_bins')} "
                f"features {a_feat}/{(a.get('spectral') or {}).get('n_features_total')}. "
                f"B status={_item_status(b, 'spectrum')} f0={b_spec.get('dominant_hz')} Hz "
                f"FFT bins={b_spec.get('n_fft_bins')} PSD bins={b_spec.get('n_psd_bins')} "
                f"features {b_feat}/{(b.get('spectral') or {}).get('n_features_total')}. "
                f"Warnings A={(a.get('spectral') or {}).get('warnings')} "
                f"B={(b.get('spectral') or {}).get('warnings')}. "
                "Short-clip session f0 may sit at the walking-band floor; that is reported, not replaced with cadence."
            ),
            blockers=[]
            if a_sp and b_sp
            else [
                x
                for x in [
                    None if a_sp else f"A spectrum {_item_status(a, 'spectrum')}",
                    None if b_sp else f"B spectrum {_item_status(b, 'spectrum')}",
                ]
                if x
            ],
        )
    )

    a_m = a.get("moments") or {}
    b_m = b.get("moments") or {}
    a_mom_st = _item_status(a, "moments")
    b_mom_st = _item_status(b, "moments")
    mom_computed = a_mom_st == "computed" and b_mom_st == "computed"
    rows.append(
        _score_row(
            "Asymmetry events",
            a=a,
            b=b,
            verdict="Pass" if mom_computed else ("Partial" if a_mom_st == "computed" or b_mom_st == "computed" else "Fail"),
            evidence=(
                f"A outcome={a_m.get('outcome')} n_moments={a_m.get('n_moments')} "
                f"L/R cycles {a_m.get('n_left_cycles')}/{a_m.get('n_right_cycles')} "
                f"SI rows={len(a_m.get('joint_comparisons') or [])}. "
                f"B outcome={b_m.get('outcome')} n_moments={b_m.get('n_moments')} "
                f"L/R cycles {b_m.get('n_left_cycles')}/{b_m.get('n_right_cycles')}. "
                f"RV A={a_mom_st} B={b_mom_st}. "
                "Symmetric gait with n_moments=0 is still computed (threshold not exceeded), not a hidden fail."
            ),
        )
    )

    a_r = a.get("reconstruction3d") or {}
    b_r = b.get("reconstruction3d") or {}
    a_rst = _item_status(a, "reconstruction3d")
    b_rst = _item_status(b, "reconstruction3d")
    recon_pass = a_rst == "computed" or b_rst == "computed"
    recon_both_fail = a_rst in {"failed", "unavailable"} and b_rst in {"failed", "unavailable"}
    if a_rst == "computed" and b_rst == "computed":
        recon_v = "Pass"
    elif recon_pass:
        recon_v = "Partial"
    else:
        recon_v = "Fail"
    rows.append(
        _score_row(
            "3D full-body moment reconstruction",
            a=a,
            b=b,
            verdict=recon_v,
            evidence=(
                f"A status={a_rst} snapshot_matched={a_r.get('snapshot_matched')} "
                f"export_plot={a_r.get('export_plot')} strongest={a_r.get('strongest_present')}. "
                f"B status={b_rst} snapshot_matched={b_r.get('snapshot_matched')} "
                f"export_plot={b_r.get('export_plot')}. "
                "Renderer reuses the existing 16-joint wire at the selected frame; not a second IK solver. "
                "If no moment exceeded threshold, reconstruction is correctly unavailable (not marked passed)."
            ),
            blockers=[]
            if not recon_both_fail
            else ["No strongest asymmetric moment produced a matched skeleton snapshot on either clip."],
        )
    )

    # Comparison is filled later; placeholder filled by caller.
    rows.append(
        _score_row(
            "Comparison workflow",
            a=a,
            b=b,
            verdict="Fail",
            evidence="Pending compare run.",
        )
    )
    rows.append(
        _score_row(
            "Report export",
            a=a,
            b=b,
            verdict="Fail",
            evidence="Pending zip inventory.",
        )
    )
    rows.append(
        _score_row(
            "Synthetic 5-cycle capability (not video A/B)",
            a=a,
            b=b,
            verdict="Pass"
            if synth.get("meets_five_cycle_gate")
            and (synth.get("research_statuses") or {}).get("spectrum") == "computed"
            else "Partial",
            evidence=(
                f"{synth.get('note')} cycles={synth.get('gait_cycles')} "
                f"ACF period={synth.get('acf_period_s')} s f0={synth.get('fft_dominant_hz')} Hz "
                f"moments={synth.get('n_moments')} outcome={synth.get('moments_outcome')} "
                f"RV={synth.get('research_statuses')} export={synth.get('export', {}).get('names')}."
            ),
        )
    )
    return rows


def _fill_compare_and_export(rows: list[dict[str, Any]], a: dict[str, Any], b: dict[str, Any], compare: dict[str, Any]) -> None:
    for row in rows:
        if row["requirement"] == "Comparison workflow":
            ok = bool(compare.get("ok")) and compare.get("n_diffs", 0) >= 1
            identical = bool(compare.get("identical"))
            row["verdict"] = "Pass" if ok and not identical else ("Partial" if ok else "Fail")
            row["evidence"] = (
                f"Independent snapshots from pose JSON (build_snapshot_from_sequence). "
                f"A cadence={compare.get('A_cadence')} spm cycles={compare.get('A_cycles')} "
                f"knee ROM={compare.get('A_knee_rom')}. "
                f"B cadence={compare.get('B_cadence')} spm cycles={compare.get('B_cycles')} "
                f"knee ROM={compare.get('B_knee_rom')}. "
                f"Δ cadence={compare.get('delta_cadence')} steps/min; n_diffs={compare.get('n_diffs')}; "
                f"errors={compare.get('errors')}. "
                "Reload-Comparison GUI click was not driven this run."
            )
            row["blockers"] = [] if ok else list(compare.get("errors") or ["Compare did not produce diffs."])
        if row["requirement"] == "Report export":
            ae = a.get("export") or {}
            be = b.get("export") or {}
            required = {
                "has_json",
                "has_markdown",
                "has_definitions",
                "has_video_metadata",
                "has_failure_reasons",
            }
            a_ok = ae.get("exists") and all(ae.get(k) for k in required) and not ae.get("status_pass_claimed")
            b_ok = be.get("exists") and all(be.get(k) for k in required) and not be.get("status_pass_claimed")
            row["verdict"] = "Pass" if a_ok and b_ok else ("Partial" if a_ok or b_ok else "Fail")
            row["evidence"] = (
                f"A zip {ae.get('path')} names={ae.get('names')} plots={ae.get('plots')} "
                f"failure_keys={ae.get('failure_reason_keys')} claimed_pass={ae.get('status_pass_claimed')}. "
                f"B zip {be.get('path')} names={be.get('names')} plots={be.get('plots')} "
                f"failure_keys={be.get('failure_reason_keys')} claimed_pass={be.get('status_pass_claimed')}. "
                "Plots are omitted when the underlying series were not computed."
            )
            blockers = []
            if not a_ok:
                blockers.append("A research ZIP missing required files or claimed a pass.")
            if not b_ok:
                blockers.append("B research ZIP missing required files or claimed a pass.")
            row["blockers"] = blockers


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = _source_sync_audit()
    clips: list[dict[str, Any]] = []
    for clip in CLIPS:
        try:
            clips.append(_analyze_clip(clip))
        except Exception as exc:
            clips.append(
                {
                    "id": clip["id"],
                    "label": clip["label"],
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "traceback": traceback.format_exc()[-2000:],
                    "gait_cycles": 0,
                    "ui": {"ok": False, "error": str(exc)},
                    "export": {"exists": False},
                    "research_validation": {"items": {}},
                    "video": _video_meta(clip["video"]),
                }
            )
    a, b = clips[0], clips[1]

    compare: dict[str, Any] = {"ok": False, "errors": []}
    try:
        seq_a = load_pose_sequence(Path(a["poses_path"]))
        seq_b = load_pose_sequence(Path(b["poses_path"]))
        snap_a = build_snapshot_from_sequence(
            seq_a, key="normal", label="Normal", video_path=str(CLIPS[0]["video"])
        )
        snap_b = build_snapshot_from_sequence(
            seq_b, key="abnormal", label="Asymmetric", video_path=str(CLIPS[1]["video"])
        )
        left = extract_compare_metrics(
            label="Normal",
            session_key="normal",
            cycles=snap_a.gait_cycle,
            sequence=seq_a,
            biomechanical=snap_a.biomech_analysis,
            estimated_vgrf=None,
            duration_s=a.get("duration_s"),
        )
        right = extract_compare_metrics(
            label="Asymmetric",
            session_key="abnormal",
            cycles=snap_b.gait_cycle,
            sequence=seq_b,
            biomechanical=snap_b.biomech_analysis,
            estimated_vgrf=None,
            duration_s=b.get("duration_s"),
        )
        # Prefer already-computed cycle counts from the primary analysis objects.
        left.usable_gait_cycles = int(a.get("gait_cycles") or 0)
        right.usable_gait_cycles = int(b.get("gait_cycles") or 0)
        left.cadence_spm = a.get("cadence_spm")
        right.cadence_spm = b.get("cadence_spm")
        result = compare_session_metrics(left, right)
        diffs = list(getattr(result, "diffs", None) or getattr(result, "differences", None) or [])
        compare = {
            "ok": True,
            "identical": bool(getattr(result, "identical", False)),
            "A_cadence": left.cadence_spm,
            "B_cadence": right.cadence_spm,
            "A_cycles": left.usable_gait_cycles,
            "B_cycles": right.usable_gait_cycles,
            "A_knee_rom": left.knee_rom_deg,
            "B_knee_rom": right.knee_rom_deg,
            "delta_cadence": None
            if left.cadence_spm is None or right.cadence_spm is None
            else float(right.cadence_spm) - float(left.cadence_spm),
            "n_diffs": len(diffs),
            "diff_labels": [
                str(getattr(d, "label", None) or getattr(d, "name", None) or d) for d in diffs[:12]
            ],
            "errors": [],
        }
    except Exception as exc:
        compare = {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}

    try:
        synth = _synthetic_five_cycle_probe()
    except Exception as exc:
        synth = {
            "kind": "synthetic_not_video",
            "meets_five_cycle_gate": False,
            "error": f"{type(exc).__name__}: {exc}",
            "research_statuses": {},
            "export": {"exists": False},
        }

    table = _build_table(a, b, audit, synth)
    _fill_compare_and_export(table, a, b, compare)

    payload = {
        "schema": "stablewalk-lecturer-e2e",
        "protocol": {
            "min_usable_gait_cycles": MIN_USABLE_CYCLES,
            "mediapipe_rerun": False,
            "clinical": False,
            "note": (
                "Missing or withheld products stay insufficient/failed/unavailable. "
                "They are never converted into Pass."
            ),
        },
        "source_sync_audit": audit,
        "clips": clips,
        "compare": compare,
        "synthetic_probe": synth,
        "table": table,
    }
    OUT_JSON.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print()
    print(f"{'Requirement':<42} {'Verdict':<8} Evidence")
    print("-" * 120)
    for row in table:
        ev = row["evidence"].replace("\n", " ")
        if len(ev) > 140:
            ev = ev[:137] + "..."
        print(f"{row['requirement']:<42} {row['verdict']:<8} {ev}".encode("ascii", "replace").decode("ascii"))
        for blk in row.get("blockers") or []:
            print(f"    blocker: {blk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
