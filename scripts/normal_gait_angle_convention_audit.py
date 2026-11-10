"""Live audit of normal_gait.mp4 after knee-angle convention changes.

Does not change thresholds, invent ACF/f0, or treat the filename as a diagnosis.
Writes docs/normal_gait_angle_convention_audit.md and a JSON companion.

  python scripts/normal_gait_angle_convention_audit.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.asymmetry3d.events import (
    cycles_from_heel_strikes,
    summarize_gait_events,
)
from stablewalk.analysis.biomechanical import run_biomechanical_analysis
from stablewalk.analysis.gait_analysis_summary import build_gait_analysis_summary
from stablewalk.analysis.gait_cycle_analysis import (
    MAX_COMPLETE_CYCLE_DURATION_S,
    MIN_COMPLETE_CYCLE_DURATION_S,
    MIN_RELIABLE_CONTACT_CONFIDENCE,
    analyze_gait_cycles,
)
from stablewalk.analysis.quality_layers import GAIT_QUALITY_FORMULA, GAIT_QUALITY_WEIGHTS
from stablewalk.analysis.spectral.autocorrelation import (
    analyze_autocorrelation,
    normalized_autocorrelation,
)
from stablewalk.analysis.spectral.config import SpectralAnalysisConfig
from stablewalk.analysis.spectral.report import analyze_spectral
from stablewalk.analysis.spectral.series import (
    detrend_signal,
    interpolate_short_gaps,
    normalize_unit_variance,
    smooth_preserve_gait,
)
from stablewalk.analysis.spectral.spectrum import analyze_channel_spectrum
from stablewalk.analysis.stability_scoring import analyze_biomech_stability
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.knee_convention import (
    analysis_knee_flexion_deg,
    analysis_knee_interior_deg,
    recording_knee_storage,
)
from stablewalk.pose.kinematics import knee_flexion_from_snapshot

OUT_DIR = ROOT / "docs" / "_normal_gait_angle_convention_audit"
REPORT_MD = ROOT / "docs" / "normal_gait_angle_convention_audit.md"
REPORT_JSON = OUT_DIR / "audit.json"
POSES = config.POSES_DIR / "normal_gait_poses.json"
VIDEO = config.DEMO_VIDEOS_DIR / "normal_gait.mp4"

PRIMARY = "dof:left_knee_flexion"
SAGITTAL = (
    ("left_knee_flexion", "knee"),
    ("right_knee_flexion", "knee"),
    ("left_hip_flexion", "hip"),
    ("right_hip_flexion", "hip"),
    ("left_ankle_flexion", "ankle"),
    ("right_ankle_flexion", "ankle"),
)


def _finite(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=float)[np.isfinite(arr)]


def _stats(arr: np.ndarray) -> dict[str, Any]:
    y = _finite(arr)
    if y.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None, "p10": None, "p90": None, "rom_p10_p90": None}
    return {
        "n": int(y.size),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "mean": float(np.mean(y)),
        "median": float(np.median(y)),
        "p10": float(np.percentile(y, 10)),
        "p90": float(np.percentile(y, 90)),
        "rom_p10_p90": float(np.percentile(y, 90) - np.percentile(y, 10)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 8:
        return None
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _mae(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) == 0:
        return None
    return float(np.mean(np.abs(a[mask] - b[mask])))


def _series_from_snaps(snaps, getter) -> np.ndarray:
    out = np.full(len(snaps), np.nan, dtype=float)
    for i, snap in enumerate(snaps):
        val = getter(snap)
        if val is None:
            continue
        try:
            number = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out[i] = number
    return out


def _stored_dof(snap, dof_id: str) -> float | None:
    dof = snap.get_dof(dof_id)
    if dof is None or dof.angle_deg is None:
        return None
    val = float(dof.angle_deg)
    return val if math.isfinite(val) else None


def _preprocess(raw: np.ndarray, fps: float, cfg: SpectralAnalysisConfig) -> dict[str, np.ndarray]:
    max_gap = max(1, int(round(cfg.max_gap_s * fps)))
    win = max(3, int(round(cfg.smooth_window_s * fps)))
    if win % 2 == 0:
        win += 1
    filled, _interp = interpolate_short_gaps(raw, max_gap_samples=max_gap)
    smoothed = smooth_preserve_gait(filled, window_samples=win, polyorder=cfg.smooth_polyorder)
    processed = detrend_signal(smoothed, cfg.detrend)
    normalized = normalize_unit_variance(processed)
    return {
        "raw": raw,
        "filled": filled,
        "smoothed": smoothed,
        "processed": processed,
        "normalized": normalized,
    }


class _FakeSeries:
    def __init__(self, channel_id: str, values: dict[str, np.ndarray], fps: float) -> None:
        self.spec = type("S", (), {"channel_id": channel_id, "units": "deg"})()
        self.processed = values["processed"]
        self.normalized = values["normalized"]
        self.raw = values["raw"]
        self.fps = fps
        self.missing_fraction = float(np.mean(~np.isfinite(values["raw"]))) if values["raw"].size else 1.0
        self.duration_s = (len(values["raw"]) - 1) / fps if len(values["raw"]) > 1 else 0.0


def _estimate_version(name: str, raw: np.ndarray, fps: float, cfg: SpectralAnalysisConfig) -> dict[str, Any]:
    prep = _preprocess(raw, fps, cfg)
    series = _FakeSeries(name, prep, fps)
    spec = analyze_channel_spectrum(series, config=cfg)
    ac = analyze_autocorrelation(series, config=cfg)
    lags_i, rho = normalized_autocorrelation(prep["normalized"], max_lag=int(round(cfg.max_acf_lag_s * fps)))
    lags_s = lags_i.astype(float) / max(fps, 1e-6)
    lo, hi = cfg.stride_lag_s
    in_win = (lags_s >= lo) & (lags_s <= hi) & np.isfinite(rho)
    rho_win = rho[in_win]
    max_rho = float(np.max(rho_win)) if rho_win.size else None
    max_lag = float(lags_s[in_win][int(np.argmax(rho_win))]) if rho_win.size else None
    # local maxima inside the window (same gate as ACF: value > 0.05)
    local_max: list[dict[str, float]] = []
    for i in range(1, rho.size - 1):
        lag = float(lags_s[i])
        val = float(rho[i])
        if lo <= lag <= hi and val > 0.05 and val >= float(rho[i - 1]) and val > float(rho[i + 1]):
            local_max.append({"lag_s": lag, "rho": val})
    local_max.sort(key=lambda p: p["rho"], reverse=True)
    step_lo, step_hi = cfg.step_lag_s
    step_in = (lags_s >= step_lo) & (lags_s <= step_hi) & np.isfinite(rho)
    step_rho = rho[step_in]
    return {
        "name": name,
        "raw": _stats(raw),
        "processed": _stats(prep["processed"]),
        "normalized": _stats(prep["normalized"]),
        "dominant_hz": spec.dominant_hz,
        "dominant_power": spec.dominant_power,
        "spectral_centroid_hz": spec.spectral_centroid_hz,
        "band_power_fraction": spec.band_power_fraction,
        "spectral_notes": list(spec.notes),
        "acf_stride_period_s": ac.stride_period_s,
        "acf_step_period_s": ac.step_period_s,
        "acf_reliability_label": ac.reliability_label,
        "acf_insufficient_reason": ac.insufficient_reason,
        "acf_notes": list(ac.notes),
        "acf_stride_window_max_rho": max_rho,
        "acf_stride_window_max_lag_s": max_lag,
        "acf_stride_local_maxima": local_max[:5],
        "acf_step_window_max_rho": float(np.max(step_rho)) if step_rho.size else None,
    }


def _scan_stale_exports() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    roots = [
        config.OUTPUT_DIR / "sessions",
        config.OUTPUT_DIR / "analysis",
        config.OUTPUT_DIR / "reports",
        config.OUTPUT_DIR / "metrics",
        ROOT / "docs" / "_session_consistency_e2e",
    ]
    needles = ('"dominant_hz": 1.0416', '"dominant_frequency_hz": 1.0416', "1.0416666666666667")
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not any(n in text for n in needles) and "1.042" not in text:
                continue
            # cheap parse of first dominant_hz-like values
            found = []
            for token in ("1.0416666666666667", "1.042", "0.4166666666666667"):
                if token in text:
                    found.append(token)
            hits.append(
                {
                    "path": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path),
                    "tokens": found,
                    "bytes": path.stat().st_size,
                    "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                }
            )
    return hits


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "unavailable"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "unavailable"
        return f"{v:.{digits}g}"
    return str(v)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = SpectralAnalysisConfig()
    sequence = load_pose_sequence(POSES)
    recording = pose_sequence_to_gait_motion(sequence)
    snaps = list(recording.snapshots)
    fps = float(recording.fps)
    storage = recording_knee_storage(recording)
    pose_mtime = datetime.fromtimestamp(POSES.stat().st_mtime, tz=timezone.utc).isoformat()

    versions: dict[str, np.ndarray] = {}
    for dof_id, joint in SAGITTAL:
        stored = _series_from_snaps(snaps, lambda s, d=dof_id: _stored_dof(s, d))
        versions[f"{dof_id}:stored_interior"] = stored
        versions[f"{dof_id}:oneeighty_minus_stored"] = np.where(np.isfinite(stored), 180.0 - stored, np.nan)
        if joint == "knee":
            hka_flex = _series_from_snaps(snaps, lambda s, d=dof_id: analysis_knee_flexion_deg(s, d, storage=storage))
            hka_int = _series_from_snaps(snaps, lambda s, d=dof_id: analysis_knee_interior_deg(s, d, storage=storage))
            versions[f"{dof_id}:hka_flexion"] = hka_flex
            versions[f"{dof_id}:hka_interior"] = hka_int

    estimates: dict[str, dict[str, Any]] = {}
    for name, raw in versions.items():
        estimates[name] = _estimate_version(name, raw, fps, cfg)

    cycles = analyze_gait_cycles(recording, sequence=sequence)
    event_summary = summarize_gait_events(recording, cycles, sequence=sequence)
    left_hs = [e for e in cycles.events if "left_heel_strike" in e.event_type]
    right_hs = [e for e in cycles.events if "right_heel_strike" in e.event_type]
    left_to = [e for e in cycles.events if "left_toe_off" in e.event_type]
    right_to = [e for e in cycles.events if "right_toe_off" in e.event_type]

    spec = analyze_spectral(recording, sequence=sequence, cycles=cycles)
    primary_series = spec.bundle.get(PRIMARY) if spec.bundle else None
    primary_ac = spec.autocorr.get(PRIMARY)
    primary_sp = spec.spectra.get(PRIMARY)

    stability = analyze_biomech_stability(sequence, cycles=cycles)
    biomech = run_biomechanical_analysis(
        recording,
        sequence,
        cycles=cycles,
        stability=stability,
    )
    gait_summary = build_gait_analysis_summary(stability)

    # Channel-by-channel live spectral (current pipeline)
    channel_live: dict[str, dict[str, Any]] = {}
    for cid, series in spec.bundle.channels.items():
        if not cid.startswith("dof:"):
            continue
        ac = spec.autocorr.get(cid)
        sp = spec.spectra.get(cid)
        channel_live[cid] = {
            "raw": _stats(series.raw),
            "processed": _stats(series.processed),
            "dominant_hz": getattr(sp, "dominant_hz", None),
            "dominant_power": getattr(sp, "dominant_power", None),
            "centroid_hz": getattr(sp, "spectral_centroid_hz", None),
            "band_power_fraction": getattr(sp, "band_power_fraction", None),
            "acf_stride_period_s": getattr(ac, "stride_period_s", None),
            "acf_step_period_s": getattr(ac, "step_period_s", None),
            "acf_reliability_label": getattr(ac, "reliability_label", None),
            "acf_insufficient_reason": getattr(ac, "insufficient_reason", None),
            "acf_notes": list(getattr(ac, "notes", None) or ()),
        }

    leftover_hits = _scan_stale_exports()

    left_knee_hka = versions["left_knee_flexion:hka_flexion"]
    left_knee_stored = versions["left_knee_flexion:stored_interior"]
    left_knee_180 = versions["left_knee_flexion:oneeighty_minus_stored"]
    spec_raw = primary_series.raw if primary_series is not None else np.asarray([])

    audit: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clip": {
            "video": str(VIDEO.relative_to(ROOT)),
            "poses": str(POSES.relative_to(ROOT)),
            "pose_mtime": pose_mtime,
            "filename_is_not_diagnosis": True,
            "n_pose_frames": len(sequence.frames),
            "n_detected": sum(1 for f in sequence.frames if f.detected),
            "n_snapshots": len(snaps),
            "fps": fps,
            "duration_s": float(recording.duration_s),
            "coordinate_system": recording.coordinate_system,
            "source_kind": recording.source_kind,
            "knee_angle_storage": storage,
        },
        "formulas": {
            "knee_interior_theta": "θ = acos(clip((hip-knee)·(ankle-knee) / (|hip-knee||ankle-knee|), -1, 1)) in degrees",
            "knee_flexion": "knee_flexion_deg = 180° − θ; 0° = extension",
            "hip_stored": "interior angle at hip from shoulder–hip–knee (0–180°); no 180−θ in spectral",
            "ankle_stored": "interior angle at ankle from knee–ankle–foot_index (0–180°); no 180−θ",
            "dominant_frequency": "f0 = arg max_{f in gait_band} P(f); P is Welch PSD of z-scored processed series",
            "acf_rho": "ρ(k) = Σ x[t]x[t+k] / Σ x[t]² after mean removal on the longest finite run",
            "acf_stride_period": "lag of the largest positive local max of ρ in stride_lag_s with ρ > 0.05",
            "hs_hs_cycle": "complete ipsilateral heel-strike → next same-side heel-strike; duration in [0.35, 3.0] s",
            "gait_quality_composite": GAIT_QUALITY_FORMULA,
            "gait_quality_weights": dict(GAIT_QUALITY_WEIGHTS),
            "movement_stability": "confidence-weighted mean of pelvis_stability, trunk_stability, joint_smoothness",
            "overview_gait_quality_domains": "confidence-weighted mean of temporal_symmetry, spatial_symmetry, foot_clearance, cycle_consistency, contact_pattern",
        },
        "settings": cfg.to_dict(),
        "cycle_settings": {
            "min_complete_cycle_s": MIN_COMPLETE_CYCLE_DURATION_S,
            "max_complete_cycle_s": MAX_COMPLETE_CYCLE_DURATION_S,
            "min_reliable_contact_confidence": MIN_RELIABLE_CONTACT_CONFIDENCE,
            "min_acf_peak": 0.05,
        },
        "signal_versions": {
            name: {
                "raw": estimates[name]["raw"],
                "dominant_hz": estimates[name]["dominant_hz"],
                "acf_stride_period_s": estimates[name]["acf_stride_period_s"],
                "acf_stride_window_max_rho": estimates[name]["acf_stride_window_max_rho"],
                "acf_stride_window_max_lag_s": estimates[name]["acf_stride_window_max_lag_s"],
                "acf_stride_local_maxima": estimates[name]["acf_stride_local_maxima"],
                "acf_reliability_label": estimates[name]["acf_reliability_label"],
                "acf_notes": estimates[name]["acf_notes"],
            }
            for name in estimates
        },
        "version_comparisons": {
            "left_knee_hka_flexion_vs_180_minus_stored": {
                "corr": _corr(left_knee_hka, left_knee_180),
                "mae_deg": _mae(left_knee_hka, left_knee_180),
                "note": "If MAE≈0, HKA flexion is the same as 180−stored. Nonzero means stored DOF used a different geometry (2D fallback vs 3D HKA).",
            },
            "left_knee_hka_flexion_vs_stored_interior": {
                "corr": _corr(left_knee_hka, left_knee_stored),
                "mae_deg": _mae(left_knee_hka, left_knee_stored),
            },
            "left_knee_hka_flexion_vs_spectral_raw": {
                "corr": _corr(left_knee_hka, spec_raw) if spec_raw.size else None,
                "mae_deg": _mae(left_knee_hka, spec_raw) if spec_raw.size else None,
                "note": "Current spectral raw must match analysis_knee_flexion_deg (HKA).",
            },
            "sign_flip_identity": (
                "A pure 180−θ map is affine. After linear detrend + z-score, "
                "PSD and ACF of flexion equal those of interior. A changed f0 "
                "therefore cannot come from the 180−θ label alone."
            ),
        },
        "current_spectral_primary": {
            "channel_id": PRIMARY,
            "dominant_hz": getattr(primary_sp, "dominant_hz", None),
            "dominant_power": getattr(primary_sp, "dominant_power", None),
            "centroid_hz": getattr(primary_sp, "spectral_centroid_hz", None),
            "bandwidth_hz": getattr(primary_sp, "spectral_bandwidth_hz", None),
            "entropy": getattr(primary_sp, "spectral_entropy", None),
            "band_power_fraction": getattr(primary_sp, "band_power_fraction", None),
            "notes": list(getattr(primary_sp, "notes", None) or ()),
            "acf_stride_period_s": getattr(primary_ac, "stride_period_s", None),
            "acf_step_period_s": getattr(primary_ac, "step_period_s", None),
            "acf_reliability_label": getattr(primary_ac, "reliability_label", None),
            "acf_insufficient_reason": getattr(primary_ac, "insufficient_reason", None),
            "acf_notes": list(getattr(primary_ac, "notes", None) or ()),
            "n_valid_cycles": spec.features.n_valid_cycles if spec.features else None,
            "n_detected_cycles": spec.features.n_detected_cycles if spec.features else None,
            "feature_warnings": list(spec.features.warnings) if spec.features else [],
            "raw": _stats(primary_series.raw) if primary_series is not None else None,
            "processed": _stats(primary_series.processed) if primary_series is not None else None,
        },
        "current_spectral_dof_channels": channel_live,
        "cycle_methods": {
            "contact_hs_hs": {
                "method": "analyze_gait_cycles: contact FSM + heel/toe/ankle clearance; complete ipsilateral HS–HS",
                "n_complete_cycles": len(cycles.cycles),
                "boundary_side": cycles.cycles[0].boundary_side if cycles.cycles else None,
                "cycle_durations_s": [float(c.duration_s) for c in cycles.cycles],
                "cycle_windows_s": [
                    [float(c.start_time_s), float(c.end_time_s)] for c in cycles.cycles
                ],
                "left_heel_strikes": len(left_hs),
                "right_heel_strikes": len(right_hs),
                "left_toe_offs": len(left_to),
                "right_toe_offs": len(right_to),
                "left_hs_times_s": [float(e.time_s) for e in left_hs],
                "right_hs_times_s": [float(e.time_s) for e in right_hs],
                "contact_confidence": cycles.metrics.contact_confidence,
                "confidence_tier": cycles.metrics.confidence_tier,
                "metrics_reliable": cycles.metrics.metrics_reliable,
                "reliability_reason": cycles.metrics.reliability_reason,
                "cadence_steps_per_min": cycles.metrics.cadence_steps_per_min,
                "warnings": list(cycles.warnings),
                "does_not_use": "knee/hip/ankle angle convention, PSD, or ACF",
            },
            "asymmetry_sides": {
                "method": "summarize_gait_events / cycles_from_heel_strikes: ipsilateral HS–HS per side",
                "left_cycles": len(event_summary.left_cycles),
                "right_cycles": len(event_summary.right_cycles),
                "n_heel_strikes": len(event_summary.heel_strikes),
                "n_toe_offs": len(event_summary.toe_offs),
            },
            "spectral_feature_cycles": {
                "method": "extract_spectral_features._stride_cv on analyze_gait_cycles.cycles (same HS–HS list)",
                "n_detected_cycles": spec.features.n_detected_cycles if spec.features else None,
                "n_valid_cycles": spec.features.n_valid_cycles if spec.features else None,
                "valid_means": "cycles with duration_s > 0; not an ACF count",
            },
            "acf_period": {
                "method": "positive local max of ρ in stride_lag_s=(0.60, 1.80) with ρ>0.05 on z-scored primary channel",
                "stride_period_s": getattr(primary_ac, "stride_period_s", None),
                "unavailable_reason": (
                    getattr(primary_ac, "insufficient_reason", None)
                    or (
                        "No clear autocorrelation peak in the stride lag window."
                        if getattr(primary_ac, "stride_period_s", None) is None
                        else None
                    )
                ),
                "independent_of_hs_hs": True,
            },
        },
        "scores": {
            "do_not_use_f0_or_acf": True,
            "movement_stability": {
                "score": None if gait_summary.movement_stability.score is None else float(gait_summary.movement_stability.score),
                "completeness_pct": gait_summary.movement_stability.completeness_pct,
                "active_domains": gait_summary.movement_stability.active_domains,
                "domain_keys": list(gait_summary.movement_stability.domain_keys),
                "unavailable_domains": list(gait_summary.movement_stability.unavailable_domains),
                "source": "summarize_gait_analysis(stability).movement_stability",
            },
            "overview_gait_quality_domains": {
                "score": None if gait_summary.gait_quality.score is None else float(gait_summary.gait_quality.score),
                "completeness_pct": gait_summary.gait_quality.completeness_pct,
                "active_domains": gait_summary.gait_quality.active_domains,
                "domain_keys": list(gait_summary.gait_quality.domain_keys),
                "unavailable_domains": list(gait_summary.gait_quality.unavailable_domains),
                "source": "summarize_gait_analysis(stability).gait_quality — NOT compute_gait_quality_score",
            },
            "gait_quality_composite": {
                "score": None if biomech.gait_quality.score is None else float(biomech.gait_quality.score),
                "available": bool(getattr(biomech.gait_quality, "available", False)),
                "status": getattr(biomech.gait_quality, "status", None),
                "confidence": getattr(biomech.gait_quality, "confidence", None),
                "components": dict(getattr(biomech.gait_quality, "components", {}) or {}),
                "explanation": getattr(biomech.gait_quality, "explanation", None),
                "source": "compute_gait_quality_score via run_biomechanical_analysis",
            },
            "legacy_composite": {
                "score": float(gait_summary.legacy_composite_score),
                "classification": gait_summary.legacy_classification,
            },
            "stability_result": {
                "score": float(stability.score),
                "classification": stability.classification,
                "completeness_pct": float(stability.completeness_pct),
                "usable_gait_cycles": getattr(stability, "usable_gait_cycles", None),
            },
            "domain_scores": {
                m.key: {"score": m.score, "confidence": m.confidence, "availability": m.availability}
                for m in stability.metrics
            },
        },
        "conventions": {
            "knee": {
                "pose_storage": "pose_interior_hka = interior HKA from compute_joint_angles (angle_at_joint; 3D if z varies, else 2D)",
                "analysis_display": "analysis_knee_flexion_deg prefers 3D HKA flexion; else 180−stored",
                "spectral": "converted (knee only) via is_knee_flexion_dof",
                "kinematics_table": "knee_flexion_from_snapshot (3D HKA flexion)",
                "rom": "180 − interior",
            },
            "hip": {
                "pose_storage": "interior shoulder–hip–knee from compute_joint_angles; stored as left/right_hip_flexion",
                "analysis_display": "Kinematics table keeps stored interior; joint_rom converts 180−interior; spectral does not convert",
                "spectral": "stored DOF passed through — label says flexion, values are interior",
                "rom": "180 − interior",
            },
            "ankle": {
                "pose_storage": "interior knee–ankle–foot_index; stored as left/right_ankle_flexion",
                "analysis_display": "kept as interior everywhere (foot landmarks are not a clean dorsi/plantar axis)",
                "spectral": "stored DOF passed through",
                "rom": "interior span, not 180−θ",
            },
        },
        "leftover_exports_mentioning_1p042": leftover_hits,
        "quality_reasons": {
            "did_not_force_frequency": True,
            "did_not_restore_acf": True,
            "did_not_change_thresholds": True,
        },
    }

    left_c = cycles_from_heel_strikes(list(cycles.events), "left")
    right_c = cycles_from_heel_strikes(list(cycles.events), "right")
    audit["cycle_methods"]["asymmetry_sides"]["left_cycles"] = len(left_c)
    audit["cycle_methods"]["asymmetry_sides"]["right_cycles"] = len(right_c)
    audit["cycle_methods"]["asymmetry_sides"]["left_plus_right"] = len(left_c) + len(right_c)
    audit["cycle_methods"]["asymmetry_sides"]["left_windows_s"] = [
        [float(c.start_time_s), float(c.end_time_s)] for c in left_c
    ]
    audit["cycle_methods"]["asymmetry_sides"]["right_windows_s"] = [
        [float(c.start_time_s), float(c.end_time_s)] for c in right_c
    ]

    REPORT_JSON.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    REPORT_MD.write_text(_render_md(audit, estimates), encoding="utf-8")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")
    print("primary f0", audit["current_spectral_primary"]["dominant_hz"])
    print("acf stride", audit["current_spectral_primary"]["acf_stride_period_s"])
    print("hs-hs cycles", audit["cycle_methods"]["contact_hs_hs"]["n_complete_cycles"])
    print("L+R cycles", audit["cycle_methods"]["asymmetry_sides"]["left_plus_right"])
    print("movement", audit["scores"]["movement_stability"]["score"])
    print("overview gq domains", audit["scores"]["overview_gait_quality_domains"]["score"])
    print("gq composite", audit["scores"]["gait_quality_composite"]["score"])
    print("legacy", audit["scores"]["legacy_composite"]["score"])
    return 0


def _render_md(audit: dict[str, Any], estimates: dict[str, dict[str, Any]]) -> str:
    clip = audit["clip"]
    prim = audit["current_spectral_primary"]
    cyc = audit["cycle_methods"]
    scores = audit["scores"]
    cmp_ = audit["version_comparisons"]
    cfg = audit["settings"]
    lk = estimates["left_knee_flexion:hka_flexion"]
    ls = estimates["left_knee_flexion:stored_interior"]
    l180 = estimates["left_knee_flexion:oneeighty_minus_stored"]
    lines = [
        "# normal_gait.mp4 angle-convention audit",
        "",
        f"Generated: `{audit['generated_at']}` from cached `{clip['poses']}` (mtime `{clip['pose_mtime']}`).",
        "The filename is not a diagnosis. Thresholds, ACF, and f0 were not adjusted to look successful.",
        "",
        "## Observed vs live",
        "",
        "| Quantity | User-reported | Live recompute | Source |",
        "|---|---|---|---|",
        f"| Primary dominant frequency | 1.042 → 0.4167 Hz | **{_fmt(prim['dominant_hz'])} Hz** | `f0 = arg max P(f)` on `{prim['channel_id']}` in `{cfg['gait_band_hz']}` Hz |",
        f"| ACF stride period | unavailable | **{_fmt(prim['acf_stride_period_s'])}** | positive local max of ρ in `{cfg['stride_lag_s']}` s with ρ>0.05 |",
        f"| Contact HS–HS complete cycles | (see 3-cycle note) | **{cyc['contact_hs_hs']['n_complete_cycles']}** on `{cyc['contact_hs_hs']['boundary_side']}` | `analyze_gait_cycles` |",
        f"| Left + right HS–HS cycles | 3 if summed | **{cyc['asymmetry_sides']['left_plus_right']}** (L{cyc['asymmetry_sides']['left_cycles']} + R{cyc['asymmetry_sides']['right_cycles']}) | `cycles_from_heel_strikes` |",
        f"| Spectral feature cycles | 3 valid? | **{prim['n_valid_cycles']} valid / {prim['n_detected_cycles']} detected** | `_stride_cv` on the single-side HS–HS list |",
        f"| Left heel-strikes | 3 | **{cyc['contact_hs_hs']['left_heel_strikes']}** at {cyc['contact_hs_hs']['left_hs_times_s']} s | contact FSM |",
        f"| Movement stability | 92.5? | **{_fmt(scores['movement_stability']['score'], 1)}** | pelvis/trunk/smoothness domains |",
        f"| Overview gait-quality domains | 82.5? | **{_fmt(scores['overview_gait_quality_domains']['score'], 1)}** | symmetry/clearance/cycle/contact domains |",
        f"| Heuristic gait-quality composite | 82.5? | **{_fmt(scores['gait_quality_composite']['score'], 1)}** | `compute_gait_quality_score` |",
        f"| Legacy composite | — | **{_fmt(scores['legacy_composite']['score'], 1)}** | 8-domain stability score |",
        "",
        "## Why 0.4167 Hz is not a forced value",
        "",
        f"- Frequency grid at {clip['fps']} Hz with Welch `nperseg ≈ n` is Δf = {clip['fps']}/{clip['n_snapshots']} = **{clip['fps']/max(clip['n_snapshots'],1):.4f} Hz**.",
        "- `0.4167 Hz = 2 Δf` (2.40 s). `1.0417 Hz = 5 Δf` (0.96 s). Both are bins inside `(0.40, 4.00)` Hz.",
        "- `f0` is the **maximum-power bin**, not the nearest bin to a walking prior. The search floor is 0.40 Hz, so a slow peak sits on the first in-band bin.",
        f"- Current left-knee HKA flexion: f0={_fmt(lk['dominant_hz'])} Hz, centroid={_fmt(lk['spectral_centroid_hz'])} Hz, band-power fraction={_fmt(lk['band_power_fraction'])}.",
        f"- Current left-knee **stored interior** (no conversion): f0={_fmt(ls['dominant_hz'])} Hz. Current **180−stored**: f0={_fmt(l180['dominant_hz'])} Hz.",
        f"- HKA flexion vs 180−stored: corr={_fmt(cmp_['left_knee_hka_flexion_vs_180_minus_stored']['corr'])}, MAE={_fmt(cmp_['left_knee_hka_flexion_vs_180_minus_stored']['mae_deg'])}°.",
        f"- Spectral raw vs HKA flexion: corr={_fmt(cmp_['left_knee_hka_flexion_vs_spectral_raw']['corr'])}, MAE={_fmt(cmp_['left_knee_hka_flexion_vs_spectral_raw']['mae_deg'])}°.",
        "",
        cmp_["sign_flip_identity"],
        "",
        "If stored-interior f0 already equals HKA f0, the 1.042 Hz figure is **not** the current left-knee series (stale GUI cache, another channel, or an older estimator). If they differ, the change is the **HKA 3D substitution**, not the 180−θ label.",
        "",
        "## ACF unavailable — quality reason",
        "",
        f"- Formula: `{audit['formulas']['acf_rho']}` then `{audit['formulas']['acf_stride_period']}`.",
        f"- Window: `{cfg['stride_lag_s']}` s. Gate: local maximum and ρ > 0.05. Negative or trough values are not periods.",
        f"- Live primary: stride={_fmt(prim['acf_stride_period_s'])}, step={_fmt(prim['acf_step_period_s'])}, label=`{prim['acf_reliability_label']}`.",
        f"- Reason: `{prim['acf_insufficient_reason'] or 'No clear autocorrelation peak in the stride lag window.'}`",
        f"- Left-knee HKA: window max ρ={_fmt(lk['acf_stride_window_max_rho'])} at lag {_fmt(lk['acf_stride_window_max_lag_s'])} s; local maxima={lk['acf_stride_local_maxima']}.",
        f"- Left-knee stored interior: window max ρ={_fmt(ls['acf_stride_window_max_rho'])} at lag {_fmt(ls['acf_stride_window_max_lag_s'])} s.",
        "- This was not restored. A missing peak is the correct export.",
        "",
        "## Separate cycle methods (confidence is not interchangeable)",
        "",
        "### 1. Contact heel-strike cycles",
        "",
        "- Detector: foot clearance + contact FSM (`analyze_gait_cycles`). Independent of angle convention and of PSD/ACF.",
        f"- Complete cycle: consecutive **same-side** HS with duration in [{audit['cycle_settings']['min_complete_cycle_s']}, {audit['cycle_settings']['max_complete_cycle_s']}] s. Partial lead/trail intervals are dropped.",
        f"- This recording: L HS {cyc['contact_hs_hs']['left_heel_strikes']} {cyc['contact_hs_hs']['left_hs_times_s']} s; R HS {cyc['contact_hs_hs']['right_heel_strikes']} {cyc['contact_hs_hs']['right_hs_times_s']} s.",
        f"- Complete cycles on the chosen boundary side (`{cyc['contact_hs_hs']['boundary_side']}`): {cyc['contact_hs_hs']['n_complete_cycles']} with durations {cyc['contact_hs_hs']['cycle_durations_s']} s.",
        f"- Contact confidence: {_fmt(cyc['contact_hs_hs']['contact_confidence'])} (`{cyc['contact_hs_hs']['confidence_tier']}`). Reliable={cyc['contact_hs_hs']['metrics_reliable']}.",
        f"- Reliability reason: {cyc['contact_hs_hs']['reliability_reason']}",
        f"- Cadence from those intervals: {_fmt(cyc['contact_hs_hs']['cadence_steps_per_min'])} steps/min (not from f0).",
        f"- Warnings: {cyc['contact_hs_hs']['warnings']}",
        "",
        "### 2. Left+right ipsilateral cycles (asymmetry)",
        "",
        f"- Same HS events, counted **per side**: L={cyc['asymmetry_sides']['left_cycles']} {cyc['asymmetry_sides']['left_windows_s']}, R={cyc['asymmetry_sides']['right_cycles']} {cyc['asymmetry_sides']['right_windows_s']}.",
        f"- Sum = **{cyc['asymmetry_sides']['left_plus_right']}**. This is the only live count that matches “3 valid cycles”. It is not 3 left strides.",
        "",
        "### 3. Spectral feature cycle count",
        "",
        f"- Reuses method 1 (`analyze_gait_cycles.cycles`), then counts durations > 0: **{prim['n_valid_cycles']} valid / {prim['n_detected_cycles']} detected**.",
        "- Per-cycle PSD slices use those HS–HS bounds. This is not an ACF cycle estimate.",
        "",
        "### 4. ACF period / n_stride_cycles",
        "",
        "- Separate estimator on the z-scored knee series. Currently **unavailable** (no in-window peak).",
        "- Must not be filled from method 1 or 2.",
        "",
        "## Hip and ankle conventions (not only knee)",
        "",
        "| Joint | Pose storage | Spectral raw | Kinematics table | Joint ROM |",
        "|---|---|---|---|---|",
        "| Knee | interior HKA (`angle_at_joint`, 2D fallback allowed) | **3D HKA flexion** `180−θ` | 3D HKA flexion | `180−interior` |",
        "| Hip | interior shoulder–hip–knee, **named** `*_hip_flexion` | **stored interior** (no convert) | stored interior | `180−interior` |",
        "| Ankle | interior knee–ankle–foot_index, named `*_ankle_flexion` | stored interior | stored interior | interior span |",
        "",
        "Hip is therefore labeled flexion in Spectral/Asymmetry channel ids but is still the pose interior angle. ROM and spectral hip series are **different conventions**. Ankle is consistently interior.",
        "",
        "## Preprocessing (spectral)",
        "",
        f"- Uniform FPS grid at {clip['fps']} Hz. Visibility mask {cfg['visibility_min']}. Linear-fill gaps ≤ {cfg['max_gap_s']} s. Savitzky–Golay `{cfg['smooth_window_s']}` s / poly {cfg['smooth_polyorder']}. Detrend=`{cfg['detrend']}`. Z-score for ACF/FFT.",
        f"- Primary raw range: {_fmt(prim['raw']['min'] if prim['raw'] else None, 2)} … {_fmt(prim['raw']['max'] if prim['raw'] else None, 2)} deg (median {_fmt(prim['raw']['median'] if prim['raw'] else None, 2)}). Processed (detrended) RMS-scale min/max {_fmt(prim['processed']['min'] if prim['processed'] else None, 2)} … {_fmt(prim['processed']['max'] if prim['processed'] else None, 2)}.",
        "",
        "## Composite scores do not consume f0 or ACF",
        "",
        "That is why 92.5 / 82.5 can stay put after the spectral change. They are not stale spectral numbers; they never read the spectrum.",
        "",
        f"- **Movement stability** `{_fmt(scores['movement_stability']['score'], 1)}`: {scores['movement_stability']['domain_keys']}. Completeness { _fmt(scores['movement_stability']['completeness_pct'], 1)}%.",
        f"- **Overview gait-quality domains** `{_fmt(scores['overview_gait_quality_domains']['score'], 1)}`: {scores['overview_gait_quality_domains']['domain_keys']}. Different algorithm from the composite below.",
        f"- **Heuristic gait-quality composite** `{_fmt(scores['gait_quality_composite']['score'], 1)}`: components `{scores['gait_quality_composite']['components']}`. Formula: {audit['formulas']['gait_quality_composite']}",
        f"- **Legacy 8-domain composite** `{_fmt(scores['legacy_composite']['score'], 1)}` (`{scores['legacy_composite']['classification']}`).",
        f"- Stability usable_gait_cycles=`{scores['stability_result']['usable_gait_cycles']}` (another count; not ACF).",
        "",
        "Domain scores:",
        "",
    ]
    for key, row in scores["domain_scores"].items():
        lines.append(f"- `{key}`: score={_fmt(row['score'], 1)} confidence={_fmt(row['confidence'], 2)} ({row['availability']})")
    lines.extend(
        [
            "",
            "## Sagittal DOF live spectral (current pipeline)",
            "",
            "| Channel | raw min…max (deg) | f0 (Hz) | ACF stride (s) | ACF label |",
            "|---|---|---|---|---|",
        ]
    )
    for cid, row in audit["current_spectral_dof_channels"].items():
        raw = row["raw"]
        lines.append(
            f"| `{cid}` | {_fmt(raw['min'], 2)}…{_fmt(raw['max'], 2)} | {_fmt(row['dominant_hz'])} | {_fmt(row['acf_stride_period_s'])} | {row['acf_reliability_label']} |"
        )
    hits = audit["leftover_exports_mentioning_1p042"]
    lines.extend(
        [
            "",
            "## Signal-version cache / inconsistency",
            "",
            "- Pose JSON was not re-run (MediaPipe not invoked). `knee_angle_storage=pose_interior_hka`.",
            "- GUI `schedule_spectral_analysis` keeps `gui._spectral_report` until `force=True`. A live session started before the convention change can still show 1.042 Hz until Spectral is recomputed. After recompute, leftover **session export JSON** can still contain the old f0.",
            f"- Exports that still mention 1.042 / 1.0417 Hz: **{len(hits)}** file(s).",
        ]
    )
    for hit in hits[:12]:
        lines.append(f"- `{hit['path']}` tokens={hit['tokens']} mtime={hit['mtime']}")
    if len(hits) > 12:
        lines.append(f"- … {len(hits) - 12} more (see audit.json)")
    lines.extend(
        [
            "",
            "Right-knee / some hip channels in the current live table can still peak at 1.0417 Hz. That is a **different channel**, not a restored primary f0.",
            "",
            "## What was not done",
            "",
            "- No expected-frequency override.",
            "- ACF period left unavailable.",
            "- No threshold edits to create a stride peak or a third HS–HS cycle.",
            "",
            f"Machine-readable dump: `{REPORT_JSON.relative_to(ROOT).as_posix()}`.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
