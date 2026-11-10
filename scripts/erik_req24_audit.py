"""Audit requirements 2–4 on demo poses + synthetic ground-truth signals."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _session_vector(report) -> dict[str, float | None]:
    feats = getattr(report, "features", None)
    if feats is None:
        return {}
    vec = getattr(feats, "session_vector", None)
    if isinstance(vec, dict):
        return {k: (None if v is None else float(v)) for k, v in vec.items()}
    return {}


def _clip_report(name: str) -> dict:
    from stablewalk import config
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.analysis.asymmetry3d.report import analyze_asymmetry3d
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
    from stablewalk.analysis.gait_view_analysis import estimate_gait_view
    from stablewalk.analysis.opensim_id_readiness import assess_opensim_id_readiness
    from stablewalk.analysis.spectral.report import analyze_spectral
    from stablewalk.io.pose_loader import load_pose_sequence

    video = config.DEMO_VIDEOS_DIR / f"{name}.mp4"
    poses = config.POSES_DIR / f"{name}_poses.json"
    out: dict = {
        "name": name,
        "video_exists": video.is_file(),
        "poses_exists": poses.is_file(),
    }
    if not poses.is_file():
        out["error"] = f"missing {poses}"
        return out
    seq = load_pose_sequence(poses)
    rec = pose_sequence_to_gait_motion(seq)
    cycles = analyze_gait_cycles(rec)
    spec = analyze_spectral(rec, sequence=seq, cycles=cycles)
    asym = analyze_asymmetry3d(rec, cycles=cycles, sequence=seq)
    pose_frames = getattr(seq, "frames", None) or getattr(seq, "pose_frames", None) or []
    view = estimate_gait_view(pose_frames) if pose_frames else None
    n = len(cycles.per_frame)
    contact_frac = None
    if n:
        lc = sum(s.left_contact for s in cycles.per_frame)
        rc = sum(s.right_contact for s in cycles.per_frame)
        contact_frac = (lc + rc) / max(2 * n, 1)
    hs = [e for e in cycles.events if e.event_type.endswith("_heel_strike")]
    to = [e for e in cycles.events if e.event_type.endswith("_toe_off")]
    out.update(
        {
            "frames": rec.frame_count,
            "fps": rec.fps,
            "duration_s": float(getattr(rec, "duration_s", 0.0) or 0.0),
            "view": None if view is None else str(view.view_type),
            "view_confidence": None if view is None else float(view.view_confidence),
            "contact_frac": contact_frac,
            "hs_left": sum(1 for e in hs if e.side == "left"),
            "hs_right": sum(1 for e in hs if e.side == "right"),
            "to_left": sum(1 for e in to if e.side == "left"),
            "to_right": sum(1 for e in to if e.side == "right"),
            "hs_times_s": [round(e.time_s, 3) for e in hs],
            "to_times_s": [round(e.time_s, 3) for e in to],
            "cycle_count": cycles.metrics.gait_cycle_count,
            "cadence": cycles.metrics.cadence_steps_per_min,
            "metrics_reliable": cycles.metrics.metrics_reliable,
            "event_sufficiency": cycles.event_sufficiency,
            "event_sufficiency_reason": cycles.event_sufficiency_reason,
            "cycle_warnings": list(cycles.warnings)[:8],
            "spectral_available": spec.available,
            "spectral_warnings": list(spec.warnings)[:6],
            "session_vector": _session_vector(spec),
            "univariate_channel": spec.univariate_channel_id,
            "asymmetry_available": bool(getattr(asym, "available", True)),
            "asymmetry_warnings": list(getattr(asym, "warnings", None) or [])[:6],
            "opensim_id_ready": assess_opensim_id_readiness(run_name=name).ready_for_traditional_id,
        }
    )
    compact: dict = {}
    for rec_a in getattr(asym, "scalars", None) or []:
        name_a = str(getattr(rec_a, "name", "") or "")
        compact[name_a] = {
            "left": getattr(rec_a, "left", None),
            "right": getattr(rec_a, "right", None),
            "asi_percent": getattr(rec_a, "asi_percent", None),
            "si_percent": getattr(rec_a, "si_percent", None),
            "reliable": getattr(rec_a, "reliable", None),
            "note": getattr(rec_a, "note", ""),
        }
    out["asymmetry_indices"] = compact
    r0 = getattr(getattr(spec.matrices, "r0", None), "correlation", None)
    if r0 is not None:
        import numpy as np

        out["r0_shape"] = list(r0.shape)
        out["r0_symmetric"] = bool(np.allclose(r0, r0.T, equal_nan=True, atol=1e-8))
    eigen = spec.eigen.get("correlation") if spec.eigen else None
    if eigen is not None:
        ev = getattr(eigen, "eigenvalues", None)
        if ev is not None:
            out["r0_eigenvalues"] = [float(x) for x in list(ev)[:4]]
    return out


def main() -> int:
    from tests.test_spectral_analysis import _sine_recording
    from stablewalk.analysis.spectral.report import analyze_spectral
    from stablewalk.analysis.spectral.spectrum import welch_psd

    results = {
        "clips": [_clip_report("normal_gait"), _clip_report("abnormal_gait")],
        "synthetic": {},
    }
    rec = _sine_recording(fps=50.0, duration_s=8.0, freq_hz=1.0)
    spec = analyze_spectral(rec)
    vec = _session_vector(spec)
    results["synthetic"]["sine_1hz"] = {
        "dominant_frequency_hz": vec.get("dominant_frequency_hz"),
        "spectral_entropy": vec.get("spectral_entropy"),
        "acf_stride_lag_s": vec.get("acf_stride_lag_s"),
        "available": spec.available,
        "warnings": list(spec.warnings)[:4],
    }
    rec_n = _sine_recording(fps=50.0, duration_s=8.0, freq_hz=1.0, irregular=True)
    spec_n = analyze_spectral(rec_n)
    vec_n = _session_vector(spec_n)
    results["synthetic"]["irregular"] = {
        "dominant_frequency_hz": vec_n.get("dominant_frequency_hz"),
        "spectral_entropy": vec_n.get("spectral_entropy"),
        "acf_stride_lag_s": vec_n.get("acf_stride_lag_s"),
    }
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
