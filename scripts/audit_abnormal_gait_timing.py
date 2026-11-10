#!/usr/bin/env python3
"""Trace abnormal_gait_v2 timing numbers without changing analysis formulas."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/output/example_v2_verification/abnormal_gait_timing_audit.json"


def _latest_poses(filename: str) -> Path | None:
    from stablewalk import config

    matches = []
    for path in config.POSES_DIR.glob("*_poses.json"):
        try:
            with path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            continue
        src = str(raw.get("source_video") or "")
        if Path(src).name == filename:
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _peak_rows(ac) -> list[dict]:
    rows = []
    for peak in getattr(ac, "peaks", None) or []:
        lag = float(peak.lag_s)
        in_step = 0.30 <= lag <= 0.90
        in_stride = 0.55 <= lag <= 2.20
        in_both = in_step and in_stride
        rows.append(
            {
                "lag_s": lag,
                "rho": float(peak.value),
                "prominence": float(peak.prominence),
                "in_step_window": in_step,
                "in_stride_window": in_stride,
                "in_overlap_0_55_to_0_90": in_both,
            }
        )
    return rows


def main() -> int:
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.analysis.four_gait_signals import analyze_four_gait_signals
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
    from stablewalk.analysis.spectral.config import SpectralAnalysisConfig, default_config
    from stablewalk.analysis.spectral.report import analyze_spectral
    from stablewalk.io.pose_loader import load_pose_sequence

    poses = _latest_poses("abnormal_gait_v2.mp4")
    report: dict = {"pose_json": str(poses) if poses else None}
    if poses is None:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("No pose JSON for abnormal_gait_v2.mp4", flush=True)
        return 1

    seq = load_pose_sequence(poses)
    rec = pose_sequence_to_gait_motion(seq, detected_only=True)
    cycles = analyze_gait_cycles(rec)
    spectral = analyze_spectral(rec, sequence=seq, cycles=cycles)
    four = analyze_four_gait_signals(rec, cycles=cycles)

    primary = spectral.univariate_channel_id
    ac = spectral.autocorr.get(primary)
    cfg = spectral.config or SpectralAnalysisConfig()
    duration = float(spectral.bundle.metadata.get("duration_s") or rec.duration_s)
    metrics = getattr(cycles, "metrics", None)

    channel_acf = {}
    for cid, result in (spectral.autocorr or {}).items():
        channel_acf[cid] = {
            "gait_period_s": result.gait_period_s,
            "stride_period_s": result.stride_period_s,
            "step_period_s": result.step_period_s,
            "cadence_steps_per_min": result.cadence_steps_per_min,
            "n_stride_cycles": result.n_stride_cycles,
            "reliability_label": result.reliability_label,
            "insufficient_reason": result.insufficient_reason,
            "peaks": _peak_rows(result),
            "notes": list(result.notes or []),
        }

    four_acf = {}
    for cid, result in (four.autocorr_filtered or {}).items():
        four_acf[cid] = {
            "gait_period_s": result.gait_period_s,
            "stride_period_s": result.stride_period_s,
            "step_period_s": result.step_period_s,
            "reliability_label": result.reliability_label,
            "insufficient_reason": result.insufficient_reason,
            "peaks": _peak_rows(result),
        }

    step_s = None if ac is None else ac.step_period_s
    stride_s = None if ac is None else ac.stride_period_s
    cadence_from_step = (60.0 / step_s) if step_s else None
    cadence_from_stride_assumed_2 = (120.0 / stride_s) if stride_s else None
    est_from_channel = (duration / stride_s) if stride_s else None
    est_from_session = (
        duration / four.gait_period_s if four.gait_period_s else None
    )

    cycle_list = list(getattr(cycles, "cycles", None) or [])
    detected = len(cycle_list)
    valid = sum(1 for c in cycle_list if getattr(c, "duration_s", 0) and c.duration_s > 1e-6)
    hs_events = [
        {
            "type": e.event_type,
            "side": e.side,
            "time_s": e.time_s,
        }
        for e in (getattr(cycles, "events", None) or [])
        if "heel_strike" in str(getattr(e, "event_type", ""))
    ]

    report.update(
        {
            "duration_s": duration,
            "fps": float(getattr(seq, "fps", 0) or rec.fps),
            "windows": {
                "stride_lag_s": list(cfg.stride_lag_s),
                "step_lag_s": list(cfg.step_lag_s),
                "overlap_s": [max(cfg.step_lag_s[0], cfg.stride_lag_s[0]), min(cfg.step_lag_s[1], cfg.stride_lag_s[1])],
            },
            "spectral": {
                "channel_id": primary,
                "detrend": cfg.detrend,
                "cache": "gui._spectral_report (not session cache)",
                "primary": channel_acf.get(primary),
            },
            "four_gait": {
                "channels": list(four.channel_ids),
                "detrend": "none",
                "cache": "gui._four_gait_report (not session cache)",
                "session_gait_period_s": four.gait_period_s,
                "session_reason": four.gait_period_reason,
                "event_stride_period_s": four.event_stride_period_s,
                "event_stride_n": four.event_stride_n,
                "acf_event_note": four.acf_event_note,
                "per_channel": four_acf,
            },
            "events": {
                "detected_cycles": detected,
                "valid_cycles": valid,
                "cycle_durations_s": [float(c.duration_s) for c in cycle_list],
                "boundary_sides": [getattr(c, "boundary_side", None) for c in cycle_list],
                "heel_strikes": hs_events,
                "event_cadence_steps_per_min": getattr(metrics, "cadence_steps_per_min", None),
                "event_step_time_s": getattr(metrics, "step_time_s", None),
                "event_stride_time_s": getattr(metrics, "stride_time_s", None),
            },
            "conversions": {
                "60_over_step": cadence_from_step,
                "120_over_channel_stride_assumes_two_steps": cadence_from_stride_assumed_2,
                "duration_over_channel_stride": est_from_channel,
                "duration_over_session_stride": est_from_session,
                "step_over_channel_stride": (
                    None
                    if not (step_s and stride_s)
                    else float(step_s) / float(stride_s)
                ),
            },
            "ui_match": {
                "session_1_58": four.gait_period_s,
                "channel_0_667": None if ac is None else ac.gait_period_s,
                "step_0_542": step_s,
                "cadence_110_7": None if ac is None else ac.cadence_steps_per_min,
                "est_cycles_40_5": None if ac is None else ac.n_stride_cycles,
                "detected_3": detected,
            },
            "all_spectral_channels": channel_acf,
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["ui_match"], indent=2, default=str), flush=True)
    print(json.dumps(report["conversions"], indent=2, default=str), flush=True)
    print("session", four.gait_period_s, four.gait_period_reason, flush=True)
    print("primary peaks", report["spectral"]["primary"]["peaks"] if report["spectral"]["primary"] else None, flush=True)
    print("wrote", OUT, flush=True)
    _ = default_config
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
