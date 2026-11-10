"""One-shot supervisor validation. Prints JSON. Does not invent results."""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _row(name: str, status: str, evidence: str, notes: str = "") -> dict:
    return {"name": name, "status": status, "evidence": evidence, "notes": notes}


def _video_meta(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "cannot_open"}
    meta = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    meta["duration_s"] = meta["frames"] / meta["fps"] if meta["fps"] > 1e-6 else 0.0
    return meta


def _finite_frac(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(a)))


def _clipping(fig) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    overflow = []
    for ax in fig.axes:
        tb = ax.get_tightbbox(renderer)
        if tb is None:
            continue
        pad = 2.0
        if tb.x0 < fb.x0 - pad or tb.y0 < fb.y0 - pad or tb.x1 > fb.x1 + pad or tb.y1 > fb.y1 + pad:
            overflow.append(str(ax.get_title() or ax.get_ylabel() or "axes"))
    return {
        "n_axes": len(fig.axes),
        "overflow_axes": overflow,
        "clipped": bool(overflow),
    }


def analyze_stem(stem: str) -> dict:
    from stablewalk import config
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.analysis.asymmetry3d.report import analyze_asymmetry3d
    from stablewalk.analysis.four_gait_signals import (
        FOUR_GAIT_CHANNEL_IDS,
        analyze_four_gait_signals,
        correlation_at_lag,
    )
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
    from stablewalk.io.pose_loader import load_pose_sequence
    from stablewalk.models.gait_wireframe import (
        GAIT_WIRE_JOINT_COUNT,
        GAIT_WIRE_JOINT_IDS,
        build_gait_wireframe,
    )
    from stablewalk.ui.tk.gait_correlation_tabs import recording_cache_key
    from stablewalk.ui.viewers.four_gait_charts import (
        build_signal_axes,
        render_four_gait_figure,
    )

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    video = config.DEMO_VIDEOS_DIR / f"{stem}.mp4"
    pose = config.POSES_DIR / f"{stem}_poses.json"
    out: dict = {"stem": stem, "video": str(video), "pose": str(pose), "checks": []}

    if not video.is_file():
        out["checks"].append(_row("video", "missing", f"{video.name} not on disk"))
        return out
    meta = _video_meta(video)
    if "error" in meta:
        out["checks"].append(_row("video", "fail", "OpenCV could not open the container"))
        return out
    out["video_meta"] = meta
    out["checks"].append(
        _row(
            "video",
            "pass",
            f"{meta['frames']} frames @ {meta['fps']:.2f} Hz, {meta['width']}×{meta['height']}",
        )
    )

    if not pose.is_file():
        out["checks"].append(
            _row(
                "pose_cache",
                "missing",
                f"{pose.name} absent — MediaPipe was not re-run in this validation",
            )
        )
        return out

    seq = load_pose_sequence(pose)
    rec = pose_sequence_to_gait_motion(seq)
    out["n_pose_frames"] = len(seq.frames)
    out["n_rec_frames"] = rec.frame_count
    out["fps"] = rec.fps
    out["source"] = rec.source
    out["rec_id"] = id(rec)
    out["seq_id"] = id(seq)
    out["cache_key"] = list(recording_cache_key(rec))
    if rec.frame_count != len(seq.frames):
        out["checks"].append(
            _row(
                "pose_sync",
                "fail",
                f"recording {rec.frame_count} vs pose {len(seq.frames)}",
            )
        )
    else:
        out["checks"].append(
            _row("pose_sync", "pass", f"{rec.frame_count} snapshots match pose frames")
        )

    wire = build_gait_wireframe(rec)
    n_wire = len(wire.frames)
    ids = list(GAIT_WIRE_JOINT_IDS)
    mid = wire.frames[len(wire.frames) // 2] if wire.frames else None
    proj_n = len(mid.joints) if mid is not None else 0
    wire_ok = GAIT_WIRE_JOINT_COUNT == 16 and n_wire == rec.frame_count and proj_n >= 12
    out["wire"] = {
        "declared": GAIT_WIRE_JOINT_COUNT,
        "ids": ids,
        "n_frames": n_wire,
        "projected_joints": proj_n,
    }
    out["checks"].append(
        _row(
            "wire_16",
            "pass" if wire_ok else "fail",
            f"{GAIT_WIRE_JOINT_COUNT} declared joints, {n_wire} frames, {proj_n} projected on frame 0",
        )
    )

    four = analyze_four_gait_signals(rec, sequence=seq)
    out["four_id"] = id(four)
    n_ok = 0
    lengths = {}
    for cid in FOUR_GAIT_CHANNEL_IDS:
        col = np.asarray(four.filtered.get(cid), dtype=float)
        lengths[cid] = int(col.size)
        if col.size == four.times_s.size and _finite_frac(col) >= 0.50:
            n_ok += 1
    time_ok = four.times_s.size == rec.frame_count and abs(four.fps - rec.fps) < 0.6
    four_ok = four.available and n_ok == 4 and time_ok
    out["four"] = {
        "available": four.available,
        "n_ok": n_ok,
        "n_times": int(four.times_s.size),
        "fps": four.fps,
        "gait_period_s": four.gait_period_s,
        "gait_period_reason": four.gait_period_reason,
        "warnings": list(four.warnings)[:8],
    }
    out["checks"].append(
        _row(
            "four_signals",
            "pass" if four_ok else "fail",
            f"{n_ok}/4 channels finite on {four.times_s.size} samples @ {four.fps:.2f} Hz",
        )
    )

    c0 = np.asarray(correlation_at_lag(four, "filtered", 0), dtype=float)
    diag = np.diag(c0)
    sym = float(np.nanmax(np.abs(c0 - c0.T))) if c0.size else 1.0
    unit = bool(diag.size == 4 and np.all(np.isfinite(diag)) and np.allclose(diag, 1.0, atol=1e-6))
    bounded = bool(np.all((c0[~np.isnan(c0)] >= -1.0 - 1e-9) & (c0[~np.isnan(c0)] <= 1.0 + 1e-9)))
    mat_ok = c0.shape == (4, 4) and unit and bounded and sym < 1e-9
    out["pearson"] = {
        "shape": list(c0.shape),
        "diag": [float(x) for x in diag],
        "max_asym": sym,
        "unit_diag": unit,
        "bounded": bounded,
    }
    out["checks"].append(
        _row(
            "pearson_c0",
            "pass" if mat_ok else "fail",
            f"4×4 symmetric (Δ={sym:.1e}), unit diagonal={unit}, |r|≤1={bounded}",
        )
    )

    acf_ok = True
    acf_notes = []
    for cid in FOUR_GAIT_CHANNEL_IDS:
        ac = four.autocorr_filtered.get(cid)
        if ac is None or ac.rho.size < 1:
            acf_ok = False
            acf_notes.append(f"{cid} missing")
            continue
        rho0 = float(ac.rho[0])
        if not math.isfinite(rho0) or abs(rho0 - 1.0) > 1e-5:
            acf_ok = False
            acf_notes.append(f"{cid} ρ(0)={rho0}")
        else:
            acf_notes.append(f"{cid} ρ(0)=1")
    out["acf"] = {
        "period_s": four.gait_period_s,
        "reason": four.gait_period_reason,
        "notes": acf_notes,
    }
    out["checks"].append(
        _row(
            "acf",
            "pass" if acf_ok else "fail",
            "; ".join(acf_notes)
            + (
                f". Period withheld: {four.gait_period_reason}"
                if four.gait_period_s is None
                else f". Period {four.gait_period_s:.3f} s"
            ),
        )
    )

    xcorr_ok = True
    xcorr_notes = []
    for name, xc in (four.xcorr_filtered or {}).items():
        rho = np.asarray(getattr(xc, "rho", []), dtype=float)
        if rho.size < 3 or not np.isfinite(rho).any():
            xcorr_ok = False
            xcorr_notes.append(f"{name} empty")
            continue
        mid = int(np.argmin(np.abs(np.asarray(xc.lags_s))))
        xcorr_notes.append(
            f"{name} peak τ={xc.peak_lag_s} r={xc.peak_value}"
        )
    out["checks"].append(
        _row(
            "xcorr",
            "pass" if xcorr_ok and four.xcorr_filtered else "fail",
            "; ".join(xcorr_notes[:4]) or "no xcorr series",
        )
    )

    spec_ok = True
    spec_notes = []
    for cid in FOUR_GAIT_CHANNEL_IDS:
        spec = (four.spectrum_filtered or {}).get(cid)
        if spec is None:
            spec_ok = False
            spec_notes.append(f"{cid} missing")
            continue
        q = str(getattr(spec, "quality_status", "") or "")
        f0 = getattr(spec, "dominant_hz", None)
        if q in {"insufficient"} and f0 not in (None,):
            # honest: insufficient should not fabricate f0
            if f0 is not None and math.isfinite(float(f0)):
                spec_ok = False
                spec_notes.append(f"{cid} fabricated f0 under {q}")
            else:
                spec_notes.append(f"{cid} {q}")
        else:
            spec_notes.append(f"{cid} {q}")
    spec_corr = np.asarray(four.spectral_corr_filtered, dtype=float) if four.spectral_corr_filtered is not None else np.zeros((0, 0))
    spec_shape_ok = spec_corr.shape == (4, 4)
    out["spectral"] = {
        "notes": spec_notes,
        "corr_shape": list(spec_corr.shape),
        "freq_compare": bool(four.freq_compare_filtered),
    }
    out["checks"].append(
        _row(
            "fft_spectral",
            "pass" if spec_ok else "fail",
            "; ".join(spec_notes),
        )
    )
    out["checks"].append(
        _row(
            "spectral_corr",
            "pass" if spec_shape_ok else "warn",
            f"spectral-shape matrix {list(spec_corr.shape)}"
            + ("" if spec_shape_ok else " — withheld or incomplete on this clip"),
        )
    )

    cycles = analyze_gait_cycles(rec)
    asym = analyze_asymmetry3d(rec, cycles=cycles, sequence=seq)
    qnames = {c.name: c.status for c in asym.quality.checks}
    out["asymmetry"] = {
        "available": asym.available,
        "provenance": asym.kinematics.provenance,
        "n_left": len(asym.events.left_cycles),
        "n_right": len(asym.events.right_cycles),
        "quality": qnames,
        "ground_reliable": asym.quality.ground_reliable,
        "warnings": list(asym.warnings)[:8],
    }
    out["checks"].append(
        _row(
            "reconstruction_3d",
            "pass" if asym.available and asym.kinematics.n_frames == rec.frame_count else "fail",
            f"{asym.kinematics.n_frames} frames, {asym.kinematics.provenance}",
        )
    )
    out["checks"].append(
        _row(
            "asymmetry",
            "pass" if asym.available else "fail",
            f"L/R cycles {len(asym.events.left_cycles)}/{len(asym.events.right_cycles)}; "
            + ", ".join(f"{k}={v}" for k, v in qnames.items()),
        )
    )

    # Figure layout / clipping on the four required views.
    clip_notes = []
    clip_fail = False
    from stablewalk.ui.viewers.four_gait_charts import (
        build_corr_axes,
        build_signal_axes,
        draw_correlation_heatmap,
        draw_spectral_correlation_heatmap,
        render_four_gait_corr_figure,
        render_four_gait_spectrum_figure,
    )

    for label, builder in (
        ("time_2x2", "time"),
        ("acf_xcorr", "acf"),
        ("spectrum", "spec"),
        ("pearson", "pearson"),
        ("spectral_corr", "scorr"),
    ):
        fig = Figure(figsize=(7.2, 6.0), dpi=100)
        try:
            if builder == "time":
                axes = build_signal_axes(fig, combined=False)
                render_four_gait_figure(fig, axes, four, "filtered", combined=False)
            elif builder == "acf":
                axes = build_corr_axes(fig)
                render_four_gait_corr_figure(fig, axes, four, "filtered")
            elif builder == "spec":
                axes = build_signal_axes(fig, combined=False)
                render_four_gait_spectrum_figure(fig, axes, four, "filtered")
            elif builder == "pearson":
                ax = fig.add_subplot(111)
                draw_correlation_heatmap(ax, four, "filtered", lag_samples=0)
            else:
                ax = fig.add_subplot(111)
                draw_spectral_correlation_heatmap(ax, four, "filtered")
            info = _clipping(fig)
            if info["clipped"]:
                clip_fail = True
                clip_notes.append(f"{label} overflow {info['overflow_axes']}")
            else:
                clip_notes.append(f"{label} {info['n_axes']} axes ok")
        except Exception as exc:
            clip_fail = True
            clip_notes.append(f"{label} {type(exc).__name__}: {exc}")
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)
    out["checks"].append(
        _row(
            "plot_layout",
            "fail" if clip_fail else "pass",
            "; ".join(clip_notes),
        )
    )
    return out


def main() -> int:
    from stablewalk.research.compare import compare_independent_clips
    from stablewalk.research.pipeline import analyze_independent_clip, slots_are_independent
    from stablewalk.io.pose_loader import load_pose_sequence
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk import config

    report: dict = {
        "schema": "stablewalk-supervisor-validation",
        "clips": [],
        "compare": {},
        "summary": {},
    }
    stems = ("normal_gait", "abnormal_gait")
    per = {}
    for stem in stems:
        try:
            per[stem] = analyze_stem(stem)
        except Exception as exc:
            per[stem] = {
                "stem": stem,
                "checks": [_row("pipeline", "fail", f"{type(exc).__name__}: {exc}")],
                "traceback": traceback.format_exc(),
            }
        report["clips"].append(per[stem])

    a = per.get("normal_gait") or {}
    b = per.get("abnormal_gait") or {}
    both = a.get("rec_id") and b.get("rec_id")
    if both:
        pose_a = config.POSES_DIR / "normal_gait_poses.json"
        pose_b = config.POSES_DIR / "abnormal_gait_poses.json"
        seq_a = load_pose_sequence(pose_a)
        seq_b = load_pose_sequence(pose_b)
        rec_a = pose_sequence_to_gait_motion(seq_a)
        rec_b = pose_sequence_to_gait_motion(seq_b)
        slot_a = analyze_independent_clip(
            sequence=seq_a, recording=rec_a, video_id="normal_gait", label="normal"
        )
        slot_b = analyze_independent_clip(
            sequence=seq_b, recording=rec_b, video_id="abnormal_gait", label="abnormal"
        )
        independent = slots_are_independent(slot_a, slot_b)
        cmp = compare_independent_clips(slot_a, slot_b)
        shared_ids = (
            slot_a.sequence_id == slot_b.sequence_id
            or slot_a.gait_motion_id == slot_b.gait_motion_id
        )
        four_same = a.get("four_id") == b.get("four_id")
        cache_same = a.get("cache_key") == b.get("cache_key")
        cad = cmp.scalar_by_name("cadence")
        report["compare"] = {
            "independent": independent and cmp.independent and not shared_ids,
            "same_file": False,
            "shared_objects": shared_ids,
            "four_gait_ids_equal": four_same,
            "cache_keys_equal": cache_same,
            "cadence_a": None if cad is None else cad.value_a,
            "cadence_b": None if cad is None else cad.value_b,
            "cadence_delta": None if cad is None else cad.delta,
            "scalars": [
                {
                    "name": s.name,
                    "a": s.value_a,
                    "b": s.value_b,
                    "delta": s.delta,
                }
                for s in cmp.scalars
            ],
            "warnings": list(cmp.warnings)[:8],
        }
        report["clips"].append(
            {
                "stem": "compare",
                "checks": [
                    _row(
                        "compare_isolation",
                        "pass" if independent and not shared_ids and not four_same and not cache_same else "fail",
                        f"independent={independent}, shared_objects={shared_ids}, "
                        f"four_ids_equal={four_same}, cache_equal={cache_same}",
                    ),
                    _row(
                        "compare_values",
                        "pass"
                        if cad and cad.value_a is not None and cad.value_b is not None and cad.delta not in (0, None)
                        else "warn",
                        f"cadence A={getattr(cad, 'value_a', None)} B={getattr(cad, 'value_b', None)} Δ={getattr(cad, 'delta', None)}",
                    ),
                ],
            }
        )
    else:
        report["compare"] = {
            "independent": False,
            "note": "One or both demo pose caches were missing — no invented comparison.",
        }

    statuses = [c["status"] for clip in report["clips"] for c in clip.get("checks", [])]
    report["summary"] = {
        "pass": statuses.count("pass"),
        "fail": statuses.count("fail"),
        "warn": statuses.count("warn"),
        "missing": statuses.count("missing"),
    }
    dest = ROOT / "docs" / "_supervisor_final_validation.json"
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["summary"]))
    print(f"wrote {dest}")
    return 0 if report["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
