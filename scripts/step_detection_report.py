"""Step detection debug report for Abnormal / Normal / Athletic demo videos."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from stablewalk import config
from stablewalk.analysis.biomech_stability import analyze_biomech_stability
from stablewalk.pose.estimation import PoseEstimator
from stablewalk.pose.gait_step_detection import (
    MIN_STEP_INTERVAL_S,
    detect_gait_steps,
    export_foot_landmark_signals,
    format_step_detection_report,
)
from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path

LANDMARKS = (
    "left_heel", "right_heel",
    "left_ankle", "right_ankle",
    "left_foot_index", "right_foot_index",
)


def _print_signal_summary(name: str, sig) -> None:
    valid = sig.raw_y[~np.isnan(sig.raw_y)]
    if valid.size == 0:
        print(f"  {name}: no data")
        return
    print(
        f"  {name}: n={valid.size}  "
        f"y range={float(np.min(valid)):.4f}..{float(np.max(valid)):.4f}  "
        f"std={float(np.std(valid)):.4f}"
    )


def analyze_video(key: str) -> dict:
    ex = next(e for e in DEMO_GAIT_EXAMPLES if e.key == key)
    path = demo_path(ex)
    with PoseEstimator(video_mode=True) as est:
        seq = est.process_video(path, enrich_gait=False)

    gait = detect_gait_steps(seq)
    stability = analyze_biomech_stability(seq)
    signals = export_foot_landmark_signals(seq)

    return {
        "key": key,
        "path": path,
        "gait": gait,
        "stability": stability,
        "signals": signals,
    }


def main() -> int:
    config.ensure_output_dirs()
    print("=" * 72)
    print("STEP DETECTION INVESTIGATION — StableWalk demo videos")
    print("=" * 72)
    print()
    print("Implementation: stablewalk/pose/gait_step_detection.py")
    print("Input: vertical foot landmark y (image coords, down = positive)")
    print("Signal: linear detrend + moving-average low-pass smoothing")
    print(f"Thresholds: min interval = fps × {MIN_STEP_INTERVAL_S}s")
    print("  prominence >= 14% of signal range; swing >= 16% of range (hysteresis)")
    print("  body-height-scaled minimum oscillation amplitude")
    print()

    import stablewalk.pose.gait_step_detection as gait_step_detection  # noqa: F401

    results = {}
    for key in ("abnormal", "normal", "athletic"):
        if not demo_path(next(e for e in DEMO_GAIT_EXAMPLES if e.key == key)).is_file():
            print(f"MISSING: {key}")
            return 1
        results[key] = analyze_video(key)

    for key, data in results.items():
        gait = data["gait"]
        print(format_step_detection_report(str(data["path"]), gait))
        print()
        print("RAW LANDMARK SIGNALS (vertical y, normalized image coords):")
        for lm in LANDMARKS:
            _print_signal_summary(lm, data["signals"][lm])
        print()
        print("LEFT EVENTS:")
        print(gait.left.event_frame_indices)
        print("RIGHT EVENTS:")
        print(gait.right.event_frame_indices)
        print()
        sc = data["stability"].metric("step_consistency")
        print(
            f"Stability step-consistency score: {sc.score:.1f}/100  "
            f"(left={sc.values.get('left_steps')}, right={sc.values.get('right_steps')}, "
            f"confidence={sc.values.get('step_detection_confidence')})"
        )
        print("-" * 72)
        print()

    print("SUMMARY — detected gait steps (left + right)")
    print(f"{'Demo':<12} {'Duration':>10} {'L steps':>8} {'R steps':>8} {'Total':>8} {'Cadence':>12} {'Conf':>6}")
    print("-" * 72)
    for key, data in results.items():
        g = data["gait"]
        cad = f"{g.cadence_hz:.2f} Hz" if g.cadence_hz else "N/A"
        print(
            f"{key.capitalize():<12} {g.duration_s:>9.2f}s "
            f"{g.left.step_count:>8} {g.right.step_count:>8} {g.total_steps:>8} "
            f"{cad:>12} {g.confidence:>6}"
        )

    print()
    print("NOTE: The Walk Summary column formerly labeled 'Steps' shows the")
    print("Step Regularity score (0–100), NOT the raw step count. Actual detected")
    print("step counts are shown under 'Detected gait steps' in the UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
