#!/usr/bin/env python3
"""
Example: load AMP reference and contact-sync exports from StableWalk.

Run after:
    python main.py --real-to-sim data/demo_videos/normal_gait.mp4

Usage:
    python scripts/load_amp_reference_example.py normal_gait
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect StableWalk AMP / contact-sync exports.")
    parser.add_argument(
        "run_name",
        nargs="?",
        default="normal_gait",
        help="Run folder under output/motion_reference/ (default: normal_gait)",
    )
    args = parser.parse_args()
    run_dir = config.MOTION_REFERENCE_EXPORT_DIR / args.run_name

    amp_path = run_dir / "amp_reference_motion.npz"
    sync_path = run_dir / "contact_sync_reward.npz"
    report_path = run_dir / "real_to_sim_pipeline_report.json"

    if not amp_path.is_file():
        print(f"AMP reference not found: {amp_path}")
        print("Run: python main.py --real-to-sim <video>")
        return 1

    amp = np.load(amp_path, allow_pickle=False)
    meta = json.loads(str(amp["amp_metadata_json"]))
    print(f"AMP reference: {amp_path.name}")
    print(f"  frames: {len(amp['timestamps'])}  fps: {float(amp['fps']):.1f}")
    print(f"  robot: {meta.get('robot_name')}  scale: {meta.get('scale_factor', '—')}")
    style = meta.get("gait_style", {})
    if style:
        print(f"  gait style: {style.get('style_summary', '—')}")

    if sync_path.is_file():
        sync = np.load(sync_path, allow_pickle=False)
        mean_r = float(sync["combined_reward"].mean())
        print(f"\nContact sync: {sync_path.name}")
        print(f"  mean combined reward: {mean_r:.0%}")
        print(f"  left contact frames: {int(sync['left_contact_mask'].sum())}")
        print(f"  right contact frames: {int(sync['right_contact_mask'].sum())}")
    else:
        print("\nContact sync NPZ not found (pose sequence may be missing).")

    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"\nPipeline report: {report_path.name}")
        for stage in report.get("stages", []):
            print(f"  {stage['stage']}: {stage['status']} — {stage['detail'][:70]}")

    print("\nNext step: load amp_reference_motion.npz in Isaac Lab AMP training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
