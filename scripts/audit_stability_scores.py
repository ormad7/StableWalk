"""
Audit stability scoring — domain availability, contribution table, pelvis metrics.

Usage:
  python scripts/audit_stability_scores.py abnormal_gait normal_gait athletic_walking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.biomech_stability import analyze_biomech_stability
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.analysis.stability_scoring import (
    _build_context,
    compute_pelvis_motion_metrics,
)
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.enrichment import enrich_pose_sequence


def _resolve_poses(name: str) -> Path:
    stem = name.replace(".mp4", "")
    path = config.POSES_DIR / f"{stem}_poses.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing poses: {path}")
    return path


def audit_video(name: str) -> None:
    seq = load_pose_sequence(_resolve_poses(name))
    enrich_pose_sequence(seq)
    recording = pose_sequence_to_gait_motion(seq)
    cycles = analyze_gait_cycles(recording)
    result = analyze_biomech_stability(seq, cycles=cycles)
    ctx = _build_context(seq, cycles=cycles, recording=recording)
    pelvis = compute_pelvis_motion_metrics(ctx) if ctx else {}

    print("=" * 72)
    print(f"STABILITY AUDIT — {name}")
    print("=" * 72)
    print(f"Overall Stability:     {result.score:.0f}/100 ({result.classification})")
    print(f"Analysis completeness: {result.completeness_pct:.0f}%")
    print(f"Confidence badge:      {result.confidence_badge}")
    print()
    print(result.contribution_table_text())
    print()

    print("Domain documentation (computed values):")
    for m in result.metrics:
        print(f"\n  {m.name} [{m.availability}, conf={m.confidence:.2f}]")
        print(f"    score={m.score}  weight={m.weight:.2f}  contribution={m.contribution:.2f}")
        print(f"    summary: {m.summary}")
        if m.findings:
            for f in m.findings[:3]:
                print(f"    finding: {f}")
        for k, v in list(m.values.items())[:8]:
            if k != "debug_features":
                print(f"    {k}: {v}")

    print("\nPelvis motion audit (hip-center keypoints, image-normalized coords):")
    if pelvis:
        for key in (
            "pelvis_mediolateral_range",
            "pelvis_vertical_range",
            "pelvis_mediolateral_rms",
            "pelvis_velocity_variability",
            "pelvis_acceleration_variability",
            "pelvis_jerk_metric",
            "normalized_pelvis_sway",
            "pelvis_score",
        ):
            val = pelvis.get(key)
            if key == "pelvis_score":
                pm = result.metric("pelvis_stability")
                val = pm.score if pm else None
            if isinstance(val, float):
                print(f"  {key}: {val:.6f}")
            else:
                print(f"  {key}: {val}")
    else:
        print("  (context unavailable)")

    if result.data_limitations:
        print("\nData limitations:")
        for note in result.data_limitations:
            print(f"  • {note}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit stability score domains")
    parser.add_argument(
        "videos",
        nargs="*",
        default=["abnormal_gait", "normal_gait", "athletic_walking"],
    )
    args = parser.parse_args()
    config.ensure_output_dirs()
    for name in args.videos:
        audit_video(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
