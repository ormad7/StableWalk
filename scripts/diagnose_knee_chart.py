"""Diagnose knee angle chart data for demo videos."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.io.pose_loader import load_pose_sequence, detected_frame_indices
from stablewalk.pose.enrichment import enrich_pose_sequence
from stablewalk.ui.viewers.knee_angle_chart import (
    ANGLE_CONVENTION_SUMMARY,
    build_knee_angle_series,
    opensim_ik_available,
)
from stablewalk.ui.viewers.knee_chart_interpretation import (
    cycle_mode_is_available,
    format_diagnostic_report,
    usable_knee_cycle_count,
)


def _analyze_cycles(name: str, poses: Path) -> None:
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
    from stablewalk.analysis.gait_feature_analysis import analyze_gait_features
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion

    seq = load_pose_sequence(poses)
    enrich_pose_sequence(seq)
    motion = pose_sequence_to_gait_motion(seq)
    if motion is None:
        print("Gait motion: unavailable")
        return
    gc = analyze_gait_cycles(motion)
    ik = config.OPENSIM_DIR / name / f"{name}_ik.mot"
    ik_path = ik if ik.is_file() else None
    gf = analyze_gait_features(motion, gc, sequence=seq, ik_mot_path=ik_path)
    print(f"Usable knee cycles:     {usable_knee_cycle_count(gf)}")
    print(f"Cycle mode available:   {cycle_mode_is_available(gf)}")
    print(f"Contact confidence:     {gc.metrics.contact_confidence:.2f}")
    print(f"Gait events:            {len(gc.events)} (L HS {gc.metrics.left_heel_strike_count}, "
          f"R HS {gc.metrics.right_heel_strike_count})")


def main() -> int:
    config.ensure_output_dirs()
    for name in ("abnormal_gait", "normal_gait", "athletic_walking"):
        poses = config.POSES_DIR / f"{name}_poses.json"
        if not poses.is_file():
            print(f"MISSING {poses}")
            continue
        seq = load_pose_sequence(poses)
        enrich_pose_sequence(seq)
        indices = detected_frame_indices(seq)
        ik = config.OPENSIM_DIR / name / f"{name}_ik.mot"
        ik_path = ik if ik.is_file() else None
        ik_avail = opensim_ik_available(ik_path, ik_completed=True)
        series_pose = build_knee_angle_series(
            seq, indices, ik_mot_path=ik_path, source_preference="pose_derived"
        )
        series_auto = build_knee_angle_series(
            seq, indices, ik_mot_path=ik_path, source_preference="auto"
        )
        print("=" * 60)
        print(name.upper())
        print("=" * 60)
        print(f"Convention (documented): {ANGLE_CONVENTION_SUMMARY}")
        print(f"OpenSim IK offered:      {ik_avail}")
        print()
        print("--- Pose-derived source ---")
        print(format_diagnostic_report(series_pose))
        print()
        print("--- Auto source selection ---")
        print(format_diagnostic_report(series_auto))
        print()
        _analyze_cycles(name, poses)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
