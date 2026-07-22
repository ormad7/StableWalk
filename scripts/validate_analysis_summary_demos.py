#!/usr/bin/env python3
"""Validate Analysis Summary field population across demo pose files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.analysis_summary import build_analysis_summary
from stablewalk.analysis.biomechanical import run_biomechanical_analysis
from stablewalk.analysis.estimated_vgrf_analysis import analyze_estimated_vgrf
from stablewalk.analysis.foot_contact_analysis import analyze_foot_contact
from stablewalk.analysis.gait_comparison_validation import COMPARISON_VIDEOS
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.analysis.gait_feature_analysis import analyze_gait_features
from stablewalk.io.pose_loader import load_pose_sequence


def _find_poses(stem: str) -> Path | None:
    poses_dir = ROOT / "data" / "output" / "poses"
    for name in (f"{stem}_poses.json", f"{stem}.json", "walk_stream_poses.json"):
        p = poses_dir / name
        if p.is_file():
            return p
    matches = list(poses_dir.glob(f"*{stem}*_poses.json"))
    return matches[0] if matches else None


def _field_status(summary, attr: str) -> dict:
    field = getattr(summary, attr, None)
    if field is None:
        return {"available": False, "value": None}
    return {"available": field.available, "value": field.value, "tier": field.tier}


def main() -> int:
    report: dict[str, dict] = {}
    missing: list[str] = []

    for label, stem in COMPARISON_VIDEOS:
        poses_path = _find_poses(stem)
        if poses_path is None:
            missing.append(f"{label} ({stem})")
            continue
        seq = load_pose_sequence(poses_path)
        rec = pose_sequence_to_gait_motion(seq)
        cycles = analyze_gait_cycles(rec)
        contact = analyze_foot_contact(rec, cycles=cycles)
        vgrf = analyze_estimated_vgrf(rec, contact)
        features = analyze_gait_features(rec, cycles, sequence=seq)
        bio = run_biomechanical_analysis(
            rec, seq, cycles=cycles, contact=contact, features=features
        )
        summary = build_analysis_summary(
            source=str(poses_path),
            biomechanical=bio,
            estimated_vgrf=vgrf,
            contact=contact,
            cycles=cycles,
        )
        report[label] = {
            "poses_path": str(poses_path),
            "overall_gait_quality": _field_status(summary, "overall_gait_quality"),
            "cadence": _field_status(summary, "cadence"),
            "walking_speed": _field_status(summary, "walking_speed"),
            "symmetry": _field_status(summary, "symmetry"),
            "stability_margin": _field_status(summary, "stability_margin"),
            "center_of_mass": _field_status(summary, "center_of_mass"),
            "estimated_virtual_grf": _field_status(summary, "estimated_virtual_grf"),
            "gait_events": [
                {"name": e.name, "detected": e.detected} for e in summary.gait_events
            ],
            "video_quality": _field_status(summary, "video_quality"),
            "analysis_confidence": _field_status(summary, "analysis_confidence"),
        }

    out = {"results": report, "missing": missing}
    out_path = ROOT / "data" / "output" / "reports" / "analysis_summary_demo_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if missing:
        print("\nMissing pose data:", ", ".join(missing))
    return 0 if report else 1


if __name__ == "__main__":
    raise SystemExit(main())
