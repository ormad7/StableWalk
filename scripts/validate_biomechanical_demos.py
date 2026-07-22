#!/usr/bin/env python3
"""
Validate biomechanical analysis across demo videos (Normal, Athletic, Abnormal).

Requires processed pose JSON under data/output/poses/ or demo videos with pipeline run.
Does not force expected outcomes — reports measured vs expected patterns honestly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.biomechanical import run_biomechanical_analysis
from stablewalk.analysis.biomechanical.walking_speed import (
    format_walking_speed_display,
    is_plausible_walking_speed,
    is_reportable_walking_speed,
)
from stablewalk.analysis.gait_comparison_validation import COMPARISON_VIDEOS
from stablewalk.io.pose_loader import load_pose_sequence


def _find_poses(stem: str) -> Path | None:
    poses_dir = ROOT / "data" / "output" / "poses"
    for name in (f"{stem}_poses.json", f"{stem}.json", "walk_stream_poses.json"):
        p = poses_dir / name
        if p.is_file():
            return p
    matches = list(poses_dir.glob(f"*{stem}*_poses.json"))
    return matches[0] if matches else None


def main() -> int:
    results: dict[str, dict] = {}
    missing: list[str] = []

    for label, stem in COMPARISON_VIDEOS:
        poses_path = _find_poses(stem)
        if poses_path is None:
            missing.append(f"{label} ({stem})")
            continue
        seq = load_pose_sequence(poses_path)
        rec = pose_sequence_to_gait_motion(seq)
        bio = run_biomechanical_analysis(rec, seq)
        ws = bio.gait_metrics.walking_speed if bio.gait_metrics else None
        results[label] = {
            "poses_path": str(poses_path),
            "gait_quality_score": bio.gait_quality.score if bio.gait_quality else None,
            "symmetry_pct": (
                bio.symmetry.overall_symmetry_pct.value
                if bio.symmetry and bio.symmetry.overall_symmetry_pct
                else None
            ),
            "stable_margin_pct": (
                bio.stability_margin.stable_pct if bio.stability_margin else None
            ),
            "cadence": (
                bio.gait_metrics.cadence.value
                if bio.gait_metrics and bio.gait_metrics.cadence
                else None
            ),
            "walking_speed_m_s": ws.value if ws else None,
            "walking_speed_display": format_walking_speed_display(ws),
            "walking_speed_plausible": (
                is_plausible_walking_speed(ws.value) if ws and ws.value is not None else False
            ),
            "walking_speed_reportable": is_reportable_walking_speed(ws),
            "walking_speed_confidence": ws.confidence if ws else None,
            "video_quality": (
                bio.video_quality.overall_quality_score if bio.video_quality else None
            ),
            "abnormalities": bio.abnormalities,
        }

    out_path = ROOT / "data" / "output" / "reports" / "biomechanical_demo_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": results, "missing": missing}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if missing:
        print("\nNote: Missing pose data for:", ", ".join(missing))
        print("Run pipeline on demo videos or place *_poses.json in data/output/poses/")

    # Pattern check (informational — not pass/fail gates)
    if "normal" in results and "abnormal" in results:
        n, a = results["normal"], results["abnormal"]
        if n.get("symmetry_pct") and a.get("symmetry_pct"):
            if n["symmetry_pct"] < a["symmetry_pct"]:
                print(
                    "\nObservation: Abnormal symmetry index higher than Normal — "
                    "may reflect clip length, view, or label mismatch; not forced."
                )
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
