#!/usr/bin/env python3
"""Verify gait phase consistency against contact mask on demo videos."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
from stablewalk.analysis.gait_phase_classification import (
    classify_gait_phase_from_contacts,
    contact_to_display_state,
    format_gait_phase_display,
    validate_phase_contact_consistency,
)
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.pose.enrichment import enrich_pose_sequence

DEMO_POSE_FILES = {
    "Abnormal": "run_f17fb2308965836b_1783601673324_poses.json",
    "Normal": "run_787305e91e7251c7_1783601717711_poses.json",
    "Performance": "run_3ef649f04859ef2c_1783601885066_poses.json",
}


def audit_video(label: str, poses_path: Path, output_dir: Path) -> dict:
    sequence = load_pose_sequence(poses_path)
    enrich_pose_sequence(sequence)
    recording = pose_sequence_to_gait_motion(sequence)
    result = analyze_gait_cycles(recording)
    m = result.metrics

    rows: list[dict] = []
    inconsistent = 0
    blank_phase = 0

    for state in result.per_frame:
        phase = classify_gait_phase_from_contacts(
            state.left_contact,
            state.right_contact,
            contact_confidence=m.contact_confidence,
            confidence_tier=m.confidence_tier,
            left_foot_clearance_m=state.left.foot_clearance_m,
            right_foot_clearance_m=state.right.foot_clearance_m,
        )
        display = format_gait_phase_display(phase)
        ok, expected = validate_phase_contact_consistency(
            state.left_contact, state.right_contact, phase
        )
        if not ok:
            inconsistent += 1
        if display == "—":
            blank_phase += 1

        rows.append(
            {
                "frame_index": state.frame_index,
                "timestamp_s": round(state.time_s, 4),
                "left_contact": state.left_contact,
                "right_contact": state.right_contact,
                "left_display": contact_to_display_state(state.left_contact),
                "right_display": contact_to_display_state(state.right_contact),
                "contact_confidence": round(m.contact_confidence, 3),
                "derived_phase": phase,
                "display_phase": display,
                "consistent": ok,
            }
        )

    csv_path = output_dir / f"gait_phase_debug_{label.lower()}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "video": label,
        "frames": len(rows),
        "inconsistent": inconsistent,
        "blank_phase": blank_phase,
        "csv": str(csv_path),
        "phases": sorted({r["derived_phase"] for r in rows}),
    }


def main() -> int:
    output_dir = config.OUTPUT_DIR / "reports" / "gait_phase_audit"
    summaries: list[dict] = []

    print("# Gait Phase Consistency Audit\n")
    for label, fname in DEMO_POSE_FILES.items():
        path = config.POSES_DIR / fname
        if not path.is_file():
            print(f"SKIP {label}: {path}")
            continue
        summary = audit_video(label, path, output_dir)
        summaries.append(summary)
        print(
            f"## {label}: {summary['frames']} frames, "
            f"{summary['inconsistent']} inconsistent, "
            f"{summary['blank_phase']} blank phases"
        )
        print(f"   Phases: {', '.join(summary['phases'])}")
        print(f"   CSV: {summary['csv']}\n")

    total_bad = sum(s["inconsistent"] + s["blank_phase"] for s in summaries)
    print(f"Total inconsistencies: {total_bad}")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
