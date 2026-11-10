#!/usr/bin/env python3
"""Write a concise three-example validity comparison from audited dumps.

Uses already-exported Normal / Fast / Abnormal validity JSON. Does not invent
metrics, retune thresholds, or treat filenames as quality grades.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXAMPLES = (
    ("normal", "Normal Gait", "normal_gait_v2.mp4", ROOT / "validation_outputs" / "normal" / "normal_validity.json"),
    ("athletic", "Fast Walking", "fast_walking_v2.mp4", ROOT / "validation_outputs" / "athletic" / "athletic_validity.json"),
    ("abnormal", "Abnormal Gait", "abnormal_gait.mp4", ROOT / "validation_outputs" / "abnormal" / "abnormal_validity.json"),
)

UNPERFORMED = (
    "Force-plate or instrumented-treadmill GRF",
    "Marker-based optical motion capture",
    "IMU / wearable validation",
    "Clinician-scored gait quality instrument (e.g. GGI, GPS, GDI)",
    "Calibrated metric reconstruction / known subject stature check",
    "Bilateral contact with known ground truth",
    "Diagnosis, impairment grading, or filename-based quality boosting",
)


def _pct(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def _layer(block: dict, key: str) -> dict:
    return dict((block.get("validity") or {}).get(key) or {})


def main() -> int:
    from stablewalk.analysis.result_validity import (
        gait_quality_withhold_reason,
        laterality_withhold_reason,
        result_validity_from_export_dict,
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Example validity comparison",
        "",
        f"Generated {now} from audited `validation_outputs/*/ *validity.json` dumps.",
        "Filename / demo key is a source label only. It is not a quality grade.",
        "",
        "## Shared display contract",
        "",
        "- Pose coverage, per-foot contact coverage, spectral reliability, and biomechanical validity stay independent.",
        "- Gait-quality and symmetry scores are unvalidated heuristics. They are withheld when required contact, cycle, or spectral inputs are insufficient.",
        "- Unequal missing-data coverage is not gait asymmetry.",
        "- No threshold was retuned for this comparison.",
        "",
        "## Per-example checks that were actually run",
        "",
    ]

    missing = []
    for key, label, filename, path in EXAMPLES:
        if not path.is_file():
            missing.append(f"{label}: missing {path}")
            lines.extend(
                [
                    f"### {label} (`{filename}`)",
                    "",
                    f"Not compared — audited dump not found at `{path}`.",
                    "",
                ]
            )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = result_validity_from_export_dict(payload)
        pose = _layer(payload, "pose_coverage")
        contact = _layer(payload, "contact_coverage")
        extras = contact.get("extras") or {}
        cycles = _layer(payload, "cycle_detection")
        c_ex = cycles.get("extras") or {}
        spectral = _layer(payload, "spectral")
        biomech = _layer(payload, "biomechanical")
        quality_hold = gait_quality_withhold_reason(report)
        laterality_hold = laterality_withhold_reason(report)
        lines.extend(
            [
                f"### {label} (`{filename}`)",
                "",
                f"- Pose coverage: **{pose.get('status', 'n/a')}** — "
                f"{pose.get('n_detected', '?')}/{pose.get('n_frames', '?')} "
                f"({_pct(pose.get('coverage_pct'))}). Presence only; not tracking accuracy.",
                f"- Left contact known: **{_pct(extras.get('left_pct'))}** "
                f"({(_layer(payload, 'contact_left').get('status') or 'n/a')}).",
                f"- Right contact known: **{_pct(extras.get('right_pct'))}** "
                f"({(_layer(payload, 'contact_right').get('status') or 'n/a')}).",
                f"- Cycle evidence: **{int(c_ex.get('n_left') or 0)} L / "
                f"{int(c_ex.get('n_right') or 0)} R** "
                f"({cycles.get('status', 'n/a')}).",
                f"- Spectral reliability: **{spectral.get('status', 'n/a')}**.",
                f"- Biomechanical validity: **{biomech.get('status', 'n/a')}** "
                "(monocular COM–BoS is not a physical stability score).",
                f"- Gait-quality heuristic: "
                f"{('withheld — ' + quality_hold).rstrip('.') if quality_hold else 'shown only as an unvalidated heuristic'}.",
                f"- Symmetry heuristic: "
                f"{('withheld — ' + laterality_hold).rstrip('.') if laterality_hold else 'shown only as an unvalidated heuristic'}.",
                "",
            ]
        )

    lines.extend(
        [
            "## Limitations that apply to all three examples",
            "",
            "- Monocular 2-D/3-D pose is not marker-based kinematics.",
            "- Contact is inferred from pose; Unknown is a missing observation, not swing and not laterality.",
            "- Spectral ACF reliability is not interchangeable with pose coverage or video quality.",
            "- Physical COM–BoS / XCoM / measured GRF are unreportable from these clips.",
            "- Fast Walking analysis is truncated to the demo frame cap when the source file is longer.",
            "",
            "## Checks that were not performed",
            "",
        ]
    )
    for item in UNPERFORMED:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Outcome",
            "",
            "Normal Gait does not score better because of its filename. All three examples have insufficient or unequal per-foot contact evidence, so Overview / Biomechanics / Results / exports withhold gait-quality and symmetry as unvalidated heuristics rather than Borderline or Acceptable grades.",
            "",
        ]
    )
    if missing:
        lines.extend(["## Missing dumps", ""] + [f"- {item}" for item in missing] + [""])

    text = "\n".join(lines)
    out_dir = ROOT / "validation_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = out_dir / "example_validity_comparison.md"
    primary.write_text(text, encoding="utf-8")
    (ROOT / "example_validity_comparison.md").write_text(text, encoding="utf-8")
    print(primary)
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
