"""Export StableWalk Analysis Summary to JSON and human-readable text."""

from __future__ import annotations

import json
from pathlib import Path

from stablewalk.analysis.analysis_summary import AnalysisSummary
from stablewalk.ui.scientific_labels import TIER_DEFINITIONS, format_tier_badge

SUMMARY_SCHEMA_VERSION = "1.2"


def export_analysis_summary_json(summary: AnalysisSummary, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "kind": "analysis_summary",
        "methodology_note": (
            "All metrics are computed from monocular pose estimation unless explicitly "
            "marked measured. Tiers: measured, estimated, derived, calculated."
        ),
        **summary.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _append_field(lines: list[str], field, *, indent: str = "") -> None:
    if field is None:
        return
    tier = format_tier_badge(field.tier)
    tier_tag = f" [{tier}]" if tier else ""
    lines.append(f"{indent}{field.label}: {field.value}{tier_tag}")
    if not field.available and field.reason:
        lines.append(f"{indent}  Note: {field.reason}")


def format_analysis_summary_report(summary: AnalysisSummary) -> str:
    """Plain-text report suitable for thesis appendices and sharing."""
    lines: list[str] = [
        "Gait Analysis Report",
        "StableWalk Analysis Summary",
        "=" * 40,
        "",
        "Methodology",
        "-" * 40,
        "StableWalk analyzes walking from monocular video using MediaPipe pose",
        "estimation. Metrics are labeled by data tier:",
        "  measured   — instrumented or calibrated sensor data (not from video alone)",
        "  estimated  — pose-derived proxies (COM, vGRF, walking speed scale)",
        "  derived    — composite indices from estimated inputs",
        "  calculated — deterministic timing/geometry (cadence, phase %)",
        "",
        "This report does not constitute a clinical diagnosis.",
    ]

    if summary.source:
        lines.extend(["", f"Source: {summary.source}"])
    if summary.timestamp_s is not None:
        lines.append(
            f"Playhead sync: frame {summary.frame_index} at {summary.timestamp_s:.2f} s"
        )

    lines.extend(["", "Gait Performance", "-" * 40])
    _append_field(lines, summary.overall_gait_quality)
    _append_field(lines, summary.walking_speed)
    _append_field(lines, summary.cadence)
    _append_field(lines, summary.symmetry)
    _append_field(lines, summary.stability_margin)

    lines.extend(["", "Biomechanics", "-" * 40])
    _append_field(lines, summary.center_of_mass)
    _append_field(lines, summary.estimated_virtual_grf)
    _append_field(lines, summary.joint_rom_summary)

    if summary.gait_events:
        lines.extend(["", "Detected Events", "-" * 40])
        for ev in summary.gait_events:
            mark = "✓ detected" if ev.detected else "✗ not detected"
            lines.append(f"  {ev.name}: {mark} — {ev.detail}")

    lines.extend(["", "Analysis Confidence", "-" * 40])
    _append_field(lines, summary.video_quality)
    _append_field(lines, summary.tracking_confidence)
    _append_field(lines, summary.pipeline_confidence)
    if summary.analysis_confidence:
        _append_field(lines, summary.analysis_confidence)

    if summary.scientific_interpretation:
        lines.extend([
            "",
            "Interpretation",
            "-" * 40,
            summary.scientific_interpretation,
        ])

    if summary.pipeline_status:
        diagram = summary.pipeline_status.get("diagram") or []
        if diagram:
            lines.extend(["", "Pipeline Status", "-" * 40])
            for stage in diagram:
                symbol = stage.get("status_symbol", "")
                label = stage.get("status_label", "")
                lines.append(
                    f"  {stage.get('label', '')}: {symbol} {label} — {stage.get('detail', '')}"
                )

    lines.extend(["", "Data Tier Definitions", "-" * 40])
    terminology = summary.terminology or TIER_DEFINITIONS
    for tier, definition in terminology.items():
        lines.append(f"  {tier}: {definition}")

    lines.append("")
    return "\n".join(lines)


def export_analysis_summary_report(summary: AnalysisSummary, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_analysis_summary_report(summary), encoding="utf-8")
    return path


__all__ = [
    "SUMMARY_SCHEMA_VERSION",
    "export_analysis_summary_json",
    "export_analysis_summary_report",
    "format_analysis_summary_report",
]
