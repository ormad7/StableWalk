"""Export a research comparison report (JSON / CSV / Markdown)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stablewalk.research.compare import ResearchComparison
from stablewalk.research.categories import CLINICAL_POLICY


def export_comparison_json(report: ResearchComparison, path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def export_comparison_csv(report: ResearchComparison, path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[Any]] = [
        ["# research_video_comparison"],
        ["independent", report.independent],
        ["delta", "B - A"],
        [],
        ["# scalars"],
        ["name", "display", "units", "A", "B", "delta", "relative_pct", "family", "note"],
    ]
    for s in report.scalars:
        rows.append(
            [
                s.name,
                s.display,
                s.units,
                s.value_a,
                s.value_b,
                s.delta,
                s.relative_pct,
                s.family,
                s.note,
            ]
        )
    rows.extend([[], ["# limitations"]])
    for line in report.limitations:
        rows.append([line])
    rows.extend([[], ["# warnings"]])
    for line in report.warnings:
        rows.append([line])
    with dest.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return dest


def export_comparison_markdown(report: ResearchComparison, path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    a = report.slot_a
    b = report.slot_b
    lines = [
        "# Gait video comparison report",
        "",
        f"- Video A: `{a.video_id}` ({a.label})",
        f"- Video B: `{b.video_id}` ({b.label})",
        f"- Independent pipelines: **{'yes' if report.independent else 'NO'}**",
        f"- Δ definition: B − A",
        f"- Exported: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Data quality",
        "",
        f"- A: {a.n_frames} frames @ {a.fps:.3g} Hz · detected {a.quality.get('detected_fraction', 0):.0%}",
        f"- B: {b.n_frames} frames @ {b.fps:.3g} Hz · detected {b.quality.get('detected_fraction', 0):.0%}",
        f"- A cycles: {a.quality.get('gait_cycle_count', 0)} · B cycles: {b.quality.get('gait_cycle_count', 0)}",
        "",
        "## Numerical differences",
        "",
        "| Metric | A | B | Δ (B−A) | Units | Family |",
        "|---|---:|---:|---:|---|---|",
    ]
    for s in report.scalars:
        lines.append(
            f"| {s.display} | {_fmt(s.value_a)} | {_fmt(s.value_b)} | {_fmt(s.delta)} | {s.units} | {s.family} |"
        )
    lines.extend(["", "## Waveforms", ""])
    for w in report.waveforms:
        lines.append(
            f"- {w.name}: aligned={w.aligned}, RMSE={_fmt(w.rmse)} {w.units}. {w.note}"
        )
    lines.extend(["", "## Limitations", ""])
    for item in report.limitations:
        lines.append(f"- {item}")
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        for item in report.warnings:
            lines.append(f"- {item}")
    lines.extend(["", "## Clinical labeling", "", CLINICAL_POLICY, ""])
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4g}"
