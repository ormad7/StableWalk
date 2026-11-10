"""Export frame-level contact observations and Unknown rejection reasons."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from stablewalk.analysis.foot_contact_analysis import (
    CONTACT_UNKNOWN_REASON_LABELS,
    FootContactAnalysisResult,
    iter_contact_frame_evidence,
)

CONTACT_EVIDENCE_FIELDS: tuple[str, ...] = (
    "frame_index",
    "time_s",
    "side",
    "visibility",
    "clearance_m",
    "probability",
    "binary",
    "known",
    "unknown_reason",
    "unknown_reason_label",
    "confidence",
)


def contact_evidence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts by side and Unknown reason. Not a laterality score."""
    out: dict[str, Any] = {
        "n_rows": len(rows),
        "by_side": {},
        "note": (
            "Unknown is a missing observation. Unequal L/R coverage is not "
            "a gait-laterality finding."
        ),
        "reason_labels": dict(CONTACT_UNKNOWN_REASON_LABELS),
    }
    for side in ("left", "right"):
        side_rows = [r for r in rows if r.get("side") == side]
        unknown = [r for r in side_rows if not r.get("known")]
        reasons: dict[str, int] = {}
        for row in unknown:
            code = str(row.get("unknown_reason") or "unspecified")
            reasons[code] = reasons.get(code, 0) + 1
        out["by_side"][side] = {
            "n_frames": len(side_rows),
            "n_known": len(side_rows) - len(unknown),
            "n_unknown": len(unknown),
            "known_pct": (100.0 * (len(side_rows) - len(unknown)) / len(side_rows))
            if side_rows
            else 0.0,
            "unknown_reasons": reasons,
        }
    return out


def write_contact_frame_evidence(
    contact: FootContactAnalysisResult,
    path: str | Path,
    *,
    also_json: bool = True,
) -> Path:
    """Write CSV (and optional JSON sidecar) of per-frame contact evidence."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = iter_contact_frame_evidence(contact)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTACT_EVIDENCE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CONTACT_EVIDENCE_FIELDS})
    if also_json:
        sidecar = dest.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {"summary": contact_evidence_summary(rows), "rows": rows},
                indent=2,
            ),
            encoding="utf-8",
        )
    return dest


__all__ = [
    "CONTACT_EVIDENCE_FIELDS",
    "contact_evidence_summary",
    "write_contact_frame_evidence",
]
