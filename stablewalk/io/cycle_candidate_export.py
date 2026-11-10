"""Export HS–HS cycle candidates with valid-contact coverage and acceptance."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

CYCLE_CANDIDATE_FIELDS: tuple[str, ...] = (
    "candidate_index",
    "side",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "n_frames",
    "n_known_contact",
    "n_known_stance",
    "n_known_swing",
    "known_contact_coverage",
    "n_valid_pose_frames",
    "valid_pose_coverage",
    "longest_internal_gap_s",
    "accepted",
    "acceptance_reason",
    "source",
)


def write_cycle_candidates_csv(
    candidates: Sequence[Any],
    path: str | Path,
    *,
    also_json: bool = True,
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = [_candidate_row(item) for item in candidates]
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CYCLE_CANDIDATE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CYCLE_CANDIDATE_FIELDS})
    if also_json:
        sidecar = dest.with_suffix(".json")
        n_accepted = sum(1 for row in rows if row.get("accepted"))
        sidecar.write_text(
            json.dumps(
                {
                    "n_candidates": len(rows),
                    "n_accepted": n_accepted,
                    "n_rejected": len(rows) - n_accepted,
                    "note": (
                        "Candidates are binary-contact HS–HS intervals. "
                        "Accepted cycles require known stance, confirmed swing, "
                        "and no unsupported pose gap (internal invalid-pose gap "
                        "larger than one missed sample). Left and right accepted "
                        "cycles are both published. Gaps are not interpolated."
                    ),
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return dest


def _candidate_row(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        payload = dict(item)
    else:
        payload = {
            "candidate_index": getattr(item, "candidate_index", ""),
            "side": getattr(item, "side", ""),
            "start_frame": getattr(item, "start_frame", ""),
            "end_frame": getattr(item, "end_frame", ""),
            "start_time_s": getattr(item, "start_time_s", ""),
            "end_time_s": getattr(item, "end_time_s", ""),
            "duration_s": getattr(item, "duration_s", ""),
            "n_frames": getattr(item, "n_frames", ""),
            "n_known_contact": getattr(item, "n_known_contact", ""),
            "n_known_stance": getattr(item, "n_known_stance", ""),
            "n_known_swing": getattr(item, "n_known_swing", ""),
            "known_contact_coverage": getattr(item, "known_contact_coverage", ""),
            "n_valid_pose_frames": getattr(item, "n_valid_pose_frames", ""),
            "valid_pose_coverage": getattr(item, "valid_pose_coverage", ""),
            "longest_internal_gap_s": getattr(item, "longest_internal_gap_s", ""),
            "accepted": getattr(item, "accepted", False),
            "acceptance_reason": getattr(item, "acceptance_reason", ""),
            "source": getattr(item, "source", "binary_contact_hs_hs"),
        }
    return {key: payload.get(key, "") for key in CYCLE_CANDIDATE_FIELDS}
