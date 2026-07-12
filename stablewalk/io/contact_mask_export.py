"""Export foot-contact masks and gait events to ``contact_mask.npz``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult

CONTACT_MASK_SCHEMA_VERSION = "1.0"


def export_contact_mask_npz(
    contact: FootContactAnalysisResult,
    output_path: Path,
    *,
    run_name: str = "",
) -> Path:
    """
    Export per-frame contact probabilities, binary masks, events, and phases.

    Output: ``data/output/motion_reference/<run_name>/contact_mask.npz``
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(contact.per_frame)
    payload: dict[str, Any] = {
        "schema_version": CONTACT_MASK_SCHEMA_VERSION,
        "run_name": run_name or output_path.parent.name,
        "frame_count": np.int32(n),
        "fps": np.float64(contact.fps),
        "timestamps": contact.timestamps,
        "left_contact_probability": contact.left_contact_probability,
        "right_contact_probability": contact.right_contact_probability,
        "left_contact_binary": contact.left_contact_binary,
        "right_contact_binary": contact.right_contact_binary,
        "left_heel_strike": np.array([f.left_heel_strike for f in contact.per_frame], dtype=np.int8),
        "right_heel_strike": np.array(
            [f.right_heel_strike for f in contact.per_frame], dtype=np.int8
        ),
        "left_toe_off": np.array([f.left_toe_off for f in contact.per_frame], dtype=np.int8),
        "right_toe_off": np.array([f.right_toe_off for f in contact.per_frame], dtype=np.int8),
        "left_foot_substate": np.array(
            [f.left_foot_substate for f in contact.per_frame], dtype="U16"
        ),
        "right_foot_substate": np.array(
            [f.right_foot_substate for f in contact.per_frame], dtype="U16"
        ),
        "macro_phase": np.array([f.macro_phase for f in contact.per_frame], dtype="U20"),
        "left_confidence": np.array(
            [f.left_confidence for f in contact.per_frame], dtype=np.float64
        ),
        "right_confidence": np.array(
            [f.right_confidence for f in contact.per_frame], dtype=np.float64
        ),
        "stance_phase": np.array(
            [1 if f.macro_phase == "stance" else 0 for f in contact.per_frame], dtype=np.int8
        ),
        "swing_phase": np.array(
            [1 if f.macro_phase == "swing" else 0 for f in contact.per_frame], dtype=np.int8
        ),
        "double_support_phase": np.array(
            [1 if f.macro_phase == "double_support" else 0 for f in contact.per_frame],
            dtype=np.int8,
        ),
        "metrics_json": np.array([str(contact.metrics.to_dict())], dtype="U4096"),
        "gait_event_confidence": np.float64(contact.metrics.gait_event_confidence),
        "valid_gait_cycle_count": np.int32(contact.metrics.valid_gait_cycle_count),
    }

    if contact.events:
        payload["event_types"] = np.array([e.event_type for e in contact.events], dtype="U24")
        payload["event_frame_indices"] = np.array(
            [e.frame_index for e in contact.events], dtype=np.int32
        )
        payload["event_times"] = np.array([e.time_s for e in contact.events], dtype=np.float64)

    np.savez_compressed(output_path, **payload)
    return output_path


__all__ = ["CONTACT_MASK_SCHEMA_VERSION", "export_contact_mask_npz"]
