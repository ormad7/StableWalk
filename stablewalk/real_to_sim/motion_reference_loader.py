"""Load and validate ``stablewalk_motion.npz`` motion reference files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stablewalk.io.motion_reference_export import MOTION_REFERENCE_SCHEMA_VERSION

REQUIRED_ARRAYS = (
    "timestamps",
    "fps",
    "root_positions",
    "canonical_joint_positions",
    "left_contact_mask",
    "right_contact_mask",
    "body_scale_metadata_json",
    "joint_ids_json",
)


@dataclass(frozen=True)
class MotionReferenceData:
    """Loaded motion reference arrays."""

    path: Path
    timestamps: np.ndarray
    fps: float
    root_positions: np.ndarray
    canonical_joint_positions: np.ndarray
    left_contact_mask: np.ndarray
    right_contact_mask: np.ndarray
    joint_ids: list[str]
    metadata: dict[str, Any]
    root_orientations: np.ndarray | None = None
    gait_style_json: str | None = None

    @property
    def frame_count(self) -> int:
        return int(len(self.timestamps))


def validate_motion_reference(path: Path) -> tuple[bool, list[str]]:
    """Check NPZ schema and array shapes."""
    issues: list[str] = []
    path = Path(path)
    if not path.is_file():
        return False, [f"File not found: {path}"]

    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        return False, [f"Cannot load NPZ: {exc}"]

    for key in REQUIRED_ARRAYS:
        if key not in data:
            issues.append(f"Missing array: {key}")

    if issues:
        return False, issues

    n = len(data["timestamps"])
    if data["root_positions"].shape != (n, 3):
        issues.append(f"root_positions shape mismatch: expected ({n}, 3)")
    if data["left_contact_mask"].shape != (n,):
        issues.append("left_contact_mask length mismatch")
    if data["right_contact_mask"].shape != (n,):
        issues.append("right_contact_mask length mismatch")

    try:
        meta = json.loads(str(data["body_scale_metadata_json"]))
        if "schema_version" not in meta:
            issues.append("metadata missing schema_version")
    except json.JSONDecodeError:
        issues.append("body_scale_metadata_json is not valid JSON")

    return len(issues) == 0, issues


def load_motion_reference(path: Path) -> MotionReferenceData:
    """Load a validated motion reference NPZ."""
    path = Path(path)
    ok, issues = validate_motion_reference(path)
    if not ok:
        raise ValueError("; ".join(issues))

    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["body_scale_metadata_json"]))
    joint_ids = json.loads(str(data["joint_ids_json"]))

    root_orient = None
    if "root_orientations" in data and len(data["root_orientations"]):
        root_orient = np.asarray(data["root_orientations"], dtype=np.float64)

    gait_style = None
    if "gait_style_characteristics_json" in data:
        gait_style = str(data["gait_style_characteristics_json"])

    return MotionReferenceData(
        path=path,
        timestamps=np.asarray(data["timestamps"], dtype=np.float64),
        fps=float(data["fps"]),
        root_positions=np.asarray(data["root_positions"], dtype=np.float64),
        canonical_joint_positions=np.asarray(
            data["canonical_joint_positions"], dtype=np.float64
        ),
        left_contact_mask=np.asarray(data["left_contact_mask"], dtype=np.int8),
        right_contact_mask=np.asarray(data["right_contact_mask"], dtype=np.int8),
        joint_ids=joint_ids,
        metadata=meta,
        root_orientations=root_orient,
        gait_style_json=gait_style,
    )


__all__ = [
    "MOTION_REFERENCE_SCHEMA_VERSION",
    "MotionReferenceData",
    "load_motion_reference",
    "validate_motion_reference",
]
