"""Canonical knee-angle names used by Kinematics, Asymmetry, Spectral, and exports.

Two geometric quantities exist at the knee. They must not share a label.

``knee_interior_deg``
    Internal 3D Hip–Knee–Ankle angle θ ∈ (0, 180]. Straight limb ≈ 180°.
    This is what pose ``JointAngles.left_knee`` / stored pose DOFs contain.

``knee_flexion_deg`` (canonical display / analysis angle)
    Anatomical flexion = 180° − θ. 0° = full extension; larger = more flexion.
    Kinematics already plots this. Asymmetry and Spectral must reuse it.

Processed spectral copies are not angles of the same kind:

``knee_flexion_detrended_deg``
    Linear (or mean) detrend of ``knee_flexion_deg``. Units still deg, but
    signed about zero. Not a posture angle.

``knee_flexion_zscore``
    Unit-variance copy of the detrended series. Dimensionless. Used for ACF/FFT.

Do not convert OpenSim / mock / synthetic series that already store flexion.
Those recordings set ``metadata['knee_angle_storage'] = anatomical_flexion``.
Pose recordings set ``pose_interior_hka``.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from stablewalk.pose.kinematics import (
    KNEE_FLEXION_CONVENTION,
    KNEE_FLEXION_UNITS,
    knee_flexion_from_snapshot,
    knee_hka_joint_ids,
)

KneeStorage = Literal["pose_interior_hka", "anatomical_flexion"]

KNEE_STORAGE_METADATA_KEY = "knee_angle_storage"
KNEE_STORAGE_INTERIOR = "pose_interior_hka"
KNEE_STORAGE_FLEXION = "anatomical_flexion"

KNEE_FLEXION_DOF_IDS = frozenset({"left_knee_flexion", "right_knee_flexion"})
_KNEE_DOF_TO_ITEM = {
    "left_knee_flexion": "left_knee",
    "right_knee_flexion": "right_knee",
}

KNEE_FLEXION_AXIS = "Anatomical knee flexion (deg); 0° = extension"
KNEE_INTERIOR_AXIS = "Pose interior HKA angle (deg); 180° ≈ extension"
KNEE_DETRENDED_AXIS = "Detrended anatomical knee flexion (deg); signed about 0"
KNEE_ZSCORE_AXIS = "Z-scored detrended knee flexion (dimensionless)"


def is_knee_flexion_dof(dof_id: str | None) -> bool:
    return str(dof_id or "") in KNEE_FLEXION_DOF_IDS


def recording_knee_storage(recording: Any) -> KneeStorage:
    """Return how ``left/right_knee_flexion`` DOFs were stored on this recording."""
    meta = getattr(recording, "metadata", None) or {}
    tagged = str(meta.get(KNEE_STORAGE_METADATA_KEY) or "").strip()
    if tagged == KNEE_STORAGE_INTERIOR:
        return KNEE_STORAGE_INTERIOR
    if tagged == KNEE_STORAGE_FLEXION:
        return KNEE_STORAGE_FLEXION
    kind = str(getattr(recording, "source_kind", "") or "").strip().lower()
    if kind == "pose_estimation":
        return KNEE_STORAGE_INTERIOR
    if kind in {"mock", "opensim"}:
        return KNEE_STORAGE_FLEXION
    return KNEE_STORAGE_FLEXION


def _stored_dof_deg(snapshot: Any, dof_id: str) -> float | None:
    getter = getattr(snapshot, "get_dof", None)
    dof = getter(dof_id) if callable(getter) else (getattr(snapshot, "dofs", None) or {}).get(dof_id)
    if dof is None:
        return None
    value = getattr(dof, "angle_deg", None)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def analysis_knee_flexion_deg(
    snapshot: Any,
    dof_id: str,
    *,
    storage: KneeStorage,
) -> float | None:
    """Anatomical knee flexion for analysis views. Never invents landmarks.

    * ``pose_interior_hka``: prefer 3D HKA flexion (same as Kinematics). If
      Hip–Knee–Ankle is missing, convert the stored interior DOF with 180° − θ.
    * ``anatomical_flexion``: use the stored DOF as-is (mock / OpenSim / sine).
      HKA is not substituted, because synthetic joint XYZ may not match the DOF.
    """
    if not is_knee_flexion_dof(dof_id):
        return _stored_dof_deg(snapshot, dof_id)
    stored = _stored_dof_deg(snapshot, dof_id)
    if storage == KNEE_STORAGE_FLEXION:
        return stored
    item_id = _KNEE_DOF_TO_ITEM.get(dof_id)
    hka = knee_flexion_from_snapshot(snapshot, item_id) if item_id else None
    if hka is not None:
        return float(hka)
    if stored is None:
        return None
    return float(180.0 - stored)


def analysis_knee_interior_deg(
    snapshot: Any,
    dof_id: str,
    *,
    storage: KneeStorage,
) -> float | None:
    """Raw geometric interior HKA (deg), or None if it cannot be recovered."""
    if not is_knee_flexion_dof(dof_id):
        return None
    item_id = _KNEE_DOF_TO_ITEM.get(dof_id)
    if item_id and knee_hka_joint_ids(item_id) is not None:
        flex = knee_flexion_from_snapshot(snapshot, item_id)
        if flex is not None:
            return float(180.0 - flex)
    stored = _stored_dof_deg(snapshot, dof_id)
    if stored is None:
        return None
    if storage == KNEE_STORAGE_INTERIOR:
        return float(stored)
    return float(180.0 - stored)


def knee_channel_axis_labels(*, processed: bool = False) -> tuple[str, str]:
    """(title, ylabel) for a knee time-series panel."""
    if processed:
        return "Detrended knee flexion", KNEE_DETRENDED_AXIS
    return "Anatomical knee flexion", KNEE_FLEXION_AXIS


__all__ = [
    "KNEE_DETRENDED_AXIS",
    "KNEE_FLEXION_AXIS",
    "KNEE_FLEXION_CONVENTION",
    "KNEE_FLEXION_DOF_IDS",
    "KNEE_FLEXION_UNITS",
    "KNEE_INTERIOR_AXIS",
    "KNEE_STORAGE_FLEXION",
    "KNEE_STORAGE_INTERIOR",
    "KNEE_STORAGE_METADATA_KEY",
    "KNEE_ZSCORE_AXIS",
    "KneeStorage",
    "analysis_knee_flexion_deg",
    "analysis_knee_interior_deg",
    "is_knee_flexion_dof",
    "knee_channel_axis_labels",
    "recording_knee_storage",
]
