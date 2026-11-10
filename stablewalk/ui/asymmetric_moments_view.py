"""Display helpers for gait asymmetry moments on the existing 3D skeleton.

Does not create a second renderer. Joint IDs are mapped onto the active
skeleton display mode (full vs 16-joint gait wire).
"""

from __future__ import annotations

from typing import Any, Iterable

from stablewalk.analysis.gait_asymmetry_moments import (
    AsymmetryMoment,
    difference_joint_pairs,
    format_contribution_line,
)
from stablewalk.models.joint_registry import ROOT_JOINT_ID
from stablewalk.ui.theme import DANGER

# Affected joints on the 3D skeleton / video overlay.
ASYMMETRY_HIGHLIGHT_COLOR = DANGER  # "#d95555"
# OpenCV BGR of DANGER (#d95555 → R217 G85 B85).
ASYMMETRY_HIGHLIGHT_BGR: tuple[int, int, int] = (85, 85, 217)

_PELVIS_ALIASES = frozenset({ROOT_JOINT_ID, "pelvis", "pelvis_center", "mid_hip"})


def display_joint_id(joint_id: str, display_mode: str) -> str:
    """Map analysis joint ids onto the joints the current skeleton view draws."""
    if joint_id in _PELVIS_ALIASES:
        return "pelvis_center" if display_mode == "gait_16" else ROOT_JOINT_ID
    return joint_id


def overlay_joint_ids(joint_id: str, display_mode: str) -> set[str]:
    """Names that video-overlay landmarks may use for the same joint."""
    mapped = display_joint_id(joint_id, display_mode)
    names = {joint_id, mapped}
    if joint_id in _PELVIS_ALIASES or mapped in _PELVIS_ALIASES:
        names.update(_PELVIS_ALIASES)
    return names


_LIMB_CHAIN: dict[str, tuple[str, ...]] = {
    "hip": ("hip", "knee", "ankle", "heel", "toe"),
    "knee": ("hip", "knee", "ankle", "heel", "toe"),
    "ankle": ("hip", "knee", "ankle", "heel", "toe"),
    "heel": ("hip", "knee", "ankle", "heel", "toe"),
    "toe": ("hip", "knee", "ankle", "heel", "toe"),
    "foot": ("hip", "knee", "ankle", "heel", "toe"),
}


def _limb_chain_ids(joint_id: str) -> set[str]:
    """Expand a joint to the full ipsilateral wire limb (not a single marker)."""
    text = str(joint_id)
    side = "left_" if text.startswith("left_") else "right_" if text.startswith("right_") else ""
    token = text.split("_", 1)[-1] if side else text
    chain = _LIMB_CHAIN.get(token)
    if not chain or not side:
        return {text}
    return {f"{side}{part}" for part in chain}


def moment_highlight_joints(
    moment: AsymmetryMoment | None,
    display_mode: str,
) -> set[str]:
    if moment is None:
        return set()
    joints: set[str] = set()
    for jid in moment.involved_joints:
        for expanded in _limb_chain_ids(jid) | {jid}:
            joints.add(display_joint_id(expanded, display_mode))
            joints.update(overlay_joint_ids(expanded, display_mode))
    return {j for j in joints if j}


def moment_difference_pairs(
    moment: AsymmetryMoment | None,
    display_mode: str,
) -> list[tuple[str, str]]:
    if moment is None:
        return []
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for now_id, match_id in difference_joint_pairs(moment):
        key = (
            display_joint_id(now_id, display_mode),
            display_joint_id(match_id, display_mode),
        )
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def moment_measurement_labels(
    moment: AsymmetryMoment | None,
    display_mode: str,
    *,
    limit: int = 3,
) -> dict[str, str]:
    """Short Δ labels at the current-frame (right / pelvis) joints."""
    if moment is None:
        return {}
    labels: dict[str, str] = {}
    for item in moment.contributing_measurements[: max(1, int(limit))]:
        jid = next(
            (j for j in item.joints if not str(j).startswith("left_")),
            item.joints[0] if item.joints else None,
        )
        if not jid:
            continue
        mapped = display_joint_id(str(jid), display_mode)
        if mapped in labels:
            continue
        line = format_contribution_line(item)
        # Compact canvas callout: keep the Δ token.
        if "  Δ " in line:
            labels[mapped] = line.split("  Δ ", 1)[-1].strip()
            labels[mapped] = f"Δ {labels[mapped]}" if not labels[mapped].startswith("Δ") else labels[mapped]
        else:
            labels[mapped] = line
    return labels


def moment_measurement_lines(moment: AsymmetryMoment | None, *, limit: int = 6) -> list[str]:
    if moment is None:
        return []
    lines: list[str] = []
    for item in moment.contributing_measurements[: max(1, int(limit))]:
        lines.append(format_contribution_line(item))
    return lines


def format_joint_list(joint_ids: Iterable[str]) -> str:
    pretty: list[str] = []
    seen: set[str] = set()
    for jid in joint_ids:
        label = str(jid).replace("_", " ")
        if label in seen:
            continue
        seen.add(label)
        pretty.append(label)
    return ", ".join(pretty) if pretty else "—"


def _primary_values_label(moment: AsymmetryMoment) -> str:
    ranked = sorted(
        moment.contributing_measurements,
        key=lambda item: float(item.score),
        reverse=True,
    )
    for item in ranked:
        if item.left_value is None or item.right_value is None:
            continue
        if item.units == "deg":
            return f"L {item.left_value:.1f}°   R {item.right_value:.1f}°"
        return f"L {item.left_value:.3g}   R {item.right_value:.3g}"
    lines = moment_measurement_lines(moment, limit=1)
    return lines[0] if lines else "L / R —"


def moment_card_payload(moment: AsymmetryMoment, index: int) -> dict[str, Any]:
    """Plain data used by the Overview cards (no Tk)."""
    frame = int(moment.frame_index)
    conf = float(moment.confidence)
    return {
        "index": int(index),
        "frame_index": frame,
        "time_s": float(moment.time_s),
        "score": float(moment.score),
        "confidence": conf,
        "severity": str(moment.severity),
        "involved_joints": list(moment.involved_joints),
        "joints_label": format_joint_list(moment.involved_joints),
        "explanation": str(moment.explanation),
        "measurements": moment_measurement_lines(moment),
        "values_label": _primary_values_label(moment),
        "timestamp_label": f"{moment.time_s:.2f} s  ·  frame {frame}",
        "score_label": f"{moment.score:.0f}",
        "confidence_label": f"{conf:.2f}",
    }


__all__ = [
    "ASYMMETRY_HIGHLIGHT_BGR",
    "ASYMMETRY_HIGHLIGHT_COLOR",
    "display_joint_id",
    "format_joint_list",
    "moment_card_payload",
    "moment_difference_pairs",
    "moment_highlight_joints",
    "moment_measurement_labels",
    "moment_measurement_lines",
    "overlay_joint_ids",
]
