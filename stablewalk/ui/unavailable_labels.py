"""Contextual unavailable labels for dashboard readouts.

Prefer a short, situation-specific phrase over a bare ``Unavailable``.
Avoid repeating the same placeholder across neighbouring widgets.
"""

from __future__ import annotations

from typing import Literal

# Shared phrases — keep these stable so tests and UX stay consistent.
WAITING_COMPARISON = "Waiting for comparison data"
METRIC_FOR_RECORDING = "Metric unavailable for this recording"
REQUIRES_GAIT_CYCLES = "Requires synchronized gait cycles"
REQUIRES_JOINT_ANGLES = "Requires estimated joint angles"
REQUIRES_ANALYZED_VIDEO = "Requires analyzed video"
MISSING_SESSION_A = "Missing in Session A"
MISSING_SESSION_B = "Missing in Session B"
OPENSIM_SDK_MISSING = "OpenSim SDK not installed"
MEASUREMENT_FOR_RECORDING = "Measurement unavailable for this recording"

CompareMissing = Literal["both", "a", "b", "none"]

# Metric → short “requires …” copy when both sides lack data.
_COMPARE_REQUIRES: dict[str, str] = {
    "Cadence": REQUIRES_GAIT_CYCLES,
    "Stride Length": REQUIRES_GAIT_CYCLES,
    "Step Length": REQUIRES_GAIT_CYCLES,
    "Symmetry": REQUIRES_GAIT_CYCLES,
    "Stability": "Requires valid COM–BoS frames",
    "Speed": "Requires reportable walking scale",
    "Joint ROM": REQUIRES_JOINT_ANGLES,
    "Knee ROM": REQUIRES_JOINT_ANGLES,
    "Gait Quality": "Requires scored gait domains",
    "COM Excursion": "Requires COM trajectory",
}


def _clean(text: str | None) -> str:
    return " ".join(str(text or "").split()).strip()


def compare_missing_side(
    *,
    left_available: bool,
    right_available: bool,
) -> CompareMissing:
    if left_available and right_available:
        return "none"
    if not left_available and not right_available:
        return "both"
    return "a" if not left_available else "b"


def _session_metric_unavailable(metric: str | None) -> str:
    key = _clean(metric)
    if key in _COMPARE_REQUIRES:
        return _COMPARE_REQUIRES[key]
    return METRIC_FOR_RECORDING


def compare_unavailable_label(
    metric: str | None = None,
    *,
    left_available: bool = False,
    right_available: bool = False,
    detail: str | None = None,
    left_session_ready: bool = False,
    right_session_ready: bool = False,
) -> str:
    """Short label for a Compare metric / Δ cell that cannot be shown."""
    side = compare_missing_side(
        left_available=left_available, right_available=right_available
    )
    if side == "a":
        if left_session_ready:
            return _session_metric_unavailable(metric)
        return MISSING_SESSION_A
    if side == "b":
        if right_session_ready:
            return _session_metric_unavailable(metric)
        return MISSING_SESSION_B
    if side == "both":
        key = _clean(metric)
        if key in _COMPARE_REQUIRES:
            return _COMPARE_REQUIRES[key]
        return WAITING_COMPARISON
    # Values exist but still not comparable (non-finite, etc.).
    detail_txt = _clean(detail)
    if detail_txt:
        return detail_txt
    return METRIC_FOR_RECORDING


def compare_reason_label(
    metric: str,
    *,
    left_available: bool,
    right_available: bool,
    reason_default: str,
    left_session_ready: bool = False,
    right_session_ready: bool = False,
) -> str:
    """Longer tooltip / table reason — contextual, not ``Unavailable in …``."""
    side = compare_missing_side(
        left_available=left_available, right_available=right_available
    )
    default = _clean(reason_default) or METRIC_FOR_RECORDING
    # Prefer the shared short requires-phrase when both sides are empty.
    if side == "both":
        short = _COMPARE_REQUIRES.get(_clean(metric))
        if short and short.lower() not in default.lower():
            return f"{short}. {default}" if default else short
        return default
    if side == "a":
        if left_session_ready:
            short = _session_metric_unavailable(metric)
            if short.lower() not in default.lower():
                return f"{short}. {default}" if default else short
            return default
        return f"{MISSING_SESSION_A} — {default}"
    if side == "b":
        if right_session_ready:
            short = _session_metric_unavailable(metric)
            if short.lower() not in default.lower():
                return f"{short}. {default}" if default else short
            return default
        return f"{MISSING_SESSION_B} — {default}"
    return default


def foot_clearance_unavailable_label(raw_reason: str | None = None) -> str:
    """Primary Overview foot-distance readout when clearance cannot be shown."""
    from stablewalk.ui.foot_clearance_display import overview_unavailable_reason

    mapped = overview_unavailable_reason(raw_reason or "")
    if not mapped or mapped.lower() in ("unavailable", "measurement unavailable"):
        return MEASUREMENT_FOR_RECORDING
    # overview_unavailable_reason already returns a short contextual phrase.
    return mapped


def opensim_unavailable_label(
    *,
    sdk: bool = False,
    has_session: bool = False,
) -> str:
    """Compact OpenSim status when the toolkit is not ready."""
    if not sdk and not has_session:
        return REQUIRES_ANALYZED_VIDEO
    if not sdk:
        return OPENSIM_SDK_MISSING
    return REQUIRES_ANALYZED_VIDEO


def session_slot_unavailable_label(prefix: str = "Session") -> str:
    """Compare session card header when a slot has no finished analysis."""
    return f"{prefix} — waiting for a finished analysis"


__all__ = [
    "MEASUREMENT_FOR_RECORDING",
    "METRIC_FOR_RECORDING",
    "MISSING_SESSION_A",
    "MISSING_SESSION_B",
    "OPENSIM_SDK_MISSING",
    "REQUIRES_ANALYZED_VIDEO",
    "REQUIRES_GAIT_CYCLES",
    "REQUIRES_JOINT_ANGLES",
    "WAITING_COMPARISON",
    "compare_missing_side",
    "compare_reason_label",
    "compare_unavailable_label",
    "foot_clearance_unavailable_label",
    "opensim_unavailable_label",
    "session_slot_unavailable_label",
]
