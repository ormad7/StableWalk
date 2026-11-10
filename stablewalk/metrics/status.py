"""Shared validity / display-status classifiers for StableWalk metrics.

Rules:
- Never invent clinical labels without an explicit threshold definition.
- Heuristic / pose-derived thresholds must be labeled as such.
- Unavailable / failed / insufficient data must not become numeric zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from stablewalk.metrics.definitions import BandThresholds, MetricDefinition, get_metric_definition


class ValidityState(str, Enum):
    VALID = "valid"
    LOW_CONFIDENCE = "low confidence"
    INSUFFICIENT_DATA = "insufficient data"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class DisplayStatus(str, Enum):
    """User-facing status. Prefer these exact strings in UI."""

    ACCEPTABLE = "Acceptable"
    BORDERLINE = "Borderline"
    REDUCED = "Reduced"
    UNSTABLE = "Unstable"  # engine frame/session stability only
    OUT_OF_RANGE = "Out of range"
    HEURISTIC = "Heuristic"
    UNVALIDATED_HEURISTIC = "Unvalidated heuristic"
    LOW_CONFIDENCE = "Low confidence"
    INSUFFICIENT_DATA = "Insufficient data"
    UNAVAILABLE = "Unavailable"
    FAILED = "Failed"
    NOT_APPLICABLE = "N/A"


# Legacy / divergent labels → canonical DisplayStatus (for migrations).
_LEGACY_STATUS_MAP: dict[str, DisplayStatus] = {
    "normal": DisplayStatus.ACCEPTABLE,
    "acceptable": DisplayStatus.ACCEPTABLE,
    "high": DisplayStatus.ACCEPTABLE,
    "moderate": DisplayStatus.BORDERLINE,
    "borderline": DisplayStatus.BORDERLINE,
    "reduced": DisplayStatus.REDUCED,
    "low": DisplayStatus.REDUCED,
    "abnormal": DisplayStatus.OUT_OF_RANGE,
    "unstable": DisplayStatus.UNSTABLE,
    "insufficient": DisplayStatus.INSUFFICIENT_DATA,
    "insufficient data": DisplayStatus.INSUFFICIENT_DATA,
    "unavailable": DisplayStatus.UNAVAILABLE,
    "n/a": DisplayStatus.NOT_APPLICABLE,
    "failed": DisplayStatus.FAILED,
    "low confidence": DisplayStatus.LOW_CONFIDENCE,
}


@dataclass(frozen=True, slots=True)
class MetricStatus:
    """Full display contract for one metric instance."""

    key: str
    display_name: str
    value: float | int | str | None
    unit: str
    source: str
    confidence: float | None
    completeness: float | None
    validity: ValidityState
    threshold_source: str
    display_status: DisplayStatus
    user_explanation: str
    technical_explanation: str
    threshold_note: str = ""
    # True when value is a genuine computed number (including legitimate 0).
    value_is_numeric: bool = False
    # Provenance tags for UI chips.
    provenance_tag: str = ""

    @property
    def display_value(self) -> str:
        """Formatted value for cards; never coerces missing → '0'."""
        if self.validity in (ValidityState.UNAVAILABLE, ValidityState.FAILED):
            return "—"
        # Insufficient data may still show a genuine computed number (e.g. 0 cycles).
        if (
            self.validity == ValidityState.INSUFFICIENT_DATA
            and not self.value_is_numeric
            and self.value is None
        ):
            return "—"
        if self.value is None:
            return "—"
        if isinstance(self.value, str):
            return self.value
        if isinstance(self.value, float):
            if abs(self.value) >= 100:
                return f"{self.value:.0f}"
            if abs(self.value) >= 10:
                return f"{self.value:.1f}"
            return f"{self.value:.2f}"
        return str(self.value)

    @property
    def status_label(self) -> str:
        return self.display_status.value

    def to_export_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "value": self.value,
            "display_value": self.display_value,
            "unit": self.unit,
            "source": self.source,
            "confidence": self.confidence,
            "completeness": self.completeness,
            "validity": self.validity.value,
            "threshold_source": self.threshold_source,
            "display_status": self.display_status.value,
            "user_explanation": self.user_explanation,
            "technical_explanation": self.technical_explanation,
            "threshold_note": self.threshold_note,
            "value_is_numeric": self.value_is_numeric,
            "provenance_tag": self.provenance_tag,
        }


def canonicalize_status_label(raw: str | None) -> DisplayStatus | None:
    if raw is None:
        return None
    return _LEGACY_STATUS_MAP.get(str(raw).strip().lower())


def classify_band(
    value: float,
    bands: BandThresholds,
) -> DisplayStatus:
    """Map a numeric value onto Acceptable / Borderline / Reduced|Out of range."""
    if bands.normal_max is not None or bands.borderline_max is not None:
        # Range metric (e.g. cadence).
        n_lo = bands.normal_min
        n_hi = bands.normal_max
        b_lo = bands.borderline_min if bands.borderline_min is not None else n_lo
        b_hi = bands.borderline_max if bands.borderline_max is not None else n_hi
        if n_lo is not None and n_hi is not None and n_lo <= value <= n_hi:
            return DisplayStatus.ACCEPTABLE
        if b_lo is not None and b_hi is not None and b_lo <= value <= b_hi:
            return DisplayStatus.BORDERLINE
        return DisplayStatus.OUT_OF_RANGE

    if bands.higher_is_better:
        if bands.normal_min is not None and value >= bands.normal_min:
            return DisplayStatus.ACCEPTABLE
        if bands.borderline_min is not None and value >= bands.borderline_min:
            return DisplayStatus.BORDERLINE
        return DisplayStatus.REDUCED
    # Lower is better
    if bands.normal_min is not None and value <= bands.normal_min:
        return DisplayStatus.ACCEPTABLE
    if bands.borderline_min is not None and value <= bands.borderline_min:
        return DisplayStatus.BORDERLINE
    return DisplayStatus.OUT_OF_RANGE


def validity_from_flags(
    *,
    available: bool | None = None,
    failed: bool = False,
    insufficient: bool = False,
    low_confidence: bool = False,
    confidence: float | None = None,
    confidence_floor: float = 0.35,
) -> ValidityState:
    if failed:
        return ValidityState.FAILED
    if available is False:
        return ValidityState.UNAVAILABLE
    if insufficient:
        return ValidityState.INSUFFICIENT_DATA
    if low_confidence:
        return ValidityState.LOW_CONFIDENCE
    if confidence is not None and 0.0 <= confidence <= 1.0 and confidence < confidence_floor:
        return ValidityState.LOW_CONFIDENCE
    if confidence is not None and confidence > 1.0 and confidence < confidence_floor * 100.0:
        return ValidityState.LOW_CONFIDENCE
    return ValidityState.VALID


def status_for_validity(validity: ValidityState) -> DisplayStatus:
    return {
        ValidityState.VALID: DisplayStatus.ACCEPTABLE,  # overridden when bands applied
        ValidityState.LOW_CONFIDENCE: DisplayStatus.LOW_CONFIDENCE,
        ValidityState.INSUFFICIENT_DATA: DisplayStatus.INSUFFICIENT_DATA,
        ValidityState.UNAVAILABLE: DisplayStatus.UNAVAILABLE,
        ValidityState.FAILED: DisplayStatus.FAILED,
    }[validity]


def build_metric_status(
    key: str,
    *,
    value: float | int | str | None,
    available: bool | None = None,
    failed: bool = False,
    insufficient: bool = False,
    low_confidence: bool = False,
    confidence: float | None = None,
    completeness: float | None = None,
    definition: MetricDefinition | None = None,
    force_status: DisplayStatus | None = None,
    extra_user_note: str = "",
    categorical_value: str | None = None,
) -> MetricStatus:
    """Build a MetricStatus from a definition + raw analysis value."""
    definition = definition or get_metric_definition(key)
    if definition is None:
        # Fallback for unregistered keys — still avoid zero-coercion.
        validity = validity_from_flags(
            available=available if available is not None else value is not None,
            failed=failed,
            insufficient=insufficient,
            low_confidence=low_confidence,
            confidence=confidence,
        )
        disp = force_status or (
            status_for_validity(validity)
            if validity != ValidityState.VALID
            else DisplayStatus.HEURISTIC
        )
        return MetricStatus(
            key=key,
            display_name=key.replace("_", " ").title(),
            value=None if validity != ValidityState.VALID and validity != ValidityState.LOW_CONFIDENCE else value,
            unit="",
            source="derived",
            confidence=confidence,
            completeness=completeness,
            validity=validity,
            threshold_source="none",
            display_status=disp,
            user_explanation=extra_user_note or "No shared definition registered for this metric.",
            technical_explanation="",
            value_is_numeric=isinstance(value, (int, float))
            and validity in (ValidityState.VALID, ValidityState.LOW_CONFIDENCE),
        )

    # Infer availability if not provided.
    if available is None:
        if value is None and categorical_value is None:
            available = False
        else:
            available = True

    validity = validity_from_flags(
        available=available,
        failed=failed,
        insufficient=insufficient,
        low_confidence=low_confidence,
        confidence=confidence,
    )

    # Suppress numeric display when unavailable/failed. Insufficient data may
    # still carry a genuine computed number (including legitimate zero).
    show_value: float | int | str | None
    if validity in (ValidityState.UNAVAILABLE, ValidityState.FAILED):
        show_value = None
        value_is_numeric = False
    elif categorical_value is not None:
        show_value = categorical_value
        value_is_numeric = False
    elif validity == ValidityState.INSUFFICIENT_DATA and isinstance(value, (int, float)):
        show_value = value
        value_is_numeric = True
    elif validity == ValidityState.INSUFFICIENT_DATA:
        show_value = None
        value_is_numeric = False
    else:
        show_value = value
        value_is_numeric = isinstance(value, (int, float))

    if force_status is not None:
        disp = force_status
    elif validity != ValidityState.VALID and validity != ValidityState.LOW_CONFIDENCE:
        disp = status_for_validity(validity)
    elif validity == ValidityState.LOW_CONFIDENCE:
        disp = DisplayStatus.LOW_CONFIDENCE
    elif categorical_value:
        mapped = canonicalize_status_label(categorical_value)
        disp = mapped or DisplayStatus.HEURISTIC
    elif isinstance(value, (int, float)) and definition.bands is not None:
        disp = classify_band(float(value), definition.bands)
        # Heuristic bands have no documented clinical basis — do not say Acceptable.
        if definition.threshold_source == "heuristic" and disp == DisplayStatus.ACCEPTABLE:
            disp = DisplayStatus.HEURISTIC
        # Stability % uses Reduced for low share; engine Unstable is reserved
        # for frame/margin categorical states.
        if definition.key == "stable_pct" and disp == DisplayStatus.REDUCED:
            disp = DisplayStatus.REDUCED
    elif definition.threshold_source in ("heuristic", "literature"):
        disp = DisplayStatus.HEURISTIC
    else:
        disp = DisplayStatus.ACCEPTABLE

    # Label heuristic thresholds explicitly in status when Acceptable/Borderline/Reduced.
    if (
        definition.threshold_source == "heuristic"
        and disp
        in (DisplayStatus.ACCEPTABLE, DisplayStatus.BORDERLINE, DisplayStatus.REDUCED, DisplayStatus.OUT_OF_RANGE)
        and force_status is None
        and not categorical_value
    ):
        # Keep band label; UI shows "Heuristic" provenance separately.
        pass

    provenance = {
        "measured": "Measured",
        "estimated": "Estimated",
        "derived": "Derived",
        "opensim": "OpenSim",
    }.get(definition.source, definition.source.title())

    user_ex = definition.user_explanation
    if extra_user_note:
        user_ex = f"{user_ex} {extra_user_note}".strip()
    if definition.threshold_source == "heuristic":
        user_ex = f"{user_ex} (Pose-derived heuristic — not a clinical cut-off.)"

    return MetricStatus(
        key=definition.key,
        display_name=definition.display_name,
        value=show_value,
        unit=definition.unit,
        source=definition.source,
        confidence=confidence,
        completeness=completeness,
        validity=validity,
        threshold_source=definition.threshold_source,
        display_status=disp,
        user_explanation=user_ex,
        technical_explanation=definition.technical_explanation,
        threshold_note=definition.threshold_note,
        value_is_numeric=value_is_numeric,
        provenance_tag=provenance,
    )


def engine_stability_state_status(
    state: str | None,
    *,
    available: bool = True,
    confidence: float | None = None,
) -> MetricStatus:
    """Map engine Stable/Reduced/Unstable categorical to MetricStatus."""
    if not available or not state:
        return build_metric_status(
            "stability_frame_state",
            value=None,
            available=False,
            confidence=confidence,
        )
    raw = str(state).strip()
    key = raw.lower()
    mapped = {
        "stable": DisplayStatus.HEURISTIC,
        "reduced": DisplayStatus.REDUCED,
        "unstable": DisplayStatus.UNSTABLE,
        "unknown": DisplayStatus.UNAVAILABLE,
    }.get(key, DisplayStatus.HEURISTIC)
    # Keep engine vocabulary on the value. Do not relabel Stable as Acceptable.
    label = {
        "stable": "Stable",
        "reduced": "Reduced",
        "unstable": "Unstable",
        "unknown": "Unavailable",
    }.get(key, raw)
    st = build_metric_status(
        "stability_frame_state",
        value=label,
        available=True,
        confidence=confidence,
        categorical_value=label,
        force_status=mapped,
    )
    return MetricStatus(
        key=st.key,
        display_name=st.display_name,
        value=label,
        unit="",
        source=st.source,
        confidence=confidence,
        completeness=st.completeness,
        validity=ValidityState.VALID,
        threshold_source=st.threshold_source,
        display_status=mapped,
        user_explanation=st.user_explanation,
        technical_explanation=st.technical_explanation,
        threshold_note=st.threshold_note,
        value_is_numeric=False,
        provenance_tag=st.provenance_tag,
    )


def display_status_label(status: DisplayStatus | str) -> str:
    if isinstance(status, DisplayStatus):
        return status.value
    mapped = canonicalize_status_label(status)
    return mapped.value if mapped is not None else str(status)


def format_metric_value(
    value: float | int | str | None,
    *,
    unit: str = "",
    validity: ValidityState | None = None,
    value_is_numeric: bool = False,
) -> str:
    """Format for UI/export; never turns missing into '0'."""
    if validity in (ValidityState.UNAVAILABLE, ValidityState.FAILED):
        return "—"
    if (
        validity == ValidityState.INSUFFICIENT_DATA
        and not value_is_numeric
        and value is None
    ):
        return "—"
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    text = (
        f"{value:.0f}"
        if isinstance(value, float) and abs(value) >= 100
        else f"{value:.1f}"
        if isinstance(value, float) and abs(value) >= 10
        else f"{value:.2f}"
        if isinstance(value, float)
        else str(value)
    )
    return f"{text}{unit}" if unit and not str(unit).startswith("/") else (
        f"{text}{unit}" if unit else text
    )


def is_genuine_zero(status: MetricStatus) -> bool:
    """True only when displayed zero is a real computed result."""
    return (
        status.value_is_numeric
        and status.value == 0
        and status.validity
        in (ValidityState.VALID, ValidityState.LOW_CONFIDENCE, ValidityState.INSUFFICIENT_DATA)
    )


def never_zero_from_missing(status: MetricStatus) -> bool:
    """Guard: missing/failed metrics must not present as numeric zero."""
    if status.validity in (
        ValidityState.UNAVAILABLE,
        ValidityState.FAILED,
    ):
        return status.value is None and status.display_value == "—"
    if status.validity == ValidityState.INSUFFICIENT_DATA and not status.value_is_numeric:
        return status.display_value == "—"
    return True


def metric_visual_level(status: DisplayStatus | MetricStatus) -> str:
    """Map to legacy color keys: normal / borderline / abnormal / unavailable / neutral."""
    disp = status.display_status if isinstance(status, MetricStatus) else status
    return {
        DisplayStatus.ACCEPTABLE: "normal",
        DisplayStatus.BORDERLINE: "borderline",
        DisplayStatus.REDUCED: "abnormal",
        DisplayStatus.OUT_OF_RANGE: "abnormal",
        DisplayStatus.UNSTABLE: "abnormal",
        DisplayStatus.LOW_CONFIDENCE: "abnormal",
        DisplayStatus.INSUFFICIENT_DATA: "unavailable",
        DisplayStatus.UNAVAILABLE: "unavailable",
        DisplayStatus.FAILED: "unavailable",
        DisplayStatus.HEURISTIC: "neutral",
        DisplayStatus.NOT_APPLICABLE: "unavailable",
    }.get(disp, "neutral")


__all__ = [
    "DisplayStatus",
    "MetricStatus",
    "ValidityState",
    "build_metric_status",
    "canonicalize_status_label",
    "classify_band",
    "display_status_label",
    "engine_stability_state_status",
    "format_metric_value",
    "is_genuine_zero",
    "metric_visual_level",
    "never_zero_from_missing",
    "status_for_validity",
    "validity_from_flags",
]
