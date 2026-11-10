"""
Stability margin — projected COM relative to Base of Support.

Margin and state are **derived estimates** from pose-based COM and foot polygons.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from stablewalk.analysis.biomechanical.base_of_support import (
    BaseOfSupportAnalysis,
    _align_feet_to_support_plane,
    _convex_hull_2d,
)
from stablewalk.analysis.biomechanical.com_estimation import CenterOfMassAnalysis
from stablewalk.analysis.biomechanical.types import BiomechanicalFrameStability, StabilityState

logger = logging.getLogger(__name__)

# Internal diagnostic bins only. 0.04 m is NOT a documented Hof MoS, Winter,
# or clinical-normative threshold for this monocular 2-D COM–BoS proxy.
# UI must not present Stable / Reduced / Unstable or a "Normal" band from these.
_STABLE_MARGIN_M = 0.04
_REDUCED_MARGIN_M = 0.0

BALANCE_UNIT = "m"
BALANCE_COORDINATE_FRAME = (
    "pelvis-centred body-normalized XZ; BoS mean Z aligned to COM Z; metres"
)
BALANCE_METHOD = (
    "experimental monocular 2-D COM vs estimated foot-polygon BoS "
    "(signed distance). Not Hof margin of stability; not force-plate MoS"
)
BALANCE_CLASSIFICATION_REASON = (
    "Normal/Stable/Unstable bands and the 0.04 m threshold have no documented "
    "Hof, Winter, or clinical-normative basis for this monocular 2-D COM–BoS "
    "proxy, so they are not used as classifications."
)
VALIDITY_VALID = "valid"
VALIDITY_UNVALIDATED = "unvalidated"
VALIDITY_UNAVAILABLE = "unavailable"
DISPLAY_KPI = "kpi"
DISPLAY_PREVIEW = "unvalidated_preview"
DISPLAY_HIDDEN = "hidden"
_MIN_VALID_FRAME_RATIO = 0.50
_MIN_FRAME_CONFIDENCE = 0.35
_LOW_CONFIDENCE = 0.50
# Show diagnostics when Stable% is below session borderline even if confidence is OK.
_DIAG_STABLE_PCT_MIN = 55.0
# DEPTH_LAYERS heel vs toe differ by ~0.01; that is not measured foot length.
_MIN_MEASURED_AP_SPAN = 0.04
# Physical half-width for zero-area (heel–toe line) support.
# Matches prior body-scale 0.03 · ~1.7 m stature ≈ 5 cm (must be ≥ Stable threshold).
_FOOT_HALF_WIDTH_M = 0.05

def _depth_aligned_polygon(
    com_x: float,
    com_z: float,
    polygon: list[tuple[float, float]],
) -> tuple[tuple[float, float], list[tuple[float, float]], float]:
    """
    Align BoS depth to the COM so visualization DEPTH_LAYERS are not a lean.

    Canonical Z includes a joint-specific depth heuristic (feet behind the
    pelvis). That offset is not a measured anterior COM lean. After alignment
    the polygon's mean Z equals COM Z; left/right and heel/toe relative Z
    (and all X) stay as measured.
    """
    if not polygon:
        return (com_x, com_z), polygon, 0.0
    mean_z = sum(p[1] for p in polygon) / float(len(polygon))
    bias = mean_z - com_z
    aligned = [(p[0], p[1] - bias) for p in polygon]
    return (com_x, com_z), aligned, bias


def _collapse_unmeasured_ap_span(
    com_z: float,
    polygon: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], bool]:
    """Drop DEPTH_LAYERS heel–toe Z when it is not a measured foot length.

    After mean-Z alignment the remaining AP span is often ~1 cm from the
    visualization layers. Using that as BoS depth forces Reduced/Unstable
    even when COM is over the feet in X. Collapse to the mediolateral
    interval and reuse the two-point thickened-foot model.
    """
    if len(polygon) < 2:
        return polygon, False
    z_span = max(p[1] for p in polygon) - min(p[1] for p in polygon)
    if z_span >= _MIN_MEASURED_AP_SPAN:
        return polygon, False
    xs = [p[0] for p in polygon]
    return [(min(xs), com_z), (max(xs), com_z)], True


def _point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > py) != (yj > py):
            denominator = yj - yi
            if abs(denominator) > 1e-12:
                crossing_x = (xj - xi) * (py - yi) / denominator + xi
                if px < crossing_x:
                    inside = not inside
        j = i
    return inside


def _distance_to_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5


def _signed_margin_to_polygon(
    px: float,
    py: float,
    polygon: list[tuple[float, float]],
    *,
    meters_per_unit: float,
) -> float | None:
    """
    Signed distance to polygon edge in metres.

    Positive when COM projection is inside; negative when outside.
    Two-point support (single-limb line) uses a thickened foot model so the
    margin is not forced negative solely because a line has zero area.
    Half-width is a fixed physical length so body-span variation cannot push
    the maximum two-point margin below the Stable threshold.
    """
    if len(polygon) < 2:
        return None
    if len(polygon) == 2:
        d_norm = _distance_to_segment(
            px, py, polygon[0][0], polygon[0][1], polygon[1][0], polygon[1][1]
        )
        return _FOOT_HALF_WIDTH_M - d_norm * meters_per_unit

    min_dist = float("inf")
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        d = _distance_to_segment(px, py, ax, ay, bx, by)
        min_dist = min(min_dist, d)

    inside = _point_in_polygon(px, py, polygon)
    signed = min_dist if inside else -min_dist
    return signed * meters_per_unit


def _classify_margin(margin_m: float | None) -> str:
    if margin_m is None:
        return "Unavailable"
    if margin_m >= _STABLE_MARGIN_M:
        return StabilityState.STABLE.value
    if margin_m >= _REDUCED_MARGIN_M:
        return StabilityState.REDUCED.value
    return StabilityState.UNSTABLE.value


@dataclass(frozen=True)
class BalanceDisplayContract:
    """Shared plot/card contract for COM–BoS signed distance.

    One object: values, units, coordinate frame, method, validity, reason.
    Invalid results are either hidden or shown as an unvalidated preview.
    """

    values_m: tuple[float | None, ...]
    times_s: tuple[float, ...]
    unit: str
    coordinate_frame: str
    method: str
    validity: str
    reason: str
    display_mode: str
    reportable: bool
    classification_supported: bool
    classification_reason: str
    mean_margin_m: float | None = None
    min_margin_m: float | None = None
    max_margin_m: float | None = None
    kind: str = ""

    @property
    def has_series(self) -> bool:
        return bool(self.times_s)


@dataclass
class StabilityMarginAnalysis:
    per_frame: list[BiomechanicalFrameStability] = field(default_factory=list)
    mean_margin_m: float | None = None
    min_margin_m: float | None = None
    max_margin_m: float | None = None
    stable_pct: float | None = None
    valid_frame_ratio: float = 0.0
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, float | int | str | None] = field(default_factory=dict)
    kind: str = "derived"
    reportable: bool = True
    experimental_note: str = ""
    unit: str = BALANCE_UNIT
    coordinate_frame: str = BALANCE_COORDINATE_FRAME
    method: str = BALANCE_METHOD
    validity: str = VALIDITY_UNAVAILABLE
    reason: str = ""
    display_mode: str = DISPLAY_HIDDEN
    classification_supported: bool = False
    classification_reason: str = BALANCE_CLASSIFICATION_REASON

    def display_contract(self) -> BalanceDisplayContract:
        """Single display object for plots, summary cards, and exports."""
        return _contract_from_analysis(self)

    def to_dict(self) -> dict:
        contract = self.display_contract()
        return {
            "kind": self.kind,
            "reportable": self.reportable,
            "experimental_note": self.experimental_note,
            "unit": contract.unit,
            "coordinate_frame": contract.coordinate_frame,
            "method": contract.method,
            "validity": contract.validity,
            "reason": contract.reason,
            "display_mode": contract.display_mode,
            "classification_supported": contract.classification_supported,
            "classification_reason": contract.classification_reason,
            "frame_count": len(self.per_frame),
            "mean_margin_m": self.mean_margin_m,
            "min_margin_m": self.min_margin_m,
            "max_margin_m": self.max_margin_m,
            "stable_pct": None if self.stable_pct is None else round(self.stable_pct, 2),
            "valid_frame_ratio": round(self.valid_frame_ratio, 4),
            "confidence": round(self.confidence, 4),
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
            "note": "COM–BoS signed distance from estimated pose kinematics.",
        }


def _unavailable_contract(reason: str) -> BalanceDisplayContract:
    return BalanceDisplayContract(
        values_m=(),
        times_s=(),
        unit=BALANCE_UNIT,
        coordinate_frame=BALANCE_COORDINATE_FRAME,
        method=BALANCE_METHOD,
        validity=VALIDITY_UNAVAILABLE,
        reason=reason,
        display_mode=DISPLAY_HIDDEN,
        reportable=False,
        classification_supported=False,
        classification_reason=BALANCE_CLASSIFICATION_REASON,
    )


def _compose_reason(
    *,
    experimental_note: str,
    extra: str,
    classification_supported: bool,
    classification_reason: str,
    fallback: str,
) -> str:
    stored = str(extra or "").strip()
    parts: list[str] = []
    if stored:
        parts.append(stored)
    else:
        for part in (experimental_note, fallback):
            text = str(part or "").strip()
            if text and text not in parts:
                parts.append(text)
    blob = " ".join(parts)
    if not classification_supported:
        cls = str(classification_reason or BALANCE_CLASSIFICATION_REASON).strip()
        if cls and cls not in blob:
            parts.append(cls)
    return " ".join(parts).strip()


def _contract_from_attributes(analysis: object) -> BalanceDisplayContract:
    frames = list(getattr(analysis, "per_frame", None) or ())
    values: tuple[float | None, ...] = tuple(
        None
        if getattr(frame, "stability_margin_m", None) is None
        else float(frame.stability_margin_m)
        for frame in frames
    )
    times = tuple(float(getattr(frame, "time_s", 0.0) or 0.0) for frame in frames)
    reportable = bool(getattr(analysis, "reportable", True))
    kind = str(getattr(analysis, "kind", "") or "")
    classification_supported = bool(getattr(analysis, "classification_supported", False))
    if kind == "experimental_monocular" or not reportable:
        classification_supported = False
    has_series = bool(frames)
    stored_validity = str(getattr(analysis, "validity", "") or "")
    stored_mode = str(getattr(analysis, "display_mode", "") or "")
    if not has_series:
        validity = stored_validity or VALIDITY_UNAVAILABLE
        display_mode = stored_mode or DISPLAY_HIDDEN
    elif not reportable or not classification_supported:
        validity = stored_validity or VALIDITY_UNVALIDATED
        display_mode = stored_mode or DISPLAY_PREVIEW
        if validity == VALIDITY_VALID:
            validity = VALIDITY_UNVALIDATED
        if display_mode == DISPLAY_KPI:
            display_mode = DISPLAY_PREVIEW
    else:
        validity = stored_validity or VALIDITY_VALID
        display_mode = stored_mode or DISPLAY_KPI
    classification_reason = str(
        getattr(analysis, "classification_reason", "") or BALANCE_CLASSIFICATION_REASON
    )
    reason = _compose_reason(
        experimental_note=str(getattr(analysis, "experimental_note", "") or ""),
        extra=str(getattr(analysis, "reason", "") or ""),
        classification_supported=classification_supported,
        classification_reason=classification_reason,
        fallback=(
            "Stability margin analysis has no per-frame series."
            if not has_series
            else "COM–BoS signed distance is unvalidated for this method "
            "and is not a physical stability score."
        ),
    )
    mean_margin = getattr(analysis, "mean_margin_m", None)
    min_margin = getattr(analysis, "min_margin_m", None)
    max_margin = getattr(analysis, "max_margin_m", None)
    return BalanceDisplayContract(
        values_m=values,
        times_s=times,
        unit=str(getattr(analysis, "unit", None) or BALANCE_UNIT),
        coordinate_frame=str(
            getattr(analysis, "coordinate_frame", None) or BALANCE_COORDINATE_FRAME
        ),
        method=str(getattr(analysis, "method", None) or BALANCE_METHOD),
        validity=validity,
        reason=reason,
        display_mode=display_mode,
        reportable=reportable,
        classification_supported=classification_supported,
        classification_reason=classification_reason,
        mean_margin_m=None if mean_margin is None else float(mean_margin),
        min_margin_m=None if min_margin is None else float(min_margin),
        max_margin_m=None if max_margin is None else float(max_margin),
        kind=kind,
    )


def _contract_from_analysis(analysis: StabilityMarginAnalysis) -> BalanceDisplayContract:
    return _contract_from_attributes(analysis)


def balance_display_contract(analysis: object | None) -> BalanceDisplayContract:
    """Shared display contract for Balance plots and summary cards."""
    if analysis is None:
        return _unavailable_contract(
            "Stability margin analysis was not computed for this session."
        )
    return _contract_from_attributes(analysis)


def format_stability_diagnostics(
    analysis: StabilityMarginAnalysis | object,
    *,
    low_confidence_only: bool = True,
) -> str:
    """Compact diagnostic line for low-confidence or suspiciously low Stable%."""
    confidence = float(getattr(analysis, "confidence", 0.0) or 0.0)
    stable_pct = getattr(analysis, "stable_pct", None)
    if low_confidence_only and confidence >= _LOW_CONFIDENCE:
        if stable_pct is not None and float(stable_pct) >= _DIAG_STABLE_PCT_MIN:
            return ""
    d = getattr(analysis, "diagnostics", None) or {}
    bits: list[str] = []
    valid = d.get("valid_frames")
    total = d.get("n_frames")
    if valid is not None and total:
        bits.append(f"valid {int(valid)}/{int(total)}")
    n_st = d.get("n_stable")
    n_rd = d.get("n_reduced")
    n_un = d.get("n_unstable")
    if any(v is not None for v in (n_st, n_rd, n_un)):
        bits.append(
            f"states S/R/U {int(n_st or 0)}/{int(n_rd or 0)}/{int(n_un or 0)}"
        )
    ds = d.get("double_support_frames")
    ss = d.get("single_support_frames")
    sw = d.get("swing_frames")
    if any(v is not None for v in (ds, ss, sw)):
        bits.append(
            f"support D/S/W {int(ds or 0)}/{int(ss or 0)}/{int(sw or 0)}"
        )
    carried = d.get("carried_stance_frames")
    if carried:
        bits.append(f"carried stance {int(carried)}")
    collapsed = d.get("collapsed_unmeasured_ap_frames")
    if collapsed:
        bits.append(f"AP span collapsed {int(collapsed)}")
    bias = d.get("mean_depth_bias_m")
    if isinstance(bias, (int, float)) and abs(float(bias)) >= 0.005:
        bits.append(f"depth-layer offset {float(bias):+.3f} m (removed)")
    mx = d.get("mean_com_bos_x_m")
    if isinstance(mx, (int, float)):
        bits.append(f"COM−BoS X {float(mx):+.3f} m")
    mean_margin = getattr(analysis, "mean_margin_m", None)
    if mean_margin is not None:
        bits.append(f"mean margin {float(mean_margin):+.3f} m")
    min_margin = getattr(analysis, "min_margin_m", None)
    max_margin = getattr(analysis, "max_margin_m", None)
    if min_margin is not None and max_margin is not None:
        bits.append(f"range {float(min_margin):+.3f}…{float(max_margin):+.3f} m")
    reason = d.get("low_confidence_reason")
    if reason:
        bits.append(str(reason))
    if not bits:
        return ""
    return "Diagnostics: " + " · ".join(bits)


def analyze_stability_margin(
    com: CenterOfMassAnalysis,
    bos: BaseOfSupportAnalysis,
    *,
    meters_per_unit: float = 1.0,
) -> StabilityMarginAnalysis:
    """Compute per-frame stability margin and state in absolute metres.

    ``meters_per_unit`` converts body-normalized pose coordinates to metres.
    """
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise ValueError("meters_per_unit must be finite and positive")
    bos_by_frame = {f.frame_index: f for f in bos.per_frame}
    per_frame: list[BiomechanicalFrameStability] = []
    margins: list[float] = []
    stable_count = 0
    reduced_count = 0
    unstable_count = 0
    unavailable_count = 0
    support_counts = {
        "double_support": 0,
        "left_stance": 0,
        "right_stance": 0,
        "swing": 0,
    }
    carried_count = 0
    collapsed_count = 0
    depth_biases: list[float] = []
    x_offsets: list[float] = []

    for com_f in com.per_frame:
        bos_f = bos_by_frame.get(com_f.frame_index)
        polygon_source = getattr(bos_f, "polygon_source", "contact") if bos_f else "contact"
        # Carried stance is overlay continuity only — not measured ground contact.
        if (
            bos_f is None
            or not bos_f.polygon_xy
            or polygon_source == "carried_stance"
        ):
            unavailable_count += 1
            if bos_f is not None and polygon_source == "carried_stance":
                carried_count += 1
                support_counts[bos_f.support_type] = (
                    support_counts.get(bos_f.support_type, 0) + 1
                )
            per_frame.append(
                BiomechanicalFrameStability(
                    frame_index=com_f.frame_index,
                    time_s=com_f.time_s,
                    stability_margin_m=None,
                    stability_state="Unavailable",
                    com_projection=(com_f.position[0], com_f.position[2]),
                    confidence=com_f.confidence * 0.4,
                )
            )
            continue

        support_counts[bos_f.support_type] = (
            support_counts.get(bos_f.support_type, 0) + 1
        )

        px, pz = com_f.position[0], com_f.position[2]
        left_xy = list(getattr(bos_f, "left_xy", None) or [])
        right_xy = list(getattr(bos_f, "right_xy", None) or [])
        if left_xy or right_xy:
            plane_pts = _align_feet_to_support_plane([left_xy, right_xy], pz)
            if len(plane_pts) >= 3:
                plane_polygon = _convex_hull_2d(plane_pts)
            else:
                plane_polygon = plane_pts
        else:
            plane_polygon = list(bos_f.polygon_xy)

        (px, pz), polygon, depth_bias = _depth_aligned_polygon(
            px, pz, plane_polygon
        )
        polygon, collapsed_ap = _collapse_unmeasured_ap_span(pz, polygon)
        if collapsed_ap:
            collapsed_count += 1
        depth_biases.append(depth_bias * meters_per_unit)
        bos_cx = sum(p[0] for p in polygon) / float(len(polygon))
        x_offsets.append((px - bos_cx) * meters_per_unit)
        # Overlay stays on the measured world-frame support polygon.
        world_mean_z = (
            sum(p[1] for p in bos_f.polygon_xy) / float(len(bos_f.polygon_xy))
            if bos_f.polygon_xy
            else pz
        )
        support_plane_projection = (px, world_mean_z)

        margin = _signed_margin_to_polygon(
            px, pz, polygon, meters_per_unit=meters_per_unit
        )
        conf = min(com_f.confidence, bos_f.confidence)
        state = (
            _classify_margin(margin)
            if margin is not None
            and math.isfinite(margin)
            and conf >= _MIN_FRAME_CONFIDENCE
            else "Unavailable"
        )

        if margin is not None and math.isfinite(margin) and conf >= _MIN_FRAME_CONFIDENCE:
            margins.append(margin)
            if state == StabilityState.STABLE.value:
                stable_count += 1
            elif state == StabilityState.REDUCED.value:
                reduced_count += 1
            elif state == StabilityState.UNSTABLE.value:
                unstable_count += 1
        else:
            unavailable_count += 1

        per_frame.append(
            BiomechanicalFrameStability(
                frame_index=com_f.frame_index,
                time_s=com_f.time_s,
                stability_margin_m=margin,
                stability_state=state,
                com_projection=support_plane_projection,
                confidence=conf,
            )
        )

    n = len(per_frame)
    valid_ratio = len(margins) / n if n else 0.0
    valid_confidences = [
        frame.confidence
        for frame in per_frame
        if frame.stability_margin_m is not None
        and math.isfinite(frame.stability_margin_m)
        and frame.confidence >= _MIN_FRAME_CONFIDENCE
    ]
    confidence = (
        float(sum(valid_confidences) / len(valid_confidences))
        if valid_confidences
        else 0.0
    )
    warnings: list[str] = []
    mean_margin = None
    min_margin = None
    max_margin = None
    stable_pct = None
    if valid_ratio < _MIN_VALID_FRAME_RATIO:
        warnings.append(
            "Stability margin unavailable: only "
            f"{valid_ratio:.0%} of frames had a valid COM/support polygon with "
            f"confidence ≥ {_MIN_FRAME_CONFIDENCE:.2f}."
        )
    elif margins:
        mean_margin = float(sum(margins) / len(margins))
        min_margin = float(min(margins))
        max_margin = float(max(margins))
        stable_pct = stable_count / len(margins) * 100.0

    single_support = support_counts.get("left_stance", 0) + support_counts.get(
        "right_stance", 0
    )
    mean_bias = (
        float(sum(depth_biases) / len(depth_biases)) if depth_biases else None
    )
    mean_x = float(sum(x_offsets) / len(x_offsets)) if x_offsets else None
    low_reason = ""
    if confidence < _LOW_CONFIDENCE:
        if valid_ratio < _MIN_VALID_FRAME_RATIO:
            low_reason = "few valid COM/BoS frames"
        elif not margins:
            low_reason = "no confident COM–BoS samples"
        else:
            low_reason = "low landmark/contact confidence"
    diagnostics: dict[str, float | int | str | None] = {
        "n_frames": n,
        "valid_frames": len(margins),
        "n_stable": stable_count,
        "n_reduced": reduced_count,
        "n_unstable": unstable_count,
        "n_unavailable": unavailable_count,
        "double_support_frames": support_counts.get("double_support", 0),
        "single_support_frames": single_support,
        "swing_frames": support_counts.get("swing", 0),
        "carried_stance_frames": carried_count,
        "collapsed_unmeasured_ap_frames": collapsed_count,
        "mean_depth_bias_m": None if mean_bias is None else round(mean_bias, 4),
        "mean_com_bos_x_m": None if mean_x is None else round(mean_x, 4),
        "low_confidence_reason": low_reason or None,
        "meters_per_unit": round(float(meters_per_unit), 4),
    }

    xcom_note = _experimental_xcom_note(com, meters_per_unit)
    if xcom_note:
        diagnostics["experimental_xcom"] = xcom_note

    experimental_note = (
        "Monocular COM–BoS is experimental: floor projection uses heuristic "
        "depth (Z), not a measured support polygon. The stable-frame "
        "percentage is a 2-D geometric proxy — not Hof margin of stability "
        "and not evidence from a pelvis-vertical COM graph. XCoM requires "
        "reliable horizontal COM velocity and pendulum length; those are "
        "not available from this video, so XCoM is diagnostic only."
    )
    warnings.append(
        "Physical stability % is not reported as a session KPI "
        "(monocular geometry is insufficient)."
    )
    has_series = bool(per_frame)
    reason = _compose_reason(
        experimental_note=experimental_note,
        extra="",
        classification_supported=False,
        classification_reason=BALANCE_CLASSIFICATION_REASON,
        fallback="",
    )

    return StabilityMarginAnalysis(
        per_frame=per_frame,
        mean_margin_m=mean_margin,
        min_margin_m=min_margin,
        max_margin_m=max_margin,
        stable_pct=stable_pct,
        valid_frame_ratio=valid_ratio,
        confidence=confidence,
        warnings=warnings,
        diagnostics=diagnostics,
        kind="experimental_monocular",
        reportable=False,
        experimental_note=experimental_note,
        unit=BALANCE_UNIT,
        coordinate_frame=BALANCE_COORDINATE_FRAME,
        method=BALANCE_METHOD,
        validity=VALIDITY_UNVALIDATED if has_series else VALIDITY_UNAVAILABLE,
        reason=reason,
        display_mode=DISPLAY_PREVIEW if has_series else DISPLAY_HIDDEN,
        classification_supported=False,
        classification_reason=BALANCE_CLASSIFICATION_REASON,
    )


def _experimental_xcom_note(
    com: CenterOfMassAnalysis,
    meters_per_unit: float,
) -> str:
    """Hof-style XCoM is recorded as a research note, never a session score."""
    frames = com.per_frame
    if len(frames) < 8:
        return ""
    heights = [abs(f.position[1]) * meters_per_unit for f in frames if f.position]
    if not heights:
        return ""
    length_m = max(0.45, float(sum(heights) / len(heights)))
    omega0 = math.sqrt(9.81 / length_m)
    return (
        f"Experimental XCoM uses ω0=√(g/ℓ) with ℓ≈{length_m:.2f} m "
        f"(ω0={omega0:.2f} rad/s). Horizontal COM velocity is monocular "
        "and is not used as a published margin of stability."
    )


__all__ = [
    "BALANCE_CLASSIFICATION_REASON",
    "BALANCE_COORDINATE_FRAME",
    "BALANCE_METHOD",
    "BALANCE_UNIT",
    "BalanceDisplayContract",
    "DISPLAY_HIDDEN",
    "DISPLAY_KPI",
    "DISPLAY_PREVIEW",
    "StabilityMarginAnalysis",
    "VALIDITY_UNAVAILABLE",
    "VALIDITY_UNVALIDATED",
    "VALIDITY_VALID",
    "analyze_stability_margin",
    "balance_display_contract",
    "format_stability_diagnostics",
]
