"""Runtime value provenance for displayed gait metrics.

Attaches source / units / frame / filtering / confidence metadata at display
formatters only. Does not invent values or retune analysis formulas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Scope = Literal["instantaneous", "window", "session", "cycle"]
Tier = Literal["measured", "estimated", "derived", "calculated", "unknown"]

NA_DISPLAY = "N/A"


@dataclass(frozen=True, slots=True)
class ValueProvenance:
    """Provenance for one currently displayed metric value."""

    metric_name: str
    display_value: str
    raw_value: float | str | None
    units: str
    source_frame: int | None
    """0-based playback list index when known."""

    source_function: str
    """Module.function that produced or formatted the value."""

    source_array: str = ""
    """Data structure path, e.g. SkeletonSnapshot.joints['left_knee'].position."""

    coordinate_system: str = ""
    filtering: str = "none"
    confidence: float | None = None
    confidence_note: str = ""
    scope: Scope = "instantaneous"
    tier: Tier = "unknown"
    unavailable_reason: str | None = None
    source_video_frame: int | None = None
    """Original PoseFrame / snapshot.frame_index when known."""

    def format_inspect(self) -> str:
        conf = (
            f"{self.confidence:.3f}"
            if self.confidence is not None
            else "—"
        )
        raw = "—" if self.raw_value is None else repr(self.raw_value)
        lines = [
            f"metric name:      {self.metric_name}",
            f"raw value:        {raw}",
            f"display value:    {self.display_value}",
            f"source frame:     {self.source_frame if self.source_frame is not None else '—'}"
            + (
                f" (video {self.source_video_frame})"
                if self.source_video_frame is not None
                else ""
            ),
            f"source function:  {self.source_function}",
            f"source array:     {self.source_array or '—'}",
            f"units:            {self.units or '—'}",
            f"coordinate sys:   {self.coordinate_system or '—'}",
            f"filtering:        {self.filtering or 'none'}",
            f"confidence:       {conf}"
            + (f" — {self.confidence_note}" if self.confidence_note else ""),
            f"scope:            {self.scope}",
            f"tier:             {self.tier}",
        ]
        if self.unavailable_reason:
            lines.append(f"unavailable:      {self.unavailable_reason}")
        return "\n".join(lines)


@dataclass
class ProvenanceBundle:
    """Mutable map of metric_key → provenance for the current playhead."""

    entries: dict[str, ValueProvenance] = field(default_factory=dict)

    def put(self, key: str, provenance: ValueProvenance) -> ValueProvenance:
        self.entries[key] = provenance
        return provenance

    def get(self, key: str) -> ValueProvenance | None:
        return self.entries.get(key)

    def clear(self) -> None:
        self.entries.clear()

    def format_all(self) -> str:
        if not self.entries:
            return "(no provenance recorded yet — scrub or open a session)"
        blocks = []
        for key in sorted(self.entries):
            blocks.append(f"=== {key} ===\n{self.entries[key].format_inspect()}")
        return "\n\n".join(blocks)


def format_na(reason: str, *, confidence: float | None = None) -> str:
    """Standard unavailable display — never invent a number."""
    reason = (reason or "").strip()
    if confidence is not None and confidence > 0.0:
        if reason:
            return f"{NA_DISPLAY} — {reason} ({confidence:.0%})"
        return f"{NA_DISPLAY} — low confidence ({confidence:.0%})"
    if reason:
        return f"{NA_DISPLAY} — {reason}"
    return NA_DISPLAY


def format_display_or_na(
    value: float | None,
    *,
    formatter,
    reason: str | None = None,
    confidence: float | None = None,
    min_confidence: float = 0.0,
) -> tuple[str, str | None]:
    """Return (display_text, unavailable_reason)."""
    if reason:
        return format_na(reason, confidence=confidence), reason
    if value is None:
        return format_na("value not available"), "value not available"
    if confidence is not None and confidence < min_confidence:
        msg = f"confidence below {min_confidence:.0%}"
        return format_na(msg, confidence=confidence), msg
    try:
        return str(formatter(value)), None
    except Exception:
        return format_na("display formatting failed"), "display formatting failed"


def store_provenance(gui: Any, key: str, provenance: ValueProvenance) -> None:
    bundle = getattr(gui, "_metric_provenance", None)
    if bundle is None:
        bundle = ProvenanceBundle()
        try:
            gui._metric_provenance = bundle
        except Exception:
            return
    bundle.put(key, provenance)


def clear_provenance(gui: Any) -> None:
    bundle = getattr(gui, "_metric_provenance", None)
    if isinstance(bundle, ProvenanceBundle):
        bundle.clear()


def get_provenance_bundle(gui: Any) -> ProvenanceBundle:
    bundle = getattr(gui, "_metric_provenance", None)
    if not isinstance(bundle, ProvenanceBundle):
        bundle = ProvenanceBundle()
        try:
            gui._metric_provenance = bundle
        except Exception:
            pass
    return bundle


def provenance_from_metric_with_confidence(
    *,
    metric_name: str,
    metric: Any | None,
    display_value: str,
    units: str,
    source_function: str,
    source_array: str,
    coordinate_system: str,
    scope: Scope,
    tier: Tier,
    source_frame: int | None = None,
    filtering: str = "none",
    unavailable_reason: str | None = None,
) -> ValueProvenance:
    raw = None
    conf = None
    note = ""
    if metric is not None:
        raw = getattr(metric, "value", None)
        conf = getattr(metric, "confidence", None)
        note = str(getattr(metric, "note", "") or "")
        kind = str(getattr(metric, "kind", "") or "")
        if kind and tier == "unknown":
            if kind in ("measured", "estimated", "derived", "calculated"):
                tier = kind  # type: ignore[assignment]
    return ValueProvenance(
        metric_name=metric_name,
        display_value=display_value,
        raw_value=float(raw) if isinstance(raw, (int, float)) else raw,
        units=units,
        source_frame=source_frame,
        source_function=source_function,
        source_array=source_array,
        coordinate_system=coordinate_system,
        filtering=filtering,
        confidence=float(conf) if conf is not None else None,
        confidence_note=note,
        scope=scope,
        tier=tier,
        unavailable_reason=unavailable_reason,
    )


# Static catalog of expected display contracts (audit checklist).
METRIC_DISPLAY_CONTRACTS: Mapping[str, dict[str, str]] = {
    "position_xyz": {
        "units": "cm (stature-scaled) or m (table)",
        "coordinate_system": "Root-relative (pelvis origin); not world",
        "scope": "instantaneous",
        "tier": "estimated",
        "notes": "Monocular pose; absolute meters approximate",
    },
    "joint_angle": {
        "units": "degrees (DOF angle_deg — not necessarily flexion=180−interior)",
        "coordinate_system": "Pose / DOF flexion channel when present",
        "scope": "instantaneous",
        "tier": "estimated",
        "notes": "Knee charts may convert interior→flexion; joint info uses DOF/joint angle as stored",
    },
    "angular_velocity": {
        "units": "degrees/second",
        "coordinate_system": "Same DOF as joint angle",
        "scope": "instantaneous",
        "tier": "derived",
        "notes": "Must never be labeled as degrees",
    },
    "rom_window": {
        "units": "degrees (min–max of angle series to playhead)",
        "coordinate_system": "Root-relative pose angles",
        "scope": "window",
        "tier": "derived",
    },
    "rom_session_robust": {
        "units": "degrees (robust per-cycle flexion ROM, session mean)",
        "coordinate_system": "Root-relative pose flexion",
        "scope": "session",
        "tier": "estimated",
    },
    "rom_cycle": {
        "units": "degrees (cycle-normalized)",
        "coordinate_system": "Knee flexion series",
        "scope": "cycle",
        "tier": "estimated",
    },
    "path_span": {
        "units": "cm (spatial bounding span — not joint ROM)",
        "coordinate_system": "Root-relative path",
        "scope": "window",
        "tier": "derived",
    },
    "cadence": {
        "units": "steps/min",
        "coordinate_system": "Temporal (heel-strike timing)",
        "scope": "session",
        "tier": "calculated",
    },
    "gait_phase": {
        "units": "categorical phase label",
        "coordinate_system": "Contact / cycle timeline",
        "scope": "instantaneous",
        "tier": "derived",
    },
    "contact_state": {
        "units": "CONTACT / SWING (estimated)",
        "coordinate_system": "Foot–floor contact estimate",
        "scope": "instantaneous",
        "tier": "estimated",
    },
    "symmetry": {
        "units": "%",
        "coordinate_system": "Session L/R temporal or spatial metrics",
        "scope": "session",
        "tier": "derived",
    },
    "stability_margin": {
        "units": "% stable frames or meters (chart)",
        "coordinate_system": "COM vs base-of-support in X–Z (pose-estimated)",
        "scope": "session or instantaneous series",
        "tier": "derived",
    },
    "gait_quality": {
        "units": "/100 composite score",
        "coordinate_system": "n/a (aggregate)",
        "scope": "session",
        "tier": "derived",
        "notes": "Overview Gait Coordination ≠ Biomech Composite Gait Quality",
    },
}


def format_contract_audit() -> str:
    """Human-readable audit of display contracts (no live values)."""
    blocks = []
    for key, spec in METRIC_DISPLAY_CONTRACTS.items():
        lines = [f"=== {key} ==="]
        for field_name, value in spec.items():
            lines.append(f"{field_name}: {value}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_joint_info_provenance(
    *,
    item_id: str,
    list_pos: int | None,
    source_video_frame: int | None,
    x_raw: float | None,
    y_raw: float | None,
    z_raw: float | None,
    x_display: str,
    y_display: str,
    z_display: str,
    angle_raw: float | None,
    angle_display: str,
    omega_raw: float | None,
    omega_display: str,
    rom_display: str,
    confidence: float | None,
    confidence_display: str,
    contact_display: str,
    phase_display: str,
    pos_scale: float,
    anchor: str | None,
) -> dict[str, ValueProvenance]:
    """Provenance bundle entries for Overview joint information fields."""
    joint_path = (
        f"SkeletonSnapshot.joints['{anchor}'].position"
        if anchor
        else "SkeletonSnapshot.joints[<missing>].position"
    )
    scale_note = (
        f"stature_display_scale={pos_scale:.4f} → cm"
        if abs(pos_scale - 1.0) > 1e-6
        else "stature_display_scale=1.0 → cm"
    )
    out: dict[str, ValueProvenance] = {}

    def _xyz(axis: str, raw: float | None, display: str) -> ValueProvenance:
        reason = None
        if display.startswith(NA_DISPLAY) or display in ("—", "-"):
            reason = "joint position missing" if raw is None else None
        return ValueProvenance(
            metric_name=f"{axis.upper()} position",
            display_value=display,
            raw_value=raw,
            units="cm (stature-scaled from recording meters)",
            source_frame=list_pos,
            source_video_frame=source_video_frame,
            source_function="joint_information.build_joint_information",
            source_array=f"{joint_path}.{axis.lower()}",
            coordinate_system="Root-relative (pelvis origin); not world",
            filtering=scale_note,
            confidence=confidence,
            scope="instantaneous",
            tier="estimated",
            unavailable_reason=reason,
        )

    out["position_x"] = _xyz("X", x_raw, x_display)
    out["position_y"] = _xyz("Y", y_raw, y_display)
    out["position_z"] = _xyz("Z", z_raw, z_display)

    angle_reason = None
    if angle_display in ("—", "-") or angle_display.startswith(NA_DISPLAY):
        angle_reason = "DOF/joint angle_deg not available"
    out["joint_angle"] = ValueProvenance(
        metric_name="Current DOF/joint angle (estimated)",
        display_value=angle_display,
        raw_value=angle_raw,
        units="degrees (stored angle_deg; not interior-angle conversion)",
        source_frame=list_pos,
        source_video_frame=source_video_frame,
        source_function="dof_position_table.angle_value_for_item",
        source_array="DofSample.angle_deg or JointSample.angle_deg",
        coordinate_system="Pose DOF / joint angle channel",
        filtering="none",
        confidence=confidence,
        scope="instantaneous",
        tier="estimated",
        unavailable_reason=angle_reason,
    )

    omega_reason = None
    if omega_raw is not None and abs(float(omega_raw)) > 400.0:
        omega_reason = "angular velocity spike hidden (|ω|>400 °/s)"
    elif omega_display in ("—", "-") or omega_display.startswith(NA_DISPLAY):
        omega_reason = "DofSample.velocity_deg_s not available"
    out["angular_velocity"] = ValueProvenance(
        metric_name="Angular velocity (derived)",
        display_value=(
            format_na(omega_reason) if omega_reason and omega_raw is not None and abs(float(omega_raw)) > 400.0
            else omega_display
        ),
        raw_value=omega_raw,
        units="degrees/second",
        source_frame=list_pos,
        source_video_frame=source_video_frame,
        source_function="joint_information._angular_velocity_deg_s",
        source_array="DofSample.velocity_deg_s",
        coordinate_system="Same DOF as joint angle",
        filtering="hide if |ω| > 400 °/s",
        confidence=confidence,
        scope="instantaneous",
        tier="derived",
        unavailable_reason=omega_reason if omega_display in ("—", "-",) or (omega_reason and "spike" in omega_reason) else None,
    )

    rom_reason = None
    if rom_display in ("—", "-") or rom_display.startswith(NA_DISPLAY):
        rom_reason = "insufficient angle samples in window"
    out["rom_window"] = ValueProvenance(
        metric_name="Angle range to frame (window)",
        display_value=rom_display,
        raw_value=None,
        units="degrees (min–max of angle series to playhead)",
        source_frame=list_pos,
        source_video_frame=source_video_frame,
        source_function="joint_information._rom_display",
        source_array="recording.build_time_series().angles / dof_angles",
        coordinate_system="Root-relative pose angles",
        filtering="window [0 … playhead]",
        confidence=confidence,
        scope="window",
        tier="derived",
        unavailable_reason=rom_reason,
    )

    conf_reason = None
    if confidence_display in ("—", "-") or confidence_display.startswith(NA_DISPLAY):
        conf_reason = "keypoint visibility unavailable"
    out["tracking_confidence"] = ValueProvenance(
        metric_name="Tracking confidence (keypoint visibility)",
        display_value=confidence_display,
        raw_value=confidence,
        units="percent (visibility)",
        source_frame=list_pos,
        source_video_frame=source_video_frame,
        source_function="joint_data_table.confidence_for_joint_at_frame",
        source_array="PoseSequence.frames[].keypoints[].visibility",
        coordinate_system="n/a",
        filtering="none",
        confidence=confidence,
        scope="instantaneous",
        tier="measured",
        unavailable_reason=conf_reason,
    )

    contact_reason = None
    if contact_display in ("—", "-"):
        contact_reason = "not a foot analysis point or contact unavailable"
    out["contact_state"] = ValueProvenance(
        metric_name="Contact state (estimated)",
        display_value=contact_display,
        raw_value=None,
        units="CONTACT / SWING (estimated)",
        source_frame=list_pos,
        source_video_frame=source_video_frame,
        source_function="dof_position_table._foot_table_cells",
        source_array="foot clearance / contact pipeline",
        coordinate_system="Estimated floor plane",
        filtering="foot points only",
        scope="instantaneous",
        tier="estimated",
        unavailable_reason=contact_reason,
    )

    phase_reason = None
    if phase_display in ("—", "-"):
        phase_reason = "gait phase not available at this frame"
    out["gait_phase"] = ValueProvenance(
        metric_name="Gait phase (derived)",
        display_value=phase_display,
        raw_value=None,
        units="categorical",
        source_frame=list_pos,
        source_video_frame=source_video_frame,
        source_function="joint_information.build_joint_information",
        source_array="gait_phase argument / cycle analysis",
        coordinate_system="Contact / cycle timeline",
        filtering="none",
        scope="instantaneous",
        tier="derived",
        unavailable_reason=phase_reason,
    )
    return out


__all__ = [
    "METRIC_DISPLAY_CONTRACTS",
    "NA_DISPLAY",
    "ProvenanceBundle",
    "ValueProvenance",
    "build_joint_info_provenance",
    "clear_provenance",
    "format_contract_audit",
    "format_display_or_na",
    "format_na",
    "get_provenance_bundle",
    "provenance_from_metric_with_confidence",
    "store_provenance",
]
