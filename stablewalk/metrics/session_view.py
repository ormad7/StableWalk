"""Single session metric view — one object shared by Overview, Biomech, Results, Compare, exports.

All UI surfaces should read MetricSnapshot values from here instead of
re-deriving bands or coercing missing fields to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from stablewalk.metrics.definitions import (
    METRIC_DEFINITIONS,
    resolve_metric_key,
)
from stablewalk.metrics.status import (
    DisplayStatus,
    MetricStatus,
    ValidityState,
    build_metric_status,
    engine_stability_state_status,
)


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """Canonical display contract for one metric in a session."""

    status: MetricStatus
    raw_value: float | int | str | None = None
    analysis_path: str = ""
    notes: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return self.status.key

    @property
    def display_name(self) -> str:
        return self.status.display_name

    @property
    def value(self) -> float | int | str | None:
        return self.status.value

    @property
    def unit(self) -> str:
        return self.status.unit

    @property
    def validity(self) -> ValidityState:
        return self.status.validity

    @property
    def display_status(self) -> DisplayStatus:
        return self.status.display_status

    @property
    def display_value(self) -> str:
        return self.status.display_value

    def to_export_dict(self) -> dict[str, Any]:
        out = self.status.to_export_dict()
        out["raw_value"] = self.raw_value
        out["analysis_path"] = self.analysis_path
        out["notes"] = list(self.notes)
        return out


@dataclass
class SessionMetricView:
    """All registered metrics for one analyzed session."""

    snapshots: dict[str, MetricSnapshot] = field(default_factory=dict)
    session_label: str = ""
    contradictions: list[str] = field(default_factory=list)

    def get(self, key: str) -> MetricSnapshot | None:
        return get_snapshot(self.snapshots, key)

    def require(self, key: str) -> MetricSnapshot:
        snap = self.get(key)
        if snap is None:
            raise KeyError(f"Metric {key!r} not in session view")
        return snap

    def export_dict(self) -> dict[str, Any]:
        return {
            "session_label": self.session_label,
            "metrics": {k: v.to_export_dict() for k, v in self.snapshots.items()},
            "contradictions": list(self.contradictions),
        }


def get_snapshot(
    snapshots: Mapping[str, MetricSnapshot], key: str
) -> MetricSnapshot | None:
    resolved = resolve_metric_key(key)
    return snapshots.get(resolved) or snapshots.get(key)


def _gq_available(gq: Any) -> bool:
    if gq is None:
        return False
    if getattr(gq, "available", None) is False:
        return False
    if getattr(gq, "score", None) is None:
        return False
    # Legacy: score 0 with empty components and "Insufficient" factor → unavailable.
    components = getattr(gq, "components", None) or {}
    factors = getattr(gq, "dominant_factors", None) or []
    if (
        float(gq.score) == 0.0
        and not components
        and any("insufficient" in str(f).lower() for f in factors)
    ):
        return False
    return True


def _extract_cycle_count(cycles: Any) -> int | None:
    if cycles is None:
        return None
    for attr in ("usable_cycle_count", "n_usable_cycles", "detected_cycle_count"):
        v = getattr(cycles, attr, None)
        if v is not None:
            return int(v)
    detected = getattr(cycles, "cycles", None)
    if detected is None:
        detected = getattr(cycles, "detected_cycles", None)
    if detected is not None:
        try:
            return len(detected)
        except TypeError:
            pass
    metrics = getattr(cycles, "metrics", None)
    if metrics is not None:
        for attr in ("n_cycles", "cycle_count", "detected_cycles"):
            v = getattr(metrics, attr, None)
            if v is not None:
                return int(v)
    return None


def _extract_cadence(ba: Any, cycles: Any) -> tuple[float | None, float | None]:
    """Return (cadence_spm, confidence) for summary surfaces.

    Low-confidence or unilateral cadence is withheld here; the raw number may
    still exist on cycle metrics as exploratory evidence.
    """
    from stablewalk.analysis.gait_cycle_analysis import (
        cadence_is_exploratory,
        summary_cadence_spm,
    )

    metrics = getattr(cycles, "metrics", None) if cycles is not None else None
    if cadence_is_exploratory(metrics, cycles):
        return None, None
    gm = getattr(ba, "gait_metrics", None) if ba is not None else None
    if gm is not None:
        cad = getattr(gm, "cadence", None)
        if cad is not None and getattr(cad, "value", None) is not None:
            note = str(getattr(cad, "note", "") or "").lower()
            if "exploratory" in note:
                return None, None
            return float(cad.value), float(getattr(cad, "confidence", 0.0) or 0.0)
    withheld = summary_cadence_spm(metrics, cycles)
    if withheld is not None:
        return withheld, None
    return None, None


def _stability_snapshot_notes(sm: Any) -> tuple[str, ...]:
    """Warnings plus the shared Balance contract reason — never a replacement score."""
    if sm is None:
        return ()
    notes = [str(w) for w in (getattr(sm, "warnings", None) or ()) if w]
    try:
        from stablewalk.analysis.biomechanical.stability_margin import (
            balance_display_contract,
            format_stability_diagnostics,
        )

        contract = balance_display_contract(sm)
        if contract.reason:
            notes.append(contract.reason)
        diag = format_stability_diagnostics(sm)
        if diag:
            notes.append(diag)
    except Exception:
        exp = str(getattr(sm, "experimental_note", "") or "").strip()
        if exp:
            notes.append(exp)
    return tuple(notes)


def detect_metric_contradictions(snapshots: Mapping[str, MetricSnapshot]) -> list[str]:
    """Explain real formula divergences — do not alter values."""
    notes: list[str] = []
    gq = get_snapshot(snapshots, "gait_quality_composite")
    stab = get_snapshot(snapshots, "stable_pct")
    if (
        gq is not None
        and stab is not None
        and gq.validity == ValidityState.VALID
        and stab.validity == ValidityState.VALID
        and isinstance(gq.value, (int, float))
        and isinstance(stab.value, (int, float))
    ):
        gq_ok = float(gq.value) >= 60.0
        stab_low = float(stab.value) < 55.0
        if gq_ok and stab_low:
            notes.append(
                "The heuristic gait-quality descriptor and Stability (stable-frame share) "
                "diverge because they use different formulas. The descriptor is a weighted "
                "blend of pose-derived components (stability margin / stable_pct is only "
                "~15% weight when present) and is not a validated instrument. "
                "Stable-frame share is 100×(# frames with COM–BoS margin ≥ 0.04 m) / "
                "(# valid margin frames) and is not physical stability. A session can "
                f"score ~{float(gq.value):.0f}/100 on the heuristic while stable_pct is "
                f"{float(stab.value):.0f}% when other components remain high."
            )
    return notes


def build_session_metric_snapshots(
    *,
    biomechanical: Any = None,
    cycles: Any = None,
    gait_summary: Any = None,
    analysis_confidence_pct: float | None = None,
    analysis_confidence_level: str | None = None,
    session_label: str = "",
    frame_index: int | None = None,
    opensim_available: bool | None = None,
    contact: Any = None,
    spectral: Any = None,
    validity: Any = None,
) -> SessionMetricView:
    """Build the shared metric view from analysis result objects.

    Pass the live ``BiomechanicalAnalysisResult`` (and optional cycle result).
    Does not recompute analysis formulas — only packages display contracts.
    """
    ba = biomechanical
    snaps: dict[str, MetricSnapshot] = {}
    report = validity
    if report is None and (contact is not None or spectral is not None):
        from stablewalk.analysis.result_validity import build_result_validity

        report = build_result_validity(
            biomechanical=ba,
            contact=contact,
            cycles=cycles,
            spectral=spectral,
        )

    # --- Composite Gait Quality ---
    gq = getattr(ba, "gait_quality", None) if ba is not None else None
    # gait_summary is retained for callers; it must not replace composite Q.
    _ = gait_summary
    quality_hold = ""
    laterality_hold = ""
    if report is not None:
        from stablewalk.analysis.result_validity import (
            gait_quality_withhold_reason,
            laterality_withhold_reason,
        )

        quality_hold = gait_quality_withhold_reason(report)
        laterality_hold = laterality_withhold_reason(report)
    if quality_hold:
        snaps["gait_quality_composite"] = MetricSnapshot(
            status=build_metric_status(
                "gait_quality_composite",
                value=None,
                available=False,
                insufficient=True,
            ),
            raw_value=None,
            analysis_path="BiomechanicalAnalysisResult.gait_quality.score",
            notes=(quality_hold,),
        )
    elif _gq_available(gq):
        snaps["gait_quality_composite"] = MetricSnapshot(
            status=build_metric_status(
                "gait_quality_composite",
                value=float(gq.score),
                available=True,
                confidence=float(getattr(gq, "confidence", 0.0) or 0.0),
                completeness=1.0 if getattr(gq, "components", None) else None,
            ),
            raw_value=float(gq.score),
            analysis_path="BiomechanicalAnalysisResult.gait_quality.score",
            notes=tuple(getattr(gq, "dominant_factors", ()) or ()),
        )
    else:
        insufficient = bool(
            gq is not None
            and (
                getattr(gq, "available", None) is False
                or (
                    getattr(gq, "dominant_factors", None)
                    and any(
                        "insufficient" in str(f).lower()
                        for f in gq.dominant_factors
                    )
                )
            )
        )
        snaps["gait_quality_composite"] = MetricSnapshot(
            status=build_metric_status(
                "gait_quality_composite",
                value=None,
                available=False,
                insufficient=insufficient,
                failed=False,
            ),
            raw_value=None,
            analysis_path="BiomechanicalAnalysisResult.gait_quality",
            notes=("Gait quality not scored — insufficient components.",),
        )

    # --- Stability % ---
    sm = getattr(ba, "stability_margin", None) if ba is not None else None
    stab_notes = _stability_snapshot_notes(sm)
    reportable = bool(getattr(sm, "reportable", True)) if sm is not None else False
    if (
        sm is not None
        and getattr(sm, "stable_pct", None) is not None
        and reportable
    ):
        snaps["stable_pct"] = MetricSnapshot(
            status=build_metric_status(
                "stable_pct",
                value=float(sm.stable_pct),
                available=True,
                confidence=float(getattr(sm, "confidence", 0.0) or 0.0),
                completeness=float(getattr(sm, "valid_frame_ratio", 0.0) or 0.0),
            ),
            raw_value=float(sm.stable_pct),
            analysis_path="stability_margin.stable_pct",
            notes=stab_notes,
        )
    else:
        insufficient = bool(
            sm is not None
            and getattr(sm, "warnings", None)
            and any("unavailable" in str(w).lower() for w in sm.warnings)
        )
        snaps["stable_pct"] = MetricSnapshot(
            status=build_metric_status(
                "stable_pct",
                value=None,
                available=False,
                insufficient=insufficient or sm is not None,
            ),
            raw_value=None,
            analysis_path="stability_margin.stable_pct",
            notes=stab_notes if sm is not None else ("Stability margin analysis missing.",),
        )

    # --- Mean margin ---
    if (
        sm is not None
        and getattr(sm, "mean_margin_m", None) is not None
        and reportable
    ):
        snaps["stability_margin_m"] = MetricSnapshot(
            status=build_metric_status(
                "stability_margin_m",
                value=float(sm.mean_margin_m),
                available=True,
                confidence=float(getattr(sm, "confidence", 0.0) or 0.0),
                completeness=float(getattr(sm, "valid_frame_ratio", 0.0) or 0.0),
            ),
            raw_value=float(sm.mean_margin_m),
            analysis_path="stability_margin.mean_margin_m",
            notes=stab_notes,
        )
    else:
        snaps["stability_margin_m"] = MetricSnapshot(
            status=build_metric_status(
                "stability_margin_m",
                value=None,
                available=False,
                insufficient=sm is not None,
            ),
            raw_value=None,
            analysis_path="stability_margin.mean_margin_m",
            notes=stab_notes if sm is not None else (),
        )

    # --- Frame stability state ---
    classification_supported = False
    if sm is not None:
        try:
            from stablewalk.analysis.biomechanical.stability_margin import (
                balance_display_contract,
            )

            classification_supported = bool(
                balance_display_contract(sm).classification_supported
            )
        except Exception:
            classification_supported = False
    frame_state: str | None = None
    frame_conf: float | None = None
    if (
        classification_supported
        and reportable
        and sm is not None
        and frame_index is not None
    ):
        for f in sm.per_frame or []:
            if f.frame_index == frame_index:
                frame_state = f.stability_state
                frame_conf = float(getattr(f, "confidence", 0.0) or 0.0)
                break
    if (
        classification_supported
        and reportable
        and frame_state
        and str(frame_state).lower() != "unavailable"
    ):
        snaps["stability_frame_state"] = MetricSnapshot(
            status=engine_stability_state_status(
                frame_state, available=True, confidence=frame_conf
            ),
            raw_value=frame_state,
            analysis_path="stability_margin.per_frame[].stability_state",
            notes=stab_notes,
        )
    else:
        # Session-level label from stable_pct bands (not engine per-frame word).
        stab_snap = snaps.get("stable_pct")
        if (
            classification_supported
            and reportable
            and stab_snap is not None
            and stab_snap.validity == ValidityState.VALID
            and isinstance(stab_snap.value, (int, float))
        ):
            # Reuse stable_pct status vocabulary for session Stability State card.
            snaps["stability_frame_state"] = MetricSnapshot(
                status=build_metric_status(
                    "stability_frame_state",
                    value=stab_snap.status.display_status.value,
                    available=True,
                    confidence=stab_snap.status.confidence,
                    completeness=stab_snap.status.completeness,
                    force_status=stab_snap.status.display_status,
                    categorical_value=stab_snap.status.display_status.value,
                    extra_user_note=(
                        f"Session label from stable-frame share "
                        f"({float(stab_snap.value):.0f}%), not a single-frame engine state."
                    ),
                ),
                raw_value=float(stab_snap.value),
                analysis_path="stability_margin.stable_pct→session label",
            )
        else:
            snaps["stability_frame_state"] = MetricSnapshot(
                status=build_metric_status(
                    "stability_frame_state",
                    value=None,
                    available=False,
                    insufficient=sm is not None,
                ),
                raw_value=None,
                analysis_path="stability_margin.per_frame",
                notes=stab_notes if sm is not None else (),
            )

    # --- Symmetry ---
    sym = getattr(ba, "symmetry", None) if ba is not None else None
    sym_m = getattr(sym, "overall_symmetry_pct", None) if sym is not None else None
    if laterality_hold:
        snaps["symmetry"] = MetricSnapshot(
            status=build_metric_status(
                "symmetry", value=None, available=False, insufficient=True
            ),
            raw_value=None,
            analysis_path="symmetry.overall_symmetry_pct",
            notes=(laterality_hold,),
        )
    elif sym_m is not None and getattr(sym_m, "value", None) is not None:
        snaps["symmetry"] = MetricSnapshot(
            status=build_metric_status(
                "symmetry",
                value=float(sym_m.value),
                available=True,
                confidence=float(getattr(sym_m, "confidence", 0.0) or 0.0),
            ),
            raw_value=float(sym_m.value),
            analysis_path="symmetry.overall_symmetry_pct",
        )
    else:
        snaps["symmetry"] = MetricSnapshot(
            status=build_metric_status(
                "symmetry", value=None, available=False, insufficient=sym is not None
            ),
            raw_value=None,
            analysis_path="symmetry.overall_symmetry_pct",
        )

    # --- Cadence ---
    cadence, cad_conf = _extract_cadence(ba, cycles)
    if cadence is not None:
        snaps["cadence"] = MetricSnapshot(
            status=build_metric_status(
                "cadence",
                value=float(cadence),
                available=True,
                confidence=cad_conf,
            ),
            raw_value=float(cadence),
            analysis_path="gait_metrics.cadence or cycles.metrics.cadence",
        )
    else:
        snaps["cadence"] = MetricSnapshot(
            status=build_metric_status("cadence", value=None, available=False),
            raw_value=None,
            analysis_path="gait_metrics.cadence",
        )

    # --- Confidence ---
    if analysis_confidence_pct is not None:
        snaps["analysis_confidence"] = MetricSnapshot(
            status=build_metric_status(
                "analysis_confidence",
                value=float(analysis_confidence_pct),
                available=True,
                confidence=float(analysis_confidence_pct) / 100.0,
            ),
            raw_value=float(analysis_confidence_pct),
            analysis_path="analysis_confidence",
            notes=((analysis_confidence_level or ""),)
            if analysis_confidence_level
            else (),
        )
    elif analysis_confidence_level:
        level = analysis_confidence_level.strip().upper()
        # Categorical only — do not invent a percent.
        if level.startswith("HIGH"):
            force = DisplayStatus.ACCEPTABLE
        elif level.startswith("MODERATE") or "PARTIAL" in level:
            force = DisplayStatus.BORDERLINE
        elif level.startswith("LOW"):
            force = DisplayStatus.LOW_CONFIDENCE
        else:
            force = DisplayStatus.HEURISTIC
        snaps["analysis_confidence"] = MetricSnapshot(
            status=build_metric_status(
                "analysis_confidence",
                value=analysis_confidence_level,
                available=True,
                categorical_value=analysis_confidence_level,
                force_status=force,
            ),
            raw_value=analysis_confidence_level,
            analysis_path="analysis_confidence_level",
        )
    else:
        snaps["analysis_confidence"] = MetricSnapshot(
            status=build_metric_status(
                "analysis_confidence", value=None, available=False
            ),
            raw_value=None,
            analysis_path="analysis_confidence",
        )

    # --- Cycles ---
    n_cycles = _extract_cycle_count(cycles)
    if n_cycles is not None:
        snaps["gait_cycles"] = MetricSnapshot(
            status=build_metric_status(
                "gait_cycles",
                value=int(n_cycles),
                available=True,
                # 0 cycles is a genuine computed count; still flag completeness.
                insufficient=int(n_cycles) < 1,
            ),
            raw_value=int(n_cycles),
            analysis_path="gait_cycle_analysis",
        )
    else:
        snaps["gait_cycles"] = MetricSnapshot(
            status=build_metric_status(
                "gait_cycles", value=None, available=False, insufficient=True
            ),
            raw_value=None,
            analysis_path="gait_cycle_analysis",
        )

    # --- OpenSim presence (availability marker only) ---
    if opensim_available is False:
        # Register a lightweight unavailable marker using gait_quality_coordination
        # key only if coordination score isn't already present — skip inventing.
        pass
    del opensim_available  # reserved for future OpenSim metric packaging

    # Ensure every registered definition has an entry (unavailable if missing).
    for key in METRIC_DEFINITIONS:
        if key in snaps:
            continue
        if key == "gait_quality_coordination":
            # Coordination score lives on gait_analysis_summary, not biomech.
            snaps[key] = MetricSnapshot(
                status=build_metric_status(
                    key, value=None, available=False
                ),
                raw_value=None,
                analysis_path="gait_analysis_summary.gait_quality",
            )

    contradictions = detect_metric_contradictions(snaps)
    return SessionMetricView(
        snapshots=snaps,
        session_label=session_label,
        contradictions=contradictions,
    )


def build_session_metric_snapshots_from_summary(
    summary: Any,
    *,
    biomechanical: Any = None,
    cycles: Any = None,
    session_label: str = "",
) -> SessionMetricView:
    """Prefer biomechanical object; fall back to AnalysisSummary display fields."""
    conf_pct = None
    conf_level = None
    if summary is not None:
        ac = getattr(summary, "analysis_confidence", None)
        if ac is not None and getattr(ac, "available", False):
            # Parse "72/100" or "HIGH" style.
            val = str(getattr(ac, "value", "") or "")
            if "/" in val:
                try:
                    conf_pct = float(val.split("/", 1)[0])
                except ValueError:
                    conf_level = val
            else:
                conf_level = val
    view = build_session_metric_snapshots(
        biomechanical=biomechanical,
        cycles=cycles,
        gait_summary=getattr(summary, "gait_summary", None)
        if summary is not None
        else None,
        analysis_confidence_pct=conf_pct,
        analysis_confidence_level=conf_level,
        session_label=session_label,
    )
    return view


def level_from_display_status(status: DisplayStatus) -> str:
    """Map DisplayStatus → legacy Overview/Results level keys."""
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
    }.get(status, "neutral")


def snapshot_status_label(snap: MetricSnapshot) -> str:
    """User-facing status chip; empty when unavailable/neutral."""
    if snap.validity in (
        ValidityState.UNAVAILABLE,
        ValidityState.FAILED,
        ValidityState.INSUFFICIENT_DATA,
    ):
        return DisplayStatus.UNAVAILABLE.value
    return snap.display_status.value


__all__ = [
    "MetricSnapshot",
    "SessionMetricView",
    "build_session_metric_snapshots",
    "build_session_metric_snapshots_from_summary",
    "detect_metric_contradictions",
    "get_snapshot",
    "level_from_display_status",
    "snapshot_status_label",
]
