"""Compare workflow: session catalog, validation, and presentation contracts.

Rules
-----
- Session A/B must reference completed analyzed sessions (cache snapshot or bundle),
  never a bare filename / demo label.
- Reload Comparison is enabled only when both slots are valid and distinct.
- Δ = B − A always.
- Positive Δ is not labeled “better” unless directionality is defined.
- Comparison result is fingerprinted and reused (no recalc during playback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from stablewalk.analysis.session_compare import (
    CompareMetrics,
    SessionComparisonResult,
    compare_session_metrics,
)
from stablewalk.ui.tk.session_cache import (
    AnalyzedSessionSnapshot,
    SessionCompareCache,
    sessions_are_identical,
)

Directionality = Literal["higher_better", "lower_better", "none"]


class CompareUiState(str, Enum):
    EMPTY = "empty"
    LOADING = "loading"
    INCOMPLETE = "incomplete"
    ERROR = "error"
    IDENTICAL = "identical"
    READY = "ready"


# Primary bottom-bar comparison metrics (display names).
PRIMARY_COMPARE_METRICS: tuple[str, ...] = (
    "Cadence",
    "Stride Length",
    "Symmetry",
    "Stability",
    "Speed",
    "Joint ROM",
)

# Compact demo strip — Stability / Symmetry / Cadence / ROM only.
DEMO_DELTA_METRICS: tuple[str, ...] = (
    "Stability",
    "Symmetry",
    "Cadence",
    "Joint ROM",
)
DEMO_DELTA_TITLES: dict[str, str] = {
    "Joint ROM": "Knee ROM (session span)",
    "Stability": "Mean COM–BoS Margin",
}
DEMO_COMPARE_LEFT_KEY = "normal"
DEMO_COMPARE_RIGHT_KEY = "abnormal"

# Display thresholds for "notable difference" chips (not statistical p-values).
# Magnitudes match the narrative gates in ``build_comparison_interpretation``.
NOTABLE_DIFF_THRESHOLDS: dict[str, float] = {
    "Cadence": 3.0,  # steps/min
    "Stride Length": 0.02,  # m
    "Step Length": 0.02,  # m
    "Symmetry": 5.0,  # percentage points
    "Stability": 0.002,  # m margin
    "Speed": 0.05,  # m/s
    "Joint ROM": 3.0,  # deg
    "Knee ROM": 3.0,  # deg
    "Gait Quality": 3.0,  # /100
    "COM Excursion": 0.005,  # m
}


def is_notable_difference(name: str, delta: float | None) -> bool:
    """True when |Δ| exceeds the display threshold for ``name``."""
    if delta is None:
        return False
    thr = NOTABLE_DIFF_THRESHOLDS.get(name)
    if thr is None:
        return False
    return abs(float(delta)) >= float(thr)


def notable_difference_label(name: str, delta: float | None) -> str:
    """Short note for tables / chips when a difference is notable."""
    if not is_notable_difference(name, delta):
        return ""
    thr = NOTABLE_DIFF_THRESHOLDS.get(name)
    if thr is None:
        return "Notable difference"
    return f"Notable (|Δ| ≥ {thr:g})"

# Additional metrics behind expandable section.
SECONDARY_COMPARE_METRICS: tuple[str, ...] = (
    "Gait Quality",
    "Step Length",
    "Knee ROM",
    "Hip ROM",
    "Ankle ROM",
    "COM Excursion",
    "Foot clearance",
    "Usable gait cycles",
)

# Metric → (CompareMetrics attr, unit, directionality, unavailable reason)
_METRIC_SPEC: dict[str, tuple[str, str, Directionality, str]] = {
    "Gait Quality": (
        "gait_quality",
        "/100",
        "higher_better",
        "Composite gait quality was not scored for this session.",
    ),
    "Cadence": (
        "cadence_spm",
        " steps/min",
        "higher_better",
        "Cadence unavailable — insufficient heel-strike cycles.",
    ),
    "Symmetry": (
        "symmetry_pct",
        "%",
        "higher_better",
        "Symmetry was not computed for this session.",
    ),
    "Stability": (
        "stability_margin_m",
        " m",
        "higher_better",
        "Stability margin unavailable — insufficient valid COM–BoS frames.",
    ),
    "Knee ROM": (
        "knee_rom_deg",
        "°",
        "higher_better",
        "Knee ROM unavailable — joint angles not estimated.",
    ),
    "Joint ROM": (
        "knee_rom_deg",
        "°",
        "higher_better",
        "Joint ROM unavailable — joint angles not estimated.",
    ),
    "Stride Length": (
        "stride_length_m",
        " m",
        "none",
        "Stride length unavailable — spatial gait metrics not computed.",
    ),
    "Speed": (
        "walking_speed_m_s",
        " m/s",
        "higher_better",
        "Walking speed unavailable — scale/view not reportable.",
    ),
    "Step Length": (
        "step_length_m",
        " m",
        "none",
        "Step length unavailable — spatial gait metrics not computed.",
    ),
    "COM Excursion": (
        "com_excursion_m",
        " m",
        "lower_better",
        "COM excursion unavailable — center-of-mass path missing.",
    ),
    "Hip ROM": (
        "hip_rom_deg",
        "°",
        "none",
        "Hip ROM unavailable — joint angles not estimated.",
    ),
    "Ankle ROM": (
        "ankle_rom_deg",
        "°",
        "none",
        "Ankle ROM unavailable — joint angles not estimated.",
    ),
    "Usable gait cycles": (
        "usable_gait_cycles",
        "",
        "higher_better",
        "No usable gait cycles were detected.",
    ),
    "Foot clearance": (
        "foot_clearance_m",
        " m",
        "none",
        "Foot clearance unavailable — swing height was not computed.",
    ),
}


@dataclass(frozen=True, slots=True)
class CompareSessionCandidate:
    """One selectable analyzed session for the Compare picker."""

    key: str
    session_name: str
    video_name: str
    analysis_date: str
    backend: str
    valid_frame_count: int
    confidence: str
    source_path: str = ""
    is_analyzed: bool = False
    pipeline_complete: bool = False
    incomplete_reason: str = ""
    fps: float | None = None
    duration_s: float | None = None
    kind: str = "cache"  # cache | bundle | demo_label

    def picker_label(self) -> str:
        frames = (
            f"{self.valid_frame_count} frames"
            if self.valid_frame_count > 0
            else "no frames"
        )
        conf = self.confidence or "—"
        date = self.analysis_date or "—"
        backend = self.backend or "—"
        status = "analyzed" if self.is_analyzed else "not analyzed"
        return (
            f"{self.session_name}  ·  {self.video_name}  ·  {date}  ·  "
            f"{backend}  ·  {frames}  ·  conf {conf}  ·  {status}"
        )


@dataclass(frozen=True, slots=True)
class CompareMetricRow:
    name: str
    value_a: float | None
    value_b: float | None
    delta: float | None
    unit: str
    directionality: Directionality
    display_a: str
    display_b: str
    display_delta: str
    reason: str = ""
    available_both: bool = False
    tone: Literal["better", "worse", "neutral", "na"] = "neutral"


@dataclass
class CompareValidation:
    state: CompareUiState
    message: str
    can_reload: bool = False
    can_compare: bool = False
    action: str | None = None  # analyze_session_b | analyze_session_a | pick_sessions
    action_label: str = ""
    left_ready: bool = False
    right_ready: bool = False
    left_reason: str = ""
    right_reason: str = ""
    identical: bool = False


@dataclass
class CachedComparison:
    """Fingerprinted comparison payload — safe to reuse during playback."""

    fingerprint: tuple[Any, ...]
    result: SessionComparisonResult | None
    primary_rows: list[CompareMetricRow] = field(default_factory=list)
    secondary_rows: list[CompareMetricRow] = field(default_factory=list)
    validation: CompareValidation | None = None


def snapshot_is_analyzed(snap: AnalyzedSessionSnapshot | None) -> tuple[bool, str]:
    """Return (ok, reason) for a cache snapshot."""
    if snap is None:
        return False, "No analyzed session is loaded in this slot."
    if not getattr(snap, "ready", False) or int(getattr(snap, "n_frames", 0) or 0) <= 0:
        return False, "Session has no valid pose frames."
    m = getattr(snap, "metrics", None)
    if m is None:
        return False, "Session metrics are missing — analysis incomplete."
    primary = any(
        getattr(m, attr, None) is not None
        for attr in (
            "gait_quality",
            "cadence_spm",
            "walking_speed_m_s",
            "step_length_m",
            "symmetry_pct",
            "stability_margin_m",
            "knee_rom_deg",
        )
    )
    if not primary:
        return (
            False,
            "Analysis incomplete — no primary gait KPIs were computed.",
        )
    # Demo label alone (key set, empty metrics object shared) already caught above.
    if not str(getattr(snap, "key", "") or "").strip():
        return False, "Session identity is missing."
    return True, ""


def candidate_from_snapshot(snap: AnalyzedSessionSnapshot) -> CompareSessionCandidate:
    m = snap.metrics
    conf = ""
    # Prefer explicit snapshot metadata when present.
    conf = str(getattr(snap, "confidence_label", "") or "")
    if not conf and m is not None:
        # Lightweight proxy: usable cycles / presence of gait quality.
        if m.gait_quality is not None:
            conf = "derived"
        elif m.cadence_spm is not None:
            conf = "partial"
        else:
            conf = "low"
    ok, reason = snapshot_is_analyzed(snap)
    video = Path(str(snap.video_path or snap.source or "")).name or snap.label
    return CompareSessionCandidate(
        key=snap.key,
        session_name=snap.label or snap.key,
        video_name=video,
        analysis_date=str(getattr(snap, "analysis_date", "") or ""),
        backend=str(getattr(snap, "backend", "") or "pose"),
        valid_frame_count=int(
            getattr(snap, "valid_frame_count", 0) or snap.n_frames or 0
        ),
        confidence=conf or "—",
        source_path=str(snap.poses_path or snap.video_path or snap.source or ""),
        is_analyzed=ok,
        pipeline_complete=ok,
        incomplete_reason=reason,
        fps=float(snap.fps) if snap.fps else None,
        duration_s=float(m.duration_s) if m and m.duration_s is not None else None,
        kind="cache",
    )


def candidate_from_bundle_metadata(
    path: str | Path,
    meta: dict[str, Any] | None = None,
) -> CompareSessionCandidate:
    bundle = Path(path)
    data = meta or {}
    if not data and (bundle / "session_metadata.json").is_file():
        try:
            import json

            data = json.loads((bundle / "session_metadata.json").read_text(encoding="utf-8"))
        except Exception:
            data = {}
    name = str(data.get("display_name") or data.get("bundle_name") or bundle.name)
    video = Path(str(data.get("video_source") or "")).name or "—"
    frames = int(data.get("frame_count") or 0)
    date = str(data.get("exported_at") or data.get("saved_at") or "")
    backend = str(
        (data.get("workspace") or {}).get("pose_backend")
        or data.get("pose_backend")
        or data.get("backend")
        or "—"
    )
    conf = str(
        data.get("analysis_confidence")
        or (data.get("workspace") or {}).get("analysis_confidence")
        or "—"
    )
    has_results = (bundle / "results_snapshot.json").is_file() or (
        bundle / "analysis_summary.json"
    ).is_file()
    has_poses = (bundle / "poses.json").is_file() or bool(data.get("poses_json_path"))
    analyzed = has_results and (frames > 0 or has_poses)
    reason = ""
    if not analyzed:
        if not has_results:
            reason = "Saved folder has no analysis results — analyze before comparing."
        elif frames <= 0 and not has_poses:
            reason = "Bundle has no pose frames."
    return CompareSessionCandidate(
        key=f"bundle:{bundle.resolve()}",
        session_name=name,
        video_name=video,
        analysis_date=date,
        backend=backend,
        valid_frame_count=frames,
        confidence=conf,
        source_path=str(bundle.resolve()),
        is_analyzed=analyzed,
        pipeline_complete=analyzed,
        incomplete_reason=reason,
        fps=float(data["fps"]) if data.get("fps") not in (None, "") else None,
        kind="bundle",
    )


def candidate_demo_label(key: str, *, analyzed: bool) -> CompareSessionCandidate:
    """Demo key present in selector but not yet analyzed — not selectable as ready."""
    from stablewalk.ui.tk.session_cache import DEMO_COMPARE_KEYS

    label = key.replace("_", " ").title()
    try:
        from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES

        for ex in DEMO_GAIT_EXAMPLES:
            if ex.key == key:
                label = ex.display_name
                break
    except Exception:
        pass
    return CompareSessionCandidate(
        key=key,
        session_name=label,
        video_name=f"{key}.mp4",
        analysis_date="",
        backend="demo",
        valid_frame_count=0,
        confidence="—",
        source_path="",
        is_analyzed=analyzed,
        pipeline_complete=analyzed,
        incomplete_reason=(
            ""
            if analyzed
            else "Demo label only — session has not been analyzed yet."
        ),
        kind="demo_label" if key in DEMO_COMPARE_KEYS else "label",
    )


def list_compare_candidates(
    cache: SessionCompareCache,
    *,
    include_recent_bundles: bool = True,
    include_unanalyzed_demos: bool = True,
) -> list[CompareSessionCandidate]:
    """Catalog for the session picker (analyzed first)."""
    out: list[CompareSessionCandidate] = []
    seen: set[str] = set()

    for key in cache.keys():
        snap = cache.get(key)
        if snap is None:
            continue
        cand = candidate_from_snapshot(snap)
        out.append(cand)
        seen.add(key)

    if include_recent_bundles:
        try:
            from stablewalk.io.session_registry import list_recent_sessions

            for entry in list_recent_sessions(limit=20):
                path = Path(entry.path)
                key = f"bundle:{path.resolve()}" if path.exists() else ""
                if not key or key in seen or entry.path in seen:
                    continue
                cand = candidate_from_bundle_metadata(
                    path,
                    {
                        "display_name": entry.display_name,
                        "video_source": entry.video_source,
                        "exported_at": entry.saved_at,
                        "frame_count": entry.frame_count,
                        "backend": getattr(entry, "backend", "") or "",
                        "analysis_confidence": getattr(entry, "confidence", "") or "",
                    },
                )
                out.append(cand)
                seen.add(key)
                seen.add(entry.path)
        except Exception:
            pass

    if include_unanalyzed_demos:
        from stablewalk.ui.tk.session_cache import DEMO_COMPARE_KEYS

        for key in DEMO_COMPARE_KEYS:
            if key in seen:
                continue
            out.append(candidate_demo_label(key, analyzed=False))
            seen.add(key)

    # Analyzed first, then incomplete.
    out.sort(key=lambda c: (not c.is_analyzed, c.session_name.lower()))
    return out


def validate_compare_slots(
    cache: SessionCompareCache,
    *,
    loading_keys: Sequence[str] | None = None,
    left_override: AnalyzedSessionSnapshot | None = None,
    right_override: AnalyzedSessionSnapshot | None = None,
) -> CompareValidation:
    """Gate Reload / compare rendering for Session A/B."""
    loading = set(loading_keys or ())
    left = left_override if left_override is not None else cache.left()
    right = right_override if right_override is not None else cache.right()
    left_key = cache.left_key
    right_key = cache.right_key

    if not left_key and not right_key:
        return CompareValidation(
            state=CompareUiState.EMPTY,
            message="Select two completed analysis sessions to compare.",
            action="pick_sessions",
            action_label="Choose sessions",
        )

    if left_key in loading or right_key in loading:
        which = []
        if left_key in loading:
            which.append("A")
        if right_key in loading:
            which.append("B")
        return CompareValidation(
            state=CompareUiState.LOADING,
            message=f"Loading Session {' & '.join(which)}…",
            left_ready=left is not None,
            right_ready=right is not None,
        )

    left_ok, left_reason = snapshot_is_analyzed(left)
    right_ok, right_reason = snapshot_is_analyzed(right)

    # Label-only selection (key set, no snapshot).
    if left is None:
        left_ok, left_reason = False, (
            f"Session A ('{left_key}') has not been analyzed yet."
            if left_key
            else "Session A is not selected."
        )
    if right is None:
        right_ok, right_reason = False, (
            f"Session B ('{right_key}') has not been analyzed yet."
            if right_key
            else "Session B is not selected."
        )

    if sessions_are_identical(
        left, right, left_key=left_key, right_key=right_key
    ):
        return CompareValidation(
            state=CompareUiState.IDENTICAL,
            message="Session A and Session B are the same session. Choose two different analyses.",
            can_reload=False,
            can_compare=False,
            left_ready=left_ok,
            right_ready=right_ok,
            identical=True,
            action="pick_sessions",
            action_label="Choose a different Session B",
        )

    if not left_ok and not right_ok:
        return CompareValidation(
            state=CompareUiState.INCOMPLETE,
            message=f"{left_reason} {right_reason}".strip(),
            left_ready=False,
            right_ready=False,
            left_reason=left_reason,
            right_reason=right_reason,
            action="pick_sessions",
            action_label="Choose analyzed sessions",
        )

    if not right_ok:
        return CompareValidation(
            state=CompareUiState.INCOMPLETE,
            message=right_reason or "Session B has not been analyzed yet.",
            left_ready=left_ok,
            right_ready=False,
            left_reason=left_reason,
            right_reason=right_reason,
            action="analyze_session_b",
            action_label="Analyze Session B",
        )

    if not left_ok:
        return CompareValidation(
            state=CompareUiState.INCOMPLETE,
            message=left_reason or "Session A has not been analyzed yet.",
            left_ready=False,
            right_ready=right_ok,
            left_reason=left_reason,
            right_reason=right_reason,
            action="analyze_session_a",
            action_label="Analyze Session A",
        )

    # Different FPS / lengths are allowed — comparison uses normalized progress.
    return CompareValidation(
        state=CompareUiState.READY,
        message="Comparison ready (Δ = B − A).",
        can_reload=True,
        can_compare=True,
        left_ready=True,
        right_ready=True,
    )


def _metric_value(metrics: CompareMetrics | None, attr: str) -> float | None:
    if metrics is None:
        return None
    v = getattr(metrics, attr, None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt_value(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    if abs(value) >= 100:
        body = f"{value:.0f}"
    elif abs(value) >= 10:
        body = f"{value:.1f}"
    else:
        body = f"{value:.2f}"
    return f"{body}{unit}"


def _tone_for_delta(
    delta: float | None, directionality: Directionality
) -> Literal["better", "worse", "neutral", "na"]:
    if delta is None:
        return "na"
    if abs(delta) < 1e-12:
        return "neutral"
    if directionality == "none":
        return "neutral"
    if directionality == "higher_better":
        return "better" if delta > 0 else "worse"
    if directionality == "lower_better":
        return "better" if delta < 0 else "worse"
    return "neutral"


def build_metric_row(
    name: str,
    left: CompareMetrics | None,
    right: CompareMetrics | None,
) -> CompareMetricRow:
    attr, unit, directionality, reason_default = _METRIC_SPEC[name]
    display_unit = unit

    va = _metric_value(left, attr)
    vb = _metric_value(right, attr)
    if va is None and vb is None:
        from stablewalk.ui.unavailable_labels import compare_reason_label

        reason = compare_reason_label(
            name,
            left_available=False,
            right_available=False,
            reason_default=reason_default,
            left_session_ready=left is not None,
            right_session_ready=right is not None,
        )
        return CompareMetricRow(
            name=name,
            value_a=None,
            value_b=None,
            delta=None,
            unit=display_unit,
            directionality=directionality,
            display_a="—",
            display_b="—",
            display_delta="—",
            reason=reason,
            available_both=False,
            tone="na",
        )
    if va is None or vb is None:
        from stablewalk.ui.unavailable_labels import compare_reason_label

        reason = compare_reason_label(
            name,
            left_available=va is not None,
            right_available=vb is not None,
            reason_default=reason_default,
            left_session_ready=left is not None,
            right_session_ready=right is not None,
        )
        return CompareMetricRow(
            name=name,
            value_a=va,
            value_b=vb,
            delta=None,
            unit=display_unit,
            directionality=directionality,
            display_a=_fmt_value(va, display_unit) if va is not None else "—",
            display_b=_fmt_value(vb, display_unit) if vb is not None else "—",
            display_delta="—",
            reason=reason,
            available_both=False,
            tone="na",
        )

    delta = float(vb) - float(va)  # Δ = B − A
    sign = "+" if delta >= 0 else ""
    tone = _tone_for_delta(delta, directionality)
    if abs(delta) >= 100:
        delta_body = f"{delta:.0f}"
    elif abs(delta) >= 10:
        delta_body = f"{delta:.1f}"
    else:
        delta_body = f"{delta:.2f}"
    delta_txt = f"{sign}{delta_body}{display_unit}"
    if abs(float(va)) > 1e-9:
        pct = (delta / float(va)) * 100.0
        pct_sign = "+" if pct >= 0 else ""
        delta_txt = f"{delta_txt} ({pct_sign}{pct:.1f}%)"
    reason = notable_difference_label(name, delta)
    return CompareMetricRow(
        name=name,
        value_a=va,
        value_b=vb,
        delta=delta,
        unit=display_unit,
        directionality=directionality,
        display_a=_fmt_value(va, display_unit),
        display_b=_fmt_value(vb, display_unit),
        display_delta=delta_txt,
        reason=reason,
        available_both=True,
        tone=tone,
    )


def build_demo_delta_rows(
    left: CompareMetrics | None,
    right: CompareMetrics | None,
) -> list[CompareMetricRow]:
    """Four basic Δ cards for demo comparison (no extra plots)."""
    return build_comparison_rows(left, right, names=DEMO_DELTA_METRICS)


def assign_demo_comparison_slots(cache: Any) -> tuple[str, str]:
    """Point Compare at Normal (A) vs Abnormal (B)."""
    cache.set_slot("left", DEMO_COMPARE_LEFT_KEY)
    cache.set_slot("right", DEMO_COMPARE_RIGHT_KEY)
    return DEMO_COMPARE_LEFT_KEY, DEMO_COMPARE_RIGHT_KEY


def build_comparison_rows(
    left: CompareMetrics | None,
    right: CompareMetrics | None,
    *,
    names: Sequence[str],
    only_both_available: bool = False,
) -> list[CompareMetricRow]:
    rows: list[CompareMetricRow] = []
    for name in names:
        if name not in _METRIC_SPEC:
            continue
        row = build_metric_row(name, left, right)
        if only_both_available and not row.available_both:
            # Still include with reason (requirement 8).
            pass
        rows.append(row)
    return rows


def comparison_fingerprint(
    left: AnalyzedSessionSnapshot | None,
    right: AnalyzedSessionSnapshot | None,
    *,
    left_key: str,
    right_key: str,
) -> tuple[Any, ...]:
    return (
        left_key,
        right_key,
        None if left is None else id(left),
        None if right is None else id(right),
        None if left is None else left.n_frames,
        None if right is None else right.n_frames,
        None if left is None else left.metrics.gait_quality,
        None if right is None else right.metrics.gait_quality,
        None if left is None else left.metrics.cadence_spm,
        None if right is None else right.metrics.cadence_spm,
        None if left is None else left.metrics.symmetry_pct,
        None if right is None else right.metrics.symmetry_pct,
        None if left is None else left.metrics.stability_margin_m,
        None if right is None else right.metrics.stability_margin_m,
        None if left is None else left.metrics.walking_speed_m_s,
        None if right is None else right.metrics.walking_speed_m_s,
        None if left is None else left.metrics.knee_rom_deg,
        None if right is None else right.metrics.knee_rom_deg,
    )


def build_cached_comparison(
    cache: SessionCompareCache,
    *,
    loading_keys: Sequence[str] | None = None,
    previous: CachedComparison | None = None,
    left_override: AnalyzedSessionSnapshot | None = None,
    right_override: AnalyzedSessionSnapshot | None = None,
) -> CachedComparison:
    """Build or reuse a CachedComparison for the current A/B slots."""
    validation = validate_compare_slots(
        cache,
        loading_keys=loading_keys,
        left_override=left_override,
        right_override=right_override,
    )
    left = left_override if left_override is not None else cache.left()
    right = right_override if right_override is not None else cache.right()
    fp = comparison_fingerprint(
        left, right, left_key=cache.left_key, right_key=cache.right_key
    )
    if (
        previous is not None
        and previous.fingerprint == fp
        and previous.validation is not None
        and previous.validation.state == validation.state
    ):
        return previous

    if not validation.can_compare or left is None or right is None:
        return CachedComparison(
            fingerprint=fp,
            result=None,
            primary_rows=[],
            secondary_rows=[],
            validation=validation,
        )

    result = compare_session_metrics(left.metrics, right.metrics)
    primary = build_comparison_rows(
        left.metrics, right.metrics, names=PRIMARY_COMPARE_METRICS
    )
    secondary = build_comparison_rows(
        left.metrics, right.metrics, names=SECONDARY_COMPARE_METRICS
    )
    return CachedComparison(
        fingerprint=fp,
        result=result,
        primary_rows=primary,
        secondary_rows=secondary,
        validation=validation,
    )


def joint_curves_ready(
    left: AnalyzedSessionSnapshot | None,
    right: AnalyzedSessionSnapshot | None,
) -> tuple[bool, str]:
    """Synchronized joint-angle curves only when both trajectories are valid."""
    if left is None or right is None:
        return False, "Both sessions must be analyzed before joint curves are shown."
    ok_a, reason_a = snapshot_is_analyzed(left)
    ok_b, reason_b = snapshot_is_analyzed(right)
    if not ok_a:
        return False, reason_a
    if not ok_b:
        return False, reason_b
    if sessions_are_identical(left, right, left_key=left.key, right_key=right.key):
        return False, "Sessions are identical — joint overlay is disabled."
    ya = [v for v in (left.knee_y or []) if v is not None]
    yb = [v for v in (right.knee_y or []) if v is not None]
    if len(ya) < 3 or len(yb) < 3:
        return False, "Knee angle trajectories are incomplete for one or both sessions."
    return True, ""


__all__ = [
    "DEMO_COMPARE_LEFT_KEY",
    "DEMO_COMPARE_RIGHT_KEY",
    "DEMO_DELTA_METRICS",
    "DEMO_DELTA_TITLES",
    "NOTABLE_DIFF_THRESHOLDS",
    "PRIMARY_COMPARE_METRICS",
    "SECONDARY_COMPARE_METRICS",
    "CachedComparison",
    "CompareMetricRow",
    "CompareSessionCandidate",
    "CompareUiState",
    "CompareValidation",
    "assign_demo_comparison_slots",
    "build_cached_comparison",
    "build_comparison_rows",
    "build_demo_delta_rows",
    "build_metric_row",
    "candidate_from_bundle_metadata",
    "candidate_from_snapshot",
    "comparison_fingerprint",
    "is_notable_difference",
    "joint_curves_ready",
    "list_compare_candidates",
    "notable_difference_label",
    "snapshot_is_analyzed",
    "validate_compare_slots",
]
