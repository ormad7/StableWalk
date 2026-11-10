"""Shared gait-timing model used by Spectral, Four Gait, and Overview.

Period estimates, cadence, and cycle counts are labeled by *kind* so a
session ACF consensus is never shown as a selected-channel value, and an
estimated cycle count (duration / period) is never shown as a detected
heel-strike count.

Cadence from a step period is ``60 / T_step``. Converting a stride period
with ``120 / T_stride`` assumes two steps per stride and is stored separately;
asymmetric gait is not forced into that 2:1 ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Same numbers as spectral.config. Inlined so this module does not import
# stablewalk.analysis.spectral (that package's __init__ loads report.py).
DEFAULT_STRIDE_LAG_S: tuple[float, float] = (0.55, 2.20)
DEFAULT_STEP_LAG_S: tuple[float, float] = (0.30, 0.90)

PeriodKind = Literal[
    "session_acf_stride",
    "channel_acf_stride",
    "channel_acf_step",
    "event_hs_hs",
]

ACF_CHANNEL_PREPROCESSING = (
    "z-scored processed channel (linear detrend + unit-variance); "
    "longest finite run; mean-remove; biased Pearson ρ(k)"
)
ACF_PEAK_SELECTION_STRIDE = (
    "positive local maximum of ρ(τ) in the stride-lag window; "
    "exclusive-window assignment prefers a peak above the step-window ceiling"
)
ACF_PEAK_SELECTION_STEP = (
    "positive local maximum of ρ(τ) in the step-lag window; "
    "independent of stride (not rewritten to stride/2)"
)
SESSION_STRIDE_PREPROCESSING = (
    "Each contributing thigh/shank channel uses "
    + ACF_CHANNEL_PREPROCESSING
    + ". Session value is the median of those channel stride lags."
)
SESSION_STRIDE_PEAK_SELECTION = (
    "median of reliable channel ACF stride lags; each lag is a qualifying "
    "positive local maximum of ρ(τ), not a sampled ACF point"
)
EVENT_STRIDE_PREPROCESSING = (
    "Ipsilateral heel-strike times from the contact FSM; no ACF."
)
EVENT_STRIDE_PEAK_SELECTION = (
    "median complete ipsilateral HS–HS interval inside 0.50–2.50 s"
)
CadenceMethod = Literal[
    "sixty_over_step",
    "one_twenty_over_stride_assumes_two_steps",
    "event_alternating_hs",
    "event_one_twenty_over_stride_assumes_two_steps",
    "unavailable",
]


@dataclass(frozen=True)
class EstimateProvenance:
    """Source, window, preprocessing, peak method, and confidence for one estimate."""

    source_signal: str
    time_window_s: tuple[float, float] | None
    preprocessing: str
    peak_selection: str
    confidence: float | None
    confidence_label: str = "insufficient"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_signal": self.source_signal,
            "time_window_s": list(self.time_window_s) if self.time_window_s else None,
            "preprocessing": self.preprocessing,
            "peak_selection": self.peak_selection,
            "confidence": self.confidence,
            "confidence_label": self.confidence_label,
        }


def _default_provenance() -> EstimateProvenance:
    return EstimateProvenance(
        source_signal="",
        time_window_s=None,
        preprocessing="",
        peak_selection="",
        confidence=None,
    )


@dataclass(frozen=True)
class PeriodEstimate:
    """One period with its method, channel, and search window."""

    kind: PeriodKind
    value_s: float | None
    method: str
    channel_id: str | None = None
    window_s: tuple[float, float] | None = None
    reliability_label: str = "insufficient"
    n_support: int = 0
    notes: tuple[str, ...] = ()
    provenance: EstimateProvenance = field(default_factory=_default_provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value_s": self.value_s,
            "method": self.method,
            "channel_id": self.channel_id,
            "window_s": list(self.window_s) if self.window_s else None,
            "reliability_label": self.reliability_label,
            "n_support": self.n_support,
            "notes": list(self.notes),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class CadenceEstimate:
    """Cadence in steps/min, with the conversion that produced it."""

    steps_per_min: float | None
    method: CadenceMethod
    source_period_s: float | None
    assumes_two_steps_per_stride: bool = False
    notes: tuple[str, ...] = ()
    provenance: EstimateProvenance = field(default_factory=_default_provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps_per_min": self.steps_per_min,
            "method": self.method,
            "source_period_s": self.source_period_s,
            "assumes_two_steps_per_stride": self.assumes_two_steps_per_stride,
            "notes": list(self.notes),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class CycleCountEstimate:
    """Estimated (duration/period) vs detected (heel-strike) cycle counts."""

    estimated_from_channel_stride: float | None
    estimated_from_session_stride: float | None
    period_used_for_channel_estimate_s: float | None
    detected: int
    valid: int
    usable: int | None
    detected_durations_s: tuple[float, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_from_channel_stride": self.estimated_from_channel_stride,
            "estimated_from_session_stride": self.estimated_from_session_stride,
            "period_used_for_channel_estimate_s": self.period_used_for_channel_estimate_s,
            "detected": self.detected,
            "valid": self.valid,
            "usable": self.usable,
            "detected_durations_s": list(self.detected_durations_s),
            "notes": list(self.notes),
        }


@dataclass
class GaitTimingResult:
    """Single timing product shared across tabs."""

    duration_s: float
    fps: float
    session_stride: PeriodEstimate
    channel_stride: PeriodEstimate
    channel_step: PeriodEstimate
    event_stride: PeriodEstimate
    cadence_spectral: CadenceEstimate
    cadence_event: CadenceEstimate
    cycles: CycleCountEstimate
    consistency_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "stablewalk-gait-timing",
            "schema_version": "1.0",
            "duration_s": self.duration_s,
            "fps": self.fps,
            "session_stride": self.session_stride.to_dict(),
            "channel_stride": self.channel_stride.to_dict(),
            "channel_step": self.channel_step.to_dict(),
            "event_stride": self.event_stride.to_dict(),
            "cadence_spectral": self.cadence_spectral.to_dict(),
            "cadence_event": self.cadence_event.to_dict(),
            "cycles": self.cycles.to_dict(),
            "consistency_notes": list(self.consistency_notes),
        }


def _finite(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num or num in (float("inf"), float("-inf")):
        return None
    return num


def cadence_from_step_period(step_period_s: float | None) -> CadenceEstimate:
    step = _finite(step_period_s)
    if step is None or step <= 1e-6:
        return CadenceEstimate(
            steps_per_min=None,
            method="unavailable",
            source_period_s=None,
            notes=("No independent ACF step period.",),
        )
    return CadenceEstimate(
        steps_per_min=float(60.0 / step),
        method="sixty_over_step",
        source_period_s=step,
        assumes_two_steps_per_stride=False,
        notes=("Cadence = 60 / T_step from the ACF step-lag window.",),
    )


def cadence_from_stride_two_steps(stride_period_s: float | None) -> CadenceEstimate:
    stride = _finite(stride_period_s)
    if stride is None or stride <= 1e-6:
        return CadenceEstimate(
            steps_per_min=None,
            method="unavailable",
            source_period_s=None,
            assumes_two_steps_per_stride=True,
            notes=("No stride period for a two-steps-per-stride conversion.",),
        )
    return CadenceEstimate(
        steps_per_min=float(120.0 / stride),
        method="one_twenty_over_stride_assumes_two_steps",
        source_period_s=stride,
        assumes_two_steps_per_stride=True,
        notes=(
            "Cadence = 120 / T_stride. This assumes two steps per stride and "
            "is not used when an independent step period exists.",
        ),
    )


def _cycle_list(cycles: Any | None) -> list[Any]:
    if cycles is None:
        return []
    nested = getattr(cycles, "cycles", None)
    if nested is not None and nested is not cycles:
        return list(nested)
    if isinstance(cycles, (list, tuple)):
        return list(cycles)
    return []


def compose_gait_timing(
    *,
    duration_s: float,
    fps: float,
    channel_acf: Any | None = None,
    session_period_s: float | None = None,
    session_reason: str = "",
    session_channel_ids: tuple[str, ...] = (),
    event_stride_s: float | None = None,
    event_stride_n: int = 0,
    acf_event_note: str = "",
    cycles: Any | None = None,
    usable_cycles: int | None = None,
    stride_window_s: tuple[float, float] = DEFAULT_STRIDE_LAG_S,
    step_window_s: tuple[float, float] = DEFAULT_STEP_LAG_S,
) -> GaitTimingResult:
    """Build the shared timing object from already-computed ACF / event results."""
    channel_id = str(getattr(channel_acf, "channel_id", "") or "") or None
    channel_rel = str(getattr(channel_acf, "reliability_label", "") or "insufficient")
    channel_conf = _finite(getattr(channel_acf, "reliability", None) if channel_acf else None)
    stride_s = _finite(getattr(channel_acf, "stride_period_s", None) if channel_acf else None)
    step_s = _finite(getattr(channel_acf, "step_period_s", None) if channel_acf else None)
    assign_note = str(getattr(channel_acf, "stride_assignment_note", "") or "")
    channel_notes = tuple(
        n for n in ([assign_note, *list(getattr(channel_acf, "notes", None) or [])] if channel_acf else [])
        if n
    )
    session_s = _finite(session_period_s)
    event_s = _finite(event_stride_s)
    time_window = getattr(channel_acf, "time_window_s", None) if channel_acf else None
    if not time_window and duration_s > 0:
        time_window = (0.0, float(duration_s))
    elif time_window:
        time_window = (float(time_window[0]), float(time_window[1]))
    preprocessing = str(
        getattr(channel_acf, "preprocessing", "") or ACF_CHANNEL_PREPROCESSING
    )
    source = channel_id or "selected-channel ACF"

    session_stride = PeriodEstimate(
        kind="session_acf_stride",
        value_s=session_s,
        method="median of reliable thigh/shank ACF stride lags",
        channel_id=",".join(session_channel_ids) if session_channel_ids else None,
        window_s=stride_window_s,
        reliability_label="low" if session_s is not None else "insufficient",
        n_support=len(session_channel_ids),
        notes=tuple(n for n in (session_reason,) if n),
        provenance=EstimateProvenance(
            source_signal=(
                ", ".join(session_channel_ids)
                if session_channel_ids
                else "thigh/shank ACF channels"
            ),
            time_window_s=time_window,
            preprocessing=SESSION_STRIDE_PREPROCESSING,
            peak_selection=SESSION_STRIDE_PEAK_SELECTION,
            confidence=None,
            confidence_label="low" if session_s is not None else "insufficient",
        ),
    )
    channel_stride = PeriodEstimate(
        kind="channel_acf_stride",
        value_s=stride_s,
        method="selected-channel ACF peak in the stride-lag window",
        channel_id=channel_id,
        window_s=stride_window_s,
        reliability_label=channel_rel if stride_s is not None else "insufficient",
        n_support=1 if stride_s is not None else 0,
        notes=channel_notes,
        provenance=EstimateProvenance(
            source_signal=source,
            time_window_s=time_window,
            preprocessing=preprocessing,
            peak_selection=str(
                getattr(channel_acf, "peak_selection", "") or ACF_PEAK_SELECTION_STRIDE
            ),
            confidence=channel_conf if stride_s is not None else None,
            confidence_label=channel_rel if stride_s is not None else "insufficient",
        ),
    )
    channel_step = PeriodEstimate(
        kind="channel_acf_step",
        value_s=step_s,
        method="selected-channel ACF peak in the step-lag window",
        channel_id=channel_id,
        window_s=step_window_s,
        reliability_label=channel_rel if step_s is not None else "insufficient",
        n_support=1 if step_s is not None else 0,
        notes=(),
        provenance=EstimateProvenance(
            source_signal=source,
            time_window_s=time_window,
            preprocessing=preprocessing,
            peak_selection=ACF_PEAK_SELECTION_STEP,
            confidence=channel_conf if step_s is not None else None,
            confidence_label=channel_rel if step_s is not None else "insufficient",
        ),
    )
    event_stride = PeriodEstimate(
        kind="event_hs_hs",
        value_s=event_s,
        method="median ipsilateral heel-strike to heel-strike interval",
        window_s=(0.50, 2.50),
        reliability_label="low" if event_s is not None else "insufficient",
        n_support=int(event_stride_n or 0),
        notes=tuple(n for n in (acf_event_note,) if n),
        provenance=EstimateProvenance(
            source_signal="contact FSM ipsilateral heel-strike times",
            time_window_s=time_window,
            preprocessing=EVENT_STRIDE_PREPROCESSING,
            peak_selection=EVENT_STRIDE_PEAK_SELECTION,
            confidence=None,
            confidence_label="low" if event_s is not None else "insufficient",
        ),
    )

    if step_s is not None:
        cadence_spectral = cadence_from_step_period(step_s)
        cadence_spectral = CadenceEstimate(
            steps_per_min=cadence_spectral.steps_per_min,
            method=cadence_spectral.method,
            source_period_s=cadence_spectral.source_period_s,
            assumes_two_steps_per_stride=False,
            notes=cadence_spectral.notes,
            provenance=EstimateProvenance(
                source_signal=source,
                time_window_s=time_window,
                preprocessing=preprocessing,
                peak_selection="60 / T_step from a qualifying step-lag ACF local maximum",
                confidence=channel_conf,
                confidence_label=channel_rel if step_s is not None else "insufficient",
            ),
        )
    else:
        cadence_spectral = cadence_from_stride_two_steps(stride_s)
        cadence_spectral = CadenceEstimate(
            steps_per_min=cadence_spectral.steps_per_min,
            method=cadence_spectral.method,
            source_period_s=cadence_spectral.source_period_s,
            assumes_two_steps_per_stride=True,
            notes=cadence_spectral.notes,
            provenance=EstimateProvenance(
                source_signal=source,
                time_window_s=time_window,
                preprocessing=preprocessing,
                peak_selection=(
                    "120 / T_stride (assumes two steps per stride) because no "
                    "independent step-lag local maximum exists"
                ),
                confidence=channel_conf if stride_s is not None else None,
                confidence_label=channel_rel if stride_s is not None else "insufficient",
            ),
        )

    metrics = getattr(cycles, "metrics", None)
    event_step = _finite(getattr(metrics, "step_time_s", None) if metrics else None)
    event_stride_time = _finite(getattr(metrics, "stride_time_s", None) if metrics else None)
    event_cadence = _finite(getattr(metrics, "cadence_steps_per_min", None) if metrics else None)
    if event_cadence is not None and event_step is not None:
        cadence_event = CadenceEstimate(
            steps_per_min=event_cadence,
            method="event_alternating_hs",
            source_period_s=event_step,
            assumes_two_steps_per_stride=False,
            notes=("Event cadence = 60 / mean alternating heel-strike interval.",),
        )
    elif event_cadence is not None:
        cadence_event = CadenceEstimate(
            steps_per_min=event_cadence,
            method="event_one_twenty_over_stride_assumes_two_steps",
            source_period_s=event_stride_time,
            assumes_two_steps_per_stride=True,
            notes=(
                "No alternating step intervals; event cadence used 120 / mean "
                "same-side HS–HS duration (two steps per stride assumed).",
            ),
        )
    else:
        cadence_event = CadenceEstimate(
            steps_per_min=None,
            method="unavailable",
            source_period_s=None,
            notes=("Event cadence was not computed.",),
        )

    cycle_list = _cycle_list(cycles)
    durations = tuple(
        float(c.duration_s)
        for c in cycle_list
        if getattr(c, "duration_s", 0) and float(c.duration_s) > 1e-6
    )
    est_channel = (float(duration_s) / stride_s) if stride_s and stride_s > 1e-6 else None
    est_session = (float(duration_s) / session_s) if session_s and session_s > 1e-6 else None
    cycle_notes = [
        "Estimated cycle count is clip duration / ACF stride period, not a heel-strike count.",
        "Detected count is complete ipsilateral HS–HS intervals from the contact FSM.",
    ]
    if est_channel is not None and len(durations) and abs(est_channel - len(durations)) > 2:
        cycle_notes.append(
            f"Estimated {est_channel:.1f} channel-stride cycles vs {len(durations)} "
            "detected HS–HS cycles. These estimators are independent; a large "
            "gap is a quality warning, not a reason to rewrite either value."
        )

    consistency: list[str] = []
    if stride_s is not None and step_s is not None and step_s > 1e-6:
        ratio = stride_s / step_s
        if 1.7 <= ratio <= 2.3:
            consistency.append(
                f"Channel step {step_s:.3f} s is about half the channel stride "
                f"{stride_s:.3f} s (ratio {ratio:.2f}). That is a check, not a constraint."
            )
        else:
            consistency.append(
                f"Channel step {step_s:.3f} s is not half of channel stride "
                f"{stride_s:.3f} s (ratio {ratio:.2f}). Cadence uses 60/T_step; "
                "stride is not rewritten to 2×step."
            )
    if session_s is not None and stride_s is not None:
        rel = abs(session_s - stride_s) / session_s if session_s > 1e-9 else float("inf")
        if rel > 0.25:
            consistency.append(
                f"Session ACF stride {session_s:.3f} s differs from selected-channel "
                f"ACF stride {stride_s:.3f} s. Different channels and preprocessing; "
                "neither value is overwritten."
            )
    if acf_event_note:
        consistency.append(acf_event_note)
    out_of_band = [d for d in durations if d < 0.50 or d > 2.50]
    if out_of_band:
        consistency.append(
            f"{len(out_of_band)} detected HS–HS interval(s) sit outside the "
            "0.50–2.50 s event-stride band used for the independent heel-strike "
            "interval. Detected count still includes them; the event stride "
            "median does not. Thresholds were not retuned."
        )

    return GaitTimingResult(
        duration_s=float(duration_s),
        fps=float(fps),
        session_stride=session_stride,
        channel_stride=channel_stride,
        channel_step=channel_step,
        event_stride=event_stride,
        cadence_spectral=cadence_spectral,
        cadence_event=cadence_event,
        cycles=CycleCountEstimate(
            estimated_from_channel_stride=est_channel,
            estimated_from_session_stride=est_session,
            period_used_for_channel_estimate_s=stride_s,
            detected=len(cycle_list),
            valid=len(durations),
            usable=usable_cycles,
            detected_durations_s=durations,
            notes=tuple(cycle_notes),
        ),
        consistency_notes=consistency,
    )


def merge_gait_timing(spectral: Any | None, four_gait: Any | None, cycles: Any | None) -> GaitTimingResult | None:
    """Recompose timing after either tab report lands."""
    spec_timing = getattr(spectral, "timing", None) if spectral is not None else None
    four_timing = getattr(four_gait, "timing", None) if four_gait is not None else None
    duration = 0.0
    fps = 0.0
    if spec_timing is not None:
        duration = float(spec_timing.duration_s)
        fps = float(spec_timing.fps)
    elif four_timing is not None:
        duration = float(four_timing.duration_s)
        fps = float(four_timing.fps)
    elif four_gait is not None:
        times = getattr(four_gait, "times_s", None)
        if times is not None and getattr(times, "size", 0) >= 2:
            duration = float(times[-1] - times[0])
        fps = float(getattr(four_gait, "fps", 0.0) or 0.0)
    channel_acf = None
    if spectral is not None:
        getter = getattr(spectral, "primary_autocorr", None)
        channel_acf = getter() if callable(getter) else None
    usable = None
    if cycles is not None:
        biomech_usable = getattr(cycles, "usable_cycle_count", None)
        if biomech_usable is None:
            metrics = getattr(cycles, "metrics", None)
            biomech_usable = getattr(metrics, "gait_cycle_count", None)
        if biomech_usable is not None:
            usable = int(biomech_usable)
    if spectral is None and four_gait is None and cycles is None:
        return None
    return compose_gait_timing(
        duration_s=duration,
        fps=fps,
        channel_acf=channel_acf,
        session_period_s=getattr(four_gait, "gait_period_s", None) if four_gait is not None else (
            spec_timing.session_stride.value_s if spec_timing is not None else None
        ),
        session_reason=str(getattr(four_gait, "gait_period_reason", "") or "") if four_gait is not None else "",
        session_channel_ids=tuple(
            getattr(four_gait, "gait_period_channel_ids", ()) or ()
        ) if four_gait is not None else (),
        event_stride_s=getattr(four_gait, "event_stride_period_s", None) if four_gait is not None else (
            spec_timing.event_stride.value_s if spec_timing is not None else None
        ),
        event_stride_n=int(getattr(four_gait, "event_stride_n", 0) or 0) if four_gait is not None else 0,
        acf_event_note=str(getattr(four_gait, "acf_event_note", "") or "") if four_gait is not None else "",
        cycles=cycles,
        usable_cycles=usable,
    )


def refresh_shared_gait_timing(gui: Any) -> GaitTimingResult | None:
    """Store one timing object on the GUI for every tab to read."""
    timing = merge_gait_timing(
        getattr(gui, "_spectral_report", None),
        getattr(gui, "_four_gait_report", None),
        getattr(gui, "_gait_cycle", None),
    )
    gui._gait_timing = timing
    if timing is not None:
        spectral = getattr(gui, "_spectral_report", None)
        if spectral is not None:
            spectral.timing = timing
        four = getattr(gui, "_four_gait_report", None)
        if four is not None:
            four.timing = timing
    return timing
