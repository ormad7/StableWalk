"""Motion Analysis — temporal gait metrics as modern KPI cards."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.analysis.biomechanical.advanced_gait_metrics import AdvancedGaitMetrics
from stablewalk.analysis.biomechanical.types import MetricWithConfidence
from stablewalk.analysis.biomechanical.walking_speed import (
    is_reportable_walking_speed,
)
from stablewalk.analysis.gait_cycle_analysis import GaitTemporalMetrics
from stablewalk.ui.scientific_labels import (
    LABEL_CADENCE,
    LABEL_WALKING_SPEED,
    UNIT_CADENCE,
    format_walking_speed_value_numeric,
)
from stablewalk.ui.summary_metric_style import interpret_summary_metric
from stablewalk.ui.theme import DASHBOARD_CARD_PAD, PANEL
from stablewalk.ui.tk.kpi_cards import create_kpi_card, update_kpi_card
from stablewalk.ui.metric_tooltips import get_metric_tooltip

_METRIC_SPECS: tuple[tuple[str, str, str, float, str], ...] = (
    ("stance_pct", "Stance", "percent", 100.0, "calculated"),
    ("swing_pct", "Swing", "percent", 100.0, "calculated"),
    ("double_support_pct", "Double Support", "percent", 100.0, "calculated"),
    ("single_support_pct", "Single Support (ipsilateral)", "percent", 100.0, "calculated"),
    ("step_time", "Step Time", "seconds", 1.4, "calculated"),
    ("stride_time", "Stride Time", "seconds", 2.8, "calculated"),
    ("contact_duration", "Contact Time", "seconds", 1.2, "calculated"),
    ("cadence", LABEL_CADENCE, "cadence", 180.0, "calculated"),
    ("walking_speed", LABEL_WALKING_SPEED, "speed", 2.0, "estimated"),
)


def _card_title(text: str) -> str:
    return f"  {text}  "


def build_motion_temporal_metrics_panel(gui, parent: tk.Misc) -> ttk.LabelFrame:
    """Install the right-side temporal gait metrics panel on Motion Analysis."""
    panel = ttk.LabelFrame(
        parent,
        text=_card_title("Temporal Gait Metrics"),
        style="Card.TLabelframe",
        padding=DASHBOARD_CARD_PAD,
    )
    panel.columnconfigure(0, weight=1)

    host = tk.Frame(panel, bg=PANEL, highlightthickness=0)
    host.pack(fill=tk.BOTH, expand=True)
    host.columnconfigure(0, weight=1)

    gui._motion_metrics_host = host
    gui._motion_kpi_cards = {}
    # Compatibility shims for older update helpers / tests.
    gui._motion_metric_values = {}
    gui._motion_metric_fills = {}
    gui._motion_metric_tiers = {}

    tip_map = {
        "cadence": "cadence",
        "walking_speed": "walking_speed",
        "stance_pct": "gait_cycle",
        "swing_pct": "gait_cycle",
        "double_support_pct": "gait_cycle",
        "single_support_pct": "gait_cycle",
        "step_time": "gait_cycle",
        "stride_time": "gait_cycle",
        "contact_duration": "heel_strike",
    }

    for key, title, _kind, _scale, tier in _METRIC_SPECS:
        tip = get_metric_tooltip(tip_map.get(key, "")) or f"{title} — updates after gait analysis"
        show_bar = _kind in ("percent", "cadence", "speed")
        kpi = create_kpi_card(
            host,
            key=key,
            title=title,
            tooltip=tip,
            compact=True,
            show_bar=show_bar,
            fill=False,
        )
        gui._motion_kpi_cards[key] = kpi
        gui._motion_metric_values[key] = kpi.value_lbl
        gui._motion_metric_fills[key] = kpi.bar_fill or kpi.accent
        gui._motion_metric_tiers[key] = kpi.quality_lbl
        # Seed quality label with data tier until live quality is known.
        kpi.quality_lbl.configure(text=tier.replace("_", " ").title())

    gui._motion_temporal_metrics_panel = panel
    return panel


def _metric_value(
    metric: MetricWithConfidence | None,
) -> float | None:
    if metric is None or metric.value is None:
        return None
    return float(metric.value)


def _split_display(key: str, value: float, *, walking_speed_metric=None) -> tuple[str, str]:
    if key.endswith("_pct") or key in ("double_support_pct", "single_support_pct"):
        return f"{value:.0f}", "%"
    if key == "cadence":
        return f"{value:.0f}", UNIT_CADENCE
    if key == "walking_speed":
        if walking_speed_metric is not None and walking_speed_metric.value is not None:
            text = format_walking_speed_value_numeric(walking_speed_metric.value)
            parts = text.split(" ", 1)
            if len(parts) == 2:
                return parts[0], parts[1]
            return text, ""
        return f"{value:.2f}", "m/s"
    if key in ("step_time", "stride_time", "contact_duration"):
        return f"{value:.2f}", "s"
    return f"{value:.2f}", ""


def _session_view_type(gui) -> str | None:
    biomech = getattr(gui, "_biomech", None)
    return getattr(biomech, "view_type", None) if biomech is not None else None


def _quality_for_motion(key: str, value: float | None, *, view_type: str | None = None) -> str:
    if value is None:
        return "unavailable"
    from types import SimpleNamespace

    from stablewalk.ui.summary_metric_style import (
        is_view_limited_for_quality,
        soften_view_limited_level,
    )

    view_limited = is_view_limited_for_quality(view_type)

    if key == "cadence":
        field = SimpleNamespace(available=True, value=f"{value:.0f} spm", tier="calculated")
        level = interpret_summary_metric("cadence", field, view_type=view_type)  # type: ignore[arg-type]
        return soften_view_limited_level(level, view_limited=view_limited)
    if key == "walking_speed":
        field = SimpleNamespace(available=True, value=f"{value:.2f} m/s", tier="estimated")
        return interpret_summary_metric("walking_speed", field, view_type=view_type)  # type: ignore[arg-type]
    if key in ("stance_pct", "swing_pct"):
        # Typical adult walking: stance ~60%, swing ~40%.
        # Frontal monocular contact stretches these — use a wide band.
        target = 60.0 if key == "stance_pct" else 40.0
        delta = abs(value - target)
        normal_max = 14.0 if view_limited else 10.0
        borderline_max = 28.0 if view_limited else 20.0
        if delta <= normal_max:
            return "normal"
        if delta <= borderline_max:
            return "borderline"
        return soften_view_limited_level("abnormal", view_limited=view_limited)
    if key in ("double_support_pct", "single_support_pct"):
        # Total DS ≈ 20%; ipsilateral SS ≈ 40% of the gait cycle.
        target = 20.0 if key == "double_support_pct" else 40.0
        delta = abs(value - target)
        normal_max = 14.0 if view_limited else 10.0
        borderline_max = 30.0 if view_limited else 18.0
        if delta <= normal_max:
            return "normal"
        if delta <= borderline_max:
            return "borderline"
        return soften_view_limited_level("abnormal", view_limited=view_limited)
    return "neutral"


def _collect_metric_data(
    gait_metrics: AdvancedGaitMetrics | None,
    temporal: GaitTemporalMetrics | None,
) -> dict[str, float | None]:
    data: dict[str, float | None] = {
        key: None for key, *_ in _METRIC_SPECS
    }
    if gait_metrics is not None:
        data["stance_pct"] = _metric_value(gait_metrics.stance_pct)
        data["swing_pct"] = _metric_value(gait_metrics.swing_pct)
        data["double_support_pct"] = _metric_value(gait_metrics.double_support_pct)
        data["single_support_pct"] = _metric_value(gait_metrics.single_support_pct)
        data["step_time"] = _metric_value(gait_metrics.step_time)
        data["stride_time"] = _metric_value(gait_metrics.stride_time)
        data["cadence"] = _metric_value(gait_metrics.cadence)
        if is_reportable_walking_speed(gait_metrics.walking_speed):
            data["walking_speed"] = _metric_value(gait_metrics.walking_speed)

    if temporal is not None and temporal.metrics_reliable:
        if data["double_support_pct"] is None:
            data["double_support_pct"] = temporal.double_support_pct
        if data["single_support_pct"] is None:
            data["single_support_pct"] = temporal.single_support_pct
        if data["step_time"] is None:
            data["step_time"] = temporal.step_time_s
        if data["stride_time"] is None:
            data["stride_time"] = temporal.stride_time_s
        if data["cadence"] is None:
            data["cadence"] = temporal.cadence_steps_per_min
        stance_values = [
            value
            for value in (temporal.left_stance_pct, temporal.right_stance_pct)
            if value is not None
        ]
        swing_values = [
            value
            for value in (temporal.left_swing_pct, temporal.right_swing_pct)
            if value is not None
        ]
        if data["stance_pct"] is None and stance_values:
            data["stance_pct"] = sum(stance_values) / len(stance_values)
        if data["swing_pct"] is None and swing_values:
            data["swing_pct"] = sum(swing_values) / len(swing_values)
        data["contact_duration"] = temporal.average_stance_duration_s

    return data


def update_motion_temporal_metrics_panel(gui) -> None:
    """Refresh KPI cards from the current gait analysis bundle."""
    cards = getattr(gui, "_motion_kpi_cards", None) or {}
    if not cards:
        return

    biomech = getattr(gui, "_biomech_analysis", None)
    gait_metrics = biomech.gait_metrics if biomech is not None else None
    cycles = getattr(gui, "_gait_cycle", None)
    temporal = cycles.metrics if cycles is not None else None
    ws_metric = gait_metrics.walking_speed if gait_metrics is not None else None

    data = _collect_metric_data(gait_metrics, temporal)
    scales = {key: scale for key, _t, _k, scale, _tier in _METRIC_SPECS}
    view_type = _session_view_type(gui)

    for key, _title, _kind, scale, _tier in _METRIC_SPECS:
        card = cards.get(key)
        if card is None:
            continue
        raw = data.get(key)
        if raw is None or (
            key == "walking_speed" and not is_reportable_walking_speed(ws_metric)
        ):
            update_kpi_card(card, value="—", available=False, fraction=0.0)
            continue

        value, unit = _split_display(key, raw, walking_speed_metric=ws_metric)
        # Keep value+unit as one readable string so compact cards never hide the number.
        if unit and value not in ("—", "N/A", ""):
            display = f"{value} {unit}".strip()
            value, unit = display, ""
        fraction = raw / scale if scale > 0 else 0.0
        if key.endswith("_pct") or "support" in key:
            fraction = raw / 100.0
        update_kpi_card(
            card,
            value=value,
            unit=unit,
            quality=_quality_for_motion(key, raw, view_type=view_type),  # type: ignore[arg-type]
            available=True,
            fraction=fraction,
            numeric=float(raw),
        )


__all__ = [
    "build_motion_temporal_metrics_panel",
    "update_motion_temporal_metrics_panel",
]
