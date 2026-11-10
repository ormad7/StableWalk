"""Phase-timing gait metrics as Primary / Secondary / Technical KPI cards."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.analysis.biomechanical.advanced_gait_metrics import AdvancedGaitMetrics
from stablewalk.analysis.biomechanical.types import MetricWithConfidence
from stablewalk.analysis.biomechanical.walking_speed import (
    is_reportable_walking_speed,
)
from stablewalk.analysis.gait_cycle_analysis import GaitTemporalMetrics
from stablewalk.ui.metric_tooltips import get_metric_tooltip
from stablewalk.ui.scientific_labels import (
    LABEL_CADENCE,
    LABEL_WALKING_SPEED,
    UNIT_CADENCE,
    format_walking_speed_value_numeric,
)
from stablewalk.ui.summary_metric_style import interpret_summary_metric
from stablewalk.ui.theme import DASHBOARD_CARD_PAD, PANEL
from stablewalk.ui.tk.kpi_cards import (
    create_kpi_card,
    create_metric_group_label,
    create_technical_details_panel,
    update_kpi_card,
)

# (key, title, kind, scale, data_tier, visual_tier)
_METRIC_SPECS: tuple[tuple[str, str, str, float, str, str], ...] = (
    ("cadence", LABEL_CADENCE, "cadence", 180.0, "calculated", "primary"),
    ("stance_pct", "Stance", "percent", 100.0, "calculated", "primary"),
    ("walking_speed", LABEL_WALKING_SPEED, "speed", 2.0, "estimated", "secondary"),
    ("swing_pct", "Swing", "percent", 100.0, "calculated", "secondary"),
    ("double_support_pct", "Double Support", "percent", 100.0, "calculated", "secondary"),
    ("single_support_pct", "Single Support", "percent", 100.0, "calculated", "secondary"),
    ("step_length", "Step Length", "meters", 1.2, "estimated", "technical"),
    ("step_width", "Step Width", "meters", 0.35, "estimated", "technical"),
    ("step_time", "Step Time", "seconds", 1.4, "calculated", "technical"),
    ("stride_time", "Stride Time", "seconds", 2.8, "calculated", "technical"),
    ("contact_duration", "Contact Time", "seconds", 1.2, "calculated", "technical"),
)

_GROUP_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Primary", ("cadence", "stance_pct")),
    (
        "Secondary",
        ("walking_speed", "swing_pct", "double_support_pct", "single_support_pct"),
    ),
    (
        "Technical",
        (
            "step_length",
            "step_width",
            "step_time",
            "stride_time",
            "contact_duration",
        ),
    ),
)


def _card_title(text: str) -> str:
    from stablewalk.ui.theme import format_labelframe_title

    return format_labelframe_title(text)


def build_motion_temporal_metrics_panel(gui, parent: tk.Misc) -> ttk.LabelFrame:
    """Install phase-timing KPIs (Primary / Secondary / Technical Details)."""
    panel = ttk.LabelFrame(
        parent,
        text=_card_title("Phase Timing & Stability Drivers"),
        style="Card.TLabelframe",
        padding=DASHBOARD_CARD_PAD,
    )
    panel.columnconfigure(0, weight=1)

    from stablewalk.ui.tk.tab_grid import grid_fill, stretch_columns, stretch_rows

    host = tk.Frame(panel, bg=PANEL, highlightthickness=0)
    stretch_columns(panel)
    stretch_rows(panel)
    grid_fill(host)
    stretch_columns(host)

    gui._motion_metrics_host = host
    gui._motion_kpi_cards = {}
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
    specs_by_key = {row[0]: row for row in _METRIC_SPECS}

    details = create_technical_details_panel(
        host,
        title="Technical details",
        bg=PANEL,
        start_expanded=False,
    )
    gui._motion_technical_details = details

    for group_title, keys in _GROUP_ORDER:
        if group_title == "Technical":
            group_parent = details.content
            create_metric_group_label(group_parent, group_title, bg=PANEL)
        else:
            group_parent = host
            create_metric_group_label(group_parent, group_title, bg=PANEL)

        for key in keys:
            _key, title, kind, _scale, data_tier, visual_tier = specs_by_key[key]
            tip = get_metric_tooltip(tip_map.get(key, "")) or f"{title} — updates after gait analysis"
            show_bar = kind in ("percent", "cadence", "speed") and visual_tier != "technical"
            kpi = create_kpi_card(
                group_parent,
                key=key,
                title=title,
                tooltip=tip,
                compact=visual_tier != "primary",
                show_bar=show_bar,
                fill=False,
                tier=visual_tier,
            )
            gui._motion_kpi_cards[key] = kpi
            gui._motion_metric_values[key] = kpi.value_lbl
            gui._motion_metric_fills[key] = kpi.bar_fill or kpi.accent
            gui._motion_metric_tiers[key] = kpi.quality_lbl
            if visual_tier != "technical":
                kpi.quality_lbl.configure(text=data_tier.replace("_", " ").title())

    # Keep Details toggle after Primary/Secondary groups.
    details.frame.pack_forget()
    details.frame.pack(fill=tk.X, pady=(8, 0))

    gui._motion_temporal_metrics_panel = panel
    return panel


def _metric_value(
    metric: MetricWithConfidence | None,
) -> float | None:
    if metric is None or metric.value is None:
        return None
    return float(metric.value)


def _unavailable_with_confidence(metric: MetricWithConfidence | None) -> str:
    """Show N/A with reason instead of inventing a number when gated."""
    from stablewalk.ui.value_provenance import format_na

    if metric is None:
        return format_na("metric not computed")
    conf = float(getattr(metric, "confidence", 0.0) or 0.0)
    note = str(getattr(metric, "note", "") or "").strip()
    if note:
        # Prefer analysis note; avoid inventing clinical values.
        if note.upper().startswith("N/A"):
            return format_na(note[3:].lstrip(" —:-") or "unavailable", confidence=conf or None)
        return format_na(note, confidence=conf or None)
    if conf > 0.0:
        return format_na("value gated", confidence=conf)
    return format_na("value not available")


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
    if key in ("step_length", "step_width", "stride_length"):
        return f"{value * 100.0:.0f}", "cm"
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
        # Frontal monocular contact stretches these Γאפ use a wide band.
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
        # Total DS Γיט 20%; ipsilateral SS Γיט 40% of the gait cycle.
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
        data["step_length"] = _metric_value(gait_metrics.step_length)
        data["step_width"] = _metric_value(gait_metrics.step_width)
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
    scales = {key: scale for key, _t, _k, scale, _data, _tier in _METRIC_SPECS}
    view_type = _session_view_type(gui)
    metric_objs = {
        "stance_pct": getattr(gait_metrics, "stance_pct", None) if gait_metrics else None,
        "swing_pct": getattr(gait_metrics, "swing_pct", None) if gait_metrics else None,
        "double_support_pct": getattr(gait_metrics, "double_support_pct", None)
        if gait_metrics
        else None,
        "single_support_pct": getattr(gait_metrics, "single_support_pct", None)
        if gait_metrics
        else None,
        "step_length": getattr(gait_metrics, "step_length", None) if gait_metrics else None,
        "step_width": getattr(gait_metrics, "step_width", None) if gait_metrics else None,
        "step_time": getattr(gait_metrics, "step_time", None) if gait_metrics else None,
        "stride_time": getattr(gait_metrics, "stride_time", None) if gait_metrics else None,
        "cadence": getattr(gait_metrics, "cadence", None) if gait_metrics else None,
        "walking_speed": ws_metric,
    }

    for key, _title, _kind, scale, _data, _tier in _METRIC_SPECS:
        card = cards.get(key)
        if card is None:
            continue
        raw = data.get(key)
        metric_obj = metric_objs.get(key)
        if raw is None or (
            key == "walking_speed" and not is_reportable_walking_speed(ws_metric)
        ):
            if key == "walking_speed":
                from stablewalk.analysis.biomechanical.walking_speed import (
                    format_walking_speed_display,
                )

                update_kpi_card(
                    card,
                    value=format_walking_speed_display(ws_metric),
                    available=False,
                    fraction=0.0,
                )
            else:
                update_kpi_card(
                    card,
                    value=_unavailable_with_confidence(metric_obj),
                    available=False,
                    fraction=0.0,
                )
            continue

        value, unit = _split_display(key, raw, walking_speed_metric=ws_metric)
        # Keep value+unit as one readable string so compact cards never hide the number.
        if unit and value not in ("Γאפ", "N/A", ""):
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
