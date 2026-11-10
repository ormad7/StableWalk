"""Biomechanics tab layout for the StableWalk dashboard."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from stablewalk.ui.theme import (
    FONT_KPI_TITLE,
    FONT_UI,
    FONT_UI_SM,
    FONT_UI_XS,
    KPI_CARD_GAP,
    MUTED,
    MUTED_DIM,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    TEXT_SECONDARY,
    bind_responsive_wrap,
)
from stablewalk.ui.scientific_labels import (
    DISCLAIMER_MONOCULAR,
    DISCLAIMER_RESEARCH,
    LABEL_CADENCE,
    LABEL_CADENCE_SHORT,
    LABEL_COMPOSITE_GAIT_QUALITY,
    LABEL_GAIT_QUALITY,
    LABEL_GAIT_SYMMETRY,
    LABEL_JOINT_ROM,
    LABEL_STABILITY_MARGIN,
    LABEL_STABILITY_STATE,
    LABEL_VIDEO_QUALITY,
    LABEL_WALKING_SPEED,
    UNITS_COORD_LEGEND,
)
from stablewalk.ui.tk.dashboard_layout import (
    _card_title,
)
from stablewalk.ui.metric_tooltips import attach_metric_tooltip, get_metric_tooltip
from stablewalk.ui.summary_metric_style import interpret_summary_metric
from stablewalk.ui.tk.analysis_split import MIN_H_CHART, MIN_W_CHART

from stablewalk.ui.tk.kpi_cards import (
    create_kpi_card,
    create_kpi_grid_card,
    create_metric_group_label,
    parse_numeric,
    update_kpi_card,
)

# Tight pad so the graph can claim unused chrome around the plot.
_ANALYSIS_PANEL_PAD = (PAD_SM, PAD_SM)
# Side rails: compact Key Findings / readable interpretation; graph keeps leftover.
BIOMECH_FINDINGS_MIN_W = 188
BIOMECH_FINDINGS_MAX_W = 220
BIOMECH_FINDINGS_WIDTH = 204
BIOMECH_CLINICAL_MIN_W = 300
BIOMECH_CLINICAL_MAX_W = 320
BIOMECH_CLINICAL_WIDTH = 320
# Collapsed rail floors so the graph stays usable on narrow desktops.
BIOMECH_FINDINGS_COLLAPSED_W = 176
BIOMECH_CLINICAL_COLLAPSED_W = 220
# User-collapsed rail on a wide layout: keep the ▶ control, give leftover to the graph.
BIOMECH_RAIL_HEADER_ONLY_W = 40
# Grid floor for the chart column — figure may still prefer a taller natural size.
BIOMECH_CHART_GRID_MIN_H = MIN_H_CHART
BIOMECH_CHART_COL_MIN_W = MIN_W_CHART
# Collapse side rails before the three columns overflow a typical laptop width.
BIOMECH_STACK_BELOW_W = 1180
# Stacked text rows: share of body height, hard cap so they do not dominate.
BIOMECH_STACK_SIDE_SHARE = 0.28
BIOMECH_STACK_SIDE_MAX_H = 200
BIOMECH_STACK_SIDE_MIN_H = 140
# Inner text inset; wraplength = rail width − this margin.
_BIOMECH_CLINICAL_WRAP_MARGIN = 36
BIOMECH_INTERPRETATION_EMPTY = (
    "No session interpretation yet. Analyze a walking video to fill this panel "
    "for the selected graph."
)
# Wide, tall figure; `_fit_figure_to_host` stretches to the chart host.
BIOMECH_CHART_FIGSIZE = (14.0, 9.0)

# (key, caption, metric-tooltip key) for the compact summary strip above the graphs.
_BIOMECH_SUMMARY_FIELDS: tuple[tuple[str, str, str | None], ...] = (
    ("heel_strikes", "Heel Strikes (estimated)", None),
    ("cadence", LABEL_CADENCE_SHORT, "cadence"),
    ("stance", "Stance % (calculated)", None),
    ("swing", "Swing % (calculated)", None),
    ("gait_quality", LABEL_GAIT_QUALITY, "gait_quality"),
    ("stability", LABEL_STABILITY_STATE, "stability"),
)

_BIOMECH_SIDEBAR_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("gait_quality", LABEL_COMPOSITE_GAIT_QUALITY, "gait_quality", "lbl_biomech_gait_quality"),
    ("stability_state", LABEL_STABILITY_STATE, "stability", "lbl_biomech_stability_state"),
    ("symmetry", LABEL_GAIT_SYMMETRY, "symmetry", "lbl_biomech_symmetry"),
    ("stability_margin", LABEL_STABILITY_MARGIN, "stability", "lbl_biomech_stability_margin"),
    ("cadence", LABEL_CADENCE, "cadence", "lbl_biomech_cadence"),
    ("walking_speed", LABEL_WALKING_SPEED, "walking_speed", "lbl_biomech_walking_speed"),
)

# Key Findings card key → SessionMetricView snapshot key (shared with Overview/Results).
_KEY_FINDINGS_SNAPSHOT_KEYS: dict[str, str] = {
    "gait_quality": "gait_quality_composite",
    "stability_state": "stability_frame_state",
    "symmetry": "symmetry",
    "stability_margin": "stable_pct",
    "cadence": "cadence",
}

# Engine frame labels on the KPI — do not rename Stable to clinical Acceptable.
_STABILITY_SHORT = {
    "Stable": "Stable",
    "Acceptable": "Heuristic",
    "Reduced Stability": "Reduced",
    "Borderline": "Borderline",
    "Unstable": "Unstable",
    "Reduced": "Reduced",
}


def biomech_three_col_min_width() -> int:
    """Narrowest workspace that can keep Key Findings at its readable floor."""
    return (
        int(BIOMECH_FINDINGS_MIN_W)
        + int(BIOMECH_CHART_COL_MIN_W)
        + int(BIOMECH_CLINICAL_MIN_W)
        + int(PAD_XS) * 4
    )


def biomech_balanced_column_widths(total_w: int) -> tuple[int, int, int]:
    """Key findings | chart | interpretation pixel widths.

    Chart keeps leftover space and at least ``BIOMECH_CHART_COL_MIN_W`` when
    the window can support it. Side rails stay between min and max so copy
    does not collapse or stretch into a tall skinny column.
    """
    pad = int(PAD_XS) * 2
    usable = max(1, int(total_w) - pad)
    findings = int(usable * 0.15)
    clinical = int(usable * 0.28)
    findings = max(BIOMECH_FINDINGS_MIN_W, min(BIOMECH_FINDINGS_MAX_W, findings))
    clinical = max(BIOMECH_CLINICAL_MIN_W, min(BIOMECH_CLINICAL_MAX_W, clinical))
    chart = usable - findings - clinical
    if chart < BIOMECH_CHART_COL_MIN_W:
        deficit = BIOMECH_CHART_COL_MIN_W - chart
        # Never shrink Key Findings below its readable floor; take leftover
        # from the clinical rail only, then let the caller stack instead.
        c_flex = max(0, clinical - BIOMECH_CLINICAL_MIN_W)
        take_c = min(c_flex, deficit)
        clinical -= take_c
        chart = usable - findings - clinical
    return findings, max(1, chart), clinical


def _install_biomech_overflow_scroll(host: tk.Misc, inner: tk.Frame) -> None:
    """Vertical overflow only — keeps side text from stretching the page."""
    host.columnconfigure(0, weight=1)
    host.columnconfigure(1, weight=0)
    host.rowconfigure(0, weight=1)
    # width/height=1 so the default ~378px Canvas does not inflate the rail
    # and steal graph width (that leftover band reads as empty left space).
    canvas = tk.Canvas(
        host, bg=PANEL, highlightthickness=0, bd=0, width=1, height=1
    )
    vsb = ttk.Scrollbar(host, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    vsb.grid_remove()
    win = canvas.create_window((0, 0), window=inner, anchor="nw")
    scroll_active = {"value": False}
    sync_busy = {"value": False}
    try:
        from stablewalk.ui.tk.clip_viewport import ensure_widget_fills_host

        ensure_widget_fills_host(canvas)
    except Exception:
        pass

    def _sync(_event: object | None = None) -> None:
        if sync_busy["value"]:
            return
        sync_busy["value"] = True
        try:
            from stablewalk.ui.tk.scroll_shell import apply_overflow_scroll_geometry

            host_w = max(1, int(host.winfo_width() or 0))
            host_h = max(1, int(host.winfo_height() or 0))
            view_w = max(1, int(canvas.winfo_width() or 0))
            view_h = max(1, int(canvas.winfo_height() or 0))
            # Until the rail is mapped, the canvas still reports 1×1. Do not
            # collapse Key Findings to one pixel or fake a scrollbar.
            if host_w >= 40 and view_w < 40:
                canvas.configure(width=max(1, host_w - 4))
                view_w = host_w
            if host_h >= 40 and view_h < 40:
                canvas.configure(height=max(1, host_h - 4))
                view_h = host_h
            if view_w < 40 or view_h < 40:
                vsb.grid_remove()
                scroll_active["value"] = False
                return
            try:
                canvas.itemconfigure(win, width=view_w)
                inner.update_idletasks()
            except tk.TclError:
                pass
            inner_h = max(1, int(inner.winfo_reqheight() or 0))
            scroll_active["value"] = apply_overflow_scroll_geometry(
                canvas, win, vsb, content_height=inner_h
            )
        except tk.TclError:
            pass
        finally:
            sync_busy["value"] = False

    def _wheel(event: tk.Event) -> str | None:
        if not scroll_active["value"] or not event.delta:
            return None
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    inner.bind("<Configure>", _sync, add="+")
    canvas.bind("<Configure>", _sync, add="+")
    canvas.bind("<MouseWheel>", _wheel, add="+")
    inner.bind("<MouseWheel>", _wheel, add="+")
    host._overflow_canvas = canvas
    host._overflow_vsb = vsb
    host._overflow_sync = _sync


def _quality_for_key(
    key: str,
    value_text: str,
    *,
    available: bool,
    view_type: str | None = None,
    stability_valid_ratio: float | None = None,
) -> str:
    if not available:
        return "unavailable"
    from types import SimpleNamespace

    from stablewalk.ui.summary_metric_style import (
        is_view_limited_for_quality,
        soften_view_limited_level,
    )

    view_limited = is_view_limited_for_quality(view_type)

    if key in ("heel_strikes",):
        return "neutral" if available else "unavailable"
    if key in ("stance", "swing"):
        num = parse_numeric(value_text)
        if num is None:
            return "neutral"
        target = 60.0 if key == "stance" else 40.0
        delta = abs(num - target)
        normal_max = 14.0 if view_limited else 10.0
        borderline_max = 28.0 if view_limited else 20.0
        if delta <= normal_max:
            return "normal"
        if delta <= borderline_max:
            return "borderline"
        return soften_view_limited_level("abnormal", view_limited=view_limited)
    if key == "stability_state":
        lower = value_text.lower()
        if "unavailable" in lower:
            return "unavailable"
        if "unstable" in lower:
            return soften_view_limited_level("abnormal", view_limited=view_limited)
        if "reduced" in lower:
            return "borderline"
        if "stable" in lower:
            return "normal"
        return "neutral"

    field = SimpleNamespace(available=True, value=value_text, tier="calculated")
    mapped = {
        "gait_quality": "gait_quality",
        "cadence": "cadence",
        "walking_speed": "walking_speed",
        "symmetry": "symmetry",
        "stability": "stability_margin",
        "stability_margin": "stability_margin",
        "video_quality": "video_quality",
    }.get(key, key)
    return interpret_summary_metric(
        mapped,
        field,  # type: ignore[arg-type]
        view_type=view_type,
        stability_valid_ratio=stability_valid_ratio,
    )


def _build_biomech_summary_card(gui, parent: tk.Misc) -> tk.Frame:
    """At-a-glance strip: Primary large, Secondary compact, Technical in Details."""
    from stablewalk.ui.tk.kpi_cards import create_technical_details_panel

    card = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    card.columnconfigure(0, weight=1)
    card.columnconfigure(1, weight=1)
    card.columnconfigure(2, weight=1)
    card.columnconfigure(3, weight=1)

    tiers = {
        "gait_quality": "primary",
        "stability": "primary",
        "cadence": "secondary",
        "heel_strikes": "technical",
        "stance": "technical",
        "swing": "technical",
    }

    gui._biomech_summary_kpis = {}
    gui._biomech_summary_value_labels = {}

    primary = [f for f in _BIOMECH_SUMMARY_FIELDS if tiers[f[0]] == "primary"]
    for col, (key, caption, tip) in enumerate(primary):
        tip_text = get_metric_tooltip(tip) if tip else f"{caption} — live session metric"
        kpi = create_kpi_grid_card(
            card,
            key=key,
            title=caption,
            tooltip=tip_text,
            compact=False,
            show_bar=key in ("gait_quality",),
            tier="primary",
        )
        kpi.frame.grid(
            row=0,
            column=col * 2,
            columnspan=2,
            sticky="nsew",
            padx=(0 if col == 0 else KPI_CARD_GAP, 0),
            pady=(0, KPI_CARD_GAP),
        )
        gui._biomech_summary_kpis[key] = kpi
        gui._biomech_summary_value_labels[key] = kpi.value_lbl

    secondary = [f for f in _BIOMECH_SUMMARY_FIELDS if tiers[f[0]] == "secondary"]
    for col, (key, caption, tip) in enumerate(secondary):
        tip_text = get_metric_tooltip(tip) if tip else f"{caption} — live session metric"
        kpi = create_kpi_grid_card(
            card,
            key=key,
            title=caption,
            tooltip=tip_text,
            compact=True,
            show_bar=False,
            tier="secondary",
        )
        kpi.frame.grid(
            row=1,
            column=col,
            sticky="nsew",
            padx=(0 if col == 0 else PAD_XS, 0),
            pady=(0, PAD_XS),
        )
        gui._biomech_summary_kpis[key] = kpi
        gui._biomech_summary_value_labels[key] = kpi.value_lbl

    details = create_technical_details_panel(
        card,
        title="Technical details",
        bg=PANEL,
        start_expanded=False,
        pack=False,
    )
    details.frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(PAD_XS, 0))
    details.content.columnconfigure(0, weight=1)
    details.content.columnconfigure(1, weight=1)
    details.content.columnconfigure(2, weight=1)
    gui._biomech_summary_details = details

    technical = [f for f in _BIOMECH_SUMMARY_FIELDS if tiers[f[0]] == "technical"]
    for col, (key, caption, tip) in enumerate(technical):
        tip_text = get_metric_tooltip(tip) if tip else f"{caption} — live session metric"
        kpi = create_kpi_grid_card(
            details.content,
            key=key,
            title=caption,
            tooltip=tip_text,
            compact=True,
            show_bar=False,
            tier="technical",
        )
        kpi.frame.grid(
            row=0,
            column=col,
            sticky="nsew",
            padx=(0 if col == 0 else PAD_XS, 0),
        )
        gui._biomech_summary_kpis[key] = kpi
        gui._biomech_summary_value_labels[key] = kpi.value_lbl

    return card


def _metric_value_exists(value: object) -> bool:
    """True when a computed numeric/categorical value is present (not missing)."""
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip()
        if not text or text in ("—", "-", "N/A", "Unavailable", "Not available"):
            return False
        if text.lower().startswith("n/a"):
            return False
        return True
    if isinstance(value, (int, float)):
        try:
            import math

            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False
    return True


def bind_biomech_key_findings_from_session_view(
    gui,
    session_view,
    *,
    ba=None,
    view_type: str | None = None,
    stability_valid_ratio: float | None = None,
) -> None:
    """Bind Key Findings KPI cards to computed SessionMetricView snapshots.

    Shows the numeric/categorical value whenever it exists. Uses ``N/A`` only
    when the snapshot has no computed value (data truly missing).
    """
    from stablewalk.metrics.session_view import get_snapshot
    from stablewalk.metrics.status import ValidityState
    from stablewalk.ui.tk.kpi_cards import split_value_unit
    from stablewalk.ui.value_provenance import NA_DISPLAY

    kpis = getattr(gui, "_biomech_kpi_cards", None) or {}
    if not kpis:
        return

    def _paint(
        key: str,
        *,
        value: str,
        unit: str = "",
        available: bool,
        fraction: float | None = None,
        numeric: float | None = None,
        tip: str | None = None,
    ) -> None:
        card = kpis.get(key)
        if card is None:
            return
        if not available:
            value = NA_DISPLAY
            unit = ""
            if not (tip or "").strip():
                tip = {
                    "gait_quality": (
                        "N/A — Composite Gait Quality was not scored because no "
                        "pose-derived components were available. Not a 0/100 score."
                    ),
                    "symmetry": (
                        "N/A — overall symmetry needs at least two valid bilateral "
                        "components. Not a 0% symmetry finding."
                    ),
                    "stability_state": (
                        "N/A — Normal/Stable/Unstable classifications are not "
                        "supported for monocular COM–BoS. Physical stability is "
                        "unreportable."
                    ),
                    "stability_margin": (
                        "N/A — physical COM–BoS margin is unreportable from this "
                        "monocular method. The Balance plot is an unvalidated preview."
                    ),
                    "cadence": "N/A — cadence was not computed for this session.",
                    "walking_speed": "N/A — walking speed was not estimated.",
                }.get(
                    key,
                    "N/A — this value was not computed. It is not a score of 0.",
                )
        display = f"{value}{(' ' + unit) if unit and available else ''}".strip()
        update_kpi_card(
            card,
            value=value,
            unit=unit if available else "",
            quality=_quality_for_key(
                key,
                display,
                available=available,
                view_type=view_type,
                stability_valid_ratio=stability_valid_ratio,
            ),  # type: ignore[arg-type]
            available=available,
            fraction=fraction if available else None,
            numeric=numeric if available else None,
            tooltip=tip,
        )

    if session_view is None:
        for key in kpis:
            if key == "video_quality":
                continue
            _paint(key, value=NA_DISPLAY, available=False)
    else:
        # Snapshot-backed cards (shared computed metrics).
        unit_overrides = {
            "gait_quality": "/100",
            "symmetry": "%",
            "stability_margin": "% stable",
            "cadence": "steps/min",
        }
        for card_key, snap_key in _KEY_FINDINGS_SNAPSHOT_KEYS.items():
            snap = get_snapshot(session_view.snapshots, snap_key)
            if snap is None:
                _paint(card_key, value=NA_DISPLAY, available=False)
                continue
            has_value = _metric_value_exists(snap.value) or (
                snap.validity == ValidityState.VALID and _metric_value_exists(snap.display_value)
            )
            # Prefer formatted display_value for categoricals; numeric for gauges.
            if has_value:
                raw = snap.value
                if isinstance(raw, (int, float)):
                    value = snap.display_value
                    unit = unit_overrides.get(card_key, snap.unit or "")
                    numeric = float(raw)
                    fraction = None
                    if card_key in ("gait_quality", "symmetry", "stability_margin"):
                        fraction = max(0.0, min(1.0, numeric / 100.0))
                    elif card_key == "cadence":
                        fraction = min(1.0, numeric / 180.0)
                    tip = snap.status.user_explanation or None
                    if card_key in ("stability_margin", "stability_state"):
                        extra = " ".join(n for n in (snap.notes or ()) if n)
                        if extra:
                            tip = f"{tip} {extra}".strip() if tip else extra
                    _paint(
                        card_key,
                        value=value,
                        unit=unit,
                        available=True,
                        fraction=fraction,
                        numeric=numeric,
                        tip=tip,
                    )
                else:
                    # Categorical (e.g. Acceptable / Borderline stability state).
                    text = str(snap.display_value)
                    tip = str(raw) if raw is not None else text
                    if card_key in ("stability_margin", "stability_state"):
                        extra = " ".join(n for n in (snap.notes or ()) if n)
                        if extra:
                            tip = f"{tip} {extra}".strip()
                    _paint(
                        card_key,
                        value=text,
                        available=True,
                        tip=tip,
                    )
            else:
                extra = " ".join(n for n in (snap.notes or ()) if n)
                _paint(
                    card_key,
                    value=NA_DISPLAY,
                    available=False,
                    tip=extra or None,
                )

    # Keep Key Findings aligned with Overview when the session view has
    # no numeric snapshot yet (gait_summary / stability result fallback).
    try:
        from stablewalk.ui.overview_clinical_summary import extract_overview_summary_fields

        overview_fields = extract_overview_summary_fields(gui)
    except Exception:
        overview_fields = {}
    overview_aliases = {
        "gait_quality": "gait_quality",
        "stability_state": "stability_state",
        "symmetry": "symmetry",
        "cadence": "cadence",
        "stability_margin": "stability_state",
    }
    for card_key, overview_key in overview_aliases.items():
        card = kpis.get(card_key)
        if card is None:
            continue
        try:
            current = str(card.value_lbl.cget("text") or "")
        except Exception:
            current = ""
        if _metric_value_exists(current) and current not in (NA_DISPLAY, "—"):
            continue
        pair = overview_fields.get(overview_key)
        if not pair:
            continue
        display, tip = pair
        if not _metric_value_exists(display):
            continue
        value, unit = split_value_unit(str(display))
        numeric = None
        fraction = None
        try:
            numeric = float("".join(ch for ch in value if ch.isdigit() or ch in ".-"))
            if card_key in ("gait_quality", "symmetry", "stability_margin", "stability_state"):
                if 0.0 <= numeric <= 100.0 and (
                    "%" in str(display) or "/100" in str(display)
                ):
                    fraction = max(0.0, min(1.0, numeric / 100.0))
        except (TypeError, ValueError):
            numeric = None
        _paint(
            card_key,
            value=value,
            unit=unit,
            available=True,
            numeric=numeric,
            fraction=fraction,
            tip=str(tip) if tip else None,
        )

    # Walking speed — show the computed estimate whenever a numeric value exists.
    ws = None
    gm = getattr(ba, "gait_metrics", None) if ba is not None else None
    if gm is not None:
        ws = getattr(gm, "walking_speed", None)
    speed_val = getattr(ws, "value", None) if ws is not None else None
    if _metric_value_exists(speed_val):
        from stablewalk.analysis.biomechanical.walking_speed import (
            is_plausible_walking_speed,
        )
        from stablewalk.ui.scientific_labels import format_walking_speed_value_numeric

        if is_plausible_walking_speed(float(speed_val)):
            display = format_walking_speed_value_numeric(float(speed_val))
            value, unit = split_value_unit(display)
            _paint(
                "walking_speed",
                value=value,
                unit=unit,
                available=True,
                numeric=float(speed_val),
                tip=display,
            )
        else:
            _paint("walking_speed", value=NA_DISPLAY, available=False)
    else:
        _paint("walking_speed", value=NA_DISPLAY, available=False)


def update_biomechanics_summary_card(gui, ba, gait_cycle, *, frame_index=None) -> None:
    """Refresh the compact summary strip; safe to call every playback frame."""
    kpis = getattr(gui, "_biomech_summary_kpis", None)
    if not kpis:
        return

    biomech = getattr(gui, "_biomech", None)
    view_type = getattr(biomech, "view_type", None) if biomech is not None else None
    sm = getattr(ba, "stability_margin", None) if ba is not None else None
    valid_ratio = getattr(sm, "valid_frame_ratio", None) if sm is not None else None

    def _set(
        key: str,
        value: str,
        *,
        unit: str = "",
        available: bool = True,
        fraction: float | None = None,
        numeric: float | None = None,
        tip: str | None = None,
    ) -> None:
        card = kpis.get(key)
        if card is None:
            return
        display = f"{value}{(' ' + unit) if unit else ''}".strip()
        update_kpi_card(
            card,
            value=value,
            unit=unit,
            quality=_quality_for_key(
                key,
                display,
                available=available,
                view_type=view_type,
                stability_valid_ratio=valid_ratio,
            ),  # type: ignore[arg-type]
            available=available,
            fraction=fraction,
            numeric=numeric if numeric is not None else parse_numeric(value),
            tooltip=tip,
        )

    metrics = getattr(gait_cycle, "metrics", None) if gait_cycle is not None else None
    gm = getattr(ba, "gait_metrics", None) if ba is not None else None

    if ba is None and metrics is None:
        for key in kpis:
            _set(key, "—", available=False)
        return

    if metrics is not None:
        hs = int((metrics.left_heel_strike_count or 0) + (metrics.right_heel_strike_count or 0))
        _set("heel_strikes", str(hs) if hs > 0 else "—", available=hs > 0, numeric=float(hs))
    else:
        _set("heel_strikes", "—", available=False)

    from stablewalk.analysis.gait_cycle_analysis import (
        cadence_is_exploratory,
        summary_cadence_spm,
    )

    cadence = None
    if cadence_is_exploratory(metrics, gait_cycle):
        cadence = None
    else:
        if gm is not None and gm.cadence is not None:
            cadence = gm.cadence.value
        if cadence is None:
            cadence = summary_cadence_spm(metrics, gait_cycle)
    if cadence is not None:
        _set("cadence", f"{cadence:.0f}", unit="steps/min", available=True, numeric=float(cadence))
    else:
        _set("cadence", "N/A", available=False)

    stance = swing = None
    if gm is not None:
        if gm.stance_pct is not None:
            stance = gm.stance_pct.value
        if gm.swing_pct is not None:
            swing = gm.swing_pct.value
    if (stance is None or swing is None) and metrics is not None:
        st = metrics.average_stance_duration_s
        sw = metrics.average_swing_duration_s
        cycle = (st or 0.0) + (sw or 0.0)
        if cycle > 0:
            if stance is None:
                stance = (st or 0.0) / cycle * 100.0
            if swing is None:
                swing = (sw or 0.0) / cycle * 100.0
    if stance is not None:
        _set(
            "stance",
            f"{stance:.0f}",
            unit="%",
            available=True,
            fraction=float(stance) / 100.0,
            numeric=float(stance),
        )
    else:
        _set("stance", "—", available=False)
    if swing is not None:
        _set(
            "swing",
            f"{swing:.0f}",
            unit="%",
            available=True,
            fraction=float(swing) / 100.0,
            numeric=float(swing),
        )
    else:
        _set("swing", "—", available=False)

    gq = getattr(ba, "gait_quality", None) if ba is not None else None
    score = (
        gq.score
        if gq is not None and getattr(gq, "available", True) and gq.score is not None
        else None
    )
    if score is not None:
        _set(
            "gait_quality",
            f"{score:.0f}",
            unit="/100",
            available=True,
            fraction=float(score) / 100.0,
            numeric=float(score),
        )
    else:
        _set("gait_quality", "—", available=False)

    text = "—"
    available = False
    numeric = None
    tip = None
    sm = getattr(ba, "stability_margin", None) if ba is not None else None
    contract = None
    if sm is not None:
        try:
            from stablewalk.analysis.biomechanical.stability_margin import (
                balance_display_contract,
            )

            contract = balance_display_contract(sm)
        except Exception:
            contract = None
    classification_ok = bool(
        contract is not None
        and contract.reportable
        and contract.classification_supported
        and contract.validity == "valid"
    )
    if sm is not None and classification_ok:
        if frame_index is not None:
            for f in sm.per_frame:
                if f.frame_index == frame_index:
                    text = _STABILITY_SHORT.get(f.stability_state, f.stability_state)
                    available = True
                    tip = f.stability_state
                    break
        if text == "—":
            pct = getattr(sm, "stable_pct", None)
            if pct is not None:
                from stablewalk.metrics.status import build_metric_status

                st = build_metric_status("stable_pct", value=float(pct), available=True)
                text = st.display_status.value
                available = True
                numeric = float(pct)
                tip = (
                    f"{pct:.0f}% stable frames (pose-derived heuristic band: "
                    f"{st.display_status.value})"
                )
        try:
            from stablewalk.analysis.biomechanical.stability_margin import (
                format_stability_diagnostics,
            )

            diag = format_stability_diagnostics(sm)
            if diag:
                tip = f"{tip} {diag}".strip() if tip else diag
        except Exception:
            pass
    elif sm is not None:
        from stablewalk.ui.value_provenance import NA_DISPLAY

        text = NA_DISPLAY
        available = False
        tip = (
            contract.reason
            if contract is not None and contract.reason
            else "Physical COM–BoS stability is unreportable from this method."
        )
    _set("stability", text, available=available, numeric=numeric, tip=tip)


# Internal Biomechanics workspace tabs → chart source + series/panel mode.
_BIOMECH_WORKSPACE_TABS: tuple[tuple[str, str, str, str], ...] = (
    # tab_id, label, chart_family ("biomech"|"contact"), series_or_panel
    ("com", "COM", "biomech", "com"),
    ("grf", "Ground Reaction Force", "contact", "vgrf"),
    ("balance", "Balance", "biomech", "stability"),
    ("contact", "Contact", "contact", "contact"),
    ("stability", "Gait Phase", "biomech", "phase"),
)

_BIOMECH_TAB_COPY: dict[str, dict[str, str]] = {
    "com": {
        "interp": (
            "Center of mass height (body-normalized). Smooth oscillation with heel-strike "
            "dips is typical; large vertical excursions or flat segments can indicate "
            "compensatory gait or tracking limits."
        ),
        "stats_hint": "COM height range and mean (body heights).",
    },
    "grf": {
        "interp": (
            "Estimated vertical ground reaction force (virtual GRF in body weights). "
            "Not a force-plate measurement — use for relative left/right loading patterns only."
        ),
        "stats_hint": "Peak and mean vGRF (BW) by side when available.",
    },
    "balance": {
        "interp": (
            "Unvalidated preview of monocular COM versus estimated support. "
            "This is not a force-plate margin of stability and is not reportable "
            "as a session balance score. Normal/Stable/Unstable bands and the "
            "0.04 m threshold have no documented basis for this method."
        ),
        "stats_hint": "Unvalidated preview — summary cards stay N/A.",
    },
    "contact": {
        "interp": (
            "Left/right foot contact probability and stance events over time. "
            "Clear alternating contact with brief double support is expected in walking."
        ),
        "stats_hint": "Stance / swing share and heel-strike counts.",
    },
    "stability": {
        "interp": (
            "Gait phase timeline (swing, stance, double support). Regular phase structure "
            "supports temporal stability; prolonged double support or irregular switching "
            "may reflect caution or asymmetry."
        ),
        "stats_hint": "Phase timing and temporal symmetry cues.",
    },
}


def _biomech_chart_note_text(gui, tab_id: str) -> str:
    """Tab-specific caption for the note row outside the plot axes."""
    from stablewalk.ui.viewers.contact_gait_viewer import (
        CONTACT_PANEL_CONTACT,
        CONTACT_PANEL_VGRF,
        contact_panel_note,
    )

    contact = getattr(gui, "_foot_contact", None)
    vgrf = getattr(gui, "_estimated_vgrf", None)
    if tab_id == "grf":
        return contact_panel_note(CONTACT_PANEL_VGRF, contact=contact, vgrf=vgrf)
    if tab_id == "contact":
        return contact_panel_note(CONTACT_PANEL_CONTACT, contact=contact, vgrf=vgrf)
    copy = _BIOMECH_TAB_COPY.get(tab_id) or {}
    return str(copy.get("stats_hint") or copy.get("interp") or "")


def _format_biomech_tab_stats(gui, tab_id: str) -> str:
    """Compact Max / Min / Avg / SD line for the active Biomechanics workspace tab."""
    from stablewalk.ui.viewers.biomech_chart_enrichment import build_tab_enrichment

    return build_tab_enrichment(gui, tab_id).stats_line


def _clinical_paragraph(text: str) -> str:
    """Breathable paragraph copy — sentence breaks instead of dense blocks."""
    raw = " ".join(str(text or "").split()).strip()
    if not raw:
        return "—"
    # Split on sentence endings; keep short single-sentence sections intact.
    parts: list[str] = []
    buf = ""
    for ch in raw:
        buf += ch
        if ch in ".!?" and len(buf.strip()) > 28:
            parts.append(buf.strip())
            buf = ""
    if buf.strip():
        parts.append(buf.strip())
    if len(parts) <= 1:
        return raw
    return "\n\n".join(parts)


def _workspace_tab_label(tab_id: str) -> str:
    for key, label, _family, _mode in _BIOMECH_WORKSPACE_TABS:
        if key == tab_id:
            return label
    return tab_id


def _clinical_display_is_empty(clinical: Any) -> bool:
    """True when generation returned the no-data card — display only."""
    obs = str(getattr(clinical, "current_observation", "") or "").strip()
    return obs.startswith("No observation yet")


def _chart_panel_row_height(panel: tk.Misc | None, row: int) -> int:
    """Pixel height of one chart-panel grid row (tab strip, tools, footer)."""
    if panel is None:
        return 0
    try:
        bbox = panel.grid_bbox(0, int(row))
    except tk.TclError:
        return 0
    if not bbox or len(bbox) < 4:
        return 0
    return max(0, int(bbox[3]))


def _sync_clinical_overflow(gui: Any) -> None:
    sync = getattr(gui, "_biomech_clinical_overflow_sync", None)
    if not callable(sync):
        return
    try:
        sync()
    except tk.TclError:
        pass


def _set_clinical_body_mode(gui: Any, *, empty: bool) -> None:
    """Show either the interpretation sections or a neutral empty state."""
    empty_lbl = getattr(gui, "lbl_biomech_clinical_empty", None)
    host = getattr(gui, "_biomech_clinical_body_host", None)
    canvas = getattr(host, "_overflow_canvas", None) if host is not None else None
    vsb = getattr(host, "_overflow_vsb", None) if host is not None else None
    if empty:
        if canvas is not None:
            try:
                canvas.grid_remove()
            except tk.TclError:
                pass
        if vsb is not None:
            try:
                vsb.grid_remove()
            except tk.TclError:
                pass
        if empty_lbl is not None:
            try:
                empty_lbl.grid()
            except tk.TclError:
                pass
        return
    if empty_lbl is not None:
        try:
            empty_lbl.grid_remove()
        except tk.TclError:
            pass
    if canvas is not None:
        try:
            canvas.grid()
        except tk.TclError:
            pass
    _sync_clinical_overflow(gui)


def _build_clinical_interpretation_card(gui, parent: tk.Misc) -> tk.Frame:
    """Right-rail Biomechanical Interpretation card (educational sections)."""
    from stablewalk.ui.viewers.biomech_chart_enrichment import (
        CLINICAL_CARD_DISCLAIMER,
        CLINICAL_CARD_SECTION_KEYS,
        CLINICAL_CARD_SECTION_TITLES,
    )

    wrap0 = max(180, BIOMECH_CLINICAL_WIDTH - _BIOMECH_CLINICAL_WRAP_MARGIN)

    card = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    card.columnconfigure(0, weight=1)
    card.rowconfigure(1, weight=1, minsize=140)
    card.rowconfigure(2, weight=0)

    subtitle = tk.Label(
        card,
        text="Active graph · educational reference",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=wrap0,
    )
    subtitle.grid(row=0, column=0, sticky="ew", padx=(PAD_XS, 0), pady=(0, PAD_SM))
    gui.lbl_biomech_clinical_graph = subtitle

    # Overflow scroll when interpretation is taller than the graph-aligned rail.
    body_host = tk.Frame(card, bg=PANEL, highlightthickness=0)
    body_host.grid(row=1, column=0, sticky="nsew")
    body_host.columnconfigure(0, weight=1)
    body_host.rowconfigure(0, weight=1)
    try:
        # Size from the card row, not the 1×1 canvas request (that collapsed
        # the interpretation body to a blank band under the header).
        body_host.grid_propagate(False)
    except tk.TclError:
        pass
    body = tk.Frame(body_host, bg=PANEL, highlightthickness=0)
    body.columnconfigure(0, weight=1)
    _install_biomech_overflow_scroll(body_host, body)
    gui._biomech_clinical_body_host = body_host
    gui._biomech_clinical_overflow_sync = getattr(body_host, "_overflow_sync", None)

    empty = tk.Label(
        body_host,
        text=BIOMECH_INTERPRETATION_EMPTY,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI,
        anchor="nw",
        justify=tk.LEFT,
        wraplength=wrap0,
    )
    empty.grid(row=0, column=0, sticky="nsew", padx=(PAD_XS, 0), pady=(0, PAD_SM))
    gui.lbl_biomech_clinical_empty = empty

    gui._biomech_clinical_labels: dict[str, tk.Label] = {}
    gui._biomech_clinical_headers: dict[str, tk.Label] = {}
    for i, key in enumerate(CLINICAL_CARD_SECTION_KEYS):
        section = tk.Frame(body, bg=PANEL, highlightthickness=0)
        section.grid(
            row=i,
            column=0,
            sticky="ew",
            pady=(0 if i == 0 else PAD_MD, 0),
        )
        section.columnconfigure(0, weight=1)
        heading = tk.Label(
            section,
            text=CLINICAL_CARD_SECTION_TITLES.get(key, key),
            bg=PANEL,
            fg=TEXT_SECONDARY,
            font=FONT_KPI_TITLE,
            anchor="w",
            justify=tk.LEFT,
            wraplength=wrap0,
        )
        heading.grid(row=0, column=0, sticky="ew", padx=(PAD_XS, 0), pady=(0, PAD_XS))
        gui._biomech_clinical_headers[key] = heading
        value = tk.Label(
            section,
            text="—",
            bg=PANEL,
            fg=TEXT,
            font=FONT_UI,
            anchor="nw",
            justify=tk.LEFT,
            wraplength=wrap0,
        )
        value.grid(row=1, column=0, sticky="ew", padx=(PAD_XS, 0), pady=(0, PAD_SM))
        gui._biomech_clinical_labels[key] = value
        setattr(gui, f"lbl_biomech_clinical_{key}", value)

    disclaimer = tk.Label(
        card,
        text=CLINICAL_CARD_DISCLAIMER,
        bg=PANEL,
        fg=MUTED_DIM,
        font=FONT_UI_XS,
        anchor="nw",
        justify=tk.LEFT,
        wraplength=wrap0,
    )
    disclaimer.grid(row=2, column=0, sticky="ew", padx=(PAD_XS, 0), pady=(PAD_SM, 0))
    gui.lbl_biomech_clinical_disclaimer = disclaimer

    def _on_clinical_resize(event: object) -> None:
        try:
            width = int(getattr(event, "width", 0) or 0)
        except (TypeError, ValueError):
            return
        if width < 80:
            return
        # Generous wrap margin → cleaner line breaks, fewer dense mid-word wraps.
        wrap = max(160, width - _BIOMECH_CLINICAL_WRAP_MARGIN)
        for lbl in gui._biomech_clinical_labels.values():
            try:
                lbl.configure(wraplength=wrap)
            except tk.TclError:
                pass
        for hdr in gui._biomech_clinical_headers.values():
            try:
                hdr.configure(wraplength=wrap)
            except tk.TclError:
                pass
        try:
            subtitle.configure(wraplength=wrap)
        except tk.TclError:
            pass
        try:
            disclaimer.configure(wraplength=wrap)
        except tk.TclError:
            pass
        try:
            empty.configure(wraplength=wrap)
        except tk.TclError:
            pass
        _sync_clinical_overflow(gui)

    card.bind("<Configure>", _on_clinical_resize, add="+")
    _set_clinical_body_mode(gui, empty=True)
    return card


def update_biomech_workspace_panels(gui) -> None:
    """Refresh interpretation + statistics for the active Biomechanics tab."""
    from stablewalk.ui.viewers.biomech_chart_enrichment import (
        build_clinical_interpretation,
        build_tab_enrichment,
    )

    tab_id = str(getattr(gui, "_biomech_workspace_tab", "com") or "com")
    enrichment = build_tab_enrichment(gui, tab_id)
    gui._biomech_export_name = enrichment.export_name
    # Keep tools-bar export stems in sync with the active graph.
    for bar_attr in ("biomech_chart_tools", "contact_gait_tools"):
        bar = getattr(gui, bar_attr, None)
        if bar is not None:
            try:
                bar._chart_export_name = enrichment.export_name
            except Exception:
                pass
    interp = getattr(gui, "lbl_biomech_interpretation", None)
    if interp is not None:
        try:
            interp.configure(text=enrichment.interpretation)
        except Exception:
            pass
    stats = getattr(gui, "lbl_biomech_tab_stats", None)
    if stats is not None:
        try:
            line = enrichment.stats_line
            try:
                from stablewalk.ui.overview_clinical_summary import (
                    extract_overview_summary_fields,
                )

                fields = extract_overview_summary_fields(gui)
                left = (fields.get("contact_coverage_left") or ("", ""))[0]
                right = (fields.get("contact_coverage_right") or ("", ""))[0]
                cycles = (fields.get("cycle_evidence") or ("", ""))[0]
                bits = []
                if left and left != "Not available":
                    bits.append(f"L contact {left}")
                if right and right != "Not available":
                    bits.append(f"R contact {right}")
                if cycles and cycles != "Not available":
                    bits.append(f"cycles {cycles}")
                if bits:
                    extra = " · ".join(bits) + " · missing coverage is not asymmetry"
                    line = f"{line}  ·  {extra}" if line else extra
            except Exception:
                pass
            stats.configure(text=line)
        except Exception:
            pass
    note_lbl = getattr(gui, "lbl_biomech_chart_note", None)
    if note_lbl is not None:
        try:
            note_lbl.configure(text=_biomech_chart_note_text(gui, tab_id))
        except Exception:
            pass

    # Right-rail interpretation card (auto for selected graph).
    clinical = build_clinical_interpretation(gui, tab_id, enrichment=enrichment)
    graph_lbl = getattr(gui, "lbl_biomech_clinical_graph", None)
    if graph_lbl is not None:
        try:
            graph_lbl.configure(
                text=f"{_workspace_tab_label(tab_id)} · educational reference"
            )
        except Exception:
            pass
    labels = getattr(gui, "_biomech_clinical_labels", None) or {}
    headers = getattr(gui, "_biomech_clinical_headers", None) or {}
    reference_heading = headers.get("normal_range")
    if reference_heading is not None:
        try:
            reference_heading.configure(
                text="Illustrative reference" if tab_id == "com" else "Normal range"
            )
        except Exception:
            pass
    if _clinical_display_is_empty(clinical):
        _set_clinical_body_mode(gui, empty=True)
        return
    for key, text in clinical.as_sections().items():
        lbl = labels.get(key)
        if lbl is None:
            continue
        try:
            lbl.configure(text=_clinical_paragraph(text))
        except Exception:
            pass
    _set_clinical_body_mode(gui, empty=False)
    _sync_clinical_overflow(gui)


def build_biomechanics_tab(gui, parent: ttk.Frame) -> None:
    """Install Biomechanics tab: internal tabs with one large graph each."""
    from stablewalk.ui.design_system.components import CollapsibleSection, WarningBanner
    from stablewalk.ui.tk.dashboard_page_frame import install_page_header
    from stablewalk.ui.tk.kpi_cards import create_technical_details_panel
    from stablewalk.ui.viewers.biomechanics_charts import (
        BIOMECH_SERIES_OPTIONS,
        DEFAULT_BIOMECH_SERIES,
    )
    from stablewalk.ui.viewers.chart_interactions import build_chart_tools_bar
    from stablewalk.ui.viewers.contact_gait_viewer import (
        CONTACT_PANEL_CONTACT,
        CONTACT_PANEL_VGRF,
    )

    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=0)
    parent.rowconfigure(1, weight=0)
    parent.rowconfigure(2, weight=1)

    install_page_header(parent, "biomechanics", row=0)

    # Short page banner; full research disclaimer lives on the status-bar chip.
    banner = WarningBanner(
        parent,
        "Research / education estimates — not a clinical diagnosis.",
        tone="warn",
        tooltip=DISCLAIMER_RESEARCH or DISCLAIMER_MONOCULAR or "",
    )
    banner.frame.grid(row=1, column=0, sticky="ew", pady=(0, PAD_XS))
    gui._biomech_disclaimer_banner = banner

    body = ttk.Frame(parent)
    body.grid(row=2, column=0, sticky="nsew")
    body.columnconfigure(0, weight=0)
    body.columnconfigure(1, weight=1)
    body.columnconfigure(2, weight=0)
    body.rowconfigure(0, weight=1)
    # Key Findings stays a sibling of a chart|clinical PanedWindow so stacking
    # can re-grid the rail. Tests without a paned still use the 3-column grid.
    from stablewalk.ui.tk.analysis_split import create_h_split

    gui._biomech_split = None
    gui._biomech_chart_split = create_h_split(body)
    gui._biomech_sash_placed = False
    gui._biomech_sash_for_width = None
    gui._biomech_body = body

    sidebar = ttk.Frame(
        body, style="Card.TFrame", padding=(PAD_XS, PAD_XS, PAD_XS, PAD_XS)
    )
    sidebar.columnconfigure(0, weight=1)
    sidebar.rowconfigure(0, weight=1)
    gui._biomech_sidebar = sidebar

    sidebar_host = tk.Frame(sidebar, bg=PANEL, highlightthickness=0)
    sidebar_host.grid(row=0, column=0, sticky="nsew")
    sidebar_host.columnconfigure(0, weight=1)
    sidebar_host.rowconfigure(0, weight=1)
    sidebar_inner = tk.Frame(sidebar_host, bg=PANEL, highlightthickness=0)
    sidebar_inner.grid(row=0, column=0, sticky="nsew")
    gui._biomech_sidebar_inner = sidebar_inner
    gui._biomech_findings_host = sidebar_host

    chart_panel = ttk.Frame(body, style="Card.TFrame", padding=_ANALYSIS_PANEL_PAD)
    chart_panel.columnconfigure(0, weight=1)
    chart_panel.rowconfigure(0, weight=0)  # tab strip
    chart_panel.rowconfigure(1, weight=0)  # export
    chart_panel.rowconfigure(2, weight=1, minsize=BIOMECH_CHART_GRID_MIN_H)  # graph (primary)
    chart_panel.rowconfigure(3, weight=0)  # tab-specific note outside axes
    chart_panel.rowconfigure(4, weight=0)  # compact stats footer

    # Clinical rail mirrors the chart column so the interpretation body
    # stays vertically aligned with the graph host (not the tab chrome).
    clinical_host = ttk.Frame(body)
    clinical_host.columnconfigure(0, weight=1)
    clinical_host.rowconfigure(0, weight=0)  # chrome spacer (tabs + tools)
    clinical_host.rowconfigure(1, weight=1)  # interpretation ↔ graph
    clinical_host.rowconfigure(2, weight=0)  # footer spacer
    gui._biomech_clinical_host = clinical_host

    clinical_chrome = tk.Frame(clinical_host, bg=PANEL, highlightthickness=0)
    clinical_chrome.grid(row=0, column=0, sticky="ew")
    gui._biomech_clinical_chrome = clinical_chrome

    clinical_viewport = tk.Frame(clinical_host, bg=PANEL, highlightthickness=0)
    clinical_viewport.grid(row=1, column=0, sticky="nsew")
    clinical_viewport.columnconfigure(0, weight=1)
    clinical_viewport.rowconfigure(0, weight=1)
    clinical_inner = tk.Frame(clinical_viewport, bg=PANEL, highlightthickness=0)
    clinical_inner.columnconfigure(0, weight=1)
    clinical_inner.rowconfigure(0, weight=1)
    clinical_inner.grid(row=0, column=0, sticky="nsew")
    clinical_section = CollapsibleSection(
        clinical_inner,
        "Biomechanical Interpretation",
        expanded=True,
        bg=PANEL,
        on_toggle=lambda expanded: _on_biomech_rail_toggled(gui, "clinical", expanded),
    )
    clinical_section.frame.grid(row=0, column=0, sticky="nsew")
    clinical_section.body.columnconfigure(0, weight=1)
    clinical_section.body.rowconfigure(0, weight=1)
    try:
        clinical_section.title_lbl.configure(
            wraplength=max(140, BIOMECH_CLINICAL_MIN_W - 48),
            justify=tk.LEFT,
            anchor="w",
        )
    except tk.TclError:
        pass
    clinical_card = _build_clinical_interpretation_card(gui, clinical_section.body)
    clinical_card.grid(row=0, column=0, sticky="nsew", padx=(PAD_XS, 0))
    try:
        clinical_section.frame.rowconfigure(1, weight=1)
    except tk.TclError:
        pass
    gui._biomech_clinical_section = clinical_section
    gui._biomech_clinical_viewport = clinical_viewport
    gui._biomech_clinical_card = clinical_card

    clinical_footer_spacer = tk.Frame(clinical_host, bg=PANEL, highlightthickness=0)
    clinical_footer_spacer.grid(row=2, column=0, sticky="ew")
    gui._biomech_clinical_footer_spacer = clinical_footer_spacer

    from stablewalk.ui.tk.analysis_split import split_add

    sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, PAD_XS), pady=0)
    gui._biomech_chart_split.grid(row=0, column=1, sticky="nsew")
    split_add(gui._biomech_chart_split, chart_panel, minsize=BIOMECH_CHART_COL_MIN_W)
    split_add(gui._biomech_chart_split, clinical_host, minsize=BIOMECH_CLINICAL_MIN_W)
    body.columnconfigure(0, weight=0)
    body.columnconfigure(1, weight=1)
    body.columnconfigure(2, weight=0)
    gui._biomech_chart_panel = chart_panel

    def _sync_clinical_graph_alignment(_event: object | None = None) -> None:
        """Keep interpretation body top/bottom aligned with the graph."""
        try:
            stacked = bool(getattr(gui, "_biomech_stacked", False))
            collapsed = bool(getattr(gui, "_biomech_rails_collapsed", False))
            footer_hidden = bool(
                getattr(gui, "_biomech_footer_compact_hidden", False)
            )
            if stacked or collapsed:
                clinical_chrome.grid_remove()
                clinical_footer_spacer.grid_remove()
                _sync_clinical_overflow(gui)
                return
            clinical_chrome.grid()
            chart = getattr(gui, "_biomech_chart_panel", None)
            # Tab strip + tools only — never the notebook client/page height.
            chrome_h = _chart_panel_row_height(chart, 0) + _chart_panel_row_height(
                chart, 1
            )
            if chrome_h <= 0:
                tools = getattr(gui, "biomech_chart_tools", None)
                tools_row = getattr(tools, "master", None) if tools is not None else None
                try:
                    if tools_row is not None:
                        chrome_h = max(0, int(tools_row.winfo_reqheight() or 0)) + 28
                except tk.TclError:
                    chrome_h = 0
            chrome_h += int(PAD_XS) * 2
            note_h = _chart_panel_row_height(chart, 3)
            if note_h <= 0:
                note_lbl = getattr(gui, "lbl_biomech_chart_note", None)
                if note_lbl is not None:
                    try:
                        note_h = int(note_lbl.winfo_reqheight() or 0)
                    except tk.TclError:
                        note_h = 0
            footer = getattr(gui, "_biomech_chart_footer", None)
            footer_h = 0
            if footer is not None and not footer_hidden:
                footer_h = _chart_panel_row_height(chart, 4)
                if footer_h <= 0:
                    try:
                        footer_h = int(footer.winfo_height() or 0)
                        if footer_h <= 1:
                            footer_h = int(footer.winfo_reqheight() or 0)
                    except tk.TclError:
                        footer_h = 0
            spacer_h = note_h + footer_h
            if spacer_h > 0:
                spacer_h += int(PAD_XS)
            if chrome_h > 0:
                clinical_chrome.configure(height=max(1, chrome_h))
                clinical_chrome.grid_propagate(False)
            if spacer_h > 0:
                clinical_footer_spacer.grid()
                clinical_footer_spacer.configure(height=max(1, spacer_h))
                clinical_footer_spacer.grid_propagate(False)
            else:
                clinical_footer_spacer.grid_remove()
            _sync_clinical_overflow(gui)
        except tk.TclError:
            pass

    gui._sync_biomech_clinical_graph_alignment = _sync_clinical_graph_alignment

    # Sidebar owns KPIs — do not build a duplicate summary strip.
    gui._biomech_summary_card = None
    gui._biomech_summary_kpis = {}
    gui._biomech_summary_value_labels = {}

    findings_section = CollapsibleSection(
        sidebar_inner,
        "Key Findings",
        expanded=True,
        bg=PANEL,
        on_toggle=lambda expanded: _on_biomech_rail_toggled(gui, "findings", expanded),
    )
    findings_section.frame.pack(fill=tk.BOTH, expand=True)
    findings_body = findings_section.body
    findings_body.columnconfigure(0, weight=1)
    gui._biomech_findings_section = findings_section

    gui._biomech_kpi_cards = {}
    _SIDEBAR_TIERS = {
        "gait_quality": "primary",
        "stability_state": "primary",
        "symmetry": "secondary",
        "stability_margin": "secondary",
        "cadence": "technical",
        "walking_speed": "technical",
    }
    primary_keys = {"gait_quality", "stability_state"}
    secondary_keys = {"symmetry", "stability_margin"}

    for key, title, tip_key, attr in _BIOMECH_SIDEBAR_FIELDS:
        tip = get_metric_tooltip(tip_key) or title
        tier = _SIDEBAR_TIERS.get(key, "secondary")
        if key not in primary_keys and key not in secondary_keys:
            continue
        kpi = create_kpi_card(
            findings_body,
            key=key,
            title=title,
            tooltip=tip,
            compact=True,
            show_bar=False,
            fill=False,
            tier="secondary",
            title_wrap_lines=2,
        )
        gui._biomech_kpi_cards[key] = kpi
        setattr(gui, attr, kpi.value_lbl)

    details = create_technical_details_panel(
        findings_body,
        title="Technical details",
        bg=PANEL,
        start_expanded=False,
    )
    gui._biomech_sidebar_details = details

    for key, title, tip_key, attr in _BIOMECH_SIDEBAR_FIELDS:
        if _SIDEBAR_TIERS.get(key) != "technical":
            continue
        tip = get_metric_tooltip(tip_key) or title
        kpi = create_kpi_card(
            details.content,
            key=key,
            title=title,
            tooltip=tip,
            compact=True,
            show_bar=False,
            fill=False,
            tier="technical",
            title_wrap_lines=2,
        )
        gui._biomech_kpi_cards[key] = kpi
        setattr(gui, attr, kpi.value_lbl)

    rom_title = ttk.Label(details.content, text=LABEL_JOINT_ROM, style="SideMuted.TLabel")
    rom_title.pack(anchor=tk.W, pady=(PAD_SM, PAD_XS // 2))
    attach_metric_tooltip(rom_title, "rom")
    gui.lbl_biomech_rom = tk.Label(
        details.content, text="—", bg=PANEL, fg=MUTED, font=FONT_UI_XS, anchor="w", justify=tk.LEFT
    )
    gui.lbl_biomech_rom.pack(anchor=tk.W, fill=tk.X)
    attach_metric_tooltip(gui.lbl_biomech_rom, "rom")
    bind_responsive_wrap(gui, details.content, ("lbl_biomech_rom",), margin=12)

    video_kpi = create_kpi_card(
        details.content,
        key="video_quality",
        title=LABEL_VIDEO_QUALITY,
        tooltip=get_metric_tooltip("video_quality") or LABEL_VIDEO_QUALITY,
        compact=True,
        show_bar=True,
        fill=False,
        tier="technical",
        title_wrap_lines=2,
    )
    gui._biomech_kpi_cards["video_quality"] = video_kpi
    gui.lbl_biomech_video_score = video_kpi.value_lbl
    gui.lbl_biomech_video_checks = tk.Label(
        details.content,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=max(140, BIOMECH_FINDINGS_MIN_W - 24),
    )
    gui.lbl_biomech_video_checks.pack(anchor=tk.W, fill=tk.X, pady=(0, PAD_XS))

    # Flat evidence block inside Technical details (no nested LabelFrame card).
    evidence = tk.Frame(details.content, bg=PANEL, highlightthickness=0)
    evidence.pack(fill=tk.X, pady=(PAD_SM, 0))
    create_metric_group_label(evidence, "Analysis evidence", bg=PANEL)
    gui.lbl_advanced_temporal = ttk.Label(
        evidence, text="Temporal Symmetry: —", style="Card.TLabel"
    )
    gui.lbl_advanced_temporal.pack(anchor=tk.W)
    gui.lbl_advanced_pelvis = ttk.Label(
        evidence, text="Pelvis Stability: —", style="Card.TLabel"
    )
    gui.lbl_advanced_pelvis.pack(anchor=tk.W, pady=(PAD_XS, 0))
    gui.lbl_advanced_evidence = tk.Label(
        evidence,
        text="Evidence: —",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=max(140, BIOMECH_FINDINGS_MIN_W - 24),
        justify=tk.LEFT,
    )
    gui.lbl_advanced_evidence.pack(anchor=tk.W, fill=tk.X, pady=(PAD_XS, 0))
    gui._section_advanced_info = evidence

    # Hidden host for temporal metrics builder (stats surface uses workspace footer).
    gui._biomech_temporal_host = tk.Frame(chart_panel, bg=PANEL, highlightthickness=0)

    gui.fig_biomech = None
    gui.canvas_biomech = None
    label_by_id = {sid: lab for sid, lab in BIOMECH_SERIES_OPTIONS}
    gui._biomech_series_key = DEFAULT_BIOMECH_SERIES
    gui.var_biomech_series = tk.StringVar(
        value=label_by_id.get(DEFAULT_BIOMECH_SERIES, "COM Vertical Position")
    )
    gui.cmb_biomech_series = None  # replaced by workspace tabs
    gui._biomech_workspace_tab = "com"
    gui._contact_chart_panel = CONTACT_PANEL_CONTACT
    gui._biomech_chart_family = "biomech"

    # ── Internal tabs ─────────────────────────────────────────────────────
    notebook = ttk.Notebook(chart_panel)
    notebook.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
    gui._biomech_notebook = notebook
    gui._biomech_tab_frames = {}
    for tab_id, title, _family, _mode in _BIOMECH_WORKSPACE_TABS:
        page = ttk.Frame(notebook)
        notebook.add(page, text=title)
        gui._biomech_tab_frames[tab_id] = page

    # Export tools for the active chart family.
    tools_row = tk.Frame(chart_panel, bg=PANEL, highlightthickness=0)
    tools_row.grid(row=1, column=0, sticky="ew", pady=(0, PAD_XS))
    tools_row.columnconfigure(0, weight=1)
    def _biomech_export_name() -> str:
        return str(getattr(gui, "_biomech_export_name", None) or "biomech_com")

    gui._biomech_export_name = "biomech_com"
    gui.biomech_chart_tools = build_chart_tools_bar(
        gui,
        tools_row,
        fig_attr="fig_biomech",
        canvas_attr="canvas_biomech",
        export_name="biomech_com",
        export_name_getter=_biomech_export_name,
        ensure_chart=lambda: getattr(gui, "_ensure_biomech_chart", lambda: False)(),
    )
    gui.biomech_chart_tools.grid(row=0, column=0, sticky="ew")
    gui._biomech_contact_tools_host = tk.Frame(tools_row, bg=PANEL, highlightthickness=0)
    gui._biomech_contact_tools_host.columnconfigure(0, weight=1)

    # Shared chart stack — only one family visible (avoids overlap / compression).
    chart_stack = tk.Frame(chart_panel, bg=PANEL, highlightthickness=0)
    chart_stack.grid(row=2, column=0, sticky="nsew")
    chart_stack.columnconfigure(0, weight=1)
    chart_stack.rowconfigure(0, weight=1)
    # Do not let the matplotlib widget's 14" natural size expand the page.
    try:
        chart_stack.grid_propagate(False)
    except tk.TclError:
        pass

    gui.biomech_chart_host = tk.Frame(chart_stack, bg=PANEL, highlightthickness=0)
    gui.biomech_chart_host.grid(row=0, column=0, sticky="nsew")
    gui.biomech_chart_host.columnconfigure(0, weight=1)
    gui.biomech_chart_host.rowconfigure(0, weight=1)
    try:
        gui.biomech_chart_host.grid_propagate(False)
    except tk.TclError:
        pass

    gui._biomech_contact_host = tk.Frame(chart_stack, bg=PANEL, highlightthickness=0)
    gui._biomech_contact_host.columnconfigure(0, weight=1)
    gui._biomech_contact_host.rowconfigure(0, weight=1)
    try:
        gui._biomech_contact_host.grid_propagate(False)
    except tk.TclError:
        pass

    note_row = tk.Frame(chart_panel, bg=PANEL, highlightthickness=0)
    note_row.grid(row=3, column=0, sticky="ew", pady=(PAD_XS, 0))
    note_row.columnconfigure(0, weight=1)
    gui._biomech_chart_note_row = note_row
    gui.lbl_biomech_chart_note = tk.Label(
        note_row,
        text=_BIOMECH_TAB_COPY["com"]["stats_hint"],
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
    )
    gui.lbl_biomech_chart_note.grid(row=0, column=0, sticky="ew")

    # Responsive Interpretation / Statistics cards below the dedicated note row.
    footer = tk.Frame(chart_panel, bg=PANEL, highlightthickness=0)
    footer.grid(row=4, column=0, sticky="ew", pady=(PAD_XS, 0))
    footer.columnconfigure(0, weight=3)
    footer.columnconfigure(1, weight=2)
    gui._biomech_chart_footer = footer

    interp_box = ttk.LabelFrame(
        footer,
        text=_card_title("Interpretation"),
        style="Card.TLabelframe",
        padding=(PAD_SM, PAD_SM),
    )
    interp_box.grid(row=0, column=0, sticky="nsew", padx=(0, PAD_XS))
    gui.lbl_biomech_interpretation = tk.Label(
        interp_box,
        text=_BIOMECH_TAB_COPY["com"]["interp"],
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI,
        anchor="nw",
        justify=tk.LEFT,
        wraplength=640,
    )
    interp_box.columnconfigure(0, weight=1)
    gui.lbl_biomech_interpretation.grid(row=0, column=0, sticky="ew")

    stats_box = ttk.LabelFrame(
        footer,
        text=_card_title("Statistics"),
        style="Card.TLabelframe",
        padding=(PAD_SM, PAD_SM),
    )
    stats_box.grid(row=0, column=1, sticky="nsew")
    gui.lbl_biomech_tab_stats = tk.Label(
        stats_box,
        text=_BIOMECH_TAB_COPY["com"]["stats_hint"],
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI,
        anchor="nw",
        justify=tk.LEFT,
        wraplength=360,
    )
    stats_box.columnconfigure(0, weight=1)
    gui.lbl_biomech_tab_stats.grid(row=0, column=0, sticky="ew")

    gui.lbl_biomech_units_legend = tk.Label(
        footer,
        text=f"{UNITS_COORD_LEGEND}  ·  Cyan shade = low tracking confidence",
        bg=PANEL,
        fg=MUTED_DIM,
        font=FONT_UI_XS,
        anchor="w",
    )
    attach_metric_tooltip(gui.lbl_biomech_units_legend, "com")
    gui.lbl_biomech_units_legend.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(PAD_XS // 2, 0))

    def _wrap_footer_text(
        event: object, label: tk.Label, *, max_line_width: int
    ) -> None:
        try:
            width = int(getattr(event, "width", 0) or 0)
            if width >= 80:
                available = max(120, width - 2 * int(PAD_SM))
                label.configure(wraplength=min(max_line_width, available))
        except (tk.TclError, TypeError, ValueError):
            pass

    interp_box.bind(
        "<Configure>",
        lambda event: _wrap_footer_text(
            event,
            gui.lbl_biomech_interpretation,
            max_line_width=560,
        ),
        add="+",
    )
    stats_box.bind(
        "<Configure>",
        lambda event: _wrap_footer_text(
            event,
            gui.lbl_biomech_tab_stats,
            max_line_width=420,
        ),
        add="+",
    )

    def _reflow_biomech_footer(event: object | None = None) -> None:
        try:
            width = int(getattr(event, "width", 0) or footer.winfo_width() or 0)
        except (tk.TclError, TypeError, ValueError):
            return
        stacked = width < 720
        if bool(getattr(gui, "_biomech_footer_stacked", False)) == stacked:
            return
        gui._biomech_footer_stacked = stacked
        try:
            if stacked:
                interp_box.grid_configure(
                    row=0, column=0, columnspan=2, padx=0, pady=(0, PAD_XS)
                )
                stats_box.grid_configure(
                    row=1, column=0, columnspan=2, padx=0, pady=0
                )
                gui.lbl_biomech_units_legend.grid_configure(row=2)
            else:
                interp_box.grid_configure(
                    row=0,
                    column=0,
                    columnspan=1,
                    padx=(0, PAD_XS),
                    pady=0,
                )
                stats_box.grid_configure(
                    row=0, column=1, columnspan=1, padx=0, pady=0
                )
                gui.lbl_biomech_units_legend.grid_configure(row=1)
        except tk.TclError:
            pass

    footer.bind("<Configure>", _reflow_biomech_footer, add="+")
    try:
        gui.root.after_idle(_reflow_biomech_footer)
    except (AttributeError, tk.TclError):
        _reflow_biomech_footer()

    gui._biomech_chart_ready = False
    gui._biomechanics_header = None
    gui._biomech_chart_panel = chart_panel

    def _show_chart_family(family: str) -> None:
        gui._biomech_chart_family = family
        contact_tools_host = getattr(gui, "_biomech_contact_tools_host", None)
        if family == "contact":
            gui.biomech_chart_host.grid_remove()
            gui._biomech_contact_host.grid(row=0, column=0, sticky="nsew")
            try:
                gui.biomech_chart_tools.grid_remove()
            except Exception:
                pass
            if contact_tools_host is not None:
                ctools = getattr(gui, "contact_gait_tools", None)
                if ctools is not None:
                    try:
                        ctools.grid(row=0, column=0, sticky="ew")
                    except Exception:
                        pass
                try:
                    contact_tools_host.grid(row=0, column=0, sticky="ew")
                except Exception:
                    pass
        else:
            gui._biomech_contact_host.grid_remove()
            gui.biomech_chart_host.grid(row=0, column=0, sticky="nsew")
            if contact_tools_host is not None:
                try:
                    contact_tools_host.grid_remove()
                except Exception:
                    pass
            try:
                gui.biomech_chart_tools.grid(row=0, column=0, sticky="ew")
            except Exception:
                pass

    def _schedule_active_chart_refresh(*, tries: int = 0) -> None:
        """Refresh the selected chart once Tk has assigned real host geometry."""
        pending = getattr(gui, "_biomech_chart_refresh_after", None)
        if pending is not None:
            try:
                gui.root.after_cancel(pending)
            except (tk.TclError, ValueError):
                pass

        def _host_ready(host: tk.Misc | None) -> bool:
            if host is None:
                return False
            try:
                if not bool(host.winfo_ismapped()):
                    return False
                return int(host.winfo_width() or 0) >= 80 and int(host.winfo_height() or 0) >= 80
            except tk.TclError:
                return False

        def _refresh() -> None:
            gui._biomech_chart_refresh_after = None
            from stablewalk.ui.tk.layout_playback import force_layout_pass

            family = str(
                getattr(gui, "_biomech_chart_family", "biomech") or "biomech"
            )
            host = (
                getattr(gui, "_biomech_contact_host", None)
                if family == "contact"
                else getattr(gui, "biomech_chart_host", None)
            )
            with force_layout_pass(gui):
                try:
                    if host is not None:
                        host.update_idletasks()
                except tk.TclError:
                    return

                if family == "contact":
                    ensure = getattr(gui, "_ensure_contact_gait_chart", None)
                    update = getattr(gui, "_update_contact_gait_chart", None)
                    fit = getattr(gui, "_fit_contact_gait_canvas", None)
                    canvas_attr = "canvas_contact_gait"
                else:
                    ensure = getattr(gui, "_ensure_biomech_chart", None)
                    update = getattr(gui, "_update_biomechanics_chart", None)
                    fit = getattr(gui, "_fit_biomech_canvas", None)
                    canvas_attr = "canvas_biomech"

                if callable(ensure):
                    try:
                        ensure()
                    except Exception:
                        return
                if callable(update):
                    try:
                        update(force_rebuild=True)
                    except Exception:
                        pass
                if callable(fit):
                    try:
                        fit()
                    except Exception:
                        pass
                canvas = getattr(gui, canvas_attr, None)
                if canvas is not None:
                    try:
                        widget = canvas.get_tk_widget()
                        widget.grid(row=0, column=0, sticky="nsew")
                        widget.lift()
                        canvas.draw()
                    except Exception:
                        pass

            if not _host_ready(host) and int(tries) < 12:
                try:
                    gui._biomech_chart_refresh_after = gui.root.after(
                        50,
                        lambda: _schedule_active_chart_refresh(tries=int(tries) + 1),
                    )
                except tk.TclError:
                    pass

        try:
            if int(tries) <= 0:
                gui._biomech_chart_refresh_after = gui.root.after_idle(_refresh)
            else:
                gui._biomech_chart_refresh_after = gui.root.after(50, _refresh)
        except tk.TclError:
            _refresh()

    gui._refresh_active_biomech_chart = _schedule_active_chart_refresh

    def _activate_biomech_tab(tab_id: str) -> None:
        meta = {t[0]: t for t in _BIOMECH_WORKSPACE_TABS}.get(tab_id)
        if meta is None:
            return
        _tid, _title, family, mode = meta
        gui._biomech_workspace_tab = tab_id
        if family == "biomech":
            gui._biomech_series_key = mode
            gui.var_biomech_series.set(label_by_id.get(mode, mode))
            gui._biomech_chart_data_key = None
            _show_chart_family("biomech")
        else:
            gui._contact_chart_panel = (
                CONTACT_PANEL_VGRF if mode == "vgrf" else CONTACT_PANEL_CONTACT
            )
            gui._contact_chart_data_key = None
            _show_chart_family("contact")
        update_biomech_workspace_panels(gui)
        _schedule_active_chart_refresh()

    def _on_notebook_tab(_event: object | None = None) -> None:
        try:
            idx = int(notebook.index(notebook.select()))
        except Exception:
            return
        if idx < 0 or idx >= len(_BIOMECH_WORKSPACE_TABS):
            return
        tab_key = _BIOMECH_WORKSPACE_TABS[idx][0]
        _activate_biomech_tab(tab_key)
        try:
            from stablewalk.ui.tk.ui_motion import reveal_host

            family = _BIOMECH_WORKSPACE_TABS[idx][2]
            host = (
                getattr(gui, "_biomech_contact_host", None)
                if family == "contact"
                else getattr(gui, "biomech_chart_host", None)
            )
            reveal_host(gui, host, key=f"biomech-tab:{tab_key}")
        except Exception:
            pass

    notebook.bind("<<NotebookTabChanged>>", _on_notebook_tab)
    gui._activate_biomech_workspace_tab = _activate_biomech_tab
    _show_chart_family("biomech")

    def _fit_biomech_canvas() -> None:
        canvas = getattr(gui, "canvas_biomech", None)
        fig = getattr(gui, "fig_biomech", None)
        host = getattr(gui, "biomech_chart_host", None)
        if canvas is None or fig is None or host is None:
            return
        if getattr(gui, "_biomech_chart_family", "biomech") != "biomech":
            return
        from stablewalk.ui.tk.dashboard_layout import _fit_figure_to_host
        from stablewalk.ui.viewers.biomechanics_charts import layout_biomechanics_figure

        fitted = _fit_figure_to_host(
            canvas,
            fig,
            host=host,
            min_px=80,
        )
        _pin_biomech_mpl_widget(canvas, host)
        if fitted:
            layout_biomechanics_figure(fig, size_known=True)
            try:
                # Synchronous draw — draw_idle can leave a 1×1 raster after pin.
                canvas.draw()
            except Exception:
                try:
                    canvas.draw_idle()
                except Exception:
                    pass
            _pin_biomech_mpl_widget(canvas, host)
            if getattr(gui, "_biomech_chart_undersized", False):
                gui._biomech_chart_undersized = False
                gui._biomech_chart_data_key = None
            try:
                from stablewalk.ui.tk.ui_motion import consume_viz_reveal, reveal_canvas

                if consume_viz_reveal(gui, "biomech"):
                    reveal_canvas(gui, canvas, key="biomech")
            except Exception:
                pass

    def _ensure_biomech_chart() -> bool:
        if getattr(gui, "_biomech_chart_ready", False) and gui.fig_biomech is not None:
            return True
        from stablewalk.ui.viewers.chart_interactions import attach_chart_interactions

        # Wide/tall default; fit-to-host expands into the graph column.
        gui.fig_biomech = Figure(figsize=BIOMECH_CHART_FIGSIZE, dpi=120, facecolor=PANEL)
        gui.canvas_biomech = FigureCanvasTkAgg(
            gui.fig_biomech, master=gui.biomech_chart_host
        )
        w = gui.canvas_biomech.get_tk_widget()
        # 1×1 requested size — BIOMECH_CHART_FIGSIZE must not inflate the tab.
        w.configure(bg=PANEL, highlightthickness=0, width=1, height=1)
        w.grid(row=0, column=0, sticky="nsew")
        from stablewalk.ui.tk.dashboard_layout import _bind_figure_resize

        _bind_figure_resize(gui.canvas_biomech, gui.fig_biomech, min_px=80)
        attach_chart_interactions(gui.fig_biomech, gui.canvas_biomech)
        gui._fit_biomech_canvas = _fit_biomech_canvas
        gui._biomech_chart_ready = True
        try:
            gui.root.after_idle(_fit_biomech_canvas)
        except tk.TclError:
            pass
        return True

    gui._ensure_biomech_chart = _ensure_biomech_chart
    gui._fit_biomech_canvas = _fit_biomech_canvas
    _biomech_cfg_job: dict[str, object] = {"id": None}

    def _on_biomech_panel_configure(_event: object | None = None) -> None:
        from stablewalk.ui.tk.layout_playback import should_defer_layout

        if should_defer_layout(gui):
            return
        job = _biomech_cfg_job.get("id")
        if job is not None:
            try:
                gui.root.after_cancel(job)  # type: ignore[arg-type]
            except Exception:
                pass

        def _run() -> None:
            _biomech_cfg_job["id"] = None
            family = getattr(gui, "_biomech_chart_family", "biomech")
            if family == "contact":
                fit = getattr(gui, "_fit_contact_gait_canvas", None)
            else:
                fit = getattr(gui, "_fit_biomech_canvas", None)
            if callable(fit):
                try:
                    fit()
                except Exception:
                    pass
            align = getattr(gui, "_sync_biomech_clinical_graph_alignment", None)
            if callable(align):
                try:
                    align()
                except Exception:
                    pass

        try:
            _biomech_cfg_job["id"] = gui.root.after(140, _run)
        except Exception:
            _run()

    chart_panel.bind("<Configure>", _on_biomech_panel_configure, add="+")
    gui.biomech_chart_host.bind("<Configure>", _on_biomech_panel_configure, add="+")
    gui._biomech_contact_host.bind("<Configure>", _on_biomech_panel_configure, add="+")
    # Footer labels wrap to their own card widths; binding them to the wider
    # chart panel would turn long copy back into one compressed line.
    bind_responsive_wrap(gui, chart_panel, ("lbl_biomech_units_legend",), margin=24)
    update_biomech_workspace_panels(gui)
    try:
        _sync_clinical_graph_alignment()
    except Exception:
        pass
    _sync_clinical_overflow(gui)

    _biomech_body_job: dict[str, object] = {"id": None}

    def _on_biomech_body_configure(event: object | None = None) -> None:
        try:
            w = int(getattr(event, "width", 0) or body.winfo_width() or 0)
            h = int(getattr(event, "height", 0) or body.winfo_height() or 0)
        except (tk.TclError, TypeError, ValueError):
            return
        if w < 80:
            return
        job = _biomech_body_job.get("id")
        if job is not None:
            try:
                gui.root.after_cancel(job)  # type: ignore[arg-type]
            except Exception:
                pass

        def _run() -> None:
            _biomech_body_job["id"] = None
            apply_biomech_workspace_geometry(gui, width=w, height=h)

        try:
            _biomech_body_job["id"] = gui.root.after(40, _run)
        except Exception:
            _run()

    body.bind("<Configure>", _on_biomech_body_configure, add="+")
    apply_biomech_workspace_geometry(gui, width=max(body.winfo_reqwidth(), 1200), height=0)


def _pin_biomech_mpl_widget(canvas: FigureCanvasTkAgg, host: tk.Misc | None) -> None:
    """Keep the TkAgg widget from requesting the figure's natural pixel size."""
    if canvas is None or host is None:
        return
    try:
        widget = canvas.get_tk_widget()
        hw = max(1, int(host.winfo_width() or 1))
        hh = max(1, int(host.winfo_height() or 1))
        if hw < 8 or hh < 8:
            widget.configure(width=1, height=1)
        else:
            widget.configure(width=hw, height=hh)
    except (tk.TclError, AttributeError):
        pass


def _set_biomech_rail_expanded(section: Any, expanded: bool) -> None:
    setter = getattr(section, "set_expanded", None)
    if callable(setter):
        try:
            setter(bool(expanded))
        except tk.TclError:
            pass


def _sync_biomech_rail_header_mode(section: Any, *, header_only: bool) -> None:
    """Header-only rails keep the toggle; the title would clip at ~40 px."""
    if section is None:
        return
    title = getattr(section, "title_lbl", None)
    if title is None:
        return
    try:
        if header_only:
            title.grid_remove()
        else:
            title.grid()
    except tk.TclError:
        pass


def _on_biomech_rail_toggled(gui: Any, _key: str, _expanded: bool) -> None:
    """User collapse/expand of Key Findings or Interpretation reclaims graph width."""
    gui._biomech_sash_placed = False
    body = getattr(gui, "_biomech_body", None)
    width = 0
    height = 0
    if body is not None:
        try:
            width = int(body.winfo_width() or 0)
            height = int(body.winfo_height() or 0)
        except tk.TclError:
            width = 0
            height = 0
    apply_biomech_workspace_geometry(gui, width=width, height=height)


def apply_biomech_workspace_geometry(gui: Any, *, width: int, height: int = 0) -> None:
    """Balance three columns, collapsing side rails before stacking."""
    body = getattr(gui, "_biomech_body", None)
    sidebar = getattr(gui, "_biomech_sidebar", None)
    chart = getattr(gui, "_biomech_chart_panel", None)
    clinical = getattr(gui, "_biomech_clinical_host", None)
    if body is None or sidebar is None or chart is None or clinical is None:
        return
    try:
        measured = int(body.winfo_width() or 0)
    except tk.TclError:
        measured = 0
    if measured > 80:
        width = measured if int(width) < 80 else min(int(width), measured)
    pad = int(PAD_XS) * 2
    usable = max(1, int(width) - pad)
    findings_w, chart_w, clinical_w = biomech_balanced_column_widths(width)
    from stablewalk.ui.tk.layout_scale import layout_scale_from_gui, scaled_breakpoints

    scale = layout_scale_from_gui(gui)
    stack_below = int(BIOMECH_STACK_BELOW_W)
    if abs(scale - 1.0) >= 0.08:
        # Cap DPI growth so 1366-class laptops keep the chart|interpretation sash
        # instead of collapsing both rails to headers.
        stack_below = min(int(scaled_breakpoints(scale)["biomech_stack"]), 1200)
    collapse_rails = int(width) < max(
        stack_below, biomech_three_col_min_width()
    )
    if not collapse_rails and chart_w < BIOMECH_CHART_COL_MIN_W:
        collapse_rails = True
    if collapse_rails:
        findings_w = min(int(findings_w), int(BIOMECH_FINDINGS_COLLAPSED_W))
        clinical_w = min(int(clinical_w), int(BIOMECH_CLINICAL_COLLAPSED_W))
        chart_w = max(1, usable - findings_w - clinical_w)
    # Last resort: stack only when collapsed rails still cannot keep a usable graph.
    stack_floor = (
        int(BIOMECH_FINDINGS_COLLAPSED_W)
        + int(BIOMECH_CHART_COL_MIN_W)
        + int(BIOMECH_CLINICAL_COLLAPSED_W)
        + pad
    )
    stack = bool(collapse_rails and chart_w < BIOMECH_CHART_COL_MIN_W and int(width) < stack_floor)
    findings_section = getattr(gui, "_biomech_findings_section", None)
    clinical_section = getattr(gui, "_biomech_clinical_section", None)
    header_only_findings = (
        not stack
        and not collapse_rails
        and findings_section is not None
        and not bool(getattr(findings_section, "expanded", True))
    )
    header_only_clinical = (
        not stack
        and not collapse_rails
        and clinical_section is not None
        and not bool(getattr(clinical_section, "expanded", True))
    )
    if header_only_findings:
        findings_w = BIOMECH_RAIL_HEADER_ONLY_W
    if header_only_clinical:
        clinical_w = BIOMECH_RAIL_HEADER_ONLY_W
    if header_only_findings or header_only_clinical:
        chart_w = max(1, usable - findings_w - clinical_w)
    _sync_biomech_rail_header_mode(findings_section, header_only=header_only_findings)
    _sync_biomech_rail_header_mode(clinical_section, header_only=header_only_clinical)
    body_h = int(height or 0)
    if body_h <= 1:
        try:
            body_h = int(body.winfo_height() or 0)
        except tk.TclError:
            body_h = 0
    if body_h <= 1:
        body_h = 360
    side_h = 0
    clinical_h = 0
    chart_h = BIOMECH_CHART_GRID_MIN_H
    if stack:
        usable_h = max(96, body_h - (PAD_XS * 2))
        side_h = min(
            BIOMECH_STACK_SIDE_MAX_H,
            max(BIOMECH_STACK_SIDE_MIN_H, int(body_h * BIOMECH_STACK_SIDE_SHARE)),
        )
        clinical_h = side_h
        chart_floor = 80
        if side_h + clinical_h + chart_floor > usable_h:
            side_h = min(side_h, BIOMECH_STACK_SIDE_MIN_H)
            leftover = max(chart_floor, usable_h - side_h)
            clinical_h = min(clinical_h, max(80, leftover - chart_floor))
            chart_h = max(chart_floor, usable_h - side_h - clinical_h)
        else:
            chart_h = max(chart_floor, min(BIOMECH_CHART_GRID_MIN_H, usable_h - side_h * 2))
    paned = getattr(gui, "_biomech_chart_split", None)
    has_paned = False
    if paned is not None:
        try:
            has_paned = callable(getattr(paned, "panes", None))
        except tk.TclError:
            has_paned = False
    try:
        if stack:
            sidebar.configure(width=max(80, int(width)), height=side_h)
            clinical.configure(width=max(80, int(width)), height=clinical_h)
            sidebar.grid_propagate(False)
            clinical.grid_propagate(False)
        else:
            sidebar.configure(width=int(findings_w), height=max(1, int(body_h)))
            sidebar.grid_propagate(False)
            clinical.configure(width=int(clinical_w), height=max(1, int(body_h)))
            clinical.grid_propagate(not has_paned)
    except tk.TclError:
        pass
    try:
        if stack:
            if has_paned:
                from stablewalk.ui.tk.analysis_split import split_clear

                split_clear(paned)
                paned.grid_remove()
                gui._biomech_sash_placed = False
            body.columnconfigure(0, weight=1, minsize=0)
            body.columnconfigure(1, weight=0, minsize=0)
            body.columnconfigure(2, weight=0, minsize=0)
            body.rowconfigure(0, weight=0, minsize=side_h)
            body.rowconfigure(1, weight=1, minsize=chart_h)
            body.rowconfigure(2, weight=0, minsize=clinical_h)
            sidebar.grid(
                row=0,
                column=0,
                columnspan=3,
                sticky="nsew",
                padx=0,
                pady=(0, PAD_XS),
            )
            chart.grid(
                row=1, column=0, columnspan=3, sticky="nsew", padx=0, pady=0
            )
            clinical.grid(
                row=2,
                column=0,
                columnspan=3,
                sticky="nsew",
                padx=0,
                pady=(PAD_XS, 0),
            )
        elif has_paned:
            from stablewalk.ui.tk.analysis_split import split_add

            body.columnconfigure(0, weight=0, minsize=findings_w)
            body.columnconfigure(1, weight=1, minsize=BIOMECH_CHART_COL_MIN_W)
            body.columnconfigure(2, weight=0, minsize=0)
            chart_row_min = max(80, min(BIOMECH_CHART_GRID_MIN_H, max(80, int(body_h))))
            body.rowconfigure(0, weight=1, minsize=chart_row_min)
            body.rowconfigure(1, weight=0, minsize=0)
            body.rowconfigure(2, weight=0, minsize=0)
            sidebar.grid(
                row=0,
                column=0,
                columnspan=1,
                sticky="nsew",
                padx=(0, PAD_XS),
                pady=0,
            )
            try:
                chart.grid_forget()
            except tk.TclError:
                pass
            try:
                clinical.grid_forget()
            except tk.TclError:
                pass
            clinical_min = (
                BIOMECH_RAIL_HEADER_ONLY_W
                if header_only_clinical
                else BIOMECH_CLINICAL_MIN_W
            )
            split_add(paned, chart, minsize=BIOMECH_CHART_COL_MIN_W)
            split_add(paned, clinical, minsize=clinical_min)
            try:
                paned.paneconfigure(chart, minsize=BIOMECH_CHART_COL_MIN_W)
                paned.paneconfigure(clinical, minsize=max(40, int(clinical_min)))
            except tk.TclError:
                pass
            paned.grid(row=0, column=1, columnspan=1, sticky="nsew", padx=0, pady=0)
            bucket = int(width) // 80
            header_flag = bool(header_only_clinical)
            if (
                not getattr(gui, "_biomech_sash_placed", False)
                or getattr(gui, "_biomech_sash_for_width", None) != bucket
                or getattr(gui, "_biomech_sash_header_clinical", None) != header_flag
            ):
                try:
                    paned.update_idletasks()
                    pw = int(paned.winfo_width() or 0)
                    if pw < 80:
                        pw = max(80, int(chart_w) + int(clinical_w))
                    pos = max(
                        BIOMECH_CHART_COL_MIN_W,
                        min(pw - max(int(clinical_w), 40), pw - int(clinical_min)),
                    )
                    paned.sash_place(0, pos, 0)
                    gui._biomech_sash_placed = True
                    gui._biomech_sash_for_width = bucket
                    gui._biomech_sash_header_clinical = header_flag
                except tk.TclError:
                    pass
        else:
            body.columnconfigure(0, weight=0, minsize=findings_w)
            body.columnconfigure(1, weight=4, minsize=BIOMECH_CHART_COL_MIN_W)
            body.columnconfigure(2, weight=1, minsize=clinical_w)
            chart_row_min = max(80, min(BIOMECH_CHART_GRID_MIN_H, max(80, int(body_h))))
            body.rowconfigure(0, weight=1, minsize=chart_row_min)
            body.rowconfigure(1, weight=0, minsize=0)
            body.rowconfigure(2, weight=0, minsize=0)
            sidebar.grid(
                row=0,
                column=0,
                columnspan=1,
                sticky="nsew",
                padx=(0, PAD_XS),
                pady=0,
            )
            chart.grid(
                row=0, column=1, columnspan=1, sticky="nsew", padx=0, pady=0
            )
            clinical.grid(
                row=0,
                column=2,
                columnspan=1,
                sticky="nsew",
                padx=(PAD_XS, 0),
                pady=0,
            )
    except tk.TclError:
        pass

    # On short three-column workspaces the right clinical rail already carries
    # the interpretation. Hide the duplicate chart footer so the graph keeps a
    # useful height; restore it automatically on taller or stacked layouts.
    footer = getattr(gui, "_biomech_chart_footer", None)
    hide_footer = not stack and body_h < 560
    if footer is not None:
        try:
            if hide_footer:
                footer.grid_remove()
            else:
                footer.grid(row=4, column=0, sticky="ew", pady=(PAD_XS, 0))
        except tk.TclError:
            pass
    note_row = getattr(gui, "_biomech_chart_note_row", None)
    if note_row is not None:
        try:
            note_row.grid(row=3, column=0, sticky="ew", pady=(PAD_XS, 0))
        except tk.TclError:
            pass
    gui._biomech_footer_compact_hidden = hide_footer
    gui._biomech_stacked = stack
    now_collapsed = bool(collapse_rails and not stack)
    # Narrower rails only — do not auto-collapse Key Findings / Interpretation.
    # Users expand, collapse, and drag the chart|clinical sash themselves.
    gui._biomech_rails_collapsed = now_collapsed

    sync = getattr(gui, "_sync_biomech_clinical_graph_alignment", None)
    if callable(sync):
        sync()

    clinical_wrap = max(160, (width if stack else clinical_w) - _BIOMECH_CLINICAL_WRAP_MARGIN)
    for lbl in (getattr(gui, "_biomech_clinical_labels", None) or {}).values():
        try:
            lbl.configure(wraplength=clinical_wrap)
        except tk.TclError:
            pass
    for hdr in (getattr(gui, "_biomech_clinical_headers", None) or {}).values():
        try:
            hdr.configure(wraplength=clinical_wrap)
        except tk.TclError:
            pass
    graph_lbl = getattr(gui, "lbl_biomech_clinical_graph", None)
    if graph_lbl is not None:
        try:
            graph_lbl.configure(wraplength=clinical_wrap)
        except tk.TclError:
            pass
    title = getattr(getattr(gui, "_biomech_clinical_section", None), "title_lbl", None)
    if title is not None:
        try:
            title.configure(
                wraplength=max(120, clinical_wrap - 12),
                justify=tk.LEFT,
                anchor="w",
            )
        except tk.TclError:
            pass
    disclaimer = getattr(gui, "lbl_biomech_clinical_disclaimer", None)
    if disclaimer is not None:
        try:
            disclaimer.configure(wraplength=clinical_wrap)
        except tk.TclError:
            pass
    empty_lbl = getattr(gui, "lbl_biomech_clinical_empty", None)
    if empty_lbl is not None:
        try:
            empty_lbl.configure(wraplength=clinical_wrap)
        except tk.TclError:
            pass

    footer_wrap = max(160, (width if stack else chart_w) - 32)
    interp = getattr(gui, "lbl_biomech_interpretation", None)
    if interp is not None:
        try:
            interp.configure(wraplength=footer_wrap)
        except tk.TclError:
            pass
    stats_lbl = getattr(gui, "lbl_biomech_tab_stats", None)
    if stats_lbl is not None:
        try:
            stats_lbl.configure(wraplength=max(120, footer_wrap // 2))
        except tk.TclError:
            pass
    note_lbl = getattr(gui, "lbl_biomech_chart_note", None)
    if note_lbl is not None:
        try:
            note_lbl.configure(wraplength=footer_wrap)
        except tk.TclError:
            pass

    wrap = max(140, (width if stack else findings_w) - 28)
    for attr in ("lbl_biomech_video_checks", "lbl_advanced_evidence", "lbl_biomech_rom"):
        lbl = getattr(gui, attr, None)
        if lbl is None:
            continue
        try:
            lbl.configure(wraplength=wrap)
        except tk.TclError:
            pass

    from stablewalk.ui.theme import set_wraplength_safe

    title_wrap = max(96, (width if stack else findings_w) - 36)
    for kpi in (getattr(gui, "_biomech_kpi_cards", None) or {}).values():
        title_lbl = getattr(kpi, "title_lbl", None)
        if title_lbl is None:
            continue
        try:
            set_wraplength_safe(title_lbl, title_wrap)
            title_lbl.configure(anchor="nw", justify=tk.LEFT)
        except tk.TclError:
            pass

    gui._biomech_rail_widths = (findings_w, chart_w, clinical_w)

    # After the rails have a real size, stretch the interpretation overflow
    # canvas so section text is not clipped to the initial 1×1 window.
    try:
        body.after_idle(lambda: _sync_clinical_overflow(gui))
    except tk.TclError:
        _sync_clinical_overflow(gui)
    fit = getattr(gui, "_fit_biomech_canvas", None)
    if callable(fit):
        try:
            body.after_idle(fit)
        except (tk.TclError, AttributeError):
            try:
                fit()
            except Exception:
                pass


def reflow_biomech_workspace(gui: Any, *, width: int, height: int = 0) -> None:
    """Resize Key Findings / graph / interpretation with the window."""
    body_h = 0
    body = getattr(gui, "_biomech_body", None)
    if body is not None:
        try:
            body_h = int(body.winfo_height() or 0)
            measured = int(body.winfo_width() or 0)
            if measured > 80:
                width = measured
        except tk.TclError:
            body_h = 0
    if body_h <= 1 and int(height or 0) > 1:
        body_h = max(1, int(height) - 48)
    apply_biomech_workspace_geometry(gui, width=width, height=body_h)


__all__ = [
    "BIOMECH_CHART_COL_MIN_W",
    "BIOMECH_CHART_FIGSIZE",
    "BIOMECH_CLINICAL_MAX_W",
    "BIOMECH_CLINICAL_MIN_W",
    "BIOMECH_CLINICAL_WIDTH",
    "BIOMECH_FINDINGS_MAX_W",
    "BIOMECH_FINDINGS_MIN_W",
    "BIOMECH_FINDINGS_WIDTH",
    "BIOMECH_RAIL_HEADER_ONLY_W",
    "BIOMECH_INTERPRETATION_EMPTY",
    "BIOMECH_STACK_BELOW_W",
    "apply_biomech_workspace_geometry",
    "biomech_balanced_column_widths",
    "biomech_three_col_min_width",
    "build_biomechanics_tab",
    "reflow_biomech_workspace",
    "update_biomechanics_summary_card",
    "update_biomech_workspace_panels",
    "_clinical_display_is_empty",
    "_clinical_paragraph",
    "_workspace_tab_label",
]
