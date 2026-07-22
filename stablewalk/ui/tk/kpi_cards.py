"""Modern KPI metric cards — shared design language for every dashboard page.

Each card shows: name, large value, unit, trend, quality color, and tooltip.
Presentation only — callers keep supplying the same underlying metrics.
"""

from __future__ import annotations

import re
import tkinter as tk
from dataclasses import dataclass
from typing import Literal

from stablewalk.ui.summary_metric_style import (
    InterpretationLevel,
    metric_visual_style,
)
from stablewalk.ui.theme import (
    BORDER,
    CARD_BORDER_WIDTH,
    ELEVATED,
    FONT_BADGE,
    FONT_KPI_TITLE,
    FONT_KPI_TREND,
    FONT_KPI_UNIT,
    FONT_KPI_VALUE,
    FONT_KPI_VALUE_SM,
    MUTED,
    MUTED_DIM,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    create_tooltip,
)

Quality = InterpretationLevel

_NUMBER_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")

_TREND_UP = "▲"
_TREND_DOWN = "▼"
_TREND_FLAT = "●"


@dataclass
class KpiCard:
    """Handles for one KPI card instance."""

    key: str
    frame: tk.Frame
    accent: tk.Frame
    title_lbl: tk.Label
    value_lbl: tk.Label
    unit_lbl: tk.Label
    trend_lbl: tk.Label
    quality_lbl: tk.Label
    bar_track: tk.Frame | None = None
    bar_fill: tk.Frame | None = None
    previous_numeric: float | None = None
    last_paint: tuple | None = None


def create_kpi_card(
    parent: tk.Misc,
    *,
    key: str,
    title: str,
    tooltip: str | None = None,
    compact: bool = False,
    show_bar: bool = False,
    fill: bool = False,
) -> KpiCard:
    """Build one elevated KPI card matching the StableWalk dashboard language."""
    outer = tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=CARD_BORDER_WIDTH,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    if fill:
        outer.pack(fill=tk.BOTH, expand=True, pady=(0, PAD_SM))
    else:
        outer.pack(fill=tk.X, pady=(0, PAD_SM))

    accent = tk.Frame(outer, bg=BORDER, width=3, highlightthickness=0)
    accent.pack(side=tk.LEFT, fill=tk.Y)
    accent.pack_propagate(False)

    body = tk.Frame(outer, bg=ELEVATED, highlightthickness=0)
    body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=PAD_SM, pady=PAD_XS + 1)

    header = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
    header.pack(fill=tk.X)

    title_lbl = tk.Label(
        header,
        text=title,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_KPI_TITLE,
        anchor="w",
    )
    title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

    trend_lbl = tk.Label(
        header,
        text=_TREND_FLAT,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_KPI_TREND,
        anchor="e",
        width=2,
    )
    trend_lbl.pack(side=tk.RIGHT)

    value_row = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
    value_row.pack(fill=tk.X, pady=(2, 0))

    value_font = FONT_KPI_VALUE_SM if compact else FONT_KPI_VALUE
    value_lbl = tk.Label(
        value_row,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=value_font,
        anchor="w",
    )
    value_lbl.pack(side=tk.LEFT)

    unit_lbl = tk.Label(
        value_row,
        text="",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_KPI_UNIT,
        anchor="sw",
    )
    unit_lbl.pack(side=tk.LEFT, padx=(6, 0), pady=(0, 2))

    quality_lbl = tk.Label(
        body,
        text="",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_BADGE,
        anchor="w",
    )
    quality_lbl.pack(fill=tk.X, pady=(1, 0))

    bar_track = bar_fill = None
    if show_bar:
        bar_track = tk.Frame(body, bg=BORDER, height=4, highlightthickness=0)
        bar_track.pack(fill=tk.X, pady=(4, 0))
        bar_track.pack_propagate(False)
        bar_fill = tk.Frame(bar_track, bg=BORDER, height=4, highlightthickness=0)
        bar_fill.place(relx=0, rely=0, relheight=1.0, relwidth=0.0, anchor="nw")

    card = KpiCard(
        key=key,
        frame=outer,
        accent=accent,
        title_lbl=title_lbl,
        value_lbl=value_lbl,
        unit_lbl=unit_lbl,
        trend_lbl=trend_lbl,
        quality_lbl=quality_lbl,
        bar_track=bar_track,
        bar_fill=bar_fill,
    )
    if tooltip:
        set_kpi_tooltip(card, tooltip)
    return card


def create_kpi_grid_card(
    parent: tk.Misc,
    *,
    key: str,
    title: str,
    tooltip: str | None = None,
    compact: bool = True,
    show_bar: bool = False,
) -> KpiCard:
    """KPI card intended for grid placement (caller manages ``.grid``)."""
    outer = tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=CARD_BORDER_WIDTH,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )

    accent = tk.Frame(outer, bg=BORDER, width=3, highlightthickness=0)
    accent.pack(side=tk.LEFT, fill=tk.Y)
    accent.pack_propagate(False)

    body = tk.Frame(outer, bg=ELEVATED, highlightthickness=0)
    body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=PAD_SM, pady=PAD_XS + 1)

    header = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
    header.pack(fill=tk.X)

    title_lbl = tk.Label(
        header,
        text=title,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_KPI_TITLE,
        anchor="w",
    )
    title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

    trend_lbl = tk.Label(
        header,
        text=_TREND_FLAT,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_KPI_TREND,
        anchor="e",
        width=2,
    )
    trend_lbl.pack(side=tk.RIGHT)

    value_row = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
    value_row.pack(fill=tk.X, pady=(2, 0))

    value_font = FONT_KPI_VALUE_SM if compact else FONT_KPI_VALUE
    value_lbl = tk.Label(
        value_row,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=value_font,
        anchor="w",
    )
    value_lbl.pack(side=tk.LEFT)

    unit_lbl = tk.Label(
        value_row,
        text="",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_KPI_UNIT,
        anchor="sw",
    )
    unit_lbl.pack(side=tk.LEFT, padx=(6, 0), pady=(0, 2))

    quality_lbl = tk.Label(
        body,
        text="",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_BADGE,
        anchor="w",
    )
    quality_lbl.pack(fill=tk.X, pady=(1, 0))

    bar_track = bar_fill = None
    if show_bar:
        bar_track = tk.Frame(body, bg=BORDER, height=4, highlightthickness=0)
        bar_track.pack(fill=tk.X, pady=(4, 0))
        bar_track.pack_propagate(False)
        bar_fill = tk.Frame(bar_track, bg=BORDER, height=4, highlightthickness=0)
        bar_fill.place(relx=0, rely=0, relheight=1.0, relwidth=0.0, anchor="nw")

    card = KpiCard(
        key=key,
        frame=outer,
        accent=accent,
        title_lbl=title_lbl,
        value_lbl=value_lbl,
        unit_lbl=unit_lbl,
        trend_lbl=trend_lbl,
        quality_lbl=quality_lbl,
        bar_track=bar_track,
        bar_fill=bar_fill,
    )
    if tooltip:
        set_kpi_tooltip(card, tooltip)
    return card


def set_kpi_tooltip(card: KpiCard, text: str) -> None:
    """Attach / refresh the shared hover explanation on the whole card."""
    tip = (text or "").strip() or "Metric updates after analysis"
    for widget in (
        card.frame,
        card.title_lbl,
        card.value_lbl,
        card.unit_lbl,
        card.trend_lbl,
        card.quality_lbl,
    ):
        create_tooltip(widget, tip, wraplength=340)


def split_value_unit(display: str) -> tuple[str, str]:
    """Split a combined readout like ``71/100`` or ``63 steps/min`` into parts."""
    text = (display or "").strip()
    if not text or text in ("—", "-", "N/A"):
        return text or "—", ""

    if "/" in text and text.count("/") == 1:
        left, right = text.split("/", 1)
        if _NUMBER_RE.fullmatch(left.strip()) and right.strip():
            return left.strip(), f"/{right.strip()}"

    match = _NUMBER_RE.match(text)
    if not match:
        return text, ""
    value = match.group(1)
    rest = text[match.end() :].strip()
    return value, rest


def _trend_for(
    card: KpiCard,
    numeric: float | None,
    *,
    quality: Quality,
) -> tuple[str, str]:
    """Return (glyph, color) for the trend chip — only after a real numeric delta."""
    del quality  # Quality is shown on the accent/bar; do not invent trends from it.
    if numeric is None:
        return "", MUTED
    prev = card.previous_numeric
    if prev is None:
        return "", MUTED
    delta = numeric - prev
    if abs(delta) < 1e-6:
        return _TREND_FLAT, MUTED
    if delta > 0:
        return _TREND_UP, MUTED
    return _TREND_DOWN, MUTED


def update_kpi_card(
    card: KpiCard,
    *,
    value: str,
    unit: str = "",
    quality: Quality = "neutral",
    tooltip: str | None = None,
    fraction: float | None = None,
    numeric: float | None = None,
    available: bool = True,
) -> None:
    """Refresh value, unit, trend, quality color, optional bar, and tooltip."""
    display_value = value if value else "—"
    paint_key = (
        display_value,
        unit or "",
        quality,
        available,
        None if fraction is None else round(float(fraction), 4),
        tooltip,
        None if numeric is None else round(float(numeric), 6),
    )
    if card.last_paint == paint_key:
        return
    card.last_paint = paint_key

    style = metric_visual_style(quality if available else "unavailable")
    if not available and display_value not in ("—", "N/A"):
        # Keep the provided explanation text; mute the styling.
        pass

    card.value_lbl.configure(
        text=display_value,
        fg=style.value_fg if available else MUTED,
    )
    card.unit_lbl.configure(text=unit or "", fg=MUTED if available else MUTED_DIM)
    card.quality_lbl.configure(
        text=style.status_label if available and style.status_label else "",
        fg=style.value_fg if available else MUTED_DIM,
    )
    card.accent.configure(bg=style.border if available else BORDER)
    card.frame.configure(
        highlightbackground=style.border if available else BORDER,
        highlightcolor=style.border if available else BORDER,
    )

    glyph, trend_color = _trend_for(
        card,
        numeric if available else None,
        quality=quality if available else "unavailable",
    )
    card.trend_lbl.configure(text=glyph, fg=trend_color)

    if card.bar_fill is not None:
        frac = 0.0 if fraction is None or not available else max(0.0, min(1.0, fraction))
        card.bar_fill.configure(bg=style.value_fg if available and frac > 0 else BORDER)
        card.bar_fill.place_configure(relwidth=frac)

    if tooltip is not None:
        set_kpi_tooltip(card, tooltip)

    if available and numeric is not None:
        card.previous_numeric = float(numeric)


def parse_numeric(value: str | float | None) -> float | None:
    """Best-effort numeric extraction for trend tracking."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER_RE.search(str(value))
    return float(match.group(1)) if match else None


__all__ = [
    "KpiCard",
    "Quality",
    "create_kpi_card",
    "create_kpi_grid_card",
    "parse_numeric",
    "set_kpi_tooltip",
    "split_value_unit",
    "update_kpi_card",
]
