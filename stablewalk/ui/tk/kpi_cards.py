"""Modern KPI metric cards — shared design language for every dashboard page.

Visual hierarchy (presentation only — no metrics removed):
  primary   — large hero cards (examiner attention)
  secondary — compact cards (supporting clinical context)
  technical — lighter chrome; typically housed in expandable Details panels

``supporting`` is accepted as a legacy alias of ``technical``.
"""

from __future__ import annotations

import re
import tkinter as tk
from dataclasses import dataclass
from typing import Literal

from stablewalk.ui.summary_metric_style import (
    InterpretationLevel,
    kpi_status_label,
    metric_visual_style,
)
from stablewalk.ui.theme import (
    BORDER,
    CARD_BORDER_WIDTH,
    ELEVATED,
    FONT_BADGE,
    FONT_KPI_GROUP,
    FONT_KPI_HERO,
    FONT_KPI_SUPPORTING,
    FONT_KPI_TITLE,
    FONT_KPI_TITLE_PRIMARY,
    FONT_KPI_TREND,
    FONT_KPI_UNIT,
    FONT_KPI_VALUE,
    FONT_KPI_VALUE_SM,
    FONT_UI_SM,
    GLYPH_TREND_DOWN,
    GLYPH_TREND_FLAT,
    GLYPH_TREND_UP,
    KPI_ACCENT_PRIMARY,
    KPI_ACCENT_SECONDARY,
    KPI_ACCENT_SUPPORTING,
    KPI_CARD_GAP,
    MUTED,
    MUTED_DIM,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    SURFACE,
    TEXT,
    TEXT_SECONDARY,
    create_tooltip,
    disclosure_label,
)

Quality = InterpretationLevel
KpiTier = Literal["primary", "secondary", "technical", "supporting"]


def normalize_kpi_tier(tier: str | None) -> Literal["primary", "secondary", "technical"]:
    """Map caller tiers onto the three examiner-facing groups."""
    key = (tier or "secondary").strip().lower()
    if key in ("primary",):
        return "primary"
    if key in ("technical", "supporting"):
        return "technical"
    return "secondary"


_NUMBER_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")

_TREND_UP = GLYPH_TREND_UP
_TREND_DOWN = GLYPH_TREND_DOWN
_TREND_FLAT = GLYPH_TREND_FLAT


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
    tier: Literal["primary", "secondary", "technical"] = "secondary"


@dataclass
class TechnicalDetailsPanel:
    """Collapsible host for technical metrics (all information retained)."""

    frame: tk.Frame
    toggle: tk.Button
    content: tk.Frame
    expanded: bool = False


def _tier_chrome(tier: str, *, compact: bool) -> dict:
    """Surface / type / accent for a visual priority tier."""
    resolved = normalize_kpi_tier(tier)
    if resolved == "primary":
        return {
            "bg": ELEVATED,
            "title_font": FONT_KPI_TITLE_PRIMARY,
            "title_fg": TEXT_SECONDARY,
            "value_font": FONT_KPI_HERO if not compact else FONT_KPI_VALUE,
            "value_fg": TEXT,
            "unit_fg": MUTED,
            "accent_w": KPI_ACCENT_PRIMARY,
            "padx": PAD_MD,
            "pady": PAD_SM,
            "show_trend": True,
            "show_quality": True,
            "bar_h": 4,
        }
    if resolved == "technical":
        return {
            "bg": SURFACE,
            "title_font": FONT_KPI_TITLE,
            "title_fg": MUTED_DIM,
            "value_font": FONT_KPI_SUPPORTING if compact else FONT_KPI_VALUE_SM,
            "value_fg": TEXT_SECONDARY,
            "unit_fg": MUTED_DIM,
            "accent_w": KPI_ACCENT_SUPPORTING,
            "padx": PAD_SM,
            "pady": PAD_XS,
            "show_trend": False,
            "show_quality": False,
            "bar_h": 3,
        }
    return {
        "bg": ELEVATED,
        "title_font": FONT_KPI_TITLE,
        "title_fg": MUTED,
        "value_font": FONT_KPI_VALUE_SM if compact else FONT_KPI_VALUE,
        "value_fg": TEXT,
        "unit_fg": MUTED,
        "accent_w": KPI_ACCENT_SECONDARY,
        "padx": PAD_SM,
        "pady": PAD_XS,
        "show_trend": True,
        "show_quality": True,
        "bar_h": 4,
    }


def create_metric_group_label(
    parent: tk.Misc,
    text: str,
    *,
    bg: str = PANEL,
    pack: bool = True,
) -> tk.Label:
    """Small group caption for KPI stacks (Primary / Secondary / Technical)."""
    lbl = tk.Label(
        parent,
        text=text.strip().title() if text.isupper() else text,
        bg=bg,
        fg=MUTED,
        font=FONT_KPI_GROUP,
        anchor="w",
    )
    if pack:
        lbl.pack(fill=tk.X, pady=(KPI_CARD_GAP, PAD_XS))
    return lbl


def create_technical_details_panel(
    parent: tk.Misc,
    *,
    title: str = "Technical details",
    bg: str = PANEL,
    start_expanded: bool = False,
    pack: bool = True,
) -> TechnicalDetailsPanel:
    """
    Expandable Details host for technical metrics.

    All child metrics remain in the widget tree; only visibility toggles.
    """
    shell = tk.Frame(parent, bg=bg, highlightthickness=0)
    if pack:
        shell.pack(fill=tk.X, pady=(PAD_SM, 0))

    header = tk.Frame(shell, bg=bg, highlightthickness=0)
    header.pack(fill=tk.X)

    state = {"expanded": bool(start_expanded)}

    def _label() -> str:
        return disclosure_label(title, expanded=state["expanded"])

    content = tk.Frame(shell, bg=bg, highlightthickness=0)

    def _sync() -> None:
        toggle.configure(text=_label())
        if state["expanded"]:
            content.pack(fill=tk.X, pady=(PAD_XS, 0))
        else:
            content.pack_forget()

    def _toggle() -> None:
        state["expanded"] = not state["expanded"]
        panel.expanded = state["expanded"]
        _sync()

    toggle = tk.Button(
        header,
        text=_label(),
        bg=bg,
        fg=TEXT_SECONDARY,
        font=FONT_UI_SM,
        relief=tk.FLAT,
        bd=0,
        activebackground=bg,
        activeforeground=TEXT,
        anchor="w",
        cursor="hand2",
        command=_toggle,
        highlightthickness=0,
        padx=0,
        pady=PAD_XS // 2,
    )
    toggle.pack(side=tk.LEFT, fill=tk.X, expand=True)
    create_tooltip(
        toggle,
        "Show or hide technical metrics. All values remain available — nothing is removed.",
        wraplength=320,
    )

    panel = TechnicalDetailsPanel(
        frame=shell,
        toggle=toggle,
        content=content,
        expanded=bool(start_expanded),
    )
    _sync()
    return panel


def _build_kpi_shell(
    parent: tk.Misc,
    *,
    key: str,
    title: str,
    tooltip: str | None,
    compact: bool,
    show_bar: bool,
    details_key: str | None,
    tier: str,
    pack: bool,
    fill: bool,
    title_wrap_lines: int = 0,
) -> KpiCard:
    resolved = normalize_kpi_tier(tier)
    # Primary = large cards (unless caller forces compact); secondary/technical = compact.
    use_compact = bool(compact) if resolved == "primary" else True

    chrome = _tier_chrome(resolved, compact=use_compact)
    bg = chrome["bg"]

    outer = tk.Frame(
        parent,
        bg=bg,
        highlightthickness=CARD_BORDER_WIDTH,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    if pack:
        if fill:
            outer.pack(fill=tk.BOTH, expand=True, pady=(0, KPI_CARD_GAP))
        else:
            outer.pack(fill=tk.X, pady=(0, KPI_CARD_GAP))

    accent = tk.Frame(outer, bg=BORDER, width=chrome["accent_w"], highlightthickness=0)
    accent.pack(side=tk.LEFT, fill=tk.Y)
    accent.pack_propagate(False)

    body = tk.Frame(outer, bg=bg, highlightthickness=0)
    body.pack(
        side=tk.LEFT,
        fill=tk.BOTH,
        expand=True,
        padx=chrome["padx"],
        pady=chrome["pady"],
    )

    from stablewalk.ui.metric_details import resolve_metric_key
    from stablewalk.ui.theme import set_wraplength_safe
    from stablewalk.ui.tk.metric_details_dialog import attach_metric_details_control

    wrap_title = max(0, int(title_wrap_lines or 0)) > 0

    header = tk.Frame(body, bg=bg, highlightthickness=0)
    header.pack(fill=tk.X)

    title_lbl = tk.Label(
        header,
        text=title,
        bg=bg,
        fg=chrome["title_fg"],
        font=chrome["title_font"],
        anchor="nw" if wrap_title else "w",
        justify=tk.LEFT,
        wraplength=160 if wrap_title else 0,
    )

    value_row = tk.Frame(body, bg=bg, highlightthickness=0)
    value_row.pack(fill=tk.X, pady=(PAD_XS // 2, 0))

    trend_lbl = tk.Label(
        value_row if wrap_title else header,
        text=_TREND_FLAT if chrome["show_trend"] else "",
        bg=bg,
        fg=MUTED,
        font=FONT_KPI_TREND,
        anchor="e",
        width=2 if chrome["show_trend"] else 0,
    )

    dk = details_key or key
    if resolve_metric_key(dk):
        details_parent = value_row if wrap_title else header
        attach_metric_details_control(details_parent, dk, bg=bg, side=tk.RIGHT)

    if wrap_title:
        title_lbl.pack(fill=tk.X, anchor="w")
        if chrome["show_trend"]:
            trend_lbl.pack(side=tk.RIGHT, padx=(0, PAD_XS))

        def _sync_title_wrap(_event: object | None = None) -> None:
            try:
                width = int(header.winfo_width() or 0)
            except tk.TclError:
                return
            if width < 40:
                return
            # Full inner width so long metric names wrap instead of clipping.
            set_wraplength_safe(title_lbl, max(96, width - 4))

        header.bind("<Configure>", _sync_title_wrap, add="+")
    else:
        title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if chrome["show_trend"]:
            trend_lbl.pack(side=tk.RIGHT)

    value_lbl = tk.Label(
        value_row,
        text="—",
        bg=bg,
        fg=chrome["value_fg"],
        font=chrome["value_font"],
        anchor="w",
    )
    value_lbl.pack(side=tk.LEFT)

    unit_lbl = tk.Label(
        value_row,
        text="",
        bg=bg,
        fg=chrome["unit_fg"],
        font=FONT_KPI_UNIT,
        anchor="sw",
    )
    unit_lbl.pack(side=tk.LEFT, padx=(PAD_SM - 2, 0), pady=(0, PAD_XS // 2))

    quality_lbl = tk.Label(
        body,
        text="",
        bg=bg,
        fg=MUTED,
        font=FONT_BADGE,
        anchor="w",
    )
    if chrome["show_quality"]:
        quality_lbl.pack(fill=tk.X, pady=(PAD_XS // 4, 0))

    bar_track = bar_fill = None
    if show_bar:
        bar_track = tk.Frame(body, bg=BORDER, height=chrome["bar_h"], highlightthickness=0)
        bar_track.pack(fill=tk.X, pady=(PAD_XS, 0))
        bar_track.pack_propagate(False)
        bar_fill = tk.Frame(bar_track, bg=BORDER, height=chrome["bar_h"], highlightthickness=0)
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
        tier=resolved,
    )
    if tooltip:
        set_kpi_tooltip(card, tooltip)
    return card


def create_kpi_card(
    parent: tk.Misc,
    *,
    key: str,
    title: str,
    tooltip: str | None = None,
    compact: bool = False,
    show_bar: bool = False,
    fill: bool = False,
    details_key: str | None = None,
    tier: str = "secondary",
    title_wrap_lines: int = 0,
) -> KpiCard:
    """Build one KPI card. Primary = large; secondary = compact; technical = light."""
    return _build_kpi_shell(
        parent,
        key=key,
        title=title,
        tooltip=tooltip,
        compact=compact,
        show_bar=show_bar,
        details_key=details_key,
        tier=tier,
        pack=True,
        fill=fill,
        title_wrap_lines=title_wrap_lines,
    )


def create_kpi_grid_card(
    parent: tk.Misc,
    *,
    key: str,
    title: str,
    tooltip: str | None = None,
    compact: bool = True,
    show_bar: bool = False,
    details_key: str | None = None,
    tier: str = "secondary",
) -> KpiCard:
    """KPI card intended for grid placement (caller manages ``.grid``)."""
    return _build_kpi_shell(
        parent,
        key=key,
        title=title,
        tooltip=tooltip,
        compact=compact,
        show_bar=show_bar,
        details_key=details_key,
        tier=tier,
        pack=False,
        fill=False,
    )


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
    del quality
    if card.tier == "technical":
        return "", MUTED_DIM
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
        card.tier,
    )
    if card.last_paint == paint_key:
        return
    card.last_paint = paint_key

    style = metric_visual_style(quality if available else "unavailable")
    status_text = kpi_status_label(
        quality if available else "unavailable",
        key=card.key,
    )
    chrome = _tier_chrome(card.tier, compact=True)
    value_fg = style.value_fg if available else MUTED
    if card.tier == "technical" and available and quality in ("neutral", "unavailable"):
        value_fg = chrome["value_fg"]

    card.value_lbl.configure(text=display_value, fg=value_fg)
    card.unit_lbl.configure(
        text=unit or "",
        fg=chrome["unit_fg"] if available else MUTED_DIM,
    )
    # Subtle value flash — only when the displayed number/text actually changed.
    try:
        from stablewalk.ui.tk.ui_motion import flash_metric_value

        flash_metric_value(
            None,
            card.value_lbl,
            settle_fg=value_fg,
            key=f"kpi:{card.key}",
        )
    except Exception:
        pass
    if card.tier != "technical":
        card.quality_lbl.configure(
            text=status_text,
            fg=style.value_fg if available else MUTED_DIM,
        )
    else:
        card.quality_lbl.configure(text="", fg=MUTED_DIM)

    accent_bg = style.border if available else BORDER
    if card.tier == "technical" and available:
        accent_bg = BORDER
    card.accent.configure(bg=accent_bg)
    card.frame.configure(
        highlightbackground=style.border if available and card.tier != "technical" else BORDER,
        highlightcolor=style.border if available and card.tier != "technical" else BORDER,
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

    if not available and not (tooltip or "").strip():
        tooltip = (
            "N/A — this value was not computed for the current session. "
            "It is not a score of 0 and is not a clinical finding."
        )
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
    "KpiTier",
    "Quality",
    "TechnicalDetailsPanel",
    "create_kpi_card",
    "create_kpi_grid_card",
    "create_metric_group_label",
    "create_technical_details_panel",
    "normalize_kpi_tier",
    "parse_numeric",
    "set_kpi_tooltip",
    "split_value_unit",
    "update_kpi_card",
]
