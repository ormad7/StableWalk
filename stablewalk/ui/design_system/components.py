"""Reusable clinical-dashboard UI components (presentation only)."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Any, Callable, Literal

from stablewalk.ui.design_system import tokens as T

StatusTone = Literal["ok", "warn", "fail", "info", "neutral"]


def _tooltip(widget: tk.Misc, text: str, *, wraplength: int = 360) -> None:
    """Lazy import avoids theme ↔ design_system circular import at load time."""
    from stablewalk.ui.theme import create_tooltip

    create_tooltip(widget, text, wraplength=wraplength)

_STATUS_FG: dict[StatusTone, str] = {
    "ok": T.STATUS_OK,
    "warn": T.STATUS_WARN,
    "fail": T.STATUS_FAIL,
    "info": T.STATUS_INFO,
    "neutral": T.MUTED,
}

_STATUS_BG: dict[StatusTone, str] = {
    "ok": "#1a2e24",
    "warn": "#2e2818",
    "fail": "#2e1a1c",
    "info": "#1a2530",
    "neutral": T.CONTROL,
}


def _tone_fg(tone: StatusTone | str | None) -> str:
    key = (tone or "neutral").strip().lower()  # type: ignore[union-attr]
    if key in _STATUS_FG:
        return _STATUS_FG[key]  # type: ignore[index]
    return T.MUTED


# ── SectionHeader ─────────────────────────────────────────────────────────


@dataclass
class SectionHeaderHandles:
    frame: tk.Frame
    title_lbl: tk.Label
    subtitle_lbl: tk.Label | None


def SectionHeader(
    parent: tk.Misc,
    title: str,
    *,
    subtitle: str = "",
    bg: str = T.PANEL,
    tooltip: str = "",
) -> SectionHeaderHandles:
    """Section title with optional muted supporting line."""
    frame = tk.Frame(parent, bg=bg, highlightthickness=0)
    frame.columnconfigure(0, weight=1)

    title_lbl = tk.Label(
        frame,
        text=title,
        bg=bg,
        fg=T.TEXT,
        font=T.FONT_SECTION_TITLE,
        anchor="w",
    )
    title_lbl.grid(row=0, column=0, sticky="ew")

    subtitle_lbl = None
    if subtitle:
        subtitle_lbl = tk.Label(
            frame,
            text=subtitle,
            bg=bg,
            fg=T.MUTED,
            font=T.FONT_SUPPORTING,
            anchor="w",
        )
        subtitle_lbl.grid(row=1, column=0, sticky="ew", pady=(T.SPACE_1, 0))

    if tooltip:
        _tooltip(title_lbl, tooltip, wraplength=360)

    return SectionHeaderHandles(frame=frame, title_lbl=title_lbl, subtitle_lbl=subtitle_lbl)


# ── StatusBadge ───────────────────────────────────────────────────────────


@dataclass
class StatusBadgeHandles:
    frame: tk.Frame
    label: tk.Label


def StatusBadge(
    parent: tk.Misc,
    text: str,
    *,
    tone: StatusTone = "neutral",
    bg_parent: str = T.PANEL,
    tooltip: str = "",
) -> StatusBadgeHandles:
    """Compact semantic status chip (ok / warn / fail / info)."""
    chip_bg = _STATUS_BG.get(tone, T.CONTROL)
    chip_fg = _tone_fg(tone)
    frame = tk.Frame(parent, bg=chip_bg, highlightthickness=0)
    label = tk.Label(
        frame,
        text=text,
        bg=chip_bg,
        fg=chip_fg,
        font=T.FONT_META,
        padx=T.SPACE_2,
        pady=T.SPACE_1,
        anchor="center",
    )
    label.pack()
    if tooltip:
        _tooltip(frame, tooltip, wraplength=320)
        _tooltip(label, tooltip, wraplength=320)
    return StatusBadgeHandles(frame=frame, label=label)


def update_status_badge(
    handles: StatusBadgeHandles, text: str, *, tone: StatusTone = "neutral"
) -> None:
    chip_bg = _STATUS_BG.get(tone, T.CONTROL)
    chip_fg = _tone_fg(tone)
    try:
        handles.frame.configure(bg=chip_bg)
        handles.label.configure(text=text, bg=chip_bg, fg=chip_fg)
    except tk.TclError:
        pass


# ── MetricCard ────────────────────────────────────────────────────────────


def MetricCard(
    parent: tk.Misc,
    *,
    title: str,
    unit: str = "",
    description: str = "",
    details_key: str | None = None,
    gui: Any | None = None,
    tooltip: str = "",
    compact: bool = False,
) -> dict[str, Any]:
    """
    Clinical metric card (Overview Session Metrics).

    Returns the legacy dict keys (frame, accent, title, value, unit, desc, status)
    so existing refresh paths keep working. ``compact=True`` densifies padding for
    the Overview single-row KPI strip.
    """
    from stablewalk.ui.tk.metric_details_dialog import attach_metric_details_control

    pad_x = T.PAD_SM if compact else T.PAD_MD
    pad_y = (T.PAD_XS, T.PAD_SM) if compact else (T.PAD_MD, T.PAD_MD)
    value_gap = (2, 0) if compact else (T.PAD_XS, T.PAD_XS)
    value_font = T.FONT_METRIC_MD if compact else T.FONT_METRIC

    frame = tk.Frame(
        parent,
        bg=T.ELEVATED,
        highlightthickness=0,
    )
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)
    if compact:
        # Keep a readable height without locking width — propagate=False
        # plus a natural width made cards overlap in a shrinking 4-column row.
        try:
            frame.rowconfigure(1, minsize=52)
        except tk.TclError:
            pass

    # Compact Overview KPIs keep a thin top accent for laboratory alignment.
    accent = tk.Frame(
        frame,
        bg=T.ACCENT if compact else T.BORDER_SUBTLE,
        height=T.CARD_ACCENT_HEIGHT,
        highlightthickness=0,
    )
    accent.grid(row=0, column=0, sticky="ew")

    body = tk.Frame(frame, bg=T.ELEVATED, highlightthickness=0)
    body.grid(
        row=1,
        column=0,
        sticky="nsew",
        padx=pad_x,
        pady=pad_y,
    )
    body.columnconfigure(0, weight=1)
    if not compact:
        body.rowconfigure(2, weight=1)

    title_row = tk.Frame(body, bg=T.ELEVATED, highlightthickness=0)
    title_row.grid(row=0, column=0, sticky="ew")
    title_row.columnconfigure(0, weight=1)
    title_lbl = tk.Label(
        title_row,
        text=title,
        bg=T.ELEVATED,
        fg=T.MUTED,
        font=T.FONT_SUPPORTING_SEMIBOLD if compact else T.FONT_CARD_TITLE,
        anchor="w",
    )
    title_lbl.grid(row=0, column=0, sticky="ew")
    if compact:
        try:
            title_lbl.configure(wraplength=140)
        except tk.TclError:
            pass
    if details_key:
        details_host = tk.Frame(title_row, bg=T.ELEVATED, highlightthickness=0)
        details_host.grid(row=0, column=1, sticky="e")
        attach_metric_details_control(
            details_host,
            details_key,
            bg=T.ELEVATED,
            side=tk.RIGHT,
            gui=gui,
            overview_kpi=bool(compact),
        )

    value_row = tk.Frame(body, bg=T.ELEVATED, highlightthickness=0)
    value_row.grid(row=1, column=0, sticky="ew", pady=value_gap)
    value_lbl = tk.Label(
        value_row,
        text="—",
        bg=T.ELEVATED,
        fg=T.TEXT,
        font=value_font,
        anchor="w",
    )
    value_lbl.pack(side=tk.LEFT, anchor="s")
    unit_lbl = tk.Label(
        value_row,
        text=unit,
        bg=T.ELEVATED,
        fg=T.MUTED,
        font=T.FONT_META,
        anchor="sw",
    )
    unit_lbl.pack(
        side=tk.LEFT,
        padx=(T.SPACE_1, 0),
        pady=(0, 0 if compact else T.SPACE_1),
    )

    desc_lbl = tk.Label(
        body,
        text=description,
        bg=T.ELEVATED,
        fg=T.MUTED,
        font=T.FONT_SUPPORTING,
        anchor="nw",
        justify=tk.LEFT,
        wraplength=160,
    )
    if not compact:
        desc_lbl.grid(row=2, column=0, sticky="new", pady=(T.PAD_XS, T.PAD_SM))

    status_lbl = tk.Label(
        body,
        text="",
        bg=T.ELEVATED,
        fg=T.MUTED,
        font=T.FONT_META,
        anchor="w",
    )
    if not compact:
        status_lbl.grid(row=3, column=0, sticky="ew")

    tip = tooltip or description
    if tip:
        _tooltip(frame, tip, wraplength=360)
        _tooltip(value_lbl, tip, wraplength=360)

    return {
        "frame": frame,
        "accent": accent,
        "title": title_lbl,
        "value": value_lbl,
        "unit": unit_lbl,
        "desc": desc_lbl,
        "status": status_lbl,
    }


# ── EmptyState ────────────────────────────────────────────────────────────


def EmptyState(
    parent: tk.Misc,
    text: str,
    *,
    bg: str = T.PANEL,
    padx: int = T.SPACE_3,
    pady: int = T.SPACE_4,
    pack: bool = False,
) -> tk.Label:
    """Centered empty-state copy. Caller grids/packs unless ``pack=True``."""
    lbl = tk.Label(
        parent,
        text=text,
        bg=bg,
        fg=T.MUTED,
        font=T.FONT_BODY,
        anchor="center",
        justify=tk.CENTER,
        wraplength=440,
        padx=padx,
        pady=pady,
    )
    if pack:
        lbl.pack(fill=tk.BOTH, expand=True)
    return lbl


# ── WarningBanner ─────────────────────────────────────────────────────────


@dataclass
class WarningBannerHandles:
    frame: tk.Frame
    label: tk.Label


def WarningBanner(
    parent: tk.Misc,
    text: str,
    *,
    tone: StatusTone = "warn",
    tooltip: str = "",
) -> WarningBannerHandles:
    """Persistent scientific / clinical warning strip (never decorative-only)."""
    bar_bg = _STATUS_BG.get(tone, _STATUS_BG["warn"])
    bar_fg = _tone_fg(tone)
    frame = tk.Frame(parent, bg=bar_bg, highlightthickness=0)
    label = tk.Label(
        frame,
        text=text,
        bg=bar_bg,
        fg=T.TEXT,
        font=T.FONT_SUPPORTING,
        anchor="w",
        justify=tk.LEFT,
        wraplength=720,
        padx=T.PAD_MD,
        pady=T.PAD_SM,
    )
    label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    if tooltip:
        _tooltip(label, tooltip, wraplength=400)
    return WarningBannerHandles(frame=frame, label=label)


# ── ExportButton ──────────────────────────────────────────────────────────


def ExportButton(
    parent: tk.Misc,
    text: str,
    command: Callable[[], None] | None = None,
    *,
    primary: bool = False,
    style: str | None = None,
) -> ttk.Button:
    """Consistent export / action button height and padding."""
    btn_style = style or ("ExportAccent.TButton" if primary else "Export.TButton")
    btn = ttk.Button(parent, text=text, style=btn_style, command=command)
    return btn


# ── CollapsibleSection ────────────────────────────────────────────────────


@dataclass
class CollapsibleSectionHandles:
    frame: tk.Frame
    header: tk.Frame
    toggle: tk.Button
    title_lbl: tk.Label
    body: tk.Frame
    expanded: bool


def CollapsibleSection(
    parent: tk.Misc,
    title: str,
    *,
    expanded: bool = False,
    bg: str = T.PANEL,
    on_toggle: Callable[[bool], None] | None = None,
) -> CollapsibleSectionHandles:
    """Disclosure section with shared ▶/▼ chrome."""
    from stablewalk.ui.theme import GLYPH_COLLAPSE, GLYPH_EXPAND

    frame = tk.Frame(parent, bg=bg, highlightthickness=0)
    frame.columnconfigure(0, weight=1)

    header = tk.Frame(frame, bg=bg, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(1, weight=1)

    state = {"expanded": bool(expanded)}

    body = tk.Frame(frame, bg=bg, highlightthickness=0)
    body.columnconfigure(0, weight=1)

    title_lbl = tk.Label(
        header,
        text=title,
        bg=bg,
        fg=T.TEXT_SECONDARY,
        font=T.FONT_CARD_TITLE,
        anchor="w",
    )
    title_lbl.grid(row=0, column=1, sticky="w", padx=(T.SPACE_1, 0))

    def _apply() -> None:
        arrow = GLYPH_COLLAPSE if state["expanded"] else GLYPH_EXPAND
        try:
            toggle.configure(text=arrow)
            if state["expanded"]:
                body.grid(row=1, column=0, sticky="nsew", pady=(T.SPACE_2, 0))
                frame.rowconfigure(1, weight=1)
            else:
                body.grid_remove()
                frame.rowconfigure(1, weight=0)
        except tk.TclError:
            pass

    def _click() -> None:
        state["expanded"] = not state["expanded"]
        handles.expanded = bool(state["expanded"])
        _apply()
        if on_toggle is not None:
            try:
                on_toggle(bool(state["expanded"]))
            except Exception:
                pass

    toggle = tk.Button(
        header,
        text=GLYPH_COLLAPSE if expanded else GLYPH_EXPAND,
        command=_click,
        bg=bg,
        fg=T.MUTED,
        activebackground=T.CONTROL,
        activeforeground=T.TEXT,
        relief=tk.FLAT,
        borderwidth=0,
        font=T.FONT_META,
        padx=T.SPACE_1,
        pady=0,
        cursor="hand2",
    )
    toggle.grid(row=0, column=0, sticky="w")

    handles = CollapsibleSectionHandles(
        frame=frame,
        header=header,
        toggle=toggle,
        title_lbl=title_lbl,
        body=body,
        expanded=bool(expanded),
    )

    def set_expanded(expanded_value: bool) -> None:
        want = bool(expanded_value)
        if bool(state["expanded"]) == want:
            handles.expanded = want
            return
        state["expanded"] = want
        handles.expanded = want
        _apply()

    handles.set_expanded = set_expanded  # type: ignore[attr-defined]
    _apply()
    return handles


# ── Quiet panel shell ─────────────────────────────────────────────────────


def QuietPanel(parent: tk.Misc, *, bg: str = T.ELEVATED) -> tk.Frame:
    """Borderless elevated surface — separation via background only."""
    return tk.Frame(parent, bg=bg, highlightthickness=0)


def SoftSeparator(parent: tk.Misc, *, bg: str = T.BORDER_SUBTLE) -> tk.Frame:
    """1 px horizontal rule for calm section breaks."""
    sep = tk.Frame(parent, bg=bg, height=T.SEPARATOR_HEIGHT, highlightthickness=0)
    sep.pack_propagate(False)
    sep.grid_propagate(False)
    return sep


__all__ = [
    "CollapsibleSection",
    "CollapsibleSectionHandles",
    "EmptyState",
    "ExportButton",
    "MetricCard",
    "QuietPanel",
    "SectionHeader",
    "SectionHeaderHandles",
    "SoftSeparator",
    "StatusBadge",
    "StatusBadgeHandles",
    "StatusTone",
    "WarningBanner",
    "WarningBannerHandles",
    "update_status_badge",
]
