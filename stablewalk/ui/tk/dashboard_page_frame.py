"""Shared page framing — one question per dashboard tab (presentation only)."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass

import stablewalk.ui.design_system.tokens as T
from stablewalk.ui.theme import (
    MUTED,
    PAD_MD,
    PAGE_HEADER_PADY,
    PAGE_HEADER_SUBTITLE_PADY,
    PANEL,
    SECTION_TITLE_PADY,
    TEXT,
)


@dataclass(frozen=True)
class PageBriefing:
    """Title + examiner-facing question for one notebook page."""

    key: str
    title: str
    question: str
    subtitle: str = ""


PAGE_BRIEFINGS: dict[str, PageBriefing] = {
    "overview": PageBriefing(
        key="overview",
        title="Overview",
        question="",
        subtitle="",
    ),
    "motion": PageBriefing(
        key="motion",
        title="Motion Analysis",
        question="",
        subtitle="Synced kinematics, 3D joint path, and spectral analysis",
    ),
    "biomechanics": PageBriefing(
        key="biomechanics",
        title="Biomechanics",
        question="",
        subtitle="COM, ground reaction, balance, contact, and stability — with interpretation",
    ),
    "results": PageBriefing(
        key="results",
        title="Results",
        question="",
        subtitle="Session findings, confidence, and exportable report",
    ),
    "compare": PageBriefing(
        key="compare",
        title="Compare",
        question="",
        subtitle="Research Video A/B · independent pose pipelines · gait-phase sync · laboratory compare",
    ),
    "advanced": PageBriefing(
        key="advanced",
        title="Advanced",
        question="",
        subtitle="Technical data, OpenSim, research outputs, and session artifacts",
    ),
}


def install_page_header(
    parent: tk.Misc,
    page_key: str,
    *,
    actions: tk.Misc | None = None,
    row: int = 0,
    pady: tuple[int, int] = PAGE_HEADER_PADY,
) -> tk.Frame:
    """
    Install a consistent page header: title, optional subtitle + actions.

    Returns the header frame (already gridded at ``row``).
    For Overview, ``gui._overview_session_badge_host`` is created for a compact
    session badge (filled by the clinical refresh).
    """
    briefing = PAGE_BRIEFINGS[page_key]
    header = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    header.grid(row=row, column=0, sticky="ew", pady=pady)
    header.columnconfigure(0, weight=1)

    title_row = tk.Frame(header, bg=PANEL, highlightthickness=0)
    title_row.grid(row=0, column=0, sticky="ew")
    title_row.columnconfigure(0, weight=1)

    tk.Label(
        title_row,
        text=briefing.title,
        bg=PANEL,
        fg=TEXT,
        font=T.FONT_PAGE_TITLE,
        anchor="w",
    ).grid(row=0, column=0, sticky="nsw")

    actions_host: tk.Frame | None = None
    if actions:
        # Must be a real child of title_row — Tk ``in_=`` still reserves page columns.
        actions_host = tk.Frame(title_row, bg=PANEL, highlightthickness=0)
        actions_host.grid(row=0, column=1, sticky="nse", padx=(PAD_MD, 0))
    header.actions_host = actions_host  # type: ignore[attr-defined]

    next_row = 1
    # Prefer a single muted subtitle — skip question+subtitle+separator stack.
    support = briefing.subtitle or briefing.question
    if support:
        tk.Label(
            header,
            text=support,
            bg=PANEL,
            fg=MUTED,
            font=T.FONT_SUPPORTING,
            anchor="w",
        ).grid(row=next_row, column=0, sticky="w", pady=PAGE_HEADER_SUBTITLE_PADY)
        next_row += 1

    if page_key == "overview":
        # Built for session badge wiring; layout hides it on the clean Overview.
        badge_host = tk.Frame(header, bg=PANEL, highlightthickness=0)
        header._overview_session_badge_host = badge_host  # type: ignore[attr-defined]

    return header


def page_section_caption(parent: tk.Misc, text: str, *, bg: str = PANEL) -> tk.Label:
    """Muted Title Case section caption (never ALL CAPS chrome)."""
    lbl = tk.Label(
        parent,
        text=(text or "").strip(),
        bg=bg,
        fg=MUTED,
        font=T.FONT_SECTION_TITLE,
        anchor="w",
    )
    lbl.pack(fill=tk.X, pady=SECTION_TITLE_PADY)
    return lbl


__all__ = [
    "PAGE_BRIEFINGS",
    "PageBriefing",
    "install_page_header",
    "page_section_caption",
]
