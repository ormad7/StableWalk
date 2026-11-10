"""Smoke tests for the clinical design system (presentation only)."""

from __future__ import annotations

import tkinter as tk

import pytest

from stablewalk.ui.design_system import tokens as T
from stablewalk.ui.design_system.components import (
    CollapsibleSection,
    EmptyState,
    ExportButton,
    MetricCard,
    QuietPanel,
    SectionHeader,
    SoftSeparator,
    StatusBadge,
    WarningBanner,
    update_status_badge,
)
from stablewalk.ui import theme as theme_mod
from stablewalk.ui.tk.dashboard_page_frame import (
    PAGE_BRIEFINGS,
    install_page_header,
    page_section_caption,
)


def test_spacing_is_4_8_16_24_scale() -> None:
    assert T.PAD_XS == 4
    assert T.PAD_SM == 8
    assert T.PAD_MD == 16
    assert T.PAD_LG == 24
    assert T.PAD_XL == 32
    assert T.SPACE_1 == 4
    assert T.SPACE_2 == 8
    assert T.SPACE_3 == 16
    assert T.SPACE_4 == 24
    assert T.SPACE_5 == 32
    assert T.PAD_XS < T.PAD_SM < T.PAD_MD < T.PAD_LG


def test_layout_aliases_map_to_spacing_scale() -> None:
    assert T.INLINE_GAP == T.PAD_XS
    assert T.CONTROL_GAP == T.PAD_SM
    assert T.SECTION_GAP == T.PAD_MD
    assert T.PAGE_GUTTER == T.PAD_MD
    assert T.CARD_GAP == T.PAD_SM
    assert T.KPI_GAP == T.PAD_SM
    assert T.PAGE_HEADER_PADY == (0, T.PAD_MD)
    assert T.PAGE_HEADER_SUBTITLE_PADY == (T.PAD_SM, 0)
    assert T.SECTION_TITLE_PADY == (T.PAD_MD, T.PAD_XS)
    assert T.DASHBOARD_TAB_PAD == (T.PAD_MD, T.PAD_SM)
    assert T.CARD_PAD == (T.PAD_MD, T.PAD_SM)


def test_control_sizes_unified() -> None:
    assert T.BUTTON_HEIGHT == 32
    assert T.CONTROL_HEIGHT == 32
    assert T.INPUT_HEIGHT == 32
    assert T.COMBO_HEIGHT == 32
    assert T.RADIUS_PX == 4
    assert T.BTN_PAD == (12, 7)
    assert T.BTN_PAD_COMPACT == (10, 5)
    assert T.BTN_PAD_TRANSPORT == T.BTN_PAD_COMPACT


def test_typography_floor_at_9pt() -> None:
    for font in (
        T.FONT_SUPPORTING,
        T.FONT_META,
        T.FONT_BODY,
        T.FONT_CARD_TITLE,
        T.FONT_SECTION_TITLE,
        T.FONT_PAGE_TITLE,
        T.FONT_METRIC,
        T.FONT_METRIC_MD,
        T.FONT_METRIC_SM,
        T.FONT_BUTTON,
        T.FONT_BUTTON_SM,
    ):
        assert int(font[1]) >= 9


def test_theme_mirrors_design_tokens() -> None:
    """theme.py must stay an alias layer — not a second type/spacing system."""
    assert theme_mod.PAD_XS == T.PAD_XS
    assert theme_mod.PAD_SM == T.PAD_SM
    assert theme_mod.PAD_MD == T.PAD_MD
    assert theme_mod.PAD_LG == T.PAD_LG
    assert theme_mod.PAD_XL == T.PAD_XL
    assert theme_mod.INLINE_GAP == T.INLINE_GAP
    assert theme_mod.SECTION_GAP == T.SECTION_GAP
    assert theme_mod.PAGE_HEADER_PADY == T.PAGE_HEADER_PADY
    assert theme_mod.BTN_PAD == T.BTN_PAD
    assert theme_mod.BTN_PAD_COMPACT == T.BTN_PAD_COMPACT
    assert theme_mod.BTN_PAD_TRANSPORT == T.BTN_PAD_TRANSPORT
    assert theme_mod.BUTTON_HEIGHT == T.BUTTON_HEIGHT
    assert theme_mod.RADIUS_PX == T.RADIUS_PX
    assert theme_mod.FONT_TITLE == T.FONT_PAGE_TITLE
    assert theme_mod.FONT_SECTION == T.FONT_SECTION_TITLE
    assert theme_mod.FONT_HEADING == T.FONT_CARD_TITLE
    assert theme_mod.FONT_KPI_HERO == T.FONT_METRIC
    assert theme_mod.FONT_KPI_VALUE == T.FONT_METRIC_MD
    assert theme_mod.FONT_BUTTON == T.FONT_BUTTON
    assert theme_mod.BG == T.BG
    assert theme_mod.PANEL == T.PANEL
    assert theme_mod.ACCENT == T.ACCENT
    assert theme_mod.SUCCESS == T.STATUS_OK
    assert theme_mod.WARNING == T.STATUS_WARN
    assert theme_mod.DANGER == T.STATUS_FAIL


def test_semantic_status_colors_distinct() -> None:
    assert T.STATUS_OK != T.STATUS_WARN
    assert T.STATUS_WARN != T.STATUS_FAIL
    assert T.STATUS_FAIL != T.STATUS_INFO


def test_page_briefings_cover_all_tabs() -> None:
    assert set(PAGE_BRIEFINGS) >= {
        "overview",
        "motion",
        "biomechanics",
        "results",
        "compare",
        "advanced",
    }
    for briefing in PAGE_BRIEFINGS.values():
        assert briefing.title
        assert not briefing.title.isupper() or len(briefing.title) <= 3


@pytest.fixture()
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk not available")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


def test_components_construct(root) -> None:
    host = QuietPanel(root)
    SectionHeader(host, "Session Metrics", subtitle="Supporting")
    SoftSeparator(host)
    badge = StatusBadge(host, "Acceptable", tone="ok")
    update_status_badge(badge, "Borderline", tone="warn")
    WarningBanner(host, "Research use only", tone="warn")
    EmptyState(host, "No analysis yet")
    ExportButton(host, "Export CSV")
    CollapsibleSection(host, "Details", expanded=False)
    card = MetricCard(host, title="Cadence", unit="steps/min", description="Step rate")
    assert set(card) >= {"frame", "accent", "title", "value", "desc", "status"}
    assert card["title"].cget("text") == "Cadence"
    assert "18" in str(card["value"].cget("font"))
    assert "Semibold" in str(card["value"].cget("font"))


def test_page_header_uses_token_spacing(root) -> None:
    import inspect

    sig = inspect.signature(install_page_header)
    assert sig.parameters["pady"].default == T.PAGE_HEADER_PADY

    host = tk.Frame(root)
    host.columnconfigure(0, weight=1)
    install_page_header(host, "overview", row=0)
    caption_host = tk.Frame(root)
    caption = page_section_caption(caption_host, "Clinical Findings")
    assert caption.cget("text") == "Clinical Findings"
    assert caption.cget("font") == T.FONT_SECTION_TITLE or "11" in str(
        caption.cget("font")
    )
