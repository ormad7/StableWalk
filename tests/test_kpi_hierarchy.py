"""Tests for KPI tier hierarchy and technical Details panels."""

from __future__ import annotations

import tkinter as tk

from stablewalk.ui.tk.kpi_cards import (
    create_kpi_card,
    create_technical_details_panel,
    normalize_kpi_tier,
    update_kpi_card,
)


def test_normalize_kpi_tier_aliases() -> None:
    assert normalize_kpi_tier("primary") == "primary"
    assert normalize_kpi_tier("secondary") == "secondary"
    assert normalize_kpi_tier("technical") == "technical"
    assert normalize_kpi_tier("supporting") == "technical"


def test_primary_secondary_technical_card_sizes() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        primary = create_kpi_card(root, key="p", title="Primary", tier="primary")
        secondary = create_kpi_card(root, key="s", title="Secondary", tier="secondary")
        technical = create_kpi_card(root, key="t", title="Technical", tier="technical")
        assert primary.tier == "primary"
        assert secondary.tier == "secondary"
        assert technical.tier == "technical"
        # Primary uses the hero font; secondary/technical use compact value fonts.
        assert primary.value_lbl.cget("font") != secondary.value_lbl.cget("font")
        update_kpi_card(primary, value="80", unit="/100", available=True, numeric=80.0)
        update_kpi_card(technical, value="1.20", unit="s", available=True, numeric=1.2)
        assert primary.value_lbl.cget("text") == "80"
        assert technical.value_lbl.cget("text") == "1.20"
    finally:
        root.destroy()


def test_technical_details_panel_toggles_without_destroying_content() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        panel = create_technical_details_panel(root, title="Technical details")
        child = create_kpi_card(panel.content, key="x", title="Hidden", tier="technical")
        assert panel.expanded is False
        # Collapsed: content is not packed into the shell.
        assert panel.content.winfo_manager() == ""
        panel.toggle.invoke()
        root.update_idletasks()
        assert panel.expanded is True
        assert panel.content.winfo_manager() == "pack"
        panel.toggle.invoke()
        root.update_idletasks()
        assert panel.expanded is False
        assert panel.content.winfo_manager() == ""
        # Widget still exists — information was not removed.
        assert child.value_lbl.winfo_exists()
    finally:
        root.destroy()
