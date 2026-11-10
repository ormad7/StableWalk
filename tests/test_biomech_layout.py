"""Biomechanics workspace column balance — layout only."""

from __future__ import annotations


def test_biomech_side_rails_stay_readable_without_eating_the_graph() -> None:
    from stablewalk.ui.tk.dashboard_biomechanics import (
        BIOMECH_CHART_COL_MIN_W,
        BIOMECH_CHART_FIGSIZE,
        BIOMECH_CLINICAL_MAX_W,
        BIOMECH_CLINICAL_MIN_W,
        BIOMECH_FINDINGS_MAX_W,
        BIOMECH_FINDINGS_MIN_W,
        BIOMECH_RAIL_HEADER_ONLY_W,
        BIOMECH_STACK_BELOW_W,
        BIOMECH_STACK_SIDE_MAX_H,
        biomech_balanced_column_widths,
    )

    assert BIOMECH_FINDINGS_MIN_W >= 180
    assert BIOMECH_FINDINGS_MAX_W <= 220
    assert BIOMECH_FINDINGS_MIN_W < BIOMECH_FINDINGS_MAX_W
    assert BIOMECH_CLINICAL_MIN_W >= 240
    assert BIOMECH_CLINICAL_MAX_W <= 380
    assert BIOMECH_CLINICAL_MIN_W < BIOMECH_CLINICAL_MAX_W
    assert 32 <= BIOMECH_RAIL_HEADER_ONLY_W < BIOMECH_FINDINGS_MIN_W
    assert BIOMECH_CHART_COL_MIN_W >= 320
    assert BIOMECH_STACK_BELOW_W >= 960
    assert BIOMECH_STACK_SIDE_MAX_H <= 280
    assert BIOMECH_CHART_FIGSIZE[0] >= 13.0
    assert BIOMECH_CHART_FIGSIZE[1] >= 8.5

    findings, chart, clinical = biomech_balanced_column_widths(1440)
    assert BIOMECH_FINDINGS_MIN_W <= findings <= BIOMECH_FINDINGS_MAX_W
    assert BIOMECH_CLINICAL_MIN_W <= clinical <= BIOMECH_CLINICAL_MAX_W
    assert chart > findings
    assert chart > clinical
    assert chart >= BIOMECH_CHART_COL_MIN_W
    assert chart / (findings + chart + clinical) >= 0.48

    f2, c2, i2 = biomech_balanced_column_widths(1100)
    assert f2 >= BIOMECH_FINDINGS_MIN_W
    assert i2 >= BIOMECH_CLINICAL_MIN_W
    assert c2 >= BIOMECH_CHART_COL_MIN_W

    f_narrow, _c_narrow, _i_narrow = biomech_balanced_column_widths(900)
    assert f_narrow >= BIOMECH_FINDINGS_MIN_W


def test_biomech_geometry_updates_on_every_width() -> None:
    import tkinter as tk
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_biomechanics import (
        BIOMECH_STACK_BELOW_W,
        apply_biomech_workspace_geometry,
    )

    try:
        root = tk.Tk()
    except tk.TclError:  # pragma: no cover - headless CI without display
        return
    root.withdraw()
    try:
        body = tk.Frame(root, width=1440, height=800)
        body.grid_propagate(False)
        body.grid(row=0, column=0)
        sidebar = tk.Frame(body, width=200)
        chart = tk.Frame(body)
        clinical = tk.Frame(body, width=280)
        footer = tk.Frame(chart)
        footer.grid(row=3, column=0, sticky="ew")
        sidebar.grid(row=0, column=0, sticky="nsew")
        chart.grid(row=0, column=1, sticky="nsew")
        clinical.grid(row=0, column=2, sticky="nsew")
        gui = SimpleNamespace(
            _biomech_body=body,
            _biomech_sidebar=sidebar,
            _biomech_chart_panel=chart,
            _biomech_clinical_host=clinical,
            _biomech_chart_footer=footer,
            _biomech_split=None,
        )
        apply_biomech_workspace_geometry(gui, width=1440, height=800)
        assert gui._biomech_stacked is False
        assert int(float(sidebar.cget("height") or 0)) >= 240
        apply_biomech_workspace_geometry(gui, width=1600, height=800)
        assert gui._biomech_stacked is False
        assert getattr(gui, "_biomech_rails_collapsed", False) is False
        assert 180 <= int(float(sidebar.cget("width") or 0)) <= 220
        assert 250 <= int(float(clinical.cget("width") or 0)) <= 380
        assert footer.winfo_manager() == "grid"
        apply_biomech_workspace_geometry(gui, width=1600, height=500)
        assert gui._biomech_footer_compact_hidden is True
        assert footer.winfo_manager() == ""
        apply_biomech_workspace_geometry(gui, width=1600, height=800)
        assert footer.winfo_manager() == "grid"

        apply_biomech_workspace_geometry(
            gui, width=BIOMECH_STACK_BELOW_W - 40, height=800
        )
        assert gui._biomech_stacked is False
        assert getattr(gui, "_biomech_rails_collapsed", False) is True
        assert sidebar.winfo_manager() == "grid"
        collapsed_info = sidebar.grid_info()
        chart_info = chart.grid_info()
        assert int(collapsed_info["column"]) == 0
        assert int(chart_info["column"]) == 1
        assert int(collapsed_info["row"]) == 0
        assert int(chart_info["row"]) == 0
        assert body.grid_rowconfigure(0)["weight"] == 1
        assert int(sidebar.cget("width") or 0) <= 188
        assert int(clinical.cget("width") or 0) <= 230

        apply_biomech_workspace_geometry(gui, width=1440, height=800)
        assert gui._biomech_stacked is False
        assert sidebar.winfo_manager() == "grid"
        wide_info = sidebar.grid_info()
        wide_chart = chart.grid_info()
        assert int(wide_info["column"]) == 0
        assert int(wide_info["row"]) == 0
        assert int(wide_chart["column"]) == 1
        assert int(wide_info["columnspan"]) == 1

        apply_biomech_workspace_geometry(gui, width=1440, height=200)
        assert int(float(sidebar.cget("height") or 0)) <= 200
        assert int(float(clinical.cget("height") or 0)) <= 200
    finally:
        root.destroy()


def test_biomechanics_page_gives_graph_most_width() -> None:
    from stablewalk.ui.tk.dashboard_biomechanics import (
        BIOMECH_CHART_COL_MIN_W,
        BIOMECH_CHART_FIGSIZE,
        BIOMECH_CLINICAL_MAX_W,
        BIOMECH_FINDINGS_MAX_W,
        biomech_balanced_column_widths,
    )
    from stablewalk.ui.viewers.biomechanics_charts import BIOMECH_TECH_CHART_MIN_H

    findings, chart, clinical = biomech_balanced_column_widths(1600)
    assert chart > findings + clinical * 0.5
    assert BIOMECH_FINDINGS_MAX_W < BIOMECH_CHART_COL_MIN_W
    assert BIOMECH_CLINICAL_MAX_W <= 380
    assert BIOMECH_CLINICAL_MAX_W < BIOMECH_CHART_COL_MIN_W + 40
    assert BIOMECH_CHART_FIGSIZE[0] >= 13.0
    assert BIOMECH_CHART_FIGSIZE[1] >= 8.5
    assert BIOMECH_TECH_CHART_MIN_H >= 680


def test_biomech_desktop_sizes_keep_three_col_proportions() -> None:
    from stablewalk.ui.tk.dashboard_biomechanics import (
        BIOMECH_CLINICAL_MAX_W,
        BIOMECH_CLINICAL_MIN_W,
        BIOMECH_FINDINGS_MAX_W,
        BIOMECH_FINDINGS_MIN_W,
        BIOMECH_STACK_BELOW_W,
        biomech_balanced_column_widths,
    )
    from stablewalk.ui.theme import PAD_XS

    for total in (1920, 1600, 1366):
        assert total >= BIOMECH_STACK_BELOW_W
        findings, chart, clinical = biomech_balanced_column_widths(total)
        assert BIOMECH_FINDINGS_MIN_W <= findings <= BIOMECH_FINDINGS_MAX_W
        assert BIOMECH_CLINICAL_MIN_W <= clinical <= BIOMECH_CLINICAL_MAX_W
        assert chart > findings
        assert chart > clinical
        usable = total - int(PAD_XS) * 2
        assert findings + chart + clinical == usable


def test_key_findings_panel_is_parented_on_body() -> None:
    """Stacking must re-grid Key Findings on body — never hide a paned parent."""
    import inspect

    from stablewalk.ui.tk import dashboard_biomechanics as mod

    build_source = inspect.getsource(mod.build_biomechanics_tab)
    geom_source = inspect.getsource(mod.apply_biomech_workspace_geometry)
    # Chart | interpretation are a PanedWindow; Key Findings stays on body.
    assert "create_h_split" in build_source
    assert "split_add(gui._biomech_chart_split, chart_panel" in build_source
    assert "split_add(gui._biomech_chart_split, clinical_host" in build_source
    assert "split_add(gui._biomech_chart_split, sidebar" not in build_source
    assert "Biomechanical Interpretation" in build_source
    assert "Clinical Interpretation" not in build_source
    assert "split_clear" in geom_source
    assert "paned.grid_remove" in geom_source
    compact = [
        key
        for key, _title, _tip, _attr in mod._BIOMECH_SIDEBAR_FIELDS
        if key
        in {"gait_quality", "stability_state", "symmetry", "stability_margin"}
    ]
    assert compact == [
        "gait_quality",
        "stability_state",
        "symmetry",
        "stability_margin",
    ]


def test_biomech_chart_clinical_splitter_keeps_findings_on_body() -> None:
    import tkinter as tk
    from types import SimpleNamespace

    from stablewalk.ui.tk.analysis_split import create_h_split, split_add
    from stablewalk.ui.tk.dashboard_biomechanics import apply_biomech_workspace_geometry

    try:
        root = tk.Tk()
    except tk.TclError:
        return
    root.withdraw()
    try:
        body = tk.Frame(root, width=1440, height=800)
        body.grid_propagate(False)
        body.grid(row=0, column=0)
        sidebar = tk.Frame(body, width=200)
        paned = create_h_split(body)
        chart = tk.Frame(paned)
        clinical = tk.Frame(paned, width=280)
        split_add(paned, chart, minsize=340)
        split_add(paned, clinical, minsize=260)
        sidebar.grid(row=0, column=0, sticky="nsew")
        paned.grid(row=0, column=1, sticky="nsew")
        gui = SimpleNamespace(
            _biomech_body=body,
            _biomech_sidebar=sidebar,
            _biomech_chart_panel=chart,
            _biomech_clinical_host=clinical,
            _biomech_chart_footer=None,
            _biomech_split=None,
            _biomech_chart_split=paned,
        )
        apply_biomech_workspace_geometry(gui, width=1440, height=800)
        assert gui._biomech_stacked is False
        assert str(sidebar.master) == str(body)
        panes = [str(p) for p in paned.panes()]
        assert str(chart) in panes
        assert str(clinical) in panes
        assert int(sidebar.grid_info()["column"]) == 0
    finally:
        root.destroy()


def test_clinical_rail_uses_one_vertical_overflow_shell() -> None:
    import inspect

    from stablewalk.ui.tk import dashboard_biomechanics as mod

    build_source = inspect.getsource(mod.build_biomechanics_tab)
    card_source = inspect.getsource(mod._build_clinical_interpretation_card)

    assert "_install_biomech_overflow_scroll(clinical_viewport" not in build_source
    assert "_install_biomech_overflow_scroll(sidebar_host, sidebar_inner)" not in build_source
    assert "sidebar_inner.grid(row=0, column=0, sticky=\"nsew\")" in build_source
    assert "width=1, height=1" in inspect.getsource(mod._install_biomech_overflow_scroll)
    assert "_install_biomech_overflow_scroll(body_host, body)" in card_source


def test_clinical_alignment_removes_spacers_in_stacked_layout() -> None:
    import inspect

    from stablewalk.ui.tk import dashboard_biomechanics as mod

    source = inspect.getsource(mod.build_biomechanics_tab)
    assert "if stacked or collapsed:" in source
    assert "clinical_chrome.grid_remove()" in source
    assert "clinical_footer_spacer.grid_remove()" in source
    assert "_chart_panel_row_height" in source
    assert "notebook.winfo_height()" not in source


def test_interpretation_empty_state_is_neutral() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_biomechanics import (
        BIOMECH_INTERPRETATION_EMPTY,
        _clinical_display_is_empty,
        _workspace_tab_label,
    )

    assert "No session interpretation yet" in BIOMECH_INTERPRETATION_EMPTY
    assert "diagnosis" not in BIOMECH_INTERPRETATION_EMPTY.lower()
    empty = SimpleNamespace(
        current_observation="No observation yet — waiting for biomechanics analysis."
    )
    filled = SimpleNamespace(
        current_observation="Current observation: mean 0.52 BH; range 0.48–0.56 BH."
    )
    assert _clinical_display_is_empty(empty) is True
    assert _clinical_display_is_empty(filled) is False
    assert _workspace_tab_label("stability") == "Gait Phase"
    assert _workspace_tab_label("grf") == "Ground Reaction Force"


def test_interpretation_refresh_fills_or_shows_empty_state() -> None:
    import inspect

    from stablewalk.ui.tk import dashboard_biomechanics as mod

    source = inspect.getsource(mod.update_biomech_workspace_panels)
    assert "_set_clinical_body_mode" in source
    assert "_clinical_display_is_empty" in source
    assert "_workspace_tab_label" in source
    assert "build_clinical_interpretation" in source
    card = inspect.getsource(mod._build_clinical_interpretation_card)
    assert "lbl_biomech_clinical_empty" in card
    assert "minsize=140" in card
    assert "grid_propagate(False)" in card
    assert "disclaimer.grid" in card
    assert "_set_clinical_body_mode(gui, empty=True)" in card


def test_deferred_layout_refits_only_the_active_biomechanics_chart_family() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from stablewalk.ui.tk.layout_playback import _run_deferred_layout

    biomech_fit = MagicMock()
    contact_fit = MagicMock()
    gui = SimpleNamespace(
        _biomech_chart_family="contact",
        _fit_biomech_canvas=biomech_fit,
        _fit_contact_gait_canvas=contact_fit,
    )

    _run_deferred_layout(gui)

    contact_fit.assert_called_once_with()
    biomech_fit.assert_not_called()


def test_biomech_key_findings_titles_wrap_without_clipping_details() -> None:
    import tkinter as tk

    from stablewalk.ui.scientific_labels import LABEL_COMPOSITE_GAIT_QUALITY
    from stablewalk.ui.tk.dashboard_biomechanics import BIOMECH_FINDINGS_MIN_W
    from stablewalk.ui.tk.kpi_cards import create_kpi_card

    try:
        root = tk.Tk()
    except tk.TclError:
        return
    root.withdraw()
    try:
        rail = tk.Frame(root, width=BIOMECH_FINDINGS_MIN_W, height=180, bg="#111")
        rail.pack()
        rail.pack_propagate(False)
        kpi = create_kpi_card(
            rail,
            key="gait_quality",
            title=LABEL_COMPOSITE_GAIT_QUALITY,
            compact=True,
            fill=True,
            tier="secondary",
            title_wrap_lines=2,
        )
        root.update_idletasks()
        kpi.title_lbl.master.event_generate("<Configure>")
        root.update_idletasks()
        wrap = int(float(str(kpi.title_lbl.cget("wraplength") or 0)))
        assert wrap >= 96
        assert int(kpi.title_lbl.winfo_reqheight() or 0) >= int(
            kpi.value_lbl.winfo_reqheight() or 0
        )
        details = None
        for child in kpi.value_lbl.master.winfo_children():
            try:
                if str(child.cget("text") or "") == "Details":
                    details = child
                    break
            except tk.TclError:
                continue
        assert details is not None
        assert details.winfo_manager() == "pack"
        assert str(details.pack_info().get("side") or "") == "right"
        assert kpi.quality_lbl.winfo_manager() == "pack"
        # Status sits under the value so value / unit / Details stay unclipped.
        assert kpi.quality_lbl.master is kpi.title_lbl.master.master
    finally:
        root.destroy()
